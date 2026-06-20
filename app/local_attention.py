"""Local (sliding-window) attention for nano-parakeet's FastConformer encoder.

Port of NeMo's RelPositionMultiHeadAttentionLongformer to nano-parakeet's
module interface. Replaces O(T^2) full self-attention with O(T*w) banded
attention, enabling transcription of long audio without OOM.

Source: github.com/NVIDIA/NeMo
  nemo/collections/asr/parts/submodules/multi_head_attention.py

Ported verbatim:
  _skew, _skew2, _chunk_overlap, sliding_chunks_matmul_qk,
  sliding_chunks_matmul_pv

Adapted to nano-parakeet's interface:
  RelPositionLocalMHA.forward: same computation as NeMo's Longformer.forward.
    nano-parakeet calls self_attn(x, pos_emb, pad_mask) with q=k=v=x projected
    inline. NeMo calls self_attn(query, key, value, pad_mask, pos_emb, cache)
    with forward_qkv() inherited from base class. Cache is for NeMo's streaming
    inference; nano-parakeet runs full sequence, so cache is not ported.
  RelPositionLocalMHA.__init__: matches nano-parakeet's RelPositionMHA param
    layout (no bias, no dropout) so weights transfer via load_state_dict.
  LocalRelPositionalEncoding: NeMo's LocalAttRelPositionalEncoding without
    xscale, dropout, or dynamic extend_pe.

Rewritten:
  _get_invalid_locations_mask / _mask_invalid_locations: NeMo uses lru_cache
    with runtime .to(device) (CPU-GPU sync). Replaced with _build_boundary_masks
    at init, registered as module buffers, applied once in forward after all
    score contributions are combined.

Dropped:
  Global attention (global_tokens, _compute_global_key_attn, etc.),
  avoid_float16_autocast_context, SDPA backend dispatch.
"""

import math
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

INF_VAL = 10000.0


# ---------------------------------------------------------------------------
# Positional encoding for local attention (fixed window, not sequence-length)
# ---------------------------------------------------------------------------

class LocalRelPositionalEncoding(nn.Module):
    """Relative positional encoding sized to the attention window, not the sequence."""

    def __init__(self, d_model: int, att_context_size: list[int]):
        super().__init__()
        left, right = att_context_size
        positions = torch.arange(left, -right - 1, -1, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / d_model)
        )
        pe = torch.zeros(left + right + 1, d_model)
        pe[:, 0::2] = torch.sin(positions * div)
        pe[:, 1::2] = torch.cos(positions * div)
        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)  # [1, 2w+1, d]

    def forward(self, x: torch.Tensor):
        return x, self.pe


# ---------------------------------------------------------------------------
# Sliding-window helpers (ported from NeMo's Longformer attention)
# ---------------------------------------------------------------------------

def _skew(x: torch.Tensor, direction: list[int], padding_value: float) -> torch.Tensor:
    x_padded = F.pad(x, direction, value=padding_value)
    return x_padded.view(*x_padded.size()[:-2], x_padded.size(-1), x_padded.size(-2))


def _skew2(x: torch.Tensor, padding_value: float) -> torch.Tensor:
    B, C, M, L = x.size()
    x = F.pad(x, (0, M + 1), value=padding_value)
    x = x.view(B, C, -1)
    x = x[:, :, :-M]
    x = x.view(B, C, M, M + L)
    return x[:, :, :, :-1]


def _chunk_overlap(x: torch.Tensor, w: int) -> torch.Tensor:
    x = x.view(x.size(0), x.size(1) // (w * 2), w * 2, x.size(2))
    s0, s1, s2, s3 = x.stride()
    return x.as_strided(
        size=(x.size(0), x.size(1) * 2 - 1, w * 2, x.size(3)),
        stride=(s0, s1 // 2, s2, s3),
    )


def _build_boundary_masks(w: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Build masks for positions near sequence edges with truncated windows.

    Returns (beginning_mask, ending_mask), both [1, 1, w, w+1] bool tensors.
    Registered as buffers on the module so they move with .to(device).
    """
    diagonals_list = []
    for j in range(-w, 1):
        diagonal_mask = torch.zeros(w, dtype=torch.bool)
        diagonal_mask[:-j] = True
        diagonals_list.append(diagonal_mask)
    mask = torch.stack(diagonals_list, dim=-1)[None, None, :, :]
    return mask, mask.flip(dims=(2, 3))


def _apply_boundary_masks(
    input_tensor: torch.Tensor, w: int,
    beginning_mask: torch.Tensor, ending_mask: torch.Tensor,
):
    """Mask invalid boundary locations in-place on a banded attention tensor."""
    seq_len = input_tensor.size(2)
    begin_input = input_tensor[:, :, :w, :w + 1]
    begin_input.masked_fill_(beginning_mask[:, :, :seq_len].expand(begin_input.size()), -float('inf'))
    end_input = input_tensor[:, :, -w:, -(w + 1):]
    end_input.masked_fill_(ending_mask[:, :, -seq_len:].expand(end_input.size()), -float('inf'))


def sliding_chunks_matmul_qk(
    q: torch.Tensor, k: torch.Tensor, w: int, padding_value: float,
) -> torch.Tensor:
    """Sliding-window Q·Kᵀ → banded attention scores [B, h, T, 2w+1]."""
    bsz, num_heads, seqlen, head_dim = q.size()
    chunks_count = seqlen // w - 1

    q = q.reshape(bsz * num_heads, seqlen, head_dim)
    k = k.reshape(bsz * num_heads, seqlen, head_dim)

    chunk_q = _chunk_overlap(q, w)
    chunk_k = _chunk_overlap(k, w)
    chunk_attn = torch.einsum('bcxd,bcyd->bcxy', chunk_q, chunk_k)
    diagonal_chunk_attn = _skew(chunk_attn, direction=(0, 0, 0, 1), padding_value=padding_value)

    diagonal_attn = diagonal_chunk_attn.new_empty(
        (bsz * num_heads, chunks_count + 1, w, w * 2 + 1)
    )
    diagonal_attn[:, :-1, :, w:] = diagonal_chunk_attn[:, :, :w, :w + 1]
    diagonal_attn[:, -1, :, w:] = diagonal_chunk_attn[:, -1, w:, :w + 1]
    diagonal_attn[:, 1:, :, :w] = diagonal_chunk_attn[:, :, -(w + 1):-1, w + 1:]
    diagonal_attn[:, 0, 1:w, 1:w] = diagonal_chunk_attn[:, 0, :w - 1, 1 - w:]

    return diagonal_attn.view(bsz, num_heads, seqlen, 2 * w + 1)


def sliding_chunks_matmul_pv(
    prob: torch.Tensor, v: torch.Tensor, w: int,
) -> torch.Tensor:
    """Sliding-window attn_probs · V → context [B, T, h, d_k]."""
    bsz, num_heads, seqlen, head_dim = v.size()
    chunks_count = seqlen // w - 1
    chunk_prob = prob.reshape(bsz * num_heads, seqlen // w, w, 2 * w + 1)

    v = v.reshape(bsz * num_heads, seqlen, head_dim)
    padded_v = F.pad(v, (0, 0, w, w), value=-1)
    chunk_v_size = (bsz * num_heads, chunks_count + 1, 3 * w, head_dim)
    chunk_v_stride = padded_v.stride()
    chunk_v_stride = (chunk_v_stride[0], w * chunk_v_stride[1], chunk_v_stride[1], chunk_v_stride[2])
    chunk_v = padded_v.as_strided(size=chunk_v_size, stride=chunk_v_stride)

    skewed_prob = _skew2(chunk_prob, padding_value=0)
    context = torch.einsum('bcwd,bcdh->bcwh', skewed_prob, chunk_v)
    return context.view(bsz, num_heads, seqlen, head_dim).transpose(1, 2)


# ---------------------------------------------------------------------------
# Local attention module (drop-in replacement for nano-parakeet's RelPositionMHA)
# ---------------------------------------------------------------------------

class RelPositionLocalMHA(nn.Module):
    """Sliding-window relative-position multi-head attention.

    Same learned parameters as RelPositionMHA — weights transfer via load_state_dict.
    """

    def __init__(self, d_model: int = 1024, n_heads: int = 8,
                 att_context_size: list[int] | None = None):
        super().__init__()
        if att_context_size is None:
            att_context_size = [512, 512]
        self.h = n_heads
        self.d_k = d_model // n_heads
        self.s_d_k = math.sqrt(self.d_k)
        self.linear_q = nn.Linear(d_model, d_model, bias=False)
        self.linear_k = nn.Linear(d_model, d_model, bias=False)
        self.linear_v = nn.Linear(d_model, d_model, bias=False)
        self.linear_out = nn.Linear(d_model, d_model, bias=False)
        self.linear_pos = nn.Linear(d_model, d_model, bias=False)
        self.pos_bias_u = nn.Parameter(torch.zeros(n_heads, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.zeros(n_heads, self.d_k))
        self.att_context_size = att_context_size
        w = max(att_context_size)
        beginning_mask, ending_mask = _build_boundary_masks(w)
        self.register_buffer('_begin_mask', beginning_mask, persistent=False)
        self.register_buffer('_end_mask', ending_mask, persistent=False)

    def forward(self, x: torch.Tensor, pos_emb: torch.Tensor,
                pad_mask: torch.Tensor) -> torch.Tensor:
        """
        x:        [B, T, d_model]
        pos_emb:  [1, 2w+1, d_model]
        pad_mask: [B, T] bool (True = padding)
        """
        B, T, _ = x.shape
        w = max(self.att_context_size)

        q = self.linear_q(x).view(B, T, self.h, self.d_k).transpose(1, 2)
        k = self.linear_k(x).view(B, T, self.h, self.d_k).transpose(1, 2)
        v = self.linear_v(x).view(B, T, self.h, self.d_k).transpose(1, 2)

        # pad time to multiple of 2w for chunked matmul
        pad_len = (2 * w - T % (2 * w)) % (2 * w)
        q = F.pad(q, (0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, pad_len))
        # the 2w alignment padding must be masked (value=True), else it leaks into attention
        mask = F.pad(pad_mask, (0, pad_len), value=True)

        q_u = q + self.pos_bias_u.unsqueeze(1)
        q_v = q + self.pos_bias_v.unsqueeze(1)

        # banded content attention  [B, h, T_pad, 2w+1]
        diagonal_ac = sliding_chunks_matmul_qk(q_u, k, w, padding_value=0.0)

        # relative position scores  [B, h, T_pad, 2w+1]
        p = self.linear_pos(pos_emb).view(1, -1, self.h, self.d_k).transpose(1, 2)
        diagonal_bd = torch.matmul(q_v, p.transpose(-2, -1))

        # merge position info into content scores (left / right halves)
        left_ctx = self.att_context_size[0]
        right_ctx = self.att_context_size[1]
        diagonal_ac[:, :, :, :left_ctx] += diagonal_bd[:, :, :, :left_ctx]
        diagonal_ac[:, :, :, -(right_ctx + 1):] += diagonal_bd[:, :, :, left_ctx:]

        scores = diagonal_ac / self.s_d_k

        # mask positions outside the asymmetric window
        start_pos = w - left_ctx
        end_pos = w + right_ctx
        scores[:, :, :, :start_pos] = -INF_VAL
        scores[:, :, :, end_pos + 1:] = -INF_VAL

        # windowed padding mask
        mask_col = mask.unsqueeze(1).unsqueeze(-1)  # [B, 1, T_pad, 1]
        float_mask = mask_col.type_as(scores).masked_fill(mask_col, -INF_VAL)
        ones = float_mask.new_ones(float_mask.size())
        d_mask = sliding_chunks_matmul_qk(ones, float_mask, w, padding_value=0.0)
        scores += d_mask

        # mask boundary positions (truncated windows at sequence edges)
        _apply_boundary_masks(scores, w, self._begin_mask, self._end_mask)

        attn = torch.softmax(scores, dim=-1).masked_fill(mask_col, 0.0)
        out = sliding_chunks_matmul_pv(attn, v, w)
        out = out.reshape(B, -1, self.h * self.d_k)[:, :T]
        return self.linear_out(out)


# ---------------------------------------------------------------------------
# Encoder forward replacement (passes [B, T] pad mask for local attention)
# ---------------------------------------------------------------------------

def _local_attn_encoder_forward(self, features: torch.Tensor, lengths: torch.Tensor):
    x = features.transpose(1, 2)
    x, lengths = self.pre_encode(x, lengths)
    x, pos_emb = self.pos_enc(x)

    B, T, _ = x.shape
    # each layer pads time to a multiple of 2w; the mask must cover those frames too
    pad_mask = torch.arange(T, device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)

    for layer in self.layers:
        x = layer(x, pos_emb, pad_mask)
    return x, lengths


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def enable_local_attention(model, att_context_size: list[int] | None = None):
    """Patch a ParakeetTDT model to use sliding-window local attention.

    Replaces full O(T²) self-attention with O(T·w) local attention,
    usage at ~2.88MB/s or 10GB/h.
    Weights are transferred from the existing attention modules (same param names).

    Args:
        model: ParakeetTDT from from_pretrained().
        att_context_size: [left, right] in encoder frames (1 frame = 80 ms).
            Default [512, 512] ~= 41 s context each side.
            Peak measured allocated VRAM ~13GB for 80min audio.
    """
    if getattr(model, '_local_attention_enabled', False):
        return
    if att_context_size is None:
        att_context_size = [512, 512]

    encoder = model.encoder
    d_model = encoder.layers[0].self_attn.linear_q.in_features
    n_heads = encoder.layers[0].self_attn.h
    device = next(encoder.parameters()).device
    dtype = next(encoder.parameters()).dtype

    # replace positional encoding (buffer-only, no learned weights)
    encoder.pos_enc = LocalRelPositionalEncoding(d_model, att_context_size).to(
        device=device, dtype=dtype,
    )

    # replace attention in each layer (identical param names → direct state_dict copy)
    for layer in encoder.layers:
        old_attn = layer.self_attn
        new_attn = RelPositionLocalMHA(d_model, n_heads, att_context_size)
        new_attn.load_state_dict(old_attn.state_dict())
        layer.self_attn = new_attn.to(device=device, dtype=dtype)

    # swap encoder forward to pass [B, T] pad mask instead of [B, T, T]
    encoder.forward = types.MethodType(_local_attn_encoder_forward, encoder)

    model._local_attention_enabled = True
