"""
Model 3 — Decoder-Only Autoregressive Transformer (GPT-style).

Architecture overview
---------------------
A single unified vocabulary covers both condition tokens and date tokens.
The full input sequence is:

    [day_tok] [month_tok] [leap_tok] [decade_tok] [SOS] Y1 Y2 Y3 Y4 - M1 M2 - D1 D2 [EOS]

All tokens share one token embedding table.  A causal (upper-triangular) mask
prevents any position from attending to future positions.

Loss is computed ONLY on the date positions (SOS index + 1 onwards), not on
the condition prefix — condition tokens are part of the "prompt" and are not
predicted.

Key design choices
------------------
* Unified vocabulary with learned positional embeddings (max_len=32).
* Pre-LayerNorm transformer blocks for stable training.
* AdamW optimiser with weight decay = 0.01.
* CosineAnnealingLR scheduler.
* Loss mask: ignore condition prefix tokens, ignore PAD_ID tokens.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ─────────────────────────────────────────────────────────────────────────────
# Unified vocabulary
# ─────────────────────────────────────────────────────────────────────────────
#
# Condition tokens (input prefix):
#   Day    (0-6)   : MON=0, TUE=1, WED=2, THU=3, FRI=4, SAT=5, SUN=6
#   Month  (7-18)  : JAN=7, FEB=8, ..., DEC=18
#   Leap   (19-20) : False=19, True=20
#   Decade (21-61) : 180→21, 181→22, ..., 220→61
#
# Date tokens (output):
#   Digits 0-9  →  62-71
#   SEP '-'     →  72
#   PAD         →  73
#   SOS         →  74
#   EOS         →  75
#
# Total vocab: 76 tokens.

DAY_OFFSET    = 0
MONTH_OFFSET  = 7
LEAP_OFFSET   = 19
DECADE_OFFSET = 21   # decade id 0 (='180') maps to token 21

DIGIT_OFFSET  = 62   # digit 0 maps to token 62, digit 9 to token 71
SEP_TOK       = 72
PAD_TOK       = 73
SOS_TOK       = 74
EOS_TOK       = 75
UNIFIED_VOCAB = 76

NUM_COND_TOKENS = 4   # the prefix length (day, month, leap, decade)


def cond_to_unified(
    day: Tensor,
    month: Tensor,
    leap: Tensor,
    decade: Tensor,
) -> Tensor:
    """
    Map per-condition integer IDs to unified vocabulary token IDs.

    Parameters
    ----------
    day    : LongTensor (B,)   values 0-6
    month  : LongTensor (B,)   values 0-11
    leap   : LongTensor (B,)   values 0-1
    decade : LongTensor (B,)   values 0-40

    Returns
    -------
    LongTensor (B, 4)   — the four condition tokens in unified vocab
    """
    return torch.stack(
        [
            day    + DAY_OFFSET,
            month  + MONTH_OFFSET,
            leap   + LEAP_OFFSET,
            decade + DECADE_OFFSET,
        ],
        dim=1,
    )


def date_toks_to_unified(date_toks: Tensor) -> Tensor:
    """
    Map date token IDs (output vocab 0-13) to unified vocab.

    Mapping:
      0-9   → DIGIT_OFFSET + digit   (62-71)
      10    → SEP_TOK (72)
      11    → PAD_TOK (73)
      12    → SOS_TOK (74)
      13    → EOS_TOK (75)

    Parameters
    ----------
    date_toks : LongTensor (any shape)

    Returns
    -------
    LongTensor (same shape)
    """
    mapping = torch.tensor(
        [62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75],
        dtype=torch.long,
        device=date_toks.device,
    )
    return mapping[date_toks]


def unified_to_date_toks(unified: Tensor) -> Tensor:
    """
    Reverse map from unified vocab IDs back to date token IDs (0-13).

    Non-date tokens (condition tokens) map to PAD_ID=11.

    Parameters
    ----------
    unified : LongTensor (any shape)

    Returns
    -------
    LongTensor (same shape)
    """
    out = torch.full_like(unified, 11)          # default to PAD
    mask = (unified >= DIGIT_OFFSET) & (unified <= EOS_TOK)
    out[mask] = unified[mask] - DIGIT_OFFSET    # 62→0, 63→1, ..., 75→13
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Pre-LN Transformer Block
# ─────────────────────────────────────────────────────────────────────────────

class PreLNBlock(nn.Module):
    """
    Pre-LayerNorm Transformer decoder block (no cross-attention).

    Pre-LN: LayerNorm before attention / FFN → more stable training.

    Parameters
    ----------
    d_model, nhead, dim_feedforward, dropout : standard Transformer parameters.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor, causal_mask: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x           : FloatTensor (batch, seq, d_model)
        causal_mask : BoolTensor  (seq, seq)  — True = mask out

        Returns
        -------
        FloatTensor (batch, seq, d_model)
        """
        # Self-attention with residual
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.attn(x, x, x, attn_mask=causal_mask, is_causal=True)
        x = residual + attn_out
        # FFN with residual
        x = x + self.ffn(self.norm2(x))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Autoregressive Transformer
# ─────────────────────────────────────────────────────────────────────────────

class AutoregressiveDateTransformer(nn.Module):
    """
    GPT-style decoder-only Transformer for conditional date generation.

    Full sequence during training:
        [day] [month] [leap] [decade] [SOS] Y1 Y2 Y3 Y4 - M1 M2 - D1 D2 [EOS] [PAD]...

    Loss is computed only on positions after [SOS] (i.e., the date tokens).

    Parameters
    ----------
    d_model : int
    nhead : int
    num_layers : int
    dim_feedforward : int
    dropout : float
    max_len : int       Maximum total sequence length (conditions + date).
    label_smoothing : float
    """

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_len: int = 32,
        label_smoothing: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model    = d_model
        self.max_len    = max_len
        self.num_cond   = NUM_COND_TOKENS

        # Shared token + position embeddings
        self.tok_emb = nn.Embedding(UNIFIED_VOCAB, d_model, padding_idx=PAD_TOK)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.drop    = nn.Dropout(dropout)
        self.norm_in = nn.LayerNorm(d_model)

        self.blocks = nn.ModuleList(
            [PreLNBlock(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)]
        )
        self.norm_out  = nn.LayerNorm(d_model)
        self.lm_head   = nn.Linear(d_model, UNIFIED_VOCAB, bias=False)

        # Weight tying: share token embedding with output projection (common in LMs)
        self.lm_head.weight = self.tok_emb.weight

        self.criterion = nn.CrossEntropyLoss(
            ignore_index=PAD_TOK,
            label_smoothing=label_smoothing,
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Standard GPT-style weight initialisation."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ── Forward ──────────────────────────────────────────────────────────────

    def _embed(self, token_ids: Tensor) -> Tensor:
        """Embed tokens and add positional encoding."""
        T = token_ids.size(1)
        positions = torch.arange(T, device=token_ids.device).unsqueeze(0)
        x = self.tok_emb(token_ids) + self.pos_emb(positions)
        return self.norm_in(self.drop(x))

    def _causal_mask(self, T: int, device: torch.device) -> Tensor:
        """Upper-triangular causal mask (True = ignore) of shape (T, T)."""
        return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, token_ids: Tensor) -> Tensor:
        """
        Forward pass over a full token sequence.

        Parameters
        ----------
        token_ids : LongTensor (batch, seq)
            Unified-vocab token ids.

        Returns
        -------
        FloatTensor (batch, seq, UNIFIED_VOCAB) — raw logits
        """
        T = token_ids.size(1)
        mask = self._causal_mask(T, token_ids.device)
        x = self._embed(token_ids)
        for block in self.blocks:
            x = block(x, mask)
        return self.lm_head(self.norm_out(x))

    def compute_loss(
        self,
        day: Tensor,
        month: Tensor,
        leap: Tensor,
        decade: Tensor,
        target: Tensor,
    ) -> Tensor:
        """
        Build the full sequence, run forward pass, compute cross-entropy loss
        on date positions only (not on condition prefix tokens).

        Parameters
        ----------
        day, month, leap, decade : LongTensor (batch,)
        target : LongTensor (batch, MAX_OUTPUT_LEN)
            Padded date tokens in the output vocabulary (0-13 ids).

        Returns
        -------
        Scalar FloatTensor
        """
        # Map condition tokens to unified vocab
        cond_toks = cond_to_unified(day, month, leap, decade)           # (B, 4)
        # Map date output tokens to unified vocab
        date_toks = date_toks_to_unified(target)                         # (B, T_date)

        # Full sequence: [cond(4)] + [date(T_date)]
        # During training we feed all except the last date token as input
        seq_in  = torch.cat([cond_toks, date_toks[:, :-1]], dim=1)      # (B, 4+T_date-1)
        seq_tgt = torch.cat(
            [torch.full_like(cond_toks, PAD_TOK), date_toks[:, 1:]], dim=1
        )  # (B, 4+T_date-1) — condition positions set to PAD so loss ignores them

        logits = self.forward(seq_in)                                    # (B, seq, V)
        return self.criterion(
            logits.reshape(-1, UNIFIED_VOCAB),
            seq_tgt.reshape(-1),
        )

    # ── Inference ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        day: Tensor,
        month: Tensor,
        leap: Tensor,
        decade: Tensor,
        max_date_len: int = 12,
        top_k: int = 1,
    ) -> Tensor:
        """
        Generate date token ids autoregressively given condition tokens as prompt.

        Parameters
        ----------
        day, month, leap, decade : LongTensor (batch,)
        max_date_len : int   Maximum number of date tokens to generate.
        top_k : int          1 = greedy; >1 = top-k sampling.

        Returns
        -------
        LongTensor (batch, generated_date_len)   — in output vocab (0-13)
        """
        B      = day.size(0)
        device = day.device

        # Prompt: condition tokens in unified vocab + SOS
        cond_toks = cond_to_unified(day, month, leap, decade)           # (B, 4)
        sos       = torch.full((B, 1), SOS_TOK, dtype=torch.long, device=device)
        generated = torch.cat([cond_toks, sos], dim=1)                  # (B, 5)

        done         = torch.zeros(B, dtype=torch.bool, device=device)
        date_tokens  : list[Tensor] = []

        for _ in range(max_date_len - 1):
            logits    = self.forward(generated)[:, -1, :]               # (B, V)

            if top_k == 1:
                next_tok = logits.argmax(dim=-1)
            else:
                top_logits, top_ids = logits.topk(min(top_k, UNIFIED_VOCAB), dim=-1)
                probs    = F.softmax(top_logits, dim=-1)
                idx      = torch.multinomial(probs, 1).squeeze(-1)
                next_tok = top_ids.gather(1, idx.unsqueeze(1)).squeeze(1)

            next_tok = torch.where(done, torch.full_like(next_tok, PAD_TOK), next_tok)
            date_tokens.append(next_tok)
            generated = torch.cat([generated, next_tok.unsqueeze(1)], dim=1)
            done = done | (next_tok == EOS_TOK)
            if done.all():
                break

        # Stack and convert from unified vocab back to output vocab (0-13)
        date_unified = torch.stack(date_tokens, dim=1)                  # (B, T)
        return unified_to_date_toks(date_unified)