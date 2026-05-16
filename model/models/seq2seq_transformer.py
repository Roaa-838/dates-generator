"""
Model 2 — Encoder-Decoder Transformer (Seq2Seq) for conditional date generation.

Architecture overview
---------------------
Encoder : 4-token condition sequence → TransformerEncoder layers → memory
Decoder : autoregressive date token sequence → TransformerDecoder layers → logits

The encoder receives the four condition tokens (day, month, leap, decade) as a
4-element sequence, each with its own learnable embedding + positional encoding.

The decoder generates the date token-by-token (year-first order) using
teacher forcing during training and greedy / sampled decoding at inference.

Key design choices
------------------
* Label smoothing (ε=0.1) prevents overconfident predictions and improves
  generalisation to unseen condition combinations.
* Noam (warm-up) learning rate schedule: lr = d_model^-0.5 × min(step^-0.5, step × warm^-1.5)
* Sinusoidal positional encoding for the encoder; learned embeddings for decoder positions.
* Causal mask on decoder prevents attending to future tokens.
* Pad mask on both encoder and decoder ignores PAD tokens in loss and attention.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ─────────────────────────────────────────────────────────────────────────────
# Positional Encoding  (sinusoidal, as in "Attention Is All You Need")
# ─────────────────────────────────────────────────────────────────────────────

class SinusoidalPositionalEncoding(nn.Module):
    """
    Adds fixed sinusoidal positional encodings to token embeddings.

    Parameters
    ----------
    d_model : int   Embedding dimensionality.
    max_len : int   Maximum sequence length to pre-compute.
    dropout : float
    """

    def __init__(self, d_model: int, max_len: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : FloatTensor (batch, seq, d_model)

        Returns
        -------
        FloatTensor (batch, seq, d_model)
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ─────────────────────────────────────────────────────────────────────────────
# Condition Encoder
# ─────────────────────────────────────────────────────────────────────────────

class ConditionEncoder(nn.Module):
    """
    Embeds the 4 input conditions as a token sequence for the Transformer encoder.

    Each condition type has its own embedding table.  A type-embedding (like BERT's
    segment embedding) is added to distinguish the four condition positions.

    Parameters
    ----------
    d_model : int
    day_vocab, month_vocab, leap_vocab, decade_vocab : int
    dropout : float
    """

    def __init__(
        self,
        d_model: int,
        day_vocab: int = 7,
        month_vocab: int = 12,
        leap_vocab: int = 2,
        decade_vocab: int = 41,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.day_emb    = nn.Embedding(day_vocab,    d_model)
        self.month_emb  = nn.Embedding(month_vocab,  d_model)
        self.leap_emb   = nn.Embedding(leap_vocab,   d_model)
        self.decade_emb = nn.Embedding(decade_vocab, d_model)
        # Type embedding: distinguishes the 4 condition slots
        self.type_emb = nn.Embedding(4, d_model)
        self.pos_enc  = SinusoidalPositionalEncoding(d_model, max_len=8, dropout=dropout)
        self.norm     = nn.LayerNorm(d_model)
        self._d_model = d_model

    def forward(
        self, day: Tensor, month: Tensor, leap: Tensor, decade: Tensor
    ) -> Tensor:
        """
        Parameters
        ----------
        day, month, leap, decade : LongTensor (batch,)

        Returns
        -------
        FloatTensor (batch, 4, d_model)   — encoder input sequence
        """
        B = day.size(0)
        device = day.device
        types = torch.arange(4, device=device).unsqueeze(0).expand(B, -1)  # (B, 4)

        token_embs = torch.stack(
            [
                self.day_emb(day),
                self.month_emb(month),
                self.leap_emb(leap),
                self.decade_emb(decade),
            ],
            dim=1,
        ) * math.sqrt(self._d_model)  # (B, 4, d_model)

        type_embs = self.type_emb(types)  # (B, 4, d_model)
        x = token_embs + type_embs
        return self.norm(self.pos_enc(x))


# ─────────────────────────────────────────────────────────────────────────────
# Seq2Seq Transformer
# ─────────────────────────────────────────────────────────────────────────────

class Seq2SeqDateTransformer(nn.Module):
    """
    Encoder-Decoder Transformer for conditional date generation.

    Training  : teacher forcing — decoder receives shifted ground-truth tokens.
    Inference : greedy autoregressive decoding (or top-k sampling).

    Parameters
    ----------
    d_model : int               Transformer hidden size.
    nhead : int                 Number of attention heads.
    num_encoder_layers : int
    num_decoder_layers : int
    dim_feedforward : int       FFN inner dimension.
    dropout : float
    vocab_size : int            Output token vocabulary size (14).
    max_dec_len : int           Maximum decoder sequence length (12).
    label_smoothing : float     Cross-entropy label smoothing ε.
    day_vocab, month_vocab, leap_vocab, decade_vocab : int
    """

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        vocab_size: int = 14,
        max_dec_len: int = 12,
        label_smoothing: float = 0.1,
        day_vocab: int = 7,
        month_vocab: int = 12,
        leap_vocab: int = 2,
        decade_vocab: int = 41,
    ) -> None:
        super().__init__()
        self.d_model     = d_model
        self.vocab_size  = vocab_size
        self.max_dec_len = max_dec_len

        # ── Encoder side ────────────────────────────────────────────────────
        self.cond_encoder = ConditionEncoder(
            d_model, day_vocab, month_vocab, leap_vocab, decade_vocab, dropout
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # Pre-LN: more stable training
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        # ── Decoder side ────────────────────────────────────────────────────
        self.dec_token_emb = nn.Embedding(vocab_size, d_model, padding_idx=11)
        self.dec_pos_emb   = nn.Embedding(max_dec_len + 2, d_model)  # learned
        self.dec_dropout   = nn.Dropout(dropout)
        self.dec_norm      = nn.LayerNorm(d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_decoder_layers
        )

        # ── Output projection ───────────────────────────────────────────────
        self.output_proj = nn.Linear(d_model, vocab_size)

        # ── Loss ────────────────────────────────────────────────────────────
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=11,              # PAD_ID
            label_smoothing=label_smoothing,
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier-uniform initialisation for all linear / embedding weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ── Masks ────────────────────────────────────────────────────────────────

    @staticmethod
    def _causal_mask(sz: int, device: torch.device) -> Tensor:
        """Upper-triangular causal mask (True = ignore) of shape (sz, sz)."""
        return torch.triu(torch.ones(sz, sz, device=device, dtype=torch.bool), diagonal=1)

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        day: Tensor,
        month: Tensor,
        leap: Tensor,
        decade: Tensor,
        tgt: Tensor,
    ) -> Tensor:
        """
        Forward pass with teacher forcing (training mode).

        Parameters
        ----------
        day, month, leap, decade : LongTensor (batch,)
        tgt : LongTensor (batch, tgt_len)
            Decoder input — ground-truth tokens shifted right (SOS prepended).
            The last token (EOS) is excluded from the input; prediction of EOS
            is evaluated from the previous token.

        Returns
        -------
        FloatTensor (batch, tgt_len, vocab_size)   — raw logits
        """
        device = day.device
        # ── Encode conditions ────────────────────────────────────────────────
        enc_in  = self.cond_encoder(day, month, leap, decade)   # (B, 4, d_model)
        memory  = self.transformer_encoder(enc_in)              # (B, 4, d_model)

        # ── Decode ──────────────────────────────────────────────────────────
        T = tgt.size(1)
        positions = torch.arange(T, device=device).unsqueeze(0)  # (1, T)

        dec_emb = (
            self.dec_token_emb(tgt) * math.sqrt(self.d_model)
            + self.dec_pos_emb(positions)
        )
        dec_emb = self.dec_norm(self.dec_dropout(dec_emb))

        causal = self._causal_mask(T, device)
        pad_mask = tgt.eq(11)  # (B, T) — PAD_ID = 11

        dec_out = self.transformer_decoder(
            dec_emb,
            memory,
            tgt_mask=causal,
            tgt_key_padding_mask=pad_mask,
        )
        return self.output_proj(dec_out)  # (B, T, vocab_size)

    def compute_loss(
        self,
        day: Tensor,
        month: Tensor,
        leap: Tensor,
        decade: Tensor,
        target: Tensor,
    ) -> Tensor:
        """
        Compute cross-entropy loss for a training batch.

        Parameters
        ----------
        day, month, leap, decade : LongTensor (batch,)
        target : LongTensor (batch, MAX_OUTPUT_LEN)
            Full padded target sequence including SOS and EOS.

        Returns
        -------
        Scalar FloatTensor
        """
        # Decoder input: all tokens except the last (EOS)
        dec_in  = target[:, :-1]    # (B, T-1)
        # Decoder target: all tokens except the first (SOS)
        dec_tgt = target[:, 1:]     # (B, T-1)

        logits = self.forward(day, month, leap, decade, dec_in)  # (B, T-1, V)
        # Flatten for CrossEntropyLoss
        return self.criterion(
            logits.reshape(-1, self.vocab_size),
            dec_tgt.reshape(-1),
        )

    # ── Inference ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        day: Tensor,
        month: Tensor,
        leap: Tensor,
        decade: Tensor,
        max_len: int | None = None,
        top_k: int = 1,
    ) -> Tensor:
        """
        Autoregressively generate a date token sequence.

        Parameters
        ----------
        day, month, leap, decade : LongTensor (batch,)
        max_len : int, optional   Override maximum generation length.
        top_k : int
            1 → greedy decoding.  >1 → top-k sampling (adds output diversity).

        Returns
        -------
        LongTensor (batch, generated_len)
        """
        SOS_ID, EOS_ID, PAD_ID = 12, 13, 11
        max_len = max_len or self.max_dec_len
        device  = day.device
        B       = day.size(0)

        # Encode conditions once
        enc_in = self.cond_encoder(day, month, leap, decade)
        memory = self.transformer_encoder(enc_in)

        generated = torch.full((B, 1), SOS_ID, dtype=torch.long, device=device)
        done      = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            T = generated.size(1)
            positions = torch.arange(T, device=device).unsqueeze(0)
            dec_emb = (
                self.dec_token_emb(generated) * math.sqrt(self.d_model)
                + self.dec_pos_emb(positions)
            )
            dec_emb = self.dec_norm(dec_emb)
            causal  = self._causal_mask(T, device)
            dec_out = self.transformer_decoder(dec_emb, memory, tgt_mask=causal)
            logits  = self.output_proj(dec_out[:, -1, :])  # (B, V)

            if top_k == 1:
                next_tok = logits.argmax(dim=-1)
            else:
                top_logits, top_ids = logits.topk(top_k, dim=-1)
                probs = F.softmax(top_logits, dim=-1)
                idx   = torch.multinomial(probs, 1).squeeze(-1)
                next_tok = top_ids.gather(1, idx.unsqueeze(1)).squeeze(1)

            # Replace with PAD for sequences that already emitted EOS
            next_tok = torch.where(done, torch.full_like(next_tok, PAD_ID), next_tok)
            generated = torch.cat([generated, next_tok.unsqueeze(1)], dim=1)
            done = done | (next_tok == EOS_ID)
            if done.all():
                break

        return generated  # (B, T)


# ─────────────────────────────────────────────────────────────────────────────
# Noam learning-rate scheduler
# ─────────────────────────────────────────────────────────────────────────────

class NoamScheduler:
    """
    Noam warm-up schedule from "Attention Is All You Need".

    lr = d_model^{-0.5} × min(step^{-0.5}, step × warmup_steps^{-1.5})

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
    d_model : int
    warmup_steps : int
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        d_model: int,
        warmup_steps: int = 400,
    ) -> None:
        self.optimizer     = optimizer
        self.d_model       = d_model
        self.warmup_steps  = warmup_steps
        self._step         = 0

    def step(self) -> float:
        """Advance one step and update the learning rate. Returns current lr."""
        self._step += 1
        lr = (
            self.d_model ** -0.5
            * min(self._step ** -0.5, self._step * self.warmup_steps ** -1.5)
        )
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr