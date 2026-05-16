"""
Model 4 — Conditional Variational Autoencoder (CVAE) for date generation.

Architecture overview
---------------------
Encoder : (condition_vector ⊕ date_embedding) → μ, log σ²
Decoder : (z sampled via reparameterisation ⊕ condition_vector) → per-position logits

Loss (ELBO)
-----------
    ELBO = Reconstruction Loss + β · KL-divergence
    Reconstruction = CrossEntropy summed over all output positions
    KL = −½ Σ (1 + log σ² − μ² − σ²)
    β is annealed linearly from 0 → 1 over kl_anneal_epochs to prevent
    posterior collapse (the decoder ignoring z).

Key design choices
------------------
* Date embedding for the encoder: mean-pooling of per-token embeddings
  (bag-of-tokens) — simple and effective for short sequences.
* Reparameterisation trick: z = μ + ε · exp(0.5 · log σ²), ε ~ N(0, I).
* Decoder outputs independent logits per position → argmax for token selection.
* Unlike GANs, CVAE training is stable (single objective, no adversarial game).
* The latent space allows diversity at inference time by sampling different z.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ─────────────────────────────────────────────────────────────────────────────
# Condition Embedder  (same design as in WGAN-GP for consistency)
# ─────────────────────────────────────────────────────────────────────────────

class ConditionEmbedder(nn.Module):
    """
    Project the four input conditions into a single dense condition vector.

    Parameters
    ----------
    day_vocab, month_vocab, leap_vocab, decade_vocab : int
    embed_dim : int   Per-condition embedding size.
    cond_dim : int    Output condition vector size.
    """

    def __init__(
        self,
        day_vocab: int = 7,
        month_vocab: int = 12,
        leap_vocab: int = 2,
        decade_vocab: int = 41,
        embed_dim: int = 32,
        cond_dim: int = 128,
    ) -> None:
        super().__init__()
        self.day_emb    = nn.Embedding(day_vocab,    embed_dim)
        self.month_emb  = nn.Embedding(month_vocab,  embed_dim)
        self.leap_emb   = nn.Embedding(leap_vocab,   embed_dim)
        self.decade_emb = nn.Embedding(decade_vocab, embed_dim)

        self.proj = nn.Sequential(
            nn.Linear(4 * embed_dim, cond_dim),
            nn.LayerNorm(cond_dim),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        day: Tensor,
        month: Tensor,
        leap: Tensor,
        decade: Tensor,
    ) -> Tensor:
        """
        Returns
        -------
        FloatTensor (batch, cond_dim)
        """
        x = torch.cat(
            [
                self.day_emb(day),
                self.month_emb(month),
                self.leap_emb(leap),
                self.decade_emb(decade),
            ],
            dim=-1,
        )
        return self.proj(x)


# ─────────────────────────────────────────────────────────────────────────────
# Encoder
# ─────────────────────────────────────────────────────────────────────────────

class CVAEEncoder(nn.Module):
    """
    Encode a (condition, date) pair into the latent Gaussian parameters μ and log σ².

    Date tokens are embedded and mean-pooled into a single vector.
    The pooled date embedding is concatenated with the condition vector and
    passed through an MLP to produce μ and log σ².

    Parameters
    ----------
    vocab_size : int     Output token vocabulary size (14).
    tok_embed_dim : int  Dimensionality of each token embedding.
    cond_dim : int       Condition vector size.
    hidden : int         MLP hidden size.
    latent_dim : int     Dimensionality of the latent variable z.
    """

    def __init__(
        self,
        vocab_size: int = 14,
        tok_embed_dim: int = 32,
        cond_dim: int = 128,
        hidden: int = 256,
        latent_dim: int = 64,
    ) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, tok_embed_dim, padding_idx=11)

        self.net = nn.Sequential(
            nn.Linear(tok_embed_dim + cond_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
        )
        self.mu_head      = nn.Linear(hidden, latent_dim)
        self.logvar_head  = nn.Linear(hidden, latent_dim)

    def forward(self, tokens: Tensor, cond: Tensor) -> tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        tokens : LongTensor (batch, seq_len)   — padded date token ids
        cond   : FloatTensor (batch, cond_dim)

        Returns
        -------
        mu     : FloatTensor (batch, latent_dim)
        logvar : FloatTensor (batch, latent_dim)
        """
        # Mask out PAD tokens before mean-pooling
        pad_mask = tokens.ne(11).float().unsqueeze(-1)          # (B, T, 1)
        emb      = self.tok_emb(tokens) * pad_mask              # (B, T, E)
        pooled   = emb.sum(dim=1) / pad_mask.sum(dim=1).clamp(min=1)  # (B, E)

        h = self.net(torch.cat([pooled, cond], dim=-1))
        return self.mu_head(h), self.logvar_head(h)


# ─────────────────────────────────────────────────────────────────────────────
# Decoder
# ─────────────────────────────────────────────────────────────────────────────

class CVAEDecoder(nn.Module):
    """
    Decode latent vector z + condition into per-position token logits.

    Output: independent logit distributions for each position in the
    date sequence (non-autoregressive).

    Parameters
    ----------
    latent_dim : int
    cond_dim : int
    hidden : int
    seq_len : int      Output sequence length (MAX_OUTPUT_LEN = 12).
    vocab_size : int   Output token vocabulary size (14).
    """

    def __init__(
        self,
        latent_dim: int = 64,
        cond_dim: int = 128,
        hidden: int = 256,
        seq_len: int = 12,
        vocab_size: int = 14,
    ) -> None:
        super().__init__()
        self.seq_len   = seq_len
        self.vocab_size = vocab_size

        self.net = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden * 2),
            nn.LayerNorm(hidden * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden * 2, hidden * 2),
            nn.LayerNorm(hidden * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden * 2, seq_len * vocab_size),
        )

    def forward(self, z: Tensor, cond: Tensor) -> Tensor:
        """
        Parameters
        ----------
        z    : FloatTensor (batch, latent_dim)
        cond : FloatTensor (batch, cond_dim)

        Returns
        -------
        FloatTensor (batch, seq_len, vocab_size)  — raw logits per position
        """
        return self.net(torch.cat([z, cond], dim=-1)).view(
            -1, self.seq_len, self.vocab_size
        )


# ─────────────────────────────────────────────────────────────────────────────
# Conditional VAE
# ─────────────────────────────────────────────────────────────────────────────

class ConditionalVAE(nn.Module):
    """
    Full Conditional VAE: ConditionEmbedder + CVAEEncoder + CVAEDecoder.

    Training
    --------
    Compute ELBO loss via compute_loss().
    Use β-annealing: start with beta=0, linearly increase to beta_max
    over kl_anneal_epochs.  Set beta externally each epoch.

    Inference
    ---------
    Sample z ~ N(0, I) and decode to get diverse valid dates.
    Or pass through encoder to get the posterior z for a known date (reconstruction).

    Parameters
    ----------
    latent_dim : int
    embed_dim : int      Per-condition embedding size.
    cond_dim : int       Condition vector size.
    tok_embed_dim : int  Encoder date-token embedding size.
    hidden : int         MLP hidden size for encoder and decoder.
    seq_len : int        Output sequence length.
    vocab_size : int     Output vocabulary size.
    beta : float         KL weight in ELBO (updated externally for annealing).
    day_vocab, month_vocab, leap_vocab, decade_vocab : int
    """

    def __init__(
        self,
        latent_dim: int = 64,
        embed_dim: int = 32,
        cond_dim: int = 128,
        tok_embed_dim: int = 32,
        hidden: int = 256,
        seq_len: int = 12,
        vocab_size: int = 14,
        beta: float = 1.0,
        day_vocab: int = 7,
        month_vocab: int = 12,
        leap_vocab: int = 2,
        decade_vocab: int = 41,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.seq_len    = seq_len
        self.vocab_size = vocab_size
        self.beta       = beta   # set externally each epoch for KL annealing

        self.cond_embedder = ConditionEmbedder(
            day_vocab, month_vocab, leap_vocab, decade_vocab, embed_dim, cond_dim
        )
        self.encoder = CVAEEncoder(vocab_size, tok_embed_dim, cond_dim, hidden, latent_dim)
        self.decoder = CVAEDecoder(latent_dim, cond_dim, hidden, seq_len, vocab_size)

    # ── Reparameterisation ───────────────────────────────────────────────────

    def reparameterise(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Sample z via the reparameterisation trick.

        z = μ + ε · exp(0.5 · log σ²),  ε ~ N(0, I)

        During inference (eval mode) simply return μ for deterministic output,
        or call with a random ε for diverse sampling.

        Parameters
        ----------
        mu     : FloatTensor (batch, latent_dim)
        logvar : FloatTensor (batch, latent_dim)

        Returns
        -------
        FloatTensor (batch, latent_dim)
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu  # deterministic at eval time

    # ── ELBO loss ────────────────────────────────────────────────────────────

    def compute_loss(
        self,
        day: Tensor,
        month: Tensor,
        leap: Tensor,
        decade: Tensor,
        target: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Compute the ELBO loss = reconstruction + β · KL.

        Parameters
        ----------
        day, month, leap, decade : LongTensor (batch,)
        target : LongTensor (batch, seq_len)   — padded date token ids (0-13)

        Returns
        -------
        total_loss : Scalar FloatTensor
        recon_loss : Scalar FloatTensor   (for logging)
        kl_loss    : Scalar FloatTensor   (for logging)
        """
        cond   = self.cond_embedder(day, month, leap, decade)       # (B, cond_dim)
        mu, logvar = self.encoder(target, cond)                      # (B, latent_dim)
        z      = self.reparameterise(mu, logvar)                    # (B, latent_dim)
        logits = self.decoder(z, cond)                              # (B, seq_len, V)

        # Reconstruction: cross-entropy over all positions, ignoring PAD (11)
        recon_loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            target.reshape(-1),
            ignore_index=11,
            reduction="mean",
        )

        # KL divergence: −½ Σ (1 + log σ² − μ² − σ²)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        total_loss = recon_loss + self.beta * kl_loss
        return total_loss, recon_loss, kl_loss

    # ── Inference ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        day: Tensor,
        month: Tensor,
        leap: Tensor,
        decade: Tensor,
        sample: bool = True,
    ) -> Tensor:
        """
        Generate hard token ids from sampled or prior z.

        Parameters
        ----------
        day, month, leap, decade : LongTensor (batch,)
        sample : bool
            True  → sample z ~ N(0, I) for diverse outputs.
            False → use z = 0 (prior mean) for deterministic output.

        Returns
        -------
        LongTensor (batch, seq_len)
        """
        cond = self.cond_embedder(day, month, leap, decade)
        if sample:
            z = torch.randn(day.size(0), self.latent_dim, device=cond.device)
        else:
            z = torch.zeros(day.size(0), self.latent_dim, device=cond.device)
        logits = self.decoder(z, cond)          # (B, seq_len, V)
        return logits.argmax(dim=-1)            # (B, seq_len)