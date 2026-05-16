"""
Model 1 — Conditional WGAN-GP for structured date generation.

Architecture overview
---------------------
ConditionEmbedder : 4 × nn.Embedding → MLP → condition vector (cond_dim)
Generator         : noise (latent_dim) + cond → MLP → Gumbel-Softmax logits per position
Discriminator     : date one-hot + cond → MLP (SpectralNorm) → scalar Wasserstein score

Key design choices
------------------
* Gumbel-Softmax (straight-through) enables gradients through discrete token sampling.
* Temperature τ is annealed 1.0 → 0.1 over training to sharpen distributions.
* Spectral Normalisation on the discriminator replaces Lipschitz clipping.
* NO BatchNorm in the discriminator — breaks gradient penalty computation.
  LayerNorm is used in the generator instead.
* WGAN-GP loss with λ=10; critic run n_critic=5 steps per generator step.
* Adam(lr=1e-4, betas=(0.0, 0.9)) — β₁=0 is mandatory for WGAN-GP stability.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils import spectral_norm


# ─────────────────────────────────────────────────────────────────────────────
# Condition Embedder  (shared by Generator and Discriminator)
# ─────────────────────────────────────────────────────────────────────────────

class ConditionEmbedder(nn.Module):
    """
    Embed the four input conditions into a single dense condition vector.

    Each condition (day, month, leap, decade) gets its own nn.Embedding.
    The four embeddings are concatenated and projected via a small MLP.

    Parameters
    ----------
    day_vocab : int     (default 7)
    month_vocab : int   (default 12)
    leap_vocab : int    (default 2)
    decade_vocab : int  (default 41)
    embed_dim : int     Per-condition embedding dimensionality.
    cond_dim : int      Output dimensionality of the condition vector.
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
        Parameters
        ----------
        day, month, leap, decade : LongTensor of shape (batch,)

        Returns
        -------
        Tensor of shape (batch, cond_dim)
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
# Generator
# ─────────────────────────────────────────────────────────────────────────────

class Generator(nn.Module):
    """
    Map noise + condition → soft date token sequence via Gumbel-Softmax.

    Input  : z ~ N(0, I) of shape (batch, latent_dim)
             cond of shape (batch, cond_dim)
    Output : soft token probabilities of shape (batch, seq_len, vocab_size)
             Use argmax over vocab_size for hard token selection at inference.

    Parameters
    ----------
    latent_dim : int    Dimensionality of the noise vector.
    cond_dim : int      Condition vector size (matches ConditionEmbedder output).
    hidden : int        Hidden layer width.
    seq_len : int       Output sequence length (MAX_OUTPUT_LEN = 12).
    vocab_size : int    Output token vocabulary size (14).
    tau_init : float    Initial Gumbel temperature (annealed externally).
    """

    def __init__(
        self,
        latent_dim: int = 100,
        cond_dim: int = 128,
        hidden: int = 256,
        seq_len: int = 12,
        vocab_size: int = 14,
        tau_init: float = 1.0,
    ) -> None:
        super().__init__()
        self.seq_len   = seq_len
        self.vocab_size = vocab_size
        self.tau        = tau_init

        self.net = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, hidden),
            nn.LayerNorm(hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, hidden * 2),
            nn.LayerNorm(hidden * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden * 2, hidden * 2),
            nn.LayerNorm(hidden * 2),
            nn.LeakyReLU(0.2, inplace=True),
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
        FloatTensor (batch, seq_len, vocab_size) — soft Gumbel-Softmax probs.
        """
        x = torch.cat([z, cond], dim=-1)
        logits = self.net(x).view(-1, self.seq_len, self.vocab_size)
        # Gumbel-Softmax: differentiable discrete sampling
        return F.gumbel_softmax(logits, tau=self.tau, hard=False, dim=-1)

    def hard_sample(self, z: Tensor, cond: Tensor) -> Tensor:
        """
        Return hard (argmax) token ids for inference (no gradient needed).

        Returns
        -------
        LongTensor (batch, seq_len)
        """
        soft = self.forward(z, cond)
        return soft.argmax(dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Discriminator
# ─────────────────────────────────────────────────────────────────────────────

class Discriminator(nn.Module):
    """
    Wasserstein critic: scores (real/fake date, condition) pairs.

    Input  : date one-hot FloatTensor (batch, seq_len, vocab_size)
             cond FloatTensor         (batch, cond_dim)
    Output : scalar score FloatTensor (batch, 1)  — NO sigmoid activation.

    Design rules
    ------------
    * All linear layers use SpectralNorm for Lipschitz control.
    * NO BatchNorm (would invalidate the gradient penalty).
    * LeakyReLU activations throughout.
    """

    def __init__(
        self,
        seq_len: int = 12,
        vocab_size: int = 14,
        cond_dim: int = 128,
        hidden: int = 256,
    ) -> None:
        super().__init__()
        flat_dim = seq_len * vocab_size
        self.date_proj = spectral_norm(nn.Linear(flat_dim, hidden))

        self.net = nn.Sequential(
            spectral_norm(nn.Linear(hidden + cond_dim, hidden)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Linear(hidden, hidden // 2)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Linear(hidden // 2, hidden // 4)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Linear(hidden // 4, 1)),
        )

    def forward(self, date_soft: Tensor, cond: Tensor) -> Tensor:
        """
        Parameters
        ----------
        date_soft : FloatTensor (batch, seq_len, vocab_size)
            Real data: one-hot encoded. Fake data: Gumbel-Softmax output.
        cond      : FloatTensor (batch, cond_dim)

        Returns
        -------
        FloatTensor (batch, 1)
        """
        d = F.leaky_relu(self.date_proj(date_soft.view(date_soft.size(0), -1)), 0.2)
        return self.net(torch.cat([d, cond], dim=-1))


# ─────────────────────────────────────────────────────────────────────────────
# WGAN-GP wrapper
# ─────────────────────────────────────────────────────────────────────────────

class ConditionalWGANGP(nn.Module):
    """
    Wrapper that bundles Generator + Discriminator with WGAN-GP loss helpers.

    Typical training loop
    ---------------------
    model = ConditionalWGANGP(...)
    opt_G = Adam(model.generator.parameters(),     lr=1e-4, betas=(0.0, 0.9))
    opt_D = Adam(model.discriminator.parameters(), lr=1e-4, betas=(0.0, 0.9))

    for batch in loader:
        cond = model.embedder(day, month, leap, decade)
        # ----- Critic update (×n_critic) -----
        for _ in range(n_critic):
            z = torch.randn(B, latent_dim)
            d_loss = model.critic_loss(real_onehot, cond, z, lambda_gp=10)
            opt_D.zero_grad(); d_loss.backward(); opt_D.step()
        # ----- Generator update -----
        z = torch.randn(B, latent_dim)
        g_loss = model.generator_loss(cond, z)
        opt_G.zero_grad(); g_loss.backward(); opt_G.step()
        # ----- Anneal Gumbel temperature -----
        model.anneal_temperature()

    Parameters
    ----------
    latent_dim : int
    embed_dim, cond_dim, hidden : int
    seq_len, vocab_size : int
    tau_init, tau_min, tau_decay : float
        Gumbel temperature annealing: τ ← max(τ_min, τ × τ_decay) each step.
    """

    def __init__(
        self,
        latent_dim: int = 100,
        embed_dim: int = 32,
        cond_dim: int = 128,
        hidden: int = 256,
        seq_len: int = 12,
        vocab_size: int = 14,
        day_vocab: int = 7,
        month_vocab: int = 12,
        leap_vocab: int = 2,
        decade_vocab: int = 41,
        tau_init: float = 1.0,
        tau_min: float = 0.1,
        tau_decay: float = 0.9995,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.seq_len    = seq_len
        self.vocab_size = vocab_size
        self.tau_min    = tau_min
        self.tau_decay  = tau_decay

        self.embedder = ConditionEmbedder(
            day_vocab, month_vocab, leap_vocab, decade_vocab, embed_dim, cond_dim
        )
        self.generator = Generator(
            latent_dim, cond_dim, hidden, seq_len, vocab_size, tau_init
        )
        self.discriminator = Discriminator(seq_len, vocab_size, cond_dim, hidden)

    # ── Gumbel temperature annealing ─────────────────────────────────────────

    def anneal_temperature(self) -> None:
        """Decay the Gumbel-Softmax temperature by tau_decay, floored at tau_min."""
        new_tau = max(self.tau_min, self.generator.tau * self.tau_decay)
        self.generator.tau = new_tau

    # ── Gradient penalty ─────────────────────────────────────────────────────

    def _gradient_penalty(
        self,
        real: Tensor,
        fake: Tensor,
        cond: Tensor,
    ) -> Tensor:
        """
        Compute the WGAN-GP gradient penalty on interpolated samples.

        GP = E[(||∇_x̂ D(x̂, c)||₂ − 1)²]  where x̂ = εx_real + (1-ε)x_fake

        Parameters
        ----------
        real, fake : FloatTensor (batch, seq_len, vocab_size)
        cond       : FloatTensor (batch, cond_dim)

        Returns
        -------
        Scalar FloatTensor
        """
        batch = real.size(0)
        alpha = torch.rand(batch, 1, 1, device=real.device)
        interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)

        d_interp = self.discriminator(interp, cond)
        grads = torch.autograd.grad(
            outputs=d_interp,
            inputs=interp,
            grad_outputs=torch.ones_like(d_interp),
            create_graph=True,
            retain_graph=True,
        )[0]
        grads = grads.view(batch, -1)
        return ((grads.norm(2, dim=1) - 1) ** 2).mean()

    # ── Loss helpers ─────────────────────────────────────────────────────────

    def critic_loss(
        self,
        real_onehot: Tensor,
        cond: Tensor,
        z: Tensor,
        lambda_gp: float = 10.0,
    ) -> Tensor:
        """
        Wasserstein critic loss + gradient penalty.

        L_D = E[D(fake)] − E[D(real)] + λ · GP

        Parameters
        ----------
        real_onehot : FloatTensor (batch, seq_len, vocab_size)
        cond        : FloatTensor (batch, cond_dim)
        z           : FloatTensor (batch, latent_dim)
        lambda_gp   : float

        Returns
        -------
        Scalar FloatTensor
        """
        fake = self.generator(z, cond).detach()
        d_real = self.discriminator(real_onehot, cond)
        d_fake = self.discriminator(fake, cond)
        gp = self._gradient_penalty(real_onehot, fake, cond)
        return d_fake.mean() - d_real.mean() + lambda_gp * gp

    def generator_loss(self, cond: Tensor, z: Tensor) -> Tensor:
        """
        Wasserstein generator loss.

        L_G = −E[D(G(z, c))]

        Parameters
        ----------
        cond : FloatTensor (batch, cond_dim)
        z    : FloatTensor (batch, latent_dim)

        Returns
        -------
        Scalar FloatTensor
        """
        fake = self.generator(z, cond)
        return -self.discriminator(fake, cond).mean()

    # ── Inference ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        day: Tensor,
        month: Tensor,
        leap: Tensor,
        decade: Tensor,
        device: torch.device | None = None,
    ) -> Tensor:
        """
        Generate hard token ids for a batch of conditions.

        Parameters
        ----------
        day, month, leap, decade : LongTensor (batch,)
        device : torch.device, optional

        Returns
        -------
        LongTensor (batch, seq_len)
        """
        if device is not None:
            day, month, leap, decade = (
                t.to(device) for t in (day, month, leap, decade)
            )
        cond = self.embedder(day, month, leap, decade)
        z = torch.randn(day.size(0), self.latent_dim, device=cond.device)
        return self.generator.hard_sample(z, cond)