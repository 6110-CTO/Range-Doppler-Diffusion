# src/ddim_sampler.py
import torch
import numpy as np

class DDIMSampler:
    """
    DDIM sampler (Song et al., 2020) — deterministic, ~20x faster than DDPM.
    Works with any trained DDPM model without retraining.

    Args:
        model:      your trained UNet / denoising model
        betas:      1D tensor of betas used during training (length T)
        device:     torch.device
    """

    def __init__(self, model, betas: torch.Tensor, device: torch.device):
        self.model = model
        self.device = device
        T = len(betas)

        # Precompute the same noise schedule your DDPM training used
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)          # ᾱ_t

        self.T = T
        self.alphas_cumprod = alphas_cumprod.to(device)        # shape [T]

    @torch.no_grad()
    def sample(
        self,
        shape,                  # e.g. (batch, channels, H, W) or (batch, seq_len)
        num_steps: int = 50,    # how many DDIM steps (vs 1000 for DDPM)
        eta: float = 0.0,       # 0 = fully deterministic DDIM; 1 = recovers DDPM noise
        condition=None,         # optional conditioning input (clutter map, etc.)
        clip_denoised: bool = False,
    ):
        """
        Run the DDIM reverse process.

        Returns:
            x0_pred: tensor of shape `shape`, the denoised output
        """
        device = self.device

        # Choose a subsequence of timesteps (uniformly spaced, descending)
        timesteps = torch.linspace(self.T - 1, 0, num_steps + 1).long()
        # e.g. for T=1000, num_steps=50: [999, 979, 959, ..., 19, 0]

        # Start from pure noise
        x = torch.randn(shape, device=device)

        for i in range(len(timesteps) - 1):
            t_curr = timesteps[i]
            t_next = timesteps[i + 1]

            # Broadcast scalar timestep to batch dimension
            t_batch = torch.full((shape[0],), t_curr, device=device, dtype=torch.long)

            # --- Get ᾱ values ---
            alpha_bar_t    = self.alphas_cumprod[t_curr]
            alpha_bar_t_next = self.alphas_cumprod[t_next] if t_next >= 0 else torch.tensor(1.0, device=device)

            # --- Predict noise with your model ---
            if condition is not None:
                eps_pred = self.model(x, t_batch, condition)
            else:
                eps_pred = self.model(x, t_batch)

            # --- Predict x0 from current x and predicted noise ---
            x0_pred = (x - (1 - alpha_bar_t).sqrt() * eps_pred) / alpha_bar_t.sqrt()

            if clip_denoised:
                x0_pred = x0_pred.clamp(-1, 1)

            # --- DDIM update step ---
            # sigma controls stochasticity: 0 → deterministic DDIM, >0 → adds noise
            sigma = eta * (
                (1 - alpha_bar_t_next) / (1 - alpha_bar_t) *
                (1 - alpha_bar_t / alpha_bar_t_next)
            ).sqrt()

            # Direction pointing to x_t
            dir_xt = (1 - alpha_bar_t_next - sigma**2).sqrt() * eps_pred

            # Optional noise injection
            noise = sigma * torch.randn_like(x) if eta > 0 else 0.0

            # New x at next (earlier) timestep
            x = alpha_bar_t_next.sqrt() * x0_pred + dir_xt + noise

        return x