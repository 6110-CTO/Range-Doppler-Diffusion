import os
import math
import logging

import torch
import torch.nn.functional as F

from src.data import prep_dataset
from src.models import DetUNet, StudentTDiffusion

logger = logging.getLogger(__name__)


def train_det_epoch(diffusion, dataloader, optimizer, device,
                    lambda_det=0.01):
    """
    Run one epoch of training for the combined diffusion + detection model.
    """
    diffusion.train()
    total_loss = 0.0
    for batch in dataloader:
        # Unpack batch
        signals_norm, rd_signals_norm, IQs_norm, RDs_norm, \
        clutter_all, gauss_all, labels, scnr_dBs = batch

        # Build x0 (clean RD) and cond (noisy RD)
        signal = rd_signals_norm.to(device)
        cond   = RDs_norm.to(device)

        # Ensure shape (B,2,H,W)
        if signal.real.ndim == 3:
            x0 = torch.cat([signal.real.unsqueeze(1), signal.imag.unsqueeze(1)], dim=1)
            cond = torch.cat([cond.real.unsqueeze(1), cond.imag.unsqueeze(1)], dim=1)
        else:
            x0 = torch.cat([signal.real, signal.imag], dim=1)
            cond = torch.cat([cond.real, cond.imag], dim=1)
        x0   = x0.to(device)
        cond = cond.to(device)

        # Detection labels: (B, H, W) -> add channel dim
        mask = labels.to(device).unsqueeze(1)  # (B,1,H,W)

        # Sample random timesteps
        t = torch.randint(0, diffusion.T, (x0.size(0),), device=device)
        # Add noise
        x_noisy, noise = diffusion.q_sample(x0, t)

        # Normalize timestep and forward
        t_norm = t.float() / diffusion.T
        inp    = torch.cat([x_noisy, cond], dim=1)
        noise_pred, det_pred = diffusion.model(inp, t_norm)

        # Noise prediction loss
        mse_loss = F.mse_loss(noise_pred, noise)
        # Detection loss (per-pixel BCE)
        det_loss = F.binary_cross_entropy_with_logits(det_pred, mask)

        loss = mse_loss + lambda_det * det_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

@torch.no_grad()
def det_validate(diffusion, dataloader, device, lambda_det=0):
    """
    Validate both denoising and detection performance.
    """
    diffusion.eval()
    total_loss = 0.0
    gen_mse, gen_psnr = None, None

    for i, batch in enumerate(dataloader):
        # Unpack batch
        signals_norm, rd_signals_norm, IQs_norm, RDs_norm, \
        clutter_all, gauss_all, labels, scnr_dBs = batch

        # Build x0 (clean RD) and cond (noisy RD)
        signal = rd_signals_norm.to(device)
        cond   = RDs_norm.to(device)

        # Real/imag channels
        if signal.real.ndim == 3:
            x0   = torch.cat([signal.real.unsqueeze(1),
                              signal.imag.unsqueeze(1)], dim=1)
            cond = torch.cat([cond.real.unsqueeze(1),
                              cond.imag.unsqueeze(1)], dim=1)
        else:
            x0   = torch.cat([signal.real, signal.imag], dim=1)
            cond = torch.cat([cond.real,   cond.imag],   dim=1)
        x0   = x0.to(device)
        cond = cond.to(device)

        # Build mask (B,1,H,W)
        mask = labels.to(device).unsqueeze(1)

        # Sample timesteps & noise
        t       = torch.randint(0, diffusion.T, (x0.size(0),), device=device)
        x_noisy, noise = diffusion.q_sample(x0, t)

        # Forward through model
        t_norm       = t.float() / diffusion.T
        model_input  = torch.cat([x_noisy, cond], dim=1)
        noise_pred, det_pred = diffusion.model(model_input, t_norm)

        # Losses
        mse_loss = F.mse_loss(noise_pred, noise)
        det_loss = F.binary_cross_entropy_with_logits(det_pred, mask)
        loss     = mse_loss + lambda_det * det_loss
        total_loss += loss.item()

        # Generation metrics on first batch
        if i == 0:
            generated = diffusion.sample(cond, x0.shape)
            mse_val   = F.mse_loss(generated, x0).item()
            psnr_val  = 20 * math.log10(x0.max().item() / math.sqrt(mse_val)) \
                        if mse_val > 0 else float('inf')
            gen_mse, gen_psnr = mse_val, psnr_val

    avg_loss = total_loss / len(dataloader)
    return avg_loss, gen_mse, gen_psnr


def train_model(config, device, run_name="model"):
    logger.info("Preparing dataset...")
    train_loader, val_loader, train_dataset, val_dataset = prep_dataset(config)
    logger.info("Successfully created diffusion dataset.")

    # Model
    cond_unet = DetUNet(
        in_channels=4,
        out_channels=2,
        time_emb_dim=config.time_emb_dim
    ).to(device)

    cond_diffusion = StudentTDiffusion(
        model=cond_unet,
        scheduler_type=config.scheduler_type,
        T=config.noise_steps,
        beta_start=config.beta_start,
        beta_end=config.beta_end,
    ).to(device)

    optimizer = torch.optim.Adam(cond_diffusion.parameters(), config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10,
        threshold=1e-4, cooldown=15, min_lr=1e-6
    )

    num_epochs = config.num_epochs
    best_mse_loss = float("inf")

    # Determine root for saving checkpoints (project root = parent of src/)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    for epoch in range(num_epochs):
        train_loss = train_det_epoch(cond_diffusion, train_loader, optimizer, device)
        val_loss, gen_mse, gen_psnr = det_validate(cond_diffusion, val_loader, device)

        logger.info(f"Epoch {epoch+1}/{num_epochs}: "
                    f"Train Loss = {train_loss:.4f} | Val Loss = {val_loss:.4f}")

        if gen_mse is not None:
            logger.info(f"   [Generation Metrics] MSE: {gen_mse:.4f} | PSNR: {gen_psnr:.2f} dB")

        scheduler.step(val_loss)

        # Save best model
        if gen_mse is not None and gen_mse < best_mse_loss:
            best_mse_loss = gen_mse
            save_path = os.path.join(root, f"{run_name}.pth")
            torch.save(cond_diffusion.state_dict(), save_path)
            logger.info(f"   --> Best MSE model saved at {save_path}.")

    return cond_diffusion, val_dataset
