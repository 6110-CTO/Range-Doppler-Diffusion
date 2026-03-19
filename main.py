import os
import sys
import json
import torch
import logging
from types import SimpleNamespace
from torch.utils.data import Subset

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT, "src"))

from cfar import *
from dataset import prep_dataset
from trainer import train_det_epoch, det_validate
from visfuncs import visualize_sample
from inference import run_inference
from models.unet import DetUNet
from models.diffusion import StudentTDiffusion

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load JSON config
def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    return SimpleNamespace(**config_dict)

# Training Loop
def train_model(config, device, run_name="model"):
    logger.info("Preparing dataset...")
    train_loader, val_loader, train_dataset, val_dataset = prep_dataset(config)
    logger.info("Successfully created diffusion dataset.")
    
    # uncomment the next line to visualize a sample
    # visualize_sample(train_dataset, sample_index=1500)
    
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
            save_path = os.path.join(ROOT, f"{run_name}.pth")
            torch.save(cond_diffusion.state_dict(), save_path)
            logger.info(f"   --> Best MSE model saved at {save_path}.")
    
    return cond_diffusion, val_dataset

# Inference
def run_eval(model, val_dataset, checkpoint_path, device):
    if checkpoint_path and not os.path.exists(checkpoint_path):
        logger.warning(f"Checkpoint file not found at {checkpoint_path}")
        logger.info("Skipping inference with checkpoint.")
        return
    
    logger.info("Running inference...")
    run_inference(model, val_dataset, checkpoint_path, device)
    logger.info("Inference completed.")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_path = os.path.join(ROOT, "configs", "base_config.json")
    config = load_config(config_path)
    
    model, val_dataset = train_model(config, device, run_name="dmodel")
    checkpoint_path = "/home/hawk/Desktop/Range-Doppler-Diffusion/det_diff_model.pth"
    run_eval(model, val_dataset, checkpoint_path, device)

if __name__ == "__main__":
    main()