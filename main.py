import os
import torch
import logging

from src.config import load_config
from src.train import train_model
from src.inference import run_inference

ROOT = os.path.dirname(os.path.abspath(__file__))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


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
    config = load_config(os.path.join(ROOT, "configs", "base_config.json"))

    model, val_dataset = train_model(config, device, run_name="dmodel")

    checkpoint_path = os.path.join(ROOT, "dmodel.pth")
    run_eval(model, val_dataset, checkpoint_path, device)


if __name__ == "__main__":
    main()
