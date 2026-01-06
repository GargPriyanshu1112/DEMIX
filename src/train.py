import os
import torch
import hydra
import wandb
import logging
import torch.nn.functional as F
from tqdm import tqdm
from time import time
from omegaconf import OmegaConf
from dotenv import load_dotenv
from contextlib import nullcontext
from torch.utils.data import DataLoader

from dataset import load_dataset
from model import DiffusionUNetConfig, DiffusionUNet
from noise_scheduler import DDPM
from utils import get_device, save_grid, get_ist_time_now

OmegaConf.register_new_resolver("now_ist", get_ist_time_now)

@hydra.main(config_name="default", config_path="../config", version_base=None)
def main(config):
    logger = logging.getLogger("ddpm")
    device = get_device(config.device_type)
    logger.info(f"Using {device.type.upper()}.")
    log_dir = None

    logger.info(f"Loading {config.dataset.name} dataset.")
    torch_ds = load_dataset(config.dataset, False)
    dataloader = DataLoader(
        torch_ds,
        shuffle=True,
        batch_size=config.batch_size,
        drop_last=config.dataloader.drop_last,
        num_workers=config.dataloader.workers,
        pin_memory=config.dataloader.pin_memory
    )

    logger.info("Loading model.")
    model_config = DiffusionUNetConfig(**config.model)
    if config.init_from == "scratch":
        model = DiffusionUNet(model_config)
        model.to(device)
    else:
        # TODO
        ckpt = torch.load(config.init_from, map_location=device, weights_only=False)

    if config.torch_compile:
        model = torch.compile(model)

    ddpm = DDPM(
        img_size=config.dataset.img_size,
        img_chls=config.dataset.img_chls,
        device=device
    )

    optimizer = hydra.utils.instantiate(config.optimizer, params=model.parameters())
    if config.init_from != "scratch":
        optimizer.load_state_dict(ckpt["optimizer"])

    if config.enable_tf32:
        torch.set_float32_matmul_precision("high")

    if device.type == "cuda" and config.autocast_dtype == "bf16":
        amp_ctx = torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16)
    else:
        amp_ctx = nullcontext()

    if config.logging.wandb.enable:
        load_dotenv()
        _wandb_key = os.getenv("WANDB_API_KEY")
        assert _wandb_key is not None, "Load WANDB_API_KEY in .env"
        wandb.login(_wandb_key)
        wandb.init(
            project=config.logging.wandb.project,
            name=hydra.core.hydra_config.HydraConfig.get().job.name,
            config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
        )
        wandb.define_metric("epoch")
        wandb.define_metric("loss", step_metric="epoch")
        wandb.define_metric("epoch_time", step_metric="epoch")

    model.train()
    for epoch in range(1, config.n_epochs+1):
        logger.info(f"Epoch {epoch}/{config.n_epochs}")

        t0 = time()
        cum_loss = 0.0
        progress_bar = tqdm(dataloader, dynamic_ncols=True, desc=f"Epoch {epoch}", leave=False)

        for step, batch in enumerate(progress_bar):
            imgs, lbls = batch[0].to(device), None

            B, C, H, W = imgs.shape
            t = ddpm.sample_timesteps(B)
            x_t, eps = ddpm.forward(imgs, t)

            optimizer.zero_grad(set_to_none=True)

            with amp_ctx:
                eps_theta = model(x_t, t, lbls)
                loss = F.mse_loss(eps, eps_theta)

            loss.backward()
            optimizer.step()
            cum_loss += loss.item()

            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

            if device.type == "cuda":
                torch.cuda.synchronize()

            epoch_time = time() - t0
            avg_loss = cum_loss / len(dataloader)
            logger.info(f"Loss: {avg_loss:.4f} Time: {epoch_time:.2f}s")

            if config.logging.wandb.enable:
                wandb.log({'epoch': epoch, 'loss': avg_loss, 'epoch_time': epoch_time})

            if (epoch == config.n_epoch) or (epoch%config.vis_every_epoch == 0):
                model.eval()
                with torch.no_grad():
                    lbls = None
                    imgs, _ = ddpm.reverse(model, config.vis_n_samples, amp_ctx, lbls)
                    img_path = log_dir / f"{config.model_name}={epoch:05d}.png"
                    img = save_grid(imgs, img_path, n_row=config.dataset.n_classes).permute(1, 2, 0).numpy()
                    if config.logging.wandb.enable and config.logging.wandb.log_imgs:
                        wandb.log({"samples": wandb.Image(img)}, step=epoch)
            model.train()

            if (epoch == config.n_epoch) or (epoch%config.save_every_epoch == 0):
                ckpt_path = log_dir / f"{config.model_name}.pt"
                torch.save(
                    obj={
                        "model": model.state_dict(),
                        "config": config,
                        "epoch": epoch,
                        "loss": avg_loss,
                        "optimizer": optimizer.state_dict()
                    },
                    f=ckpt_path
                )
                logger.info(f"Saved checkpoint to {str(ckpt_path)}")


if __name__ == "__main__":
    main()