import os
import torch
import hydra
import wandb
import random
import logging
import torch.nn.functional as F
from tqdm import tqdm
from time import time
from pathlib import Path
from omegaconf import OmegaConf
from dotenv import load_dotenv
from contextlib import nullcontext
from torch.utils.data import DataLoader

from dataset import load_dataset
from model import DiffusionUNetConfig, DiffusionUNet
from moe import MoEConfig
from noise_scheduler import DDPM
from utils import (
    get_device,
    create_grid,
    get_ist_time_now,
    sample_lbls,
    torch_compile_ckpt_fix,
    plot_most_used_expert,
    plot_expert_usage_heatmap,
    plot_routing_perplexity
)

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

    if not config.ddpm.classifier_free_guidance.enable:
        config.model.n_classes = 0
    model_config = DiffusionUNetConfig(**config.model)
    moe_config = MoEConfig(**config.moe)

    logger.info("Loading model.")
    start_epoch = 1
    if config.init_from == "scratch":
        model = DiffusionUNet(config.dataset.img_size[0], model_config, moe_config)
        model.to(device)
    else:
        ckpt = torch.load(config.init_from, map_location=device, weights_only=False)
        ckpt_cfg = ckpt['config']
        model_config = DiffusionUNetConfig(**ckpt_cfg.model)
        moe_config = MoEConfig(**ckpt_cfg.moe)
        assert config.ddpm.timesteps == ckpt_cfg.ddpm.timesteps, f"Different timesteps: ckpt timesteps {ckpt_cfg.ddpm.timesteps}"
        assert model_config.n_classes > 0 if config.ddpm.classifier_free_guidance.enable else model_config.n_classes == 0, f"Incompatible model {config.ddpm.classifier_free_guidance.enable=} {model_config.n_classes=}"
        assert config.dataset.name == ckpt_cfg.dataset.name, f"Different dataset: {ckpt_cfg.dataset.name}"
        model = DiffusionUNet(config.dataset.img_size[0], model_config, moe_config)
        model.to(device)
        model.load_state_dict(torch_compile_ckpt_fix(ckpt['model']))
        logger.info(f"Loaded checkpoint from {config.init_from}")
        start_epoch = ckpt['epoch'] + 1

    if config.torch_compile:
        model = torch.compile(model)

    ddpm = DDPM(
        config,
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
        wandb.define_metric("metrics/*", step_metric="epoch")
        wandb.define_metric("norms/*", step_metric="epoch")

    model.train()
    p_keep = 1. - config.ddpm.classifier_free_guidance.p_drop # prob at which labels are retained during cfg enabled training
    for epoch in range(start_epoch, config.n_epochs+1):
        logger.info(f"Epoch {epoch}/{config.n_epochs}")

        epoch_router_norm = 0.0
        epoch_expert_norm = 0.0
        epoch_total_norm  = 0.0

        t0 = time()
        cum_loss = 0.0
        progress_bar = tqdm(dataloader, dynamic_ncols=True, desc=f"Epoch {epoch}", leave=True)

        for step, batch in enumerate(progress_bar):
            imgs, lbls = batch[0].to(device, non_blocking=True), None
            if config.ddpm.classifier_free_guidance.enable and random.random() < p_keep:
                lbls = batch[1].to(device, non_blocking=True)

            B, C, H, W = imgs.shape
            t = ddpm.sample_timesteps(B)
            x_t, noise = ddpm.forward(imgs, t)

            optimizer.zero_grad(set_to_none=True)

            with amp_ctx:
                outputs = model(x_t, t, lbls)
                mse_term = F.mse_loss(outputs.pred_noise, noise)
                aux_term = moe_config.lambda_aux * outputs.aux_loss
                z_term   = moe_config.lambda_z * outputs.z_loss
                total_loss = mse_term + aux_term + z_term

            total_loss.backward()

            if moe_config.enable:
                named_params = list(model.named_parameters())
                residual_scale = [p for n, p in named_params if 'residual_scale' in n and p.grad is not None]
                router_params  = [p for n, p in named_params if 'experts_proj' in n and p.grad is not None]
                expert_params  = [p for n, p in named_params if ('fc1' in n or 'fc2' in n) and p.grad is not None]

                router_norm = torch.nn.utils.get_total_norm(router_params)
                expert_norm = torch.nn.utils.get_total_norm(expert_params)

                epoch_router_norm += router_norm.item()
                epoch_expert_norm += expert_norm.item()

            total_norm  = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            epoch_total_norm += total_norm.item()

            optimizer.step()
            cum_loss += total_loss.item()

            postfix = {
                'mse': f"{mse_term.item():.4f}",
                'scaled_aux': f"{aux_term.item():.4f}",
                'scaled_z': f"{z_term.item():.4f}",
                'total_norm': f"{total_norm.item():.4f}",
            }
            if moe_config.enable:
                postfix['router_norm'] = f"{router_norm:.4f}"
                postfix['expert_norm'] = f"{expert_norm:.4f}"
                postfix['scale'] = f"{residual_scale[0].item():.4f}" if residual_scale else "N/A"
            progress_bar.set_postfix(postfix)

        if device.type == "cuda":
            torch.cuda.synchronize()

        epoch_time = time() - t0

        avg_router_norm = epoch_router_norm / len(dataloader)
        avg_expert_norm = epoch_expert_norm / len(dataloader)
        avg_total_norm  = epoch_total_norm  / len(dataloader)
        avg_loss = cum_loss / len(dataloader)

        if moe_config.enable:
            logger.info(
                f"[Epoch {epoch}] "
                f"Loss={avg_loss:.4f}, "
                f"Time={epoch_time:.2f}s "
                f"router_norm={avg_router_norm:.4f}, "
                f"expert_norm={avg_expert_norm:.4f}, "
                f"total_norm={avg_total_norm:.4f}"
            )
        else:
            logger.info(
                f"[Epoch {epoch}] "
                f"Loss={avg_loss:.4f}, "
                f"Time={epoch_time:.2f}s "
                f"total_norm={avg_total_norm:.4f}"
            )

        if config.logging.wandb.enable:
            log_dict = {
                "epoch": epoch,
                "norms/total": avg_total_norm,
                "metrics/loss": avg_loss,
                "metrics/time": epoch_time,
            }
            if moe_config.enable:
                log_dict.update({
                    "norms/router": avg_router_norm,
                    "norms/expert": avg_expert_norm,
                })
            wandb.log(log_dict)

        if (epoch == config.n_epochs) or (epoch%config.vis_every_epoch == 0):
            model.eval()
            with torch.no_grad():
                lbls = None
                if config.ddpm.classifier_free_guidance.enable:
                    lbls = sample_lbls(config.dataset.n_classes, config.vis_n_samples, device)
                imgs, experts_used_per_t, routing_weights_per_t, _ = ddpm.reverse(model, config.vis_n_samples, amp_ctx, lbls)
                img_path = log_dir / f"{config.model_name}-{epoch:05d}.png"
                img = create_grid(imgs, n_row=config.dataset.n_classes, img_path=img_path).permute(1, 2, 0).numpy()
                most_used_expert_per_t_fig = plot_most_used_expert(
                    experts_used_per_t,
                    save_name=f"{config.model_name}-{epoch:05d}_top1_expert.png",
                    save_dir=log_dir
                )
                heatmap_fig = plot_expert_usage_heatmap(
                    routing_weights_per_t,
                    save_name=f"{config.model_name}-{epoch:05d}_heatmap.png",
                    save_dir=log_dir
                )
                routing_entropy_fig = plot_routing_perplexity(
                    routing_weights_per_t,
                    save_name=f"{config.model_name}-{epoch:05d}_entropy.png",
                    save_dir=log_dir
                )
                logger.info(f"Saved sample images generated to {str(img_path)}")
                if config.logging.wandb.enable and config.logging.wandb.log_imgs:
                    wandb.log({
                        "samples": wandb.Image(img),
                        "most_used_expert": wandb.Image(most_used_expert_per_t_fig),
                        "expert_usage_heatmap": wandb.Image(heatmap_fig),
                        "routing_entropy": wandb.Image(routing_entropy_fig),
                    }, step=epoch)
        model.train()

        if (epoch == config.n_epochs) or (epoch%config.save_every_epoch == 0):
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