import torch
import matplotlib.pyplot as plt
from pathlib import Path

from noise_scheduler import DDPM
from model import DiffusionUNetConfig, DiffusionUNet
from utils import get_device, torch_compile_ckpt_fix, sample_lbls, create_grid

def sample(ckpt_path, steps=-1, n=20, save_path=None):
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    device = get_device(device_type)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ckpt_cfg = ckpt['config']
    model_config = DiffusionUNetConfig(**ckpt_cfg.model)
    model = DiffusionUNet(model_config)
    model.to(device)
    model.load_state_dict(torch_compile_ckpt_fix(ckpt['model']))
    print(f"Loaded checkpoint from '{ckpt_path}'")

    dataset_name = ckpt_cfg.dataset.name
    img_size, img_chls = ckpt_cfg.dataset.img_size, ckpt_cfg.dataset.img_chls
    n_class = ckpt_cfg.model.n_classes
    is_classifier_free_guidance_enabled = ckpt_cfg.ddpm.classifier_free_guidance.enable
    schedule = ckpt_cfg.ddpm.schedule

    if steps != -1:
        ckpt_cfg.ddpm.timesteps = steps
    ddpm = DDPM(ckpt_cfg, img_size=img_size, img_chls=img_chls, device=device)

    lbls = (
        sample_lbls(n_class, n, device)
        if n_class > 0 else
        None
    )
    imgs, _ = ddpm.reverse(model, n, lbls=lbls)

    grid = create_grid(imgs, n_row=n_class).to("cpu").permute(1, 2, 0).numpy()
    plt.figure(facecolor='black')
    plt.imshow(grid)
    plt.axis("off")
    plt.tight_layout()
    title = f"{dataset_name} ({schedule=})"
    if is_classifier_free_guidance_enabled:
        title += " with CFG"
    plt.title(title, color='white')
    if save_path.exists:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.show()


if __name__ == "__main__":
    sample(r"./models/cifar10_ddpm_600epochs.pt", steps=1000, n=100, save_path=Path("./outputs.png"))