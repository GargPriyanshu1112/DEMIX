import cv2
import math
import pytz
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Optional, List
from dataclasses import dataclass, field
from torchvision.utils import make_grid, save_image

@dataclass
class RoutingStats:
    topk_indices: torch.Tensor
    routing_weights: torch.Tensor
    expert_cap_util: torch.Tensor
    routing_mask: torch.Tensor

@dataclass
class MoEStats:
    aux_loss: torch.Tensor
    z_loss: torch.Tensor
    scale_reg: torch.Tensor
    routing: Optional[RoutingStats] = None

    @staticmethod
    def empty_like(x: torch.Tensor):
        zero = x.new_zeros(())
        return MoEStats(aux_loss=zero, z_loss=zero, scale_reg=zero, routing=None)

@dataclass
class Outputs:
    pred_noise: torch.Tensor
    aux_loss: torch.Tensor
    z_loss: torch.Tensor
    scale_reg: torch.Tensor
    moe_routing_info: List[torch.Tensor] = field(default_factory=list)

def get_device(device_type):
    device_type = device_type.lower()
    if device_type == "cuda":
        assert torch.cuda.is_available(), "CUDA is not available."
        device = torch.device("cuda")
    elif device_type == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    elif device_type == "cpu":
        device = torch.device("cpu")
    else:
        raise ValueError("Only supports 'cuda', 'auto' and 'cpu'.")
    return device

def generate_fwd_process_vizualization(x_0, ts, noise_scheduler, dst, fps=10):
    _, C, H, W = x_0.shape

    out = cv2.VideoWriter(
        dst,
        fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
        fps=fps,
        frameSize=(W, H),
        isColor=(C == 3)
    )

    x_ts = torch.empty((len(ts), H, W, C), dtype=x_0.dtype, device=x_0.device) # pre-allocate
    for idx, t in enumerate(ts):
        x_t = noise_scheduler.add_noise(x_0, t)[0] # [C, H, W]
        x_ts[idx] = x_t.permute(1, 2, 0) # [H, W, C]
    x_ts = ((x_ts.clamp(-1.0, 1.0) + 1.0) / 2.0) * 255.0 # convert tensor from [-1., 1.] to [0., 255.]
    x_ts = x_ts.to(torch.uint8).cpu().numpy()

    for frame in x_ts:
        if frame.shape[2] == 1: # grayscale image
            frame = frame.squeeze(2)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)
    out.release()
    print(f"Saved video to '{dst}'")


def create_grid(imgs, n_row=0, img_path=None):
    n = imgs.shape[0]
    n_row = int(math.ceil(math.sqrt(n))) if n_row==0 else n_row
    grid = make_grid(imgs.float(), nrow=n_row, padding=2, normalize=True)
    if img_path:
        save_image(grid, img_path)
    return grid

def get_ist_time_now(fmt="%d-%m-%Y-%H%M%S"):
    ist = pytz.timezone('Asia/Kolkata')
    now_ist = datetime.now(ist)
    return now_ist.strftime(fmt)

def sample_lbls(n_class, n, device="cpu"):
    return torch.arange(0, n_class, dtype=torch.long, device=device).repeat((n + n_class - 1) // n_class)[:n]

def torch_compile_ckpt_fix(state_dict):
    # Remove '_orig_mod.' prefix added to state_dict keys when saving from a torch.compile-wrapped model.
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    return state_dict

def plot_most_used_expert(experts_used_per_t, save_name="top1-routing.png", save_dir=None, figsize=(14, 4)):
    T, n_experts = experts_used_per_t.shape
    timesteps = np.arange(T, 0, -1)
    top1_expert_per_t = np.argmax(experts_used_per_t, axis=1)

    fig = plt.figure(figsize=figsize)
    for i in range(n_experts):
        mask = (top1_expert_per_t == i)
        if not np.any(mask):
            continue
        x = timesteps[mask]
        y = [i+1] * len(x)
        plt.scatter(x, y, label=f"E{i+1}")

    plt.xlim(T, 0)
    plt.xlabel("t")

    plt.ylabel("top-1 expert")
    plt.yticks(np.arange(1, n_experts + 1))
    plt.ylim(0.5, n_experts + 0.5)

    plt.tight_layout()
    if save_dir:
        plt.savefig(f"{save_dir}/{save_name}", bbox_inches='tight')
    return fig

def plot_expert_usage_heatmap(experts_avg_topk_assignment_per_t, save_name="heatmap.png", save_dir=None, figsize=(14, 4)):
    T, n_experts = experts_avg_topk_assignment_per_t.shape
    experts_avg_topk_assignment_per_t_norm = experts_avg_topk_assignment_per_t / (experts_avg_topk_assignment_per_t.sum(axis=1, keepdims=True) + 1e-8)

    fig = plt.figure(figsize=figsize)
    plt.imshow(experts_avg_topk_assignment_per_t_norm.T, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(label="Fraction of top-k assignments")

    ax = plt.gca()
    ticks = list(range(0, T, 100)) + [T-1]
    labels = [str(max(T - t, 0)) for t in ticks]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)

    plt.xlabel("Timestep (t)  |  high noise → low noise")
    plt.ylabel("Expert")
    plt.yticks(range(n_experts), [f"E{i+1}" for i in range(n_experts)])

    plt.title("Expert Usage Heatmap (Inference)", fontsize=10)
    plt.tight_layout()

    if save_dir:
        plt.savefig(f"{save_dir}/{save_name}", bbox_inches='tight', pad_inches=0.1)
    return fig

def plot_routing_perplexity(experts_avg_topk_rout_load_per_t, save_name="routing_perplexity.png", save_dir=None, figsize=(12, 4)):
    T, n_experts = experts_avg_topk_rout_load_per_t.shape
    experts_avg_topk_rout_load_per_t_norm = experts_avg_topk_rout_load_per_t / (experts_avg_topk_rout_load_per_t.sum(axis=1, keepdims=True) + 1e-8)
    entropy_per_t = -np.sum(experts_avg_topk_rout_load_per_t_norm * np.log(experts_avg_topk_rout_load_per_t_norm + 1e-8), axis=1)
    perplexity_per_t = np.exp(entropy_per_t)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(perplexity_per_t, linewidth=2, color='steelblue')

    ax.axhline(
        y=n_experts,
        color='red',
        linestyle='--',
        linewidth=1,
        label=f"Uniform routing (PP={n_experts})"
    )
    ax.axhline(
        y=1,
        color='gray',
        linestyle='--',
        linewidth=1,
        label="Complete collapse (PP=1)"
    )

    ticks = list(range(0, T, 100)) + [T-1]
    labels = [str(max(T - t, 0)) for t in ticks]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)

    ax.set_ylim(0, n_experts * 1.05)
    ax.set_ylabel("Routing Perplexity", fontsize=10)
    ax.set_xlabel("Timestep (high → low noise)", fontsize=10)
    ax.set_title("Routing Perplexity Across Denoising Stages", fontsize=10)
    ax.legend(fontsize=8)
    plt.tight_layout()

    if save_dir:
        plt.savefig(f"{save_dir}/{save_name}", bbox_inches='tight', pad_inches=0.1)
    return fig