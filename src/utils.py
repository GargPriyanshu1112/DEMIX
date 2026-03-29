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
    expert_probs: torch.Tensor
    expert_cap_util: torch.Tensor
    expert_mask: torch.Tensor

@dataclass
class MoEStats:
    aux_loss: torch.Tensor
    z_loss: torch.Tensor
    routing: Optional[RoutingStats] = None

    @staticmethod
    def empty_like(x: torch.Tensor):
        zero = x.new_zeros(())
        return MoEStats(aux_loss=zero, z_loss=zero, routing=None)

@dataclass
class Outputs:
    pred_noise: torch.Tensor
    aux_loss: torch.Tensor
    z_loss: torch.Tensor
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

def plot_most_used_expert(experts_used_per_t, save_path=None, figsize=(14, 4)):
    T, n_experts = experts_used_per_t.shape
    timesteps = np.arange(T) + 1
    top1_expert_per_t = np.argmax(experts_used_per_t, axis=1)

    fig = plt.figure(figsize=figsize)
    for i in range(n_experts):
        mask = (top1_expert_per_t == i)
        if not np.any(mask):
            continue
        x = timesteps[mask]
        y = [i+1] * len(x)
        plt.scatter(x, y, label=f"Expert_{i+1}")

    plt.gca().invert_xaxis()
    plt.xlabel("t")

    plt.ylabel("top-1 expert")
    plt.yticks(np.arange(1, n_experts + 1))
    plt.ylim(0.5, n_experts + 0.5)

    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    return fig

def plot_expert_usage_heatmap(routing_weights_per_t, save_path=None, figsize=(14, 4)):
    n_experts = routing_weights_per_t.shape[1]
    normalized = routing_weights_per_t / (routing_weights_per_t.sum(axis=1, keepdims=True) + 1e-8)

    fig = plt.figure(figsize=figsize)
    plt.imshow(normalized.T, aspect='auto', origin='upper', cmap='viridis')
    plt.colorbar(label="Fraction of tokens")

    plt.xlabel(f"Timestep (high→low noise)")
    plt.ylabel("Expert")
    plt.yticks(range(n_experts), [f"E{i+1}" for i in range(n_experts)])

    plt.title("Expert Usage Heatmap (Inference)", fontsize=10)
    plt.gca().invert_xaxis()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    return fig

def plot_routing_entropy(routing_weights_per_t, save_path=None, figsize=(14, 4)):
    """
    Validates: At which denoising stage does specialization emerge?
    """
    T, n_experts = routing_weights_per_t.shape
    normalized = routing_weights_per_t / (routing_weights_per_t.sum(axis=1, keepdims=True) + 1e-8)
    entropy_per_t = -np.sum(normalized * np.log(normalized + 1e-8), axis=1)
    max_entropy = np.log(n_experts) # uniform routing

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(entropy_per_t[::-1], linewidth=2, color='steelblue', label="Routing entropy")

    # Max entropy reference line
    ax.axhline(
        y=max_entropy,
        color='red',
        linestyle='--',
        linewidth=1,
        label=f"Max entropy = {max_entropy:.2f} (uniform routing)"
    )

    # Shade regions: high noise (early), low noise (late)
    ax.axvspan(T - T//3, T, alpha=0.08, color='red', label="High noise")
    ax.axvspan(0, T - 2*T//3, alpha=0.08, color='green', label="Low noise")

    ax.set_xlabel("Timestep (high → low noise)")
    ax.set_ylabel("Routing Entropy")
    ax.set_ylim(0, max_entropy * 1.1)
    ax.invert_xaxis()

    ax.set_title(
        "Routing Entropy Across Denoising Stages\n"
        "High = uniform routing  |  Low = specialized routing",
        fontsize=10
    )

    ax.legend(fontsize=9)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    return fig
