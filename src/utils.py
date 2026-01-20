import cv2
import pytz
import torch
from datetime import datetime
from torchvision.utils import make_grid, save_image

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


def save_grid(imgs, img_path=None, n_row=10):
    n = imgs.shape[0]
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