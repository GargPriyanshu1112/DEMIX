import torch
from tqdm import tqdm

class LinearNoiseScheduler(torch.nn.Module):
    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

        # Pre-calculate coeffs
        betas = torch.linspace(beta_start, beta_end, num_timesteps)
        alphas = 1. - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        sqrt_alpha_bars = torch.sqrt(alpha_bars)
        sqrt_one_minus_alpha_bars = torch.sqrt(1. - alpha_bars)

        # Register buffers
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alpha_bars", sqrt_alpha_bars)
        self.register_buffer("sqrt_one_minus_alpha_bars", sqrt_one_minus_alpha_bars)

    def add_noise(self, x_0, t):
        eps = torch.randn_like(x_0) # [B, C, H, W]
        sqrt_alpha_bar_t = self.sqrt_alpha_bars[t].reshape(-1, 1, 1, 1) # [B, 1, 1, 1]
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bars[t].reshape(-1, 1, 1, 1) # [B, 1, 1, 1]
        x_t = sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * eps
        return x_t, eps

    def sample_prev_timestep(self, t, x_t, pred_eps_t):
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bars[t].reshape(-1, 1, 1, 1)
        beta_t = self.betas[t].reshape(-1, 1, 1, 1)
        alpha_t = self.alphas[t].reshape(-1, 1, 1, 1)

        mean = x_t - ((beta_t * pred_eps_t) / sqrt_one_minus_alpha_bar_t)
        mean /= torch.sqrt(alpha_t)

        if t[0].item() == 0:
            return mean
        else:
            alpha_bar_t = self.alpha_bars[t].reshape(-1, 1, 1, 1)
            alpha_bar_prev = self.alpha_bars[t-1].reshape(-1, 1, 1, 1)
            variance = beta_t * (1 - alpha_bar_prev) / (1 - alpha_bar_t)
            std = torch.sqrt(variance)
            z = torch.randn_like(x_t)
            return mean + std*z


class DDPM:
    def __init__(self, img_size, img_chls, device):
        super().__init__()
        self.img_size = img_size
        self.img_chls = img_chls
        self.device = device
        self.scheduler = LinearNoiseScheduler().to(device)

    def sample_timesteps(self, num_t):
        ts = torch.randint(low=1, high=self.scheduler.num_timesteps, size=(num_t,), device=self.device)
        return ts

    @torch.no_grad()
    def forward(self, x_0, t):
        return self.scheduler.add_noise(x_0, t)

    @torch.no_grad()
    def reverse(self, model, n, amp_ctx, lbls=None, debug=False, debug_steps=10):
        x = torch.randn((n, self.img_chls, *self.img_size), device=self.device)

        debug_steps = min(debug_steps, self.scheduler.num_timesteps)
        debug_stepsize = max(1, self.scheduler.num_timesteps // debug_steps)

        debug_ret = None
        if debug:
            debug_ret = torch.zeros((debug_steps, n, self.img_chls, *self.img_size), dtype=torch.uint8, device="cpu")

        step_idx = 0
        for i in tqdm(reversed(range(self.scheduler.num_timesteps)), dynamic_ncols=True, desc="sampling", leave=False, total=self.scheduler.num_timesteps):
            t = torch.full((n,), fill_value=i, dtype=torch.long, device=self.device)
            with amp_ctx:
                pred_eps_t = model(x, t, lbls)
            x = self.scheduler.sample_prev_timestep(t, x, pred_eps_t)
            if debug and (i%debug_stepsize == 0) and (step_idx < debug_steps):
                x_debug = ((x.clamp(-1., 1.) + 1) * 127.5).to("cpu").to(torch.uint8)
                debug_ret[step_idx] = x_debug
                step_idx += 1

        x = x.clamp(-1., 1.)
        x = ((x + 1) * 127.5).to("cpu").to(torch.uint8)
        return x, debug_ret
