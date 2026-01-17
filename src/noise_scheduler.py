import tqdm
import torch


class LinearNoiseScheduler:
    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

        # Pre-calculate coeffs
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1. - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1. - self.alpha_bars)

    def add_noise(self, x_0, t):
        eps = torch.randn_like(x_0) # [B, C, H, W]
        sqrt_alpha_bar_t = self.sqrt_alpha_bars[t].reshape(-1, 1, 1, 1) # [B, 1, 1, 1]
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bars[t].reshape(-1, 1, 1, 1) # [B, 1, 1, 1]
        x_t = sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * eps
        return x_t, eps

    def sample_prev_timestep(self, t, x_t, pred_eps_t):
        x_0 = (x_t - (self.sqrt_one_minus_alpha_bars[t] * pred_eps_t)) / self.sqrt_alpha_bars[t]
        x_0 = torch.clamp(x_0, -1, 1)

        mean = x_t - ((self.betas[t] * pred_eps_t) / self.sqrt_one_minus_alpha_bars[t])
        mean /= torch.sqrt(self.alphas[t])

        if t == 1:
            return mean
        else:
            variance = (self.betas[t] * (1 - self.alpha_bars[t-1])) / (1 - self.alpha_bars[t])
            std = variance ** 0.5
            z = torch.randn(x_t.shape).to(x_t.device)
            return mean + std*z


class DDPM(LinearNoiseScheduler):
    def __init__(self, img_size, img_chls, device):
        super().__init__()
        self.img_size = img_size
        self.img_chls = img_chls
        self.device = device

    def sample_timesteps(self, num_t):
        ts = torch.randint(low=1, high=self.num_timesteps, size=(num_t,), device=self.device)
        return ts

    @torch.no_grad()
    def forward(self, x_0, t):
        return self.add_noise(x_0, t)

    @torch.no_grad()
    def reverse(self, model, n, amp_ctx, lbls=None, debug=False, debug_steps=10):
        x = torch.randn((n, self.img_chls, *self.img_size), device=self.device)

        debug_steps = min(debug_steps, self.num_timesteps)
        debug_stepsize = max(1, self.num_timesteps // debug_steps)

        debug_ret = None
        if debug:
            debug_ret = torch.zeros((debug_steps, n, self.img_chls, *self.img_size), dtype=torch.uint8, device="cpu")

        step_idx = 0
        for i in tqdm(reversed(range(1, self.num_timesteps)), dynamic_cols=True, desc="sampling", leave=False, total=self.num_timesteps-1):
            t = torch.full((n,), fill_value=i, dtype=torch.long, device=self.device)
            with amp_ctx:
                pred_eps_t = model(x, t, lbls)
            x = self.sample_prev_timestep(t, x, pred_eps_t)
            if debug and (i%debug_stepsize == 0) and (step_idx < debug_steps):
                x_debug = ((x.clamp(-1., 1.) + 1) * 127.5).to("cpu").to(torch.uint8)
                debug_ret[step_idx] = x_debug
                step_idx += 1

        x = x.clamp(-1., 1.)
        x = ((x + 1) * 127.5).to("cpu").to(torch.uint8)
        return x, debug_ret
