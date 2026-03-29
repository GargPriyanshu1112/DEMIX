import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from contextlib import nullcontext

class NoiseScheduler(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_timesteps = cfg.ddpm.timesteps
        self.beta_start = cfg.ddpm.beta_start
        self.beta_end = cfg.ddpm.beta_end
        self.schedule = cfg.ddpm.schedule

        # Pre-calculate coeffs
        betas = {"linear": self._linear_schedule(), "cosine": self._cosine_schedule()}[self.schedule]
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

    def _linear_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.num_timesteps)

    def _cosine_schedule(self):
        T = self.num_timesteps
        t = torch.linspace(0, T, T+1)
        s = 0.008
        f_ts = torch.square(torch.cos((((t/T) + s) / (1+s)) * torch.pi/2))
        alpha_bars = f_ts / f_ts[0]
        betas = 1 - (alpha_bars[1:] / alpha_bars[:-1])
        betas = betas.clip(self.beta_start, self.beta_end)
        return betas

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
    def __init__(self, cfg, img_size, img_chls, device):
        super().__init__()
        self.img_size = img_size
        self.img_chls = img_chls
        self.device = device
        self.scheduler = NoiseScheduler(cfg).to(device)
        self.enable_cfg = cfg.ddpm.classifier_free_guidance.enable
        self.scale_cfg = cfg.ddpm.classifier_free_guidance.scale
        self.enable_moe = cfg.moe.enable
        self.n_experts = cfg.moe.n_experts if self.enable_moe else None

    def sample_timesteps(self, num_t):
        ts = torch.randint(low=1, high=self.scheduler.num_timesteps, size=(num_t,), device=self.device)
        return ts

    @torch.no_grad()
    def forward(self, x_0, t):
        return self.scheduler.add_noise(x_0, t)

    @torch.no_grad()
    def reverse(self, model, n, amp_ctx=nullcontext(), lbls=None, debug=False, debug_steps=10):
        assert lbls is not None if self.enable_cfg else lbls is None
        x = torch.randn((n, self.img_chls, *self.img_size), device=self.device)

        debug_steps = min(debug_steps, self.scheduler.num_timesteps)
        debug_stepsize = max(1, self.scheduler.num_timesteps // debug_steps)

        debug_ret = None
        if debug:
            debug_ret = torch.zeros((debug_steps, n, self.img_chls, *self.img_size), dtype=torch.uint8, device="cpu")

        step_idx = 0
        routing_weights_per_t = []
        experts_used_per_t = []
        for i in tqdm(reversed(range(self.scheduler.num_timesteps)), dynamic_ncols=True, desc="sampling", leave=False, total=self.scheduler.num_timesteps):
            t = torch.full((n,), fill_value=i, dtype=torch.long, device=self.device)
            with amp_ctx:
                outputs = model(x, t, lbls)
                pred_noise = outputs.pred_noise
                if self.enable_cfg:
                    uncond_outputs = model(x, t, None)
                    uncond_pred_noise = uncond_outputs.pred_noise
                    pred_noise = torch.lerp(uncond_pred_noise, pred_noise, self.scale_cfg) # (1−s)*ϵ_uncond + s*ϵ_cond​

            if self.enable_moe:
                routing = outputs.moes_routing_info[0]
                expert_probs = routing.expert_probs # [B*T, n_experts, expert_capacity], one capacity slot per (token, expert) pair

                topk_indices = routing.topk_indices # [B, T, K]
                B, T = topk_indices.shape[0],topk_indices.shape[1]

                top1 = topk_indices[:, :, 0] # [B, T]
                one_hot = F.one_hot(top1, num_classes=self.n_experts) # [B, T, n_experts]
                one_hot = one_hot.sum(dim=(0, 1)) # [n_experts]
                experts_used_per_t.append(one_hot)

                token_exp_probs = expert_probs.sum(dim=2) # [B*T, n_experts]
                token_exp_probs = token_exp_probs.reshape(B, T, self.n_experts) # [B, T, n_experts]
                routing_weight = token_exp_probs.sum(dim=(0, 1)) # Total routing prob/weight assigned to an expert across all tokens
                routing_weights_per_t.append(routing_weight)

            x = self.scheduler.sample_prev_timestep(t, x, pred_noise)
            if debug and (i%debug_stepsize == 0) and (step_idx < debug_steps):
                x_debug = ((x.clamp(-1., 1.) + 1) * 127.5).to("cpu").to(torch.uint8)
                debug_ret[step_idx] = x_debug
                step_idx += 1

        if self.enable_moe:
            routing_weights_per_t = torch.stack(routing_weights_per_t, dim=0).detach().cpu().numpy() # [num_timesteps, n_experts]
            experts_used_per_t = torch.stack(experts_used_per_t, dim=0).detach().cpu().numpy() # [num_timesteps, n_experts]
        else:
            routing_weights_per_t, experts_used_per_t = [], []

        x = x.clamp(-1., 1.)
        x = ((x + 1) * 127.5).to("cpu").to(torch.uint8)
        return x, experts_used_per_t, routing_weights_per_t, debug_ret
