import math
import torch
import torch.nn.functional as F
from torch import nn
from typing import List
from dataclasses import dataclass, field

@dataclass
class MoEBlockPlacementConfig:
    resolutions: List[int] = field(default_factory=list)
    blocks: List[int] = field(default_factory=list)

@dataclass
class MoEPlacementConfig:
    downblock: MoEBlockPlacementConfig = field(default_factory=MoEBlockPlacementConfig)
    middleblock: MoEBlockPlacementConfig = field(default_factory=MoEBlockPlacementConfig)
    upblock: MoEBlockPlacementConfig = field(default_factory=MoEBlockPlacementConfig)

@dataclass
class MoEConfig:
    enable: bool = False
    n_experts: int = 8
    topk: int = 2
    use_noisy_topk: bool = True
    min_expert_capacity: int = 4
    capacity_factor: float = 3.0
    bias: bool = False
    dropout: float = 0.0
    placement: MoEPlacementConfig = field(default_factory=MoEPlacementConfig)


class Router(nn.Module):
    def __init__(self, d, cfg):
        super().__init__()
        assert cfg.topk >= 1 and cfg.topk <= cfg.n_experts
        self.n_experts = cfg.n_experts
        self.topk = cfg.topk
        self.use_noisy_topk = cfg.use_noisy_topk
        self.min_expert_capacity = cfg.min_expert_capacity
        self.capacity_factor = cfg.capacity_factor
        self.experts_proj = nn.Linear(d, cfg.n_experts, bias=False)
        self.experts_proj_noisy = nn.Linear(d, cfg.n_experts, bias=False) if cfg.use_noisy_topk else None

    def forward(self, x): # x: [B, T, emb_dim]
        batch_num_tokens = x.shape[0] * x.shape[1] # total tokens in the input batch

        logits = self.experts_proj(x) # [B, T, n_experts]
        if self.use_noisy_topk:
            # Add noise into the router
            noise = F.softplus(self.experts_proj_noisy(x))
            noise = torch.randn_like(noise)
            logits += noise # [B, T, n_experts]

        # Top-k experts for each token
        topk_logits, topk_indices = logits.topk(self.topk, dim=-1) # [B, T, K]

        # Probabilities of chosen experts for each token
        router_probs = torch.full_like(logits, -torch.inf) # [B, T, n_experts]
        router_probs.scatter_(-1, topk_indices, topk_logits) # [B, T, n_experts]
        router_probs = F.softmax(router_probs, -1) # [B, T, n_experts]

        # Expert capacity
        expert_capacity = math.floor(self.topk * batch_num_tokens * self.capacity_factor / self.n_experts)
        expert_capacity += expert_capacity % 2 # make sure expert capacity is an even number
        expert_capacity = max(expert_capacity, self.min_expert_capacity)
        expert_capacity = int(expert_capacity)
        assert expert_capacity > 0

        # One-hot mask of chosen experts for each token
        mask = F.one_hot(topk_indices, num_classes=self.n_experts) # [B, T, K, n_experts]
        mask = mask.view(batch_num_tokens, self.topk, self.n_experts) # [B*T, K, n_experts]
        mask = mask.permute(1, 0, 2) # [K, B*T, n_experts]

        # Token's index for its chosen expert. Top experts prioritized (top-1 first, top-2 second, etc.)
        token_idx_for_experts = mask.reshape(self.topk*batch_num_tokens, self.n_experts) # [K*B*T, n_experts]
        token_idx_for_experts = torch.cumsum(token_idx_for_experts, dim=0) - 1 # subtracted -1 as queue is 0-indexed, [K*B*T, n_experts]
        token_idx_for_experts = token_idx_for_experts.reshape(self.topk, batch_num_tokens, self.n_experts) # [K, B*T, n_experts]

        # Mask to zero-out token indexes beyond expert capacity
        mask *= torch.lt(token_idx_for_experts, expert_capacity) # [K, B*T, n_experts]
        used_capacity = torch.sum(mask, dim=(0, 1)) # [n_experts]

        # Mask indexes to only include selected tokens (those withen expert capacity)
        token_idx_for_experts = torch.sum(mask * token_idx_for_experts, dim=-1)  # [K, B*T]

        # Token's one-hot position within the capacity of the chosen expert
        token_capacity_idx_for_expert = F.one_hot(token_idx_for_experts, num_classes=expert_capacity) # [K, B*T, expert_capacity]

        # Mask router probs. to zero-out probabilities for redundant tokens (those beyond expert capacity)
        router_probs = router_probs.view(batch_num_tokens, self.n_experts)[None, :] # [B, T, n_experts] -> [1, B*T, n_experts]
        expert_weights = mask * router_probs # [K, B*T, n_experts]

        # Weight of selected expert for each token at position the capacity of that expert
        # [K, B*T, n_experts, 1] * [K, B*T, 1, expert_capacity] -> [K, B*T, n_experts, expert_capacity]
        expert_weights = torch.sum(expert_weights.unsqueeze(3) * token_capacity_idx_for_expert.unsqueeze(2), dim=0) # [B*T, n_experts, expert_capacity]

        # Binary mask of selected experts for each token
        mask = expert_weights.bool() # [B*T, n_experts, expert_capacity]

        return used_capacity, expert_weights, mask


class Experts(nn.Module):
    def __init__(self, d, cfg):
        super().__init__()
        self.bias = cfg.bias
        self.fc1 = nn.Parameter(torch.empty(cfg.n_experts, d, d*4))
        self.fc2 = nn.Parameter(torch.empty(cfg.n_experts, d*4, d))
        self.fc1_bias = nn.Parameter(torch.empty(cfg.n_experts, 1, d*4)) if self.bias else None
        self.fc2_bias = nn.Parameter(torch.empty(cfg.n_experts, 1, d)) if self.bias else None
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        x = torch.bmm(x, self.fc1)
        if self.bias:
            x += self.fc1_bias
        x = self.gelu(x)
        x = torch.bmm(x, self.fc2)
        if self.bias:
            x += self.fc2_bias
        x = self.dropout(x)
        return x


class MOELayer(nn.Module):
    def __init__(self, c, cfg):
        super().__init__()
        self.router = Router(c, cfg)
        self.experts = Experts(c, cfg)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H*W, C)

        B, T, emb_dim = x.size()
        batch_num_tokens = B*T # total tokens in the input batch

        _, expert_weights, expert_mask = self.router(x)

        # Reshape inputs into batches for each expert
        x = x.reshape(batch_num_tokens, emb_dim) # [B*T, emb_dim]
        expert_mask = expert_mask.permute(1, 2, 0).type_as(x) # [n_experts, expert_capacity, B*T]
        x = expert_mask @ x # [n_experts, expert_capacity, emb_dim]

        outputs = self.experts(x) # [n_experts, expert_capacity, emb_dim]
        outputs = outputs.view(-1, emb_dim) # [n_experts*expert_capacity, emb_dim]

        expert_weights = expert_weights.view(batch_num_tokens, -1) # [B*T, n_experts*expert_capacity]
        outputs = expert_weights @ outputs # [B*T, emb_dim]

        outputs = outputs.reshape(B, C, H, W)
        return outputs