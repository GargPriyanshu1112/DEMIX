import math
import torch
import torch.nn.functional as F
from torch import nn
from typing import List
from dataclasses import dataclass, field

from utils import RoutingStats, MoEStats

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
    use_aux_loss: bool = True
    lambda_aux: float = 0.01
    use_z_loss: bool = True
    lambda_z: float = 0.001
    n_experts: int = 8
    k: int = 2
    use_noisy_topk: bool = True
    min_expert_capacity: int = 4
    capacity_factor: float = 3.0
    bias: bool = False
    dropout: float = 0.0
    placement: MoEPlacementConfig = field(default_factory=MoEPlacementConfig)


class Router(nn.Module):
    def __init__(self, d, cfg):
        super().__init__()
        assert cfg.k >= 1 and cfg.k <= cfg.n_experts
        self.n_experts = cfg.n_experts
        self.k = cfg.k
        self.use_noisy_topk = cfg.use_noisy_topk
        self.min_expert_capacity = cfg.min_expert_capacity
        self.capacity_factor = cfg.capacity_factor
        self.experts_proj = nn.Linear(d, cfg.n_experts, bias=False)
        self.experts_proj_noisy = nn.Linear(d, cfg.n_experts, bias=False) if cfg.use_noisy_topk else None

        self.use_aux_loss = cfg.use_aux_loss
        self.use_z_loss = cfg.use_z_loss
        self._init_weights()

    def _init_weights(self, scale=0.1):
        """
        Purpose:
        Sparse MoE models are highly sensitive to weight scale because large router
        weights produce high-magnitude logits, causing softmax to become overly peaky.
        This leads to imbalanced expert utilization and can result in expert collapse.
        Reducing the initialization scale reduces variance and improves stability.
        """
        fan_in = self.experts_proj.weight.shape[1] # num input connections contributing to each output unit
        std = math.sqrt(scale / fan_in)

        torch.nn.init.trunc_normal_(tensor=self.experts_proj.weight, mean=0.0, std=std, a=-2*std, b=2*std)

        if self.use_noisy_topk:
            torch.nn.init.trunc_normal_(tensor=self.experts_proj_noisy.weight, mean=0.0, std=std, a=-2*std, b=2*std)

    def forward(self, x): # x: [B, T, emb_dim]
        zero = x.new_zeros(())
        batch_num_tokens = x.shape[0] * x.shape[1] # total tokens in the input batch

        logits = self.experts_proj(x) # [B, T, n_experts]
        if self.use_noisy_topk:
            # Add noise into the router
            noise = F.softplus(self.experts_proj_noisy(x))
            noise *= torch.randn_like(noise)
            logits += noise # [B, T, n_experts]

        if self.training and self.use_z_loss:
            z_loss = self.compute_z_loss(logits=logits)
        else:
            z_loss = zero

        # Top-k experts for each token
        topk_logits, topk_indices = logits.topk(self.k, dim=-1) # [B, T, K]

        # Probability of top-k experts for each token
        topk_probs = torch.full_like(logits, -torch.inf) # [B, T, n_experts]
        topk_probs.scatter_(-1, topk_indices, topk_logits) # [B, T, n_experts]
        topk_probs = F.softmax(topk_probs, -1) # [B, T, n_experts]

        if self.training and self.use_aux_loss:
            aux_loss = self.compute_aux_loss(topk_indices=topk_indices, topk_probs=topk_probs)
        else:
            aux_loss = zero

        # Expert capacity
        expert_capacity = math.floor(self.k * batch_num_tokens * self.capacity_factor / self.n_experts)
        expert_capacity += expert_capacity % 2 # make sure expert capacity is an even number
        expert_capacity = max(expert_capacity, self.min_expert_capacity)
        expert_capacity = int(expert_capacity)
        assert expert_capacity > 0

        # One-hot expert mask per token, for each k
        expert_mask = F.one_hot(topk_indices, num_classes=self.n_experts) # [B, T, K, n_experts]
        expert_mask = expert_mask.view(batch_num_tokens, self.k, self.n_experts) # [B*T, K, n_experts]
        expert_mask = expert_mask.permute(1, 0, 2) # [K, B*T, n_experts]

        # Per-expert token routing order. Tokens are assigned to experts in top-k priority order (top-1 first, then top-2,...).
        routing_order = expert_mask.reshape(self.k*batch_num_tokens, self.n_experts) # [K*B*T, n_experts]
        routing_order = torch.cumsum(routing_order, dim=0) - 1 # [K*B*T, n_experts], subtracted -1 as queue is 0-indexed
        routing_order = routing_order.reshape(self.k, batch_num_tokens, self.n_experts) # [K, B*T, n_experts]

        # Enforce per-expert capacity constraint. Mask out tokens beyond `expert_capacity`.
        expert_mask *= torch.lt(routing_order, expert_capacity) # [K, B*T, n_experts], element wise less-than operation

        # Utilized expert capacity
        expert_cap_util = torch.sum(expert_mask, dim=(0, 1)) # [n_experts]

        # Mask out routing probs for tokens beyond `expert_capacity`
        topk_probs = topk_probs.view(batch_num_tokens, self.n_experts)[None, :] # [1, B*T, n_experts]
        topk_probs = expert_mask * topk_probs # [K, B*T, n_experts]

        # One-hot vector of each token's capacity slot index (position) within its assigned expert
        cap_slot_idx = F.one_hot(
            torch.sum(expert_mask * routing_order, dim=-1), # [K, B*T], for each token, its position inside the assigned expert
            num_classes=expert_capacity
        ) # [K, B*T, expert_capacity]

        # Scatter each token’s top-k routing probabilities into its assigned (expert, capacity slot)
        # to build a structured routing tensor for batched expert computation.
        topk_probs.unsqueeze_(3) # [K, B*T, n_experts, 1]
        cap_slot_idx.unsqueeze_(2) # [K, B*T, 1, expert_capacity]
        routing_weights = torch.sum(topk_probs * cap_slot_idx, dim=0) # [B*T, n_experts, expert_capacity]

        # Binary mask - True where a token is routed to an expert at a specific capacity slot, False everywhere else.
        routing_mask = routing_weights.bool() # [B*T, n_experts, expert_capacity]

        return (
            aux_loss,
            z_loss,
            RoutingStats(
                topk_indices=topk_indices,
                routing_weights=routing_weights,
                expert_cap_util=expert_cap_util,
                routing_mask=routing_mask
            )
        )

    def compute_aux_loss(self, topk_indices, topk_probs):
        with torch.no_grad():
            one_hot_indices = F.one_hot(topk_indices, num_classes=self.n_experts) # [B, T, K, n_experts]
            one_hot_indices = torch.sum(one_hot_indices.float(), dim=2) # [B, T, n_experts]
            tokens_per_expert = torch.mean(one_hot_indices.float(), dim=(0, 1)) # [n_experts]
        probs_per_expert = torch.mean(topk_probs.float(), dim=(0, 1)) # [n_experts]
        return self.n_experts * torch.sum(tokens_per_expert * probs_per_expert)

    def compute_z_loss(self, logits):
        return torch.mean(
            torch.logsumexp(logits, dim=-1) ** 2.0
        )


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

        self._init_weights()

    def _init_weights(self, scale=0.1):
        """
        Purpose:
        In MoE models, experts are updated using only a subset of tokens due to top-k
        routing, leading to higher gradient variance. A reduced initialization scale
        helps mitigate this by stabilizing activations and gradients during early training.
        """
        fan_in = self.fc1.shape[1]
        std = math.sqrt(scale / fan_in)
        torch.nn.init.trunc_normal_(tensor=self.fc1, mean=0.0, std=std, a=-2*std, b=2*std)

        fan_in = self.fc2.shape[1]
        std = math.sqrt(scale / fan_in)
        torch.nn.init.trunc_normal_(tensor=self.fc2, mean=0.0, std=std, a=-2*std, b=2*std)

        if self.bias:
            torch.nn.init.zeros_(self.fc1_bias)
            torch.nn.init.zeros_(self.fc2_bias)

    def forward(self, x):
        x = torch.bmm(x, self.fc1) # batched matrix multiplication
        if self.bias:
            x += self.fc1_bias
        x = self.gelu(x)
        x = torch.bmm(x, self.fc2) # batched matrix multiplication
        if self.bias:
            x += self.fc2_bias
        x = self.dropout(x)
        return x


class IdentityMoELayer(nn.Module):
    def forward(self, x):
        return x, MoEStats.empty_like(x)

class MoELayer(nn.Module):
    def __init__(self, d, cfg):
        super().__init__()
        self.router = Router(d, cfg)
        self.experts = Experts(d, cfg)
        # Learnable residual scale for MoE output. Initialized small (0.1) to ensure
        # training stability early on — preventing random expert outputs from corrupting
        # the feature map before experts have learned meaningful specializations.
        # As training progresses, the scale should adapt, allowing experts to contribute
        # more strongly once their weights and routing are stabilized.
        self.residual_scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x):
        residual = x # for residual scaling

        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H*W, C) # [B, H*W, C]
        B, T, emb_dim = x.size()

        # Run router in full precision to avoid training instability (https://arxiv.org/abs/2101.03961, pg.9-10)
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            aux_loss, z_loss, routing_stats = self.router(x.float()) # x.float() casts the tensor to torch.float32

        routing_weights = routing_stats.routing_weights.view(B*T, -1) # [B*T, n_experts*expert_capacity]
        routing_mask = routing_stats.routing_mask.permute(1, 2, 0).type_as(x) # [n_experts, expert_capacity, B*T]

        # Reshape tokens into batches for each expert
        x = x.reshape(B*T, emb_dim) # [B*T, emb_dim]
        x = routing_mask @ x # [n_experts, expert_capacity, emb_dim]

        # Expert output
        outputs = self.experts(x) # [n_experts, expert_capacity, emb_dim]

        outputs = outputs.view(-1, emb_dim) # [n_experts*expert_capacity, emb_dim]
        outputs = routing_weights @ outputs # [B*T, emb_dim]
        outputs = outputs.reshape(B, H, W, C).permute(0, 3, 1, 2) # [B, C, H, W]

        outputs = residual + (self.residual_scale * outputs)

        return (
            outputs,
            MoEStats(
                aux_loss=aux_loss,
                z_loss=z_loss,
                routing=routing_stats
            )
        )