# DEMIX — Denoising Diffusion with Mixture of Experts

---

## Overview

In Denoising Diffusion Probabilistic Models (DDPM), a UNet is trained to predict the noise added to an image at a given timestep.

- **Input:** Timestep *t* and the corresponding noisy image $x_t$
- **Output:** Predicted noise $\epsilon_\theta(x_t, t)$

The UNet **does not reconstruct the input image**. Instead, it learns to estimate the noise component present at each timestep.

---

## The Nature of the Problem

Each timestep corresponds to a fundamentally different prediction task:

| Timestep           | Input condition    | Task description                                          |
| ------------------ | ------------------ | --------------------------------------------------------- |
| $t \approx 1000$ | Pure noise         | Predict large-magnitude, unstructured noise               |
| $t \approx 500$  | Partially noisy    | Predict structured noise conditioned on underlying signal |
| $t \approx 100$  | Slight noise       | Predict low-magnitude noise                               |
| $t \approx 1$    | Nearly clean image | Predict small residual noise                              |

The UNet effectively solves 1,000 distinct noise prediction problems — one per timestep — using the same set of weights across all these noise regimes.

---

## Limitations of Dense Architectures

In a standard UNet block, a single set of weights processes all timesteps. The network must average behaviour across every noise regime, which leads to two failure modes:

- **Suboptimal per-timestep performance:** no regime receives dedicated capacity.
- **Conflicting gradients:** coarse and fine denoising tasks interfere with one another during training.

---

## Why Mixture of Experts?

Mixture of Experts (MoE) addresses these limitations through *conditional computation*. Rather than sharing one transformation across all inputs, multiple expert networks exist in parallel. A lightweight router selects which experts to activate for each token.

The key principle is to **allocate specialized capacity to different denoising regimes**, rather than forcing a single network to handle all of them.

---

## What Do Experts Learn?

There are three proposed hypotheses for how experts may specialize:

**Hypothesis 1 — Timestep specialization**

Experts specialize according to noise level.

- Expert A → high-noise inputs
- Expert B → low-noise inputs
- Expert C → near-clean inputs

**Hypothesis 2 — Content specialization**

Experts specialize according to spatial structure.

- Expert A → edges and boundaries
- Expert B → smooth or homogeneous regions

**Hypothesis 3 — Joint specialization**

Experts specialize on (timestep, content) combinations.

- Expert A → edges at low $t$
- Expert B → smooth regions at high $t$
- Expert C → textures at mid $t$

Because expert weights do not interfere with one another, each expert can overfit to its specific regime without compromising the others.

---

## Architecture

### Placement within the ResBlock

MoE is applied **immediately after timestep embedding injection** in the ResNet block:

```python
h = self.g_norm1(x)
h = F.silu(h)
h = self.conv1(h)
h = h + t                         # timestep embedding injected
h, moe_stats = self.moe_layer(h)  # MoE applied here
h = self.g_norm2(h)
h = F.silu(h)
h = self.dropout(h)
h = self.conv2(h)
h = h + self.shortcut(x)
```

### Design rationale

Two natural insertion points exist within a ResBlock of the diffusion UNet: immediately after timestep injection, or at the end of the residual block. Each carries distinct tradeoffs.

**Immediately after timestep injection (adopted)**

In diffusion models, $t$ is the primary conditioning signal — it directly governs denoising behaviour at every step. Placing MoE at this point ensures routing decisions are made on representations that jointly encode both spatial content and the current denoising stage. This encourages the router to select experts based on *what* a feature represents (content) as well as *where* the model is in the denoising process (timestep), enabling explicit timestep-aware specialization.

The tradeoff is that the router receives partially processed representations — before the second normalization, second convolution, and residual addition — rather than fully refined outputs. Additionally, the timestep embedding shifts the activation distribution prior to routing, which can introduce early instability.

In practice, this instability is mitigated by initializing the gating network as a lightweight linear projection with small-scale weights, producing low-variance logits at initialization. This prevents overly sharp routing decisions early in training and allows the router to adapt gradually as learning progresses [[ST-MoE, pg. 10](https://arxiv.org/abs/2101.03961)].

**At the end of the residual block (not adopted)**

Routing on fully processed residual outputs provides more stable, statistically well-behaved representations. However, the timestep signal injected earlier becomes diluted after passing through multiple nonlinear and convolutional operations, making it less explicitly accessible to the router. Routing decisions become less directly conditioned on the denoising stage — which is the primary quantity of interest in this work.

This work adopts placement after timestep injection as a deliberate choice to prioritize **explicit timestep-aware routing**, which is central to the goal of learning and analysing expert specialization across the denoising trajectory.

---

### Resolution

MoE routing requires meaningful token diversity to learn useful specialization. When all tokens appear similar, routing collapses.

| Resolution           | Spatial diversity                                      | Expected routing behaviour                                  |
| -------------------- | ------------------------------------------------------ | ----------------------------------------------------------- |
| Bottleneck (4×4)    | Low — features are highly compressed and global       | All positions likely routed identically; low specialization |
| Early layer (32×32) | High — local structure (edges, textures) is preserved | Rich specialization more likely to emerge                   |
