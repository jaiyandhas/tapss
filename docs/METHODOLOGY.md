# TAPSS Methodology Notes

These are my working notes on the design of each estimator. I'm writing this as I go, so it'll get updated as experiments reveal things.

---

## Why not just use EWC?

EWC is the obvious baseline and I implement it here for comparison. The Fisher Information Matrix measures how sensitive the loss is to each parameter — a high Fisher value means moving that weight hurts the loss a lot. That's a reasonable importance proxy.

But the Fisher is an optimization-space quantity. It's high for parameters near sharp loss curvature. That's not quite the same as "the model needs this parameter to make correct predictions."

The hypothesis I want to test: gradient-based + perturbation-based + activation-based importance might identify a different (and complementary) set of "important" parameters — ones that encode factual knowledge rather than just being near the bottom of a sharp valley.

Whether that turns out to be true is what the experiments are for.

---

## Estimator 1: Gradient Magnitude

For each parameter `θ_i`, compute the average absolute gradient over a calibration set:

```
S_grad(θ_i) = E_{x ~ D_A} [ |∂L/∂θ_i| ]
```

This is the simplest possible signal. Parameters that consistently get large gradients during inference are the ones the loss is "trying to fix" — so they're doing something that matters.

The downside is noise. Gradient magnitudes are sensitive to learning rate, batch composition, and the specific loss landscape. A parameter might get large gradients just because it happens to be in a bad region at the time of measurement, not because it's structurally important.

I run this over a fixed calibration subset of Task A data (default: 50 batches, no updates — just forward + backward to collect gradients).

---

## Estimator 2: Activation Frequency

Register forward hooks on all `nn.Linear` layers and compute the mean absolute activation magnitude:

```
S_act(θ_i) = E_{x ~ D_A} [ mean |h_i(x)| ]
```

where `h_i` is the output of the layer containing `θ_i`.

The idea: if a weight matrix feeds into a layer whose neurons are consistently firing at high magnitude, that weight is "in use." Weights feeding into near-zero activations can probably be changed without breaking anything.

This is cheap to compute and doesn't require any gradient computation. The signal is coarser though — it's per-layer, not per-parameter.

---

## Estimator 3: Perturbation Sensitivity

This one is slow but the most direct. For each parameter tensor, add Gaussian noise and measure the KL divergence between the original and perturbed output distributions:

```
S_pert(θ_i) = E_{x ~ D_A} [ KL( f(x; θ) || f(x; θ + ε_i) ) ]
```

where `ε_i ~ N(0, σ²)` is applied in-place, then reversed after measurement.

If changing `θ_i` doesn't change predictions, it doesn't matter. This is model-agnostic — doesn't care about the loss function or gradient flow.

The O(params × batches) complexity is brutal on larger models. On DistilBERT with 50 calibration batches it takes a few minutes. I added `--skip-perturbation` for quick iteration.

---

## Estimator 4: Layer Contribution (Taylor Criterion)

Based on Molchanov et al. (2017). The first-order Taylor approximation of what happens to the loss if you zero out a weight:

```
S_layer(θ_i) ≈ 0.7 * E[ (∂L/∂θ_i) · |h_parent| ] + 0.3 * layer_mean
```

The gradient × activation product is a proxy for ΔL from removing that feature. I add a layer-mean term (weighted 0.3) because pure Taylor can be noisy for individual weights — averaging over the layer smooths it out a bit.

The `0.7 / 0.3` split is a heuristic I picked because it seemed stable across a few quick tests. It should get ablated properly.

---

## Combined TAPSS Score

Combine the four signals with configurable weights:

```
TAPSS(θ_i) = Σ_k  w_k · S_k(θ_i)
```

Each component is independently min-max normalised to `[0, 1]` before combining, so the weights are comparable. The weights are renormalised to sum to 1 if they don't already.

Default: `gradient=0.40, perturbation=0.30, activation=0.20, layer=0.10`

I chose these by intuition: perturbation is the most direct signal but also the noisiest at small calibration set sizes, so gradient gets a slight edge. Activation is coarser so it gets less weight. Layer contribution fills in the rest.

This is a hypothesis. The ablations will tell me if the combination actually outperforms single components.

---

## Protection Policies

### Policy A: Freeze Top-K
Zero out the gradient for the top-K% parameters (by TAPSS score) before the optimizer step. Simplest possible protection.

Upside: guaranteed no movement. Downside: reduces the model's ability to adapt to Task B, since some frozen params might actually be useful for both tasks.

### Policy B: LR Scaling
Build separate optimizer param groups with learning rate scaled by `(1 - importance)`. High-importance params get a much lower effective LR.

```
lr_eff(θ_i) = base_lr × max(min_ratio, 1 - importance_i × scale)
```

Softer than freezing. The optimizer can still update protected weights, just slowly.

### Policy C: Regularization (TAPSS-EWC)
Add an importance-weighted L2 pull toward the Task A weights:

```
L_reg = λ · Σ_i  importance_i · ||θ_i - θ_i^A||²
```

Directly analogous to EWC, but using TAPSS scores instead of Fisher. This is the cleanest comparison point — same policy family, different importance signal.

### Policy D: Soft Gradient Dampening
Scale down gradients after the backward pass:

```
grad_eff(θ_i) = grad(θ_i) × max(min_ratio, 1 - importance_i × strength)
```

Similar effect to LR scaling but applied per-parameter at the gradient level rather than the optimizer level. Easier to implement with any optimizer.

### Policy E: Adaptive
Combines C + D and schedules them to relax over the course of Task B training. Starts strict (high λ, high dampening) and loosens as training progresses.

Intuition: early in Task B fine-tuning, the model is far from the Task B optimum and gradients are large. That's when forgetting is most dangerous. Late in training, gradients are smaller and the model has already adapted — less need for strict protection.

---

## Experimental setup

Task sequence: `AG News (4-class) → SST-2 (2-class)`

Both are sentence classification, so the model architecture stays fixed between tasks. The classifier head is re-initialised when switching to Task B (this is a known limitation — a proper CL setup would handle this differently).

Forgetting is measured by re-evaluating on the Task A test set after Task B training:

```
Forgetting   = acc_A_pre − acc_A_post
BWT          = acc_A_post − acc_A_pre    (= −Forgetting)
Avg Accuracy = (acc_A_post + acc_B) / 2
```

All baselines use identical data splits, model checkpoints, and LoRA config. The only thing that changes is the protection mechanism.

---

## Known issues / things to fix

- The re-initialisation of the classifier head between tasks means Task A and Task B evaluations use different heads. A task-incremental setup with task IDs would be cleaner.
- Diagonal Fisher in the EWC baseline is rough — full Fisher would be more principled.
- I haven't tested forgetting over sequences longer than 2 tasks.
- The `0.7/0.3` split in the Taylor criterion and the default component weights both need ablation, not just intuition.
