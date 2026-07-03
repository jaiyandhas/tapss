# TAPSS — Task-Adaptive Parameter Saliency Score

> An exploratory prototype investigating whether explainability-guided parameter importance can help reduce catastrophic forgetting during continual LoRA fine-tuning.

*Python 3.12 · PyTorch · HuggingFace PEFT*

---

## What this is

I built this during my gap year to dig into a specific question that kept nagging at me while reading the EWC paper: the Fisher Information Matrix is elegant, but it measures *optimization sensitivity*, not *functional importance*. A parameter can have a huge Fisher value just because it sits near a sharp loss landscape — that doesn't necessarily mean the model *relies* on it for its predictions.

TAPSS is my attempt to take the explainability angle instead. The idea is simple:

> If a parameter consistently receives large gradients, feeds into high-activation neurons, and causes prediction changes when perturbed — that parameter is probably storing something the model actually needs. Protect those during Task B fine-tuning.

This is a proof-of-concept, not a production system. The experiments are small-scale and run on CPU/MPS. The code is designed so each piece is easy to read and swap out.

---

## The core formula

```
TAPSS = w₁·S_gradient + w₂·S_perturbation + w₃·S_activation + w₄·S_layer
```

| Component | What it's measuring |
|-----------|-------------------|
| Gradient Magnitude | How hard the loss is pushing on this parameter |
| Perturbation Sensitivity | Does changing this weight break predictions? |
| Activation Frequency | Is this weight feeding into neurons that actually fire? |
| Layer Contribution | Taylor approximation of what happens if this weight is gone |

Default weights are `0.40 : 0.30 : 0.20 : 0.10`. These are a starting point, not optimized values — the ablation section below is where I plan to actually evaluate this.

---

## Project layout

```
TAPSS/
├── configs/               # YAML experiment configs
│   ├── model/
│   ├── dataset/
│   ├── saliency/
│   ├── protection/
│   ├── lora/
│   └── experiment/
│
├── task_datasets/         # Dataset loading / tokenisation
├── models/                # Model registry + TAPSSModel wrapper
│
├── saliency/              # The actual research
│   ├── base.py            # ImportanceEstimator ABC + ImportanceScores dataclass
│   ├── gradient.py        # Gradient magnitude
│   ├── activation.py      # Activation frequency via forward hooks
│   ├── perturbation.py    # Perturbation sensitivity (slow but direct)
│   ├── layer_contribution.py  # Taylor criterion
│   ├── tapss.py           # Weighted combination
│   ├── rankings.py        # Ranking + CSV/JSON/HTML export
│   └── knowledge_probe.py # KL divergence / embedding distance probe
│
├── peft_modules/
│   ├── protection.py      # 5 protection policies
│   └── lora_trainer.py    # Training loop
│
├── continual_learning/
│   ├── pipeline.py        # Task A → Task B → forgetting measurement
│   ├── baselines.py       # Vanilla LoRA, naive fine-tuning, random protection
│   └── ewc.py             # EWC baseline
│
├── evaluation/            # Metrics, tables, SQLite experiment tracker
├── visualization/         # Matplotlib + Plotly plots, HTML report generator
├── scripts/               # CLI entry points
├── tests/                 # Unit tests (45, all passing)
└── outputs/               # Saved results (gitignored except .gitkeep)
```

---

## Installation

```bash
# with poetry
poetry install

# or plain pip
pip install -r requirements.txt
```

Tested on Python 3.12, PyTorch 2.4, macOS (MPS) and Linux (CUDA). Should work on CPU too, just slower.

---

## Running experiments

### Compute importance scores only
```bash
python scripts/compute_saliency.py \
    --model distilbert-base-uncased \
    --dataset ag_news \
    --method tapss \
    --output-dir outputs/saliency
```

### Full TAPSS experiment (AG News → SST-2)
```bash
python scripts/run_experiment.py \
    --config configs/experiment/cl_ag_sst2.yaml
```

For a quick sanity check with tiny subsets:
```bash
python scripts/run_experiment.py \
    --config configs/experiment/cl_ag_sst2.yaml \
    -o dataset.train_size=64 -o training.num_epochs=1 \
    --skip-perturbation
```

### Compare all baselines
```bash
python scripts/compare_methods.py \
    --config configs/experiment/baseline_comparison.yaml
```

### Visualise saved results
```bash
python scripts/visualize.py --results-dir outputs/cl_ag_sst2
```

---

## Config system

Everything is YAML-driven. Override any value from CLI:

```bash
python scripts/run_experiment.py \
    --config configs/experiment/cl_ag_sst2.yaml \
    -o training.num_epochs=5 \
    -o protection.topk_percent=30 \
    -o saliency.weights.gradient=0.5
```

A snapshot of the exact config used is saved alongside every run for reproducibility.

Example (`configs/experiment/cl_ag_sst2.yaml`):
```yaml
experiment:
  name: "cl_ag_sst2"
  seed: 42

model:
  name: "distilbert-base-uncased"

continual_learning:
  task_a_dataset: "ag_news"
  task_b_dataset: "sst2"

saliency:
  method: "tapss"
  weights:
    gradient: 0.40
    perturbation: 0.30
    activation: 0.20
    layer_contribution: 0.10

protection:
  policy: "freeze_topk"
  topk_percent: 20.0

lora:
  r: 8
  lora_alpha: 16
```

---

## What gets saved

```
outputs/<experiment_name>/
├── config.yaml
├── report.html               # self-contained interactive HTML report
├── experiments.db            # SQLite log (all runs, queryable)
├── rankings/
│   ├── parameter_rankings.csv
│   ├── parameter_rankings.json
│   └── parameter_rankings.html
├── plots/
│   ├── layer_heatmap.png
│   ├── importance_histogram.png
│   ├── protection_map.png
│   └── task_b_training.png
└── checkpoints/
```

---

## Metrics tracked

| Metric | Definition |
|--------|------------|
| Forgetting | `acc_A_pre − acc_A_post` (lower is better) |
| Backward Transfer | `acc_A_post − acc_A_pre` |
| Average Accuracy | `(acc_A_post + acc_B) / 2` |
| Training Time | wall-clock seconds |
| Trainable Params | LoRA adapter count during Task B |

---

## Tests

```bash
pytest tests/ -v
```

45 tests, all on toy models — no HuggingFace downloads needed. Runs in ~7 seconds.

---

## Research questions I'm trying to answer

**RQ1.** Does TAPSS reduce catastrophic forgetting vs. Vanilla LoRA?

**RQ2.** Does score quality matter? (TAPSS vs. Random Protection with identical policy)

**RQ3.** Is combining multiple signals better than any single one? (needs the ablations below)

**RQ4.** How does TAPSS compare to an EWC-style Fisher baseline on forgetting metrics?

---

## Ablation plan

The component weights are a hypothesis right now, not a conclusion. The ablation matrix I want to run:

| Run | Config |
|-----|--------|
| gradient-only | `w1=1, w2=0, w3=0, w4=0` |
| perturbation-only | `w1=0, w2=1, w3=0, w4=0` |
| activation-only | `w1=0, w2=0, w3=1, w4=0` |
| layer-only | `w1=0, w2=0, w3=0, w4=1` |
| grad+perturb | `w1=0.5, w2=0.5, w3=0, w4=0` |
| full TAPSS | default weights |

```bash
python scripts/compute_saliency.py --method gradient ...
python scripts/compute_saliency.py --method perturbation ...
python scripts/compute_saliency.py --method tapss ...
```

---

## Limitations (honest ones)

- Classifier head is re-initialised between tasks — a real CL setup would need task-specific heads or a shared head with proper masking.
- Only two tasks. Forgetting compounds over longer sequences; haven't tested that yet.
- Protection of LoRA weights isn't well-motivated — the interesting thing is protecting *base model* weights that the adapters perturb indirectly.
- Perturbation sensitivity is O(params × batches). On a full BERT it's painfully slow; use `--skip-perturbation` for anything larger than DistilBERT.
- EWC baseline uses diagonal Fisher. Full Fisher would be more principled but requires O(params²) memory.

---

## Things I want to add

- [ ] Ablation runs (see above — this is the highest priority)
- [ ] Attention head-level importance (instead of weight-matrix-level)
- [ ] Longer task sequences (5+ tasks)
- [ ] fp16/bf16 training support
- [ ] Weight optimisation via Bayesian search over the TAPSS component weights

---

## Citation

If this is useful to you:

```bibtex
@misc{tapss2026,
  title  = {TAPSS: Task-Adaptive Parameter Saliency Score for Continual Learning},
  year   = {2026},
  note   = {Undergraduate research prototype}
}
```

---

## References

1. Kirkpatrick et al., "Overcoming catastrophic forgetting in neural networks", PNAS 2017.
2. Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022.
3. Li & Hoiem, "Learning without Forgetting", TPAMI 2018.
4. Molchanov et al., "Pruning CNNs for Resource Efficient Inference", ICLR 2017.
5. Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization", ICCV 2017.
6. Lopez-Paz & Ranzato, "Gradient Episodic Memory for Continual Learning", NeurIPS 2017.

---

*This is a research prototype, not a production system.*
