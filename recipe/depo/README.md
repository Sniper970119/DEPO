# DEPO: Difficulty-Estimated Policy Optimization

DEPO extends GRPO / DAPO with an **online difficulty filter** that steers training toward prompts in the model's current learning zone — neither too easy nor too hard.

## Key Idea

During training, a lightweight **DualHeadScoringModel** (a small BERT encoder with dual output heads) co-evolves with the policy:

1. **Before rollout**: the scorer predicts a *difficulty score* (≈ mean@k accuracy) for every candidate prompt.  Prompts outside the configured window `[min_thresh, max_thresh]` are discarded — only "just-right" prompts enter the rollout.
2. **After PPO update**: the scorer is updated with the *actual* mean@k labels from the current step, plus a knowledge-distillation signal from the policy's own log-probs.

The scorer's three-part loss is:

```
L = BCE(score_head, mean@k)                     # accuracy prediction
  + w_distill × BCE(distill_head, exp(avg_logp)) # policy KD
  + w_ranking × pairwise_ranking_loss             # ordinal consistency
```

## Repository Layout

```
recipe/depo/
├── __init__.py
├── main_depo.py              # Hydra entry point
├── depo_ray_trainer.py       # RayDEPOTrainer (extends RayPPOTrainer)
├── run_depo_qwen2_5_7b.sh    # Reference launch script
└── config/
    └── depo_trainer.yaml     # DEPO config (inherits ppo_trainer)

verl/workers/
└── online_scoring_worker.py  # OnlineScoringWorker + DualHeadScoringModel
```

> **No upstream files are modified.**  `RayDEPOTrainer` inherits from `RayPPOTrainer` directly and overrides only `fit()`, `_save_checkpoint()`, and `_load_checkpoint()`.

## Quick Start

### 1. Prepare data

DEPO uses the same parquet format as VERL's GRPO trainer.  See `examples/data_preprocess/` for preparation scripts.

### 2. Configure paths

Edit `run_depo_qwen2_5_7b.sh` (or set environment variables):

```bash
export MODEL_PATH=/path/to/Qwen2.5-7B-Instruct
export SCORER_PATH=/path/to/distilbert-base-uncased   # any small BERT encoder
export TRAIN_DATA="['/path/to/train.parquet']"
export VAL_DATA="['/path/to/test.parquet']"
export CKPTS_DIR=./checkpoints/depo_run
```

### 3. Launch

```bash
# Start Ray cluster first (or use an existing one)
ray start --head --num-gpus 8

# Run DEPO with DAPO features (recommended for math tasks)
bash recipe/depo/run_depo_qwen2_5_7b.sh

# Run pure DEPO (standard GRPO clip, no overlong buffer)
USE_DAPO=false bash recipe/depo/run_depo_qwen2_5_7b.sh

# Or launch via Python directly
python -m recipe.depo.main_depo \
    actor_rollout_ref.model.path=$MODEL_PATH \
    reward_model.model_name_or_path=$SCORER_PATH \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$VAL_DATA" \
    depo.use_dapo=true
```

## `use_dapo` Toggle

| Setting | Clip | Overlong buffer | Use case |
|---------|------|-----------------|----------|
| `depo.use_dapo=true` (default) | Asymmetric `[clip_ratio_low, clip_ratio_high]` | ✅ enabled | Math / reasoning tasks |
| `depo.use_dapo=false` | Symmetric (standard GRPO) | ❌ disabled | General tasks |

## Key Hyper-parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data.online_rl.warmup_steps` | 100 | Steps before filtering activates (scorer warm-up) |
| `data.online_rl.filter_thresholds` | [0.15, 0.85] | Difficulty window |
| `reward_model.w_distill` | 0.5 | KD loss weight |
| `reward_model.w_ranking` | 3.0 | Ranking loss weight |
| `reward_model.ranking_threshold` | 0.05 | Min difficulty gap for ranking constraint |
| `reward_model.ranking_margin` | 0.5 | Ranking loss margin |
| `reward_model.learning_rate` | 2e-4 | Scorer optimizer LR |

## Monitoring

All scorer metrics are logged under the `training/score_model/` prefix in WandB / console:

- `training/score_model/total_loss` — combined scorer loss
- `training/score_model/target_mean` — mean of actual mean@k labels in the batch
- `training/score_model/calib_mae` — calibration error (predicted vs. actual difficulty)
- `training/filter_survival_rate` — fraction of prompts surviving the difficulty filter
- `training/score_model/raw_predicted_mean` — mean predicted difficulty before filtering

## Requirements

DEPO has no additional dependencies beyond VERL's standard requirements.  The online scorer runs inside a standard `OnlineScoringWorker` (a VERL `Worker` subclass) and requires 1 GPU allocated to the `RewardModel` role.
