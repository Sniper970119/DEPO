#!/usr/bin/env bash
# =============================================================================
# DEPO Training Script — Qwen2.5-7B-Instruct (reference config, 2 nodes × 8 GPUs)
# =============================================================================
# Usage:
#   bash recipe/depo/run_depo_qwen2_5_7b.sh
#
# Key environment variables you can override:
#   MODEL_PATH        Path to the policy model (Qwen2.5-7B-Instruct or similar)
#   SCORER_PATH       Path to the scorer backbone (distilbert-base-uncased or similar)
#   TRAIN_DATA        Path to training parquet file(s)
#   VAL_DATA          Path to validation parquet file(s)
#   CKPTS_DIR         Where to save checkpoints
#   NNODES            Number of nodes (default: 2)
#   USE_DAPO          Whether to enable DAPO-style features (default: true)
# =============================================================================
set -x

# ---- Experiment identity ----
project_name='verl-depo'
exp_name='depo_qwen2_5_7b'

# ---- Paths ----
MODEL_PATH=${MODEL_PATH:-"/path/to/Qwen2.5-7B-Instruct"}
SCORER_PATH=${SCORER_PATH:-"/path/to/distilbert-base-uncased"}
CKPTS_DIR=${CKPTS_DIR:-"./checkpoints/${exp_name}"}

TRAIN_DATA=${TRAIN_DATA:-"['/path/to/train.parquet']"}
VAL_DATA=${VAL_DATA:-"['/path/to/test.parquet']"}

# ---- DAPO / DEPO toggle ----
USE_DAPO=${USE_DAPO:-true}

# ---- Sequence lengths ----
max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 10))

# ---- DEPO difficulty filter ----
warmup_steps=100
filter_min=0.15
filter_max=0.85

# ---- Scorer hyper-parameters ----
scorer_lr=2e-4
w_distill=0.5
w_ranking=3.0
ranking_threshold=0.05
ranking_margin=0.5

# ---- Training batch sizes ----
train_prompt_bsz=128
gen_prompt_bsz=128
n_resp_per_prompt=8
train_prompt_mini_bsz=64

# ---- Rollout algorithm (GRPO) ----
adv_estimator=grpo
temperature=1.0
top_p=1.0
top_k=-1           # -1 for vLLM
val_top_p=0.7

# ---- DAPO-specific clip (only meaningful when USE_DAPO=true) ----
clip_ratio_low=0.2
clip_ratio_high=0.28
enable_overlong_buffer=true
overlong_buffer_len=$((1024 * 4))
overlong_penalty_factor=1.0

# ---- Performance ----
sp_size=2
use_dynamic_bsz=true
actor_ppo_max_token_len=$((max_prompt_length + max_response_length))
offload=true
gen_tp=2

# ---- Ray ----
RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
WORKING_DIR=${WORKING_DIR:-"${PWD}"}
NNODES=${NNODES:-2}

python3 -m recipe.depo.main_depo \
    depo.use_dapo=${USE_DAPO} \
    data.train_files="${TRAIN_DATA}" \
    data.val_files="${VAL_DATA}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.val_max_samples=200 \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.train_batch_size=${train_prompt_bsz} \
    data.filter_overlong_prompts=false \
    data.online_rl.enabled=true \
    data.online_rl.warmup_steps=${warmup_steps} \
    data.online_rl.filter_thresholds="[${filter_min}, ${filter_max}]" \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode="token-mean" \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=true \
    actor_rollout_ref.rollout.max_num_batched_tokens=13000 \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k="${top_k}" \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=true \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=false \
    algorithm.kl_ctrl.kl_coef=0.0 \
    algorithm.filter_groups.enable=false \
    reward_model.reward_manager=dapo \
    reward_model.model_name_or_path="${SCORER_PATH}" \
    reward_model.learning_rate=${scorer_lr} \
    reward_model.micro_batch_size_per_gpu=8 \
    reward_model.predict_batch_size=16 \
    reward_model.w_distill=${w_distill} \
    reward_model.w_ranking=${w_ranking} \
    reward_model.ranking_threshold=${ranking_threshold} \
    reward_model.ranking_margin=${ranking_margin} \
    reward_model.overlong_buffer.enable=${enable_overlong_buffer} \
    reward_model.overlong_buffer.len=${overlong_buffer_len} \
    reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor} \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.logger="['wandb']" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=true \
    trainer.test_freq=100 \
    trainer.save_freq=100 \
    trainer.total_epochs=10 \
    trainer.default_local_dir="${CKPTS_DIR}"
