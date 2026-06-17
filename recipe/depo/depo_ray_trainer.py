# Copyright 2024 Alibaba Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
DEPO Ray Trainer — Difficulty-Estimated Policy Optimization.

DEPO extends GRPO / DAPO with an online difficulty filter:
  - A lightweight DualHeadScoringModel (hosted in OnlineScoringWorker) predicts
    how likely a prompt is to produce mixed correct/incorrect responses (mean@k).
  - Each training step only uses prompts whose predicted difficulty falls within
    the configured [min_thresh, max_thresh] window.
  - After the PPO update the scorer is updated with the actual mean@k labels and
    policy log-probs (BCE + knowledge distillation + pairwise ranking loss).

Config toggles (set in depo_trainer.yaml or via CLI):
  depo.use_dapo        (bool, default True)
      True  → asymmetric clip ratios + overlong_buffer (DAPO features on top of DEPO)
      False → standard GRPO clip, no overlong_buffer (pure DEPO)

Inheritance chain (all upstream, nothing modified):
  RayPPOTrainer  ←  RayDEPOTrainer
"""

import os
import uuid
from collections import defaultdict
from pprint import pprint

import numpy as np
import ray
import torch
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.metric_utils import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    RayPPOTrainer,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.trainer.ppo.reward import compute_reward
from verl.utils.fs import local_mkdir_safe
from verl.utils.logger import print_rank_0
from verl.utils.metric import reduce_metrics
from verl.utils.profiler import marked_timer


class RayDEPOTrainer(RayPPOTrainer):
    """
    DEPO trainer: adds online difficulty filtering and scorer co-training on top of RayPPOTrainer.

    Key additions vs. the base trainer:
      1. _filtered_batch_generator() — yields batches whose prompts are within the
         target difficulty window as predicted by OnlineScoringWorker.
      2. fit() — replaces the base training loop, calling the filtered generator and
         running the scorer update after each PPO step.
      3. _save_checkpoint() / _load_checkpoint() — extend the base methods to
         persist/restore the scorer alongside the actor checkpoint.
      4. _compute_mean_k() — aggregates per-trajectory accuracy into per-prompt
         mean@k labels used as scorer training targets.
    """

    # ------------------------------------------------------------------
    # Checkpoint overrides
    # ------------------------------------------------------------------

    def _save_checkpoint(self):
        """Save actor (+ critic if used) and the online scorer."""
        super()._save_checkpoint()

        if not self._depo_enabled():
            return

        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )
        scorer_ckpt_path = os.path.join(local_global_step_folder, "score_model")
        local_mkdir_safe(scorer_ckpt_path)
        print_rank_0(f"[DEPO] Saving scorer checkpoint to {scorer_ckpt_path}")
        ray.get(self.rm_wg.execute_rank_zero_async("rm_save_checkpoint", scorer_ckpt_path))
        print_rank_0("[DEPO] Scorer checkpoint saved.")

    def _load_checkpoint(self):
        """Load actor (+ critic if used) and the online scorer."""
        super()._load_checkpoint()

        if not self._depo_enabled():
            return

        checkpoint_folder = self.config.trainer.default_local_dir
        if not os.path.isabs(checkpoint_folder):
            checkpoint_folder = os.path.join(os.getcwd(), checkpoint_folder)

        from verl.trainer.ppo.ray_trainer import find_latest_ckpt_path
        global_step_folder = find_latest_ckpt_path(checkpoint_folder)

        if global_step_folder is None:
            return

        scorer_ckpt_path = os.path.join(global_step_folder, "score_model")
        scorer_config_path = os.path.join(scorer_ckpt_path, "config.json")
        if os.path.isdir(scorer_ckpt_path) and os.path.exists(scorer_config_path):
            print_rank_0(f"[DEPO] Loading scorer checkpoint from {scorer_ckpt_path}")
            ray.get(self.rm_wg.execute_all_async("rm_load_checkpoint", scorer_ckpt_path))
            print_rank_0("[DEPO] Scorer checkpoint loaded.")
        else:
            print_rank_0(f"[DEPO] No scorer checkpoint at {scorer_ckpt_path}, starting from scratch.")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def fit(self):
        """
        Main DEPO training loop.

        Per epoch:
          1. _filtered_batch_generator() produces batches pre-filtered by the scorer.
          2. For each batch: rollout → reward → advantage → actor update.
          3. After the actor update: update the scorer with actual mean@k labels.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        is_rank_zero = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 1
        self.gen_steps = 0
        self._total_samples_trained = 0
        self._total_samples_scanned = 0

        self._load_checkpoint()

        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        total_progress_bar = tqdm(
            total=self.total_training_steps,
            initial=self.global_steps,
            desc="Total Progress",
            disable=not is_rank_zero,
        )

        last_val_metrics = None

        for epoch in range(self.config.trainer.total_epochs):
            print_rank_0(f"--- Starting Epoch {epoch} ---")

            epoch_progress_bar = tqdm(
                self._filtered_batch_generator(),
                desc=f"Training Epoch {epoch}",
                disable=not is_rank_zero,
                leave=False,
            )

            for batch_dict, relevant_prompts, predicted_scores_kept, filter_stats in epoch_progress_bar:

                # ---- Throughput tracking ----
                num_trained = len(batch_dict["input_ids"])
                num_scanned = filter_stats.get("total_processed", num_trained) if filter_stats else num_trained
                self._total_samples_trained += num_trained
                self._total_samples_scanned += num_scanned

                # ---- Skip non-divisible batches ----
                if num_trained % self.actor_rollout_wg.world_size != 0:
                    print_rank_0(
                        f"[DEPO] Skipping batch of size {num_trained}: not divisible by "
                        f"world_size={self.actor_rollout_wg.world_size}."
                    )
                    continue

                metrics: dict = {}
                timing_raw: dict = {}
                is_depo_active = self._depo_enabled()

                # ---- Filter stats metrics ----
                self._log_filter_stats(metrics, filter_stats, predicted_scores_kept, is_depo_active)
                metrics["training/cumulative_samples_trained"] = self._total_samples_trained
                metrics["training/cumulative_samples_scanned"] = self._total_samples_scanned

                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )
                original_uids = batch.non_tensor_batch["uid"].copy()

                gen_batch = self._get_gen_batch(batch)
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # ---- Generation ----
                    with marked_timer("gen", timing_raw, color="red"):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)

                        if "response_mask" not in gen_batch_output.batch:
                            gen_batch_output.batch["response_mask"] = compute_response_mask(gen_batch_output)

                        # Extract per-sample average log-prob for scorer distillation target
                        if is_depo_active and "rollout_log_probs" in gen_batch_output.batch:
                            log_probs = gen_batch_output.batch["rollout_log_probs"]
                            resp_mask = gen_batch_output.batch["response_mask"]
                            if log_probs.device != resp_mask.device:
                                resp_mask = resp_mask.to(log_probs.device)
                            avg_log_probs = (log_probs * resp_mask).sum(dim=-1) / (resp_mask.sum(dim=-1) + 1e-8)
                            gen_batch_output.batch["avg_log_probs"] = avg_log_probs.detach()

                        timing_raw.update(gen_batch_output.meta_info.get("timing", {}))
                        gen_batch_output.meta_info.pop("timing", None)

                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch:
                        batch.batch["response_mask"] = compute_response_mask(batch)

                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # ---- Reward ----
                    with marked_timer("reward", timing_raw, color="yellow"):
                        reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)
                        if reward_extra_infos_dict is None:
                            reward_extra_infos_dict = {}

                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update(
                                {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                            )
                            if "is_correct" in reward_extra_infos_dict:
                                batch.batch["acc_trace"] = torch.tensor(
                                    reward_extra_infos_dict["is_correct"],
                                    dtype=torch.float32,
                                    device=reward_tensor.device,
                                )

                    # ---- Old log-probs & reference log-probs ----
                    batch = self._recompute_log_probs(batch, metrics, timing_raw)

                    # ---- KL penalty ----
                    with marked_timer("reward_penalty", timing_raw):
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch,
                                kl_ctrl=self.kl_ctrl_in_reward,
                                kl_penalty=self.config.algorithm.kl_penalty,
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                    # ---- Advantage ----
                    with marked_timer("adv", timing_raw, color="brown"):
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    # ---- Critic update ----
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, "pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        metrics.update(reduce_metrics(critic_output.meta_info["metrics"]))

                    # ---- Actor update ----
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = (
                                self.config.actor_rollout_ref.rollout.multi_turn.enable
                            )
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        metrics.update(reduce_metrics(actor_output.meta_info["metrics"]))

                    # ---- Scorer update ----
                    if is_depo_active and relevant_prompts:
                        scorer_metrics = self._update_scorer(
                            batch, reward_extra_infos_dict, original_uids,
                            relevant_prompts, predicted_scores_kept,
                        )
                        metrics.update(scorer_metrics)

                # ---- Validation ----
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, "green"):
                        val_metrics = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # ---- Checkpoint ----
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                ):
                    with marked_timer("save_checkpoint", timing_raw, "green"):
                        self._save_checkpoint()

                # ---- Metrics ----
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                logger.log(data=metrics, step=self.global_steps)

                total_progress_bar.update(1)
                self.global_steps += 1

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    epoch_progress_bar.close()
                    return

            print_rank_0(f"--- Finished Epoch {epoch}. Global steps: {self.global_steps} ---")

    # ------------------------------------------------------------------
    # Difficulty filtering
    # ------------------------------------------------------------------

    def _filtered_batch_generator(self):
        """
        Generator that yields batches filtered to the target difficulty window.

        When DEPO is disabled (use_rm=False or online_rl.enabled=False), this
        passes through the raw dataloader unchanged so the trainer degrades
        gracefully to standard GRPO.

        Yields:
            batch_dict (dict):                   collated batch ready for PPO.
            relevant_prompts (list[str] | None): prompt texts in this batch.
            predicted_scores_kept (Tensor | None): scorer predictions for logging.
            filter_stats (dict):                 filtering statistics.
        """
        if not self._depo_enabled():
            print_rank_0("[DEPO] Online filtering disabled — using standard dataloader.")
            for batch_dict in self.train_dataloader:
                yield batch_dict, None, None, {"total_processed": len(batch_dict["input_ids"])}
            return

        print_rank_0("[DEPO] Online filtering ENABLED.")

        target_batch_size = self.config.data.train_batch_size
        warmup_steps = self.config.data.online_rl.get("warmup_steps", 0)
        min_thresh, max_thresh = self.config.data.online_rl.filter_thresholds

        accumulated_samples: list = []
        accumulated_prompts: list = []
        accumulated_scores: list = []
        cumulative_scanned = 0
        last_yield_scanned = 0
        raw_scores_buffer: list = []

        for raw_batch in self.train_dataloader:
            prompts_text = raw_batch["full_prompts"].tolist()
            num_in_batch = len(prompts_text)

            # Score all prompts in this mini-batch
            all_worker_scores = self.rm_wg.predict(prompts_text)
            predicted_scores = torch.tensor(all_worker_scores[0])
            raw_scores_buffer.append(predicted_scores.detach().cpu())

            # During warm-up: accept everything (scorer not yet calibrated)
            if self.global_steps <= warmup_steps:
                keep_mask = torch.ones(num_in_batch, dtype=torch.bool)
            else:
                keep_mask = (predicted_scores >= min_thresh) & (predicted_scores <= max_thresh)

            for idx in range(num_in_batch):
                cumulative_scanned += 1
                if not keep_mask[idx]:
                    continue

                sample = {k: v[idx] for k, v in raw_batch.items()}
                accumulated_samples.append(sample)
                accumulated_prompts.append(prompts_text[idx])
                accumulated_scores.append(predicted_scores[idx])

                if len(accumulated_samples) >= target_batch_size:
                    delta_scanned = cumulative_scanned - last_yield_scanned
                    combined_raw = torch.cat(raw_scores_buffer)

                    yield (
                        self.collate_fn(accumulated_samples[:target_batch_size]),
                        accumulated_prompts[:target_batch_size],
                        torch.stack(accumulated_scores[:target_batch_size]),
                        {
                            "total_processed": delta_scanned,
                            "survival_rate": target_batch_size / delta_scanned,
                            "raw_stats": {
                                "raw_mean": combined_raw.mean().item(),
                                "raw_std": combined_raw.std().item(),
                                "raw_max": combined_raw.max().item(),
                                "raw_min": combined_raw.min().item(),
                            },
                        },
                    )

                    accumulated_samples = accumulated_samples[target_batch_size:]
                    accumulated_prompts = accumulated_prompts[target_batch_size:]
                    accumulated_scores = accumulated_scores[target_batch_size:]
                    last_yield_scanned = cumulative_scanned
                    raw_scores_buffer = []

        # Handle remaining samples at epoch end
        if accumulated_samples:
            final_size = len(accumulated_samples)  # size before padding (for survival_rate)
            num_workers = self.actor_rollout_wg.world_size
            remainder = final_size % num_workers
            if remainder != 0:
                padding_count = num_workers - remainder
                print_rank_0(
                    f"[DEPO] Padding final batch: {final_size} → {final_size + padding_count} samples."
                )
                pad_indices = np.random.choice(final_size, padding_count, replace=True)
                for pad_idx in pad_indices:
                    accumulated_samples.append(accumulated_samples[pad_idx])
                    accumulated_prompts.append(accumulated_prompts[pad_idx])
                    accumulated_scores.append(accumulated_scores[pad_idx])

            print_rank_0(f"[DEPO] Yielding final batch of size {len(accumulated_samples)}.")
            delta_scanned = cumulative_scanned - last_yield_scanned
            yield (
                self.collate_fn(accumulated_samples),
                accumulated_prompts,
                torch.stack(accumulated_scores),
                {
                    "total_processed": delta_scanned,
                    "survival_rate": final_size / delta_scanned if delta_scanned > 0 else 1.0,
                    "raw_stats": {},
                },
            )

    # ------------------------------------------------------------------
    # Scorer update helpers
    # ------------------------------------------------------------------

    def _update_scorer(
        self,
        batch: DataProto,
        reward_extra_infos_dict: dict,
        original_uids: np.ndarray,
        relevant_prompts: list,
        predicted_scores_kept,
    ) -> dict:
        """
        Compute mean@k labels and avg log-probs, then update the scorer.

        Returns a dict of scorer metrics prefixed with "training/score_model/".
        """
        actual_mean_k = self._compute_mean_k(batch, reward_extra_infos_dict, original_uids)
        avg_lp_per_prompt = self._aggregate_avg_log_probs(batch, original_uids)

        assert len(relevant_prompts) == len(actual_mean_k) == len(avg_lp_per_prompt), (
            f"Length mismatch: prompts={len(relevant_prompts)}, "
            f"mean_k={len(actual_mean_k)}, avg_lp={len(avg_lp_per_prompt)}"
        )

        raw_results = self.rm_wg.update(relevant_prompts, actual_mean_k, avg_lp_per_prompt)
        result = raw_results[0] if isinstance(raw_results, list) else raw_results

        metrics = {
            "training/score_model/total_loss": result.get("loss", 0),
            "training/score_model/score_loss": result.get("loss_score_bce", 0),
            "training/score_model/distill_loss": result.get("loss_distill_bce", 0),
            "training/score_model/ranking_loss": result.get("loss_rank", 0),
            "training/score_model/pred_std": result.get("std", 0),
            "training/score_model/target_mean": actual_mean_k.mean().item(),
            "training/score_model/target_std": actual_mean_k.std().item(),
            "training/score_model/target_min": actual_mean_k.min().item(),
            "training/score_model/target_max": actual_mean_k.max().item(),
            "training/score_model/avg_log_prob": avg_lp_per_prompt.mean().item(),
        }
        if "calib_mae" in result:
            metrics["training/score_model/calib_mae"] = result["calib_mae"]

        return metrics

    def _compute_mean_k(
        self,
        batch: DataProto,
        extra_infos: dict,
        original_uids: np.ndarray,
    ) -> torch.Tensor:
        """
        Aggregate per-trajectory accuracy into per-prompt mean@k scores.

        Uses ``acc_trace`` (written during reward computation) if available,
        otherwise falls back to the ``is_correct`` field in extra_infos.
        """
        if "acc_trace" in batch.batch:
            acc_tensor = batch.batch["acc_trace"]
            shuffled_uids = batch.non_tensor_batch["uid"]

            uid_to_acc: dict = defaultdict(list)
            for uid, acc in zip(shuffled_uids, acc_tensor.detach().float().cpu().tolist()):
                uid_to_acc[uid].append(acc)

            mean_k_list = []
            for uid in original_uids:
                accs = uid_to_acc.get(uid, [0.0])
                mean_k_list.append(sum(accs) / len(accs))
            # Return bfloat16 on same device as acc_tensor (consistent with original)
            return torch.tensor(mean_k_list, device=acc_tensor.device, dtype=torch.bfloat16)

        # Fallback: use is_correct from extra_infos
        if "is_correct" in extra_infos:
            is_correct = torch.tensor(extra_infos["is_correct"], dtype=torch.float32)
            n = self.config.actor_rollout_ref.rollout.n
            return is_correct.view(-1, n).mean(dim=-1)

        return torch.zeros(len(original_uids), dtype=torch.float32)

    def _aggregate_avg_log_probs(self, batch: DataProto, original_uids: np.ndarray) -> torch.Tensor:
        """
        Re-aggregate per-trajectory avg log-probs into per-prompt averages,
        correctly handling any shuffling that happened during the PPO update.
        """
        if "avg_log_probs" not in batch.batch:
            return torch.zeros(len(original_uids), dtype=torch.float32)

        shuffled_lps = batch.batch["avg_log_probs"].detach().float().cpu()
        shuffled_uids = batch.non_tensor_batch["uid"]

        uid_to_lp: dict = defaultdict(list)
        for uid, lp in zip(shuffled_uids, shuffled_lps.tolist()):
            uid_to_lp[uid].append(lp)

        avg_lp_list = [
            sum(uid_to_lp[uid]) / len(uid_to_lp[uid]) if uid in uid_to_lp else 0.0
            for uid in original_uids
        ]
        return torch.tensor(avg_lp_list, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _depo_enabled(self) -> bool:
        """Return True when the online scoring / filtering is active."""
        return self.use_rm and self.config.data.get("online_rl", {}).get("enabled", False)

    def _recompute_log_probs(self, batch: DataProto, metrics: dict, timing_raw: dict) -> DataProto:
        """Recompute old log-probs and optionally reference log-probs."""
        batch.batch["response_mask"] = compute_response_mask(batch)

        with marked_timer("old_log_prob", timing_raw, "blue"):
            from verl.trainer.ppo.core_algos import agg_loss

            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
            entropys = old_log_prob.batch["entropys"]
            response_masks = batch.batch["response_mask"]
            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
            entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
            metrics["actor/entropy"] = entropy_agg.detach().item()
            old_log_prob.batch.pop("entropys")
            batch = batch.union(old_log_prob)

        if self.use_reference_policy:
            with marked_timer("ref", timing_raw, "olive"):
                if not self.ref_in_actor:
                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                else:
                    ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                batch = batch.union(ref_log_prob)

        return batch

    def _log_filter_stats(
        self,
        metrics: dict,
        filter_stats: dict | None,
        predicted_scores_kept,
        is_depo_active: bool,
    ):
        """Write difficulty filter statistics into the metrics dict."""
        if not is_depo_active or not filter_stats:
            return

        if "survival_rate" in filter_stats:
            metrics["training/filter_survival_rate"] = filter_stats["survival_rate"]

        raw_stats = filter_stats.get("raw_stats", {})
        if raw_stats:
            metrics["training/score_model/raw_predicted_mean"] = raw_stats["raw_mean"]
            metrics["training/score_model/raw_predicted_std"] = raw_stats["raw_std"]
            metrics["training/score_model/raw_predicted_max"] = raw_stats["raw_max"]
            metrics["training/score_model/raw_predicted_min"] = raw_stats["raw_min"]

        if predicted_scores_kept is not None:
            metrics["training/score_model_predicted_mean"] = predicted_scores_kept.mean().item()
            metrics["training/score_model_predicted_std"] = predicted_scores_kept.std().item()
            metrics["training/score_model_predicted_min"] = predicted_scores_kept.min().item()
            metrics["training/score_model_predicted_max"] = predicted_scores_kept.max().item()

        warmup_steps = self.config.data.online_rl.get("warmup_steps", 0)
        if self.global_steps <= warmup_steps:
            metrics["training/score_model_warmup_active"] = 1.0
