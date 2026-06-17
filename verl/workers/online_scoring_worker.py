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
Online Scoring Worker for DEPO (Difficulty-Estimated Policy Optimization).

This module implements the OnlineScoringWorker, which hosts a lightweight
DualHeadScoringModel that is co-trained alongside the policy model.
The scorer predicts prompt difficulty (mean@k accuracy) and is used to
filter training batches to an appropriate difficulty range.

Architecture:
    DualHeadScoringModel:
        - Backbone: small BERT-family encoder (e.g., distilbert-base-uncased)
        - score_head:   predicts difficulty score via BCE (target = mean@k accuracy)
        - distill_head: predicts policy log-prob via BCE (knowledge distillation)

Training loss (per step):
    L = BCE(score_head, mean_k) + w_distill * BCE(distill_head, exp(avg_logprob))
      + w_ranking * ranking_loss(score_head, mean_k, threshold, margin)
"""

import json
import os
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoConfig, AutoModel, AutoTokenizer, PreTrainedModel

from verl.single_controller.base.decorator import Dispatch, register
from verl.single_controller.base.worker import Worker
from verl.utils.logger import print_rank_0


# ===========================================================================
# Model Architecture
# ===========================================================================


class ResidualBlock(nn.Module):
    """A single residual block with LayerNorm and GELU activation."""

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gelu(x + self.net(x))


class DualHeadScoringModel(PreTrainedModel):
    """
    A dual-head scoring model built on top of a BERT-family backbone.

    Outputs:
        score_logit:   raw logit for difficulty score prediction
        distill_logit: raw logit for policy log-prob distillation
    """

    config_class = AutoConfig

    def __init__(self, config):
        super().__init__(config)

        self.bert = AutoModel.from_config(config)
        hidden_size = config.hidden_size

        # Width expansion
        self.input_proj = nn.Sequential(
            nn.Linear(hidden_size, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
        )

        # Depth residual refinement
        self.res_refiner = nn.Sequential(
            ResidualBlock(1024),
            ResidualBlock(1024),
        )

        # Feature compression
        self.bottleneck = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # Dual output heads
        self.score_head = nn.Linear(512, 1)
        self.distill_head = nn.Linear(512, 1)

        self._init_custom_weights()

    def _init_custom_weights(self):
        for module in [self.input_proj, self.res_refiner, self.bottleneck]:
            module.apply(self._normal_init)
        for head in [self.score_head, self.distill_head]:
            head.apply(self._near_zero_init)

    @staticmethod
    def _normal_init(module: nn.Module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()

    @staticmethod
    def _near_zero_init(module: nn.Module):
        """Near-zero init ensures initial output is close to 0.5 (sigmoid)."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.001)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Take CLS token representation
        cls_repr = outputs.last_hidden_state[:, 0, :]

        features = self.input_proj(cls_repr)
        features = self.res_refiner(features)
        shared = self.bottleneck(features)

        score_logit = self.score_head(shared).squeeze(-1)
        distill_logit = self.distill_head(shared).squeeze(-1)

        return score_logit, distill_logit


# ===========================================================================
# Worker
# ===========================================================================


class OnlineScoringWorker(Worker):
    """
    Ray Worker that hosts a DualHeadScoringModel co-trained with the policy.

    Lifecycle:
        1. init_model()  — called once after Ray actor is created
        2. predict()     — called before each training batch to score prompts
        3. update()      — called after each training batch to update the scorer
        4. save/load_checkpoint() — called by the trainer for persistence
    """

    def __init__(self, config: Dict):
        print_rank_0(
            f"[PID:{os.getpid()}] OnlineScoringWorker initializing "
            f"(micro_bs={config.get('micro_batch_size_per_gpu', 4)}, "
            f"predict_bs={config.get('predict_batch_size', 16)})"
        )
        self.config = config
        self.device: torch.device = None
        self.model: DualHeadScoringModel = None
        self.tokenizer = None
        self.optimizer = None

        self.train_batch_size: int = self.config.get("micro_batch_size_per_gpu", 4)
        self.predict_batch_size: int = self.config.get("predict_batch_size", 16)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """Initialize the scoring model from a pretrained backbone or a saved checkpoint."""
        self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
        model_path = self.config["model_name_or_path"]
        print_rank_0(f"[OnlineScoringWorker] Loading model from: {model_path}")

        is_custom_checkpoint = self._is_custom_checkpoint(model_path)

        if is_custom_checkpoint:
            print_rank_0("[OnlineScoringWorker] MODE: RESUME — loading full custom checkpoint.")
            self.model = DualHeadScoringModel.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
            ).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        else:
            print_rank_0("[OnlineScoringWorker] MODE: SCRATCH — loading backbone + random heads.")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            backbone_config = AutoConfig.from_pretrained(model_path)
            self.model = DualHeadScoringModel(backbone_config)
            self.model.bert = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16)
            self.model.to(torch.bfloat16).to(self.device)

        self._fix_tokenizer()
        self._build_optimizer()

        if is_custom_checkpoint:
            self._load_optimizer(model_path)

        print_rank_0("[OnlineScoringWorker] Initialization complete.")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def predict(self, prompts: List[str]) -> List[float]:
        """Score a list of prompts, returning predicted difficulty in [0, 1]."""
        self.model.eval()
        all_scores: List[float] = []
        try:
            with torch.no_grad():
                for start in range(0, len(prompts), self.predict_batch_size):
                    batch_prompts = prompts[start : start + self.predict_batch_size]
                    inputs = self.tokenizer(
                        batch_prompts, padding=True, truncation=True, max_length=512, return_tensors="pt"
                    ).to(self.device)
                    score_logit, _ = self.model(inputs["input_ids"], inputs["attention_mask"])
                    all_scores.extend(torch.sigmoid(score_logit).cpu().tolist())
        finally:
            self.model.train()
        return all_scores

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def update(
        self,
        relevant_prompts: List[str],
        actual_mean_k: torch.Tensor,
        avg_log_probs: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Update the scorer using the training batch outcomes.

        Args:
            relevant_prompts: Raw prompt texts that passed the difficulty filter.
            actual_mean_k:    Ground-truth mean@k accuracy for each prompt (shape: [N]).
            avg_log_probs:    Per-prompt average log-prob from the policy rollout (shape: [N]).

        Returns:
            Dictionary of scalar metrics for logging.
        """
        self.model.train()
        num_samples = len(relevant_prompts)
        if num_samples == 0:
            return {}

        actual_mean_k = actual_mean_k.float().cpu()
        # Convert log-probs to probability space for BCE distillation target
        distill_target = torch.clamp(torch.exp(avg_log_probs.float().cpu()), min=1e-4, max=1.0 - 1e-4)

        # --- Phase 1: Full-batch inference for ranking gradient (no grad, on GPU) ---
        all_logits_list = []
        with torch.no_grad():
            for start in range(0, num_samples, self.predict_batch_size):
                batch_prompts = relevant_prompts[start : start + self.predict_batch_size]
                inputs = self.tokenizer(
                    batch_prompts, padding=True, truncation=True, max_length=512, return_tensors="pt"
                ).to(self.device)
                score_logit, _ = self.model(inputs["input_ids"], inputs["attention_mask"])
                all_logits_list.append(score_logit.float().cpu())

        full_logits = torch.cat(all_logits_list, dim=0)  # [N]

        # --- Phase 2: Compute ranking loss gradient on CPU ---
        ranking_threshold = self.config.get("ranking_threshold", 0.1)
        ranking_margin = self.config.get("ranking_margin", 0.5)
        w_ranking = self.config.get("w_ranking", 3.0)

        full_logits_with_grad = full_logits.detach().requires_grad_(True)
        global_rank_grads = torch.zeros_like(full_logits)
        loss_rank_value = 0.0

        # Build pairwise mask: pairs where t_i > t_j + threshold
        scores_i = actual_mean_k.unsqueeze(1)   # [N, 1]
        scores_j = actual_mean_k.unsqueeze(0)   # [1, N]
        logits_i = full_logits_with_grad.unsqueeze(1)
        logits_j = full_logits_with_grad.unsqueeze(0)
        pair_mask = (scores_i > scores_j + ranking_threshold)

        if pair_mask.any():
            rank_loss = torch.relu(ranking_margin - (logits_i - logits_j)) * pair_mask.float()
            final_rank_loss = rank_loss.sum() / (pair_mask.sum() + 1e-8)
            loss_rank_value = final_rank_loss.item()
            final_rank_loss.backward()
            global_rank_grads = full_logits_with_grad.grad.clone()

        # --- Phase 3: Mini-batch gradient accumulation ---
        self.optimizer.zero_grad()
        w_distill = self.config.get("w_distill", 0.5)
        num_steps = (num_samples + self.train_batch_size - 1) // self.train_batch_size

        accum_total_loss = 0.0
        accum_score_loss = 0.0
        accum_distill_loss = 0.0
        accum_mae = 0.0

        for start in range(0, num_samples, self.train_batch_size):
            end = min(start + self.train_batch_size, num_samples)
            batch_prompts = relevant_prompts[start:end]

            target_score = actual_mean_k[start:end].to(self.device, dtype=torch.bfloat16)
            target_distill = distill_target[start:end].to(self.device, dtype=torch.bfloat16)
            batch_rank_grads = global_rank_grads[start:end].to(self.device, dtype=torch.bfloat16)

            inputs = self.tokenizer(
                batch_prompts, padding=True, truncation=True, max_length=512, return_tensors="pt"
            ).to(self.device)
            score_logit, distill_logit = self.model(inputs["input_ids"], inputs["attention_mask"])

            loss_score = F.binary_cross_entropy_with_logits(score_logit, target_score)
            loss_distill = F.binary_cross_entropy_with_logits(distill_logit, target_distill)
            local_loss = loss_score + w_distill * loss_distill

            # Inject ranking gradient as a surrogate loss term
            surrogate = local_loss
            if w_ranking > 0:
                surrogate = local_loss + w_ranking * torch.sum(score_logit * batch_rank_grads.detach())

            (surrogate / num_steps).backward()

            accum_total_loss += local_loss.item()
            accum_score_loss += loss_score.item()
            accum_distill_loss += loss_distill.item()
            accum_mae += torch.abs(torch.sigmoid(score_logit) - target_score).mean().item()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return {
            "loss": accum_total_loss / num_steps,
            "loss_score_bce": accum_score_loss / num_steps,
            "loss_distill_bce": accum_distill_loss / num_steps,
            "loss_rank": loss_rank_value,
            "calib_mae": accum_mae / num_steps,
            "std": torch.sigmoid(full_logits).std().item(),
        }

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, checkpoint_dir: str) -> None:
        """Save model, tokenizer, and optimizer state in HuggingFace format."""
        print_rank_0(f"[OnlineScoringWorker] Saving checkpoint to {checkpoint_dir}")
        self.model.save_pretrained(checkpoint_dir, safe_serialization=True)
        if self.tokenizer:
            self.tokenizer.save_pretrained(checkpoint_dir)
        torch.save(self.optimizer.state_dict(), os.path.join(checkpoint_dir, "optimizer.pt"))

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, checkpoint_dir: str) -> None:
        """Load model, tokenizer, and optimizer state from a HuggingFace checkpoint."""
        if not os.path.isdir(checkpoint_dir):
            print_rank_0(f"[OnlineScoringWorker] Checkpoint not found at {checkpoint_dir}, skipping.")
            return
        print_rank_0(f"[OnlineScoringWorker] Loading checkpoint from {checkpoint_dir}")
        self.model = DualHeadScoringModel.from_pretrained(
            checkpoint_dir, torch_dtype=torch.bfloat16
        ).to(self.device)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        except Exception:
            print_rank_0("[OnlineScoringWorker] Tokenizer not found in checkpoint, skipping.")
        self._load_optimizer(checkpoint_dir)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _is_custom_checkpoint(self, model_path: str) -> bool:
        """Determine whether model_path is one of our saved checkpoints."""
        config_file = os.path.join(model_path, "config.json")
        if not (os.path.isdir(model_path) and os.path.exists(config_file)):
            return False
        try:
            with open(config_file) as fp:
                cfg_data = json.load(fp)
            if "DualHeadScoringModel" in cfg_data.get("architectures", []):
                return True
            if os.path.exists(os.path.join(model_path, "optimizer.pt")):
                return True
        except Exception as err:
            print_rank_0(f"[OnlineScoringWorker] Warning during config check: {err}")
        return False

    def _fix_tokenizer(self):
        """Ensure right-padding and a valid pad token."""
        if self.tokenizer.padding_side == "left":
            self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _build_optimizer(self):
        """Build AdamW with differential learning rates for backbone vs. heads."""
        base_lr = self.config.get("learning_rate", 2e-4)
        self.optimizer = AdamW(
            [
                {"params": self.model.bert.parameters(), "lr": base_lr * 0.2},
                {"params": self.model.input_proj.parameters(), "lr": base_lr},
                {"params": self.model.res_refiner.parameters(), "lr": base_lr},
                {"params": self.model.bottleneck.parameters(), "lr": base_lr},
                {"params": self.model.score_head.parameters(), "lr": base_lr},
                {"params": self.model.distill_head.parameters(), "lr": base_lr},
            ],
            lr=base_lr,
        )

    def _load_optimizer(self, checkpoint_dir: str):
        """Load optimizer state if available."""
        opt_path = os.path.join(checkpoint_dir, "optimizer.pt")
        if os.path.exists(opt_path):
            print_rank_0("[OnlineScoringWorker] Loading optimizer state...")
            try:
                self.optimizer.load_state_dict(torch.load(opt_path, map_location=self.device))
            except Exception as err:
                print_rank_0(f"[OnlineScoringWorker] Failed to load optimizer: {err}. Starting fresh.")
