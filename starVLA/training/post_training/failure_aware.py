"""Failure-aware post-training interface.

This module keeps the public post-training knobs in one place and maps them
onto the existing StarVLA trainer, dataloader, and action-head configs.
"""

from __future__ import annotations

import json
import os
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from omegaconf import DictConfig, OmegaConf


DEFAULT_ACTION_DIM_WEIGHTS = "1.0,1.0,1.0,0.7,0.7,0.7,1.5"


@dataclass(frozen=True)
class FailureAwarePostTrainingInputs:
    """External inputs expected by the failure-aware post-training interface."""

    pretrained_checkpoint: str
    failure_log_path: str = ""
    max_train_steps: int = 5000
    action_lr: float = 5.0e-5
    per_device_batch_size: int = 8
    failure_aware_weight: float = 2.0
    failure_aware_top_k: int = 20
    action_loss_dim_weights: str = DEFAULT_ACTION_DIM_WEIGHTS
    action_loss_time_weights: str = ""
    action_loss_early_step_weight: float = 1.0
    action_loss_late_step_weight: float = 1.25
    freeze_modules: str = "qwen_vl_interface"
    reload_modules: str = "action_model"


def normalize_lang(text: Any) -> str:
    """Normalize language instructions for failure-aware matching."""

    return " ".join(str(text).lower().strip().split())


def parse_float_sequence(value: Any) -> list[float] | None:
    """Parse comma, JSON-list, tuple, or list inputs into floats."""

    if value is None or value == "":
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped.lower() in {"none", "null"}:
            return None
        if stripped.startswith("["):
            return [float(v) for v in json.loads(stripped)]
        return [float(v.strip()) for v in stripped.split(",") if v.strip()]
    return [float(v) for v in value]


def loss_weight_vector(config: Any, key: str, expected_len: int, device, dtype) -> torch.Tensor | None:
    """Build a loss-weight vector from an action-head config key."""

    getter = config.get if hasattr(config, "get") else lambda item, default=None: getattr(config, item, default)
    weights = parse_float_sequence(getter(key, None))
    if weights is None:
        return None
    if len(weights) != expected_len:
        raise ValueError(f"{key} must have length {expected_len}, got {len(weights)}: {weights}")
    return torch.tensor(weights, device=device, dtype=dtype)


def weighted_velocity_loss(pred_actions: torch.Tensor, velocity: torch.Tensor, config: Any) -> torch.Tensor:
    """Flow-matching velocity loss with optional action-dim and chunk-time weights."""

    squared_error = (pred_actions - velocity) ** 2
    _, horizon, action_dim = squared_error.shape
    weights = torch.ones((1, horizon, action_dim), device=squared_error.device, dtype=squared_error.dtype)

    dim_weights = loss_weight_vector(
        config,
        "action_loss_dim_weights",
        action_dim,
        squared_error.device,
        squared_error.dtype,
    )
    if dim_weights is not None:
        weights = weights * dim_weights.view(1, 1, action_dim)

    time_weights = loss_weight_vector(
        config,
        "action_loss_time_weights",
        horizon,
        squared_error.device,
        squared_error.dtype,
    )
    getter = config.get if hasattr(config, "get") else lambda item, default=None: getattr(config, item, default)
    if time_weights is None:
        early_weight = float(getter("action_loss_early_step_weight", 1.0))
        late_weight = float(getter("action_loss_late_step_weight", 1.0))
        if early_weight != 1.0 or late_weight != 1.0:
            time_weights = torch.linspace(
                early_weight,
                late_weight,
                steps=horizon,
                device=squared_error.device,
                dtype=squared_error.dtype,
            )
    if time_weights is not None:
        weights = weights * time_weights.view(1, horizon, 1)

    return (squared_error * weights).mean() / weights.mean().clamp_min(1e-8)


def _parse_lang_list(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("["):
            return [str(item) for item in json.loads(stripped)]
        return [part.strip() for part in stripped.split("||") if part.strip()]
    return [str(item) for item in value]


def _load_failure_events(log_path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    path = Path(log_path)
    events = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            events.append(json.loads(line))
    return events


def mine_failure_langs(
    failure_log_path: str = "",
    *,
    top_k: int = 20,
    extra_langs: Iterable[str] | str | None = None,
) -> set[str]:
    """Return normalized hard instructions from manual entries and a failure log."""

    hard_langs = {normalize_lang(lang) for lang in _parse_lang_list(extra_langs)}
    if not failure_log_path:
        return {lang for lang in hard_langs if lang}

    path = Path(failure_log_path)
    if not path.exists():
        print(f"Warning: failure_aware_log_path not found: {path}")
        return {lang for lang in hard_langs if lang}

    counts: Counter[str] = Counter()
    for event in _load_failure_events(path):
        lang = event.get("lang_annotation", "")
        if lang:
            counts[normalize_lang(lang)] += 1
    for lang, _ in counts.most_common(int(top_k)):
        hard_langs.add(lang)
    return {lang for lang in hard_langs if lang}


class FailureAwareSampler:
    """Small reusable accept/reject sampler for hard instruction reweighting."""

    def __init__(
        self,
        *,
        hard_langs: Iterable[str] | None = None,
        weight: float = 1.0,
        seed: int | None = None,
    ):
        self.hard_langs = {normalize_lang(lang) for lang in hard_langs or [] if normalize_lang(lang)}
        self.weight = max(1.0, float(weight))
        self.rng = random.Random(seed)

    @property
    def enabled(self) -> bool:
        return bool(self.hard_langs) and self.weight > 1.0

    def accept(self, sample: dict[str, Any]) -> bool:
        if not self.enabled:
            return True
        lang = normalize_lang(sample.get("lang", ""))
        if lang in self.hard_langs:
            return True
        return self.rng.random() < (1.0 / self.weight)


def build_sampler_from_data_cfg(data_cfg: Any, *, seed: int | None = None) -> FailureAwareSampler:
    """Build a FailureAwareSampler from cfg.datasets.vla_data."""

    if data_cfg is None:
        return FailureAwareSampler(seed=seed)

    getter = data_cfg.get if hasattr(data_cfg, "get") else lambda item, default=None: getattr(data_cfg, item, default)
    hard_langs = mine_failure_langs(
        getter("failure_aware_log_path", ""),
        top_k=int(getter("failure_aware_top_k", 20)),
        extra_langs=getter("failure_aware_langs", None),
    )
    weight = float(getter("failure_aware_weight", 1.0))
    sampler = FailureAwareSampler(hard_langs=hard_langs, weight=weight, seed=seed)
    if sampler.hard_langs:
        print(f"Failure-aware sampling enabled for {len(sampler.hard_langs)} language instructions")
    return sampler


def _set_if_missing(cfg: DictConfig, key: str, value: Any) -> None:
    if cfg.get(key, None) is None:
        cfg[key] = value


def _ensure_post_training_node(cfg: DictConfig) -> DictConfig:
    if cfg.get("post_training", None) is None:
        cfg.post_training = OmegaConf.create({})
    if cfg.post_training.get("failure_aware", None) is None:
        cfg.post_training.failure_aware = OmegaConf.create({"enabled": False})
    return cfg.post_training.failure_aware


def apply_failure_aware_post_training(cfg: DictConfig) -> DictConfig:
    """Expand cfg.post_training.failure_aware onto trainer/dataset/action configs.

    Public input surface:
        cfg.post_training.failure_aware.enabled
        cfg.post_training.failure_aware.pretrained_checkpoint
        cfg.post_training.failure_aware.failure_log_path
        cfg.post_training.failure_aware.max_train_steps
        cfg.post_training.failure_aware.action_lr
        cfg.post_training.failure_aware.per_device_batch_size
        cfg.post_training.failure_aware.failure_aware_weight
        cfg.post_training.failure_aware.failure_aware_top_k
        cfg.post_training.failure_aware.action_loss_dim_weights
        cfg.post_training.failure_aware.action_loss_time_weights
        cfg.post_training.failure_aware.action_loss_early_step_weight
        cfg.post_training.failure_aware.action_loss_late_step_weight
        cfg.post_training.failure_aware.freeze_modules
        cfg.post_training.failure_aware.reload_modules
    """

    post_cfg = _ensure_post_training_node(cfg)
    if not bool(post_cfg.get("enabled", False)):
        return cfg

    trainer_cfg = cfg.trainer
    data_cfg = cfg.datasets.vla_data
    action_cfg = cfg.framework.action_model

    pretrained_checkpoint = post_cfg.get("pretrained_checkpoint", "")
    if pretrained_checkpoint:
        trainer_cfg.pretrained_checkpoint = pretrained_checkpoint

    trainer_cfg.max_train_steps = int(post_cfg.get("max_train_steps", 5000))
    trainer_cfg.freeze_modules = post_cfg.get("freeze_modules", "qwen_vl_interface")
    trainer_cfg.reload_modules = post_cfg.get("reload_modules", "action_model")
    trainer_cfg.learning_rate.action_model = float(post_cfg.get("action_lr", 5.0e-5))
    data_cfg.per_device_batch_size = int(post_cfg.get("per_device_batch_size", 8))

    data_cfg.failure_aware_log_path = post_cfg.get("failure_log_path", "")
    data_cfg.failure_aware_weight = float(post_cfg.get("failure_aware_weight", 2.0))
    data_cfg.failure_aware_top_k = int(post_cfg.get("failure_aware_top_k", 20))
    if post_cfg.get("failure_aware_langs", None) is not None:
        data_cfg.failure_aware_langs = post_cfg.failure_aware_langs

    action_cfg.action_loss_dim_weights = post_cfg.get("action_loss_dim_weights", DEFAULT_ACTION_DIM_WEIGHTS)
    action_cfg.action_loss_time_weights = post_cfg.get("action_loss_time_weights", "")
    action_cfg.action_loss_early_step_weight = float(post_cfg.get("action_loss_early_step_weight", 1.0))
    action_cfg.action_loss_late_step_weight = float(post_cfg.get("action_loss_late_step_weight", 1.25))

    _set_if_missing(trainer_cfg, "num_warmup_steps", 1000)
    return cfg

