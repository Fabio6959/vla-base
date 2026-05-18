# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License").
# Implemented for CALVIN Scene D zero-shot generalization.

"""
两阶段冷启动训练脚本

训练策略：
    Stage 1 (Cold Start, Epoch 0-1):
        - 冻结: Vision Encoder, Qwen, LoRA
        - 训练: Bridge Network, Action Head
        - 学习率: Bridge 1e-4, Action Head 1e-4
    
    Stage 2 (Joint Training, Epoch >= 2):
        - 冻结: Vision Encoder, Qwen 主体
        - 训练: LoRA, Bridge, Action Head
        - 学习率: LoRA 1e-5, Bridge 5e-5, Action Head 1e-4

特性：
    - 自动阶段切换
    - 不同模块使用不同学习率
    - DeepSpeed ZeRO-2 支持
    - 每 0.5 epoch 保存 checkpoint
    - 支持自动 resume
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import torch
import torch.distributed as dist
import wandb
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import get_scheduler

from starVLA.dataloader import build_dataloader
from starVLA.model.framework.base_framework import build_framework
from starVLA.model.framework.share_tools import apply_config_compat
from starVLA.training.trainer_utils.config_tracker import AccessTrackedConfig, wrap_config
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils, normalize_dotlist_args

logger = get_logger(__name__)

# DeepSpeed 配置
deepspeed_plugin = DeepSpeedPlugin()
accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin)
accelerator.print(accelerator.state)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# =============================================================================
# 两阶段训练配置
# =============================================================================

STAGE1_EPOCHS = 2  # Stage 1 持续的 epoch 数 (Epoch 0, 1)
STAGE2_START_EPOCH = 2  # Stage 2 开始的 epoch

# 学习率配置
STAGE1_LR = {
    "bridge": 1e-4,
    "action_model": 1e-4,
}

STAGE2_LR = {
    "lora": 1e-5,
    "bridge": 5e-5,
    "action_model": 1e-4,
}


def get_current_stage(epoch: int) -> int:
    """
    根据当前 epoch 确定训练阶段
    
    Args:
        epoch: 当前 epoch
    
    Returns:
        训练阶段 (1 或 2)
    """
    if epoch < STAGE2_START_EPOCH:
        return 1
    else:
        return 2


def get_stage_name(stage: int) -> str:
    """获取阶段名称"""
    return "Stage 1 (Cold Start)" if stage == 1 else "Stage 2 (Joint Training)"


def setup_directories(cfg) -> Path:
    """创建输出目录"""
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    output_dir = Path(cfg.output_dir)
    
    if not dist.is_initialized() or dist.get_rank() == 0:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)
    
    return output_dir


def prepare_data(cfg, accelerator, output_dir) -> DataLoader:
    """准备数据加载器"""
    logger.info(f"Creating VLA Dataset with Mixture `{cfg.datasets.vla_data.data_mix}`")
    vla_train_dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)
    
    accelerator.dataloader_config.dispatch_batches = False
    dist.barrier()
    return vla_train_dataloader


def build_optimizer_for_stage(
    model, 
    stage: int, 
    base_lr: float = 1e-4,
    betas: Tuple[float, float] = (0.9, 0.95),
    weight_decay: float = 0.0,
    eps: float = 1e-8,
) -> torch.optim.Optimizer:
    """
    根据训练阶段构建优化器
    
    Args:
        model: 模型
        stage: 训练阶段 (1 或 2)
        base_lr: 基础学习率
        betas: Adam betas
        weight_decay: 权重衰减
        eps: Adam eps
    
    Returns:
        优化器
    """
    param_groups = []
    
    if stage == 1:
        # Stage 1: 仅训练 Bridge 和 Action Head
        lr_config = STAGE1_LR
        
        # Bridge 参数
        bridge_params = list(model.bridge.parameters())
        if bridge_params:
            param_groups.append({
                "params": bridge_params,
                "lr": lr_config["bridge"],
                "name": "bridge"
            })
        
        # Action Model 参数
        action_params = list(model.action_model.parameters())
        if action_params:
            param_groups.append({
                "params": action_params,
                "lr": lr_config["action_model"],
                "name": "action_model"
            })
        
    else:
        # Stage 2: 训练 LoRA + Bridge + Action Head
        lr_config = STAGE2_LR
        
        # LoRA 参数
        lora_params = []
        for name, param in model.qwen_vl_interface.model.named_parameters():
            if 'lora' in name.lower() and param.requires_grad:
                lora_params.append(param)
        
        if lora_params:
            param_groups.append({
                "params": lora_params,
                "lr": lr_config["lora"],
                "name": "lora"
            })
        
        # Bridge 参数
        bridge_params = list(model.bridge.parameters())
        if bridge_params:
            param_groups.append({
                "params": bridge_params,
                "lr": lr_config["bridge"],
                "name": "bridge"
            })
        
        # Action Model 参数
        action_params = list(model.action_model.parameters())
        if action_params:
            param_groups.append({
                "params": action_params,
                "lr": lr_config["action_model"],
                "name": "action_model"
            })
    
    # 使用 Fused AdamW (如果可用)
    fused_available = torch.cuda.is_available()
    
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=base_lr,
        betas=betas,
        weight_decay=weight_decay,
        eps=eps,
        fused=fused_available,
    )
    
    if dist.is_initialized() and dist.get_rank() == 0:
        for group in optimizer.param_groups:
            logger.info(f"LR Group {group['name']}: lr={group['lr']}, num_params={len(group['params'])}")
    
    return optimizer


class TwoStageVLATrainer(TrainerUtils):
    """
    两阶段 VLA 训练器
    
    支持：
        - 自动阶段切换
        - 不同模块不同学习率
        - 每 0.5 epoch 保存 checkpoint
        - 自动 resume
    """
    
    def __init__(
        self, 
        cfg, 
        model, 
        dataloader, 
        accelerator,
        checkpoint_dir: str = None,
    ):
        self.config = cfg
        self.model = model
        self.dataloader = dataloader
        self.accelerator = accelerator
        
        # 训练状态
        self.completed_steps = 0
        self.current_epoch = 0
        self.current_stage = 1
        self.global_step = 0
        
        # Checkpoint 目录
        self.checkpoint_dir = checkpoint_dir or os.path.join(cfg.output_dir, "checkpoints")
        
        # 计算每个 epoch 的步数
        self.steps_per_epoch = len(dataloader)
        self.half_epoch_steps = self.steps_per_epoch // 2
        
        # 计算总 batch size
        self.total_batch_size = (
            cfg.datasets.vla_data.per_device_batch_size
            * accelerator.num_processes
            * accelerator.gradient_accumulation_steps
        )
        
        # 初始化优化器和调度器
        self.optimizer = None
        self.lr_scheduler = None
        
        # 数据迭代器
        self.data_iter = None
        self.epoch_count = 0
    
    def setup_optimizer_and_scheduler(self):
        """设置优化器和学习率调度器"""
        # 根据当前阶段创建优化器
        self.current_stage = get_current_stage(self.current_epoch)
        self.optimizer = build_optimizer_for_stage(
            self.model,
            stage=self.current_stage,
            base_lr=self.config.trainer.learning_rate.base,
            betas=tuple(self.config.trainer.optimizer.betas),
            weight_decay=self.config.trainer.optimizer.weight_decay,
            eps=self.config.trainer.optimizer.eps,
        )
        
        # 创建学习率调度器
        sched_kwargs = self.config.trainer.scheduler_specific_kwargs
        self.lr_scheduler = get_scheduler(
            name=self.config.trainer.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=self.config.trainer.num_warmup_steps,
            num_training_steps=self.config.trainer.max_train_steps,
            scheduler_specific_kwargs=sched_kwargs,
        )
    
    def prepare_training(self):
        """准备训练"""
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = self.config.seed + rank if hasattr(self.config, "seed") else rank + 3047
        set_seed(seed)
        
        # 保存初始配置
        self._save_initial_configs()
        
        # 初始化 checkpoint
        self._init_checkpointing()
        
        # 设置优化器
        self.setup_optimizer_and_scheduler()
        
        # 设置模型训练阶段
        self._set_model_stage(self.current_stage)
        
        # 打印可训练参数
        self.print_trainable_parameters(self.model)
        
        # 分布式训练准备
        self.model, self.optimizer, self.dataloader = self.setup_distributed_training(
            self.accelerator,
            self.model,
            self.optimizer,
            self.dataloader,
        )
        
        # 初始化 wandb
        self._init_wandb()
        
        # 打印训练阶段信息
        self._log_stage_info()
    
    def _set_model_stage(self, stage: int):
        """设置模型的训练阶段"""
        if hasattr(self.model, 'get_trainable_parameters_by_stage'):
            self.model.get_trainable_parameters_by_stage(stage)
    
    def _log_stage_info(self):
        """打印当前训练阶段信息"""
        stage_name = get_stage_name(self.current_stage)
        logger.info("=" * 60)
        logger.info(f"Current Training Stage: {stage_name}")
        logger.info(f"Current Epoch: {self.current_epoch}")
        logger.info(f"Completed Steps: {self.completed_steps}")
        logger.info("=" * 60)
        
        # 打印学习率
        if self.optimizer is not None:
            for group in self.optimizer.param_groups:
                logger.info(f"  {group['name']}: lr = {group['lr']}")
    
    def _save_initial_configs(self):
        """保存初始配置"""
        if not self.accelerator.is_main_process:
            return
        
        output_dir = Path(self.config.output_dir)
        
        if isinstance(self.config, AccessTrackedConfig):
            full_cfg = self.config.unwrap()
        else:
            full_cfg = self.config
        
        full_yaml_path = output_dir / "config.full.yaml"
        OmegaConf.save(full_cfg, full_yaml_path, resolve=True)
        logger.info(f"Full config saved at {full_yaml_path}")
        
        if isinstance(self.config, AccessTrackedConfig):
            self.config.save_accessed_config(output_dir / "config.yaml", use_original_values=False)
    
    def _init_checkpointing(self):
        """初始化 checkpoint"""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        is_resume = getattr(self.config.trainer, "is_resume", False)
        
        if is_resume:
            resume_ckpt, self.completed_steps, self.current_epoch = self._get_latest_checkpoint()
            if resume_ckpt:
                self._load_checkpoint(resume_ckpt)
                logger.info(f"Resuming from checkpoint: {resume_ckpt}")
                logger.info(f"  Epoch: {self.current_epoch}, Steps: {self.completed_steps}")
                return
        
        pretrained_checkpoint = getattr(self.config.trainer, "pretrained_checkpoint", None)
        if pretrained_checkpoint:
            self.model = self.load_pretrained_backbones(
                self.model, pretrained_checkpoint
            )
            logger.info(f"Loaded pretrained checkpoint: {pretrained_checkpoint}")
    
    def _get_latest_checkpoint(self) -> Tuple[Optional[str], int, int]:
        """获取最新的 checkpoint"""
        import re
        
        if not os.path.exists(self.checkpoint_dir):
            return None, 0, 0
        
        checkpoints = []
        for f in os.listdir(self.checkpoint_dir):
            match = re.match(r"epoch_(\d+)_step_(\d+)", f)
            if match:
                epoch = int(match.group(1))
                step = int(match.group(2))
                checkpoints.append((f, epoch, step))
        
        if not checkpoints:
            return None, 0, 0
        
        # 按步数排序
        checkpoints.sort(key=lambda x: x[2], reverse=True)
        latest = checkpoints[0]
        
        return os.path.join(self.checkpoint_dir, latest[0]), latest[2], latest[1]
    
    def _load_checkpoint(self, checkpoint_path: str):
        """加载 checkpoint"""
        # 加载模型权重
        if os.path.isdir(checkpoint_path):
            # DeepSpeed 格式
            self.accelerator.load_state(checkpoint_path)
        else:
            # PyTorch 格式
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            self.model.load_state_dict(state_dict, strict=False)
    
    def _save_checkpoint(self, is_half_epoch: bool = False):
        """保存 checkpoint"""
        if not self.accelerator.is_main_process:
            return
        
        checkpoint_name = f"epoch_{self.current_epoch}_step_{self.completed_steps}"
        if is_half_epoch:
            checkpoint_name += "_half"
        
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        os.makedirs(checkpoint_path, exist_ok=True)
        
        # 保存模型权重
        state_dict = self.accelerator.get_state_dict(self.model)
        torch.save(state_dict, os.path.join(checkpoint_path, "pytorch_model.pt"))
        
        # 保存训练状态
        training_state = {
            "epoch": self.current_epoch,
            "global_step": self.completed_steps,
            "stage": self.current_stage,
        }
        with open(os.path.join(checkpoint_path, "training_state.json"), "w") as f:
            json.dump(training_state, f, indent=2)
        
        # 保存优化器状态
        torch.save(
            self.optimizer.state_dict(),
            os.path.join(checkpoint_path, "optimizer.pt")
        )
        
        # 保存调度器状态
        torch.save(
            self.lr_scheduler.state_dict(),
            os.path.join(checkpoint_path, "scheduler.pt")
        )
        
        logger.info(f"Checkpoint saved at {checkpoint_path}")
    
    def _init_wandb(self):
        """初始化 wandb"""
        if self.accelerator.is_main_process:
            wandb.init(
                name=self.config.run_id,
                dir=os.path.join(self.config.output_dir, "wandb"),
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                group="vla-train",
            )
    
    def _create_data_iterator(self):
        """创建数据迭代器"""
        self.data_iter = iter(self.dataloader)
    
    def _get_next_batch(self):
        """获取下一个 batch"""
        try:
            batch = next(self.data_iter)
        except StopIteration:
            self.epoch_count += 1
            if hasattr(self.dataloader, "sampler") and callable(getattr(self.dataloader.sampler, "set_epoch", None)):
                self.dataloader.sampler.set_epoch(self.epoch_count)
            self.data_iter = iter(self.dataloader)
            batch = next(self.data_iter)
        return batch
    
    def _check_stage_transition(self):
        """检查是否需要切换训练阶段"""
        new_epoch = self.completed_steps // self.steps_per_epoch
        new_stage = get_current_stage(new_epoch)
        
        if new_stage != self.current_stage:
            logger.info(f"\n{'='*60}")
            logger.info(f"Stage Transition: {get_stage_name(self.current_stage)} -> {get_stage_name(new_stage)}")
            logger.info(f"{'='*60}\n")
            
            self.current_stage = new_stage
            self.current_epoch = new_epoch
            
            # 重新创建优化器
            self.optimizer = build_optimizer_for_stage(
                self.model,
                stage=self.current_stage,
                base_lr=self.config.trainer.learning_rate.base,
                betas=tuple(self.config.trainer.optimizer.betas),
                weight_decay=self.config.trainer.optimizer.weight_decay,
                eps=self.config.trainer.optimizer.eps,
            )
            
            # 重新创建调度器
            sched_kwargs = self.config.trainer.scheduler_specific_kwargs
            self.lr_scheduler = get_scheduler(
                name=self.config.trainer.lr_scheduler_type,
                optimizer=self.optimizer,
                num_warmup_steps=self.config.trainer.num_warmup_steps,
                num_training_steps=self.config.trainer.max_train_steps,
                scheduler_specific_kwargs=sched_kwargs,
            )
            
            # 更新模型训练阶段
            self._set_model_stage(self.current_stage)
            
            # 重新准备分布式训练
            self.model, self.optimizer, self.dataloader = self.setup_distributed_training(
                self.accelerator,
                self.model,
                self.optimizer,
                self.dataloader,
            )
            
            # 打印阶段信息
            self._log_stage_info()
        
        elif new_epoch != self.current_epoch:
            self.current_epoch = new_epoch
            self._log_stage_info()
    
    def train(self):
        """执行训练循环"""
        self._log_training_config()
        self._create_data_iterator()
        
        progress_bar = tqdm(
            total=self.config.trainer.max_train_steps,
            initial=self.completed_steps,
            disable=not self.accelerator.is_local_main_process,
        )
        
        # 记录上次保存的步数
        last_save_step = self.completed_steps
        
        while self.completed_steps < self.config.trainer.max_train_steps:
            t_start = time.perf_counter()
            batch = self._get_next_batch()
            t_data = time.perf_counter() - t_start
            
            t_start_model = time.perf_counter()
            metrics = self._train_step(batch)
            t_model = time.perf_counter() - t_start_model
            
            if self.accelerator.sync_gradients:
                progress_bar.update(1)
                self.completed_steps += 1
                
                # 检查阶段切换
                self._check_stage_transition()
            
            # 更新进度条
            if self.accelerator.is_local_main_process:
                progress_bar.set_postfix({
                    "epoch": self.current_epoch,
                    "stage": self.current_stage,
                    "loss": f"{metrics['action_loss']:.4f}",
                    "data_t": f"{t_data:.3f}",
                    "model_t": f"{t_model:.3f}",
                })
            
            # 记录指标
            self._log_metrics(metrics)
            
            # 每 0.5 epoch 保存 checkpoint
            steps_since_save = self.completed_steps - last_save_step
            if steps_since_save >= self.half_epoch_steps and self.completed_steps > 0:
                self._save_checkpoint(is_half_epoch=True)
                last_save_step = self.completed_steps
            
            # 定期保存
            if self.completed_steps % self.config.trainer.save_interval == 0 and self.completed_steps > 0:
                self._save_checkpoint()
            
            if self.completed_steps >= self.config.trainer.max_train_steps:
                break
        
        self._finalize_training()
    
    def _train_step(self, batch) -> dict:
        """单步训练"""
        with self.accelerator.accumulate(self.model):
            self.optimizer.zero_grad()
            
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_dict = self.model.forward(batch)
                action_loss = output_dict["action_loss"]
            
            self.accelerator.backward(action_loss)
            
            if self.config.trainer.gradient_clipping is not None:
                self.accelerator.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.trainer.gradient_clipping
                )
            
            self.optimizer.step()
            
            if self.accelerator.sync_gradients:
                self.lr_scheduler.step()
        
        return {
            "action_loss": action_loss.item(),
        }
    
    def _log_training_config(self):
        """打印训练配置"""
        if self.accelerator.is_main_process:
            logger.info("=" * 60)
            logger.info("Training Configuration")
            logger.info("=" * 60)
            logger.info(f"  Total optimization steps: {self.config.trainer.max_train_steps}")
            logger.info(f"  Steps per epoch: {self.steps_per_epoch}")
            logger.info(f"  Per device batch size: {self.config.datasets.vla_data.per_device_batch_size}")
            logger.info(f"  Total batch size: {self.total_batch_size}")
            logger.info(f"  Gradient accumulation steps: {self.accelerator.gradient_accumulation_steps}")
            logger.info(f"  Stage 1 epochs: {STAGE1_EPOCHS}")
            logger.info(f"  Stage 2 starts at epoch: {STAGE2_START_EPOCH}")
            logger.info("=" * 60)
    
    def _log_metrics(self, metrics: dict):
        """记录指标"""
        if self.completed_steps % self.config.trainer.logging_frequency == 0:
            if dist.get_rank() == 0:
                # 添加学习率
                if self.optimizer is not None:
                    last_lrs = self.lr_scheduler.get_last_lr()
                    for i, group in enumerate(self.optimizer.param_groups):
                        group_name = group.get("name", str(i))
                        metrics[f"learning_rate/{group_name}"] = last_lrs[i] if i < len(last_lrs) else last_lrs[-1]
                
                # 添加 epoch 和 stage
                metrics["epoch"] = self.current_epoch
                metrics["stage"] = self.current_stage
                metrics["step"] = self.completed_steps
                
                # GPU 内存
                if torch.cuda.is_available():
                    metrics["gpu_memory/allocated"] = torch.cuda.memory_allocated() / 1024**3
                    metrics["gpu_memory/reserved"] = torch.cuda.memory_reserved() / 1024**3
                
                wandb.log(metrics, step=self.completed_steps)
                logger.info(f"Step {self.completed_steps}: {metrics}")
    
    def _finalize_training(self):
        """训练结束处理"""
        if self.accelerator.is_main_process:
            final_checkpoint = os.path.join(self.config.output_dir, "final_model")
            os.makedirs(final_checkpoint, exist_ok=True)
            
            state_dict = self.accelerator.get_state_dict(self.model)
            torch.save(state_dict, os.path.join(final_checkpoint, "pytorch_model.pt"))
            
            logger.info(f"Training complete. Final model saved at {final_checkpoint}")
            wandb.finish()
        
        self.accelerator.wait_for_everyone()


def main(cfg) -> None:
    """主函数"""
    logger.info("VLA Two-Stage Training :: Warming Up")
    
    cfg = wrap_config(cfg)
    logger.info("Configuration wrapped for access tracking")
    
    output_dir = setup_directories(cfg=cfg)
    
    # 构建模型
    vla = build_framework(cfg)
    
    # 准备数据
    vla_train_dataloader = prepare_data(cfg=cfg, accelerator=accelerator, output_dir=output_dir)
    
    # 创建训练器
    trainer = TwoStageVLATrainer(
        cfg=cfg,
        model=vla,
        dataloader=vla_train_dataloader,
        accelerator=accelerator,
    )
    
    # 准备训练
    trainer.prepare_training()
    
    # 开始训练
    trainer.train()
    
    logger.info("Training finished!")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/calvin/train_files/starvla_train_calvin_lora.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()
    
    cfg = OmegaConf.load(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(cfg, cli_cfg)
    
    # 应用配置兼容性
    cfg = apply_config_compat(cfg)
    
    # 保存源配置路径
    cfg.config_yaml = args.config_yaml
    
    # Debug 模式
    if cfg.is_debug and dist.is_initialized() and dist.get_rank() == 0:
        import debugpy
        debugpy.listen(("0.0.0.0", 10092))
        print("Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()
    
    main(cfg)
