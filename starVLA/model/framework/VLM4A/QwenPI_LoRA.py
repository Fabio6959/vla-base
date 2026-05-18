# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License").
# Implemented for CALVIN Scene D zero-shot generalization.

"""
QwenPI-LoRA 框架 - 支持多种 VLM 模型

提供两个主要接口：
1. QwenPI_LoRA_Qwen35 - 使用 Qwen3.5-VL-4B-Instruct
2. QwenPI_LoRA_Qwen3 - 使用 Qwen3-VL-4B-Instruct

架构：
    Frozen Vision Encoder (内置在 VLM)
        +
    VLM (LoRA Finetuning)
        +
    Residual MLP Bridge
        +
    PI Flow-Matching Action Head (DiT)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import os

import numpy as np
import torch
from PIL import Image

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

IGNORE_INDEX = -100

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config, populate_layerwise_dit_cfg
from starVLA.model.modules.action_model.LayerwiseFM_ActionHeader import LayerwiseFlowmatchingActionHead, get_action_model
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.modules.projector.residual_mlp_bridge import build_bridge
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images

# LoRA 相关导入
try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    logger.warning("PEFT not available. LoRA will be disabled.")


# =============================================================================
# 模型特定配置
# =============================================================================

@dataclass
class Qwen35VLConfig:
    """Qwen3.5-VL-4B 模型配置"""
    name: str = "QwenPI_LoRA_Qwen35"
    
    qwenvl: dict = field(
        default_factory=lambda: {
            "base_vlm": "./playground/Pretrained_models/Qwen3.5-VL-4B-Instruct",
            "attn_implementation": "flash_attention_2",
            "vl_hidden_dim": 2560,
            "num_vl_layers": 40,
        }
    )
    
    lora: dict = field(
        default_factory=lambda: {
            "enabled": True,
            "r": 32,
            "lora_alpha": 64,
            "lora_dropout": 0.05,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "bias": "none",
        }
    )
    
    bridge: dict = field(
        default_factory=lambda: {
            "bridge_type": "shared",
            "hidden_dim": 2048,
            "use_layer_norm": True,
            "dropout": 0.0,
        }
    )
    
    action_model: dict = field(
        default_factory=lambda: {
            "action_model_type": "LayerwiseFM",
            "action_dim": 7,
            "state_dim": 7,
            "action_horizon": 16,
            "repeated_diffusion_steps": 2,
            "num_inference_timesteps": 4,
            "add_pos_embed": True,
            "max_seq_len": 1024,
            "num_target_vision_tokens": 32,
            "noise_beta_alpha": 1.5,
            "noise_beta_beta": 1.0,
            "noise_s": 0.999,
            "num_timestep_buckets": 1000,
            "diffusion_model_cfg": {
                "dropout": 0.2,
                "final_dropout": True,
                "interleave_self_attention": True,
                "norm_type": "ada_norm",
                "positional_embeddings": None,
                "attention_head_dim": 64,
            },
        }
    )


@dataclass
class Qwen3VLConfig:
    """Qwen3-VL-4B 模型配置"""
    name: str = "QwenPI_LoRA_Qwen3"
    
    qwenvl: dict = field(
        default_factory=lambda: {
            "base_vlm": "./playground/Pretrained_models/Qwen3-VL-4B-Instruct",
            "attn_implementation": "flash_attention_2",
            "vl_hidden_dim": 2560,
            "num_vl_layers": 36,
        }
    )
    
    lora: dict = field(
        default_factory=lambda: {
            "enabled": True,
            "r": 32,
            "lora_alpha": 64,
            "lora_dropout": 0.05,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "bias": "none",
        }
    )
    
    bridge: dict = field(
        default_factory=lambda: {
            "bridge_type": "shared",
            "hidden_dim": 2048,
            "use_layer_norm": True,
            "dropout": 0.0,
        }
    )
    
    action_model: dict = field(
        default_factory=lambda: {
            "action_model_type": "LayerwiseFM",
            "action_dim": 7,
            "state_dim": 7,
            "action_horizon": 16,
            "repeated_diffusion_steps": 2,
            "num_inference_timesteps": 4,
            "add_pos_embed": True,
            "max_seq_len": 1024,
            "num_target_vision_tokens": 32,
            "noise_beta_alpha": 1.5,
            "noise_beta_beta": 1.0,
            "noise_s": 0.999,
            "num_timestep_buckets": 1000,
            "diffusion_model_cfg": {
                "dropout": 0.2,
                "final_dropout": True,
                "interleave_self_attention": True,
                "norm_type": "ada_norm",
                "positional_embeddings": None,
                "attention_head_dim": 64,
            },
        }
    )


# =============================================================================
# 通用基类
# =============================================================================

class BaseQwenPILoRA(baseframework):
    """
    QwenPI-LoRA 基类
    
    支持多种 Qwen VLM 模型的通用实现。
    """
    
    default_config_class = None
    
    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        
        if self.default_config_class is None:
            raise ValueError("default_config_class must be set in subclass")
        
        self.config = merge_framework_config(self.default_config_class, config)
        
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        
        vlm_hf_cfg = self.qwen_vl_interface.model.config
        text_cfg = getattr(vlm_hf_cfg, "text_config", vlm_hf_cfg)
        num_vl_layers = int(text_cfg.num_hidden_layers)
        llm_hidden_size = int(vlm_hf_cfg.hidden_size)
        
        self.config.framework.qwenvl.vl_hidden_dim = llm_hidden_size
        self.config.framework.qwenvl.num_vl_layers = num_vl_layers
        
        lora_config = self.config.framework.get("lora", {})
        if lora_config.get("enabled", True) and PEFT_AVAILABLE:
            self.qwen_vl_interface.model = self._apply_lora(
                self.qwen_vl_interface.model, 
                lora_config
            )
            self.lora_enabled = True
        else:
            self.lora_enabled = False
        
        self.qwen_vl_interface.model = self._freeze_vision_encoder(self.qwen_vl_interface.model)
        
        populate_layerwise_dit_cfg(
            self.config,
            dit_hidden_dim=llm_hidden_size,
            num_dit_layers=num_vl_layers,
        )
        
        self.action_model: LayerwiseFlowmatchingActionHead = get_action_model(config=self.config)
        self.bridge = build_bridge(self.config)
        self.action_horizon = int(self.config.framework.action_model.action_horizon)
        self.lora_config = lora_config
        
        logger.info(f"{self.__class__.__name__} initialized with:")
        logger.info(f"  - VLM: {self.config.framework.qwenvl.base_vlm}")
        logger.info(f"  - VLM hidden size: {llm_hidden_size}")
        logger.info(f"  - VLM layers: {num_vl_layers}")
        logger.info(f"  - LoRA enabled: {self.lora_enabled}")
        logger.info(f"  - Action horizon: {self.action_horizon}")
    
    def _apply_lora(self, model, lora_config: dict):
        if not PEFT_AVAILABLE:
            return model
        
        if not lora_config.get("enabled", True):
            return model
        
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_config.get("r", 32),
            lora_alpha=lora_config.get("lora_alpha", 64),
            lora_dropout=lora_config.get("lora_dropout", 0.05),
            target_modules=lora_config.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
            bias=lora_config.get("bias", "none"),
        )
        
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        
        return model
    
    def _freeze_vision_encoder(self, model):
        frozen_count = 0
        vision_modules = []
        
        if hasattr(model, 'model') and hasattr(model.model, 'visual'):
            vision_modules.append(model.model.visual)
        elif hasattr(model, 'visual'):
            vision_modules.append(model.visual)
        elif hasattr(model, 'vision_tower'):
            vision_modules.append(model.vision_tower)
        
        if hasattr(model, 'base_model'):
            base = model.base_model
            if hasattr(base, 'model'):
                if hasattr(base.model, 'visual'):
                    vision_modules.append(base.model.visual)
                if hasattr(base.model, 'vision_tower'):
                    vision_modules.append(base.model.vision_tower)
        
        for vision_module in vision_modules:
            for param in vision_module.parameters():
                param.requires_grad = False
                frozen_count += param.numel()
        
        logger.info(f"Frozen {frozen_count:,} parameters in Vision Encoder")
        return model
    
    def _encode_vl_hidden_states(
        self, batch_images: List, instructions: List[str]
    ) -> List[torch.Tensor]:
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images, instructions=instructions
        )
        
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            expected_layers = len(self.action_model.model.transformer_blocks)
            vl_embs_list = list(qwenvl_outputs.hidden_states[-expected_layers:])
        
        return vl_embs_list
    
    def set_lora_trainable(self, trainable: bool):
        if not self.lora_enabled:
            return
        
        for name, param in self.qwen_vl_interface.model.named_parameters():
            if 'lora' in name.lower():
                param.requires_grad = trainable
        
        status = "trainable" if trainable else "frozen"
        logger.info(f"LoRA parameters set to {status}")
    
    def get_trainable_parameters_by_stage(self, stage: int) -> dict:
        param_groups = {}
        
        if stage == 1:
            self.set_lora_trainable(False)
            param_groups['bridge'] = [p for p in self.bridge.parameters() if p.requires_grad]
            param_groups['action_model'] = [p for p in self.action_model.parameters() if p.requires_grad]
            logger.info("Stage 1 (Cold Start): Training Bridge + Action Head only")
            
        elif stage == 2:
            self.set_lora_trainable(True)
            
            lora_params = []
            for name, param in self.qwen_vl_interface.model.named_parameters():
                if 'lora' in name.lower() and param.requires_grad:
                    lora_params.append(param)
            
            param_groups['lora'] = lora_params
            param_groups['bridge'] = [p for p in self.bridge.parameters() if p.requires_grad]
            param_groups['action_model'] = [p for p in self.action_model.parameters() if p.requires_grad]
            logger.info("Stage 2 (Joint Training): Training LoRA + Bridge + Action Head")
        
        return param_groups
    
    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        actions = [example["action"] for example in examples]
        
        state = [example["state"] for example in examples] if "state" in examples[0] else None
        
        vl_embs_list = self._encode_vl_hidden_states(batch_images, instructions)
        base_hidden = vl_embs_list[-1]
        
        bridge_embs_list = self.bridge(vl_embs_list)
        
        with torch.autocast("cuda", dtype=torch.float32):
            actions = torch.tensor(
                np.array(actions), device=base_hidden.device, dtype=base_hidden.dtype
            )
            actions_target = actions[:, -self.action_horizon:, :]
            
            repeated_diffusion_steps = self.config.framework.action_model.get("repeated_diffusion_steps", 2)
            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            
            bridge_embs_repeated = [h.repeat(repeated_diffusion_steps, 1, 1) for h in bridge_embs_list]
            
            state_repeated = None
            if state is not None:
                state = torch.tensor(np.array(state), device=base_hidden.device, dtype=base_hidden.dtype)
                state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)
            
            action_loss = self.action_model(
                bridge_embs_repeated,
                actions_target_repeated,
                state_repeated,
            )
        
        return {"action_loss": action_loss}
    
    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict] = None,
        **kwargs: str,
    ) -> np.ndarray:
        if type(examples) is not list:
            examples = [examples]
        
        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]
        
        state = [example["state"] for example in examples] if "state" in examples[0] else None
        
        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)
        
        vl_embs_list = self._encode_vl_hidden_states(batch_images, instructions)
        
        bridge_embs_list = self.bridge(vl_embs_list)
        
        state_tensor = None
        if state is not None:
            base_hidden = vl_embs_list[-1]
            state_tensor = torch.from_numpy(np.array(state)).to(
                base_hidden.device, dtype=base_hidden.dtype
            )
        
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(
                bridge_embs_list, state_tensor
            )
        
        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}


# =============================================================================
# 具体模型实现
# =============================================================================

@FRAMEWORK_REGISTRY.register("QwenPI_LoRA_Qwen35")
class QwenPI_LoRA_Qwen35(BaseQwenPILoRA):
    """
    QwenPI-LoRA 框架 - 使用 Qwen3.5-VL-4B-Instruct
    
    特点：
        - Qwen3.5-VL 是最新的视觉语言模型
        - 支持 FlashAttention-2
        - 更强的视觉理解能力
    """
    
    default_config_class = Qwen35VLConfig


@FRAMEWORK_REGISTRY.register("QwenPI_LoRA_Qwen3")
class QwenPI_LoRA_Qwen3(BaseQwenPILoRA):
    """
    QwenPI-LoRA 框架 - 使用 Qwen3-VL-4B-Instruct
    
    特点：
        - Qwen3-VL 是稳定的视觉语言模型
        - 支持 FlashAttention-2
        - 良好的性能表现
    """
    
    default_config_class = Qwen3VLConfig


# =============================================================================
# 兼容性别名
# =============================================================================

@FRAMEWORK_REGISTRY.register("QwenPILoRA")
class Qwen_PI_LoRA(QwenPI_LoRA_Qwen35):
    """
    QwenPI-LoRA 框架（兼容性别名）
    
    默认使用 Qwen3.5-VL-4B-Instruct
    """
    pass


# =============================================================================
# 测试代码
# =============================================================================

if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="qwen35",
        choices=["qwen35", "qwen3"],
        help="Which VLM to use",
    )
    parser.add_argument(
        "--config_yaml",
        type=str,
        default=None,
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()
    
    if args.config_yaml:
        cfg = OmegaConf.load(args.config_yaml)
    else:
        if args.model == "qwen35":
            cfg = OmegaConf.structured(Qwen35VLConfig())
        else:
            cfg = OmegaConf.structured(Qwen3VLConfig())
        cfg = OmegaConf.create({"framework": OmegaConf.to_container(cfg)})
    
    if args.model == "qwen35":
        model = QwenPI_LoRA_Qwen35(cfg)
    else:
        model = QwenPI_LoRA_Qwen3(cfg)
    
    print(f"\nModel: {model.__class__.__name__}")
    print(f"VLM: {model.config.framework.qwenvl.base_vlm}")
    print(f"LoRA enabled: {model.lora_enabled}")
