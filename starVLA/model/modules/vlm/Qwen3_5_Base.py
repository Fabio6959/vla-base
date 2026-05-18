# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License").

"""
Qwen3.5-4B-base 接口

架构：
    SigLIP Vision Encoder (独立)
        +
    Projector (视觉特征 -> LLM embedding)
        +
    Qwen3.5-4B (纯文本 LLM)

特点：
    - Vision Encoder 独立于 LLM
    - 使用 SigLIP 作为视觉编码器
    - 需要额外的 Projector 将视觉特征投影到 LLM 空间
"""

from typing import Optional, List
import os

import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoModel, AutoTokenizer, AutoConfig
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

IGNORE_INDEX = -100
DEFAULT_IMAGE_TOKEN = "<image>"


class SigLIPVisionEncoder(nn.Module):
    """
    SigLIP 视觉编码器
    
    使用 Google 的 SigLIP 模型提取图像特征。
    支持多种 SigLIP 变体。
    
    Args:
        model_name: SigLIP 模型名称，默认 "google/siglip-so400m-patch14-384"
        image_size: 输入图像大小
    """
    
    def __init__(
        self,
        model_name: str = "google/siglip-so400m-patch14-384",
        image_size: int = 384,
    ):
        super().__init__()
        
        self.model_name = model_name
        self.image_size = image_size
        
        # 加载 SigLIP 模型
        try:
            from transformers import SiglipModel, SiglipProcessor
            self.model = SiglipModel.from_pretrained(model_name)
            self.processor = SiglipProcessor.from_pretrained(model_name)
        except ImportError:
            logger.warning("SigLIP not available in transformers, using AutoModel")
            self.model = AutoModel.from_pretrained(model_name)
            self.processor = None
        
        # 获取特征维度
        self.hidden_size = self.model.config.vision_config.hidden_size
        self.num_patches = (image_size // 14) ** 2  # SigLIP patch size = 14
        
        logger.info(f"SigLIP Vision Encoder loaded: {model_name}")
        logger.info(f"  - Hidden size: {self.hidden_size}")
        logger.info(f"  - Image size: {image_size}")
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            images: 图像张量 [B, C, H, W]
        
        Returns:
            视觉特征 [B, num_patches, hidden_size]
        """
        # SigLIP 的 vision_model 直接输出
        vision_outputs = self.model.vision_model(pixel_values=images)
        # last_hidden_state: [B, num_patches+1, hidden_size]
        # 去掉 CLS token
        patch_features = vision_outputs.last_hidden_state[:, 1:, :]
        return patch_features
    
    def preprocess_images(self, images: List[Image.Image]) -> torch.Tensor:
        """
        预处理图像
        
        Args:
            images: PIL 图像列表
        
        Returns:
            预处理后的张量 [B, C, H, W]
        """
        if self.processor is not None:
            inputs = self.processor(images=images, return_tensors="pt")
            return inputs['pixel_values']
        else:
            # 手动预处理
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
            return torch.stack([transform(img) for img in images])


class VisionProjector(nn.Module):
    """
    视觉特征投影器
    
    将视觉编码器的特征投影到 LLM embedding 空间。
    
    Args:
        vision_hidden_size: 视觉特征维度
        llm_hidden_size: LLM embedding 维度
        projector_type: 投影器类型，可选 "linear", "mlp", "mlp_gelu"
    """
    
    def __init__(
        self,
        vision_hidden_size: int,
        llm_hidden_size: int,
        projector_type: str = "mlp",
    ):
        super().__init__()
        
        self.projector_type = projector_type
        
        if projector_type == "linear":
            self.projector = nn.Linear(vision_hidden_size, llm_hidden_size)
        elif projector_type == "mlp":
            self.projector = nn.Sequential(
                nn.Linear(vision_hidden_size, llm_hidden_size),
                nn.GELU(),
                nn.Linear(llm_hidden_size, llm_hidden_size),
            )
        elif projector_type == "mlp_gelu":
            self.projector = nn.Sequential(
                nn.Linear(vision_hidden_size, llm_hidden_size),
                nn.GELU(),
                nn.Linear(llm_hidden_size, llm_hidden_size),
                nn.GELU(),
                nn.Linear(llm_hidden_size, llm_hidden_size),
            )
        else:
            raise ValueError(f"Unknown projector type: {projector_type}")
        
        logger.info(f"Vision Projector created: {projector_type}")
        logger.info(f"  - Vision hidden size: {vision_hidden_size}")
        logger.info(f"  - LLM hidden size: {llm_hidden_size}")
    
    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            vision_features: 视觉特征 [B, num_patches, vision_hidden_size]
        
        Returns:
            投影后的特征 [B, num_patches, llm_hidden_size]
        """
        return self.projector(vision_features)


class _Qwen3_5_Base_Interface(nn.Module):
    """
    Qwen3.5-4B-base 接口
    
    架构：
        SigLIP Vision Encoder -> Projector -> Qwen3.5-4B
    
    特点：
        - 纯文本 LLM + 独立视觉编码器
        - Vision Encoder 可冻结
        - 支持多视角输入
    
    Args:
        config: 全局配置对象
    """
    
    def __init__(self, config: Optional[dict] = None, **kwargs):
        super().__init__()
        
        self.config = config
        qwenvl_config = config.framework.get("qwenvl", {})
        
        # 模型路径
        model_id = qwenvl_config.get("base_vlm", "Qwen/Qwen3.5-4B")
        attn_implementation = qwenvl_config.get("attn_implementation", "sdpa")
        
        # 视觉编码器配置
        vision_config = qwenvl_config.get("vision", {})
        self.vision_model_name = vision_config.get("model_name", "google/siglip-so400m-patch14-384")
        self.image_size = vision_config.get("image_size", 384)
        
        # 加载 Qwen3.5-4B
        logger.info(f"Loading Qwen3.5-4B from {model_id}")
        
        # 检查是否使用 flash_attention_2
        if attn_implementation == "flash_attention_2":
            try:
                import flash_attn
            except ImportError:
                logger.warning("flash_attn not installed, falling back to sdpa")
                attn_implementation = "sdpa"
        
        # 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.tokenizer.padding_side = "left"
        
        # 添加特殊 token
        if DEFAULT_IMAGE_TOKEN not in self.tokenizer.get_vocab():
            self.tokenizer.add_tokens([DEFAULT_IMAGE_TOKEN])
            self.image_token_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
        else:
            self.image_token_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
        
        # 加载 LLM
        from transformers import Qwen2ForCausalLM
        self.model = Qwen2ForCausalLM.from_pretrained(
            model_id,
            attn_implementation=attn_implementation,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        
        # 调整 embedding 大小
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # LLM hidden size
        self.llm_hidden_size = self.model.config.hidden_size
        
        # 加载视觉编码器
        logger.info(f"Loading SigLIP from {self.vision_model_name}")
        self.vision_encoder = SigLIPVisionEncoder(
            model_name=self.vision_model_name,
            image_size=self.image_size,
        )
        self.vision_hidden_size = self.vision_encoder.hidden_size
        
        # 创建投影器
        projector_type = vision_config.get("projector_type", "mlp")
        self.projector = VisionProjector(
            vision_hidden_size=self.vision_hidden_size,
            llm_hidden_size=self.llm_hidden_size,
            projector_type=projector_type,
        )
        
        # 保存配置
        self.num_image_tokens = self.vision_encoder.num_patches
        
        logger.info(f"Qwen3.5-4B-base Interface initialized")
        logger.info(f"  - LLM hidden size: {self.llm_hidden_size}")
        logger.info(f"  - Vision hidden size: {self.vision_hidden_size}")
        logger.info(f"  - Num image tokens: {self.num_image_tokens}")
    
    def freeze_vision_encoder(self):
        """冻结视觉编码器"""
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        logger.info("Vision Encoder frozen")
    
    def unfreeze_vision_encoder(self):
        """解冻视觉编码器"""
        for param in self.vision_encoder.parameters():
            param.requires_grad = True
        logger.info("Vision Encoder unfrozen")
    
    def encode_images(self, images: List[List[Image.Image]]) -> torch.Tensor:
        """
        编码图像
        
        Args:
            images: 图像列表，每个元素是一个样本的多视角图像列表
        
        Returns:
            视觉特征 [B, num_views * num_patches, llm_hidden_size]
        """
        # 展平图像
        batch_size = len(images)
        num_views = len(images[0])
        all_images = [img for sample in images for img in sample]
        
        # 预处理
        image_tensor = self.vision_encoder.preprocess_images(all_images)
        image_tensor = image_tensor.to(next(self.vision_encoder.parameters()).device)
        
        # 编码
        vision_features = self.vision_encoder(image_tensor)
        
        # 投影
        projected_features = self.projector(vision_features)
        
        # 重塑形状
        projected_features = projected_features.view(
            batch_size, num_views * self.num_image_tokens, self.llm_hidden_size
        )
        
        return projected_features
    
    def build_qwenvl_inputs(
        self,
        images: List[List[Image.Image]],
        instructions: List[str],
        solutions: Optional[List[str]] = None,
        **kwargs,
    ) -> dict:
        """
        构建模型输入
        
        Args:
            images: 图像列表
            instructions: 指令列表
            solutions: 可选的解决方案列表
        
        Returns:
            模型输入字典
        """
        batch_size = len(images)
        num_views = len(images[0])
        
        # 构建文本
        # 为每个图像添加 <image> token
        num_image_tokens_per_sample = num_views * self.num_image_tokens
        image_tokens = DEFAULT_IMAGE_TOKEN * num_views
        
        texts = []
        for instruction in instructions:
            text = f"{image_tokens}\n{instruction}"
            texts.append(text)
        
        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            return_tensors="pt",
            add_special_tokens=True,
        )
        
        # 编码图像
        vision_features = self.encode_images(images)
        
        # 准备 inputs_embeds
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        # 获取 word embeddings
        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        
        # 替换 image token 的 embedding
        for i in range(batch_size):
            image_token_mask = input_ids[i] == self.image_token_id
            if image_token_mask.any():
                # 找到 image token 的位置
                image_indices = torch.where(image_token_mask)[0]
                # 替换为视觉特征
                num_tokens_to_replace = min(len(image_indices), vision_features.shape[1])
                inputs_embeds[i, image_indices[:num_tokens_to_replace]] = vision_features[i, :num_tokens_to_replace]
        
        return {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
        }
    
    def forward(
        self,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        前向传播
        
        Args:
            inputs_embeds: 输入 embeddings
            attention_mask: 注意力掩码
        
        Returns:
            模型输出
        """
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
                **kwargs,
            )
        return outputs
    
    def generate(self, **kwargs):
        """生成"""
        with torch.autocast("cuda", dtype=torch.float16):
            outputs = self.model.generate(**kwargs)
        return outputs


if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default=None)
    args = parser.parse_args()
    
    if args.config_yaml:
        cfg = OmegaConf.load(args.config_yaml)
    else:
        cfg = OmegaConf.create({
            "framework": {
                "qwenvl": {
                    "base_vlm": "Qwen/Qwen3.5-4B",
                    "attn_implementation": "sdpa",
                    "vision": {
                        "model_name": "google/siglip-so400m-patch14-384",
                        "image_size": 384,
                        "projector_type": "mlp",
                    }
                }
            }
        })
    
    model = _Qwen3_5_Base_Interface(cfg)
    print("Model loaded successfully!")
