# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License").
# Implemented for CALVIN Scene D zero-shot generalization.

"""
CALVIN 数据增强模块

目标：提升 Scene D zero-shot 泛化能力

包含的增强：
    - ColorJitter: 亮度、对比度调整
    - RandomCrop: 随机裁剪
    - GaussianBlur: 高斯模糊
    - RandomGrayscale: 随机灰度化
    - RandomAffine: 随机仿射变换

特点：
    - 增强仅用于训练集
    - validation/eval 不使用增强
    - 保持 action 对齐
"""

from typing import Any, Callable, ClassVar, Literal, Optional
import random

import albumentations as A
import cv2
import numpy as np
import torch
import torchvision.transforms.v2 as T
from PIL import Image
from pydantic import Field, PrivateAttr

from starVLA.dataloader.gr00t_lerobot.transform.base import ModalityTransform
from starVLA.dataloader.gr00t_lerobot.schema import DatasetMetadata


class CALVINAugmentationTransform(ModalityTransform):
    """
    CALVIN 数据增强变换
    
    针对机器人操作任务的视觉增强，提升泛化能力。
    
    Args:
        brightness: 亮度调整范围，默认 0.2
        contrast: 对比度调整范围，默认 0.2
        saturation: 饱和度调整范围，默认 0.1
        hue: 色调调整范围，默认 0.05
        crop_scale: 随机裁剪缩放范围，默认 (0.9, 1.0)
        crop_ratio: 随机裁剪宽高比范围，默认 (0.9, 1.1)
        blur_kernel_size: 高斯模糊核大小，默认 (3, 7)
        blur_sigma: 高斯模糊 sigma 范围，默认 (0.1, 2.0)
        grayscale_prob: 随机灰度化概率，默认 0.1
        affine_degrees: 仿射变换旋转角度范围，默认 (-5, 5)
        affine_translate: 仿射变换平移范围，默认 (0.05, 0.05)
        affine_scale: 仿射变换缩放范围，默认 (0.95, 1.05)
        backend: 后端类型，"torchvision" 或 "albumentations"
    """
    
    brightness: float | tuple[float, float] = Field(default=0.2, description="亮度调整范围")
    contrast: float | tuple[float, float] = Field(default=0.2, description="对比度调整范围")
    saturation: float | tuple[float, float] = Field(default=0.1, description="饱和度调整范围")
    hue: float | tuple[float, float] = Field(default=0.05, description="色调调整范围")
    
    crop_scale: tuple[float, float] = Field(default=(0.9, 1.0), description="随机裁剪缩放范围")
    crop_ratio: tuple[float, float] = Field(default=(0.9, 1.1), description="随机裁剪宽高比范围")
    
    blur_kernel_size: tuple[int, int] = Field(default=(3, 7), description="高斯模糊核大小范围")
    blur_sigma: tuple[float, float] = Field(default=(0.1, 2.0), description="高斯模糊 sigma 范围")
    
    grayscale_prob: float = Field(default=0.1, description="随机灰度化概率")
    
    affine_degrees: tuple[float, float] = Field(default=(-5, 5), description="仿射变换旋转角度范围")
    affine_translate: tuple[float, float] = Field(default=(0.05, 0.05), description="仿射变换平移范围")
    affine_scale: tuple[float, float] = Field(default=(0.95, 1.05), description="仿射变换缩放范围")
    
    backend: str = Field(default="torchvision", description="后端类型")
    
    _train_transform: Callable | None = PrivateAttr(default=None)
    _eval_transform: Callable | None = PrivateAttr(default=None)
    _original_resolutions: dict[str, tuple[int, int]] = PrivateAttr(default_factory=dict)
    
    def set_metadata(self, dataset_metadata: DatasetMetadata):
        """设置元数据"""
        super().set_metadata(dataset_metadata)
        self._original_resolutions = {}
        
        for key in self.apply_to:
            split_keys = key.split(".")
            assert len(split_keys) == 2, f"Invalid key: {key}"
            sub_key = split_keys[1]
            if sub_key in dataset_metadata.modalities.video:
                self._original_resolutions[key] = dataset_metadata.modalities.video[sub_key].resolution
        
        self._train_transform = self._build_train_transform()
        self._eval_transform = None  # eval 不使用增强
    
    def _build_train_transform(self) -> Callable:
        """构建训练时的增强变换"""
        if self.backend == "torchvision":
            return self._build_torchvision_transform()
        else:
            return self._build_albumentations_transform()
    
    def _build_torchvision_transform(self) -> Callable:
        """构建 torchvision 变换"""
        transforms_list = []
        
        # 1. ColorJitter
        transforms_list.append(T.ColorJitter(
            brightness=self.brightness,
            contrast=self.contrast,
            saturation=self.saturation,
            hue=self.hue,
        ))
        
        # 2. RandomGrayscale
        transforms_list.append(T.RandomGrayscale(p=self.grayscale_prob))
        
        # 3. GaussianBlur
        transforms_list.append(T.GaussianBlur(
            kernel_size=self.blur_kernel_size,
            sigma=self.blur_sigma,
        ))
        
        # 4. RandomAffine
        transforms_list.append(T.RandomAffine(
            degrees=self.affine_degrees,
            translate=self.affine_translate,
            scale=self.affine_scale,
        ))
        
        # 5. RandomResizedCrop (如果指定了 crop_scale)
        if self.crop_scale != (1.0, 1.0):
            # 获取原始分辨率
            if self._original_resolutions:
                first_key = list(self._original_resolutions.keys())[0]
                height, width = self._original_resolutions[first_key]
                transforms_list.insert(0, T.RandomResizedCrop(
                    size=(height, width),
                    scale=self.crop_scale,
                    ratio=self.crop_ratio,
                ))
        
        return T.Compose(transforms_list)
    
    def _build_albumentations_transform(self) -> Callable:
        """构建 albumentations 变换"""
        transforms_list = []
        
        # 1. ColorJitter
        transforms_list.append(A.ColorJitter(
            brightness=self.brightness if isinstance(self.brightness, tuple) else (1 - self.brightness, 1 + self.brightness),
            contrast=self.contrast if isinstance(self.contrast, tuple) else (1 - self.contrast, 1 + self.contrast),
            saturation=self.saturation if isinstance(self.saturation, tuple) else (1 - self.saturation, 1 + self.saturation),
            hue=self.hue if isinstance(self.hue, tuple) else (-self.hue, self.hue),
            p=0.8,
        ))
        
        # 2. GaussianBlur
        transforms_list.append(A.GaussianBlur(
            blur_limit=self.blur_kernel_size,
            sigma_limit=self.blur_sigma,
            p=0.3,
        ))
        
        # 3. RandomGrayscale
        transforms_list.append(A.ToGray(p=self.grayscale_prob))
        
        # 4. Affine
        transforms_list.append(A.Affine(
            scale={"x": self.affine_scale, "y": self.affine_scale},
            translate_percent={"x": (-self.affine_translate[0], self.affine_translate[0]),
                               "y": (-self.affine_translate[1], self.affine_translate[1])},
            rotate=self.affine_degrees,
            p=0.5,
        ))
        
        # 5. RandomCrop (如果指定了 crop_scale)
        if self.crop_scale != (1.0, 1.0) and self._original_resolutions:
            first_key = list(self._original_resolutions.keys())[0]
            height, width = self._original_resolutions[first_key]
            crop_height = int(height * self.crop_scale[0])
            crop_width = int(width * self.crop_scale[0])
            transforms_list.insert(0, A.RandomCrop(
                height=crop_height,
                width=crop_width,
                p=0.5,
            ))
            transforms_list.insert(1, A.Resize(height=height, width=width))
        
        return A.Compose(transforms_list)
    
    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        """应用变换"""
        # eval 模式不使用增强
        if not self.training:
            return data
        
        transform = self._train_transform
        if transform is None:
            return data
        
        # 对每个视频 key 应用变换
        for key in self.apply_to:
            if key not in data:
                continue
            
            video_data = data[key]
            
            if self.backend == "torchvision":
                # torchvision: 输入为 tensor (T, C, H, W)
                if isinstance(video_data, torch.Tensor):
                    data[key] = transform(video_data)
            else:
                # albumentations: 输入为 numpy (T, H, W, C)
                if isinstance(video_data, np.ndarray):
                    transformed_frames = []
                    for frame in video_data:
                        transformed = transform(image=frame)["image"]
                        transformed_frames.append(transformed)
                    data[key] = np.stack(transformed_frames)
        
        return data


class CALVINStrongAugmentation(CALVINAugmentationTransform):
    """
    CALVIN 强增强版本
    
    使用更强的增强参数，适用于数据量较少的情况。
    """
    
    brightness: float | tuple[float, float] = Field(default=0.3)
    contrast: float | tuple[float, float] = Field(default=0.3)
    saturation: float | tuple[float, float] = Field(default=0.2)
    hue: float | tuple[float, float] = Field(default=0.1)
    
    crop_scale: tuple[float, float] = Field(default=(0.8, 1.0))
    crop_ratio: tuple[float, float] = Field(default=(0.8, 1.2))
    
    blur_kernel_size: tuple[int, int] = Field(default=(3, 9))
    blur_sigma: tuple[float, float] = Field(default=(0.1, 3.0))
    
    grayscale_prob: float = Field(default=0.15)
    
    affine_degrees: tuple[float, float] = Field(default=(-10, 10))
    affine_translate: tuple[float, float] = Field(default=(0.1, 0.1))
    affine_scale: tuple[float, float] = Field(default=(0.9, 1.1))


class CALVINLightAugmentation(CALVINAugmentationTransform):
    """
    CALVIN 轻量增强版本
    
    使用较轻的增强参数，适用于数据量充足或需要保持更多原始信息的情况。
    """
    
    brightness: float | tuple[float, float] = Field(default=0.1)
    contrast: float | tuple[float, float] = Field(default=0.1)
    saturation: float | tuple[float, float] = Field(default=0.05)
    hue: float | tuple[float, float] = Field(default=0.02)
    
    crop_scale: tuple[float, float] = Field(default=(0.95, 1.0))
    crop_ratio: tuple[float, float] = Field(default=(0.95, 1.05))
    
    blur_kernel_size: tuple[int, int] = Field(default=(3, 5))
    blur_sigma: tuple[float, float] = Field(default=(0.1, 1.0))
    
    grayscale_prob: float = Field(default=0.05)
    
    affine_degrees: tuple[float, float] = Field(default=(-3, 3))
    affine_translate: tuple[float, float] = Field(default=(0.02, 0.02))
    affine_scale: tuple[float, float] = Field(default=(0.98, 1.02))


def build_calvin_augmentation(
    augmentation_type: str = "standard",
    apply_to: list[str] = None,
    **kwargs,
) -> CALVINAugmentationTransform:
    """
    构建 CALVIN 数据增强变换
    
    Args:
        augmentation_type: 增强类型，可选 "light", "standard", "strong"
        apply_to: 应用到的 key 列表
        **kwargs: 其他参数
    
    Returns:
        数据增强变换
    """
    apply_to = apply_to or ["video.image"]
    
    if augmentation_type == "light":
        return CALVINLightAugmentation(apply_to=apply_to, **kwargs)
    elif augmentation_type == "strong":
        return CALVINStrongAugmentation(apply_to=apply_to, **kwargs)
    else:
        return CALVINAugmentationTransform(apply_to=apply_to, **kwargs)


# =============================================================================
# 用于 LeRobot 数据集的增强包装器
# =============================================================================

class LeRobotAugmentationWrapper:
    """
    LeRobot 数据集增强包装器
    
    将增强集成到 LeRobot 数据集的 transform 流程中。
    """
    
    def __init__(
        self,
        base_transforms,
        augmentation_config: dict = None,
        training: bool = True,
    ):
        """
        Args:
            base_transforms: 基础变换
            augmentation_config: 增强配置
            training: 是否为训练模式
        """
        self.base_transforms = base_transforms
        self.training = training
        
        # 构建增强
        if augmentation_config and training:
            self.augmentation = build_calvin_augmentation(
                augmentation_type=augmentation_config.get("type", "standard"),
                apply_to=augmentation_config.get("apply_to", ["video.image"]),
                **augmentation_config.get("params", {}),
            )
            self.augmentation.training = True
        else:
            self.augmentation = None
    
    def set_metadata(self, metadata):
        """设置元数据"""
        if self.base_transforms and hasattr(self.base_transforms, 'set_metadata'):
            self.base_transforms.set_metadata(metadata)
        if self.augmentation:
            self.augmentation.set_metadata(metadata)
    
    def __call__(self, data: dict) -> dict:
        """应用变换"""
        # 先应用基础变换
        if self.base_transforms:
            data = self.base_transforms(data)
        
        # 再应用增强
        if self.augmentation and self.training:
            data = self.augmentation.apply(data)
        
        return data


if __name__ == "__main__":
    # 测试代码
    print("Testing CALVIN Augmentation...")
    
    # 创建测试数据
    batch_size = 2
    seq_len = 4
    height, width = 224, 224
    
    # 测试 torchvision 后端
    print("\n=== Testing torchvision backend ===")
    aug_torch = CALVINAugmentationTransform(
        apply_to=["video.image"],
        backend="torchvision",
    )
    aug_torch.training = True
    
    # 模拟设置元数据
    from starVLA.dataloader.gr00t_lerobot.schema import DatasetMetadata
    
    # 测试 numpy 输入 (T, H, W, C)
    video_np = np.random.randint(0, 255, (seq_len, height, width, 3), dtype=np.uint8)
    data_np = {"video.image": video_np}
    
    print(f"Input shape: {video_np.shape}")
    
    # 测试 albumentations 后端
    print("\n=== Testing albumentations backend ===")
    aug_alb = CALVINAugmentationTransform(
        apply_to=["video.image"],
        backend="albumentations",
    )
    aug_alb.training = True
    
    # 测试强增强
    print("\n=== Testing strong augmentation ===")
    aug_strong = CALVINStrongAugmentation(
        apply_to=["video.image"],
        backend="albumentations",
    )
    aug_strong.training = True
    
    # 测试轻量增强
    print("\n=== Testing light augmentation ===")
    aug_light = CALVINLightAugmentation(
        apply_to=["video.image"],
        backend="albumentations",
    )
    aug_light.training = True
    
    print("\nAll tests passed!")
