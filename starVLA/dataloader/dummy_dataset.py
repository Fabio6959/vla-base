# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License").

"""
伪数据集模块

用于在没有真实数据集的情况下测试训练 pipeline。
生成随机的图像、指令和动作数据。
"""

import json
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm


class DummyCALVINDataset(Dataset):
    """
    伪 CALVIN 数据集
    
    生成随机数据用于测试训练 pipeline。
    
    Args:
        num_samples: 样本数量
        image_size: 图像尺寸 (H, W)
        action_dim: 动作维度
        action_horizon: 动作序列长度
        state_dim: 状态维度
        num_cameras: 相机数量
    """
    
    def __init__(
        self,
        num_samples: int = 1000,
        image_size: tuple = (200, 200),
        action_dim: int = 7,
        action_horizon: int = 16,
        state_dim: int = 7,
        num_cameras: int = 2,
        instructions: Optional[list] = None,
    ):
        self.num_samples = num_samples
        self.image_size = image_size
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.state_dim = state_dim
        self.num_cameras = num_cameras
        
        # 默认指令列表
        self.instructions = instructions or [
            "pick up the red block",
            "open the drawer",
            "push the blue button",
            "rotate the knob",
            "slide the door",
            "lift the green cube",
            "press the switch",
            "turn on the light",
            "move the object left",
            "close the container",
        ]
        
        print(f"[DummyCALVINDataset] Created with {num_samples} samples")
        print(f"  - Image size: {image_size}")
        print(f"  - Action dim: {action_dim}, horizon: {action_horizon}")
        print(f"  - State dim: {state_dim}")
        print(f"  - Cameras: {num_cameras}")
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> dict:
        """获取单个样本"""
        # 生成随机图像
        images = []
        for _ in range(self.num_cameras):
            # 生成带有一些结构的随机图像
            img = np.random.randint(0, 50, (*self.image_size, 3), dtype=np.uint8)
            
            # 添加一些随机形状
            for _ in range(random.randint(2, 5)):
                color = tuple(random.randint(100, 255) for _ in range(3))
                center = (random.randint(20, self.image_size[1]-20), 
                         random.randint(20, self.image_size[0]-20))
                radius = random.randint(10, 30)
                cv2_circle(img, center, radius, color, -1)
            
            images.append(Image.fromarray(img))
        
        # 生成随机动作序列
        actions = np.random.uniform(-1, 1, (self.action_horizon, self.action_dim)).astype(np.float32)
        
        # 生成随机状态
        state = np.random.uniform(-1, 1, (1, self.state_dim)).astype(np.float32)
        
        # 随机选择指令
        instruction = random.choice(self.instructions)
        
        return {
            "image": images,
            "action": actions,
            "state": state,
            "lang": instruction,
        }


def cv2_circle(img, center, radius, color, thickness):
    """简单的画圆函数（避免依赖 cv2）"""
    y, x = np.ogrid[:img.shape[0], :img.shape[1]]
    mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
    img[mask] = color


def create_dummy_lerobot_dataset(
    output_dir: str,
    num_episodes: int = 10,
    episode_length: int = 50,
    image_size: tuple = (200, 200),
    action_dim: int = 7,
    state_dim: int = 7,
):
    """
    创建 LeRobot 格式的伪数据集
    
    Args:
        output_dir: 输出目录
        num_episodes: episode 数量
        episode_length: 每个 episode 的长度
        image_size: 图像尺寸
        action_dim: 动作维度
        state_dim: 状态维度
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 创建 meta 目录
    meta_dir = output_path / "meta"
    meta_dir.mkdir(exist_ok=True)
    
    # 创建 data 目录
    data_dir = output_path / "data"
    data_dir.mkdir(exist_ok=True)
    
    print(f"Creating dummy LeRobot dataset at {output_dir}")
    
    # 创建 tasks.jsonl
    tasks = [
        {"task_index": i, "task": f"dummy_task_{i}"} 
        for i in range(10)
    ]
    with open(meta_dir / "tasks.jsonl", "w") as f:
        for task in tasks:
            f.write(json.dumps(task) + "\n")
    
    # 创建 episodes.jsonl
    episodes = []
    for i in range(num_episodes):
        episodes.append({
            "episode_index": i,
            "length": episode_length,
        })
    with open(meta_dir / "episodes.jsonl", "w") as f:
        for ep in episodes:
            f.write(json.dumps(ep) + "\n")
    
    # 创建 modality.json
    modality = {
        "video": {
            "image": {
                "original_key": "image",
                "start": 0,
                "end": 3,
            }
        },
        "state": {
            "state": {
                "original_key": "state",
                "start": 0,
                "end": state_dim,
                "absolute": True,
                "rotation_type": None,
            }
        },
        "action": {
            "action": {
                "original_key": "action",
                "start": 0,
                "end": action_dim,
                "absolute": False,
                "rotation_type": None,
            }
        },
        "annotation": {
            "human": {
                "original_key": "task_index",
            }
        }
    }
    with open(meta_dir / "modality.json", "w") as f:
        json.dump(modality, f, indent=2)
    
    # 创建 info.json
    info = {
        "codebase_version": "v2.0",
        "fps": 30,
        "video_path": "videos/{episode_chunk}/{episode_index}/{video_key}.mp4",
        "data_path": "data/{episode_chunk}/{episode_index}.parquet",
        "chunks_size": 1000,
        "total_videos": 0,
        "features": {
            "image": {
                "shape": [image_size[0], image_size[1], 3],
                "names": ["height", "width", "channel"],
            },
            "state": {
                "shape": [state_dim],
                "names": ["state"],
            },
            "action": {
                "shape": [action_dim],
                "names": ["action"],
            },
        }
    }
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)
    
    # 创建 parquet 数据文件
    try:
        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq
        
        chunk_dir = data_dir / "chunk-000"
        chunk_dir.mkdir(exist_ok=True)
        
        for ep_idx in tqdm(range(num_episodes), desc="Creating episodes"):
            # 生成随机数据
            data = {
                "episode_index": [ep_idx] * episode_length,
                "frame_index": list(range(episode_length)),
                "timestamp": [i / 30.0 for i in range(episode_length)],
                "task_index": [random.randint(0, 9) for _ in range(episode_length)],
                "action": [np.random.uniform(-1, 1, action_dim).tolist() for _ in range(episode_length)],
                "state": [np.random.uniform(-1, 1, state_dim).tolist() for _ in range(episode_length)],
            }
            
            # 保存为 parquet
            df = pd.DataFrame(data)
            df.to_parquet(chunk_dir / f"episode_{ep_idx:06d}.parquet")
        
        print(f"Created {num_episodes} episodes with {episode_length} frames each")
        
    except ImportError as e:
        print(f"Warning: Could not create parquet files: {e}")
        print("Install pandas and pyarrow for full functionality")
    
    print(f"Dummy dataset created at {output_dir}")
    return output_path


class SimpleDummyDataset(Dataset):
    """
    简单的伪数据集
    
    直接返回字典格式，无需 LeRobot 格式。
    用于快速测试模型 forward/backward。
    """
    
    def __init__(
        self,
        num_samples: int = 100,
        image_size: tuple = (224, 224),
        action_dim: int = 7,
        action_horizon: int = 16,
        state_dim: int = 7,
    ):
        self.num_samples = num_samples
        self.image_size = image_size
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.state_dim = state_dim
        
        # 预生成一些随机图像
        self.images = [
            Image.fromarray(
                np.random.randint(0, 255, (*image_size, 3), dtype=np.uint8)
            )
            for _ in range(10)
        ]
        
        self.instructions = [
            "pick up the block",
            "open the drawer",
            "push the button",
        ]
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # 随机选择图像
        images = [self.images[idx % 10], self.images[(idx + 1) % 10]]
        
        return {
            "image": images,
            "action": np.random.uniform(-1, 1, (self.action_horizon, self.action_dim)).astype(np.float32),
            "state": np.random.uniform(-1, 1, (1, self.state_dim)).astype(np.float32),
            "lang": self.instructions[idx % len(self.instructions)],
        }


def collate_fn(batch):
    """简单的 collate 函数"""
    return batch


if __name__ == "__main__":
    # 测试伪数据集
    print("Testing DummyCALVINDataset...")
    dataset = DummyCALVINDataset(num_samples=10)
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Image: {len(sample['image'])} images of size {sample['image'][0].size}")
    print(f"Action shape: {sample['action'].shape}")
    print(f"State shape: {sample['state'].shape}")
    print(f"Instruction: {sample['lang']}")
    
    # 测试创建 LeRobot 格式数据集
    print("\nTesting create_dummy_lerobot_dataset...")
    create_dummy_lerobot_dataset(
        output_dir="playground/Datasets/dummy_calvin",
        num_episodes=5,
        episode_length=20,
    )
