# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License").
# Implemented for CALVIN Scene D zero-shot generalization.

"""
Residual MLP Bridge Network

功能：
    将 Qwen VLM 的 hidden state 映射到 PI Action Head 的 condition vector。
    采用轻量级的 Residual MLP 结构，支持 LayerNorm 和可配置的隐藏维度。

结构：
    Input (LLM hidden_dim)
        ↓
    Linear → SiLU → Linear
        ↓
    + Residual Connection
        ↓
    Output (PI condition_dim)
"""

import torch
import torch.nn as nn
from typing import Optional


class ResidualMLPBridge(nn.Module):
    """
    残差 MLP Bridge 网络
    
    将 VLM 的 hidden state 映射到 Action Head 的 condition vector。
    使用残差连接和 LayerNorm 提高训练稳定性。
    
    Args:
        input_dim: 输入维度（VLM hidden size）
        output_dim: 输出维度（PI condition dim）
        hidden_dim: 隐藏层维度，默认为 input_dim
        use_layer_norm: 是否使用 LayerNorm，默认 True
        dropout: dropout 概率，默认 0.0
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: Optional[int] = None,
        use_layer_norm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        hidden_dim = hidden_dim or input_dim
        self.hidden_dim = hidden_dim
        
        # 主分支：Linear -> SiLU -> Linear
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        # LayerNorm
        self.layer_norm = nn.LayerNorm(output_dim) if use_layer_norm else None
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
        # 残差连接：如果维度不匹配，使用投影层
        self.residual_proj = None
        if input_dim != output_dim:
            self.residual_proj = nn.Linear(input_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (B, seq_len, input_dim) 或 (B, input_dim)
        
        Returns:
            输出张量，形状为 (B, seq_len, output_dim) 或 (B, output_dim)
        """
        # 保存残差
        residual = x if self.residual_proj is None else self.residual_proj(x)
        
        # 主分支
        out = self.fc1(x)
        out = self.act(out)
        
        if self.dropout is not None:
            out = self.dropout(out)
        
        out = self.fc2(out)
        
        # 残差连接
        out = out + residual
        
        # LayerNorm
        if self.layer_norm is not None:
            out = self.layer_norm(out)
        
        return out


class MultiLayerBridge(nn.Module):
    """
    多层 Bridge 网络
    
    为 Layer-wise Action Head 提供每层的 condition 投影。
    每层使用独立的 Bridge 参数。
    
    Args:
        input_dim: 输入维度（VLM hidden size）
        output_dim: 输出维度（PI condition dim）
        num_layers: 层数
        hidden_dim: 隐藏层维度
        use_layer_norm: 是否使用 LayerNorm
        dropout: dropout 概率
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_layers: int,
        hidden_dim: Optional[int] = None,
        use_layer_norm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.num_layers = num_layers
        
        # 每层使用独立的 Bridge
        self.bridges = nn.ModuleList([
            ResidualMLPBridge(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dim=hidden_dim,
                use_layer_norm=use_layer_norm,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
    
    def forward(self, hidden_states_list: list) -> list:
        """
        前向传播
        
        Args:
            hidden_states_list: 各层 hidden states 列表，
                每个元素形状为 (B, seq_len, input_dim)
        
        Returns:
            各层 condition vectors 列表，
                每个元素形状为 (B, seq_len, output_dim)
        """
        assert len(hidden_states_list) == self.num_layers, \
            f"Expected {self.num_layers} hidden states, got {len(hidden_states_list)}"
        
        output_list = []
        for i, (hidden_state, bridge) in enumerate(zip(hidden_states_list, self.bridges)):
            output_list.append(bridge(hidden_state))
        
        return output_list


class SharedBridge(nn.Module):
    """
    共享 Bridge 网络
    
    所有层共享同一个 Bridge 参数，减少参数量。
    
    Args:
        input_dim: 输入维度（VLM hidden size）
        output_dim: 输出维度（PI condition dim）
        hidden_dim: 隐藏层维度
        use_layer_norm: 是否使用 LayerNorm
        dropout: dropout 概率
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: Optional[int] = None,
        use_layer_norm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.bridge = ResidualMLPBridge(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            use_layer_norm=use_layer_norm,
            dropout=dropout,
        )
    
    def forward(self, hidden_states_list: list) -> list:
        """
        前向传播
        
        Args:
            hidden_states_list: 各层 hidden states 列表
        
        Returns:
            各层 condition vectors 列表
        """
        return [self.bridge(h) for h in hidden_states_list]


def build_bridge(config) -> nn.Module:
    """
    根据配置构建 Bridge 网络
    
    Args:
        config: 全局配置对象，需要包含:
            - config.framework.bridge.bridge_type: "multi_layer" 或 "shared"
            - config.framework.bridge.hidden_dim: 隐藏层维度
            - config.framework.bridge.use_layer_norm: 是否使用 LayerNorm
            - config.framework.bridge.dropout: dropout 概率
            - config.framework.qwenvl.vl_hidden_dim: VLM hidden size
            - config.framework.action_model.hidden_size: Action head hidden size
            - config.framework.qwenvl.num_vl_layers: VLM 层数（用于 multi_layer）
    
    Returns:
        Bridge 网络模块
    """
    bridge_config = config.framework.get("bridge", {})
    
    input_dim = config.framework.qwenvl.vl_hidden_dim
    output_dim = config.framework.action_model.hidden_size
    
    bridge_type = bridge_config.get("bridge_type", "shared")
    hidden_dim = bridge_config.get("hidden_dim", None)
    use_layer_norm = bridge_config.get("use_layer_norm", True)
    dropout = bridge_config.get("dropout", 0.0)
    
    if bridge_type == "multi_layer":
        num_layers = config.framework.qwenvl.num_vl_layers
        return MultiLayerBridge(
            input_dim=input_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            use_layer_norm=use_layer_norm,
            dropout=dropout,
        )
    else:
        return SharedBridge(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            use_layer_norm=use_layer_norm,
            dropout=dropout,
        )


if __name__ == "__main__":
    # 测试代码
    batch_size = 2
    seq_len = 100
    input_dim = 2560
    output_dim = 2048
    num_layers = 36
    
    # 测试 ResidualMLPBridge
    print("Testing ResidualMLPBridge...")
    bridge = ResidualMLPBridge(input_dim, output_dim, hidden_dim=2048)
    x = torch.randn(batch_size, seq_len, input_dim)
    out = bridge(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    assert out.shape == (batch_size, seq_len, output_dim)
    
    # 测试 MultiLayerBridge
    print("\nTesting MultiLayerBridge...")
    multi_bridge = MultiLayerBridge(input_dim, output_dim, num_layers)
    hidden_states = [torch.randn(batch_size, seq_len, input_dim) for _ in range(num_layers)]
    outputs = multi_bridge(hidden_states)
    print(f"Number of outputs: {len(outputs)}")
    print(f"Each output shape: {outputs[0].shape}")
    assert len(outputs) == num_layers
    assert outputs[0].shape == (batch_size, seq_len, output_dim)
    
    # 测试 SharedBridge
    print("\nTesting SharedBridge...")
    shared_bridge = SharedBridge(input_dim, output_dim)
    outputs = shared_bridge(hidden_states)
    print(f"Number of outputs: {len(outputs)}")
    print(f"Each output shape: {outputs[0].shape}")
    
    print("\nAll tests passed!")
