# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License").

"""
Projector 模块

包含各种投影器和 Bridge 网络，用于连接 VLM 和 Action Head。
"""

from .residual_mlp_bridge import (
    ResidualMLPBridge,
    MultiLayerBridge,
    SharedBridge,
    build_bridge,
)

try:
    from .QFormer import QFormerProjector
except ImportError:
    pass

__all__ = [
    "ResidualMLPBridge",
    "MultiLayerBridge",
    "SharedBridge",
    "build_bridge",
]
