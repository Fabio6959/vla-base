#!/bin/bash
# ================================================================================
# CALVIN QwenPI-LoRA 训练启动脚本
# ================================================================================
# 
# 使用方法：
#   bash run_calvin_lora_train.sh [NUM_GPUS]
#
# 示例：
#   bash run_calvin_lora_train.sh 8
#
# ================================================================================

NUM_GPUS=${1:-8}
CONFIG_FILE="examples/calvin/train_files/starvla_train_calvin_lora.yaml"
DEEPSPEED_CONFIG="examples/calvin/train_files/deepspeed_config.json"

echo "=========================================="
echo "CALVIN QwenPI-LoRA Training"
echo "=========================================="
echo "Number of GPUs: $NUM_GPUS"
echo "Config file: $CONFIG_FILE"
echo "DeepSpeed config: $DEEPSPEED_CONFIG"
echo "=========================================="

# 设置环境变量
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# Flash Attention 2
export FLASH_ATTENTION_FORCE_BUILD=FALSE

# DeepSpeed 环境变量
export DS_ACCELERATOR=deepspeed

# 启动训练
deepspeed --num_gpus=$NUM_GPUS \
    --master_port=29500 \
    starVLA/training/train_starvla_lora.py \
    --config_yaml $CONFIG_FILE \
    trainer.learning_rate.base=1e-4 \
    trainer.gradient_accumulation_steps=4 \
    datasets.vla_data.per_device_batch_size=8

echo "=========================================="
echo "Training completed!"
echo "=========================================="
