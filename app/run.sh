#!/bin/bash

# LatentQA Gradio App 启动脚本

echo "🚀 启动 LatentQA Gradio 应用..."

# 设置项目根目录
export PROJECT_ROOT=$(dirname $(dirname $(realpath $0)))
cd $PROJECT_ROOT

echo "📁 项目根目录: $PROJECT_ROOT"

# 激活 deception 环境
echo "🔄 激活 conda 环境: deception"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate deception

if [ $? -ne 0 ]; then
    echo "❌ 无法激活 deception 环境，请确保该环境存在"
    exit 1
fi

echo "✅ 已激活 deception 环境"

# 启动应用
echo "🌐 启动Gradio应用..."
CUDA_VISIBLE_DEVICES=2 python3 app/app.py