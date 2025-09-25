#!/bin/bash

echo "🚀 启动 RTP-LLM React 项目 (Vite)..."

# 检查是否已安装 Node.js
if ! command -v node &> /dev/null; then
    echo "❌ 错误: 未找到 Node.js，请先安装 Node.js"
    exit 1
fi

# 检查是否已安装 npm
if ! command -v npm &> /dev/null; then
    echo "❌ 错误: 未找到 npm，请先安装 npm"
    exit 1
fi

# 进入项目目录
cd "$(dirname "$0")"

echo "📦 安装依赖..."
npm install

if [ $? -eq 0 ]; then
    echo "✅ 依赖安装成功"
    echo "⚡ 启动 Vite 开发服务器..."
    npm run dev
else
    echo "❌ 依赖安装失败"
    exit 1
fi