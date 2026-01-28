# RTP-LLM 安装指南

## 依赖管理

RTP-LLM 使用 `pyproject.toml` 统一管理所有 Python 依赖。

## 基础安装

```bash
# 使用 virtualenv（推荐，可选）
/opt/conda310/bin/virtualenv ~/venv
source ~/venv/bin/activate
```

```bash
# 使用 uv 加速 pip
pip install uv 

# 安装前置依赖（torch 等）
# 会自动检测环境依赖（cuda12，rocm 等）
RTP_SKIP_BAZEL_BUILD=1 uv pip install . -e 

# 开始 build
# 默认自动检测平台信息
# 指定 [dev] 安装开发相关 wheel（如 pytest）
# 一定要有 --no-build-isolation，bazel 需要依赖 torch
uv pip install .[dev] -e --no-build-isolation

# 开始启动
python -m rtp_llm.start_server

```

## 平台特定安装

```bash
# CUDA 12.6 (内部版本)
pip install .[cuda12] --no-build-isolation

# CUDA 12.9 (内部版本)
pip install .[cuda12_9] --no-build-isolation

# ROCm/AMD (内部版本)
pip install .[rocm] --no-build-isolation

# CUDA 12 ARM (内部版本)
pip install .[cuda12_arm] --no-build-isolation
```

## 构建 whl

```bash
uv build
```

## 高级选项

``` bash
# 使用 RTP_BAZEL_CONFIG 指定 c++ bazel 构建参数
RTP_BAZEL_CONFIG="--disk_cache=~/.cache/disk_cache --config=cuda12_6" uv pip install -e . --no-build-isolation
```