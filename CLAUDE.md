# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RTP-LLM is a high-performance LLM inference acceleration engine by Alibaba. It's a hybrid C++/CUDA/Python codebase supporting NVIDIA (CUDA), AMD (ROCm), ARM, and CPU backends.

## Build System

Bazel is the build system. All builds require a config flag for the target platform.

```bash
# Build the main shared library (CUDA 12.x)
bazelisk build //:th_transformer --config=cuda12

# Build with specific CUDA version
bazelisk build //:th_transformer --config=cuda12_6
bazelisk build //:th_transformer --config=cuda12_9

# Other platforms
bazelisk build //:th_transformer --config=rocm
bazelisk build //:th_transformer --config=arm
bazelisk build //:th_transformer --config=cpu

# Debug build (append to any config)
bazelisk build //:th_transformer --config=cuda12_6 --config=debug

# Generate compile_commands.json
bazelisk run //:refresh_compdb
```

Key build targets:
- `//:th_transformer` — main shared library
- `//:rtp_compute_ops` — compute operations library
- `//:th_transformer_config` — config bindings library

## Running Tests

C++ tests use GoogleTest, wrapped via `cc_test_wrapper` (defined in `def.bzl`) which builds a `cc_binary` then runs it as an `sh_test`. Python tests use `py_test`.

```bash
# Run a single C++ test
bazelisk test //rtp_llm/cpp/normal_engine/test:engine_test --config=cuda12

# Run a single Python test
bazelisk test //rtp_llm/test:generate_config_test --config=cuda12

# Run all tests in a directory
bazelisk test //rtp_llm/cpp/core/test/... --config=cuda12

# Run with ASAN
bazelisk test //rtp_llm/cpp/core/test:... --config=cuda12 --config=asan
```

Most tests require GPU hardware (exec_properties specify GPU type like `{'gpu':'A10'}`).

## Architecture

### Two-Layer Design

**Python layer** (`rtp_llm/`): Model definitions, configuration, server orchestration, weight loading.
- `model_factory.py` / `model_factory_register.py` — model registration and instantiation
- `config/` — engine, model, quantization, KV cache configuration
- `models/` — 60+ model definitions (Qwen, LLaMA, DeepSeek, GLM, etc.)
- `server/`, `frontend/` — HTTP/gRPC serving infrastructure
- `start_server.py` → `start_backend_server.py` / `start_frontend_server.py` — entry points

**C++ layer** (`rtp_llm/cpp/`): Core inference engine, GPU kernels, memory management.
- `core/` — Buffer, Types, memory tracking primitives
- `devices/` — Hardware abstraction layer with implementations in `cuda_impl/`, `rocm_impl/`, `cpu_impl/`, `arm_impl/`
- `kernels/` — 100+ CUDA kernels (attention, sampling, quantization, MoE)
- `models/` — GptModel, Sampler, LoRA, logits processors, weight handling
- `engine_base/` — Base engine with schedulers, stream management, system prompt cache
- `normal_engine/` — Standard inference engine (NormalEngine, NormalExecutor)
- `speculative_engine/` — Speculative decoding engine
- `embedding_engine/` — Embedding model engine
- `cache/` — KV cache management with local and remote connectors
- `api_server/` — HTTP server with OpenAI-compatible endpoints
- `cuda/` — cuBLAS, FlashAttention (cufmha), CUTLASS, DeepGEMM, NCCL wrappers
- `pybind/` — Python bindings connecting C++ to Python

### Request Flow

HTTP request → `api_server/HttpApiServer` → Engine (`NormalEngine`/`SpeculativeEngine`/`EmbeddingEngine`) → Scheduler → Executor → `GptModel` → Device ops (CUDA kernels) → Response stream

### Device Abstraction

`DeviceBase` defines the interface. Each backend (`cuda_impl`, `rocm_impl`, etc.) implements device-specific operations. `DeviceFactory` instantiates the correct backend. Tests in `devices/base_tests/` define device-agnostic test cases reused across backends.

## Code Conventions

- C++17 standard, compiled with `-Wall -Werror`
- CUDA kernels in `.cu` files, headers in `.h`
- Bazel BUILD files use `copts()` / `cuda_copts()` from `def.bzl`
- C++ tests use `cc_test_wrapper` (not raw `cc_test`) — see `def.bzl`
- Python uses `/opt/conda310/bin/python3` (hardcoded in `.bazelrc`)
- Config flags like `using_cuda`, `using_rocm`, `using_arm` gate platform-specific code via `select()` in BUILD files and `#ifdef` in C++
