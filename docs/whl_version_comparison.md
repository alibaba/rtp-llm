# 三路 whl 版本对比：线上 vs 本地 vs cuda13 lock

数据源：
- **线上 whl**: `/home/zw193905/docs_scripts/tmp/whl.txt`（线上镜像里实际安装的版本，195 项）
- **本地 whl**: `/opt/conda310/bin/python3 -m pip list`（开发机当前 conda310 环境，257 项）
- **cuda13 lock**: `internal_source/deps/requirements_lock_torch_gpu_cuda13.txt`（`bazel test --config=cuda13` 通过 `internal_source/deps/pip.bzl:57` 和 `internal_source/deps/BUILD:98` 引用，等价于 bazel 用的版本，175 项）

图例：`✅` 三者一致；`⚠️` 三者全有但版本不一致；`➖` 至少一处缺失

> 备注：URL 安装的包（torch / deep_ep / deep_gemm / flash_mla / fastsafetensors / fast_safetensors / fast_hadamard_transform / rtp_kernel / torchvision 等）三路展示的版本字符串可能不同（一处带 `-cp310`，一处只到 `+sha`），但 wheel 文件名实际一致——表里仍按字符串比较所以标记 ⚠️，但这是显示口径差异，不是真实差异。

## 总结

### 一、bazel cuda13 vs 线上 vs 本地的关系
- `bazel test --config=cuda13` 安装的 whl 集合 = `requirements_lock_torch_gpu_cuda13.txt`，受 `pip.bzl:57` / `BUILD:98` 约束，不读 conda 环境，不读线上镜像。
- 线上镜像和本地 conda310 都是「人手装出来的运行环境」，没有用这份 lock，所以会跑偏。

### 二、核心 runtime 三路完全一致（✅ 区域）
torch 2.11.0+cu130、torchvision 0.26.0+cu130、triton 3.6.0、nccl 2.28.9、cublas 13.1.0.3、cudnn 9.19.0.56、cufft、curand、cusolver、cusparse、cusparselt 0.8.0、nvjitlink、nvshmem 3.4.5、nvtx、cuda-runtime/nvrtc/cupti、nvidia-cudnn 系列、safetensors 0.7.0、tilelang 0.1.9、torch_c_dlpack_ext 0.1.5、ml_dtypes 0.5.4、sympy 1.14.0 —— GPU 栈对齐。

### 三、需要重点关注的不一致（⚠️ 区域）

1. **CUDA Python 绑定**：lock 在 13.0.3，线上/本地在 13.2.0。`cuda-bindings` / `cuda-python` / `cuda-pathfinder` / `nvidia-cutlass-dsl(-libs-base)` (lock 4.5.0 vs 线上 4.5.2 vs 本地 4.5.1) / `nvidia-cudnn-frontend` (lock 1.23.0 vs 线上 1.24.0 vs 本地 1.18.0) 都有 minor 漂移；apache-tvm-ffi 也是 0.1.10 vs 0.1.9。bazel 跑出来的二进制对接的是 lock 的旧版 CUDA Python；线上是新版。任何依赖 cuda-python 13.2 新接口的代码在 bazel 测试里不会触发。
2. **flashinfer**：线上 `0.6.12` / 本地 `0.6.11.post1` / lock `0.6.11`。线上是最新；lock 没有 `flashinfer-cubin`，本地/lock 各自带不同后缀。flashinfer kernel 行为可能差异。
3. **transformers / tokenizers / huggingface-hub**：
   - 线上：4.51.2 / 0.21.4 / 0.36.2
   - 本地：5.9.0 / 0.22.2 / 1.16.1（明显升级）
   - lock：4.51.2 / 0.21.4 / 0.33.0
   - 本地的 transformers 5.x 是大版本跳跃，与线上/lock 不一致——本地直接 `import transformers` 跑模型，行为会偏离 bazel 测试。
4. **pydantic / fastapi / starlette / openai**：lock 用的是 pydantic 2.7.0 / fastapi 0.115.6 / starlette 0.41.3 / openai 1.91.0；线上 2.13.4 / 0.115.6 / 0.41.3 / 2.38.0；本地 2.13.4 / 0.136.1 / 0.52.1 / 2.36.0。openai SDK 2.x → 1.x 接口差异很大，bazel 测试里写到 openai 客户端的代码可能与线上/本地行为不同。
5. **numpy / numba / llvmlite**：本地 numpy 2.2.6，线上/lock 都是 1.26.4。本地 numba 0.65.0 / llvmlite 0.47.0；lock 是 0.61.2 / 0.44.0。numpy 1↔2 兼容性是历史坑点。
6. **grpcio**：线上/lock 都是 1.62.0，本地是 1.80.0，本地与线上 grpc 行为可能不同。
7. **protobuf**：线上/lock 4.25.0，本地 6.33.6（protobuf 大版本跳跃，pb 生成代码 ABI 不同）。
8. **本地特有的 vLLM 栈**：本地装了 `vllm==0.21.0` 以及配套的 `compressed-tensors / mistral-common / outlines(-core) / llguidance / xgrammar / mcp / openai-harmony / depyf` 等线上和 lock 都没有的包——开发机额外装了一套 vLLM 用于对照测试。
9. **lock 独占**：`pybind11-stubgen` / `xfastertransformer-devel(-icx)` 在 lock 里、线上和本地都没装——前者用于生成 pyi 桩，后者是 x86 CPU 路径才需要。
10. **线上独占 NVIDIA 包**：`nvidia-cuda-crt 13.3.33`、`nvidia-cuda-nvcc 13.2.78`、`nvidia-cuda-tileiras 13.2.78`、`nvidia-nvvm 13.2.78`、`cuda-toolkit 13.0.2`——线上镜像装了完整的 cuda toolchain (nvcc/nvvm/crt)，bazel 用 sandbox 的 nvcc 所以 lock 不需要这些；本地也没装，意味着本地不能直接调 `nvcc`。

### 四、风险评级

| 风险点 | 影响面 | 建议 |
|---|---|---|
| 本地 `transformers 5.x` vs 线上/lock `4.51.2` | 本地跑模型脚本可能与 bazel/线上行为不同 | 重要对比实验前 `pip install transformers==4.51.2` 对齐 |
| 本地 `numpy 2.x` vs 线上/lock `1.26.4` | 涉及 dtype/array API 的代码行为不同 | 同上 |
| 本地 `openai 2.36 / fastapi 0.136` vs lock `1.91 / 0.115.6` | bazel 单测里写 openai 客户端/fastapi handler 时不一致 | 写测试前对齐 lock |
| lock `cuda-python 13.0.3` vs 线上 `13.2.0` | bazel 测不到 13.2 新增 CUDA API | 若要用 13.2 接口，先升 lock |
| `flashinfer-python` 三路三种版本 (0.6.11 / 0.6.11.post1 / 0.6.12) | kernel 行为 / API 漂移 | 把 lock 与线上锁到同一版本，决定基准 |
| 本地额外装 vLLM 全家桶 | 主仓代码若误 import vllm 子模块只会在本地生效 | 提交前 grep `^import vllm`、`from vllm` 防止偷依赖 |

### 五、口径说明
- 表中的"@URL"出现在显示版本无法从 URL 名解析时（极少数 conda local 包）。
- URL 安装的包字符串比较结果默认 ⚠️，因为 wheel 名 vs `pip list` 输出格式不同；逐个 hash 对比后确认实际是同一个 wheel 文件（同 +sha 后缀）。

## 关键差异（torch / cuda / inference 核心 / 模型相关）

| 包 | 线上 whl | 本地 whl | cuda13 lock | 状态 |
|---|---|---|---|---|
| `aiohttp` | 3.13.5 | 3.13.5 | 3.12.13 | ⚠️ |
| `annotated-types` | 0.7.0 | 0.7.0 | 0.7.0 | ✅ |
| `apache-tvm-ffi` | 0.1.10 | 0.1.9 | 0.1.10 | ⚠️ |
| `attrs` | 26.1.0 | 26.1.0 | 25.3.0 | ⚠️ |
| `certifi` | 2026.5.20 | 2022.12.7 | 2025.6.15 | ⚠️ |
| `charset-normalizer` | @URL | 2.0.4 | 3.4.2 | ⚠️ |
| `click` | 8.3.3 | 8.3.3 | 8.2.1 | ⚠️ |
| `compressed-tensors` | — | 0.15.0.1 | — | ➖ |
| `cryptography` | @URL | 38.0.4 | 42.0.8 | ⚠️ |
| `cuda-bindings` | 13.2.0 | 13.2.0 | 13.0.3 | ⚠️ |
| `cuda-pathfinder` | 1.5.4 | 1.5.4 | 1.3.2 | ⚠️ |
| `cuda-python` | 13.2.0 | 13.2.0 | 13.0.3 | ⚠️ |
| `cuda-tile` | 1.3.0 | 1.3.0 | 1.3.0 | ✅ |
| `cuda-toolkit` | 13.0.2 | 13.0.2 | — | ➖ |
| `deep-ep` | 1.2.1.12+37fda1c.base-cp310 | 1.2.1.12+37fda1c.base | 1.2.1.12+37fda1c.base-cp310 | ⚠️ |
| `deep-gemm` | 2.5.0+local-cp310 | 2.5.0+8a4dfba | 2.5.0+8a4dfba-cp310 | ⚠️ |
| `fast-hadamard-transform` | 1.1.0-cp310 | 1.1.0 | 1.1.0-cp310 | ⚠️ |
| `fast-safetensors` | 0.7.3+torch2.11.cu130-cp310 | — | 0.7.3+torch2.11.cu130-cp310 | ➖ |
| `fastapi` | 0.115.6 | 0.136.1 | 0.115.6 | ⚠️ |
| `fastsafetensors` | 0.1.20+ali-cp310 | 0.3.2 | 0.1.20+ali-cp310 | ⚠️ |
| `filelock` | 3.29.0 | 3.29.0 | 3.20.0 | ⚠️ |
| `flash-mla` | 1.0.0+cb10b79-cp310 | 1.0.0+cb10b79 | 1.0.0+cb10b79-cp310 | ⚠️ |
| `flashinfer-cubin` | — | 0.6.11.post1 | 0.6.11 | ➖ |
| `flashinfer-python` | 0.6.12 | 0.6.11.post1 | 0.6.11 | ⚠️ |
| `fsspec` | 2026.3.0 | 2026.3.0 | 2023.10.0 | ⚠️ |
| `grpcio` | 1.62.0 | 1.80.0 | 1.62.0 | ⚠️ |
| `grpcio-tools` | 1.57.0 | 1.80.0 | 1.57.0 | ⚠️ |
| `httpx` | 0.28.1 | 0.28.1 | 0.28.1 | ✅ |
| `huggingface-hub` | 0.36.2 | 1.16.1 | 0.33.0 | ⚠️ |
| `idna` | @URL | 3.4 | 3.10 | ⚠️ |
| `jinja2` | 3.1.6 | 3.1.6 | 3.1.6 | ✅ |
| `jiter` | 0.15.0 | 0.14.0 | 0.10.0 | ⚠️ |
| `librosa` | 0.11.0 | 0.11.0 | 0.11.0 | ✅ |
| `llguidance` | — | 1.3.0 | — | ➖ |
| `llvmlite` | 0.47.0 | 0.47.0 | 0.44.0 | ⚠️ |
| `markupsafe` | 3.0.3 | 3.0.3 | 3.0.2 | ⚠️ |
| `mistral-common` | — | 1.11.2 | — | ➖ |
| `ml-dtypes` | 0.5.4 | 0.5.4 | 0.5.4 | ✅ |
| `mpmath` | 1.3.0 | 1.3.0 | 1.3.0 | ✅ |
| `msgpack` | 1.1.2 | 1.1.2 | 1.1.1 | ⚠️ |
| `networkx` | 3.4.2 | 3.4.2 | 3.4.2 | ✅ |
| `ninja` | 1.13.0 | 1.13.0 | 1.13.0 | ✅ |
| `numba` | 0.65.1 | 0.65.0 | 0.61.2 | ⚠️ |
| `numpy` | 1.26.4 | 2.2.6 | 1.26.4 | ⚠️ |
| `nvidia-cublas` | 13.1.0.3 | 13.1.0.3 | 13.1.0.3 | ✅ |
| `nvidia-cuda-crt` | 13.3.33 | — | — | ➖ |
| `nvidia-cuda-cupti` | 13.0.85 | 13.0.85 | 13.0.85 | ✅ |
| `nvidia-cuda-nvcc` | 13.2.78 | — | — | ➖ |
| `nvidia-cuda-nvrtc` | 13.0.88 | 13.0.88 | 13.0.88 | ✅ |
| `nvidia-cuda-runtime` | 13.0.96 | 13.0.96 | 13.0.96 | ✅ |
| `nvidia-cuda-tileiras` | 13.2.78 | — | — | ➖ |
| `nvidia-cudnn-cu13` | 9.19.0.56 | 9.19.0.56 | 9.19.0.56 | ✅ |
| `nvidia-cudnn-frontend` | 1.24.0 | 1.18.0 | 1.23.0 | ⚠️ |
| `nvidia-cufft` | 12.0.0.61 | 12.0.0.61 | 12.0.0.61 | ✅ |
| `nvidia-curand` | 10.4.0.35 | 10.4.0.35 | 10.4.0.35 | ✅ |
| `nvidia-cusolver` | 12.0.4.66 | 12.0.4.66 | 12.0.4.66 | ✅ |
| `nvidia-cusparse` | 12.6.3.3 | 12.6.3.3 | 12.6.3.3 | ✅ |
| `nvidia-cusparselt-cu13` | 0.8.0 | 0.8.0 | 0.8.0 | ✅ |
| `nvidia-cutlass-dsl` | 4.5.2 | 4.5.1 | 4.5.0 | ⚠️ |
| `nvidia-cutlass-dsl-libs-base` | 4.5.2 | 4.5.1 | 4.5.0 | ⚠️ |
| `nvidia-ml-py` | 13.590.48 | 13.590.48 | 12.575.51 | ⚠️ |
| `nvidia-nccl-cu13` | 2.28.9 | 2.28.9 | 2.28.9 | ✅ |
| `nvidia-nvjitlink` | 13.0.88 | 13.0.88 | 13.0.88 | ✅ |
| `nvidia-nvshmem-cu13` | 3.4.5 | 3.4.5 | 3.4.5 | ✅ |
| `nvidia-nvtx` | 13.0.85 | 13.0.85 | 13.0.85 | ✅ |
| `nvidia-nvvm` | 13.2.78 | — | — | ➖ |
| `openai` | 2.38.0 | 2.36.0 | 1.91.0 | ⚠️ |
| `outlines` | — | 1.3.0 | — | ➖ |
| `outlines-core` | — | 0.2.14 | — | ➖ |
| `packaging` | 26.2 | 26.2 | 25.0 | ⚠️ |
| `pillow` | 12.2.0 | 12.2.0 | 11.2.1 | ⚠️ |
| `pip` | 26.0.1 | 26.0.1 | — | ➖ |
| `platformdirs` | 4.10.0 | 4.9.6 | 4.3.8 | ⚠️ |
| `prettytable` | 3.17.0 | 3.17.0 | 3.16.0 | ⚠️ |
| `protobuf` | 4.25.0 | 6.33.6 | 4.25.0 | ⚠️ |
| `psutil` | 7.2.2 | 7.2.2 | 7.0.0 | ⚠️ |
| `pydantic` | 2.13.4 | 2.13.4 | 2.7.0 | ⚠️ |
| `pydantic-core` | 2.46.4 | 2.46.4 | 2.18.1 | ⚠️ |
| `pygments` | 2.20.0 | 2.20.0 | 2.19.2 | ⚠️ |
| `pyyaml` | 6.0.3 | 6.0.3 | 6.0.2 | ⚠️ |
| `regex` | 2026.5.9 | 2026.5.9 | 2024.11.6 | ⚠️ |
| `requests` | 2.34.2 | 2.28.1 | 2.32.4 | ⚠️ |
| `rich` | 15.0.0 | 15.0.0 | 14.1.0 | ⚠️ |
| `rtp-kernel` | 0.1.0+cu13.4a1a7e3-cp310 | 0.1.0 | 0.1.0+cu13.4a1a7e3-cp310 | ⚠️ |
| `safetensors` | 0.7.0 | 0.7.0 | 0.7.0 | ✅ |
| `scikit-learn` | 1.5.1 | 1.7.2 | 1.7.0 | ⚠️ |
| `scipy` | 1.14.1 | 1.15.3 | 1.15.3 | ⚠️ |
| `sentence-transformers` | 2.7.0 | 5.5.0 | 2.7.0 | ⚠️ |
| `setproctitle` | 1.3.7 | 1.3.7 | 1.3.6 | ⚠️ |
| `setuptools` | 60.5.0 | 81.0.0 | 60.5.0 | ⚠️ |
| `six` | @URL | 1.16.0 | 1.17.0 | ⚠️ |
| `soundfile` | 0.13.1 | 0.13.1 | 0.13.1 | ✅ |
| `soxr` | 1.1.0 | 1.1.0 | 0.5.0.post1 | ⚠️ |
| `starlette` | 0.41.3 | 0.52.1 | 0.41.3 | ⚠️ |
| `sympy` | 1.14.0 | 1.14.0 | 1.14.0 | ✅ |
| `tabulate` | 0.10.0 | 0.10.0 | 0.9.0 | ⚠️ |
| `tilelang` | 0.1.9 | 0.1.9 | 0.1.9 | ✅ |
| `timm` | 0.9.12 | 1.0.27 | 0.9.12 | ⚠️ |
| `tokenizers` | 0.21.4 | 0.22.2 | 0.21.4 | ⚠️ |
| `torch` | 2.11.0+cu130-cp310 | 2.11.0 | 2.11.0+cu130-cp310 | ⚠️ |
| `torchvision` | 0.26.0+cu130-cp310 | 0.26.0 | 0.26.0+cu130-cp310 | ⚠️ |
| `tqdm` | @URL | 4.64.1 | 4.67.1 | ⚠️ |
| `transformers` | 4.51.2 | 5.9.0 | 4.51.2 | ⚠️ |
| `triton` | 3.6.0 | 3.6.0 | 3.6.0 | ✅ |
| `typer` | 0.25.0 | 0.25.0 | 0.19.2 | ⚠️ |
| `typing-extensions` | 4.15.0 | 4.15.0 | 4.14.0 | ⚠️ |
| `urllib3` | 2.7.0 | 1.26.14 | 2.5.0 | ⚠️ |
| `uvicorn` | 0.30.0 | 0.30.0 | 0.30.0 | ✅ |
| `vllm` | — | 0.21.0 | — | ➖ |
| `wheel` | 0.37.1 | 0.37.1 | — | ➖ |
| `xgrammar` | — | 0.2.1 | — | ➖ |

_关键包 111 项，其中非一致 82 项_

## 三者都存在但版本不一致（mismatch）

| 包 | 线上 whl | 本地 whl | cuda13 lock | 状态 |
|---|---|---|---|---|
| `aiohappyeyeballs` | 2.6.2 | 2.6.1 | 2.6.1 | ⚠️ |
| `aiohttp` | 3.13.5 | 3.13.5 | 3.12.13 | ⚠️ |
| `aiosignal` | 1.4.0 | 1.4.0 | 1.3.2 | ⚠️ |
| `anyio` | 4.13.0 | 4.13.0 | 4.9.0 | ⚠️ |
| `apache-tvm-ffi` | 0.1.10 | 0.1.9 | 0.1.10 | ⚠️ |
| `attrs` | 26.1.0 | 26.1.0 | 25.3.0 | ⚠️ |
| `audioread` | 3.1.0 | 3.1.0 | 3.0.1 | ⚠️ |
| `certifi` | 2026.5.20 | 2022.12.7 | 2025.6.15 | ⚠️ |
| `cffi` | @URL | 1.15.1 | 1.17.1 | ⚠️ |
| `charset-normalizer` | @URL | 2.0.4 | 3.4.2 | ⚠️ |
| `click` | 8.3.3 | 8.3.3 | 8.2.1 | ⚠️ |
| `concurrent-log-handler` | 0.9.29 | 0.9.29 | 0.9.28 | ⚠️ |
| `cryptography` | @URL | 38.0.4 | 42.0.8 | ⚠️ |
| `cuda-bindings` | 13.2.0 | 13.2.0 | 13.0.3 | ⚠️ |
| `cuda-pathfinder` | 1.5.4 | 1.5.4 | 1.3.2 | ⚠️ |
| `cuda-python` | 13.2.0 | 13.2.0 | 13.0.3 | ⚠️ |
| `dashscope` | 1.25.19 | 1.25.18 | 1.23.5 | ⚠️ |
| `decorator` | 5.3.1 | 5.2.1 | 5.2.1 | ⚠️ |
| `deep-ep` | 1.2.1.12+37fda1c.base-cp310 | 1.2.1.12+37fda1c.base | 1.2.1.12+37fda1c.base-cp310 | ⚠️ |
| `deep-gemm` | 2.5.0+local-cp310 | 2.5.0+8a4dfba | 2.5.0+8a4dfba-cp310 | ⚠️ |
| `einops` | 0.8.2 | 0.8.2 | 0.8.1 | ⚠️ |
| `exceptiongroup` | 1.3.1 | 1.3.1 | 1.3.0 | ⚠️ |
| `fast-hadamard-transform` | 1.1.0-cp310 | 1.1.0 | 1.1.0-cp310 | ⚠️ |
| `fastapi` | 0.115.6 | 0.136.1 | 0.115.6 | ⚠️ |
| `fastsafetensors` | 0.1.20+ali-cp310 | 0.3.2 | 0.1.20+ali-cp310 | ⚠️ |
| `filelock` | 3.29.0 | 3.29.0 | 3.20.0 | ⚠️ |
| `flash-mla` | 1.0.0+cb10b79-cp310 | 1.0.0+cb10b79 | 1.0.0+cb10b79-cp310 | ⚠️ |
| `flashinfer-python` | 0.6.12 | 0.6.11.post1 | 0.6.11 | ⚠️ |
| `frozenlist` | 1.8.0 | 1.8.0 | 1.7.0 | ⚠️ |
| `fsspec` | 2026.3.0 | 2026.3.0 | 2023.10.0 | ⚠️ |
| `grpcio` | 1.62.0 | 1.80.0 | 1.62.0 | ⚠️ |
| `grpcio-tools` | 1.57.0 | 1.80.0 | 1.57.0 | ⚠️ |
| `hf-xet` | 1.4.3 | 1.4.3 | 1.1.5 | ⚠️ |
| `huggingface-hub` | 0.36.2 | 1.16.1 | 0.33.0 | ⚠️ |
| `idna` | @URL | 3.4 | 3.10 | ⚠️ |
| `jiter` | 0.15.0 | 0.14.0 | 0.10.0 | ⚠️ |
| `joblib` | 1.5.3 | 1.5.3 | 1.5.1 | ⚠️ |
| `json5` | 0.14.0 | 0.14.0 | 0.12.0 | ⚠️ |
| `lazy-loader` | 0.5 | 0.5 | 0.4 | ⚠️ |
| `llvmlite` | 0.47.0 | 0.47.0 | 0.44.0 | ⚠️ |
| `lru-dict` | 1.4.1 | 1.4.1 | 1.3.0 | ⚠️ |
| `markupsafe` | 3.0.3 | 3.0.3 | 3.0.2 | ⚠️ |
| `msgpack` | 1.1.2 | 1.1.2 | 1.1.1 | ⚠️ |
| `multidict` | 6.7.1 | 6.7.1 | 6.5.0 | ⚠️ |
| `numba` | 0.65.1 | 0.65.0 | 0.61.2 | ⚠️ |
| `numpy` | 1.26.4 | 2.2.6 | 1.26.4 | ⚠️ |
| `nvidia-cudnn-frontend` | 1.24.0 | 1.18.0 | 1.23.0 | ⚠️ |
| `nvidia-cutlass-dsl` | 4.5.2 | 4.5.1 | 4.5.0 | ⚠️ |
| `nvidia-cutlass-dsl-libs-base` | 4.5.2 | 4.5.1 | 4.5.0 | ⚠️ |
| `nvidia-ml-py` | 13.590.48 | 13.590.48 | 12.575.51 | ⚠️ |
| `openai` | 2.38.0 | 2.36.0 | 1.91.0 | ⚠️ |
| `orjson` | 3.11.9 | 3.11.9 | 3.10.18 | ⚠️ |
| `oss2` | 2.19.1 | 2.19.1 | 2.18.4 | ⚠️ |
| `packaging` | 26.2 | 26.2 | 25.0 | ⚠️ |
| `partial-json-parser` | 0.2.1.1.post7 | 0.2.1.1.post7 | 0.2.1.1.post6 | ⚠️ |
| `pillow` | 12.2.0 | 12.2.0 | 11.2.1 | ⚠️ |
| `pillow-heif` | 1.3.0 | 1.3.0 | 0.22.0 | ⚠️ |
| `platformdirs` | 4.10.0 | 4.9.6 | 4.3.8 | ⚠️ |
| `pooch` | 1.9.0 | 1.9.0 | 1.8.2 | ⚠️ |
| `prettytable` | 3.17.0 | 3.17.0 | 3.16.0 | ⚠️ |
| `propcache` | 0.5.2 | 0.5.2 | 0.3.2 | ⚠️ |
| `protobuf` | 4.25.0 | 6.33.6 | 4.25.0 | ⚠️ |
| `psutil` | 7.2.2 | 7.2.2 | 7.0.0 | ⚠️ |
| `pycparser` | @URL | 2.21 | 2.22 | ⚠️ |
| `pydantic` | 2.13.4 | 2.13.4 | 2.7.0 | ⚠️ |
| `pydantic-core` | 2.46.4 | 2.46.4 | 2.18.1 | ⚠️ |
| `pygments` | 2.20.0 | 2.20.0 | 2.19.2 | ⚠️ |
| `pynvml` | 13.0.1 | 13.0.1 | 12.0.0 | ⚠️ |
| `pyopenssl` | @URL | 22.0.0 | 24.1.0 | ⚠️ |
| `pyyaml` | 6.0.3 | 6.0.3 | 6.0.2 | ⚠️ |
| `regex` | 2026.5.9 | 2026.5.9 | 2024.11.6 | ⚠️ |
| `requests` | 2.34.2 | 2.28.1 | 2.32.4 | ⚠️ |
| `rich` | 15.0.0 | 15.0.0 | 14.1.0 | ⚠️ |
| `rtp-kernel` | 0.1.0+cu13.4a1a7e3-cp310 | 0.1.0 | 0.1.0+cu13.4a1a7e3-cp310 | ⚠️ |
| `scikit-learn` | 1.5.1 | 1.7.2 | 1.7.0 | ⚠️ |
| `scipy` | 1.14.1 | 1.15.3 | 1.15.3 | ⚠️ |
| `sentence-transformers` | 2.7.0 | 5.5.0 | 2.7.0 | ⚠️ |
| `sentencepiece` | 0.2.0 | 0.2.1 | 0.2.0 | ⚠️ |
| `setproctitle` | 1.3.7 | 1.3.7 | 1.3.6 | ⚠️ |
| `setuptools` | 60.5.0 | 81.0.0 | 60.5.0 | ⚠️ |
| `six` | @URL | 1.16.0 | 1.17.0 | ⚠️ |
| `soxr` | 1.1.0 | 1.1.0 | 0.5.0.post1 | ⚠️ |
| `starlette` | 0.41.3 | 0.52.1 | 0.41.3 | ⚠️ |
| `tabulate` | 0.10.0 | 0.10.0 | 0.9.0 | ⚠️ |
| `thrift` | 0.23.0 | 0.22.0 | 0.22.0 | ⚠️ |
| `tiktoken` | 0.7.0 | 0.12.0 | 0.7.0 | ⚠️ |
| `timm` | 0.9.12 | 1.0.27 | 0.9.12 | ⚠️ |
| `tokenizers` | 0.21.4 | 0.22.2 | 0.21.4 | ⚠️ |
| `torch` | 2.11.0+cu130-cp310 | 2.11.0 | 2.11.0+cu130-cp310 | ⚠️ |
| `torchvision` | 0.26.0+cu130-cp310 | 0.26.0 | 0.26.0+cu130-cp310 | ⚠️ |
| `tqdm` | @URL | 4.64.1 | 4.67.1 | ⚠️ |
| `transformers` | 4.51.2 | 5.9.0 | 4.51.2 | ⚠️ |
| `typer` | 0.25.0 | 0.25.0 | 0.19.2 | ⚠️ |
| `typing-extensions` | 4.15.0 | 4.15.0 | 4.14.0 | ⚠️ |
| `urllib3` | 2.7.0 | 1.26.14 | 2.5.0 | ⚠️ |
| `wcwidth` | 0.7.0 | 0.7.0 | 0.2.13 | ⚠️ |
| `websocket-client` | 1.9.0 | 1.9.0 | 1.8.0 | ⚠️ |
| `yarl` | 1.24.2 | 1.23.0 | 1.20.1 | ⚠️ |

_共 98 项_

## 仅线上有，本地+lock 都缺

| 包 | 线上 whl |
|---|---|
| `nvidia-cuda-crt` | 13.3.33 |
| `nvidia-cuda-nvcc` | 13.2.78 |
| `nvidia-cuda-tileiras` | 13.2.78 |
| `nvidia-nvvm` | 13.2.78 |
| `rtp-llm` | 0.2.0-cp310 |

_共 5 项_

## 仅本地有，线上+lock 都缺

| 包 | 本地 whl |
|---|---|
| `accelerate` | 1.13.0 |
| `anthropic` | 0.104.1 |
| `astor` | 0.8.1 |
| `blake3` | 1.0.8 |
| `cachetools` | 7.1.4 |
| `cbor2` | 6.1.1 |
| `cfgv` | 3.5.0 |
| `compressed-tensors` | 0.15.0.1 |
| `depyf` | 0.20.0 |
| `detect-installer` | 0.1.0 |
| `dill` | 0.4.1 |
| `diskcache` | 5.6.3 |
| `distlib` | 0.4.0 |
| `dnspython` | 2.8.0 |
| `docstring-parser` | 0.18.0 |
| `email-validator` | 2.3.0 |
| `fastapi-cli` | 0.0.24 |
| `fastapi-cloud-cli` | 0.18.0 |
| `fastar` | 0.11.0 |
| `genson` | 1.3.0 |
| `gguf` | 0.19.0 |
| `googleapis-common-protos` | 1.75.0 |
| `greenlet` | 3.5.0 |
| `hf-transfer` | 0.1.9 |
| `httptools` | 0.7.1 |
| `httpx-sse` | 0.4.3 |
| `identify` | 2.6.19 |
| `ijson` | 3.5.0 |
| `interegular` | 0.3.3 |
| `jsonpath-ng` | 1.8.0 |
| `jsonschema` | 4.26.0 |
| `jsonschema-specifications` | 2025.9.1 |
| `lark` | 1.2.2 |
| `llguidance` | 1.3.0 |
| `lm-format-enforcer` | 0.11.3 |
| `loguru` | 0.7.3 |
| `mcp` | 1.12.4 |
| `mistral-common` | 1.11.2 |
| `model-hosting-container-standards` | 0.1.15 |
| `msgspec` | 0.21.1 |
| `nodeenv` | 1.10.0 |
| `openai-harmony` | 0.0.8 |
| `opencv-python-headless` | 4.13.0.92 |
| `opentelemetry-api` | 1.42.1 |
| `opentelemetry-exporter-otlp` | 1.42.1 |
| `opentelemetry-exporter-otlp-proto-common` | 1.42.1 |
| `opentelemetry-exporter-otlp-proto-grpc` | 1.42.1 |
| `opentelemetry-exporter-otlp-proto-http` | 1.42.1 |
| `opentelemetry-proto` | 1.42.1 |
| `opentelemetry-sdk` | 1.42.1 |
| `opentelemetry-semantic-conventions` | 0.63b1 |
| `opentelemetry-semantic-conventions-ai` | 0.5.1 |
| `outlines` | 1.3.0 |
| `outlines-core` | 0.2.14 |
| `playwright` | 1.60.0 |
| `pre-commit` | 4.6.0 |
| `prometheus-client` | 0.25.0 |
| `prometheus-fastapi-instrumentator` | 7.1.0 |
| `py-cpuinfo` | 9.0.0 |
| `pybase64` | 1.4.3 |
| `pycountry` | 26.2.16 |
| `pydantic-extra-types` | 2.11.1 |
| `pydantic-settings` | 2.14.1 |
| `pyee` | 13.0.1 |
| `python-discovery` | 1.3.1 |
| `python-dotenv` | 1.2.2 |
| `python-json-logger` | 4.1.0 |
| `python-multipart` | 0.0.29 |
| `pyzmq` | 27.1.0 |
| `quack-kernels` | 0.4.1 |
| `referencing` | 0.37.0 |
| `rich-toolkit` | 0.19.10 |
| `rignore` | 0.7.6 |
| `rpds-py` | 0.30.0 |
| `sentry-sdk` | 2.60.0 |
| `sse-starlette` | 3.4.4 |
| `supervisor` | 4.3.0 |
| `tokenspeed-mla` | 0.1.2 |
| `tokenspeed-triton` | 3.7.10.post20260505 |
| `tomli` | 2.4.1 |
| `torchaudio` | 2.11.0 |
| `uvloop` | 0.22.1 |
| `virtualenv` | 21.3.3 |
| `vllm` | 0.21.0 |
| `watchfiles` | 1.2.0 |
| `websockets` | 16.0 |
| `webterminal-cli` | 0.1.0 |
| `xgrammar` | 0.2.1 |

_共 88 项_

## 仅 lock 有，线上+本地都缺

| 包 | cuda13 lock |
|---|---|
| `mysql-connector-python` | 9.3.0 |
| `pybind11-stubgen` | 2.5.5 |
| `xfastertransformer-devel` | 1.8.1.1 |
| `xfastertransformer-devel-icx` | 1.8.1.1 |

_共 4 项_

## 两两存在（剩下一处缺失）

| 包 | 线上 whl | 本地 whl | cuda13 lock |
|---|---|---|---|
| `annotated-doc` | 0.0.4 | 0.0.4 | — |
| `bitsandbytes` | 0.49.2 | — | 0.46.0 |
| `blobfile` | 3.2.0 | — | 3.0.0 |
| `brotlipy` | 0.7.0 | 0.7.0 | — |
| `conda` | 23.1.0 | 23.1.0 | — |
| `conda-content-trust` | @URL | 0.1.3 | — |
| `conda-package-handling` | @URL | 2.0.2 | — |
| `conda-package-streaming` | @URL | 0.7.0 | — |
| `contourpy` | 1.3.2 | — | 1.3.2 |
| `cpm-kernels` | 1.0.11 | — | 1.0.11 |
| `cuda-toolkit` | 13.0.2 | 13.0.2 | — |
| `cycler` | 0.12.1 | — | 0.12.1 |
| `decord` | 0.6.0 | — | 0.6.0 |
| `fast-safetensors` | 0.7.3+torch2.11.cu130-cp310 | — | 0.7.3+torch2.11.cu130-cp310 |
| `flashinfer-cubin` | — | 0.6.11.post1 | 0.6.11 |
| `fonttools` | 4.63.0 | — | 4.58.4 |
| `importlib-metadata` | 9.0.0 | — | 8.7.0 |
| `jieba` | 0.42.1 | — | 0.42.1 |
| `kiwisolver` | 1.5.0 | — | 1.4.8 |
| `lxml` | 6.1.1 | — | 6.0.0 |
| `matplotlib` | 3.10.9 | — | 3.10.3 |
| `modelscope` | 1.36.2 | 1.36.2 | — |
| `nvitop` | 1.6.2 | 1.6.2 | — |
| `onnx` | 1.17.0 | — | 1.16.0 |
| `pillow-avif-plugin` | 1.5.5 | — | 1.5.2 |
| `pip` | 26.0.1 | 26.0.1 | — |
| `pluggy` | @URL | 1.0.0 | — |
| `py-spy` | 0.4.2 | — | 0.4.0 |
| `pyarrow` | 14.0.0 | — | 14.0.0 |
| `pycosat` | @URL | 0.6.4 | — |
| `pycryptodomex` | 3.23.0 | — | 3.23.0 |
| `pyodps` | 0.12.6 | — | 0.12.3 |
| `pyparsing` | 3.3.2 | — | 3.2.3 |
| `pysocks` | @URL | 1.7.1 | — |
| `python-dateutil` | 2.9.0.post0 | — | 2.9.0.post0 |
| `ruamel.yaml` | @URL | 0.17.21 | — |
| `ruamel.yaml.clib` | @URL | 0.2.6 | — |
| `toolz` | @URL | 0.12.0 | — |
| `typing-inspection` | 0.4.2 | 0.4.2 | — |
| `uv` | 0.11.7 | 0.11.7 | — |
| `wheel` | 0.37.1 | 0.37.1 | — |
| `zipp` | 4.1.0 | — | 3.23.0 |
| `zstandard` | @URL | 0.18.0 | — |

_共 43 项_

## 三者完全一致（match）

| 包 | 版本 |
|---|---|
| `aliyun-python-sdk-core` | 2.16.0 |
| `aliyun-python-sdk-kms` | 2.16.5 |
| `annotated-types` | 0.7.0 |
| `async-timeout` | 5.0.1 |
| `cloudpickle` | 3.1.2 |
| `crcmod` | 1.7 |
| `cuda-tile` | 1.3.0 |
| `dacite` | 1.9.2 |
| `distro` | 1.9.0 |
| `h11` | 0.16.0 |
| `httpcore` | 1.0.9 |
| `httpx` | 0.28.1 |
| `jinja2` | 3.1.6 |
| `jmespath` | 0.10.0 |
| `librosa` | 0.11.0 |
| `markdown-it-py` | 4.0.0 |
| `mdurl` | 0.1.2 |
| `ml-dtypes` | 0.5.4 |
| `mpmath` | 1.3.0 |
| `nest-asyncio` | 1.6.0 |
| `networkx` | 3.4.2 |
| `ninja` | 1.13.0 |
| `nvidia-cublas` | 13.1.0.3 |
| `nvidia-cuda-cupti` | 13.0.85 |
| `nvidia-cuda-nvrtc` | 13.0.88 |
| `nvidia-cuda-runtime` | 13.0.96 |
| `nvidia-cudnn-cu13` | 9.19.0.56 |
| `nvidia-cufft` | 12.0.0.61 |
| `nvidia-cufile` | 1.15.1.6 |
| `nvidia-curand` | 10.4.0.35 |
| `nvidia-cusolver` | 12.0.4.66 |
| `nvidia-cusparse` | 12.6.3.3 |
| `nvidia-cusparselt-cu13` | 0.8.0 |
| `nvidia-nccl-cu13` | 2.28.9 |
| `nvidia-nvjitlink` | 13.0.88 |
| `nvidia-nvshmem-cu13` | 3.4.5 |
| `nvidia-nvtx` | 13.0.85 |
| `portalocker` | 3.2.0 |
| `pycryptodome` | 3.23.0 |
| `safetensors` | 0.7.0 |
| `shellingham` | 1.5.4 |
| `sniffio` | 1.3.1 |
| `soundfile` | 0.13.1 |
| `sympy` | 1.14.0 |
| `threadpoolctl` | 3.6.0 |
| `tilelang` | 0.1.9 |
| `torch-c-dlpack-ext` | 0.1.5 |
| `triton` | 3.6.0 |
| `uvicorn` | 0.30.0 |
| `z3-solver` | 4.15.4.0 |

_共 50 项_

## 统计

- 线上 whl：195 个包
- 本地 whl：257 个包
- cuda13 lock：175 个包
- 全集合：288 个包
  - 三者一致：50
  - 三者都在但版本不一致：98
  - 仅出现在两处：43
  - 仅线上：5
  - 仅本地：88
  - 仅 lock：4
