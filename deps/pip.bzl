load("@rules_python//python:pip.bzl", "pip_parse")

# Index configuration (opensource side). Strict separation rule:
#   - NO artlab.alibaba-inc.com (intranet-only host, not reachable from public)
#   - rtp-opensource/rtp_llm/simple: RTP-LLM custom wheels (flash_attn, deep_gemm,
#     deep_ep, flash_mla, flashinfer_*, rtp_kernel, aiter, ...) — single source of
#     truth, shared with internal side
#   - download.pytorch.org/whl/<cfg>: official PyTorch PEP 503 indexes. Reachable
#     directly from CN (~9 MB/s), no proxy needed. We do NOT use the SJTU mirror
#     (mirror.sjtu.edu.cn/pytorch-wheels/...) because SJTU 301-redirects unknown
#     packages to download.pytorch.org which returns 403 — uv treats 403 as a
#     hard error and the resolve fails for non-pytorch packages (decord, etc.)
#     even though they exist on the PyPI mirror.
#   - mirrors.aliyun.com/pypi/simple: China-friendly PyPI mirror for base packages
# Shared pip/pip-compile flags. Does NOT include PyTorch indexes, which differ
# by target config (CUDA vs ROCm). Keeping them separate avoids uv asking every
# package against both cu129 and rocm7.2 indexes during lockfile regeneration.
PIP_BASE_ARGS = [
    "--cache-dir=~/.cache/pip",
    # --index-url overrides the env's PIP_INDEX_URL (which intranet containers
    # set to artlab via /etc/profile.d/alibaba_pypi_env.sh). Without this line,
    # opensource pip-compile would silently pull from artlab — forbidden.
    "--index-url=https://mirrors.aliyun.com/pypi/simple/",
    # Extra indexes (PEP 503), used for transitive resolution:
    "--extra-index-url=https://rtp-opensource.oss-cn-hangzhou.aliyuncs.com/rtp_llm/simple/",
    "--verbose",
]

PIP_CUDA_EXTRA_ARGS = PIP_BASE_ARGS + [
    "--extra-index-url=https://download.pytorch.org/whl/cu129/",
]

PIP_ROCM_EXTRA_ARGS = PIP_BASE_ARGS + [
    "--extra-index-url=https://download.pytorch.org/whl/rocm7.2/",
]

# Backwards-compatible alias for callers that do not yet care about config.
# NOTE: This alias implies the CUDA context. ROCm callers MUST use
# PIP_ROCM_EXTRA_ARGS explicitly; do not rely on this default.
PIP_EXTRA_ARGS = PIP_CUDA_EXTRA_ARGS

def pip_deps():
    # PPU support is internal-only and the opensource-side arch_select.bzl has
    # no `using_ppu` select branch. But the shared WORKSPACE (repo root is
    # github-opensource/) still calls `pip_ppu_torch_install_deps()`
    # unconditionally — internal builds resolve this via `--override_repository
    # =rtp_deps=internal_source/deps`, where internal pip.bzl registers the
    # real PPU lockfile. Opensource-only builds don't apply that override, so
    # we must declare `pip_ppu_torch` here or WORKSPACE load fails with
    # "Failed to load Starlark extension '@pip_ppu_torch//:requirements.bzl'".
    #
    # The lockfile intentionally aliases the cuda12_9 lockfile. pip_parse is
    # lazy: no PPU wheel is downloaded unless a target depends on
    # `@pip_ppu_torch`, and no opensource target does. The Starlark parse
    # overhead of reading the same lockfile twice is negligible compared to the
    # actual wheel fetch during a real build.
    pip_parse(
        name = "pip_ppu_torch",
        requirements_lock = "@rtp_deps//:requirements_lock_torch_gpu_cuda12_9.txt",
        python_interpreter = "/opt/conda310/bin/python3",
        extra_pip_args = PIP_CUDA_EXTRA_ARGS,
        timeout = 3600,
        quiet = False,
    )

    pip_parse(
        name = "pip_gpu_cuda12_9_torch",
        requirements_lock = "@rtp_deps//:requirements_lock_torch_gpu_cuda12_9.txt",
        python_interpreter = "/opt/conda310/bin/python3",
        extra_pip_args = PIP_CUDA_EXTRA_ARGS,
        timeout = 3600,
        quiet = False,
    )

    # TODO(pip_unify): Backwards-compatible alias for internal source that still
    # references @pip_gpu_cuda12_torch. Points to the same cuda12_9 lockfile.
    # Remove once internal overlays are migrated to @pip_gpu_cuda12_9_torch.
    pip_parse(
        name = "pip_gpu_cuda12_torch",
        requirements_lock = "@rtp_deps//:requirements_lock_torch_gpu_cuda12_9.txt",
        python_interpreter = "/opt/conda310/bin/python3",
        extra_pip_args = PIP_CUDA_EXTRA_ARGS,
        timeout = 3600,
        quiet = False,
    )

    pip_parse(
        name = "pip_cuda12_arm_torch",
        requirements_lock = "@rtp_deps//:requirements_lock_cuda12_arm.txt",
        python_interpreter = "/opt/conda310/bin/python3",
        extra_pip_args = PIP_CUDA_EXTRA_ARGS,
        timeout = 3600,
        quiet = False,
    )

    pip_parse(
        name = "pip_gpu_rocm_torch",
        requirements_lock = "@rtp_deps//:requirements_lock_rocm.txt",
        python_interpreter = "/opt/conda310/bin/python3",
        extra_pip_args = PIP_ROCM_EXTRA_ARGS,
        timeout = 12000,
        quiet = False,
    )
