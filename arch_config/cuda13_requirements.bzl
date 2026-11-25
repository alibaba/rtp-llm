"""Default pip locks and wheel metadata for the CUDA 13 dependency profile."""

CUDA13_X86_WHEEL_REQUIREMENTS = [
    "nvidia-cutlass-dsl==4.4.2",
    "nvidia-cutlass-dsl-libs-cu13==4.4.2",
]
CUDA13_ARM_WHEEL_REQUIREMENTS = [
    "nvidia-cutlass-dsl==4.5.0",
    "nvidia-cutlass-dsl-libs-base==4.5.0",
]

CUDA13_X86_REQUIREMENTS_LOCK = "@rtp_pip_deps//:requirements_lock_torch_gpu_cuda13.txt"
CUDA13_ARM_REQUIREMENTS_LOCK = "@rtp_pip_deps//:requirements_lock_cuda13_arm.txt"
