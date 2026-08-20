# GPU compiler selection constraints + target platforms.
# A custom constraint_setting (gpu_compiler) gates the CUDA / ROCm nvcc/hipcc wrapper
# toolchains (3rdparty/cuda_config/crosstool, 3rdparty/gpus/crosstool) so that they win
# resolution only when the target platform explicitly carries the cuda_compiler /
# rocm_compiler constraint; otherwise (CPU / arm / host tools) resolution falls back to the
# default local_config_cc gcc, avoiding hijacking every linux-x86_64 build.
package(default_visibility = ["//visibility:public"])

# Default value = none (neither cuda nor rocm) → the auto-detected host platform carries no such constraint → gcc.
constraint_setting(
    name = "gpu_compiler",
)

constraint_value(
    name = "cuda_compiler",
    constraint_setting = ":gpu_compiler",
)

constraint_value(
    name = "rocm_compiler",
    constraint_setting = ":gpu_compiler",
)

# Target platforms: inherit the host's os/cpu constraints + append gpu_compiler.
# build:cuda / build:rocm in .bazelrc point at them via --platforms.
platform(
    name = "cuda_linux_x86_64",
    constraint_values = [
        "@platforms//os:linux",
        "@platforms//cpu:x86_64",
        ":cuda_compiler",
    ],
)

# Used by cuda12_9_arm; only valid for native builds on aarch64 hosts (cross-compiling nvcc from an x86 host is unsupported).
platform(
    name = "cuda_linux_aarch64",
    constraint_values = [
        "@platforms//os:linux",
        "@platforms//cpu:aarch64",
        ":cuda_compiler",
    ],
)

# Used by cpu / arm (pure CPU). No gpu_compiler → cc resolution still falls back to gcc.
# Platforms must be declared, not taken from the environment: without --platforms the target
# CPU = machine CPU, while each profile's lock only has wheels for the arch declared in deps.json.
platform(
    name = "linux_x86_64",
    constraint_values = [
        "@platforms//os:linux",
        "@platforms//cpu:x86_64",
    ],
)

platform(
    name = "linux_aarch64",
    constraint_values = [
        "@platforms//os:linux",
        "@platforms//cpu:aarch64",
    ],
)

platform(
    name = "rocm_linux_x86_64",
    constraint_values = [
        "@platforms//os:linux",
        "@platforms//cpu:x86_64",
        ":rocm_compiler",
    ],
)
