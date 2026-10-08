load("//bazel:cutlass_dsl.bzl", "cutlass_install")
load("@pip_gpu_cuda13_torch//:requirements.bzl", x86_requirement = "requirement", x86_wheel = "whl_requirement")
load("@pip_cuda13_arm_torch//:requirements.bzl", arm_requirement = "requirement", arm_wheel = "whl_requirement")

package(default_visibility = ["//visibility:public"])

_COMPONENTS = [
    "nvidia-cutlass-dsl",
    "nvidia-cutlass-dsl-libs-core",
    "nvidia-cutlass-dsl-libs-base",
    "nvidia-cutlass-dsl-libs-cu12",
    "nvidia-cutlass-dsl-libs-cu13",
]

_CUDA13_ONLY = select({
    "//:using_cuda13_x86": [],
    "//:using_cuda13_arm": [],
    "//conditions:default": ["@platforms//:incompatible"],
})

cutlass_install(
    name = "cutlass_site",
    version = "4.7.1",
    target_compatible_with = _CUDA13_ONLY,
    wheels = select({
        "//:using_cuda13_x86": [x86_wheel(name) for name in _COMPONENTS],
        "//:using_cuda13_arm": [arm_wheel(name) for name in _COMPONENTS],
        "//conditions:default": [],
    }),
)

py_library(
    name = "cutlass_dsl",
    target_compatible_with = _CUDA13_ONLY,
    data = [":cutlass_site"],
    imports = [
        "cutlass_site",
        "cutlass_site/nvidia_cutlass_dsl/dsl_packages",
    ],
    deps = select({
        "//:using_cuda13_x86": [x86_requirement("nvidia-cutlass-dsl")],
        "//:using_cuda13_arm": [arm_requirement("nvidia-cutlass-dsl")],
        "//conditions:default": [],
    }),
)

py_test(
    name = "import_test",
    target_compatible_with = _CUDA13_ONLY,
    srcs = ["import_test.py"],
    data = ["//rtp_llm/models_py:utils"],
    deps = [":cutlass_dsl"],
)
