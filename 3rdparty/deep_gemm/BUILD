load("@rules_python//python:defs.bzl", "py_library")
load("@//bazel:arch_select.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

requirement([
    "torch",
    "setuptools",
])

genrule(
    name = "build_deep_gemm_cpp",
    srcs = glob([
        "csrc/**/*",
        "deep_gemm/**/*",
        "third-party/**/*",
        "setup.py",
        "*.py",
        "*.md", 
        "*.txt",
    ]),
    outs = ["deep_gemm_cpp.so"],
    cmd = """
    set -e
    PYTHON_BIN=$${PYTHON_BIN_PATH:-python3}
    export PYTHONPATH="$${PYTHONPATH:-}"
    
    TORCH_PATH=$$(find /root/.cache/bazel -path "*pip_gpu_cuda12_torch_torch/site-packages" 2>/dev/null | head -1)
    if [ -n "$$TORCH_PATH" ]; then
        export PYTHONPATH="$$TORCH_PATH:$$PYTHONPATH"
    else
        echo "Error: bazel-managed torch not found"
        exit 1
    fi
    
    $$PYTHON_BIN -c "import torch; print('torch:', torch.__version__)"
    
    SETUP_DIR=$$(dirname $(location setup.py))
    cd $$SETUP_DIR
    $$PYTHON_BIN setup.py build_ext --inplace
    
    SO_FILE=$$(find . -name "deep_gemm_cpp*.so" | head -1)
    if [ -f "$$SO_FILE" ]; then
        cd -
        cp "$$SETUP_DIR/$$SO_FILE" "$@"
    else
        echo "Error: Could not find generated .so file"
        exit 1
    fi
    """,
    tools = select({
        "@//:using_cuda12": ["@pip_gpu_cuda12_torch//torch"],
        "@//:using_cuda11": ["@pip_gpu_torch//torch"],
        "@//:using_rocm": ["@pip_gpu_rocm_torch//torch"],
        "@//:using_arm": ["@pip_arm_torch//torch"],
        "//conditions:default": ["@pip_cpu_torch//torch"],
    }),
    visibility = ["//visibility:public"],
)

alias(
    name = "deep_gemm_cpp",
    actual = ":build_deep_gemm_cpp",
    visibility = ["//visibility:public"],
)

py_library(
    name = "deep_gemm",
    srcs = glob(["deep_gemm/**/*.py"]),
    data = glob([
        "deep_gemm/include/**/*",
        "deep_gemm/**/*",
        "third-party/**/*",
    ]) + [":deep_gemm_cpp.so"],
    imports = ["deep_gemm"],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "deep_gemm_files",
    srcs = glob([
        "deep_gemm/**/*",
        "deep_gemm/include/**/*",
        "third-party/**/*",
        "csrc/**/*",
    ]),
    visibility = ["//visibility:public"],
) 