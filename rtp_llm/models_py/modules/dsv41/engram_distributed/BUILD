package(default_visibility = ["//visibility:public"])

# Importing the control plane and storage lifecycle is stdlib-only. Torch and
# CUDA are loaded only by explicit checkpoint/attach operations.
py_library(
    name = "core",
    srcs = glob(["*.py"]),
    deps = ["//rtp_llm:numa"],
)

py_library(
    name = "runtime",
    deps = [
        ":core",
        "//rtp_llm:numpy",
        "//rtp_llm:ops",
        "//rtp_llm:safetensors",
        "//rtp_llm:torch",
    ],
)
