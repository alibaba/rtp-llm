load("@arch_config//:arch_select.bzl", "deep_ep_py_deps")

deep_ep_py_deps()

# Authoritative rank coordinate system: pure stdlib, zero deps.
py_library(
    name = "rank_layout",
    srcs = ["rank_layout.py"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "collective_torch",
    srcs = [
        "collective_torch.py",
        "rocm_rccl.py",
        "symm_mem.py",
    ],
    deps = [
        ":rank_layout",
        "//rtp_llm:torch",
    ],
    visibility = ["//visibility:public"],
)

# Stage-local expert-parallel context: the CP4EP4PP2 stage roster plus the
# strict resolver for the experimental DSV4_PP_EP_* opt-in. Pure metadata over
# rank_layout; imports torch only lazily inside the verification helper.
py_library(
    name = "ep_stage_context",
    srcs = ["ep_stage_context.py"],
    deps = [
        ":rank_layout",
    ],
    visibility = ["//visibility:public"],
)

py_library(
    name = "deepep_wrapper",
    srcs = ["deepep_wrapper.py"],
    deps = [
        ":collective_torch",
        "//rtp_llm:torch",
    ] + select({
        "@//:using_cuda12":["deep_ep_py"],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
)

py_library(
    name = "moriep_wrapper",
    srcs = ["moriep_wrapper.py"],
    deps = [
        "//rtp_llm:torch",
    ],
    visibility = ["//visibility:public"],
)

py_library(
    name = "user_buffers",
    srcs = ["user_buffers.py"],
    deps = [
        "//rtp_llm:torch",
    ],
    visibility = ["//visibility:public"],
)
