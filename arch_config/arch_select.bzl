# to wrapper target relate with different system config
load("@pip_cpu_torch//:requirements.bzl", requirement_cpu="requirement")
load("@pip_arm_torch//:requirements.bzl", requirement_arm="requirement")
load("@pip_gpu_cuda12_torch//:requirements.bzl", requirement_gpu_cuda12="requirement")
load("@pip_gpu_cuda12_9_torch//:requirements.bzl", requirement_gpu_cuda12_9="requirement")
load("@pip_cuda12_arm_torch//:requirements.bzl", requirement_cuda12_arm="requirement")
load("@pip_gpu_rocm_torch//:requirements.bzl", requirement_gpu_rocm="requirement")
load("@rtp_llm//bazel:defs.bzl", "copy_so")
# Absence completeness: exceptions[].exists_in in deps.json derives @rtp_llm//deps:absent_map.bzl,
# which requirement() uses to route absent (dependency, profile) branches to the absent_dep
# stub that fails explicitly at analysis time.
# The decision implementation is shared with the internal overlay; only this repo's branch table is given here.
load("@rtp_llm//deps:requirement.bzl", "requirement_libs")

def copy_all_so():
    copy_so("@rtp_llm//:th_transformer")
    copy_so("@rtp_llm//:th_transformer_config")
    copy_so("@rtp_llm//:th_grammar_tokenizer_info")
    copy_so("@rtp_llm//:rtp_compute_ops")

# cuda13 (x86 + arm) has no public pip supply: torch 2.11+cu130 and every wheel built against
# it are published only to the internal index, so this tree declares no cuda13 hub or lock and
# the internal overlay is what supplies cuda13. These two "hubs" therefore hand every name the
# absence stub, which fails explicitly at analysis time -- without them the cuda13 branches
# would be missing from the select and land on another profile's wheels (cuda13_x86 on the
# default cpu hub, cuda13_arm on the cuda12_9_arm hub, whose config_setting it also matches).
def _cuda13_absent(_name):
    return "@arch_config//:python_cuda13_absent"

def _cuda13_arm_absent(_name):
    return "@arch_config//:python_cuda13_arm_absent"

# Six hubs + two absence branches: every accelerator profile is explicit, otherwise a missing
# branch silently falls to default (cpu hub). One table for every requirement_libs caller —
# a per-caller copy drifts. default keeps only cpu (the internal version adds a ppu branch).
_REQUIREMENT_BRANCHES = [
    ("@rtp_llm//:cuda_pre_12_9", "cuda12_6", requirement_gpu_cuda12),
    ("@rtp_llm//:using_cuda12_9_x86", "cuda12_9", requirement_gpu_cuda12_9),
    ("@rtp_llm//:using_cuda12_arm", "cuda12_9_arm", requirement_cuda12_arm),
    ("@rtp_llm//:using_cuda13_x86", "cuda13", _cuda13_absent),
    ("@rtp_llm//:using_cuda13_arm", "cuda13_arm", _cuda13_arm_absent),
    ("@rtp_llm//:using_rocm", "rocm", requirement_gpu_rocm),
    ("@rtp_llm//:using_arm", "arm", requirement_arm),
]

# xgrammar's wheel metadata pulls apache-tvm-ffi (and triton on x86), which only
# the cuda12_9/cuda13 locks carry; dash_sc imports it optionally and degrades
# gracefully, so the other platforms resolve it to nothing.
_DSV4_PLATFORM_ONLY = ["xgrammar"]

def requirement(names):
    requirement_libs(names, _REQUIREMENT_BRANCHES, "cpu", requirement_cpu)

def cache_store_deps():
    native.alias(
        name = "cache_store_arch_select_impl",
        actual = "@rtp_llm//rtp_llm/cpp/disaggregate/cache_store:cache_store_base_impl"
    )

def transfer_backend_deps():
    native.alias(
        name = "transfer_backend_arch_select_impl",
        actual = "@rtp_llm//rtp_llm/cpp/cache/connector/p2p/transfer:transfer_backend_base_impl",
    )

def embedding_arpc_deps():
    native.alias(
        name = "embedding_arpc_deps",
        actual = "@rtp_llm//rtp_llm/cpp/embedding_engine:embedding_engine_arpc_server_impl"
    )


def _torch_view(repo_root):
    # Takes the repo-root label ("@repo//") so check_torch_repos.py still sees
    # a literal "@repo//" reference in every branch.
    return [repo_root + ":torch_api", repo_root + ":torch", repo_root + ":torch_libs"]

def torch_deps():
    # The open-source side only references the public C++ torch view repos derived from
    # deps.json cc_view:
    #   default→@torch_py310_cpu, using_arm→@torch_py310_cpu_aarch64,
    #   cuda12_6→@torch_2.6_py310_cuda, cuda12_9_x86→@torch_2.8_py310_cuda.
    # cuda12_9_arm has no public C++ torch view (only the internal overlay's private aarch64
    # torch flavor provides one); that branch explicitly points at the analysis-time-failing
    # absent stub (@arch_config//:torch_cuda12_9_arm_absent), no fallback to the default cpu
    # view (avoids private-name leakage + version cross-contamination).
    # cuda13 x86/ARM are the same story for torch 2.11+cu130 (published to the internal index
    # only, so this tree declares no cc_view for either profile).
    deps = select({
        "@rtp_llm//:using_rocm": _torch_view("@torch_rocm//"),
        "@rtp_llm//:using_arm": _torch_view("@torch_py310_cpu_aarch64//"),
        "@rtp_llm//:cuda_pre_12_9": _torch_view("@torch_2.6_py310_cuda//"),
        "@rtp_llm//:using_cuda12_9_x86": _torch_view("@torch_2.8_py310_cuda//"),
        "@rtp_llm//:using_cuda12_arm": [
            "@arch_config//:torch_cuda12_9_arm_absent",
        ],
        "@rtp_llm//:using_cuda13_x86": [
            "@arch_config//:torch_cuda13_absent",
        ],
        "@rtp_llm//:using_cuda13_arm": [
            "@arch_config//:torch_cuda13_arm_absent",
        ],
        "//conditions:default": _torch_view("@torch_py310_cpu//"),
    })
    return deps

def flashinfer_deps():
    # The CUDA-13 C++ flashinfer view is its own source build (separate BUILD file + cuda13
    # patches) and this tree declares no such repo, so that branch points at the
    # analysis-time-failing absent stub rather than the CUDA-12 build -- linking the cu12
    # flashinfer into a cuda13 target would be a silent ABI swap.
    native.alias(
        name = "flashinfer",
        actual = select({
            "@rtp_llm//:using_cuda13_x86": "@arch_config//:flashinfer_cuda13_absent",
            "@rtp_llm//:using_cuda13_arm": "@arch_config//:flashinfer_cuda13_arm_absent",
            "//conditions:default": "@flashinfer_cpp//:flashinfer",
        })
    )

def flashmla_deps():
    native.alias(
        name = "flashmla",
        actual = "@flashmla//:flashmla"
    )

def deep_ep_py_deps():
    # Keep the standalone DeepEP wrapper on the same profile/absence map as the other
    # Python requirements.  An empty target here used to hide a real CUDA12 dependency
    # and silently made the CUDA13/default cases indistinguishable from optional code.
    requirement_libs(["deep-ep"], _REQUIREMENT_BRANCHES, "cpu", requirement_cpu)
    native.alias(
        name = "deep_ep_py",
        actual = ":deep-ep",
        tags = ["manual"],
    )

def cuda_register():
    native.alias(
        name = "cuda_register",
        actual = select({
            "//conditions:default": "@rtp_llm//rtp_llm/models_py/bindings/cuda/ops:gpu_register",
        }),
        visibility = ["//visibility:public"],
    )

def triton_deps(names):
    return select({
        "//conditions:default": [],
    })

def internal_deps():
    return []

def jit_deps():
    return []

def torch_nvshmem_deps():
    return select({
        "@rtp_llm//:using_cuda13_x86": ["@arch_config//:torch_nvshmem_cuda13"],
        "//conditions:default": [],
    })

# Injection point for PPU wheel Requires-Dist (open-source stub): an open-source clone has no
# ppu.json overlay/ppu lock, and the using_ppu branch is unreachable; return empty srcs + empty
# flag, and if actually selected, gen_wheel_requires fails clearly due to the missing --lock
# (a missing private overlay fails at the entry point instead of passing off another profile's
# facts as ppu).
# Private profile wheel metadata is only available through the internal overlay. The public
# build keeps these inputs empty; selecting a private profile still fails through its explicit
# dependency absence stubs instead of borrowing another profile's lock.
def whl_reqs_profile_srcs(_profile):
    return []

def whl_reqs_profile_flag(_profile):
    return ""

def select_py_bindings():
    return select({
        "@rtp_llm//:using_cuda12": [
            "@rtp_llm//rtp_llm/models_py/bindings/cuda:cuda_bindings_register"
        ],
        "@rtp_llm//:using_rocm": [
            "@rtp_llm//rtp_llm/models_py/bindings/rocm:rocm_bindings_register"
        ],
        "//conditions:default": [
            "@rtp_llm//rtp_llm/models_py/bindings:dummy_register",
        ],
    })

def no_block_copy_link_deps():
    """Deps for the cc_library that defines execNoBlockCopy / warmupNoBlockCopy (per device)."""
    return select({
        "@rtp_llm//:using_cuda12": [
            "@rtp_llm//rtp_llm/models_py/bindings/cuda:no_block_copy",
        ],
        "@rtp_llm//:using_rocm": [
            "@rtp_llm//rtp_llm/models_py/bindings:no_block_copy_default",
        ],
        "//conditions:default": [
            "@rtp_llm//rtp_llm/models_py/bindings:no_block_copy_default",
        ],
    })

def vipserver_deps():
    """VIPServer client dependency: the open-source view has no such intranet repo; returns empty.

    The consumption point (remote_connector:subscriber) only pulls it under the RECO_INTERNAL
    guard; the repo name is hidden inside the seam instead of written in an open-source BUILD,
    so the open-source tree has zero private names.
    """
    return []
