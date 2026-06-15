load("//bazel:arch_select.bzl", "torch_deps")

cc_library(
    name = "flashinfer",
    hdrs = [
        "flashinfer.h",
    ],
    visibility = ["//visibility:public"],
)

genrule(
    name = "flashinfer_cuda13_attention_stubs_src",
    outs = ["flashinfer_cuda13_attention_stubs.cc"],
    cmd = """cat > "$@" <<'EOF'
#include "3rdparty/flashinfer/flashinfer.h"

#include <stdexcept>

namespace {
[[noreturn]] void unsupported() {
    throw std::runtime_error("FlashInfer attention C++ kernels are not linked in the CUDA 13 slim wheel");
}
}  // namespace

at::Tensor BatchDecodeWithPagedKVCachePlan(
    at::Tensor, at::Tensor, at::Tensor, at::Tensor, int64_t, int64_t, int64_t, int64_t, bool, int64_t, double, int64_t, int64_t, at::Tensor, at::Tensor, int64_t) {
    unsupported();
}

void BatchDecodeWithPagedKVCacheRun(at::Tensor,
                                    at::Tensor,
                                    at::Tensor,
                                    at::Tensor,
                                    at::Tensor,
                                    at::Tensor,
                                    at::Tensor,
                                    at::Tensor,
                                    at::Tensor,
                                    at::Tensor,
                                    std::optional<at::Tensor>,
                                    int64_t,
                                    int64_t,
                                    std::optional<at::Tensor>,
                                    double,
                                    double,
                                    double,
                                    double,
                                    int64_t) {
    unsupported();
}

at::Tensor BatchPrefillWithKVCachePlan(
    at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t, bool, int64_t, int64_t, bool, int64_t) {
    unsupported();
}

void BatchPrefillWithPagedKVCacheRun(at::Tensor,
                                     at::Tensor,
                                     at::Tensor,
                                     at::Tensor,
                                     at::Tensor,
                                     at::Tensor,
                                     at::Tensor,
                                     at::Tensor,
                                     at::Tensor,
                                     at::Tensor,
                                     at::Tensor,
                                     std::optional<at::Tensor>,
                                     int64_t,
                                     int64_t,
                                     int64_t,
                                     std::optional<at::Tensor>,
                                     std::optional<at::Tensor>,
                                     std::optional<at::Tensor>,
                                     double,
                                     double,
                                     double,
                                     double,
                                     int64_t) {
    unsupported();
}

at::Tensor BatchMLAPagedAttentionPlan(
    at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, int64_t, int64_t, bool, int64_t) {
    unsupported();
}

void BatchMLAPagedAttentionRun(at::Tensor,
                               at::Tensor,
                               at::Tensor,
                               at::Tensor,
                               at::Tensor,
                               at::Tensor,
                               at::Tensor,
                               at::Tensor,
                               std::optional<at::Tensor>,
                               int64_t,
                               int64_t,
                               int64_t,
                               double,
                               int64_t) {
    unsupported();
}
EOF
""",
)

cc_library(
    name = "flashinfer_cuda13_stubs",
    srcs = [
        ":flashinfer_cuda13_attention_stubs_src",
    ],
    deps = [
        ":flashinfer",
        "@flashinfer_cpp_cu13//:flashinfer_minimal",
    ] + torch_deps(),
    visibility = ["//visibility:public"],
)
