#include "rtp_llm/models_py/bindings/rocm/CachedF16Gemm.h"

#include <c10/core/DeviceGuard.h>
#include <hipblaslt/hipblaslt.h>
#include <algorithm>
#include <climits>
#include <cstdint>
#include <map>
#include <memory>
#include <tuple>

namespace rtp_llm {
namespace {
// Match AITER's heuristic workspace budget, but allocate only the selected
// algorithm's requirement. No tensor data/weight pointers are stored in a plan.
constexpr size_t kWorkspaceLimit = 256ULL * 1024 * 1024;
constexpr size_t kPlanLimit      = 256;
constexpr size_t kContextLimit   = 8;
using PlanKey                    = std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int, int, int, bool, bool>;
using ContextKey                 = std::pair<int, uintptr_t>;

void checkBlas(hipblasStatus_t status) {
    TORCH_CHECK(status == HIPBLAS_STATUS_SUCCESS, "cached F16 hipBLASLt error: ", static_cast<int>(status));
}
void checkHip(hipError_t status) {
    TORCH_CHECK(status == hipSuccess, "cached F16 HIP error: ", hipGetErrorString(status));
}

struct Plan {
    hipblasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
    hipblasLtMatmulDesc_t   desc = nullptr;
    hipblasLtMatmulAlgo_t   algo{};
    size_t                  workspace_bytes = 0;
    ~Plan() {
        if (desc)
            hipblasLtMatmulDescDestroy(desc);
        if (a)
            hipblasLtMatrixLayoutDestroy(a);
        if (b)
            hipblasLtMatrixLayoutDestroy(b);
        if (c)
            hipblasLtMatrixLayoutDestroy(c);
    }
};

struct Context {
    int                                      device;
    hipblasLtHandle_t                        handle     = nullptr;
    hipblasLtMatmulPreference_t              preference = nullptr;
    at::Tensor                               workspace;
    std::map<PlanKey, std::unique_ptr<Plan>> plans;
    int64_t                                  hits = 0, misses = 0;
    explicit Context(int device): device(device) {}
    ~Context() {
        // Resources belong to this device, including when a worker thread exits
        // with a different current device. Destructors must not throw.
        int previous = device;
        hipGetDevice(&previous);
        if (previous != device)
            hipSetDevice(device);
        plans.clear();
        workspace = at::Tensor();
        if (preference)
            hipblasLtMatmulPreferenceDestroy(preference);
        if (handle)
            hipblasLtDestroy(handle);
        if (previous != device)
            hipSetDevice(previous);
    }
    void initialize() {
        checkBlas(hipblasLtCreate(&handle));
        checkBlas(hipblasLtMatmulPreferenceCreate(&preference));
        const uint64_t bytes = kWorkspaceLimit;
        checkBlas(hipblasLtMatmulPreferenceSetAttribute(
            preference, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &bytes, sizeof(bytes)));
    }
};

// Thread AND stream isolation prevents sharing mutable bias descriptors or
// scratch memory across concurrent launches. Bounds avoid unbounded retention.
// Nothing from this cache is captured by a graph; capture uses the old path.
thread_local std::map<ContextKey, std::unique_ptr<Context>> contexts;

bool aligned(const at::Tensor& tensor) {
    return reinterpret_cast<uintptr_t>(tensor.data_ptr()) % 16 == 0;
}

// Equivalent to AITER's transpose-result mapping: out^T = weight^T * input^T.
// Prefer column-major for size-one dimensions, as AITER does.
bool layout(const at::Tensor& tensor, hipblasOperation_t& op, int64_t& ld) {
    if (tensor.stride(0) == 1 && tensor.stride(1) >= std::max<int64_t>(1, tensor.size(0))) {
        op = HIPBLAS_OP_T;
        ld = tensor.stride(1);
    } else if (tensor.stride(1) == 1 && tensor.stride(0) >= std::max<int64_t>(1, tensor.size(1))) {
        op = HIPBLAS_OP_N;
        ld = tensor.stride(0);
    } else {
        return false;
    }
    return ld > 0 && ld <= INT_MAX;
}

std::unique_ptr<Plan> makePlan(Context& context, const PlanKey& key, const void* bias) {
    auto [m, n, k, lda, ldb, trans_a, trans_b, dtype, preshuffle, has_bias] = key;
    const auto type                                                         = static_cast<hipDataType>(dtype);
    const auto op_a                                                         = static_cast<hipblasOperation_t>(trans_a);
    const auto op_b                                                         = static_cast<hipblasOperation_t>(trans_b);
    auto       plan                                                         = std::make_unique<Plan>();
    checkBlas(
        hipblasLtMatrixLayoutCreate(&plan->a, type, op_a == HIPBLAS_OP_N ? m : k, op_a == HIPBLAS_OP_N ? k : m, lda));
    checkBlas(
        hipblasLtMatrixLayoutCreate(&plan->b, type, op_b == HIPBLAS_OP_N ? k : n, op_b == HIPBLAS_OP_N ? n : k, ldb));
    checkBlas(hipblasLtMatrixLayoutCreate(&plan->c, type, m, n, m));
    if (preshuffle) {
#if (HIPBLASLT_VERSION_MAJOR >= 1) || (HIPBLASLT_VERSION_MAJOR == 0 && HIPBLASLT_VERSION_MINOR >= 15)
        const hipblasLtOrder_t order = HIPBLASLT_ORDER_COL16_4R8;
        checkBlas(hipblasLtMatrixLayoutSetAttribute(plan->a, HIPBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
#else
        return nullptr;
#endif
    }
    checkBlas(hipblasLtMatmulDescCreate(&plan->desc, HIPBLAS_COMPUTE_32F, HIP_R_32F));
    checkBlas(hipblasLtMatmulDescSetAttribute(plan->desc, HIPBLASLT_MATMUL_DESC_TRANSA, &op_a, sizeof(op_a)));
    checkBlas(hipblasLtMatmulDescSetAttribute(plan->desc, HIPBLASLT_MATMUL_DESC_TRANSB, &op_b, sizeof(op_b)));
    if (has_bias) {
        const hipblasLtEpilogue_t epilogue = HIPBLASLT_EPILOGUE_BIAS;
        checkBlas(
            hipblasLtMatmulDescSetAttribute(plan->desc, HIPBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
        checkBlas(hipblasLtMatmulDescSetAttribute(plan->desc, HIPBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
    }
    hipblasLtMatmulHeuristicResult_t result{};
    int                              count = 0;
    checkBlas(hipblasLtMatmulAlgoGetHeuristic(
        context.handle, plan->desc, plan->a, plan->b, plan->c, plan->c, context.preference, 1, &result, &count));
    if (count == 0 || result.state != HIPBLAS_STATUS_SUCCESS || result.workspaceSize > kWorkspaceLimit) {
        return nullptr;
    }
    plan->algo            = result.algo;
    plan->workspace_bytes = result.workspaceSize;
    return plan;
}
}  // namespace

std::optional<at::Tensor> rocmCachedF16Gemm(const at::Tensor&                input,
                                            const at::Tensor&                weight,
                                            const std::optional<at::Tensor>& bias,
                                            bool                             preshuffle) {
    if (!input.is_cuda() || input.device() != weight.device() || input.dim() != 2 || weight.dim() != 2
        || input.scalar_type() != weight.scalar_type()
        || (input.scalar_type() != at::kHalf && input.scalar_type() != at::kBFloat16) || input.size(1) != weight.size(0)
        || !aligned(input) || !aligned(weight)
        || (at::GradMode::is_enabled()
            && (input.requires_grad() || weight.requires_grad() || (bias && bias->requires_grad())))) {
        return std::nullopt;
    }
    const int64_t m = weight.size(1), n = input.size(0), k = input.size(1);
    if (m <= 0 || n <= 0 || k <= 0 || m > INT_MAX || n > INT_MAX || k > INT_MAX)
        return std::nullopt;
    if (bias
        && (bias->device() != input.device() || bias->scalar_type() != input.scalar_type() || bias->dim() != 1
            || bias->numel() != m || !bias->is_contiguous() || !aligned(*bias))) {
        return std::nullopt;
    }
    hipblasOperation_t op_a, op_b;
    int64_t            lda, ldb;
    if (!layout(weight, op_a, lda) || !layout(input, op_b, ldb))
        return std::nullopt;
    if (preshuffle && (op_a != HIPBLAS_OP_T || lda != k || m % 16 || k % 32))
        return std::nullopt;

    const c10::DeviceGuard guard(input.device());
    const auto             stream = at::hip::getCurrentHIPStream(input.get_device()).stream();
    hipStreamCaptureStatus capture;
    checkHip(hipStreamIsCapturing(stream, &capture));
    if (capture != hipStreamCaptureStatusNone)
        return std::nullopt;

    const ContextKey context_key{input.get_device(), reinterpret_cast<uintptr_t>(stream)};
    auto             found = contexts.find(context_key);
    if (found == contexts.end()) {
        if (contexts.size() >= kContextLimit)
            return std::nullopt;
        auto context = std::make_unique<Context>(input.get_device());
        context->initialize();
        found = contexts.emplace(context_key, std::move(context)).first;
    }
    auto&         context = *found->second;
    const auto    dtype   = input.scalar_type() == at::kHalf ? HIP_R_16F : HIP_R_16BF;
    const PlanKey key{m, n, k, lda, ldb, op_a, op_b, dtype, preshuffle, bias.has_value()};
    auto          hit      = context.plans.find(key);
    const void*   bias_ptr = bias ? bias->data_ptr() : nullptr;
    if (hit == context.plans.end()) {
        if (context.plans.size() >= kPlanLimit)
            return std::nullopt;
        ++context.misses;
        auto plan = makePlan(context, key, bias_ptr);
        if (!plan)
            return std::nullopt;
        if (plan->workspace_bytes
            && (!context.workspace.defined()
                || context.workspace.numel() < static_cast<int64_t>(plan->workspace_bytes))) {
            // Allocation/growth takes place on this context's stream. The Torch
            // allocator safely retires any old storage after preceding work.
            context.workspace =
                at::empty({static_cast<int64_t>(plan->workspace_bytes)}, input.options().dtype(at::kByte));
        }
        hit = context.plans.emplace(key, std::move(plan)).first;
    } else {
        ++context.hits;
    }
    auto& plan = *hit->second;
    if (bias) {
        // Same shape can belong to different layers. Never reuse a stale pointer.
        checkBlas(hipblasLtMatmulDescSetAttribute(
            plan.desc, HIPBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_ptr, sizeof(bias_ptr)));
    }
    auto        output = at::empty({n, m}, input.options());
    const float alpha = 1.0f, beta = 0.0f;
    checkBlas(hipblasLtMatmul(context.handle,
                              plan.desc,
                              &alpha,
                              weight.data_ptr(),
                              plan.a,
                              input.data_ptr(),
                              plan.b,
                              &beta,
                              output.data_ptr(),
                              plan.c,
                              output.data_ptr(),
                              plan.c,
                              &plan.algo,
                              plan.workspace_bytes ? context.workspace.data_ptr() : nullptr,
                              plan.workspace_bytes,
                              stream));
    return output;
}

std::vector<int64_t> rocmF16GemmCacheStats() {
    const auto stream = at::hip::getCurrentHIPStream();
    auto       it     = contexts.find({stream.device_index(), reinterpret_cast<uintptr_t>(stream.stream())});
    if (it == contexts.end())
        return {0, 0, 0, 0};
    const auto& ctx = *it->second;
    return {ctx.hits,
            ctx.misses,
            static_cast<int64_t>(ctx.plans.size()),
            ctx.workspace.defined() ? ctx.workspace.numel() : 0};
}
}  // namespace rtp_llm
