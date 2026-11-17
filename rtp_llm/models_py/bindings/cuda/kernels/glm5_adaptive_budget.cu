#include "rtp_llm/models_py/bindings/cuda/kernels/glm5_adaptive_budget.h"

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <cmath>
#include <tuple>

namespace rtp_llm {
namespace {

__global__ void orderedSurvival(const float* confidence, const int64_t* slots, const int32_t* caps,
                                double* scores, bool* valid_rows, int64_t batch, int64_t slot_count,
                                int64_t steps, int64_t row_stride) {
    const int64_t request = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (request >= batch) return;
    const int64_t slot = slots[request];
    const int64_t cap = caps[request];
    bool valid = slot >= 0 && slot < slot_count && cap >= 0 && cap <= steps;
    double product = 1.0;
    for (int64_t step = 0; step < steps; ++step) {
        const double probability = slot >= 0 && slot < slot_count ?
            static_cast<double>(confidence[slot * row_stride + step]) : 0.0;
        const bool finite_probability = isfinite(probability) && probability >= 0.0 && probability <= 1.0;
        valid = valid && finite_probability;
        // NumPy cumprod(float64) multiplies sequentially within each row.
        product = __dmul_rn(product, finite_probability ? probability : 0.0);
        scores[request * steps + step] = step < cap ? product : -1.0;
    }
    valid_rows[request] = valid;
}

__global__ void orderedEconomicBudget(const double* sorted_scores, const bool* valid_rows,
                                      const double* costs, int64_t* output, int64_t batch,
                                      int64_t limit, int64_t sampling_requests) {
    for (int64_t request = 0; request < batch; ++request) {
        if (!valid_rows[request]) { output[0] = -1; return; }
    }
    for (int64_t budget = 0; budget <= limit; ++budget) {
        if (!isfinite(costs[budget]) || costs[budget] <= 0.0) { output[0] = -2; return; }
    }
    double prefix = 0.0;
    double best = __ddiv_rn(static_cast<double>(sampling_requests), costs[0]);
    int64_t selected = 0;
    for (int64_t budget = 1; budget <= limit; ++budget) {
        const double score = sorted_scores[budget - 1];
        if (score < 0.0) { output[0] = -1; return; }
        // Preserve NumPy's sequential cumsum, then its separate addition of
        // the anchor count. A parallel scan changes close-budget decisions.
        prefix = __dadd_rn(prefix, score);
        const double expected = __dadd_rn(static_cast<double>(sampling_requests), prefix);
        const double utility = __ddiv_rn(expected, costs[budget]);
        if (utility > best) { best = utility; selected = budget; }
    }
    output[0] = selected;
}

}  // namespace

torch::Tensor glm5SelectDraftTokenBudgetDevice(const torch::Tensor& confidence,
                                               const torch::Tensor& request_slots,
                                               const torch::Tensor& capacities,
                                               const torch::Tensor& total_costs,
                                               int64_t limit,
                                               int64_t sampling_requests) {
    TORCH_CHECK(confidence.is_cuda() && confidence.scalar_type() == torch::kFloat32
                    && confidence.dim() == 2 && confidence.stride(1) == 1,
                "adaptive confidence must be CUDA float32 rows");
    const c10::cuda::CUDAGuard guard(confidence.device());
    const int64_t batch = request_slots.numel();
    const int64_t steps = confidence.size(1);
    TORCH_CHECK(request_slots.device() == confidence.device() && request_slots.scalar_type() == torch::kInt64
                    && request_slots.dim() == 1 && request_slots.is_contiguous(), "invalid adaptive request slots");
    TORCH_CHECK(capacities.device() == confidence.device() && capacities.scalar_type() == torch::kInt32
                    && capacities.dim() == 1 && capacities.is_contiguous() && capacities.numel() == batch,
                "invalid adaptive capacities");
    TORCH_CHECK(total_costs.device() == confidence.device() && total_costs.scalar_type() == torch::kFloat64
                    && total_costs.dim() == 1 && total_costs.is_contiguous() && total_costs.numel() == limit + 1,
                "invalid adaptive total costs");
    TORCH_CHECK(batch > 0 && steps > 0 && limit >= 0 && limit <= batch * steps
                    && sampling_requests >= 0 && sampling_requests <= batch, "invalid adaptive budget dimensions");
    const auto stream = c10::cuda::getCurrentCUDAStream();
    auto scores = torch::empty({batch * steps}, confidence.options().dtype(torch::kFloat64));
    auto valid = torch::empty({batch}, confidence.options().dtype(torch::kBool));
    auto output = torch::empty({1}, confidence.options().dtype(torch::kInt64));
    for (const auto& source : {confidence, request_slots, capacities, total_costs}) {
        c10::cuda::CUDACachingAllocator::recordStream(source.storage().data_ptr(), stream);
    }
    orderedSurvival<<<static_cast<unsigned>((batch + 127) / 128), 128, 0, stream.stream()>>>(
        confidence.data_ptr<float>(), request_slots.data_ptr<int64_t>(), capacities.data_ptr<int32_t>(),
        scores.data_ptr<double>(), valid.data_ptr<bool>(), batch, confidence.size(0), steps, confidence.stride(0));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    auto sorted = limit > 0 ? std::get<0>(scores.sort(/*dim=*/0, /*descending=*/true)) : scores;
    orderedEconomicBudget<<<1, 1, 0, stream.stream()>>>(sorted.data_ptr<double>(), valid.data_ptr<bool>(),
        total_costs.data_ptr<double>(), output.data_ptr<int64_t>(), batch, limit, sampling_requests);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

}  // namespace rtp_llm
