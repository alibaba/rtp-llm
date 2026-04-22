#include "rtp_llm/models_py/bindings/rocm/FakeBalanceExpertOp.h"
#include "rtp_llm/models_py/bindings/common/Torch_ext.h"
#include "rtp_llm/cpp/kernels/moe_kernels.h"

namespace rtp_llm {

void fake_balance_expert_op(
    at::Tensor& expert_ids,
    at::Tensor& expert_scales,
    int64_t     dp_rank,
    int64_t     dp_size,
    int64_t     ep_size,
    int64_t     expert_num,
    int64_t     hip_stream) {

    CHECK_INPUT(expert_ids);
    CHECK_INPUT(expert_scales);

    TORCH_CHECK(expert_ids.dim() == 2, "expert_ids must be a 2D tensor");
    TORCH_CHECK(expert_scales.dim() == 2, "expert_scales must be a 2D tensor");
    TORCH_CHECK(expert_ids.sizes() == expert_scales.sizes(),
                "expert_ids and expert_scales must have the same shape");

    int size = static_cast<int>(expert_ids.numel());
    hipStream_t stream = reinterpret_cast<hipStream_t>(hip_stream);

    if (expert_ids.dtype() == torch::kInt64) {
        fake_balance_expert(expert_ids.data_ptr<int64_t>(),
                            expert_scales.data_ptr<float>(),
                            static_cast<int>(dp_rank),
                            static_cast<int>(dp_size),
                            static_cast<int>(ep_size),
                            static_cast<int>(expert_num),
                            size,
                            stream);
    } else if (expert_ids.dtype() == torch::kInt32) {
        fake_balance_expert(expert_ids.data_ptr<int32_t>(),
                            expert_scales.data_ptr<float>(),
                            static_cast<int>(dp_rank),
                            static_cast<int>(dp_size),
                            static_cast<int>(ep_size),
                            static_cast<int>(expert_num),
                            size,
                            stream);
    } else {
        throw std::runtime_error("Unsupported dtype for fake_balance_expert_op: " +
                                 std::string(expert_ids.dtype().name()));
    }
}

}  // namespace rtp_llm
