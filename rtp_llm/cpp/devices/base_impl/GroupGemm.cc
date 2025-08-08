#include "rtp_llm/cpp/devices/DeviceBase.h"

namespace rtp_llm {

GroupedGemmOutput DeviceBase::groupedGemm(const GroupedGemmParams& params) {
    RTP_LLM_LOG_DEBUG("use default group gemm");
    params.check();
    size_t                 num = params.A.size();
    std::vector<BufferPtr> output(num);
    for (int i = 0; i < num; i++) {
        BufferPtr c = nullptr;
        if (params.C != std::nullopt) {
            output[i] = gemm({*params.A[i],
                              *params.B[i],
                              std::nullopt,
                              params.C.value()[i],
                              DataType::TYPE_INVALID,
                              params.C.value()[i]->type(),
                              TransposeOperation::NONE,
                              TransposeOperation::NONE,
                              ActivationType::Identity,
                              1.0f,
                              1.0f});
        } else {
            output[i] = gemm({*params.A[i], *params.B[i]});
        }
    }
    return GroupedGemmOutput({std::move(output)});
}

}  // namespace rtp_llm
