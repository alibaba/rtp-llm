#include "rtp_llm/models_py/bindings/core/ExecOps.h"

namespace rtp_llm {

void invokeMtpSpecSamplerTokenMatrix(const torch::Tensor& history,
                                     const torch::Tensor& draft,
                                     const torch::Tensor& base_lengths,
                                     torch::Tensor&       output,
                                     int64_t              gamma) {
    const auto batch = history.size(0);
    const auto rows  = output.size(1);
    const auto width = history.size(1);
    output.copy_(history.unsqueeze(1).expand({batch, rows, width}));
    auto offsets = torch::arange(gamma, base_lengths.options()).view({1, 1, gamma});
    output.scatter_(2,
                    (base_lengths.view({batch, 1, 1}) + offsets).expand({batch, rows, gamma}),
                    draft.unsqueeze(1).expand({batch, rows, gamma}));
}

}  // namespace rtp_llm
