#include "rtp_llm/models_py/bindings/core/WeightPreprocess.h"

namespace rtp_llm {

torch::Tensor
preprocessGemmWeightByKey(const std::string& /*key*/, torch::Tensor weight, bool /*user_arm_gemm_use_kai*/) {
    return weight;
}

torch::Tensor preprocessWeightScale(torch::Tensor weight, torch::Tensor /*scale*/) {
    return weight;
}

}  // namespace rtp_llm
