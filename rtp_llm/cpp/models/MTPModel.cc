#include "rtp_llm/cpp/models/MTPModel.h"

#include "rtp_llm/cpp/core/Buffer.h"
#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/devices/OpData.h"
#include "rtp_llm/cpp/core/BufferHelper.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/devices/utils/DevicePerfWrapper.h"
#include "rtp_llm/cpp/devices/utils/DebugUtils.h"
#include "rtp_llm/cpp/models/models_weight/W.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/StringUtil.h"

#include <memory>

using namespace std;

using namespace rtp_llm;

namespace rtp_llm {

EmbeddingPostOutput MTPModel::embeddingPost(const rtp_llm::BufferPtr& hidden_states, const GptModelInputs& inputs) {
    DevicePerfWrapper wrapper(device_, "mtp_embeddingPost");
    auto              last_hidden_states = inputs.last_hidden_states;

    if (last_hidden_states == nullptr) {
        RTP_LLM_LOG_DEBUG("last hidden states is null in mtp model");
        return {hidden_states, nullptr};
    }

    if ((weights_.layers[0].enorm == nullptr) || (weights_.layers[0].hnorm == nullptr)
        || (weights_.layers[0].eh_proj == nullptr)) {
        RTP_LLM_LOG_DEBUG("mtp model weights is null");
        return {hidden_states, nullptr};
    }

    auto e_norm = device_->layernorm(LayernormParams(hidden_states,
                                                     nullptr,
                                                     *weights_.layers[0].enorm,
                                                     std::nullopt,
                                                     std::nullopt,
                                                     std::nullopt,
                                                     0.f,
                                                     1e-6,
                                                     false,
                                                     false,
                                                     NormType::rmsnorm));

    auto h_norm = device_->layernorm(LayernormParams(last_hidden_states,
                                                     nullptr,
                                                     *weights_.layers[0].hnorm,
                                                     std::nullopt,
                                                     std::nullopt,
                                                     std::nullopt,
                                                     0.f,
                                                     1e-6,
                                                     false,
                                                     false,
                                                     NormType::rmsnorm));

    auto e_norm_tensor = rtp_llm::Buffer2torchTensor(*e_norm.output, false);
    auto h_norm_tensor = rtp_llm::Buffer2torchTensor(*h_norm.output, false);

    torch::Tensor cat_tensor;
    if (reverse_e_h_norm_) {
        cat_tensor = torch::cat({e_norm_tensor, h_norm_tensor}, -1);
    } else {
        cat_tensor = torch::cat({h_norm_tensor, e_norm_tensor}, -1);
    }
    auto cat_buffer = rtp_llm::torchTensor2Buffer(cat_tensor);

    auto final_hidden_states = device_->gemm({*cat_buffer, *(weights_.layers[0].eh_proj->kernel)});

    return {final_hidden_states, nullptr};
}

}  // namespace rtp_llm
