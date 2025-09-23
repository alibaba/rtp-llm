#include "rtp_llm/cpp/models/Eagle3Model.h"

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

EmbeddingPostOutput Eagle3Model::embeddingPost(const rtp_llm::BufferPtr& hidden_states, const GptModelInputs& inputs) {
    DevicePerfWrapper wrapper(device_, "eagle3_embedding_post");
    auto              last_hidden_states = inputs.last_hidden_states;

    if (last_hidden_states == nullptr) {
        RTP_LLM_LOG_DEBUG("last hidden states is null in eagle3 model");
        // For eagle3, attention q_proj is (2 * hidden_size -> hidden_size), thereforce we need to duplicate hidden
        // states here
        BufferPtr duplicate_hidden = device_->clone(
            {*torchTensor2Buffer(
                 torch::cat({Buffer2torchTensor(hidden_states, false), Buffer2torchTensor(hidden_states, false)}, 1)),
             AllocationType::DEVICE});
        return {duplicate_hidden, hidden_states};
    }

    if ((weights_.layers[0].eagle3_input_norm == nullptr) || (weights_.layers[0].eagle3_fc_norm == nullptr)
        || (weights_.layers[0].eagle3_fc_proj == nullptr)) {
        RTP_LLM_FAIL("eagle3_input_norm | eagle3_fc_norm | eagle3_fc_proj doesn't exist");
        return {hidden_states, nullptr};
    }

    // printBufferData(*inputs.last_hidden_states, "last_hidden_states");

    auto proj_last_hidden_states =
        device_->gemm({*inputs.last_hidden_states, *(weights_.layers[0].eagle3_fc_proj->kernel)});
    printBufferData(*proj_last_hidden_states, "proj_last_hidden_states");

    auto proj_norm = device_->layernorm(LayernormParams(proj_last_hidden_states,
                                                        nullptr,
                                                        *weights_.layers[0].eagle3_fc_norm,
                                                        std::nullopt,
                                                        std::nullopt,
                                                        std::nullopt,
                                                        0.f,
                                                        1e-6,
                                                        false,
                                                        false,
                                                        NormType::rmsnorm));

    // printBufferData(*proj_norm.output, "proj_norm");
    // printBufferData(*weights_.layers[0].eagle3_fc_norm->gamma, "weights_.layers[0].eagle3_fc_norm");

    auto input_norm = device_->layernorm(LayernormParams(hidden_states,
                                                         nullptr,
                                                         *weights_.layers[0].eagle3_input_norm,
                                                         std::nullopt,
                                                         std::nullopt,
                                                         std::nullopt,
                                                         0.f,
                                                         1e-6,
                                                         false,
                                                         false,
                                                         NormType::rmsnorm));
    // printBufferData(*input_norm.output, "input_norm");
    // printBufferData(*weights_.layers[0].eagle3_input_norm->gamma, "weights_.layers[0].eagle3_input_norm");

    auto proj_norm_tensor  = rtp_llm::Buffer2torchTensor(*proj_norm.output, false);
    auto input_norm_tensor = rtp_llm::Buffer2torchTensor(*input_norm.output, false);

    torch::Tensor cat_tensor      = torch::cat({input_norm_tensor, proj_norm_tensor}, -1);
    BufferPtr final_hidden_states = device_->clone({*rtp_llm::torchTensor2Buffer(cat_tensor), AllocationType::DEVICE});

    // printBufferData(*final_hidden_states, "final_hidden_states");
    return {final_hidden_states, proj_last_hidden_states};
}

}  // namespace rtp_llm
