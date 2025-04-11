#include "maga_transformer/cpp/models/MTPModel.h"

#include "src/fastertransformer/core/Buffer.h"
#include "src/fastertransformer/core/Types.h"
#include "src/fastertransformer/devices/OpData.h"
#include "src/fastertransformer/core/BufferHelper.h"
#include "src/fastertransformer/core/torch_utils/BufferTorchUtils.h"
#include "src/fastertransformer/devices/utils/DevicePerfWrapper.h"
#include "src/fastertransformer/devices/utils/DebugUtils.h"
#include "src/fastertransformer/models/W.h"
#include "maga_transformer/cpp/utils/AssertUtils.h"
#include "maga_transformer/cpp/utils/StringUtil.h"

#include <memory>


using namespace std;
using namespace fastertransformer;
using namespace rtp_llm;

namespace rtp_llm {

ft::BufferPtr MTPModel::embeddingPost(const ft::BufferPtr& hidden_states, const GptModelInputs& inputs) {
    DevicePerfWrapper wrapper(device_, "mtp_embeddingPost");
    auto last_hidden_states = inputs.last_hidden_states;

    if (last_hidden_states == nullptr) {
        FT_LOG_WARNING("last hidden states is null in mtp model");
        return hidden_states;
    }


    if ((weights_.layers[0].enorm == nullptr) ||
        (weights_.layers[0].hnorm == nullptr) ||
        (weights_.layers[0].eh_proj == nullptr))
    {
        FT_LOG_WARNING("mtp model weights is null");
        return hidden_states;
    }

    auto e_norm = device_->layernorm(LayernormParams(hidden_states, nullptr, *weights_.layers[0].enorm, std::nullopt,
        std::nullopt, std::nullopt, 0.f, 1e-6, false, false, NormType::rmsnorm));

    auto h_norm = device_->layernorm(LayernormParams(last_hidden_states, nullptr, *weights_.layers[0].hnorm, std::nullopt,
        std::nullopt, std::nullopt, 0.f, 1e-6, false, false, NormType::rmsnorm));

    auto e_norm_tensor = ft::Buffer2torchTensor(*e_norm.output, false);
    auto h_norm_tensor = ft::Buffer2torchTensor(*h_norm.output, false);

    torch::Tensor cat_tensor;
    if (reverse_e_h_norm_) {
        cat_tensor = torch::cat({e_norm_tensor, h_norm_tensor}, -1);
    } else {
        cat_tensor = torch::cat({h_norm_tensor, e_norm_tensor}, -1);
    }
    auto cat_buffer = ft::torchTensor2Buffer(cat_tensor);

    auto final_hidden_states = device_->gemm({*cat_buffer, *(weights_.layers[0].eh_proj->kernel)});

    return final_hidden_states;

}

} // namespace rtp_llm
