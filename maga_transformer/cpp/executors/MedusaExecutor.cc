#include "maga_transformer/cpp/executors/MedusaExecutor.h"
#include "maga_transformer/cpp/batch_stream_processor/SpeculativeBatchStreamProcessor.h"
#include "maga_transformer/cpp/common/status_util.h"
#include "maga_transformer/cpp/executors/NormalExecutor.h"
#include "maga_transformer/cpp/models/GptModel.h"
#include "maga_transformer/cpp/models/Sampler.h"

using namespace std;

namespace rtp_llm {

MedusaModelExecutor::MedusaModelExecutor(
    const MagaInitParams&                                                   params,
    const std::vector<std::unordered_map<std::string, ft::ConstBufferPtr>>& layer_weights,
    const std::unordered_map<std::string, ft::ConstBufferPtr>&              weights):
    NormalExecutor(params, layer_weights, weights) {
    unique_ptr<GptModelInitParams> model_params;
    model_.reset(new GptModel(*model_params));
    SamplerInitParams sampler_params;
    sampler_.reset(new MedusaTreeSampler(sampler_params));
    batch_stream_processor_.reset(new SpeculativeBatchStreamProcessor(*params.gpt_init_parameter));
}

}  // namespace rtp_llm
