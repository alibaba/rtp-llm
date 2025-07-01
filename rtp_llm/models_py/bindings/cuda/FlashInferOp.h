#pragma once

#include <memory>
// #include "rtp_llm/cpp/devices/cuda_impl/CudaFlashInfer.h"
#include "rtp_llm/cpp/th_op/GptInitParameter.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"

namespace rtp_llm {

class FlashInferOp {
public:
    FlashInferOp(const GptInitParameter& gpt_init_parameter);
    void forward(torch::Tensor input, torch::Tensor output,
                 torch::Tensor k_cache, torch::Tensor v_cache,
                 torch_ext::PyAttentionInputs attn_params);
private:
    GptInitParameter configs;
    RopeConfig rope_config;
    // const AttentionLayerWeights weights;
    // std::unique_ptr<FlashInferAttnParams> params;
};
void register_attn_params(pybind11::module& m);

void registerFlashInferOp(const py::module& m);
}
