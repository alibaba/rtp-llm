#pragma once

#include <memory>
#include "rtp_llm/cpp/devices/cuda_impl/CudaFlashInfer.h"
#include "rtp_llm/cpp/config/GptInitParameter.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"
#include "rtp_llm/models_py/bindings/cuda/FMHACudaBase.h"

namespace rtp_llm {

class FlashInferPrefillOp: public FMHACudaBase {
public:
    FlashInferPrefillOp(const GptInitParameter& gpt_init_parameter);

    bool support(torch_ext::PyAttentionInputs attn_inputs);

    ParamsBasePtr prepare(torch_ext::PyAttentionInputs attn_inputs);

    torch::Tensor
    forward(const torch::Tensor& q, std::optional<torch_ext::KVCache> kv_cache, const FlashInferAttnParamsPtr& params);
};

class FlashInferDecodeOp: public FMHACudaBase {
public:
    FlashInferDecodeOp(const GptInitParameter& gpt_init_parameter);
    bool          support(torch_ext::PyAttentionInputs attn_inputs);
    ParamsBasePtr prepare(torch_ext::PyAttentionInputs attn_inputs);

    torch::Tensor
    forward(const torch::Tensor& q, std::optional<torch_ext::KVCache> kv_cache, const FlashInferAttnParamsPtr& params);
};

void registerFlashInferOp(const py::module& m);

}  // namespace rtp_llm
