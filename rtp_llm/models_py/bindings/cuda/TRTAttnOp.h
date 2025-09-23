#pragma once

#include <memory>
#include "rtp_llm/cpp/config/GptInitParameter.h"
#include "rtp_llm/cpp/devices/cuda_impl/CudaDevice.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"
#include "rtp_llm/models_py/bindings/cuda/FMHACudaBase.h"

namespace rtp_llm {

class TRTPrefillOp: public FMHACudaBase {
public:
    TRTPrefillOp(const GptInitParameter& gpt_init_parameter);
    bool support(torch_ext::PyAttentionInputs attn_inputs);

    ParamsBasePtr prepare(torch_ext::PyAttentionInputs attn_inputs);

    torch::Tensor
    forward(const torch::Tensor& input, std::optional<torch_ext::KVCache> kv_cache, const TRTAttnPtr& params);

private:
    std::shared_ptr<cufmha> cufmha_runner_;
    torch::Tensor           static_scale_;
};

void registerTRTAttnOp(const py::module& m);

}  // namespace rtp_llm
