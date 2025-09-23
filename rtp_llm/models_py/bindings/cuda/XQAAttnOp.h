#pragma once

#ifdef USING_CUDA12

#include "rtp_llm/cpp/config/GptInitParameter.h"
#include "rtp_llm/cpp/devices/cuda_impl/CudaXqa.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"
#include "rtp_llm/models_py/bindings/cuda/FMHACudaBase.h"

namespace rtp_llm {

class XQAAttnOp: public FMHACudaBase {
public:
    XQAAttnOp(const GptInitParameter& gpt_init_parameter);
    bool support(torch_ext::PyAttentionInputs attn_inputs);

    ParamsBasePtr prepare(torch_ext::PyAttentionInputs attn_inputs);

    torch::Tensor
    forward(const torch::Tensor& input, std::optional<torch_ext::KVCache> kv_cache, const XQAParamsPtr& params);
};

void registerXQAAttnOp(const py::module& m);

}  // namespace rtp_llm
#endif
