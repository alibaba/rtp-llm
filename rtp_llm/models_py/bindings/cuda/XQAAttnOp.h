#pragma once

#ifdef USING_CUDA12

#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/model_utils/AttentionConfig.h"
#include "rtp_llm/cpp/devices/cuda_impl/CudaXqa.h"
#include "rtp_llm/cpp/devices/cuda_impl/CudaDevice.h"
#include "rtp_llm/cpp/devices/DeviceFactory.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"

namespace rtp_llm {

class XQAAttnOp {
public:
    XQAAttnOp(const AttentionConfigs& attn_configs);
    bool support(torch_ext::PyAttentionInputs attn_inputs);

    ParamsBasePtr prepare(torch_ext::PyAttentionInputs attn_inputs);

    torch::Tensor
    forward(const torch::Tensor& input, std::optional<torch_ext::KVCache> kv_cache, const XQAParamsPtr& params);

protected:
    AttentionConfigs attn_configs_;
};

void registerXQAAttnOp(const py::module& m);

}  // namespace rtp_llm
#endif
