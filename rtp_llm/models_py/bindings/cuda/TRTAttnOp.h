#pragma once

#include <memory>
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/model_utils/AttentionConfig.h"
#include "rtp_llm/cpp/devices/cuda_impl/CudaDevice.h"
#include "rtp_llm/cpp/devices/DeviceFactory.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"

namespace rtp_llm {

class TRTPrefillOp {
public:
    TRTPrefillOp(const ModelConfig& model_config, const ParallelismConfig& parallelism_config);
    bool support(torch_ext::PyAttentionInputs attn_inputs);

    ParamsBasePtr prepare(torch_ext::PyAttentionInputs attn_inputs);

    torch::Tensor
    forward(const torch::Tensor& input, std::optional<torch_ext::KVCache> kv_cache, const TRTAttnPtr& params);

protected:
    AttentionConfigs attn_configs_;
    CudaDevice*      device_;

private:
    std::shared_ptr<cufmha> cufmha_runner_;
    torch::Tensor           static_scale_;
};

void registerTRTAttnOp(const py::module& m);

}  // namespace rtp_llm
