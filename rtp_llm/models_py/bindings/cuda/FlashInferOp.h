#pragma once

#include <memory>
#include "rtp_llm/cpp/devices/cuda_impl/CudaFlashInfer.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"
#include "rtp_llm/cpp/devices/cuda_impl/CudaDevice.h"
#include "rtp_llm/cpp/devices/DeviceFactory.h"
#include "rtp_llm/cpp/model_utils/AttentionConfig.h"

namespace rtp_llm {

class FlashInferPrefillOp {
public:
    FlashInferPrefillOp(const AttentionConfigs& attn_configs);

    bool support(torch_ext::PyAttentionInputs attn_inputs);

    ParamsBasePtr prepare(torch_ext::PyAttentionInputs attn_inputs);

    torch::Tensor forward(const torch::Tensor&                   q,
                          std::optional<torch_ext::LayerKVCache> kv_cache,
                          const FlashInferAttnParamsPtr&         params);

protected:
    AttentionConfigs attn_configs_;
    CudaDevice*      device_;
};

class FlashInferDecodeOp {
public:
    FlashInferDecodeOp(const AttentionConfigs& attn_configs);
    bool          support(torch_ext::PyAttentionInputs attn_inputs);
    ParamsBasePtr prepare(torch_ext::PyAttentionInputs attn_inputs);

    torch::Tensor forward(const torch::Tensor&                   q,
                          std::optional<torch_ext::LayerKVCache> kv_cache,
                          const FlashInferAttnParamsPtr&         params);

protected:
    AttentionConfigs attn_configs_;
    CudaDevice*      device_;
};

void registerFlashInferOp(const py::module& m);

}  // namespace rtp_llm
