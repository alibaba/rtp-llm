#pragma once

#include <memory>
#include <torch/extension.h>
#include "rtp_llm/models_py/bindings/cuda/ops/CudaFlashInfer.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"
#include "rtp_llm/cpp/model_utils/AttentionConfig.h"

namespace rtp_llm {

class FlashInferPrefillOp {
public:
    FlashInferPrefillOp(const AttentionConfigs& attn_configs,
                        MlaOpsType              mla_ops_type      = MlaOpsType::AUTO,
                        bool                    enable_cuda_graph = false);

    bool support(torch_ext::PyAttentionInputs attn_inputs);

    ParamsBasePtr prepare(torch_ext::PyAttentionInputs attn_inputs);

    torch::Tensor forward(const torch::Tensor&                   q,
                          std::optional<torch_ext::LayerKVCache> kv_cache,
                          const FlashInferAttnParamsPtr&         params);

protected:
    AttentionConfigs attn_configs_;
    MlaOpsType       mla_ops_type_;
    bool             enable_cuda_graph_;
};

class FlashInferDecodeOp {
public:
    FlashInferDecodeOp(const AttentionConfigs& attn_configs,
                       MlaOpsType              mla_ops_type      = MlaOpsType::AUTO,
                       bool                    enable_cuda_graph = false);
    bool          support(torch_ext::PyAttentionInputs attn_inputs);
    ParamsBasePtr prepare(torch_ext::PyAttentionInputs attn_inputs);

    torch::Tensor forward(const torch::Tensor&                   q,
                          std::optional<torch_ext::LayerKVCache> kv_cache,
                          const FlashInferAttnParamsPtr&         params);

protected:
    AttentionConfigs attn_configs_;
    MlaOpsType       mla_ops_type_;
    bool             enable_cuda_graph_;
};

void registerFlashInferOp(const py::module& m);

}  // namespace rtp_llm
