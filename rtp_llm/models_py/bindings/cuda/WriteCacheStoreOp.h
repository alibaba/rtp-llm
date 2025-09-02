#pragma once

#include "rtp_llm/models_py/bindings/cuda/FMHACudaBase.h"
#include "rtp_llm/cpp/devices/OpData.h"

namespace rtp_llm {
class WriteCacheStoreOp: public FMHACudaBase {
public:
    WriteCacheStoreOp(const GptInitParameter& gpt_init_parameter);
    void forward(torch_ext::PyAttentionInputs attn_inputs, std::optional<torch_ext::KVCache> kv_cache);
};

void registerWriteCacheStoreOp(const py::module& m);

}  // namespace rtp_llm
