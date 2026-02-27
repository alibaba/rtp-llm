#pragma once

#include "rtp_llm/cpp/devices/OpData.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"

namespace rtp_llm {

void WriteCacheStoreOp(const torch::Tensor&                         input_lengths,
                       const torch::Tensor&                         prefix_lengths,
                       const torch::Tensor&                         kv_cache_block_id_host,
                       std::optional<torch_ext::PyCacheStoreInputs> cache_store_member,
                       std::optional<torch_ext::LayerKVCache>       kv_cache);

}  // namespace rtp_llm
