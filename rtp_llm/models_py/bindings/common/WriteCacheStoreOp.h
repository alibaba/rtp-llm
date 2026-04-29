#pragma once

#include "rtp_llm/models_py/bindings/core/OpData.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"

namespace rtp_llm {

// group_id and kv_block_stride_bytes default to overrides. group_id=-1 falls
// back to looking up via kv_cache_layer_to_group_host[layer_id]; otherwise the
// caller-supplied gid is used for both the offset table (when 3D) and the
// "_g{gid}" key suffix. kv_block_stride_bytes=0 falls back to
// PyCacheStoreInputs::kv_block_stride_bytes; otherwise the caller-supplied
// per-pool stride is used. Both are needed by DSV4 where one layer touches
// multiple pools, each with its own layout.
void WriteCacheStoreOp(const torch::Tensor&                         input_lengths,
                       const torch::Tensor&                         prefix_lengths,
                       const torch::Tensor&                         kv_cache_block_id_host,
                       std::optional<torch_ext::PyCacheStoreInputs> cache_store_member,
                       std::optional<torch_ext::LayerKVCache>       kv_cache,
                       int32_t                                      group_id              = -1,
                       int64_t                                      kv_block_stride_bytes = 0);

}  // namespace rtp_llm
