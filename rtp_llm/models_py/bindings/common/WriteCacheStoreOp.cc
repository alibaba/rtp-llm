#include "rtp_llm/models_py/bindings/common/WriteCacheStoreOp.h"
#include "rtp_llm/cpp/core/ExecOps.h"

namespace rtp_llm {

using namespace torch_ext;

void WriteCacheStoreOp(const torch::Tensor&                         input_lengths,
                       const torch::Tensor&                         prefix_lengths,
                       const torch::Tensor&                         kv_cache_block_id_host,
                       std::optional<torch_ext::PyCacheStoreInputs> cache_store_member,
                       std::optional<torch_ext::LayerKVCache>       kv_cache) {
    if (kv_cache.has_value() && cache_store_member.has_value()) {
        const PyCacheStoreInputs& cache_store_inputs = cache_store_member.value();

        CacheStoreInputs inputs{input_lengths,
                                prefix_lengths,
                                kv_cache_block_id_host,
                                cache_store_inputs.kv_cache_layer_to_group,
                                cache_store_inputs.kv_cache_group_types,
                                cache_store_inputs.context_batch_size,
                                cache_store_inputs.decoder_batch_size,
                                cache_store_inputs.request_id,
                                cache_store_inputs.request_pd_separation,
                                cache_store_inputs.cache_keys,
                                cache_store_inputs.tokens_per_block,
                                cache_store_inputs.kv_block_stride_bytes,
                                cache_store_inputs.kv_scale_stride_bytes,
                                cache_store_inputs.pd_separation,
                                cache_store_inputs.model_id,
                                cache_store_inputs.decode_entrance,
                                cache_store_inputs.warmup,
                                kv_cache.value().layer_id};

        KvCacheInfo kv_cache_info;
        // kv_cache_buffer uses kv block base address (compatible with existing cache store writer which writes "k_").
        kv_cache_info.kv_cache_buffer = kv_cache.value().kv_cache_base;
        kv_cache_info.kv_scale_buffer =
            (kv_cache.value().kv_scale_base.defined() && kv_cache.value().kv_scale_base.numel() > 0) ?
                kv_cache.value().kv_scale_base :
                torch::Tensor();
        execWriteCacheStore(inputs, kv_cache_info, cache_store_inputs.mla_kvcache, cache_store_inputs.cache_store);
    }
}

}  // namespace rtp_llm
