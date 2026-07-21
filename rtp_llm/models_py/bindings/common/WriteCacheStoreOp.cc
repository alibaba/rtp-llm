#include "rtp_llm/models_py/bindings/common/WriteCacheStoreOp.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/models_py/bindings/core/CacheStoreAsyncWriter.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

using namespace torch_ext;

void WriteCacheStoreOp(const torch::Tensor&                         input_lengths,
                       const torch::Tensor&                         prefix_lengths,
                       const torch::Tensor&                         kv_cache_block_id_host,
                       std::optional<torch_ext::PyCacheStoreInputs> cache_store_member,
                       std::optional<torch_ext::LayerKVCache>       kv_cache) {
    if (!kv_cache.has_value() || !cache_store_member.has_value()) {
        return;
    }

    const PyCacheStoreInputs& cache_store_inputs = cache_store_member.value();

    // Capture all torch::Tensors by value so the underlying memory stays alive
    // in the background thread. torch::Tensor copy is a cheap refcount bump.
    auto captured_input_lengths          = input_lengths;
    auto captured_prefix_lengths         = prefix_lengths;
    auto captured_kv_cache_block_id_host = kv_cache_block_id_host;
    auto captured_cache_store            = cache_store_inputs;
    auto captured_kv_cache               = kv_cache.value();
    RTP_LLM_CHECK_WITH_INFO(!captured_kv_cache.tag.empty(),
                            "cache-store write requires a cache tag for layer=%d",
                            captured_kv_cache.layer_id);

    // Create event in main thread to avoid cudaEventRecord contention on background threads.
    auto event = runtimeCreateEvent();

    auto run = [captured_input_lengths,
                captured_prefix_lengths,
                captured_kv_cache_block_id_host,
                captured_cache_store,
                captured_kv_cache,
                event = std::move(event)]() mutable {
        const auto   tokens_it              = captured_cache_store.tokens_per_block_by_tag.find(captured_kv_cache.tag);
        const size_t store_tokens_per_block = tokens_it != captured_cache_store.tokens_per_block_by_tag.end() ?
                                                  tokens_it->second :
                                                  captured_cache_store.tokens_per_block;
        const size_t layer_tokens_per_block = static_cast<size_t>(captured_kv_cache.seq_size_per_block);
        RTP_LLM_CHECK_WITH_INFO(layer_tokens_per_block > 0,
                                "LayerKVCache.seq_size_per_block must be positive for tag=%s",
                                captured_kv_cache.tag.c_str());
        RTP_LLM_CHECK_WITH_INFO(store_tokens_per_block > 0,
                                "cache-store tokens_per_block must be positive for tag=%s",
                                captured_kv_cache.tag.c_str());

        CacheStoreInputs inputs;
        inputs.input_lengths_host                             = captured_input_lengths;
        inputs.prefix_lengths_host                            = captured_prefix_lengths;
        inputs.host_kv_cache_offset                           = captured_kv_cache_block_id_host;
        inputs.kv_cache_group_policies                        = captured_cache_store.kv_cache_group_policies;
        inputs.tokens_per_block_by_tag                        = captured_cache_store.tokens_per_block_by_tag;
        inputs.kv_block_stride_bytes_by_tag                   = captured_cache_store.kv_block_stride_bytes_by_tag;
        inputs.kv_scale_stride_bytes_by_tag                   = captured_cache_store.kv_scale_stride_bytes_by_tag;
        inputs.kv_block_transfer_bytes_by_tag                 = captured_cache_store.kv_block_transfer_bytes_by_tag;
        inputs.kv_scale_transfer_bytes_by_tag                 = captured_cache_store.kv_scale_transfer_bytes_by_tag;
        inputs.context_batch_size                             = captured_cache_store.context_batch_size;
        inputs.decoder_batch_size                             = captured_cache_store.decoder_batch_size;
        inputs.request_id                                     = captured_cache_store.request_id;
        inputs.request_pd_separation                          = captured_cache_store.request_pd_separation;
        inputs.cache_keys                                     = captured_cache_store.cache_keys;
        inputs.tokens_per_block                               = captured_cache_store.tokens_per_block;
        inputs.kv_block_stride_bytes                          = captured_cache_store.kv_block_stride_bytes;
        inputs.kv_scale_stride_bytes                          = captured_cache_store.kv_scale_stride_bytes;
        inputs.pd_separation                                  = captured_cache_store.pd_separation;
        inputs.model_id                                       = captured_cache_store.model_id;
        inputs.decode_entrance                                = captured_cache_store.decode_entrance;
        inputs.warmup                                         = captured_cache_store.warmup;
        inputs.use_opaque_kv_cache_store                      = captured_cache_store.use_opaque_kv_cache_store;
        inputs.layer_id                                       = captured_kv_cache.layer_id;
        inputs.tag                                            = captured_kv_cache.tag;
        inputs.cp_rank                                        = captured_cache_store.cp_rank;
        inputs.cp_size                                        = captured_cache_store.cp_size;
        inputs.pre_created_event                              = std::move(event);
        inputs.tokens_per_block_by_tag[captured_kv_cache.tag] = layer_tokens_per_block;

        KvCacheInfo kv_cache_info;
        kv_cache_info.kv_cache_buffer = captured_kv_cache.kv_cache_base;
        kv_cache_info.kv_scale_buffer =
            (captured_kv_cache.kv_scale_base.defined() && captured_kv_cache.kv_scale_base.numel() > 0) ?
                captured_kv_cache.kv_scale_base :
                torch::Tensor();
        execWriteCacheStore(inputs, kv_cache_info, captured_cache_store.mla_kvcache, captured_cache_store.cache_store);
    };

    auto* async_writer = cache_store_inputs.cache_store_async_writer;
    if (async_writer) {
        async_writer->submit(std::move(run));
    } else {
        run();
    }
}

}  // namespace rtp_llm
