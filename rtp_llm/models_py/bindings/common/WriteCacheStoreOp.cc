#include "rtp_llm/models_py/bindings/common/WriteCacheStoreOp.h"
#include "rtp_llm/cpp/core/ExecOps.h"
#include "rtp_llm/cpp/core/CacheStoreAsyncWriter.h"

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

    // Create event in main thread to avoid cudaEventRecord contention on background threads.
    auto event = runtimeCreateEvent();

    auto run = [captured_input_lengths,
                captured_prefix_lengths,
                captured_kv_cache_block_id_host,
                captured_cache_store,
                captured_kv_cache,
                event = std::move(event)]() mutable {
        CacheStoreInputs inputs;
        inputs.input_lengths_host           = captured_input_lengths;
        inputs.prefix_lengths_host          = captured_prefix_lengths;
        inputs.host_kv_cache_offset         = captured_kv_cache_block_id_host;
        inputs.kv_cache_layer_to_group_host = captured_cache_store.kv_cache_layer_to_group;
        inputs.kv_cache_group_types_host    = captured_cache_store.kv_cache_group_types;
        inputs.context_batch_size           = captured_cache_store.context_batch_size;
        inputs.decoder_batch_size           = captured_cache_store.decoder_batch_size;
        inputs.request_id                   = captured_cache_store.request_id;
        inputs.request_pd_separation        = captured_cache_store.request_pd_separation;
        inputs.cache_keys                   = captured_cache_store.cache_keys;
        inputs.tokens_per_block             = captured_cache_store.tokens_per_block;
        inputs.kv_block_stride_bytes        = captured_cache_store.kv_block_stride_bytes;
        inputs.kv_scale_stride_bytes        = captured_cache_store.kv_scale_stride_bytes;
        inputs.pd_separation                = captured_cache_store.pd_separation;
        inputs.model_id                     = captured_cache_store.model_id;
        inputs.decode_entrance              = captured_cache_store.decode_entrance;
        inputs.warmup                       = captured_cache_store.warmup;
        inputs.layer_id                     = captured_kv_cache.layer_id;
        inputs.pre_created_event            = std::move(event);
        inputs.cp_slot_mapper               = captured_cache_store.cp_slot_mapper;

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
