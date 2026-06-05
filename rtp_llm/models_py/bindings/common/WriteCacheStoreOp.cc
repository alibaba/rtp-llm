#include "rtp_llm/models_py/bindings/common/WriteCacheStoreOp.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/models_py/bindings/core/CacheStoreAsyncWriter.h"

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
        auto resolve_store_stride = [&](const torch::Tensor& tensor, size_t fallback_stride, const char* name) {
            size_t stride_bytes = fallback_stride;
            if (!tensor.defined() || tensor.dim() != 2) {
                return stride_bytes;
            }
            const size_t row_stride_bytes = static_cast<size_t>(tensor.stride(0)) * tensor.element_size();
            const int    layer_tokens_per_block = captured_kv_cache.seq_size_per_block;
            RTP_LLM_CHECK_WITH_INFO(layer_tokens_per_block > 0,
                                    "LayerKVCache.seq_size_per_block must be positive for cache-store %s write",
                                    name);
            const size_t layer_tokens = static_cast<size_t>(layer_tokens_per_block);
            const size_t store_tokens = captured_cache_store.tokens_per_block;
            RTP_LLM_CHECK_WITH_INFO(store_tokens > 0, "cache-store tokens_per_block must be positive");

            if (store_tokens >= layer_tokens) {
                RTP_LLM_CHECK_WITH_INFO(store_tokens % layer_tokens == 0,
                                        "cache-store tokens_per_block=%zu must be divisible by layer tokens_per_block=%zu "
                                        "for cache-store %s write",
                                        store_tokens,
                                        layer_tokens,
                                        name);
                return row_stride_bytes * (store_tokens / layer_tokens);
            }

            // CP compact STATE/SWA rows can cover cp_size canonical cache-key
            // blocks. CacheStore keys still use the canonical store_tokens
            // granularity; the compact row is stored as one opaque slice block.
            RTP_LLM_CHECK_WITH_INFO(layer_tokens % store_tokens == 0,
                                    "LayerKVCache.seq_size_per_block=%zu must be divisible by cache-store "
                                    "tokens_per_block=%zu for compact cache-store %s write",
                                    layer_tokens,
                                    store_tokens,
                                    name);
            return row_stride_bytes;
        };

        size_t kv_block_stride_bytes = captured_cache_store.kv_block_stride_bytes;
        if (captured_kv_cache.kv_cache_base.defined() && captured_kv_cache.kv_cache_base.dim() == 2) {
            kv_block_stride_bytes =
                resolve_store_stride(captured_kv_cache.kv_cache_base, kv_block_stride_bytes, "kv");
        }
        size_t kv_scale_stride_bytes = captured_cache_store.kv_scale_stride_bytes;
        if (captured_kv_cache.kv_scale_base.defined() && captured_kv_cache.kv_scale_base.dim() == 2) {
            kv_scale_stride_bytes =
                resolve_store_stride(captured_kv_cache.kv_scale_base, kv_scale_stride_bytes, "scale");
        }

        const size_t layer_tokens_per_block = static_cast<size_t>(captured_kv_cache.seq_size_per_block);
        RTP_LLM_CHECK_WITH_INFO(layer_tokens_per_block > 0,
                                "LayerKVCache.seq_size_per_block must be positive for cache-store write");
        const size_t store_tokens_per_block = captured_cache_store.tokens_per_block;
        RTP_LLM_CHECK_WITH_INFO(store_tokens_per_block > 0, "cache-store tokens_per_block must be positive");

        CacheStoreInputs inputs{captured_input_lengths,
                                captured_prefix_lengths,
                                captured_kv_cache_block_id_host,
                                captured_cache_store.kv_cache_layer_to_group,
                                captured_cache_store.kv_cache_layer_region_to_group,
                                captured_cache_store.kv_cache_group_types,
                                captured_cache_store.context_batch_size,
                                captured_cache_store.decoder_batch_size,
                                captured_cache_store.request_id,
                                captured_cache_store.request_pd_separation,
                                captured_cache_store.cache_keys,
                                store_tokens_per_block,
                                kv_block_stride_bytes,
                                kv_scale_stride_bytes,
                                captured_cache_store.pd_separation,
                                captured_cache_store.model_id,
                                captured_cache_store.decode_entrance,
                                captured_cache_store.warmup,
                                captured_cache_store.use_opaque_kv_cache_store,
                                captured_kv_cache.layer_id,
                                captured_kv_cache.region_name,
                                captured_cache_store.cp_rank,
                                captured_cache_store.cp_size,
                                std::move(event)};

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
