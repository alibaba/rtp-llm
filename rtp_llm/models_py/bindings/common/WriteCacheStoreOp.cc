#include "rtp_llm/models_py/bindings/common/WriteCacheStoreOp.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/models_py/bindings/core/CacheStoreAsyncWriter.h"
#include <algorithm>

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
        auto calcPhysicalBlockStrideBytes = [](const torch::Tensor& block_view,
                                               size_t               tokens_per_block,
                                               int                  layer_seq_size_per_block,
                                               size_t               fallback_stride_bytes) -> size_t {
            if (!block_view.defined() || block_view.dim() == 0) {
                return fallback_stride_bytes;
            }
            const int64_t stride0 = block_view.stride(0);
            if (stride0 <= 0) {
                return fallback_stride_bytes;
            }
            size_t stride_bytes = static_cast<size_t>(stride0) * block_view.element_size();
            if (tokens_per_block > 0 && layer_seq_size_per_block > 0
                && tokens_per_block % static_cast<size_t>(layer_seq_size_per_block) == 0) {
                // LayerKVCache for full-attention may expose kernel-block views while cache-store
                // addresses physical blocks. Recover physical stride by folding kernel blocks.
                const size_t kernel_blocks_per_physical =
                    tokens_per_block / static_cast<size_t>(layer_seq_size_per_block);
                stride_bytes *= std::max<size_t>(kernel_blocks_per_physical, 1);
            }
            return stride_bytes;
        };

        const size_t effective_kv_block_stride_bytes =
            calcPhysicalBlockStrideBytes(captured_kv_cache.kv_cache_base,
                                         captured_cache_store.tokens_per_block,
                                         captured_kv_cache.seq_size_per_block,
                                         captured_cache_store.kv_block_stride_bytes);
        const size_t effective_kv_scale_stride_bytes =
            calcPhysicalBlockStrideBytes(captured_kv_cache.kv_scale_base,
                                         captured_cache_store.tokens_per_block,
                                         captured_kv_cache.seq_size_per_block,
                                         captured_cache_store.kv_scale_stride_bytes);

        CacheStoreInputs inputs{captured_input_lengths,
                                captured_prefix_lengths,
                                captured_kv_cache_block_id_host,
                                captured_cache_store.kv_cache_layer_to_group,
                                captured_cache_store.kv_cache_group_types,
                                captured_cache_store.context_batch_size,
                                captured_cache_store.decoder_batch_size,
                                captured_cache_store.request_id,
                                captured_cache_store.request_pd_separation,
                                captured_cache_store.cache_keys,
                                captured_cache_store.tokens_per_block,
                                effective_kv_block_stride_bytes,
                                effective_kv_scale_stride_bytes,
                                captured_cache_store.pd_separation,
                                captured_cache_store.model_id,
                                captured_cache_store.decode_entrance,
                                captured_cache_store.warmup,
                                captured_kv_cache.layer_id,
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
