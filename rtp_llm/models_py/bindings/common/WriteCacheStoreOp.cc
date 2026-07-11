#include "rtp_llm/models_py/bindings/common/WriteCacheStoreOp.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/models_py/bindings/core/CacheStoreAsyncWriter.h"

#include <algorithm>

namespace rtp_llm {

using namespace torch_ext;

namespace {

size_t inferPhysicalBlockStrideBytes(const torch::Tensor& tensor,
                                     size_t               physical_tokens_per_block,
                                     size_t               layer_tokens_per_block,
                                     size_t               fallback) {
    if (!tensor.defined() || tensor.numel() <= 0 || tensor.dim() <= 0 || physical_tokens_per_block == 0
        || layer_tokens_per_block == 0) {
        return fallback;
    }

    const auto leading_blocks = static_cast<size_t>(tensor.size(0));
    if (leading_blocks == 0) {
        return fallback;
    }

    const size_t kernel_blocks_per_physical_block =
        std::max<size_t>(1, physical_tokens_per_block / layer_tokens_per_block);
    if (leading_blocks % kernel_blocks_per_physical_block != 0) {
        return fallback;
    }

    const size_t physical_blocks = leading_blocks / kernel_blocks_per_physical_block;
    if (physical_blocks == 0 || static_cast<size_t>(tensor.nbytes()) % physical_blocks != 0) {
        return fallback;
    }

    return static_cast<size_t>(tensor.nbytes()) / physical_blocks;
}

}  // namespace

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
    auto captured_input_lengths           = input_lengths;
    auto captured_prefix_lengths          = prefix_lengths;
    auto captured_kv_cache_block_id_host  = kv_cache_block_id_host;
    auto captured_cache_store             = cache_store_inputs;
    auto captured_kv_cache                = kv_cache.value();
    auto captured_kv_cache_layer_to_group = captured_cache_store.kv_cache_layer_to_group;
    if (captured_kv_cache.layer_id >= 0 && captured_kv_cache.group_id >= 0) {
        if (captured_kv_cache_layer_to_group.defined()
            && captured_kv_cache_layer_to_group.numel() > captured_kv_cache.layer_id) {
            captured_kv_cache_layer_to_group = captured_kv_cache_layer_to_group.clone();
        } else {
            captured_kv_cache_layer_to_group = torch::full({captured_kv_cache.layer_id + 1},
                                                           captured_kv_cache.group_id,
                                                           torch::TensorOptions(torch::kInt32).device(torch::kCPU));
        }
        captured_kv_cache_layer_to_group.data_ptr<int32_t>()[captured_kv_cache.layer_id] = captured_kv_cache.group_id;
    }

    // Create event in main thread to avoid cudaEventRecord contention on background threads.
    auto event = runtimeCreateEvent();

    auto run = [captured_input_lengths,
                captured_prefix_lengths,
                captured_kv_cache_block_id_host,
                captured_cache_store,
                captured_kv_cache,
                captured_kv_cache_layer_to_group,
                event = std::move(event)]() mutable {
        const size_t kv_block_stride_bytes =
            inferPhysicalBlockStrideBytes(captured_kv_cache.kv_cache_base,
                                          captured_cache_store.tokens_per_block,
                                          static_cast<size_t>(captured_kv_cache.seq_size_per_block),
                                          captured_cache_store.kv_block_stride_bytes);
        const size_t kv_scale_stride_bytes =
            inferPhysicalBlockStrideBytes(captured_kv_cache.kv_scale_base,
                                          captured_cache_store.tokens_per_block,
                                          static_cast<size_t>(captured_kv_cache.seq_size_per_block),
                                          captured_cache_store.kv_scale_stride_bytes);
        CacheStoreInputs inputs{captured_input_lengths,
                                captured_prefix_lengths,
                                captured_kv_cache_block_id_host,
                                captured_kv_cache_layer_to_group,
                                captured_cache_store.kv_cache_group_types,
                                captured_cache_store.context_batch_size,
                                captured_cache_store.decoder_batch_size,
                                captured_cache_store.request_id,
                                captured_cache_store.request_pd_separation,
                                captured_cache_store.cache_keys,
                                captured_cache_store.tokens_per_block,
                                kv_block_stride_bytes,
                                kv_scale_stride_bytes,
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
