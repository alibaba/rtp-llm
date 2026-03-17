#include "rtp_llm/models_py/bindings/common/WriteCacheStoreOp.h"
#include "rtp_llm/cpp/devices/DeviceFactory.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"

namespace rtp_llm {

using namespace torch_ext;

namespace {

// Build CacheStoreInputs + KvCacheInfo from captured torch::Tensors and call
// writeCacheStore. Executed on the async writer's background thread. All
// torch::Tensors are captured by value (refcount++) so underlying memory stays
// alive until this function returns.
void runWriteCacheStore(DeviceBase*        device,
                        torch::Tensor      input_lengths,
                        torch::Tensor      prefix_lengths,
                        torch::Tensor      kv_cache_block_id_host,
                        PyCacheStoreInputs cache_store_inputs,
                        KVCache            kv_cache) {

    auto layer_to_group_buf = (cache_store_inputs.kv_cache_layer_to_group.defined()
                               && cache_store_inputs.kv_cache_layer_to_group.numel() > 0) ?
                                  torchTensor2Buffer(cache_store_inputs.kv_cache_layer_to_group) :
                                  nullptr;
    auto group_types_buf =
        (cache_store_inputs.kv_cache_group_types.defined() && cache_store_inputs.kv_cache_group_types.numel() > 0) ?
            torchTensor2Buffer(cache_store_inputs.kv_cache_group_types) :
            nullptr;

    CacheStoreInputs inputs{torchTensor2Buffer(input_lengths),
                            torchTensor2Buffer(prefix_lengths),
                            torchTensor2Buffer(kv_cache_block_id_host),
                            layer_to_group_buf,
                            group_types_buf,
                            cache_store_inputs.context_batch_size,
                            cache_store_inputs.decoder_batch_size,
                            torchTensor2Buffer(cache_store_inputs.request_id),
                            torchTensor2Buffer(cache_store_inputs.request_pd_separation),
                            cache_store_inputs.cache_keys,
                            cache_store_inputs.tokens_per_block,
                            cache_store_inputs.kv_block_stride_bytes,
                            cache_store_inputs.kv_scale_stride_bytes,
                            cache_store_inputs.pd_separation,
                            cache_store_inputs.model_id,
                            cache_store_inputs.decode_entrance,
                            cache_store_inputs.warmup,
                            kv_cache.layer_id};

    KvCacheInfo kv_cache_info;
    kv_cache_info.kv_cache_buffer = torchTensor2Buffer(kv_cache.kv_cache_base);
    kv_cache_info.kv_scale_buffer = (kv_cache.kv_scale_base.defined() && kv_cache.kv_scale_base.numel() > 0) ?
                                        torchTensor2Buffer(kv_cache.kv_scale_base) :
                                        nullptr;

    device->writeCacheStore(inputs, kv_cache_info, cache_store_inputs.mla_kvcache);
}

}  // anonymous namespace

void WriteCacheStoreOp(const torch::Tensor&                         input_lengths,
                       const torch::Tensor&                         prefix_lengths,
                       const torch::Tensor&                         kv_cache_block_id_host,
                       std::optional<torch_ext::PyCacheStoreInputs> cache_store_member,
                       std::optional<torch_ext::KVCache>            kv_cache) {
    if (!kv_cache.has_value() || !cache_store_member.has_value()) {
        return;
    }

    auto device = DeviceFactory::getDefaultDevice();

    // Capture all torch::Tensors by value so the underlying memory stays alive
    // in the background thread. torch::Tensor copy is a cheap refcount bump.
    auto captured_input_lengths          = input_lengths;
    auto captured_prefix_lengths         = prefix_lengths;
    auto captured_kv_cache_block_id_host = kv_cache_block_id_host;
    auto captured_cache_store            = cache_store_member.value();
    auto captured_kv_cache               = kv_cache.value();

    device->submitAsyncCacheStoreTask([device,
                                       captured_input_lengths,
                                       captured_prefix_lengths,
                                       captured_kv_cache_block_id_host,
                                       captured_cache_store,
                                       captured_kv_cache]() {
        runWriteCacheStore(device,
                           captured_input_lengths,
                           captured_prefix_lengths,
                           captured_kv_cache_block_id_host,
                           captured_cache_store,
                           captured_kv_cache);
    });
}

}  // namespace rtp_llm
