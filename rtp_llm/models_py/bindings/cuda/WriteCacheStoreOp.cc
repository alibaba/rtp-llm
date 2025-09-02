#include "rtp_llm/models_py/bindings/cuda/WriteCacheStoreOp.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"

namespace rtp_llm {

WriteCacheStoreOp::WriteCacheStoreOp(const GptInitParameter& gpt_init_parameter): FMHACudaBase(gpt_init_parameter) {}

void WriteCacheStoreOp::forward(torch_ext::PyAttentionInputs attn_inputs, std::optional<torch_ext::KVCache> kv_cache) {
    if (kv_cache.has_value() && attn_inputs.cache_store_inputs.has_value()) {
        const PyCacheStoreInputs& cache_store_inputs = attn_inputs.cache_store_inputs.value();
        CacheStoreInputs          inputs{torchTensor2Buffer(attn_inputs.input_lengths),
                                torchTensor2Buffer(attn_inputs.prefix_lengths),
                                torchTensor2Buffer(attn_inputs.kv_cache_block_id_host),
                                cache_store_inputs.context_batch_size,
                                cache_store_inputs.decoder_batch_size,
                                // (size_t)(attn_inputs.input_lengths.size(0) - attn_inputs.sequence_lengths.size(0)),
                                // (size_t)attn_inputs.sequence_lengths.size(0),
                                torchTensor2Buffer(cache_store_inputs.request_id),
                                torchTensor2Buffer(cache_store_inputs.request_pd_separation),
                                cache_store_inputs.cache_keys,
                                cache_store_inputs.tokens_per_block,
                                cache_store_inputs.k_block_size,
                                cache_store_inputs.v_block_size,
                                cache_store_inputs.scale_block_size,
                                cache_store_inputs.pd_separation,
                                cache_store_inputs.model_id,
                                cache_store_inputs.decode_entrance,
                                cache_store_inputs.warmup,
                                kv_cache.value().layer_id};
        KvCacheInfo               kv_cache_info;
        kv_cache_info.k_cache_buffer = torchTensor2Buffer(kv_cache.value().k_cache_base);
        kv_cache_info.v_cache_buffer = torchTensor2Buffer(kv_cache.value().v_cache_base);
        device_->writeCacheStore(inputs, kv_cache_info, cache_store_inputs.mla_kvcache);
    }
}

void registerWriteCacheStoreOp(const py::module& m) {
    pybind11::class_<WriteCacheStoreOp>(m, "WriteCacheStoreOp")
        .def(pybind11::init<GptInitParameter>(), py::arg("gpt_init_parameter"))
        .def("forward", &WriteCacheStoreOp::forward, py::arg("attn_inputs"), py::arg("kv_cache"));
}

}  // namespace rtp_llm
