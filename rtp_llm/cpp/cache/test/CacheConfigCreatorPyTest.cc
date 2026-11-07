#include <pybind11/pybind11.h>

#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/cache/CoordinatorCacheManager.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"

namespace py = pybind11;

namespace rtp_llm::test {

void validateBasicConfig(const ModelConfig& model_config) {
    ParallelismConfig parallelism_config;
    parallelism_config.tp_size = 1;
    (void)CacheConfigCreator::createWarmupConfig(model_config, parallelism_config, /*gen_num_per_cycle=*/0);
}

torch_ext::KVCache makeOffloadTestCache(bool fp8, int max_selected_rows, int initial_hot_rows, bool dynamic_hot_cache) {
    RTP_LLM_CHECK_WITH_INFO(dynamic_hot_cache || initial_hot_rows < 0 || initial_hot_rows == max_selected_rows,
                            "initial_hot_rows requires dynamic hot cache unless the full capacity is requested");
    AttentionConfigs attention;
    attention.kv_lora_rank  = 512;
    attention.rope_head_dim = 64;
    KVCacheSpecDesc desc;
    desc.tag        = "default";
    desc.cache_type = MultiHeadLatentAttention;
    desc.dtype      = fp8 ? TYPE_FP8_E4M3 : TYPE_BF16;
    SpecBuildContext context;
    context.seq_size_per_block      = 256;
    context.kernel_tokens_per_block = 128;
    context.attn_config             = &attention;
    CacheConfig config;
    config.layer_num          = 2;
    config.seq_size_per_block = 256;
    config.use_mla            = true;
    config.dtype              = desc.dtype;
    config.fromGroupedSpecs({SpecBuilder::build(desc, context)}, {{0, 1}}, {CacheGroupType::FULL}, {desc.tag});
    config.finalizeBlockNums(6, RuntimeConfig{});
    config.tiered_cache_configs.emplace(desc.tag, TieredCacheConfig{6, 2, max_selected_rows, 0, dynamic_hot_cache});
    CoordinatorCacheManager manager(config);
    RTP_LLM_CHECK(manager.init());
    for (const auto& pool : manager.groupBlockPools()) {
        if (pool->isOffload() && dynamic_hot_cache) {
            pool->allLayerOffloadStorage().front()->hot_cache->resize(
                initial_hot_rows < 0 ? max_selected_rows : initial_hot_rows);
        }
    }
    return torch_ext::KVCache(manager.allLayerCacheBase());
}

PYBIND11_MODULE(libcache_config_creator_py_test, m) {
    m.def("validate_basic_config", &validateBasicConfig);
    m.def("make_offload_test_cache", &makeOffloadTestCache, py::arg("fp8"), py::arg("max_selected_rows") = 256,
          py::arg("initial_hot_rows") = -1, py::arg("dynamic_hot_cache") = false);
}

}  // namespace rtp_llm::test
