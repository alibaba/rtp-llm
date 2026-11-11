#include <pybind11/pybind11.h>

#include "rtp_llm/cpp/cache/CacheConfigCreator.h"

namespace py = pybind11;

namespace rtp_llm::test {

void validateBasicConfig(const ModelConfig& model_config) {
    ParallelismConfig parallelism_config;
    parallelism_config.tp_size = 1;
    (void)CacheConfigCreator::createWarmupConfig(model_config, parallelism_config, /*gen_num_per_cycle=*/0);
}

py::dict describeBasicConfig(const ModelConfig& model_config, int cp_size, bool prefill) {
    RTP_LLM_CHECK_WITH_INFO(cp_size > 0, "cache test CP size must be positive");
    ParallelismConfig parallelism_config;
    parallelism_config.tp_size                            = prefill ? cp_size : 1;
    parallelism_config.world_size                         = parallelism_config.tp_size;
    parallelism_config.local_world_size                   = parallelism_config.tp_size;
    parallelism_config.role_type                          = prefill ? RoleType::PREFILL : RoleType::DECODE;
    parallelism_config.prefill_cp_config.prefill_cp_size  = cp_size;
    parallelism_config.prefill_cp_config.kv_cache_sharded = cp_size > 1;
    parallelism_config.prefill_cp_config.method           = cp_size == 1 ? CPRotateMethod::DISABLED :
                                                            prefill      ? CPRotateMethod::ALL_GATHER :
                                                                           CPRotateMethod::PREFILL_CP;
    const auto config =
        CacheConfigCreator::createWarmupConfig(model_config, parallelism_config, /*gen_num_per_cycle=*/0);
    py::dict groups;
    for (const auto& group : config.topology().groups()) {
        py::dict result;
        result["seq_size_per_block"]         = group.seqSizePerBlock();
        result["kernel_seq_size_per_block"]  = group.kernelSeqSizePerBlock();
        result["kernel_blocks_per_kv_block"] = group.kernelBlocksPerKvBlock();
        result["block_stride_bytes"]         = group.kvBlockStrideBytes();
        result["block_payload_bytes"]        = group.spec->block_payload_bytes();
        if (const auto opaque = std::dynamic_pointer_cast<const OpaqueKVCacheSpec>(group.spec)) {
            result["entries_per_block"]        = opaque->entriesPerBlock();
            result["entries_per_kernel_block"] = opaque->entriesPerBlock() / group.kernelBlocksPerKvBlock();
        }
        py::list layers;
        for (const auto layer_id : config.layerIdsForGroup(group.tag)) {
            layers.append(layer_id);
        }
        result["layer_ids"]        = std::move(layers);
        groups[py::str(group.tag)] = std::move(result);
    }
    return groups;
}

PYBIND11_MODULE(libcache_config_creator_py_test, m) {
    m.def("validate_basic_config", &validateBasicConfig);
    m.def("describe_basic_config",
          &describeBasicConfig,
          py::arg("model_config"),
          py::arg("cp_size") = 1,
          py::arg("prefill") = true);
}

}  // namespace rtp_llm::test
