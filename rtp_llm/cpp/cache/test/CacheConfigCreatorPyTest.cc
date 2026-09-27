#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <map>
#include <stdexcept>
#include <vector>

#include "rtp_llm/cpp/cache/CacheConfigCreator.h"

namespace py = pybind11;

namespace rtp_llm::test {

void validateBasicConfig(const ModelConfig& model_config) {
    ParallelismConfig parallelism_config;
    parallelism_config.tp_size = 1;
    (void)CacheConfigCreator::createBasicConfig(
        model_config, parallelism_config, /*is_mtp=*/false, /*gen_num_per_cycle=*/0);
}

// Probe the production SpecBuilder with caller-supplied descriptors and return
// per-layer/tag wire geometry. Unsharded PREFILL_CP needs no CP scaling. Report
// payload and padded transfer extents separately to catch alignment mismatches.
py::dict dsv4BlockGeometry(const LayerKVCacheSpecDescs& layer_descs,
                           uint32_t                     seq_size_per_block,
                           uint32_t                     kernel_tokens_per_block,
                           uint32_t                     gen_num_per_cycle) {
    ParallelismConfig parallelism_config;
    parallelism_config.prefill_cp_config.kv_cache_sharded = false;
    SpecBuildContext ctx;
    ctx.dtype                   = DataType::TYPE_FP8_E4M3;
    ctx.seq_size_per_block      = seq_size_per_block;
    ctx.kernel_tokens_per_block = kernel_tokens_per_block;
    ctx.parallelism_config      = &parallelism_config;
    ctx.gen_num_per_cycle       = gen_num_per_cycle;

    const auto specs =
        CacheConfigCreator::buildLayerSpecsFromDescs(layer_descs, ctx, static_cast<int64_t>(layer_descs.size()));

    struct TagRec {
        size_t              block_size_bytes    = 0;
        size_t              block_payload_bytes = 0;
        uint32_t            seq_size_per_block  = 0;
        std::vector<size_t> layers;
    };
    std::map<std::string, TagRec> by_tag;
    for (size_t layer = 0; layer < specs.size(); ++layer) {
        for (const auto& spec : specs[layer]) {
            if (!spec) {
                continue;
            }
            auto& rec = by_tag[spec->tag];
            if (rec.layers.empty()) {
                rec.block_size_bytes    = spec->block_size_bytes();
                rec.block_payload_bytes = spec->block_payload_bytes();
                rec.seq_size_per_block  = spec->seq_size_per_block;
            } else if (rec.block_size_bytes != spec->block_size_bytes()
                       || rec.block_payload_bytes != spec->block_payload_bytes()) {
                // same tag must have identical geometry on every layer
                throw std::runtime_error("tag " + spec->tag + " has layer-varying geometry");
            }
            rec.layers.push_back(layer);
        }
    }

    py::dict out;
    for (const auto& [tag, rec] : by_tag) {
        py::dict d;
        d["block_size_bytes"]    = rec.block_size_bytes;
        d["block_payload_bytes"] = rec.block_payload_bytes;
        d["seq_size_per_block"]  = rec.seq_size_per_block;
        d["layers"]              = rec.layers;
        out[tag.c_str()]         = d;
    }
    return out;
}

PYBIND11_MODULE(libcache_config_creator_py_test, m) {
    m.def("validate_basic_config", &validateBasicConfig);
    m.def("dsv4_block_geometry", &dsv4BlockGeometry);
}

}  // namespace rtp_llm::test
