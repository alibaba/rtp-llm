#pragma once

#include <array>
#include <string>

#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/config/ModelConfig.h"

namespace rtp_llm {

// DSV4 expected spec tags and their metadata. Single source of truth for
// all DSV4 cache config validation. Co-located with DSV4CacheConfigHelper
// because both are transitional and will be removed once the declarative
// spec pipeline fully replaces this helper.
constexpr size_t kDsv4PoolNum = 7;

struct ExpectedDSV4Spec {
    const char*       tag;
    KVCacheRegionName region_name;
    bool              is_paged;
};

constexpr std::array<ExpectedDSV4Spec, kDsv4PoolNum> kExpectedDsv4Specs = {
    ExpectedDSV4Spec{"csa_kv",        KVCacheRegionName::CSA_KV,        true},
    ExpectedDSV4Spec{"hca_kv",        KVCacheRegionName::HCA_KV,        true},
    ExpectedDSV4Spec{"indexer_kv",    KVCacheRegionName::INDEXER_KV,    true},
    ExpectedDSV4Spec{"indexer_state", KVCacheRegionName::INDEXER_STATE, false},
    ExpectedDSV4Spec{"csa_state",     KVCacheRegionName::CSA_STATE,     false},
    ExpectedDSV4Spec{"hca_state",     KVCacheRegionName::HCA_STATE,     false},
    ExpectedDSV4Spec{"swa_kv",        KVCacheRegionName::SWA_KV,        false},
};

inline bool isDsv4SpecTag(const std::string& tag) {
    for (const auto& expected : kExpectedDsv4Specs) {
        if (tag == expected.tag) {
            return true;
        }
    }
    return false;
}

// Whether the model uses a DSV4 typed hybrid-pool layout (i.e. declares any
// DSV4 KV cache spec by tag). Single source of truth — callers must use this
// instead of re-implementing tag checks in .cc anonymous namespaces. Performs
// the transitional fail-fast: when layer_compress_ratios is set the model must
// also declare DSV4 kv_cache_specs (the ratios-only fallback is disabled).
// Co-located here so every DSV4 transitional predicate is removed together
// once the declarative spec pipeline fully replaces this helper.
inline bool hasTypedHybridPoolLayout(const ModelConfig& model_config) {
    bool has_dsv4 = false;
    for (const auto& layer_specs : model_config.kv_cache_specs) {
        for (const auto& spec : layer_specs.second) {
            if (spec != nullptr && isDsv4SpecTag(spec->tag)) {
                has_dsv4 = true;
                break;
            }
        }
        if (has_dsv4) {
            break;
        }
    }
    RTP_LLM_CHECK_WITH_INFO(model_config.attn_config.layer_compress_ratios.empty() || has_dsv4,
                            "DSV4 cache config requires model_config.kv_cache_specs; "
                            "layer_compress_ratios fallback is disabled");
    return has_dsv4;
}

class DSV4CacheConfigHelper {
public:
    static void applyConfig(CacheConfig&             config,
                            const ModelConfig&       model_config,
                            const ParallelismConfig& parallelism_config,
                            const KVCacheConfig&     kv_cache_config,
                            int                      gen_num_per_cycle);
};

}  // namespace rtp_llm
