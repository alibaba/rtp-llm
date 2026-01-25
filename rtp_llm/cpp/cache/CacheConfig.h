#pragma once

#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/Types.h"
#include "rtp_llm/cpp/cache/KVCacheSpec.h"
#include "rtp_llm/cpp/cache/MemoryLayout.h"
#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/utils/StringUtil.h"

namespace rtp_llm {

struct CacheConfig {
    std::vector<KVCacheSpecPtr>   cache_specs;
    std::vector<std::vector<int>> global_layer_ids;  // including mtp module layers

    rtp_llm::DataType dtype;
    uint32_t          layer_num;      // TODO(xinfei.sxf) the number of main model layers
    uint32_t          layer_all_num;  // TODO(xinfei.sxf) the number of all layers including mtp modules

    uint32_t block_num;

    // ---- Per-block sizes (all layers) ----
    size_t kv_block_size_bytes = 0;
    size_t kv_scale_size_bytes = 0;
    size_t block_size_bytes    = 0;

    // ---- Per-block strides (one layer) ----
    size_t kv_block_stride_bytes = 0;
    size_t kv_scale_stride_bytes = 0;

    size_t seq_size_per_block = 1;

    // for adpation to MLA
    bool use_mla = false;

    std::vector<std::shared_ptr<CacheConfig>> mtp_sub_configs;

    CacheConfig() {}

    std::string debugString(size_t indent = 0) const {
        const std::string indent_str = std::string(indent, ' ');
        const std::string indent1    = indent_str + "  ";
        const std::string indent2    = indent_str + "    ";

        std::ostringstream os;
        os << indent_str << "CacheConfig{\n";
        os << indent1 << "dtype=" << static_cast<int>(dtype) << "\n";
        os << indent1 << "layer_num=" << layer_num << "\n";
        os << indent1 << "layer_all_num=" << layer_all_num << "\n";
        os << indent1 << "block_num=" << block_num << "\n";
        os << indent1 << "seq_size_per_block=" << seq_size_per_block << "\n";
        os << indent1 << "use_mla=" << (use_mla ? "true" : "false") << "\n";

        os << indent1 << "kv_block_size_bytes=" << kv_block_size_bytes << "\n";
        os << indent1 << "kv_scale_size_bytes=" << kv_scale_size_bytes << "\n";
        os << indent1 << "block_size_bytes=" << block_size_bytes << "\n";

        os << indent1 << "kv_block_stride_bytes=" << kv_block_stride_bytes << "\n";
        os << indent1 << "kv_scale_stride_bytes=" << kv_scale_stride_bytes << "\n";

        os << indent1 << "cache_specs.size=" << cache_specs.size() << "\n";
        for (size_t i = 0; i < cache_specs.size(); ++i) {
            const auto& spec = cache_specs[i];
            if (!spec) {
                os << indent1 << "cache_specs[" << i << "]=null\n";
                continue;
            }

            os << indent1 << "cache_specs[" << i << "] {\n";
            os << spec->debugString(indent + 2);
            os << indent1 << "}\n";
        }

        os << indent1 << "global_layer_ids.size=" << global_layer_ids.size() << "\n";
        os << indent1 << "global_layer_ids=" << rtp_llm::vectorsToString(global_layer_ids) << "\n";

        os << indent1 << "mtp_sub_configs.size=" << mtp_sub_configs.size() << "\n";
        for (size_t i = 0; i < mtp_sub_configs.size(); ++i) {
            const auto& sub = mtp_sub_configs[i];
            if (!sub) {
                os << indent1 << "mtp_sub_configs[" << i << "]=null\n";
                continue;
            }
            os << indent1 << "mtp_sub_configs[" << i << "]:\n";
            os << sub->debugString(indent + 4);
        }

        os << indent_str << "}\n";
        return os.str();
    }
};
}  // namespace rtp_llm
