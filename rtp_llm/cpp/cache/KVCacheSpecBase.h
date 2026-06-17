#pragma once

#include <memory>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/Types.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/models_py/bindings/core/Types.h"
#include "rtp_llm/cpp/model_utils/AttentionConfig.h"

namespace rtp_llm {

enum KVCacheSpecType {
    MultiHeadAttention,
    MultiHeadLatentAttention,
    LinearAttention,
};

inline const char* KVCacheSpecTypeToString(KVCacheSpecType t) {
    switch (t) {
        case KVCacheSpecType::MultiHeadAttention:
            return "MultiHeadAttention";
        case KVCacheSpecType::MultiHeadLatentAttention:
            return "MultiHeadLatentAttention";
        case KVCacheSpecType::LinearAttention:
            return "LinearAttention";
        default:
            return "Unknown";
    }
}

struct KVCacheSpec;
using KVCacheSpecPtr = std::shared_ptr<KVCacheSpec>;

struct KVCacheSpec {
    std::string tag;
    std::vector<int> layers;
    uint32_t local_head_num_kv = 1;
    uint32_t seq_size_per_block = 1;

    KVCacheSpecType   type = KVCacheSpecType::MultiHeadAttention;
    rtp_llm::DataType dtype = rtp_llm::DataType::TYPE_INVALID;

    virtual size_t block_size() const   = 0;
    virtual size_t k_block_size() const = 0;
    virtual size_t v_block_size() const = 0;

    virtual size_t block_size_bytes() const   = 0;
    virtual size_t k_block_size_bytes() const = 0;
    virtual size_t v_block_size_bytes() const = 0;

    virtual size_t scale_block_size_bytes() const {
        return 0;
    }
    virtual size_t k_scale_block_size_bytes() const {
        return 0;
    }
    virtual size_t v_scale_block_size_bytes() const {
        return 0;
    }

    virtual KVCacheSpecPtr clone() const = 0;

    std::string fingerprint() const {
        std::ostringstream os;
        os << "tag=" << tag << ";type=" << static_cast<int>(type) << ";dtype=" << static_cast<int>(dtype)
           << ";local_head_num_kv=" << local_head_num_kv << ";seq_size_per_block=" << seq_size_per_block;
        os << fingerprintExtra();
        return os.str();
    }

    virtual std::string debugString(size_t indent = 0) const = 0;

protected:
    virtual std::string fingerprintExtra() const {
        return "";
    }

    // Helper method to generate common parts of debug string
    std::string commonDebugString(size_t indent = 0) const {
        const std::string indent_str = std::string(indent, ' ');
        const std::string indent1    = indent_str + "  ";

        std::ostringstream os;
        os << indent1 << "tag=" << tag << "\n";
        os << indent1 << "type=" << KVCacheSpecTypeToString(type) << "(" << static_cast<int>(type) << ")\n";
        os << indent1 << "dtype=" << static_cast<int>(dtype) << "\n";
        os << indent1 << "layers.size=" << layers.size() << "\n";
        os << indent1 << "local_head_num_kv=" << local_head_num_kv << "\n";
        os << indent1 << "seq_size_per_block=" << seq_size_per_block << "\n";
        os << indent1 << "block_size=" << block_size() << "\n";
        os << indent1 << "k_block_size=" << k_block_size() << "\n";
        os << indent1 << "v_block_size=" << v_block_size() << "\n";
        os << indent1 << "block_size_bytes=" << block_size_bytes() << "\n";
        os << indent1 << "k_block_size_bytes=" << k_block_size_bytes() << "\n";
        os << indent1 << "v_block_size_bytes=" << v_block_size_bytes() << "\n";
        return os.str();
    }
};

}  // namespace rtp_llm
