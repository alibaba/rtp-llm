#pragma once

#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/KVCacheSpecType.h"
#include "rtp_llm/cpp/cache/Types.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/models_py/bindings/core/Types.h"

namespace rtp_llm {

struct KVCacheSpec;
using KVCacheSpecPtr    = std::shared_ptr<KVCacheSpec>;
using LayerKVCacheSpecs = std::vector<std::vector<KVCacheSpecPtr>>;

struct KVCacheSpec {
    std::string tag;
    uint32_t    seq_size_per_block = 1;

    KVCacheSpecType type = KVCacheSpecType::MultiHeadAttention;

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

    virtual rtp_llm::DataType memoryLayoutDType() const = 0;

    virtual KVCacheSpecPtr clone() const = 0;

    std::string fingerprint() const {
        std::ostringstream os;
        os << "tag=" << tag << ";type=" << static_cast<int>(type) << ";dtype=" << static_cast<int>(memoryLayoutDType())
           << ";seq_size_per_block=" << seq_size_per_block << ";block_elems=" << block_size()
           << ";k_block_elems=" << k_block_size() << ";v_block_elems=" << v_block_size()
           << ";block_bytes=" << block_size_bytes() << ";k_block_bytes=" << k_block_size_bytes()
           << ";v_block_bytes=" << v_block_size_bytes() << ";scale_block_bytes=" << scale_block_size_bytes()
           << ";k_scale_block_bytes=" << k_scale_block_size_bytes()
           << ";v_scale_block_bytes=" << v_scale_block_size_bytes();
        return os.str();
    }

    virtual std::string debugString(size_t indent = 0) const = 0;

protected:
    std::string commonDebugString(size_t indent = 0) const {
        const std::string indent_str(indent, ' ');
        const std::string indent1 = indent_str + "  ";

        std::ostringstream os;
        os << indent1 << "tag=" << tag << "\n";
        os << indent1 << "type=" << KVCacheSpecTypeToString(type) << "(" << static_cast<int>(type) << ")\n";
        os << indent1 << "dtype=" << static_cast<int>(memoryLayoutDType()) << "\n";
        os << indent1 << "seq_size_per_block=" << seq_size_per_block << "\n";
        os << indent1 << "block_size=" << block_size() << "\n";
        os << indent1 << "k_block_size=" << k_block_size() << "\n";
        os << indent1 << "v_block_size=" << v_block_size() << "\n";
        os << indent1 << "block_size_bytes=" << block_size_bytes() << "\n";
        os << indent1 << "k_block_size_bytes=" << k_block_size_bytes() << "\n";
        os << indent1 << "v_block_size_bytes=" << v_block_size_bytes() << "\n";
        os << indent1 << "scale_block_size_bytes=" << scale_block_size_bytes() << "\n";
        os << indent1 << "k_scale_block_size_bytes=" << k_scale_block_size_bytes() << "\n";
        os << indent1 << "v_scale_block_size_bytes=" << v_scale_block_size_bytes() << "\n";
        return os.str();
    }
};

}  // namespace rtp_llm
