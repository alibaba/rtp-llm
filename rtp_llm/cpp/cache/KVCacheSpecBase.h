#pragma once

#include <memory>
#include <vector>
#include <map>
#include <sstream>
#include <string>
#include <utility>

#include "rtp_llm/cpp/cache/BlockInfo.h"
#include "rtp_llm/cpp/cache/Types.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/models_py/bindings/core/Types.h"
#include "rtp_llm/cpp/model_utils/AttentionConfig.h"

namespace rtp_llm {

enum KVCacheSpecType {
    MultiHeadAttention,        // MHAKVCacheSpec: standard multi-head attention KV cache
    MultiHeadLatentAttention,  // MLAKVCacheSpec: MLA compressed latent KV cache
    LinearAttention,           // LinearKVCacheSpec: linear / SSM attention state cache
    OpaqueKV,                  // Byte-addressed opaque paged KV pool
    OpaqueState,               // Fixed-allocation opaque state cache
};

inline KVPartitionBytes splitKVPartitionBytes(size_t      full_block_bytes,
                                              size_t      k_block_bytes,
                                              size_t      v_block_bytes,
                                              int         heads,
                                              int         partition_count,
                                              int         partition_id,
                                              const char* debug_name) {
    RTP_LLM_CHECK_WITH_INFO(partition_count > 0, "partition_count must be > 0");
    RTP_LLM_CHECK_WITH_INFO(partition_id >= 0 && partition_id < partition_count,
                            "partition_id out of range: %d / %d",
                            partition_id,
                            partition_count);
    RTP_LLM_CHECK_WITH_INFO(heads > 0, "heads must be > 0, got=%d (%s)", heads, debug_name);
    RTP_LLM_CHECK_WITH_INFO(k_block_bytes + v_block_bytes == full_block_bytes,
                            "block bytes mismatch (%s): full=%zu k_partition=%zu v_partition=%zu",
                            debug_name,
                            full_block_bytes,
                            k_block_bytes,
                            v_block_bytes);
    RTP_LLM_CHECK_WITH_INFO(k_block_bytes % static_cast<size_t>(heads) == 0,
                            "k_block_bytes must be divisible by heads (%s): k_partition=%zu heads=%d",
                            debug_name,
                            k_block_bytes,
                            heads);
    RTP_LLM_CHECK_WITH_INFO(v_block_bytes % static_cast<size_t>(heads) == 0,
                            "v_block_bytes must be divisible by heads (%s): v_partition=%zu heads=%d",
                            debug_name,
                            v_block_bytes,
                            heads);
    RTP_LLM_CHECK_WITH_INFO(heads % partition_count == 0,
                            "heads must be divisible by partition_count (%s): heads=%d partition_count=%d",
                            debug_name,
                            heads,
                            partition_count);

    const size_t k_partition_bytes_per_head = k_block_bytes / static_cast<size_t>(heads);
    const size_t v_partition_bytes_per_head = v_block_bytes / static_cast<size_t>(heads);
    const int    head_cnt                   = heads / partition_count;
    const int    head_begin                 = partition_id * head_cnt;

    const size_t k_partition_off = static_cast<size_t>(head_begin) * k_partition_bytes_per_head;
    const size_t v_partition_off = k_block_bytes + static_cast<size_t>(head_begin) * v_partition_bytes_per_head;
    const size_t k_partition_sz  = static_cast<size_t>(head_cnt) * k_partition_bytes_per_head;
    const size_t v_partition_sz  = static_cast<size_t>(head_cnt) * v_partition_bytes_per_head;
    return {k_partition_off, k_partition_sz, v_partition_off, v_partition_sz};
}

inline const char* KVCacheSpecTypeToString(KVCacheSpecType t) {
    switch (t) {
        case KVCacheSpecType::MultiHeadAttention:
            return "MultiHeadAttention";
        case KVCacheSpecType::MultiHeadLatentAttention:
            return "MultiHeadLatentAttention";
        case KVCacheSpecType::LinearAttention:
            return "LinearAttention";
        case KVCacheSpecType::OpaqueKV:
            return "OpaqueKV";
        case KVCacheSpecType::OpaqueState:
            return "OpaqueState";
        default:
            return "Unknown";
    }
}

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

    virtual size_t block_payload_bytes() const {
        return block_size_bytes();
    }
    virtual size_t k_block_payload_bytes() const {
        return k_block_size_bytes();
    }
    virtual size_t v_block_payload_bytes() const {
        return v_block_size_bytes();
    }

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
           << ";v_block_bytes=" << v_block_size_bytes() << ";block_payload_bytes=" << block_payload_bytes()
           << ";k_block_payload_bytes=" << k_block_payload_bytes()
           << ";v_block_payload_bytes=" << v_block_payload_bytes() << ";scale_block_bytes=" << scale_block_size_bytes()
           << ";k_scale_block_bytes=" << k_scale_block_size_bytes()
           << ";v_scale_block_bytes=" << v_scale_block_size_bytes();
        return os.str();
    }

    virtual std::string debugString(size_t indent = 0) const = 0;

protected:
    // Helper method to generate common parts of debug string
    std::string commonDebugString(size_t indent = 0) const {
        const std::string indent_str = std::string(indent, ' ');
        const std::string indent1    = indent_str + "  ";

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
        os << indent1 << "block_payload_bytes=" << block_payload_bytes() << "\n";
        os << indent1 << "k_block_payload_bytes=" << k_block_payload_bytes() << "\n";
        os << indent1 << "v_block_payload_bytes=" << v_block_payload_bytes() << "\n";
        os << indent1 << "scale_block_size_bytes=" << scale_block_size_bytes() << "\n";
        os << indent1 << "k_scale_block_size_bytes=" << k_scale_block_size_bytes() << "\n";
        os << indent1 << "v_scale_block_size_bytes=" << v_scale_block_size_bytes() << "\n";
        return os.str();
    }
};

}  // namespace rtp_llm
