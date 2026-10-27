#pragma once

#include <memory>
#include <vector>
#include <map>
#include <sstream>
#include <string>
#include <utility>

#include "rtp_llm/cpp/cache/BlockInfo.h"
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

// A plane is a view into one native kernel page, with entry as its first axis.
// K/V, heads and quantization scales keep their original offsets and strides.
struct CacheEntryPlane {
    bool                 scale        = false;
    DataType             dtype        = TYPE_INVALID;
    size_t               offset_bytes = 0;
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;
};

struct CacheEntryLayout {
    uint32_t                     rows_per_page = 0;
    std::vector<CacheEntryPlane> planes;
    bool                         partition_heads = false;

    void validate(size_t kv_page_bytes, size_t scale_page_bytes) const {
        RTP_LLM_CHECK_WITH_INFO(rows_per_page > 0 && !planes.empty(),
                                "cache group does not expose addressable KV entries");
        for (const auto& plane : planes) {
            const size_t bytes = plane.scale ? scale_page_bytes : kv_page_bytes;
            RTP_LLM_CHECK_WITH_INFO(plane.dtype != TYPE_INVALID && !plane.shape.empty()
                                        && plane.shape.front() == rows_per_page
                                        && plane.shape.size() == plane.strides.size(),
                                    "invalid cache entry plane shape");
            const size_t element = getTypeSize(plane.dtype);
            RTP_LLM_CHECK_WITH_INFO(element > 0 && bytes % element == 0 && plane.offset_bytes % element == 0
                                        && plane.offset_bytes < bytes,
                                    "invalid cache entry plane offset or dtype");
            size_t remaining = (bytes - plane.offset_bytes) / element - 1;
            for (size_t axis = 0; axis < plane.shape.size(); ++axis) {
                RTP_LLM_CHECK_WITH_INFO(plane.shape[axis] > 0 && plane.strides[axis] > 0
                                            && static_cast<uint64_t>(plane.shape[axis] - 1)
                                                   <= remaining / plane.strides[axis],
                                        "cache entry plane exceeds its native page");
                remaining -= (plane.shape[axis] - 1) * plane.strides[axis];
            }
        }
    }
};

struct KVCacheSpec {
    std::string tag;
    uint32_t    seq_size_per_block;         // tokens/physical block of this group
    uint32_t    kernel_seq_size_per_block;  // tokens/kernel page of this group
    uint32_t    local_kv_head_num;          // layout heads on this rank
    // Cache-key coverage may differ from physical allocation under CP. A zero
    // value means one key covers one physical group block.
    size_t cache_key_token_stride = 0;

    KVCacheSpecType type = KVCacheSpecType::MultiHeadAttention;

    KVCacheSpec(std::string tag,
                uint32_t    seq_size_per_block,
                uint32_t    kernel_seq_size_per_block,
                uint32_t    local_kv_head_num):
        tag(std::move(tag)),
        seq_size_per_block(seq_size_per_block),
        kernel_seq_size_per_block(kernel_seq_size_per_block),
        local_kv_head_num(local_kv_head_num) {
        RTP_LLM_CHECK_WITH_INFO(
            seq_size_per_block > 0, "KVCacheSpecDesc tag=%s requires positive seq_size_per_block", this->tag.c_str());
        RTP_LLM_CHECK_WITH_INFO(kernel_seq_size_per_block > 0,
                                "KVCacheSpecDesc tag=%s requires positive kernel_seq_size_per_block",
                                this->tag.c_str());
        RTP_LLM_CHECK_WITH_INFO(
            seq_size_per_block % kernel_seq_size_per_block == 0,
            "KVCacheSpecDesc tag=%s seq_size_per_block=%u must be divisible by kernel_seq_size_per_block=%u",
            this->tag.c_str(),
            seq_size_per_block,
            kernel_seq_size_per_block);
        RTP_LLM_CHECK_WITH_INFO(
            local_kv_head_num > 0, "KVCacheSpecDesc tag=%s requires positive local_kv_head_num", this->tag.c_str());
    }

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

    // State/ring specs remain unsupported unless they provide an entry view.
    virtual CacheEntryLayout entryLayout() const {
        return {};
    }

    virtual KVCacheSpecPtr clone() const = 0;

    size_t cacheKeyTokenStride() const {
        return cache_key_token_stride == 0 ? seq_size_per_block : cache_key_token_stride;
    }

    std::string fingerprint() const {
        std::ostringstream os;
        os << "tag=" << tag << ";type=" << static_cast<int>(type) << ";dtype=" << static_cast<int>(memoryLayoutDType())
           << ";seq_size_per_block=" << seq_size_per_block << ";kernel_seq_size_per_block=" << kernel_seq_size_per_block
           << ";local_kv_head_num=" << local_kv_head_num << ";cache_key_token_stride=" << cacheKeyTokenStride()
           << ";block_elems=" << block_size() << ";k_block_elems=" << k_block_size()
           << ";v_block_elems=" << v_block_size() << ";block_bytes=" << block_size_bytes()
           << ";k_block_bytes=" << k_block_size_bytes() << ";v_block_bytes=" << v_block_size_bytes()
           << ";block_payload_bytes=" << block_payload_bytes() << ";k_block_payload_bytes=" << k_block_payload_bytes()
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
        os << indent1 << "kernel_seq_size_per_block=" << kernel_seq_size_per_block << "\n";
        os << indent1 << "local_kv_head_num=" << local_kv_head_num << "\n";
        os << indent1 << "cache_key_token_stride=" << cacheKeyTokenStride() << "\n";
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
