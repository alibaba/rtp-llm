#include "rtp_llm/cpp/cache/CPSlotMapper.h"

#include <algorithm>
#include <stdexcept>

#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {
namespace {

size_t groupSeqSize(const CacheConfig& config, size_t gid, size_t fallback) {
    return gid < static_cast<size_t>(config.groupNums()) ? config.seqSizePerBlockForGroup(gid) : fallback;
}

bool isCompactFullBlockList(const KVCacheResource&  source,
                            const BlockIndicesType& src_blocks,
                            const CacheKeysType&    selected_keys) {
    return src_blocks.size() <= selected_keys.size() || src_blocks.size() < source.cacheKeys().size();
}

bool selectedLastRankKeysAreAligned(const KVCacheResource& source, int cp_size) {
    if (source.lastBlockAligned()) {
        return true;
    }
    const auto& keys = source.cacheKeys();
    if (keys.empty() || cp_size <= 1) {
        return source.lastBlockAligned();
    }
    const int partial_key_pos = static_cast<int>(keys.size() - 1);
    const int last_rank       = cp_size - 1;
    return partial_key_pos % cp_size != last_rank;
}

}  // namespace

CPSlotMapper::CPSlotMapper(): cp_rank_(0), cp_size_(1), block_size_(1), virtual_block_size_(1) {}

CPSlotMapper::CPSlotMapper(int cp_rank, int cp_size, int block_size):
    cp_rank_(cp_rank), cp_size_(cp_size), block_size_(block_size), virtual_block_size_(block_size * cp_size) {
    if (cp_size <= 0) {
        throw std::invalid_argument("CPSlotMapper cp_size must be positive");
    }
    if (block_size <= 0) {
        throw std::invalid_argument("CPSlotMapper block_size must be positive");
    }
    if (cp_rank < 0 || cp_rank >= cp_size) {
        throw std::invalid_argument("CPSlotMapper cp_rank out of range");
    }
}

CpGroupLayout CPSlotMapper::layoutForGroup(const CacheConfig& config, size_t gid) const {
    CpGroupLayout layout;
    const auto    policy      = gid < static_cast<size_t>(config.groupNums()) ? config.policyForGroup(gid) :
                                                                                defaultCacheGroupPolicy(CacheGroupType::FULL);
    layout.active_tail_blocks = policy.active_tail_blocks > 0 ? static_cast<size_t>(policy.active_tail_blocks) : 0;
    if (!isSharded() || gid >= static_cast<size_t>(config.groupNums())) {
        return layout;
    }
    layout.mapping = policy.cp_mapping;
    // FULL groups use page/block-level CP mapping. Byte slicing is only valid for
    // state/SWA-style groups whose writer stores matching sliced payloads.
    layout.slice = policy.group_type == CacheGroupType::FULL ? CpBlockSliceMode::NONE : policy.cp_slice;
    return layout;
}

bool CPSlotMapper::usesCpCanonicalKeys(const CacheConfig& config, size_t gid) const {
    return layoutForGroup(config, gid).usesCpCanonicalKeys();
}

bool CPSlotMapper::blockRoundRobinGroup(const CacheConfig& config, size_t gid) const {
    return layoutForGroup(config, gid).mapping == CpBlockMappingMode::BLOCK_ROUND_ROBIN;
}

bool CPSlotMapper::compactLastRankGroup(const CacheConfig& config, size_t gid) const {
    return layoutForGroup(config, gid).mapping == CpBlockMappingMode::COMPACT_LAST_RANK;
}

int CPSlotMapper::localBlockCount(int seq_len) const {
    if (seq_len <= 0) {
        return 0;
    }
    // All CP ranks keep the same block count = ceil(total_blocks / cp_size).
    // rank0 is the controller: it allocates blocks and broadcasts block_ids
    // to all ranks.  Using a uniform count simplifies KV cache management.
    int total_blocks = (seq_len + block_size_ - 1) / block_size_;
    return (total_blocks + cp_size_ - 1) / cp_size_;
}

int CPSlotMapper::effectiveSeqLenForAlloc(int actual_seq_len) const {
    return localBlockCount(actual_seq_len) * block_size_;
}

int CPSlotMapper::effectiveSeqLenForAlloc(const CacheConfig& config, size_t gid, int seq_len) const {
    if (!blockRoundRobinGroup(config, gid)) {
        return seq_len;
    }
    return effectiveSeqLenForAlloc(seq_len);
}

size_t CPSlotMapper::logicalSeqSizePerBlock(const CacheConfig& config, size_t gid) const {
    if (blockRoundRobinGroup(config, gid)) {
        return static_cast<size_t>(virtual_block_size_);
    }
    return groupSeqSize(config, gid, config.seq_size_per_block);
}

CacheKeysType CPSlotMapper::canonicalCacheKeys(const CacheKeysType& full_keys) const {
    if (!isSharded()) {
        return full_keys;
    }
    CacheKeysType local;
    const int     start = cp_size_ - 1;
    for (int i = start; i < static_cast<int>(full_keys.size()); i += cp_size_) {
        local.push_back(full_keys[static_cast<size_t>(i)]);
    }
    return local;
}

CacheKeysType
CPSlotMapper::localCacheKeys(const CacheConfig& config, size_t gid, const CacheKeysType& full_keys) const {
    return usesCpCanonicalKeys(config, gid) ? canonicalCacheKeys(full_keys) : full_keys;
}

std::vector<CacheStoreBlockPair> CPSlotMapper::buildStorePlan(const CacheConfig& config,
                                                              size_t             gid,
                                                              size_t             total_logical_blocks,
                                                              size_t             reuse_block_size,
                                                              bool               use_hybrid) const {
    auto policy = gid < static_cast<size_t>(config.groupNums()) ? config.policyForGroup(gid) :
                                                                  defaultCacheGroupPolicy(CacheGroupType::FULL);
    if (!isSharded() || gid >= static_cast<size_t>(config.groupNums())) {
        policy.cp_mapping = CpBlockMappingMode::NONE;
    }
    return buildCacheStorePlan(policy, total_logical_blocks, reuse_block_size, use_hybrid, cp_rank_, cp_size_);
}

std::vector<CacheStoreBlockPair> CPSlotMapper::buildStorePlan(CacheGroupType group_type,
                                                              size_t         total_logical_blocks,
                                                              size_t         reuse_block_size,
                                                              bool           use_hybrid) const {
    return buildStorePlan(defaultCacheGroupPolicy(group_type), total_logical_blocks, reuse_block_size, use_hybrid);
}

std::vector<CacheStoreBlockPair> CPSlotMapper::buildStorePlan(const CacheGroupPolicy& policy,
                                                              size_t                  total_logical_blocks,
                                                              size_t                  reuse_block_size,
                                                              bool                    use_hybrid) const {
    return buildCacheStorePlan(policy, total_logical_blocks, reuse_block_size, use_hybrid, cp_rank_, cp_size_);
}

std::vector<BlockInfo> CPSlotMapper::sliceBlockForPeer(const CacheConfig&     config,
                                                       size_t                 gid,
                                                       std::vector<BlockInfo> parts,
                                                       size_t                 peer_idx) const {
    const auto layout = layoutForGroup(config, gid);
    if (!isSharded() || layout.slice == CpBlockSliceMode::NONE) {
        return parts;
    }
    RTP_LLM_CHECK_WITH_INFO(parts.size() == 1, "CP byte slicing expects one block part, got %zu", parts.size());
    RTP_LLM_CHECK_WITH_INFO(
        peer_idx < static_cast<size_t>(cp_size_), "CP slice peer_idx=%zu out of cp_size=%d", peer_idx, cp_size_);
    auto spec = config.specForGroup(gid);
    RTP_LLM_CHECK_WITH_INFO(spec != nullptr, "CP slice got null spec for gid=%zu", gid);
    auto& block = parts[0];
    RTP_LLM_CHECK_WITH_INFO(block.addr != nullptr, "CP byte slicing got null block addr");

    size_t slice_bytes = 0;
    if (layout.slice == CpBlockSliceMode::EQUAL_BYTES) {
        RTP_LLM_CHECK_WITH_INFO(block.size_bytes % static_cast<size_t>(cp_size_) == 0,
                                "CP block bytes %zu not divisible by cp_size %d",
                                block.size_bytes,
                                cp_size_);
        slice_bytes = block.size_bytes / static_cast<size_t>(cp_size_);
    } else {
        const auto payload_bytes = spec->k_block_payload_bytes();
        RTP_LLM_CHECK_WITH_INFO(payload_bytes > 0, "CP payload slicing requires positive payload bytes");
        RTP_LLM_CHECK_WITH_INFO(payload_bytes % static_cast<size_t>(cp_size_) == 0,
                                "CP payload bytes %zu not divisible by cp_size %d",
                                payload_bytes,
                                cp_size_);
        slice_bytes = payload_bytes / static_cast<size_t>(cp_size_);
    }

    const size_t slice_offset = slice_bytes * peer_idx;
    RTP_LLM_CHECK_WITH_INFO(slice_offset + slice_bytes <= block.size_bytes,
                            "CP slice [%zu, %zu) exceeds block bytes %zu",
                            slice_offset,
                            slice_offset + slice_bytes,
                            block.size_bytes);
    block.addr       = static_cast<void*>(static_cast<char*>(block.addr) + slice_offset);
    block.size_bytes = slice_bytes;
    return parts;
}

KVCacheResource CPSlotMapper::projectConnectorResource(const KVCacheResource& source,
                                                       const CacheConfig&     config,
                                                       const CacheKeysType&   selected_keys) const {
    KVCacheResource selected = source;
    selected.initGroups(config.topologyPtr());
    selected.setCacheKeys(selected_keys);
    const bool selected_aligned = selectedLastRankKeysAreAligned(source, cp_size_);
    selected.setLastBlockAligned(selected_aligned);

    // Memory connector intentionally drops the last key to avoid matching a
    // partial tail.  After CP Page-RR remap, a source partial can belong to a
    // non-last rank, making the selected last-rank key complete.  Append the
    // original partial key as a connector-only dummy tail so the drop-last
    // contract discards the dummy, not the usable selected key.
    if (!source.lastBlockAligned() && selected_aligned && !source.cacheKeys().empty()) {
        selected.cacheKeys().push_back(source.cacheKeys().back());
        selected.rebuildLinearBlockDependencies();
        selected.setLastBlockAligned(false);
    }

    for (int gid = 0; gid < source.groupNums(); ++gid) {
        const auto&      src_blocks = source.blocks(gid);
        BlockIndicesType dst_blocks;
        dst_blocks.reserve(selected_keys.size());

        const auto layout = layoutForGroup(config, static_cast<size_t>(gid));
        if (layout.slice != CpBlockSliceMode::NONE) {
            for (size_t i = 0; i < selected_keys.size(); ++i) {
                dst_blocks.push_back(i < src_blocks.size() ? src_blocks[i] : NULL_BLOCK_IDX);
            }
        } else if (layout.mapping == CpBlockMappingMode::BLOCK_ROUND_ROBIN) {
            if (isCompactFullBlockList(source, src_blocks, selected_keys)) {
                for (size_t i = 0; i < selected_keys.size(); ++i) {
                    dst_blocks.push_back(i < src_blocks.size() ? src_blocks[i] : NULL_BLOCK_IDX);
                }
            } else {
                for (size_t logical_pos = static_cast<size_t>(cp_size_ - 1); dst_blocks.size() < selected_keys.size();
                     logical_pos += static_cast<size_t>(cp_size_)) {
                    dst_blocks.push_back(logical_pos < src_blocks.size() ? src_blocks[logical_pos] : NULL_BLOCK_IDX);
                }
            }
        } else {
            for (size_t logical_pos = static_cast<size_t>(cp_size_ - 1); dst_blocks.size() < selected_keys.size();
                 logical_pos += static_cast<size_t>(cp_size_)) {
                dst_blocks.push_back(logical_pos < src_blocks.size() ? src_blocks[logical_pos] : NULL_BLOCK_IDX);
            }
        }

        selected.mutableBlockIds(gid).assign(std::move(dst_blocks));
    }

    return selected;
}

}  // namespace rtp_llm
