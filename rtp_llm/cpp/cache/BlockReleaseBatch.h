#pragma once

#include <cstddef>
#include <unordered_map>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/IBlockPool.h"

namespace rtp_llm {

struct BlockReleaseReceipt {
    size_t       group_id{0};
    BlockIdxType block_id{NULL_BLOCK_IDX};
    BlockRefType released_ref_type{BlockRefType::REQUEST};
    uint32_t     old_total_ref_count{0};
    uint32_t     new_total_ref_count{0};
    bool         block_released{false};
};

class BlockReleaseBatch {
public:
    void append(size_t group_id, const std::vector<BlockRefTransition>& transitions);

    std::vector<BlockReleaseReceipt> finish();

private:
    struct ReceiptKey {
        size_t       group_id;
        BlockIdxType block_id;

        bool operator==(const ReceiptKey& other) const {
            return group_id == other.group_id && block_id == other.block_id;
        }
    };

    struct ReceiptKeyHash {
        size_t operator()(const ReceiptKey& key) const {
            const size_t group_hash = std::hash<size_t>{}(key.group_id);
            const size_t block_hash = std::hash<BlockIdxType>{}(key.block_id);
            return group_hash ^ (block_hash << 1);
        }
    };

    std::vector<BlockReleaseReceipt>                        receipts_;
    std::unordered_map<ReceiptKey, size_t, ReceiptKeyHash> receipt_index_map_;
};

}  // namespace rtp_llm
