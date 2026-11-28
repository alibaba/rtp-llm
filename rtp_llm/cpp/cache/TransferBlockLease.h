// Copyright (c) 2026 Alibaba Cloud.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/IBlockPool.h"

#include <algorithm>
#include <memory>
#include <utility>

namespace rtp_llm {

// Keep physical pages allocated until publication and every READ grant retire.
class TransferBlockLease {
public:
    TransferBlockLease(std::shared_ptr<IBlockPool> pool, BlockIndicesType blocks):
        pool_(std::move(pool)), blocks_(std::move(blocks)) {
        blocks_.erase(std::remove_if(blocks_.begin(), blocks_.end(),
                                     [](int32_t block) { return block == 0 || block == NULL_BLOCK_IDX; }),
                      blocks_.end());
        std::sort(blocks_.begin(), blocks_.end());
        blocks_.erase(std::unique(blocks_.begin(), blocks_.end()), blocks_.end());
        pool_->incTreeRef(blocks_, BlockTreeRefType::STORE);
    }

    ~TransferBlockLease() {
        pool_->decTreeRef(blocks_, BlockTreeRefType::STORE);
    }

    TransferBlockLease(const TransferBlockLease&) = delete;
    TransferBlockLease& operator=(const TransferBlockLease&) = delete;

private:
    std::shared_ptr<IBlockPool> pool_;
    BlockIndicesType blocks_;
};

}  // namespace rtp_llm
