#pragma once

#include <map>
#include <optional>
#include <set>

#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/StorageBackend.h"

namespace rtp_llm {

// Only the tree decides whether the Host catalog can drop a committed row.
// StorageBackend holds the source and admits, dispatches and retires its I/O.
class HostStorageWriteback {
public:
    explicit HostStorageWriteback(IBlockPool& pool): writes(pool) {}

    std::optional<BlockIdxType> settle(uint64_t ticket, bool success) {
        std::optional<BlockIdxType> slot;
        writes.settle(ticket, success, [&](const StorageBackend::HostWrites::Source& source, bool committed) {
            // Concurrent readers can defer retirement of a committed page.
            // A failed publication stays readable and retries after a new access.
            if (committed)
                committed_.emplace(source.slot, source.group);
            else
                failed_.insert(source.slot);
            slot = source.slot;
        });
        return slot;
    }

    size_t expectedFrees(const std::set<BlockIdxType>& excluded) const {
        size_t count = writes.countIf([&](BlockIdxType slot) { return excluded.count(slot) == 0; });
        for (const auto& [slot, group] : committed_)
            count += excluded.count(slot) == 0;
        return count;
    }

    bool busy() const { return writes.busy(); }
    bool failed(BlockIdxType slot) const { return failed_.count(slot) != 0; }
    void accessed(BlockIdxType slot) { failed_.erase(slot); }

    std::optional<size_t> committedGroup(BlockIdxType slot) const {
        auto it = committed_.find(slot);
        if (it == committed_.end())
            return std::nullopt;
        return it->second;
    }

    void unpublish(BlockIdxType slot) {
        committed_.erase(slot);
        failed_.erase(slot);
    }

    StorageBackend::HostWrites writes;

private:
    std::map<BlockIdxType, size_t> committed_;
    std::set<BlockIdxType>         failed_;
};

}  // namespace rtp_llm
