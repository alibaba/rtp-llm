#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.h"
#include "rtp_llm/cpp/cache/block_tree_cache/ScopeRollback.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferDispatcher.h"

#include <set>

namespace rtp_llm {
namespace {
struct StagePage {
    size_t       group, row;
    uint64_t     node_identity;
    BlockIdxType disk, host;
};
struct StageWaiter {
    size_t                    remaining;
    bool                      success = true;
    std::function<void(bool)> done;
    std::mutex                mutex;

    StageWaiter(size_t n, std::function<void(bool)> callback): remaining(n), done(std::move(callback)) {}
    void finish(bool ok) {
        std::unique_lock lock(mutex);
        success &= ok;
        if (--remaining)
            return;
        auto       notify = std::move(done);
        const bool result = success;
        lock.unlock();
        notify(result);
    }
};
}  // namespace

DiskStage BlockTreeCache::prefetchDisk(const BlockTreeKeys&                          keys,
                                       const std::vector<std::pair<size_t, size_t>>& positions,
                                       std::optional<int64_t>                        request,
                                       std::function<void(bool)>                     done) {
    if (stopping_.load() || !config_.enable_disk_cache || !config_.enable_host_cache)
        return DiskStage::None;
    return loader_.prefetchDisk(
        keys, positions, request, std::chrono::milliseconds(config_.disk_cache_sync_timeout_ms), std::move(done));
}

DiskStage BlockTreeLoader::prefetchDisk(const BlockTreeKeys&                          keys,
                                        const std::vector<std::pair<size_t, size_t>>& positions,
                                        std::optional<int64_t>                        request,
                                        std::chrono::milliseconds                     timeout,
                                        std::function<void(bool)>                     done) {
    using Slot = std::pair<size_t, BlockIdxType>;
    const std::set<std::pair<size_t, size_t>> requested(positions.begin(), positions.end());
    std::vector<StagePage>                    staged;
    std::vector<Slot>                         waiting;
    std::vector<TransferDescriptor>           descriptors;
    staged.reserve(requested.size());
    waiting.reserve(requested.size());
    descriptors.reserve(requested.size());
    std::unique_lock lock(mutex_);
    const auto&      groups = tree_->groupSets();
    const auto       path   = tree_->findNode(keys);
    block_tree_cache_detail::ScopeRollback undo_staged([&] {
        for (const auto& page : staged) {
            const auto& pools = groups[page.group]->pools();
            pools.host->decTreeRef(page.host, BlockTreeRefType::LOAD);
            pools.disk->decTreeRef(page.disk, BlockTreeRefType::LOAD);
        }
    });
    // Protect the Host rows of the chosen match while reserving its Disk
    // targets. Capacity recovery may evict other cached Host rows.
    std::vector<Slot> host_sources;
    host_sources.reserve(requested.size());
    block_tree_cache_detail::ScopeRollback release_host_sources([&] {
        for (const auto& [group, block] : host_sources) {
            groups[group]->pools().host->decTreeRef(block, BlockTreeRefType::LOAD);
        }
    });
    for (auto [group, row] : requested) {
        if (group < groups.size() && row < path.size()) {
            const auto& resource = path[row]->group_set_resources[group];
            if (resource.getTopTier() == Tier::HOST && !resource.transfer_detached) {
                groups[group]->pools().host->incTreeRef(resource.host_block, BlockTreeRefType::LOAD);
                host_sources.emplace_back(group, resource.host_block);
            }
        }
    }
    for (auto [g, row] : requested) {
        if (g >= groups.size() || row >= path.size())
            continue;
        auto*       node     = path[row];
        const auto& resource = node->group_set_resources[g];
        if (resource.getTopTier() != Tier::DISK || resource.transfer_detached)
            continue;
        const Slot source{g, resource.disk_block};
        if (disk_waiters_.count(source)) {
            waiting.push_back(source);
            continue;
        }
        if (resource.transfer_state != GroupSetTransferState::IDLE)
            continue;
        const auto& group     = groups[g];
        const auto  source_id = resource.disk_block;
        group->pools().disk->incTreeRef(source_id, BlockTreeRefType::LOAD);
        block_tree_cache_detail::ScopeRollback undo_source([&] {
            group->pools().disk->decTreeRef(source_id, BlockTreeRefType::LOAD);
        });
        auto host = group->allocateSingleBlock(Tier::HOST, BlockTreeRefType::LOAD);
        if (isNullBlockIdx(host)) {
            const bool freed = evictor_.dropColdHostPageLocked(g, "disk_restore");
            if (freed)
                host = group->allocateSingleBlock(Tier::HOST, BlockTreeRefType::LOAD);
            if (isNullBlockIdx(host)) {
                return freed ? DiskStage::Retry : DiskStage::None;
            }
        }
        staged.push_back({g, row, node->identity, source_id, host});
        undo_source.dismiss();
        TransferDescriptor desc(node, g, row, Tier::DISK, Tier::HOST, {source_id}, true);
        desc.target_blocks = {host};
        // Only the immutable transfer identity crosses the worker boundary.
        desc.node = nullptr;
        descriptors.push_back(std::move(desc));
        waiting.push_back(source);
    }
    if (waiting.empty())
        return DiskStage::None;
    auto waiter = std::make_shared<StageWaiter>(waiting.size(), std::move(done));
    std::vector<Slot> registered;
    registered.reserve(waiting.size());
    block_tree_cache_detail::ScopeRollback undo_waiters([&] {
        for (auto slot : registered) {
            const auto entry = disk_waiters_.find(slot);
            entry->second.pop_back();
            if (entry->second.empty()) {
                disk_waiters_.erase(entry);
            }
        }
    });
    for (auto slot : waiting) {
        const auto entry = disk_waiters_.try_emplace(slot).first;
        block_tree_cache_detail::ScopeRollback undo_empty([&] {
            if (entry->second.empty()) {
                disk_waiters_.erase(entry);
            }
        });
        entry->second.emplace_back([waiter](bool ok) { waiter->finish(ok); });
        registered.push_back(slot);
        undo_empty.dismiss();
    }
    if (staged.empty()) {
        undo_waiters.dismiss();
        return DiskStage::Pending;
    }
    TransferTask task(std::move(descriptors), timeout, request);
    BlockTransferDispatcher::TransferDoneCallback complete = [this, keys, staged](TransferResult result) {
            std::vector<std::pair<std::function<void(bool)>, bool>> callbacks;
            {
                std::lock_guard lock(mutex_);
                const auto      current = tree_->findNode(keys);
                for (const auto& page : staged) {
                    const auto& group   = tree_->groupSets()[page.group];
                    bool        adopted = false;
                    if (result.ok() && result.buffer_safe && page.row < current.size()
                        && current[page.row]->identity == page.node_identity) {
                        auto* node     = current[page.row];
                        auto& resource = node->group_set_resources[page.group];
                        if (resource.transfer_state == GroupSetTransferState::IDLE && !resource.transfer_detached
                            && resource.disk_block == page.disk && resource.getTopTier() == Tier::DISK) {
                            group->pools().host->incTreeRef(page.host, BlockTreeRefType::CACHE);
                            resource.host_block            = page.host;
                            resource.retained_disk_replica = true;
                            evictor_.admitCandidate(node, page.group, Tier::HOST);
                            evictor_.admitCandidate(node, page.group, Tier::DISK);
                            tree_->refreshPublishedState(node);
                            adopted = true;
                        }
                    }
                    group->pools().host->decTreeRef(page.host, BlockTreeRefType::LOAD);
                    group->pools().disk->decTreeRef(page.disk, BlockTreeRefType::LOAD);
                    auto subscribers = disk_waiters_.extract({page.group, page.disk});
                    for (auto& callback : subscribers.mapped())
                        callbacks.emplace_back(std::move(callback), adopted);
                }
            }
            for (auto& [callback, ok] : callbacks)
                callback(ok);
        };
    release_host_sources.run();
    undo_staged.dismiss();
    undo_waiters.dismiss();
    lock.unlock();
    transfer_dispatcher_->runTransfer(std::move(task), std::move(complete), "disk_restore");
    return DiskStage::Pending;
}
}  // namespace rtp_llm
