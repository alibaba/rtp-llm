#include "rtp_llm/cpp/cache/block_tree_cache/transfer/CrcCopyExecutor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/ScopeRollback.h"

namespace rtp_llm {

TransferResult CrcCopyExecutor::execute(int                                    device,
                                        const std::vector<CrcCopyItem>&        items,
                                        CrcCopyDirection                       direction,
                                        const std::vector<TransferDescriptor>& descriptors) {
    Slot* slot = nullptr;
    {
        std::unique_lock<std::mutex> lock(mutex_);
        const auto                   found = slots_.find(device);
        if (found == slots_.end() || !accepting_)
            return TransferStatus::INVALID_ARGS;
        // Blocking compatibility path only. Async admission never waits here.
        cv_.wait(lock, [&] {
            if (!accepting_)
                return true;
            for (auto& candidate : found->second)
                if (!candidate.updating) {
                    candidate.updating = true;
                    slot               = &candidate;
                    return true;
                }
            return false;
        });
        if (!slot)
            return TransferStatus::INVALID_ARGS;
    }
    block_tree_cache_detail::ScopeRollback release([&] {
        std::lock_guard<std::mutex> lock(mutex_);
        slot->updating = false;
        cv_.notify_all();
    });
    std::vector<CrcCopyFailure>            failures;
    const auto                             result =
        direction == CrcCopyDirection::EVICT ? slot->copy->evict(items, failures) : slot->copy->load(items, failures);
    return mapResult(result, direction, descriptors, failures);
}

TransferResult CrcCopyExecutor::copy(const std::vector<HostBufferView>&     hosts,
                                     const std::vector<TransferDescriptor>& descriptors,
                                     const std::vector<const GroupSet*>&    groups) {
    const bool               evict = descriptors.front().source_tier == Tier::DEVICE;
    std::vector<CrcCopyItem> items;
    items.reserve(hosts.size());
    for (size_t i = 0; i < hosts.size(); ++i) {
        const auto& group = groups_.at(groups[i]);
        items.push_back(makeItem(hosts[i], descriptors[i], group));
    }
    return execute(
        deviceFor(*groups.front()), items, evict ? CrcCopyDirection::EVICT : CrcCopyDirection::LOAD, descriptors);
}

}  // namespace rtp_llm
