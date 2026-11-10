#include "rtp_llm/cpp/cache/block_tree_cache/transfer/CrcTransferService.h"
#include "rtp_llm/cpp/cache/block_tree_cache/ScopeRollback.h"

namespace rtp_llm {
namespace {
TransferResult rejectedWithoutIo() {
    TransferResult result(TransferStatus::INVALID_ARGS);
    result.buffer_safe = true;
    return result;
}

bool validCopyBuffers(const HostBufferView& host, const TransferDescriptor& descriptor, const GroupSet& group) {
    const auto& blocks = descriptor.blocksAt(Tier::DEVICE);
    if (!isValidHostBufferView(host, group.payloadBytes(), group.storageBytes())
        || blocks.size() != group.devicePools().size())
        return false;
    for (size_t member = 0; member < blocks.size(); ++member)
        if (!group.devicePools()[member]->validBlock(blocks[member]))
            return false;
    return true;
}
}  // namespace

TransferResult CrcTransferService::execute(int                                    device,
                                           const std::vector<CrcCopyItem>&        items,
                                           CrcCopyDirection                       direction,
                                           const std::vector<TransferDescriptor>& descriptors) {
    Slot* slot = nullptr;
    {
        std::unique_lock<std::mutex> lock(mutex_);
        const auto                   found = slots_.find(device);
        if (found == slots_.end() || !accepting_)
            return rejectedWithoutIo();
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
            return rejectedWithoutIo();
    }
    block_tree_cache_detail::ScopeRollback release([&] {
        std::lock_guard<std::mutex> lock(mutex_);
        slot->updating = false;
        cv_.notify_all();
    });
    std::vector<CrcCopyFailure>            failures;
    const auto                             result =
        direction == CrcCopyDirection::STORE ? slot->copy->store(items, failures) : slot->copy->load(items, failures);
    return mapResult(result, direction, descriptors, failures);
}

TransferResult CrcTransferService::copy(const std::vector<HostBufferView>&     hosts,
                                        const std::vector<TransferDescriptor>& descriptors,
                                        const std::vector<const GroupSet*>&    groups) {
    if (async_ || hosts.empty() || hosts.size() != descriptors.size() || hosts.size() != groups.size())
        return rejectedWithoutIo();
    const bool               store = descriptors.front().source_tier == Tier::DEVICE;
    std::vector<CrcCopyItem> items;
    items.reserve(hosts.size());
    for (size_t i = 0; i < hosts.size(); ++i) {
        const auto& group = groups_.at(groups[i]);
        if (!group->crcEnabled() || (descriptors[i].source_tier == Tier::DEVICE) != store
            || !validCopyBuffers(hosts[i], descriptors[i], *group))
            return rejectedWithoutIo();
        items.push_back(makeItem(hosts[i], descriptors[i], group));
    }
    return execute(
        deviceFor(*groups.front()), items, store ? CrcCopyDirection::STORE : CrcCopyDirection::LOAD, descriptors);
}

}  // namespace rtp_llm
