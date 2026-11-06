#include "rtp_llm/cpp/cache/block_tree_cache/transfer/CrcTransferService.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <limits>
#include <stdexcept>
#include <thread>
#include <utility>

#include "autil/CommonMacros.h"

#include "rtp_llm/cpp/cache/block_tree_cache/transfer/DeviceHostCopyGeometry.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace {
size_t checkedMultiply(size_t a, size_t b) {
    if (a && b > std::numeric_limits<size_t>::max() / a)
        throw std::overflow_error("CRC service capacity overflow");
    return a * b;
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

int CrcTransferService::deviceFor(const GroupSet& group) {
    return group.devicePools().front()->deviceIndex();
}

CrcTransferService::CrcTransferService(const std::vector<GroupSetPtr>& groups,
                                       size_t                          max_batch,
                                       size_t                          slots_per_device,
                                       int64_t                         world_rank,
                                       std::shared_ptr<CrcDumpWriter>  dump_writer,
                                       bool                            async):
    async_(async) {
    initialize(groups, max_batch, slots_per_device, world_rank, std::move(dump_writer));
}

void CrcTransferService::initialize(const std::vector<GroupSetPtr>& groups,
                                    size_t                          max_batch,
                                    size_t                          slots_per_device,
                                    int64_t                         world_rank,
                                    std::shared_ptr<CrcDumpWriter>  dump_writer) {
    if (!CrcBlockCopyBatch::available() || !max_batch || !slots_per_device)
        throw std::invalid_argument("CRC requires CUDA 13 and positive workspace limits");
    if (!dump_writer)
        dump_writer = CrcDumpWriter::forRank(world_rank);
    struct Geometry {
        size_t payload{0}, tiles{0};
    };
    std::map<int, Geometry> geometry;
    for (const auto& group : groups) {
        if (!group)
            throw std::invalid_argument("null CRC GroupSet");
        groups_.emplace(group.get(), group);
        if (!group->crcEnabled())
            continue;
        const auto& device_pools = group->devicePools();
        if (device_pools.empty())
            throw std::invalid_argument("CRC requires GPU-resident device pools");
        const int device = deviceFor(*group);
        if (device < 0 || std::any_of(device_pools.begin(), device_pools.end(), [device](const auto& pool) {
                return pool->deviceIndex() != device;
            }))
            throw std::invalid_argument("CRC GroupSet requires backings on a single GPU device");
        if ((group->hostPool() && group->hostPool()->strideBytes() < group->storageBytes())
            || (group->diskPool() && group->diskPool()->strideBytes() < group->storageBytes()))
            throw std::invalid_argument("CRC pool stride cannot hold payload and footer");
        auto& shape   = geometry[device];
        shape.payload = std::max(shape.payload, group->payloadBytes());
        const std::vector<BlockIdxType> blocks(device_pools.size(), 0);
        shape.tiles = std::max(shape.tiles, getDeviceHostCopyTiles(*group, blocks).size());
    }
    for (const auto& [device, shape] : geometry) {
        const auto tiles              = checkedMultiply(shape.tiles, max_batch);
        auto&      slots              = slots_[device];
        size_t     total_device_bytes = 0, total_host_bytes = 0;
        for (size_t i = 0; i < slots_per_device; ++i) {
            Slot slot;
            slot.device = device;
            slot.copy   = std::make_unique<CrcBlockCopyBatch>(device, max_batch, shape.payload, tiles, dump_writer);
            if (i == 0) {
                total_device_bytes = checkedMultiply(slots_per_device, slot.copy->deviceMemoryBytes());
                total_host_bytes   = checkedMultiply(slots_per_device, slot.copy->hostMemoryBytes());
                slots.reserve(slots_per_device);
            }
            slots.push_back(std::move(slot));
        }
        constexpr double mib = 1024.0 * 1024.0;
        RTP_LLM_LOG_INFO("BlockTree CRC32C: device=%d async=%d batch=%zu payload=%zu slots_per_device_direction=%zu "
                         "device_total=%.2f MiB host_total=%.2f MiB",
                         device,
                         async_,
                         max_batch,
                         shape.payload,
                         slots_per_device,
                         total_device_bytes / mib,
                         total_host_bytes / mib);
    }
}

CrcTransferService::~CrcTransferService() {
    drain();
}

TransferStatus CrcTransferService::statusFor(CrcCopyState status) {
    switch (status) {
        case CrcCopyState::OK:
            return TransferStatus::OK;
        case CrcCopyState::INVALID_ARGS:
            return TransferStatus::INVALID_ARGS;
        case CrcCopyState::CRC_MISMATCH:
            return TransferStatus::CRC_MISMATCH;
        default:
            return TransferStatus::DEVICE_IO_ERROR;
    }
}

TransferResult CrcTransferService::mapResult(CrcCopyState                           result,
                                             CrcCopyDirection                       direction,
                                             const std::vector<TransferDescriptor>& descriptors,
                                             const std::vector<CrcCopyFailure>&     failures) {
    TransferResult transfer_result(statusFor(result));
    transfer_result.expandBlockErrors(descriptors.size());
    if (result == CrcCopyState::CRC_MISMATCH) {
        // A failed load may have scattered every item. Keep the whole batch
        // uncommitted so callers discard its targets, including healthy items.
        std::fill(transfer_result.block_errors.begin(),
                  transfer_result.block_errors.end(),
                  direction == CrcCopyDirection::STORE ? CacheCopyError::NONE : CacheCopyError::NOT_EXECUTED);
    }
    for (const auto& failure : failures) {
        const auto& descriptor                           = descriptors[failure.item_index];
        const auto  failure_status                       = statusFor(failure.status);
        transfer_result.block_errors[failure.item_index] = cacheCopyErrorFor(failure_status);
        const auto failure_error                         = transferStatusToErrorInfo(failure_status);
        transfer_result.merge({ErrorInfo(failure_error.code(),
                                         failure_error.ToString() + ", descriptor_index="
                                             + std::to_string(failure.item_index) + ", " + descriptor.debugString()),
                               cacheCopyErrorFor(failure_status)});
    }
    return transfer_result;
}

CrcCopyItem CrcTransferService::makeItem(const HostBufferView&     host,
                                         const TransferDescriptor& descriptor,
                                         const GroupSetPtr&        group) {
    CrcCopyItem item{host.base, group->payloadBytes(), host.capacity_bytes, {}};
    for (const auto& tile : getDeviceHostCopyTiles(*group, descriptor.blocksAt(Tier::DEVICE)))
        item.tiles.push_back({tile.buffer.addr, tile.host_offset, tile.buffer.size_bytes});
    // Capture both inputs by value so diagnostics outlive the caller's vectors.
    item.dump_metadata = [descriptor, group_owner = group] {
        return CrcDumpWriter::dumpMetadata(descriptor, *group_owner);
    };
    return item;
}

bool CrcTransferService::trySubmit(const std::vector<HostBufferView>&     hosts,
                                   const std::vector<TransferDescriptor>& descriptors,
                                   const GroupSetPtr&                     group,
                                   DoneCallback                           done) {
    if (!done)
        throw std::invalid_argument("CRC stage callback is required");
    const auto reject = [&] {
        done(TransferStatus::INVALID_ARGS);
        return true;
    };
    if (unlikely(!async_) || !group || !groups_.count(group.get()) || !group->crcEnabled()
        || group->devicePools().empty() || hosts.empty() || hosts.size() != descriptors.size())
        return reject();
    const bool store     = descriptors.front().source_tier == Tier::DEVICE;
    const auto direction = store ? CrcCopyDirection::STORE : CrcCopyDirection::LOAD;
    const auto host_tier = store ? descriptors.front().target_tier : descriptors.front().source_tier;
    if (host_tier != Tier::HOST && host_tier != Tier::DISK)
        return reject();

    // Every allocation and capture precedes the first possible CUDA enqueue.
    auto operation         = std::make_unique<Operation>();
    operation->group       = group;
    operation->descriptors = descriptors;
    std::vector<CrcCopyItem> items;
    items.reserve(hosts.size());
    for (size_t i = 0; i < descriptors.size(); ++i) {
        const auto& descriptor = descriptors[i];
        if (descriptor.group_set_id != group->groupSetId() || descriptor.source_tier != descriptors.front().source_tier
            || descriptor.target_tier != descriptors.front().target_tier
            || !validCopyBuffers(hosts[i], descriptor, *group))
            return reject();
        items.push_back(makeItem(hosts[i], descriptor, group));
    }
    TransferResult invalid(TransferStatus::INVALID_ARGS);
    TransferResult failed(TransferStatus::DEVICE_IO_ERROR);
    Slot*          slot = nullptr;
    if (!accepting_)
        return reject();
    for (auto& candidate : slots_.at(deviceFor(*group)))
        if (!candidate.operation) {
            slot                  = &candidate;
            slot->operation       = std::move(operation);
            slot->operation->done = std::move(done);
            break;
        }
    if (!slot)
        return false;

    // The primitive returns only after draining any partial submission error.
    const auto                 submitted = slot->copy->trySubmit(items, direction);
    std::unique_ptr<Operation> immediate;
    if (submitted != CrcCopyState::PENDING)
        immediate = std::move(slot->operation);
    if (immediate && submitted != CrcCopyState::BUSY)
        immediate->done(submitted == CrcCopyState::INVALID_ARGS ? std::move(invalid) : std::move(failed));
    return submitted != CrcCopyState::BUSY;
}

void CrcTransferService::advance(Slot& slot, bool deliver) {
    if (!slot.operation)
        return;
    auto& operation = *slot.operation;
    if (!operation.result) {
        const auto state = slot.copy->poll();
        if (state == CrcCopyState::PENDING)
            return;
        // Construct the result before retiring the terminal workspace.
        // Allocation failures propagate to the caller.
        const auto direction = operation.descriptors.front().source_tier == Tier::DEVICE ? CrcCopyDirection::STORE :
                                                                                           CrcCopyDirection::LOAD;
        operation.result     = mapResult(state, direction, operation.descriptors, slot.copy->failures());
        slot.copy->finish();
    }
    if (!deliver)
        return;
    auto completed = std::move(slot.operation);
    completed->done(std::move(*completed->result));
}

bool CrcTransferService::poll() {
    if (unlikely(!async_))
        throw std::logic_error("poll requires an async CRC service");
    for (auto& [device, slots] : slots_)
        for (auto& slot : slots)
            advance(slot, true);
    for (const auto& [device, slots] : slots_)
        for (const auto& slot : slots)
            if (slot.operation)
                return true;
    return false;
}

void CrcTransferService::stopAdmission() {
    if (async_) {
        accepting_ = false;
        return;
    }
    // TODO(legacy removal): remove this locked legacy admission branch.
    std::lock_guard<std::mutex> lock(mutex_);
    accepting_ = false;
    cv_.notify_all();
}

void CrcTransferService::drain() {
    stopAdmission();
    // TODO(legacy removal): remove this synchronous wait branch.
    if (unlikely(!async_)) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [&] {
            for (const auto& [device, slots] : slots_)
                for (const auto& slot : slots)
                    if (slot.updating)
                        return false;
            return true;
        });
        return;
    }
    for (;;) {
        for (auto& [device, slots] : slots_)
            for (auto& slot : slots)
                advance(slot, false);
        bool pending = false;
        for (const auto& [device, slots] : slots_)
            for (const auto& slot : slots)
                pending = pending || (slot.operation && !slot.operation->result);
        if (!pending)
            return;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

}  // namespace rtp_llm
