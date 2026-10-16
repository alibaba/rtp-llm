#include "rtp_llm/cpp/cache/block_tree_cache/transfer/CrcTransferService.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

#include "rtp_llm/cpp/cache/block_tree_cache/ScopeRollback.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/DeviceHostCopyGeometry.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

int CrcTransferService::deviceFor(const GroupSet& group) {
    // The factory enables CRC only for GPU-resident backings on this rank.
    return group.devicePools().front()->deviceIndex();
}

CrcTransferService::CrcTransferService(const std::vector<GroupSetPtr>& groups,
                                       size_t                          max_batch,
                                       size_t                          workers,
                                       int64_t                         world_rank,
                                       std::shared_ptr<CrcDumpWriter>  dump_writer) {
    if (!dump_writer)
        dump_writer = CrcDumpWriter::forRank(world_rank);
    if (!CrcBlockCopyBatch::available() || !max_batch || !workers) {
        throw std::invalid_argument("CRC requires CUDA 13 and nonempty workspace limits");
    }
    struct Geometry {
        size_t payload{0};
        size_t tiles{0};
    };
    std::map<int, Geometry> geometry;
    for (const auto& group : groups) {
        if (!group->crcEnabled()) {
            continue;
        }
        if ((group->hostPool() && group->hostPool()->strideBytes() < group->storageBytes())
            || (group->diskPool() && group->diskPool()->strideBytes() < group->storageBytes())) {
            throw std::invalid_argument("CRC pool stride cannot hold payload and footer");
        }
        auto& shape   = geometry[deviceFor(*group)];
        shape.payload = std::max(shape.payload, group->payloadBytes());
        // Block zero is reserved but physically present in every pool. Inspect
        // its layout without allocating or reading payload bytes.
        const std::vector<BlockIdxType> geometry_blocks(group->devicePools().size(), 0);
        const size_t                    tiles = getDeviceHostCopyTiles(*group, geometry_blocks).size();
        shape.tiles                           = std::max(shape.tiles, tiles);
    }
    for (const auto& [device, shape] : geometry) {
        if (!shape.tiles || shape.tiles > std::numeric_limits<size_t>::max() / max_batch) {
            throw std::invalid_argument("invalid CRC batch tile capacity");
        }
        auto& slots = slots_[device];
        slots.reserve(workers);
        for (size_t i = 0; i < workers; ++i) {
            slots.push_back({std::make_unique<CrcBlockCopyBatch>(
                                 device, max_batch, shape.payload, shape.tiles * max_batch, dump_writer),
                             false});
        }
        const size_t     slot_device_bytes  = slots.front().copy->deviceMemoryBytes();
        const size_t     total_device_bytes = slots.size() * slot_device_bytes;
        constexpr double bytes_per_mib      = 1024.0 * 1024.0;
        RTP_LLM_LOG_INFO("BlockTree CRC32C enabled: device=%d batch=%zu max_payload=%zu max_tiles=%zu "
                         "[slot_count=%zu] [slot_device_bytes=%zu] [total_device_bytes=%zu] "
                         "[slot_device_mib=%.3f] [total_device_mib=%.3f]",
                         device,
                         max_batch,
                         shape.payload,
                         shape.tiles * max_batch,
                         slots.size(),
                         slot_device_bytes,
                         total_device_bytes,
                         slot_device_bytes / bytes_per_mib,
                         total_device_bytes / bytes_per_mib);
    }
}

TransferStatus CrcTransferService::statusFor(CrcCopyStatus status) {
    switch (status) {
        case CrcCopyStatus::OK:
            return TransferStatus::OK;
        case CrcCopyStatus::INVALID_ARGS:
            return TransferStatus::INVALID_ARGS;
        case CrcCopyStatus::CRC_COMPUTE_ERROR:
            return TransferStatus::CRC_COMPUTE_ERROR;
        case CrcCopyStatus::CRC_MISMATCH:
            return TransferStatus::CRC_MISMATCH;
        default:
            return TransferStatus::DEVICE_IO_ERROR;
    }
}

TransferResult CrcTransferService::execute(int                                    device,
                                           const std::vector<CrcCopyItem>&        items,
                                           Operation                              operation,
                                           const std::vector<TransferDescriptor>& descriptors) {
    Slot* slot = nullptr;
    {
        std::unique_lock<std::mutex> lock(mutex_);
        const auto                   found = slots_.find(device);
        if (found == slots_.end()) {
            return TransferStatus::INVALID_ARGS;
        }
        cv_.wait(lock, [&] {
            for (auto& candidate : found->second) {
                if (!candidate.busy) {
                    candidate.busy = true;
                    slot           = &candidate;
                    return true;
                }
            }
            return false;
        });
    }
    block_tree_cache_detail::ScopeRollback release([&] {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            slot->busy = false;
        }
        // Waiters share this condition variable but need slots on different devices.
        // Wake every waiter so a free slot cannot be stranded by waking another device.
        cv_.notify_all();
    });
    std::vector<CrcCopyFailure>            failures;
    const auto                             result =
        operation == Operation::STORE ? slot->copy->store(items, &failures) : slot->copy->load(items, &failures);
    TransferResult transfer_result(statusFor(result));
    transfer_result.expandBlockErrors(items.size());
    if (result == CrcCopyStatus::CRC_MISMATCH || result == CrcCopyStatus::CRC_COMPUTE_ERROR) {
        // A failed load may have scattered every item. Keep the whole batch
        // uncommitted so callers discard its targets, including healthy items.
        std::fill(transfer_result.block_errors.begin(),
                  transfer_result.block_errors.end(),
                  operation == Operation::STORE ? CacheCopyError::NONE : CacheCopyError::NOT_EXECUTED);
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

TransferResult CrcTransferService::copy(const std::vector<HostBufferView>&     hosts,
                                        const std::vector<TransferDescriptor>& descriptors,
                                        const std::vector<const GroupSet*>&    groups) {
    if (hosts.empty() || hosts.size() != groups.size() || hosts.size() != descriptors.size()) {
        return TransferStatus::INVALID_ARGS;
    }
    const bool               store = descriptors.front().source_tier == Tier::DEVICE;
    std::vector<CrcCopyItem> items;
    items.reserve(hosts.size());
    for (size_t i = 0; i < hosts.size(); ++i) {
        const auto& group = *groups[i];
        if (!group.crcEnabled() || (descriptors[i].source_tier == Tier::DEVICE) != store
            || !isValidHostBufferView(hosts[i], group.payloadBytes(), group.storageBytes())) {
            return TransferStatus::INVALID_ARGS;
        }
        const auto& blocks = descriptors[i].blocksAt(Tier::DEVICE);
        if (blocks.size() != group.devicePools().size()) {
            return TransferStatus::INVALID_ARGS;
        }
        // RPC workers use the coordinator's block indices without mirroring
        // its allocation bitmap. Validate the physical index on each rank;
        // the transfer task retains the coordinator's allocation lifetime.
        for (size_t member = 0; member < blocks.size(); ++member) {
            if (!group.devicePools()[member]->validBlock(blocks[member])) {
                RTP_LLM_LOG_WARNING("CRC transfer has invalid device block: group=%zu member=%zu block=%d",
                                    descriptors[i].group_set_id,
                                    member,
                                    blocks[member]);
                return TransferStatus::INVALID_ARGS;
            }
        }
        CrcCopyItem item{hosts[i].base, group.payloadBytes(), hosts[i].capacity_bytes, {}};
        for (const auto& tile : getDeviceHostCopyTiles(group, blocks)) {
            item.tiles.push_back({tile.buffer.addr, tile.host_offset, tile.buffer.size_bytes});
        }
        item.dump_metadata = [&, i] { return CrcDumpWriter::dumpMetadata(descriptors[i], *groups[i]); };
        items.push_back(std::move(item));
    }
    return execute(deviceFor(*groups.front()), items, store ? Operation::STORE : Operation::LOAD, descriptors);
}

}  // namespace rtp_llm
