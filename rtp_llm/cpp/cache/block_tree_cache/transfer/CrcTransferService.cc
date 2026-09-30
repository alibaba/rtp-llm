#include "rtp_llm/cpp/cache/block_tree_cache/transfer/CrcTransferService.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

#include "rtp_llm/cpp/cache/block_tree_cache/ScopeRollback.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/DeviceHostCopyGeometry.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace {
CrcDumpWriter::Metadata
dumpMetadata(int device, const char* operation, const TransferDescriptor& descriptor, const GroupSet& group) {
    const bool              store = descriptor.source_tier == Tier::DEVICE;
    CrcDumpWriter::Metadata metadata{
        {"device", std::to_string(device)},
        {"operation", operation},
        {"direction", std::string(tierName(descriptor.source_tier)) + "->" + tierName(descriptor.target_tier)},
        {"copy_item", descriptor.debugString()},
        {"group_set_id", std::to_string(descriptor.group_set_id)},
        {"kind", cacheGroupTypeName(group.groupType())},
        {"layout", group.usesPhysicalPayloadGeometry() ? "physical_pool_geometry" : "topology_geometry"},
        {"member_count", std::to_string(group.groupTags().size())},
        {"output_written", store ? "true" : "false"},
        {"cpu_snapshot_role", store ? "candidate_output_after_failure" : "restore_source_after_failure"},
        {"gpu_footer_role", store ? "candidate_output" : "checked_source"},
        {"gpu_snapshot_role", store ? "gathered_candidate_output" : "checked_source_before_scatter"},
        {"tile_copy_stage", store ? "gather_completed" : "scatter_not_started"},
        {"cpu_backing",
         descriptor.source_tier == Tier::DISK || descriptor.target_tier == Tier::DISK ? "disk_staging_buffer" :
                                                                                        "memory_pool"},
    };
    for (size_t member = 0; member < group.groupTags().size(); ++member) {
        const auto& tag                                = group.groupTags()[member];
        const auto& base                               = group.group(tag);
        const auto  prefix                             = "member_" + std::to_string(member) + "_";
        metadata[prefix + "tag"]                       = tag;
        metadata[prefix + "kind"]                      = cacheGroupTypeName(base.policy.group_type);
        metadata[prefix + "pool"]                      = group.devicePools()[member]->poolName();
        metadata[prefix + "seq_size_per_block"]        = std::to_string(base.seqSizePerBlock());
        metadata[prefix + "kernel_seq_size_per_block"] = std::to_string(base.kernelSeqSizePerBlock());
        metadata[prefix + "kv_stride_bytes"]           = std::to_string(base.kvBlockStrideBytes());
        metadata[prefix + "scale_stride_bytes"]        = std::to_string(base.kvScaleStrideBytes());
    }
    // HOST<->DISK validates staging without a device endpoint. Block zero gives
    // the same immutable geometry; its addresses are intentionally not recorded.
    const bool has_device = descriptor.source_tier == Tier::DEVICE || descriptor.target_tier == Tier::DEVICE;
    const auto blocks =
        has_device ? descriptor.blocksAt(Tier::DEVICE) : std::vector<BlockIdxType>(group.devicePools().size(), 0);
    size_t tile = 0;
    visitDeviceHostCopyTiles(group, blocks, [&](const BlockInfo& buffer, size_t offset, size_t member, size_t layer) {
        const auto  prefix            = "layout_tile_" + std::to_string(tile++) + "_";
        const auto& tag               = group.groupTags()[member];
        metadata[prefix + "member"]   = std::to_string(member);
        metadata[prefix + "tag"]      = tag;
        metadata[prefix + "layer_id"] = std::to_string(group.topologyPtr()->layerIdsForGroup(tag)[layer]);
        metadata[prefix + "offset"]   = std::to_string(offset);
        metadata[prefix + "bytes"]    = std::to_string(buffer.size_bytes);
    });
    metadata["layout_tile_count"] = std::to_string(tile);
    for (size_t i = 0; i < descriptor.source_blocks.size(); ++i)
        metadata["source_block_" + std::to_string(i)] = std::to_string(descriptor.source_blocks[i]);
    for (size_t i = 0; i < descriptor.target_blocks.size(); ++i)
        metadata["target_block_" + std::to_string(i)] = std::to_string(descriptor.target_blocks[i]);
    if (group.hostPool())
        metadata["host_pool_stride_bytes"] = std::to_string(group.hostPool()->strideBytes());
    if (group.diskPool()) {
        metadata["disk_file"]              = group.diskPool()->filePath();
        metadata["disk_pool_stride_bytes"] = std::to_string(group.diskPool()->strideBytes());
        if (descriptor.source_tier == Tier::DISK || descriptor.target_tier == Tier::DISK) {
            const auto block        = descriptor.singleBlockAt(Tier::DISK);
            metadata["disk_slot"]   = std::to_string(block);
            metadata["disk_offset"] = std::to_string(group.diskPool()->blockOffset(block));
        }
    }
    return metadata;
}
}  // namespace

int CrcTransferService::deviceFor(const GroupSet& group) {
    // The factory enables CRC only for GPU-resident backings on this rank.
    return group.devicePools().front()->deviceIndex();
}

CrcTransferService::CrcTransferService(const std::vector<GroupSetPtr>& groups,
                                       size_t                          max_batch,
                                       size_t                          workers,
                                       int64_t                         world_rank):
    dump_writer_(CrcDumpWriter::forRank(world_rank)) {
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
        const size_t                    tiles =
            visitDeviceHostCopyTiles(*group, geometry_blocks, [](const BlockInfo&, size_t, size_t, size_t) {});
        shape.tiles = std::max(shape.tiles, tiles);
    }
    for (const auto& [device, shape] : geometry) {
        if (!shape.tiles || shape.tiles > std::numeric_limits<size_t>::max() / max_batch) {
            throw std::invalid_argument("invalid CRC batch tile capacity");
        }
        auto& slots = slots_[device];
        slots.reserve(workers);
        for (size_t i = 0; i < workers; ++i) {
            slots.push_back(
                {std::make_unique<CrcBlockCopyBatch>(device, max_batch, shape.payload, shape.tiles * max_batch),
                 false});
        }
        RTP_LLM_LOG_INFO("BlockTree CRC32C enabled: device=%d workspaces=%zu batch=%zu max_payload=%zu max_tiles=%zu",
                         device,
                         workers,
                         max_batch,
                         shape.payload,
                         shape.tiles * max_batch);
    }
}

TransferStatus CrcTransferService::statusFor(CrcCopyStatus status) {
    switch (status) {
        case CrcCopyStatus::OK:
            return TransferStatus::OK;
        case CrcCopyStatus::INVALID_ARGS:
            return TransferStatus::INVALID_ARGS;
        case CrcCopyStatus::CRC_MISMATCH:
            return TransferStatus::CRC_MISMATCH;
        default:
            return TransferStatus::DEVICE_IO_ERROR;
    }
}

TransferStatus CrcTransferService::execute(int                                    device,
                                           const std::vector<CrcCopyItem>&        items,
                                           Operation                              operation,
                                           const std::vector<TransferDescriptor>& descriptors,
                                           const std::vector<const GroupSet*>&    groups) {
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
    block_tree_cache_detail::ScopeRollback                   release([&] {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            slot->busy = false;
        }
        // Waiters share this condition variable but need slots on different devices.
        // Wake every waiter so a free slot cannot be stranded by waking another device.
        cv_.notify_all();
    });
    std::vector<CrcCopyFailure>                              failures;
    std::vector<std::unique_ptr<CrcDumpWriter::Reservation>> reservations(items.size());
    const CrcCopyCapture                                     capture = [&](size_t index) {
        reservations[index] = dump_writer_->reserve(CrcBlockCopyBatch::encodedBytes(items[index].payload_bytes));
        return reservations[index] != nullptr;
    };
    const auto  result         = operation == Operation::STORE ? slot->copy->store(items, &failures, capture) :
                                 operation == Operation::LOAD  ? slot->copy->load(items, &failures, capture) :
                                                                 slot->copy->validate(items, &failures, capture);
    const char* operation_name = operation == Operation::STORE ? "store" :
                                 operation == Operation::LOAD  ? "load" :
                                                                 "validate";
    for (auto& failure : failures) {
        const auto& descriptor = descriptors[failure.item_index];
        if (failure.status == CrcCopyStatus::CRC_MISMATCH)
            descriptor.markCorrupted();
        else if (failure.status == CrcCopyStatus::CRC_COMPUTE_ERROR)
            descriptor.markCrcComputeFailed();
        // Mark diagnostics before allocating metadata or doing I/O: a failed
        // dump must never prevent invalidation/fallback for the original verdict.
        try {
            auto& reservation = reservations[failure.item_index];
            if (reservation) {
                failure.dump_path =
                    dump_writer_->write(*reservation,
                                        items[failure.item_index],
                                        failure,
                                        dumpMetadata(device, operation_name, descriptor, *groups[failure.item_index]));
                reservation.reset();
            }
        } catch (const std::exception& error) {
            RTP_LLM_LOG_WARNING("CRC diagnostic metadata failed: %s", error.what());
        } catch (...) {
            RTP_LLM_LOG_WARNING("CRC diagnostic metadata failed");
        }
        RTP_LLM_LOG_WARNING("CRC backing failure: %s status=%d dump=%s expected=%08x actual=%08x",
                            descriptor.debugString().c_str(),
                            static_cast<int>(failure.status),
                            failure.dump_path.c_str(),
                            failure.expected,
                            failure.actual);
    }
    if (result != CrcCopyStatus::OK) {
        RTP_LLM_LOG_WARNING("BlockTree CRC transfer failed: device=%d operation=%d items=%zu status=%d",
                            device,
                            static_cast<int>(operation),
                            items.size(),
                            static_cast<int>(result));
    }
    if (result == CrcCopyStatus::OK) {
        const unsigned bit = 1u << static_cast<unsigned>(operation);
        if (!(logged_operations_.fetch_or(bit, std::memory_order_relaxed) & bit)) {
            const char* name = operation == Operation::STORE ? "store" :
                               operation == Operation::LOAD  ? "load" :
                                                               "validate";
            RTP_LLM_LOG_INFO(
                "BlockTree CRC32C completed: operation=%s device=%d backing_count=%zu", name, device, items.size());
        }
    }
    return statusFor(result);
}

TransferStatus CrcTransferService::copy(const std::vector<HostBufferView>&     hosts,
                                        const std::vector<TransferDescriptor>& descriptors,
                                        const std::vector<const GroupSet*>&    groups) {
    try {
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
            visitDeviceHostCopyTiles(group, blocks, [&](const BlockInfo& buffer, size_t offset, size_t, size_t) {
                item.tiles.push_back({buffer.addr, offset, buffer.size_bytes});
            });
            items.push_back(std::move(item));
        }
        return execute(
            deviceFor(*groups.front()), items, store ? Operation::STORE : Operation::LOAD, descriptors, groups);
    } catch (const std::exception& error) {
        RTP_LLM_LOG_WARNING("CRC transfer exception: %s", error.what());
        return TransferStatus::DEVICE_IO_ERROR;
    }
}

TransferStatus CrcTransferService::validate(const std::vector<HostBufferView>&     hosts,
                                            const std::vector<TransferDescriptor>& descriptors,
                                            const std::vector<const GroupSet*>&    groups) {
    try {
        if (hosts.empty() || hosts.size() != groups.size() || hosts.size() != descriptors.size()) {
            return TransferStatus::INVALID_ARGS;
        }
        std::vector<CrcCopyItem> items;
        items.reserve(hosts.size());
        for (size_t i = 0; i < hosts.size(); ++i) {
            const auto& group = *groups[i];
            if (!group.crcEnabled() || !isValidHostBufferView(hosts[i], group.payloadBytes(), group.storageBytes())) {
                return TransferStatus::INVALID_ARGS;
            }
            items.push_back({hosts[i].base, group.payloadBytes(), hosts[i].capacity_bytes, {}});
        }
        return execute(deviceFor(*groups.front()), items, Operation::VALIDATE, descriptors, groups);
    } catch (const std::exception& error) {
        RTP_LLM_LOG_WARNING("CRC validation exception: %s", error.what());
        return TransferStatus::DEVICE_IO_ERROR;
    }
}

}  // namespace rtp_llm
