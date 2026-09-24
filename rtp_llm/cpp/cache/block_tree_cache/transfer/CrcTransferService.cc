#include "rtp_llm/cpp/cache/block_tree_cache/transfer/CrcTransferService.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

#include "rtp_llm/cpp/cache/block_tree_cache/ScopeRollback.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

int CrcTransferService::deviceFor(const GroupSet& group) {
    int device = -1;
    for (const auto& pool : group.devicePools()) {
        if (!pool || pool->where() != MemoryType::MEMORY_GPU || pool->deviceIndex() < 0
            || (device >= 0 && device != pool->deviceIndex())) {
            throw std::invalid_argument("CRC backing must reside entirely on one CUDA device");
        }
        device = pool->deviceIndex();
    }
    if (device < 0) {
        throw std::invalid_argument("CRC backing has no CUDA pool");
    }
    return device;
}

CrcTransferService::CrcTransferService(const std::vector<GroupSetPtr>& groups, size_t max_batch, size_t workers) {
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
        size_t tiles  = 0;
        for (size_t member = 0; member < group->groupIds().size(); ++member) {
            const auto&  base      = group->groupAt(member);
            const size_t per_layer = (base.kv_block_stride_bytes > 0) + (base.kv_scale_stride_bytes > 0);
            if (per_layer && base.layer_ids.size() > (std::numeric_limits<size_t>::max() - tiles) / per_layer) {
                throw std::overflow_error("CRC tile count overflow");
            }
            tiles += base.layer_ids.size() * per_layer;
        }
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
    // Every protected-path failure is terminal for consumers. Never allow a failed
    // protected load to enter PREFILL's generic transfer-error fallback.
    return status == CrcCopyStatus::OK ? TransferStatus::OK : TransferStatus::CACHE_INTEGRITY_ERROR;
}

TransferStatus CrcTransferService::execute(int device, const std::vector<CrcCopyItem>& items, Operation operation) {
    Slot* slot = nullptr;
    {
        std::unique_lock<std::mutex> lock(mutex_);
        const auto                   found = slots_.find(device);
        if (found == slots_.end()) {
            return TransferStatus::CACHE_INTEGRITY_ERROR;
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
    const auto                             result = operation == Operation::STORE ? slot->copy->store(items) :
                                                    operation == Operation::LOAD  ? slot->copy->load(items) :
                                                                                    slot->copy->validate(items);
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
            return TransferStatus::CACHE_INTEGRITY_ERROR;
        }
        const bool                              store = descriptors.front().source_tier == Tier::DEVICE;
        std::map<int, std::vector<CrcCopyItem>> batches;
        for (size_t i = 0; i < hosts.size(); ++i) {
            const auto& group = *groups[i];
            if (!group.crcEnabled() || (descriptors[i].source_tier == Tier::DEVICE) != store
                || !isValidHostBufferView(hosts[i], group.payloadBytes(), group.storageBytes())) {
                return TransferStatus::CACHE_INTEGRITY_ERROR;
            }
            const int   device = deviceFor(group);
            const auto& blocks = descriptors[i].blocksAt(Tier::DEVICE);
            if (blocks.size() != group.devicePools().size()) {
                return TransferStatus::CACHE_INTEGRITY_ERROR;
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
                    return TransferStatus::CACHE_INTEGRITY_ERROR;
                }
            }
            CrcCopyItem item{hosts[i].base, group.payloadBytes(), hosts[i].capacity_bytes, {}};
            size_t      offset = 0;
            for (size_t member = 0; member < group.groupIds().size(); ++member) {
                const auto& base = group.groupAt(member);
                for (size_t layer = 0; layer < base.layer_ids.size(); ++layer) {
                    const auto   buffers   = group.devicePools()[member]->convertIndexToBuffer(layer, blocks[member]);
                    const size_t lengths[] = {base.kv_block_stride_bytes, base.kv_scale_stride_bytes};
                    for (size_t k = 0; k < 2; ++k) {
                        if (!lengths[k]) {
                            continue;
                        }
                        if (k >= buffers.size() || !buffers[k].addr || buffers[k].size_bytes < lengths[k]
                            || offset > item.payload_bytes || lengths[k] > item.payload_bytes - offset) {
                            return TransferStatus::CACHE_INTEGRITY_ERROR;
                        }
                        item.tiles.push_back({buffers[k].addr, offset, lengths[k]});
                        offset += lengths[k];
                    }
                }
            }
            if (offset != item.payload_bytes) {
                return TransferStatus::CACHE_INTEGRITY_ERROR;
            }
            batches[device].push_back(std::move(item));
        }
        for (const auto& [device, items] : batches) {
            const auto result = execute(device, items, store ? Operation::STORE : Operation::LOAD);
            if (result != TransferStatus::OK) {
                return result;
            }
        }
        return TransferStatus::OK;
    } catch (const std::exception& error) {
        RTP_LLM_LOG_WARNING("CRC transfer exception: %s", error.what());
        return TransferStatus::CACHE_INTEGRITY_ERROR;
    }
}

TransferStatus CrcTransferService::validate(const std::vector<HostBufferView>&  hosts,
                                            const std::vector<const GroupSet*>& groups) {
    try {
        if (hosts.empty() || hosts.size() != groups.size()) {
            return TransferStatus::CACHE_INTEGRITY_ERROR;
        }
        std::map<int, std::vector<CrcCopyItem>> batches;
        for (size_t i = 0; i < hosts.size(); ++i) {
            const auto& group = *groups[i];
            if (!group.crcEnabled() || !isValidHostBufferView(hosts[i], group.payloadBytes(), group.storageBytes())) {
                return TransferStatus::CACHE_INTEGRITY_ERROR;
            }
            batches[deviceFor(group)].push_back({hosts[i].base, group.payloadBytes(), hosts[i].capacity_bytes, {}});
        }
        for (const auto& [device, items] : batches) {
            const auto result = execute(device, items, Operation::VALIDATE);
            if (result != TransferStatus::OK) {
                return result;
            }
        }
        return TransferStatus::OK;
    } catch (const std::exception& error) {
        RTP_LLM_LOG_WARNING("CRC validation exception: %s", error.what());
        return TransferStatus::CACHE_INTEGRITY_ERROR;
    }
}

}  // namespace rtp_llm
