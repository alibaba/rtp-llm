#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferWorkLoop.h"

#include <cassert>
#include <limits>
#include <utility>

#include <c10/core/DeviceGuard.h>
#include <torch/torch.h>
#if defined(USING_ROCM) && USING_ROCM
#include <c10/core/impl/VirtualGuardImpl.h>
#endif

#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/HostBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/HostStagingBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/metrics/BlockTreeCacheMetricsReporter.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/CrcCopyExecutor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/DeviceHostCopyGeometry.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/DiskTransferUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/models_py/bindings/NoBlockCopy.h"

namespace rtp_llm {

namespace {

TransferResult unsubmittedResult(TransferResult result) {
    result.buffer_safe = true;
    return result;
}

}  // namespace

TransferWorkLoop::TransferWorkLoop(std::vector<GroupSetPtr> groups,
                                   std::unique_ptr<CrcCopyExecutor> crc,
                                   DiskIOExecutor* disk_io_executor,
                                   TransferOperation operation,
                                   size_t batch_limit,
                                   size_t queue_capacity,
                                   std::shared_ptr<HostStagingBlockPool> staging,
                                   std::shared_ptr<BlockTreeCacheMetricsReporter> metrics_reporter,
                                   int64_t world_rank,
                                   size_t worker_index,
                                   std::shared_ptr<HostStagingBlockPool> swa_staging):
    TransferWorkLoop(operation, batch_limit, queue_capacity, std::move(metrics_reporter), world_rank, worker_index, {}) {
    static const RtpIo io{
        &TransferWorkLoop::acquireRtpStaging,
        [](const GroupSet& group, BlockIdxType block) -> HostBufferView {
            const auto buffer = group.hostPool()->blockBuffer(block);
            return {buffer.addr, buffer.payload_bytes, buffer.stride_bytes};
        },
        [](const GroupSet& group) { return group.devicePools().front()->deviceIndex(); },
        &TransferWorkLoop::submitRtpCrc,
        &TransferWorkLoop::pollRtpCrc,
        &TransferWorkLoop::freeRtpCrcSlots,
        &TransferWorkLoop::stopRtpCrc,
        &TransferWorkLoop::submitPlainDeviceCopy,
        &TransferWorkLoop::submitDisk,
    };
    groups_ = std::move(groups);
    crc_ = std::move(crc);
    disk_io_executor_ = disk_io_executor;
    rtp_io_ = &io;
    abortOnOom([&] {
        if (operation_ == TransferOperation::EVICT && (staging || swa_staging))
            throw std::invalid_argument("evict WorkLoop must not own staging pools");
        // A single supplied pool remains useful for standalone load owners.
        staging_.pools = {staging, swa_staging ? std::move(swa_staging) : std::move(staging)};
        for (const auto& group : groups_) {
            if (!group->devicePools().empty())
                crc_waiting_.try_emplace(io.device_index(*group));
        }
    });
}

bool TransferWorkLoop::acquireRtpStaging(Entry& entry) {
    auto leases = staging_.pools[stagingIndex(entry)]->tryMallocBatch(entry.task.descriptors().size());
    if (!leases)
        return false;
    entry.staging = std::move(*leases);
    entry.staging_timer.finish(true);
    for (const auto& lease : entry.staging)
        entry.hosts.push_back(lease.blockBuffer(group(entry)->payloadBytes()));
    return true;
}

bool TransferWorkLoop::submitRtpCrc(Entry& entry,
                                   TransferAsyncContext::DoneCallback done,
                                   TransferCrcTiming* timing) {
    return crc_->trySubmit(entry.hosts, entry.task.descriptors(), group(entry), std::move(done), timing);
}

bool TransferWorkLoop::pollRtpCrc() {
    return crc_->poll();
}

size_t TransferWorkLoop::freeRtpCrcSlots(int device) const {
    return crc_->availableSlots(device);
}

void TransferWorkLoop::stopRtpCrc() {
    crc_->stopAndWaitForIo();
}

void TransferWorkLoop::publishDisk(Entry& entry, BlockIOStatus status) noexcept {
    recordDiskResult(entry, status);
    publishCompletion(entry);
}

void TransferWorkLoop::submitPlainDeviceCopy(Entry& entry) {
    const auto& descriptors = entry.task.descriptors();
    const Tier  source      = descriptors.front().source_tier;
    const Tier  target      = descriptors.front().target_tier;
    const bool  to_device   = target == Tier::DEVICE;
    const bool  valid_route =
        to_device ? (source == Tier::HOST || source == Tier::DISK) : (source == Tier::DEVICE && target == Tier::HOST);
    const auto& owner       = *group(entry);
    const auto  reject      = [&] { fail(entry, unsubmittedResult(TransferStatus::INVALID_ARGS)); };
    if (!valid_route || !isPlainDeviceTransferGroup(owner) || entry.hosts.size() != descriptors.size()) {
        reject();
        return;
    }

    const auto             device_index = owner.devicePools().front()->deviceIndex();
    const auto             device       = torch::Device(torch::kCUDA, static_cast<c10::DeviceIndex>(device_index));
    const c10::DeviceGuard device_guard(device);
    const auto             bytes_on_host   = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
    const auto             bytes_on_device = torch::TensorOptions().dtype(torch::kUInt8).device(device);
    MultiCopyParams copy;
    startStepTimer(entry, to_device ? "h2d" : "d2h");
    // Build and validate every view before enqueuing any device access.
    for (size_t i = 0; i < descriptors.size(); ++i) {
        const auto& descriptor = descriptors[i];
        const auto& host       = entry.hosts[i];
        if (descriptor.source_tier != source || descriptor.target_tier != target
            || !isValidHostBufferView(host, owner.payloadBytes(), owner.payloadBytes())) {
            reject();
            return;
        }
        for (const auto& tile : getDeviceHostCopyTiles(owner, descriptor.blocksAt(Tier::DEVICE))) {
            const auto& pool    = *owner.devicePools()[tile.member_group_id];
            const auto  base    = reinterpret_cast<uintptr_t>(pool.getBaseAddress());
            const auto  address = reinterpret_cast<uintptr_t>(tile.buffer.addr);
            const auto  size    = pool.getTotalSizeBytes();
            if (!tile.buffer.is_cuda || tile.buffer.device_index != device_index
                || tile.buffer.size_bytes > static_cast<size_t>(std::numeric_limits<int64_t>::max())
                || address < base || address - base > size || tile.buffer.size_bytes > size - (address - base)) {
                reject();
                return;
            }
            const auto shape       = static_cast<int64_t>(tile.buffer.size_bytes);
            auto       host_view   = torch::from_blob(
                static_cast<uint8_t*>(host.base) + tile.host_offset, {shape}, bytes_on_host);
            auto       device_view = torch::from_blob(tile.buffer.addr, {shape}, bytes_on_device);
            copy.multi_src.push_back(to_device ? std::move(host_view) : std::move(device_view));
            copy.multi_dst.push_back(to_device ? std::move(device_view) : std::move(host_view));
        }
    }
    if (entry.task.expired()) {
        fail(entry,
             unsubmittedResult(
                 TransferResult(ErrorInfo(ErrorCode::DEADLINE_EXCEEDED, "transfer expired before stage submission"),
                                CacheCopyError::NOT_EXECUTED)));
        return;
    }

    // CUDA and PPU drain their copy streams on failure. The ROCm implementation
    // uses blocking tensor copies on the current stream. If a copy throws,
    // require a confirmed drain before withEntry can release the backing.
#if defined(USING_ROCM) && USING_ROCM
    const auto stream = c10::impl::VirtualGuardImpl(device.type()).getStream(device);
    try {
        execNoBlockCopy(copy);
    } catch (...) {
        try {
            stream.synchronize();
        } catch (...) {
            RTP_LLM_LOG_ERROR("could not drain failed plain device copy on device %d", device_index);
            Logger::getEngineLogger().flush();
            std::abort();
        }
        throw;
    }
#else
    execNoBlockCopy(copy);
#endif
    TransferResult result(TransferStatus::OK);
    result.buffer_safe = true;
    recordResult(entry, std::move(result));
    completed_.push(entry);
}

void TransferWorkLoop::submitDisk(Entry& entry) {
    startStepTimer(entry, entry.step == Step::DISK_READ ? "disk_read" : "disk_write");
    auto*                    pool   = group(entry)->diskPool().get();
    const bool               write  = entry.step == Step::DISK_WRITE;
    const auto               stride = pool->strideBytes();
    BlockIdList              blocks;
    std::vector<void*>       buffers;
    std::vector<const void*> write_buffers;
    const auto&              descriptors = entry.task.descriptors();
    blocks.reserve(descriptors.size());
    buffers.reserve(descriptors.size());
    write_buffers.reserve(descriptors.size());
    for (size_t i = 0; i < descriptors.size(); ++i) {
        const auto& host  = entry.hosts[i];
        const auto& owner = *group(entry);
        if (owner.storageBytes() > stride || !isValidHostBufferView(host, owner.payloadBytes(), stride)) {
            fail(entry, unsubmittedResult(TransferStatus::DISK_IO_ERROR));
            return;
        }
        blocks.push_back(descriptors[i].singleBlockAt(Tier::DISK));
        if (write)
            write_buffers.push_back(host.base);
        else
            buffers.push_back(host.base);
    }
    DiskBlockPool::Callback done = [this, &entry](BlockIOStatus status) { publishDisk(entry, status); };
    // Ordinary rejection has no I/O/callback. Accepted workers own completion.
    // Submission exceptions propagate to withEntry; it handles OOM locally.
    assert(disk_io_executor_ != nullptr);
    const auto status = write ? pool->asyncWrite(*disk_io_executor_, blocks, write_buffers, stride, std::move(done)) :
                                pool->asyncRead(*disk_io_executor_, blocks, buffers, stride, std::move(done));
    if (status != BlockIOStatus::OK) {
        recordDiskResult(entry, status);
        completed_.push(entry);
    }
}

void TransferWorkLoop::recordDiskResult(Entry& entry, BlockIOStatus status) noexcept {
    assert(entry.result.ok());
    // Only called after a zero-I/O rejection or the final DiskIOExecutor callback,
    // which joins every accepted worker even on partial failure or exception.
    // This stage contributes true to the AND; preserve the earlier stages' proof.
    const auto transfer_status = blockIOStatusToTransferStatus(status);
    const auto copy_error      = cacheCopyErrorFor(transfer_status);
    const bool intermediate    = transfer_status == TransferStatus::OK && entry.step == Step::DISK_READ
                              && entry.task.descriptors().front().target_tier == Tier::DEVICE;
    if (!intermediate)
        std::fill(entry.result.block_errors.begin(), entry.result.block_errors.end(), copy_error);
    // Write only existing storage. Error strings are constructed by the owner
    // after taking the completion queue. OOM there exits the process without retry.
    entry.result.copy_error = copy_error;
    entry.result.status.setErrorCode(transfer_status == TransferStatus::OK           ? ErrorCode::NONE_ERROR :
                                     transfer_status == TransferStatus::INVALID_ARGS ? ErrorCode::INVALID_PARAMS :
                                                                                       ErrorCode::EXECUTION_EXCEPTION);
}

}  // namespace rtp_llm
