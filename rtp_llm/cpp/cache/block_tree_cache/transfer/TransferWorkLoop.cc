#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferWorkLoop.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <limits>
#include <pthread.h>
#include <stdexcept>
#include <utility>

#include <c10/core/DeviceGuard.h>
#if defined(USING_ROCM) && USING_ROCM
#include <c10/core/impl/VirtualGuardImpl.h>
#endif

#include "rtp_llm/cpp/cache/block_tree_cache/ScopeRollback.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/DeviceHostCopyGeometry.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/DiskTransferUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/models_py/bindings/NoBlockCopy.h"

namespace rtp_llm {

namespace {

TransferResult unsubmittedResult(TransferResult result) {
    // This stage has not submitted I/O. recordResult still ANDs any earlier
    // stage's proof, including when a successor expires while waiting.
    result.buffer_safe = true;
    return result;
}

}  // namespace

void TransferWorkLoop::Queue::push(Entry& entry) {
    assert(entry.next == nullptr && entry.previous == nullptr);
    entry.previous = tail;
    if (tail)
        tail->next = &entry;
    else
        head = &entry;
    tail = &entry;
}

TransferWorkLoop::Entry* TransferWorkLoop::Queue::pop() {
    auto* entry = head;
    if (entry)
        remove(*entry);
    return entry;
}

void TransferWorkLoop::Queue::remove(Entry& entry) {
    if (entry.previous)
        entry.previous->next = entry.next;
    else {
        assert(head == &entry);
        head = entry.next;
    }
    if (entry.next)
        entry.next->previous = entry.previous;
    else {
        assert(tail == &entry);
        tail = entry.previous;
    }
    entry.next = entry.previous = nullptr;
}

TransferWorkLoop::Queue TransferWorkLoop::Queue::take() {
    auto result = *this;
    head = tail = nullptr;
    return result;
}

TransferWorkLoop::Entry::Entry(TransferTask task, BeforeCompleteCallback before_complete):
    task(std::move(task)),
    context(std::make_shared<TransferBatchAsyncContext>(
        this->task.descriptors().size(), nullptr, this->task.completionProof())),
    result(TransferResult::emptyAggregate(this->task.descriptors().size())),
    before_complete(std::move(before_complete)) {}

bool TransferWorkLoop::isDeviceCopy(Step step) {
    return step == Step::CRC_LOAD || step == Step::CRC_EVICT || step == Step::PLAIN_DEVICE_COPY;
}

TransferWorkLoop::TransferWorkLoop(std::vector<GroupSetPtr>                       groups,
                                   std::unique_ptr<CrcCopyExecutor>               crc,
                                   DiskIOExecutor*                                disk_io_executor,
                                   TransferOperation                              operation,
                                   size_t                                         batch_limit,
                                   size_t                                         queue_capacity,
                                   std::shared_ptr<HostStagingBlockPool>          staging,
                                   std::shared_ptr<BlockTreeCacheMetricsReporter> metrics_reporter,
                                   int64_t                                        world_rank,
                                   size_t                                         worker_index,
                                   std::shared_ptr<HostStagingBlockPool>          swa_staging):
    groups_(std::move(groups)),
    crc_(std::move(crc)),
    disk_io_executor_(disk_io_executor),
    operation_(operation),
    batch_limit_(batch_limit),
    capacity_(queue_capacity),
    metrics_reporter_(std::move(metrics_reporter)),
    world_rank_(world_rank),
    thread_name_((operation == TransferOperation::LOAD ? "btc_load_" : "btc_evict_") + std::to_string(worker_index)) {
    abortOnOom([&] {
        if (!batch_limit_ || !capacity_)
            throw std::invalid_argument("async transfer requires positive M/Q");
        if (capacity_ > std::numeric_limits<size_t>::max() / batch_limit_)
            throw std::overflow_error("async transfer descriptor capacity overflow");
        if (operation_ == TransferOperation::EVICT && (staging || swa_staging))
            throw std::invalid_argument("evict WorkLoop must not own staging pools");
        // A single supplied pool remains useful for standalone load owners.
        staging_.pools = {staging, swa_staging ? std::move(swa_staging) : std::move(staging)};
        for (const auto& group : groups_) {
            if (!group->devicePools().empty())
                crc_waiting_.try_emplace(group->devicePools().front()->deviceIndex());
        }
    });
}

TransferWorkLoop::~TransferWorkLoop() {
    shutdown();
}

void TransferWorkLoop::waitForIdle() {
    assert(thread_.get_id() != std::this_thread::get_id());
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] { return entry_count_ == 0; });
}

void TransferWorkLoop::start() {
    abortOnOom([&] {
        thread_ = std::thread([this] {
            const auto error = pthread_setname_np(pthread_self(), thread_name_.c_str());
            if (error)
                RTP_LLM_LOG_WARNING("cannot name transfer owner %s: %s", thread_name_.c_str(), std::strerror(error));
            RTP_LLM_LOG_INFO("BlockTree transfer owner: name=%s rank=%ld", thread_name_.c_str(), world_rank_);
            run();
        });
    });
}

std::shared_ptr<TransferAsyncContext>
TransferWorkLoop::reject(const TransferTask& task, const char* message, ErrorCode code) {
    const auto count = task.descriptors().size();
    auto       total = timer(task, "worker_task_total", 0, count, AsyncProfileKind::Lifecycle, "worker_task");
    auto       queue = timer(
        task, "transfer_queue_wait", 0, count, AsyncProfileKind::Wait, "worker_task", nullptr, total.startSteadyUs());
    const auto end_us = (total || queue) ? AsyncProfileSession::steadyNowUs() : -1;
    queue.finishAtSteady(end_us, false);
    total.finishAtSteady(end_us, false);
    return std::make_shared<CompletedTransferAsyncContext>(
        unsubmittedResult(TransferResult(ErrorInfo(code, message), CacheCopyError::NOT_EXECUTED)),
        count,
        task.completionProof());
}

std::shared_ptr<TransferAsyncContext>
TransferWorkLoop::execute(TransferTask&& task, BeforeCompleteCallback&& before_complete, bool* retryable) {
    if (retryable) {
        *retryable = false;
    }
    block_tree_cache_detail::ScopeRollback unsubmitted([proof = task.completionProof()] {
        if (proof)
            proof->report(true);
    });
    return abortOnOom([&]() -> std::shared_ptr<TransferAsyncContext> {
        const auto count = task.descriptors().size();
        if (!count)
            return reject(task, "empty WorkLoop transfer", ErrorCode::INVALID_PARAMS);
        if (task.expired())
            return reject(task, "transfer expired before WorkLoop admission", ErrorCode::DEADLINE_EXCEEDED);
        std::unique_lock<std::mutex> lock(mutex_);
        if (!accepting_) {
            lock.unlock();
            return reject(task, "async transfer owner stopped", ErrorCode::EXECUTION_EXCEPTION);
        }
        if (entry_count_ == capacity_) {
            if (retryable) {
                *retryable = true;
                unsubmitted.dismiss();  // The caller will retry the same physical batch.
                return nullptr;
            }
            lock.unlock();
            return reject(task, "RESOURCE_EXHAUSTED: async transfer queue full", ErrorCode::EXECUTION_EXCEPTION);
        }
        // Admission and the move are atomic with respect to competing submitters
        // and shutdown. The resident budget counts unfinished batches.
        auto    total_timer = timer(task, "worker_task_total", 0, count, AsyncProfileKind::Lifecycle, "worker_task");
        auto    queue_or_finalize_timer = timer(task,
                                             "transfer_queue_wait",
                                             0,
                                             count,
                                             AsyncProfileKind::Wait,
                                             "worker_task",
                                             &total_timer.trace(),
                                             total_timer.startSteadyUs());
        Entries incoming;
        incoming.push_back(std::make_unique<Entry>(std::move(task), std::move(before_complete)));
        incoming.front()->task_timer              = std::move(total_timer);
        incoming.front()->queue_or_finalize_timer = std::move(queue_or_finalize_timer);
        auto context                              = incoming.front()->context;
        unsubmitted.dismiss();
        incoming_.splice(incoming_.end(), incoming);
        ++entry_count_;
        cv_.notify_all();
        return context;
    });
}

void TransferWorkLoop::stopAdmission() {
    std::lock_guard<std::mutex> lock(mutex_);
    accepting_ = false;
    cv_.notify_all();
}

void TransferWorkLoop::cancelPendingStagingTransfers() {
    std::lock_guard<std::mutex> lock(mutex_);
    cancel_staging_ = true;
    cv_.notify_all();
}

void TransferWorkLoop::shutdown() {
    assert(thread_.get_id() != std::this_thread::get_id());
    stopAdmission();
    if (thread_.joinable())
        thread_.join();
    if (crc_)
        crc_->stopAndWaitForIo();
}

size_t TransferWorkLoop::outstandingBatches() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return entry_count_;
}

TransferStageTimer TransferWorkLoop::timer(const TransferTask&     task,
                                           const char*             stage,
                                           size_t                  begin,
                                           size_t                  end,
                                           AsyncProfileKind        kind,
                                           const char*             scope,
                                           const AsyncProfileSpan* parent,
                                           int64_t                 start_us) const noexcept {
    return TransferStageTimer(
        metrics_reporter_.get(),
        stage,
        task.requestId().value_or(-1),
        begin,
        end,
        world_rank_,
        scope,
        parent ? parent->transferId() : task.traceId(),
        task.descriptors().empty() ?
            CacheStageContext::local(Tier::NONE, Tier::NONE) :
            CacheStageContext::local(task.descriptors().front().source_tier, task.descriptors().front().target_tier),
        kind,
        parent,
        start_us);
}

void TransferWorkLoop::startStepTimer(Entry& entry, const char* stage) {
    if (!entry.step_started) {
        entry.step_started = true;
        entry.step_timer   = timer(entry.task,
                                 stage,
                                 0,
                                 entry.task.descriptors().size(),
                                 AsyncProfileKind::Lifecycle,
                                 "worker_batch",
                                 &entry.task_timer.trace());
    }
}

const GroupSetPtr& TransferWorkLoop::group(const Entry& entry) const {
    return groups_.at(entry.task.descriptors().front().group_set_id);
}

bool TransferWorkLoop::plan(Entry& entry) {
    if (!std::exchange(entry.initial_plan, false))
        return true;  // Keep the same buffers and step across retries and successors.
    {
        const bool expired = entry.task.expired();
        entry.queue_or_finalize_timer.finish(!expired);
        if (expired) {
            fail(entry,
                 unsubmittedResult(
                     TransferResult(ErrorInfo(ErrorCode::DEADLINE_EXCEEDED, "transfer expired in WorkLoop queue"),
                                    CacheCopyError::NOT_EXECUTED)));
            return false;
        }
    }
    const auto& descriptors = entry.task.descriptors();
    const auto& first       = descriptors.front();
    const bool  combined    = first.source_tier == Tier::DISK && first.target_tier == Tier::DEVICE;
    entry.hosts.clear();
    entry.hosts.reserve(descriptors.size());
    entry.step = first.source_tier == Tier::DISK   ? Step::DISK_READ :
                 first.source_tier == Tier::DEVICE ? Step::CRC_EVICT :
                 first.target_tier == Tier::DISK   ? Step::DISK_WRITE :
                                                     Step::CRC_LOAD;
    if (isDeviceCopy(entry.step) && !group(entry)->crcEnabled()) {
        entry.step = Step::PLAIN_DEVICE_COPY;
    }
    if (combined) {
        if (!staging_.pools[stagingIndex(entry)])
            throw std::invalid_argument("async disk staging is unavailable");
        entry.batch_timer   = timer(entry.task,
                                  "worker_batch_total",
                                  0,
                                  descriptors.size(),
                                  AsyncProfileKind::Lifecycle,
                                  "worker_batch",
                                  &entry.task_timer.trace());
        entry.staging_timer = timer(entry.task,
                                    "staging_wait",
                                    0,
                                    descriptors.size(),
                                    AsyncProfileKind::Wait,
                                    "worker_batch",
                                    &entry.task_timer.trace(),
                                    entry.batch_timer.startSteadyUs());
        if (entry.staging.empty() && !acquireStaging(entry))
            return false;
    } else {
        for (const auto& descriptor : descriptors) {
            const auto buffer = group(entry)->hostPool()->blockBuffer(descriptor.singleBlockAt(Tier::HOST));
            entry.hosts.push_back({buffer.addr, buffer.payload_bytes, buffer.stride_bytes});
        }
    }
    return true;
}

size_t TransferWorkLoop::stagingIndex(const Entry& entry) const {
    return group(entry)->groupType() == CacheGroupType::SWA ? 1 : 0;
}

bool TransferWorkLoop::acquireStaging(Entry& entry, bool waiting) {
    if (!waiting && entry.task.expired()) {
        fail(entry,
             unsubmittedResult(TransferResult(ErrorInfo(ErrorCode::DEADLINE_EXCEEDED, "staging admission timed out"),
                                              CacheCopyError::NOT_EXECUTED)));
        return false;
    }
    if (!waiting && !staging_.waiting.empty()) {
        waitForResource(entry, staging_.waiting);
        retryStaging();
        return false;
    }
    auto leases = staging_.pools[stagingIndex(entry)]->tryMallocBatch(entry.task.descriptors().size());
    if (!leases) {
        if (!waiting)
            waitForResource(entry, staging_.waiting);
        return false;
    }
    entry.staging = std::move(*leases);
    entry.staging_timer.finish(true);
    for (const auto& lease : entry.staging)
        entry.hosts.push_back(lease.blockBuffer(group(entry)->payloadBytes()));
    return true;
}

void TransferWorkLoop::releaseStaging(Entry& entry) {
    if (entry.staging.empty())
        return;
    entry.staging.clear();
    retryStaging();
}

void TransferWorkLoop::retryStaging() {
    // Preserve FIFO within each pool, while a blocked FULL waiter must not
    // prevent SWA from using its independent capacity (and vice versa).
    std::array<bool, 2> blocked{};
    for (auto* waiting = staging_.waiting.head; waiting;) {
        auto* next = waiting->next;
        withEntry(*waiting, [&] {
            // Remove a failed waiter before withEntry finishes and destroys it.
            block_tree_cache_detail::ScopeRollback unqueue([&] { removeWaiter(*waiting); });
            if (waiting->task.expired()) {
                unqueue.run();
                fail(*waiting,
                     unsubmittedResult(
                         TransferResult(ErrorInfo(ErrorCode::DEADLINE_EXCEEDED, "staging admission timed out"),
                                        CacheCopyError::NOT_EXECUTED)));
                return;
            }
            const auto lane = stagingIndex(*waiting);
            if (blocked[lane] || !acquireStaging(*waiting, true)) {
                unqueue.dismiss();
                blocked[lane] = true;
                return;
            }
            unqueue.run();
            ready_.push(*waiting);
        });
        waiting = next;
    }
}

void TransferWorkLoop::tryStep(Entry& entry) {
    assert(entry.result.ok());
    withEntry(entry, [&] {
        if (!plan(entry))
            return;
        if (entry.task.expired()) {
            fail(entry,
                 unsubmittedResult(
                     TransferResult(ErrorInfo(ErrorCode::DEADLINE_EXCEEDED, "transfer expired before stage submission"),
                                    CacheCopyError::NOT_EXECUTED)));
            return;
        }
        if (entry.step == Step::PLAIN_DEVICE_COPY)
            submitPlainDeviceCopy(entry);
        else if (isDeviceCopy(entry.step))
            submitCrc(entry);
        else
            submitDisk(entry);
    });
}

void TransferWorkLoop::submitCrc(Entry& entry) {
    startStepTimer(entry, entry.step == Step::CRC_LOAD ? "h2d" : "d2h");
    const auto& owner   = group(entry);
    auto&       waiting = crc_waiting_.at(owner->devicePools().front()->deviceIndex());
    // Settle the stable Entry and intrusive indexes. Optional observations use
    // the existing task span and preallocated profiling buffer.
    CrcCopyExecutor::DoneCallback done = [this, &entry](TransferResult result) noexcept {
        assert(entry.step == Step::CRC_LOAD || entry.step == Step::CRC_EVICT);
        const auto& timing = entry.crc_timing;
        if (timing.completed_us && result.ok()) {
            const auto& trace             = entry.task_timer.trace();
            const auto  released_us       = trace.active() ? AsyncProfileSession::steadyNowUs() : 0;
            entry.queue_or_finalize_timer = timer(entry.task,
                                                  "transfer_finalize_latency",
                                                  0,
                                                  entry.task.descriptors().size(),
                                                  AsyncProfileKind::Lifecycle,
                                                  "worker_batch",
                                                  &entry.task_timer.trace(),
                                                  timing.completed_us);
            if (released_us) {
                const auto& descriptor = entry.task.descriptors().front();
                const auto  identity   = CacheStageContext::local(descriptor.source_tier, descriptor.target_tier);
                auto        released   = AsyncProfileSpan::fromSteady("slot_released",
                                                             entry.task.requestId().value_or(-1),
                                                             0,
                                                             entry.task.descriptors().size(),
                                                             released_us,
                                                             world_rank_,
                                                             trace.transferId(),
                                                             identity.path,
                                                             "worker_batch",
                                                             identity.operation,
                                                             AsyncProfileKind::Derived,
                                                             &trace);
                released.finishAtSteady(released_us);
            }
        }
        recordResult(entry, std::move(result));
        completed_.push(entry);
    };
    auto*      timing  = entry.task_timer ? &entry.crc_timing : nullptr;
    const bool handled = crc_->trySubmit(entry.hosts, entry.task.descriptors(), owner, std::move(done), timing);
    if (timing && timing->submit_return_us) {
        auto submit = timer(entry.task,
                            "crc_submit_cpu",
                            0,
                            entry.task.descriptors().size(),
                            AsyncProfileKind::Execution,
                            "worker_batch",
                            &entry.task_timer.trace(),
                            timing->slot_acquired_us);
        submit.finishAtSteady(timing->submit_return_us, true);
    }
    if (!handled) {
        waitForResource(entry, waiting);
    } else {
        // Keep handled work active until the next poll. Immediate callbacks
        // already queued completion; no separate per-Entry pending state is needed.
        has_crc_pending_ = true;
    }
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

void TransferWorkLoop::publishDisk(Entry& entry, BlockIOStatus status) noexcept {
    recordDiskResult(entry, status);
    // Hold the lock from publication through the final notification so the owner
    // cannot release admission or exit before this callback finishes accessing the WorkLoop.
    std::lock_guard<std::mutex> lock(mutex_);
    disk_completed_.push(entry);
    // Do not touch entry after publication: the owner may already consume it.
    cv_.notify_all();
}

void TransferWorkLoop::pollOnce() {
    expireResourceWaiters();
    Queue disk_completed;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        disk_completed = disk_completed_.take();
    }
    while (auto* entry = disk_completed.pop())
        completed_.push(*entry);
    has_crc_pending_ = crc_ && crc_->poll();
    auto completed   = completed_.take();
    while (auto* entry = completed.pop())
        consumeStage(*entry);
    if (crc_) {
        // Retry from actual free capacity. A previously woken waiter can expire
        // before submitting, so a completion callback alone is not a wakeup.
        for (auto& [device, waiting] : crc_waiting_) {
            if (waiting.empty())
                continue;
            auto free_slots = crc_->availableSlots(device);
            while (free_slots > 0 && waiting.head) {
                auto& entry = *waiting.head;
                removeWaiter(entry);
                ready_.push(entry);
                --free_slots;
            }
        }
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

void TransferWorkLoop::recordResult(Entry& entry, TransferResult result) noexcept {
    assert(entry.result.ok());  // The first failure ends this Entry.
    const bool intermediate =
        result.ok() && entry.step == Step::DISK_READ && entry.task.descriptors().front().target_tier == Tier::DEVICE;
    if (!intermediate)
        for (size_t i = 0; i < entry.task.descriptors().size(); ++i)
            entry.result.block_errors[i] = result.blockError(i);
    entry.result.buffer_safe = entry.result.buffer_safe && result.buffer_safe;
    entry.result.status      = std::move(result.status);
    entry.result.copy_error  = result.copy_error;
}

void TransferWorkLoop::consumeStage(Entry& entry) {
    auto& terminal = entry.result;
    // End asynchronous stages only after their safe result reaches the owner.
    // Busy retries retain the same timer and do not produce extra samples.
    const auto end_us = entry.task_timer ? AsyncProfileSession::steadyNowUs() : -1;
    entry.step_timer.finishAtSteady(end_us, terminal.ok());
    entry.step_timer   = {};
    entry.step_started = false;
    if (!terminal.ok()) {
        entry.batch_timer.finishAtSteady(end_us, false);
        entry.task_timer.finishAtSteady(end_us, false);
        const auto& descriptor = entry.task.descriptors().front();
        if (!isDeviceCopy(entry.step))
            terminal.status = transferStatusToErrorInfo(terminal.copy_error == CacheCopyError::INVALID_REQUEST ?
                                                            TransferStatus::INVALID_ARGS :
                                                            TransferStatus::DISK_IO_ERROR);
        if (descriptor.source_tier == Tier::DISK && descriptor.target_tier == Tier::DEVICE) {
            const char* stage = entry.step == Step::DISK_READ ? "disk-to-staging" : "staging-to-device";
            terminal.status =
                ErrorInfo(terminal.status.code(),
                          std::string("disk-to-device transfer failed, stage=") + stage + ", descriptor_range=[0,"
                              + std::to_string(entry.task.descriptors().size()) + "): " + terminal.status.ToString());
        }
    }
    if (!terminal.ok()) {
        releaseStaging(entry);
        finish(entry);
        return;
    }
    const auto& descriptor = entry.task.descriptors().front();
    if (entry.step == Step::DISK_READ && descriptor.target_tier == Tier::DEVICE) {
        entry.step = group(entry)->crcEnabled() ? Step::CRC_LOAD : Step::PLAIN_DEVICE_COPY;
        tryStep(entry);
    } else {
        entry.batch_timer.finishAtSteady(end_us, true);
        entry.task_timer.finishAtSteady(end_us, true);
        releaseStaging(entry);
        finish(entry);
    }
}

void TransferWorkLoop::fail(Entry& entry, TransferResult result) {
    removeWaiter(entry);
    recordResult(entry, std::move(result));
    const auto end_us = entry.task_timer ? AsyncProfileSession::steadyNowUs() : -1;
    entry.step_timer.finishAtSteady(end_us, false);
    entry.staging_timer.finishAtSteady(end_us, false);
    entry.batch_timer.finishAtSteady(end_us, false);
    entry.task_timer.finishAtSteady(end_us, false);
    releaseStaging(entry);
    finish(entry);
}

void TransferWorkLoop::finish(Entry& entry) {
    auto finalize_timer =
        entry.crc_timing.completed_us ? std::move(entry.queue_or_finalize_timer) : TransferStageTimer{};
    entry.task_timer.finish(entry.result.ok());
    try {
        abortOnOom([&] {
            if (entry.before_complete)
                entry.before_complete(entry.result);
        });
    } catch (const std::exception& error) {
        RTP_LLM_LOG_WARNING("async transfer before_complete callback failed: %s", error.what());
    } catch (...) {
        RTP_LLM_LOG_WARNING("async transfer before_complete callback failed");
    }
    entry.context->complete(std::move(entry.result));
    // Context ready precedes callback return. Neither permits early reuse of
    // admission capacity or destruction of the callback's captured resources.
    entries_.erase(entry.position);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        --entry_count_;
        cv_.notify_all();
    }
    finalize_timer.finish(true);
}

void TransferWorkLoop::cancelStaging() {
    while (auto* entry = staging_.waiting.head) {
        removeWaiter(*entry);
        fail(*entry,
             unsubmittedResult(TransferResult(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "staging admission cancelled"),
                                              CacheCopyError::NOT_EXECUTED)));
    }
}

void TransferWorkLoop::waitForResource(Entry& entry, Queue& waiting) {
    assert(!entry.waiting_on && !entry.wait_deadline);
    entry.wait_deadline = wait_deadlines_.emplace(entry.task.deadline(), &entry);
    waiting.push(entry);
    entry.waiting_on = &waiting;
}

void TransferWorkLoop::removeWaiter(Entry& entry) {
    if (entry.wait_deadline) {
        wait_deadlines_.erase(*entry.wait_deadline);
        entry.wait_deadline.reset();
    }
    if (entry.waiting_on) {
        entry.waiting_on->remove(entry);
        entry.waiting_on = nullptr;
    }
}

void TransferWorkLoop::expireResourceWaiters() {
    if (wait_deadlines_.empty())
        return;
    const auto now = TransferTask::Clock::now();
    if (wait_deadlines_.begin()->first > now)
        return;
    while (!wait_deadlines_.empty() && wait_deadlines_.begin()->first <= now) {
        auto& entry = *wait_deadlines_.begin()->second;
        removeWaiter(entry);
        fail(entry,
             unsubmittedResult(
                 TransferResult(ErrorInfo(ErrorCode::DEADLINE_EXCEEDED, "transfer resource wait timed out"),
                                CacheCopyError::NOT_EXECUTED)));
    }
    // An expired FIFO head may have needed more blocks than the next waiter.
    // Retry against existing free capacity without waiting for another release.
    retryStaging();
}

void TransferWorkLoop::run() {
    for (;;) {
        Entries incoming;
        bool    cancel;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            incoming.splice(incoming.end(), incoming_);
            cancel = std::exchange(cancel_staging_, false);
        }
        while (!incoming.empty()) {
            auto  position = incoming.begin();
            auto& entry    = **position;
            entries_.splice(entries_.end(), incoming, position);
            entry.position = position;
            ready_.push(entry);
        }
        if (cancel)
            cancelStaging();
        Queue cuda_work, disk_work;
        auto  ready = ready_.take();
        while (auto* entry = ready.pop()) {
            withEntry(*entry, [&] {
                if (plan(*entry))
                    (isDeviceCopy(entry->step) ? cuda_work : disk_work).push(*entry);
            });
        }
        auto submit = [&](Queue& work) {
            while (auto* entry = work.pop())
                tryStep(*entry);
        };
        if (operation_ == TransferOperation::LOAD) {
            submit(disk_work);
            submit(cuda_work);
        } else {
            submit(cuda_work);
            submit(disk_work);
        }
        pollOnce();
        std::unique_lock<std::mutex> lock(mutex_);
        if (!accepting_ && entry_count_ == 0 && incoming_.empty() && disk_completed_.empty())
            return;
        if (ready_.empty() && completed_.empty() && !has_crc_pending_) {
            const auto wake_up = [this] {
                return !incoming_.empty() || !disk_completed_.empty() || cancel_staging_
                       || (!accepting_ && entry_count_ == 0);
            };
            if (wait_deadlines_.empty()) {
                // No resource deadline exists. Converting steady_clock::time_point::max()
                // to system_clock in older condition_variable implementations can
                // overflow and spin without releasing mutex_, blocking shutdown.
                cv_.wait(lock, wake_up);
            } else {
                cv_.wait_until(lock, wait_deadlines_.begin()->first, wake_up);
            }
            lock.unlock();
            // Give already completed Disk stages and their successors the first
            // opportunity after waking, before admitting this turn's new work.
            pollOnce();
        }
    }
}

}  // namespace rtp_llm
