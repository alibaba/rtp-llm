#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferWorkLoop.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstring>
#include <limits>
#include <pthread.h>
#include <stdexcept>
#include <utility>

#include "rtp_llm/cpp/cache/block_tree_cache/ScopeRollback.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/DiskTransferUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

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
    context(std::make_shared<TransferBatchAsyncContext>(this->task.descriptors().size())),
    result(this->task.descriptors().size()),
    before_complete(std::move(before_complete)) {}

bool TransferWorkLoop::isCrc(Step step) {
    return step == Step::CRC_LOAD || step == Step::CRC_STORE;
}

TransferWorkLoop::TransferWorkLoop(std::vector<GroupSetPtr>                       groups,
                                   std::unique_ptr<CrcTransferService>            crc,
                                   DiskIOExecutor*                                disk_io_executor,
                                   TransferDirection                              direction,
                                   size_t                                         batch_limit,
                                   size_t                                         queue_capacity,
                                   std::shared_ptr<HostStagingBlockPool>          staging,
                                   std::shared_ptr<BlockTreeCacheMetricsReporter> metrics_reporter,
                                   int64_t                                        world_rank,
                                   size_t                                         worker_index):
    groups_(std::move(groups)),
    crc_(std::move(crc)),
    disk_io_executor_(disk_io_executor),
    direction_(direction),
    batch_limit_(batch_limit),
    capacity_(queue_capacity),
    metrics_reporter_(std::move(metrics_reporter)),
    world_rank_(world_rank),
    thread_name_((direction == TransferDirection::LOAD ? "btc_load_" : "btc_offload_") + std::to_string(worker_index)) {
    abortOnOom([&] {
        if (!batch_limit_ || !capacity_)
            throw std::invalid_argument("async transfer requires positive M/Q");
        if (capacity_ > std::numeric_limits<size_t>::max() / batch_limit_)
            throw std::overflow_error("async transfer descriptor capacity overflow");
        staging_.pool = std::move(staging);
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

std::shared_ptr<TransferAsyncContext> TransferWorkLoop::execute(TransferTask           task,
                                                                BeforeCompleteCallback before_complete) {
    return abortOnOom([&]() -> std::shared_ptr<TransferAsyncContext> {
        const auto count = task.descriptors().size();
        // Reuse task-level observations for both direct and combined routes.
        auto       total_timer = timer(task, "worker_task_total", 0, count, AsyncProfileKind::Lifecycle, "worker_task");
        auto       queue_timer = timer(task, "transfer_queue_wait", 0, count, AsyncProfileKind::Wait, "worker_task");
        const auto reject      = [&](const char* message,
                                ErrorCode   code =
                                    ErrorCode::EXECUTION_EXCEPTION) -> std::shared_ptr<TransferAsyncContext> {
            queue_timer.finish(false);
            total_timer.finish(false);
            return std::make_shared<CompletedTransferAsyncContext>(
                TransferResult(ErrorInfo(code, message), CacheCopyError::NOT_EXECUTED), count);
        };
        if (!count)
            return reject("empty WorkLoop transfer", ErrorCode::INVALID_PARAMS);
        if (task.expired())
            return reject("transfer expired before WorkLoop admission", ErrorCode::DEADLINE_EXCEEDED);
        if (count > batch_limit_)
            return reject("WorkLoop batch exceeds configured limit", ErrorCode::INVALID_PARAMS);
        bool rejected;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            rejected = !accepting_ || entry_count_ == capacity_;
        }
        if (rejected)
            return reject("RESOURCE_EXHAUSTED: async transfer full or stopped");
        // Copy logical sizes, including nested block vectors. Do not retain a
        // caller's arbitrary excess vector capacity inside the resident budget.
        auto    owned_task = task.subtask(std::vector<TransferDescriptor>(task.descriptors()));
        Entries incoming;
        incoming.push_back(std::make_unique<Entry>(std::move(owned_task), std::move(before_complete)));
        incoming.front()->task_timer  = std::move(total_timer);
        incoming.front()->queue_timer = std::move(queue_timer);
        auto context                  = incoming.front()->context;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            rejected = !accepting_ || entry_count_ == capacity_;
            if (!rejected) {
                incoming_.splice(incoming_.end(), incoming);
                ++entry_count_;
                cv_.notify_all();
            }
        }
        if (rejected) {
            // The second admission check races with other submitters/stop. The
            // local Entry owns the observers and is destroyed before returning.
            incoming.front()->queue_timer.finish(false);
            incoming.front()->task_timer.finish(false);
            return reject("RESOURCE_EXHAUSTED: async transfer full or stopped");
        }
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

TransferStageTimer TransferWorkLoop::timer(const TransferTask& task,
                                           const char*         stage,
                                           size_t              begin,
                                           size_t              end,
                                           AsyncProfileKind    kind,
                                           const char*         scope) const noexcept {
    return TransferStageTimer(
        metrics_reporter_.get(),
        stage,
        task.requestId().value_or(-1),
        begin,
        end,
        world_rank_,
        scope,
        task.traceId(),
        task.descriptors().empty() ?
            CacheStageContext::local(Tier::NONE, Tier::NONE) :
            CacheStageContext::local(task.descriptors().front().source_tier, task.descriptors().front().target_tier),
        kind);
}

void TransferWorkLoop::startStepTimer(Entry& entry, const char* stage) {
    if (!entry.step_started) {
        entry.step_started = true;
        entry.step_timer   = timer(entry.task, stage, 0, entry.task.descriptors().size(), AsyncProfileKind::Lifecycle);
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
        const auto elapsed = entry.queue_timer.finish(!expired);
        if (metrics_reporter_) {
            const auto& first = entry.task.descriptors().front();
            metrics_reporter_->reportWorkLoopQueueWait(first.source_tier, first.target_tier, elapsed);
        }
        if (expired) {
            fail(entry,
                 TransferResult(ErrorInfo(ErrorCode::DEADLINE_EXCEEDED, "transfer expired in WorkLoop queue"),
                                CacheCopyError::NOT_EXECUTED));
            return false;
        }
    }
    const auto& descriptors = entry.task.descriptors();
    const auto& first       = descriptors.front();
    const bool  combined    = (first.source_tier == Tier::DEVICE && first.target_tier == Tier::DISK)
                          || (first.source_tier == Tier::DISK && first.target_tier == Tier::DEVICE);
    entry.hosts.clear();
    entry.hosts.reserve(descriptors.size());
    entry.step = first.source_tier == Tier::DISK   ? Step::DISK_READ :
                 first.source_tier == Tier::DEVICE ? Step::CRC_STORE :
                 first.target_tier == Tier::DISK   ? Step::DISK_WRITE :
                                                     Step::CRC_LOAD;
    if (combined) {
        if (!staging_.pool)
            throw std::invalid_argument("async disk staging is unavailable");
        entry.batch_timer = timer(entry.task, "worker_batch_total", 0, descriptors.size(), AsyncProfileKind::Lifecycle);
        entry.staging_timer = timer(entry.task, "staging_wait", 0, descriptors.size(), AsyncProfileKind::Wait);
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

bool TransferWorkLoop::acquireStaging(Entry& entry, bool waiting) {
    if (!waiting && entry.task.expired()) {
        fail(entry,
             TransferResult(ErrorInfo(ErrorCode::DEADLINE_EXCEEDED, "staging admission timed out"),
                            CacheCopyError::NOT_EXECUTED));
        return false;
    }
    auto& resource = staging_;
    if (!waiting && !resource.waiting.empty()) {
        waitForResource(entry, resource.waiting);
        return false;
    }
    auto leases = resource.pool->tryMallocBatch(entry.task.descriptors().size());
    if (!leases) {
        if (!waiting)
            waitForResource(entry, resource.waiting);
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
    auto& resource = staging_;
    entry.staging.clear();
    retryStaging(resource);
}

void TransferWorkLoop::retryStaging(StagingResource& resource) {
    // Only the owner uses these pools; no requestBatch callback can run here.
    while (auto* waiting = resource.waiting.head) {
        bool blocked = false;
        withEntry(*waiting, [&] {
            // Remove a failed waiter before withEntry finishes and destroys it.
            block_tree_cache_detail::ScopeRollback unqueue([&] { removeWaiter(*waiting); });
            if (waiting->task.expired()) {
                unqueue.run();
                fail(*waiting,
                     TransferResult(ErrorInfo(ErrorCode::DEADLINE_EXCEEDED, "staging admission timed out"),
                                    CacheCopyError::NOT_EXECUTED));
                return;
            }
            if (!acquireStaging(*waiting, true)) {
                unqueue.dismiss();
                blocked = true;
                return;
            }
            unqueue.run();
            ready_.push(*waiting);
        });
        if (blocked)
            break;
    }
}

void TransferWorkLoop::tryStep(Entry& entry) {
    assert(entry.result.ok());
    withEntry(entry, [&] {
        if (!plan(entry))
            return;
        if (entry.task.expired()) {
            fail(entry,
                 TransferResult(ErrorInfo(ErrorCode::DEADLINE_EXCEEDED, "transfer expired before stage submission"),
                                CacheCopyError::NOT_EXECUTED));
            return;
        }
        if (isCrc(entry.step))
            submitCrc(entry);
        else
            submitDisk(entry);
    });
}

void TransferWorkLoop::submitCrc(Entry& entry) {
    startStepTimer(entry, entry.step == Step::CRC_LOAD ? "h2d" : "d2h");
    const auto& owner   = group(entry);
    auto&       waiting = crc_waiting_.at(owner->devicePools().front()->deviceIndex());
    // The callback only fills a stable Entry and preallocated intrusive indexes.
    CrcTransferService::DoneCallback done = [this, &entry, &waiting](TransferResult result) noexcept {
        assert(isCrc(entry.step));
        recordResult(entry, std::move(result));
        completed_.push(entry);
        if (auto* next = waiting.head) {
            removeWaiter(*next);
            ready_.push(*next);
        }
    };
    const bool handled = crc_->trySubmit(entry.hosts, entry.task.descriptors(), owner, std::move(done));
    if (!handled) {
        waitForResource(entry, waiting);
    } else {
        // Keep handled work active until the next poll. Immediate callbacks
        // already queued completion; no separate per-Entry pending state is needed.
        has_crc_pending_ = true;
    }
}

void TransferWorkLoop::submitDisk(Entry& entry) {
    startStepTimer(entry, entry.step == Step::DISK_READ ? "disk_read" : "disk_write");
    auto*                    pool   = group(entry)->diskPool().get();
    const bool               write  = entry.step == Step::DISK_WRITE;
    const auto               stride = pool->strideBytes();
    BlockIdList              blocks;
    std::vector<void*>       buffers;
    const auto&              descriptors = entry.task.descriptors();
    blocks.reserve(descriptors.size());
    buffers.reserve(descriptors.size());
    for (size_t i = 0; i < descriptors.size(); ++i) {
        const auto& host  = entry.hosts[i];
        const auto& owner = *group(entry);
        if (owner.storageBytes() > stride || !isValidHostBufferView(host, owner.payloadBytes(), stride)) {
            fail(entry, TransferStatus::DISK_IO_ERROR);
            return;
        }
        blocks.push_back(descriptors[i].singleBlockAt(Tier::DISK));
        buffers.push_back(host.base);
    }
    BlockTreeDiskBlockPool::Callback done = [this, &entry](BlockIOStatus status) { publishDisk(entry, status); };
    // Ordinary rejection has no I/O/callback. Accepted workers own completion.
    // Submission exceptions propagate to withEntry; it handles OOM locally.
    assert(disk_io_executor_ != nullptr);
    const auto status =
        write ? pool->asyncWrite(
                    *disk_io_executor_, blocks, buffers, stride, group(entry)->storageBytes(), std::move(done)) :
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
    // Busy can only recover through a record owned by this direction. Do not
    // sleep forever or spin retrying an unregistered resource reservation.
    if (!has_crc_pending_ && ready_.empty() && completed_.empty())
        for (auto& [device, waiting] : crc_waiting_)
            while (auto* entry = waiting.head) {
                removeWaiter(*entry);
                fail(*entry, ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "CRC Busy without a pending direction record"));
            }
}

void TransferWorkLoop::recordDiskResult(Entry& entry, BlockIOStatus status) noexcept {
    assert(entry.result.ok());
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
        result.ok()
        && ((entry.step == Step::DISK_READ && entry.task.descriptors().front().target_tier == Tier::DEVICE)
            || (entry.step == Step::CRC_STORE && entry.task.descriptors().front().target_tier == Tier::DISK));
    if (!intermediate)
        for (size_t i = 0; i < entry.task.descriptors().size(); ++i)
            entry.result.block_errors[i] = result.blockError(i);
    entry.result.status     = std::move(result.status);
    entry.result.copy_error = result.copy_error;
}

void TransferWorkLoop::consumeStage(Entry& entry) {
    auto& terminal = entry.result;
    // End asynchronous stages only after their safe result reaches the owner.
    // Busy retries retain the same timer and do not produce extra samples.
    entry.step_timer.finish(terminal.ok());
    entry.step_timer   = {};
    entry.step_started = false;
    if (!terminal.ok()) {
        entry.batch_timer.finish(false);
        entry.task_timer.finish(false);
        const auto& descriptor = entry.task.descriptors().front();
        if (!isCrc(entry.step))
            terminal.status = transferStatusToErrorInfo(terminal.copy_error == CacheCopyError::INVALID_REQUEST ?
                                                            TransferStatus::INVALID_ARGS :
                                                            TransferStatus::DISK_IO_ERROR);
        if (descriptor.source_tier == Tier::DISK && descriptor.target_tier == Tier::DEVICE) {
            const char* stage = entry.step == Step::DISK_READ ? "disk-to-staging" : "staging-to-device";
            terminal.status =
                ErrorInfo(terminal.status.code(),
                          std::string("disk-to-device transfer failed, stage=") + stage + ", descriptor_range=[0,"
                              + std::to_string(entry.task.descriptors().size()) + "): " + terminal.status.ToString());
        } else if (descriptor.source_tier == Tier::DEVICE && descriptor.target_tier == Tier::DISK) {
            const char* stage = entry.step == Step::CRC_STORE ? "device-to-staging" : "staging-to-disk";
            terminal.status =
                ErrorInfo(terminal.status.code(),
                          std::string("device-to-disk ") + stage + " failed: " + terminal.status.ToString());
        }
    }
    if (!terminal.ok()) {
        releaseStaging(entry);
        finish(entry);
        return;
    }
    const auto& descriptor = entry.task.descriptors().front();
    if (entry.step == Step::DISK_READ && descriptor.target_tier == Tier::DEVICE) {
        entry.step = Step::CRC_LOAD;
        tryStep(entry);
    } else if (entry.step == Step::CRC_STORE && descriptor.target_tier == Tier::DISK) {
        entry.step = Step::DISK_WRITE;
        tryStep(entry);
    } else {
        entry.batch_timer.finish(true);
        entry.task_timer.finish(true);
        releaseStaging(entry);
        finish(entry);
    }
}

void TransferWorkLoop::fail(Entry& entry, TransferResult result) {
    removeWaiter(entry);
    recordResult(entry, std::move(result));
    entry.step_timer.finish(false);
    entry.staging_timer.finish(false);
    entry.batch_timer.finish(false);
    entry.task_timer.finish(false);
    releaseStaging(entry);
    finish(entry);
}

void TransferWorkLoop::finish(Entry& entry) {
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
    std::lock_guard<std::mutex> lock(mutex_);
    --entry_count_;
    cv_.notify_all();
}

void TransferWorkLoop::cancelStaging() {
    while (auto* entry = staging_.waiting.head) {
        removeWaiter(*entry);
        fail(*entry,
             TransferResult(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "staging admission cancelled"),
                            CacheCopyError::NOT_EXECUTED));
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
             TransferResult(ErrorInfo(ErrorCode::DEADLINE_EXCEEDED, "transfer resource wait timed out"),
                            CacheCopyError::NOT_EXECUTED));
    }
    // An expired FIFO head may have needed more blocks than the next waiter.
    // Retry against existing free capacity without waiting for another release.
    retryStaging(staging_);
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
                    (isCrc(entry->step) ? cuda_work : disk_work).push(*entry);
            });
        }
        auto submit = [&](Queue& work) {
            while (auto* entry = work.pop())
                tryStep(*entry);
        };
        if (direction_ == TransferDirection::LOAD) {
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
            const auto wake =
                wait_deadlines_.empty() ? TransferTask::Clock::time_point::max() : wait_deadlines_.begin()->first;
            cv_.wait_until(lock, wake, [&] {
                return !incoming_.empty() || !disk_completed_.empty() || cancel_staging_
                       || (!accepting_ && entry_count_ == 0);
            });
            lock.unlock();
            // Give already completed Disk stages and their successors the first
            // opportunity after waking, before admitting this turn's new work.
            pollOnce();
        }
    }
}

}  // namespace rtp_llm
