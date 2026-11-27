#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferWorkLoop.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <limits>
#include <pthread.h>
#include <stdexcept>
#include <utility>

#include "rtp_llm/cpp/cache/block_tree_cache/ScopeRollback.h"
#include "rtp_llm/cpp/cache/block_tree_cache/diagnostic/CacheDiagnostics.h"

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
    return step == Step::CRC_LOAD || step == Step::CRC_EVICT || step == Step::PLAIN_DEVICE_COPY
           || step == Step::DEVICE_COPY;
}

TransferWorkLoop::TransferWorkLoop(TransferOperation operation,
                                   size_t batch_limit,
                                   size_t queue_capacity,
                                   std::shared_ptr<CacheObserver> observer,
                                   int64_t world_rank,
                                   size_t worker_index,
                                   SubmitBatch submit_batch):
    submit_batch_(std::move(submit_batch)),
    operation_(operation),
    batch_limit_(batch_limit),
    capacity_(queue_capacity),
    metrics_reporter_(std::move(observer)),
    world_rank_(world_rank),
    thread_name_((operation == TransferOperation::LOAD ? "btc_load_" : "btc_evict_") + std::to_string(worker_index)) {
    if (!batch_limit_ || !capacity_)
        throw std::invalid_argument("async transfer requires positive M/Q");
    if (capacity_ > std::numeric_limits<size_t>::max() / batch_limit_)
        throw std::overflow_error("async transfer descriptor capacity overflow");
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
#if defined(__APPLE__)
            const auto error = pthread_setname_np(thread_name_.c_str());
#else
            const auto error = pthread_setname_np(pthread_self(), thread_name_.c_str());
#endif
            if (error)
                BTC_LOG_WARNING("cannot name transfer owner %s: %s", thread_name_.c_str(), std::strerror(error));
            BTC_LOG_INFO("BlockTree transfer owner: name=%s rank=%ld", thread_name_.c_str(), world_rank_);
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
        (this->*rtp_io_->stop_crc)();
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
    if (submit_batch_) {
        entry.step = Step::DEVICE_COPY;
        return true;
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
            entry.hosts.push_back(rtp_io_->host_buffer(*group(entry), descriptor.singleBlockAt(Tier::HOST)));
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
    if (!(this->*rtp_io_->acquire_staging)(entry)) {
        if (!waiting)
            waitForResource(entry, staging_.waiting);
        return false;
    }
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
        if (entry.step == Step::DEVICE_COPY)
            submitDeviceCopy(entry);
        else if (entry.step == Step::PLAIN_DEVICE_COPY)
            (this->*rtp_io_->plain_copy)(entry);
        else if (isDeviceCopy(entry.step))
            submitCrc(entry);
        else
            (this->*rtp_io_->disk_copy)(entry);
    });
}

void TransferWorkLoop::submitCrc(Entry& entry) {
    startStepTimer(entry, entry.step == Step::CRC_LOAD ? "h2d" : "d2h");
    auto& waiting = crc_waiting_.at(rtp_io_->device_index(*group(entry)));
    // Settle the stable Entry and intrusive indexes. Optional observations use
    // the existing task span and preallocated profiling buffer.
    TransferAsyncContext::DoneCallback done = [this, &entry](TransferResult result) noexcept {
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
    const bool handled = (this->*rtp_io_->submit_crc)(entry, std::move(done), timing);
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

void TransferWorkLoop::submitDeviceCopy(Entry& entry) {
    const auto& desc = entry.task.descriptors().front();
    startStepTimer(entry, desc.target_tier == Tier::DEVICE ? "h2d" :
                          desc.source_tier == Tier::DEVICE ? "d2h" :
                          desc.source_tier == Tier::DISK ? "disk_read" : "disk_write");
    auto claimed = std::make_shared<std::atomic<bool>>(false);
    try {
        submit_batch_(entry.task, [this, &entry, claimed](TransferResult result) {
            if (claimed->exchange(true, std::memory_order_acq_rel))
                return;
            if (!result.block_errors.empty() && result.block_errors.size() != entry.task.descriptors().size())
                result = TransferResult::executionException(result.buffer_safe, "invalid worker receipt size");
            if (!result.buffer_safe)
                result.status.setErrorCode(ErrorCode::TRANSFER_COMPLETION_UNKNOWN);
            recordResult(entry, std::move(result));
            publishCompletion(entry);
        });
    } catch (...) {
        if (!claimed->exchange(true, std::memory_order_acq_rel))
            throw;
    }
}

void TransferWorkLoop::publishCompletion(Entry& entry) noexcept {
    // The producer does not access the entry after publishing it to the owner.
    std::lock_guard<std::mutex> lock(mutex_);
    io_completed_.push(entry);
    cv_.notify_all();
}

void TransferWorkLoop::pollOnce() {
    expireResourceWaiters();
    Queue io_completed;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        io_completed = io_completed_.take();
    }
    while (auto* entry = io_completed.pop())
        completed_.push(*entry);
    has_crc_pending_ = crc_ && (this->*rtp_io_->poll_crc)();
    auto completed   = completed_.take();
    while (auto* entry = completed.pop())
        consumeStage(*entry);
    if (crc_) {
        // Retry from actual free capacity. A previously woken waiter can expire
        // before submitting, so a completion callback alone is not a wakeup.
        for (auto& [device, waiting] : crc_waiting_) {
            if (waiting.empty())
                continue;
            auto free_slots = (this->*rtp_io_->free_crc_slots)(device);
            while (free_slots > 0 && waiting.head) {
                auto& entry = *waiting.head;
                removeWaiter(entry);
                ready_.push(entry);
                --free_slots;
            }
        }
    }
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
        BTC_LOG_WARNING("async transfer before_complete callback failed: %s", error.what());
    } catch (...) {
        BTC_LOG_WARNING("async transfer before_complete callback failed");
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
        Queue device_work, disk_work;
        auto  ready = ready_.take();
        while (auto* entry = ready.pop()) {
            withEntry(*entry, [&] {
                if (plan(*entry))
                    (isDeviceCopy(entry->step) ? device_work : disk_work).push(*entry);
            });
        }
        auto submit = [&](Queue& work) {
            while (auto* entry = work.pop())
                tryStep(*entry);
        };
        if (operation_ == TransferOperation::LOAD) {
            submit(disk_work);
            submit(device_work);
        } else {
            submit(device_work);
            submit(disk_work);
        }
        pollOnce();
        std::unique_lock<std::mutex> lock(mutex_);
        if (!accepting_ && entry_count_ == 0 && incoming_.empty() && io_completed_.empty())
            return;
        if (ready_.empty() && completed_.empty() && !has_crc_pending_) {
            const auto wake_up = [this] {
                return !incoming_.empty() || !io_completed_.empty() || cancel_staging_
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
