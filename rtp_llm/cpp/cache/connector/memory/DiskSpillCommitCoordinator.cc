#include "rtp_llm/cpp/cache/connector/memory/DiskSpillCommitCoordinator.h"

#include <cstring>

#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace {

size_t alignUpTo(size_t value, size_t alignment) {
    RTP_LLM_CHECK_WITH_INFO(alignment != 0, "disk spill alignment must not be zero");
    return ((value + alignment - 1) / alignment) * alignment;
}

}  // namespace

DiskSpillCommitCoordinator::DiskSpillCommitCoordinator(std::shared_ptr<DiskSpillBlockCache> cache,
                                                       Config                               config,
                                                       int                                  worker_count,
                                                       BroadcastSpillFn                     spill_fn,
                                                       BroadcastDeleteFn                    delete_fn,
                                                       PollWorkerStatusFn                   poll_fn):
    cache_(std::move(cache)),
    config_(config),
    worker_count_(worker_count),
    spill_fn_(std::move(spill_fn)),
    delete_fn_(std::move(delete_fn)),
    poll_fn_(std::move(poll_fn)) {}

DiskSpillCommitCoordinator::~DiskSpillCommitCoordinator() {
    stop();
}

bool DiskSpillCommitCoordinator::start() {
    bool expected = false;
    if (!running_.compare_exchange_strong(expected, true)) {
        return true;
    }
    thread_ = std::thread([this]() { mainLoop(); });
    RTP_LLM_LOG_INFO("disk spill commit coordinator started, worker_count=%d stage_ack_timeout=%ld commit_timeout=%ld",
                     worker_count_,
                     config_.stage_ack_timeout_ms,
                     config_.commit_timeout_ms);
    return true;
}

void DiskSpillCommitCoordinator::stop() {
    bool expected = true;
    if (!running_.compare_exchange_strong(expected, false)) {
        return;
    }
    {
        std::lock_guard<std::mutex> lk(mutex_);
        cv_.notify_all();
    }
    if (thread_.joinable()) {
        thread_.join();
    }
    // abort all remaining jobs
    {
        std::unique_lock<std::mutex> lock(mutex_);
        for (auto& [id, job] : jobs_) {
            if (job.state != SpillStageState::COMMITTED && job.state != SpillStageState::FREE
                && job.state != SpillStageState::LEAKED) {
                terminate(job, SpillStageState::ABORTING, lock);
            }
        }
        jobs_.clear();
    }
    // Wait for all dispatched local pwrite callbacks to finish. The IoWorker
    // owned by DiskSpillBlockCache may outlive this coordinator (especially in
    // tests where coordinator is stack-allocated), so callbacks holding `this`
    // must drain before we return.
    std::unique_lock<std::mutex> lk(pending_callbacks_mutex_);
    pending_callbacks_cv_.wait(lk, [this]() { return pending_callbacks_ == 0; });
}

SpillJobId DiskSpillCommitCoordinator::submitSpill(const DiskSpillBlockCache::DiskItem& slot,
                                                   BlockIdxType                         source_mem_block,
                                                   std::vector<char>                    staging_data,
                                                   OnCompleteFn                         on_complete) {
    if (!running_.load()) {
        if (on_complete) {
            on_complete(0, SpillStageState::ABORTING);
        }
        return 0;
    }
    std::unique_lock<std::mutex> lock(mutex_);
    if (jobs_.size() >= config_.max_inflight_jobs) {
        // backpressure; refuse and let caller drop the spill
        if (on_complete) {
            on_complete(0, SpillStageState::ABORTING);
        }
        return 0;
    }
    const auto id = next_job_id_.fetch_add(1);
    SpillJob   job;
    job.id               = id;
    job.slot             = slot;
    job.source_mem_block = source_mem_block;
    job.staging_data     = std::make_shared<std::vector<char>>(std::move(staging_data));
    job.on_complete      = std::move(on_complete);
    job.state            = SpillStageState::RESERVED;
    job.created_at       = std::chrono::steady_clock::now();
    job.staging_done_at   = job.created_at;
    job.pwrite_inflight_at = job.created_at;
    for (int r = 0; r < worker_count_; ++r) {
        job.worker_status[r] = SpillWriteStatus::PENDING;
    }
    jobs_[id] = std::move(job);
    cv_.notify_one();
    return id;
}

void DiskSpillCommitCoordinator::notifyWorkerStatus(SpillJobId job_id, int rank, SpillWriteStatus status) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto                  it = jobs_.find(job_id);
    if (it == jobs_.end()) {
        return;
    }
    it->second.worker_status[rank] = status;
    cv_.notify_one();
}

SpillStageState DiskSpillCommitCoordinator::getJobState(SpillJobId job_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto                  it = jobs_.find(job_id);
    if (it == jobs_.end()) {
        return SpillStageState::FREE;
    }
    return it->second.state;
}

SpillWriteStatus DiskSpillCommitCoordinator::getJobWriteStatus(SpillJobId job_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto                  it = jobs_.find(job_id);
    if (it == jobs_.end()) {
        return SpillWriteStatus::UNKNOWN_JOB;
    }
    const auto& job = it->second;
    if (job.state == SpillStageState::COMMITTED) {
        return SpillWriteStatus::SUCCESS;
    }
    if (job.state == SpillStageState::LEAKED || job.state == SpillStageState::ABORTING) {
        return SpillWriteStatus::FAILED;
    }
    return SpillWriteStatus::PENDING;
}

size_t DiskSpillCommitCoordinator::inflightJobs() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return jobs_.size();
}

bool DiskSpillCommitCoordinator::drainForTest(int64_t timeout_ms) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (std::chrono::steady_clock::now() < deadline) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            // count non-terminal jobs
            size_t pending = 0;
            for (const auto& [_, job] : jobs_) {
                if (job.state != SpillStageState::COMMITTED && job.state != SpillStageState::FREE
                    && job.state != SpillStageState::LEAKED) {
                    ++pending;
                }
            }
            if (pending == 0) {
                return true;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    return false;
}

void DiskSpillCommitCoordinator::mainLoop() {
    while (running_.load()) {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait_for(lock, std::chrono::milliseconds(config_.poll_interval_ms), [this]() {
                return !running_.load() || !jobs_.empty();
            });
            if (!running_.load()) {
                return;
            }
            tickLocked(lock);
        }
    }
}

void DiskSpillCommitCoordinator::tickLocked(std::unique_lock<std::mutex>& lock) {
    struct StageTask {
        LocalPwriteTask pwrite;
        BlockIdxType    source_mem_block{NULL_BLOCK_IDX};
        bool            broadcast{false};
    };

    const auto now = std::chrono::steady_clock::now();
    std::vector<SpillJobId> terminate_ids;
    std::vector<StageTask> stage_tasks;
    for (auto& [id, job] : jobs_) {
        // RESERVED -> STAGING: dispatch local pwrite + broadcast spill
        if (job.state == SpillStageState::RESERVED) {
            stage_tasks.push_back(
                StageTask{LocalPwriteTask{id, job.slot, job.staging_data}, job.source_mem_block, spill_fn_ != nullptr});
            job.state           = SpillStageState::STAGING;
            job.staging_done_at = now;
            if (spill_fn_) {
                job.spill_broadcast_sent = true;
            } else {
                // no worker fanout (e.g. TP=1) — treat as instantly acked
                for (auto& [rank, status] : job.worker_status) {
                    status = SpillWriteStatus::SUCCESS;
                }
                job.spill_broadcast_sent = true;
            }
        }

        // STAGING: enter pwrite collection as soon as local pwrite progresses, or
        // when stage ack times out. The previous implementation waited for the
        // full stage_ack_timeout even when every ack was already complete, which
        // delayed commits by minutes in production configs.
        if (job.state == SpillStageState::STAGING) {
            if (shouldEnterPwriteInflight(job, now)) {
                job.state              = SpillStageState::PWRITE_INFLIGHT;
                job.pwrite_inflight_at = now;
            }
        }

        // PWRITE_INFLIGHT: poll workers + check timeout
        if (job.state == SpillStageState::PWRITE_INFLIGHT) {
            if (poll_fn_) {
                for (auto& [rank, status] : job.worker_status) {
                    if (status == SpillWriteStatus::PENDING) {
                        status = poll_fn_(rank, job.id);
                    }
                }
            }
            if (job.local_pwrite_done && allWorkersDone(job)) {
                if (job.local_pwrite_ok && !anyWorkerFailed(job)) {
                    // commit
                    if (cache_->commit(job.slot)) {
                        job.state = SpillStageState::COMMITTED;
                    } else {
                        job.state = SpillStageState::ABORTING;
                    }
                } else {
                    job.state = SpillStageState::ABORTING;
                }
                terminate_ids.push_back(id);
            } else {
                const auto elapsed_ms =
                    std::chrono::duration_cast<std::chrono::milliseconds>(now - job.pwrite_inflight_at).count();
                if (elapsed_ms > config_.commit_timeout_ms) {
                    job.state = SpillStageState::ABORTING;
                    terminate_ids.push_back(id);
                }
            }
        }
    }

    if (!stage_tasks.empty()) {
        lock.unlock();
        std::vector<SpillJobId> failed_stage_tasks;
        failed_stage_tasks.reserve(stage_tasks.size());
        for (const auto& task : stage_tasks) {
            if (!dispatchLocalPwrite(task.pwrite)) {
                failed_stage_tasks.push_back(task.pwrite.id);
                continue;
            }
            if (task.broadcast) {
                bool ok = false;
                try {
                    ok = spill_fn_(task.pwrite.id, task.pwrite.slot, task.source_mem_block);
                } catch (const std::exception& e) {
                    RTP_LLM_LOG_WARNING("disk spill broadcast callback threw: %s", e.what());
                } catch (...) {
                    RTP_LLM_LOG_WARNING("disk spill broadcast callback threw unknown");
                }
                if (!ok) {
                    failed_stage_tasks.push_back(task.pwrite.id);
                }
            }
        }
        lock.lock();
        for (auto id : failed_stage_tasks) {
            auto it = jobs_.find(id);
            if (it == jobs_.end()) {
                continue;
            }
            if (it->second.state != SpillStageState::COMMITTED && it->second.state != SpillStageState::FREE
                && it->second.state != SpillStageState::LEAKED) {
                it->second.state = SpillStageState::ABORTING;
                terminate_ids.push_back(id);
            }
        }
    }

    // Run terminations outside the per-job loop to avoid mutating the map during
    // iteration.
    for (auto id : terminate_ids) {
        auto it = jobs_.find(id);
        if (it == jobs_.end()) {
            continue;
        }
        auto& job = it->second;
        if (job.state == SpillStageState::ABORTING) {
            cache_->abort(job.slot);
            if (delete_fn_) {
                if (!delete_fn_(job.slot)) {
                    job.state = SpillStageState::LEAKED;
                }
            }
            if (job.state == SpillStageState::ABORTING) {
                job.state = SpillStageState::FREE;
            }
        }
        if (job.on_complete) {
            try {
                job.on_complete(job.id, job.state);
            } catch (const std::exception& e) {
                RTP_LLM_LOG_WARNING("disk spill on_complete callback threw: %s", e.what());
            }
        }
        jobs_.erase(it);
    }
}

bool DiskSpillCommitCoordinator::dispatchLocalPwrite(const LocalPwriteTask& task) {
    if (!cache_) {
        return false;
    }
    const auto fms = cache_->fileManagers();
    if (task.slot.disk_id < 0 || static_cast<size_t>(task.slot.disk_id) >= fms.size()) {
        return false;
    }
    const auto file_manager = fms[task.slot.disk_id];
    if (!file_manager) {
        return false;
    }
    const auto workers = cache_->ioWorkers();
    if (static_cast<size_t>(task.slot.disk_id) >= workers.size() || !workers[task.slot.disk_id]) {
        return false;
    }
    auto staging = task.staging_data;
    if (!staging || staging->size() < task.slot.block_size) {
        return false;
    }
    if (file_manager->ioMode() == DiskSpillFileManager::IoMode::DIRECT) {
        auto aligned_staging = file_manager->acquireStagingBuffer();
        if (!aligned_staging || !aligned_staging->valid() || aligned_staging->size() < task.slot.block_size) {
            if (aligned_staging) {
                file_manager->releaseStagingBuffer(aligned_staging);
            }
            return false;
        }
        const auto write_bytes = alignUpTo(task.slot.block_size, file_manager->alignBytes());
        if (write_bytes > aligned_staging->size()) {
            file_manager->releaseStagingBuffer(aligned_staging);
            return false;
        }
        std::memcpy(aligned_staging->data(), staging->data(), task.slot.block_size);
        if (write_bytes > task.slot.block_size) {
            std::memset(static_cast<char*>(aligned_staging->data()) + task.slot.block_size,
                        0,
                        write_bytes - task.slot.block_size);
        }

        const auto id   = task.id;
        auto*      self = this;
        {
            std::lock_guard<std::mutex> lk(pending_callbacks_mutex_);
            ++pending_callbacks_;
        }
        const bool ok = workers[task.slot.disk_id]->submitWrite(
            task.slot.slot_id,
            aligned_staging->data(),
            write_bytes,
            [self, id, staging, aligned_staging, file_manager](bool ok, const std::string& /*err*/) {
                file_manager->releaseStagingBuffer(aligned_staging);
                self->onLocalPwriteComplete(id, ok);
                std::lock_guard<std::mutex> lk(self->pending_callbacks_mutex_);
                --self->pending_callbacks_;
                self->pending_callbacks_cv_.notify_all();
            });
        return ok;
    }

    const auto id   = task.id;
    auto*      self = this;
    {
        std::lock_guard<std::mutex> lk(pending_callbacks_mutex_);
        ++pending_callbacks_;
    }
    const bool ok = workers[task.slot.disk_id]->submitWrite(
        task.slot.slot_id,
        staging->data(),
        task.slot.block_size,
        [self, id, staging](bool ok, const std::string& /*err*/) {
            self->onLocalPwriteComplete(id, ok);
            std::lock_guard<std::mutex> lk(self->pending_callbacks_mutex_);
            --self->pending_callbacks_;
            self->pending_callbacks_cv_.notify_all();
        });
    return ok;
}

void DiskSpillCommitCoordinator::onLocalPwriteComplete(SpillJobId id, bool ok) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto                  it = jobs_.find(id);
    if (it == jobs_.end()) {
        return;
    }
    auto& job             = it->second;
    job.local_pwrite_done = true;
    job.local_pwrite_ok   = ok;
    cv_.notify_one();
}

bool DiskSpillCommitCoordinator::shouldEnterPwriteInflight(
    const SpillJob& job, std::chrono::steady_clock::time_point now) const {
    if (job.local_pwrite_done || allWorkersDone(job) || anyWorkerFailed(job)) {
        return true;
    }
    const auto elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(now - job.staging_done_at).count();
    return elapsed_ms > config_.stage_ack_timeout_ms;
}

bool DiskSpillCommitCoordinator::allWorkersDone(const SpillJob& job) const {
    for (const auto& [rank, status] : job.worker_status) {
        if (status == SpillWriteStatus::PENDING) {
            return false;
        }
    }
    return true;
}

bool DiskSpillCommitCoordinator::anyWorkerFailed(const SpillJob& job) const {
    for (const auto& [rank, status] : job.worker_status) {
        if (status == SpillWriteStatus::FAILED || status == SpillWriteStatus::UNKNOWN_JOB) {
            return true;
        }
    }
    return false;
}

void DiskSpillCommitCoordinator::terminate(SpillJob&                     job,
                                           SpillStageState               final_state,
                                           std::unique_lock<std::mutex>& /*lock*/) {
    job.state = final_state;
    if (final_state == SpillStageState::ABORTING) {
        cache_->abort(job.slot);
        if (delete_fn_) {
            if (!delete_fn_(job.slot)) {
                job.state = SpillStageState::LEAKED;
            } else {
                job.state = SpillStageState::FREE;
            }
        }
    }
    if (job.on_complete) {
        try {
            job.on_complete(job.id, job.state);
        } catch (const std::exception& e) {
            RTP_LLM_LOG_WARNING("disk spill on_complete callback (terminate) threw: %s", e.what());
        }
    }
}

}  // namespace rtp_llm
