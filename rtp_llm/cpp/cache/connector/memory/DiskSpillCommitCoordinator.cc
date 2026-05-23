#include "rtp_llm/cpp/cache/connector/memory/DiskSpillCommitCoordinator.h"

#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

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
    job.id              = id;
    job.slot            = slot;
    job.staging_data    = std::make_shared<std::vector<char>>(std::move(staging_data));
    job.on_complete     = std::move(on_complete);
    job.state           = SpillStageState::RESERVED;
    job.created_at      = std::chrono::steady_clock::now();
    job.staging_done_at = job.created_at;
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
            tickLocked();
        }
    }
}

void DiskSpillCommitCoordinator::tickLocked() {
    const auto now = std::chrono::steady_clock::now();
    std::vector<SpillJobId> terminate_ids;
    for (auto& [id, job] : jobs_) {
        // RESERVED -> STAGING: dispatch local pwrite + broadcast spill
        if (job.state == SpillStageState::RESERVED) {
            if (!tryDispatchLocalPwrite(job)) {
                job.state = SpillStageState::ABORTING;
                terminate_ids.push_back(id);
                continue;
            }
            if (!tryBroadcastSpill(job)) {
                job.state = SpillStageState::ABORTING;
                terminate_ids.push_back(id);
                continue;
            }
            job.state = SpillStageState::STAGING;
        }

        // STAGING: timeout if no progress
        if (job.state == SpillStageState::STAGING) {
            const auto elapsed_ms =
                std::chrono::duration_cast<std::chrono::milliseconds>(now - job.staging_done_at).count();
            if (elapsed_ms > config_.stage_ack_timeout_ms) {
                if (!job.local_pwrite_done) {
                    // local pwrite not done -> push to PWRITE_INFLIGHT so we wait commit_timeout next
                }
                job.state = SpillStageState::PWRITE_INFLIGHT;
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
                    std::chrono::duration_cast<std::chrono::milliseconds>(now - job.created_at).count();
                if (elapsed_ms > config_.commit_timeout_ms) {
                    job.state = SpillStageState::ABORTING;
                    terminate_ids.push_back(id);
                }
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
        std::unique_lock<std::mutex> dummy_lock(mutex_, std::adopt_lock);  // already locked
        dummy_lock.release();
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

bool DiskSpillCommitCoordinator::tryDispatchLocalPwrite(SpillJob& job) {
    if (!cache_) {
        return false;
    }
    const auto fms = cache_->fileManagers();
    if (job.slot.disk_id < 0 || static_cast<size_t>(job.slot.disk_id) >= fms.size()) {
        return false;
    }
    const auto workers = cache_->ioWorkers();
    if (static_cast<size_t>(job.slot.disk_id) >= workers.size() || !workers[job.slot.disk_id]) {
        return false;
    }
    const auto id      = job.id;
    auto*      self    = this;
    auto       staging = job.staging_data;  // shared_ptr capture keeps buffer alive past job erase
    {
        std::lock_guard<std::mutex> lk(pending_callbacks_mutex_);
        ++pending_callbacks_;
    }
    const bool ok = workers[job.slot.disk_id]->submitWrite(
        job.slot.slot_id,
        staging->data(),
        job.slot.block_size,
        [self, id, staging](bool ok, const std::string& /*err*/) {
            self->onLocalPwriteComplete(id, ok);
            std::lock_guard<std::mutex> lk(self->pending_callbacks_mutex_);
            --self->pending_callbacks_;
            self->pending_callbacks_cv_.notify_all();
        });
    if (!ok) {
        std::lock_guard<std::mutex> lk(pending_callbacks_mutex_);
        --pending_callbacks_;
        pending_callbacks_cv_.notify_all();
    }
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

bool DiskSpillCommitCoordinator::tryBroadcastSpill(SpillJob& job) {
    if (!spill_fn_) {
        // no worker fanout (e.g. TP=1) — treat as instantly acked
        for (auto& [rank, status] : job.worker_status) {
            status = SpillWriteStatus::SUCCESS;
        }
        job.spill_broadcast_sent = true;
        return true;
    }
    job.spill_broadcast_sent = spill_fn_(job.id, job.slot);
    return job.spill_broadcast_sent;
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
