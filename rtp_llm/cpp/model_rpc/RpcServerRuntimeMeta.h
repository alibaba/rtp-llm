#pragma once
#include <algorithm>
#include <atomic>
#include <functional>
#include <list>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "autil/EnvUtil.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/engine_base/schedulers/EngineScheduleInfo.h"

namespace rtp_llm {

struct TaskIdentity {
    const int64_t request_id;
    const int64_t batch_id;
};

struct RunningEntry {
    EngineScheduleInfo::TaskInfo task_info;
    GenerateStreamPtr            stream;
    // S-1 single-record invariant: latched when the aging sweep published the
    // typed CANCELED record while this live entry was preserved. The stream's
    // own later dequeue() then removes the entry without publishing a second
    // (untyped) finished record. C++-side bookkeeping only; never serialized.
    bool canceled_published = false;
};

// Control-record overlay installed by priority Cancel. installed_at_ms anchors
// the aging sweep: it is written once at installation and deliberately NOT
// refreshed by dequeue()'s metric refresh, so an overlay whose finalizer chain
// stalled keeps aging toward sweepStalePriorityOverlays.
struct PriorityPreemptionOverlay {
    EngineScheduleInfo::TaskInfo task_info;
    int64_t                      installed_at_ms = 0;
};

class RpcServerRuntimeMeta {
public:
    // Engine execution time: wall time from task begin to finish, minus the time
    // spent queued. This isolates the NormalEngine execution cost from queueing.
    static int64_t computeExecutionTimeMs(int64_t finish_time_ms, int64_t begin_time_us, int64_t waiting_time_ms) {
        return finish_time_ms - begin_time_us / 1000 - waiting_time_ms;
    }

    static TaskPhase derivePhase(const GenerateStreamPtr& stream) {
        if (!stream)
            return TaskPhase::PENDING;
        if (stream->getStatus() == StreamState::RUNNING)
            return TaskPhase::RUNNING;
        if (stream->curBlocksNum() > 0)
            return TaskPhase::KV_ALLOCATED;
        return TaskPhase::RECEIVED;
    }

    EngineScheduleInfo getEngineScheduleInfo(int64_t latest_finished_version) {
        // Ledger backstop for every zombie path (stuck Fetch thread, stuck
        // RemoteFinish, wedged scheduler, dropped finalizer): a CANCELING
        // overlay whose finalizer never completes is aged out here so
        // WorkerStatus accounting cannot pin the request forever. Runs in its
        // own critical section before the read lock below (no lock nesting).
        sweepStalePriorityOverlays(autil::TimeUtility::currentTimeInMilliSeconds(), priorityOverlaySweepMaxAgeMs());
        std::shared_lock<std::shared_mutex> lock(read_write_lock_);
        EngineScheduleInfo                  info;
        std::unordered_set<int64_t>         emitted_preemption_overlays;
        const int64_t                       now_ms              = autil::TimeUtility::currentTimeInMilliSeconds();
        size_t                              stale_overlay_count = 0;
        int64_t                             oldest_stale_age_ms = 0;
        for (auto& [id, entry] : running_streams_) {
            auto task_info  = entry.task_info;
            task_info.phase = derivePhase(entry.stream);
            auto overlay    = priority_preemption_overlays_.find(id);
            if (overlay != priority_preemption_overlays_.end()) {
                task_info.priority_preemption_progress = PriorityPreemptionProgress::CANCELING;
                emitted_preemption_overlays.insert(id);
                accountOverlayAge(now_ms, overlay->second, stale_overlay_count, oldest_stale_age_ms);
            }
            info.running_task_info_list.push_back(std::move(task_info));
        }
        // A Prefill request remains the priority-cancel control record even
        // when it has no local stream yet (Stage 2), or its local stream has
        // already been dequeued while Decode is generating (Stage 4). Emit an
        // overlay-only TaskInfo without inserting a synthetic engine runtime
        // entry; it therefore does not change scheduler/load accounting.
        for (const auto& [request_id, overlay] : priority_preemption_overlays_) {
            if (emitted_preemption_overlays.find(request_id) == emitted_preemption_overlays.end()) {
                info.running_task_info_list.push_back(overlay.task_info);
                accountOverlayAge(now_ms, overlay, stale_overlay_count, oldest_stale_age_ms);
            }
        }
        logStaleOverlaysRateLimited(now_ms, stale_overlay_count, oldest_stale_age_ms);
        int64_t version = latest_finished_version;
        for (auto& iter : finished_streams_) {
            if (iter.first > latest_finished_version) {
                info.finished_task_info_list.push_back(iter.second);
                if (iter.first > version) {
                    version = iter.first;
                }
            }
        }
        info.latest_finished_version = version;
        return info;
    }

    void enqueue(int64_t request_id, const GenerateStreamPtr& stream) {
        enqueue(TaskIdentity{request_id, stream->generateInput()->group_id}, stream);
    }

    void enqueue(const TaskIdentity& identity, const GenerateStreamPtr& stream) {
        std::unique_lock<std::shared_mutex> lock(read_write_lock_);
        const auto                          stream_batch_id = stream->generateInput()->group_id;
        const auto                          batch_id        = resolveBatchId(identity, stream_batch_id);
        auto                                new_task        = makeTaskInfo(TaskIdentity{identity.request_id, batch_id},
                                     stream->prefixLength(),
                                     stream->inputLength(),
                                     stream->getTimeInfo().wait_time_us);
        running_streams_[identity.request_id]               = RunningEntry{std::move(new_task), stream};
    }

    // WorkerStatus control overlay for the original Prefill. This does not
    // mutate running_streams_, so accepting Cancel cannot inflate engine load
    // or resource accounting.
    void markPriorityPreemptionCanceling(const TaskIdentity& identity) {
        std::unique_lock<std::shared_mutex> lock(read_write_lock_);
        const auto                          request_id = identity.request_id;
        if (priority_preemption_overlays_.find(request_id) != priority_preemption_overlays_.end()) {
            return;
        }
        auto task_info = makeTaskInfo(identity,
                                      /*prefix_length=*/0,
                                      /*input_length=*/0,
                                      /*waiting_time_ms=*/0);
        auto running   = running_streams_.find(request_id);
        if (running != running_streams_.end()) {
            task_info = running->second.task_info;
        } else {
            for (auto it = finished_streams_.rbegin(); it != finished_streams_.rend(); ++it) {
                if (it->second.request_id == request_id) {
                    task_info = it->second;
                    break;
                }
            }
        }
        task_info.end_time_ms = -1;
        task_info.error_code  = 0;
        task_info.error_message.clear();
        task_info.priority_preemption_progress = PriorityPreemptionProgress::CANCELING;
        PriorityPreemptionOverlay overlay;
        overlay.task_info       = std::move(task_info);
        overlay.installed_at_ms = autil::TimeUtility::currentTimeInMilliSeconds();
        priority_preemption_overlays_.emplace(request_id, std::move(overlay));
    }

    // Publish the single authoritative completion delta for priority Cancel.
    // The caller must invoke this only after the Prefill request execution has
    // quiesced and its local/downstream cleanup path has returned.
    //
    // Return semantics (P1-1 defect A fix): true when this call closed the
    // control record — including the sweep-meets-late-finalizer rendezvous
    // where the aging sweep already consumed the overlay but the paired
    // running entry still lingers: that entry is removed HERE so it can never
    // leak permanently (the late finalizer's stream_.reset() skips the
    // ordinary dequeue teardown). false only when neither overlay nor running
    // entry exists, i.e. the request is already fully closed.
    bool markPriorityPreemptionCanceled(int64_t request_id, int64_t error_code, const std::string& error_message) {
        std::unique_lock<std::shared_mutex> lock(read_write_lock_);
        auto                                overlay = priority_preemption_overlays_.find(request_id);
        if (overlay == priority_preemption_overlays_.end()) {
            auto running = running_streams_.find(request_id);
            if (running == running_streams_.end()) {
                return false;
            }
            if (running->second.canceled_published) {
                // The sweep already published the typed CANCELED record while
                // preserving this live entry (single-record invariant): remove
                // the entry WITHOUT publishing anything.
                running_streams_.erase(running);
                return true;
            }
            // Defensive branch, unreachable through current callers (the only
            // overlay consumer besides this method is the sweep, which always
            // latches canceled_published on a preserved entry). If some future
            // path still drops the overlay without publishing, guarantee at
            // least one terminal record by reusing the finalizer's own
            // merge-and-publish body rather than silently erasing the entry.
            publishCanceledRecordLocked(request_id,
                                        running->second.task_info,
                                        error_code,
                                        error_message,
                                        /*keep_live_running_entry=*/false);
            return true;
        }
        auto task_info = overlay->second.task_info;
        priority_preemption_overlays_.erase(overlay);
        // Complete the control record and remove a still-visible Prefill
        // runtime entry in one critical section. Calling dequeue() first would
        // publish an untyped finished record and then a second typed CANCELED
        // record for the same request.
        publishCanceledRecordLocked(request_id,
                                    std::move(task_info),
                                    error_code,
                                    error_message,
                                    /*keep_live_running_entry=*/false);
        return true;
    }

    void dequeue(int64_t request_id, const GenerateStreamPtr& stream) {
        std::unique_lock<std::shared_mutex> lock(read_write_lock_);
        auto                                ptr = running_streams_.find(request_id);
        if (ptr == running_streams_.end()) {
            return;
        }
        // S-1 single-record invariant (P1-1 defect B fix): the aging sweep
        // already published the typed CANCELED record while this live stream
        // kept running. Ordinary teardown must only remove the entry — never
        // emit a second untyped finished record for the same request.
        if (ptr->second.canceled_published) {
            running_streams_.erase(ptr);
            return;
        }
        auto&   task_info           = ptr->second.task_info;
        int64_t current             = autil::TimeUtility::currentTimeInMilliSeconds();
        task_info.end_time_ms       = current;
        task_info.prefix_length     = stream->prefixLength();
        task_info.input_length      = stream->inputLength();
        task_info.waiting_time_ms   = stream->getTimeInfo().wait_time_us / 1000;
        task_info.iterate_count     = stream->iterCount();
        task_info.execution_time_ms = computeExecutionTimeMs(current, stream->beginTimeUs(), task_info.waiting_time_ms);

        auto overlay = priority_preemption_overlays_.find(request_id);
        if (overlay != priority_preemption_overlays_.end()) {
            // Once priority Cancel has published CANCELING, ordinary stream
            // teardown must not emit an untyped terminal record. Preserve the
            // latest runtime metrics in the control overlay; the priority
            // finalizer will publish the one authoritative CANCELED record.
            overlay->second.task_info             = task_info;
            overlay->second.task_info.end_time_ms = -1;
            overlay->second.task_info.error_code  = 0;
            overlay->second.task_info.error_message.clear();
            overlay->second.task_info.priority_preemption_progress = PriorityPreemptionProgress::CANCELING;
            // installed_at_ms is intentionally NOT refreshed here: the aging
            // anchor must stay at installation time so a stalled finalizer
            // keeps aging toward sweepStalePriorityOverlays.
            running_streams_.erase(ptr);
            return;
        }

        if (finished_streams_.size() >= finished_capacity_) {
            finished_streams_.pop_front();
        }
        if (stream->hasError()) {
            task_info.error_code    = static_cast<int64_t>(stream->statusInfo().code());
            task_info.error_message = stream->statusInfo().ToString();
        }

        int64_t version = version_.fetch_add(1, std::memory_order_relaxed);
        finished_streams_.push_back(std::make_pair(version, task_info));
        running_streams_.erase(ptr);
    }

    void finishTask(int64_t            request_id,
                    int64_t            input_length  = 0,
                    int64_t            prefix_length = 0,
                    int64_t            error_code    = 0,
                    const std::string& error_message = "") {
        std::unique_lock<std::shared_mutex> lock(read_write_lock_);
        EngineScheduleInfo::TaskInfo        task_info{request_id,
                                               prefix_length,
                                               input_length,
                                               /*waiting_time_ms=*/0,
                                               /*iterate_count=*/0,
                                               /*end_time_ms=*/-1};
        auto                                ptr = running_streams_.find(request_id);
        if (ptr != running_streams_.end()) {
            task_info = ptr->second.task_info;
            if (input_length > 0) {
                task_info.input_length = input_length;
            }
            if (prefix_length > 0) {
                task_info.prefix_length = prefix_length;
            }
            running_streams_.erase(ptr);
        }
        if (finished_streams_.size() >= finished_capacity_) {
            finished_streams_.pop_front();
        }
        task_info.end_time_ms   = autil::TimeUtility::currentTimeInMilliSeconds();
        task_info.error_code    = error_code;
        task_info.error_message = error_message;
        int64_t version         = version_.fetch_add(1, std::memory_order_relaxed);
        finished_streams_.push_back(std::make_pair(version, task_info));
    }

    // Ledger backstop for every zombie path (Z1..Z5): if the finalizer chain
    // never completes, age the CANCELING overlay out and publish the typed
    // CANCELED record (error_code=PRIORITY_PREEMPTED) so WorkerStatus/finished
    // accounting cannot pin the request forever. Public because tests drive it
    // with an explicit clock; production entry is getEngineScheduleInfo above.
    // Idempotent with the finalizer chain: once the overlay is erased here, a
    // late markPriorityPreemptionCanceled either finds a preserved running
    // entry (canceled_published latch -> silent removal) or nothing (false).
    //
    // S-2 DRIVER CAVEAT: this sweep is driven by getEngineScheduleInfo, i.e.
    // the master's WorkerStatus polling loop. If the master stops polling,
    // this backstop stops firing as well — it is not a local timer.
    size_t sweepStalePriorityOverlays(int64_t now_ms, int64_t max_age_ms) {
        if (max_age_ms <= 0) {
            return 0;
        }
        std::vector<int64_t> stale_request_ids;
        int64_t              oldest_stale_age_ms = 0;
        {
            std::unique_lock<std::shared_mutex> lock(read_write_lock_);
            if (priority_preemption_overlays_.empty()) {
                return 0;
            }
            for (const auto& [request_id, overlay] : priority_preemption_overlays_) {
                if (now_ms - overlay.installed_at_ms > max_age_ms) {
                    stale_request_ids.push_back(request_id);
                }
            }
            // Two passes: collect first, then erase, so the map iterator is
            // never invalidated mid-scan.
            for (int64_t request_id : stale_request_ids) {
                auto overlay = priority_preemption_overlays_.find(request_id);
                if (overlay == priority_preemption_overlays_.end()) {
                    continue;
                }
                auto task_info      = overlay->second.task_info;
                oldest_stale_age_ms = std::max(oldest_stale_age_ms, now_ms - overlay->second.installed_at_ms);
                priority_preemption_overlays_.erase(overlay);
                // keep_live_running_entry=true: never touch a live stream's
                // runtime entry. Its own teardown will close that entry later;
                // the sweep only closes the control record (and latches
                // canceled_published on the preserved entry for the S-1
                // single-record invariant).
                publishCanceledRecordLocked(request_id,
                                            std::move(task_info),
                                            static_cast<int64_t>(ErrorCode::PRIORITY_PREEMPTED),
                                            "priority preemption overlay aged out",
                                            /*keep_live_running_entry=*/true);
            }
        }
        if (!stale_request_ids.empty()) {
            // P2: sweep counter as a log record (this class has no metrics
            // reporter channel; LocalRpcServer only aggregates engine-level
            // reporters, not runtime-meta internals).
            RTP_LLM_LOG_WARNING("swept %zu stale priority-preemption overlay(s), oldest age [%ld ms], "
                                "max_age_ms [%ld]",
                                stale_request_ids.size(),
                                oldest_stale_age_ms,
                                max_age_ms);
            // W-2 registry linkage: mirror the aged-CANCELED publication into
            // the deferred-context registry (drop the active entry so a lost
            // finalizer task cannot keep it referenced until shutdown).
            // Deliberately OUTSIDE read_write_lock_: the cancel chain takes
            // registry-mu -> terminal-transition-mu -> read_write_lock_, and
            // calling the hook under the sweep's lock would invert that order.
            if (stale_overlay_swept_hook_) {
                for (int64_t request_id : stale_request_ids) {
                    stale_overlay_swept_hook_(request_id);
                }
            }
        }
        return stale_request_ids.size();
    }

    // W-2 linkage hook (see sweepStalePriorityOverlays). Installed once during
    // server init — before the WorkerStatus polling threads that drive
    // getEngineScheduleInfo start — and only read afterwards, so plain
    // std::function assignment is sufficient (no lock needed).
    void setStaleOverlaySweptHook(std::function<void(int64_t)> hook) {
        stale_overlay_swept_hook_ = std::move(hook);
    }

protected:
    static int64_t resolveBatchId(const TaskIdentity& identity, int64_t stream_batch_id) {
        if (identity.batch_id >= 0) {
            if (stream_batch_id >= 0 && stream_batch_id != identity.batch_id) {
                RTP_LLM_LOG_WARNING("task batch identity mismatch: request_id=%ld envelope_batch_id=%ld "
                                    "stream_batch_id=%ld; keeping envelope identity",
                                    identity.request_id,
                                    identity.batch_id,
                                    stream_batch_id);
            }
            return identity.batch_id;
        }
        return stream_batch_id;
    }

    static EngineScheduleInfo::TaskInfo
    makeTaskInfo(const TaskIdentity& identity, int64_t prefix_length, int64_t input_length, int64_t waiting_time_ms) {
        EngineScheduleInfo::TaskInfo task_info{identity.request_id, prefix_length, input_length, waiting_time_ms};
        task_info.batch_id = identity.batch_id;
        return task_info;
    }

    // read_write_lock_ must be held (unique). Shared publish body of the
    // authoritative typed CANCELED record, used by both the finalizer path
    // (markPriorityPreemptionCanceled) and the aging sweep.
    //
    // keep_live_running_entry=false reproduces the historical finalizer
    // semantics: the paired running entry is always merged and removed (the
    // finalizer only runs after the stream quiesced).
    // keep_live_running_entry=true (sweep) removes the running entry only when
    // its stream is already gone or FINISHED; a live stream is never disturbed
    // by the sweep — the published record then carries the overlay snapshot
    // and the preserved entry is latched canceled_published so its own later
    // dequeue() cannot emit a second untyped record (S-1).
    void publishCanceledRecordLocked(int64_t                      request_id,
                                     EngineScheduleInfo::TaskInfo task_info,
                                     int64_t                      error_code,
                                     const std::string&           error_message,
                                     bool                         keep_live_running_entry) {
        auto running = running_streams_.find(request_id);
        if (running != running_streams_.end()) {
            const auto& stream    = running->second.stream;
            const bool  removable = !keep_live_running_entry || !stream || stream->getStatus() == StreamState::FINISHED;
            if (removable) {
                task_info = running->second.task_info;
                if (stream) {
                    const int64_t current     = autil::TimeUtility::currentTimeInMilliSeconds();
                    task_info.end_time_ms     = current;
                    task_info.prefix_length   = stream->prefixLength();
                    task_info.input_length    = stream->inputLength();
                    task_info.waiting_time_ms = stream->getTimeInfo().wait_time_us / 1000;
                    task_info.iterate_count   = stream->iterCount();
                    task_info.execution_time_ms =
                        computeExecutionTimeMs(current, stream->beginTimeUs(), task_info.waiting_time_ms);
                }
                running_streams_.erase(running);
            } else {
                // Sweep publication over a LIVE stream: the typed CANCELED
                // record is published now; latch canceled_published so this
                // entry's later ordinary dequeue() only removes it (S-1
                // single-record invariant).
                running->second.canceled_published = true;
            }
        }
        if (task_info.end_time_ms < 0) {
            task_info.end_time_ms = autil::TimeUtility::currentTimeInMilliSeconds();
        }
        task_info.error_code                   = error_code;
        task_info.error_message                = error_message;
        task_info.priority_preemption_progress = PriorityPreemptionProgress::CANCELED;
        if (finished_streams_.size() >= finished_capacity_) {
            finished_streams_.pop_front();
        }
        int64_t version = version_.fetch_add(1, std::memory_order_relaxed);
        finished_streams_.push_back(std::make_pair(version, std::move(task_info)));
    }

    // RTP_LLM_PRIORITY_OVERLAY_SWEEP_MS: max age of a CANCELING overlay before
    // the ledger sweep publishes CANCELED for it; <=0 disables the sweep.
    static int64_t priorityOverlaySweepMaxAgeMs() {
        static const int64_t max_age_ms = autil::EnvUtil::getEnv("RTP_LLM_PRIORITY_OVERLAY_SWEEP_MS", int64_t(300000));
        return max_age_ms;
    }

    // A CANCELING overlay older than this is considered stuck (the finalizer
    // chain should settle within seconds) and becomes an observability signal.
    static constexpr int64_t kStalePriorityOverlayAgeMs = 60 * 1000;
    // Rate limit for the stale-overlay summary log (one line per window).
    static constexpr int64_t kStaleOverlayLogIntervalMs = 60 * 1000;

    static void accountOverlayAge(int64_t                          now_ms,
                                  const PriorityPreemptionOverlay& overlay,
                                  size_t&                          stale_count,
                                  int64_t&                         oldest_stale_age_ms) {
        const int64_t age_ms = now_ms - overlay.installed_at_ms;
        if (age_ms > kStalePriorityOverlayAgeMs) {
            ++stale_count;
            oldest_stale_age_ms = std::max(oldest_stale_age_ms, age_ms);
        }
    }

    void logStaleOverlaysRateLimited(int64_t now_ms, size_t stale_count, int64_t oldest_stale_age_ms) {
        if (stale_count == 0) {
            return;
        }
        int64_t last = last_stale_overlay_log_ms_.load(std::memory_order_relaxed);
        if (now_ms - last < kStaleOverlayLogIntervalMs) {
            return;
        }
        if (last_stale_overlay_log_ms_.compare_exchange_strong(last, now_ms, std::memory_order_relaxed)) {
            RTP_LLM_LOG_INFO("priority-preemption overlays stuck in CANCELING: count [%zu], oldest age [%ld ms], "
                             "stale threshold [%ld ms]",
                             stale_count,
                             oldest_stale_age_ms,
                             kStalePriorityOverlayAgeMs);
        }
    }

    void trimFinishedStreams() {
        auto current = autil::TimeUtility::currentTimeInMilliSeconds();
        auto iter    = finished_streams_.begin();
        while (iter != finished_streams_.end()) {
            int64_t end_time_ms = iter->second.end_time_ms;
            if (end_time_ms > current) {
                RTP_LLM_LOG_WARNING("find task: %ld end time: %ld bigger than current time: %ld",
                                    iter->second.request_id,
                                    end_time_ms,
                                    current);
                iter = finished_streams_.erase(iter);
            } else if (end_time_ms + timeout_ms_ <= current) {
                iter = finished_streams_.erase(iter);
            } else {
                break;
            }
        }
    }
    std::unordered_map<int64_t, RunningEntry>                   running_streams_;
    std::unordered_map<int64_t, PriorityPreemptionOverlay>      priority_preemption_overlays_;
    std::list<std::pair<int64_t, EngineScheduleInfo::TaskInfo>> finished_streams_;
    std::atomic<int64_t>      version_{autil::TimeUtility::currentTimeInMicroSeconds()};
    mutable std::shared_mutex read_write_lock_;
    int64_t                   timeout_ms_        = 5000;
    int64_t                   finished_capacity_ = 1000;
    // Rate-limit anchor for the stale-overlay observability log (P2).
    std::atomic<int64_t> last_stale_overlay_log_ms_{0};
    // W-2 registry linkage hook, invoked by sweepStalePriorityOverlays after
    // releasing read_write_lock_ (see the method comment for lock-order
    // rationale). Installed once at init; see setStaleOverlaySweptHook.
    std::function<void(int64_t)> stale_overlay_swept_hook_;
};

};  // namespace rtp_llm
