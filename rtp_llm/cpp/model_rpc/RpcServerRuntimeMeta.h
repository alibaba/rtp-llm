#pragma once
#include <atomic>
#include <cstdint>
#include <list>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/engine_base/schedulers/EngineScheduleInfo.h"

namespace rtp_llm {

// Verification-log sampling state for the A2 event-driven runtime-meta
// promotion fix (the finish callback registered by
// PrefillGenerateContext::setStream). Pure observability: cumulative atomic
// counters plus time-throttled INFO lines only; no behavior change.
// C++17 inline variables keep one process-wide instance across every
// translation unit that includes this header. Emission policy: at most one
// INFO line per event per kPromotionLogIntervalMs (<= 2 lines/s combined), so
// thousands of callback fires per second cannot flood the log, while every
// emitted line carries the cumulative counters so the evidence stays complete
// between samples.
namespace promotion_log_detail {
constexpr int64_t kPromotionLogIntervalMs = 1000;

inline std::atomic<uint64_t> g_finish_callback_fires{0};     // callback-driven dequeue() invocations
inline std::atomic<uint64_t> g_finish_callback_promoted{0};  // fires that migrated the entry out of running_streams_
inline std::atomic<int64_t>  g_finish_callback_last_log_ms{0};
inline std::atomic<uint64_t> g_fetch_dequeue_miss_total{0};     // fetch/teardown-driven find-miss count
inline std::atomic<uint64_t> g_fetch_dequeue_miss_promoted{0};  // misses where finished_streams_ already held the entry
inline std::atomic<int64_t>  g_fetch_dequeue_miss_last_log_ms{0};
}  // namespace promotion_log_detail

struct TaskIdentity {
    const int64_t request_id;
    const int64_t batch_id;
};

struct RunningEntry {
    EngineScheduleInfo::TaskInfo task_info;
    GenerateStreamPtr            stream;
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
        std::shared_lock<std::shared_mutex> lock(read_write_lock_);
        EngineScheduleInfo                  info;
        std::unordered_set<int64_t>         emitted_preemption_overlays;
        for (auto& [id, entry] : running_streams_) {
            auto task_info  = entry.task_info;
            task_info.phase = derivePhase(entry.stream);
            auto overlay    = priority_preemption_overlays_.find(id);
            if (overlay != priority_preemption_overlays_.end()) {
                task_info.priority_preemption_progress = PriorityPreemptionProgress::CANCELING;
                emitted_preemption_overlays.insert(id);
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
                info.running_task_info_list.push_back(overlay);
            }
        }
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
        priority_preemption_overlays_.emplace(request_id, std::move(task_info));
    }

    // Publish the single authoritative completion delta for priority Cancel.
    // The caller must invoke this only after the Prefill request execution has
    // quiesced and its local/downstream cleanup path has returned.
    bool markPriorityPreemptionCanceled(int64_t request_id, int64_t error_code, const std::string& error_message) {
        std::unique_lock<std::shared_mutex> lock(read_write_lock_);
        auto                                overlay = priority_preemption_overlays_.find(request_id);
        if (overlay == priority_preemption_overlays_.end()) {
            return false;
        }
        auto task_info = overlay->second;
        priority_preemption_overlays_.erase(overlay);
        // Complete the control record and remove a still-visible Prefill
        // runtime entry in one critical section. Calling dequeue() first would
        // publish an untyped finished record and then a second typed CANCELED
        // record for the same request.
        auto running = running_streams_.find(request_id);
        if (running != running_streams_.end()) {
            task_info          = running->second.task_info;
            const auto& stream = running->second.stream;
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
        return true;
    }

    // Migrates the runtime-meta entry of request_id out of running_streams_:
    // publishes a finished record, or hands the entry to the priority-
    // preemption overlay when Cancel installed one. Returns true when this
    // call was the migrator; false on an idempotent find-miss (the entry was
    // already migrated, e.g. by the A2 finish callback, or was never
    // enqueued). from_finish_callback=true marks the A2 event-driven
    // promotion call site (PrefillGenerateContext::setStream) so its
    // effectiveness is verifiable via sampled INFO logs; it never changes the
    // migration semantics.
    bool dequeue(int64_t request_id, const GenerateStreamPtr& stream, bool from_finish_callback = false) {
        if (from_finish_callback) {
            promotion_log_detail::g_finish_callback_fires.fetch_add(1, std::memory_order_relaxed);
        }
        std::unique_lock<std::shared_mutex> lock(read_write_lock_);
        auto                                ptr = running_streams_.find(request_id);
        if (ptr == running_streams_.end()) {
            // A find-miss on the fetch/teardown path with the entry already in
            // finished_streams_ is the direct counter-evidence that the A2
            // finish callback promoted it first (see
            // noteFetchDequeueMissLocked).
            if (!from_finish_callback) {
                noteFetchDequeueMissLocked(request_id, stream);
            }
            return false;
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
            overlay->second             = task_info;
            overlay->second.end_time_ms = -1;
            overlay->second.error_code  = 0;
            overlay->second.error_message.clear();
            overlay->second.priority_preemption_progress = PriorityPreemptionProgress::CANCELING;
            const int64_t migrated_batch_id              = task_info.batch_id;
            running_streams_.erase(ptr);
            if (from_finish_callback) {
                // The callback still migrated the entry out of running_streams_
                // (handed over to the priority finalizer), so it counts as a
                // promotion and proves the callback fired.
                noteFinishCallbackPromotedLocked(request_id, migrated_batch_id);
            }
            return true;
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
        const int64_t migrated_batch_id = task_info.batch_id;
        running_streams_.erase(ptr);
        if (from_finish_callback) {
            noteFinishCallbackPromotedLocked(request_id, migrated_batch_id);
        }
        return true;
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

    // Verification log for the A2 finish callback being the migrator: this
    // dequeue call removed the entry from running_streams_ (published a
    // finished record, or handed it to the priority-preemption overlay).
    // Time-throttled to at most one INFO per kPromotionLogIntervalMs; the
    // counters are cumulative so each sampled line still proves how many
    // fires happened and how many were promotions.
    void noteFinishCallbackPromotedLocked(int64_t request_id, int64_t batch_id) {
        const uint64_t fires = promotion_log_detail::g_finish_callback_fires.load(std::memory_order_relaxed);
        const uint64_t promoted =
            promotion_log_detail::g_finish_callback_promoted.fetch_add(1, std::memory_order_relaxed) + 1;
        const int64_t now_ms  = autil::TimeUtility::currentTimeInMilliSeconds();
        int64_t       last_ms = promotion_log_detail::g_finish_callback_last_log_ms.load(std::memory_order_relaxed);
        if (now_ms - last_ms >= promotion_log_detail::kPromotionLogIntervalMs
            && promotion_log_detail::g_finish_callback_last_log_ms.compare_exchange_strong(
                last_ms, now_ms, std::memory_order_acq_rel)) {
            RTP_LLM_LOG_INFO("event=finish_callback_promoted request_id=%ld batch_id=%ld "
                             "callback_fires_total=%lu callback_promoted_total=%lu (sampled 1/s): "
                             "A2 finish callback migrated the runtime-meta entry out of running_streams_",
                             request_id,
                             batch_id,
                             fires,
                             promoted);
        }
    }

    // Counter-evidence verification log: a fetch/teardown-driven dequeue
    // find-miss whose entry is already in finished_streams_ (with an error
    // code consistent with this stream) proves the A2 finish callback
    // promoted the entry before the FetchResponse arrived. A find-miss with
    // no matching finished record (never enqueued, or a finishTask record
    // with a different error code) stays silent by design.
    // Called with read_write_lock_ held; scans finished_streams_ from the
    // newest end, so the common case (the callback migrated this very entry
    // moments before the fetch) terminates after a few nodes.
    void noteFetchDequeueMissLocked(int64_t request_id, const GenerateStreamPtr& stream) {
        promotion_log_detail::g_fetch_dequeue_miss_total.fetch_add(1, std::memory_order_relaxed);
        if (!stream) {
            return;
        }
        for (auto it = finished_streams_.rbegin(); it != finished_streams_.rend(); ++it) {
            if (it->second.request_id != request_id) {
                continue;
            }
            const bool error_consistent =
                (it->second.error_code == 0 && !stream->hasError())
                || (stream->hasError() && it->second.error_code == static_cast<int64_t>(stream->statusInfo().code()));
            if (!error_consistent) {
                break;
            }
            const uint64_t miss_total =
                promotion_log_detail::g_fetch_dequeue_miss_total.load(std::memory_order_relaxed);
            const uint64_t promoted =
                promotion_log_detail::g_fetch_dequeue_miss_promoted.fetch_add(1, std::memory_order_relaxed) + 1;
            const int64_t now_ms = autil::TimeUtility::currentTimeInMilliSeconds();
            int64_t last_ms = promotion_log_detail::g_fetch_dequeue_miss_last_log_ms.load(std::memory_order_relaxed);
            if (now_ms - last_ms >= promotion_log_detail::kPromotionLogIntervalMs
                && promotion_log_detail::g_fetch_dequeue_miss_last_log_ms.compare_exchange_strong(
                    last_ms, now_ms, std::memory_order_acq_rel)) {
                RTP_LLM_LOG_INFO("event=fetch_found_already_promoted request_id=%ld batch_id=%ld "
                                 "dequeue_miss_total=%lu already_promoted_total=%lu (sampled 1/s): "
                                 "fetch/teardown dequeue found the runtime-meta entry already migrated "
                                 "by the A2 finish callback",
                                 request_id,
                                 it->second.batch_id,
                                 miss_total,
                                 promoted);
            }
            break;
        }
    }

    static EngineScheduleInfo::TaskInfo
    makeTaskInfo(const TaskIdentity& identity, int64_t prefix_length, int64_t input_length, int64_t waiting_time_ms) {
        EngineScheduleInfo::TaskInfo task_info{identity.request_id, prefix_length, input_length, waiting_time_ms};
        task_info.batch_id = identity.batch_id;
        return task_info;
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
    std::unordered_map<int64_t, EngineScheduleInfo::TaskInfo>   priority_preemption_overlays_;
    std::list<std::pair<int64_t, EngineScheduleInfo::TaskInfo>> finished_streams_;
    std::atomic<int64_t>      version_{autil::TimeUtility::currentTimeInMicroSeconds()};
    mutable std::shared_mutex read_write_lock_;
    int64_t                   timeout_ms_        = 5000;
    int64_t                   finished_capacity_ = 1000;
};

};  // namespace rtp_llm
