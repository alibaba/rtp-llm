#pragma once
#include <atomic>
#include <list>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
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
        const auto time_info       = stream->getTimeInfo();
        const auto stream_batch_id = stream->generateInput()->group_id;
        const auto batch_id        = resolveBatchId(identity, stream_batch_id);
        auto       new_task        = makeTaskInfo(TaskIdentity{identity.request_id, batch_id},
                                     stream->prefixLength(),
                                     stream->inputLength(),
                                     time_info.wait_time_us / 1000);

        std::unique_lock<std::shared_mutex> lock(read_write_lock_);
        running_streams_[identity.request_id] = RunningEntry{std::move(new_task), stream};
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
    // quiesced and its local/downstream cleanup path has returned. `stream`
    // must be the registered stream, or null when no local stream was enqueued.
    bool markPriorityPreemptionCanceled(int64_t                  request_id,
                                        int64_t                  error_code,
                                        const std::string&       error_message,
                                        const GenerateStreamPtr& stream) {
        StreamRuntimeSnapshot stream_snapshot;
        const bool            has_stream_snapshot = stream != nullptr;
        if (has_stream_snapshot) {
            stream_snapshot = captureStreamRuntimeSnapshot(stream);
        }

        std::unique_lock<std::shared_mutex> lock(read_write_lock_);
        auto                                overlay = priority_preemption_overlays_.find(request_id);
        if (overlay == priority_preemption_overlays_.end()) {
            return false;
        }
        auto running   = running_streams_.find(request_id);
        auto task_info = overlay->second;
        priority_preemption_overlays_.erase(overlay);
        // Complete the control record and remove a still-visible Prefill
        // runtime entry in one critical section. Calling dequeue() first would
        // publish an untyped finished record and then a second typed CANCELED
        // record for the same request.
        if (running != running_streams_.end() && has_stream_snapshot && running->second.stream == stream) {
            task_info = running->second.task_info;
            applyStreamRuntimeSnapshot(task_info, stream_snapshot);
            running_streams_.erase(running);
        } else if (has_stream_snapshot) {
            // The request id may already belong to a replacement stream. The
            // overlay still belongs to the canceled request, so finalize it
            // from the preserved stream without consuming the replacement.
            applyStreamRuntimeSnapshot(task_info, stream_snapshot);
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

    void dequeue(int64_t request_id, const GenerateStreamPtr& stream) {
        if (!stream) {
            return;
        }
        {
            std::shared_lock<std::shared_mutex> lock(read_write_lock_);
            const auto                          running = running_streams_.find(request_id);
            if (running == running_streams_.end() || running->second.stream != stream) {
                return;
            }
        }
        const auto stream_snapshot = captureStreamRuntimeSnapshot(stream);
        commitDequeueSnapshot(request_id, stream, stream_snapshot);
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
    struct StreamRuntimeSnapshot {
        int64_t   end_time_ms     = -1;
        int64_t   begin_time_us   = 0;
        int64_t   waiting_time_ms = 0;
        int64_t   prefix_length   = 0;
        int64_t   input_length    = 0;
        size_t    iterate_count   = 0;
        ErrorInfo status;
    };

    static StreamRuntimeSnapshot captureStreamRuntimeSnapshot(const GenerateStreamPtr& stream) {
        // Read the coherent TimeInfo before the wall clock. resetBeginTime()
        // uses the same stream mutex, so the sampled current time cannot precede
        // the captured begin epoch during a production reset.
        const auto time_info = stream->getTimeInfo();

        StreamRuntimeSnapshot snapshot;
        snapshot.end_time_ms     = autil::TimeUtility::currentTimeInMilliSeconds();
        snapshot.begin_time_us   = time_info.begin_time_us;
        snapshot.waiting_time_ms = time_info.wait_time_us / 1000;
        snapshot.prefix_length   = stream->prefixLength();
        snapshot.input_length    = stream->inputLength();
        snapshot.iterate_count   = stream->iterCount();
        snapshot.status          = stream->statusInfo();
        return snapshot;
    }

    static void applyStreamRuntimeSnapshot(EngineScheduleInfo::TaskInfo& task_info,
                                           const StreamRuntimeSnapshot&  snapshot) {
        task_info.end_time_ms     = snapshot.end_time_ms;
        task_info.prefix_length   = snapshot.prefix_length;
        task_info.input_length    = snapshot.input_length;
        task_info.waiting_time_ms = snapshot.waiting_time_ms;
        task_info.iterate_count   = snapshot.iterate_count;
        task_info.execution_time_ms =
            computeExecutionTimeMs(snapshot.end_time_ms, snapshot.begin_time_us, snapshot.waiting_time_ms);
    }

    void commitDequeueSnapshot(int64_t                      request_id,
                               const GenerateStreamPtr&     stream,
                               const StreamRuntimeSnapshot& stream_snapshot) {
        std::unique_lock<std::shared_mutex> lock(read_write_lock_);
        auto                                ptr = running_streams_.find(request_id);
        if (ptr == running_streams_.end() || ptr->second.stream != stream) {
            return;
        }
        auto& task_info = ptr->second.task_info;
        applyStreamRuntimeSnapshot(task_info, stream_snapshot);

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
            running_streams_.erase(ptr);
            return;
        }

        if (finished_streams_.size() >= finished_capacity_) {
            finished_streams_.pop_front();
        }
        if (stream_snapshot.status.hasError()) {
            task_info.error_code    = static_cast<int64_t>(stream_snapshot.status.code());
            task_info.error_message = stream_snapshot.status.ToString();
        }

        int64_t version = version_.fetch_add(1, std::memory_order_relaxed);
        finished_streams_.push_back(std::make_pair(version, task_info));
        running_streams_.erase(ptr);
    }

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
