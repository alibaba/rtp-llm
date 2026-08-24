#pragma once
#include <atomic>
#include <list>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/engine_base/schedulers/EngineScheduleInfo.h"

namespace rtp_llm {

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

    // Current KV usage in tokens = allocated blocks * block size. Block size
    // comes from stream->seqSizePerBlock(), i.e. the KVCacheManager config the
    // stream was created with (production streams always carry a cache_manager;
    // unit-test streams built with an empty ResourceContext do not, so guard
    // and report 0 instead of dereferencing a null cache_manager).
    static int64_t deriveKvTokens(const GenerateStreamPtr& stream) {
        if (!stream || !stream->resourceContext().cache_manager) {
            return 0;
        }
        return static_cast<int64_t>(stream->curBlocksNum()) * stream->seqSizePerBlock();
    }

    // Final KV usage in tokens for a finished stream. By the time dequeue runs,
    // GenerateStateMachine has already released the KV blocks (clearBlocks),
    // so curBlocksNum() would read 0. Derive the peak block-aligned footprint
    // from the final sequence length instead: ceil(seq_len / block_size)
    // blocks, each holding block_size tokens (the last partially-filled block
    // still occupies a full block), matching the running-entry semantics of
    // blocks * block size.
    static int64_t deriveFinalKvTokens(const GenerateStreamPtr& stream) {
        if (!stream || !stream->resourceContext().cache_manager) {
            return 0;
        }
        const int64_t block_size = stream->seqSizePerBlock();
        if (block_size <= 0) {
            return 0;
        }
        const int64_t seq_len = stream->seqLength();
        return ((seq_len + block_size - 1) / block_size) * block_size;
    }

    EngineScheduleInfo getEngineScheduleInfo(int64_t latest_finished_version) {
        std::shared_lock<std::shared_mutex> lock(read_write_lock_);
        EngineScheduleInfo                  info;
        // Value copy (not `auto&`): the snapshot must not write the derived
        // phase / kv_tokens back into running_streams_. This keeps the protocol
        // invariant that a finished entry keeps phase == PENDING (the value it
        // was enqueued with) instead of leaking the last snapshot's derived
        // phase into finished reports.
        for (const auto& [id, entry] : running_streams_) {
            auto task_info      = entry.task_info;
            task_info.phase     = derivePhase(entry.stream);
            task_info.kv_tokens = deriveKvTokens(entry.stream);
            info.running_task_info_list.push_back(std::move(task_info));
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
        std::unique_lock<std::shared_mutex> lock(read_write_lock_);
        auto                                new_task = EngineScheduleInfo::TaskInfo(
            {request_id, stream->prefixLength(), stream->inputLength(), stream->getTimeInfo().wait_time_us});
        new_task.batch_id            = stream->generateInput()->group_id;
        running_streams_[request_id] = RunningEntry{new_task, stream};
    }

    void dequeue(int64_t request_id, const GenerateStreamPtr& stream) {
        std::unique_lock<std::shared_mutex> lock(read_write_lock_);
        auto                                ptr = running_streams_.find(request_id);
        if (ptr == running_streams_.end()) {
            return;
        }
        auto& task_info = ptr->second.task_info;
        if (finished_streams_.size() >= finished_capacity_) {
            finished_streams_.pop_front();
        }
        int64_t current             = autil::TimeUtility::currentTimeInMilliSeconds();
        task_info.end_time_ms       = current;
        task_info.prefix_length     = stream->prefixLength();
        task_info.input_length      = stream->inputLength();
        task_info.waiting_time_ms   = stream->getTimeInfo().wait_time_us / 1000;
        task_info.iterate_count     = stream->iterCount();
        task_info.execution_time_ms = computeExecutionTimeMs(current, stream->beginTimeUs(), task_info.waiting_time_ms);
        task_info.kv_tokens         = deriveFinalKvTokens(stream);
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

protected:
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
    std::list<std::pair<int64_t, EngineScheduleInfo::TaskInfo>> finished_streams_;
    std::atomic<int64_t>      version_{autil::TimeUtility::currentTimeInMicroSeconds()};
    mutable std::shared_mutex read_write_lock_;
    int64_t                   timeout_ms_        = 5000;
    int64_t                   finished_capacity_ = 1000;
};

};  // namespace rtp_llm
