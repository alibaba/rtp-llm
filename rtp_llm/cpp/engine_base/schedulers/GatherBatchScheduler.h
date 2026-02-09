#pragma once
#include "rtp_llm/cpp/engine_base/schedulers/FIFOScheduler.h"
#include <list>
#include <unordered_map>
#include <chrono>

namespace rtp_llm {


// GatherBatchScheduler is used to gather batch of streams, and sort streams by requestId.
class GatherBatchScheduler: virtual public FIFOScheduler {
public:
    explicit GatherBatchScheduler(const RuntimeConfig&                   runtime_config,
                                  const ModelConfig&                     model_config,
                                  const PDSepConfig&                     pd_sep_config,
                                  const ParallelismConfig&               parallelism_config,
                                  const ModelSpecificConfig&             model_specific_config,
                                  const std::shared_ptr<KVCacheManager>& cache_manager,
                                  const kmonitor::MetricsReporterPtr     metrics_reporter,
                                  const int                              max_score_len = 1):
        FIFOScheduler(runtime_config,
                      model_config,
                      pd_sep_config,
                      parallelism_config,
                      model_specific_config,
                      cache_manager,
                      metrics_reporter,
                      max_score_len) {
        RTP_LLM_LOG_INFO("GatherBatchScheduler init");
    }


protected:
    void reportMetrics() override {
        if (metrics_reporter_) {
            RtpLLMSchedulerMetricsCollector collector;
            int64_t                         waiting_size = 0;
            // Lock is held by caller (schedule)
            for (const auto& group : waiting_groups_) {
                waiting_size += group.streams.size();
            }
            collector.wait_stream_size           = waiting_size;
            collector.running_stream_size        = running_streams_.size();
            collector.remote_running_stream_size = remote_running_streams_.size();
            metrics_reporter_->report<RtpLLMSchedulerMetrics, RtpLLMSchedulerMetricsCollector>(nullptr, &collector);
        }
    }

    struct ScheduleGroup {
        int64_t                      batch_group_id;
        bool                         force_batch;
        int                          expected_size;
        int64_t                      timeout_us;
        int64_t                      first_enqueue_time;
        std::list<GenerateStreamPtr> streams;
    };

    std::list<ScheduleGroup>                                        waiting_groups_;
    std::unordered_map<int64_t, std::list<ScheduleGroup>::iterator> group_map_;

protected:
    std::atomic<bool> schedule_trigger_ = false;

    bool waitPredicate() override {
        return stop_ || !running_streams_.empty() || !remote_running_streams_.empty() || schedule_trigger_;
    }

public:
    absl::StatusOr<std::list<GenerateStreamPtr>> schedule(size_t reserve_step = 0) override {
        std::unique_lock<std::mutex> lock(lock_);

        if (need_fill_fake_stream_ || !waiting_groups_.empty()) {
            cond_.wait_for(lock, std::chrono::milliseconds(10), [this] { return waitPredicate(); });
        } else {
            cond_.wait(lock, [this] { return waitPredicate(); });
        }

        schedule_trigger_ = false;

        evaluateRunningRemote();
        evictDoneStreams(waiting_streams_);
        evictDoneStreams(running_streams_);
        evictDoneStreams(remote_running_streams_);

        evaluateRunningNext(reserve_step);
        auto new_streams = scheduleNew(reserve_step);

        if (!new_streams.empty()) {
            schedule_trigger_ = true;
        }

        accountBatchMetrics(new_streams, running_streams_);
        running_streams_.insert(running_streams_.end(), new_streams.begin(), new_streams.end());
        reportMetrics();
        last_schedule_time_ = autil::TimeUtility::currentTimeInMilliSeconds();
        return running_streams_;
    }

    absl::Status enqueue(const GenerateStreamPtr& stream) override {
        {
            std::lock_guard<std::mutex> lock(lock_);
            int64_t                     gid = stream->batchGroupId();
            auto                        it  = group_map_.find(gid);
            if (it == group_map_.end()) {
                ScheduleGroup group;
                group.batch_group_id     = gid;
                group.force_batch        = stream->generateConfig()->force_batch;
                group.expected_size      = stream->batchGroupSize();
                group.timeout_us         = (int64_t)stream->batchGroupTimeout() * 1000;
                group.first_enqueue_time = stream->enqueueTime();

                waiting_groups_.emplace_back(std::move(group));
                group_map_[gid] = std::prev(waiting_groups_.end());
                it              = group_map_.find(gid);
            }
            it->second->streams.push_back(stream);
            schedule_trigger_ = true;
        }
        cond_.notify_all();
        return absl::OkStatus();
    }

    absl::Status batchEnqueue(const std::vector<GenerateStreamPtr>& streams) override {
        {
            std::lock_guard<std::mutex> lock(lock_);
            for (const auto& stream : streams) {
                int64_t gid = stream->batchGroupId();
                auto    it  = group_map_.find(gid);
                if (it == group_map_.end()) {
                    ScheduleGroup group;
                    group.batch_group_id     = gid;
                    group.force_batch        = stream->generateConfig()->force_batch;
                    group.expected_size      = stream->batchGroupSize();
                    group.timeout_us         = (int64_t)stream->batchGroupTimeout() * 1000;
                    group.first_enqueue_time = stream->enqueueTime();

                    waiting_groups_.emplace_back(std::move(group));
                    group_map_[gid] = std::prev(waiting_groups_.end());
                    it              = group_map_.find(gid);
                }
                it->second->streams.push_back(stream);
            }
            schedule_trigger_ = true;
        }
        cond_.notify_all();
        return absl::OkStatus();
    }

    bool empty() override {
        std::lock_guard<std::mutex> lock(lock_);
        return waiting_groups_.empty() && running_streams_.empty();
    }

    int64_t waitingStreamsSize() override {
        std::lock_guard<std::mutex> lock(lock_);
        int64_t                     size = 0;
        for (const auto& group : waiting_groups_) {
            size += group.streams.size();
        }
        return size;
    }

    int64_t onflightStreams() override {
        std::lock_guard<std::mutex> lock(lock_);
        int64_t                     waiting_size = 0;
        for (const auto& group : waiting_groups_) {
            waiting_size += group.streams.size();
        }
        return waiting_size + running_streams_.size();
    }

    std::vector<EngineScheduleInfo::TaskInfo> waitingTaskList() override {
        std::lock_guard<std::mutex>               lock(lock_);
        std::vector<EngineScheduleInfo::TaskInfo> task_list;
        task_list.reserve(waiting_groups_.size());
        for (const auto& group : waiting_groups_) {
            for (const auto& stream : group.streams) {
                EngineScheduleInfo::TaskInfo task_info;
                task_info.inter_request_id = stream->interRequestId();
                task_info.prefix_length    = stream->prefixLength();
                task_info.input_length     = stream->inputLength();
                task_list.emplace_back(task_info);
            }
        }
        return task_list;
    }

protected:
    void evictDoneStreams(std::list<GenerateStreamPtr>& streams) override {
        if (&streams == &waiting_streams_) {
            // Evict done streams in waiting groups
            for (auto git = waiting_groups_.begin(); git != waiting_groups_.end();) {
                for (auto sit = git->streams.begin(); sit != git->streams.end();) {
                    (*sit)->checkTimeout();
                    if ((*sit)->stopped() || (*sit)->finished()) {
                        (*sit)->releaseResource();
                        RTP_LLM_LOG_DEBUG("evict waiting stream [%ld]", (*sit)->streamId());
                        sit = git->streams.erase(sit);
                    } else {
                        ++sit;
                    }
                }
                if (git->streams.empty()) {
                    group_map_.erase(git->batch_group_id);
                    git = waiting_groups_.erase(git);
                } else {
                    ++git;
                }
            }
        } else {
            FIFOScheduler::evictDoneStreams(streams);
        }
    }

    bool canFitGroupInMemory(const std::list<GenerateStreamPtr>& current_new_streams,
                             const std::list<GenerateStreamPtr>& group_streams) const {
        std::list<GenerateStreamPtr> temp_streams = current_new_streams;
        for (const auto& s : group_streams) {
            if (!evaluateRunningMemory(temp_streams, s)) {
                return false;
            }
            temp_streams.push_back(s);
        }
        return true;
    }

    std::list<GenerateStreamPtr> scheduleNew(size_t reserve_step) override {
        int64_t current_waiting_size = 0;
        for (const auto& group : waiting_groups_) {
            current_waiting_size += group.streams.size();
        }

        std::list<GenerateStreamPtr> new_streams;
        int64_t                      current_time_us = autil::TimeUtility::currentTimeInMicroSeconds();

        for (auto git = waiting_groups_.begin(); git != waiting_groups_.end();) {
            auto& group = *git;

            bool is_timeout     = (current_time_us - group.first_enqueue_time) > group.timeout_us;
            bool is_size_enough = (int)group.streams.size() >= group.expected_size;

            RTP_LLM_LOG_DEBUG(
                "Checking Group [%ld]: size [%lu], expected [%d], force [%d], timeout [%d] (elapsed %ld us, limit %ld us)",
                group.batch_group_id,
                group.streams.size(),
                group.expected_size,
                group.force_batch,
                is_timeout,
                (current_time_us - group.first_enqueue_time),
                group.timeout_us);

            // Only force_batch groups need to wait for size or timeout.
            // Non-force_batch groups follow greedy scheduling (run immediately if resources allow).
            if (group.force_batch && !is_timeout && !is_size_enough) {
                RTP_LLM_LOG_DEBUG("Group [%ld] skipped: force_batch waiting for size or timeout.",
                                  group.batch_group_id);
                ++git;
                continue;
            }

            // Atomic Pre-check & Isolation Logic:
            // Only for Force Batch groups that haven't timed out.
            if (group.force_batch && !is_timeout) {
                // 1. Isolation Check: Must execute alone.
                // If new_streams is not empty (others scheduled), skip this group to run in next batch.
                if (!new_streams.empty()) {
                    RTP_LLM_LOG_DEBUG("Group [%ld] skipped: force_batch requires isolation but batch is not empty.",
                                      group.batch_group_id);
                    ++git;
                    continue;
                }

                // 2. Resource Check: Must fit entirely.
                if (!canFitGroupInMemory(new_streams, group.streams)) {
                    // If running_streams is not empty (system busy), skip and wait.
                    // Only run if system is empty to avoid deadlock (partial run allowed in worst case).
                    if (!running_streams_.empty() || !remote_running_streams_.empty()) {
                        RTP_LLM_LOG_DEBUG("Group [%ld] skipped: atomic fit failed and system busy.",
                                          group.batch_group_id);
                        ++git;
                        continue;
                    }
                }
            }

            size_t scheduled_count = 0;
            for (auto sit = group.streams.begin(); sit != group.streams.end();) {
                auto stream = *sit;
                if (evaluateNewStream(new_streams, stream, reserve_step)) {
                    RTP_LLM_LOG_DEBUG("stream [%ld] add to new queue", stream->streamId());
                    if (stream->setRunning()) {
                        new_streams.emplace_back(stream);
                        sit = group.streams.erase(sit);
                        scheduled_count++;
                    } else {
                        RTP_LLM_LOG_WARNING("stream [%ld] set running failed", stream->streamId());
                        stream->releaseResource();
                        sit = group.streams.erase(sit);
                    }
                } else if (running_streams_.empty() && new_streams.empty() && remote_running_streams_.empty()) {
                    RTP_LLM_LOG_WARNING("stream [%ld] can not add to new queue", stream->streamId());
                    if (stream->inputLength() > cache_manager_->maxAvailableTokensNum()) {
                        stream->stopAndRelease(ErrorCode::EXCEEDS_KV_CACHE_MAX_LEN,
                                               "input len " + std::to_string(stream->inputLength())
                                                   + " is greater than kv cache max available tokens num "
                                                   + std::to_string(cache_manager_->maxAvailableTokensNum()));
                    } else if ((size_t)stream->inputLength() * stream->currentBatchSize() > max_batch_tokens_size_) {
                        auto error_info = autil::StringUtil::formatString(
                            "input len [%d] * batch size [%d] > max_batch_tokens_size [%d]",
                            stream->inputLength(),
                            stream->currentBatchSize(),
                            max_batch_tokens_size_);
                        stream->stopAndRelease(ErrorCode::MALLOC_FAILED, error_info);
                    } else {
                        stream->stopAndRelease(ErrorCode::MALLOC_FAILED, "LACK MEM");
                    }
                    sit = group.streams.erase(sit);
                } else {
                    RTP_LLM_LOG_INFO(
                        "Group [%ld] partial schedule stopped: memory limit reached. Scheduled %lu streams.",
                        group.batch_group_id,
                        scheduled_count);

                    // Downgrade strategy: If a force_batch group is split due to resource limits,
                    // the remaining part loses its 'force' privilege and isolation requirements.
                    // This allows it to be scheduled greedily (piggybacking) to minimize latency.
                    if (group.force_batch) {
                        RTP_LLM_LOG_INFO("Group [%ld] downgraded to normal batch due to partial scheduling.",
                                         group.batch_group_id);
                        group.force_batch = false;
                    }

                    return new_streams;
                }
            }

            if (scheduled_count > 0) {
                RTP_LLM_LOG_DEBUG("Group [%ld] scheduled %lu streams.", group.batch_group_id, scheduled_count);
            }

            bool is_force_batch = group.force_batch;

            if (group.streams.empty()) {
                group_map_.erase(group.batch_group_id);
                git = waiting_groups_.erase(git);
            } else {
                ++git;
            }

            // Strict Isolation: If a force_batch group was scheduled (and not timed out),
            // return immediately to ensure no other groups piggyback on this batch.
            if (is_force_batch && !is_timeout && scheduled_count > 0) {
                return new_streams;
            }
        }

        if (!new_streams.empty()) {
            std::string stream_ids = "";
            for (const auto& s : new_streams) {
                stream_ids += std::to_string(s->streamId()) + "(G" + std::to_string(s->batchGroupId()) + ") ";
            }
            RTP_LLM_LOG_DEBUG("ScheduleNew Result: %s", stream_ids.c_str());
        }

        return new_streams;
    }

};

}  // namespace rtp_llm