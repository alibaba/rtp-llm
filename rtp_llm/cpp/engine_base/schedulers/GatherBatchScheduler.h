#pragma once
#include "rtp_llm/cpp/engine_base/schedulers/FIFOScheduler.h"
#include <list>
#include <unordered_map>
#include <chrono>

namespace rtp_llm {

struct GatherBatchSchedulerConfigLocal: public autil::legacy::Jsonizable {
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) override {
        json.Jsonize("batch_size", batch_size_);
    }
    uint32_t batch_size_;
};

// GatherBatchScheduler is used to gather batch of streams, and sort streams by streamId.
// Currently it is only used in CI with prompt_batch input, which may occur unstable result
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
        gather_batch_size_ = 1;
    }

    void updateSchedulerInfo(const std::string& scheduler_info) override {
        GatherBatchSchedulerConfigLocal config;
        autil::legacy::FromJsonString(config, scheduler_info);
        gather_batch_size_ = config.batch_size_;
        RTP_LLM_LOG_INFO("GatherBatchScheduler update batch size to %d", gather_batch_size_);
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
                        if (git->expected_size > 0)
                            git->expected_size--;
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

        if (current_waiting_size > 0 && current_waiting_size < gather_batch_size_) {
            RTP_LLM_LOG_INFO("GatherBatchScheduler scheduleNew, waitingStreamsSize [%ld] < gather_batch_size_ [%d]",
                             current_waiting_size,
                             gather_batch_size_);
            return {};
        }

        // reset gather_batch_size_ to 1
        gather_batch_size_ = 1;

        std::list<GenerateStreamPtr> new_streams;
        int64_t                      current_time_us = autil::TimeUtility::currentTimeInMicroSeconds();

        for (auto git = waiting_groups_.begin(); git != waiting_groups_.end();) {
            auto& group = *git;

            bool is_timeout     = (current_time_us - group.first_enqueue_time) > group.timeout_us;
            bool is_size_enough = (int)group.streams.size() >= group.expected_size;

            if (!is_timeout && !is_size_enough) {
                ++git;
                continue;
            }

            // Atomic Pre-check:
            // Only for Force Batch groups that haven't timed out.
            // If they fit, great. If not, we skip them to wait for more resources (try to run atomically).
            // Unless the system is empty (deadlock prevention), then we fall through to split.
            // Normal groups (or timed-out Force Batch) always skip this and go straight to greedy scheduling.
            if (group.force_batch && !is_timeout) {
                if (!canFitGroupInMemory(new_streams, group.streams)) {
                    if (!running_streams_.empty() || !new_streams.empty() || !remote_running_streams_.empty()) {
                        ++git;
                        continue;
                    }
                }
            }

            for (auto sit = group.streams.begin(); sit != group.streams.end();) {
                auto stream = *sit;
                if (evaluateNewStream(new_streams, stream, reserve_step)) {
                    RTP_LLM_LOG_DEBUG("stream [%ld] add to new queue", stream->streamId());
                    if (stream->setRunning()) {
                        new_streams.emplace_back(stream);
                        sit = group.streams.erase(sit);
                        if (group.expected_size > 0)
                            group.expected_size--;
                    } else {
                        RTP_LLM_LOG_WARNING("stream [%ld] set running failed", stream->streamId());
                        stream->releaseResource();
                        sit = group.streams.erase(sit);
                        if (group.expected_size > 0)
                            group.expected_size--;
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
                    if (group.expected_size > 0)
                        group.expected_size--;
                } else {
                    return new_streams;
                }
            }

            if (group.streams.empty()) {
                group_map_.erase(group.batch_group_id);
                git = waiting_groups_.erase(git);
            } else {
                ++git;
            }
        }

        return new_streams;
    }

protected:
    int gather_batch_size_;
};

}  // namespace rtp_llm