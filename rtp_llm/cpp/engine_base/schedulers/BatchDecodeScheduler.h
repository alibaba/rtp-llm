#pragma once

#include "autil/legacy/jsonizable.h"
#include "rtp_llm/cpp/engine_base/schedulers/SchedulerBase.h"
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/Types.h"
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <list>

namespace rtp_llm {

struct BatchDecodeSchedulerConfigLocal: public autil::legacy::Jsonizable {
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) override {
        json.Jsonize("batch_size", batch_size_);
        json.Jsonize("mode", mode_, "decode");
    }
    uint32_t    batch_size_;
    std::string mode_;
};
class BatchDecodeScheduler: public SchedulerBase {
public:
    enum SchedulerType : std::uint8_t {
        kBatchDecode  = 0,
        kBatchPrefill = 1
    };
    BatchDecodeScheduler(const RuntimeConfig&                   runtime_config,
                         const std::shared_ptr<KVCacheManager>& cache_manager,
                         const kmonitor::MetricsReporterPtr     metrics_reporter,
                         int                                    dp_rank = 0) {
        cache_manager_    = cache_manager;
        metrics_reporter_ = metrics_reporter;
        batch_size_       = runtime_config.batch_decode_scheduler_config.batch_decode_scheduler_batch_size;
        scheduler_type_   = SchedulerType::kBatchDecode;
        dp_rank_          = dp_rank;
    }
    virtual ~BatchDecodeScheduler() = default;

    absl::Status enqueue(const GenerateStreamPtr& stream) override {
        {
            std::lock_guard<std::mutex> lock(lock_);
            waiting_streams_.emplace_back(stream);
            if (waiting_streams_.size() % 16 == 0) {
                RTP_LLM_LOG_DEBUG("BatchDecodeScheduler::enqueue: waiting_streams_.size() = %d",
                                  waiting_streams_.size());
            }
        }
        cond_.notify_all();
        return absl::OkStatus();
    }

    std::vector<GenerateStreamPtr> batchEnqueue(const std::vector<GenerateStreamPtr>& streams) override {
        {
            std::lock_guard<std::mutex> lock(lock_);
            waiting_streams_.insert(waiting_streams_.end(), streams.begin(), streams.end());
        }
        cond_.notify_all();
        return streams;
    }

    void updateSchedulerInfo(const std::string& scheduler_info) override {
        BatchDecodeSchedulerConfigLocal config;
        autil::legacy::FromJsonString(config, scheduler_info);
        batch_size_ = config.batch_size_;
        if (config.mode_ == "decode") {
            scheduler_type_ = SchedulerType::kBatchDecode;
        } else if (config.mode_ == "prefill") {
            scheduler_type_ = SchedulerType::kBatchPrefill;
        }
        RTP_LLM_LOG_INFO("BatchDecodeScheduler update batch size to %d, mode to %d", batch_size_, int(scheduler_type_));
    }

    // 根据状态机转移后的目标状态，将 stream 路由到对应的队列
    void addStreamToNewState(const GenerateStreamPtr& stream, StreamState new_state) {
        switch (new_state) {
            case StreamState::WAITING:
                waiting_streams_.push_back(stream);
                break;
            case StreamState::LOADING_CACHE:
                loading_cache_streams_.push_back(stream);
                break;
            case StreamState::RUNNING:
                running_streams_.push_back(stream);
                break;
            case StreamState::FINISHED:
                break;
            default:
                RTP_LLM_LOG_ERROR(
                    "Unknown state: %d for stream [%ld]", static_cast<int>(new_state), stream->streamId());
                break;
        }
    }

    // 通过 GenerateStateMachine 驱动每个 stream 的状态转移，状态变化的 stream 移入对应队列
    void evaluateAndUpdateStreams(std::list<GenerateStreamPtr>& streams) {
        for (auto it = streams.begin(); it != streams.end();) {
            auto state     = (*it)->getStatus();
            auto new_state = (*it)->moveToNext();
            if (new_state != state) {
                addStreamToNewState(*it, new_state);
                it = streams.erase(it);
            } else {
                it++;
            }
        }
    }

    void evaluateWaitingStreams() {
        // 清理 waiting_streams_ 中有错误的 stream
        waiting_streams_.remove_if([](const auto& s) { return s->hasError(); });

        std::list<GenerateStreamPtr> new_streams;
        for (auto it = waiting_streams_.begin(); it != waiting_streams_.end(); it++) {
            // 先检查是否有错误，避免错误请求占用资源
            if (!(*it)->hasError()) {
                new_streams.push_back(*it);
            }
            if (new_streams.size() >= batch_size_) {
                break;
            }
        }
        // 凑到batch_size_个stream再统一入队
        if (new_streams.size() >= batch_size_) {
            for (auto& stream : new_streams) {
                stream->reportEvent(StreamEvents::CanRun);
                // 忙等stream load cache done, 和原有SyncLoadCache逻辑等效
                while (stream->getStatus() != StreamState::FINISHED && stream->moveToNext() != StreamState::RUNNING) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
            }
            // 过滤 FINISHED stream，仅将 RUNNING stream 加入 running_streams_
            new_streams.remove_if([](const auto& s) { return s->getStatus() == StreamState::FINISHED; });
            running_streams_.insert(running_streams_.end(), new_streams.begin(), new_streams.end());
            // 从waiting_streams_中移除已调度的stream
            for (auto& stream : new_streams) {
                waiting_streams_.remove(stream);
            }
        }
    }

    void initRunningStreams() {
        // set kvcache block
        for (auto it = running_streams_.begin(); it != running_streams_.end(); it++) {
            (*it)->setPerfTest(true);
            // reset start time，to get more accurate avg token time
            (*it)->resetBeginTime(autil::TimeUtility::currentTimeInMicroSeconds());
            // only set gen_timeline = True for first rank
            if (dp_rank_ != 0) {
                (*it)->setGenTimeline(false);
            }
            if (scheduler_type_ == SchedulerType::kBatchDecode) {
                (*it)->setIsContextStream(false);
                // for linear attn, incrKVBlock to clear unused linear block
                (*it)->moveToNext();
            }
        }
    }

    absl::StatusOr<std::list<GenerateStreamPtr>> schedule() override {
        std::unique_lock<std::mutex> lock(lock_);
        cond_.wait_for(lock, std::chrono::seconds(30), [this] {
            return waiting_streams_.size() >= batch_size_ || running_streams_.size() > 0
                   || !loading_cache_streams_.empty();
        });

        // 统一通过状态机驱动各队列中 stream 的状态转移
        // LOADING_CACHE -> DONE/WAITING: error / load cache done
        evaluateAndUpdateStreams(loading_cache_streams_);
        evaluateAndUpdateStreams(running_streams_);

        if (running_streams_.empty() && waiting_streams_.size() >= batch_size_) {
            evaluateWaitingStreams();
            if (!running_streams_.empty()) {
                initRunningStreams();
                RTP_LLM_LOG_INFO("BatchDecodeScheduler::schedule: running_streams_.size() = %d, start run",
                                 running_streams_.size());
            }
        }

        return running_streams_;
    }

    absl::Status stop() override {
        // Not implemented
        return absl::UnimplementedError("BatchDecodeScheduler::stop not implemented");
    }

    bool empty() override {
        // Not implemented
        return true;  // 默认返回值
    }

    int64_t lastScheduleTime() override {
        return 0;  // 默认返回值
    }

    int64_t onflightStreams() override {
        std::lock_guard<std::mutex> lock(lock_);
        return waiting_streams_.size() + loading_cache_streams_.size() + running_streams_.size();
    }

private:
    std::mutex                   lock_;
    std::condition_variable      cond_;
    std::list<GenerateStreamPtr> waiting_streams_;
    std::list<GenerateStreamPtr> loading_cache_streams_;
    std::list<GenerateStreamPtr> running_streams_;
    uint32_t                     batch_size_;
    bool                         reorder_request_;
    uint32_t                     current_step_ = 0;

    std::shared_ptr<KVCacheManager> cache_manager_;
    kmonitor::MetricsReporterPtr    metrics_reporter_;
    SchedulerType                   scheduler_type_;
    int                             dp_rank_ = 0;
};

}  // namespace rtp_llm
