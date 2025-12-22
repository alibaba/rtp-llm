#pragma once

#include "autil/legacy/jsonizable.h"
#include "rtp_llm/cpp/engine_base/schedulers/SchedulerBase.h"
#include "rtp_llm/cpp/devices/DeviceBase.h"
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/types.h"
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
    BatchDecodeScheduler(const RuntimeConfig&                 runtime_config,
                        const std::shared_ptr<KVCacheManager>& cache_manager,
                         const kmonitor::MetricsReporterPtr   metrics_reporter,
                         rtp_llm::DeviceBase*                 device) {
        cache_manager_    = cache_manager;
        device_           = device;
        metrics_reporter_ = metrics_reporter;
        batch_size_       = runtime_config.batch_decode_scheduler_config.batch_decode_scheduler_batch_size;
        scheduler_type_   = SchedulerType::kBatchDecode;
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

    absl::Status batchEnqueue(const std::vector<GenerateStreamPtr>& streams) override {
        return absl::InternalError("Not implement yet");
    }

    void evictAllDoneStreams() {
        for (auto it = running_streams_.begin(); it != running_streams_.end();) {
            (*it)->checkTimeout();
            if ((*it)->stopped() || (*it)->finished()) {
                // Immediately free resources to run more streams
                (*it)->releaseResource();
                RTP_LLM_LOG_DEBUG("evict stream [%ld]", (*it)->streamId());
                it = running_streams_.erase(it);
            } else {
                ++it;
            }
        }
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

    void initRunningStreams() {
        // set kvcache block
        for (auto it = running_streams_.begin(); it != running_streams_.end(); it++) {
            // reset start time，to get more accurate avg token time
            (*it)->resetBeginTime(autil::TimeUtility::currentTimeInMicroSeconds());
            // only set gen_timeline = True for first rank
            if (device_->getDeviceProperties().dp_rank != 0) {
                (*it)->setGenTimeline(false);
            }
            auto result = (*it)->initKVBlock(0, 0);
            if (!result.ok()) {
                (*it)->setStop(ErrorCode::MALLOC_FAILED,
                               "BatchDecodeScheduler::initRunningStreams: initKVBlock failed");
            }
        }
        // incr kvcache block to decode
        if (scheduler_type_ == SchedulerType::kBatchDecode) {
            for (auto it = running_streams_.begin(); it != running_streams_.end(); it++) {
                auto stream = *it;
                stream->setIsContextStream(false);
            }
        }
    }

    void incrRunningStream() {
        for (auto it = running_streams_.begin(); it != running_streams_.end();) {
            auto result = (*it)->incrKVBlock(0, 0);
            if (!result.ok()) {
                (*it)->stopAndRelease(ErrorCode::MALLOC_FAILED, "incrKVBlock failed");
                RTP_LLM_LOG_WARNING("stream [%ld] incr block failed", (*it)->streamId());
                it = running_streams_.erase(it);
            } else {
                it++;
            }
        }
    }

    absl::StatusOr<std::list<GenerateStreamPtr>> schedule(size_t reserve_step = 0) override {
        std::unique_lock<std::mutex> lock(lock_);
        cond_.wait_for(lock, std::chrono::seconds(30), [this] {
            return waiting_streams_.size() >= batch_size_ || running_streams_.size() > 0;
        });
        if (running_streams_.size() == 0 && waiting_streams_.size() >= batch_size_) {
            auto it = waiting_streams_.begin();
            std::advance(it, batch_size_);
            running_streams_.insert(running_streams_.end(), waiting_streams_.begin(), it);
            waiting_streams_.erase(waiting_streams_.begin(), it);
            initRunningStreams();
            RTP_LLM_LOG_INFO("BatchDecodeScheduler::schedule: running_streams_.size() = %d, start run",
                             running_streams_.size());
        } else {
            incrRunningStream();
        }
        evictAllDoneStreams();
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
        return waiting_streams_.size() + running_streams_.size();
    }

private:
    std::mutex                   lock_;
    std::condition_variable      cond_;
    std::list<GenerateStreamPtr> waiting_streams_;
    std::list<GenerateStreamPtr> running_streams_;
    uint32_t                     batch_size_;
    bool                         reorder_request_;
    uint32_t                     current_step_ = 0;

    std::shared_ptr<KVCacheManager> cache_manager_;
    kmonitor::MetricsReporterPtr    metrics_reporter_;
    rtp_llm::DeviceBase*            device_;
    SchedulerType                   scheduler_type_;
};

}  // namespace rtp_llm
