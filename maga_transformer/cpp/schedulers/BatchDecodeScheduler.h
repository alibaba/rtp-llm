#pragma once

#include "autil/legacy/jsonizable.h"
#include "maga_transformer/cpp/schedulers/SchedulerBase.h"
#include "maga_transformer/cpp/devices/DeviceBase.h"
#include <mutex>
#include <condition_variable>
#include <list>

namespace rtp_llm {

struct BatchDecodeSchedulerConfig: public autil::legacy::Jsonizable {
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) override {
        json.Jsonize("batch_size", batch_size_);
    }
    uint32_t batch_size_;
};

class BatchDecodeScheduler : public SchedulerBase {
public:
    BatchDecodeScheduler(const rtp_llm::GptInitParameter&          params,
                         const std::shared_ptr<CacheManager>& cache_manager,
                         const kmonitor::MetricsReporterPtr   metrics_reporter,
                         rtp_llm::DeviceBase*  device) {
        cache_manager_    = cache_manager;
        device_           = device;
        metrics_reporter_ = metrics_reporter;
        batch_size_ = 1;
        char* batch_size = std::getenv("SCHEDULER_RUN_BATCH_SIZE");
        if (batch_size) {
            batch_size_ = std::atoi(batch_size);
        }        
    }
    virtual ~BatchDecodeScheduler() = default;

    absl::Status enqueue(const GenerateStreamPtr& stream) override {
        {
            std::lock_guard<std::mutex> lock(lock_);
            waiting_streams_.emplace_back(stream);
            if (waiting_streams_.size() % 16 == 0) {
                RTP_LLM_LOG_DEBUG("BatchDecodeScheduler::enqueue: waiting_streams_.size() = %d", waiting_streams_.size());
            }
        }
        cond_.notify_all();
        return absl::OkStatus();
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
        BatchDecodeSchedulerConfig config;
        autil::legacy::FromJsonString(config, scheduler_info);
        batch_size_ = config.batch_size_;        
        RTP_LLM_LOG_INFO("BatchDecodeScheduler update batch size to %d", batch_size_);
    }

    void initRunningStreams() {
        // set kvcache block
        for (auto it = running_streams_.begin(); it != running_streams_.end(); it++) {
            // reset start time，to get more accurate avg token time
            (*it)->resetBeginTime(autil::TimeUtility::currentTimeInMicroSeconds());
            auto result = (*it)->initKVBlock(0, 0);
            if (!result.ok()) {
                (*it)->setStop(ErrorCode::MALLOC_FAILED, "BatchDecodeScheduler::initRunningStreams: initKVBlock failed");
            }
        }
        // incr kvcache block to decode
        for (auto it = running_streams_.begin(); it != running_streams_.end(); it++) {
            auto stream = *it;
            stream->setIsContextStream(false);
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
        cond_.wait_for(lock, std::chrono::seconds(300), [this]{
            return waiting_streams_.size() >= batch_size_ || running_streams_.size() > 0;
        });
        if (running_streams_.size() == 0 && waiting_streams_.size() >= batch_size_) {
            auto it = waiting_streams_.begin();
            std::advance(it, batch_size_);
            running_streams_.insert(running_streams_.end(), waiting_streams_.begin(), it);
            waiting_streams_.erase(waiting_streams_.begin(), it);
            auto unused_buffer = device_->allocateBuffer({rtp_llm::DataType::TYPE_INT32, {1,1}, rtp_llm::AllocationType::DEVICE}, {});
            device_->allReduce({unused_buffer, rtp_llm::ReduceOp::Sum, false, rtp_llm::ParallelMode::DP});
            device_->syncCommunication(false);
            initRunningStreams();
            RTP_LLM_LOG_INFO("BatchDecodeScheduler::schedule: running_streams_.size() = %d, start run", running_streams_.size());
        } else {
            incrRunningStream();
        }
        evictAllDoneStreams();        
        return running_streams_;
    }

    bool canLoadBalance() override {
        return false;
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
    std::mutex lock_;
    std::condition_variable cond_;
    std::list<GenerateStreamPtr> waiting_streams_;
    std::list<GenerateStreamPtr> running_streams_;

    uint32_t batch_size_;

    uint32_t current_step_ = 0;

    std::shared_ptr<CacheManager> cache_manager_;
    kmonitor::MetricsReporterPtr metrics_reporter_;
    rtp_llm::DeviceBase* device_;
};


}  // namespace rtp_llm
