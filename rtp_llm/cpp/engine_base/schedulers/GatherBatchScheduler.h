#pragma once
#include "rtp_llm/cpp/engine_base/schedulers/FIFOScheduler.h"

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
        {
            std::lock_guard<std::mutex> lock(lock_);
            gather_batch_size_ = config.batch_size_;
        }
        cond_.notify_all();
        RTP_LLM_LOG_INFO("GatherBatchScheduler update batch size to %d", gather_batch_size_);
    }

    absl::StatusOr<std::list<GenerateStreamPtr>> schedule() override {
        std::unique_lock<std::mutex> lock(lock_);
        cond_.wait_for(lock, std::chrono::seconds(30), [this] {
            return waiting_streams_.size() >= static_cast<size_t>(gather_batch_size_) || running_streams_.size() > 0
                   || !loading_cache_streams_.empty();
        });

        // LOADING_CACHE -> DONE/WAITING: error / load cache done
        evaluateAndUpdateStreams(loading_cache_streams_);
        // RUNNING -> DONE: error / finished
        evaluateAndUpdateStreams(running_streams_);

        // Defer the gather until running streams drain so the next batch is pure prefill.
        const bool python_model_busy = !running_streams_.empty();
        if (waiting_streams_.size() >= static_cast<size_t>(gather_batch_size_) && !python_model_busy) {
            // Gather exactly gather_batch_size_ streams
            std::list<GenerateStreamPtr> new_streams;
            for (auto it = waiting_streams_.begin(); it != waiting_streams_.end(); it++) {
                if (!(*it)->hasError()) {
                    new_streams.push_back(*it);
                }
                if (new_streams.size() >= static_cast<size_t>(gather_batch_size_)) {
                    break;
                }
            }
            // Only schedule when we have enough streams
            if (new_streams.size() >= static_cast<size_t>(gather_batch_size_)) {
                for (auto& stream : new_streams) {
                    stream->reportEvent(StreamEvents::CanRun);
                    // busy wait for loading cache done, equivalent to to original logic.
                    while (stream->getStatus() != StreamState::FINISHED
                           && stream->moveToNext() != StreamState::RUNNING) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    }
                }
                // 过滤 FINISHED stream，仅将 RUNNING stream 加入 running_streams_
                new_streams.remove_if([](const auto& s) { return s->getStatus() == StreamState::FINISHED; });
                // 按 streamId 排序以保证 CI 确定性结果
                new_streams.sort([](const GenerateStreamPtr& a, const GenerateStreamPtr& b) {
                    return a->streamId() < b->streamId();
                });
                running_streams_.insert(running_streams_.end(), new_streams.begin(), new_streams.end());
                // Remove scheduled streams from waiting_streams_
                for (auto& stream : new_streams) {
                    waiting_streams_.remove(stream);
                }
            }
            gather_batch_size_ = 1;
        }

        return running_streams_;
    }

protected:
    int gather_batch_size_;
};

}  // namespace rtp_llm