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
    explicit GatherBatchScheduler(const RuntimeConfig&                 runtime_config,
                                  const ModelConfig&                   model_config,
                                  const PDSepConfig&                  pd_sep_config,
                                  const ParallelismConfig&            parallelism_config,
                                  const ModelSpecificConfig&          model_specific_config,
                                  const std::shared_ptr<KVCacheManager>& cache_manager,
                                  const kmonitor::MetricsReporterPtr   metrics_reporter,
                                  const int                            max_score_len = 1):
        FIFOScheduler(runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager, metrics_reporter, max_score_len) {
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

protected:
    std::list<GenerateStreamPtr> scheduleNew(size_t reserve_step) override {
        if (waiting_streams_.empty()) {
            return {};
        }
        if ((int)waiting_streams_.size() < gather_batch_size_) {
            RTP_LLM_LOG_INFO("GatherBatchScheduler scheduleNew, waiting_streams_.size() [%d] < gather_batch_size_ [%d]",
                             waiting_streams_.size(),
                             gather_batch_size_);
            return {};
        }
        RTP_LLM_LOG_INFO(
            "GatherBatchScheduler scheduleNew, waiting_streams_.size() [%d] >= gather_batch_size_ [%d], start run",
            waiting_streams_.size(),
            gather_batch_size_);
        waiting_streams_.sort(
            [](const GenerateStreamPtr& a, const GenerateStreamPtr& b) { return a->streamId() < b->streamId(); });
        gather_batch_size_ = 1;
        return FIFOScheduler::scheduleNew(reserve_step);
    }

protected:
    int gather_batch_size_;
};

}  // namespace rtp_llm