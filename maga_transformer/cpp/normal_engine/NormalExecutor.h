#pragma once

#include <memory>
#include "kmonitor/client/MetricsReporter.h"
#include "maga_transformer/cpp/engine_base/Executor.h"
#include "maga_transformer/cpp/dataclass/EngineInitParameter.h"
#include "maga_transformer/cpp/normal_engine/NormalBatchStreamProcessor.h"
#include "src/fastertransformer/core/Types.h"
#include "maga_transformer/cpp/metrics/RtpLLMMetrics.h"

namespace rtp_llm {

class NormalExecutor: public Executor {
public:
    explicit NormalExecutor(const EngineInitParams& params, const std::shared_ptr<CacheManager>& cache_manager, ft::DeviceBase* device);
    absl::Status process(const std::list<GenerateStreamPtr>& streams) override;
    absl::Status addLoRA(const int64_t                                                           lora_id,
                         const std::vector<std::unordered_map<std::string, ft::ConstBufferPtr>>& lora_a_weights,
                         const std::vector<std::unordered_map<std::string, ft::ConstBufferPtr>>& lora_b_weights) override;
    absl::Status removeLoRA(const int64_t lora_id) override;
    void         reportMetrics(const StreamGroups& stream_groups);
private:
    std::unique_ptr<GptModel>                   model_;
    std::unique_ptr<Sampler>                    sampler_;
    std::unique_ptr<NormalBatchStreamProcessor> batch_stream_processor_;
    std::shared_ptr<CacheManager>               cache_manager_;
    kmonitor::MetricsReporterPtr                metrics_reporter_ = nullptr;
    MetricsLoopReporter<RtpLLMTokenPSMetrics, RtpLLMTokenPSMetricsCollector> tps_reporter_;

    ft::DataType                                dtype_               = ft::DataType::TYPE_FP16 ;
    bool                                        is_causal_           = true;
    bool                                        need_attention_mask_ = false;
};

}  // namespace rtp_llm
