#pragma once

#include <functional>
#include <memory>
#include <optional>
#include "kmonitor/client/MetricsReporter.h"
#include "rtp_llm/cpp/engine_base/Executor.h"
#include "rtp_llm/cpp/engine_base/EngineInitParams.h"
#include "rtp_llm/cpp/engine_base/ProposeModelEngineInitParams.h"
#include "rtp_llm/cpp/normal_engine/NormalBatchStreamProcessor.h"
#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/core/DeviceData.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/models/eplb/ExpertBalancer.h"

namespace rtp_llm {

class KVCacheManager;
struct GptModelInitParams;

class NormalExecutor: public Executor {
public:
    explicit NormalExecutor(const EngineInitParams&                params,
                            const std::shared_ptr<KVCacheManager>& cache_manager,
                            bool                                   warm_up             = false,
                            bool                                   is_propose          = false,
                            int                                    propose_model_index = 0,
                            const ExecInitParams&                  exec_init_params    = ExecInitParams{});
    ~NormalExecutor();
    absl::Status process(const std::list<GenerateStreamPtr>& streams) override;
    void         reportMetrics(const StreamGroups&             stream_groups,
                               RtpLLMExecutorMetricsCollector& executor_collector,
                               RtpLLMTokenPSMetricsCollector&  tps_collector);

    void setBatchProcessor(std::unique_ptr<NormalBatchStreamProcessor> processor) {
        batch_stream_processor_ = std::move(processor);
    }

    void setModel(std::unique_ptr<ModelBase> model) {
        model_ = std::move(model);
    }

    // Test hook: if set, used to create model when py_model is None
    using ModelFactory = std::function<std::unique_ptr<ModelBase>(const GptModelInitParams&)>;
    static ModelFactory test_model_factory;

    bool updateEplbConfig(const EPLBConfig& config) override;

private:
    std::unique_ptr<ModelBase>                                               model_;
    std::unique_ptr<Sampler>                                                 sampler_;
    std::unique_ptr<NormalBatchStreamProcessor>                              batch_stream_processor_;
    std::shared_ptr<KVCacheManager>                                          cache_manager_;
    std::shared_ptr<ExpertBalancer>                                          expert_balancer_;
    bool                                                                     warm_up_;
    bool                                                                     use_all_gather_;
    kmonitor::MetricsReporterPtr                                             metrics_reporter_ = nullptr;
    MetricsLoopReporter<RtpLLMTokenPSMetrics, RtpLLMTokenPSMetricsCollector> tps_reporter_;
    bool                                                                     enable_ffn_disaggregate_ = false;
    bool                                                                     enable_detail_log_       = false;

    bool              is_propose_          = false;
    int               propose_model_index_ = 0;
    int               tp_rank_             = 0;
    ParallelismConfig parallelism_config_;
};

}  // namespace rtp_llm
