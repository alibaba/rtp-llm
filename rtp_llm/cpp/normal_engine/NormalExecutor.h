#pragma once

#include <functional>
#include <memory>
#include <optional>
#include "kmonitor/client/MetricsReporter.h"
#include "rtp_llm/cpp/cuda_graph/cuda_graph_base.h"
#include "rtp_llm/cpp/engine_base/Executor.h"
#include "rtp_llm/cpp/engine_base/EngineInitParams.h"
#include "rtp_llm/cpp/engine_base/ProposeModelEngineInitParams.h"
#include "rtp_llm/cpp/normal_engine/AsyncRunner.h"
#include "rtp_llm/cpp/normal_engine/NormalBatchStreamProcessor.h"
#include "rtp_llm/cpp/models/ModelTypes.h"
#include "rtp_llm/models_py/bindings/core/Types.h"
#include "rtp_llm/models_py/bindings/core/DeviceData.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/models/eplb/ExpertBalancer.h"

namespace rtp_llm {

class KVCacheManager;
class ModelInputsLogger;
struct GptModelInitParams;

class NormalExecutor: public Executor {
public:
    explicit NormalExecutor(const EngineInitParams&                params,
                            const std::shared_ptr<KVCacheManager>& cache_manager,
                            bool                                   warm_up             = false,
                            bool                                   is_propose          = false,
                            int                                    propose_model_index = 0,
                            MlaOpsType                             mla_ops_type        = MlaOpsType::AUTO,
                            std::function<void()>                  profile_step_start  = nullptr,
                            std::function<void()>                  profile_step_finish = nullptr);
    ~NormalExecutor();
    absl::Status process(const std::list<GenerateStreamPtr>& streams, int64_t schedule_time_us = 0) override;
    void         reportMetrics(const StreamGroups&                        stream_groups,
                               RtpLLMExecutorMetricsCollector&            executor_collector,
                               RtpLLMTokenPSMetricsCollector&             tps_collector,
                               int64_t                                    tps_execute_time_us,
                               const StreamGroups::TokenCountsByPriority& token_counts_by_priority);

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

protected:
    // Stream-async dispatch. Records sampler_event on the main stream after
    // sampler_->forward, then forks the bookkeeping worker to wait on the
    // event and run dispatch + per-stream update off the main thread.
    absl::Status dispatchOutputAsync(const StreamGroups&           stream_groups,
                                     GptModelOutputs               model_output,
                                     SamplerOutput                 sampler_output,
                                     std::shared_ptr<torch::Event> sampler_event,
                                     std::function<void()>         profile_step_finish = nullptr);

    void publishNormalDeviceState(const StreamGroups& stream_groups, const SamplerOutput& sampler_output);
    void prepareGrpcNormalDeviceState(const StreamGroups& stream_groups);

    // Mirror the use_normal_device_state condition in processDecodeStreams.
    // When false, gather falls back to host accessors still mutated by the
    // worker, so callers must sync before gather.
    bool gatherCanUseDeviceState(const StreamGroups& stream_groups) const;

    // Moves the tensors selected by execution_policy_.model_input_placement to
    // CUDA before tpSyncModelInputs. This routes them through one GPU packed-
    // buffer broadcast instead of CPU execBroadcastCpu plus unpack/copy work.
    void ensureModelInputsOnCuda(GptModelInputs& model_input, const char* tag);
    void checkModelInputsOnCuda(const GptModelInputs& model_input, const char* tag) const;

private:
    // Immutable policy resolved once when the executor is constructed. Runtime
    // code asks semantic questions instead of repeatedly combining environment
    // gates, while lower layers receive only their relevant derived policy.
    struct ExecutionPolicy {
        bool                         dispatch_async{false};
        bool                         overlap_bookkeeping{false};
        bool                         use_device_input{false};
        bool                         validate_device_input{false};
        ModelInputPlacement          model_input_placement{ModelInputPlacement::HOST};
        CudaGraphReplayPreparePolicy replay_prepare_policy{CudaGraphReplayPreparePolicy::WAIT_FOR_PREVIOUS_REPLAY};

        bool waitForPreviousDispatchBeforeGather() const {
            return dispatch_async && !overlap_bookkeeping;
        }

        bool tryGatherWhileBookkeepingRuns() const {
            return overlap_bookkeeping;
        }

        bool useAsyncDispatch(bool is_decode_only) const {
            return dispatch_async && is_decode_only;
        }
    };

    static ExecutionPolicy loadExecutionPolicy();

    std::unique_ptr<ModelBase>                                               model_;
    std::unique_ptr<Sampler>                                                 sampler_;
    std::unique_ptr<NormalBatchStreamProcessor>                              batch_stream_processor_;
    std::shared_ptr<KVCacheManager>                                          cache_manager_;
    std::shared_ptr<ModelInputsLogger>                                       model_inputs_logger_;
    std::shared_ptr<ExpertBalancer>                                          expert_balancer_;
    bool                                                                     warm_up_;
    bool                                                                     use_all_gather_;
    kmonitor::MetricsReporterPtr                                             metrics_reporter_ = nullptr;
    MetricsLoopReporter<RtpLLMTokenPSMetrics, RtpLLMTokenPSMetricsCollector> tps_reporter_;
    WallClockMetricsLoopReporter<RtpLLMWallClockTokenPSMetrics, RtpLLMTokenPSMetricsCollector> wall_tps_reporter_;
    bool enable_ffn_disaggregate_ = false;
    bool enable_detail_log_       = false;

    bool                  is_propose_          = false;
    int                   propose_model_index_ = 0;
    int                   tp_rank_             = 0;
    ParallelismConfig     parallelism_config_;
    RoleType              role_type_ = RoleType::PDFUSION;
    const ExecutionPolicy execution_policy_;
    std::function<void()> profile_step_start_;
    std::function<void()> profile_step_finish_;

    // Stream-async worker owns a CUDA stream/thread for pinned D2H,
    // per-stream update, and KV release off the main thread.
    AsyncRunner dispatch_runner_;

    // Keeps async copy source tensors alive across release points. NormalExecutor
    // uses this for model-input H2D staging and sampler-input staging.
    TensorHolder buffer_holder_;
};

}  // namespace rtp_llm
