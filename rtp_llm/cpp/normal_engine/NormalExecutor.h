#pragma once

#include <functional>
#include <memory>
#include <optional>
#include "kmonitor/client/MetricsReporter.h"
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
struct GptModelInitParams;

class NormalExecutor: public Executor {
public:
    explicit NormalExecutor(const EngineInitParams&                params,
                            const std::shared_ptr<KVCacheManager>& cache_manager,
                            bool                                   warm_up                 = false,
                            bool                                   is_propose              = false,
                            int                                    propose_model_index     = 0,
                            MlaOpsType                             mla_ops_type            = MlaOpsType::AUTO,
                            int32_t                                kv_cache_group_num      = 1,
                            const std::vector<int32_t>&            kv_cache_layer_to_group = {});
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

protected:
    // Stream-async dispatch gate. Reuses the same env var as MtpExecutor so a
    // single launcher knob (`RTP_LLM_STREAM_ASYNC=1`) flips both paths.
    // Default off — production behaviour unchanged unless explicitly enabled.
    bool useStreamAsync() const;

    // Skip the front-loaded sync of the previous step's dispatch worker at
    // the start of process(). Same env var as MtpExecutor
    // (`RTP_LLM_DROP_BROAD_SYNC=1`). When dropped, the worker may still
    // be writing host stream state while the next step's gather reads it; the
    // normal async device state covers the sampled token and seq_len for the
    // batch-1 decode path.
    bool useDropBroadSync() const;

    // Stream-async dispatch. Records sampler_event on the main stream after
    // sampler_->forward, then forks the bookkeeping worker to wait on the
    // event and run dispatch + per-stream update off the main thread.
    absl::Status dispatchOutputAsync(const StreamGroups&           stream_groups,
                                     GptModelOutputs               model_output,
                                     SamplerOutput                 sampler_output,
                                     std::shared_ptr<torch::Event> sampler_event);

    void publishNormalDeviceState(const StreamGroups& stream_groups, const SamplerOutput& sampler_output);

    // Mirrors the use_normal_device_state condition in
    // NormalModelInputGatherer::processDecodeStreams. When this returns false,
    // gatherModelInput falls back to host-side stream accessors (currentExecuteTokens
    // / seqLength) that the previous step's dispatch worker is still mutating
    // — the caller MUST sync the worker before invoking gather. Uses
    // config-only proxies (hasNumBeams / numReturnSequences) for batch checks
    // to avoid touching the racy outputTokenLen()-derived currentBatchSize().
    bool gatherCanUseDeviceState(const StreamGroups& stream_groups) const;

    // Env-gated path that pushes metadata tensors (combo_tokens,
    // input_lengths, sequence_lengths, prefix_lengths,
    // sequence_lengths_plus_1, lm_output_indexes) to CUDA before
    // tpSyncModelInputs. Without it, those tensors travel through the CPU
    // packed-buffer broadcast (one execBroadcastCpu call with implicit
    // cudaSyncAndCheck on cross-node fallback, plus per-tensor copy/unpack
    // on each rank). With it, they ride along the GPU packed buffer in a
    // single execBroadcast — fewer kernel launches and no CPU-side sync.
    // Shares the env var name `RTP_LLM_DEVICE_INPUT` with MtpExecutor
    // so a single launcher knob toggles both paths.
    bool useDeviceInput() const;
    bool checkDeviceInput() const;
    void ensureModelInputsOnCuda(GptModelInputs& model_input, const char* tag);
    void checkModelInputsOnCuda(const GptModelInputs& model_input, const char* tag) const;

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

    // Bookkeeping worker for stream-async dispatch. Owns a CUDA stream + thread
    // and runs pinned token_ids/success D2H / per-stream update / KV release
    // off the main thread. Eagerly constructed regardless of env gate so the lifetime
    // matches NormalExecutor (the worker only does work when launch() is
    // called).
    AsyncRunner dispatch_runner_;

    // Keeps async copy source tensors alive across release points. NormalExecutor
    // uses this for model-input H2D staging and sampler-input staging.
    TensorHolder buffer_holder_;
};

}  // namespace rtp_llm
