#pragma once

#include <list>
#include <map>
#include <memory>
#include <vector>
#include "kmonitor/client/MetricsReporter.h"
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/engine_base/Executor.h"
#include "rtp_llm/cpp/normal_engine/NormalBatchStreamProcessor.h"
#include "rtp_llm/models_py/bindings/core/Types.h"
#include "rtp_llm/models_py/bindings/core/DeviceData.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/models/eplb/ExpertBalancer.h"
#include "rtp_llm/cpp/normal_engine/speculative/MtpBatchStreamProcessor.h"
#include "rtp_llm/cpp/engine_base/ProposeModelEngineInitParams.h"
#include "rtp_llm/cpp/models/ModelTypes.h"
#include "rtp_llm/cpp/normal_engine/AsyncRunner.h"
#include "rtp_llm/cpp/models/logits_processor/SpecLogitsVerifyRunner.h"
#include "rtp_llm/cpp/normal_engine/speculative/SpeculativeSampler.h"

namespace rtp_llm {

enum class ModelInputsModelRole;

class ModelInputsLogger;

struct MtpMetricsCollector {
    RtpLLMExecutorMetricsCollector          executor_collector;
    RtpLLMTokenPSMetricsCollector           tps_collector;
    RtpLLMSpeculativeEngineMetricsCollector sp_engine_collector;

    // Accepted decode tokens bucketed by stream priority; the bucket sums add
    // up exactly to sp_engine_collector.total_accepted_token_num.
    std::map<int32_t, int64_t> accepted_token_num_by_priority;

    bool not_skip = false;
};

class MtpExecutor: public Executor {
public:
    explicit MtpExecutor(const EngineInitParams&                        params,
                         std::unique_ptr<ProposeModelEngineInitParams>& propose_params,
                         const std::shared_ptr<KVCacheManager>&         cache_manager,
                         MlaOpsType                                     mla_ops_type       = MlaOpsType::AUTO,
                         int32_t                                        kv_cache_group_num = 1,
                         bool                                           warm_up            = false);

    absl::Status process(const std::list<GenerateStreamPtr>& streams, int64_t schedule_time_us = 0) override;
    bool         updateEplbConfig(const EPLBConfig& config) override;

    void setTargetModel(std::unique_ptr<ModelBase> model) {
        model_ = std::move(model);
    }

    void setDraftModel(std::unique_ptr<ModelBase> model) {
        draft_model_ = std::move(model);
    }

    void setBatchProcessor(std::unique_ptr<MtpBatchStreamProcessor> processor) {
        batch_stream_processor_ = std::move(processor);
    }

    void setDraftPrefillModel(std::unique_ptr<ModelBase> model) {
        sp_prefill_draft_model_ = std::move(model);
    }

    void setFastTopKSampler(std::unique_ptr<speculative::FastTopKSampler> sampler) {
        fast_topk_sampler_ = std::move(sampler);
    }

    void setSpeculativeSampler(std::unique_ptr<speculative::SpeculativeSampler> sampler) {
        speculative_sampler_ = std::move(sampler);
    }

    void setSampler(std::unique_ptr<Sampler> sampler) {
        sampler_ = std::move(sampler);
    }

public:
    static GenerateStreamPtr createMinFakePrefillStream(int                    max_new_tokens,
                                                        const ModelConfig&     model_config,
                                                        const RuntimeConfig&   runtime_config,
                                                        const ResourceContext& resource_context);
    static GenerateStreamPtr createMinFakeDecodeStream(int                    max_new_tokens,
                                                       const ModelConfig&     model_config,
                                                       const RuntimeConfig&   runtime_config,
                                                       const ResourceContext& resource_context,
                                                       int                    vocab_size,
                                                       bool                   is_dspark = false);

protected:
    static bool dsparkPrefillCPRoleIsValid(const PrefillCPConfig& prefill_cp_config, RoleType role_type);
    static bool dsparkDraftGraphAllowed(bool is_dspark, RoleType role_type);
    struct AcceptLenMetricsSnapshot {
        int64_t total_accept_len        = 0;
        int64_t total_stream_num        = 0;
        int64_t total_propose_token_num = 0;
        // Per-priority accept counts derived from the same staged accept_len
        // rows as total_accept_len, so the bucket sums match it exactly.
        std::map<int32_t, int64_t> accept_len_by_priority;
        bool                       valid = false;
    };

    bool isTpRank0() const;

    void maybeOverrideLastHiddenWithMtpBuffer(GptModelInputs& model_input,
                                              ModelBase&      source,
                                              bool            request_actual_rows = false);
    // Normalize the model's optional pre-output-projection MTP buffer into the
    // forward result. Callers then use the regular all_hidden_states ->
    // last_hidden_states hand-off. hidden_rows == 0 means "use the tensor's own
    // row count"; target verify passes the explicit combo row count because a
    // graph replay does not advance the Python-side row counter.
    void maybeOverrideLastHiddenWithMtpBuffer(GptModelOutputs& model_output,
                                              ModelBase&       source,
                                              int64_t          hidden_rows = 0);

    void maybePrintModelInput(const GptModelInputs& model_input, const std::string& prefix) const;

    absl::Status prefillStep(const std::list<GenerateStreamPtr>& streams,
                             MtpMetricsCollector&                metrics_collector,
                             int64_t                             schedule_time_us);

    absl::Status decodeStep(const std::list<GenerateStreamPtr>& streams, MtpMetricsCollector& metrics_collector);

    // decodeStep helpers — extracted to keep decodeStep readable. Each helper
    // owns a single phase (sync, prepare, forward, broadcast, dispatch) and
    // preserves the original PROFILE_SCOPE labels.
    void            waitPreviousBookkeepingAndKvSwaps(const std::list<GenerateStreamPtr>& streams);
    void            prepareGrpcMtpDeviceState(const std::list<GenerateStreamPtr>& streams, TensorHolder& host_holder);
    void            launchTargetVerifyPrepareAsync(const GptModelInputs& model_input, size_t batch_size);
    void            launchDraftPrefillPrepareAsync(const GptModelInputs& model_input);
    GptModelOutputs runTargetVerifyForward(GptModelInputs& model_input, const StreamGroups& stream_groups);
    void            debugCheckLinearBlockMapAtKernelRead(const GptModelInputs& model_input,
                                                         const StreamGroups&   stream_groups) const;
    void            broadcastPostRejectionInputs(GptModelInputs& model_input);
    GptModelOutputs runDSparkProposeForward(GptModelInputs& model_input);
    SamplerOutput   sampleDSparkDraft(const StreamGroups&  stream_groups,
                                      const torch::Tensor& base_logits,
                                      const torch::Tensor& anchors);
    void            dsparkModelDecode(GptModelInputs&                                 model_input,
                                      const StreamGroups&                             stream_groups,
                                      const MtpBatchStreamProcessor::DSparkRoundHead& round_head,
                                      SamplerOutput&                                  draft_sampler_output,
                                      torch::Tensor&                                  draft_token_ids_t,
                                      int64_t&                                        model_forward_us);
    GptModelOutputs runDraftPrefillForward(GptModelInputs& model_input);
    SpecLogitsVerifyRunner::LaunchResult
                 buildSpecLogitsVerifyInline(const std::list<GenerateStreamPtr>& streams,
                                             const torch::Tensor&                draft_tokens,
                                             std::shared_ptr<torch::Event>       draft_tokens_ready_event);
    void         collectDecodeMetrics(const StreamGroups&                          stream_groups,
                                      torch::Event&                                accept_len_ready_event,
                                      const speculative::SpeculativeSamplerOutput& speculative_sampler_output,
                                      MtpMetricsCollector&                         metrics_collector,
                                      int64_t                                      model_forward_us);
    absl::Status dispatchDecodeOutput(const StreamGroups&                          stream_groups,
                                      const std::list<GenerateStreamPtr>&          streams,
                                      const speculative::SpeculativeSamplerOutput& speculative_sampler_output,
                                      GptModelOutputs                              draft_prefill_model_output,
                                      SamplerOutput                                draft_prefill_sampler_output,
                                      std::shared_ptr<torch::Event>                rejection_event,
                                      std::shared_ptr<torch::Event>                draft_event);

    void draftModelDecode(GptModelInputs&             model_input,
                          const StreamGroups&         stream_groups,
                          std::vector<torch::Tensor>& draft_probs_list,
                          torch::Tensor&              draft_token_ids_t,
                          int64_t&                    model_forward_us);

    bool useDeviceInput() const;
    bool checkDeviceInput() const;
    void ensureModelInputsOnCuda(GptModelInputs& model_input, const char* tag);
    void checkModelInputsOnCuda(const GptModelInputs& model_input, const char* tag) const;

    AcceptLenMetricsSnapshot consumePendingAcceptLenMetrics();
    void                     stageAcceptLenMetrics(const torch::Tensor& accept_len,
                                                   torch::Event&        accept_len_ready_event,
                                                   size_t               stream_count,
                                                   std::vector<int32_t> row_priorities);

    void prepareStreams(const std::list<GenerateStreamPtr>& streams,
                        std::list<GenerateStreamPtr>&       prefill_streams,
                        std::list<GenerateStreamPtr>&       decode_streams);

    // Env-gated stream-async switch. Default off unless
    // RTP_LLM_STREAM_ASYNC=1 is exported at server start.
    bool useStreamAsync() const;

    // Device-state feature gates. Defaults stay off; each consumer keeps its
    // own runtime guard before taking an async path.
    bool useAsyncDeviceState() const;

    bool useAsyncPrepare() const;

    // Opt-in gate to skip the broad sync at decodeStep start.
    // Device state, epoch-guarded clears, and single-slotted workers preserve
    // correctness while bookkeeping overlaps the next step.
    bool useDropBroadSync() const;

    // Attach next-step device state, then fork a worker that waits on caller-
    // recorded rejection/draft events and runs D2H/specUpdate/KV release off
    // the main thread.
    absl::Status dispatchDecodeAsync(const StreamGroups&                          stream_groups,
                                     const speculative::SpeculativeSamplerOutput& spec_decode_output,
                                     MergedOutput                                 draft_prefill_output,
                                     std::shared_ptr<torch::Event>                rejection_event,
                                     std::shared_ptr<torch::Event>                draft_event);

    // Synchronous dispatch also publishes the same per-stream device state as
    // the async path, using the host seqLength after specUpdate as truth.
    void publishSyncMtpDeviceState(const StreamGroups&                          stream_groups,
                                   const speculative::SpeculativeSamplerOutput& spec_decode_output,
                                   const MergedOutput&                          draft_prefill_output);

    void releaseAllModelBuffers();
private:
    static torch::Tensor snapshotMutableHostInputToCuda(const torch::Tensor& tensor, TensorHolder& holder);

    GptModelOutputs forwardModel(ModelBase* model, const GptModelInputs& inputs, ModelInputsModelRole role);

    std::unique_ptr<ModelBase>               model_;
    std::unique_ptr<Sampler>                 sampler_;
    std::unique_ptr<MtpBatchStreamProcessor> batch_stream_processor_;
    std::shared_ptr<KVCacheManager>          cache_manager_;
    std::shared_ptr<ModelInputsLogger>       model_inputs_logger_;
    bool                                     enable_ffn_disaggregate_ = false;
    bool                                     enable_detail_log_       = false;
    int                                      tp_rank_                 = 0;
    ParallelismConfig                        parallelism_config_;
    kmonitor::MetricsReporterPtr             metrics_reporter_ = nullptr;
    MetricsLoopReporter<RtpLLMTokenPSMetrics, RtpLLMTokenPSMetricsCollector> tps_reporter_;
    WallClockMetricsLoopReporter<RtpLLMWallClockTokenPSMetrics, RtpLLMTokenPSMetricsCollector>
        wall_tps_reporter_;
    std::shared_ptr<ExpertBalancer>                                          expert_balancer_;
    size_t                                                                   vocab_size_;

    // for mtp
    DataType data_type_;
    size_t   hidden_size_;
    size_t   propose_step_;
    // Fixed-width block diffusion: one draft forward emits gamma proposals;
    // unlike MTP there is no autoregressive draft loop or hidden-state chain.
    bool          is_dspark_ = false;
    size_t        draft_vocab_size_;
    torch::Tensor dspark_markov_w1_;
    torch::Tensor dspark_markov_w2_;
    // The two wrappers keep stable runtime roles. Ordinary MTP uses decode and
    // post-verify prefill roles; DSpARK supplies propose and commit construction
    // parameters to the same two slots.
    std::shared_ptr<ModelBase>                       draft_model_;
    std::shared_ptr<ModelBase>                       sp_prefill_draft_model_;
    std::unique_ptr<speculative::SpeculativeSampler> speculative_sampler_;
    std::unique_ptr<speculative::FastTopKSampler>    fast_topk_sampler_;

    // Keeps async copy source tensors alive across release points.
    TensorHolder buffer_holder_;

    bool     warm_up_;
    RoleType role_type_;
    // True when any KV-cache group is CacheGroupType::LINEAR (RWKV / Mamba /
    // hybrid linear+full). Per-step state advances every token, so the page
    // table must be re-gathered between draft propose and target verify.
    bool is_linear_attention_model_ = false;

    torch::Tensor d2t_map_;

    torch::Stream collect_metrics_stream_;
    torch::Tensor metrics_accept_len_sum_gpu_;
    torch::Tensor metrics_accept_len_sum_cpu_;
    // Per-stream accept_len rows staged next to the scalar sum so priority
    // buckets can be rebuilt on consume without an extra sync.
    torch::Tensor                 metrics_accept_len_rows_gpu_;
    torch::Tensor                 metrics_accept_len_rows_cpu_;
    std::vector<int32_t>          metrics_accept_len_row_priorities_;
    std::shared_ptr<torch::Event> metrics_accept_len_ready_event_;
    int64_t                       metrics_accept_len_stream_num_        = 0;
    int64_t                       metrics_accept_len_propose_token_num_ = 0;

    AsyncRunner target_verify_prepare_runner_;
    AsyncRunner draft_prefill_prepare_runner_;
    // Declare the worker after its target so destruction joins the worker first.
    std::unique_ptr<SpecLogitsVerifyRunner> spec_logits_verify_runner_;
    AsyncRunner                             spec_logits_verify_async_runner_;

    // Bookkeeping worker for stream-async decode dispatch. It owns a CUDA
    // stream + thread and runs D2H/specUpdate/KV release off the main thread.
    AsyncRunner spec_bookkeeping_runner_;
};
}  // namespace rtp_llm
