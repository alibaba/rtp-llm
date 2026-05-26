#pragma once

#include <atomic>
#include <memory>
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
#include "rtp_llm/cpp/normal_engine/speculative/SpeculativeSampler.h"

namespace rtp_llm {

struct MtpMetricsCollector {
    RtpLLMExecutorMetricsCollector          executor_collector;
    RtpLLMTokenPSMetricsCollector           tps_collector;
    RtpLLMSpeculativeEngineMetricsCollector sp_engine_collector;

    bool not_skip = false;
};

class MtpExecutor: public Executor {
public:
    explicit MtpExecutor(const EngineInitParams&                        params,
                         std::unique_ptr<ProposeModelEngineInitParams>& propose_params,
                         const std::shared_ptr<KVCacheManager>&         cache_manager,
                         MlaOpsType                                     mla_ops_type            = MlaOpsType::AUTO,
                         int32_t                                        kv_cache_group_num      = 1,
                         const std::vector<int32_t>&                    kv_cache_layer_to_group = {},
                         bool                                           warm_up                 = false);

    absl::Status process(const std::list<GenerateStreamPtr>& streams, int64_t schedule_time_us = 0) override;
    bool         updateEplbConfig(const EPLBConfig& config) override;
    void         notifyStop() override;

    void setTargetModel(std::unique_ptr<ModelBase> model) {
        model_ = std::move(model);
    }

    void setDraftModel(std::unique_ptr<ModelBase> model) {
        draft_model_ = std::move(model);
    }

    void setBatchProcessor(std::unique_ptr<MtpBatchStreamProcessor> processor) {
        batch_stream_processor_ = std::move(processor);
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
                                                       int                    vocab_size);

protected:
    struct AcceptLenMetricsSnapshot {
        int64_t total_accept_len        = 0;
        int64_t total_stream_num        = 0;
        int64_t total_propose_token_num = 0;
        bool    valid                   = false;
    };

    bool isTpRank0() const;

    void maybePrintModelInput(const GptModelInputs& model_input, const std::string& prefix) const;

    absl::Status prefillStep(const std::list<GenerateStreamPtr>& streams,
                             MtpMetricsCollector&                metrics_collector,
                             int64_t                             schedule_time_us);

    absl::Status decodeStep(const std::list<GenerateStreamPtr>& streams, MtpMetricsCollector& metrics_collector);

    // decodeStep helpers — extracted to keep decodeStep readable. Each helper
    // owns a single phase (sync, prepare, forward, broadcast, dispatch) and
    // preserves the original PROFILE_SCOPE labels.
    void            waitPreviousBookkeepingAndKvSwaps(const std::list<GenerateStreamPtr>& streams);
    GptModelOutputs runTargetVerifyForward(GptModelInputs& model_input, const StreamGroups& stream_groups);
    void            debugCheckLinearBlockMapAtKernelRead(const GptModelInputs& model_input,
                                                         const StreamGroups&   stream_groups) const;
    void            broadcastPostRejectionInputs(GptModelInputs& model_input, const StreamGroups& stream_groups);
    GptModelOutputs runDraftPrefillForward(GptModelInputs& model_input);
    void            collectDecodeMetrics(const StreamGroups&                          stream_groups,
                                         torch::Event&                                accept_len_ready_event,
                                         const speculative::SpeculativeSamplerOutput& speculative_sampler_output,
                                         MtpMetricsCollector&                         metrics_collector);
    absl::Status    dispatchDecodeOutput(const StreamGroups&                          stream_groups,
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
    void
    stageAcceptLenMetrics(const torch::Tensor& accept_len, torch::Event& accept_len_ready_event, size_t stream_count);

    void prepareStreams(const std::list<GenerateStreamPtr>& streams,
                        std::list<GenerateStreamPtr>&       prefill_streams,
                        std::list<GenerateStreamPtr>&       decode_streams);

    // Spec-decode hand-off: when the source model exposes a pre-output-projection
    // residual buffer (DSv4 pre-hc [T, hc*D]), swap it into the C++ hidden-state
    // carrier. The source returns the full buffer; consumers slice as needed.
    void maybeOverrideLastHiddenWithMtpBuffer(GptModelInputs& model_input,
                                              ModelBase&      source,
                                              bool            request_actual_rows = false);
    void maybeOverrideLastHiddenWithMtpBuffer(GptModelOutputs& model_output, ModelBase& source);

    // Env-gated stream-async switch. Default off unless
    // RTP_LLM_STREAM_ASYNC=1 is exported at server start.
    bool useStreamAsync() const;

    // Device-state feature gates. Defaults stay off; each consumer keeps its
    // own runtime guard before taking an async path.
    bool useAsyncDeviceState() const;

    // Opt-in gate to skip the broad sync at decodeStep start.
    // Device state, epoch-guarded clears, and single-slotted workers preserve
    // correctness while bookkeeping overlaps the next step.
    bool useDropBroadSync() const;

    bool shouldSkipFakeStreamForStop(const GptModelInputs& model_input, const char* phase) const;

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
    std::unique_ptr<ModelBase>                                               model_;
    std::unique_ptr<Sampler>                                                 sampler_;
    std::unique_ptr<MtpBatchStreamProcessor>                                 batch_stream_processor_;
    std::shared_ptr<KVCacheManager>                                          cache_manager_;
    bool                                                                     enable_ffn_disaggregate_ = false;
    bool                                                                     enable_detail_log_       = false;
    int                                                                      tp_rank_                 = 0;
    ParallelismConfig                                                        parallelism_config_;
    kmonitor::MetricsReporterPtr                                             metrics_reporter_ = nullptr;
    MetricsLoopReporter<RtpLLMTokenPSMetrics, RtpLLMTokenPSMetricsCollector> tps_reporter_;
    std::shared_ptr<ExpertBalancer>                                          expert_balancer_;
    size_t                                                                   vocab_size_;

    // for mtp
    DataType                                         data_type_;
    size_t                                           hidden_size_;
    size_t                                           propose_step_;
    size_t                                           draft_vocab_size_;
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

    // group id tensors
    torch::Tensor target_kv_cache_layer_to_group;
    torch::Tensor draft_kv_cache_layer_to_group;

    torch::Tensor d2t_map_;

    torch::Stream                 collect_metrics_stream_;
    torch::Tensor                 metrics_accept_len_sum_gpu_;
    torch::Tensor                 metrics_accept_len_sum_cpu_;
    std::shared_ptr<torch::Event> metrics_accept_len_ready_event_;
    int64_t                       metrics_accept_len_stream_num_        = 0;
    int64_t                       metrics_accept_len_propose_token_num_ = 0;

    // Bookkeeping worker for stream-async decode dispatch. It owns a CUDA
    // stream + thread and runs D2H/specUpdate/KV release off the main thread.
    AsyncRunner spec_bookkeeping_runner_;

    std::atomic<bool> stop_requested_{false};
};
};  // namespace rtp_llm
