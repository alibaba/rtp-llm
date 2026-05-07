#pragma once

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
#include "rtp_llm/cpp/normal_engine/AsyncRunner.h"
#include "rtp_llm/cpp/normal_engine/speculative/SpeculativeSampler.h"

namespace rtp_llm {

struct MtpMetricsCollector {
    RtpLLMExecutorMetricsCollector          executor_collector;
    RtpLLMTokenPSMetricsCollector           tps_collector;
    RtpLLMSpeculativeEngineMetricsCollector sp_engine_collector;

    bool not_skip = false;
};

class MtpBufferHolder {
public:
    void hold(const torch::Tensor& tensor) {
        tensor_holder_.push_back(tensor);
    }

    void hold(const GptModelInputs& model_input) {
        tensor_holder_.push_back(model_input.combo_tokens);
        tensor_holder_.push_back(model_input.input_lengths);
        tensor_holder_.push_back(model_input.sequence_lengths);
        tensor_holder_.push_back(model_input.lm_output_indexes);
        tensor_holder_.push_back(model_input.prefix_lengths);
        tensor_holder_.push_back(model_input.sequence_lengths_plus_1);
    }

    void release() {
        tensor_holder_.clear();
    }

private:
    std::vector<torch::Tensor> tensor_holder_;
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

    absl::Status process(const std::list<GenerateStreamPtr>& streams) override;
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
    bool isTpRank0() const;

    void maybePrintModelInput(const GptModelInputs& model_input, const std::string& prefix) const;

    absl::Status prefillStep(const std::list<GenerateStreamPtr>& streams, MtpMetricsCollector& metrics_collector);

    absl::Status decodeStep(const std::list<GenerateStreamPtr>& streams, MtpMetricsCollector& metrics_collector);

    void draftModelDecode(GptModelInputs&             model_input,
                          const StreamGroups&         stream_groups,
                          std::vector<torch::Tensor>& draft_probs_list,
                          torch::Tensor&              draft_token_ids_t);

    bool useMtpDeviceInput() const;
    bool checkMtpDeviceInput() const;
    void ensureMtpModelInputsOnCuda(GptModelInputs& model_input, const char* tag);
    void checkMtpModelInputsOnCuda(const GptModelInputs& model_input, const char* tag) const;

    void prepareStreams(const std::list<GenerateStreamPtr>& streams,
                        std::list<GenerateStreamPtr>&       prefill_streams,
                        std::list<GenerateStreamPtr>&       decode_streams);

    // Env-gated stream-async switch. Default off unless
    // RTP_LLM_STREAM_ASYNC=1 is exported at server start.
    bool useStreamAsync() const;

    // Device-state feature gates. Defaults stay off; each consumer keeps its
    // own runtime guard before taking an async path.
    bool useAsyncDeviceState() const;
    bool useAsyncHostMirror() const;
    bool useAsyncStopExtra() const;

    // Opt-in gate to skip the broad spec_bookkeeping_runner_.sync() at the
    // start of decodeStep. When enabled, worker bookkeeping overlaps the next
    // step's main-thread work; epoch-guarded clears and device state preserve
    // correctness until the next dispatchDecodeAsync single-slots the worker.
    bool useDropBroadSync() const;

    // Stream-async dispatch. The caller records rejection_event after rejection
    // sampling and draft_event after draft_model_sample. This method attaches
    // device-resident next-step state to each stream, then forks a worker that
    // waits on those events and performs D2H/specUpdate/KV release off the
    // main thread.
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

private:
    std::unique_ptr<ModelBase>               model_;
    std::unique_ptr<Sampler>                 sampler_;
    std::unique_ptr<MtpBatchStreamProcessor> batch_stream_processor_;
    std::shared_ptr<KVCacheManager>          cache_manager_;
    bool                                     enable_ffn_disaggregate_ = false;
    bool                                     enable_detail_log_       = false;
    int                                      tp_rank_                 = 0;
    ParallelismConfig                        parallelism_config_;
    kmonitor::MetricsReporterPtr             metrics_reporter_ = nullptr;
    std::shared_ptr<ExpertBalancer>          expert_balancer_;
    size_t                                   vocab_size_;

    // for mtp
    DataType                                         data_type_;
    size_t                                           hidden_size_;
    size_t                                           propose_step_;
    size_t                                           draft_vocab_size_;
    std::shared_ptr<ModelBase>                       draft_model_;
    std::shared_ptr<ModelBase>                       sp_prefill_draft_model_;
    std::unique_ptr<speculative::SpeculativeSampler> speculative_sampler_;
    std::unique_ptr<speculative::FastTopKSampler>    fast_topk_sampler_;

    // holder for host buffers to avoid early free before H2D copy kernel execution
    MtpBufferHolder buffer_holder_;

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

    torch::Stream collect_metrics_stream_;

    AsyncRunner target_verify_prepare_runner_;
    AsyncRunner draft_prefill_prepare_runner_;

    // Bookkeeping worker for stream-async decode dispatch. It owns a CUDA
    // stream + thread and runs D2H/specUpdate/KV release off the main thread.
    AsyncRunner spec_bookkeeping_runner_;
};
};  // namespace rtp_llm
