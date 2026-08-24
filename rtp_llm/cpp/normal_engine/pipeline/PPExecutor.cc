#include "rtp_llm/cpp/normal_engine/pipeline/PPExecutor.h"

#include <algorithm>
#include <utility>

#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cuda_graph/cuda_graph_device_shims.h"
#include "rtp_llm/cpp/engine_base/EngineInitParams.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/models/ModelTypes.h"
#include "rtp_llm/cpp/models/PyWrappedModel.h"
#include "rtp_llm/cpp/models/Sampler.h"
#include "rtp_llm/cpp/models/eplb/ExpertBalancer.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"

namespace rtp_llm {
namespace {

torch::Tensor cloneIfDefined(const torch::Tensor& tensor) {
    return tensor.defined() ? tensor.clone() : torch::Tensor();
}

// Initial PP only supports one input beam and one output beam. Stateful logits
// processors are not represented by the initial PP sampling plan.
SamplerInputs makeSamplerInputs(const PPSamplingPlan& plan, const GptModelOutputs& model_output) {
    SamplerInputs inputs;
    inputs.logits               = model_output.logits;
    inputs.token_ids            = cloneIfDefined(plan.token_ids);
    inputs.input_lengths        = cloneIfDefined(plan.input_lengths);
    inputs.sequence_lengths     = cloneIfDefined(plan.sequence_lengths);
    inputs.top_k                = cloneIfDefined(plan.top_k);
    inputs.top_p                = cloneIfDefined(plan.top_p);
    inputs.temperature          = cloneIfDefined(plan.temperature);
    inputs.repetition_penalty   = cloneIfDefined(plan.repetition_penalty);
    inputs.presence_penalty     = cloneIfDefined(plan.presence_penalty);
    inputs.frequency_penalty    = cloneIfDefined(plan.frequency_penalty);
    inputs.no_repeat_ngram_size = cloneIfDefined(plan.no_repeat_ngram_size);
    inputs.do_sample            = cloneIfDefined(plan.do_sample);
    inputs.finished_mask        = cloneIfDefined(plan.finished_mask);
    inputs.cum_log_probs        = cloneIfDefined(plan.cum_log_probs);

    const auto batch_size = static_cast<int64_t>(plan.batch_size);
    inputs.batch_size     = plan.batch_size;
    inputs.batch_size_out = plan.batch_size;
    inputs.step           = plan.step;
    inputs.vocab_size     = static_cast<size_t>(model_output.logits.size(-1));
    inputs.num_beams_in   = torch::ones({batch_size}, torch::kLong);
    inputs.num_beams_out  = torch::ones({batch_size}, torch::kLong);
    inputs.generator.resize(plan.batch_size);

    // The current Sampler API consumes at::Generator objects. PP transports
    // seed/offset tensors instead, so stochastic sampling will be wired when
    // Sampler accepts explicit Philox state. The initial greedy path does not
    // consume generator state.
    (void)plan.random_seeds;
    (void)plan.random_offsets;
    return inputs;
}

PPSampleResult makeSampleResult(const PPSamplingPlan& plan, SamplerOutput sampler_output) {
    PPSampleResult result;

    const auto& token_ids = sampler_output.token_ids;
    if (token_ids.defined() && token_ids.dim() > 1 && token_ids.size(-1) > 0) {
        result.new_token_ids = token_ids.narrow(-1, token_ids.size(-1) - 1, 1);
    } else {
        result.new_token_ids = token_ids;
    }
    result.success       = std::move(sampler_output.success);
    result.cum_log_probs = sampler_output.cum_log_probs.defined() ? std::move(sampler_output.cum_log_probs) :
                                                                    torch::empty({0}, torch::kFloat32);
    // Greedy sampling does not advance Philox state. Keep the plan-provided
    // offset unchanged until explicit seed/offset sampling is connected.
    result.next_random_offsets =
        plan.random_offsets.defined() ? plan.random_offsets.clone() : torch::empty({0}, torch::kInt64);
    return result;
}

}  // namespace

PPExecutor::InflightBatch::InflightBatch(PPExecutionPlan execution_plan):
    plan(std::move(execution_plan)), forward_done(cuda_graph::makeGraphEvent()) {}

PPExecutor::PPExecutor(const EngineInitParams&                params,
                       const std::shared_ptr<KVCacheManager>& cache_manager,
                       MlaOpsType                             mla_ops_type,
                       std::function<void()>                  profile_step_start,
                       std::function<void()>                  profile_step_finish):
    parallelism_config_(params.parallelism_config),
    pp_rank_(parallelism_config_.world_rank / parallelism_config_.tp_size),
    cache_manager_(cache_manager),
    profile_step_start_(std::move(profile_step_start)),
    profile_step_finish_(std::move(profile_step_finish)),
    slots_(static_cast<size_t>(parallelism_config_.pp_size + 1)) {
    RTP_LLM_FAIL(
        "PPTransport backend is not implemented: per-rank plan receive and same-TP-lane tensor P2P are required");

    const size_t runtime_tokens_per_block        = cache_manager_ ? cache_manager_->cacheConfig().seq_size_per_block :
                                                                    params.model_config_.attn_config.tokens_per_block;
    const size_t runtime_kernel_tokens_per_block = cache_manager_ ?
                                                       cache_manager_->cacheConfig().kernel_seq_size_per_block :
                                                       params.model_config_.attn_config.kernel_tokens_per_block;

    GptModelInitParams model_init_params(
        {params.gpt_weights,
         genModelDescription(params.model_config_, params.parallelism_config, params.eplb_config, params.moe_config),
         cache_manager_ ? std::make_optional(cache_manager_->getMainModelGroupedCacheLayerLayout()) : std::nullopt,
         params.model_id,
         params.parallelism_config,
         params.hw_kernel_config,
         params.profiling_debug_logging_config,
         params.runtime_config,
         params.concurrency_config,
         params.sp_config,
         params.device_resource_config,
         mla_ops_type,
         params.model_config_.max_seq_len,
         params.model_config_.hidden_size,
         runtime_tokens_per_block,
         runtime_kernel_tokens_per_block,
         cache_manager_,
         std::nullopt,
         params.model_config_.hc_mult});

    if (!params.py_model.is_none()) {
        model_ = std::make_unique<PyWrappedModel>(model_init_params, params.py_model);
    } else {
        RTP_LLM_LOG_WARNING("PPExecutor py_model is None; stage forward is unavailable");
    }

    const bool is_last_stage = pp_rank_ + 1 == parallelism_config_.pp_size;
    if (is_last_stage && parallelism_config_.tp_rank == 0) {
        const auto initial_sampler_batch_size =
            static_cast<size_t>(std::max<int64_t>(1, params.runtime_config.max_generate_batch_size));
        sampler_ = std::make_unique<Sampler>(SamplerInitParams{initial_sampler_batch_size, false});
    }

    if (params.eplb_config.enable_eplb() && params.model_config_.moe_style != 0) {
        const int   first_moe_layer = params.model_config_.moe_layer_index.front();
        const auto& moe_kernel      = params.gpt_weights.layers[first_moe_layer].ffn_weights.moe_gate_weight->kernel;
        const auto  moe_weight_type = torchDTypeToDataType(moe_kernel.dtype());
        const bool  is_gated_activation = params.model_config_.isGatedActivation();
        const auto  moe_inter_size      = is_gated_activation ? moe_kernel.size(1) / 2 : moe_kernel.size(1);

        expert_balancer_ =
            std::make_shared<ExpertBalancer>(params.model_config_.expert_num,
                                             params.eplb_config.phy_exp_num(params.model_config_.expert_num),
                                             params.model_config_.num_layers,
                                             moe_inter_size,
                                             params.model_config_.hidden_size,
                                             params.parallelism_config.ep_rank,
                                             params.parallelism_config.ep_size,
                                             params.parallelism_config.world_size,
                                             params.py_eplb,
                                             moe_weight_type,
                                             params.model_config_.quant_algo,
                                             params.metrics_reporter,
                                             params.eplb_config);
    }
}

PPExecutor::~PPExecutor() {
    requestStop();
    for (auto& slot : slots_) {
        if (slot.has_value()) {
            slot->forward_done.synchronize();
        }
    }
}

void PPExecutor::requestStop() {
    if (!stop_requested_.exchange(true, std::memory_order_acq_rel) && transport_) {
        transport_->abort();
    }
}

void PPExecutor::runStage(InflightBatch& batch, GptModelInputs& local_model_input) {
    RTP_LLM_CHECK_WITH_INFO(model_ != nullptr, "PP stage model is not initialized");

    const bool is_last_stage = pp_rank_ + 1 == parallelism_config_.pp_size;
    if (!is_last_stage) {
        batch.output_tensors.emplace();
    }

    // runStage executes only the model work owned by this PP stage. Plan
    // construction and inter-stage communication remain in process().
    model_->releaseBuffers();
    if (cache_manager_ && local_model_input.kv_cache_update_mapping.defined()) {
        cache_manager_->blockBatchCopy(local_model_input.kv_cache_update_mapping);
    }

    batch.model_output = model_->forwardPP(local_model_input,
                                           batch.input_tensors ? &*batch.input_tensors : nullptr,
                                           batch.output_tensors ? &*batch.output_tensors : nullptr);
    if (expert_balancer_) {
        RtpLLMExecutorMetricsCollector collector;
        expert_balancer_->stepForward(*model_, collector);
    }

    if (!is_last_stage) {
        return;
    }

    if (parallelism_config_.tp_rank == 0) {
        auto sampler_inputs = makeSamplerInputs(batch.plan.sampling_plan, batch.model_output);
        auto sampler_output = sampler_->forward(sampler_inputs);
        batch.sample_result = makeSampleResult(batch.plan.sampling_plan, std::move(sampler_output));
    }
}

absl::Status PPExecutor::process(const std::list<GenerateStreamPtr>& streams, int64_t) {
    if (stop_requested_.load(std::memory_order_acquire)) {
        return absl::OkStatus();
    }

    const bool is_stage_root  = parallelism_config_.tp_rank == 0;
    const bool is_first_stage = pp_rank_ == 0;
    const bool is_last_stage  = pp_rank_ + 1 == parallelism_config_.pp_size;

    PPExecutionPlan plan;
    if (is_first_stage) {
        // The first-stage TP root creates the execution plan that starts the
        // pipeline batch selected by Scheduler.
        if (is_stage_root && !streams.empty()) {
            // TODO: Convert the selected streams into the inputs needed by
            // stage forward and last-stage sampling.
        }
    } else {
        // Receiving one plan advances this TP lane to the next batch. TP rank 0
        // receives the populated plan; other TP lanes receive an empty plan.
        plan = transport_->receivePlan();
    }

    // Keep each batch's resources for a full turn of the circular window.
    // Only the slot selected for reuse is waited on and released.
    auto& slot = slots_[current_slot_];
    if (slot.has_value()) {
        auto& previous_batch = *slot;
        if (previous_batch.plan_send) {
            previous_batch.plan_send->wait();
        }
        if (previous_batch.output_send) {
            // The output send is ordered after forward_done, so completing the
            // ticket also guarantees that the forward resources are reusable.
            previous_batch.output_send->wait();
        } else {
            previous_batch.forward_done.synchronize();
        }
        slot.reset();
    }

    slot.emplace(std::move(plan));
    auto& batch = *slot;

    // Keep the wire plan unchanged while asyncSendPlan borrows it. Stage
    // execution uses a shallow copy whose fields TP synchronization may replace.
    GptModelInputs local_model_input = batch.plan.logical_model_input;

    // Send one plan on every TP lane so the next stage can advance. TP rank 0
    // carries the populated plan; other TP lanes carry an empty plan.
    if (!is_last_stage) {
        batch.plan_send = transport_->asyncSendPlan(batch.plan);
    }

    // Starts the tensor receive before the TP model-input broadcast so the
    // receive overlaps with stage-local input synchronization.
    if (!is_first_stage) {
        batch.input_tensors.emplace();
        batch.tensors_recv = transport_->asyncReceiveTensors(*batch.input_tensors);
    }

    tpSyncModelInputs(local_model_input, parallelism_config_);

    if (batch.tensors_recv) {
        batch.tensors_recv->wait();
        batch.tensors_recv.reset();
    }

    if (profile_step_start_) {
        profile_step_start_();
    }

    runStage(batch, local_model_input);

    batch.forward_done.record(cuda_graph::graphGetCurrentStream());

    if (!is_last_stage) {
        batch.output_send = transport_->asyncSendTensors(*batch.output_tensors, batch.forward_done);
    } else if (is_stage_root) {
        batch.output_send = transport_->asyncSendResult(*batch.sample_result, batch.forward_done);
    }

    if (profile_step_finish_) {
        profile_step_finish_();
    }

    current_slot_ = (current_slot_ + 1) % slots_.size();
    return absl::OkStatus();
}

}  // namespace rtp_llm
