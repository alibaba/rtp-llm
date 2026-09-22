#include "rtp_llm/cpp/normal_engine/pipeline/PPExecutor.h"

#include "rtp_llm/cpp/normal_engine/pipeline/PPSerialization.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <utility>

#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cuda_graph/cuda_graph_device_shims.h"
#include "rtp_llm/cpp/engine_base/EngineInitParams.h"
#include "rtp_llm/cpp/engine_base/ProposeModelEngineInitParams.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/models/ModelTypes.h"
#include "rtp_llm/cpp/models/ModelInputsLogger.h"
#include "rtp_llm/cpp/models/PyWrappedModel.h"
#include "rtp_llm/cpp/models/Sampler.h"
#include "rtp_llm/cpp/models/eplb/ExpertBalancer.h"
#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorFactory.h"
#include "rtp_llm/cpp/models/logits_processor/SpecLogitsVerifyRunner.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/normal_engine/pipeline/PPBatchStreamProcessor.h"
#include "rtp_llm/cpp/normal_engine/speculative/MtpCompute.h"
#include "rtp_llm/cpp/normal_engine/speculative/MtpExecutor.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#include "rtp_llm/cpp/utils/StatusUtil.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"

namespace rtp_llm {

PPExecutor::ModelFactory PPExecutor::test_model_factory = nullptr;

GenerateStreamPtr PPExecutor::createMinFakePrefillStream(const ModelConfig&                model_config,
                                                         const RuntimeConfig&              runtime_config,
                                                         const ResourceContext&            resource_context,
                                                         const SpeculativeExecutionConfig& sp_config,
                                                         RoleType                          role_type) {
    const auto propose_step = sp_config.type == SP_TYPE_NONE ? 0 : sp_config.gen_num_per_cycle;
    /** Entries reference block 0. PDFUSION also covers the draft proposal window;
     * a separate P stage only needs one MTP/EAGLE draft forward or a DSpARK commit. */
    const size_t reserved_blocks = role_type == RoleType::PDFUSION ? propose_step + 1 : 1;
    return makeFakeStream(1, reserved_blocks, model_config, runtime_config, resource_context);
}

GenerateStreamPtr PPExecutor::createMinFakeDecodeStream(const ModelConfig&                model_config,
                                                        const RuntimeConfig&              runtime_config,
                                                        const ResourceContext&            resource_context,
                                                        const SpeculativeExecutionConfig& sp_config) {
    const auto propose_step = sp_config.type == SP_TYPE_NONE ? 0 : sp_config.gen_num_per_cycle;
    /** Cover target verification and the next draft round, including DSpARK's
     * proposal window. All entries reference block 0 without allocating real KV blocks. */
    const size_t reserved_blocks = 2 * (propose_step + 1);
    auto         fake_stream     = makeFakeStream(1, reserved_blocks, model_config, runtime_config, resource_context);

    /** Seed [prompt, t0] with a one-token request budget. The initialization may
     * mark the request done; PP fake execution skips request sampling and dispatch. */
    StreamUpdateInfo update_info{torch::zeros({1, 1}, torch::kInt32),
                                 1,
                                 torch::Tensor(),
                                 torch::Tensor(),
                                 torch::Tensor(),
                                 torch::Tensor(),
                                 torch::Tensor(),
                                 torch::Tensor(),
                                 torch::Tensor(),
                                 torch::Tensor(),
                                 false};
    fake_stream->update(update_info);

    if (sp_config.type != SP_TYPE_NONE) {
        /** PP verifies immediately; proposals stay separate from the formal history. */
        auto sp_buffer          = std::make_shared<SpeculativeExecutorStreamOutput>();
        sp_buffer->propose_step = propose_step;
        sp_buffer->tokens       = torch::zeros({1, propose_step + 1}, torch::kInt32);
        fake_stream->setSPOutputBuffer(sp_buffer);
    }
    return fake_stream;
}

void PPExecutor::InflightBatch::reset() {
    skip_run         = true;
    stream_groups    = StreamGroups();
    schedule_time_us = 0;
}

void PPExecutor::sendObject(const torch::Tensor& object, PPTickets& tickets) {
    auto object_size = torch::tensor({object.numel()}, torch::kInt64);
    tickets.push_back(transport_->asyncSend(object_size));
    tickets.push_back(transport_->asyncSend(object));
}

torch::Tensor PPExecutor::receiveObject() {
    auto object_size  = torch::empty({1}, torch::TensorOptions().dtype(torch::kInt64));
    auto size_receive = transport_->asyncReceive(object_size);
    waitTicket(*size_receive, "object size from previous stage");

    auto object         = torch::empty({object_size.item<int64_t>()}, torch::TensorOptions().dtype(torch::kUInt8));
    auto object_receive = transport_->asyncReceive(object);
    waitTicket(*object_receive, "object payload from previous stage");
    return object;
}

void PPExecutor::asyncSendPlan(const PPExecutionPlan& plan, bool empty_plan, PPTickets& tickets) {
    sendObject(pp_serialization::serializePlan(plan, empty_plan), tickets);
}

PPExecutionPlan PPExecutor::receivePlan() {
    auto plan = pp_serialization::deserializePlan(receiveObject());
    if (plan.model_input.shutdown) {
        RTP_LLM_LOG_INFO("received pipeline shutdown sentinel from previous stage");
    }
    return plan;
}

void PPExecutor::asyncSendExecutionResult(const PPExecutionResult& result, PPTickets& tickets) {
    sendObject(pp_serialization::serializeExecutionResult(result), tickets);
}

void PPExecutor::asyncSendTensors(const PPIntermediateTensors& tensors, PPTickets& tickets) {
    sendObject(pp_serialization::serializeTensorsMetadata(tensors), tickets);
    for (const auto& tensor_entry : tensors.tensors) {
        if (tensor_entry.second.numel() != 0) {
            tickets.push_back(transport_->asyncSend(tensor_entry.second));
        }
    }
}

PPIntermediateTensors PPExecutor::receiveTensors(PPTickets& tickets) {
    auto tensors = pp_serialization::deserializeTensorsMetadata(receiveObject());
    for (auto& tensor_entry : tensors.tensors) {
        if (tensor_entry.second.numel() != 0) {
            tickets.push_back(transport_->asyncReceive(tensor_entry.second));
        }
    }
    return tensors;
}

void PPExecutor::waitTicket(PPCommTicket& ticket, const char* what, bool throw_on_timeout) {
    const bool armed = stopping_;
    if (!armed) {
        /* Running: unbounded wait, which also drives backend progress; the shutdown
         * window is re-evaluated by the next wait once data arrives. */
        try {
            ticket.wait();
        } catch (const std::exception& e) {
            RTP_LLM_LOG_ERROR("PP comm failed while running (peer likely died): %s", e.what());
            throw PPCommWatchdogTimeout(std::string("peer communication failed: ") + e.what());
        }
        return;
    }
    /* Stopping: bound the wait for the final frames/sentinel. Not delivering within the
     * bound means the peer is gone and this stage must exit. */
    if (ticket.wait(std::chrono::milliseconds(comm_watchdog_timeout_ms_))) {
        return;
    }
    RTP_LLM_LOG_ERROR("PP comm watchdog: %s not received within %ld ms after shutdown started%s",
                      what,
                      static_cast<long>(comm_watchdog_timeout_ms_),
                      throw_on_timeout ? "; aborting wait" : "; ignored during teardown");
    if (throw_on_timeout) {
        throw PPCommWatchdogTimeout(std::string("timed out waiting for ") + what);
    }
}

void PPExecutor::waitAll(PPTickets& tickets, const char* what, bool throw_on_timeout) {
    for (auto& ticket : tickets) {
        waitTicket(*ticket, what, throw_on_timeout);
    }
    tickets.clear();
}

absl::Status PPExecutor::processExecutionResult(InflightBatch& batch) {
    /** Fake batches still complete the PP round trip before their result is discarded. */
    auto result = pp_serialization::deserializeExecutionResult(receiveObject());
    if (batch.stream_groups.isFakeStream()) {
        RTP_LLM_LOG_DEBUG("PP fake batch completed: dp_rank=%ld", parallelism_config_.dp_rank);
        return absl::OkStatus();
    }
    return batch_stream_processor_->dispatchExecutionResult(batch.stream_groups, result);
}

PPExecutor::PPExecutor(const EngineInitParams&                params,
                       const std::shared_ptr<KVCacheManager>& cache_manager,
                       bool                                   warm_up,
                       MlaOpsType                             mla_ops_type,
                       std::function<void()>                  profile_step_start,
                       std::function<void()>                  profile_step_finish,
                       ProposeModelEngineInitParams*          propose_params):
    Executor(),
    warm_up_(warm_up),
    role_type_(params.pd_sep_config.role_type),
    cache_manager_(cache_manager),
    sp_enabled_(params.sp_config.type != SP_TYPE_NONE),
    is_dspark_(params.sp_config.type == SP_TYPE_DSPARK),
    dspark_mask_token_id_(static_cast<int32_t>(params.sp_config.sp_dspark_mask_token_id)),
    propose_step_(params.sp_config.gen_num_per_cycle),
    position_id_len_factor_(params.model_config_.attn_config.rope_config.index_factor),
    parallelism_config_(params.parallelism_config),
    pp_layout_(RankLayout::fromParallelismConfig(parallelism_config_)),
    slots_(parallelism_config_.pp_size + 1),
    profile_step_start_(std::move(profile_step_start)),
    profile_step_finish_(std::move(profile_step_finish)),
    metrics_reporter_(params.metrics_reporter),
    tps_reporter_(MetricsLoopReporter<RtpLLMTokenPSMetrics, RtpLLMTokenPSMetricsCollector>(
        isFirstStage() && isStageRoot() && !warm_up_ ? metrics_reporter_ : nullptr)),
    wall_tps_reporter_(WallClockMetricsLoopReporter<RtpLLMWallClockTokenPSMetrics, RtpLLMTokenPSMetricsCollector>(
        isFirstStage() && isStageRoot() && !warm_up_ ? metrics_reporter_ : nullptr)) {
    const char* stream_async = std::getenv("RTP_LLM_STREAM_ASYNC");
    RTP_LLM_CHECK_WITH_INFO(stream_async == nullptr || std::strcmp(stream_async, "1") != 0,
                            "pipeline parallelism does not support async runner (RTP_LLM_STREAM_ASYNC)");
    RTP_LLM_CHECK_WITH_INFO(params.sp_config.type == SP_TYPE_NONE || params.sp_config.type == SP_TYPE_MTP
                                || params.sp_config.type == SP_TYPE_EAGLE || params.sp_config.type == SP_TYPE_DSPARK,
                            "pipeline parallelism only supports MTP, EAGLE and DSpARK speculative decoding");
    RTP_LLM_CHECK_WITH_INFO(!params.ffn_disaggregate_config.enable_ffn_disaggregate,
                            "pipeline parallelism does not support FFN disaggregation");
    RTP_LLM_CHECK_WITH_INFO(!parallelism_config_.enable_sp && parallelism_config_.ffn_sp_size == 1,
                            "pipeline parallelism does not support sequence parallelism");
    RTP_LLM_CHECK_WITH_INFO(!parallelism_config_.use_ub_comm,
                            "pipeline parallelism does not support user-buffer communication");
    RTP_LLM_CHECK_WITH_INFO(role_type_ == RoleType::PDFUSION || role_type_ == RoleType::PREFILL
                                || role_type_ == RoleType::DECODE,
                            "pipeline parallelism requires the PDFUSION, PREFILL, or DECODE role");
    // Active CP is a P-side execution mode. D-side PREFILL_CP only describes
    // imported KV; PDFUSION would also send verify/decode through this model.
    RTP_LLM_CHECK_WITH_INFO(!parallelism_config_.prefill_cp_config.is_enabled() || role_type_ == RoleType::PREFILL,
                            "PP context parallel execution requires the PREFILL role; "
                            "DECODE imports CP KV with cp_rotate_method=PREFILL_CP");
    RTP_LLM_CHECK_WITH_INFO(!params.runtime_config.use_batch_decode_scheduler,
                            "pipeline parallelism does not support BatchDecodeScheduler");
    const bool has_multi_task_prompt = !params.kv_cache_config.multi_task_prompt.empty()
                                       || !params.kv_cache_config.multi_task_prompt_tokens.empty()
                                       || !params.kv_cache_config.multi_task_prompt_str.empty();
    RTP_LLM_CHECK_WITH_INFO(!has_multi_task_prompt
                                || (params.sp_config.type == SP_TYPE_NONE
                                    && !parallelism_config_.prefill_cp_config.is_enabled()
                                    && parallelism_config_.dp_size <= 1),
                            "pipeline parallelism multi-task system prompts currently require "
                            "SP_NONE speculative decoding, no prefill context parallelism, and dp_size==1");
    const char* device_input = std::getenv("RTP_LLM_DEVICE_INPUT");
    RTP_LLM_CHECK_WITH_INFO(device_input == nullptr || std::strcmp(device_input, "1") != 0,
                            "pipeline parallelism does not support device-input mode (RTP_LLM_DEVICE_INPUT)");

    if (const char* watchdog_timeout_env = std::getenv("RTP_LLM_PP_COMM_WATCHDOG_TIMEOUT_MS")) {
        comm_watchdog_timeout_ms_ = std::max<int64_t>(1, std::strtoll(watchdog_timeout_env, nullptr, 10));
    }

    RTP_LLM_CHECK_WITH_INFO(!sp_enabled_ || params.sp_config.gen_num_per_cycle > 0,
                            "PP speculative decoding requires a positive gen_num_per_cycle, got %ld",
                            params.sp_config.gen_num_per_cycle);
    RTP_LLM_CHECK_WITH_INFO(!is_dspark_ || dspark_mask_token_id_ >= 0,
                            "PP DSpARK requires sp_dspark_mask_token_id, got %d",
                            dspark_mask_token_id_);
    // forwardMicroBatched bypasses the CP input/output processing in forward().
    RTP_LLM_CHECK_WITH_INFO((!is_dspark_ && !parallelism_config_.prefill_cp_config.is_enabled())
                                || params.device_resource_config.enable_layer_micro_batch == 0,
                            "PP CP and DSpARK do not support layer micro-batching");

    if (!warm_up_) {
        transport_ = std::make_unique<TorchDistributedPPTransport>(pp_layout_.prevRank(), pp_layout_.nextRank());
    }

    enable_detail_log_ = params.profiling_debug_logging_config.enable_detail_log;
    RTP_LLM_LOG_INFO("enable_detail_log_ = %d, tp_rank_ = %d", enable_detail_log_, parallelism_config_.tp_rank);
    if (params.profiling_debug_logging_config.enable_model_inputs_log) {
        model_inputs_logger_ =
            std::make_shared<ModelInputsLogger>(params.parallelism_config.world_rank,
                                                params.profiling_debug_logging_config.log_file_backup_count,
                                                metrics_reporter_);
    }

    if (params.eplb_config.enable_eplb() && params.model_config_.moe_style != 0) {
        const auto first_moe_layer =
            pp_layout_.firstMoeLayer(params.model_config_.num_layers, params.model_config_.moe_layer_index);
        if (first_moe_layer >= 0) {
            const auto& moe_kernel = params.gpt_weights.layers[first_moe_layer].ffn_weights.moe_gate_weight->kernel;
            auto        moe_weight_type     = torchDTypeToDataType(moe_kernel.dtype());
            bool        is_gated_activation = params.model_config_.isGatedActivation();
            auto        moe_inter_size      = is_gated_activation ? moe_kernel.size(1) / 2 : moe_kernel.size(1);

            expert_balancer_ =
                std::make_shared<ExpertBalancer>(params.model_config_.expert_num,
                                                 params.eplb_config.phy_exp_num(params.model_config_.expert_num),
                                                 params.model_config_.num_layers,
                                                 moe_inter_size,
                                                 params.model_config_.hidden_size,
                                                 params.parallelism_config,
                                                 pp_layout_.myLayerRange(params.model_config_.num_layers),
                                                 params.py_eplb,
                                                 moe_weight_type,
                                                 params.model_config_.quant_algo,
                                                 metrics_reporter_,
                                                 params.eplb_config);
        }
    }

    if (!warm_up_ && isLastStage() && isStageRoot()) {
        const auto initial_sampler_batch_size =
            static_cast<size_t>(std::max<int64_t>(1, params.runtime_config.max_generate_batch_size));
        sampler_ = std::make_unique<Sampler>(SamplerInitParams{initial_sampler_batch_size, false});
    }

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
        RTP_LLM_LOG_INFO("init executor with python model");
        model_ = std::make_unique<PyWrappedModel>(
            model_init_params, params.py_model, false, sp_enabled_ && isLastStage() && !warm_up_);
    } else if (test_model_factory) {
        RTP_LLM_LOG_INFO("init executor with test model factory");
        model_ = test_model_factory(model_init_params);
    } else {
        RTP_LLM_LOG_WARNING("py_model is None — model will not be initialized (test mode)");
    }

    if (propose_params && propose_params->draftModel() && !warm_up_) {
        for (auto& draft_params : *propose_params->mtp_model_params_) {
            if (is_dspark_) {
                RTP_LLM_CHECK_WITH_INFO(draft_params->model_config_.vocab_size == params.model_config_.vocab_size,
                                        "PP DSpARK requires identical draft/target vocabularies, got %ld and %ld",
                                        draft_params->model_config_.vocab_size,
                                        params.model_config_.vocab_size);
            }
            std::optional<GroupedCacheLayerLayout> draft_cache_layer_layout;
            size_t draft_tokens_per_block        = draft_params->model_config_.attn_config.tokens_per_block;
            size_t draft_kernel_tokens_per_block = draft_params->model_config_.attn_config.kernel_tokens_per_block;
            if (cache_manager_) {
                draft_cache_layer_layout       = cache_manager_->getMTPModuleGroupedCacheLayerLayout(0);
                const auto& draft_cache_config = *cache_manager_->cacheConfig().mtp_sub_configs[0];
                draft_tokens_per_block         = draft_cache_config.seq_size_per_block;
                draft_kernel_tokens_per_block  = draft_cache_config.kernel_seq_size_per_block;
            }

            GptModelInitParams draft_init_params({draft_params->gpt_weights,
                                                  genModelDescription(draft_params->model_config_,
                                                                      draft_params->parallelism_config,
                                                                      draft_params->eplb_config,
                                                                      draft_params->moe_config),
                                                  draft_cache_layer_layout,
                                                  draft_params->model_id,
                                                  draft_params->parallelism_config,
                                                  params.hw_kernel_config,
                                                  params.profiling_debug_logging_config,
                                                  params.runtime_config,
                                                  params.concurrency_config,
                                                  params.sp_config,
                                                  params.device_resource_config,
                                                  mla_ops_type,
                                                  draft_params->model_config_.max_seq_len,
                                                  draft_params->model_config_.hidden_size,
                                                  draft_tokens_per_block,
                                                  draft_kernel_tokens_per_block,
                                                  cache_manager_,
                                                  std::make_optional(0),
                                                  draft_params->model_config_.hc_mult});

            if (!params.py_sp_model.is_none()) {
                RTP_LLM_LOG_INFO("init PP executor with python draft model");
                draft_model_ =
                    std::make_unique<PyWrappedModel>(draft_init_params,
                                                     params.py_sp_model,
                                                     false,
                                                     false,
                                                     is_dspark_,
                                                     is_dspark_ ? DSparkCallPhase::COMMIT : DSparkCallPhase::NONE);
            } else if (test_model_factory) {
                draft_model_ = test_model_factory(draft_init_params);
            } else {
                RTP_LLM_LOG_WARNING("py_sp_model is None — draft model will not be initialized (test mode)");
            }
            /** Runtime uses active module 0. */
            break;
        }
        if (isStageRoot()) {
            const auto& draft_weights = propose_params->getEngineInitParams().gpt_weights;
            const auto& d2t_map       = draft_model_ ? draft_model_->weights_.d2t_map : draft_weights.d2t_map;
            if (!is_dspark_) {
                fast_topk_sampler_ = std::make_unique<speculative::FastTopKSampler>(d2t_map);
            }
            speculative_sampler_       = std::make_unique<speculative::SpeculativeSampler>(d2t_map, propose_step_);
            spec_logits_verify_runner_ = std::make_unique<SpecLogitsVerifyRunner>();
        }
    }

    const auto& cache_config = cache_manager_ ? cache_manager_->cacheConfig() : CacheConfig();
    batch_stream_processor_  = std::make_unique<PPBatchStreamProcessor>(params.model_config_,
                                                                       params.pd_sep_config,
                                                                       params.profiling_debug_logging_config,
                                                                       cache_config,
                                                                       warm_up_,
                                                                       params.sp_config.type);
    LogitsProcessorFactory::init(params.model_config_, params.grammar_config, params.sp_config.tree_decode_config);
    cudaProfilerBegin();
}

PPExecutor::~PPExecutor() {
    for (auto& slot : slots_) {
        waitAll(slot.plan_sends, "plan send completion", false);
        waitAll(slot.activation_sends, "activation send completion", false);
        waitAll(slot.execution_result_sends, "execution result send completion", false);
    }
    cudaProfilerEnd();
}

void PPExecutor::releaseAllModelBuffers() {
    buffer_holder_.release();
    // model_ is unset in test mode (py_model is None); nothing to release then.
    if (model_) {
        model_->releaseBuffers();
    }
    if (draft_model_) {
        draft_model_->releaseBuffers();
    }
}

absl::Status PPExecutor::warmUp(const ScheduleOutput& schedule_output) {
    RTP_LLM_CHECK_WITH_INFO(model_ != nullptr, "model is not initialized for PP warmup");

    StreamGroups stream_groups(schedule_output.streams);
    auto         model_input_status = batch_stream_processor_->gatherModelInput(stream_groups, buffer_holder_);
    RETURN_IF_STATUS_OR_ERROR(model_input_status);
    auto model_input = std::move(model_input_status.value());

    /* Each stage warms up the same fake request locally and only requires TP synchronization. */
    tpSyncModelInputs(model_input, parallelism_config_);

    releaseAllModelBuffers();
    if (cache_manager_) {
        cache_manager_->zeroBlocks(model_input.kv_cache_blocks_to_zero);
        model_input.kv_cache_blocks_to_zero = torch::Tensor();
        if (model_input.kv_cache_update_mapping.defined()) {
            cache_manager_->blockBatchCopy(model_input.kv_cache_update_mapping);
        }
    }

    PPIntermediateTensors input_tensors;
    PPIntermediateTensors output_tensors;
    if (!isFirstStage()) {
        input_tensors =
            model_->makePPWarmUpInputTensors(model_input, parallelism_config_.prefill_cp_config.is_enabled());
    }

    (void)model_->forwardPP(
        model_input, isFirstStage() ? nullptr : &input_tensors, isLastStage() ? nullptr : &output_tensors);
    if (expert_balancer_) {
        RtpLLMExecutorMetricsCollector collector;
        expert_balancer_->stepForward(*model_, collector);
    }

    /* Keep model tensors alive until lazy initialization kernels finish. */
    cudaSyncAndCheck();
    releaseAllModelBuffers();
    return absl::OkStatus();
}

void PPExecutor::prepareStreams(std::list<GenerateStreamPtr>& streams) {
    if (!sp_enabled_) {
        return;
    }

    const auto token_count = static_cast<int64_t>(propose_step_ + 1);
    for (auto it = streams.begin(); it != streams.end();) {
        const auto& stream = *it;
        auto        error  = LogitsProcessorFactory::validateMtpCompatibility(stream->getAllLogitsProcessorPtr());
        if (error.has_value()) {
            stream->reportError(error->code(), error->ToString());
            // The scheduler marked this request in flight, but no plan will carry it.
            stream->clearPPInflight();
            it = streams.erase(it);
            continue;
        }
        auto       sp_output_buffer = stream->getSPOutputBuffer();
        const bool is_fake_stream   = stream->isFakeStream() || stream->isPerfTest();
        if (!sp_output_buffer) {
            RTP_LLM_CHECK_WITH_INFO(stream->isContextStream() || is_fake_stream,
                                    "PP speculative decode requires proposal tokens, request_id=%ld",
                                    stream->streamId());
            sp_output_buffer         = std::make_shared<SpeculativeExecutorStreamOutput>();
            sp_output_buffer->tokens = torch::zeros({1, token_count}, torch::kInt32);
            stream->setSPOutputBuffer(sp_output_buffer);
        } else if (is_fake_stream
                   && (!sp_output_buffer->tokens.defined() || sp_output_buffer->tokens.numel() != token_count)) {
            sp_output_buffer->tokens = torch::zeros({1, token_count}, torch::kInt32);
        }
        RTP_LLM_CHECK_WITH_INFO(
            sp_output_buffer->tokens.defined() && sp_output_buffer->tokens.device().is_cpu()
                && sp_output_buffer->tokens.scalar_type() == torch::kInt32 && sp_output_buffer->tokens.is_contiguous()
                && sp_output_buffer->tokens.numel() == token_count,
            "PP speculative token buffer must contain one target token and %zu draft tokens, request_id=%ld",
            propose_step_,
            stream->streamId());
        sp_output_buffer->propose_step = propose_step_;
        ++it;
    }
}

absl::StatusOr<PPExecutionPlan> PPExecutor::buildPlan(const StreamGroups&         stream_groups,
                                                      const std::vector<int64_t>& finished_request_ids) {
    RTP_LLM_CHECK_WITH_INFO(isFirstStage(), "only the first PP stage can build an execution plan from streams");

    PPExecutionPlan plan;
    plan.finished_request_ids = finished_request_ids;

    const auto streams     = stream_groups.allStreams();
    plan.is_decode         = !streams.empty() && !streams.front()->isContextStream();
    const bool need_verify = sp_enabled_ && plan.is_decode;

    auto model_input_status =
        need_verify ?
            batch_stream_processor_->gatherTargetVerifyModelInput(stream_groups, propose_step_, buffer_holder_) :
            batch_stream_processor_->gatherModelInput(stream_groups, buffer_holder_);
    RETURN_IF_STATUS_OR_ERROR(model_input_status);
    plan.model_input          = std::move(model_input_status.value());
    plan.model_input.skip_run = stream_groups.empty();

    if (!plan.model_input.skip_run) {
        plan.sampling_plan = batch_stream_processor_->gatherSamplingPlan(stream_groups);
        plan.output_config = batch_stream_processor_->gatherOutputConfig(stream_groups);
        plan.draft_next_position_ids =
            batch_stream_processor_->gatherDraftNextPositionIds(stream_groups, plan.model_input);
    }

    return plan;
}

void PPExecutor::advanceSamplingStates(const PPSamplingPlan& sampling_plan, PPExecutionResult& result) {
    const auto stream_count = sampling_plan.request_ids.size(0);

    const auto* request_ids      = sampling_plan.request_ids.data_ptr<int64_t>();
    const auto* input_lengths    = sampling_plan.input_lengths.data_ptr<int32_t>();
    const auto* sequence_lengths = sampling_plan.sequence_lengths.data_ptr<int32_t>();
    const auto& cum_log_probs    = result.cum_log_probs;
    int64_t     batch_idx        = 0;
    for (int64_t stream_idx = 0; stream_idx < stream_count; ++stream_idx) {
        const int64_t stream_batch_size = std::max<int32_t>(sampling_plan.num_return_sequences[stream_idx], 1);

        if (result.request_errors[stream_idx].hasError()) {
            batch_idx += stream_batch_size;
            continue;
        }
        auto& state = sampling_states_.at(request_ids[stream_idx]);

        std::optional<ErrorInfo> error;
        const auto num_new_tokens = result.accept_len.defined() ? result.accept_len.data_ptr<int32_t>()[batch_idx] : 1;
        RTP_LLM_CHECK(num_new_tokens > 0 && num_new_tokens <= result.new_token_ids.size(1));
        const auto new_tokens =
            result.new_token_ids.narrow(0, batch_idx, stream_batch_size).narrow(1, 0, num_new_tokens);
        for (const auto& processor : state.logits_processors) {
            error = processor->updateStatus(new_tokens, num_new_tokens);
            if (error.has_value()) {
                break;
            }
        }
        const int64_t expected_output_len = sequence_lengths[batch_idx] - input_lengths[batch_idx] + num_new_tokens;
        if (!error.has_value()) {
            for (size_t processor_index = 0; processor_index < state.logits_processors.size(); ++processor_index) {
                const auto processor_output_len = state.logits_processors[processor_index]->committedOutputLen();
                if (processor_output_len.has_value() && processor_output_len.value() != expected_output_len) {
                    error = ErrorInfo(ErrorCode::UNKNOWN_ERROR,
                                      "logits processor committed output length mismatch: processor_index="
                                          + std::to_string(processor_index)
                                          + ", processor=" + std::to_string(processor_output_len.value())
                                          + ", expected=" + std::to_string(expected_output_len));
                    break;
                }
            }
        }
        if (error.has_value()) {
            result.request_errors[stream_idx] = std::move(error.value());
        } else if (cum_log_probs.defined()) {
            state.cum_log_probs.copy_(cum_log_probs.narrow(0, batch_idx, stream_batch_size));
        }
        batch_idx += stream_batch_size;
    }
}

void PPExecutor::clipMtpAcceptedLengths(const PPSamplingPlan& sampling_plan, PPExecutionResult& result) const {
    const auto  count            = sampling_plan.request_ids.numel();
    auto*       lengths          = result.accept_len.data_ptr<int32_t>();
    const auto* max_tokens       = sampling_plan.max_tokens.data_ptr<int32_t>();
    const auto* sequence_lengths = sampling_plan.sequence_lengths.data_ptr<int32_t>();
    for (int64_t row = 0; row < count; ++row) {
        lengths[row] = std::min(lengths[row], max_tokens[row] - sequence_lengths[row]);
    }
}

void PPExecutor::verifyDraftTokens(const PPExecutionPlan& plan,
                                   const torch::Tensor&   target_logits,
                                   PPExecutionResult&     result) {
    const auto     batch_size         = plan.sampling_plan.request_ids.size(0);
    const auto     verify_token_count = static_cast<int64_t>(propose_step_ + 1);
    const auto     vocab_size         = target_logits.size(1);
    PPOutputConfig output_config;
    output_config.return_all_probs = ReturnAllProbsMode::DEFAULT;
    auto inputs                    = batch_stream_processor_->gatherSamplerInputs(plan.sampling_plan,
                                                               output_config,
                                                               target_logits,
                                                               sampling_states_,
                                                               true,
                                                               propose_step_,
                                                               plan.model_input.combo_tokens);
    inputs.logits_processor_states_ptr.reset();

    SamplerOutput draft_sampler_output;
    draft_sampler_output.token_ids = plan.model_input.combo_tokens.reshape({batch_size, verify_token_count})
                                         .narrow(1, 1, propose_step_)
                                         .contiguous();
    /** Draft argmax and fixed PD padding both define point-mass proposals. */
    draft_sampler_output.token_ids_are_point_mass = true;
    SpecLogitsVerifyRunner::LaunchTask task;
    task.total_streams = batch_size;
    task.propose_step  = propose_step_;
    task.vocab_size    = vocab_size;
    task.draft_tokens  = draft_sampler_output.token_ids;
    speculative::SpeculativeSamplingParams params;
    params.do_sample    = plan.sampling_plan.spec_do_sample;
    params.force_accept = plan.sampling_plan.force_sp_accept;
    params.generators.reserve(batch_size);
    const auto* request_ids = plan.sampling_plan.request_ids.data_ptr<int64_t>();
    for (int64_t row = 0; row < batch_size; ++row) {
        const auto& state = sampling_states_.at(request_ids[row]);
        params.generators.push_back(state.generator);
        for (const auto& processor : state.logits_processors) {
            const auto capability = processor->mtpCapability();
            if (capability.mode == MtpProcessorMode::SPEC_VERIFY) {
                task.active.push_back({processor, static_cast<size_t>(row)});
            }
        }
    }
    SpecLogitsVerifyRunner::LaunchResult verify_result;
    if (!task.active.empty()) {
        verify_result = spec_logits_verify_runner_->run(task);
        if (verify_result.ready_event) {
            verify_result.ready_event->block(cuda_graph::graphGetCurrentStream());
        }
        SpecLogitsVerifyRunner::applyMaskToLogits(inputs.logits, verify_result, vocab_size);
    }

    auto target_sampler_output = sampler_->forward(inputs);
    target_sampler_output.all_probs =
        target_sampler_output.all_probs.reshape({batch_size, verify_token_count, vocab_size});
    speculative::SpeculativeSamplerOutput accepted;
    mtp::runRejectionSampling(
        *speculative_sampler_, params, draft_sampler_output, target_sampler_output, verify_result, accepted);
    accepted.transfer_done_event->synchronize();

    result.new_token_ids         = std::move(accepted.accept_tokens_cpu);
    result.accept_len            = std::move(accepted.accept_len_cpu);
    const auto  verify_success   = target_sampler_output.success.to(torch::kCPU).contiguous();
    const auto* max_tokens       = plan.sampling_plan.max_tokens.data_ptr<int32_t>();
    const auto* sequence_lengths = plan.sampling_plan.sequence_lengths.data_ptr<int32_t>();
    for (int64_t row = 0; row < batch_size; ++row) {
        auto& error = result.request_errors[row];
        if (error.hasError()) {
            continue;
        }
        if (static_cast<size_t>(row) < accepted.processor_errors.size() && accepted.processor_errors[row].has_value()) {
            error = std::move(accepted.processor_errors[row].value());
            continue;
        }
        // Suffix rows beyond the accepted prefix (or length cap) are not consumed.
        // A sampler failure there must not turn an earlier valid rejection into a request error.
        const auto used_rows =
            std::min(result.accept_len.data_ptr<int32_t>()[row], max_tokens[row] - sequence_lengths[row]);
        for (int64_t step = 0; step < used_rows; ++step) {
            const auto index = row * verify_token_count + step;
            if (!verify_success.data_ptr<bool>()[index]) {
                error = ErrorInfo(ErrorCode::UNKNOWN_ERROR, "sampler generate token id failed");
                break;
            }
        }
    }
}

GptModelInputs PPExecutor::prepareDraftInputForPrefill(const GptModelInputs&  target_input,
                                                       const GptModelOutputs& target_output,
                                                       const torch::Tensor&   sampled_token_ids,
                                                       const torch::Tensor&   next_position_ids) {
    auto          draft_input = target_input;
    torch::Tensor target_hidden_states;
    // Under CP, each rank binds its own target hidden after the draft input
    // broadcast. The root only constructs the full shifted tokens here.
    if (!parallelism_config_.prefill_cp_config.is_enabled()) {
        target_hidden_states = model_->getMtpTargetHiddenStates(target_input.combo_tokens.numel());
        if (!target_hidden_states.defined() || target_hidden_states.numel() == 0) {
            target_hidden_states = target_output.all_hidden_states;
        }
    }
    mtp::prepareDraftInputForPrefill(draft_input,
                                     target_hidden_states,
                                     sampled_token_ids,
                                     next_position_ids,
                                     position_id_len_factor_,
                                     buffer_holder_);
    return draft_input;
}

GptModelInputs PPExecutor::prepareDraftInputForDecode(const GptModelInputs&  target_input,
                                                      const GptModelOutputs& target_output,
                                                      const torch::Tensor&   accepted_token_ids,
                                                      const torch::Tensor&   accepted_lengths) {
    auto draft_input          = target_input;
    auto target_hidden_states = model_->getMtpTargetHiddenStates(target_input.combo_tokens.numel());
    if (!target_hidden_states.defined() || target_hidden_states.numel() == 0) {
        target_hidden_states = target_output.all_hidden_states;
    }

    mtp::prepareDraftInputForDecode(draft_input,
                                    target_hidden_states,
                                    accepted_token_ids,
                                    accepted_lengths,
                                    mtp::DraftInputLayout::COMPACT,
                                    position_id_len_factor_,
                                    buffer_holder_);
    return draft_input;
}

void PPExecutor::runDSparkCommit(const GptModelInputs& target_input, const GptModelOutputs& target_output) {
    RTP_LLM_PROFILE_SCOPE("executor.pp.dspark_commit");
    RTP_LLM_CHECK_WITH_INFO(draft_model_ != nullptr, "PP DSpARK draft model is not initialized");

    const bool cp_enabled      = parallelism_config_.prefill_cp_config.is_enabled();
    auto       target_features = model_->getMtpTargetHiddenStates(cp_enabled ? -1 : target_input.combo_tokens.numel());
    if (!target_features.defined() || target_features.numel() == 0) {
        target_features = target_output.all_hidden_states;
    }
    RTP_LLM_CHECK_WITH_INFO(target_features.defined() && target_features.dim() == 2,
                            "PP DSpARK commit requires 2-D target features");
    // CP validates feature rows against the padded local token count in handleInputs.
    RTP_LLM_CHECK_WITH_INFO(cp_enabled || target_features.size(0) == target_input.combo_tokens.numel(),
                            "PP DSpARK commit feature rows %ld do not match input rows %ld",
                            target_features.size(0),
                            target_input.combo_tokens.numel());

    auto commit_input               = target_input;
    commit_input.last_hidden_states = torch::Tensor();
    commit_input.is_target_verify   = false;
    commit_input.dspark_call_phase  = DSparkCallPhase::COMMIT;
    tpSyncModelInputs(commit_input, parallelism_config_);
    /** Sync shared geometry before binding rank-local target features. */
    mtp::prepareDSparkCommitInput(commit_input, target_features);
    if (cache_manager_) {
        const auto& draft_cache_config     = cache_manager_->getMTPModuleCacheConfig(0);
        commit_input.kv_block_stride_bytes = draft_cache_config.kv_block_stride_bytes;
        commit_input.kv_scale_stride_bytes = draft_cache_config.kv_scale_stride_bytes;
    }
    if (model_inputs_logger_) {
        model_inputs_logger_->log(commit_input, ModelInputsModelRole::DRAFT, draft_model_->model_id_);
    }
    (void)draft_model_->forward(commit_input);
}

torch::Tensor PPExecutor::proposeDraftTokens(GptModelInputs draft_input, size_t num_draft_tokens) {
    RTP_LLM_PROFILE_SCOPE("executor.pp.propose_draft_tokens");
    RTP_LLM_CHECK_WITH_INFO(draft_model_ != nullptr, "PP draft model is not initialized");

    if (is_dspark_) {
        tpSyncModelInputs(draft_input, parallelism_config_);
        draft_input.dspark_call_phase = DSparkCallPhase::PROPOSE;
        if (cache_manager_) {
            const auto& draft_cache_config    = cache_manager_->getMTPModuleCacheConfig(0);
            draft_input.kv_block_stride_bytes = draft_cache_config.kv_block_stride_bytes;
            draft_input.kv_scale_stride_bytes = draft_cache_config.kv_scale_stride_bytes;
        }
        if (model_inputs_logger_) {
            model_inputs_logger_->log(draft_input, ModelInputsModelRole::DRAFT, draft_model_->model_id_);
        }
        auto          draft_output = draft_model_->forward(draft_input);
        torch::Tensor proposed_tokens;
        if (isStageRoot()) {
            const auto batch_size = draft_input.input_lengths.numel();
            RTP_LLM_CHECK_WITH_INFO(
                draft_output.draft_tokens.defined() && draft_output.draft_tokens.scalar_type() == torch::kInt32
                    && draft_output.draft_tokens.dim() == 2 && draft_output.draft_tokens.size(0) == batch_size
                    && draft_output.draft_tokens.size(1) == static_cast<int64_t>(num_draft_tokens),
                "PP DSpARK proposal must be int32 [%ld, %zu]",
                batch_size,
                num_draft_tokens);
            proposed_tokens = draft_output.draft_tokens.to(torch::kCPU).contiguous();
        }
        cudaSyncAndCheck();
        return proposed_tokens;
    }

    torch::Tensor proposed_tokens;
    for (size_t step = 0; step < num_draft_tokens; ++step) {
        torch::Tensor cp_target_hidden;
        if (step == 0 && parallelism_config_.prefill_cp_config.is_enabled()) {
            cp_target_hidden = model_->getMtpTargetHiddenStates(-1);
            RTP_LLM_CHECK_WITH_INFO(cp_target_hidden.defined() && cp_target_hidden.numel() > 0,
                                    "PP CP draft prefill requires this rank's target hidden states");
            draft_input.last_hidden_states = torch::Tensor();
        }
        tpSyncModelInputs(draft_input, parallelism_config_);
        if (cp_target_hidden.defined()) {
            draft_input.last_hidden_states = cp_target_hidden;
        }
        if (cache_manager_) {
            const auto& draft_cache_config    = cache_manager_->getMTPModuleCacheConfig(0);
            draft_input.kv_block_stride_bytes = draft_cache_config.kv_block_stride_bytes;
            draft_input.kv_scale_stride_bytes = draft_cache_config.kv_scale_stride_bytes;
        }
        if (model_inputs_logger_) {
            model_inputs_logger_->log(draft_input, ModelInputsModelRole::DRAFT, draft_model_->model_id_);
        }
        auto draft_output = draft_model_->forward(draft_input);

        if (isStageRoot()) {
            /** The PP verifier consumes deterministic proposals without a probability tensor. */
            auto draft_tokens = fast_topk_sampler_->forward(draft_output.logits, 1).token_ids.to(torch::kInt32);
            if (step == 0) {
                proposed_tokens = torch::empty({draft_tokens.size(0), static_cast<int64_t>(num_draft_tokens)},
                                               draft_tokens.options());
            }
            proposed_tokens.select(1, step).copy_(draft_tokens.flatten());
            if (step + 1 < num_draft_tokens) {
                auto draft_hidden_states = draft_model_->getMtpTargetHiddenStates(draft_input.combo_tokens.numel());
                if (draft_hidden_states.defined() && draft_hidden_states.numel() > 0) {
                    draft_output.all_hidden_states = draft_hidden_states;
                }
                if (step == 0) {
                    const auto batch_size     = draft_input.input_lengths.numel();
                    const auto output_indexes = draft_input.lm_output_indexes.to(torch::kLong);
                    // Recover each request's valid prefix length from its selected output row.
                    const auto physical_lengths = draft_input.input_lengths.to(torch::kLong);
                    const auto starts           = physical_lengths.cumsum(0) - physical_lengths;
                    const auto valid_lengths    = output_indexes.to(starts.device()) - starts + 1;

                    draft_input.combo_tokens       = draft_tokens.reshape({batch_size});
                    draft_input.last_hidden_states = draft_output.all_hidden_states.index_select(
                        0, output_indexes.to(draft_output.all_hidden_states.device()));
                    draft_input.sequence_lengths =
                        (draft_input.prefix_lengths.to(valid_lengths.device()) + valid_lengths).to(torch::kInt32);
                    draft_input.input_lengths           = torch::ones_like(draft_input.input_lengths);
                    draft_input.prefix_lengths          = torch::empty({0}, draft_input.prefix_lengths.options());
                    draft_input.sequence_lengths_plus_1 = torch::Tensor();
                    draft_input.lm_output_indexes = torch::arange(batch_size, draft_input.lm_output_indexes.options());
                    draft_input.request_id        = torch::Tensor();
                    draft_input.request_pd_separation = torch::Tensor();
                    draft_input.cache_keys            = torch::Tensor();
                    if (draft_input.combo_position_ids.defined()) {
                        const auto positions =
                            draft_input.combo_position_ids.reshape({-1, static_cast<int64_t>(position_id_len_factor_)});
                        draft_input.combo_position_ids =
                            (positions.index_select(0, output_indexes.to(positions.device())) + 1)
                                .flatten()
                                .pin_memory();
                    }
                } else {
                    mtp::advanceDraftInput(draft_input,
                                           draft_output.all_hidden_states,
                                           draft_tokens,
                                           position_id_len_factor_,
                                           buffer_holder_);
                }
            }
        }
    }
    if (isStageRoot()) {
        proposed_tokens = proposed_tokens.to(torch::kCPU);
    }
    cudaSyncAndCheck();
    return proposed_tokens;
}

void PPExecutor::sampleTokens(const PPExecutionPlan& plan,
                              const GptModelOutputs& model_output,
                              PPExecutionResult&     result) {
    const auto stream_count = plan.sampling_plan.request_ids.size(0);
    result.request_ids      = plan.sampling_plan.request_ids.to(torch::kCPU).contiguous();
    result.request_errors.assign(stream_count, ErrorInfo::OkStatus());
    result.prompt_logits.resize(stream_count);
    if (plan.model_input.is_fake_stream) {
        /** Supply target-result shapes for the draft chain without creating request sampling state. */
        if (sp_enabled_) {
            const auto token_count = static_cast<int64_t>(plan.is_decode ? propose_step_ + 1 : 1);
            result.new_token_ids   = torch::zeros({stream_count, token_count}, torch::kInt32);
            result.accept_len      = torch::full({stream_count}, token_count, torch::kInt32);
        }
        return;
    }
    batch_stream_processor_->initSamplingStates(plan.sampling_plan, sampling_states_, result);
    if (sp_enabled_ && plan.is_decode) {
        verifyDraftTokens(plan, model_output.logits, result);
    } else {
        auto inputs = batch_stream_processor_->gatherSamplerInputs(
            plan.sampling_plan, plan.output_config, model_output.logits, sampling_states_);
        auto sampler_output = sampler_->forward(inputs);
        batch_stream_processor_->fillExecutionResult(plan, model_output, sampler_output, result);
    }
    if (sp_enabled_ && !plan.is_decode) {
        result.accept_len = torch::ones({result.new_token_ids.size(0)}, torch::kInt32);
    }
}

void PPExecutor::runDraftStep(const PPExecutionPlan& plan,
                              const GptModelOutputs& model_output,
                              PPExecutionResult&     execution_result) {
    if (is_dspark_) {
        runDSparkCommit(plan.model_input, model_output);
        if (role_type_ == RoleType::PREFILL) {
            /** P-side prefill commits draft KV without generating proposals. */
            execution_result.propose_token_ids = torch::Tensor();
            return;
        }
    }
    auto draft_input = plan.model_input;
    if (isStageRoot()) {
        /** Replace failed rows with draft placeholders; request_errors prevents their commit. */
        auto& accepted_tokens  = execution_result.new_token_ids;
        auto& accepted_lengths = execution_result.accept_len;
        for (int64_t row = 0; row < accepted_lengths.numel(); ++row) {
            if (execution_result.request_errors[row].hasError()) {
                accepted_tokens[row].zero_();
                accepted_lengths[row] = 1;
            }
        }
        if (is_dspark_) {
            RTP_LLM_CHECK_WITH_INFO(execution_result.new_token_ids.defined()
                                        && execution_result.new_token_ids.scalar_type() == torch::kInt32
                                        && execution_result.new_token_ids.dim() == 2,
                                    "PP DSpARK requires 2-D int32 target token output");
            const auto batch_size     = execution_result.new_token_ids.size(0);
            auto       prefix_lengths = plan.model_input.prefix_lengths.to(torch::kCPU).to(torch::kInt32).contiguous();
            RTP_LLM_CHECK_WITH_INFO(prefix_lengths.numel() == batch_size,
                                    "PP DSpARK prefix length count %ld does not match batch size %ld",
                                    prefix_lengths.numel(),
                                    batch_size);

            torch::Tensor anchors;
            torch::Tensor committed_ends;
            if (plan.is_decode) {
                RTP_LLM_CHECK_WITH_INFO(execution_result.accept_len.defined()
                                            && execution_result.accept_len.scalar_type() == torch::kInt32
                                            && execution_result.accept_len.numel() == batch_size,
                                        "PP DSpARK decode requires one int32 accept length per batch row");
                const auto min_accept = accepted_lengths.min().item<int32_t>();
                const auto max_accept = accepted_lengths.max().item<int32_t>();
                RTP_LLM_CHECK_WITH_INFO(min_accept > 0 && max_accept <= execution_result.new_token_ids.size(1),
                                        "PP DSpARK accept lengths must be in [1, %ld], got min=%d max=%d",
                                        execution_result.new_token_ids.size(1),
                                        min_accept,
                                        max_accept);
                auto last_indexes = (accepted_lengths.to(torch::kLong) - 1).unsqueeze(1);
                anchors           = accepted_tokens.gather(1, last_indexes).reshape({batch_size});
                committed_ends    = prefix_lengths + accepted_lengths;
            } else {
                RTP_LLM_CHECK_WITH_INFO(execution_result.new_token_ids.size(1) == 1,
                                        "PP DSpARK prefill expects one sampled token per batch row");
                auto input_lengths = plan.model_input.input_lengths.to(torch::kCPU).to(torch::kInt32).contiguous();
                RTP_LLM_CHECK_WITH_INFO(input_lengths.numel() == batch_size,
                                        "PP DSpARK input length count %ld does not match batch size %ld",
                                        input_lengths.numel(),
                                        batch_size);
                anchors        = accepted_tokens.reshape({batch_size}).contiguous();
                committed_ends = prefix_lengths + input_lengths;
            }
            mtp::prepareDSparkProposeInput(draft_input,
                                           anchors,
                                           committed_ends,
                                           propose_step_,
                                           dspark_mask_token_id_,
                                           dspark_propose_input_buffers_,
                                           buffer_holder_);
            /** Only COMMIT publishes persistent draft KV state. */
            draft_input.request_id            = torch::Tensor();
            draft_input.request_pd_separation = torch::Tensor();
            draft_input.cache_keys            = torch::Tensor();
        } else if (plan.is_decode) {
            draft_input = prepareDraftInputForDecode(plan.model_input, model_output, accepted_tokens, accepted_lengths);
        } else {
            draft_input = prepareDraftInputForPrefill(
                plan.model_input, model_output, accepted_tokens, plan.draft_next_position_ids);
        }
    }
    /** MTP/EAGLE PD prefill hands off d1 after one draft forward; D pads the remaining candidate slots. */
    const size_t draft_count           = !is_dspark_ && role_type_ == RoleType::PREFILL ? 1 : propose_step_;
    execution_result.propose_token_ids = proposeDraftTokens(std::move(draft_input), draft_count);
}

absl::Status PPExecutor::process(const ScheduleOutput& schedule_output, int64_t schedule_time_us) {
    if (warm_up_) {
        return warmUp(schedule_output);
    }

    schedule_time_us = (schedule_time_us <= 0) ? autil::TimeUtility::currentTimeInMicroSeconds() : schedule_time_us;

    const bool report_active = metrics_reporter_ && isFirstStage() && isStageRoot()
                               && std::any_of(schedule_output.streams.begin(),
                                              schedule_output.streams.end(),
                                              [](const auto& stream) { return !stream->isFakeStream(); });
    auto tps_active_guard      = tps_reporter_.makeActiveGuard(report_active);
    auto wall_tps_active_guard = wall_tps_reporter_.makeActiveGuard(report_active);
    RTP_LLM_PROFILE_FUNCTION();

    /** 0. Admit compatible streams and prepare SP buffers. */
    auto streams = schedule_output.streams;
    if (isFirstStage() && isStageRoot()) {
        prepareStreams(streams);
    }

    /** 1. recv the plan from the previous stage */
    PPExecutionPlan plan;
    StreamGroups    scheduled_stream_groups;
    if (isFirstStage()) {
        scheduled_stream_groups = StreamGroups(streams);
        auto plan_status        = buildPlan(scheduled_stream_groups, schedule_output.finished_request_ids);
        RETURN_IF_STATUS_OR_ERROR(plan_status);
        plan = std::move(plan_status.value());
        if (isStageRoot() && stopping_ && idle_streak_ >= parallelism_config_.pp_size + 1) {
            plan.model_input.shutdown = true;
            RTP_LLM_LOG_INFO("pipeline drained, emitting shutdown sentinel to next stage");
        }
    } else {
        plan = receivePlan();
    }

    if (isLastStage() && isStageRoot()) {
        for (const auto request_id : plan.finished_request_ids) {
            sampling_states_.erase(request_id);
        }
    }

    /** 2. do the sync across the all ranks in the same stage. */
    tpSyncModelInputs(plan.model_input, parallelism_config_);

    releaseAllModelBuffers();

    /** 3. Wait on this slot's previous sends before resetting it. CUDA waits order subsequent operations on the current
     * stream after communication. */
    auto& inflight = slots_[current_slot_];
    waitAll(inflight.plan_sends, "plan send completion");
    waitAll(inflight.activation_sends, "activation send completion");
    waitAll(inflight.execution_result_sends, "execution result send completion");
    inflight.reset();
    inflight.skip_run = plan.model_input.skip_run;
    if (isFirstStage() && isStageRoot()) {
        inflight.stream_groups    = std::move(scheduled_stream_groups);
        inflight.schedule_time_us = schedule_time_us;
    }

    /** 4. send the plan to next stage  */
    if (!isLastStage()) {
        asyncSendPlan(plan, !isStageRoot(), inflight.plan_sends);
    }

    /** 5. run the batch. */
    if (!plan.model_input.skip_run) {
        PPTickets             tensor_receives;
        PPIntermediateTensors input_tensors;
        PPIntermediateTensors output_tensors;

        if (!isFirstStage()) {
            input_tensors = receiveTensors(tensor_receives);
            waitAll(tensor_receives, "intermediate tensors from previous stage");
        }

        if (profile_step_start_) {
            profile_step_start_();
        }

        GptModelInputs& local_model_input = plan.model_input;
        if (cache_manager_) {
            cache_manager_->zeroBlocks(local_model_input.kv_cache_blocks_to_zero);
            local_model_input.kv_cache_blocks_to_zero = torch::Tensor();
            if (local_model_input.kv_cache_update_mapping.defined()) {
                cache_manager_->blockBatchCopy(local_model_input.kv_cache_update_mapping);
            }
        }

        const bool force = isStageRoot() && enable_detail_log_;
        if (force) {
            RTP_LLM_LOG_INFO("model_input: %s", local_model_input.debugString(force).c_str());
        } else {
            RTP_LLM_LOG_TRACE("model_input: %s", local_model_input.debugString(force).c_str());
        }
        if (model_inputs_logger_) {
            model_inputs_logger_->log(local_model_input, ModelInputsModelRole::NORMAL, model_->model_id_);
        }
        auto model_output = model_->forwardPP(
            local_model_input, isFirstStage() ? nullptr : &input_tensors, isLastStage() ? nullptr : &output_tensors);

        if (expert_balancer_) {
            RtpLLMExecutorMetricsCollector collector;
            expert_balancer_->stepForward(*model_, collector, !local_model_input.is_fake_stream);
        }

        auto forward_done = cuda_graph::makeGraphEvent();
        forward_done.record(cuda_graph::graphGetCurrentStream());
        forward_done.synchronize();

        if (!isLastStage()) {
            asyncSendTensors(output_tensors, inflight.activation_sends);
        } else {
            PPExecutionResult execution_result;
            if (isStageRoot()) {
                sampleTokens(plan, model_output, execution_result);
                if (sp_enabled_ && !plan.model_input.is_fake_stream) {
                    clipMtpAcceptedLengths(plan.sampling_plan, execution_result);
                }
            }

            if (sp_enabled_) {
                runDraftStep(plan, model_output, execution_result);
            }

            if (isStageRoot()) {
                if (!plan.model_input.is_fake_stream) {
                    advanceSamplingStates(plan.sampling_plan, execution_result);
                }
                asyncSendExecutionResult(execution_result, inflight.execution_result_sends);
            }
        }

        if (profile_step_finish_) {
            profile_step_finish_();
        }
    }

    current_slot_ = (current_slot_ + 1) % slots_.size();

    /** 6. recv the execution result of next batch and process it. */
    auto& next_batch                = slots_[current_slot_];
    bool  received_result_this_step = false;
    if (isFirstStage() && isStageRoot() && !next_batch.skip_run) {
        // A fake batch's result is not real progress; it must not reset the drain counter.
        received_result_this_step = !next_batch.stream_groups.isFakeStream();

        const auto& stream_groups            = next_batch.stream_groups;
        auto        token_counts_by_priority = stream_groups.tokenCountsByPriority();
        RETURN_IF_STATUS_ERROR(processExecutionResult(next_batch));

        const int64_t tps_execute_time_us =
            autil::TimeUtility::currentTimeInMicroSeconds() - next_batch.schedule_time_us;
        if (metrics_reporter_ && !stream_groups.isFakeStream() && tps_execute_time_us > 0) {
            RtpLLMTokenPSMetricsCollector tps_collector;
            tps_collector.addTokenSize(stream_groups.contextExecuteTokenSize(),
                                       stream_groups.contextExecuteTokenSizeWithCache(),
                                       stream_groups.totalDecodeBatchSize(),
                                       stream_groups.modelExecuteTokenSize(),
                                       tps_execute_time_us);
            tps_collector.addTokenSizeByPriority(token_counts_by_priority, tps_execute_time_us);
            tps_reporter_.report(&tps_collector);
            wall_tps_reporter_.report(&tps_collector);
        }
    }

    if (isFirstStage() && isStageRoot() && !plan.model_input.shutdown) {
        // Idle = no real request (skip_run or fake-stream placeholder) and no real result;
        // fake batches must not reset the counter, or DP+PP never emits the sentinel.
        const bool no_work   = plan.model_input.skip_run || plan.model_input.is_fake_stream;
        const bool no_result = !received_result_this_step;
        idle_streak_         = (no_work && no_result) ? idle_streak_ + 1 : 0;
    }
    if (plan.model_input.shutdown) {
        shutdown_completed_ = true;
    }
    return absl::OkStatus();
}

bool PPExecutor::updateEplbConfig(const EPLBConfig& config) {
    if (expert_balancer_) {
        return expert_balancer_->updateEplbConfig(config);
    }
    return true;
}

}  // namespace rtp_llm
