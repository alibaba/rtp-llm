#include "rtp_llm/cpp/normal_engine/pipeline/PPExecutor.h"

#include "rtp_llm/cpp/normal_engine/pipeline/PPSerialization.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <unordered_set>
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

namespace {

std::vector<TaggedBlockIdPair> decodeCacheUpdateMapping(const torch::Tensor&            copy_mapping,
                                                        const std::vector<std::string>& group_tags,
                                                        const CacheTopology&            local_topology) {
    RTP_LLM_CHECK_WITH_INFO(copy_mapping.defined() && copy_mapping.device().is_cpu()
                                && copy_mapping.scalar_type() == torch::kInt32 && copy_mapping.is_contiguous()
                                && copy_mapping.dim() == 2 && copy_mapping.size(1) == 3,
                            "cache update mapping must be a contiguous CPU int32 [N,3] tensor");
    std::unordered_set<std::string> seen;
    for (const auto& tag : group_tags) {
        RTP_LLM_CHECK_WITH_INFO(!tag.empty() && seen.insert(tag).second,
                                "cache update mapping tags must be non-empty and unique: tag=%s",
                                tag.c_str());
    }
    const std::unordered_set<std::string> local_tags(local_topology.groupTags().begin(),
                                                     local_topology.groupTags().end());
    std::vector<TaggedBlockIdPair>        mappings;
    mappings.reserve(static_cast<size_t>(copy_mapping.size(0)));
    const auto* rows = copy_mapping.data_ptr<int32_t>();
    for (int64_t i = 0; i < copy_mapping.size(0); ++i) {
        const auto row = rows[3 * i];
        RTP_LLM_CHECK_WITH_INFO(row >= 0 && static_cast<size_t>(row) < group_tags.size(),
                                "cache update mapping payload row is out of range: row=%d",
                                row);
        const auto& tag = group_tags[row];
        if (local_tags.find(tag) == local_tags.end()) {
            continue;
        }
        mappings.push_back({tag, rows[3 * i + 1], rows[3 * i + 2]});
    }
    return mappings;
}

}  // namespace

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
    result_pending     = false;
    stream_groups      = StreamGroups();
    schedule_time_us   = 0;
    executor_collector = RtpLLMExecutorMetricsCollector();
}

void PPExecutor::sendObject(const torch::Tensor& object, PPTickets& tickets) {
    auto object_size = torch::tensor({object.numel()}, torch::kInt64);
    tickets.push_back(transport_->asyncSend(object_size));
    tickets.push_back(transport_->asyncSend(object));
}

torch::Tensor PPExecutor::receiveObject() {
    auto object_size  = torch::empty({1}, torch::TensorOptions().dtype(torch::kInt64));
    auto size_receive = transport_->asyncReceive(object_size);
    size_receive->wait();

    auto object         = torch::empty({object_size.item<int64_t>()}, torch::TensorOptions().dtype(torch::kUInt8));
    auto object_receive = transport_->asyncReceive(object);
    object_receive->wait();
    return object;
}

void PPExecutor::asyncSendPlan(const PPExecutionPlan& plan, bool metadata_only, PPTickets& tickets) {
    RTP_LLM_PROFILE_SCOPE("executor.pp.send_plan");
    sendObject(pp_serialization::serializePlan(plan, metadata_only), tickets);
}

PPExecutionPlan PPExecutor::receivePlan() {
    RTP_LLM_PROFILE_SCOPE("executor.pp.recv_plan");
    auto plan = pp_serialization::deserializePlan(receiveObject());
    if (plan.shutdown) {
        RTP_LLM_LOG_INFO("received pipeline shutdown sentinel from previous stage");
    }
    return plan;
}

void PPExecutor::asyncSendExecutionResult(const PPExecutionResult& result, PPTickets& tickets) {
    RTP_LLM_PROFILE_SCOPE("executor.pp.send_result");
    sendObject(pp_serialization::serializeExecutionResult(result), tickets);
}

void PPExecutor::asyncSendTensors(const PPIntermediateTensors& tensors, PPTickets& tickets) {
    RTP_LLM_PROFILE_SCOPE("executor.pp.send_activations");
    sendObject(pp_serialization::serializeTensorsMetadata(tensors), tickets);
    for (const auto& tensor_entry : tensors.tensors) {
        if (tensor_entry.second.numel() != 0) {
            tickets.push_back(transport_->asyncSend(tensor_entry.second));
        }
    }
}

PPIntermediateTensors PPExecutor::receiveTensors(PPTickets& tickets) {
    RTP_LLM_PROFILE_SCOPE("executor.pp.recv_activations");
    auto tensors = pp_serialization::deserializeTensorsMetadata(receiveObject());
    for (auto& tensor_entry : tensors.tensors) {
        if (tensor_entry.second.numel() != 0) {
            tickets.push_back(transport_->asyncReceive(tensor_entry.second));
        }
    }
    return tensors;
}

void PPExecutor::waitAll(PPTickets& tickets) {
    for (auto& ticket : tickets) {
        ticket->wait();
    }
    tickets.clear();
}

absl::Status PPExecutor::processExecutionResult(InflightBatch& batch) {
    /** Fake batches still complete the PP round trip before their result is discarded. */
    PPExecutionResult result;
    {
        RTP_LLM_PROFILE_SCOPE("executor.pp.recv_result");
        result = pp_serialization::deserializeExecutionResult(receiveObject());
    }
    batch.result_pending = false;
    if (batch.stream_groups.isFakeStream()) {
        RTP_LLM_LOG_DEBUG("PP fake batch completed: dp_rank=%ld", parallelism_config_.dp_rank);
        return absl::OkStatus();
    }

    /** Count input work and returned tokens before dispatch changes stream state. */
    StreamGroups::TokenCountsByPriority     token_counts_by_priority;
    RtpLLMSpeculativeEngineMetricsCollector sp_collector;
    collectTokenCounts(batch.stream_groups, result, token_counts_by_priority, sp_collector);

    {
        RTP_LLM_PROFILE_SCOPE("executor.pp.dispatch_result");
        const auto start_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
        RETURN_IF_STATUS_ERROR(batch_stream_processor_->dispatchExecutionResult(batch.stream_groups, result));
        batch.executor_collector.dispatch_output_us = autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
    }

    reportResultMetrics(batch, token_counts_by_priority, sp_collector);
    return absl::OkStatus();
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
    dspark_sample_from_anchor_(params.sp_config.sp_dspark_sample_from_anchor),
    propose_step_(params.sp_config.gen_num_per_cycle),
    position_id_len_factor_(params.model_config_.attn_config.rope_config.index_factor),
    parallelism_config_(params.parallelism_config),
    pp_layout_(RankLayout::fromParallelismConfig(parallelism_config_)),
    slots_(parallelism_config_.pp_size + 1),
    track_dspark_cache_store_(is_dspark_ && role_type_ == RoleType::PREFILL && isLastStage() && !warm_up_),
    dspark_cache_store_sync_stream_(cuda_graph::graphGetStreamFromPool(true)),
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
    /**
     * Active CP is a P-side execution mode. D-side PREFILL_CP only describes
     * imported KV; PDFUSION would also send verify/decode through this model.
     */
    RTP_LLM_CHECK_WITH_INFO(!parallelism_config_.prefill_cp_config.is_enabled() || role_type_ == RoleType::PREFILL,
                            "PP context parallel execution requires the PREFILL role; "
                            "DECODE imports CP KV with cp_rotate_method=PREFILL_CP");
    RTP_LLM_CHECK_WITH_INFO(!params.runtime_config.use_batch_decode_scheduler,
                            "pipeline parallelism does not support BatchDecodeScheduler");
    const char* device_input = std::getenv("RTP_LLM_DEVICE_INPUT");
    RTP_LLM_CHECK_WITH_INFO(device_input == nullptr || std::strcmp(device_input, "1") != 0,
                            "pipeline parallelism does not support device-input mode (RTP_LLM_DEVICE_INPUT)");

    RTP_LLM_CHECK_WITH_INFO(!sp_enabled_ || params.sp_config.gen_num_per_cycle > 0,
                            "PP speculative decoding requires a positive gen_num_per_cycle, got %ld",
                            params.sp_config.gen_num_per_cycle);
    RTP_LLM_CHECK_WITH_INFO(!is_dspark_ || dspark_mask_token_id_ >= 0,
                            "PP DSpARK requires sp_dspark_mask_token_id, got %d",
                            dspark_mask_token_id_);
    /** forwardMicroBatched bypasses the CP input/output processing in forward(). */
    RTP_LLM_CHECK_WITH_INFO((!is_dspark_ && !parallelism_config_.prefill_cp_config.is_enabled())
                                || params.device_resource_config.enable_layer_micro_batch == 0,
                            "PP CP and DSpARK do not support layer micro-batching");

    if (track_dspark_cache_store_ && parallelism_config_.tp_size > 1) {
        dspark_cache_store_status_ =
            torch::empty({1}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    }
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
         cache_manager_,
         std::nullopt,
         params.model_config_.hc_mult});
    model_init_params.metrics_reporter = metrics_reporter_;

    if (!params.py_model.is_none()) {
        RTP_LLM_LOG_INFO("init executor with python model");
        /** PP target stages stay eager for both primary and generation-prefill graph runners. */
        model_ = std::make_unique<PyWrappedModel>(model_init_params,
                                                 params.py_model,
                                                 false,
                                                 sp_enabled_ && isLastStage() && !warm_up_,
                                                 DSparkModelRole::NONE,
                                                 false,
                                                 track_dspark_cache_store_);
    } else if (test_model_factory) {
        RTP_LLM_LOG_INFO("init executor with test model factory");
        model_ = test_model_factory(model_init_params);
    } else {
        RTP_LLM_LOG_WARNING("py_model is None — model will not be initialized (test mode)");
    }

    if (propose_params && propose_params->draftModel() && !warm_up_ && isLastStage()) {
        for (auto& draft_params : *propose_params->mtp_model_params_) {
            draft_vocab_size_ = draft_params->model_config_.vocab_size;
            std::optional<GroupedCacheLayerLayout> draft_cache_layer_layout;
            if (cache_manager_) {
                draft_cache_layer_layout = cache_manager_->getMTPModuleGroupedCacheLayerLayout(0);
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
                                                  cache_manager_,
                                                  std::make_optional(0),
                                                  draft_params->model_config_.hc_mult});
            draft_init_params.metrics_reporter = metrics_reporter_;

            if (!params.py_sp_model.is_none()) {
                RTP_LLM_LOG_INFO("init PP executor with python draft model");
                const bool draft_graph_allowed = !(is_dspark_ && role_type_ == RoleType::PREFILL);
                if (!is_dspark_ || role_type_ != RoleType::PREFILL) {
                    draft_model_ = std::make_unique<PyWrappedModel>(draft_init_params,
                                                                    params.py_sp_model,
                                                                    false,
                                                                    false,
                                                                    is_dspark_ ? DSparkModelRole::PROPOSE :
                                                                                 DSparkModelRole::NONE,
                                                                    draft_graph_allowed);
                }
                if (is_dspark_) {
                    draft_commit_model_ = std::make_unique<PyWrappedModel>(draft_init_params,
                                                                           params.py_sp_model,
                                                                           false,
                                                                           false,
                                                                           DSparkModelRole::COMMIT,
                                                                           draft_graph_allowed,
                                                                           track_dspark_cache_store_);
                }
            } else if (test_model_factory) {
                if (!is_dspark_ || role_type_ != RoleType::PREFILL) {
                    draft_model_ = test_model_factory(draft_init_params);
                }
                if (is_dspark_) {
                    draft_commit_model_ = test_model_factory(draft_init_params);
                }
            } else {
                RTP_LLM_LOG_WARNING("py_sp_model is None — draft model will not be initialized (test mode)");
            }
            /** Runtime uses active module 0. */
            break;
        }
        if (isStageRoot()) {
            const auto& draft_weights = propose_params->getEngineInitParams().gpt_weights;
            const auto& d2t_map       = draft_model_ ? draft_model_->weights_.d2t_map : draft_weights.d2t_map;
            if (is_dspark_ && role_type_ != RoleType::PREFILL) {
                dspark_markov_w1_ = draft_weights.dspark_markov_w1;
                dspark_markov_w2_ = draft_weights.dspark_markov_w2;
                RTP_LLM_CHECK_WITH_INFO(dspark_markov_w1_.defined() && dspark_markov_w2_.defined(),
                                        "PP DSpARK requires markov_w1 and markov_w2 weights");
                const int64_t padded_draft_vocab_size =
                    (static_cast<int64_t>(draft_vocab_size_) + 127) / 128 * 128;
                RTP_LLM_CHECK_WITH_INFO(dspark_markov_w1_.is_cuda() && dspark_markov_w2_.is_cuda()
                                            && dspark_markov_w1_.dim() == 2 && dspark_markov_w2_.dim() == 2
                                            && dspark_markov_w1_.size(1) == dspark_markov_w2_.size(1)
                                            && dspark_markov_w1_.size(0) >= params.model_config_.vocab_size
                                            && dspark_markov_w2_.size(0) >= static_cast<int64_t>(draft_vocab_size_)
                                            && dspark_markov_w2_.size(0) <= padded_draft_vocab_size
                                            && dspark_markov_w1_.scalar_type() == dspark_markov_w2_.scalar_type(),
                                        "PP DSpARK Markov weights must be CUDA [target_vocab,rank] and "
                                        "[draft_vocab,rank] tensors with matching rank and dtype");
                dspark_markov_w2_ =
                    dspark_markov_w2_.narrow(0, 0, static_cast<int64_t>(draft_vocab_size_));
                if (draft_vocab_size_ != static_cast<size_t>(params.model_config_.vocab_size)) {
                    RTP_LLM_CHECK_WITH_INFO(d2t_map.defined() && d2t_map.is_cuda() && d2t_map.dim() == 1
                                                && d2t_map.scalar_type() == torch::kInt64
                                                && d2t_map.numel() == static_cast<int64_t>(draft_vocab_size_),
                                            "reduced-vocabulary PP DSpARK requires a CUDA int64 d2t map");
                    const auto d2t_min = d2t_map.min().item<int64_t>();
                    const auto d2t_max = d2t_map.max().item<int64_t>();
                    RTP_LLM_CHECK_WITH_INFO(d2t_min >= 0 && d2t_max < params.model_config_.vocab_size,
                                            "PP DSpARK d2t target ids must be in [0,%ld), got range [%ld,%ld]",
                                            params.model_config_.vocab_size,
                                            d2t_min,
                                            d2t_max);
                }
            } else if (!is_dspark_) {
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
    cudaProfilerEnd();
}

void PPExecutor::releaseAllModelBuffers() {
    buffer_holder_.release();
    /** model_ is unset in test mode (py_model is None); nothing to release then. */
    if (model_) {
        model_->releaseBuffers();
    }
    if (draft_model_) {
        draft_model_->releaseBuffers();
    }
    if (draft_commit_model_) {
        draft_commit_model_->releaseBuffers();
    }
}

absl::Status PPExecutor::warmUp(const ScheduleOutput& schedule_output) {
    RTP_LLM_CHECK_WITH_INFO(model_ != nullptr, "model is not initialized for PP warmup");

    StreamGroups stream_groups(schedule_output.streams);
    auto         model_input_status = batch_stream_processor_->gatherModelInput(stream_groups, buffer_holder_);
    RETURN_IF_STATUS_OR_ERROR(model_input_status);
    auto model_input = std::move(model_input_status.value());

    /** Each stage warms up the same fake request locally and only requires TP synchronization. */
    tpSyncModelInputs(model_input, parallelism_config_);

    releaseAllModelBuffers();
    if (cache_manager_) {
        if (model_input.kv_cache_update_mapping.defined()) {
            cache_manager_->blockBatchCopyByGroup(
                decodeCacheUpdateMapping(model_input.kv_cache_update_mapping,
                                         model_input.kv_cache_group_tags,
                                         cache_manager_->cacheConfig().topology()));
        }
    }

    auto target_input = model_input;
    if (!isFirstStage()) {
        target_input.pp_intermediates =
            model_->makePPWarmUpInputTensors(model_input, parallelism_config_.prefill_cp_config.is_enabled()).tensors;
    }

    auto model_output = model_->forward(target_input);
    if (expert_balancer_) {
        RtpLLMExecutorMetricsCollector collector;
        expert_balancer_->stepForward(*model_, collector);
    }

    /** Keep model tensors alive until lazy initialization kernels finish. */
    cudaSyncAndCheck();
    releaseAllModelBuffers();
    return absl::OkStatus();
}

void PPExecutor::prepareStreams(std::list<GenerateStreamPtr>& streams) {
    RTP_LLM_PROFILE_SCOPE("executor.pp.prepare_streams");
    if (!sp_enabled_) {
        return;
    }

    const auto token_count = static_cast<int64_t>(propose_step_ + 1);
    for (auto it = streams.begin(); it != streams.end();) {
        const auto& stream = *it;
        if (is_dspark_) {
            RTP_LLM_CHECK_WITH_INFO(stream->maxBatchSize() == 1,
                                    "PP DSpARK does not support tiled sampling, request_id=%ld",
                                    stream->streamId());
        }
        auto error = LogitsProcessorFactory::validateMtpCompatibility(stream->getAllLogitsProcessorPtr());
        if (error.has_value()) {
            stream->reportError(error->code(), error->ToString());
            /** The scheduler marked this request in flight, but no plan will carry it. */
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
    RTP_LLM_PROFILE_SCOPE("executor.pp.build_plan");
    RTP_LLM_CHECK_WITH_INFO(isFirstStage(), "only the first PP stage can build an execution plan from streams");

    PPExecutionPlan plan;
    plan.finished_request_ids = finished_request_ids;

    const auto streams = stream_groups.allStreams();
    plan.is_decode     = !streams.empty() && !streams.front()->isContextStream();
    if (!stream_groups.empty()) {
        plan.sampling_plan = batch_stream_processor_->gatherSamplingPlan(stream_groups);
        plan.output_config = batch_stream_processor_->gatherOutputConfig(stream_groups);
    }

    const bool need_verify = sp_enabled_ && plan.is_decode;
    auto model_input_status =
        need_verify ?
            batch_stream_processor_->gatherTargetVerifyModelInput(stream_groups, propose_step_, buffer_holder_) :
            batch_stream_processor_->gatherModelInput(stream_groups, buffer_holder_);
    RETURN_IF_STATUS_OR_ERROR(model_input_status);
    plan.model_input          = std::move(model_input_status.value());
    plan.model_input.skip_run = stream_groups.empty();

    if (!plan.model_input.skip_run) {
        plan.draft_next_position_ids =
            batch_stream_processor_->gatherDraftNextPositionIds(stream_groups, plan.model_input);
    }

    return plan;
}

void PPExecutor::advanceSamplingStates(const PPSamplingPlan& sampling_plan, PPExecutionResult& result) {
    RTP_LLM_PROFILE_SCOPE("executor.pp.update_sampling_states");
    const auto stream_count = sampling_plan.request_ids.size(0);

    const auto* request_ids      = sampling_plan.request_ids.data_ptr<int64_t>();
    const auto* input_lengths    = sampling_plan.input_lengths.data_ptr<int32_t>();
    const auto* sequence_lengths = sampling_plan.sequence_lengths.data_ptr<int32_t>();
    const auto* max_tokens       = sampling_plan.max_tokens.data_ptr<int32_t>();
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
        const auto num_new_tokens = result.new_token_lengths.data_ptr<int32_t>()[batch_idx];
        /** Match the first stage's eventual write length without changing the returned acceptance count. */
        const auto num_committed_tokens =
            sp_enabled_ ? std::min(num_new_tokens, max_tokens[batch_idx] - sequence_lengths[batch_idx]) : num_new_tokens;
        const auto new_tokens =
            result.new_token_ids.narrow(0, batch_idx, stream_batch_size).narrow(1, 0, num_committed_tokens);
        for (const auto& processor : state.logits_processors) {
            error = processor->updateStatus(new_tokens, num_committed_tokens);
            if (error.has_value()) {
                break;
            }
        }
        const int64_t expected_output_len = sequence_lengths[batch_idx] - input_lengths[batch_idx] + num_committed_tokens;
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

void PPExecutor::verifyDraftTokens(const PPExecutionPlan&                 plan,
                                   const torch::Tensor&                   target_logits,
                                   SamplerOutput&                         target_sampler_output,
                                   speculative::SpeculativeSamplerOutput& accepted) {
    const auto     batch_size         = plan.sampling_plan.request_ids.size(0);
    const auto     verify_token_count = static_cast<int64_t>(propose_step_ + 1);
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
    const auto vocab_size = static_cast<int64_t>(inputs.vocab_size);

    SamplerOutput                         draft_sampler_output;
    speculative::SpeculativeSamplingParams params;
    draft_sampler_output.token_ids = plan.model_input.combo_tokens.reshape({batch_size, verify_token_count})
                                         .narrow(1, 1, propose_step_)
                                         .contiguous();
    if (is_dspark_ && !plan.model_input.is_fake_stream && plan.sampling_plan.draft_all_probs.defined()) {
        draft_sampler_output.all_probs =
            plan.sampling_plan.draft_all_probs.to(target_logits.device(), /*non_blocking=*/true);
        draft_sampler_output.token_ids_are_point_mass = false;
        params.draft_point_mass_rows                  = plan.sampling_plan.draft_point_mass_rows;
    } else {
        /** MTP argmax, DSpARK handoff padding and fake streams define point-mass proposals. */
        draft_sampler_output.token_ids_are_point_mass = true;
    }
    SpecLogitsVerifyRunner::LaunchTask task;
    task.total_streams = batch_size;
    task.propose_step  = propose_step_;
    task.vocab_size    = vocab_size;
    task.draft_tokens  = draft_sampler_output.token_ids;
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

    target_sampler_output = sampler_->forward(inputs);
    target_sampler_output.all_probs =
        target_sampler_output.all_probs.reshape({batch_size, verify_token_count, vocab_size});
    mtp::runRejectionSampling(
        *speculative_sampler_, params, draft_sampler_output, target_sampler_output, verify_result, accepted);
}

void PPExecutor::prepareDraftPrefillAfterTargetPrefill(GptModelInputs&      draft_prefill_input,
                                                       const torch::Tensor& target_hidden_states,
                                                       const torch::Tensor& sampled_token_ids,
                                                       const torch::Tensor& next_position_ids) {
    mtp::prepareDraftPrefillAfterTargetPrefill(draft_prefill_input,
                                               target_hidden_states,
                                               sampled_token_ids,
                                               next_position_ids,
                                               position_id_len_factor_,
                                               buffer_holder_);
}

void PPExecutor::prepareDraftPrefillAfterVerify(GptModelInputs&      draft_prefill_input,
                                                const torch::Tensor& target_hidden_states,
                                                const torch::Tensor& accepted_token_ids,
                                                const torch::Tensor& accepted_lengths) {
    mtp::prepareDraftPrefillAfterVerify(draft_prefill_input,
                                        target_hidden_states,
                                        accepted_token_ids,
                                        accepted_lengths,
                                        draft_input_layout_,
                                        position_id_len_factor_,
                                        buffer_holder_);
}

void PPExecutor::broadcastPostRejectionInputs(GptModelInputs& draft_input) {
    RTP_LLM_PROFILE_SCOPE("executor.pp.tp_sync_post_rejection");

    if (parallelism_config_.tp_size > 1) {
        mtp::syncDraftPrefillAfterVerify(draft_input, draft_input_layout_, parallelism_config_);
    }
}

GptModelOutputs PPExecutor::forwardDraftModel(ModelBase& draft_model, GptModelInputs& draft_input) {
    if (cache_manager_) {
        const auto& draft_cache_config   = cache_manager_->getMTPModuleCacheConfig(0);
        draft_input.kv_block_stride_bytes = 0;
        draft_input.kv_scale_stride_bytes = 0;
        if (draft_cache_config.groupNums() == 1) {
            const auto& group                = draft_cache_config.topology().groups().front();
            draft_input.kv_block_stride_bytes = group.kvBlockStrideBytes();
            draft_input.kv_scale_stride_bytes = group.kvScaleStrideBytes();
        }
    }
    if (model_inputs_logger_) {
        model_inputs_logger_->log(draft_input, ModelInputsModelRole::DRAFT, draft_model.model_id_);
    }
    return draft_model.forward(draft_input);
}

void PPExecutor::prepareDSparkCommitInput(GptModelInputs& commit_input, const torch::Tensor& target_features) {
    RTP_LLM_PROFILE_SCOPE("executor.pp.dspark_commit_prepare");

    const bool cp_enabled = parallelism_config_.prefill_cp_config.is_enabled();
    RTP_LLM_CHECK_WITH_INFO(target_features.defined() && target_features.dim() == 2,
                            "PP DSpARK commit requires 2-D target features");
    /** CP checks feature rows against the padded local token count in handleInputs. */
    RTP_LLM_CHECK_WITH_INFO(cp_enabled || target_features.size(0) == commit_input.combo_tokens.numel(),
                            "PP DSpARK commit feature rows %ld do not match input rows %ld",
                            target_features.size(0),
                            commit_input.combo_tokens.numel());

    /** COMMIT keeps the target geometry and phase, binding only this rank's auxiliary features. */
    commit_input.last_hidden_states = target_features;
}

void PPExecutor::sampleTokens(const PPExecutionPlan& plan,
                              const GptModelOutputs& model_output,
                              PPExecutionResult&     result) {
    RTP_LLM_PROFILE_SCOPE("executor.pp.target_sample");
    const auto stream_count = plan.sampling_plan.request_ids.size(0);
    result.request_ids      = plan.sampling_plan.request_ids.to(torch::kCPU).contiguous();
    result.request_errors.assign(stream_count, ErrorInfo::OkStatus());
    result.prompt_logits.resize(stream_count);
    if (plan.model_input.is_fake_stream) {
        /** Supply target-result shapes for the draft chain without creating request sampling state. */
        const auto token_count   = static_cast<int64_t>(sp_enabled_ && plan.is_decode ? propose_step_ + 1 : 1);
        result.new_token_ids     = torch::zeros({stream_count, token_count}, torch::kInt32);
        result.new_token_lengths = torch::full({stream_count}, token_count, torch::kInt32);
        return;
    }
    batch_stream_processor_->initSamplingStates(plan.sampling_plan, sampling_states_, result);

    if (sp_enabled_ && plan.model_input.is_target_verify) {
        SamplerOutput                         target_sampler_output;
        speculative::SpeculativeSamplerOutput sp_output;
        verifyDraftTokens(plan, model_output.logits, target_sampler_output, sp_output);
        batch_stream_processor_->fillExecutionResult(plan, model_output, target_sampler_output, sp_output, result);
    } else {
        auto inputs = batch_stream_processor_->gatherSamplerInputs(
            plan.sampling_plan, plan.output_config, model_output.logits, sampling_states_);
        const auto sampler_output = sampler_->forward(inputs);
        batch_stream_processor_->fillExecutionResult(plan, model_output, sampler_output, result);
    }
}

GptModelInputs PPExecutor::prepareDSparkProposeInput(const GptModelInputs& target_input,
                                                     bool                  is_decode,
                                                     const torch::Tensor&  new_token_ids,
                                                     const torch::Tensor&  new_token_lengths) {
    auto       draft_input    = target_input;
    const auto batch_size     = new_token_ids.size(0);
    auto       prefix_lengths = target_input.prefix_lengths.to(torch::kCPU).to(torch::kInt32).contiguous();

    torch::Tensor anchors;
    torch::Tensor committed_ends;
    if (is_decode) {
        auto last_indexes = (new_token_lengths.to(torch::kLong) - 1).unsqueeze(1);
        anchors           = new_token_ids.gather(1, last_indexes).reshape({batch_size});
        committed_ends    = prefix_lengths + new_token_lengths;
    } else {
        auto input_lengths = target_input.input_lengths.to(torch::kCPU).to(torch::kInt32).contiguous();
        anchors        = new_token_ids.reshape({batch_size}).contiguous();
        committed_ends = prefix_lengths + input_lengths;
    }
    mtp::prepareDSparkProposeInput(draft_input,
                                   anchors,
                                   committed_ends,
                                   propose_step_,
                                   dspark_mask_token_id_,
                                   dspark_sample_from_anchor_,
                                   dspark_propose_input_buffers_,
                                   buffer_holder_);
    /** Only COMMIT publishes persistent draft KV state. */
    draft_input.request_id            = torch::Tensor();
    draft_input.request_pd_separation = torch::Tensor();
    draft_input.cache_keys            = torch::Tensor();
    return draft_input;
}

void PPExecutor::fillFailedDraftRows(PPExecutionResult& result) {
    /** Replace failed rows with draft placeholders; request_errors prevents their commit. */
    auto& new_token_ids     = result.new_token_ids;
    auto& new_token_lengths = result.new_token_lengths;
    for (int64_t row = 0; row < new_token_lengths.numel(); ++row) {
        if (result.request_errors[row].hasError()) {
            new_token_ids[row].zero_();
            new_token_lengths[row] = 1;
        }
    }
}

void PPExecutor::runDraftStep(const GptModelInputs& target_input,
                              const torch::Tensor&  draft_next_position_ids,
                              const torch::Tensor&  target_hidden_states,
                              bool                  is_decode,
                              const PPSamplingPlan& sampling_plan,
                              PPExecutionResult&    execution_result) {
    const bool cp_enabled = parallelism_config_.prefill_cp_config.is_enabled();

    const auto& new_token_ids      = execution_result.new_token_ids;
    const auto& new_token_lengths  = execution_result.new_token_lengths;
    const auto& temperature        = sampling_plan.temperature;
    const auto& spec_do_sample     = sampling_plan.spec_do_sample;
    auto&       proposed_token_ids = execution_result.propose_token_ids;
    auto&       proposed_all_probs = execution_result.propose_all_probs;
    GptModelInputs draft_input     = target_input;

    {
        RTP_LLM_PROFILE_SCOPE("executor.pp.draft_prepare");
        /** 1. Prefill draft KV and prepare the first decode input when needed. */
        if (is_dspark_) {
            prepareDSparkCommitInput(draft_input, target_hidden_states);
        } else if (isStageRoot()) {
            if (is_decode) {
                prepareDraftPrefillAfterVerify(draft_input, target_hidden_states, new_token_ids, new_token_lengths);
            } else {
                prepareDraftPrefillAfterTargetPrefill(
                    draft_input, target_hidden_states, new_token_ids, draft_next_position_ids);
            }
        }

        if (!is_decode) {
            /** CP and DSpARK hidden is rank-local; rebind it after syncing the global inputs. */
            if (cp_enabled || is_dspark_) {
                draft_input.last_hidden_states = torch::Tensor();
            }
            tpSyncModelInputs(draft_input, parallelism_config_);
            if (cp_enabled || is_dspark_) {
                draft_input.last_hidden_states = target_hidden_states;
            }
        } else if (!is_dspark_) {
            broadcastPostRejectionInputs(draft_input);
        }
    }

    {
        RTP_LLM_PROFILE_SCOPE(is_dspark_ ? "executor.pp.dspark_commit" : "executor.pp.draft_prefill");
        /** 2. Draft Prefill. */
        auto* draft_prefill_model = is_dspark_ ? draft_commit_model_.get() : draft_model_.get();
        RTP_LLM_CHECK_WITH_INFO(draft_prefill_model != nullptr, "PP draft prefill model is not initialized");
        auto draft_prefill_output = forwardDraftModel(*draft_prefill_model, draft_input);
        if (!is_dspark_) {
            torch::Tensor draft_last_hidden_states;
            if (cp_enabled) {
                /** CP exposes request-ordered final rows; rank-local token offsets cannot
                 * recover the draft hidden required by a non-PP decode peer. */
                draft_last_hidden_states = draft_model_->getMtpLastHiddenStates(target_input.input_lengths.numel());
            } else {
                mtp::maybeOverrideLastHiddenWithMtpBuffer(
                    draft_prefill_output, *draft_model_, draft_input.combo_tokens.numel());
            }

            if (isStageRoot()) {
                auto draft_sample = fast_topk_sampler_->forward(draft_prefill_output.logits, 1);
                auto draft_tokens = draft_sample.token_ids.to(torch::kInt32);

                if (role_type_ == RoleType::PREFILL) {
                    /** P and D choose PP independently. Keep main's one-proposal handoff:
                     * q stays in draft vocabulary even after token IDs have been mapped.
                     * Snapshot the final draft hidden before reusable model buffers change;
                     * a one-step D or a PP D can ignore it without P knowing that topology. */
                    proposed_token_ids = draft_tokens;
                    proposed_all_probs = draft_sample.all_probs.unsqueeze(1).to(torch::kCPU).contiguous();
                    if (cp_enabled) {
                        RTP_LLM_CHECK_WITH_INFO(draft_last_hidden_states.defined()
                                                    && draft_last_hidden_states.dim() == 2
                                                    && draft_last_hidden_states.size(0) == draft_tokens.size(0),
                                                "CP MTP handoff requires one final draft hidden row per request");
                    } else {
                        auto last_indexes = draft_input.input_lengths.to(torch::kLong).cumsum(0) - 1;
                        draft_last_hidden_states = draft_prefill_output.all_hidden_states.index_select(
                            0, last_indexes.to(draft_prefill_output.all_hidden_states.device()));
                    }
                    execution_result.propose_hidden_states = draft_last_hidden_states.to(torch::kCPU).contiguous();
                } else {
                    const auto batch_size = new_token_ids.size(0);
                    proposed_token_ids    = torch::empty({batch_size, static_cast<int64_t>(propose_step_)},
                                                      torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
                    proposed_token_ids.select(1, 0).copy_(draft_tokens.flatten());
                    mtp::prepareDraftDecodeAfterPrefill(
                        draft_input, draft_prefill_output.all_hidden_states, draft_tokens, position_id_len_factor_);
                }
            }
        }
    }

    /** 3. Return earlier if we are prefill instances. */
    if (role_type_ == RoleType::PREFILL) {
        RTP_LLM_PROFILE_SCOPE("executor.pp.draft_output_sync");
        if (isStageRoot() && !is_dspark_) {
            proposed_token_ids = proposed_token_ids.to(torch::kCPU);
        }

        cudaSyncAndCheck();
        return;
    }

    /** 4. Generate proposals with DSpARK PROPOSE or incremental MTP decode. */
    {
        RTP_LLM_PROFILE_SCOPE_DYNAMIC("executor.pp.draft_decode(is_dspark=%d)", static_cast<int>(is_dspark_));
        if (is_dspark_) {
            if (isStageRoot()) {
                draft_input = prepareDSparkProposeInput(target_input, is_decode, new_token_ids, new_token_lengths);
            }
            tpSyncModelInputs(draft_input, parallelism_config_);
            auto draft_output = forwardDraftModel(*draft_model_, draft_input);
            if (isStageRoot()) {
                const auto batch_size  = draft_input.input_lengths.numel();
                auto draft_temperature = temperature.contiguous().clone();
                auto draft_stochastic   = spec_do_sample.contiguous();
                auto* values           = draft_temperature.data_ptr<float>();
                const auto* stochastic = draft_stochastic.data_ptr<bool>();
                constexpr float kMinDraftTemperature = 1.0e-6f;
                for (int64_t row = 0; row < batch_size; ++row) {
                    if (!std::isfinite(values[row]) || values[row] < 0.0f) {
                        values[row] = 1.0f;
                    }
                    /** Proposal temperature and rejection sampling share main's stochastic() flag. */
                    values[row] = stochastic[row] ? std::max(values[row], kMinDraftTemperature) : kMinDraftTemperature;
                }
                buffer_holder_.hold_host(draft_temperature);
                auto draft_temperature_cuda =
                    draft_temperature.to(draft_output.logits.device(), /*non_blocking=*/true);
                const auto query_width = static_cast<int64_t>(propose_step_)
                                         + static_cast<int64_t>(!dspark_sample_from_anchor_);
                auto anchors = draft_input.combo_tokens.reshape({batch_size, query_width}).select(1, 0).contiguous();
                auto draft_sampler_output = speculative_sampler_->sampleDSparkDraft(draft_output.logits,
                                                                                     anchors,
                                                                                     draft_temperature_cuda,
                                                                                     dspark_markov_w1_,
                                                                                     dspark_markov_w2_,
                                                                                     draft_vocab_size_);
                proposed_token_ids = draft_sampler_output.token_ids;
                proposed_all_probs = draft_sampler_output.all_probs;
            }
        } else {
            for (size_t step = 1; step < propose_step_; ++step) {
                tpSyncModelInputs(draft_input, parallelism_config_);
                auto draft_output = forwardDraftModel(*draft_model_, draft_input);
                mtp::maybeOverrideLastHiddenWithMtpBuffer(
                    draft_output, *draft_model_, draft_input.combo_tokens.numel());

                if (isStageRoot()) {
                    const bool has_next_step = step + 1 < propose_step_;
                    auto       draft_tokens  = fast_topk_sampler_->forward(draft_output.logits, 1).token_ids;
                    /** Keep int32 input for the next forward; copy_ casts the final proposal directly. */
                    if (has_next_step) {
                        draft_tokens = draft_tokens.to(torch::kInt32);
                    }
                    proposed_token_ids.select(1, step).copy_(draft_tokens.flatten());
                    if (has_next_step) {
                        mtp::advanceDraftInput(draft_input,
                                               draft_output.all_hidden_states,
                                               draft_tokens,
                                               position_id_len_factor_,
                                               buffer_holder_);
                    }
                }
            }
        }
    }

    {
        RTP_LLM_PROFILE_SCOPE("executor.pp.draft_output_sync");
        if (isStageRoot()) {
            proposed_token_ids = proposed_token_ids.to(torch::kCPU);
            if (proposed_all_probs.defined()) {
                proposed_all_probs = proposed_all_probs.to(torch::kCPU);
            }
        }
        cudaSyncAndCheck();
    }
}

absl::Status PPExecutor::process(const ScheduleOutput& schedule_output, int64_t schedule_time_us) {
    if (!stopped_) {
        return processImpl(schedule_output, schedule_time_us, false);
    }
    return absl::OkStatus();
}

absl::Status PPExecutor::drainPendingResults() {
    /** current_slot_ is the oldest position after processImpl advances the ring. Skip completed and non-result plans. */
    for (size_t offset = 0; offset < slots_.size(); ++offset) {
        auto& batch = slots_[(current_slot_ + offset) % slots_.size()];
        if (batch.result_pending) {
            RETURN_IF_STATUS_ERROR(processExecutionResult(batch));
        }
    }
    return absl::OkStatus();
}

absl::Status PPExecutor::finish() {
    ScheduleOutput empty_output;
    if (isFirstStage() && isStageRoot()) {
        /** An empty terminal plan skips model execution while preserving TP sync and downstream plan propagation. */
        RETURN_IF_STATUS_ERROR(processImpl(empty_output, 0, true));
        RETURN_IF_STATUS_ERROR(drainPendingResults());
    } else {
        /** Local stop may precede the terminal plan; keep serving upstream until it has been processed. */
        while (!stopped_) {
            RETURN_IF_STATUS_ERROR(process(empty_output));
        }
    }
    for (auto& slot : slots_) {
        waitAll(slot.plan_sends);
        waitAll(slot.activation_sends);
        waitAll(slot.execution_result_sends);
    }
    /** NCCL wait only orders the current stream; complete it before model/transport teardown. */
    cuda_graph::graphGetCurrentStream().synchronize();
    RTP_LLM_LOG_INFO("PP shutdown completed: pp_rank=%ld, tp_rank=%ld, dp_rank=%ld",
                     parallelism_config_.pp_rank,
                     parallelism_config_.tp_rank,
                     parallelism_config_.dp_rank);
    return absl::OkStatus();
}

absl::Status PPExecutor::processImpl(const ScheduleOutput& schedule_output, int64_t schedule_time_us, bool shutdown) {
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
    RtpLLMExecutorMetricsCollector executor_collector;

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
        /** PP input preparation covers the complete execution plan. */
        const auto start_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
        auto       plan_status   = buildPlan(scheduled_stream_groups, schedule_output.finished_request_ids);
        RETURN_IF_STATUS_OR_ERROR(plan_status);
        plan                                    = std::move(plan_status.value());
        plan.shutdown                            = shutdown;
        executor_collector.gather_model_input_us = autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
    } else {
        plan = receivePlan();
    }

    if (isLastStage() && isStageRoot()) {
        for (const auto request_id : plan.finished_request_ids) {
            sampling_states_.erase(request_id);
        }
    }

    auto& model_input = plan.model_input;

    /** 2. do the sync across the all ranks in the same stage. */
    {
        RTP_LLM_PROFILE_SCOPE("executor.pp.tp_sync_input");
        const auto start_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
        if (isFirstStage() && parallelism_config_.tp_size > 1) {
            /** First-stage TP peers enter independently; use the root's terminal plan to stop on the same round. */
            auto shutdown_tensor = torch::tensor({static_cast<int64_t>(plan.shutdown)}, torch::kInt64);
            execBroadcastCpu({{shutdown_tensor}, 0});
            plan.shutdown = shutdown_tensor.item<int64_t>() != 0;
        }
        tpSyncModelInputs(model_input, parallelism_config_);
        executor_collector.tp_sync_input_us = autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
    }

    releaseAllModelBuffers();

    /** 3. Wait on this slot's previous sends before resetting it. CUDA waits order subsequent operations on the current
     * stream after communication. */
    auto& inflight = slots_[current_slot_];
    {
        RTP_LLM_PROFILE_SCOPE("executor.pp.wait_slot_reuse");
        waitAll(inflight.plan_sends);
        waitAll(inflight.activation_sends);
        waitAll(inflight.execution_result_sends);
    }
    inflight.reset();
    if (isFirstStage() && isStageRoot()) {
        /** Keep first-stage receives paired with last-stage sends, including ordinary fake batches. */
        inflight.result_pending   = !model_input.skip_run;
        inflight.stream_groups    = std::move(scheduled_stream_groups);
        inflight.schedule_time_us = schedule_time_us;
    }

    /** 4. send the plan to next stage  */
    if (!isLastStage()) {
        asyncSendPlan(plan, !isStageRoot(), inflight.plan_sends);
    }

    /** 5. run the batch. */
    if (!model_input.skip_run) {
        if (profile_step_start_) {
            profile_step_start_();
        }

        collectExecutorMetrics(model_input, plan.sampling_plan.sequence_lengths, executor_collector);

        PPTickets             tensor_receives;
        PPIntermediateTensors input_tensors;

        if (!isFirstStage()) {
            input_tensors = receiveTensors(tensor_receives);
            waitAll(tensor_receives);
        }

        if (cache_manager_) {
            RTP_LLM_PROFILE_SCOPE("executor.pp.kv_cache_update");
            if (model_input.kv_cache_update_mapping.defined()) {
                cache_manager_->blockBatchCopyByGroup(
                    decodeCacheUpdateMapping(model_input.kv_cache_update_mapping,
                                             model_input.kv_cache_group_tags,
                                             cache_manager_->cacheConfig().topology()));
            }
        }

        const bool force = isStageRoot() && enable_detail_log_;
        if (force) {
            RTP_LLM_LOG_INFO("model_input: %s", model_input.debugString(force).c_str());
        } else {
            RTP_LLM_LOG_TRACE("model_input: %s", model_input.debugString(force).c_str());
        }
        if (model_inputs_logger_) {
            model_inputs_logger_->log(model_input, ModelInputsModelRole::NORMAL, model_->model_id_);
        }

        const bool cp_enabled = parallelism_config_.prefill_cp_config.is_enabled();
        /** model_input is TP-synchronized; peers have no local sampling plan to gate this wait. */
        mtp::DSparkCacheStoreDrainGuard cache_store_drain_guard{
            track_dspark_cache_store_ && !model_input.warmup && model_input.pd_separation,
            model_.get(), draft_commit_model_.get()};
        GptModelOutputs model_output;
        {
            /** Keep target CP field updates and upstream activations out of draft inputs. */
            auto target_input = model_input;
            target_input.pp_intermediates = std::move(input_tensors.tensors);
            {
                RTP_LLM_PROFILE_SCOPE_DYNAMIC("executor.pp.target_forward(verify=%d,batch=%zu,tokens=%zu)",
                                              static_cast<int>(model_input.is_target_verify),
                                              static_cast<size_t>(model_input.input_lengths.numel()),
                                              static_cast<size_t>(model_input.combo_tokens.numel()));
                const auto start_time_us            = autil::TimeUtility::currentTimeInMicroSeconds();
                model_output                        = model_->forward(target_input);
                executor_collector.model_forward_us = autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
            }

            if (expert_balancer_) {
                RTP_LLM_PROFILE_SCOPE("executor.pp.eplb_step");
                const auto start_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
                expert_balancer_->stepForward(*model_, executor_collector, !model_input.is_fake_stream);
                executor_collector.eplb_step_latency_us = autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
            }

            RTP_LLM_PROFILE_SCOPE("executor.pp.wait_forward_done");
            auto forward_done = cuda_graph::makeGraphEvent();
            forward_done.record(cuda_graph::graphGetCurrentStream());
            forward_done.synchronize();
        }

        if (!isLastStage()) {
            PPIntermediateTensors output_tensors{std::move(model_output.pp_intermediates)};
            asyncSendTensors(output_tensors, inflight.activation_sends);
        } else {
            PPExecutionResult execution_result;
            if (isStageRoot()) {
                sampleTokens(plan, model_output, execution_result);
                if (sp_enabled_) {
                    fillFailedDraftRows(execution_result);
                }
            }

            if (sp_enabled_) {
                mtp::maybeOverrideLastHiddenWithMtpBuffer(
                    model_output, *model_, cp_enabled ? -1 : model_input.combo_tokens.numel());

                /** Only model_input carries the target phase to every TP rank. */
                runDraftStep(model_input,
                             plan.draft_next_position_ids,
                             model_output.all_hidden_states,
                             model_input.is_target_verify,
                             plan.sampling_plan,
                             execution_result);
            }

            if (cache_store_drain_guard.armed) {
                const auto error = mtp::finishDSparkCachePublication(
                    model_.get(), draft_commit_model_.get(), parallelism_config_.tp_rank, [this](bool local_ok) {
                        return mtp::reduceDSparkCacheStoreStatus(local_ok, parallelism_config_.tp_size,
                                                               dspark_cache_store_sync_stream_,
                                                               dspark_cache_store_status_);
                    });
                cache_store_drain_guard.disarm();
                if (!error.empty() && isStageRoot()) {
                    /** Preserve earlier row errors and send the normal result so stage 0 can finish
                     * its in-flight requests without committing tokens from this failed batch. */
                    for (auto& request_error : execution_result.request_errors) {
                        if (!request_error.hasError()) {
                            request_error = ErrorInfo(ErrorCode::CACHE_STORE_STORE_FAILED, error);
                        }
                    }
                }
            }

            if (isStageRoot()) {
                if (!model_input.is_fake_stream) {
                    advanceSamplingStates(plan.sampling_plan, execution_result);
                }
                asyncSendExecutionResult(execution_result, inflight.execution_result_sends);
            }
        }
        if (isFirstStage() && isStageRoot()) {
            inflight.executor_collector = executor_collector;
        }
        if (profile_step_finish_) {
            profile_step_finish_();
        }
    }

    current_slot_ = (current_slot_ + 1) % slots_.size();

    /** 6. recv the execution result of next batch and process it. */
    auto& next_batch = slots_[current_slot_];
    if (isFirstStage() && isStageRoot() && next_batch.result_pending) {
        RETURN_IF_STATUS_ERROR(processExecutionResult(next_batch));
    }

    stopped_ = plan.shutdown;
    return absl::OkStatus();
}

bool PPExecutor::updateEplbConfig(const EPLBConfig& config) {
    if (expert_balancer_) {
        return expert_balancer_->updateEplbConfig(config);
    }
    return true;
}

void PPExecutor::collectExecutorMetrics(const GptModelInputs&           model_input,
                                       const torch::Tensor&            sequence_lengths,
                                       RtpLLMExecutorMetricsCollector& collector) const {
    if (!metrics_reporter_ || !isFirstStage() || !isStageRoot() || model_input.is_fake_stream) {
        return;
    }

    collector.generate_batch_size =
        model_input.is_target_verify ? model_input.input_lengths.numel() : model_input.sequence_lengths.numel();
    collector.context_batch_size = model_input.input_lengths.numel() - collector.generate_batch_size;
    collector.execute_token_size = model_input.combo_tokens.numel();
    /** Capture global batch dimensions before CP mutates model inputs. */
    const auto* length_data = sequence_lengths.data_ptr<int32_t>();
    collector.max_seq_len   = *std::max_element(length_data, length_data + sequence_lengths.numel());
    if (collector.context_batch_size != 0) {
        collector.context_batch_size_when_has_context  = collector.context_batch_size;
        collector.generate_batch_size_when_has_context = collector.generate_batch_size;
        collector.execute_token_size_when_has_context  = collector.execute_token_size;
        collector.max_seq_len_when_has_context         = collector.max_seq_len;
    }
}

void PPExecutor::collectTokenCounts(const StreamGroups&                      stream_groups,
                                    const PPExecutionResult&                 result,
                                    StreamGroups::TokenCountsByPriority&     token_counts_by_priority,
                                    RtpLLMSpeculativeEngineMetricsCollector& sp_collector) const {
    if (!metrics_reporter_) {
        return;
    }

    sp_collector.spec_steps = propose_step_;
    /** Result rows follow allStreams order; checked indexing precedes dispatch's result validation. */
    int64_t row = 0;
    for (const auto& stream : stream_groups.allStreams()) {
        if (stream->isContextStream()) {
            /** Sampling errors do not undo the prefill input work already executed. */
            auto&      counts         = token_counts_by_priority[stream->priority()];
            const auto execute_tokens = stream->currentExecuteTokenSize();
            counts.context += execute_tokens;
            counts.context_with_cache += execute_tokens;
            counts.total += execute_tokens;
            if (stream->reuseLength() > 0) {
                counts.context_with_cache += static_cast<int64_t>(stream->reuseLength()) * stream->currentBatchSize();
            }
        } else if (!result.request_errors.at(row).hasError()) {
            auto&         counts = token_counts_by_priority[stream->priority()];
            const int64_t generate_tokens =
                sp_enabled_ ? result.new_token_lengths[row].item<int32_t>() : stream->currentBatchSize();
            counts.generate += generate_tokens;
            counts.total += generate_tokens;
            if (sp_enabled_) {
                /** Preserve verification acceptance before request-length clipping at dispatch. */
                ++sp_collector.total_stream_num;
                sp_collector.total_propose_token_num += propose_step_;
                sp_collector.total_accepted_token_num += generate_tokens;
            }
        }
        ++row;
    }
}

void PPExecutor::reportResultMetrics(InflightBatch&                             batch,
                                     const StreamGroups::TokenCountsByPriority& token_counts_by_priority,
                                     RtpLLMSpeculativeEngineMetricsCollector&   sp_collector) {
    if (!metrics_reporter_ || warm_up_ || !isFirstStage() || !isStageRoot()) {
        return;
    }
    const int64_t batch_latency_us = autil::TimeUtility::currentTimeInMicroSeconds() - batch.schedule_time_us;
    if (batch_latency_us <= 0) {
        return;
    }

    /** Executor timings cover only first-stage local work and result dispatch; last-stage sampling is omitted.
     * TODO: Aggregate costs for the same batch across PP stages, including target and draft forward work. */
    metrics_reporter_->report<RtpLLMExecutorMetrics, RtpLLMExecutorMetricsCollector>(nullptr, &batch.executor_collector);

    int64_t generate_tokens = 0;
    int64_t total_tokens    = 0;
    for (const auto& entry : token_counts_by_priority) {
        generate_tokens += entry.second.generate;
        total_tokens += entry.second.total;
    }
    RtpLLMTokenPSMetricsCollector tps_collector;
    tps_collector.addTokenSize(batch.stream_groups.contextExecuteTokenSize(),
                               batch.stream_groups.contextExecuteTokenSizeWithCache(),
                               generate_tokens,
                               total_tokens,
                               batch_latency_us);
    tps_collector.addTokenSizeByPriority(token_counts_by_priority, batch_latency_us);
    tps_reporter_.report(&tps_collector);
    wall_tps_reporter_.report(&tps_collector);

    if (sp_enabled_) {
        sp_collector.step_latency_us = batch_latency_us;
        /** PP does not collect SP phase timings here; retain their zero placeholders, as in MtpExecutor. */
        metrics_reporter_->report<RtpLLMSpeculativeEngineMetrics, RtpLLMSpeculativeEngineMetricsCollector>(nullptr,
                                                                                                       &sp_collector);
    }
}

}  // namespace rtp_llm
