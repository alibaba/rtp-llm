#include "rtp_llm/cpp/normal_engine/pipeline/PPExecutor.h"

#include "rtp_llm/cpp/normal_engine/pipeline/PPSerialization.h"

#include <algorithm>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <unordered_set>
#include <utility>

#include <c10/core/InferenceMode.h>

#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/config/StageLayerRange.h"
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
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#include "rtp_llm/cpp/utils/StatusUtil.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"

namespace rtp_llm {

PPExecutor::ModelFactory PPExecutor::test_model_factory = nullptr;

namespace {

mtp::DraftInputLayout readPPDraftInputLayout() {
    const char* layout = std::getenv("RTP_LLM_PP_DRAFT_INPUT_LAYOUT");
    if (layout == nullptr || std::strcmp(layout, "compact") == 0) {
        return mtp::DraftInputLayout::COMPACT;
    }
    RTP_LLM_CHECK_WITH_INFO(
        std::strcmp(layout, "fixed") == 0, "RTP_LLM_PP_DRAFT_INPUT_LAYOUT must be compact or fixed, got %s", layout);
    return mtp::DraftInputLayout::FIXED_WIDTH;
}

/** Sampling can return before its input-copy stream finishes reading the pinned host buffers. */
void holdSamplerInputHostBuffers(TensorHolder& holder, const SamplerInputs& inputs) {
    holder.hold_host(inputs.token_ids);
    holder.hold_host(inputs.input_lengths);
    holder.hold_host(inputs.sequence_lengths);
    holder.hold_host(inputs.num_beams_in);
    holder.hold_host(inputs.num_beams_out);
    holder.hold_host(inputs.top_k);
    holder.hold_host(inputs.top_p);
    holder.hold_host(inputs.temperature);
    holder.hold_host(inputs.repetition_penalty);
    holder.hold_host(inputs.presence_penalty);
    holder.hold_host(inputs.frequency_penalty);
    holder.hold_host(inputs.no_repeat_ngram_size);
    holder.hold_host(inputs.do_sample);
    holder.hold_host(inputs.finished_mask);
    holder.hold_host(inputs.cum_log_probs);
}

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

void PPExecutor::asyncSendObject(const torch::Tensor& object, PPTickets& tickets, const std::string& group_name) {
    auto object_size = torch::tensor({object.numel()}, torch::kInt64);
    tickets.push_back(transport_->asyncSend(object_size, group_name));
    tickets.push_back(transport_->asyncSend(object, group_name));
}

torch::Tensor PPExecutor::receiveObject(const std::string& group_name) {
    auto object_size  = torch::empty({1}, torch::TensorOptions().dtype(torch::kInt64));
    auto size_receive = transport_->asyncReceive(object_size, group_name);
    size_receive->wait();

    auto object         = torch::empty({object_size.item<int64_t>()}, torch::TensorOptions().dtype(torch::kUInt8));
    auto object_receive = transport_->asyncReceive(object, group_name);
    object_receive->wait();
    return object;
}

void PPExecutor::asyncSendPlan(const torch::Tensor& plan_payload, PPTickets& tickets) {
    RTP_LLM_PROFILE_SCOPE("executor.pp.send_plan");
    asyncSendObject(plan_payload, tickets);
}

PPExecutionPlan PPExecutor::receivePlan(torch::Tensor& plan_payload) {
    RTP_LLM_PROFILE_SCOPE("executor.pp.recv_plan");
    plan_payload = receiveObject();
    auto plan    = pp_serialization::deserializePlan(plan_payload);
    if (plan.model_input.shutdown) {
        RTP_LLM_LOG_INFO("received pipeline shutdown sentinel from previous stage");
    }

    return plan;
}

void PPExecutor::asyncSendExecutionResult(const PPExecutionResult& result, PPTickets& tickets) {
    RTP_LLM_PROFILE_SCOPE("executor.pp.send_result");
    asyncSendObject(pp_serialization::serializeExecutionResult(result), tickets, kPPResultGroupName);
}

void PPExecutor::asyncSendTensors(const PPIntermediateTensors& tensors, PPTickets& tickets) {
    RTP_LLM_PROFILE_SCOPE("executor.pp.send_activations");
    asyncSendObject(pp_serialization::serializeTensorsMetadata(tensors), tickets);
    for (const auto& tensor_entry : tensors.tensors) {
        if (tensor_entry.second.numel() != 0) {
            /** Snapshot on the compute stream so model buffers can be reused while NCCL sends the ticket-owned copy. */
            auto snapshot = tensor_entry.second.clone(torch::MemoryFormat::Contiguous);
            tickets.push_back(transport_->asyncSend(snapshot, kPPDeviceGroupName));
        }
    }
}

PPIntermediateTensors PPExecutor::receiveTensors() {
    RTP_LLM_PROFILE_SCOPE("executor.pp.recv_activations");
    PPTickets             tickets;
    PPIntermediateTensors tensors;
    {
        cuda_graph::GraphStreamGuard stream_guard(cuda_graph::toGraphStream(recv_stream_));
        tensors = pp_serialization::deserializeTensorsMetadata(receiveObject(), recv_buffers_[next_recv_buffer_]);
        for (auto& tensor_entry : tensors.tensors) {
            if (tensor_entry.second.numel() != 0) {
                tickets.push_back(transport_->asyncReceive(tensor_entry.second, kPPDeviceGroupName));
            }
        }
    }

    /** Wait orders forward after NCCL writes; record_stream protects storage until forward's reads finish. */
    waitAll(tickets);
    const auto stream = cuda_graph::graphGetCurrentStream();
    for (auto& tensor_entry : tensors.tensors) {
        if (tensor_entry.second.numel() != 0) {
            tensor_entry.second.record_stream(stream);
        }
    }
    next_recv_buffer_ = (next_recv_buffer_ + 1) % recv_buffers_.size();
    return tensors;
}

void PPExecutor::waitAll(PPTickets& tickets) {
    for (auto& ticket : tickets) {
        ticket->wait();
    }
    tickets.clear();
}

void PPExecutor::reapCompletedActivationSends() {
    for (auto& slot : slots_) {
        auto& tickets = slot.activation_sends;
        /** Gloo metadata stays pending until waitAll(); independent ticket reclamation avoids retaining completed
         * NCCL snapshots until slot reuse. */
        for (auto it = tickets.begin(); it != tickets.end();) {
            if ((*it)->tryComplete()) {
                it = tickets.erase(it);
            } else {
                ++it;
            }
        }
    }
}

void PPExecutor::validateConfig(const EngineInitParams& params) const {
    const auto& parallelism_config = params.parallelism_config;
    RTP_LLM_CHECK_WITH_INFO(parallelism_config.pp_size > 1, "PPExecutor requires pp_size > 1");
#if !USING_CUDA
    RTP_LLM_FAIL("pipeline parallelism requires a CUDA build; set PP_SIZE=1 on non-CUDA platforms");
#endif
    if (!warm_up_ && params.hw_kernel_config.enable_cuda_graph) {
        RTP_LLM_LOG_WARNING("CUDA graph is ignored because pipeline parallelism runs eagerly");
    }
    RTP_LLM_CHECK_WITH_INFO(!params.runtime_config.use_batch_decode_scheduler,
                            "pipeline parallelism does not support use_batch_decode_scheduler; "
                            "disable USE_BATCH_DECODE_SCHEDULER or set PP_SIZE=1");
    RTP_LLM_CHECK_WITH_INFO(params.runtime_config.fifo_scheduler_config.prefill_chunk_size <= 0,
                            "pipeline parallelism does not support chunked prefill yet; "
                            "set prefill_chunk_size=0 (got %ld)",
                            params.runtime_config.fifo_scheduler_config.prefill_chunk_size);
    RTP_LLM_CHECK_WITH_INFO(params.model_config_.hidden_state_capture_layer_ids.empty(),
                            "pipeline parallelism does not support TorchSpec hidden-state capture "
                            "(hidden_state_capture_layer_ids)");

    const char* stream_async = std::getenv("RTP_LLM_STREAM_ASYNC");
    RTP_LLM_CHECK_WITH_INFO(stream_async == nullptr || std::strcmp(stream_async, "1") != 0,
                            "pipeline parallelism does not support RTP_LLM_STREAM_ASYNC=1");
    const char* device_input = std::getenv("RTP_LLM_DEVICE_INPUT");
    RTP_LLM_CHECK_WITH_INFO(device_input == nullptr || std::strcmp(device_input, "1") != 0,
                            "pipeline parallelism does not support RTP_LLM_DEVICE_INPUT=1");
    RTP_LLM_CHECK_WITH_INFO(!parallelism_config.use_ub_comm,
                            "pipeline parallelism does not support user-buffer communication");

    const auto& cp_config  = parallelism_config.prefill_cp_config;
    const bool  sp_enabled = params.sp_config.type != SP_TYPE_NONE;
    RTP_LLM_CHECK_WITH_INFO(!cp_config.is_enabled() || params.pd_sep_config.role_type == RoleType::PREFILL
                                || (!sp_enabled && !(cp_config.kv_cache_sharded && parallelism_config.tp_size > 1)),
                            "pipeline parallelism with active CP requires the PREFILL role for speculative decoding "
                            "or KV sharding with tp_size > 1");

    RTP_LLM_CHECK_WITH_INFO(params.sp_config.type == SP_TYPE_NONE || params.sp_config.type == SP_TYPE_MTP
                                || params.sp_config.type == SP_TYPE_EAGLE || params.sp_config.type == SP_TYPE_DSPARK,
                            "pipeline parallelism only supports MTP, EAGLE and DSpARK speculative decoding");
    RTP_LLM_CHECK_WITH_INFO(!sp_enabled || params.sp_config.gen_num_per_cycle > 0,
                            "PP speculative decoding requires a positive gen_num_per_cycle, got %ld",
                            params.sp_config.gen_num_per_cycle);
    RTP_LLM_CHECK_WITH_INFO(!params.kv_cache_config.enable_remote_cache,
                            "pipeline parallelism does not support enable_remote_cache");
}

int64_t PPExecutor::firstLocalMoeLayer(const ModelConfig& model_config) const {
    const auto range = stageLayerRange(parallelism_config_, parallelism_config_.pp_rank, model_config.num_layers);
    for (const auto layer_id : model_config.moe_layer_index) {
        if (layer_id >= range.begin && layer_id < range.end()) {
            return layer_id;
        }
    }
    return -1;
}

PPExecutor::PPExecutor(const EngineInitParams&                params,
                       const std::shared_ptr<KVCacheManager>& cache_manager,
                       bool                                   warm_up,
                       std::function<void()>                  profile_step_start,
                       std::function<void()>                  profile_step_finish,
                       ProposeModelEngineInitParams*          propose_params):
    Executor(),
    warm_up_(warm_up),
    role_type_(params.pd_sep_config.role_type),
    cache_manager_(cache_manager),
    sp_enabled_(params.sp_config.type != SP_TYPE_NONE),
    is_dspark_(params.sp_config.type == SP_TYPE_DSPARK),
    propose_step_(params.sp_config.gen_num_per_cycle),
    position_id_len_factor_(params.model_config_.attn_config.rope_config.index_factor),
    draft_input_layout_(readPPDraftInputLayout()),
    parallelism_config_(params.parallelism_config),
    pp_layout_(RankLayout::fromParallelismConfig(parallelism_config_)),
    slots_(parallelism_config_.pp_size + 1),
    recv_stream_(cuda_graph::graphGetStreamFromPool(false)),
    profile_step_start_(std::move(profile_step_start)),
    profile_step_finish_(std::move(profile_step_finish)),
    metrics_reporter_(params.metrics_reporter),
    tps_reporter_(MetricsLoopReporter<RtpLLMTokenPSMetrics, RtpLLMTokenPSMetricsCollector>(
        isFirstStage() && isStageRoot() && !warm_up_ ? metrics_reporter_ : nullptr)),
    wall_tps_reporter_(WallClockMetricsLoopReporter<RtpLLMWallClockTokenPSMetrics, RtpLLMTokenPSMetricsCollector>(
        isFirstStage() && isStageRoot() && !warm_up_ ? metrics_reporter_ : nullptr)) {
    validateConfig(params);

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
        const auto first_moe_layer = firstLocalMoeLayer(params.model_config_);
        if (first_moe_layer >= 0) {
            const auto range =
                stageLayerRange(parallelism_config_, parallelism_config_.pp_rank, params.model_config_.num_layers);
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
                                                 std::pair<int64_t, int64_t>{range.begin, range.end()},
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
         params.model_config_.max_seq_len,
         params.model_config_.hidden_size,
         cache_manager_,
         std::nullopt,
         params.model_config_.hc_mult});
    model_init_params.owns_py_model    = !warm_up_;
    model_init_params.metrics_reporter = metrics_reporter_;

    if (!params.py_model.is_none()) {
        RTP_LLM_LOG_INFO("init executor with python model");
        /** PP target stages stay eager for both primary and generation-prefill graph runners. */
        model_ = std::make_unique<PyWrappedModel>(model_init_params,
                                                  params.py_model,
                                                  false,
                                                  sp_enabled_ && isLastStage() && !warm_up_,
                                                  DSparkModelRole::NONE,
                                                  false);
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
                                                  draft_params->model_config_.max_seq_len,
                                                  draft_params->model_config_.hidden_size,
                                                  cache_manager_,
                                                  std::make_optional(0),
                                                  draft_params->model_config_.hc_mult});
            draft_init_params.metrics_reporter = metrics_reporter_;

            if (!params.py_sp_model.is_none()) {
                RTP_LLM_LOG_INFO("init PP executor with python draft model");
                /** PP stays eager: PyWrappedModel also requires pp_size == 1 to enable graphs. */
                const bool draft_graph_allowed = !(is_dspark_ && role_type_ == RoleType::PREFILL);
                if (!is_dspark_ || role_type_ != RoleType::PREFILL) {
                    draft_model_ =
                        std::make_unique<PyWrappedModel>(draft_init_params,
                                                         params.py_sp_model,
                                                         false,
                                                         false,
                                                         is_dspark_ ? DSparkModelRole::PROPOSE : DSparkModelRole::NONE,
                                                         draft_graph_allowed);
                }
                if (is_dspark_) {
                    draft_commit_model_ = std::make_unique<PyWrappedModel>(draft_init_params,
                                                                           params.py_sp_model,
                                                                           false,
                                                                           false,
                                                                           DSparkModelRole::COMMIT,
                                                                           draft_graph_allowed);
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
        const auto& draft_weights = propose_params->getEngineInitParams().gpt_weights;
        d2t_map_                  = draft_model_ ? draft_model_->weights_.d2t_map : draft_weights.d2t_map;
        if (is_dspark_ && role_type_ != RoleType::PREFILL) {
            dspark_markov_w1_ = draft_weights.dspark_markov_w1;
            dspark_markov_w2_ = draft_weights.dspark_markov_w2;
            RTP_LLM_CHECK_WITH_INFO(dspark_markov_w1_.defined() && dspark_markov_w2_.defined(),
                                    "PP DSpARK requires markov_w1 and markov_w2 weights");
            const int64_t padded_draft_vocab_size = (static_cast<int64_t>(draft_vocab_size_) + 127) / 128 * 128;
            RTP_LLM_CHECK_WITH_INFO(dspark_markov_w1_.is_cuda() && dspark_markov_w2_.is_cuda()
                                        && dspark_markov_w1_.dim() == 2 && dspark_markov_w2_.dim() == 2
                                        && dspark_markov_w1_.size(1) == dspark_markov_w2_.size(1)
                                        && dspark_markov_w1_.size(0) >= params.model_config_.vocab_size
                                        && dspark_markov_w2_.size(0) >= static_cast<int64_t>(draft_vocab_size_)
                                        && dspark_markov_w2_.size(0) <= padded_draft_vocab_size
                                        && dspark_markov_w1_.scalar_type() == dspark_markov_w2_.scalar_type(),
                                    "PP DSpARK Markov weights must be CUDA [target_vocab,rank] and "
                                    "[draft_vocab,rank] tensors with matching rank and dtype");
            dspark_markov_w2_ = dspark_markov_w2_.narrow(0, 0, static_cast<int64_t>(draft_vocab_size_));
            if (draft_vocab_size_ != static_cast<size_t>(params.model_config_.vocab_size)) {
                RTP_LLM_CHECK_WITH_INFO(d2t_map_.defined() && d2t_map_.is_cuda() && d2t_map_.dim() == 1
                                            && d2t_map_.scalar_type() == torch::kInt64
                                            && d2t_map_.numel() == static_cast<int64_t>(draft_vocab_size_),
                                        "reduced-vocabulary PP DSpARK requires a CUDA int64 d2t map");
                const auto d2t_min = d2t_map_.min().item<int64_t>();
                const auto d2t_max = d2t_map_.max().item<int64_t>();
                RTP_LLM_CHECK_WITH_INFO(d2t_min >= 0 && d2t_max < params.model_config_.vocab_size,
                                        "PP DSpARK d2t target ids must be in [0,%ld), got range [%ld,%ld]",
                                        params.model_config_.vocab_size,
                                        d2t_min,
                                        d2t_max);
            }
        }
        speculative_sampler_ = std::make_unique<speculative::SpeculativeSampler>(d2t_map_, propose_step_);
        if (!is_dspark_) {
            /** Each TP rank needs a greedy sampler to advance draft tokens without another broadcast. */
            fast_topk_sampler_ = std::make_unique<speculative::FastTopKSampler>(d2t_map_);
        }
        spec_logits_verify_runner_ = std::make_unique<SpecLogitsVerifyRunner>();
    }

    const auto& cache_config = cache_manager_ ? cache_manager_->cacheConfig() : CacheConfig();
    batch_stream_processor_  = std::make_unique<PPBatchStreamProcessor>(params.model_config_,
                                                                       params.pd_sep_config,
                                                                       params.profiling_debug_logging_config,
                                                                       cache_config,
                                                                       params.sp_config,
                                                                       warm_up_);
    LogitsProcessorFactory::init(params.model_config_,
                                 params.grammar_config,
                                 params.sp_config.tree_decode_config,
                                 isFirstStage() && isStageRoot() && !warm_up_ ? metrics_reporter_ : nullptr);
    if (!warm_up_ && isFirstStage() && isStageRoot()) {
        result_recv_thread_ = std::thread(&PPExecutor::resultRecvLoop, this);
    }
    if (!warm_up_ && isLastStage() && isStageRoot()) {
        result_send_thread_ = std::thread(&PPExecutor::resultSendLoop, this);
    }
    cudaProfilerBegin();
}

PPExecutor::~PPExecutor() {
    if (isFirstStage() && isStageRoot()) {
        requestResultRecvStop();
        if (result_recv_thread_.joinable()) {
            result_recv_thread_.join();
        }
    }
    stopResultSender();
    /** Finish local model-buffer consumers before destroying the models. */
    last_compute_done_.synchronize();
    cudaProfilerEnd();
}

void PPExecutor::waitLastRun() {
    RTP_LLM_PROFILE_SCOPE("executor.pp.wait_last_batch");
    if (result_send_thread_.joinable()) {
        /** Wait for CPU result processing as well as GPU completion before reusing batch state and buffers. */
        std::unique_lock<std::mutex> lock(result_send_mutex_);
        result_send_cv_.wait(lock, [this] { return !result_send_task_; });
    } else {
        last_compute_done_.synchronize();
    }
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

void PPExecutor::ensureModelInputOnGpu(GptModelInputs& model_input) {
    if (model_input.skip_run) {
        return;
    }
    /** Keep CPU multimodal tensors alive until their asynchronous H2D copies finish. */
    for (auto* tensors : {&model_input.multimodal_features, &model_input.mm_extra_input}) {
        if (!tensors->has_value()) {
            continue;
        }
        for (auto& tensor : tensors->value()) {
            buffer_holder_.hold_host(tensor);
            tensor = tensor.to(torch::kCUDA, /*non_blocking=*/true);
        }
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
            cache_manager_->blockBatchCopyByGroup(decodeCacheUpdateMapping(model_input.kv_cache_update_mapping,
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

std::optional<ErrorInfo> PPExecutor::validateStream(const GenerateStream& stream, SpeculativeType sp_type) {
    const auto& config     = *stream.generateConfig();
    const bool  sp_enabled = sp_type != SP_TYPE_NONE;
    if (stream.hasNumBeams()) {
        return ErrorInfo(ErrorCode::INVALID_PARAMS, "PP does not support beam search");
    }
    if (config.calculate_loss != 0) {
        return ErrorInfo(ErrorCode::INVALID_PARAMS, "PP does not support calculate_loss");
    }
    if (sp_enabled && config.return_all_probs != ReturnAllProbsMode::NONE) {
        return ErrorInfo(ErrorCode::INVALID_PARAMS, "PP speculative decoding does not support return_all_probs");
    }
    if (sp_enabled && (config.return_softmax_probs || config.return_prompt_logits)) {
        return ErrorInfo(ErrorCode::INVALID_PARAMS,
                         "PP speculative decoding does not support return_softmax_probs or return_prompt_logits");
    }
    /** Requests with max_new_tokens == 0 need no generated tokens, so skip sampling compatibility checks. */
    if (sp_enabled && !config.isPrefillOnly()) {
        if (sp_type == SP_TYPE_DSPARK && stream.maxBatchSize() != 1) {
            return ErrorInfo(ErrorCode::INVALID_PARAMS,
                             "PP DSpARK does not support num_return_sequences > 1");
        }
        if (auto error = LogitsProcessorFactory::validateMtpCompatibility(stream.getAllLogitsProcessorPtr());
            error.has_value()) {
            return error;
        }
    }
    return std::nullopt;
}

void PPExecutor::validateAndFilterStreams(std::list<GenerateStreamPtr>& streams, SpeculativeType sp_type) {
    RTP_LLM_PROFILE_SCOPE("executor.pp.validate_streams");
    for (auto it = streams.begin(); it != streams.end();) {
        const auto& stream = *it;
        if (auto error = validateStream(*stream, sp_type); error.has_value()) {
            stream->reportError(error->code(), error->ToString());
            /** The scheduler marked this request in flight, but no plan will carry it. */
            stream->clearPPInflight();
            it = streams.erase(it);
            continue;
        }
        ++it;
    }
}

void PPExecutor::prepareStreams(const std::list<GenerateStreamPtr>& streams) {
    RTP_LLM_PROFILE_SCOPE("executor.pp.prepare_streams");
    const auto token_count = static_cast<int64_t>(propose_step_ + 1);
    for (const auto& stream : streams) {
        if (sp_enabled_ && !stream->generateConfig()->isPrefillOnly()) {
            auto       sp_output_buffer = stream->getSPOutputBuffer();
            const bool is_fake_stream   = stream->isFakeStream() || stream->isPerfTest();
            if (!sp_output_buffer) {
                /** Real decode streams must carry the token buffer from the previous step or PD admission. */
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
                    && sp_output_buffer->tokens.scalar_type() == torch::kInt32
                    && sp_output_buffer->tokens.is_contiguous() && sp_output_buffer->tokens.numel() == token_count,
                "PP speculative token buffer must contain one target token and %ld draft tokens, request_id=%ld",
                propose_step_,
                stream->streamId());
            sp_output_buffer->propose_step = propose_step_;
        }
    }
}

absl::StatusOr<PPExecutionPlan> PPExecutor::buildPlan(const StreamGroups&           stream_groups,
                                                      const std::vector<RequestId>& finished_request_ids,
                                                      bool                          shutdown) {
    RTP_LLM_PROFILE_SCOPE("executor.pp.build_plan");
    RTP_LLM_CHECK_WITH_INFO(isFirstStage(), "only the first PP stage can build an execution plan from streams");

    for (const auto request_id : finished_request_ids) {
        token_history_lengths_.erase(request_id);
    }

    return batch_stream_processor_->gatherExecutionPlan(
        stream_groups, finished_request_ids, shutdown, token_history_lengths_, buffer_holder_);
}

void PPExecutor::advanceSamplingStates(const PPSamplingPlan& sampling_plan, PPExecutionResult& result) {
    RTP_LLM_PROFILE_SCOPE("executor.pp.update_sampling_states");
    const auto stream_count = sampling_plan.request_ids.size(0);

    const auto* request_ids      = sampling_plan.request_ids.data_ptr<RequestId>();
    const auto* input_lengths    = sampling_plan.input_lengths.data_ptr<int32_t>();
    const auto* sequence_lengths = sampling_plan.sequence_lengths.data_ptr<int32_t>();
    const auto* max_tokens       = sampling_plan.max_tokens.data_ptr<int32_t>();
    int64_t     batch_idx        = 0;
    for (int64_t stream_idx = 0; stream_idx < stream_count; ++stream_idx) {
        const int64_t stream_batch_size = std::max<int32_t>(sampling_plan.num_return_sequences[stream_idx], 1);

        if (result.request_errors[stream_idx].hasError()) {
            batch_idx += stream_batch_size;
            continue;
        }
        auto& state = sampling_states_.at(request_ids[stream_idx]).sampling;

        std::optional<ErrorInfo> error;
        const auto               num_new_tokens = result.new_token_lengths.data_ptr<int32_t>()[batch_idx];
        /** Match the first stage's eventual write length without changing the returned acceptance count. */
        const auto num_committed_tokens =
            sp_enabled_ ? std::min(num_new_tokens, max_tokens[batch_idx] - sequence_lengths[batch_idx]) :
                          num_new_tokens;
        const auto new_tokens =
            result.new_token_ids.narrow(0, batch_idx, stream_batch_size).narrow(1, 0, num_committed_tokens);
        for (const auto& processor : state.logits_processors) {
            error = processor->updateStatus(new_tokens, num_committed_tokens);
            if (error.has_value()) {
                break;
            }
        }
        const int64_t expected_output_len =
            sequence_lengths[batch_idx] - input_lengths[batch_idx] + num_committed_tokens;
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
        /**
         * PP does not support beam search, so cumulative scores do not affect token selection.
         * Only requests returning those scores need them retained across steps.
         */
        const auto& cum_log_probs = result.request_outputs[stream_idx].cum_log_probs;
        if (error.has_value()) {
            result.request_errors[stream_idx] = std::move(error.value());
        } else if (cum_log_probs.defined()) {
            state.cum_log_probs.copy_(cum_log_probs);
        }
        batch_idx += stream_batch_size;
    }
}

void PPExecutor::eraseFailedRequestStates(const PPExecutionResult& result) {
    const auto  stream_count = result.request_ids.size(0);
    const auto* request_ids  = result.request_ids.data_ptr<RequestId>();
    for (int64_t row = 0; row < stream_count; ++row) {
        if (result.request_errors[row].hasError()) {
            sampling_states_.erase(request_ids[row]);
            dspark_q_.erase(request_ids[row]);
        }
    }
}

void PPExecutor::verifyDraftTokens(const PPExecutionPlan&                        plan,
                                   const torch::Tensor&                          target_logits,
                                   SamplerOutput&                                draft_sampler_output,
                                   const speculative::SpeculativeSamplingParams& verify_params,
                                   SamplerOutput&                                target_sampler_output,
                                   speculative::SpeculativeSamplerOutput&        accepted) {
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
    holdSamplerInputHostBuffers(buffer_holder_, inputs);
    const auto vocab_size = static_cast<int64_t>(inputs.vocab_size);

    SpecLogitsVerifyRunner::LaunchTask task;
    task.total_streams      = batch_size;
    task.propose_step       = propose_step_;
    task.vocab_size         = vocab_size;
    task.draft_tokens       = draft_sampler_output.token_ids;
    const auto* request_ids = plan.sampling_plan.request_ids.data_ptr<RequestId>();
    for (int64_t row = 0; row < batch_size; ++row) {
        const auto& state = sampling_states_.at(request_ids[row]).sampling;
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
        *speculative_sampler_, verify_params, draft_sampler_output, target_sampler_output, verify_result, accepted);
}

void PPExecutor::broadcastPostRejectionInputs(GptModelInputs& draft_input) {
    RTP_LLM_PROFILE_SCOPE("executor.pp.tp_sync_post_rejection");

    mtp::syncDraftPrefillAfterVerify(draft_input, draft_input_layout_, parallelism_config_);
}

GptModelOutputs PPExecutor::forwardDraftModel(ModelBase& draft_model, GptModelInputs& draft_input) {
    if (cache_manager_) {
        mtp::applyCacheStrideToModelInput(draft_input, cache_manager_->getMTPModuleCacheConfig(0));
    }
    if (model_inputs_logger_) {
        model_inputs_logger_->log(draft_input, ModelInputsModelRole::DRAFT, draft_model.model_id_);
    }
    return draft_model.forward(draft_input);
}

PPSamplingOutput PPExecutor::sampleTokens(const PPExecutionPlan&  plan,
                                          const GptModelOutputs&  model_output,
                                          GrammarPrefetchContext* grammar_prefetch) {
    RTP_LLM_PROFILE_SCOPE("executor.pp.target_sample");
    PPSamplingOutput output;
    const auto       stream_count = plan.sampling_plan.request_ids.size(0);
    output.request_errors.assign(stream_count, ErrorInfo::OkStatus());
    if (plan.model_input.is_fake_stream || plan.model_input.skip_lm_head) {
        /** Placeholders let fake and prefill-only batches join draft collectives without creating sampling state. */
        if (sp_enabled_ && plan.is_decode) {
            output.verify_output.accept_tokens = torch::zeros({stream_count, propose_step_ + 1}, torch::kInt32);
            output.verify_output.accept_len    = torch::full({stream_count}, propose_step_ + 1, torch::kInt32);
        } else {
            output.sampler_output.token_ids = torch::zeros({stream_count, 1}, torch::kInt32);
        }
        return output;
    }
    speculative::SpeculativeSamplingParams verify_params;
    SamplerOutput                          draft_sampler_output;
    batch_stream_processor_->prepareSampling(plan,
                                             dspark_q_,
                                             d2t_map_,
                                             sampling_states_,
                                             output.request_errors,
                                             draft_sampler_output,
                                             verify_params,
                                             grammar_prefetch);
    if (sp_enabled_ && plan.is_decode) {
        verifyDraftTokens(plan,
                          model_output.logits,
                          draft_sampler_output,
                          verify_params,
                          output.sampler_output,
                          output.verify_output);
    } else {
        const auto start_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
        auto       inputs        = batch_stream_processor_->gatherSamplerInputs(
            plan.sampling_plan, plan.output_config, model_output.logits, sampling_states_);
        holdSamplerInputHostBuffers(buffer_holder_, inputs);
        if (inputs.cum_log_probs.defined()) {
            /** Accumulate scores on CUDA to defer their D2H copy until result submission. */
            inputs.cum_log_probs = inputs.cum_log_probs.to(model_output.logits.device(), /*non_blocking=*/true);
        }
        output.sampler_output  = sampler_->forward(inputs);
        output.sample_input_us = autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
    }
    return output;
}

PPDraftOutput PPExecutor::runDraftStep(const PPExecutionPlan&  plan,
                                       const GptModelOutputs&  target_output,
                                       const PPSamplingOutput& sampling_output) {
    /** Preserve batch dimensions and verify mode before draft preparation changes the input layout. */
    const bool is_verify                = plan.model_input.is_target_verify;
    const bool should_cache_draft_probs = !plan.model_input.is_fake_stream && !plan.model_input.skip_lm_head;
    const auto request_batch_size       = plan.model_input.input_lengths.numel();

    const auto& sampling_plan           = plan.sampling_plan;
    const auto& draft_next_position_ids = plan.draft_next_position_ids;
    const auto& sampler_output          = sampling_output.sampler_output;
    const auto& verify_output           = sampling_output.verify_output;
    const auto& target_hidden_states    = target_output.all_hidden_states;
    const bool  cp_enabled              = parallelism_config_.prefill_cp_config.is_enabled();

    auto to_i32_flat = [request_batch_size](const torch::Tensor& tokens) {
        return tokens.to(torch::kInt32, /*non_blocking=*/true).reshape({request_batch_size}).contiguous();
    };

    PPDraftOutput  output;
    auto&          proposed_token_ids = output.proposals.token_ids;
    GptModelInputs draft_input        = plan.model_input;
    torch::Tensor  hidden_states_d_t;
    draft_input.pd_separation = draft_input.pd_separation && !draft_input.skip_lm_head;
    draft_input.skip_lm_head  = false;

    /** 1. Run draft prefill/COMMIT and sample the first MTP/EAGLE proposal token. */
    {
        RTP_LLM_PROFILE_SCOPE("executor.pp.draft_prefill");
        if (is_dspark_) {
            /** COMMIT consumes target features; synchronize the sampled anchor later with PROPOSE inputs. */
            batch_stream_processor_->prepareDSparkCommitInput(draft_input, target_hidden_states, cp_enabled);
            forwardDraftModel(*draft_commit_model_, draft_input);
        } else {
            if (isStageRoot()) {
                if (is_verify) {
                    batch_stream_processor_->prepareDraftPrefillAfterVerify(draft_input,
                                                                            target_hidden_states,
                                                                            verify_output,
                                                                            draft_input_layout_,
                                                                            hidden_states_d_t,
                                                                            buffer_holder_);
                } else {
                    batch_stream_processor_->prepareDraftPrefillAfterTargetPrefill(draft_input,
                                                                                   target_hidden_states,
                                                                                   sampler_output.token_ids,
                                                                                   draft_next_position_ids,
                                                                                   buffer_holder_);
                }
            } else if (is_verify && draft_input_layout_ == mtp::DraftInputLayout::FIXED_WIDTH) {
                /** Fixed-width rejection sync skips shape exchange; allocate peer buffers for its CUDA broadcasts. */
                const auto cuda_i32            = torch::TensorOptions(torch::kInt32).device(torch::kCUDA);
                draft_input.is_target_verify   = false;
                draft_input.combo_tokens       = torch::empty({request_batch_size * (propose_step_ + 1)}, cuda_i32);
                draft_input.lm_output_indexes  = torch::empty({request_batch_size}, cuda_i32);
                draft_input.last_hidden_states = target_hidden_states;
            }

            if (is_verify) {
                broadcastPostRejectionInputs(draft_input);
            } else {
                /** Broadcast tokens and metadata without replacing each rank's CP hidden states. */
                draft_input.last_hidden_states = cp_enabled ? torch::Tensor() : draft_input.last_hidden_states;
                tpSyncModelInputs(draft_input, parallelism_config_);
                draft_input.last_hidden_states = cp_enabled ? target_hidden_states : draft_input.last_hidden_states;
            }

            auto draft_prefill_output = forwardDraftModel(*draft_model_, draft_input);

            if (isStageRoot()) {
                auto draft_sample  = fast_topk_sampler_->forward(draft_prefill_output.logits, 1);
                auto draft_tokens  = to_i32_flat(draft_sample.token_ids);
                proposed_token_ids = draft_tokens.unsqueeze(1);

                if (role_type_ == RoleType::PREFILL || propose_step_ > 1) {
                    /** Continue from each request's last draft hidden state, before hc_head on DeepSeek V4.
                     * CP supplies gathered last rows; otherwise select them from the full draft output. */
                    torch::Tensor draft_last_hidden_states;
                    if (cp_enabled) {
                        draft_last_hidden_states = draft_model_->getMtpLastHiddenStates(request_batch_size);
                        RTP_LLM_CHECK_WITH_INFO(draft_last_hidden_states.defined()
                                                    && draft_last_hidden_states.dim() == 2
                                                    && draft_last_hidden_states.size(0) == draft_tokens.size(0),
                                                "PP CP draft last hidden states must be a 2-D tensor with %ld rows "
                                                "(one per request)",
                                                request_batch_size);
                    } else {
                        mtp::maybeOverrideLastHiddenWithMtpBuffer(
                            draft_prefill_output, *draft_model_, draft_input.combo_tokens.numel());
                        const auto& hidden_states = draft_prefill_output.all_hidden_states;
                        const auto  indexes = draft_input.lm_output_indexes.to(hidden_states.device(), torch::kLong);
                        draft_last_hidden_states = hidden_states.index_select(0, indexes);
                    }

                    if (role_type_ == RoleType::PREFILL) {
                        output.proposals.all_probs = draft_sample.all_probs.unsqueeze(1);
                        output.hidden_states       = std::move(draft_last_hidden_states);
                    } else {
                        batch_stream_processor_->prepareDraftDecodeAfterPrefill(
                            draft_input, draft_last_hidden_states, draft_tokens);
                    }
                }
            }
        }
    }

    /** 2. Return the P-side handoff; further proposal generation belongs to the decode side. */
    if (role_type_ == RoleType::PREFILL) {
        return output;
    }

    /** 3. Generate DSpARK proposals or the remaining MTP/EAGLE draft tokens. */
    {
        RTP_LLM_PROFILE_SCOPE_DYNAMIC("executor.pp.draft_decode(is_dspark=%d)", static_cast<int>(is_dspark_));
        if (is_dspark_) {
            mtp::DSparkRoundState round_state;
            if (isStageRoot()) {
                round_state = batch_stream_processor_->buildDSparkRoundState(plan, sampling_output, buffer_holder_);
                batch_stream_processor_->prepareDSparkProposeModelInput(round_state, draft_input, buffer_holder_);
            }
            tpSyncModelInputs(draft_input, parallelism_config_);
            auto draft_output = forwardDraftModel(*draft_model_, draft_input);
            if (isStageRoot()) {
                const auto batch_size        = draft_input.input_lengths.numel();
                auto       draft_temperature = batch_stream_processor_->prepareDSparkDraftTemperature(
                    sampling_plan, draft_output.logits.device(), buffer_holder_);
                output.proposals = speculative_sampler_->sampleDSparkDraft(draft_output.logits,
                                                                           round_state.anchors,
                                                                           draft_temperature,
                                                                           dspark_markov_w1_,
                                                                           dspark_markov_w2_,
                                                                           draft_vocab_size_);
                if (should_cache_draft_probs) {
                    const auto* request_ids = sampling_plan.request_ids.data_ptr<RequestId>();
                    for (int64_t request_idx = 0; request_idx < batch_size; ++request_idx) {
                        dspark_q_.insert_or_assign(request_ids[request_idx], output.proposals.all_probs[request_idx]);
                    }
                }
            }
        } else if (propose_step_ > 1) {
            /** Root updates tokens, hidden states and lengths after prefill; synchronize them before decode. */
            tpSyncModelInputs(draft_input, parallelism_config_);

            std::vector<torch::Tensor> draft_token_columns;
            draft_token_columns.reserve(propose_step_);
            draft_token_columns.push_back(to_i32_flat(draft_input.combo_tokens));
            for (int64_t step = 1; step < propose_step_; ++step) {
                auto draft_output = forwardDraftModel(*draft_model_, draft_input);
                mtp::maybeOverrideLastHiddenWithMtpBuffer(
                    draft_output, *draft_model_, draft_input.combo_tokens.numel());

                auto draft_tokens = to_i32_flat(fast_topk_sampler_->forward(draft_output.logits, 1).token_ids);
                draft_token_columns.push_back(draft_tokens);
                const bool has_next_step = step + 1 < propose_step_;
                if (has_next_step) {
                    mtp::advanceDraftInput(draft_input, draft_output.all_hidden_states, draft_tokens);
                }
            }
            if (isStageRoot()) {
                proposed_token_ids = torch::stack(draft_token_columns, 1);
            }
        }
    }

    return output;
}

absl::Status PPExecutor::process(const ScheduleOutput& schedule_output, int64_t schedule_time_us) {
    if (!stopped_) {
        return processImpl(schedule_output, schedule_time_us, false);
    }
    return absl::OkStatus();
}

absl::Status PPExecutor::finish() {
    ScheduleOutput empty_output;
    if (isFirstStage() && isStageRoot()) {
        /** An empty terminal plan still synchronizes the first stage and propagates down every PP lane. */
        RETURN_IF_STATUS_ERROR(processImpl(empty_output, 0, true));
        if (result_recv_thread_.joinable()) {
            result_recv_thread_.join();
        }
    } else {
        /** Local stop may precede the terminal plan; keep serving upstream until it has been processed. */
        while (!stopped_) {
            RETURN_IF_STATUS_ERROR(process(empty_output));
        }
    }
    stopResultSender();
    for (auto& slot : slots_) {
        waitAll(slot.plan_sends);
        waitAll(slot.activation_sends);
        waitAll(slot.execution_result_sends);
    }
    /** NCCL wait only orders the current stream; complete it before model/transport teardown. */
    cuda_graph::graphGetCurrentStream().synchronize();
    if (isFirstStage() && isStageRoot()) {
        token_history_lengths_.clear();
    }
    if (isLastStage() && isStageRoot()) {
        sampling_states_.clear();
        dspark_q_.clear();
    }
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
    auto                           tps_guard      = tps_reporter_.makeActiveGuard(report_active);
    auto                           wall_tps_guard = wall_tps_reporter_.makeActiveGuard(report_active);
    RtpLLMExecutorMetricsCollector executor_collector;
    executor_collector.pp_rank = parallelism_config_.pp_rank;

    /** 0. Prepare SP buffers after admission and DP fake padding in the engine. */
    const auto& streams = schedule_output.streams;
    if (isFirstStage() && isStageRoot()) {
        prepareStreams(streams);
    }

    /** 1. Build or receive the plan and retain its CPU payload for forwarding. */
    PPExecutionPlan plan;
    torch::Tensor   plan_payload;
    StreamGroups    stream_groups;
    if (isFirstStage()) {
        stream_groups = StreamGroups(streams);
        /** PP input preparation covers the complete execution plan. */
        const auto start_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
        auto       plan_status   = buildPlan(stream_groups, schedule_output.finished_request_ids, shutdown);
        RETURN_IF_STATUS_OR_ERROR(plan_status);
        plan                                     = std::move(plan_status.value());
        executor_collector.gather_model_input_us = autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
        {
            /** Each lane carries the inputs its first-stage rank will execute, including empty/terminal controls. */
            RTP_LLM_PROFILE_SCOPE("executor.pp.tp_sync_input");
            const auto start_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
            tpSyncModelInputs(plan.model_input, parallelism_config_);
            executor_collector.tp_sync_input_us = autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
        }
        plan_payload = pp_serialization::serializePlan(plan);
    } else {
        const auto start_time_us        = autil::TimeUtility::currentTimeInMicroSeconds();
        plan                            = receivePlan(plan_payload);
        executor_collector.plan_wait_us = autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
    }

    /** 2. Forward before waiting on this slot's previous sends; keep this batch's tickets separate. */
    PPTickets pending_plan_sends;
    if (!isLastStage()) {
        asyncSendPlan(plan_payload, pending_plan_sends);
    }

    auto&      model_input  = plan.model_input;
    const bool prefill_only = model_input.skip_lm_head;

    /** 3. Wait on this slot's previous sends before reusing it. CUDA waits order subsequent operations on the current
     * stream after communication. */
    auto& inflight = slots_[current_slot_];
    {
        RTP_LLM_PROFILE_SCOPE("executor.pp.wait_slot_reuse");
        const auto start_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
        waitAll(inflight.plan_sends);
        waitAll(inflight.activation_sends);
        waitAll(inflight.execution_result_sends);
        executor_collector.slot_reuse_wait_us = autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
    }
    inflight.plan_sends = std::move(pending_plan_sends);

    /** 4. Overlap the next activation receive with the previous batch's GPU work and CPU result processing. */
    PPIntermediateTensors input_tensors;
    PPPendingResult       pending_result;
    if (!model_input.skip_run) {
        if (profile_step_start_) {
            profile_step_start_();
        }
        collectExecutorMetrics(model_input, plan.is_decode, plan.sampling_plan.sequence_lengths, executor_collector);
        if (!isFirstStage()) {
            const auto start_time_us              = autil::TimeUtility::currentTimeInMicroSeconds();
            input_tensors                         = receiveTensors();
            executor_collector.activation_wait_us = autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
        }
    }
    {
        const auto start_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
        waitLastRun();
        executor_collector.last_run_wait_us = autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
    }
    reapCompletedActivationSends();
    releaseAllModelBuffers();
    ensureModelInputOnGpu(model_input);

    if (isLastStage() && isStageRoot()) {
        for (const auto request_id : plan.finished_request_ids) {
            sampling_states_.erase(request_id);
            dspark_q_.erase(request_id);
        }
    }

    if (!model_input.skip_run) {
        std::optional<GrammarPrefetchContext> grammar_prefetch;
        if (isLastStage() && isStageRoot() && !prefill_only && !model_input.is_fake_stream) {
            RTP_LLM_PROFILE_SCOPE("executor.pp.grammar_prefetch");
            grammar_prefetch =
                batch_stream_processor_->prepareLogitsProcessorsAsync(plan.sampling_plan, sampling_states_);
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

        GptModelOutputs model_output;
        {
            /** Keep target CP field updates and upstream activations out of draft inputs. */
            auto target_input             = model_input;
            target_input.pp_intermediates = std::move(input_tensors.tensors);
            {
                RTP_LLM_PROFILE_SCOPE_DYNAMIC("executor.pp.target_forward(verify=%d,batch=%" PRId64 ",tokens=%" PRId64 ")",
                                              model_input.is_target_verify,
                                              model_input.input_lengths.numel(),
                                              model_input.combo_tokens.numel());
                const auto start_time_us            = autil::TimeUtility::currentTimeInMicroSeconds();
                model_output                        = model_->forward(target_input);
                executor_collector.model_forward_us = autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
            }

            if (expert_balancer_) {
                RTP_LLM_PROFILE_SCOPE("executor.pp.eplb_step");
                const auto start_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
                expert_balancer_->stepForward(*model_, executor_collector, !model_input.is_fake_stream);
                executor_collector.eplb_step_latency_us =
                    autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
            }
        }

        if (!isLastStage()) {
            PPIntermediateTensors output_tensors{std::move(model_output.pp_intermediates)};
            const auto            start_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
            asyncSendTensors(output_tensors, inflight.activation_sends);
            executor_collector.send_activation_us = autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
        } else {
            PPSamplingOutput sampling_output;
            if (isStageRoot()) {
                sampling_output = sampleTokens(
                    plan, model_output, grammar_prefetch.has_value() ? &grammar_prefetch.value() : nullptr);
            }

            PPDraftOutput draft_output;
            /** Prefill-only DP peers must still participate in the draft model's collectives. */
            if (sp_enabled_ && (!prefill_only || parallelism_config_.dp_size > 1)) {
                const bool cp_enabled = parallelism_config_.prefill_cp_config.is_enabled();
                mtp::maybeOverrideLastHiddenWithMtpBuffer(
                    model_output, *model_, cp_enabled ? -1 : model_input.combo_tokens.numel());
                draft_output = runDraftStep(plan, model_output, sampling_output);
            }

            if (isStageRoot()) {
                pending_result =
                    batch_stream_processor_->submitExecutionResult(plan, model_output, sampling_output, draft_output);
            }
        }
        last_compute_done_.record(cuda_graph::graphGetCurrentStream());
        if (!isFirstStage() && !isLastStage() && isStageRoot()) {
            reportStageMetrics(executor_collector);
        }
        if (profile_step_finish_) {
            profile_step_finish_();
        }
    }

    if (isFirstStage() && isStageRoot() && !model_input.skip_run) {
        enqueueResultRecvTask({std::move(stream_groups),
                               schedule_time_us,
                               std::move(executor_collector),
                               std::move(tps_guard),
                               std::move(wall_tps_guard)});
    } else if (isLastStage() && isStageRoot() && !model_input.skip_run) {
        enqueueResultSendTask(
            {std::move(plan), std::move(pending_result), current_slot_, std::move(executor_collector)});
    }
    current_slot_ = (current_slot_ + 1) % slots_.size();

    if (model_input.shutdown) {
        stopped_ = true;
        if (isFirstStage() && isStageRoot()) {
            requestResultRecvStop();
        }
    }
    return absl::OkStatus();
}

void PPExecutor::enqueueResultSendTask(ResultSendTask task) {
    {
        std::lock_guard<std::mutex> lock(result_send_mutex_);
        result_send_task_.emplace(std::move(task));
    }
    result_send_cv_.notify_one();
}

void PPExecutor::resultSendLoop() {
    cuda_graph::setDevice(static_cast<int>(recv_stream_.device_index()));
    c10::InferenceMode inference_guard(true);
    while (true) {
        {
            std::unique_lock<std::mutex> lock(result_send_mutex_);
            result_send_cv_.wait(lock, [this] { return result_send_stop_ || result_send_task_.has_value(); });
            /** Drain queued results before stopping so the first stage can finish its pending receives. */
            if (!result_send_task_) {
                return;
            }
        }

        /** Keep waitLastRun() blocked until sampling state and send tickets are ready for the next batch. */
        processResultAndSend(*result_send_task_);

        {
            std::lock_guard<std::mutex> lock(result_send_mutex_);
            result_send_task_.reset();
        }
        result_send_cv_.notify_all();
    }
}

void PPExecutor::processResultAndSend(ResultSendTask& task) {
    {
        RTP_LLM_PROFILE_SCOPE("executor.pp.result_sync");
        last_compute_done_.synchronize();
    }
    const auto& plan   = task.plan;
    auto&       result = task.pending.result;
    batch_stream_processor_->finishExecutionResult(plan, task.pending);
    if (!plan.model_input.is_fake_stream && !plan.model_input.skip_lm_head) {
        advanceSamplingStates(plan.sampling_plan, result);
        eraseFailedRequestStates(result);
    }
    const auto start_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
    asyncSendExecutionResult(result, slots_[task.slot].execution_result_sends);
    task.executor_collector.send_result_us = autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
    reportStageMetrics(task.executor_collector);
}

void PPExecutor::stopResultSender() {
    if (result_send_thread_.joinable()) {
        {
            std::lock_guard<std::mutex> lock(result_send_mutex_);
            result_send_stop_ = true;
        }
        result_send_cv_.notify_one();
        result_send_thread_.join();
    }
}

void PPExecutor::enqueueResultRecvTask(ResultRecvTask task) {
    {
        std::lock_guard<std::mutex> lock(result_recv_mutex_);
        result_recv_tasks_.push_back(std::move(task));
    }
    result_recv_cv_.notify_one();
}

void PPExecutor::resultRecvLoop() {
    cuda_graph::setDevice(static_cast<int>(recv_stream_.device_index()));
    /** Inference mode is thread-local; dispatch updates tensors created by the engine thread. */
    c10::InferenceMode inference_guard(true);
    while (true) {
        {
            ResultRecvTask task;
            {
                std::unique_lock<std::mutex> lock(result_recv_mutex_);
                result_recv_cv_.wait(lock, [this] { return result_recv_stop_ || !result_recv_tasks_.empty(); });
                if (result_recv_tasks_.empty()) {
                    return;
                }
                task = std::move(result_recv_tasks_.front());
                result_recv_tasks_.pop_front();
                result_recv_processing_ = true;
            }
            receiveAndProcessResult(task.stream_groups, task.schedule_time_us, task.executor_collector);
            /** Drain must include stream and guard cleanup; destroy the task before marking it idle. */
        }
        {
            std::lock_guard<std::mutex> lock(result_recv_mutex_);
            result_recv_processing_ = false;
        }
        result_recv_cv_.notify_all();
    }
}

void PPExecutor::receiveAndProcessResult(const StreamGroups&             stream_groups,
                                         int64_t                         schedule_time_us,
                                         RtpLLMExecutorMetricsCollector& executor_collector) {
    /** Fake batches still complete the PP round trip before their result is discarded. */
    PPExecutionResult result;
    {
        RTP_LLM_PROFILE_SCOPE("executor.pp.recv_result");
        result = pp_serialization::deserializeExecutionResult(receiveObject(kPPResultGroupName));
    }
    if (stream_groups.isFakeStream()) {
        RTP_LLM_LOG_DEBUG("PP fake batch completed: dp_rank=%ld", parallelism_config_.dp_rank);
        return;
    }

    batch_stream_processor_->validateExecutionResult(stream_groups, result);

    /** Count input work and returned tokens before dispatch changes stream state. */
    StreamGroups::TokenCountsByPriority     token_counts_by_priority;
    RtpLLMSpeculativeEngineMetricsCollector sp_collector;
    collectTokenCounts(stream_groups, result, token_counts_by_priority, sp_collector);

    {
        RTP_LLM_PROFILE_SCOPE("executor.pp.dispatch_result");
        const auto start_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
        batch_stream_processor_->dispatchExecutionResult(stream_groups, result);
        executor_collector.dispatch_output_us = autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
    }

    executor_collector.sample_input_us = result.sample_input_us;
    reportResultMetrics(stream_groups, schedule_time_us, executor_collector, token_counts_by_priority, sp_collector);
}

void PPExecutor::drainPendingResults() {
    std::unique_lock<std::mutex> lock(result_recv_mutex_);
    result_recv_cv_.wait(lock, [this] { return result_recv_tasks_.empty() && !result_recv_processing_; });
}

void PPExecutor::requestResultRecvStop() {
    {
        std::lock_guard<std::mutex> lock(result_recv_mutex_);
        result_recv_stop_ = true;
    }
    result_recv_cv_.notify_all();
}

bool PPExecutor::updateEplbConfig(const EPLBConfig& config) {
    if (expert_balancer_) {
        return expert_balancer_->updateEplbConfig(config);
    }
    return true;
}

void PPExecutor::collectExecutorMetrics(const GptModelInputs&           model_input,
                                        bool                            is_decode,
                                        const torch::Tensor&            sequence_lengths,
                                        RtpLLMExecutorMetricsCollector& collector) const {
    if (!metrics_reporter_ || !isFirstStage() || !isStageRoot() || model_input.is_fake_stream) {
        return;
    }

    collector.generate_batch_size =
        sp_enabled_ && is_decode ? model_input.input_lengths.numel() : model_input.sequence_lengths.numel();
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
    /** Result rows were validated against allStreams before collecting metrics. */
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
        } else if (!result.request_errors[row].hasError()) {
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

void PPExecutor::reportResultMetrics(const StreamGroups&                        stream_groups,
                                     int64_t                                    schedule_time_us,
                                     RtpLLMExecutorMetricsCollector&            executor_collector,
                                     const StreamGroups::TokenCountsByPriority& token_counts_by_priority,
                                     RtpLLMSpeculativeEngineMetricsCollector&   sp_collector) {
    if (!metrics_reporter_ || warm_up_ || !isFirstStage() || !isStageRoot()) {
        return;
    }
    const int64_t batch_latency_us = autil::TimeUtility::currentTimeInMicroSeconds() - schedule_time_us;
    if (batch_latency_us <= 0) {
        return;
    }

    /** First-stage executor timings tagged pp_rank=0; other stages report their own via reportStageMetrics.
     * This deferred report also carries the result-time dispatch and the returned last-stage sampling. */
    metrics_reporter_->report<RtpLLMExecutorMetrics, RtpLLMExecutorMetricsCollector>(nullptr, &executor_collector);

    int64_t generate_tokens = 0;
    int64_t total_tokens    = 0;
    for (const auto& entry : token_counts_by_priority) {
        generate_tokens += entry.second.generate;
        total_tokens += entry.second.total;
    }
    RtpLLMTokenPSMetricsCollector tps_collector;
    tps_collector.addTokenSize(stream_groups.contextExecuteTokenSize(),
                               stream_groups.contextExecuteTokenSizeWithCache(),
                               generate_tokens,
                               total_tokens,
                               batch_latency_us);
    tps_collector.addTokenSizeByPriority(token_counts_by_priority, batch_latency_us);
    tps_reporter_.report(&tps_collector);
    wall_tps_reporter_.report(&tps_collector);

    if (sp_enabled_) {
        sp_collector.step_latency_us = batch_latency_us;
        /** PP does not collect SP phase timings here; retain their zero placeholders, as in MtpExecutor. */
        metrics_reporter_->report<RtpLLMSpeculativeEngineMetrics, RtpLLMSpeculativeEngineMetricsCollector>(
            nullptr, &sp_collector);
    }
}

void PPExecutor::reportStageMetrics(RtpLLMExecutorMetricsCollector& collector) {
    if (!metrics_reporter_ || warm_up_ || !isStageRoot()) {
        return;
    }
    metrics_reporter_->report<RtpLLMExecutorMetrics, RtpLLMExecutorMetricsCollector>(nullptr, &collector);
}

}  // namespace rtp_llm
