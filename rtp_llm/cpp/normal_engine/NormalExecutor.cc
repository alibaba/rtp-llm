#include "rtp_llm/cpp/normal_engine/NormalExecutor.h"
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cuda_graph/cuda_graph_device_shims.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include <cstdlib>
#include <memory>
#include <string>
#include "rtp_llm/cpp/utils/StatusUtil.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#include "rtp_llm/cpp/models/ModelTypes.h"
#include "rtp_llm/cpp/models/PyWrappedModel.h"
#include "rtp_llm/cpp/models/Sampler.h"
#include "rtp_llm/cpp/config/ModelConfig.h"
#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorFactory.h"

using namespace std;

namespace rtp_llm {

NormalExecutor::ModelFactory NormalExecutor::test_model_factory = nullptr;

NormalExecutor::~NormalExecutor() {
    cudaProfilerEnd();
}

NormalExecutor::NormalExecutor(const EngineInitParams&                params,
                               const std::shared_ptr<KVCacheManager>& cache_manager,
                               bool                                   warm_up,
                               bool                                   is_propose,
                               int                                    propose_model_index,
                               MlaOpsType                             mla_ops_type,
                               int32_t                                kv_cache_group_num,
                               const std::vector<int32_t>&            kv_cache_layer_to_group):
    Executor(),
    cache_manager_(cache_manager),
    warm_up_(warm_up),
    use_all_gather_(params.moe_config.use_all_gather && !params.moe_config.use_deepep_low_latency),
    metrics_reporter_(params.metrics_reporter),
    tps_reporter_(MetricsLoopReporter<RtpLLMTokenPSMetrics, RtpLLMTokenPSMetricsCollector>(metrics_reporter_)),
    is_propose_(is_propose),
    propose_model_index_(propose_model_index),
    dispatch_runner_(cuda_graph::graphGetStreamFromPool(true)) {
    enable_detail_log_  = params.profiling_debug_logging_config.enable_detail_log;
    tp_rank_            = params.parallelism_config.tp_rank;
    parallelism_config_ = params.parallelism_config;
    RTP_LLM_LOG_INFO("enable_detail_log_ = %d, tp_rank_ = %d", enable_detail_log_, tp_rank_);

    if (params.eplb_config.enable_eplb() && params.model_config_.moe_style != 0) {
        // use first moe layer weight as moe weight type
        int         first_moe_layer = params.model_config_.moe_layer_index.front();
        const auto& moe_kernel      = params.gpt_weights.layers[first_moe_layer].ffn_weights.moe_gate_weight->kernel;
        auto        moe_weight_type = torchDTypeToDataType(moe_kernel.dtype());
        bool        is_gated_activation = params.model_config_.isGatedActivation();
        auto        moe_inter_size      = is_gated_activation ? moe_kernel.size(1) / 2 : moe_kernel.size(1);

        expert_balancer_ = make_shared<ExpertBalancer>(params.model_config_.expert_num,
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
                                                       metrics_reporter_,
                                                       params.eplb_config);
    }

    sampler_.reset(new Sampler(SamplerInitParams{}));

    GptModelInitParams model_init_params(
        {params.gpt_weights,
         genModelDescription(params.model_config_, params.parallelism_config, params.eplb_config, params.moe_config),
         cache_manager ?
             std::make_optional(is_propose_ ? cache_manager->getMTPModuleCacheLayerLayout(propose_model_index_) :
                                              cache_manager->getMainModelCacheLayerLayout()) :
             std::nullopt,
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
         params.model_config_.attn_config.tokens_per_block,
         params.model_config_.attn_config.kernel_tokens_per_block,
         kv_cache_group_num,
         kv_cache_layer_to_group,
         cache_manager});

    if (params.ffn_disaggregate_config.enable_ffn_disaggregate) {
        RTP_LLM_LOG_INFO("using ffn as service");
        enable_ffn_disaggregate_ = true;
    }
    if (!params.py_model.is_none()) {
        RTP_LLM_LOG_INFO("init executor with python model");
        model_.reset(new PyWrappedModel(model_init_params, params.py_model));
    } else if (test_model_factory) {
        RTP_LLM_LOG_INFO("init executor with test model factory");
        model_ = test_model_factory(model_init_params);
    } else {
        RTP_LLM_LOG_WARNING("py_model is None — model will not be initialized (test mode)");
    }

    // when warmup, cache manager maybe nullptr
    const auto& cache_config = cache_manager ?
                                   (is_propose_ ? cache_manager->getMTPModuleCacheConfig(propose_model_index_) :
                                                  cache_manager->cacheConfig()) :
                                   CacheConfig();

    batch_stream_processor_.reset(new NormalBatchStreamProcessor(
        params.model_config_, params.pd_sep_config, params.profiling_debug_logging_config, cache_config, warm_up_));
    LogitsProcessorFactory::init(params.model_config_.ckpt_path, params.sp_config.tree_decode_config);
    cudaProfilerBegin();
}

absl::Status NormalExecutor::process(const std::list<GenerateStreamPtr>& streams) {
    StreamGroups                   stream_groups(streams);
    RtpLLMExecutorMetricsCollector executor_collector;
    RtpLLMTokenPSMetricsCollector  tps_collector;
    GptModelInputs                 model_input;
    GptModelOutputs                model_output;
    SamplerOutput                  sampler_output;
    RTP_LLM_PROFILE_FUNCTION();

    // Release H2D source tensors held from the previous step. Safe here
    // because all H2D copies queued via .to(kCUDA, non_blocking=true) before
    // the previous tpSyncModelInputs have been consumed by that broadcast's
    // packing (the GPU path issues fusedCopy on the same default stream, so
    // ordering is preserved).
    if (useMtpDeviceInput()) {
        buffer_holder_.release();
    }

    // Cap outstanding stream-async bookkeeping to one step unless
    // RTP_LLM_DROP_BROAD_SYNC=1 explicitly drops the front-loaded sync.
    // NormalAsyncDeviceState covers the next decode step's last sample token
    // and seq_len for batch-1 streams, so DROP_BROAD_SYNC can avoid waiting for
    // the worker's host stream->update() on that path.
    if (useStreamAsync() && !useDropBroadSync()) {
        RTP_LLM_PROFILE_SCOPE_DYNAMIC("executor.wait_prev_dispatch(stream_count=%zu)", streams.size());
        dispatch_runner_.sync(cuda_graph::graphGetCurrentStream());
    }

    {
        RTP_LLM_PROFILE_SCOPE("executor.gather_model_input");
        int64_t start_time_us      = autil::TimeUtility::currentTimeInMicroSeconds();
        auto    model_input_status = batch_stream_processor_->gatherModelInput(stream_groups);
        RETURN_IF_STATUS_OR_ERROR(model_input_status);
        model_input                              = std::move(model_input_status.value());
        executor_collector.gather_model_input_us = autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
    }
    {
        RTP_LLM_PROFILE_SCOPE("executor.tp_sync_input");
        int64_t start_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
        model_input.skip_run  = streams.empty() && !enable_ffn_disaggregate_;

        // Push metadata tensors to CUDA before tpSyncModelInputs so the
        // broadcast goes through the GPU packed-buffer path (single
        // execBroadcast) instead of the CPU path (execBroadcastCpu, plus
        // per-tensor unpack/memcpy on non-root ranks). Without this gate,
        // tpSyncModelInputs sees CPU tensors and incurs noticeable kernel-
        // launch overhead per broadcast.
        ensureMtpModelInputsOnCuda(model_input, "process.before_tp_sync");

        tpSyncModelInputs(model_input, parallelism_config_);
        if (model_input.skip_run) {
            return absl::OkStatus();
        }

        // Defensive: tpSyncModelInputs honors the root device bitmap so
        // non-root ranks should already have CUDA buffers, but a few
        // device-bit-uncovered fields (sequence_lengths_plus_1 in particular)
        // can still come back as CPU. Keep the post-sync ensure to land
        // every metadata tensor on CUDA before model_->forward consumes them.
        ensureMtpModelInputsOnCuda(model_input, "process.after_tp_sync");

        executor_collector.tp_sync_input_us = autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
    }

    // make sure last model input is released before forward
    model_->releaseBuffers();

    {
        // update kv cache
        if (model_input.kv_cache_update_mapping.defined()) {
            RTP_LLM_PROFILE_SCOPE("executor.kv_cache_update");
            cache_manager_->blockBatchCopy(model_input.kv_cache_update_mapping);
        }
    }
    {
        bool force = tp_rank_ == 0 && enable_detail_log_;
        if (force) {
            RTP_LLM_LOG_INFO("model_input: %s", model_input.debugString(force).c_str());
        } else {
            RTP_LLM_LOG_TRACE("model_input: %s", model_input.debugString(force).c_str());
        }
        RTP_LLM_PROFILE_SCOPE_DYNAMIC("executor.model_forward(ctx_batch=%zu,gen_batch=%zu,tokens=%zu,max_seq=%zu)",
                                      stream_groups.totalContextBatchSize(),
                                      stream_groups.totalDecodeBatchSize(),
                                      stream_groups.modelExecuteTokenSize(),
                                      stream_groups.maxSeqLen());
        int64_t start_time_us               = autil::TimeUtility::currentTimeInMicroSeconds();
        model_output                        = std::move(model_->forward(model_input));
        executor_collector.model_forward_us = autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
    }
    if (expert_balancer_) {
        int64_t start_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
        expert_balancer_->stepForward(*model_, executor_collector);
        executor_collector.eplb_step_latency_us = autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
    }

    if (tp_rank_ > 0 || warm_up_ || streams.size() == 0) {
        model_->releaseBuffers();
        return absl::OkStatus();
    }
    {
        RTP_LLM_PROFILE_SCOPE("executor.sampler_forward");
        int64_t start_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
        CHECK_AND_RETURN_REF(sampler_input,
                             batch_stream_processor_->gatherSamplerInput(stream_groups, model_input, model_output));
        sampler_output = std::move(sampler_->forward(sampler_input));
        RTP_LLM_LOG_DEBUG("sampler forward done");
        executor_collector.sample_input_us = autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
    }

    // Stream-async dispatch is opt-in (RTP_LLM_STREAM_ASYNC=1). The user's
    // contract here: a single process() call carries either context (prefill)
    // streams or generate (decode) streams, never both. Async dispatch is most
    // useful for decode where the next iteration immediately schedules another
    // forward and can hide the .cpu() D2H + per-stream update cost behind it.
    // We don't do async dispatch for prefill (no overlap benefit — the next
    // step's gather is dominated by per-stream init, and async would add
    // worker startup overhead without corresponding savings).
    const bool is_decode_only = stream_groups.totalContextBatchSize() == 0 && stream_groups.totalDecodeBatchSize() > 0;
    if (useStreamAsync() && is_decode_only) {
        RTP_LLM_PROFILE_SCOPE("executor.dispatch_output(stream_async)");
        int64_t start_time_us = autil::TimeUtility::currentTimeInMicroSeconds();

        // Sampler outputs (token_ids/success) live on the main stream; record
        // the event at the earliest valid point so the worker can wait via
        // cudaStreamWaitEvent before the pinned D2H staging instead of
        // CPU-syncing.
        auto sampler_event = std::make_shared<torch::Event>(cuda_graph::makeGraphEvent());
        sampler_event->record(cuda_graph::graphGetCurrentStream());

        // Metrics and KV release happen on the main thread before launching;
        // the worker captures only the model/sampler outputs it needs to
        // dispatch. The dispatch_output_us metric now measures the launch
        // path, not the worker; the worker's actual dispatch time is
        // observable via the async_runner.thread profile scope.
        executor_collector.dispatch_output_us = autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
        reportMetrics(stream_groups, executor_collector, tps_collector);

        model_->releaseBuffers();

        return dispatchOutputAsync(
            stream_groups, std::move(model_output), std::move(sampler_output), std::move(sampler_event));
    }

    {
        RTP_LLM_PROFILE_SCOPE("executor.dispatch_output");
        int64_t      start_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
        MergedOutput merge_outputs{std::move(model_output), std::move(sampler_output)};
        publishNormalDeviceState(stream_groups, merge_outputs.sampler_output);
        auto result                           = batch_stream_processor_->dispatch(stream_groups, merge_outputs);
        executor_collector.dispatch_output_us = autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
        reportMetrics(stream_groups, executor_collector, tps_collector);

        model_->releaseBuffers();

        return result;
    }
}

void NormalExecutor::reportMetrics(const StreamGroups&             stream_groups,
                                   RtpLLMExecutorMetricsCollector& executor_collector,
                                   RtpLLMTokenPSMetricsCollector&  tps_collector) {
    if (tp_rank_ > 0) {
        return;
    }
    if (metrics_reporter_) {
        executor_collector.context_batch_size  = stream_groups.totalContextBatchSize();
        executor_collector.generate_batch_size = stream_groups.totalDecodeBatchSize();
        executor_collector.execute_token_size  = stream_groups.modelExecuteTokenSize();
        executor_collector.max_seq_len         = stream_groups.maxSeqLen();
        if (executor_collector.context_batch_size != 0) {
            executor_collector.context_batch_size_when_has_context  = executor_collector.context_batch_size;
            executor_collector.generate_batch_size_when_has_context = executor_collector.generate_batch_size;
            executor_collector.execute_token_size_when_has_context  = executor_collector.execute_token_size;
            executor_collector.max_seq_len_when_has_context         = executor_collector.max_seq_len;
        }
        metrics_reporter_->report<RtpLLMExecutorMetrics, RtpLLMExecutorMetricsCollector>(nullptr, &executor_collector);

        tps_collector.context_tps  = stream_groups.modelExecuteTokenSize() - stream_groups.totalDecodeBatchSize();
        tps_collector.generate_tps = stream_groups.totalDecodeBatchSize();
        tps_collector.total_tps    = stream_groups.modelExecuteTokenSize();
        tps_reporter_.report(&tps_collector);
    }
}

bool NormalExecutor::updateEplbConfig(const EPLBConfig& config) {
    if (expert_balancer_) {
        return expert_balancer_->updateEplbConfig(config);
    }
    return true;
}

bool NormalExecutor::useStreamAsync() const {
    // Process-wide gate, evaluated once. Mirrors MtpExecutor::useStreamAsync
    // and intentionally shares the same env var so a single launcher knob
    // toggles both the MTP and non-MTP async dispatch paths.
    static const bool enabled = []() {
        const char* env = std::getenv("RTP_LLM_STREAM_ASYNC");
        bool        on  = (env != nullptr && std::string(env) == "1");
        RTP_LLM_LOG_INFO("[normal-stream-async] RTP_LLM_STREAM_ASYNC=%s -> useStreamAsync=%d",
                         env ? env : "(unset)",
                         static_cast<int>(on));
        return on;
    }();
    return enabled;
}

bool NormalExecutor::useDropBroadSync() const {
    static const bool enabled = []() {
        const char* env = std::getenv("RTP_LLM_DROP_BROAD_SYNC");
        bool        on  = (env != nullptr && std::string(env) == "1");
        RTP_LLM_LOG_INFO("[normal-drop-broad-sync] RTP_LLM_DROP_BROAD_SYNC=%s -> enabled=%d",
                         env ? env : "(unset)",
                         static_cast<int>(on));
        return on;
    }();
    return enabled;
}

bool NormalExecutor::useMtpDeviceInput() const {
    static const bool enabled = []() {
        const char* env = std::getenv("RTP_LLM_DEVICE_INPUT");
        bool        on  = (env != nullptr && std::string(env) == "1");
        RTP_LLM_LOG_INFO(
            "[normal-device-input] RTP_LLM_DEVICE_INPUT=%s -> enabled=%d", env ? env : "(unset)", static_cast<int>(on));
        return on;
    }();
    return enabled;
}

bool NormalExecutor::checkMtpDeviceInput() const {
    static const bool enabled = []() {
        const char* env = std::getenv("RTP_LLM_DEVICE_INPUT_CHECK");
        bool        on  = (env != nullptr && std::string(env) == "1");
        RTP_LLM_LOG_INFO("[normal-device-input] RTP_LLM_DEVICE_INPUT_CHECK=%s -> enabled=%d",
                         env ? env : "(unset)",
                         static_cast<int>(on));
        return on;
    }();
    return enabled;
}

void NormalExecutor::ensureMtpModelInputsOnCuda(GptModelInputs& model_input, const char* tag) {
    if (!useMtpDeviceInput()) {
        return;
    }

    auto to_cuda = [this, tag](torch::Tensor& tensor, const char* name) {
        if (!tensor.defined() || tensor.is_cuda()) {
            return;
        }
        if (tensor.numel() == 0) {
            tensor = torch::empty(tensor.sizes(), torch::TensorOptions(tensor.dtype()).device(torch::kCUDA));
            return;
        }
        if (!tensor.is_pinned()) {
            // Non-pinned CPU H2D forces a synchronous copy, defeating the
            // whole point of pre-staging tensors to CUDA. Fall back rather
            // than abort, but warn loudly so we can fix the producer.
            RTP_LLM_LOG_WARNING(
                "[normal-device-input] %s.%s is CPU but not pinned; H2D falls back to blocking copy", tag, name);
            tensor = tensor.to(torch::kCUDA);
            return;
        }
        // non_blocking=true requires the source tensor to outlive the copy;
        // the holder keeps a reference until the next process() iteration
        // releases it (after the broadcast has consumed the tensor).
        buffer_holder_.hold(tensor);
        tensor = tensor.to(torch::kCUDA, /*non_blocking=*/true);
    };

    to_cuda(model_input.combo_tokens, "combo_tokens");
    to_cuda(model_input.input_lengths, "input_lengths");
    to_cuda(model_input.sequence_lengths, "sequence_lengths");
    to_cuda(model_input.prefix_lengths, "prefix_lengths");
    to_cuda(model_input.sequence_lengths_plus_1, "sequence_lengths_plus_1");
    to_cuda(model_input.lm_output_indexes, "lm_output_indexes");
    checkMtpModelInputsOnCuda(model_input, tag);
}

void NormalExecutor::checkMtpModelInputsOnCuda(const GptModelInputs& model_input, const char* tag) const {
    if (!checkMtpDeviceInput()) {
        return;
    }
    auto check = [tag](const torch::Tensor& tensor, const char* name) {
        if (!tensor.defined()) {
            return;
        }
        RTP_LLM_CHECK_WITH_INFO(tensor.is_cuda(),
                                "[normal-device-input] %s.%s expected CUDA tensor, got device=%s numel=%ld",
                                tag,
                                name,
                                tensor.device().str().c_str(),
                                tensor.numel());
    };
    check(model_input.combo_tokens, "combo_tokens");
    check(model_input.input_lengths, "input_lengths");
    check(model_input.sequence_lengths, "sequence_lengths");
    check(model_input.prefix_lengths, "prefix_lengths");
    check(model_input.sequence_lengths_plus_1, "sequence_lengths_plus_1");
    check(model_input.lm_output_indexes, "lm_output_indexes");
}

void NormalExecutor::publishNormalDeviceState(const StreamGroups& stream_groups, const SamplerOutput& sampler_output) {
    RTP_LLM_PROFILE_SCOPE("executor.publish_normal_device_state");

    auto all_streams = stream_groups.allStreams();
    if (all_streams.empty()) {
        return;
    }

    auto clear_states = [&all_streams]() {
        for (auto& stream : all_streams) {
            stream->setNormalAsyncDeviceState(GenerateStream::NormalAsyncDeviceState{});
        }
    };

    const auto& token_ids = sampler_output.token_ids;
    if (!token_ids.defined() || token_ids.numel() == 0 || token_ids.dim() < 1) {
        RTP_LLM_LOG_WARNING("[normal-device-state] skip publish: token_ids undefined/empty");
        clear_states();
        return;
    }

    for (const auto& stream : all_streams) {
        if (stream->currentBatchSize() != 1 || stream->nextBatchSize() != 1) {
            RTP_LLM_LOG_WARNING(
                "[normal-device-state] skip publish: only batch-1 decode is supported, stream=%ld cur_bs=%d next_bs=%d",
                stream->streamId(),
                stream->currentBatchSize(),
                stream->nextBatchSize());
            clear_states();
            return;
        }
    }

    const auto    cuda_i32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    torch::Tensor token_ids_gpu =
        (token_ids.is_cuda() && token_ids.scalar_type() == torch::kInt32) ? token_ids : token_ids.to(cuda_i32);
    const int64_t token_rows = token_ids_gpu.size(0);
    if (token_rows < static_cast<int64_t>(all_streams.size())) {
        RTP_LLM_LOG_WARNING(
            "[normal-device-state] skip publish: token_ids rows=%ld stream_count=%zu", token_rows, all_streams.size());
        clear_states();
        return;
    }

    int64_t batch_idx_out = 0;
    for (auto& stream : all_streams) {
        torch::Tensor last_sample_token_gpu;
        if (token_ids_gpu.dim() == 1) {
            last_sample_token_gpu = token_ids_gpu.narrow(0, batch_idx_out, 1).to(torch::kInt32);
        } else {
            const int64_t last_col = token_ids_gpu.size(-1) - 1;
            last_sample_token_gpu =
                token_ids_gpu.narrow(0, batch_idx_out, 1).select(-1, last_col).reshape({1}).to(torch::kInt32);
        }

        torch::Tensor cur_seq_len_gpu;
        const auto&   prev_next_seq_len = stream->getNormalAsyncDeviceState().next_seq_len_gpu;
        if (prev_next_seq_len.defined() && prev_next_seq_len.is_cuda()) {
            cur_seq_len_gpu = prev_next_seq_len;
        } else {
            cur_seq_len_gpu = torch::full({1}, static_cast<int64_t>(stream->seqLength()), cuda_i32);
        }

        GenerateStream::NormalAsyncDeviceState state;
        state.last_sample_token_gpu = std::move(last_sample_token_gpu);
        state.next_seq_len_gpu      = (cur_seq_len_gpu + 1).to(torch::kInt32);
        stream->setNormalAsyncDeviceState(std::move(state));
        batch_idx_out += 1;
    }
}

absl::Status NormalExecutor::dispatchOutputAsync(const StreamGroups&           stream_groups,
                                                 GptModelOutputs               model_output,
                                                 SamplerOutput                 sampler_output,
                                                 std::shared_ptr<torch::Event> sampler_event) {
    RTP_LLM_PROFILE_SCOPE("executor.dispatch_output_async");

    publishNormalDeviceState(stream_groups, sampler_output);

    auto* processor           = batch_stream_processor_.get();
    auto  stream_groups_copy  = stream_groups;
    auto  model_output_copy   = std::move(model_output);
    auto  sampler_output_copy = std::move(sampler_output);

    dispatch_runner_.launch([processor,
                             stream_groups_copy  = std::move(stream_groups_copy),
                             model_output_copy   = std::move(model_output_copy),
                             sampler_output_copy = std::move(sampler_output_copy),
                             sampler_event]() mutable {
        RTP_LLM_PROFILE_SCOPE("executor.dispatch_output_worker");

        // cudaStreamWaitEvent (NOT cudaEventSynchronize) — the worker stream
        // queues a wait; dispatch then stages token_ids/success through pinned
        // D2H and synchronizes only this worker stream. Main thread is already
        // off doing the next step's gather.
        if (sampler_event) {
            sampler_event->block(cuda_graph::graphGetCurrentStream());
        }

        auto status =
            processor->dispatch(stream_groups_copy, {std::move(model_output_copy), std::move(sampler_output_copy)});
        if (!status.ok()) {
            RTP_LLM_LOG_ERROR("[normal-stream-async] dispatch (worker) failed: %s", status.ToString().c_str());
        }
    });

    return absl::OkStatus();
}

}  // namespace rtp_llm
