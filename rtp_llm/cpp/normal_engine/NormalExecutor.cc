#include "rtp_llm/cpp/normal_engine/NormalExecutor.h"
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cuda_graph/cuda_graph_device_shims.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include <cstdlib>
#include <memory>
#include <string>
#include "rtp_llm/cpp/utils/StatusUtil.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#include "rtp_llm/cpp/models/ModelInputsLogger.h"
#include "rtp_llm/cpp/models/ModelTypes.h"
#include "rtp_llm/cpp/models/PyWrappedModel.h"
#include "rtp_llm/cpp/models/Sampler.h"
#include "rtp_llm/cpp/config/ModelConfig.h"
#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorFactory.h"
#include "rtp_llm/cpp/distribute/RpcCpuTpBroadcaster.h"

using namespace std;

namespace rtp_llm {

namespace {

bool readEnvFlagOnce(const char* env_name, const char* log_tag, const char* label) {
    const char* env = std::getenv(env_name);
    const bool  on  = (env != nullptr && std::string(env) == "1");
    RTP_LLM_LOG_INFO("[%s] %s=%s -> %s=%d", log_tag, env_name, env ? env : "(unset)", label, static_cast<int>(on));
    return on;
}

int readEnvIntOnce(const char* env_name, int default_value, const char* log_tag) {
    const char* env   = std::getenv(env_name);
    int         value = default_value;
    if (env != nullptr) {
        value = std::atoi(env);
        if (value <= 0) {
            value = default_value;
        }
    }
    RTP_LLM_LOG_INFO("[%s] %s=%s -> %d", log_tag, env_name, env ? env : "(unset)", value);
    return value;
}

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

}  // namespace

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
                               const std::vector<int32_t>&            kv_cache_layer_to_group,
                               std::function<void()>                  profile_step_start,
                               std::function<void()>                  profile_step_finish):
    Executor(),
    cache_manager_(cache_manager),
    role_type_(params.pd_sep_config.role_type),
    warm_up_(warm_up),
    use_all_gather_(params.moe_config.use_all_gather && !params.moe_config.use_deepep_low_latency),
    metrics_reporter_(params.metrics_reporter),
    tps_reporter_(MetricsLoopReporter<RtpLLMTokenPSMetrics, RtpLLMTokenPSMetricsCollector>(
        params.parallelism_config.tp_rank == 0 && !warm_up ? metrics_reporter_ : nullptr)),
    wall_tps_reporter_(WallClockMetricsLoopReporter<RtpLLMWallClockTokenPSMetrics, RtpLLMTokenPSMetricsCollector>(
        params.parallelism_config.tp_rank == 0 && !warm_up ? metrics_reporter_ : nullptr)),
    is_propose_(is_propose),
    propose_model_index_(propose_model_index),
    profile_step_start_(std::move(profile_step_start)),
    profile_step_finish_(std::move(profile_step_finish)),
    dispatch_runner_(cuda_graph::graphGetStreamFromPool(true)) {
    enable_detail_log_  = params.profiling_debug_logging_config.enable_detail_log;
    tp_rank_            = params.parallelism_config.tp_rank;
    parallelism_config_ = params.parallelism_config;
    RTP_LLM_LOG_INFO("enable_detail_log_ = %d, tp_rank_ = %d", enable_detail_log_, tp_rank_);
    if (params.profiling_debug_logging_config.enable_model_inputs_log) {
        model_inputs_logger_ =
            std::make_shared<ModelInputsLogger>(params.parallelism_config.world_rank,
                                                params.profiling_debug_logging_config.log_file_backup_count,
                                                metrics_reporter_);
    }

    const bool enable_cross_node_cpu_tp_broadcast =
        readEnvFlagOnce("RTP_LLM_CROSS_NODE_CPU_TP_BROADCAST", "NormalExecutor", "cross_node_cpu_tp_broadcast");
    if (enable_cross_node_cpu_tp_broadcast && params.parallelism_config.tp_size > 1
        && params.parallelism_config.tp_size > params.parallelism_config.local_world_size) {
        const int timeout_ms = readEnvIntOnce("RTP_LLM_CPU_TP_BROADCAST_TIMEOUT_MS", 30000, "NormalExecutor");
        RpcCpuTpBroadcaster::instance().initialize(static_cast<int>(params.parallelism_config.tp_rank),
                                                   static_cast<int>(params.parallelism_config.tp_size),
                                                   static_cast<int>(params.parallelism_config.dp_rank),
                                                   static_cast<int>(params.parallelism_config.world_size),
                                                   params.runtime_config.worker_grpc_addrs,
                                                   timeout_ms);
    }

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

    // CacheConfig is the single source of truth for tokens_per_block /
    // kernel_tokens_per_block. DSV4 promotes seq_size_per_block to a 256-token
    // physical block while attn_config still reflects the 64-token CLI flag, so
    // sourcing from attn_config makes the fused compressor index the state
    // block_table with the wrong stride. During warmup the cache_manager is
    // null — use zero-initialized block geometry so PyWrappedModel's >0
    // check catches mis-propagation (CacheConfig default is 1, not 0).
    // when warmup, cache manager maybe nullptr
    CacheConfig warmup_sentinel;
    warmup_sentinel.seq_size_per_block        = 0;
    warmup_sentinel.kernel_seq_size_per_block = 0;
    const auto& cache_config                  = cache_manager ?
                                                    (is_propose_ ? cache_manager->getMTPModuleCacheConfig(propose_model_index_) :
                                                                   cache_manager->cacheConfig()) :
                                                    warmup_sentinel;

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
         static_cast<size_t>(cache_config.seq_size_per_block),
         static_cast<size_t>(cache_config.kernel_seq_size_per_block),
         kv_cache_group_num,
         kv_cache_layer_to_group,
         cache_manager,
         params.model_config_.hc_mult});

    if (params.ffn_disaggregate_config.enable_ffn_disaggregate) {
        RTP_LLM_LOG_INFO("using ffn as service");
        enable_ffn_disaggregate_ = true;
    }
    if (!params.py_model.is_none()) {
        RTP_LLM_LOG_INFO("init executor with python model");
        model_.reset(new PyWrappedModel(model_init_params, params.py_model, false, false, {}, model_inputs_logger_));
    } else if (test_model_factory) {
        RTP_LLM_LOG_INFO("init executor with test model factory");
        model_ = test_model_factory(model_init_params);
    } else {
        RTP_LLM_LOG_WARNING("py_model is None — model will not be initialized (test mode)");
    }

    batch_stream_processor_.reset(new NormalBatchStreamProcessor(
        params.model_config_, params.pd_sep_config, params.profiling_debug_logging_config, cache_config, warm_up_));
    LogitsProcessorFactory::init(
        params.model_config_.ckpt_path, params.sp_config.tree_decode_config, params.grammar_config);
    cudaProfilerBegin();
}

absl::Status NormalExecutor::process(const std::list<GenerateStreamPtr>& streams, int64_t schedule_time_us) {
    const int64_t process_start_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
    if (schedule_time_us <= 0) {
        schedule_time_us = process_start_time_us;
    }
    RtpLLMExecutorMetricsCollector executor_collector;
    RtpLLMTokenPSMetricsCollector  tps_collector;
    auto                           tps_active_guard =
        tps_reporter_.makeActiveGuard(metrics_reporter_ && tp_rank_ == 0 && !warm_up_ && !streams.empty());
    auto wall_tps_active_guard =
        wall_tps_reporter_.makeActiveGuard(metrics_reporter_ && tp_rank_ == 0 && !warm_up_ && !streams.empty());
    GptModelInputs  model_input;
    GptModelOutputs model_output;
    SamplerOutput   sampler_output;
    RTP_LLM_PROFILE_FUNCTION();
    // Cap outstanding stream-async bookkeeping to one step unless DROP_BROAD_SYNC is on.
    // Still sync when gatherModelInput lacks NormalAsyncDeviceState; host
    // token/seq_len fallbacks race the previous worker.
    bool worker_synced = false;
    if (useStreamAsync() && !useDropBroadSync()) {
        RTP_LLM_PROFILE_SCOPE_DYNAMIC("executor.wait_prev_dispatch(stream_count=%zu)", streams.size());
        dispatch_runner_.sync(cuda_graph::graphGetCurrentStream());
        worker_synced = true;
    }

    StreamGroups stream_groups(streams);
    prepareGrpcNormalDeviceState(stream_groups);

    if (useStreamAsync() && useDropBroadSync() && !gatherCanUseDeviceState(stream_groups)) {
        RTP_LLM_PROFILE_SCOPE_DYNAMIC("executor.wait_prev_dispatch(stream_count=%zu)", streams.size());
        dispatch_runner_.sync(cuda_graph::graphGetCurrentStream());
        worker_synced = true;
        // Rebuild StreamGroups after waiting: cached maxSeqLen/batch sizes can
        // be stale while the previous worker mutates GenerateStream host state.
        stream_groups = StreamGroups(streams);
        prepareGrpcNormalDeviceState(stream_groups);
    }

    {
        RTP_LLM_PROFILE_SCOPE("executor.gather_model_input");
        int64_t start_time_us      = autil::TimeUtility::currentTimeInMicroSeconds();
        auto    model_input_status = batch_stream_processor_->gatherModelInput(stream_groups, buffer_holder_);
        RETURN_IF_STATUS_OR_ERROR(model_input_status);
        model_input                              = std::move(model_input_status.value());
        executor_collector.gather_model_input_us = autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
    }
    {
        RTP_LLM_PROFILE_SCOPE("executor.tp_sync_input");
        int64_t start_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
        model_input.skip_run  = streams.empty() && !enable_ffn_disaggregate_;

        // Move metadata to CUDA before tpSyncModelInputs so broadcasts use the
        // GPU packed-buffer path instead of CPU execBroadcastCpu/unpack loops.
        ensureModelInputsOnCuda(model_input, "process.before_tp_sync");

        tpSyncModelInputs(model_input, parallelism_config_);
        if (model_input.skip_run) {
            return absl::OkStatus();
        }

        // Post-sync guard for fields not covered by the root device bitmap
        // (notably sequence_lengths_plus_1) before model_->forward.
        ensureModelInputsOnCuda(model_input, "process.after_tp_sync");

        executor_collector.tp_sync_input_us = autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
    }

    if (profile_step_start_) {
        profile_step_start_();
    }

    // make sure last model input is released before forward
    // TensorHolder release point (NormalExecutor): after current TP sync has
    // consumed model-input H2D staging, advance the one-extra-round hold window
    // for model-input sources and previous sampler-input host tensors.
    buffer_holder_.release();
    // PyWrappedModel TensorHolder release point for the previous model forward.
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
        if (tp_rank_ == 0 && stream_groups.totalContextBatchSize() > 0) {
            RTP_LLM_LOG_INFO("prefill_batch_begin: ctx_batch=%zu gen_batch=%zu total_tokens=%zu max_seq=%zu",
                             stream_groups.totalContextBatchSize(),
                             stream_groups.totalDecodeBatchSize(),
                             stream_groups.modelExecuteTokenSize(),
                             stream_groups.maxSeqLen());
        }
        int64_t start_time_us               = autil::TimeUtility::currentTimeInMicroSeconds();
        model_output                        = std::move(model_->forward(model_input));
        executor_collector.model_forward_us = autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
        if (tp_rank_ == 0 && stream_groups.totalContextBatchSize() > 0) {
            auto        now_us = autil::TimeUtility::currentTimeInMicroSeconds();
            std::string details;
            for (auto& s : stream_groups.contextStreams()) {
                char    buf[256];
                int64_t compute_ms = (now_us - s->beginTimeUs()) / 1000 - s->getTimeInfo().wait_time_us / 1000;
                snprintf(
                    buf,
                    sizeof(buf),
                    "{id=%ld trace_id=%s input=%d prefix=%d reuse=%d ctx=%d grp=%ld/%d tokens=%d timeout=%ld compute_ms=%ld global_start_time_us=%ld} ",
                    s->streamId(),
                    s->traceId().empty() ? "-" : s->traceId().c_str(),
                    s->inputLength(),
                    s->prefixLength(),
                    s->reuseLength(),
                    s->contextLength(),
                    s->groupId(),
                    s->groupSize(),
                    s->currentExecuteTokenSize(),
                    s->getTimeoutMs(),
                    compute_ms,
                    s->generateInput()->global_start_time_us);
                details += buf;
            }
            RTP_LLM_LOG_INFO(
                "prefill_batch_end: ctx_batch=%zu gen_batch=%zu total_tokens=%zu max_seq=%zu forward_us=%ld streams=[%s]",
                stream_groups.totalContextBatchSize(),
                stream_groups.totalDecodeBatchSize(),
                stream_groups.modelExecuteTokenSize(),
                stream_groups.maxSeqLen(),
                executor_collector.model_forward_us,
                details.c_str());
        }
    }
    if (expert_balancer_) {
        int64_t start_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
        expert_balancer_->stepForward(*model_, executor_collector);
        executor_collector.eplb_step_latency_us = autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
    }

    if (tp_rank_ > 0 || warm_up_ || streams.size() == 0) {
        cudaSyncAndCheck();
        model_->releaseBuffers();
        if (profile_step_finish_) {
            profile_step_finish_();
        }
        return absl::OkStatus();
    }

    {
        RTP_LLM_PROFILE_SCOPE("executor.sampler_forward");
        int64_t start_time_us = autil::TimeUtility::currentTimeInMicroSeconds();

        // Sampler input still reads CPU stream state mutated by the previous
        // worker. DROP_BROAD_SYNC therefore needs this narrow sync unless an
        // earlier host-fallback gather already waited.
        if (useStreamAsync() && useDropBroadSync() && !worker_synced) {
            RTP_LLM_PROFILE_SCOPE_DYNAMIC("executor.wait_prev_dispatch_pre_sampler(stream_count=%zu)", streams.size());
            dispatch_runner_.sync(cuda_graph::graphGetCurrentStream());
            worker_synced = true;
            // Rebuild after waiting so sampler buffers use the current seqLength
            // instead of appending one column behind.
            stream_groups = StreamGroups(streams);
        }

        CHECK_AND_RETURN_REF(sampler_input,
                             batch_stream_processor_->gatherSamplerInput(stream_groups, model_input, model_output));
        holdSamplerInputHostBuffers(buffer_holder_, sampler_input);
        sampler_output = std::move(sampler_->forward(sampler_input));
        RTP_LLM_LOG_DEBUG("sampler forward done");
        executor_collector.sample_input_us = autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
    }

    // Stream-async dispatch is opt-in and only useful for decode-only batches:
    // the next iteration can overlap forward prep with worker D2H/update work.
    // Prefill gets no overlap benefit and would only add worker startup cost.
    const bool is_decode_only = stream_groups.totalContextBatchSize() == 0 && stream_groups.totalDecodeBatchSize() > 0;
    if (useStreamAsync() && is_decode_only) {
        RTP_LLM_PROFILE_SCOPE("executor.dispatch_output(stream_async)");
        int64_t start_time_us = autil::TimeUtility::currentTimeInMicroSeconds();

        // Record as soon as sampler outputs are valid so the worker waits with
        // cudaStreamWaitEvent before pinned D2H staging.
        auto sampler_event = std::make_shared<torch::Event>(cuda_graph::makeGraphEvent());
        sampler_event->record(cuda_graph::graphGetCurrentStream());

        // Metrics and KV release stay on the main thread; dispatch_output_us
        // measures only async launch/enqueue overhead.
        executor_collector.dispatch_output_us = autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
        int64_t tps_execute_time_us           = autil::TimeUtility::currentTimeInMicroSeconds() - schedule_time_us;
        if (tps_execute_time_us <= 0) {
            tps_execute_time_us = autil::TimeUtility::currentTimeInMicroSeconds() - process_start_time_us;
        }
        reportMetrics(stream_groups, executor_collector, tps_collector, tps_execute_time_us);

        return dispatchOutputAsync(stream_groups,
                                   std::move(model_output),
                                   std::move(sampler_output),
                                   std::move(sampler_event),
                                   profile_step_finish_);
    }

    {
        RTP_LLM_PROFILE_SCOPE("executor.dispatch_output");
        int64_t      start_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
        MergedOutput merge_outputs{std::move(model_output), std::move(sampler_output)};
        publishNormalDeviceState(stream_groups, merge_outputs.sampler_output);
        auto result                           = batch_stream_processor_->dispatch(stream_groups, merge_outputs);
        executor_collector.dispatch_output_us = autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
        int64_t tps_execute_time_us           = autil::TimeUtility::currentTimeInMicroSeconds() - schedule_time_us;
        if (tps_execute_time_us <= 0) {
            tps_execute_time_us = autil::TimeUtility::currentTimeInMicroSeconds() - process_start_time_us;
        }
        reportMetrics(stream_groups, executor_collector, tps_collector, tps_execute_time_us);

        if (profile_step_finish_) {
            profile_step_finish_();
        }
        return result;
    }
}

void NormalExecutor::reportMetrics(const StreamGroups&             stream_groups,
                                   RtpLLMExecutorMetricsCollector& executor_collector,
                                   RtpLLMTokenPSMetricsCollector&  tps_collector,
                                   int64_t                         tps_execute_time_us) {
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

        tps_collector.addTokenSize(stream_groups.contextExecuteTokenSize(),
                                   stream_groups.contextExecuteTokenSizeWithCache(),
                                   stream_groups.totalDecodeBatchSize(),
                                   stream_groups.modelExecuteTokenSize(),
                                   tps_execute_time_us);
        tps_reporter_.report(&tps_collector);
        wall_tps_reporter_.report(&tps_collector);
    }
}

bool NormalExecutor::updateEplbConfig(const EPLBConfig& config) {
    if (expert_balancer_) {
        return expert_balancer_->updateEplbConfig(config);
    }
    return true;
}

bool NormalExecutor::useStreamAsync() const {
    static const bool enabled = []() {
        return readEnvFlagOnce("RTP_LLM_STREAM_ASYNC", "normal-stream-async", "useStreamAsync");
    }();
    return enabled;
}

bool NormalExecutor::useDropBroadSync() const {
    static const bool enabled = []() {
        return readEnvFlagOnce("RTP_LLM_DROP_BROAD_SYNC", "normal-drop-broad-sync", "enabled");
    }();
    return enabled;
}

bool NormalExecutor::useDeviceInput() const {
    static const bool enabled = []() {
        return readEnvFlagOnce("RTP_LLM_DEVICE_INPUT", "normal-device-input", "enabled");
    }();
    return enabled;
}

bool NormalExecutor::checkDeviceInput() const {
    static const bool enabled = []() {
        return readEnvFlagOnce("RTP_LLM_DEVICE_INPUT_CHECK", "normal-device-input", "enabled");
    }();
    return enabled;
}

void NormalExecutor::ensureModelInputsOnCuda(GptModelInputs& model_input, const char* tag) {
    if (!useDeviceInput()) {
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
        buffer_holder_.hold_host(tensor);
        tensor = tensor.to(torch::kCUDA, /*non_blocking=*/true);
    };

    to_cuda(model_input.combo_tokens, "combo_tokens");
    to_cuda(model_input.input_lengths, "input_lengths");
    to_cuda(model_input.sequence_lengths, "sequence_lengths");
    to_cuda(model_input.prefix_lengths, "prefix_lengths");
    to_cuda(model_input.sequence_lengths_plus_1, "sequence_lengths_plus_1");
    to_cuda(model_input.lm_output_indexes, "lm_output_indexes");
    checkModelInputsOnCuda(model_input, tag);
}

void NormalExecutor::checkModelInputsOnCuda(const GptModelInputs& model_input, const char* tag) const {
    if (!checkDeviceInput()) {
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

bool NormalExecutor::gatherCanUseDeviceState(const StreamGroups& stream_groups) const {
    // Decode-only batch is the only shape the device-state path supports.
    if (stream_groups.totalContextBatchSize() != 0 || stream_groups.totalDecodeBatchSize() == 0) {
        return false;
    }
    for (const auto& stream : stream_groups.decodeStreams()) {
        // Use config-only proxies for the batch-1 check. currentBatchSize() reads
        // outputTokenLen() → seqLength() which itself races with the worker we're
        // trying to decide whether to skip syncing.
        if (stream->hasNumBeams() || stream->numReturnSequences() > 1) {
            return false;
        }
        bool has_blocking_stateful_processor = false;
        for (const auto& processor : stream->getAllLogitsProcessorPtr()) {
            if (processor != nullptr && processor->isStateful() && !processor->supportsNormalAsyncDeviceState()) {
                has_blocking_stateful_processor = true;
                break;
            }
        }
        // NormalAsyncDeviceState mirrors token ids and sequence length. Stateful
        // processors that cannot publish their own device-side next-step state
        // still need the old wait before the next sampler consumes logits.
        if (has_blocking_stateful_processor && stream->hasPendingAsyncBookkeeping()) {
            return false;
        }
        const auto& state = stream->getNormalAsyncDeviceState();
        if (!state.last_sample_token_gpu.defined() || !state.last_sample_token_gpu.is_cuda()
            || !state.next_seq_len_gpu.defined() || !state.next_seq_len_gpu.is_cuda()) {
            return false;
        }
    }
    return true;
}

void NormalExecutor::prepareGrpcNormalDeviceState(const StreamGroups& stream_groups) {
    if (role_type_ != RoleType::DECODE || stream_groups.totalContextBatchSize() != 0
        || stream_groups.totalDecodeBatchSize() == 0) {
        return;
    }

    const auto cuda_i32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    for (const auto& stream : stream_groups.decodeStreams()) {
        if (!stream->consumeGrpcNormalDeviceStatePending()) {
            continue;
        }
        const auto& state = stream->getNormalAsyncDeviceState();
        if (state.last_sample_token_gpu.defined() && state.last_sample_token_gpu.is_cuda()
            && state.next_seq_len_gpu.defined() && state.next_seq_len_gpu.is_cuda()) {
            continue;
        }
        if (stream->outputTokenLen() != 1) {
            continue;
        }
        const auto current_batch_size = stream->currentBatchSize();
        if (current_batch_size != 1) {
            RTP_LLM_LOG_WARNING("[normal-device-state] skip grpc publish: stream=%ld cur_bs=%d",
                                stream->streamId(),
                                current_batch_size);
            continue;
        }

        auto current_tokens = stream->currentExecuteTokens(0);
        if (current_tokens.size() != 1) {
            RTP_LLM_LOG_WARNING("[normal-device-state] skip grpc publish: stream=%ld token_count=%zu",
                                stream->streamId(),
                                current_tokens.size());
            continue;
        }

        const auto seq_length = stream->seqLength();
        stream->setNormalAsyncDeviceState(GenerateStream::NormalAsyncDeviceState{
            .epoch                 = 0,
            .last_sample_token_gpu = torch::full({1}, static_cast<int64_t>(current_tokens[0]), cuda_i32),
            .next_seq_len_gpu      = torch::full({1}, static_cast<int64_t>(seq_length), cuda_i32),
            .last_real_seq_len     = seq_length,
            .next_real_seq_len     = seq_length,
        });
    }
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

        // Mirror next_seq_len_gpu on host for the next iter's scheduler.
        // Fall back to live seqLength only on first publish (no prior worker).
        const auto& prev_state = stream->getNormalAsyncDeviceState();
        const int   cur_real_seq_len =
            prev_state.next_real_seq_len > 0 ? prev_state.next_real_seq_len : stream->seqLength();

        torch::Tensor cur_seq_len_gpu;
        const auto&   prev_next_seq_len = prev_state.next_seq_len_gpu;
        if (prev_next_seq_len.defined() && prev_next_seq_len.is_cuda()) {
            cur_seq_len_gpu = prev_next_seq_len;
        } else {
            cur_seq_len_gpu = torch::full({1}, static_cast<int64_t>(cur_real_seq_len), cuda_i32);
        }

        for (const auto& processor : stream->getAllLogitsProcessorPtr()) {
            if (processor != nullptr && processor->supportsNormalAsyncDeviceState()) {
                processor->prepareNormalAsyncUpdate(last_sample_token_gpu, 1);
            }
        }

        GenerateStream::NormalAsyncDeviceState state;
        state.last_sample_token_gpu = std::move(last_sample_token_gpu);
        state.next_seq_len_gpu      = (cur_seq_len_gpu + 1).to(torch::kInt32);
        state.last_real_seq_len     = cur_real_seq_len;
        state.next_real_seq_len     = cur_real_seq_len + 1;
        stream->setNormalAsyncDeviceState(std::move(state));
        batch_idx_out += 1;
    }
}

absl::Status NormalExecutor::dispatchOutputAsync(const StreamGroups&           stream_groups,
                                                 GptModelOutputs               model_output,
                                                 SamplerOutput                 sampler_output,
                                                 std::shared_ptr<torch::Event> sampler_event,
                                                 std::function<void()>         profile_step_finish) {
    RTP_LLM_PROFILE_SCOPE("executor.dispatch_output_async");

    publishNormalDeviceState(stream_groups, sampler_output);

    auto* processor           = batch_stream_processor_.get();
    auto  stream_groups_copy  = stream_groups;
    auto  model_output_copy   = std::move(model_output);
    auto  sampler_output_copy = std::move(sampler_output);

    // Claim each stream's KV resource before handing work to the runner so
    // releaseResource defers while the worker reads/updates those blocks.
    auto streams_for_inc = stream_groups_copy.allStreams();
    for (auto& s : streams_for_inc) {
        s->incPendingAsyncBookkeeping();
    }

    dispatch_runner_.launch([processor,
                             stream_groups_copy  = std::move(stream_groups_copy),
                             model_output_copy   = std::move(model_output_copy),
                             sampler_output_copy = std::move(sampler_output_copy),
                             sampler_event]() mutable {
        RTP_LLM_PROFILE_SCOPE("executor.dispatch_output_worker");

        auto worker_streams = stream_groups_copy.allStreams();

        // RAII: every captured stream is decremented exactly once, and the
        // value capture keeps GenerateStreamPtr alive until after dec runs.
        auto dec_guard = std::shared_ptr<void>(nullptr, [worker_streams](void*) {
            for (auto& s : worker_streams) {
                s->decPendingAsyncBookkeepingAndMaybeRelease();
            }
        });

        // Queue a stream wait, then do pinned D2H on the worker stream while
        // the main thread continues to the next gather.
        if (sampler_event) {
            sampler_event->block(cuda_graph::graphGetCurrentStream());
        }

        auto status =
            processor->dispatch(stream_groups_copy, {std::move(model_output_copy), std::move(sampler_output_copy)});
        if (!status.ok()) {
            RTP_LLM_LOG_ERROR("[normal-stream-async] dispatch (worker) failed: %s", status.ToString().c_str());
        }
        // dec_guard destructs here, dec'ing each stream's pending count.
    });

    // Kineto requires profiler enable/disable on the same thread. Keep finish
    // on the engine loop thread even when output dispatch runs asynchronously.
    if (profile_step_finish) {
        profile_step_finish();
    }

    return absl::OkStatus();
}

}  // namespace rtp_llm
