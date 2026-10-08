#include "rtp_llm/cpp/normal_engine/speculative/MtpExecutor.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/engine_base/EngineBase.h"
#include "rtp_llm/cpp/engine_base/stream/StreamGroups.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/cuda_graph/cuda_graph_device_shims.h"
#include "rtp_llm/cpp/utils/StatusUtil.h"
#include "rtp_llm/cpp/engine_base/schedulers/FIFOScheduler.h"
#include "rtp_llm/cpp/engine_base/schedulers/BatchDecodeScheduler.h"
#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/engine_base/system_prompt/SystemPromptConstructor.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/StringUtil.h"
#include "rtp_llm/cpp/models/PyWrappedModel.h"
#include "rtp_llm/cpp/models/logits_processor/SpecLogitsProcessor.h"
#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorFactory.h"
#include "rtp_llm/cpp/models/logits_processor/TreeLogitsProcessor.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#include <algorithm>
#include <cmath>
#if USING_CUDA
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include "rtp_llm/models_py/bindings/cuda/kernels/mtp_target_verify_prepare.h"
#endif
#include "autil/TimeUtility.h"
#include <limits>
#include <cstdlib>
#include <memory>
#include <thread>
#include <random>
#include <string>
#include <vector>
#include <atomic>

namespace rtp_llm {

namespace {

// Called on the bookkeeping producer itself, before its CPU claims are removed.
// No stream mutex is held while draining an exceptional GPU producer.
std::exception_ptr finishProtectedKvBookkeeping(const std::list<GenerateStreamPtr>& streams) {
    const torch::Stream producer_stream = cuda_graph::graphGetCurrentStream();
    std::exception_ptr  failure;
    try {
        auto done = std::make_shared<torch::Event>(cuda_graph::makeGraphEvent());
        done->record(producer_stream);
        const auto wait = [done, producer_stream] {
            cuda_graph::GraphStreamGuard guard(cuda_graph::toGraphStream(producer_stream));
            done->synchronize();
        };
        for (const auto& stream : streams) {
            if (stream->kvExecutionProtected()) {
                stream->publishKvCompletionFence(producer_stream.hash(), wait);
                stream->setPendingSwapDoneEvent(std::static_pointer_cast<void>(done));
            }
        }
    } catch (...) {
        failure = std::current_exception();
        try {
            producer_stream.synchronize();
        } catch (...) {
            failure = std::current_exception();
            for (const auto& stream : streams) {
                if (stream->kvExecutionProtected()) {
                    stream->quarantineKvExecution();
                }
            }
        }
    }
    for (const auto& stream : streams) {
        try {
            stream->decPendingAsyncBookkeepingAndMaybeRelease();
        } catch (...) {
            if (!failure) {
                failure = std::current_exception();
            }
        }
    }
    return failure;
}

struct CachedEnvFlag {
    const char* env_name;
    const char* log_tag;
    const char* label;
    bool        on;
    std::string value;
};

CachedEnvFlag cacheEnvFlag(const char* env_name, const char* log_tag, const char* label) {
    const char* env = std::getenv(env_name);
    return CachedEnvFlag{env_name, log_tag, label, env != nullptr && std::string(env) == "1", env ? env : "(unset)"};
}

void logCachedEnvFlag(const CachedEnvFlag& flag) {
    RTP_LLM_LOG_INFO(
        "[%s] %s=%s -> %s=%d", flag.log_tag, flag.env_name, flag.value.c_str(), flag.label, static_cast<int>(flag.on));
}

const CachedEnvFlag kMtpDeviceInputFlag = cacheEnvFlag("RTP_LLM_DEVICE_INPUT", "mtp-device-input", "enabled");
const CachedEnvFlag kMtpDeviceInputCheckFlag =
    cacheEnvFlag("RTP_LLM_DEVICE_INPUT_CHECK", "mtp-device-input", "enabled");
const CachedEnvFlag kStreamAsyncFlag = cacheEnvFlag("RTP_LLM_STREAM_ASYNC", "stream-async", "useStreamAsync");
const CachedEnvFlag kAsyncDeviceStateFlag =
    cacheEnvFlag("RTP_LLM_MTP_ASYNC_DEVICE_STATE", "async-device-state", "enabled");
const CachedEnvFlag kDropBroadSyncFlag = cacheEnvFlag("RTP_LLM_DROP_BROAD_SYNC", "drop-broad-sync", "enabled");
const CachedEnvFlag kAsyncPrepareFlag  = cacheEnvFlag("RTP_LLM_MTP_ASYNC_PREPARE", "async-prepare", "enabled");
const bool          kDisableSpPrefillCudaGraphByEnv = []() {
    const char* env = std::getenv("DISABLE_SP_PREFILL_CUDA_GRAPH");
    return env != nullptr && std::string(env) == "1";
}();
const bool kForceSpPrefillCudaGraphByEnv = []() {
    const char* env = std::getenv("RTP_LLM_FORCE_SP_PREFILL_CUDA_GRAPH");
    return env != nullptr && std::string(env) == "1";
}();

torch::Tensor
scatterCompactRows(const torch::Tensor& compact, const torch::Tensor& compact_to_dense, int64_t dense_rows) {
    RTP_LLM_CHECK_WITH_INFO(compact.defined() && compact.dim() >= 1, "compact scatter requires a row-major tensor");
    RTP_LLM_CHECK_WITH_INFO(compact_to_dense.defined() && compact_to_dense.is_cuda()
                                && compact_to_dense.scalar_type() == torch::kInt32
                                && compact_to_dense.numel() == compact.size(0),
                            "compact scatter mapping must be CUDA int32 with one entry per row");
    RTP_LLM_CHECK_WITH_INFO(dense_rows >= compact.size(0), "compact scatter dense capacity is too small");
    auto sizes = compact.sizes().vec();
    sizes[0]   = dense_rows;
    auto dense = torch::zeros(sizes, compact.options());
    dense.index_copy_(0, compact_to_dense.to(torch::kLong), compact);
    return dense;
}

bool isMegaMoeStrategy(const std::string& moe_strategy) {
    return moe_strategy == "mega_moe" || moe_strategy == "mega_moe_se" || moe_strategy == "mega_moe_fused"
           || moe_strategy == "mega_moe_fp8" || moe_strategy == "mega_moe_fp8_se" || moe_strategy == "mega_moe_nvfp4";
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

bool hasSpecLogitsProcessor(const std::list<GenerateStreamPtr>& streams) {
    for (const auto& stream : streams) {
        for (const auto& processor : stream->getAllLogitsProcessorPtr()) {
            if (std::dynamic_pointer_cast<SpecLogitsProcessor>(processor) != nullptr) {
                return true;
            }
        }
    }
    return false;
}

bool hasUnsupportedMtpStatefulLogitsProcessor(const std::list<GenerateStreamPtr>& streams) {
    for (const auto& stream : streams) {
        for (const auto& processor : stream->getAllLogitsProcessorPtr()) {
            if (processor == nullptr || std::dynamic_pointer_cast<SpecLogitsProcessor>(processor) != nullptr) {
                continue;
            }
            if (processor->isStateful() || std::dynamic_pointer_cast<TreeLogitsProcessor>(processor) != nullptr) {
                return true;
            }
        }
    }
    return false;
}

void recordSpecTensorUseOnCurrentStream(const torch::Tensor& tensor) {
#if USING_CUDA
    if (tensor.defined() && tensor.is_cuda()) {
        c10::cuda::CUDACachingAllocator::recordStream(tensor.storage().data_ptr(),
                                                      at::cuda::getCurrentCUDAStream(tensor.device().index()));
    }
#else
    (void)tensor;
#endif
}

torch::Tensor toCudaWithHostHold(const torch::Tensor& tensor, TensorHolder& holder) {
    if (!tensor.defined() || tensor.is_cuda()) {
        return tensor;
    }
    if (tensor.numel() == 0) {
        return torch::empty(tensor.sizes(), torch::TensorOptions(tensor.dtype()).device(torch::kCUDA));
    }
    holder.hold_host(tensor);
    return tensor.to(torch::kCUDA, /*non_blocking=*/true);
}

torch::Tensor toCudaInt32WithHostHold(const torch::Tensor& tensor, TensorHolder& holder) {
    if (!tensor.defined()) {
        return tensor;
    }
    auto cuda_i32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    if (tensor.is_cuda() && tensor.scalar_type() == torch::kInt32) {
        return tensor;
    }
    if (tensor.numel() == 0) {
        return torch::empty(tensor.sizes(), cuda_i32);
    }
    holder.hold_host(tensor);
    return tensor.to(cuda_i32, /*non_blocking=*/true);
}

void applySpecLogitsAcceptLenCap(const SamplerInputs&                   sampler_input,
                                 const SamplerOutput&                   target_sampler_output,
                                 speculative::SpeculativeSamplerOutput& output,
                                 int64_t                                batch_size,
                                 int64_t                                propose_step) {
    if (!sampler_input.spec_cap_gpu.defined()) {
        return;
    }
    RTP_LLM_CHECK_WITH_INFO(output.accept_len.defined() && output.accept_len.is_cuda(),
                            "spec logits cap requires CUDA accept_len");

    if (sampler_input.spec_mask_ready_event) {
        sampler_input.spec_mask_ready_event->block(cuda_graph::graphGetCurrentStream());
    }
    recordSpecTensorUseOnCurrentStream(sampler_input.spec_cap_gpu);
    auto cap_gpu      = sampler_input.spec_cap_gpu.to(output.accept_len.options());
    auto cap_plus_one = cap_gpu + 1;
    output.accept_len = torch::minimum(output.accept_len, cap_plus_one);

    RTP_LLM_CHECK_WITH_INFO(output.accept_tokens.defined() && output.accept_tokens.is_cuda(),
                            "spec logits cap requires CUDA accept_tokens");
    RTP_LLM_CHECK_WITH_INFO(target_sampler_output.token_ids.defined(),
                            "spec logits cap requires target sampler token_ids");
    auto target_token_ids = target_sampler_output.token_ids;
    if (!target_token_ids.is_cuda()) {
        target_token_ids = target_token_ids.to(output.accept_tokens.device(), /*non_blocking=*/true);
    }
    const int64_t token_stride  = target_token_ids.size(1);
    auto          target_tokens = target_token_ids.reshape({batch_size, propose_step + 1, token_stride})
                             .select(2, token_stride - 1)
                             .to(output.accept_tokens.options());
    auto cap_index =
        sampler_input.spec_cap_gpu.to(torch::TensorOptions().device(output.accept_tokens.device()).dtype(torch::kLong));
    auto replacement = target_tokens.gather(1, cap_index.unsqueeze(1));

    auto cols = torch::arange(propose_step + 1,
                              torch::TensorOptions().device(output.accept_tokens.device()).dtype(torch::kLong))
                    .unsqueeze(0)
                    .expand({batch_size, propose_step + 1});
    auto replace_mask = (cap_gpu < propose_step).unsqueeze(1) & (output.accept_len > cap_gpu).unsqueeze(1)
                        & (cols == cap_index.unsqueeze(1));
    output.accept_tokens =
        torch::where(replace_mask, replacement.expand({batch_size, propose_step + 1}), output.accept_tokens);

    output.accept_tokens_cpu = output.accept_tokens.to(torch::kCPU, /*non_blocking=*/true);
    output.accept_len_cpu    = output.accept_len.to(torch::kCPU, /*non_blocking=*/true);
    output.transfer_done_event->record(cuda_graph::graphGetCurrentStream());
    // The spec artifact is read by logits masking before sampling and by cap
    // application here. Recording after cap keeps future artifact pools from
    // reusing mask/cap storage before the sampler stream has consumed both.
    if (sampler_input.spec_mask_consumed_event) {
        sampler_input.spec_mask_consumed_event->record(cuda_graph::graphGetCurrentStream());
    }
}

bool isCpContextRequest(const ParallelismConfig& parallelism_config, const GptModelInputs& input) {
    return parallelism_config.prefill_cp_config.is_enabled() && input.input_lengths.defined()
           && input.sequence_lengths.defined() && input.input_lengths.size(0) != input.sequence_lengths.size(0);
}

}  // namespace

void MtpExecutor::notifyStop() {
    stop_requested_.store(true, std::memory_order_release);
}

bool MtpExecutor::shouldSkipFakeStreamForStop(const GptModelInputs& model_input, const char* phase) const {
    if (!stop_requested_.load(std::memory_order_acquire) || !model_input.is_fake_stream) {
        return false;
    }
    RTP_LLM_LOG_INFO("[MTP decode] skip fake stream during shutdown before %s", phase);
    return true;
}

bool MtpExecutor::isTpRank0() const {
    return tp_rank_ == 0;
}

bool MtpExecutor::maybeOverrideLastHiddenWithMtpBuffer(GptModelInputs&       model_input,
                                                       ModelBase&            source,
                                                       MtpHiddenStatesLayout layout,
                                                       int64_t               requested_rows) {
    if (!model_input.combo_tokens.defined() || model_input.combo_tokens.numel() == 0) {
        return false;
    }
    RTP_LLM_CHECK_WITH_INFO(layout == MtpHiddenStatesLayout::GLOBAL || layout == MtpHiddenStatesLayout::CP_LOCAL,
                            "MTP hidden buffer override requires GLOBAL or CP_LOCAL layout, got %s",
                            mtpHiddenStatesLayoutName(layout));
    RTP_LLM_CHECK_WITH_INFO(layout != MtpHiddenStatesLayout::GLOBAL || requested_rows == -1,
                            "GLOBAL MTP hidden rows are derived from combo_tokens, got explicit rows=%ld",
                            requested_rows);
    RTP_LLM_CHECK_WITH_INFO(layout != MtpHiddenStatesLayout::CP_LOCAL || requested_rows == -1 || requested_rows > 0,
                            "CP-local MTP hidden rows must be positive or -1, got %ld",
                            requested_rows);
    const auto mtp_hidden_rows =
        layout == MtpHiddenStatesLayout::CP_LOCAL ? requested_rows : model_input.combo_tokens.numel();
    auto pre_hc = source.getMtpTargetHiddenStates(mtp_hidden_rows);
    if (!pre_hc.defined() || pre_hc.numel() == 0) {
        RTP_LLM_CHECK_WITH_INFO(layout != MtpHiddenStatesLayout::CP_LOCAL || model_input.last_hidden_states.defined(),
                                "CP MTP hidden buffer must contain local rows before draft prefill");
        return false;
    }
    model_input.setLastHiddenStates(pre_hc, layout);
    return true;
}

void MtpExecutor::maybeOverrideLastHiddenWithMtpBuffer(GptModelOutputs& model_output, ModelBase& source) {
    if (!model_output.all_hidden_states.defined() || model_output.all_hidden_states.size(0) == 0) {
        return;
    }
    auto pre_hc = source.getMtpTargetHiddenStates(model_output.all_hidden_states.size(0));
    if (!pre_hc.defined() || pre_hc.numel() == 0) {
        return;
    }
    model_output.all_hidden_states = pre_hc;
}

bool MtpExecutor::useDeviceInput() const {
    static const bool logged = []() {
        logCachedEnvFlag(kMtpDeviceInputFlag);
        return true;
    }();
    (void)logged;
    return kMtpDeviceInputFlag.on;
}

bool MtpExecutor::checkDeviceInput() const {
    static const bool logged = []() {
        logCachedEnvFlag(kMtpDeviceInputCheckFlag);
        return true;
    }();
    (void)logged;
    return kMtpDeviceInputCheckFlag.on;
}

void MtpExecutor::ensureModelInputsOnCuda(GptModelInputs& model_input, const char* tag) {
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
            RTP_LLM_LOG_WARNING(
                "[mtp-device-input] %s.%s is CPU but not pinned; H2D falls back to blocking copy", tag, name);
            tensor = tensor.to(torch::kCUDA);
            return;
        }
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

void MtpExecutor::checkModelInputsOnCuda(const GptModelInputs& model_input, const char* tag) const {
    if (!checkDeviceInput()) {
        return;
    }
    auto check = [tag](const torch::Tensor& tensor, const char* name) {
        if (!tensor.defined()) {
            return;
        }
        RTP_LLM_CHECK_WITH_INFO(tensor.is_cuda(),
                                "[mtp-device-input] %s.%s expected CUDA tensor, got device=%s numel=%ld",
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
    RTP_LLM_LOG_DEBUG("[mtp-device-input] %s metadata tensors are CUDA", tag);
}

MtpExecutor::AcceptLenMetricsSnapshot MtpExecutor::consumePendingAcceptLenMetrics() {
    AcceptLenMetricsSnapshot snapshot;
    if (!metrics_accept_len_sum_cpu_.defined()) {
        return snapshot;
    }

    RTP_LLM_PROFILE_SCOPE("executor.mtp.decode_step(consume_accept_len_metrics)");
    if (metrics_accept_len_ready_event_) {
        // The current step's tiny D2H was started after rejection sampling,
        // overlapping draft commit. Consume it in this same step so an idle
        // executor cannot strand the last report or leak it into a later run.
        metrics_accept_len_ready_event_->synchronize();
    }

    snapshot.total_accept_len        = metrics_accept_len_sum_cpu_.item<int64_t>();
    snapshot.total_stream_num        = metrics_accept_len_stream_num_;
    snapshot.total_propose_token_num = metrics_accept_len_propose_token_num_;
    snapshot.valid                   = true;

    metrics_accept_len_sum_gpu_ = torch::Tensor();
    metrics_accept_len_sum_cpu_ = torch::Tensor();
    metrics_accept_len_ready_event_.reset();
    metrics_accept_len_stream_num_        = 0;
    metrics_accept_len_propose_token_num_ = 0;
    return snapshot;
}

void MtpExecutor::stageAcceptLenMetrics(const torch::Tensor& accept_len,
                                        torch::Event&        accept_len_ready_event,
                                        size_t               stream_count) {
    if (!accept_len.defined()) {
        return;
    }

    RTP_LLM_PROFILE_SCOPE("executor.mtp.decode_step(stage_accept_len_metrics)");
    metrics_accept_len_stream_num_        = static_cast<int64_t>(stream_count);
    metrics_accept_len_propose_token_num_ = static_cast<int64_t>(stream_count * propose_step_);

    if (!accept_len.is_cuda()) {
        metrics_accept_len_sum_gpu_ = torch::Tensor();
        metrics_accept_len_sum_cpu_ = accept_len.to(torch::kInt64).sum().reshape({1}).pin_memory();
        metrics_accept_len_ready_event_.reset();
        return;
    }

    cuda_graph::GraphStreamGuard stream_guard(cuda_graph::toGraphStream(collect_metrics_stream_));
    accept_len_ready_event.block(collect_metrics_stream_);
    metrics_accept_len_sum_gpu_ = accept_len.to(torch::kInt64).sum().reshape({1});
    metrics_accept_len_sum_cpu_ =
        torch::empty({1}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU).pinned_memory(true));
    metrics_accept_len_sum_cpu_.copy_(metrics_accept_len_sum_gpu_, /*non_blocking=*/true);
    metrics_accept_len_ready_event_ = std::make_shared<torch::Event>(cuda_graph::makeGraphEvent());
    metrics_accept_len_ready_event_->record(collect_metrics_stream_);
}

void MtpExecutor::maybePrintModelInput(const GptModelInputs& model_input, const std::string& prefix) const {
    bool force = tp_rank_ == 0 && enable_detail_log_;
    if (force) {
        RTP_LLM_LOG_INFO("%s model_input: %s", prefix.c_str(), model_input.debugString(force).c_str());
    } else {
        RTP_LLM_LOG_DEBUG("%s model_input: %s", prefix.c_str(), model_input.debugString(force).c_str());
    }
}

static std::shared_ptr<NormalGenerateStream> makeFakeStream(int                    max_new_tokens,
                                                            size_t                 reserved_blocks,
                                                            const ModelConfig&     model_config,
                                                            const RuntimeConfig&   runtime_config,
                                                            const ResourceContext& resource_context) {
    std::shared_ptr<GenerateInput> fake_input   = std::make_shared<GenerateInput>();
    fake_input->input_ids                       = torch::zeros({1}, torch::kInt32);
    fake_input->generate_config                 = std::make_shared<GenerateConfig>();
    fake_input->generate_config->max_new_tokens = max_new_tokens;
    fake_input->generate_config->top_k          = 1;
    fake_input->begin_time_us                   = autil::TimeUtility::currentTimeInMicroSeconds();
    fake_input->fake_query                      = true;

    auto fake_stream = std::make_shared<NormalGenerateStream>(
        fake_input, model_config, runtime_config, resource_context, nullptr, max_new_tokens);
    fake_stream->setIsFakeStream(true);
    fake_stream->setMetricsReporter(nullptr);
    fake_stream->fakeInitKVBlock(reserved_blocks);

    return fake_stream;
}

static SpeculativeExecutorStreamOutputPtr
makeFakeSPOutputBuffer(DataType data_type, size_t hidden_size, size_t vocab_size, size_t propose_step) {
    auto sp_buffer = std::make_shared<SpeculativeExecutorStreamOutput>();

    auto fake_hidden_states = torch::zeros(
        {1, (int64_t)hidden_size}, torch::TensorOptions().dtype(dataTypeToTorchType(data_type)).device(torch::kCUDA));
    auto fake_probs =
        torch::zeros({1, (int64_t)vocab_size}, torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));
    const auto cuda_i32      = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    sp_buffer->propose_step  = propose_step;
    sp_buffer->all_probs     = fake_probs;
    sp_buffer->tokens        = torch::zeros({1, 2}, torch::kInt32);
    sp_buffer->hidden_states = fake_hidden_states;
    // Pre-allocate device mirrors so the hot path never triggers a pageable
    // H2D + sync via ensureSpOutputTokenGpuMirrors().
    sp_buffer->target_token_gpu   = torch::zeros({1}, cuda_i32);
    sp_buffer->propose_tokens_gpu = torch::zeros({1}, cuda_i32);

    return sp_buffer;
}

static void ensureSpOutputTokenGpuMirrors(const SpeculativeExecutorStreamOutputPtr& sp_buffer, bool needs_proposal) {
    // PDFUSION and P2P paths publish device mirrors directly. Legacy/test
    // initialization may only carry the two CPU tokens; materialize those once
    // here rather than teaching the generic GenerateStream about MTP state.
    if (!sp_buffer || !sp_buffer->tokens.defined() || sp_buffer->tokens.numel() < 2) {
        return;
    }
    const auto cuda_i32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    if (!sp_buffer->target_token_gpu.defined() || !sp_buffer->target_token_gpu.is_cuda()) {
        sp_buffer->target_token_gpu = sp_buffer->tokens.reshape({-1}).narrow(0, 0, 1).to(cuda_i32);
    }
    // DSpARK publishes no recurrent MTP proposal. Its -1 CPU sentinel must
    // not trigger a pageable H2D copy and stream sync on every decode round.
    if (needs_proposal && (!sp_buffer->propose_tokens_gpu.defined() || !sp_buffer->propose_tokens_gpu.is_cuda())) {
        sp_buffer->propose_tokens_gpu = sp_buffer->tokens.reshape({-1}).narrow(0, 1, 1).to(cuda_i32);
    }
}

GenerateStreamPtr MtpExecutor::createMinFakePrefillStream(int                    max_new_tokens,
                                                          const ModelConfig&     model_config,
                                                          const RuntimeConfig&   runtime_config,
                                                          const ResourceContext& resource_context) {
    return makeFakeStream(max_new_tokens, 1, model_config, runtime_config, resource_context);
}

GenerateStreamPtr MtpExecutor::createMinFakeDecodeStream(int                    max_new_tokens,
                                                         const ModelConfig&     model_config,
                                                         const ModelConfig&     draft_model_config,
                                                         const RuntimeConfig&   runtime_config,
                                                         const ResourceContext& resource_context,
                                                         int                    vocab_size) {
    auto fake_stream =
        makeFakeStream(max_new_tokens, 1 + max_new_tokens, model_config, runtime_config, resource_context);

    // SPOutputBuffer stores the recurrent state consumed by draft decode.
    // Derive its width from the draft contract: target and draft widths may
    // differ (for example, EAGLE3 has a 3H target handoff and H recurrence).
    auto sp_buffer = makeFakeSPOutputBuffer(draft_model_config.data_type,
                                            draft_model_config.hidden_size * draft_model_config.hc_mult,
                                            vocab_size,
                                            max_new_tokens);

    auto new_tokens = torch::zeros({1, 1}, torch::kInt32);

    StreamUpdateInfo update_info{new_tokens,
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
    fake_stream->setSPOutputBuffer(sp_buffer);
    auto seq_len = fake_stream->seqLength();

    // set device state
    auto int32_gpu          = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    auto accept_len_gpu     = torch::ones({1}, int32_gpu);
    auto accept_tokens_gpu  = torch::zeros({1, max_new_tokens + 1}, int32_gpu);
    auto next_seq_len_gpu   = torch::ones({1}, int32_gpu) + 1;
    auto propose_tokens_gpu = torch::zeros({1, 1}, int32_gpu);

    fake_stream->setMtpAsyncDeviceState(GenerateStream::MtpAsyncDeviceState{
        .epoch                  = 0,
        .accept_len_gpu         = std::move(accept_len_gpu),
        .accept_tokens_gpu      = std::move(accept_tokens_gpu),
        .next_seq_len_gpu       = std::move(next_seq_len_gpu),
        .propose_tokens_gpu     = std::move(propose_tokens_gpu),
        .last_hidden_states_gpu = sp_buffer->hidden_states,
        .draft_all_probs_gpu    = sp_buffer->all_probs,
        .last_real_seq_len      = seq_len,
        .next_real_seq_len      = seq_len,
    });

    return fake_stream;
}

bool MtpExecutor::canSampleCompactVerifyRows(const SamplerInputs& inputs) {
    if (inputs.phase != LogitsProcessorPhase::MTP_VERIFY || !inputs.compact_token_ids
        || inputs.batch_size != inputs.batch_size_out || inputs.logits_processor_states_ptr
        || inputs.cum_log_probs.defined() || inputs.spec_vocab_mask_gpu.defined() || inputs.spec_cap_gpu.defined()
        || !inputs.spec_applied_processors.empty() || !inputs.all_probs.defined() || inputs.return_original_all_probs
        || !inputs.top_k.defined() || inputs.top_k.is_cuda() || inputs.top_k.scalar_type() != torch::kInt32
        || !inputs.top_k.is_contiguous()) {
        return false;
    }
    const auto* top_k = inputs.top_k.data_ptr<int32_t>();
    const auto  rows  = inputs.top_k.numel();
    // Large-vocab TopK renorm has non-bitwise reductions; keep that and mixed
    // TopK/TopP on the unchanged dense path until their separate gate is complete.
    return rows > 0
           && (std::all_of(top_k, top_k + rows, [](auto k) { return k <= 0; })
               || std::all_of(top_k, top_k + rows, [](auto k) { return k == 1; }));
}

void MtpExecutor::capDSparkVerifyLengths(speculative::SpeculativeSamplerOutput& output,
                                         const torch::Tensor&                   verify_lengths) {
    RTP_LLM_CHECK_WITH_INFO(output.accept_len.defined() && verify_lengths.defined()
                                && output.accept_len.sizes() == verify_lengths.sizes()
                                && output.accept_len.device() == verify_lengths.device(),
                            "adaptive DSpARK accept lengths and verify lengths must have matching shape and device");
    output.accept_len = torch::minimum(output.accept_len, verify_lengths);
    // The sampler published its CPU mirror before this cap. Bookkeeping and
    // CUDA KV updates must consume the same committed prefix.
    output.accept_len_cpu = output.accept_len.to(torch::kCPU, /*non_blocking=*/true);
    output.transfer_done_event->record(cuda_graph::graphGetCurrentStream());
}

MtpExecutor::MtpExecutor(const EngineInitParams&                        params,
                         std::unique_ptr<ProposeModelEngineInitParams>& propose_params,
                         const std::shared_ptr<KVCacheManager>&         cache_manager,
                         MlaOpsType                                     mla_ops_type,
                         int32_t                                        kv_cache_group_num,
                         const std::vector<int32_t>&                    kv_cache_layer_to_group,
                         bool                                           warm_up,
                         std::function<void()>                          prefill_profile_start,
                         std::function<void()>                          prefill_profile_finish):
    Executor(),
    cache_manager_(cache_manager),
    metrics_reporter_(params.metrics_reporter),
    tps_reporter_(MetricsLoopReporter<RtpLLMTokenPSMetrics, RtpLLMTokenPSMetricsCollector>(
        params.parallelism_config.tp_rank == 0 && !warm_up ? metrics_reporter_ : nullptr)),
    wall_tps_reporter_(WallClockMetricsLoopReporter<RtpLLMWallClockTokenPSMetrics, RtpLLMTokenPSMetricsCollector>(
        params.parallelism_config.tp_rank == 0 && !warm_up ? metrics_reporter_ : nullptr)),
    warm_up_(warm_up),
    prefill_profile_start_(std::move(prefill_profile_start)),
    prefill_profile_finish_(std::move(prefill_profile_finish)),
    role_type_(params.pd_sep_config.role_type),
    collect_metrics_stream_(cuda_graph::graphGetStreamFromPool(true)),
    // These runners intentionally do not inherit PyTorch profiler TLS from the
    // engine loop. Kineto callbacks are thread-affine; propagating an active
    // profiling state to async MTP worker threads can crash while perf timelines
    // are being recorded.
    target_verify_prepare_runner_(cuda_graph::graphGetStreamFromPool(true), false),
    draft_prefill_prepare_runner_(cuda_graph::graphGetStreamFromPool(true), false),
    spec_logits_verify_async_runner_(cuda_graph::graphGetStreamFromPool(true), false),
    spec_logits_verify_runner_(std::make_unique<SpecLogitsVerifyRunner>()),
    // Bookkeeping worker intentionally does not inherit PyTorch profiler TLS from
    // engine loop. Kineto callbacks are thread-affine; propagating an active
    // profiling state to the worker thread can crash while perf timelines are
    // being recorded.
    spec_bookkeeping_runner_(cuda_graph::graphGetStreamFromPool(true), false) {
    const auto& draft_model_config = propose_params->getEngineInitParams().model_config_;
    data_type_                     = draft_model_config.data_type;
    hidden_size_                   = draft_model_config.hidden_size * draft_model_config.hc_mult;
    propose_step_                  = propose_params->gen_num_per_circle;
    is_dspark_                     = propose_params->sp_type == SP_TYPE_DSPARK;
    dspark_verify_step_            = is_dspark_ ? params.sp_config.verifySteps() : 0;
    dspark_adaptive_verify_        = is_dspark_ && params.sp_config.isAdaptiveVerify();
    params.sp_config.validateVerifyBatchSize(params.runtime_config.max_generate_batch_size);
    dspark_verify_budget_per_request_ = is_dspark_ ? static_cast<size_t>(params.sp_config.verifyBudgetPerRequest()) : 0;
    vocab_size_                       = params.model_config_.vocab_size;
    draft_vocab_size_                 = propose_params->getEngineInitParams().model_config_.vocab_size;

    ResourceContext fake_resource;
    fake_resource.cache_manager = cache_manager;
    fake_resource.role_type     = role_type_;
    make_kv_safe_fake_stream_   = [target  = params.model_config_,
                                 draft   = draft_model_config,
                                 runtime = params.runtime_config,
                                 fake_resource,
                                 step  = propose_step_,
                                 vocab = draft_vocab_size_](bool context) {
        return context ? createMinFakePrefillStream(1, target, runtime, fake_resource) :
                           createMinFakeDecodeStream(step, target, draft, runtime, fake_resource, vocab);
    };

    RTP_LLM_LOG_INFO("[speculative decoding] vocab_size_ = %d, draft_vocab_size_ = %d", vocab_size_, draft_vocab_size_);

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
                                             metrics_reporter_,
                                             params.eplb_config);
    }

    sampler_.reset(new Sampler(SamplerInitParams{}));

    // Optional per-layer cache buffers from KVCacheManager::allLayerCacheBase().
    std::optional<CacheLayerLayout> kv_cache_layer_layout = std::nullopt;
    if (cache_manager && cache_manager->cacheConfig().groupNums() > 1) {
        kv_cache_layer_layout = cache_manager->allLayerCacheBase();
    }

    // Warmup runs MtpExecutor before CacheManager is wired up — guard every
    // cache_manager-> call here so the executor can construct with a null
    // handle. PyWrappedModel's own kernel_tokens_per_block check trips
    // loudly downstream when tokens_per_block stays 0, so we do not need a
    // soft fallback to attn_config here.
    CacheLayerLayout target_cache_layer_layout{};
    CacheLayerLayout draft_cache_layer_layout{};
    if (cache_manager) {
        target_cache_layer_layout = cache_manager->getMainModelCacheLayerLayout();
        draft_cache_layer_layout  = cache_manager->getMTPModuleCacheLayerLayout(0);
    }

    // CacheConfig is the single source of truth for tokens_per_block /
    // kernel_tokens_per_block (DSV4 promotes physical to 256 while
    // attn_config still reflects the 64-token CLI flag). Zero-init the
    // warmup sentinel so PyWrappedModel's >0 check catches mis-propagation
    // (CacheConfig default is 1, not 0).
    CacheConfig warmup_sentinel;
    warmup_sentinel.seq_size_per_block        = 0;
    warmup_sentinel.kernel_seq_size_per_block = 0;
    const auto& target_cache_config           = cache_manager ? cache_manager->cacheConfig() : warmup_sentinel;
    const auto& draft_cache_config = cache_manager ? cache_manager->getMTPModuleCacheConfig(0) : warmup_sentinel;

    GptModelInitParams model_init_params(
        {params.gpt_weights,
         genModelDescription(params.model_config_, params.parallelism_config, params.eplb_config, params.moe_config),
         cache_manager ? std::make_optional(target_cache_layer_layout) : std::nullopt,
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
         static_cast<size_t>(target_cache_config.seq_size_per_block),
         static_cast<size_t>(target_cache_config.kernel_seq_size_per_block),
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
        model_.reset(new PyWrappedModel(
            model_init_params, params.py_model, false, true, target_cache_layer_layout.layer_to_groups));
    }

    is_linear_attention_model_ = target_cache_config.linear_group_num > 0;
    batch_stream_processor_.reset(new MtpBatchStreamProcessor(params.model_config_,
                                                              params.pd_sep_config,
                                                              params.profiling_debug_logging_config,
                                                              target_cache_config,
                                                              params.sp_config,
                                                              warm_up_));

    LogitsProcessorFactory::init(
        params.model_config_.ckpt_path, params.sp_config.tree_decode_config, params.grammar_config);
    cudaProfilerBegin();

    for (auto& mtp_params : *propose_params->mtp_model_params_) {
        auto model_params =
            GptModelInitParams({mtp_params->gpt_weights,
                                Executor::genModelDescription(mtp_params->model_config_,
                                                              mtp_params->parallelism_config,
                                                              mtp_params->eplb_config,
                                                              mtp_params->moe_config),
                                cache_manager ? std::make_optional(draft_cache_layer_layout) : std::nullopt,
                                mtp_params->model_id,
                                mtp_params->parallelism_config,
                                params.hw_kernel_config,
                                params.profiling_debug_logging_config,
                                params.runtime_config,
                                params.concurrency_config,
                                params.sp_config,
                                params.device_resource_config,
                                mla_ops_type,
                                mtp_params->model_config_.max_seq_len,
                                mtp_params->model_config_.hidden_size,
                                static_cast<size_t>(draft_cache_config.seq_size_per_block),
                                static_cast<size_t>(draft_cache_config.kernel_seq_size_per_block),
                                kv_cache_group_num,
                                kv_cache_layer_to_group,
                                cache_manager,
                                mtp_params->model_config_.hc_mult});

        if (!params.py_sp_model.is_none()) {
            RTP_LLM_LOG_INFO("[speculative decoding] using py model");
            draft_model_.reset(new PyWrappedModel(model_params,
                                                  params.py_sp_model,
                                                  false,
                                                  false,
                                                  draft_cache_layer_layout.layer_to_groups,
                                                  is_dspark_ ? DSparkModelRole::PROPOSE : DSparkModelRole::NONE));
            // Create a separate draft prefill model only when the draft itself keeps CUDA graph enabled.
            const bool enable_cuda_graph         = model_params.hw_kernel_config.enable_cuda_graph;
            const bool disable_sp_prefill_by_env = kDisableSpPrefillCudaGraphByEnv;
            const bool draft_has_moe             = model_params.description.ffn_conf.moe_configs.has_value();
            // sp_prefill_cuda_graph_mode: "auto" keeps the MegaMoE policy below,
            // "on" forces the graph, "off" forces eager. The two env vars still
            // win over it so existing diagnostic flows keep working.
            const std::string& sp_prefill_mode             = model_params.hw_kernel_config.sp_prefill_cuda_graph_mode;
            const bool         mode_on                     = sp_prefill_mode == "on";
            const bool         mode_off                    = sp_prefill_mode == "off";
            const bool         force_sp_prefill_cuda_graph = kForceSpPrefillCudaGraphByEnv || mode_on;
            // Keep speculative prefill replay disabled for a distributed MegaMoE
            // engine unless it is explicitly forced for diagnostics. Even when
            // the draft itself is dense, replay has shown rank-local latency
            // spikes that stall the whole DP batch while the target uses EP.
            const bool target_uses_mega_moe = isMegaMoeStrategy(params.moe_config.moe_strategy);
            const bool draft_uses_mega_moe  = draft_has_moe && isMegaMoeStrategy(mtp_params->moe_config.moe_strategy);
            const bool uses_mega_moe        = target_uses_mega_moe || draft_uses_mega_moe;
            const bool uses_ep_collective =
                params.parallelism_config.ep_size > 1 || mtp_params->parallelism_config.ep_size > 1;
            const bool disable_sp_prefill_for_mega_moe =
                uses_mega_moe && uses_ep_collective && !force_sp_prefill_cuda_graph;
            const bool disable_sp_prefill_by_mode = mode_off && !kForceSpPrefillCudaGraphByEnv;
            const bool disable_sp_prefill_cuda_graph =
                disable_sp_prefill_by_env || disable_sp_prefill_by_mode || disable_sp_prefill_for_mega_moe;
            RTP_LLM_LOG_INFO("[speculative decoding] enable_cuda_graph=%d disable_sp_prefill_cuda_graph=%d "
                             "sp_prefill_cuda_graph_mode=%s disable_by_mode=%d "
                             "disable_by_env=%d disable_for_mega_moe=%d force_sp_prefill_cuda_graph=%d "
                             "draft_has_moe=%d target_uses_mega_moe=%d draft_uses_mega_moe=%d "
                             "uses_ep_collective=%d "
                             "(set ENABLE_CUDA_GRAPH=1 when starting server to enable sp_prefill_draft_model_; "
                             "set --sp_prefill_cuda_graph_mode on|off to override the MegaMoE policy; "
                             "set DISABLE_SP_PREFILL_CUDA_GRAPH=1 to skip the draft prefill CUDA graph capture only; "
                             "set RTP_LLM_FORCE_SP_PREFILL_CUDA_GRAPH=1 for diagnostic replay on MegaMoE)",
                             static_cast<int>(enable_cuda_graph),
                             static_cast<int>(disable_sp_prefill_cuda_graph),
                             sp_prefill_mode.c_str(),
                             static_cast<int>(disable_sp_prefill_by_mode),
                             static_cast<int>(disable_sp_prefill_by_env),
                             static_cast<int>(disable_sp_prefill_for_mega_moe),
                             static_cast<int>(force_sp_prefill_cuda_graph),
                             static_cast<int>(draft_has_moe),
                             static_cast<int>(target_uses_mega_moe),
                             static_cast<int>(draft_uses_mega_moe),
                             static_cast<int>(uses_ep_collective));
            if ((enable_cuda_graph && !disable_sp_prefill_cuda_graph) || is_dspark_) {
                RTP_LLM_LOG_INFO(
                    "[speculative decoding] creating separate prefill draft model with CUDA graph support");
                py::object sp_prefill_py_model = params.py_sp_model;
                {
                    py::gil_scoped_acquire gil;
                    if (py::hasattr(params.py_sp_model, "clone_for_cuda_graph")) {
                        try {
                            sp_prefill_py_model = params.py_sp_model.attr("clone_for_cuda_graph")();
                            RTP_LLM_LOG_INFO(
                                "[speculative decoding] cloned py_sp_model for sp_prefill CUDA graph runtime state");
                        } catch (const py::error_already_set& e) {
                            RTP_LLM_LOG_ERROR("[speculative decoding] clone_for_cuda_graph failed:\n%s", e.what());
                            throw;
                        }
                    } else {
                        RTP_LLM_LOG_WARNING(
                            "[speculative decoding] py_sp_model has no clone_for_cuda_graph(); sp_prefill CUDA graph will share Python runtime state with eager draft model");
                    }
                }
                // Draft decode consumes the draft's recurrent hidden state,
                // while draft prefill consumes the target model's output
                // contract. They are identical for legacy MTP/DSv4 models,
                // but EAGLE3 has H-wide recurrence and a 3H target handoff.
                auto sp_prefill_model_params    = model_params;
                sp_prefill_model_params.hc_mult = params.model_config_.hc_mult;
                sp_prefill_draft_model_.reset(
                    new PyWrappedModel(sp_prefill_model_params,
                                       sp_prefill_py_model,
                                       !is_dspark_,
                                       false,
                                       draft_cache_layer_layout.layer_to_groups,
                                       is_dspark_ ? DSparkModelRole::COMMIT : DSparkModelRole::NONE));
            }
        }
        break;  // NOTE: only support one mtp model now
    }

    target_kv_cache_layer_to_group =
        torch::empty({(int64_t)target_cache_layer_layout.layers_to_kv_buffer_ptrs.size()}, torch::kInt32).pin_memory();
    draft_kv_cache_layer_to_group =
        torch::empty({(int64_t)draft_cache_layer_layout.layers_to_kv_buffer_ptrs.size()}, torch::kInt32).pin_memory();

    memcpy(target_kv_cache_layer_to_group.data_ptr<int>(),
           target_cache_layer_layout.layer_to_groups.data(),
           target_cache_layer_layout.layer_to_groups.size() * sizeof(int));
    memcpy(draft_kv_cache_layer_to_group.data_ptr<int>(),
           draft_cache_layer_layout.layer_to_groups.data(),
           draft_cache_layer_layout.layer_to_groups.size() * sizeof(int));

    const auto& draft_weights = propose_params->getEngineInitParams().gpt_weights;
    d2t_map_                  = draft_model_ ? draft_model_->weights_.d2t_map : draft_weights.d2t_map;
    if (is_dspark_) {
        dspark_markov_w1_    = draft_weights.dspark_markov_w1;
        dspark_markov_w2_    = draft_weights.dspark_markov_w2;
        dspark_confidence_w_ = draft_weights.dspark_confidence_w;
        dspark_confidence_b_ = draft_weights.dspark_confidence_b;
        RTP_LLM_CHECK_WITH_INFO(dspark_markov_w1_.defined() && dspark_markov_w2_.defined(),
                                "DSpARK requires markov_w1 and markov_w2 weights");
        const int64_t padded_draft_vocab_size = (static_cast<int64_t>(draft_vocab_size_) + 127) / 128 * 128;
        RTP_LLM_CHECK_WITH_INFO(dspark_markov_w1_.is_cuda() && dspark_markov_w2_.is_cuda()
                                    && dspark_markov_w1_.dim() == 2 && dspark_markov_w2_.dim() == 2
                                    && dspark_markov_w1_.size(1) == dspark_markov_w2_.size(1)
                                    && dspark_markov_w1_.size(0) >= static_cast<int64_t>(vocab_size_)
                                    && dspark_markov_w2_.size(0) >= static_cast<int64_t>(draft_vocab_size_)
                                    && dspark_markov_w2_.size(0) <= padded_draft_vocab_size
                                    && dspark_markov_w1_.scalar_type() == dspark_markov_w2_.scalar_type(),
                                "DSpARK Markov weights must be CUDA [target_vocab,rank] and "
                                "[draft_vocab,rank] tensors with matching rank and dtype");
        dspark_markov_w2_ = dspark_markov_w2_.narrow(0, 0, static_cast<int64_t>(draft_vocab_size_));
        if (dspark_adaptive_verify_) {
            const int64_t confidence_features = static_cast<int64_t>(hidden_size_) + dspark_markov_w1_.size(1);
            RTP_LLM_CHECK_WITH_INFO(
                dspark_confidence_w_.defined() && dspark_confidence_b_.defined() && dspark_confidence_w_.is_cuda()
                    && dspark_confidence_b_.is_cuda() && dspark_confidence_w_.scalar_type() == torch::kBFloat16
                    && dspark_confidence_b_.scalar_type() == torch::kBFloat16
                    && dspark_confidence_w_.numel() == confidence_features && dspark_confidence_b_.numel() == 1,
                "DSpARK confidence head must be CUDA BF16 [hidden+markov_rank] with scalar bias");
            dspark_confidence_w_ = dspark_confidence_w_.contiguous();
            dspark_confidence_b_ = dspark_confidence_b_.contiguous();
        }
    }
    auto proposal_mode = params.sp_config.deterministic_draft_exact_match ?
                             speculative::DraftProposalMode::DETERMINISTIC :
                             speculative::DraftProposalMode::LEGACY;
#if USING_CUDA
    // Exact sampled rejection is implemented by the CUDA backend. Preserve
    // existing HIP proposal semantics for other DSpARK model families.
    if (is_dspark_) {
        proposal_mode = speculative::DraftProposalMode::SAMPLED;
    }
#endif
    speculative_sampler_.reset(new speculative::SpeculativeSampler(d2t_map_, propose_step_, proposal_mode));
    if (is_dspark_ && verifySteps() != propose_step_) {
        dspark_verify_sampler_.reset(new speculative::SpeculativeSampler(d2t_map_, verifySteps(), proposal_mode));
    }
    if (is_dspark_) {
        RTP_LLM_LOG_INFO("[DSpARK] generated_candidates=%zu verified_candidates=%zu target_commit_width=%zu "
                         "adaptive_verify=%d initial_budget_per_request=%zu",
                         propose_step_,
                         verifySteps(),
                         verifySteps() + 1,
                         static_cast<int>(dspark_adaptive_verify_),
                         dspark_verify_budget_per_request_);
    }
    if (!is_dspark_) {
        fast_topk_sampler_.reset(new speculative::FastTopKSampler(d2t_map_, proposal_mode));
    }

    RTP_LLM_LOG_INFO("[speculative decoding] d2t_map size: %ld, deterministic_draft_exact_match: %d",
                     d2t_map_.defined() ? d2t_map_.numel() : 0,
                     params.sp_config.deterministic_draft_exact_match);
}

/*
 * @brief mtp prefill step:
 *
 * +-----------------------------+
 * |     gather model input      |
 * +-----------------------------+
 *              |
 *              v
 * +-----------------------------+
 * |    target model forward     |
 * +-----------------------------+
 *              |
 *              v
 * +-----------------------------+
 * |     target model sample     |
 * +-----------------------------+
 *              |
 *              v
 * +-----------------------------+
 * |     update model input      |
 * +-----------------------------+
 *              |
 *              v
 * +-----------------------------+
 * |     draft model forward     |
 * +-----------------------------+
 *              |
 *              v
 * +-----------------------------+
 * |     draft model sample      |
 * +-----------------------------+
 *              |
 *              v
 * +-----------------------------+
 * |  dispatch output to streams |
 * +-----------------------------+
 *
 * @param streams
 * @return absl::Status
 */
absl::Status MtpExecutor::prefillStep(const std::list<GenerateStreamPtr>& streams,
                                      MtpMetricsCollector&                metrics_collector,
                                      int64_t                             schedule_time_us) {
    RTP_LLM_PROFILE_SCOPE_DYNAMIC("executor.mtp.prefill_step(prefill_stream_size=%zu)", streams.size());

    RtpLLMExecutorMetricsCollector& executor_collector = metrics_collector.executor_collector;
    RtpLLMTokenPSMetricsCollector&  tps_collector      = metrics_collector.tps_collector;

    StreamGroups                                              stream_groups(streams);
    GptModelInputs                                            model_input;
    GptModelOutputs                                           model_output;
    SamplerOutput                                             sampler_output;
    GptModelOutputs                                           draft_model_output;
    SamplerOutput                                             draft_sampler_output;
    torch::Tensor                                             draft_last_hidden_states;
    std::vector<MtpBatchStreamProcessor::PrefillTargetOutput> target_outputs;

    // placeholder for some tensors
    torch::Tensor                      draft_probs;
    torch::Tensor                      draft_token_ids;
    speculative::FastTopKSamplerOutput fast_topk_sampler_output;
    int64_t                            model_forward_us = 0;

    {
        RTP_LLM_PROFILE_SCOPE("executor.mtp.prefill_step(gather_model_input)");
        int64_t start_time_us      = autil::TimeUtility::currentTimeInMicroSeconds();
        auto    model_input_status = batch_stream_processor_->gatherModelInput(stream_groups, buffer_holder_);
        RETURN_IF_STATUS_OR_ERROR(model_input_status);
        model_input                              = std::move(model_input_status.value());
        executor_collector.gather_model_input_us = autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
    }
    {
        RTP_LLM_PROFILE_SCOPE("executor.mtp.prefill_step(tp_sync_input)");
        int64_t start_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
        model_input.skip_run  = streams.empty() && !enable_ffn_disaggregate_;
        tpSyncModelInputs(model_input, parallelism_config_);
        if (model_input.skip_run) {
            return absl::OkStatus();
        }
        executor_collector.tp_sync_input_us = autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
    }

    metrics_collector.not_skip = true;

    // TP ranks also enter prefillStep while idle. Start the window only after
    // synchronized input proves there is real work, on this execution thread.
    if (prefill_profile_start_) {
        prefill_profile_start_();
    }
    struct FinishPrefillProfile {
        const std::function<void()>& callback;
        ~FinishPrefillProfile() {
            if (callback) {
                // Profiling must not terminate the engine during stack unwind.
                try {
                    callback();
                } catch (const std::exception& error) {
                    RTP_LLM_LOG_ERROR("failed to finish prefill profile: %s", error.what());
                } catch (...) {
                    RTP_LLM_LOG_ERROR("failed to finish prefill profile: unknown exception");
                }
            }
        }
    } finish_prefill_profile{prefill_profile_finish_};

    // release model input before forward
    releaseAllModelBuffers();

    // CP+MTP: the CP processor (handleInputs) rewrites model_input in place
    // to the rank-local zigzag layout for the target forward. The post-target
    // MTP pipeline needs the original global request view, including all
    // multimodal metadata, so snapshot every field that CP can rewrite.
    const bool                                cp_enabled = parallelism_config_.prefill_cp_config.is_enabled();
    torch::Tensor                             saved_combo_tokens;
    torch::Tensor                             saved_input_lengths;
    torch::Tensor                             saved_sequence_lengths;
    torch::Tensor                             saved_combo_tokens_type_ids;
    torch::Tensor                             saved_combo_position_ids;
    torch::Tensor                             saved_text_tokens_mask;
    torch::Tensor                             saved_mm_features_locs;
    std::optional<std::vector<torch::Tensor>> saved_multimodal_features;
    std::optional<std::vector<torch::Tensor>> saved_mm_extra_input;
    if (cp_enabled) {
        saved_combo_tokens          = toCudaWithHostHold(model_input.combo_tokens, buffer_holder_);
        saved_input_lengths         = toCudaWithHostHold(model_input.input_lengths, buffer_holder_);
        saved_sequence_lengths      = model_input.sequence_lengths;
        saved_combo_tokens_type_ids = model_input.combo_tokens_type_ids;
        saved_combo_position_ids    = model_input.combo_position_ids;
        saved_text_tokens_mask      = model_input.text_tokens_mask;
        saved_mm_features_locs      = model_input.mm_features_locs;
        saved_multimodal_features   = model_input.multimodal_features;
        saved_mm_extra_input        = model_input.mm_extra_input;
    }
    auto restoreCpGlobalModelInput = [&]() {
        if (!cp_enabled) {
            return;
        }
        model_input.combo_tokens          = saved_combo_tokens;
        model_input.input_lengths         = saved_input_lengths;
        model_input.sequence_lengths      = saved_sequence_lengths;
        model_input.combo_tokens_type_ids = saved_combo_tokens_type_ids;
        model_input.combo_position_ids    = saved_combo_position_ids;
        model_input.text_tokens_mask      = saved_text_tokens_mask;
        model_input.mm_features_locs      = saved_mm_features_locs;
        model_input.multimodal_features   = saved_multimodal_features;
        model_input.mm_extra_input        = saved_mm_extra_input;
    };
    const bool saved_need_all_hidden_states = model_input.need_all_hidden_states;
    const bool use_cp_local_mtp_hidden = cp_enabled && !model_input.need_all_logits && !saved_need_all_hidden_states
                                         && model_->supportsMtpTargetHiddenStates();
    const bool capture_target_outputs = isTpRank0() && is_dspark_ && !model_input.is_fake_stream
                                        && batch_stream_processor_->needsPrefillTargetOutputs(stream_groups);
    const bool target_need_all_logits = model_input.need_all_logits;
    // CP can replace target input metadata. Only requested full-output payloads
    // need a private copy of the original row indices; the default path is empty.
    torch::Tensor original_lm_output_indexes;
    if (capture_target_outputs && target_need_all_logits) {
        original_lm_output_indexes = model_input.lm_output_indexes.clone();
    }

    // target model prefill
    {
        RTP_LLM_PROFILE_SCOPE("executor.mtp.prefill_step(target_model_forward)");
        maybePrintModelInput(model_input, "prefill target model");
        int64_t start_time_us               = autil::TimeUtility::currentTimeInMicroSeconds();
        model_input.kv_cache_layer_to_group = target_kv_cache_layer_to_group;
        model_output                        = std::move(model_->forward(model_input));
        model_forward_us += autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
    }
    model_input.need_all_hidden_states = saved_need_all_hidden_states;
    restoreCpGlobalModelInput();

    if (capture_target_outputs) {
        auto captured = batch_stream_processor_->capturePrefillTargetOutputs(
            stream_groups, model_output, target_need_all_logits, original_lm_output_indexes);
        RETURN_IF_STATUS_OR_ERROR(captured);
        target_outputs = std::move(captured.value());
    }

    // eplb
    if (expert_balancer_) {
        RTP_LLM_PROFILE_SCOPE("executor.mtp.prefill_step(eplb_step_forward)");
        int64_t start_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
        expert_balancer_->stepForward(*model_, executor_collector);
        executor_collector.eplb_step_latency_us = autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
    }

    // target model sample
    if (isTpRank0()) {
        RTP_LLM_PROFILE_SCOPE("executor.mtp.prefill_step(target_model_sample)");
        if (!model_input.is_fake_stream) {
            CHECK_AND_RETURN_REF(sampler_input,
                                 batch_stream_processor_->gatherSamplerInput(stream_groups, model_input, model_output));
            holdSamplerInputHostBuffers(buffer_holder_, sampler_input);
            sampler_output = std::move(sampler_->forward(sampler_input));
            if (!is_dspark_) {
                batch_stream_processor_->updatePrefillPostDraftModelInput(
                    model_input, model_output, sampler_output, buffer_holder_);
            }
        }
        // The executor owns the hand-off layout decision. Compact native-MTP
        // CP uses a model-owned rank-local buffer below; the target's regular
        // output is only the LM-selected rows and must not be labelled GLOBAL.
        if (is_dspark_) {
            model_input.clearLastHiddenStates();
        } else if (use_cp_local_mtp_hidden) {
            model_input.clearLastHiddenStates();
        } else {
            model_input.setLastHiddenStates(model_output.all_hidden_states, MtpHiddenStatesLayout::GLOBAL);
        }
    }

    // draft model prefill
    const bool use_target_mtp_hidden_buffer = use_cp_local_mtp_hidden;
    {
        RTP_LLM_PROFILE_SCOPE("executor.mtp.prefill_step(draft_model_forward)");
        // Native-MTP-capable models expose a rank-local target-hidden buffer.
        // Models without that buffer (GLM5/GenericMoe MTP) use the restored full
        // hidden from model_output.all_hidden_states; after tpSyncModelInputs
        // the draft PyWrappedModel CP path slices it with the same CP planner
        // used for combo_tokens.
        if (use_target_mtp_hidden_buffer || is_dspark_) {
            model_input.clearLastHiddenStates();
        }
        tpSyncModelInputs(model_input, parallelism_config_);
        model_input.mtp_iteration_step = 0;
        maybePrintModelInput(model_input, "prefill post draft model");
        int64_t     start_time_us             = autil::TimeUtility::currentTimeInMicroSeconds();
        const auto& mtp_cache_cfg             = cache_manager_->getMTPModuleCacheConfig(0);
        model_input.kv_block_stride_bytes     = mtp_cache_cfg.kv_block_stride_bytes;
        model_input.kv_scale_stride_bytes     = mtp_cache_cfg.kv_scale_stride_bytes;
        model_input.use_opaque_kv_cache_store = mtp_cache_cfg.use_opaque_kv_cache_store;
        model_input.kv_cache_layer_to_group   = draft_kv_cache_layer_to_group;
        if (is_dspark_) {
            const auto hidden_layout = cp_enabled ? MtpHiddenStatesLayout::CP_LOCAL : MtpHiddenStatesLayout::GLOBAL;
            const bool bound         = maybeOverrideLastHiddenWithMtpBuffer(model_input, *model_, hidden_layout);
            RTP_LLM_CHECK_WITH_INFO(bound, "DSpARK prefill requires target auxiliary hidden rows");
            batch_stream_processor_->validatePrefillDSparkCommitInput(model_input);
        } else if (!cp_enabled || use_target_mtp_hidden_buffer) {
            // Source = main (just ran prefill; its pre-hc buffer is current).
            const auto hidden_layout = cp_enabled ? MtpHiddenStatesLayout::CP_LOCAL : MtpHiddenStatesLayout::GLOBAL;
            maybeOverrideLastHiddenWithMtpBuffer(model_input, *model_, hidden_layout);
        } else {
            RTP_LLM_CHECK_WITH_INFO(model_input.last_hidden_states.defined()
                                        && model_input.last_hidden_states.size(0) == model_input.combo_tokens.size(0),
                                    "CP MTP restored hidden rows must match combo tokens before draft split: "
                                    "hidden_rows=%ld combo_tokens=%ld",
                                    model_input.last_hidden_states.defined() ? model_input.last_hidden_states.size(0) :
                                                                               -1,
                                    model_input.combo_tokens.size(0));
        }
        auto* prefill_draft_model = is_dspark_ ? sp_prefill_draft_model_.get() : draft_model_.get();
        RTP_LLM_CHECK_WITH_INFO(prefill_draft_model != nullptr, "speculative prefill draft model is not initialized");
        draft_model_output             = std::move(prefill_draft_model->forward(model_input));
        model_input.mtp_iteration_step = -1;
        model_forward_us += autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
    }

    if (!isTpRank0() || warm_up_ || streams.size() == 0 || model_input.is_fake_stream) {
        cudaSyncAndCheck();
        return absl::OkStatus();
    }

    if (!cp_enabled && !is_dspark_) {
        maybeOverrideLastHiddenWithMtpBuffer(draft_model_output, *draft_model_);
    }

    // draft model sample
    {
        RTP_LLM_PROFILE_SCOPE("executor.mtp.prefill_step(draft_model_sample)");
        if (!is_dspark_) {
            fast_topk_sampler_output       = fast_topk_sampler_->forward(draft_model_output.logits);
            draft_sampler_output.all_probs = fast_topk_sampler_output.all_probs;
            draft_sampler_output.token_ids = fast_topk_sampler_output.token_ids;
        }
    }

    // collect metrics
    if (metrics_reporter_) {
        RTP_LLM_PROFILE_SCOPE("executor.mtp.prefill_step(collect_metrics)");
        executor_collector.context_batch_size = stream_groups.totalContextBatchSize();
        executor_collector.execute_token_size = stream_groups.modelExecuteTokenSize();
        executor_collector.max_seq_len        = stream_groups.maxSeqLen();

        executor_collector.context_batch_size_when_has_context = executor_collector.context_batch_size;
        executor_collector.execute_token_size_when_has_context = executor_collector.execute_token_size;
        executor_collector.max_seq_len_when_has_context        = executor_collector.max_seq_len;
        executor_collector.model_forward_us += model_forward_us;
        int64_t tps_execute_time_us = autil::TimeUtility::currentTimeInMicroSeconds() - schedule_time_us;
        if (tps_execute_time_us <= 0) {
            tps_execute_time_us = model_forward_us;
        }

        tps_collector.addTokenSize(stream_groups.contextExecuteTokenSize(),
                                   stream_groups.contextExecuteTokenSizeWithCache(),
                                   0,
                                   stream_groups.modelExecuteTokenSize(),
                                   tps_execute_time_us);
    }

    // dispatch
    {
        RTP_LLM_PROFILE_SCOPE("executor.mtp.prefill_step(dispatch_output)");
        auto result =
            batch_stream_processor_->dispatchPrefill(stream_groups,
                                                     {std::move(model_output), std::move(sampler_output)},
                                                     {std::move(draft_model_output), std::move(draft_sampler_output)},
                                                     draft_last_hidden_states,
                                                     target_outputs);
        RTP_LLM_LOG_DEBUG("dispatch done");
        return result;
    }
}

/*
+-------------------------------+
|       gather model input      |
+-------------------------------+
        |
        v
+-------------------------------+
|     draft model forward       |<------------------+
+-------------------------------+                   |
        |                                           |
        v                              +------------------------+
+-------------------------------+      |    update model input  |
|     draft model sample        |      +------------------------+
+-------------------------------+                   |
        |                                           |
        |                                           |
        +---[if steps < propose_step-1] ------------+
        |
        |
        v
+-------------------------------+
|     update model input        |
+-------------------------------+
        |
        v
+-------------------------------+
|    target model forward       |
+-------------------------------+
        |
        v
+-------------------------------+
|     target model sample       |
+-------------------------------+
        |
        v
+-------------------------------+
|      rejection sample         |
+-------------------------------+
        |
        v
+-------------------------------+
|     update model input        |
+-------------------------------+
        |
        v
+-------------------------------+
|     draft model forward       |
+-------------------------------+
        |
        v
+-------------------------------+
|      draft model sample       |
+-------------------------------+
        |
        v
+-------------------------------+
|   dispatch output to streams  |
+-------------------------------+
*/

void MtpExecutor::prepareGrpcMtpDeviceState(const std::list<GenerateStreamPtr>& streams, TensorHolder& host_holder) {
    RTP_LLM_PROFILE_SCOPE("executor.mtp.decode_step(prepare grpc input)");
    const auto pinned_i32    = torch::TensorOptions().dtype(torch::kInt32).pinned_memory(true);
    auto       to_cuda_async = [&host_holder](const torch::Tensor& tensor) {
        if (!tensor.defined()) {
            return tensor;
        }
        if (tensor.is_cuda()) {
            return tensor;
        }
        if (tensor.numel() == 0) {
            return torch::empty(tensor.sizes(), torch::TensorOptions(tensor.dtype()).device(torch::kCUDA));
        }
        if (!tensor.is_pinned()) {
            RTP_LLM_LOG_WARNING("[mtp-grpc] grpc tensor is not pinned; H2D copy may block");
        }
        host_holder.hold_host(tensor);
        return tensor.to(torch::kCUDA, /*non_blocking=*/true);
    };

    for (auto& stream : streams) {
        auto sp_output_buffer = stream->getSPOutputBuffer();
        if (sp_output_buffer == nullptr) {
            continue;
        }
        auto& tensors_holder = sp_output_buffer->tensors_holder;
        if (tensors_holder.empty()) {
            continue;
        }
        if (tensors_holder.size() != 2) {
            RTP_LLM_LOG_WARNING("[mtp-grpc] skip grpc input: tensors_holder_size=%zu, stream=%ld",
                                tensors_holder.size(),
                                stream->streamId());
            tensors_holder.clear();
            continue;
        }
        if (!sp_output_buffer->tokens.defined() || sp_output_buffer->tokens.dim() != 2
            || sp_output_buffer->tokens.size(0) < 1 || sp_output_buffer->tokens.size(1) < 2) {
            RTP_LLM_LOG_WARNING("[mtp-grpc] skip grpc input: invalid tokens, stream=%ld", stream->streamId());
            tensors_holder.clear();
            continue;
        }

        const auto& propose_probs_t  = tensors_holder[0];
        const auto& propose_hidden_t = tensors_holder[1];
        RTP_LLM_CHECK_WITH_INFO(propose_probs_t.defined() && propose_probs_t.numel() > 0,
                                "[mtp-grpc] propose_probs must be non-empty, stream=%ld",
                                stream->streamId());
        if (propose_step_ > 1) {
            const int64_t hidden_dim   = propose_hidden_t.defined() ? propose_hidden_t.dim() : -1;
            const int64_t hidden_numel = propose_hidden_t.defined() ? propose_hidden_t.numel() : -1;
            const bool    valid_hidden =
                propose_hidden_t.defined() && propose_hidden_t.dim() == 2 && propose_hidden_t.size(0) > 0;
            RTP_LLM_CHECK_WITH_INFO(valid_hidden,
                                    "[mtp-grpc] propose_hidden must be non-empty 2-D for multi-step MTP, stream=%ld "
                                    "dim=%ld numel=%ld",
                                    stream->streamId(),
                                    hidden_dim,
                                    hidden_numel);
        }

        sp_output_buffer->all_probs     = to_cuda_async(propose_probs_t);
        sp_output_buffer->hidden_states = to_cuda_async(propose_hidden_t);

        auto       accept_len_cpu     = torch::ones({1}, pinned_i32);
        auto       accept_tokens_cpu  = torch::zeros({1, static_cast<int64_t>(propose_step_ + 1)}, pinned_i32);
        auto       propose_tokens_cpu = torch::empty({1}, pinned_i32);
        auto       next_seq_len_cpu   = torch::empty({1}, pinned_i32);
        auto*      token_ptr          = sp_output_buffer->tokens.data_ptr<int32_t>();
        const auto seq_length         = stream->seqLength();
        accept_tokens_cpu.data_ptr<int32_t>()[0]  = token_ptr[0];
        propose_tokens_cpu.data_ptr<int32_t>()[0] = token_ptr[1];
        next_seq_len_cpu.data_ptr<int32_t>()[0]   = seq_length;

        auto accept_len_gpu     = to_cuda_async(accept_len_cpu);
        auto accept_tokens_gpu  = to_cuda_async(accept_tokens_cpu);
        auto propose_tokens_gpu = to_cuda_async(propose_tokens_cpu);
        auto next_seq_len_gpu   = to_cuda_async(next_seq_len_cpu);

        stream->setMtpAsyncDeviceState(GenerateStream::MtpAsyncDeviceState{
            .epoch                  = 0,
            .accept_len_gpu         = std::move(accept_len_gpu),
            .accept_tokens_gpu      = std::move(accept_tokens_gpu),
            .next_seq_len_gpu       = std::move(next_seq_len_gpu),
            .propose_tokens_gpu     = std::move(propose_tokens_gpu),
            .last_hidden_states_gpu = sp_output_buffer->hidden_states,
            .draft_all_probs_gpu    = sp_output_buffer->all_probs,
            .last_real_seq_len      = seq_length,
            .next_real_seq_len      = seq_length,
        });

        tensors_holder.clear();
    }

    return;
}

absl::Status MtpExecutor::decodeStep(const std::list<GenerateStreamPtr>& streams,
                                     MtpMetricsCollector&                metrics_collector) {
    RTP_LLM_PROFILE_SCOPE_DYNAMIC("executor.mtp.decode_step(decode_stream_size=%zu)", streams.size());

    const auto verify_steps = verifySteps();

    RtpLLMExecutorMetricsCollector& executor_collector = metrics_collector.executor_collector;

    GptModelInputs  model_input;
    GptModelOutputs model_output;
    GptModelOutputs draft_prefill_model_output;

    SamplerOutput                         draft_sampler_output;
    speculative::SpeculativeSamplerOutput speculative_sampler_output;

    // Placeholders shared across draftModelDecode and the post-rejection update.
    torch::Tensor              draft_token_probs_d_t;
    torch::Tensor              hidden_states_d_t;
    torch::Tensor              draft_token_ids_t;
    torch::Tensor              spec_token_ids_t;
    torch::Tensor              dspark_verify_lengths;
    torch::Tensor              dspark_compact_to_dense;
    std::vector<torch::Tensor> draft_probs_list;
    torch::Event               accept_len_ready_event = cuda_graph::makeGraphEvent();
    int64_t                    model_forward_us       = 0;
    auto                       spec_logits_result     = std::make_shared<SpecLogitsVerifyRunner::LaunchResult>();

    // Stream-async events are recorded on the main stream as soon as tensors
    // become valid. rejection_event guards accept_len/tokens D2H; draft_event
    // guards all_probs cloning. They stay null when stream-async is off.
    std::shared_ptr<torch::Event> rejection_event;
    std::shared_ptr<torch::Event> draft_event;
    bool                          prev_bookkeeping_synced_for_spec_logits = false;
    bool                          spec_logits_async_launched              = false;
    bool                          spec_logits_processor_present           = false;

    waitPreviousBookkeepingAndKvSwaps(streams);
    // StreamGroups snapshots host-side stream state (batch sizes, execute-token
    // counts, sequence lengths, and context/decode classification) in its
    // constructor. Building it before the wait races specUpdate() in the
    // previous bookkeeping worker and permanently retains a mixed old/new
    // snapshot for this decode step.
    StreamGroups stream_groups(streams);
    prepareGrpcMtpDeviceState(streams, buffer_holder_);

    {
        RTP_LLM_PROFILE_SCOPE("executor.mtp.decode_step(gather_model_input)");
        int64_t start_time_us      = autil::TimeUtility::currentTimeInMicroSeconds();
        auto    model_input_status = batch_stream_processor_->gatherDecodeModelInput(stream_groups, buffer_holder_);
        RETURN_IF_STATUS_OR_ERROR(model_input_status);
        model_input = std::move(model_input_status.value());
        executor_collector.gather_model_input_us += autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
    }

    if (isTpRank0()) {
        RTP_LLM_PROFILE_SCOPE("executor.mtp.decode_step(tp_sync_input_rank0)");
        int64_t start_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
        model_input.skip_run  = streams.empty() && !enable_ffn_disaggregate_;
        if (model_input.skip_run) {
            tpSyncModelInputs(model_input, parallelism_config_);
            return absl::OkStatus();
        }
        executor_collector.tp_sync_input_us += autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
    }

    metrics_collector.not_skip = true;

    // TODO(yinzhi): consider beam search & lora

    MtpBatchStreamProcessor::DSparkRoundState dspark_round_state;
    size_t                                    batch_size = 0;
    if (is_dspark_) {
        GptModelInputs proposal_input;
        if (isTpRank0()) {
            dspark_round_state =
                batch_stream_processor_->buildDSparkRoundState(stream_groups, model_input, buffer_holder_);
            proposal_input = model_input;
            batch_stream_processor_->prepareDSparkProposeModelInput(dspark_round_state, proposal_input, buffer_holder_);
            ensureModelInputsOnCuda(proposal_input, "decode.prepare_dspark_proposal");
        }
        tpSyncModelInputs(proposal_input, parallelism_config_);
        if (proposal_input.skip_run) {
            return absl::OkStatus();
        }
        ensureModelInputsOnCuda(proposal_input, "decode.dspark_proposal_after_tp_sync");
        batch_size = proposal_input.input_lengths.size(0);

        releaseAllModelBuffers();
        const auto& draft_cache_cfg              = cache_manager_->getMTPModuleCacheConfig(0);
        proposal_input.kv_block_stride_bytes     = draft_cache_cfg.kv_block_stride_bytes;
        proposal_input.kv_scale_stride_bytes     = draft_cache_cfg.kv_scale_stride_bytes;
        proposal_input.use_opaque_kv_cache_store = draft_cache_cfg.use_opaque_kv_cache_store;
        proposal_input.kv_cache_layer_to_group   = draft_kv_cache_layer_to_group;
        GptModelOutputs proposal_output;
        {
            RTP_LLM_PROFILE_SCOPE("executor.mtp.decode_step(dspark_proposal_forward)");
            int64_t start_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
            proposal_output       = runDSparkProposeForward(proposal_input);
            model_forward_us += autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
        }
        if (isTpRank0()) {
            {
                RTP_LLM_PROFILE_SCOPE("executor.mtp.decode_step(dspark_markov_sample)");
                draft_sampler_output =
                    sampleDSparkDraft(stream_groups, proposal_output.logits, dspark_round_state.anchors);
            }
            draft_token_ids_t =
                torch::cat({dspark_round_state.anchors.reshape({static_cast<int64_t>(batch_size), 1}).to(torch::kInt32),
                            draft_sampler_output.token_ids.to(torch::kInt32)},
                           1);
            if (dspark_adaptive_verify_) {
                RTP_LLM_PROFILE_SCOPE("executor.mtp.decode_step(dspark_confidence_plan)");
                auto confidence = speculative_sampler_->computeDSparkConfidence(proposal_output.all_hidden_states,
                                                                                dspark_round_state.anchors,
                                                                                draft_sampler_output.token_ids,
                                                                                dspark_markov_w1_,
                                                                                dspark_confidence_w_,
                                                                                dspark_confidence_b_);
                const int64_t extra_budget =
                    std::min<int64_t>(static_cast<int64_t>(batch_size * dspark_verify_budget_per_request_),
                                      static_cast<int64_t>(batch_size * propose_step_));
                auto verify_plan        = execDSparkVerifyPlan(confidence, extra_budget);
                dspark_verify_lengths   = std::move(verify_plan.first);
                dspark_compact_to_dense = std::move(verify_plan.second);
                batch_stream_processor_->prepareCompactDSparkTargetVerifyModelInput(dspark_round_state,
                                                                                    model_input,
                                                                                    draft_sampler_output.token_ids,
                                                                                    dspark_verify_lengths,
                                                                                    dspark_compact_to_dense,
                                                                                    buffer_holder_);
            } else {
                // Full checkpoint backbone/LM-head rows remain unchanged. The
                // sampler returns only the configured verified prefix.
                batch_stream_processor_->prepareDSparkTargetVerifyModelInput(
                    dspark_round_state, model_input, draft_sampler_output.token_ids, buffer_holder_);
            }
            ensureModelInputsOnCuda(model_input, "decode.prepare_dspark_target_verify");
        }
        tpSyncModelInputs(model_input, parallelism_config_);
        ensureModelInputsOnCuda(model_input, "decode.dspark_target_verify_after_tp_sync");
    } else {
        {
            RTP_LLM_PROFILE_SCOPE("executor.mtp.decode_step(prepare_decode_input_and_tp_sync)");
            if (isTpRank0()) {
                if (propose_step_ == 1) {
                    batch_stream_processor_->prepareOneStepSpecDecodeModelInput(
                        stream_groups, model_input, buffer_holder_);
                } else {
                    batch_stream_processor_->prepareDecodeDraftModelInput(stream_groups, model_input, buffer_holder_);
                }
                ensureModelInputsOnCuda(model_input, "decode.prepare_decode_input");
            }
            tpSyncModelInputs(model_input, parallelism_config_);
            if (model_input.skip_run) {
                return absl::OkStatus();
            }
            ensureModelInputsOnCuda(model_input, "decode.after_tp_sync");
        }
        batch_size = model_input.input_lengths.size(0);
        releaseAllModelBuffers();
        launchTargetVerifyPrepareAsync(model_input, batch_size);
        if (propose_step_ > 1) {
            if (shouldSkipFakeStreamForStop(model_input, "draftModelDecode")) {
                if (useAsyncPrepare()) {
                    target_verify_prepare_runner_.sync(cuda_graph::graphGetCurrentStream());
                }
                releaseAllModelBuffers();
                return absl::OkStatus();
            }
            model_input.kv_cache_layer_to_group = draft_kv_cache_layer_to_group;
            RTP_LLM_LOG_DEBUG("[MTP decode] draftModelDecode start");
            draftModelDecode(model_input, stream_groups, draft_probs_list, draft_token_ids_t, model_forward_us);
            RTP_LLM_LOG_DEBUG("[MTP decode] draftModelDecode end");
        }
    }
    spec_logits_processor_present = isTpRank0() && !model_input.is_fake_stream && hasSpecLogitsProcessor(streams);
    if (isTpRank0() && !model_input.is_fake_stream && hasUnsupportedMtpStatefulLogitsProcessor(streams)) {
        return absl::InternalError(
            "MTP spec decode found a stateful logits processor without SpecLogitsProcessor support; "
            "disable MTP or implement spec verify for this processor");
    }

    if (shouldSkipFakeStreamForStop(model_input, "target/draft forward")) {
        return absl::OkStatus();
    }
    // For propose_step > 1, draftModelDecode builds the context-style target
    // verify input only at the end of the loop. Decide CP semantics after that
    // transformation; checking the initial one-token draft-decode input would
    // incorrectly classify it as pure decode.
    const bool cp_context_request = isCpContextRequest(parallelism_config_, model_input);
    RTP_LLM_CHECK_WITH_INFO(!parallelism_config_.prefill_cp_config.is_enabled() || cp_context_request,
                            "MTP target-verify input must be context-style when prefill CP is enabled: "
                            "input_batch=%ld decode_batch=%ld",
                            model_input.input_lengths.defined() ? model_input.input_lengths.size(0) : -1,
                            model_input.sequence_lengths.defined() ? model_input.sequence_lengths.size(0) : -1);
    const bool use_cp_local_decode_hidden = cp_context_request && model_->supportsMtpTargetHiddenStates();
    if (useAsyncPrepare() && !is_dspark_) {
        RTP_LLM_PROFILE_SCOPE("executor.mtp.decode_step(wait_target_verify_prepare)");
        target_verify_prepare_runner_.sync(cuda_graph::graphGetCurrentStream());
    }
    auto draft_tokens_ready_event = std::make_shared<torch::Event>(cuda_graph::makeGraphEvent());
    draft_tokens_ready_event->record(cuda_graph::graphGetCurrentStream());

    int64_t cp_local_decode_hidden_rows = -1;
    {
        if (shouldSkipFakeStreamForStop(model_input, "target verify forward")) {
            releaseAllModelBuffers();
            return absl::OkStatus();
        }
        const bool saved_need_all_hidden_states = model_input.need_all_hidden_states;
        if (cp_context_request && !use_cp_local_decode_hidden) {
            // Models without a rank-local target-hidden buffer must materialize
            // GLOBAL hidden states for the recurrent draft-prefill hand-off.
            model_input.need_all_hidden_states = true;
        }
        int64_t start_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
        model_output          = runTargetVerifyForward(model_input, stream_groups);
        model_forward_us += autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
        if (is_dspark_) {
            maybeOverrideLastHiddenWithMtpBuffer(model_output, *model_);
            if (dspark_adaptive_verify_ && isTpRank0()) {
                model_output.logits = scatterCompactRows(model_output.logits,
                                                         dspark_compact_to_dense,
                                                         static_cast<int64_t>(batch_size * (propose_step_ + 1)));
            }
        }
        model_input.need_all_hidden_states = saved_need_all_hidden_states;
        if (cp_context_request) {
            // handleInputs has replaced combo_tokens with this rank's exact
            // zigzag chunk. Preserve its row count before rejection sampling
            // restores the global dense [batch, propose_step + 1] tokens.
            cp_local_decode_hidden_rows = model_input.combo_tokens.numel();
        }
    }

    // trick: update draft sampler output after spec decode to avoid kernel launch overhead
    if (isTpRank0()) {
        RTP_LLM_PROFILE_SCOPE("executor.mtp.decode_step(update_draft_sampler_output)");
        if (!model_input.is_fake_stream) {
            if (is_dspark_) {
                // The round-head proposal already populated draft_sampler_output.
            } else if (propose_step_ == 1) {
                batch_stream_processor_->updateOneStepDraftSamplerOutput(
                    stream_groups, draft_sampler_output, draft_token_probs_d_t, buffer_holder_);
            } else {
                batch_stream_processor_->updateMultiStepDraftSamplerOutput(stream_groups,
                                                                           draft_sampler_output,
                                                                           draft_token_ids_t,
                                                                           spec_token_ids_t,
                                                                           draft_token_probs_d_t,
                                                                           draft_probs_list);
            }
        }
    }

    if (spec_logits_processor_present && (is_dspark_ || propose_step_ > 1) && draft_token_ids_t.defined()) {
        RTP_LLM_PROFILE_SCOPE("executor.mtp.decode_step(launch_spec_logits_verify_async)");
        if (useStreamAsync() && useDropBroadSync()) {
            RTP_LLM_PROFILE_SCOPE_DYNAMIC(
                "executor.mtp.decode_step(wait_prev_bookkeeping_pre_spec_logits,stream_count=%zu)", streams.size());
            spec_bookkeeping_runner_.sync(cuda_graph::graphGetCurrentStream());
            stream_groups                           = StreamGroups(streams);
            prev_bookkeeping_synced_for_spec_logits = true;
        }

        auto spec_streams = streams;
        auto draft_tokens = draft_token_ids_t;
        spec_logits_verify_async_runner_.launch([this,
                                                 spec_streams = std::move(spec_streams),
                                                 draft_tokens,
                                                 draft_tokens_ready_event,
                                                 spec_logits_result]() mutable {
            RTP_LLM_PROFILE_SCOPE("executor.mtp.decode_step(spec_logits_verify_async_worker)");
            try {
                *spec_logits_result =
                    buildSpecLogitsVerifyInline(spec_streams, draft_tokens, std::move(draft_tokens_ready_event));
            } catch (const std::exception& e) {
                RTP_LLM_LOG_ERROR("spec logits async worker failed: %s", e.what());
                throw;
            } catch (...) {
                RTP_LLM_LOG_ERROR("spec logits async worker failed with unknown exception");
                throw;
            }
        });
        spec_logits_async_launched = true;
    }

    if (spec_logits_processor_present && !spec_logits_async_launched) {
        RTP_LLM_PROFILE_SCOPE("executor.mtp.decode_step(spec_logits_verify_inline)");
        if (useStreamAsync() && useDropBroadSync()) {
            RTP_LLM_PROFILE_SCOPE_DYNAMIC(
                "executor.mtp.decode_step(wait_prev_bookkeeping_pre_spec_logits,stream_count=%zu)", streams.size());
            spec_bookkeeping_runner_.sync(cuda_graph::graphGetCurrentStream());
            stream_groups                           = StreamGroups(streams);
            prev_bookkeeping_synced_for_spec_logits = true;
        }
        std::shared_ptr<torch::Event> draft_tokens_ready_event;
        if (draft_sampler_output.token_ids.defined() && draft_sampler_output.token_ids.is_cuda()) {
            draft_tokens_ready_event = std::make_shared<torch::Event>(cuda_graph::makeGraphEvent());
            draft_tokens_ready_event->record(cuda_graph::graphGetCurrentStream());
        }
        *spec_logits_result =
            buildSpecLogitsVerifyInline(streams, draft_sampler_output.token_ids, std::move(draft_tokens_ready_event));
    }

    if (spec_logits_async_launched) {
        RTP_LLM_PROFILE_SCOPE("executor.mtp.decode_step(wait_spec_logits_verify_async)");
        spec_logits_verify_async_runner_.sync(cuda_graph::graphGetCurrentStream());
    }
    if (spec_logits_processor_present && !spec_logits_result->has_active_processor) {
        return absl::InternalError("MTP async spec logits processor is present but no verify artifact was produced; "
                                   "disable MTP/async or implement spec verify for this processor");
    }

    // eplb
    if (expert_balancer_) {
        RTP_LLM_PROFILE_SCOPE("executor.mtp.decode_step(eplb_step_forward)");
        int64_t start_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
        expert_balancer_->stepForward(*model_, executor_collector);
        executor_collector.eplb_step_latency_us = autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
    }

    SamplerOutput sampler_output;
    if (isTpRank0()) {
        RTP_LLM_PROFILE_SCOPE("executor.mtp.decode_step(rejection_sampling)");

        // Acceptance belongs to the requests and KV state owned by this DP rank.
        // Fake ranks keep a local full-accept placeholder only to participate in
        // the same EP model calls; they must not consume another DP rank's tokens.
        if (model_input.is_fake_stream) {
            speculative_sampler_output.accept_len = torch::full(
                {1}, (int64_t)(verify_steps + 1), torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
            speculative_sampler_output.accept_tokens = torch::zeros(
                {1, (int64_t)(verify_steps + 1)}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
        } else {
            // gatherSpecSamplerInput reads host stream state updated by the previous
            // bookkeeping worker. DROP_BROAD_SYNC therefore needs this narrow sync
            // unless the broad sync at decodeStep start already waited.
            if (useStreamAsync() && useDropBroadSync() && !prev_bookkeeping_synced_for_spec_logits) {
                RTP_LLM_PROFILE_SCOPE_DYNAMIC(
                    "executor.mtp.decode_step(wait_prev_bookkeeping_pre_sampler,stream_count=%zu)", streams.size());
                spec_bookkeeping_runner_.sync(cuda_graph::graphGetCurrentStream());
                // Rebuild after waiting so cached maxSeqLen/batch sizes reflect
                // the host stream state that sampler input is about to read.
                stream_groups = StreamGroups(streams);
            }

            // target model sample
            CHECK_AND_RETURN_REF(
                sampler_input,
                batch_stream_processor_->gatherSpecSamplerInput(stream_groups,
                                                                model_input,
                                                                model_output,
                                                                *spec_logits_result,
                                                                is_dspark_ ? draft_token_ids_t : torch::Tensor()));
#if USING_CUDA
            // Preserve processor/history/cumulative-logprob behavior on the
            // dense path. Only normalization and sampling become compact;
            // dense seed consumption is retained inside the CUDA sampler.
            if (dspark_adaptive_verify_ && canSampleCompactVerifyRows(sampler_input)) {
                sampler_input.verify_sample_rows = dspark_compact_to_dense.to(torch::kLong);
            }
#endif
            holdSamplerInputHostBuffers(buffer_holder_, sampler_input);
            sampler_output           = std::move(sampler_->forward(sampler_input));
            sampler_output.all_probs = sampler_output.all_probs.reshape(
                {(int64_t)batch_size, (int64_t)(verify_steps + 1), (int64_t)vocab_size_});

            // rejection sampling
            auto& verify_sampler = dspark_verify_sampler_ ? dspark_verify_sampler_ : speculative_sampler_;
            speculative_sampler_output =
                verify_sampler->forward(streams, draft_sampler_output, sampler_output, dspark_verify_lengths);
            applySpecLogitsAcceptLenCap(
                sampler_input, sampler_output, speculative_sampler_output, batch_size, verify_steps);
            if (dspark_adaptive_verify_) {
                RTP_LLM_CHECK_WITH_INFO(dspark_verify_lengths.defined()
                                            && dspark_verify_lengths.numel() == static_cast<int64_t>(batch_size),
                                        "adaptive DSpARK verify lengths must remain live through rejection");
                capDSparkVerifyLengths(speculative_sampler_output, dspark_verify_lengths);
            }
        }

        if (is_dspark_) {
            batch_stream_processor_->updateDecodePostDSparkCommitInput(
                model_input, model_output.all_hidden_states, batch_size);
        } else {
            batch_stream_processor_->updateDecodePostDraftModelInput(model_input,
                                                                     model_output,
                                                                     speculative_sampler_output,
                                                                     batch_size,
                                                                     hidden_states_d_t,
                                                                     buffer_holder_,
                                                                     !use_cp_local_decode_hidden);
        }
        if (metrics_reporter_ && !warm_up_ && !streams.empty() && !model_input.is_fake_stream) {
            accept_len_ready_event.record(cuda_graph::graphGetCurrentStream());
            stageAcceptLenMetrics(speculative_sampler_output.accept_len, accept_len_ready_event, streams.size());
        }
    } else {
        if (is_dspark_) {
            batch_stream_processor_->updateDecodePostDSparkCommitInput(
                model_input, model_output.all_hidden_states, batch_size);
        } else if (cp_context_request) {
            const int64_t total_tokens = static_cast<int64_t>(batch_size * (propose_step_ + 1));
            // Rank 0 rebuilds these tensors after rejection sampling. CP target
            // verify left non-root ranks with local-sized tensors, so allocate
            // matching GLOBAL receive buffers before the TP broadcasts.
            model_input.combo_tokens =
                torch::empty({total_tokens}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
            model_input.input_lengths = torch::full({static_cast<int64_t>(batch_size)},
                                                    static_cast<int64_t>(propose_step_ + 1),
                                                    torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
        }
        if (!is_dspark_) {
            model_input.lm_output_indexes =
                torch::empty({(int64_t)batch_size}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
            if (use_cp_local_decode_hidden) {
                model_input.clearLastHiddenStates();
            } else {
                model_input.setLastHiddenStates(model_output.all_hidden_states, MtpHiddenStatesLayout::GLOBAL);
            }
        }
    }

    // Record before broadcast/draft work so the worker waits only for
    // accept_len/accept_tokens, not the queue tail.
    if (useStreamAsync()) {
        rejection_event = std::make_shared<torch::Event>(cuda_graph::makeGraphEvent());
        rejection_event->record(cuda_graph::graphGetCurrentStream());
    }

    bool used_cp_local_hidden = false;
    if (is_dspark_) {
        // The commit input is already bound to the dense target verify feature rows.
    } else if (use_cp_local_decode_hidden) {
        used_cp_local_hidden = maybeOverrideLastHiddenWithMtpBuffer(
            model_input, *model_, MtpHiddenStatesLayout::CP_LOCAL, cp_local_decode_hidden_rows);
    } else if (!cp_context_request) {
        // Non-CP decode keeps the historical model-buffer override. Under CP,
        // an undeclared/opportunistic buffer has no trustworthy row layout;
        // retain the explicitly materialized GLOBAL model output instead.
        maybeOverrideLastHiddenWithMtpBuffer(model_input, *model_);
    }
    RTP_LLM_CHECK_WITH_INFO(is_dspark_ || !use_cp_local_decode_hidden || used_cp_local_hidden,
                            "CP recurrent draft-prefill requires a rank-local target-hidden buffer");
    RTP_LLM_CHECK_WITH_INFO(is_dspark_
                                || (use_cp_local_decode_hidden ?
                                        model_input.last_hidden_states_layout == MtpHiddenStatesLayout::CP_LOCAL :
                                        model_input.last_hidden_states_layout == MtpHiddenStatesLayout::GLOBAL),
                            "unexpected post-rejection MTP hidden layout=%s, expected=%s",
                            mtpHiddenStatesLayoutName(model_input.last_hidden_states_layout),
                            use_cp_local_decode_hidden ? "CP_LOCAL" : "GLOBAL");
    broadcastPostRejectionInputs(
        model_input, stream_groups, /*broadcast_hidden_states=*/is_dspark_ || !use_cp_local_decode_hidden);
    // Draft-prefill inputs are finalized only after rejection sampling updates
    // combo_tokens/last_hidden_states/lm_output_indexes and the TP broadcast
    // propagates them. Preparing before this point can leave the CUDA graph
    // attention/KV buffers stale while forward() uses the post-rejection token
    // and hidden tensors.
    launchDraftPrefillPrepareAsync(model_input);

    {
        if (useAsyncPrepare()) {
            // prepareAttentionInputs mutates PyWrappedModel state consumed by forward().
            RTP_LLM_PROFILE_SCOPE("executor.mtp.decode_step(wait_draft_prefill_prepare)");
            draft_prefill_prepare_runner_.sync(cuda_graph::graphGetCurrentStream());
        }
        if (shouldSkipFakeStreamForStop(model_input, "draft prefill forward")) {
            releaseAllModelBuffers();
            return absl::OkStatus();
        }
        int64_t start_time_us      = autil::TimeUtility::currentTimeInMicroSeconds();
        draft_prefill_model_output = runDraftPrefillForward(model_input);
        model_forward_us += autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
    }

    if (!isTpRank0() || warm_up_ || streams.size() == 0 || model_input.is_fake_stream) {
        releaseAllModelBuffers();
        return absl::OkStatus();
    }

    // draft model sample
    SamplerOutput draft_prefill_sampler_output;
    {
        RTP_LLM_PROFILE_SCOPE("executor.mtp.decode_step(draft_model_sample)");
        if (!is_dspark_) {
            auto fast_topk_sampler_output          = fast_topk_sampler_->forward(draft_prefill_model_output.logits);
            draft_prefill_sampler_output.all_probs = fast_topk_sampler_output.all_probs;
            draft_prefill_sampler_output.token_ids = fast_topk_sampler_output.token_ids;
        }
    }

    // Record after draft_model_sample so worker all_probs/token_ids reads wait
    // on the earliest valid point, not metrics or dispatch slicing.
    if (useStreamAsync()) {
        draft_event = std::make_shared<torch::Event>(cuda_graph::makeGraphEvent());
        draft_event->record(cuda_graph::graphGetCurrentStream());
    }

    return dispatchDecodeOutput(stream_groups,
                                metrics_collector,
                                streams,
                                speculative_sampler_output,
                                std::move(draft_prefill_model_output),
                                std::move(draft_prefill_sampler_output),
                                std::move(rejection_event),
                                std::move(draft_event));
}

void MtpExecutor::launchTargetVerifyPrepareAsync(const GptModelInputs& model_input, size_t batch_size) {
    if (!useAsyncPrepare()) {
        return;
    }
    if (parallelism_config_.prefill_cp_config.is_enabled()) {
        // Every MTP target-verify pass becomes a context-style [B, propose+1]
        // input. For propose_step > 1 that transformation happens later inside
        // draftModelDecode, so the current one-token shape cannot identify CP.
        RTP_LLM_LOG_DEBUG("[MTP decode] skip target-verify async prepare when prefill CP is enabled");
        return;
    }
    const auto& cache_cfg                      = cache_manager_->cacheConfig();
    auto        model_input_copy               = model_input;
    model_input_copy.kv_block_stride_bytes     = cache_cfg.kv_block_stride_bytes;
    model_input_copy.kv_scale_stride_bytes     = cache_cfg.kv_scale_stride_bytes;
    model_input_copy.use_opaque_kv_cache_store = cache_cfg.use_opaque_kv_cache_store;
    model_input_copy.kv_cache_layer_to_group   = target_kv_cache_layer_to_group;
    {
        RTP_LLM_PROFILE_SCOPE("executor.mtp.decode_step(prepare_target_verify_input)");
        const auto cuda_i32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
        model_input_copy.combo_tokens =
            torch::empty({static_cast<int64_t>(batch_size * (propose_step_ + 1))}, cuda_i32);
        torch::Tensor sequence_lengths_for_prepare = model_input.sequence_lengths;
        if ((!sequence_lengths_for_prepare.defined()
             || sequence_lengths_for_prepare.numel() < static_cast<int64_t>(batch_size))
            && model_input.prefix_lengths.defined()) {
            sequence_lengths_for_prepare = model_input.prefix_lengths;
        }
#if USING_CUDA
        const bool can_fuse_target_prepare =
            sequence_lengths_for_prepare.defined() && sequence_lengths_for_prepare.is_cuda()
            && sequence_lengths_for_prepare.scalar_type() == torch::kInt32
            && sequence_lengths_for_prepare.is_contiguous()
            && sequence_lengths_for_prepare.numel() >= static_cast<int64_t>(batch_size);
        if (can_fuse_target_prepare) {
            model_input_copy.input_lengths           = torch::empty({static_cast<int64_t>(batch_size)}, cuda_i32);
            model_input_copy.prefix_lengths          = torch::empty({static_cast<int64_t>(batch_size)}, cuda_i32);
            model_input_copy.sequence_lengths_plus_1 = torch::empty({static_cast<int64_t>(batch_size)}, cuda_i32);
            model_input_copy.lm_output_indexes       = torch::empty({static_cast<int64_t>(batch_size)}, cuda_i32);
            RTP_LLM_PROFILE_SCOPE("executor.mtp.decode_step(prepare_target_verify_input_fused)");
            invokeMtpTargetVerifyPrepare(sequence_lengths_for_prepare,
                                         model_input_copy.input_lengths,
                                         model_input_copy.prefix_lengths,
                                         model_input_copy.sequence_lengths_plus_1,
                                         model_input_copy.lm_output_indexes,
                                         static_cast<int32_t>(propose_step_ + 1),
                                         cuda_graph::graphGetCurrentStream().stream());
        } else
#endif
        {
            RTP_LLM_PROFILE_SCOPE("executor.mtp.decode_step(prepare_target_verify_input_fallback)");
            model_input_copy.input_lengths =
                torch::full({static_cast<int64_t>(batch_size)}, static_cast<int64_t>(propose_step_ + 1), cuda_i32);
            model_input_copy.lm_output_indexes = torch::arange(0,
                                                               static_cast<int64_t>(batch_size * (propose_step_ + 1)),
                                                               static_cast<int64_t>(propose_step_ + 1),
                                                               cuda_i32);
            const auto& sequence_lengths =
                sequence_lengths_for_prepare.defined() ? sequence_lengths_for_prepare : model_input.sequence_lengths;
            model_input_copy.prefix_lengths          = toCudaInt32WithHostHold(sequence_lengths, buffer_holder_);
            model_input_copy.sequence_lengths_plus_1 = model_input_copy.prefix_lengths + 1;
        }
    }
    model_input_copy.clearLastHiddenStates();
    model_input_copy.sequence_lengths =
        torch::empty({0}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    model_input_copy.is_target_verify = true;
    ensureModelInputsOnCuda(model_input_copy, "decode.target_prepare");

    auto input_ready_event = std::make_shared<torch::Event>(cuda_graph::makeGraphEvent());
    input_ready_event->record(cuda_graph::graphGetCurrentStream());
    target_verify_prepare_runner_.launch(
        [this, input_ready_event, model_input_copy = std::move(model_input_copy)]() mutable {
            RTP_LLM_PROFILE_SCOPE("executor.mtp.decode_step(target_verify_prepare_attention_inputs)");
            {
                RTP_LLM_PROFILE_SCOPE("executor.mtp.decode_step(target_verify_prepare_wait_input)");
                input_ready_event->block(cuda_graph::graphGetCurrentStream());
            }
            checkModelInputsOnCuda(model_input_copy, "decode.target_prepare.forwarded");
            {
                RTP_LLM_PROFILE_SCOPE("executor.mtp.decode_step(target_verify_prepare_model_inputs)");
                model_->prepareAttentionInputs(model_input_copy);
            }
        });
}

void MtpExecutor::launchDraftPrefillPrepareAsync(const GptModelInputs& model_input) {
    if (!useAsyncPrepare()) {
        return;
    }
    if (shouldSkipFakeStreamForStop(model_input, "draft prefill async prepare")) {
        return;
    }
    if (isCpContextRequest(parallelism_config_, model_input)) {
        // CP rewrites the input inside PyWrappedModel::forward. Preparing here
        // would capture metadata from the pre-CP layout.
        RTP_LLM_LOG_DEBUG("[MTP decode] skip draft-prefill async prepare for CP context request");
        return;
    }
    const auto& mtp_cache_cfg = cache_manager_->getMTPModuleCacheConfig(0);
    // AsyncRunner value-captures model_input on its own stream/thread, so later
    // main-stream mutations cannot affect draft prefill prepare.
    auto* prefill_model    = sp_prefill_draft_model_ ? sp_prefill_draft_model_.get() : draft_model_.get();
    auto  model_input_copy = model_input;
    // Async prepare runs before runDraftPrefillForward(), so mark the copied
    // input with the same explicit recurrent draft-prefill phase.
    model_input_copy.is_mtp_draft_prefill      = true;
    model_input_copy.kv_block_stride_bytes     = mtp_cache_cfg.kv_block_stride_bytes;
    model_input_copy.kv_scale_stride_bytes     = mtp_cache_cfg.kv_scale_stride_bytes;
    model_input_copy.use_opaque_kv_cache_store = mtp_cache_cfg.use_opaque_kv_cache_store;
    model_input_copy.kv_cache_layer_to_group   = draft_kv_cache_layer_to_group;
    ensureModelInputsOnCuda(model_input_copy, "decode.draft_prefill_prepare");
    auto input_ready_event = std::make_shared<torch::Event>(cuda_graph::makeGraphEvent());
    input_ready_event->record(cuda_graph::graphGetCurrentStream());
    draft_prefill_prepare_runner_.launch(
        [this, prefill_model, input_ready_event, model_input_copy = std::move(model_input_copy)]() mutable {
            RTP_LLM_PROFILE_SCOPE("executor.mtp.decode_step(prepare_draft_prefill_input)");
            input_ready_event->block(cuda_graph::graphGetCurrentStream());
            if (shouldSkipFakeStreamForStop(model_input_copy, "draft prefill async prepare")) {
                return;
            }
            checkModelInputsOnCuda(model_input_copy, "decode.draft_prefill_prepare.forwarded");
            prefill_model->prepareAttentionInputs(model_input_copy);
        });
}

void MtpExecutor::waitPreviousBookkeepingAndKvSwaps(const std::list<GenerateStreamPtr>& streams) {
    // Cap outstanding stream-async bookkeeping to one step unless DROP_BROAD_SYNC
    // is on. Device state handles host staleness; swap events handle linear KV.
    if (useStreamAsync() && !useDropBroadSync()) {
        RTP_LLM_PROFILE_SCOPE_DYNAMIC("executor.mtp.decode_step(wait_prev_bookkeeping,stream_count=%zu)",
                                      streams.size());
        spec_bookkeeping_runner_.sync(cuda_graph::graphGetCurrentStream());
    } else if (useStreamAsync()) {
        // DROP_BROAD_SYNC: skip CPU wait but still ensure GPU stream ordering.
        // The bookkeeping runner may have launched GPU kernels (D2H staging,
        // block table updates) on its own stream; the compute stream must wait
        // for those before reading the same buffers in forward().
        RTP_LLM_PROFILE_SCOPE("executor.mtp.decode_step(stream_wait_prev_bookkeeping)");
        spec_bookkeeping_runner_.streamWait(cuda_graph::graphGetCurrentStream());
    }

    // Linear attention may rewrite KV mappings via swapLinearBlocks; wait on
    // producer events before target verify reads KV, even when broad sync is off.
    {
        RTP_LLM_PROFILE_SCOPE_DYNAMIC("executor.mtp.decode_step(wait_pending_linear_attn_swaps,stream_count=%zu)",
                                      streams.size());
        for (auto& stream : streams) {
            auto event_handle = stream->getPendingSwapDoneEvent();
            if (event_handle) {
                auto event = std::static_pointer_cast<torch::Event>(event_handle);
                event->block(cuda_graph::graphGetCurrentStream());
                stream->clearPendingSwapDoneEvent();
            }
        }
    }
}

GptModelOutputs MtpExecutor::runTargetVerifyForward(GptModelInputs& model_input, const StreamGroups& stream_groups) {
    RTP_LLM_PROFILE_SCOPE("executor.mtp.decode_step(target_model_verify)");
    maybePrintModelInput(model_input, "decode target model");
    model_input.is_target_verify        = true;
    model_input.mtp_iteration_step      = -1;
    model_input.kv_cache_layer_to_group = target_kv_cache_layer_to_group;
    RTP_LLM_LOG_DEBUG(
        "[MTP decode] target model verify forward start, input_lengths_size=%ld, prefix_lengths_size=%ld, seq_lengths_size=%ld",
        model_input.input_lengths.size(0),
        model_input.prefix_lengths.size(0),
        model_input.sequence_lengths.size(0));

    // Linear-attention only: page table advances every token. Standard paged
    // attention (MHA/MLA) page table rarely changes within a propose+verify
    // cycle, so the re-gather is skipped there.
    if (is_linear_attention_model_) {
        RTP_LLM_PROFILE_SCOPE("executor.mtp.decode_step(update_kv_cache_kernel_block_id)");
        spec_bookkeeping_runner_.sync(cuda_graph::graphGetCurrentStream());

        if (tp_rank_ == 0) {
            model_input.kv_cache_kernel_block_id =
                batch_stream_processor_->gatherKvCacheKernelBlockId(stream_groups, buffer_holder_).value();
        }

        if (parallelism_config_.tp_size > 1) {
            execBroadcast({{model_input.kv_cache_kernel_block_id}, 0});
        }

        // Focused refresh of device block tables and graph-held buffers,
        // skipping unrelated prepareAttentionInputs work.
        model_->updateKVCacheKernelBlockId(model_input);
    }

    ensureModelInputsOnCuda(model_input, "decode.target_verify_forward");
    GptModelOutputs model_output = model_->forward(model_input);
    RTP_LLM_LOG_DEBUG("[MTP decode] target model verify forward end");
    model_input.is_target_verify = false;
    return model_output;
}

SpecLogitsVerifyRunner::LaunchResult
MtpExecutor::buildSpecLogitsVerifyInline(const std::list<GenerateStreamPtr>& streams,
                                         const torch::Tensor&                draft_tokens,
                                         std::shared_ptr<torch::Event>       draft_tokens_ready_event) {
    SpecLogitsVerifyRunner::LaunchTask task;
    task.total_streams            = streams.size();
    task.propose_step             = static_cast<int>(verifySteps());
    task.vocab_size               = vocab_size_;
    task.draft_tokens             = draft_tokens;
    task.draft_tokens_ready_event = std::move(draft_tokens_ready_event);

    size_t stream_idx = 0;
    for (const auto& stream : streams) {
        size_t processor_idx = 0;
        for (const auto& processor : stream->getAllLogitsProcessorPtr()) {
            auto spec_processor = std::dynamic_pointer_cast<SpecLogitsProcessor>(processor);
            if (spec_processor) {
                task.active.push_back({spec_processor,
                                       stream_idx,
                                       processor_idx,
                                       static_cast<uint64_t>(stream->streamId()),
                                       static_cast<int64_t>(stream->seqLength()),
                                       static_cast<int64_t>(stream->outputTokenLen())});
            }
            ++processor_idx;
        }
        ++stream_idx;
    }

    if (task.active.empty()) {
        return {};
    }
    return spec_logits_verify_runner_->buildInline(task);
}

void MtpExecutor::broadcastPostRejectionInputs(GptModelInputs&     model_input,
                                               const StreamGroups& stream_groups,
                                               bool                broadcast_hidden_states) {
    RTP_LLM_PROFILE_SCOPE("executor.mtp.decode_step(tp_sync_post_rejection)");
    const auto& mtp_cache_cfg = cache_manager_->getMTPModuleCacheConfig(0);
    // Broadcast only fields updated after rejection sampling. They are all
    // device-resident, so this stays NCCL-only and rank 0's rejection-sampled
    // view replaces non-root local target-verify outputs.
    if (parallelism_config_.tp_size > 1) {
        execBroadcast({{model_input.combo_tokens}, 0});
        execBroadcast({{model_input.input_lengths}, 0});
        if (broadcast_hidden_states) {
            RTP_LLM_CHECK_WITH_INFO(model_input.last_hidden_states_layout == MtpHiddenStatesLayout::GLOBAL,
                                    "only GLOBAL MTP hidden states may be TP-broadcast, got layout=%s",
                                    mtpHiddenStatesLayoutName(model_input.last_hidden_states_layout));
            execBroadcast({{model_input.last_hidden_states}, 0});
        } else {
            RTP_LLM_CHECK_WITH_INFO(model_input.last_hidden_states_layout == MtpHiddenStatesLayout::CP_LOCAL,
                                    "rank-local MTP hidden states must remain CP_LOCAL, got layout=%s",
                                    mtpHiddenStatesLayoutName(model_input.last_hidden_states_layout));
        }
        execBroadcast({{model_input.lm_output_indexes}, 0});
    }
    if (model_input.combo_tokens.defined() && model_input.lm_output_indexes.defined()) {
        const auto all_streams = stream_groups.allStreams();
        if (!all_streams.empty()) {
            auto target_token_gpu =
                model_input.combo_tokens.index_select(/*dim=*/0, model_input.lm_output_indexes.to(torch::kLong))
                    .to(torch::kInt32);
            int64_t batch_idx = 0;
            for (const auto& stream : all_streams) {
                auto sp_output_buffer = stream->getSPOutputBuffer();
                if (sp_output_buffer) {
                    sp_output_buffer->target_token_gpu = target_token_gpu.narrow(0, batch_idx, 1);
                }
                ++batch_idx;
            }
        }
    }
    model_input.kv_block_stride_bytes     = mtp_cache_cfg.kv_block_stride_bytes;
    model_input.kv_scale_stride_bytes     = mtp_cache_cfg.kv_scale_stride_bytes;
    model_input.use_opaque_kv_cache_store = mtp_cache_cfg.use_opaque_kv_cache_store;
    model_input.kv_cache_layer_to_group   = draft_kv_cache_layer_to_group;
}

GptModelOutputs MtpExecutor::runDSparkProposeForward(GptModelInputs& model_input) {
    RTP_LLM_CHECK_WITH_INFO(is_dspark_, "DSpARK proposal forward requires SP_TYPE_DSPARK");
    RTP_LLM_CHECK_WITH_INFO(draft_model_ != nullptr, "DSpARK proposal model is not initialized");
    maybePrintModelInput(model_input, "decode dspark propose model");
    ensureModelInputsOnCuda(model_input, "decode.dspark_propose_forward");
    return draft_model_->forward(model_input);
}

SamplerOutput MtpExecutor::sampleDSparkDraft(const StreamGroups&  stream_groups,
                                             const torch::Tensor& base_logits,
                                             const torch::Tensor& anchors) {
    RTP_LLM_CHECK_WITH_INFO(is_dspark_, "DSpARK draft sampling requires SP_TYPE_DSPARK");
    const auto         batch_size = static_cast<int64_t>(stream_groups.size());
    std::vector<float> temperatures(batch_size);
    int64_t            row                  = 0;
    constexpr float    kMinDraftTemperature = 1.0e-6f;
    for (const auto& stream : stream_groups.allStreams()) {
        RTP_LLM_CHECK_WITH_INFO(stream->maxBatchSize() == 1 && !stream->hasNumBeams(),
                                "DSpARK does not support tiled or beam sampling");
        const auto config      = stream->generateConfig();
        float      temperature = config->temperature;
        if (!std::isfinite(temperature) || temperature < 0.0f) {
            RTP_LLM_LOG_WARNING("DSpARK received invalid sampling temperature=%g for stream=%ld; using 1.0",
                                temperature,
                                stream->streamId());
            temperature = 1.0f;
        }
        temperatures[row++] = !config->top1() ? std::max(temperature, kMinDraftTemperature) : kMinDraftTemperature;
    }
    // Reuse immutable values only on their producing stream. A miss replaces
    // the tensors instead of overwriting storage still used by async sampling.
    const auto sampling_stream_id = cuda_graph::graphGetCurrentStream().id();
    if (!dspark_temperature_gpu_.defined() || dspark_temperature_gpu_.device() != base_logits.device()
        || dspark_temperature_stream_id_ != sampling_stream_id || dspark_temperatures_ != temperatures) {
        auto temperature_cpu =
            torch::empty({batch_size}, torch::TensorOptions().dtype(torch::kFloat32).pinned_memory(true));
        std::copy(temperatures.begin(), temperatures.end(), temperature_cpu.data_ptr<float>());
        buffer_holder_.hold_host(temperature_cpu);
        auto temperature_gpu          = temperature_cpu.to(base_logits.device(), /*non_blocking=*/true);
        dspark_temperature_gpu_       = std::move(temperature_gpu);
        dspark_temperatures_          = std::move(temperatures);
        dspark_temperature_stream_id_ = sampling_stream_id;
    }
    return speculative_sampler_->sampleDSparkDraft(base_logits,
                                                   anchors,
                                                   dspark_temperature_gpu_,
                                                   dspark_markov_w1_,
                                                   dspark_markov_w2_,
                                                   draft_vocab_size_,
                                                   verifySteps());
}

GptModelOutputs MtpExecutor::runDraftPrefillForward(GptModelInputs& model_input) {
    // FIX: always use sp_prefill_draft_model_ when it exists (Method B), so
    // every DP rank dispatches mega_moe on the SAME cloned _mega_buf B every
    // step. The previous gate `!model_input.is_fake_stream` sent fake-stream
    // ranks down draft_model_ (original _mega_buf A) while real-stream peers
    // went down sp_prefill_draft_model_ (cloned _mega_buf B), and the
    // peer-symmetric NVLink barrier in
    // deep_gemm/include/deep_gemm/comm/barrier.cuh trapped after timeout
    // because each rank's counter advanced on a buffer the other rank never
    // touched.
    //
    // Fake decode streams are constructed with propose_step+1 tokens — the
    // same shape as a real target-verify output — so the captured CUDA graph
    // for seq_len=propose_step+1 replays correctly for both fake and real
    // inputs. No per-step cross-rank synchronization is required: each
    // mega_moe call is itself a NVLink collective on buf B and provides its
    // own intra-kernel barrier between peers.
    //
    // See glm5_pd_sep_mtp_nvlink_barrier_crash_debug.md §12 for the
    // empirical trace and earlier Method 6.1 (DP AllReduce) attempt.
    const bool use_sp_prefill_cuda_graph    = sp_prefill_draft_model_ != nullptr;
    const bool previous_draft_prefill_phase = model_input.is_mtp_draft_prefill;
    model_input.is_mtp_draft_prefill        = true;
    model_input.mtp_iteration_step          = 0;
    RTP_LLM_PROFILE_SCOPE_DYNAMIC(
        "executor.mtp.decode_step(draft_model_forward,use_sp=%d,sp_cg=%d,sp_prefill_cg=%d,is_fake=%d)",
        static_cast<int>(use_sp_prefill_cuda_graph),
        static_cast<int>(sp_prefill_draft_model_ ? sp_prefill_draft_model_->cudaGraphEnabled() : false),
        static_cast<int>(sp_prefill_draft_model_ ? sp_prefill_draft_model_->prefillCudaGraphMode() : false),
        static_cast<int>(model_input.is_fake_stream));
    maybePrintModelInput(model_input, "decode post draft model");
    ensureModelInputsOnCuda(model_input, "decode.draft_prefill_forward");
    const bool cp_context_request = isCpContextRequest(parallelism_config_, model_input);
    // Use sp_prefill_draft_model_ if CUDA graph is enabled, otherwise use draft_model_.
    GptModelOutputs draft_prefill_model_output;
    if (use_sp_prefill_cuda_graph) {
        draft_prefill_model_output = sp_prefill_draft_model_->forward(model_input);
        if (!cp_context_request) {
            maybeOverrideLastHiddenWithMtpBuffer(draft_prefill_model_output, *sp_prefill_draft_model_);
        }
        draft_model_->copyMtpIterationTopkCacheFrom(*sp_prefill_draft_model_);
    } else {
        draft_prefill_model_output = draft_model_->forward(model_input);
        if (!cp_context_request) {
            maybeOverrideLastHiddenWithMtpBuffer(draft_prefill_model_output, *draft_model_);
        }
    }
    model_input.is_mtp_draft_prefill = previous_draft_prefill_phase;
    model_input.mtp_iteration_step   = -1;
    return draft_prefill_model_output;
}

void MtpExecutor::collectDecodeMetrics(const StreamGroups& stream_groups, MtpMetricsCollector& metrics_collector) {
    RTP_LLM_PROFILE_SCOPE("executor.mtp.decode_step(collect_metrics)");
    auto& executor_collector  = metrics_collector.executor_collector;
    auto& sp_engine_collector = metrics_collector.sp_engine_collector;

    const auto accept_len_metrics = consumePendingAcceptLenMetrics();
    if (is_dspark_ && accept_len_metrics.valid) {
        // Same-step, already-host-visible snapshot. Correlate this child range
        // with the enclosing decode scope; do not add transfers or waits.
        RTP_LLM_PROFILE_SCOPE_DYNAMIC("executor.mtp.decode_step(acceptance,sum=%lld,streams=%lld,verify_steps=%zu)",
                                      static_cast<long long>(accept_len_metrics.total_accept_len),
                                      static_cast<long long>(accept_len_metrics.total_stream_num),
                                      verifySteps());
    }
    const int64_t total_accept_len         = accept_len_metrics.total_accept_len;
    executor_collector.generate_batch_size = stream_groups.totalModelBatchSize();
    executor_collector.execute_token_size += total_accept_len;
    executor_collector.max_seq_len = stream_groups.maxSeqLen();

    executor_collector.context_batch_size_when_has_context = executor_collector.context_batch_size;
    executor_collector.execute_token_size_when_has_context = executor_collector.execute_token_size;
    executor_collector.max_seq_len_when_has_context        = executor_collector.max_seq_len;

    sp_engine_collector.total_accepted_token_num = total_accept_len;
    sp_engine_collector.total_stream_num         = accept_len_metrics.total_stream_num;
    sp_engine_collector.total_propose_token_num  = accept_len_metrics.total_propose_token_num;
    sp_engine_collector.spec_steps               = propose_step_;
}

absl::Status MtpExecutor::dispatchDecodeOutput(const StreamGroups&                          stream_groups,
                                               MtpMetricsCollector&                         metrics_collector,
                                               const std::list<GenerateStreamPtr>&          streams,
                                               const speculative::SpeculativeSamplerOutput& speculative_sampler_output,
                                               GptModelOutputs                              draft_prefill_model_output,
                                               SamplerOutput                 draft_prefill_sampler_output,
                                               std::shared_ptr<torch::Event> rejection_event,
                                               std::shared_ptr<torch::Event> draft_event) {
    auto dispatch_output = [&] {
        RTP_LLM_PROFILE_SCOPE("executor.mtp.decode_step(dispatch_output)");
        absl::Status result;
        if (useStreamAsync()) {
            // Hand off to a worker that waits on main-stream rejection/draft events
            // via cudaStreamWaitEvent; the main thread returns immediately.
            result =
                dispatchDecodeAsync(stream_groups,
                                    speculative_sampler_output,
                                    {std::move(draft_prefill_model_output), std::move(draft_prefill_sampler_output)},
                                    std::move(rejection_event),
                                    std::move(draft_event));
        } else {
            MergedOutput draft_prefill_output{std::move(draft_prefill_model_output),
                                              std::move(draft_prefill_sampler_output)};
            result = batch_stream_processor_->dispatchDecode(
                stream_groups, speculative_sampler_output, draft_prefill_output);
            if (result.ok()) {
                publishSyncMtpDeviceState(stream_groups, speculative_sampler_output, draft_prefill_output);
            }
        }
        return result;
    };

    if (is_dspark_ && useStreamAsync() && metrics_reporter_) {
        absl::Status status;
        try {
            status = dispatch_output();
        } catch (...) {
            const auto original = std::current_exception();
            try {
                collectDecodeMetrics(stream_groups, metrics_collector);
            } catch (const std::exception& error) {
                RTP_LLM_LOG_ERROR("[DSpark] metrics cleanup after dispatch failure: %s", error.what());
            } catch (...) {
                RTP_LLM_LOG_ERROR("[DSpark] metrics cleanup after dispatch failure: unknown exception");
            }
            std::rethrow_exception(original);
        }
        // Consume exactly once even for non-OK status; keep metrics failures
        // outside the dispatch catch so accepted-worker cleanup is not retried.
        collectDecodeMetrics(stream_groups, metrics_collector);
        return status;
    }

    if (metrics_reporter_) {
        collectDecodeMetrics(stream_groups, metrics_collector);
    }
    return dispatch_output();
}

void MtpExecutor::releaseAllModelBuffers() {
    // TensorHolder release point (MtpExecutor phase boundary): after the current
    // TP sync/model-input preparation has consumed staged H2D sources, advance
    // the hold window for executor-owned model/sampler staging tensors.
    buffer_holder_.release();
    // PyWrappedModel TensorHolder release points for target/draft model-internal
    // staging buffers.
    model_->releaseBuffers();
    draft_model_->releaseBuffers();
    if (sp_prefill_draft_model_) {
        sp_prefill_draft_model_->releaseBuffers();
    }
}

void MtpExecutor::prepareStreams(const std::list<GenerateStreamPtr>& streams,
                                 std::list<GenerateStreamPtr>&       prefill_streams,
                                 std::list<GenerateStreamPtr>&       decode_streams) {
    RTP_LLM_PROFILE_SCOPE_DYNAMIC("executor.mtp.prepare_streams(stream_size=%zu)", streams.size());

    for (auto& stream : streams) {
        // split streams into prefill and decode
        if (stream->isContextStream()) {
            prefill_streams.push_back(stream);
        } else {
            stream->setScoreLen(verifySteps() + 1);
            if (stream->getSPOutputBuffer() == nullptr && stream->isPerfTest()) {
                auto sp_output_buffer =
                    makeFakeSPOutputBuffer(data_type_, hidden_size_, draft_vocab_size_, propose_step_);
                stream->setSPOutputBuffer(sp_output_buffer);
            }
            decode_streams.push_back(stream);
        }

        // set base properties
        stream->setReturnAllProbs(ReturnAllProbsMode::DEFAULT);
        if (stream->getSPOutputBuffer() == nullptr) {
            const auto cuda_i32         = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
            auto       sp_output_buffer = std::make_shared<SpeculativeExecutorStreamOutput>();
            sp_output_buffer->tokens    = torch::zeros({1, 2}, torch::kInt32);
            // Pre-allocate device mirrors so ensureSpOutputTokenGpuMirrors() is a
            // no-op in steady-state (no pageable H2D + sync per stream per step).
            sp_output_buffer->target_token_gpu   = torch::zeros({1}, cuda_i32);
            sp_output_buffer->propose_tokens_gpu = torch::zeros({1}, cuda_i32);

            stream->setSPOutputBuffer(sp_output_buffer);
        }

        // set propose_step
        auto sp_output_buffer          = stream->getSPOutputBuffer();
        sp_output_buffer->propose_step = propose_step_;
        ensureSpOutputTokenGpuMirrors(sp_output_buffer, !is_dspark_);
    }
}

std::list<GenerateStreamPtr> MtpExecutor::acquireKvExecutionStreams(const std::list<GenerateStreamPtr>& streams,
                                                                    std::vector<GenerateStreamPtr>&     leases) {
    std::list<GenerateStreamPtr> active;
    bool                         had_context = false, had_decode = false;
    bool                         live_context = false, live_decode = false;
    for (const auto& stream : streams) {
        const bool context = stream->isContextStream();
        had_context |= context;
        had_decode |= !context;
        if (!stream->isFakeStream()) {
            // This must precede prepareStreams, metadata reads and GPU work.
            if (!stream->tryAcquireKvExecution()) {
                continue;
            }
            leases.push_back(stream);
        }
        active.push_back(stream);
        live_context |= context;
        live_decode |= !context;
    }
    // Preserve a globally agreed EP phase, but never add fake rows beside live
    // rows of that same phase: StreamGroups treats the entire phase as fake.
    if (parallelism_config_.dp_size > 1 && isTpRank0()) {
        if (had_context && !live_context) {
            active.push_back(make_kv_safe_fake_stream_(true));
        }
        if (had_decode && !live_decode) {
            active.push_back(make_kv_safe_fake_stream_(false));
        }
    }
    return active;
}

absl::Status MtpExecutor::process(const std::list<GenerateStreamPtr>& streams, int64_t schedule_time_us) {
    if (!is_dspark_ || !useStreamAsync() || warm_up_) {
        return processImpl(streams, schedule_time_us);
    }
    return processWithKvLease(streams, schedule_time_us);
}

absl::Status MtpExecutor::processWithKvLease(const std::list<GenerateStreamPtr>& streams, int64_t schedule_time_us) {
    std::vector<GenerateStreamPtr> leases;
    leases.reserve(streams.size());
    const torch::Stream compute_stream = cuda_graph::graphGetCurrentStream();
    const auto          producer       = compute_stream.hash();
    try {
        auto active = acquireKvExecutionStreams(streams, leases);

        auto status = processImpl(active, schedule_time_us);
        if (!leases.empty()) {
            auto done = std::make_shared<torch::Event>(cuda_graph::makeGraphEvent());
            done->record(compute_stream);
            const auto wait = [done, compute_stream] {
                cuda_graph::GraphStreamGuard guard(cuda_graph::toGraphStream(compute_stream));
                done->synchronize();
            };
            // Worker claims are already registered, or synchronous dispatch has
            // completed. Publish the GPU fence before dropping each execution hold.
            while (!leases.empty()) {
                leases.back()->finishKvExecution(producer, wait);
                leases.pop_back();
            }
        }
        return status;
    } catch (...) {
        const auto         original = std::current_exception();
        std::exception_ptr drain_failure;
        // Exceptional-only drains, including workers that threw before recording
        // their event. Keep every lease until all producers have stopped using KV.
        for (auto* runner : {&target_verify_prepare_runner_,
                             &draft_prefill_prepare_runner_,
                             &spec_logits_verify_async_runner_,
                             &spec_bookkeeping_runner_}) {
            try {
                (void)runner->joinAndDrain();
            } catch (...) {
                if (!drain_failure) {
                    drain_failure = std::current_exception();
                }
            }
        }
        // PD forward may have submitted per-layer store readers before throwing.
        // Their tensor owners do not pin allocator pages. Drain all model-owned
        // readers/callbacks while every execution lease is still held.
        for (auto* model : {model_.get(), draft_model_.get(), sp_prefill_draft_model_.get()}) {
            if (!model) {
                continue;
            }
            try {
                model->drainPendingCacheStore();
            } catch (...) {
                if (!drain_failure) {
                    drain_failure = std::current_exception();
                }
            }
        }
        try {
            compute_stream.synchronize();
        } catch (...) {
            if (!drain_failure) {
                drain_failure = std::current_exception();
            }
        }
        for (const auto& stream : leases) {
            if (drain_failure) {
                // Failed completion cannot authorize allocator reuse.
                stream->quarantineKvExecution();
                stream->finishKvExecution(producer, [drain_failure] { std::rethrow_exception(drain_failure); });
                stream->releaseResource();
            } else {
                stream->finishKvExecution(producer);
            }
        }
        std::rethrow_exception(drain_failure ? drain_failure : original);
    }
}

absl::Status MtpExecutor::processImpl(const std::list<GenerateStreamPtr>& streams, int64_t schedule_time_us) {
    RTP_LLM_PROFILE_SCOPE_DYNAMIC("executor.mtp.process(stream_size=%zu,mtp_step=%zu)", streams.size(), propose_step_);

    const int64_t process_start_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
    if (schedule_time_us <= 0) {
        schedule_time_us = process_start_time_us;
    }
    MtpMetricsCollector metrics_collector;
    auto                tps_active_guard =
        tps_reporter_.makeActiveGuard(metrics_reporter_ && isTpRank0() && !warm_up_ && !streams.empty());
    auto wall_tps_active_guard =
        wall_tps_reporter_.makeActiveGuard(metrics_reporter_ && isTpRank0() && !warm_up_ && !streams.empty());

    std::list<GenerateStreamPtr> prefill_streams;
    std::list<GenerateStreamPtr> decode_streams;

    prepareStreams(streams, prefill_streams, decode_streams);

    // step forward
    int64_t start_time_us = autil::TimeUtility::currentTimeInMicroSeconds();

    if (role_type_ == RoleType::PREFILL || role_type_ == RoleType::PDFUSION) {
        THROW_IF_STATUS_ERROR(prefillStep(prefill_streams, metrics_collector, schedule_time_us));
    }

    if (role_type_ == RoleType::DECODE || role_type_ == RoleType::PDFUSION) {
        THROW_IF_STATUS_ERROR(decodeStep(decode_streams, metrics_collector));
    }

    metrics_collector.sp_engine_collector.step_latency_us =
        autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;

    // report metrics
    if (isTpRank0() && metrics_reporter_ && metrics_collector.not_skip) {
        // decode metrics
        auto& tps_collector       = metrics_collector.tps_collector;
        auto& sp_engine_collector = metrics_collector.sp_engine_collector;
        auto  decode_time         = autil::TimeUtility::currentTimeInMicroSeconds() - schedule_time_us;
        if (sp_engine_collector.total_accepted_token_num) {
            tps_collector.addTokenSize(0,
                                       0,
                                       sp_engine_collector.total_accepted_token_num,
                                       sp_engine_collector.total_accepted_token_num,
                                       decode_time);
        }

        RTP_LLM_PROFILE_SCOPE("executor.mtp.process(report_metrics)");
        metrics_reporter_->report<RtpLLMExecutorMetrics, RtpLLMExecutorMetricsCollector>(
            nullptr, &metrics_collector.executor_collector);
        tps_reporter_.report(&metrics_collector.tps_collector);
        wall_tps_reporter_.report(&metrics_collector.tps_collector);
        metrics_reporter_->report<RtpLLMSpeculativeEngineMetrics, RtpLLMSpeculativeEngineMetricsCollector>(
            nullptr, &metrics_collector.sp_engine_collector);
    }

    return absl::OkStatus();
}

bool MtpExecutor::updateEplbConfig(const EPLBConfig& config) {
    if (expert_balancer_) {
        return expert_balancer_->updateEplbConfig(config);
    }
    return true;
}

void MtpExecutor::draftModelDecode(GptModelInputs&             model_input,
                                   const StreamGroups&         stream_groups,
                                   std::vector<torch::Tensor>& draft_probs_list,
                                   torch::Tensor&              draft_token_ids_t,
                                   int64_t&                    model_forward_us) {
    RTP_LLM_PROFILE_SCOPE_DYNAMIC("executor.mtp.draft_model_decode(batch_size=%zu)", model_input.combo_tokens.size(0));
    if (shouldSkipFakeStreamForStop(model_input, "draft decode loop")) {
        return;
    }

    const auto& mtp_cache_cfg             = cache_manager_->getMTPModuleCacheConfig(0);
    model_input.kv_block_stride_bytes     = mtp_cache_cfg.kv_block_stride_bytes;
    model_input.kv_scale_stride_bytes     = mtp_cache_cfg.kv_scale_stride_bytes;
    model_input.use_opaque_kv_cache_store = mtp_cache_cfg.use_opaque_kv_cache_store;

    GptModelOutputs            draft_decode_model_output;
    std::vector<torch::Tensor> draft_token_columns;
    torch::Tensor              spec_prefix_lengths;

    // update TP > 0 batch_size
    size_t     batch_size       = model_input.combo_tokens.size(0);
    const auto cuda_i32         = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    auto       to_cuda_i32_flat = [this, batch_size](const torch::Tensor& tensor) -> torch::Tensor {
        auto tensor_d = toCudaInt32WithHostHold(tensor, buffer_holder_);
        tensor_d      = tensor_d.reshape({static_cast<int64_t>(batch_size)});
        return tensor_d.is_contiguous() ? tensor_d : tensor_d.contiguous();
    };
    // Draft decode consumes the first proposal at the next uncommitted
    // position. Target verify starts one position earlier by replaying the last
    // committed target token, so its prefix excludes that token.
    spec_prefix_lengths = model_input.sequence_lengths.defined() ?
                              toCudaInt32WithHostHold(model_input.sequence_lengths, buffer_holder_) - 1 :
                              torch::Tensor();

    torch::Tensor pre_propose_token_t_raw;
    {
        RTP_LLM_PROFILE_SCOPE("executor.mtp.draft_model_decode(pre_propose_token)");
        // Keep the original propose token tensor alive without cloning; later
        // model_input.combo_tokens assignments do not mutate this storage.
        pre_propose_token_t_raw = to_cuda_i32_flat(model_input.combo_tokens);
    }
    const auto all_streams = stream_groups.allStreams();

    torch::Tensor pre_target_token_t;
    // Prefer device state published before the bookkeeping worker launches.
    // Batch gather: pre_target_token[i] = accept_tokens[i, accept_len[i]-1]
    {
        RTP_LLM_PROFILE_SCOPE("executor.mtp.draft_model_decode(pre_target_device_gather)");
        bool all_device_state = !all_streams.empty();
        if (all_device_state) {
            // Check all streams have device state and collect batch tensors
            std::vector<torch::Tensor> accept_tokens_slices;
            std::vector<torch::Tensor> accept_len_slices;
            accept_tokens_slices.reserve(batch_size);
            accept_len_slices.reserve(batch_size);
            for (const auto& stream : all_streams) {
                const auto& accept_tokens = stream->getAcceptTokensGpu();
                const auto& accept_len    = stream->getAcceptLenGpu();
                if (!accept_tokens.defined() || !accept_tokens.is_cuda() || !accept_len.defined()
                    || !accept_len.is_cuda()) {
                    all_device_state = false;
                    break;
                }
                accept_tokens_slices.push_back(accept_tokens.reshape({1, -1}));
                accept_len_slices.push_back(accept_len.reshape({1}));
            }
            if (all_device_state) {
                // Batch gather: [batch, propose_step+1] -> pick column [accept_len-1] per row
                auto accept_tokens_2d = torch::cat(accept_tokens_slices, 0);  // [batch, cols]
                auto accept_len_batch = torch::cat(accept_len_slices, 0);     // [batch]
                auto idx_long         = (accept_len_batch.to(torch::kInt64) - 1).reshape({(int64_t)batch_size, 1});
                pre_target_token_t =
                    accept_tokens_2d.gather(1, idx_long).reshape({(int64_t)batch_size}).to(torch::kInt32);
            }
        }
        if (!all_device_state && all_streams.empty()) {
            pre_target_token_t =
                torch::empty({(int64_t)batch_size}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
        }
    }

    if (!pre_target_token_t.defined()) {
        RTP_LLM_PROFILE_SCOPE("executor.mtp.draft_model_decode(pre_target_sp_buffer_gather)");
        std::vector<torch::Tensor> pre_target_slices_gpu;
        pre_target_slices_gpu.reserve(batch_size);
        bool all_sp_buffer_gpu = !all_streams.empty();
        for (const auto& stream : all_streams) {
            auto sp_output_buffer = stream->getSPOutputBuffer();
            if (!sp_output_buffer || !sp_output_buffer->target_token_gpu.defined()
                || !sp_output_buffer->target_token_gpu.is_cuda()) {
                all_sp_buffer_gpu = false;
                break;
            }
            pre_target_slices_gpu.push_back(sp_output_buffer->target_token_gpu.reshape({-1}));
        }
        if (all_sp_buffer_gpu && pre_target_slices_gpu.size() == batch_size && !pre_target_slices_gpu.empty()) {
            pre_target_token_t = torch::cat(pre_target_slices_gpu, 0).to(torch::kInt32);
        }
    }

    if (!pre_target_token_t.defined()) {
        // Legacy fallback for streams without MtpAsyncDeviceState, such as old
        // PD-disaggregate init paths. Unsafe while a previous worker is in
        // flight with DROP_BROAD_SYNC=1.
        RTP_LLM_PROFILE_SCOPE("executor.mtp.draft_model_decode(pre_target_host_fallback)");
        auto pre_target_token =
            torch::empty({(int64_t)batch_size}, torch::TensorOptions().dtype(torch::kInt32).pinned_memory(true));
        int batch_idx = 0;
        for (const auto& stream : all_streams) {
            int* propose_tokens                         = stream->getSPOutputBuffer()->tokens.data_ptr<int>();
            pre_target_token.data_ptr<int>()[batch_idx] = propose_tokens[0];
            batch_idx++;
        }
        pre_target_token_t = toCudaWithHostHold(pre_target_token, buffer_holder_);
    }
    draft_token_columns.push_back(to_cuda_i32_flat(pre_target_token_t));
    draft_token_columns.push_back(pre_propose_token_t_raw);

    // n-1 steps draft model decode
    for (int i = 0; i < propose_step_ - 1; i++) {
        RTP_LLM_PROFILE_SCOPE_DYNAMIC("executor.mtp.draft_model_decode(loop_iter=%d)", i);
        if (shouldSkipFakeStreamForStop(model_input, "draft decode loop forward")) {
            return;
        }
        RTP_LLM_LOG_DEBUG("[MTP draftDecode] loop step %d/%d start, batch_size %zu", i, propose_step_ - 1, batch_size);
        int64_t start_time_us          = autil::TimeUtility::currentTimeInMicroSeconds();
        model_input.mtp_iteration_step = i + 1;
        ensureModelInputsOnCuda(model_input, "draft_decode.loop_forward");
        draft_decode_model_output      = std::move(draft_model_->forward(model_input));
        model_input.mtp_iteration_step = -1;
        model_forward_us += autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
        maybeOverrideLastHiddenWithMtpBuffer(draft_decode_model_output, *draft_model_);
        RTP_LLM_LOG_DEBUG("[MTP draftDecode] loop step %d forward done", i);

        // sample
        auto fast_topk_sampler_output = fast_topk_sampler_->forward(draft_decode_model_output.logits, 1);
        auto draft_probs              = fast_topk_sampler_output.all_probs;
        auto draft_probs_reshape      = draft_probs.reshape({(int)batch_size, 1, -1});
        auto draft_token_ids          = fast_topk_sampler_output.token_ids;

        if (model_input.is_fake_stream) {
            draft_token_ids.zero_();
            draft_decode_model_output.all_hidden_states.zero_();
        }

        draft_token_ids = to_cuda_i32_flat(draft_token_ids);
        draft_token_columns.push_back(draft_token_ids);
        draft_probs_list.push_back(draft_probs_reshape);

        // update model input
        if (i != propose_step_ - 2) {
            batch_stream_processor_->updateDecodeDraftModelInput(
                model_input, draft_decode_model_output, draft_token_ids, buffer_holder_);
        }
    }

    {
        RTP_LLM_PROFILE_SCOPE("executor.mtp.draft_model_decode(build_spec_decode_input)");
        // prepare spec decode input
        const auto    tokens_per_batch = static_cast<int32_t>(propose_step_ + 1);
        torch::Tensor input_lengths;
#if USING_CUDA
        if (tokens_per_batch <= 8) {
            RTP_LLM_PROFILE_SCOPE("executor.mtp.draft_model_decode(build_spec_tokens_metadata_fused)");
            draft_token_ids_t =
                torch::empty({static_cast<int64_t>(batch_size), static_cast<int64_t>(tokens_per_batch)}, cuda_i32);
            input_lengths = torch::empty({static_cast<int64_t>(batch_size)}, cuda_i32);
            model_input.lm_output_indexes =
                torch::empty({static_cast<int64_t>(batch_size * tokens_per_batch)}, cuda_i32);
            invokeMtpSpecDecodeTokensMetadataPrepare(draft_token_columns,
                                                     draft_token_ids_t,
                                                     input_lengths,
                                                     model_input.lm_output_indexes,
                                                     tokens_per_batch,
                                                     at::cuda::getCurrentCUDAStream().stream());
        } else {
            RTP_LLM_PROFILE_SCOPE("executor.mtp.draft_model_decode(build_spec_cat_tokens)");
            draft_token_ids_t = torch::stack(draft_token_columns, 1).contiguous();
            {
                RTP_LLM_PROFILE_SCOPE("executor.mtp.draft_model_decode(build_spec_metadata_fused)");
                input_lengths = torch::empty({static_cast<int64_t>(batch_size)}, cuda_i32);
                model_input.lm_output_indexes =
                    torch::empty({static_cast<int64_t>(batch_size * tokens_per_batch)}, cuda_i32);
                invokeMtpSpecDecodeMetadataPrepare(input_lengths,
                                                   model_input.lm_output_indexes,
                                                   tokens_per_batch,
                                                   at::cuda::getCurrentCUDAStream().stream());
            }
        }
#else
        {
            RTP_LLM_PROFILE_SCOPE("executor.mtp.draft_model_decode(build_spec_lengths_indexes)");
            draft_token_ids_t = torch::stack(draft_token_columns, 1).contiguous();
            input_lengths     = torch::full({(int64_t)batch_size}, static_cast<int64_t>(propose_step_ + 1), cuda_i32);
            model_input.lm_output_indexes =
                torch::arange(0, static_cast<int64_t>(batch_size * (propose_step_ + 1)), cuda_i32);
        }
#endif

        model_input.input_lengths    = std::move(input_lengths);
        model_input.prefix_lengths   = spec_prefix_lengths;
        model_input.combo_tokens     = draft_token_ids_t.reshape({(int64_t)(batch_size * (propose_step_ + 1))});
        model_input.sequence_lengths = torch::empty({0}, torch::TensorOptions(torch::kInt32).device(torch::kCUDA));
        model_input.clearLastHiddenStates();
        ensureModelInputsOnCuda(model_input, "draft_decode.build_spec_decode_input");

        // Since other tp ranks don't have streams, its combo_tokens' first token is not correct.
        // Thus, we need to broadcast the combo_tokens to other tp ranks.
        if (parallelism_config_.tp_size > 1) {
            RTP_LLM_PROFILE_SCOPE("executor.mtp.draft_model_decode(build_spec_tp_broadcast)");
            execBroadcast({{model_input.combo_tokens}, 0});
        }

        const auto& cache_cfg                 = cache_manager_->cacheConfig();
        model_input.kv_block_stride_bytes     = cache_cfg.kv_block_stride_bytes;
        model_input.kv_scale_stride_bytes     = cache_cfg.kv_scale_stride_bytes;
        model_input.use_opaque_kv_cache_store = cache_cfg.use_opaque_kv_cache_store;
    }
}

bool MtpExecutor::useStreamAsync() const {
    static const bool logged = []() {
        logCachedEnvFlag(kStreamAsyncFlag);
        return true;
    }();
    (void)logged;
    return kStreamAsyncFlag.on;
}

bool MtpExecutor::useAsyncDeviceState() const {
    static const bool logged = []() {
        logCachedEnvFlag(kAsyncDeviceStateFlag);
        return true;
    }();
    (void)logged;
    return kAsyncDeviceStateFlag.on;
}

bool MtpExecutor::useAsyncPrepare() const {
    static const bool logged = []() {
        logCachedEnvFlag(kAsyncPrepareFlag);
        return true;
    }();
    (void)logged;
    return kAsyncPrepareFlag.on;
}

bool MtpExecutor::useDropBroadSync() const {
    static const bool logged = []() {
        logCachedEnvFlag(kDropBroadSyncFlag);
        return true;
    }();
    (void)logged;
    return kDropBroadSyncFlag.on;
}

void MtpExecutor::publishSyncMtpDeviceState(const StreamGroups&                          stream_groups,
                                            const speculative::SpeculativeSamplerOutput& spec_decode_output,
                                            const MergedOutput&                          draft_prefill_output) {
    RTP_LLM_PROFILE_SCOPE("executor.mtp.decode_step(publish_sync_mtp_device_state)");

    auto all_streams = stream_groups.allStreams();
    if (all_streams.empty()) {
        return;
    }

    const auto batch_size  = static_cast<int64_t>(all_streams.size());
    auto       to_cuda_i32 = [this](const torch::Tensor& tensor) -> torch::Tensor {
        return toCudaInt32WithHostHold(tensor, buffer_holder_);
    };

    torch::Tensor accept_len_all     = to_cuda_i32(spec_decode_output.accept_len);
    torch::Tensor accept_tokens_all  = to_cuda_i32(spec_decode_output.accept_tokens);
    torch::Tensor propose_tokens_all = to_cuda_i32(draft_prefill_output.sampler_output.token_ids);
    torch::Tensor draft_all_probs_full =
        draft_prefill_output.sampler_output.all_probs.defined() ?
            toCudaWithHostHold(draft_prefill_output.sampler_output.all_probs, buffer_holder_) :
            torch::Tensor();
    torch::Tensor draft_all_hidden_full =
        draft_prefill_output.model_output.all_hidden_states.defined() ?
            toCudaWithHostHold(draft_prefill_output.model_output.all_hidden_states, buffer_holder_) :
            torch::Tensor();

    if (!accept_len_all.defined() || !accept_tokens_all.defined()) {
        RTP_LLM_LOG_WARNING(
            "[mtp-device-state] skip sync publish: accept_len/accept_tokens undefined, stream_count=%zu",
            all_streams.size());
        return;
    }

    // Batch compute next_seq_len from host seqLength (sync path: host is authoritative)
    const auto pin_i32          = torch::TensorOptions().dtype(torch::kInt32).pinned_memory(true);
    const auto cuda_i32         = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    auto       next_seq_len_cpu = torch::empty({batch_size}, pin_i32);
    {
        int64_t i = 0;
        for (const auto& stream : all_streams) {
            next_seq_len_cpu.data_ptr<int32_t>()[i] = static_cast<int32_t>(stream->seqLength());
            ++i;
        }
    }
    buffer_holder_.hold_host(next_seq_len_cpu);
    auto next_seq_len_owned = torch::empty({batch_size}, cuda_i32);
    next_seq_len_owned.copy_(next_seq_len_cpu, /*non_blocking=*/true);

    // Batch gather the hidden row selected by each request's accepted length.
    torch::Tensor last_hidden_all;
    const auto    stream_hidden_len = static_cast<int64_t>(propose_step_ + 1);
    if (propose_step_ > 1 && !is_dspark_ && draft_all_hidden_full.defined()) {
        const auto hidden_size = draft_all_hidden_full.size(1);
        auto       hidden_3d   = draft_all_hidden_full.reshape({batch_size, stream_hidden_len, hidden_size});
        auto       accept_i32  = accept_len_all.to(torch::kInt32);
        auto       idx_long =
            (accept_i32.to(torch::kInt64) - 1).reshape({batch_size, 1, 1}).expand({batch_size, 1, hidden_size});
        last_hidden_all = hidden_3d.gather(1, idx_long).squeeze(1);
    }

    // One clone for all probs
    torch::Tensor draft_probs_all;
    if (draft_all_probs_full.defined()) {
        draft_probs_all = draft_all_probs_full.clone();
    }

    // Assign per-stream views
    int64_t probs_batch_off = 0;
    int64_t idx             = 0;
    for (auto& stream : all_streams) {
        GenerateStream::MtpAsyncDeviceState state;
        if (spec_decode_output.success_cpu.defined() && !spec_decode_output.success_cpu.data_ptr<bool>()[idx]) {
            probs_batch_off += stream->nextBatchSize();
            ++idx;
            continue;
        }
        state.accept_len_gpu    = accept_len_all.narrow(0, idx, 1);
        state.accept_tokens_gpu = accept_tokens_all.narrow(0, idx, 1);
        state.propose_tokens_gpu =
            propose_tokens_all.defined() ? propose_tokens_all.narrow(0, idx, 1) : torch::Tensor();
        state.next_seq_len_gpu       = next_seq_len_owned.narrow(0, idx, 1);
        state.last_hidden_states_gpu = last_hidden_all.defined() ? last_hidden_all.narrow(0, idx, 1) : torch::Tensor();

        const auto next_batch_size = stream->nextBatchSize();
        if (draft_probs_all.defined() && next_batch_size > 0) {
            state.draft_all_probs_gpu = draft_probs_all.narrow(0, probs_batch_off, next_batch_size);
        }

        // In the sync path dispatchDecode() has already advanced the host
        // stream by accept_len. Keep next_real_seq_len aligned with the
        // committed host length; adding accept_len here would reserve/rollback
        // against a future position that has not been verified yet.
        state.last_real_seq_len = stream->seqLength();
        state.next_real_seq_len = state.last_real_seq_len;
        // Publish per-stream GPU mirrors for the next draft-prefill step.
        auto sp_output_buffer = stream->getSPOutputBuffer();
        if (sp_output_buffer) {
            auto target_idx = (state.accept_len_gpu - 1).to(torch::kLong);
            sp_output_buffer->target_token_gpu =
                state.accept_tokens_gpu.squeeze(0).index_select(/*dim=*/0, target_idx).to(torch::kInt32);
            sp_output_buffer->propose_tokens_gpu = state.propose_tokens_gpu;
        }
        stream->setMtpAsyncDeviceState(std::move(state));

        probs_batch_off += next_batch_size;
        ++idx;
    }
}

absl::Status MtpExecutor::dispatchDecodeAsync(const StreamGroups&                          stream_groups,
                                              const speculative::SpeculativeSamplerOutput& spec_decode_output,
                                              MergedOutput                                 draft_prefill_output,
                                              std::shared_ptr<torch::Event>                rejection_event,
                                              std::shared_ptr<torch::Event>                draft_event) {
    RTP_LLM_PROFILE_SCOPE("executor.mtp.decode_step(dispatch_output_async)");

    const auto& accept_len_gpu_all     = spec_decode_output.accept_len;
    const auto& accept_tokens_gpu_all  = spec_decode_output.accept_tokens;
    const auto& propose_tokens_gpu_all = draft_prefill_output.sampler_output.token_ids;
    const auto& draft_all_hidden_full  = draft_prefill_output.model_output.all_hidden_states;
    const auto& draft_all_probs_full   = draft_prefill_output.sampler_output.all_probs;

    auto       all_streams = stream_groups.allStreams();
    const auto batch_size  = static_cast<int64_t>(all_streams.size());

    // --- Batch-level ops ---

    const auto cuda_i32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    const auto cuda_i64 = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);

    torch::Tensor prev_seq_len_all;
    torch::Tensor next_seq_len_all;
    torch::Tensor hidden_idx_all;
    if (accept_len_gpu_all.defined() && batch_size > 0) {
        // Build prev_seq_len per-stream: use device state when available,
        // fall back to host seqLength for new streams without device state.
        std::vector<torch::Tensor> prev_slices;
        prev_slices.reserve(batch_size);
        for (const auto& stream : all_streams) {
            const auto& gpu_val = stream->getNextSeqLenGpu();
            if (gpu_val.defined() && gpu_val.is_cuda()) {
                prev_slices.push_back(gpu_val.reshape({1}));
            } else {
                prev_slices.push_back(torch::tensor({static_cast<int32_t>(stream->seqLength())}, cuda_i32));
            }
        }
        prev_seq_len_all = torch::cat(prev_slices, 0);

        next_seq_len_all = torch::empty({batch_size}, cuda_i32);
        hidden_idx_all   = torch::empty({batch_size}, cuda_i64);

        auto accept_len_i32 = accept_len_gpu_all.to(torch::kInt32);
#if USING_CUDA
        invokeMtpDispatchStatePrepare(accept_len_i32,
                                      prev_seq_len_all,
                                      next_seq_len_all,
                                      hidden_idx_all,
                                      batch_size,
                                      at::cuda::getCurrentCUDAStream().stream());
#else
        next_seq_len_all = prev_seq_len_all + accept_len_i32;
        hidden_idx_all   = (accept_len_i32.to(torch::kInt64) - 1);
#endif
    }

    // 2. Batch gather hidden states (1 gather op instead of N index_selects)
    torch::Tensor last_hidden_all;
    const auto    stream_hidden_len = static_cast<int64_t>(propose_step_ + 1);
    if (propose_step_ > 1 && !is_dspark_ && draft_all_hidden_full.defined()) {
        const auto hidden_size  = draft_all_hidden_full.size(1);
        auto       hidden_3d    = draft_all_hidden_full.reshape({batch_size, stream_hidden_len, hidden_size});
        auto       idx_expanded = hidden_idx_all.reshape({batch_size, 1, 1}).expand({batch_size, 1, hidden_size});
        last_hidden_all         = hidden_3d.gather(1, idx_expanded).squeeze(1);
    }

    // DSpark publishes one request row per stream. Preserve failed requests
    // in a batch before making per-stream views; never read success on host.
    // Keep the legacy MTP publication path below unchanged.
    const bool batch_dspark_publication    = is_dspark_ && batch_size > 0 && spec_decode_output.success.defined();
    auto       published_accept_len_all    = accept_len_gpu_all;
    auto       published_accept_tokens_all = accept_tokens_gpu_all;
    auto       published_next_seq_len_all  = next_seq_len_all;
    if (batch_dspark_publication) {
        std::vector<torch::Tensor> previous_tokens;
        std::vector<torch::Tensor> previous_lengths;
        previous_tokens.reserve(batch_size);
        previous_lengths.reserve(batch_size);
        int64_t row = 0;
        for (const auto& stream : all_streams) {
            const auto previous = stream->getMtpAsyncDeviceState();
            const auto tokens   = accept_tokens_gpu_all.narrow(0, row, 1);
            const auto length   = accept_len_gpu_all.narrow(0, row, 1);
            const bool same_width =
                previous.accept_tokens_gpu.defined() && previous.accept_tokens_gpu.sizes() == tokens.sizes();
            if (same_width) {
                previous_tokens.push_back(previous.accept_tokens_gpu);
            } else {
                torch::Tensor anchor;
                if (previous.accept_tokens_gpu.defined() && previous.accept_len_gpu.defined()) {
                    anchor = previous.accept_tokens_gpu.gather(
                        1, (previous.accept_len_gpu.to(torch::kLong) - 1).reshape({1, 1}));
                } else {
                    anchor = toCudaInt32WithHostHold(
                        stream->completeTokenIds().narrow(1, stream->seqLength() - 1, 1).contiguous(), buffer_holder_);
                }
                previous_tokens.push_back(anchor.expand_as(tokens));
            }
            previous_lengths.push_back(same_width && previous.accept_len_gpu.defined() ? previous.accept_len_gpu :
                                                                                         torch::ones_like(length));
            ++row;
        }
        const auto& ok           = spec_decode_output.success;
        published_accept_len_all = torch::where(ok, accept_len_gpu_all, torch::cat(previous_lengths, 0));
        published_accept_tokens_all =
            torch::where(ok.unsqueeze(1), accept_tokens_gpu_all, torch::cat(previous_tokens, 0));
        published_next_seq_len_all = torch::where(ok, next_seq_len_all, prev_seq_len_all);
    }

    // Select all accepted target tokens in one kernel. Per-stream selection
    // otherwise launches several tiny cast/index kernels for every request.
    torch::Tensor target_tokens_all;
    if (accept_tokens_gpu_all.defined() && hidden_idx_all.defined()) {
        const auto target_indices = batch_dspark_publication ?
                                        (published_accept_len_all.to(torch::kLong) - 1).reshape({batch_size, 1}) :
                                        hidden_idx_all.reshape({batch_size, 1});
        target_tokens_all         = published_accept_tokens_all.gather(1, target_indices).squeeze(1);
        if (!batch_dspark_publication && target_tokens_all.scalar_type() != torch::kInt32) {
            target_tokens_all = target_tokens_all.to(torch::kInt32);
        }
    }

    // One batch subtraction replaces the next round's per-request launches.
    // Only publish with the DSpark success-masked anchor selected above.
    const auto dspark_committed_ends_all = batch_dspark_publication ? published_next_seq_len_all - 1 : torch::Tensor();

    // 3. One clone for all probs
    torch::Tensor draft_probs_all;
    if (draft_all_probs_full.defined()) {
        draft_probs_all = draft_all_probs_full.clone();
    }

    // 4. Assign per-stream views (narrow is metadata-only, no kernel launch)
    int64_t probs_batch_off = 0;
    int64_t idx             = 0;
    for (auto& stream : all_streams) {
        GenerateStream::MtpAsyncDeviceState state;
        state.accept_len_gpu    = published_accept_len_all.narrow(0, idx, 1);
        state.accept_tokens_gpu = published_accept_tokens_all.narrow(0, idx, 1);
        state.propose_tokens_gpu =
            propose_tokens_gpu_all.defined() ? propose_tokens_gpu_all.narrow(0, idx, 1) : torch::Tensor();
        state.next_seq_len_gpu       = published_next_seq_len_all.narrow(0, idx, 1);
        state.last_hidden_states_gpu = last_hidden_all.defined() ? last_hidden_all.narrow(0, idx, 1) : torch::Tensor();

        if (batch_dspark_publication) {
            state.dspark_anchor_gpu        = target_tokens_all.narrow(0, idx, 1);
            state.dspark_committed_end_gpu = dspark_committed_ends_all.narrow(0, idx, 1);
        }

        const auto next_batch_size = stream->nextBatchSize();
        if (draft_probs_all.defined() && next_batch_size > 0) {
            state.draft_all_probs_gpu = draft_probs_all.narrow(0, probs_batch_off, next_batch_size);
        }

        state.last_real_seq_len = stream->seqLength();
        // This is an allocation upper bound for the next iteration's
        // incrKVBlock while async bookkeeping is still in flight. It is not the
        // committed sequence length; the exact accepted length is published via
        // next_seq_len_gpu and then committed by the bookkeeping worker.
        state.next_real_seq_len = state.last_real_seq_len + static_cast<int>(propose_step_ + 1);
        if (spec_decode_output.success.defined() && !batch_dspark_publication) {
            // No host synchronization: retain the previous request state on
            // device until the same-event worker reports a failed request.
            const auto    previous = stream->getMtpAsyncDeviceState();
            const auto    ok       = spec_decode_output.success.narrow(0, idx, 1);
            torch::Tensor old_anchor;
            if (previous.accept_tokens_gpu.defined() && previous.accept_len_gpu.defined()) {
                old_anchor = previous.accept_tokens_gpu.gather(
                    1, (previous.accept_len_gpu.to(torch::kLong) - 1).reshape({1, 1}));
            } else {
                old_anchor = toCudaInt32WithHostHold(
                    stream->completeTokenIds().narrow(1, stream->seqLength() - 1, 1).contiguous(), buffer_holder_);
            }
            const auto old_tokens   = previous.accept_tokens_gpu.defined()
                                            && previous.accept_tokens_gpu.sizes() == state.accept_tokens_gpu.sizes() ?
                                          previous.accept_tokens_gpu :
                                          old_anchor.expand_as(state.accept_tokens_gpu);
            const auto old_length   = previous.accept_tokens_gpu.defined()
                                            && previous.accept_tokens_gpu.sizes() == state.accept_tokens_gpu.sizes()
                                            && previous.accept_len_gpu.defined() ?
                                          previous.accept_len_gpu :
                                          torch::ones_like(state.accept_len_gpu);
            state.accept_len_gpu    = torch::where(ok, state.accept_len_gpu, old_length);
            state.accept_tokens_gpu = torch::where(ok.unsqueeze(1), state.accept_tokens_gpu, old_tokens);
            state.next_seq_len_gpu  = torch::where(ok, state.next_seq_len_gpu, prev_seq_len_all.narrow(0, idx, 1));
        }
        // Publish per-stream GPU mirrors before launching bookkeeping.
        auto sp_output_buffer = stream->getSPOutputBuffer();
        if (sp_output_buffer) {
            sp_output_buffer->target_token_gpu =
                spec_decode_output.success.defined() && !batch_dspark_publication ?
                    state.accept_tokens_gpu.gather(1, (state.accept_len_gpu.to(torch::kLong) - 1).reshape({1, 1}))
                        .reshape({1}) :
                    target_tokens_all.narrow(0, idx, 1);
            sp_output_buffer->propose_tokens_gpu = state.propose_tokens_gpu;
        }
        stream->setMtpAsyncDeviceState(std::move(state));

        probs_batch_off += next_batch_size;
        ++idx;
    }

    // Launch bookkeeping after all next-step device views are visible.
    auto* processor          = batch_stream_processor_.get();
    auto  spec_decode_copy   = spec_decode_output;
    auto  draft_prefill_copy = std::move(draft_prefill_output);
    auto  stream_groups_copy = stream_groups;

    auto streams_for_inc = stream_groups_copy.allStreams();
    for (auto& s : streams_for_inc) {
        s->incPendingAsyncBookkeeping();
    }

    try {
        spec_bookkeeping_runner_.launch(
            [processor,
             worker_streams     = streams_for_inc,
             stream_groups_copy = std::move(stream_groups_copy),
             spec_decode_copy   = std::move(spec_decode_copy),
             draft_prefill_copy = std::move(draft_prefill_copy),
             rejection_event,
             draft_event]() mutable {
                RTP_LLM_PROFILE_SCOPE("executor.mtp.decode_step(spec_bookkeeping_worker)");

                const bool protected_kv =
                    std::any_of(worker_streams.begin(), worker_streams.end(), [](const auto& stream) {
                        return stream->kvExecutionProtected();
                    });

                auto dispatch = [&] {
                    if (rejection_event) {
                        rejection_event->block(cuda_graph::graphGetCurrentStream());
                    }
                    if (draft_event) {
                        draft_event->block(cuda_graph::graphGetCurrentStream());
                    }
                    auto status = processor->dispatchDecode(stream_groups_copy, spec_decode_copy, draft_prefill_copy);
                    if (!status.ok()) {
                        RTP_LLM_LOG_ERROR("[stream-async] dispatchDecode (worker) failed: %s",
                                          status.ToString().c_str());
                    }
                };

                if (protected_kv) {
                    std::exception_ptr dispatch_error;
                    try {
                        dispatch();
                    } catch (...) {
                        dispatch_error = std::current_exception();
                    }
                    auto completion_error = finishProtectedKvBookkeeping(worker_streams);
                    if (completion_error || dispatch_error) {
                        std::rethrow_exception(completion_error ? completion_error : dispatch_error);
                    }
                    return;
                }

                auto dec_guard = std::shared_ptr<void>(nullptr, [worker_streams](void*) {
                    for (auto& s : worker_streams) {
                        s->decPendingAsyncBookkeepingAndMaybeRelease();
                    }
                });

                dispatch();

                // Each stream exposes the exact completion event for its linear-KV swap.
                for (auto& stream : worker_streams) {
                    auto event = std::make_shared<torch::Event>(cuda_graph::makeGraphEvent());
                    event->record(cuda_graph::graphGetCurrentStream());
                    stream->setPendingSwapDoneEvent(std::static_pointer_cast<void>(event));
                }
            },
            [worker_streams = streams_for_inc](std::exception_ptr) {
                // The task was accepted but never reached its body (TLS/stream setup
                // failure). No worker GPU work was submitted. Quarantine conservatively
                // and drop only this task's registered claims before signalling task_done.
                std::exception_ptr cleanup_error;
                for (const auto& stream : worker_streams) {
                    try {
                        if (stream->kvExecutionProtected()) {
                            stream->quarantineKvExecution();
                        }
                    } catch (...) {
                        if (!cleanup_error) {
                            cleanup_error = std::current_exception();
                        }
                    }
                    try {
                        stream->decPendingAsyncBookkeepingAndMaybeRelease();
                    } catch (...) {
                        if (!cleanup_error) {
                            cleanup_error = std::current_exception();
                        }
                    }
                }
                if (cleanup_error) {
                    std::rethrow_exception(cleanup_error);
                }
            });
    } catch (...) {
        // launch can rethrow the previous task's error before accepting this
        // task. Roll back only these new CPU claims; process still owns KV.
        for (const auto& stream : streams_for_inc) {
            stream->decPendingAsyncBookkeepingAndMaybeRelease();
        }
        throw;
    }

    return absl::OkStatus();
}

}  // namespace rtp_llm
