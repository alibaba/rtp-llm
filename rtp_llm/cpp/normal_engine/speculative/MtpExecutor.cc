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
#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorFactory.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#include <sstream>
#if USING_CUDA
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
#include <ATen/cuda/CUDAContext.h>

namespace rtp_llm {

namespace {

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

bool cacheDebugFlag(const char* env_name) {
    const char* env = std::getenv(env_name);
    return env != nullptr && std::string(env) != "0";
}

const CachedEnvFlag kMtpDeviceInputFlag = cacheEnvFlag("RTP_LLM_DEVICE_INPUT", "mtp-device-input", "enabled");
const CachedEnvFlag kMtpDeviceInputCheckFlag =
    cacheEnvFlag("RTP_LLM_DEVICE_INPUT_CHECK", "mtp-device-input", "enabled");
const CachedEnvFlag kStreamAsyncFlag = cacheEnvFlag("RTP_LLM_STREAM_ASYNC", "stream-async", "useStreamAsync");
const CachedEnvFlag kAsyncDeviceStateFlag =
    cacheEnvFlag("RTP_LLM_MTP_ASYNC_DEVICE_STATE", "async-device-state", "enabled");
const CachedEnvFlag kDropBroadSyncFlag = cacheEnvFlag("RTP_LLM_DROP_BROAD_SYNC", "drop-broad-sync", "enabled");
const CachedEnvFlag kAsyncPrepareFlag  = cacheEnvFlag("RTP_LLM_MTP_ASYNC_PREPARE", "async-prepare", "enabled");
const bool          kDebugTargetVerifyInputEnabled  = cacheDebugFlag("RTP_LLM_DEBUG_TARGET_VERIFY_INPUT");
const bool          kDebugCompareSpPrefillEnabled   = cacheDebugFlag("RTP_LLM_COMPARE_SP_PREFILL");
const bool          kDebugMtpPrefillDataEnabled     = cacheDebugFlag("RTP_LLM_DEBUG_MTP_PREFILL_DATA");
const bool          kDebugMtpDecodeDataEnabled      = cacheDebugFlag("RTP_LLM_DEBUG_MTP_DECODE_DATA");
const bool          kDisableSpPrefillCudaGraphByEnv = []() {
    const char* env = std::getenv("DISABLE_SP_PREFILL_CUDA_GRAPH");
    return env != nullptr && std::string(env) == "1";
}();
const bool kForceSpPrefillCudaGraphByEnv = []() {
    const char* env = std::getenv("RTP_LLM_FORCE_SP_PREFILL_CUDA_GRAPH");
    return env != nullptr && std::string(env) == "1";
}();

bool debugTargetVerifyInputEnabled() {
    static const bool logged = []() {
        if (kDebugTargetVerifyInputEnabled) {
            RTP_LLM_LOG_WARNING("[debug-target-verify] enabled; this performs D2H copies and serializes the hot path");
        }
        return true;
    }();
    (void)logged;
    return kDebugTargetVerifyInputEnabled;
}

bool debugCompareSpPrefillEnabled() {
    static const bool logged = []() {
        if (kDebugCompareSpPrefillEnabled) {
            RTP_LLM_LOG_WARNING("[debug-sp-prefill] enabled; this runs an extra eager draft-prefill forward");
        }
        return true;
    }();
    (void)logged;
    return kDebugCompareSpPrefillEnabled;
}

bool debugMtpPrefillDataEnabled() {
    static const bool logged = []() {
        if (kDebugMtpPrefillDataEnabled) {
            RTP_LLM_LOG_WARNING("[debug-mtp-prefill-data] enabled; this records stage tensors for graph/eager diff");
        }
        return true;
    }();
    (void)logged;
    return kDebugMtpPrefillDataEnabled;
}

bool debugMtpDecodeDataEnabled() {
    static const bool logged = []() {
        if (kDebugMtpDecodeDataEnabled) {
            RTP_LLM_LOG_WARNING(
                "[debug-mtp-decode-data] enabled; tensor summaries may perform D2H copies and serialize debugging runs");
        }
        return true;
    }();
    (void)logged;
    return kDebugMtpDecodeDataEnabled;
}

std::string debugTensorSummary(const torch::Tensor& tensor, int64_t limit = 8) {
    if (!tensor.defined()) {
        return "None";
    }
    std::ostringstream oss;
    oss << "shape=[";
    for (int64_t i = 0; i < tensor.dim(); ++i) {
        if (i > 0) {
            oss << ",";
        }
        oss << tensor.size(i);
    }
    oss << "] device=" << tensor.device() << " dtype=" << tensor.dtype() << " numel=" << tensor.numel();
    if (tensor.numel() == 0) {
        return oss.str();
    }
    try {
        auto flat  = tensor.reshape({-1});
        auto count = std::min<int64_t>(limit, flat.numel());
        auto head  = flat.slice(0, 0, count);
        if (head.device().is_cuda()) {
            head = head.cpu();
        }
        oss << " head=" << head;
    } catch (const std::exception& e) {
        oss << " summary_error=" << e.what();
    }
    return oss.str();
}

void logMtpDecodeModelInput(const char* tag, const GptModelInputs& input) {
    if (!debugMtpDecodeDataEnabled()) {
        return;
    }
    if (input.is_fake_stream) {
        static std::atomic<int> fake_log_budget{16};
        if (fake_log_budget.fetch_sub(1, std::memory_order_relaxed) <= 0) {
            return;
        }
    }
    static std::atomic<int> debug_log_budget{512};
    if (debug_log_budget.fetch_sub(1, std::memory_order_relaxed) <= 0) {
        return;
    }
    RTP_LLM_LOG_INFO("[debug-mtp-decode-data][%s] combo=%s input_lengths=%s sequence_lengths=%s "
                     "prefix_lengths=%s sequence_lengths_plus_1=%s lm_output_indexes=%s "
                     "last_hidden=%s kv_kernel=%s kv_block=%s is_target_verify=%d "
                     "mtp_iteration_step=%d is_fake_stream=%d",
                     tag,
                     debugTensorSummary(input.combo_tokens).c_str(),
                     debugTensorSummary(input.input_lengths).c_str(),
                     debugTensorSummary(input.sequence_lengths).c_str(),
                     debugTensorSummary(input.prefix_lengths).c_str(),
                     debugTensorSummary(input.sequence_lengths_plus_1).c_str(),
                     debugTensorSummary(input.lm_output_indexes).c_str(),
                     debugTensorSummary(input.last_hidden_states, 0).c_str(),
                     debugTensorSummary(input.kv_cache_kernel_block_id, 0).c_str(),
                     debugTensorSummary(input.kv_cache_block_id, 0).c_str(),
                     static_cast<int>(input.is_target_verify),
                     input.mtp_iteration_step,
                     static_cast<int>(input.is_fake_stream));
}

void logMtpDecodeModelOutput(const char* tag, const GptModelOutputs& output, bool is_fake_stream = false) {
    if (!debugMtpDecodeDataEnabled()) {
        return;
    }
    if (is_fake_stream) {
        static std::atomic<int> fake_log_budget{16};
        if (fake_log_budget.fetch_sub(1, std::memory_order_relaxed) <= 0) {
            return;
        }
    }
    static std::atomic<int> debug_log_budget{512};
    if (debug_log_budget.fetch_sub(1, std::memory_order_relaxed) <= 0) {
        return;
    }
    RTP_LLM_LOG_INFO("[debug-mtp-decode-data][%s] logits=%s hidden=%s all_hidden=%s is_fake_stream=%d",
                     tag,
                     debugTensorSummary(output.logits, 0).c_str(),
                     debugTensorSummary(output.hidden_states, 0).c_str(),
                     debugTensorSummary(output.all_hidden_states, 0).c_str(),
                     static_cast<int>(is_fake_stream));
}

std::string debugTensorDiffSummary(const torch::Tensor& lhs, const torch::Tensor& rhs) {
    if (!lhs.defined() || !rhs.defined()) {
        return std::string("lhs=") + (lhs.defined() ? "defined" : "None")
               + " rhs=" + (rhs.defined() ? "defined" : "None");
    }
    if (lhs.sizes() != rhs.sizes()) {
        std::ostringstream oss;
        oss << "shape_mismatch lhs=" << debugTensorSummary(lhs, 0) << " rhs=" << debugTensorSummary(rhs, 0);
        return oss.str();
    }
    if (lhs.numel() == 0) {
        return "empty";
    }
    try {
        auto  lhs_f          = lhs.to(torch::kFloat32);
        auto  rhs_f          = rhs.to(torch::kFloat32);
        auto  lhs_nan        = torch::isnan(lhs_f);
        auto  rhs_nan        = torch::isnan(rhs_f);
        auto  both_nan       = torch::logical_and(lhs_nan, rhs_nan);
        auto  mismatch       = torch::logical_and(torch::ne(lhs_f, rhs_f), torch::logical_not(both_nan));
        auto  finite         = torch::logical_and(torch::isfinite(lhs_f), torch::isfinite(rhs_f));
        auto  finite_count   = finite.to(torch::kInt64).sum().item<int64_t>();
        auto  mismatch_count = mismatch.to(torch::kInt64).sum().item<int64_t>();
        auto  lhs_nan_count  = lhs_nan.to(torch::kInt64).sum().item<int64_t>();
        auto  rhs_nan_count  = rhs_nan.to(torch::kInt64).sum().item<int64_t>();
        float max_diff       = 0.0f;
        float mean           = 0.0f;
        if (finite_count > 0) {
            auto diff = (lhs_f - rhs_f).abs().masked_select(finite);
            max_diff  = diff.max().item<float>();
            mean      = diff.mean().item<float>();
        }
        std::ostringstream oss;
        oss << "max_abs=" << max_diff << " mean_abs=" << mean << " finite_count=" << finite_count
            << " mismatch_count=" << mismatch_count << " lhs_nan=" << lhs_nan_count << " rhs_nan=" << rhs_nan_count;
        return oss.str();
    } catch (const std::exception& e) {
        return std::string("diff_error=") + e.what();
    }
}

torch::Tensor tryGetMtpDebugTensor(ModelBase* model, const std::string& name, int64_t num_rows) {
    auto* py_model = dynamic_cast<PyWrappedModel*>(model);
    if (py_model == nullptr) {
        return torch::Tensor();
    }
    try {
        return py_model->getPythonDebugTensor(name, num_rows);
    } catch (const std::exception& e) {
        RTP_LLM_LOG_WARNING("[debug-mtp-prefill-data] get %s failed: %s", name.c_str(), e.what());
        return torch::Tensor();
    }
}

torch::Tensor tryGetMtpDebugKvCache(ModelBase* model, int64_t layer_idx, int64_t max_blocks) {
    auto* py_model = dynamic_cast<PyWrappedModel*>(model);
    if (py_model == nullptr) {
        return torch::Tensor();
    }
    try {
        return py_model->getPythonDebugKvCache(layer_idx, max_blocks);
    } catch (const std::exception& e) {
        RTP_LLM_LOG_WARNING("[debug-mtp-prefill-data] get kv cache failed: %s", e.what());
        return torch::Tensor();
    }
}

void logMtpPrefillStageDiffs(ModelBase* graph_model, ModelBase* eager_model, int64_t num_rows) {
    if (!debugMtpPrefillDataEnabled()) {
        return;
    }
    static std::atomic<int> log_budget{8};
    if (log_budget.fetch_sub(1, std::memory_order_relaxed) <= 0) {
        return;
    }
    const char* names[] = {"input_ids",
                           "input_hiddens",
                           "input_lengths",
                           "prefix_lengths",
                           "cu_seqlens",
                           "cu_kv_seqlens",
                           "kv_cache_kernel_block_id_device",
                           "kv_cache_kernel_block_id_group0",
                           "kv_cache_block_id_device",
                           "fc_hidden",
                           "layer0_hidden",
                           "layer0_residual",
                           "pre_norm_hidden"};
    for (const char* name : names) {
        auto graph_tensor = tryGetMtpDebugTensor(graph_model, name, num_rows);
        auto eager_tensor = tryGetMtpDebugTensor(eager_model, name, num_rows);
        RTP_LLM_LOG_INFO("[debug-mtp-prefill-data] %s graph=%s eager=%s diff=%s",
                         name,
                         debugTensorSummary(graph_tensor, 4).c_str(),
                         debugTensorSummary(eager_tensor, 4).c_str(),
                         debugTensorDiffSummary(graph_tensor, eager_tensor).c_str());
    }
}

std::string debugLogitsTopKSummary(const torch::Tensor& logits, int64_t k = 8) {
    if (!logits.defined() || logits.numel() == 0 || logits.dim() != 2 || logits.size(0) == 0) {
        return "None";
    }
    try {
        auto row     = logits[0].to(torch::kFloat32);
        auto topk    = row.topk(std::min<int64_t>(k, row.size(0)));
        auto values  = std::get<0>(topk);
        auto indices = std::get<1>(topk);
        if (values.device().is_cuda()) {
            values = values.cpu();
        }
        if (indices.device().is_cuda()) {
            indices = indices.cpu();
        }
        std::ostringstream oss;
        oss << "ids=" << indices << " values=" << values;
        return oss.str();
    } catch (const std::exception& e) {
        return std::string("topk_error=") + e.what();
    }
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

void MtpExecutor::maybeOverrideLastHiddenWithMtpBuffer(GptModelInputs& model_input,
                                                       ModelBase&      source,
                                                       bool            request_actual_rows) {
    if (!model_input.combo_tokens.defined() || model_input.combo_tokens.numel() == 0) {
        return;
    }
    const auto mtp_hidden_rows = request_actual_rows ? -1 : model_input.combo_tokens.numel();
    auto       pre_hc          = source.getMtpTargetHiddenStates(mtp_hidden_rows);
    if (!pre_hc.defined() || pre_hc.numel() == 0) {
        RTP_LLM_CHECK_WITH_INFO(!request_actual_rows || model_input.last_hidden_states.defined(),
                                "CP MTP hidden buffer must contain local rows before draft prefill");
        return;
    }
    model_input.last_hidden_states = pre_hc;
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
        // This waits for the previous decode step's tiny D2H only; the current
        // step's accept_len is staged below and reported on the next iteration.
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

static void ensureSpOutputTokenGpuMirrors(const SpeculativeExecutorStreamOutputPtr& sp_buffer) {
    // Mirrors should already be device-resident from the buffer's construction
    // site (MtpExecutor::prepareStreams + makeFakeSPOutputBuffer). Only the
    // legacy P2P-injection path (StreamCacheResource::applyP2PSideChannel) and
    // dispatchDecodeAsync intentionally replace them with values from a payload.
    // This function exists as a defensive fallback for older paths; it should be
    // a no-op in steady-state and never trigger an H2D sync.
    if (!sp_buffer || !sp_buffer->tokens.defined() || sp_buffer->tokens.numel() < 2) {
        return;
    }
    const auto cuda_i32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    if (!sp_buffer->target_token_gpu.defined() || !sp_buffer->target_token_gpu.is_cuda()) {
        sp_buffer->target_token_gpu = torch::zeros({1}, cuda_i32);
    }
    if (!sp_buffer->propose_tokens_gpu.defined() || !sp_buffer->propose_tokens_gpu.is_cuda()) {
        sp_buffer->propose_tokens_gpu = torch::zeros({1}, cuda_i32);
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
                                                         const RuntimeConfig&   runtime_config,
                                                         const ResourceContext& resource_context,
                                                         int                    vocab_size) {
    auto fake_stream =
        makeFakeStream(max_new_tokens, 1 + max_new_tokens, model_config, runtime_config, resource_context);

    // Fake SP buffer's hidden_states stands in for the target's pre-output
    // residual that the draft consumes (DSv4: [T, hc_mult*hidden_size];
    // non-DSv4 keeps hc_mult=1 so the shape is plain [T, hidden_size]).
    auto sp_buffer = makeFakeSPOutputBuffer(model_config.data_type,
                                            model_config.hidden_size * model_config.hc_mult,
                                            model_config.vocab_size,
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
    return fake_stream;
}

MtpExecutor::MtpExecutor(const EngineInitParams&                        params,
                         std::unique_ptr<ProposeModelEngineInitParams>& propose_params,
                         const std::shared_ptr<KVCacheManager>&         cache_manager,
                         MlaOpsType                                     mla_ops_type,
                         int32_t                                        kv_cache_group_num,
                         const std::vector<int32_t>&                    kv_cache_layer_to_group,
                         bool                                           warm_up):
    Executor(),
    cache_manager_(cache_manager),
    metrics_reporter_(params.metrics_reporter),
    tps_reporter_(MetricsLoopReporter<RtpLLMTokenPSMetrics, RtpLLMTokenPSMetricsCollector>(
        params.parallelism_config.tp_rank == 0 && !warm_up ? metrics_reporter_ : nullptr)),
    warm_up_(warm_up),
    role_type_(params.pd_sep_config.role_type),
    collect_metrics_stream_(cuda_graph::graphGetStreamFromPool(true)),
    // Bookkeeping worker intentionally does not inherit PyTorch profiler TLS from
    // the engine loop. Kineto callbacks are thread-affine; propagating an active
    // profiling state to the worker thread can crash while perf timelines are
    // being recorded.
    spec_bookkeeping_runner_(cuda_graph::graphGetStreamFromPool(true), false) {
    data_type_        = params.model_config_.data_type;
    hidden_size_      = params.model_config_.hidden_size * params.model_config_.hc_mult;
    propose_step_     = propose_params->gen_num_per_circle;
    vocab_size_       = params.model_config_.vocab_size;
    draft_vocab_size_ = propose_params->getEngineInitParams().model_config_.vocab_size;

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

    auto              target_cache_layer_layout = cache_manager->getMainModelCacheLayerLayout();
    auto              draft_cache_layer_layout  = cache_manager->getMTPModuleCacheLayerLayout(0);
    const CacheConfig cache_config              = cache_manager ? cache_manager->cacheConfig() : CacheConfig();
    const CacheConfig draft_cache_config = cache_manager ? cache_manager->getMTPModuleCacheConfig(0) : CacheConfig();

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
         cache_config.seq_size_per_block,
         cache_config.kernel_seq_size_per_block,
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

    // when warmup, cache manager maybe nullptr
    is_linear_attention_model_ = cache_config.linear_group_num > 0;
    batch_stream_processor_.reset(new MtpBatchStreamProcessor(params.model_config_,
                                                              params.pd_sep_config,
                                                              params.profiling_debug_logging_config,
                                                              cache_config,
                                                              params.sp_config,
                                                              warm_up_));

    LogitsProcessorFactory::init(params.model_config_.ckpt_path, params.sp_config.tree_decode_config);
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
                                draft_cache_config.seq_size_per_block,
                                draft_cache_config.kernel_seq_size_per_block,
                                kv_cache_group_num,
                                kv_cache_layer_to_group,
                                cache_manager,
                                mtp_params->model_config_.hc_mult});
        if (!params.py_sp_model.is_none()) {
            RTP_LLM_LOG_INFO("[speculative decoding] using py model");
            draft_model_.reset(new PyWrappedModel(
                model_params, params.py_sp_model, false, false, draft_cache_layer_layout.layer_to_groups));
            // Create separate model for speculative prefill with CUDA graph if enabled (from params)
            const bool enable_cuda_graph           = params.hw_kernel_config.enable_cuda_graph;
            const bool disable_sp_prefill_by_env   = kDisableSpPrefillCudaGraphByEnv;
            const bool force_sp_prefill_cuda_graph = kForceSpPrefillCudaGraphByEnv;
            const bool draft_uses_mega_moe         = params.moe_config.moe_strategy == "mega_moe"
                                             || params.moe_config.moe_strategy == "mega_moe_fused"
                                             || mtp_params->moe_config.moe_strategy == "mega_moe"
                                             || mtp_params->moe_config.moe_strategy == "mega_moe_fused";
            const bool draft_uses_ep_collective =
                params.parallelism_config.ep_size > 1 || mtp_params->parallelism_config.ep_size > 1;
            const bool disable_sp_prefill_for_mega_moe =
                draft_uses_mega_moe && draft_uses_ep_collective && !force_sp_prefill_cuda_graph;
            const bool disable_sp_prefill_cuda_graph = disable_sp_prefill_by_env || disable_sp_prefill_for_mega_moe;
            RTP_LLM_LOG_INFO("[speculative decoding] enable_cuda_graph=%d disable_sp_prefill_cuda_graph=%d "
                             "disable_by_env=%d disable_for_mega_moe=%d force_sp_prefill_cuda_graph=%d "
                             "draft_uses_mega_moe=%d draft_uses_ep_collective=%d "
                             "(set ENABLE_CUDA_GRAPH=1 when starting server to enable sp_prefill_draft_model_; "
                             "set DISABLE_SP_PREFILL_CUDA_GRAPH=1 to skip the draft prefill CUDA graph capture only; "
                             "set RTP_LLM_FORCE_SP_PREFILL_CUDA_GRAPH=1 for diagnostic replay on GLM5 MegaMoE)",
                             static_cast<int>(enable_cuda_graph),
                             static_cast<int>(disable_sp_prefill_cuda_graph),
                             static_cast<int>(disable_sp_prefill_by_env),
                             static_cast<int>(disable_sp_prefill_for_mega_moe),
                             static_cast<int>(force_sp_prefill_cuda_graph),
                             static_cast<int>(draft_uses_mega_moe),
                             static_cast<int>(draft_uses_ep_collective));
            if (enable_cuda_graph && !disable_sp_prefill_cuda_graph) {
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
                sp_prefill_draft_model_.reset(new PyWrappedModel(
                    model_params, sp_prefill_py_model, true, false, draft_cache_layer_layout.layer_to_groups));
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
    speculative_sampler_.reset(new speculative::SpeculativeSampler(d2t_map_, propose_step_));
    fast_topk_sampler_.reset(new speculative::FastTopKSampler(d2t_map_));

    RTP_LLM_LOG_INFO("[speculative decoding] d2t_map size: %ld", d2t_map_.defined() ? d2t_map_.numel() : 0);
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

    StreamGroups    stream_groups(streams);
    GptModelInputs  model_input;
    GptModelOutputs model_output;
    SamplerOutput   sampler_output;
    GptModelOutputs draft_model_output;
    SamplerOutput   draft_sampler_output;
    torch::Tensor   draft_last_hidden_states;

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

    // release model input before forward
    releaseAllModelBuffers();

    // CP+MTP: PyWrappedModel's CP processor (handleInputs) MUTATES
    // ``model_input.combo_tokens`` and ``model_input.input_lengths`` in
    // place to the rank-local zigzag chunk layout for the target forward.
    // The post-target MTP pipeline (updatePrefillPostDraftModelInput +
    // draft re-CP-slice) needs the FULL/global view, so snapshot both
    // tensors here while they still hold the global sequence and restore
    // on rank 0 before the second tpSync (which then broadcasts the
    // restored full view to every rank for the draft pass).
    const bool    cp_enabled = parallelism_config_.prefill_cp_config.is_enabled();
    torch::Tensor saved_combo_tokens;
    torch::Tensor saved_input_lengths;
    if (cp_enabled) {
        saved_combo_tokens  = toCudaWithHostHold(model_input.combo_tokens, buffer_holder_);
        saved_input_lengths = toCudaWithHostHold(model_input.input_lengths, buffer_holder_);
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
        if (model_input.is_fake_stream) {
            model_input.last_hidden_states = model_output.all_hidden_states;
        } else {
            CHECK_AND_RETURN_REF(sampler_input,
                                 batch_stream_processor_->gatherSamplerInput(stream_groups, model_input, model_output));
            holdSamplerInputHostBuffers(buffer_holder_, sampler_input);
            sampler_output = std::move(sampler_->forward(sampler_input));
            // Restore the full combo_tokens / input_lengths before the MTP
            // shift logic — under CP both were mutated to rank-local by the
            // target forward's handleInputs and the shift formula assumes a
            // contiguous full sequence (offset += input_length, last token
            // overwrite at offset+input_length-1).
            if (cp_enabled) {
                model_input.combo_tokens  = saved_combo_tokens;
                model_input.input_lengths = saved_input_lengths;
            }
            batch_stream_processor_->updatePrefillPostDraftModelInput(
                model_input, model_output, sampler_output, buffer_holder_);
        }
    }

    // draft model prefill
    {
        RTP_LLM_PROFILE_SCOPE("executor.mtp.prefill_step(draft_model_forward)");
        // DSv4 MTP consumes a special pre-hc residual buffer instead of the
        // normal model_output.all_hidden_states tensor. GLM MTP consumes the
        // target model's final-norm hidden states, so keep and broadcast the
        // post-layer hidden unless the source model exposes a special buffer.
        if (cp_enabled) {
            auto target_mtp_hidden = model_->getMtpTargetHiddenStates(-1);
            if (target_mtp_hidden.defined() && target_mtp_hidden.numel() > 0) {
                model_input.last_hidden_states = torch::Tensor();
            }
        }
        tpSyncModelInputs(model_input, parallelism_config_);
        model_input.mtp_iteration_step = 0;
        maybePrintModelInput(model_input, "prefill post draft model");
        int64_t     start_time_us           = autil::TimeUtility::currentTimeInMicroSeconds();
        const auto& mtp_cache_cfg           = cache_manager_->getMTPModuleCacheConfig(0);
        model_input.kv_block_stride_bytes   = mtp_cache_cfg.kv_block_stride_bytes;
        model_input.kv_scale_stride_bytes   = mtp_cache_cfg.kv_scale_stride_bytes;
        model_input.kv_cache_layer_to_group = draft_kv_cache_layer_to_group;
        // Source = main (just ran prefill; its pre-hc buffer is current).
        maybeOverrideLastHiddenWithMtpBuffer(model_input, *model_, cp_enabled);
        draft_model_output             = std::move(draft_model_->forward(model_input));
        model_input.mtp_iteration_step = -1;
        model_forward_us += autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
    }

    if (!isTpRank0() || warm_up_ || streams.size() == 0 || model_input.is_fake_stream) {
        cudaSyncAndCheck();
        return absl::OkStatus();
    }

    if (!cp_enabled) {
        maybeOverrideLastHiddenWithMtpBuffer(draft_model_output, *draft_model_);
    }

    // draft model sample
    {
        RTP_LLM_PROFILE_SCOPE("executor.mtp.prefill_step(draft_model_sample)");
        fast_topk_sampler_output       = fast_topk_sampler_->forward(draft_model_output.logits);
        draft_sampler_output.all_probs = fast_topk_sampler_output.all_probs;
        draft_sampler_output.token_ids = fast_topk_sampler_output.token_ids;
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
                                                     draft_last_hidden_states);
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

absl::Status MtpExecutor::decodeStep(const std::list<GenerateStreamPtr>& streams,
                                     MtpMetricsCollector&                metrics_collector) {
    RTP_LLM_PROFILE_SCOPE_DYNAMIC("executor.mtp.decode_step(decode_stream_size=%zu)", streams.size());

    RtpLLMExecutorMetricsCollector& executor_collector = metrics_collector.executor_collector;

    StreamGroups    stream_groups(streams);
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
    std::vector<torch::Tensor> draft_probs_list;
    torch::Event               accept_len_ready_event = cuda_graph::makeGraphEvent();
    int64_t                    model_forward_us       = 0;

    // Stream-async events are recorded on the main stream as soon as tensors
    // become valid. rejection_event guards accept_len/tokens D2H; draft_event
    // guards all_probs cloning. They stay null when stream-async is off.
    std::shared_ptr<torch::Event> rejection_event;
    std::shared_ptr<torch::Event> draft_event;

    waitPreviousBookkeepingAndKvSwaps(streams);

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

    {
        RTP_LLM_PROFILE_SCOPE("executor.mtp.decode_step(prepare_decode_input_and_tp_sync)");
        if (isTpRank0()) {
            if (propose_step_ == 1) {
                batch_stream_processor_->prepareOneStepSpecDecodeModelInput(stream_groups, model_input, buffer_holder_);
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
    size_t batch_size = model_input.input_lengths.size(0);

    // release hold buffers before draft model forward
    releaseAllModelBuffers();

    if (propose_step_ > 1) {
        if (shouldSkipFakeStreamForStop(model_input, "draftModelDecode")) {
            releaseAllModelBuffers();
            return absl::OkStatus();
        }
        model_input.kv_cache_layer_to_group = draft_kv_cache_layer_to_group;
        RTP_LLM_LOG_DEBUG("[MTP decode] draftModelDecode start");
        draftModelDecode(model_input, stream_groups, draft_probs_list, draft_token_ids_t, model_forward_us);
        RTP_LLM_LOG_DEBUG("[MTP decode] draftModelDecode end");
    }

    {
        if (shouldSkipFakeStreamForStop(model_input, "target verify forward")) {
            releaseAllModelBuffers();
            return absl::OkStatus();
        }
        int64_t start_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
        model_output          = runTargetVerifyForward(model_input, stream_groups);
        model_forward_us += autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
    }

    // trick: update draft sampler output after spec decode to avoid kernel launch overhead
    if (isTpRank0()) {
        RTP_LLM_PROFILE_SCOPE("executor.mtp.decode_step(update_draft_sampler_output)");
        if (!model_input.is_fake_stream) {
            if (propose_step_ == 1) {
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

        if (model_input.is_fake_stream) {
            speculative_sampler_output.accept_len = torch::full(
                {1}, (int64_t)(propose_step_ + 1), torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
            speculative_sampler_output.accept_tokens = torch::zeros(
                {1, (int64_t)(propose_step_ + 1)}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
        } else {
            // gatherSpecSamplerInput reads host stream state updated by the previous
            // bookkeeping worker. DROP_BROAD_SYNC therefore needs this narrow sync
            // unless the broad sync at decodeStep start already waited.
            if (useStreamAsync() && useDropBroadSync()) {
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
                batch_stream_processor_->gatherSpecSamplerInput(stream_groups, model_input, model_output));
            holdSamplerInputHostBuffers(buffer_holder_, sampler_input);
            sampler_output           = std::move(sampler_->forward(sampler_input));
            sampler_output.all_probs = sampler_output.all_probs.reshape(
                {(int64_t)batch_size, (int64_t)(propose_step_ + 1), (int64_t)vocab_size_});

            // rejection sampling
            speculative_sampler_output = speculative_sampler_->forward(streams, draft_sampler_output, sampler_output);
        }

        batch_stream_processor_->updateDecodePostDraftModelInput(
            model_input, model_output, speculative_sampler_output, batch_size, hidden_states_d_t, buffer_holder_);
        if (metrics_reporter_) {
            accept_len_ready_event.record(cuda_graph::graphGetCurrentStream());
        }
    } else {
        model_input.lm_output_indexes =
            torch::empty({(int64_t)batch_size}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
        model_input.last_hidden_states = model_output.all_hidden_states;
    }

    // Record before broadcast/draft work so the worker waits only for
    // accept_len/accept_tokens, not the queue tail.
    if (useStreamAsync()) {
        rejection_event = std::make_shared<torch::Event>(cuda_graph::makeGraphEvent());
        rejection_event->record(cuda_graph::graphGetCurrentStream());
    }

    maybeOverrideLastHiddenWithMtpBuffer(model_input, *model_);
    broadcastPostRejectionInputs(model_input, stream_groups);

    {
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
        auto fast_topk_sampler_output          = fast_topk_sampler_->forward(draft_prefill_model_output.logits);
        draft_prefill_sampler_output.all_probs = fast_topk_sampler_output.all_probs;
        draft_prefill_sampler_output.token_ids = fast_topk_sampler_output.token_ids;
    }

    // Record after draft_model_sample so worker all_probs/token_ids reads wait
    // on the earliest valid point, not metrics or dispatch slicing.
    if (useStreamAsync()) {
        draft_event = std::make_shared<torch::Event>(cuda_graph::makeGraphEvent());
        draft_event->record(cuda_graph::graphGetCurrentStream());
    }

    if (metrics_reporter_) {
        collectDecodeMetrics(stream_groups, accept_len_ready_event, speculative_sampler_output, metrics_collector);
    }

    return dispatchDecodeOutput(stream_groups,
                                streams,
                                speculative_sampler_output,
                                std::move(draft_prefill_model_output),
                                std::move(draft_prefill_sampler_output),
                                std::move(rejection_event),
                                std::move(draft_event));
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

        // Optional pre-kernel safety check. It performs D2H and can hide the
        // async race being diagnosed, so keep it out of the default hot path.
        if (debugTargetVerifyInputEnabled()) {
            debugCheckLinearBlockMapAtKernelRead(model_input, stream_groups);
        }
    }

    ensureModelInputsOnCuda(model_input, "decode.target_verify_forward");
    logMtpDecodeModelInput("target_verify_forward_input", model_input);
    GptModelOutputs model_output = model_->forward(model_input);
    logMtpDecodeModelOutput("target_verify_forward_output", model_output, model_input.is_fake_stream);
    RTP_LLM_LOG_DEBUG("[MTP decode] target model verify forward end");
    model_input.is_target_verify = false;
    return model_output;
}

void MtpExecutor::debugCheckLinearBlockMapAtKernelRead(const GptModelInputs& model_input,
                                                       const StreamGroups&   stream_groups) const {
    // Diagnose causal_conv1d_update IMA: it reads block_map[(seq_len - 2) / SBP].
    // Host-check before forward captures NULL slots without GPU coredumps.
    static const bool always_print = debugTargetVerifyInputEnabled();

    if (!model_input.kv_cache_kernel_block_id.defined() || !model_input.sequence_lengths.defined()) {
        return;
    }
    if (cache_manager_ == nullptr) {
        return;
    }
    const int sbp = static_cast<int>(cache_manager_->cacheConfig().seq_size_per_block);
    if (sbp <= 0) {
        return;
    }

    auto seq_len_cpu  = model_input.sequence_lengths.to(torch::kCPU);
    auto block_id_cpu = model_input.kv_cache_kernel_block_id.to(torch::kCPU);
    if (seq_len_cpu.scalar_type() != torch::kInt32 || block_id_cpu.scalar_type() != torch::kInt32) {
        return;
    }
    if (block_id_cpu.dim() != 3) {
        return;
    }

    const auto    all_streams = stream_groups.allStreams();
    const int64_t group_dim   = block_id_cpu.size(0);
    const int64_t batch_dim   = block_id_cpu.size(1);
    const int64_t max_blocks  = block_id_cpu.size(2);
    const int     batch       = static_cast<int>(seq_len_cpu.numel());

    auto dump_row = [&](int64_t g, int64_t b) {
        std::ostringstream oss;
        oss << "[";
        auto row = block_id_cpu.select(0, g).select(0, b);
        for (int64_t i = 0; i < row.size(0); ++i) {
            oss << row[i].item<int32_t>();
            if (i + 1 < row.size(0))
                oss << ",";
        }
        oss << "]";
        return oss.str();
    };

    const auto*        sl         = seq_len_cpu.data_ptr<int32_t>();
    bool               found_null = false;
    std::ostringstream summary;
    summary << "[debug-target-verify] batch=" << batch << " sbp=" << sbp << " group_dim=" << group_dim
            << " batch_dim=" << batch_dim << " max_blocks=" << max_blocks;
    for (int b = 0; b < batch && b < batch_dim; ++b) {
        const int seq_len   = sl[b];
        const int read_off  = (seq_len - 2) / sbp;
        int64_t   stream_id = -1;
        if (b < static_cast<int>(all_streams.size())) {
            auto it = all_streams.begin();
            std::advance(it, b);
            if (*it) {
                stream_id = (*it)->streamId();
            }
        }
        for (int64_t g = 0; g < group_dim; ++g) {
            std::string row_dump;
            if (always_print || (read_off >= 0 && read_off < max_blocks)) {
                row_dump = dump_row(g, b);
            }
            if (read_off < 0 || read_off >= max_blocks) {
                RTP_LLM_LOG_ERROR(
                    "[debug-target-verify] OOB read_off batch=%d stream=%ld group=%ld seq_len=%d read_off=%d max_blocks=%ld row=%s",
                    b,
                    stream_id,
                    g,
                    seq_len,
                    read_off,
                    max_blocks,
                    dump_row(g, b).c_str());
                found_null = true;
                continue;
            }
            const int32_t bid = block_id_cpu.select(0, g).select(0, b).index({read_off}).item<int32_t>();
            if (always_print) {
                summary << "\n  batch=" << b << " stream=" << stream_id << " group=" << g << " seq_len=" << seq_len
                        << " read_off=" << read_off << " bid=" << bid << " row=" << row_dump;
            }
            if (bid == -1) {
                RTP_LLM_LOG_ERROR(
                    "[debug-target-verify] NULL block_id at kernel read batch=%d stream=%ld group=%ld seq_len=%d read_off=%d row=%s",
                    b,
                    stream_id,
                    g,
                    seq_len,
                    read_off,
                    row_dump.c_str());
                found_null = true;
            }
        }
    }
    if (always_print) {
        RTP_LLM_LOG_INFO("%s", summary.str().c_str());
    }
    RTP_LLM_CHECK_WITH_INFO(!found_null,
                            "linear cache NULL at kernel read position — see [debug-target-verify] log lines above");
}

void MtpExecutor::broadcastPostRejectionInputs(GptModelInputs& model_input, const StreamGroups& stream_groups) {
    RTP_LLM_PROFILE_SCOPE("executor.mtp.decode_step(tp_sync_post_rejection)");
    const auto& mtp_cache_cfg = cache_manager_->getMTPModuleCacheConfig(0);
    // Broadcast only fields updated after rejection sampling. They are all
    // device-resident, so this stays NCCL-only and rank 0's rejection-sampled
    // view replaces non-root local target-verify outputs.
    if (parallelism_config_.tp_size > 1) {
        execBroadcast({{model_input.combo_tokens}, 0});
        execBroadcast({{model_input.last_hidden_states}, 0});
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
    model_input.kv_block_stride_bytes   = mtp_cache_cfg.kv_block_stride_bytes;
    model_input.kv_scale_stride_bytes   = mtp_cache_cfg.kv_scale_stride_bytes;
    model_input.kv_cache_layer_to_group = draft_kv_cache_layer_to_group;
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
    const bool use_sp_prefill_cuda_graph = sp_prefill_draft_model_ != nullptr;
    const bool any_fake_in_dp            = model_input.is_fake_stream;  // diagnostic only
    model_input.mtp_iteration_step       = 0;
    RTP_LLM_PROFILE_SCOPE_DYNAMIC(
        "executor.mtp.decode_step(draft_model_forward,use_sp=%d,sp_cg=%d,sp_prefill_cg=%d,is_fake=%d,any_fake_dp=%d)",
        static_cast<int>(use_sp_prefill_cuda_graph),
        static_cast<int>(sp_prefill_draft_model_ ? sp_prefill_draft_model_->cudaGraphEnabled() : false),
        static_cast<int>(sp_prefill_draft_model_ ? sp_prefill_draft_model_->prefillCudaGraphMode() : false),
        static_cast<int>(model_input.is_fake_stream),
        static_cast<int>(any_fake_in_dp));
    maybePrintModelInput(model_input, "decode post draft model");
    ensureModelInputsOnCuda(model_input, "decode.draft_prefill_forward");
    logMtpDecodeModelInput("draft_prefill_forward_input", model_input);
    const bool cp_context_request = isCpContextRequest(parallelism_config_, model_input);
    // Use sp_prefill_draft_model_ if CUDA graph is enabled, otherwise use draft_model_.
    GptModelOutputs         draft_prefill_model_output;
    static std::atomic<int> draft_prefill_choice_log_budget{16};
    if (draft_prefill_choice_log_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
        RTP_LLM_LOG_INFO(
            "[MTP decode] draft prefill model choice use_sp_prefill=%d sp_exists=%d sp_cg=%d "
            "sp_prefill_cg=%d is_fake_stream=%d any_fake_in_dp=%d",
            static_cast<int>(use_sp_prefill_cuda_graph),
            static_cast<int>(sp_prefill_draft_model_ != nullptr),
            static_cast<int>(sp_prefill_draft_model_ ? sp_prefill_draft_model_->cudaGraphEnabled() : false),
            static_cast<int>(sp_prefill_draft_model_ ? sp_prefill_draft_model_->prefillCudaGraphMode() : false),
            static_cast<int>(model_input.is_fake_stream),
            static_cast<int>(any_fake_in_dp));
    }
    if (use_sp_prefill_cuda_graph) {
        draft_prefill_model_output = sp_prefill_draft_model_->forward(model_input);
        if (!cp_context_request) {
            maybeOverrideLastHiddenWithMtpBuffer(draft_prefill_model_output, *sp_prefill_draft_model_);
        }
        draft_model_->copyMtpIterationTopkCacheFrom(*sp_prefill_draft_model_);
        if (debugCompareSpPrefillEnabled()) {
            auto clone_if_defined = [](const torch::Tensor& t) { return t.defined() ? t.clone() : torch::Tensor(); };
            auto graph_logits_snapshot     = clone_if_defined(draft_prefill_model_output.logits);
            auto graph_hidden_snapshot     = clone_if_defined(draft_prefill_model_output.hidden_states);
            auto graph_all_hidden_snapshot = clone_if_defined(draft_prefill_model_output.all_hidden_states);
            auto graph_kv_snapshot         = debugMtpPrefillDataEnabled() ?
                                                 tryGetMtpDebugKvCache(sp_prefill_draft_model_.get(), 0, 8) :
                                                 torch::Tensor();
            RTP_LLM_LOG_INFO(
                "[debug-sp-prefill] input combo=%s input_lengths=%s prefix_lengths=%s lm_output_indexes=%s "
                "last_hidden=%s kv_kernel=%s kv_block=%s",
                debugTensorSummary(model_input.combo_tokens).c_str(),
                debugTensorSummary(model_input.input_lengths).c_str(),
                debugTensorSummary(model_input.prefix_lengths).c_str(),
                debugTensorSummary(model_input.lm_output_indexes).c_str(),
                debugTensorSummary(model_input.last_hidden_states).c_str(),
                debugTensorSummary(model_input.kv_cache_kernel_block_id).c_str(),
                debugTensorSummary(model_input.kv_cache_block_id).c_str());
            auto eager_output = draft_model_->forward(model_input);
            if (!cp_context_request) {
                maybeOverrideLastHiddenWithMtpBuffer(eager_output, *draft_model_);
            }
            auto eager_kv_snapshot =
                debugMtpPrefillDataEnabled() ? tryGetMtpDebugKvCache(draft_model_.get(), 0, 8) : torch::Tensor();
            logMtpPrefillStageDiffs(sp_prefill_draft_model_.get(),
                                    draft_model_.get(),
                                    model_input.combo_tokens.defined() ? model_input.combo_tokens.size(0) : -1);
            if (debugMtpPrefillDataEnabled()) {
                RTP_LLM_LOG_INFO("[debug-mtp-prefill-data] layer0_kv_cache graph=%s eager=%s diff=%s",
                                 debugTensorSummary(graph_kv_snapshot, 0).c_str(),
                                 debugTensorSummary(eager_kv_snapshot, 0).c_str(),
                                 debugTensorDiffSummary(graph_kv_snapshot, eager_kv_snapshot).c_str());
            }
            RTP_LLM_LOG_INFO("[debug-sp-prefill] graph logits=%s topk=%s",
                             debugTensorSummary(draft_prefill_model_output.logits).c_str(),
                             debugLogitsTopKSummary(draft_prefill_model_output.logits).c_str());
            RTP_LLM_LOG_INFO("[debug-sp-prefill] eager logits=%s topk=%s",
                             debugTensorSummary(eager_output.logits).c_str(),
                             debugLogitsTopKSummary(eager_output.logits).c_str());
            RTP_LLM_LOG_INFO("[debug-sp-prefill] logits_diff=%s hidden_diff=%s all_hidden_diff=%s",
                             debugTensorDiffSummary(graph_logits_snapshot, eager_output.logits).c_str(),
                             debugTensorDiffSummary(graph_hidden_snapshot, eager_output.hidden_states).c_str(),
                             debugTensorDiffSummary(graph_all_hidden_snapshot, eager_output.all_hidden_states).c_str());
            RTP_LLM_LOG_INFO(
                "[debug-sp-prefill] graph_output_mutation logits=%s hidden=%s all_hidden=%s",
                debugTensorDiffSummary(draft_prefill_model_output.logits, graph_logits_snapshot).c_str(),
                debugTensorDiffSummary(draft_prefill_model_output.hidden_states, graph_hidden_snapshot).c_str(),
                debugTensorDiffSummary(draft_prefill_model_output.all_hidden_states, graph_all_hidden_snapshot)
                    .c_str());
        }
    } else {
        draft_prefill_model_output = draft_model_->forward(model_input);
        if (!cp_context_request) {
            maybeOverrideLastHiddenWithMtpBuffer(draft_prefill_model_output, *draft_model_);
        }
    }
    model_input.mtp_iteration_step = -1;
    logMtpDecodeModelOutput("draft_prefill_forward_output", draft_prefill_model_output, model_input.is_fake_stream);
    return draft_prefill_model_output;
}

void MtpExecutor::collectDecodeMetrics(const StreamGroups&                          stream_groups,
                                       torch::Event&                                accept_len_ready_event,
                                       const speculative::SpeculativeSamplerOutput& speculative_sampler_output,
                                       MtpMetricsCollector&                         metrics_collector) {
    RTP_LLM_PROFILE_SCOPE("executor.mtp.decode_step(collect_metrics)");
    auto& executor_collector  = metrics_collector.executor_collector;
    auto& sp_engine_collector = metrics_collector.sp_engine_collector;

    const auto accept_len_metrics = consumePendingAcceptLenMetrics();
    stageAcceptLenMetrics(speculative_sampler_output.accept_len, accept_len_ready_event, stream_groups.size());
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
                                               const std::list<GenerateStreamPtr>&          streams,
                                               const speculative::SpeculativeSamplerOutput& speculative_sampler_output,
                                               GptModelOutputs                              draft_prefill_model_output,
                                               SamplerOutput                 draft_prefill_sampler_output,
                                               std::shared_ptr<torch::Event> rejection_event,
                                               std::shared_ptr<torch::Event> draft_event) {
    RTP_LLM_PROFILE_SCOPE("executor.mtp.decode_step(dispatch_output)");
    absl::Status result;
    if (useStreamAsync()) {
        // Hand off to a worker that waits on main-stream rejection/draft events
        // via cudaStreamWaitEvent; the main thread returns immediately.
        result = dispatchDecodeAsync(stream_groups,
                                     speculative_sampler_output,
                                     {std::move(draft_prefill_model_output), std::move(draft_prefill_sampler_output)},
                                     std::move(rejection_event),
                                     std::move(draft_event));
    } else {
        MergedOutput draft_prefill_output{std::move(draft_prefill_model_output),
                                          std::move(draft_prefill_sampler_output)};
        result =
            batch_stream_processor_->dispatchDecode(stream_groups, speculative_sampler_output, draft_prefill_output);
        if (result.ok()) {
            publishSyncMtpDeviceState(stream_groups, speculative_sampler_output, draft_prefill_output);
        }
    }
    return result;
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
            stream->setScoreLen(propose_step_ + 1);
            if (stream->getSPOutputBuffer() == nullptr && stream->isPerfTest()) {
                auto sp_output_buffer =
                    makeFakeSPOutputBuffer(data_type_, hidden_size_, draft_vocab_size_, propose_step_);
                stream->setSPOutputBuffer(sp_output_buffer);
            }
            decode_streams.push_back(stream);
        }

        // init sp output buffer if not exist
        stream->setReturnAllProbs(true);
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
        ensureSpOutputTokenGpuMirrors(sp_output_buffer);
    }
}

absl::Status MtpExecutor::process(const std::list<GenerateStreamPtr>& streams, int64_t schedule_time_us) {
    RTP_LLM_PROFILE_SCOPE_DYNAMIC("executor.mtp.process(stream_size=%zu,mtp_step=%zu)", streams.size(), propose_step_);

    const int64_t process_start_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
    if (schedule_time_us <= 0) {
        schedule_time_us = process_start_time_us;
    }
    MtpMetricsCollector metrics_collector;
    auto                tps_active_guard =
        tps_reporter_.makeActiveGuard(metrics_reporter_ && isTpRank0() && !warm_up_ && !streams.empty());

    std::list<GenerateStreamPtr> prefill_streams;
    std::list<GenerateStreamPtr> decode_streams;

    // prepare streams
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

    const auto& mtp_cache_cfg         = cache_manager_->getMTPModuleCacheConfig(0);
    model_input.kv_block_stride_bytes = mtp_cache_cfg.kv_block_stride_bytes;
    model_input.kv_scale_stride_bytes = mtp_cache_cfg.kv_scale_stride_bytes;

    GptModelOutputs            draft_decode_model_output;
    std::vector<torch::Tensor> draft_token_columns;
    torch::Tensor              spec_prefix_lengths;

    // update TP > 0 batch_size
    size_t batch_size = model_input.combo_tokens.size(0);
    logMtpDecodeModelInput("draft_decode_begin", model_input);
    const auto cuda_i32         = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    auto       to_cuda_i32_flat = [this, batch_size](const torch::Tensor& tensor) -> torch::Tensor {
        auto tensor_d = toCudaInt32WithHostHold(tensor, buffer_holder_);
        tensor_d      = tensor_d.reshape({static_cast<int64_t>(batch_size)});
        return tensor_d.is_contiguous() ? tensor_d : tensor_d.contiguous();
    };
    spec_prefix_lengths = model_input.sequence_lengths.defined() ?
                              toCudaInt32WithHostHold(model_input.sequence_lengths, buffer_holder_) :
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
    // The legacy host fallback reads sp_output_buffer->tokens from worker-side
    // specUpdate and races when DROP_BROAD_SYNC=1.
    {
        RTP_LLM_PROFILE_SCOPE("executor.mtp.draft_model_decode(pre_target_device_gather)");
        std::vector<torch::Tensor> pre_target_slices_gpu;
        pre_target_slices_gpu.reserve(batch_size);
        bool all_device_state = !all_streams.empty();
        for (const auto& stream : all_streams) {
            const auto& accept_tokens = stream->getAcceptTokensGpu();
            const auto& accept_len    = stream->getAcceptLenGpu();
            if (!accept_tokens.defined() || !accept_tokens.is_cuda() || !accept_len.defined()
                || !accept_len.is_cuda()) {
                all_device_state = false;
                break;
            }
            auto idx_t = (accept_len - 1).to(torch::kLong);
            pre_target_slices_gpu.push_back(accept_tokens.squeeze(0).index_select(/*dim=*/0, idx_t));
        }
        if (all_device_state && pre_target_slices_gpu.size() == batch_size && !pre_target_slices_gpu.empty()) {
            pre_target_token_t = torch::cat(pre_target_slices_gpu, 0).to(torch::kInt32);
        } else if (all_streams.empty()) {
            // Non-root TP ranks have no GenerateStream objects here; rank 0
            // broadcasts the assembled combo_tokens later, so this placeholder
            // only needs to be device-resident and shape-correct.
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
        logMtpDecodeModelInput("draft_decode_loop_forward_input", model_input);
        draft_decode_model_output      = std::move(draft_model_->forward(model_input));
        model_input.mtp_iteration_step = -1;
        model_forward_us += autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
        maybeOverrideLastHiddenWithMtpBuffer(draft_decode_model_output, *draft_model_);
        logMtpDecodeModelOutput(
            "draft_decode_loop_forward_output", draft_decode_model_output, model_input.is_fake_stream);
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

        model_input.input_lengths      = std::move(input_lengths);
        model_input.prefix_lengths     = spec_prefix_lengths;
        model_input.combo_tokens       = draft_token_ids_t.reshape({(int64_t)(batch_size * (propose_step_ + 1))});
        model_input.sequence_lengths   = torch::empty({0}, torch::TensorOptions(torch::kInt32).device(torch::kCUDA));
        model_input.last_hidden_states = torch::Tensor();
        ensureModelInputsOnCuda(model_input, "draft_decode.build_spec_decode_input");
        logMtpDecodeModelInput("draft_decode_spec_input", model_input);

        // Since other tp ranks don't have streams, its combo_tokens' first token is not correct.
        // Thus, we need to broadcast the combo_tokens to other tp ranks.
        if (parallelism_config_.tp_size > 1) {
            RTP_LLM_PROFILE_SCOPE("executor.mtp.draft_model_decode(build_spec_tp_broadcast)");
            execBroadcast({{model_input.combo_tokens}, 0});
        }

        const auto& cache_cfg             = cache_manager_->cacheConfig();
        model_input.kv_block_stride_bytes = cache_cfg.kv_block_stride_bytes;
        model_input.kv_scale_stride_bytes = cache_cfg.kv_scale_stride_bytes;
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

    const auto cuda_i32    = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
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

    int64_t idx              = 0;
    int64_t hidden_token_off = 0;
    int64_t probs_batch_off  = 0;
    for (auto& stream : all_streams) {
        torch::Tensor accept_len_slice    = accept_len_all.narrow(0, idx, 1);
        torch::Tensor accept_tokens_slice = accept_tokens_all.narrow(0, idx, 1);
        torch::Tensor propose_tokens_slice =
            propose_tokens_all.defined() ? propose_tokens_all.narrow(0, idx, 1) : torch::Tensor();
        torch::Tensor target_token_slice;
        if (accept_tokens_slice.defined() && accept_len_slice.defined()) {
            auto target_idx    = (accept_len_slice - 1).to(torch::kLong);
            target_token_slice = accept_tokens_slice.squeeze(0).index_select(/*dim=*/0, target_idx).to(torch::kInt32);
        }

        auto sp_output_buffer = stream->getSPOutputBuffer();
        if (sp_output_buffer) {
            sp_output_buffer->target_token_gpu   = target_token_slice;
            sp_output_buffer->propose_tokens_gpu = propose_tokens_slice;
        }

        // Synchronous dispatch has already run GenerateStream::specUpdate, so
        // host seqLength is the authoritative committed length for this stream.
        torch::Tensor next_seq_len_gpu = torch::full({1}, static_cast<int64_t>(stream->seqLength()), cuda_i32);

        torch::Tensor last_hidden_states_gpu;
        torch::Tensor draft_all_probs_slice_gpu;
        const auto    next_batch_size = stream->nextBatchSize();
        if (propose_step_ > 1 && draft_all_hidden_full.defined()) {
            const auto stream_hidden_len = static_cast<int64_t>(propose_step_ + 1);
            if (draft_all_hidden_full.size(0) >= hidden_token_off + stream_hidden_len) {
                auto stream_hidden     = draft_all_hidden_full.narrow(0, hidden_token_off, stream_hidden_len);
                auto idx_t             = (accept_len_slice - 1).to(torch::kLong);
                last_hidden_states_gpu = stream_hidden.index_select(/*dim=*/0, idx_t);
            } else if (stream->getSPOutputBuffer() && stream->getSPOutputBuffer()->hidden_states.defined()) {
                const auto& hidden     = stream->getSPOutputBuffer()->hidden_states;
                last_hidden_states_gpu = toCudaWithHostHold(hidden, buffer_holder_);
            }
        }
        if (draft_all_probs_full.defined() && next_batch_size > 0) {
            draft_all_probs_slice_gpu = draft_all_probs_full.narrow(0, probs_batch_off, next_batch_size).clone();
        }

        GenerateStream::MtpAsyncDeviceState state;
        state.accept_len_gpu         = std::move(accept_len_slice);
        state.accept_tokens_gpu      = std::move(accept_tokens_slice);
        state.next_seq_len_gpu       = std::move(next_seq_len_gpu);
        state.propose_tokens_gpu     = std::move(propose_tokens_slice);
        state.last_hidden_states_gpu = std::move(last_hidden_states_gpu);
        state.draft_all_probs_gpu    = std::move(draft_all_probs_slice_gpu);
        // Sync dispatch already ran specUpdate; host seqLength is authoritative.
        state.last_real_seq_len = stream->seqLength();
        state.next_real_seq_len = state.last_real_seq_len;
        stream->setMtpAsyncDeviceState(std::move(state));

        hidden_token_off += static_cast<int64_t>(propose_step_ + 1);
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

    const auto& accept_len_gpu_all    = spec_decode_output.accept_len;
    const auto& accept_tokens_gpu_all = spec_decode_output.accept_tokens;
    // Publish per-stream propose token slices for next-step combo_tokens
    // without waiting for worker-side specUpdate.
    const auto& propose_tokens_gpu_all = draft_prefill_output.sampler_output.token_ids;
    // Publish hidden states and all_probs on the main thread; otherwise next-
    // step readers can race worker writes when DROP_BROAD_SYNC=1.
    const auto& draft_all_hidden_full = draft_prefill_output.model_output.all_hidden_states;
    const auto& draft_all_probs_full  = draft_prefill_output.sampler_output.all_probs;

    auto all_streams = stream_groups.allStreams();

    // Attach per-stream device state. Chain next_seq_len from prior device
    // state when available; host seqLength may be stale while the previous
    // worker is in flight under DROP_BROAD_SYNC.
    int64_t idx              = 0;
    int64_t hidden_token_off = 0;  // offset into draft_all_hidden_full per stream
    int64_t probs_batch_off  = 0;  // offset into draft_all_probs_full per stream
    for (auto& stream : all_streams) {
        torch::Tensor accept_len_slice =
            accept_len_gpu_all.defined() ? accept_len_gpu_all.narrow(0, idx, 1) : torch::Tensor();
        torch::Tensor accept_tokens_slice =
            accept_tokens_gpu_all.defined() ? accept_tokens_gpu_all.narrow(0, idx, 1) : torch::Tensor();
        torch::Tensor propose_tokens_slice =
            propose_tokens_gpu_all.defined() ? propose_tokens_gpu_all.narrow(0, idx, 1) : torch::Tensor();
        torch::Tensor target_token_slice;
        if (accept_tokens_slice.defined() && accept_len_slice.defined()) {
            auto target_idx    = (accept_len_slice - 1).to(torch::kLong);
            target_token_slice = accept_tokens_slice.squeeze(0).index_select(/*dim=*/0, target_idx).to(torch::kInt32);
        }

        auto sp_output_buffer = stream->getSPOutputBuffer();
        if (sp_output_buffer) {
            // Publish GPU mirrors before the bookkeeping worker launches. The
            // next decode step can then stay on device even with DROP_BROAD_SYNC.
            sp_output_buffer->target_token_gpu   = target_token_slice;
            sp_output_buffer->propose_tokens_gpu = propose_tokens_slice;
        }

        torch::Tensor next_seq_len_gpu;
        if (accept_len_slice.defined()) {
            const auto&   prev_next_seq_len = stream->getNextSeqLenGpu();
            torch::Tensor cur_seq_len_t;
            if (prev_next_seq_len.defined() && prev_next_seq_len.is_cuda()) {
                cur_seq_len_t = prev_next_seq_len;
            } else {
                cur_seq_len_t = torch::full({1},
                                            static_cast<int64_t>(stream->seqLength()),
                                            torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
            }
            next_seq_len_gpu = (cur_seq_len_t + accept_len_slice).to(torch::kInt32);
        }

        // Keep per-stream hidden states/all_probs on the main stream: no D2H,
        // no .item(), no synchronize.
        torch::Tensor last_hidden_states_gpu;
        torch::Tensor draft_all_probs_slice_gpu;
        const auto    next_batch_size = stream->nextBatchSize();
        if (propose_step_ > 1 && draft_all_hidden_full.defined() && accept_len_slice.defined()) {
            // Per-stream rows [hidden_token_off, hidden_token_off + propose+1)
            // contain the draft prefill hidden states. The last accepted
            // position's hidden state is at offset (accept_len - 1).
            const auto stream_hidden_len = static_cast<int64_t>(propose_step_ + 1);
            if (draft_all_hidden_full.size(0) >= hidden_token_off + stream_hidden_len) {
                auto stream_hidden     = draft_all_hidden_full.narrow(0, hidden_token_off, stream_hidden_len);
                auto idx_t             = (accept_len_slice - 1).to(torch::kLong);
                last_hidden_states_gpu = stream_hidden.index_select(/*dim=*/0, idx_t);
            }
        }
        if (draft_all_probs_full.defined() && next_batch_size > 0) {
            // Clone to break aliasing with draft_prefill_copy storage, which
            // may be released before the next step reads this view.
            draft_all_probs_slice_gpu = draft_all_probs_full.narrow(0, probs_batch_off, next_batch_size).clone();
        }

        GenerateStream::MtpAsyncDeviceState state;
        state.accept_len_gpu         = std::move(accept_len_slice);
        state.accept_tokens_gpu      = std::move(accept_tokens_slice);
        state.next_seq_len_gpu       = std::move(next_seq_len_gpu);
        state.propose_tokens_gpu     = std::move(propose_tokens_slice);
        state.last_hidden_states_gpu = std::move(last_hidden_states_gpu);
        state.draft_all_probs_gpu    = std::move(draft_all_probs_slice_gpu);
        // The next iteration may run before this step's worker-side specUpdate
        // commits accept_len on the host. Use the currently committed real
        // length as the base and add one verify window for the KV allocation
        // upper bound. Do not chain from the previous upper bound.
        state.last_real_seq_len = stream->seqLength();
        state.next_real_seq_len = state.last_real_seq_len + static_cast<int>(propose_step_ + 1);
        stream->setMtpAsyncDeviceState(std::move(state));

        hidden_token_off += static_cast<int64_t>(propose_step_ + 1);
        probs_batch_off += next_batch_size;
        ++idx;
    }

    // Do not bump host seqLength: reserve_step_ already provides speculative
    // KV budget, and next_seq_len_gpu carries post-step length for next prepare.
    // The worker stream guard makes prepareDecodeSpecUpdateInfo D2H block only the worker.
    auto* processor          = batch_stream_processor_.get();
    auto  spec_decode_copy   = spec_decode_output;
    auto  draft_prefill_copy = std::move(draft_prefill_output);
    auto  stream_groups_copy = stream_groups;

    // Claim each stream's KV resource before worker launch so releaseResource
    // defers while the worker still reads those blocks.
    auto streams_for_inc = stream_groups_copy.allStreams();
    for (auto& s : streams_for_inc) {
        s->incPendingAsyncBookkeeping();
    }

    spec_bookkeeping_runner_.launch([processor,
                                     stream_groups_copy = std::move(stream_groups_copy),
                                     spec_decode_copy   = std::move(spec_decode_copy),
                                     draft_prefill_copy = std::move(draft_prefill_copy),
                                     rejection_event,
                                     draft_event]() mutable {
        RTP_LLM_PROFILE_SCOPE("executor.mtp.decode_step(spec_bookkeeping_worker)");

        auto worker_streams = stream_groups_copy.allStreams();

        // RAII: even if dispatchDecode throws, every captured stream gets a dec
        // exactly once. Capturing worker_streams by value into the deleter keeps
        // the GenerateStreamPtr refcount alive until after dec runs.
        auto dec_guard = std::shared_ptr<void>(nullptr, [worker_streams](void*) {
            for (auto& s : worker_streams) {
                s->decPendingAsyncBookkeepingAndMaybeRelease();
            }
        });

        // Queue worker-stream waits for rejection/draft events. The worker may
        // later CPU-sync only its own stream during prepareDecodeSpecUpdateInfo.
        // Null events are allowed for defensive compatibility.
        if (rejection_event) {
            rejection_event->block(cuda_graph::graphGetCurrentStream());
        }
        if (draft_event) {
            draft_event->block(cuda_graph::graphGetCurrentStream());
        }

        // Reuse synchronous bookkeeping on the worker stream. Per-stream
        // MtpAsyncDeviceState was already published for the next prepare.
        auto status = processor->dispatchDecode(stream_groups_copy, spec_decode_copy, draft_prefill_copy);
        if (!status.ok()) {
            RTP_LLM_LOG_ERROR("[stream-async] dispatchDecode (worker) failed: %s", status.ToString().c_str());
        }

        // Keep MtpAsyncDeviceState alive after specUpdate. Both sync and async
        // decode paths use it as the canonical next-step GPU state; the next
        // dispatchDecodeAsync overwrites it with a newer epoch.

        // Record a swap-done event for each stream so the next verify can wait
        // via cudaStreamWaitEvent. Recording on all streams is cheap and keeps
        // the consumer path uniform even when a stream did not actually swap.
        for (auto& stream : worker_streams) {
            auto event = std::make_shared<torch::Event>(cuda_graph::makeGraphEvent());
            event->record(cuda_graph::graphGetCurrentStream());
            stream->setPendingSwapDoneEvent(std::static_pointer_cast<void>(event));
        }
        // dec_guard destructs here, dec'ing each stream's pending count.
    });

    // Main thread returns immediately. The next step can be dispatched while
    // this step's bookkeeping is still in flight on the worker.
    return absl::OkStatus();
}

}  // namespace rtp_llm
