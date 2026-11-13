#include "rtp_llm/cpp/models/PyWrappedModel.h"
#if USING_CUDA
#include "rtp_llm/models_py/bindings/cuda/kernels/cuda_graph_prepare.h"
#endif
#include "rtp_llm/cpp/cuda_graph/prepared_attention_inputs_guard.h"
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/models/context_parallel/ZigzagTokenLayout.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/cpp/utils/DebugUtils.h"
#include "rtp_llm/cpp/utils/utils.h"
#include "rtp_llm/cpp/utils/math_utils.h"
#include "rtp_llm/cpp/model_utils/AttentionConfig.h"
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <mutex>
#include <utility>
#include <vector>
#include <unordered_set>
#include "rtp_llm/cpp/pybind/PyUtils.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#if USING_CUDA
#include <c10/cuda/CUDAStream.h>
#include "rtp_llm/models_py/bindings/cuda/Bf16GemmOp.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/attention_input_metadata.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/scaled_fp8_quant.h"
#include <ATen/ops/_scaled_mm.h>
#endif

using namespace std;

namespace rtp_llm {

namespace {

// Pairs init() with a full drain on exceptional exits. DSpark's normal path
// finishes only writer submissions here and leaves actual publication for the
// executor's pre-dispatch barrier, preserving overlap with draft computation.
class CacheStoreWriteCycleGuard {
public:
    CacheStoreWriteCycleGuard(const std::shared_ptr<CacheStoreAsyncWriter>& writer,
                              bool                                          has_work,
                              bool                                          track_store_completions):
        writer_(writer), active_(has_work), track_store_completions_(track_store_completions) {
        if (active_) {
            writer_->init(track_store_completions_);
        }
    }

    ~CacheStoreWriteCycleGuard() {
        if (!active_) {
            return;
        }
        active_ = false;
        // Swallowed on purpose: this runs while another exception may be unwinding.
        try {
            writer_->waitAllDone();
        } catch (const std::exception& e) {
            RTP_LLM_LOG_ERROR("failed to drain CacheStore writer while unwinding forward: %s", e.what());
        } catch (...) {
            RTP_LLM_LOG_ERROR("failed to drain CacheStore writer while unwinding forward: unknown exception");
        }
    }

    // Normal path: let a drain failure propagate to the caller. Tracked cycles
    // keep publication pending for MtpExecutor; legacy cycles drain everything.
    void finish() {
        if (!active_) {
            return;
        }
        if (track_store_completions_) {
            writer_->finishSubmissions();
            active_ = false;
        } else {
            active_ = false;
            writer_->waitAllDone();
        }
    }

    CacheStoreWriteCycleGuard(const CacheStoreWriteCycleGuard&)            = delete;
    CacheStoreWriteCycleGuard& operator=(const CacheStoreWriteCycleGuard&) = delete;

private:
    std::shared_ptr<CacheStoreAsyncWriter> writer_;
    bool                                   active_{false};
    bool                                   track_store_completions_{false};
};

}  // namespace

GptModelInputs PyWrappedModel::padSequenceParallelInputs(const GptModelInputs& source) {
    if (!sequence_parallel_padding_enabled_ || source.sp_logical_tokens > 0) {
        return source;
    }
    GptModelInputs out      = source;
    const int64_t  tp       = device_props_.tp_size;
    const int64_t  requests = source.input_lengths.numel();
    const int64_t  tokens   = source.combo_tokens.numel();
    RTP_LLM_CHECK_WITH_INFO(tp > 0 && requests > 0 && tokens > 0, "SP requires nonempty input");
    // Draft updates keep complete Q-wide rows in every scheduling mode.
    // Accepted lengths select outputs and advance only the logical cache state.
    const bool prefill =
        source.sequence_lengths.numel() == 0 && !source.is_target_verify && !source.is_mtp_draft_update;
    int64_t extra_requests, extra_tokens, dummy_width;
    if (prefill) {
        extra_tokens   = pad(tokens, tp) - tokens;
        extra_requests = extra_tokens ? 1 : 0;
        dummy_width    = extra_tokens;
    } else {
        RTP_LLM_CHECK_WITH_INFO(tokens % requests == 0, "SP draft/verify requires complete request rows");
        dummy_width    = tokens / requests;
        extra_requests = padSequenceParallelRequests(requests, dummy_width, tp) - requests;
        extra_tokens   = extra_requests * dummy_width;
    }
    out.sp_logical_requests = requests;
    out.sp_logical_tokens   = tokens;
    if (!extra_tokens) {
        return out;
    }
#if USING_CUDA
    SequenceParallelPadParams pad_params;
    const auto                enqueue_pad = [&pad_params,
                              prefill](const torch::Tensor& src, const torch::Tensor& dst, int64_t dim, int64_t value) {
        // Prefill may pass temporary H2D tensors; keep its copies immediate.
        // Nonzero floating fills need typed ATen conversion rather than integer bits.
        if (prefill || !src.is_cuda() || !src.is_contiguous() || (value != 0 && src.is_floating_point())) {
            return false;
        }
        RTP_LLM_CHECK_WITH_INFO(pad_params.region_count < kMaxCudaGraphPrepareFillRegions,
                                "too many SP padding regions");
        auto& region       = pad_params.regions[pad_params.region_count++];
        region.source      = static_cast<const uint8_t*>(src.data_ptr());
        region.destination = static_cast<uint8_t*>(dst.data_ptr());
        region.groups      = 1;
        for (int64_t i = 0; i < dim; ++i)
            region.groups *= src.size(i);
        region.source_group_bytes      = src.nbytes() / region.groups;
        region.destination_group_bytes = dst.nbytes() / region.groups;
        region.element_bytes           = src.element_size();
        region.fill_bits               = static_cast<uint64_t>(value);
        return true;
    };
    const auto flush_padding = [&pad_params] {
        if (pad_params.region_count > 0) {
            invokeSequenceParallelPad(pad_params, cuda_graph::graphGetCurrentStream().stream());
        }
    };
#else
    const auto enqueue_pad   = [](const torch::Tensor&, const torch::Tensor&, int64_t, int64_t) { return false; };
    const auto flush_padding = [] {};
#endif
    const auto append = [&](const torch::Tensor& tensor, int64_t rows, int64_t value, int64_t dim = 0) {
        if (!tensor.defined() || tensor.numel() == 0 || rows == 0)
            return tensor;
        auto       shape         = tensor.sizes().vec();
        const auto original_rows = shape[dim];
        shape[dim] += rows;
        // Allocate the final host buffer pinned from the start. Concatenating
        // and then pinning copied every input twice, including long Prefill
        // token arrays, on every rank before the first GPU kernel could run.
        auto options = tensor.options();
        if (tensor.device().is_cpu() && tensor.is_pinned()) {
            options = options.pinned_memory(true);
        }
        auto padded = torch::empty(shape, options);
        if (enqueue_pad(tensor, padded, dim, value)) {
            return padded;
        }
        padded.narrow(dim, 0, original_rows).copy_(tensor);
        padded.narrow(dim, original_rows, rows).fill_(value);
        return padded;
    };
    if (const auto* target_replay = source.targetLinearReplay()) {
        auto          replay = *target_replay;
        const int64_t batch  = replay.slot_ids.numel();
        RTP_LLM_CHECK_WITH_INFO(batch == requests, "LINEAR replay request count disagrees with SP input");
        for (auto* tensor : replay.tensors()) {
            RTP_LLM_CHECK_WITH_INFO(tensor->defined() && (tensor->dim() == 1 || tensor->dim() == 2)
                                        && tensor->size(tensor->dim() - 1) == requests,
                                    "LINEAR replay request rows disagree");
            const int64_t fill = tensor == &replay.slot_ids || tensor == &replay.active_block_ids
                                         || tensor == &replay.state_read_block_ids ?
                                     -1 :
                                     0;
            *tensor            = append(*tensor, extra_requests, fill, tensor->dim() - 1);
        }
        out.linear_replay = std::move(replay);
    }
    const auto append_prefill_token =
        [this, &append, prefill](const torch::Tensor& tensor, int64_t rows, int64_t value) {
            if (!prefill || !tensor.defined() || tensor.numel() == 0 || rows == 0 || !tensor.device().is_cpu()
                || !tensor.is_pinned()) {
                return append(tensor, rows, value);
            }
            // Keep the original pinned input alive while its H2D copy is queued.
            // Padding on CUDA avoids copying the full token array into a second
            // pinned host allocation before each Prefill forward.
            buffer_holder_.hold_host(tensor);
            return append(tensor.to(torch::kCUDA, /*non_blocking=*/true), rows, value);
        };
    out.combo_tokens            = append_prefill_token(source.combo_tokens, extra_tokens, 0);
    out.last_hidden_states      = append(source.last_hidden_states, extra_tokens, 0);
    out.combo_position_ids      = append_prefill_token(source.combo_position_ids, extra_tokens, 0);
    out.combo_tokens_type_ids   = append(source.combo_tokens_type_ids, extra_tokens, 0);
    out.text_tokens_mask        = append(source.text_tokens_mask, extra_tokens, 1);
    out.input_lengths           = append(source.input_lengths, extra_requests, dummy_width);
    out.prefix_lengths          = append(source.prefix_lengths, extra_requests, 0);
    out.sequence_lengths        = append(source.sequence_lengths, extra_requests, 0);
    out.sequence_lengths_plus_1 = append(source.sequence_lengths_plus_1, extra_requests, 1);
    // Block zero is the cache manager's reserved null block, never a request block.
    if (source.kv_cache_block_id.defined())
        out.kv_cache_block_id = append(source.kv_cache_block_id, extra_requests, 0, source.kv_cache_block_id.dim() - 2);
    if (source.kv_cache_kernel_block_id.defined())
        out.kv_cache_kernel_block_id =
            append(source.kv_cache_kernel_block_id, extra_requests, 0, source.kv_cache_kernel_block_id.dim() - 2);
    flush_padding();
    return out;
}

torch::Tensor PyWrappedModel::tensorHoldHostAndToCuda(const torch::Tensor& tensor) {
    if (tensor.device().is_cuda()) {
        return tensor;
    }

    buffer_holder_.hold_host(tensor);

    if (tensor.numel() == 0) {
        return torch::empty(tensor.sizes(), torch::TensorOptions(tensor.dtype()).device(torch::kCUDA));
    }

    // NOTE: since is_pinned() operation costs a lot cpu time, we only check it when pinned_check_remaining_ > 0.
    if (pinned_check_remaining_ > 0) {
        RTP_LLM_CHECK_WITH_INFO(tensor.is_pinned(), "tensor is not pinned, fused copy requires pinned memory");
    }

    // create tensor on cuda
    auto cuda_tensor = torch::empty(tensor.sizes(), torch::TensorOptions(tensor.dtype()).device(torch::kCUDA));

    d2d_copies_.add(tensor.data_ptr(), cuda_tensor.data_ptr(), tensor.nbytes());

    return cuda_tensor;
}

GraphBase* PyWrappedModel::selectGraphRunner(const torch_ext::PyAttentionInputs& attention_inputs) const {
    if (!is_prefill_cuda_graph_mode_ && attention_inputs.is_prefill && generation_prefill_graph_runner_) {
        return generation_prefill_graph_runner_.get();
    }
    return graph_runner_.get();
}

CudaGraphState& PyWrappedModel::selectGraphState(const torch_ext::PyAttentionInputs& attention_inputs) {
    if (!is_prefill_cuda_graph_mode_ && attention_inputs.is_prefill && generation_prefill_graph_runner_) {
        return generation_prefill_cuda_graph_state_;
    }
    return graph_state_;
}

void PyWrappedModel::cleanupAfterConstructionFailure() noexcept {
    try {
        py::gil_scoped_acquire gil;
        held_attn_pyobj_   = py::object();
        py_forward_method_ = py::object();
        generation_prefill_graph_runner_.reset();
        graph_runner_.reset();
        if (py_model_ && py::hasattr(py_model_, "close")) {
            if (owns_py_model_) {
                py_model_.attr("close")();
            }
            py_model_ = py::object();
        } else {
            py_model_.release();
        }
    } catch (const std::exception& e) {
        // A failed drain must keep its storage owner alive. Never let member
        // destruction decrement this handle after the GIL guard has unwound.
        py_model_.release();
        RTP_LLM_LOG_ERROR("PyWrappedModel construction cleanup failed: %s", e.what());
    } catch (...) {
        py_model_.release();
        RTP_LLM_LOG_ERROR("PyWrappedModel construction cleanup failed with unknown exception");
    }
}

void PyWrappedModel::releaseBuffers() {
    if (held_attn_pyobj_.ptr()) {
        py::gil_scoped_acquire gil;
        held_attn_pyobj_ = py::object();
    }
    // TensorHolder release point (PyWrappedModel): advances model-internal
    // host staging buffers from tensorHoldHostAndToCuda()/holdInputsHostBuffers().
    buffer_holder_.release();
}

torch::Tensor PyWrappedModel::getMtpTargetHiddenStates(int64_t num_tokens) {
    if (!py_model_) {
        return torch::Tensor();
    }
    py::gil_scoped_acquire gil;
    if (!py::hasattr(py_model_, "get_mtp_target_hidden_states")) {
        return torch::Tensor();
    }
    py::object result = py_model_.attr("get_mtp_target_hidden_states")(num_tokens);
    return result.is_none() ? torch::Tensor() : result.cast<torch::Tensor>();
}

torch::Tensor PyWrappedModel::getMtpLastHiddenStates(int64_t num_tokens) {
    if (!py_model_) {
        return torch::Tensor();
    }
    py::gil_scoped_acquire gil;
    if (!py::hasattr(py_model_, "get_mtp_last_hidden_states")) {
        return torch::Tensor();
    }
    py::object result = py_model_.attr("get_mtp_last_hidden_states")(num_tokens);
    return result.is_none() ? torch::Tensor() : result.cast<torch::Tensor>();
}

bool PyWrappedModel::hasMtpTargetHiddenBuffer() const {
    return has_mtp_hidden_buffer_;
}

bool PyWrappedModel::requiresGlobalMtpPrefillHidden() const {
    return requires_global_mtp_prefill_hidden_;
}

torch::Tensor PyWrappedModel::restoreMtpPrefillHiddenStates(torch::Tensor hidden, const GptModelInputs& inputs) {
    RTP_LLM_CHECK_WITH_INFO(context_parallel_processor_ && mtp_prefill_cp_params_.has_value(),
                            "MTP CP restore requires the current prefill layout");
    const auto& cp = *mtp_prefill_cp_params_;
    RTP_LLM_CHECK_WITH_INFO(hidden.defined() && hidden.dim() == 2
                                && hidden.size(0) * device_props_.tp_size == cp.prefill_qkv_restore_indice.numel(),
                            "MTP CP hidden rows do not match the current rank-local layout");
    const auto rows = context_parallel_processor_->handleOutputs(hidden, inputs, cp);
    RTP_LLM_CHECK_WITH_INFO(rows == static_cast<size_t>(inputs.combo_tokens.numel()),
                            "restored MTP hidden rows must match the global draft tokens");
    return hidden;
}

bool PyWrappedModel::hasSpeculativeTargetCommitHooks() const {
    return has_speculative_target_commit_hooks_;
}

std::optional<bool> PyWrappedModel::reduceSpeculativeTargetPrepareStatus(bool local_ok) {
    if (!has_speculative_target_status_reducer_) {
        return std::nullopt;
    }
    py::gil_scoped_acquire gil;
    // Native waiting releases the GIL. Do not catch errors and switch transports:
    // the published generation may already have been consumed by the peer.
    return py_model_.attr("reduce_speculative_target_prepare_status")(local_ok).cast<bool>();
}

std::string PyWrappedModel::prepareSpeculativeTargetCommit(const torch::Tensor& accept_len) {
    if (!has_speculative_target_commit_hooks_) {
        return {};
    }
    py::gil_scoped_acquire gil;
    try {
        py_model_.attr("prepare_speculative_target_commit")(accept_len);
        return {};
    } catch (const std::exception& e) {
        RTP_LLM_LOG_ERROR("Speculative target commit prepare failed: %s", e.what());
        return e.what();
    } catch (...) {
        RTP_LLM_LOG_ERROR("Speculative target commit prepare failed with an unknown exception");
        return "unknown speculative target commit prepare failure";
    }
}

std::string PyWrappedModel::finishSpeculativeTargetCommit(bool commit) {
    if (!has_speculative_target_commit_hooks_) {
        return {};
    }
    py::gil_scoped_acquire gil;
    try {
        py_model_.attr("finish_speculative_target_commit")(commit);
        return {};
    } catch (const std::exception& e) {
        RTP_LLM_LOG_ERROR("Speculative target commit finish failed (commit=%d): %s", commit, e.what());
        return e.what();
    } catch (...) {
        RTP_LLM_LOG_ERROR("Speculative target commit finish failed with an unknown exception (commit=%d)", commit);
        return "unknown speculative target commit finish failure";
    }
}

std::string PyWrappedModel::finalizeSpeculativeTargetCommit() {
    if (!has_speculative_target_commit_hooks_) {
        return {};
    }
    py::gil_scoped_acquire gil;
    try {
        py_model_.attr("finalize_speculative_target_commit")();
        return {};
    } catch (const std::exception& e) {
        RTP_LLM_LOG_ERROR("Speculative target commit finalize failed: %s", e.what());
        return e.what();
    } catch (...) {
        RTP_LLM_LOG_ERROR("Speculative target commit finalize failed with an unknown exception");
        return "unknown speculative target commit finalize failure";
    }
}

PyWrappedModel::~PyWrappedModel() {
    try {
        py::gil_scoped_acquire gil;
        held_attn_pyobj_   = py::object();
        py_forward_method_ = py::object();
        if (hidden_state_capture_publisher_) {
            hidden_state_capture_publisher_->flushAndClose();
            hidden_state_capture_publisher_.reset();
        }
        generation_prefill_graph_runner_.reset();
        graph_runner_.reset();
        // Runners retain Python methods and graph-owned tensors. Drain and
        // destroy them before releasing the model object they reference.
        if (py_model_ && py::hasattr(py_model_, "close")) {
            if (owns_py_model_) {
                py_model_.attr("close")();
            }
            py_model_ = py::object();
        } else {
            py_model_.release();
        }
        RTP_LLM_LOG_INFO("PyWrappedModel destroyed, Python object instance released.");
    } catch (const py::error_already_set& e) {
        py_model_.release();
        RTP_LLM_LOG_ERROR("Python error during PyWrappedModel destruction: %s", e.what());
    } catch (const std::exception& e) {
        py_model_.release();
        RTP_LLM_LOG_ERROR("C++ error during PyWrappedModel destruction: %s", e.what());
    } catch (...) {
        py_model_.release();
        RTP_LLM_LOG_ERROR("Unknown error during PyWrappedModel destruction");
    }
}

bool PyWrappedModel::shouldCaptureHiddenStates(const GptModelInputs& inputs) {
    const bool capture_requested = inputs.capture_hidden_states && !inputs.warmup && !inputs.is_fake_stream
                                   && hidden_state_capture_publisher_ != nullptr;
    if (capture_requested) {
        // The explicit capture request and capture configuration are identical on
        // all TP ranks after the executor's existing input sync. Keep the
        // Python/FFN packed-hidden shape stable even when TP0 has disabled store
        // publishing.
        (void)hidden_state_capture_publisher_->shouldPublish();
    }
    return capture_requested;
}

HiddenStateCaptureStats PyWrappedModel::hiddenStateCaptureStats() const {
    return hidden_state_capture_publisher_ ? hidden_state_capture_publisher_->stats() : HiddenStateCaptureStats{};
}

std::optional<std::string> PyWrappedModel::takeDeferredHiddenStateCaptureError() {
    return hidden_state_capture_publisher_ ? hidden_state_capture_publisher_->takeDeferredError() : std::nullopt;
}

// Helper function to build PyAttentionInputs from GptModelInputs
torch_ext::PyAttentionInputs PyWrappedModel::buildPyAttentionInputs(const GptModelInputs& inputs) {
    RTP_LLM_PROFILE_SCOPE("py_model.buildPyAttentionInputs");
    torch_ext::PyAttentionInputs py_attn_inputs;
    const int64_t                token_count    = inputs.combo_tokens.defined() ? inputs.combo_tokens.numel() : 0;
    const int64_t                logical_tokens = inputs.sp_logical_tokens > 0 ? inputs.sp_logical_tokens : token_count;
    if (const auto* replay = inputs.targetLinearReplay()) {
        py_attn_inputs.linear_replay = *replay;
    }

    auto normalize_i32 = [this](const torch::Tensor& tensor) -> torch::Tensor {
        if (!tensor.defined()) {
            return tensor;
        }
        if (tensor.is_cuda()) {
            auto result = tensor.scalar_type() == torch::kInt32 ? tensor : tensor.to(torch::kInt32);
            return result.is_contiguous() ? result : result.contiguous();
        }
        auto result =
            tensor.scalar_type() == torch::kInt32 ? tensor.contiguous() : tensor.to(torch::kInt32).contiguous();
        if (!result.is_pinned()) {
            result = result.pin_memory();
        }
        buffer_holder_.hold_host(result);
        return result;
    };
    auto to_device_i32 = [this, &normalize_i32](const torch::Tensor& tensor) -> torch::Tensor {
        auto normalized = normalize_i32(tensor);
        if (!normalized.defined() || normalized.is_cuda()) {
            return normalized;
        }
        return tensorHoldHostAndToCuda(normalized);
    };

    py_attn_inputs.prefix_lengths   = normalize_i32(inputs.prefix_lengths);
    py_attn_inputs.sequence_lengths = normalize_i32(inputs.sequence_lengths);
    py_attn_inputs.input_lengths    = normalize_i32(inputs.input_lengths);
#if !USING_CUDA
    // Non-CUDA platforms only support the host metadata pipeline (the device
    // branch below needs the CUDA-only metadata kernel), so lift any
    // CUDA-resident lengths (e.g. from the MTP device-state fast path) back
    // to host before branching.
    for (auto* t : {&py_attn_inputs.prefix_lengths, &py_attn_inputs.sequence_lengths, &py_attn_inputs.input_lengths}) {
        if (t->defined() && t->is_cuda()) {
            *t = normalize_i32(t->cpu());
        }
    }
#endif
    // MTP draft-prefill hands in a CUDA prefix_lengths (device-state fast path)
    // while the rest of the host pipeline stays CPU-resident. Normalize it to
    // host here so downstream host helpers (padding offset, cu_seqlens) keep
    // their host-tensor contract; prefix_lengths_device below restores the
    // CUDA copy for device consumers.
    if (py_attn_inputs.input_lengths.defined() && !py_attn_inputs.input_lengths.is_cuda()
        && py_attn_inputs.prefix_lengths.defined() && py_attn_inputs.prefix_lengths.is_cuda()) {
        py_attn_inputs.prefix_lengths = normalize_i32(py_attn_inputs.prefix_lengths.cpu());
    }
    py_attn_inputs.prefix_lengths_device = to_device_i32(py_attn_inputs.prefix_lengths);
    py_attn_inputs.input_lengths_device  = to_device_i32(py_attn_inputs.input_lengths);

    if (inputs.combo_position_ids.defined()) {
        py_attn_inputs.combo_position_ids = tensorHoldHostAndToCuda(inputs.combo_position_ids);
    }

    // Calculate cu_seqlens
    int    batch_size                  = py_attn_inputs.input_lengths.size(0);
    size_t context_batch_size          = py_attn_inputs.prefix_lengths.size(0);
    size_t decode_batch_size           = py_attn_inputs.sequence_lengths.size(0);
    py_attn_inputs.dtype               = dataTypeToTorchType(description_.data_type);
    py_attn_inputs.is_prefill          = !decode_batch_size;
    py_attn_inputs.is_target_verify    = inputs.is_target_verify;
    py_attn_inputs.is_mtp_draft_update = inputs.is_mtp_draft_update;
    if (sequence_parallel_padding_enabled_) {
        py_attn_inputs.logical_request_count =
            inputs.sp_logical_requests > 0 ? inputs.sp_logical_requests : inputs.input_lengths.numel();
        py_attn_inputs.physical_request_count = inputs.input_lengths.numel();
        py_attn_inputs.logical_token_count    = logical_tokens;
        py_attn_inputs.physical_token_count   = token_count;
    }
    RTP_LLM_CHECK_WITH_INFO(
        context_batch_size + decode_batch_size == batch_size,
        "batch size check failed context_batch_size[%ld] decode_batch_size[%ld] total_batch_size[%ld]",
        context_batch_size,
        decode_batch_size,
        batch_size);

    // Defensive guard: PyWrappedModel currently does not support a mixed prefill+decode batch.
    // The cu_seqlens slice assignment below assumes input_lengths.cumsum spans only context streams,
    // but input_lengths actually has shape [decode + context]. When both are non-zero the sizes
    // mismatch (slice=[context_batch_size] vs cumsum=[batch_size]) and copy_ throws an opaque
    // PyTorch broadcast error. Failing here gives an actionable message and also catches any
    // future scheduler regression that lets a mixed batch reach the python model path. Schedulers
    // that talk to py_model are expected to drain decode before adding context (see
    // FIFOScheduler::evaluateRunningMemory).
    RTP_LLM_CHECK_WITH_INFO(context_batch_size == 0 || decode_batch_size == 0,
                            "PyWrappedModel received a mixed prefill+decode batch which is not supported: "
                            "context_batch_size[%ld] decode_batch_size[%ld]. The scheduler must keep prefill and "
                            "decode batches separate when load_python_model is enabled.",
                            context_batch_size,
                            decode_batch_size);

    const auto cuda_i32 = torch::TensorOptions(torch::kInt32).device(torch::kCUDA);
    const auto host_i32 = torch::TensorOptions(torch::kInt32).device(torch::kCPU).pinned_memory(true);

    if (context_batch_size > 0 && py_attn_inputs.input_lengths.is_cuda()) {
        py_attn_inputs.total_tokens = static_cast<int>(token_count);
        int64_t prefix_sum          = 0;
        if (py_attn_inputs.prefix_lengths.defined() && py_attn_inputs.prefix_lengths.numel() > 0) {
            prefix_sum = py_attn_inputs.prefix_lengths.sum().item<int64_t>();
        }
        py_attn_inputs.context_total_kv_length = py_attn_inputs.total_tokens + static_cast<int>(prefix_sum);
        py_attn_inputs.cu_seqlens              = torch::empty({0}, host_i32);
        py_attn_inputs.cu_seqlens_device       = torch::empty({batch_size + 1}, cuda_i32);
        py_attn_inputs.cu_kv_seqlens_device    = torch::empty({batch_size + 1}, cuda_i32);
        py_attn_inputs.padding_offset          = torch::empty({py_attn_inputs.total_tokens}, cuda_i32);
#if USING_CUDA
        if (!py_attn_inputs.prefix_lengths.is_cuda()) {
            // A CUDA input length can accompany a host prefix on MTP prefix
            // hits. Its device mirror is a deferred H2D destination: fill it
            // on this stream before the metadata kernel reads it.
            fusedCopy(d2d_copies_);
            d2d_copies_.clear();
        }
        if (sequence_parallel_padding_enabled_) {
            py_attn_inputs.valid_token_mask =
                torch::empty({token_count}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));
        }
        invokeBuildAttentionInputMetadata(py_attn_inputs.input_lengths_device,
                                          py_attn_inputs.prefix_lengths_device,
                                          py_attn_inputs.cu_seqlens_device,
                                          py_attn_inputs.cu_kv_seqlens_device,
                                          py_attn_inputs.padding_offset,
                                          c10::cuda::getCurrentCUDAStream().stream(),
                                          py_attn_inputs.valid_token_mask,
                                          inputs.is_fake_stream ? 0 : logical_tokens);
#else
        RTP_LLM_FAIL("device attention input metadata requires CUDA");
#endif
    } else if (context_batch_size > 0) {
        torch::Tensor cu_seqlens    = torch::zeros({batch_size + 1}, host_i32);
        torch::Tensor cu_kv_seqlens = torch::zeros({batch_size + 1}, host_i32);

        cu_seqlens.slice(0, 1, context_batch_size + 1) = py_attn_inputs.input_lengths.cumsum(0);
        cu_kv_seqlens.slice(0, 1, context_batch_size + 1) =
            py_attn_inputs.input_lengths.add(py_attn_inputs.prefix_lengths).cumsum(0);

        py_attn_inputs.context_total_kv_length = cu_kv_seqlens[context_batch_size].item<int>();
        py_attn_inputs.total_tokens            = cu_seqlens[batch_size].item<int>();
        py_attn_inputs.cu_seqlens              = cu_seqlens;
        py_attn_inputs.cu_seqlens_device       = tensorHoldHostAndToCuda(cu_seqlens);
        py_attn_inputs.cu_kv_seqlens_device    = tensorHoldHostAndToCuda(cu_kv_seqlens);
    } else {
        py_attn_inputs.total_tokens         = 0;
        py_attn_inputs.cu_seqlens_device    = torch::zeros({batch_size + 1}, cuda_i32);
        py_attn_inputs.cu_kv_seqlens_device = torch::zeros({batch_size + 1}, cuda_i32);
        if (py_attn_inputs.sequence_lengths.is_cuda()) {
            py_attn_inputs.cu_seqlens        = torch::empty({0}, host_i32);
            py_attn_inputs.decode_cu_seqlens = torch::empty({0}, host_i32);
            py_attn_inputs.decode_cu_seqlens_device =
                torch::arange(0, py_attn_inputs.sequence_lengths.size(0) + 1, 1, cuda_i32);
        } else {
            py_attn_inputs.cu_seqlens = torch::zeros({batch_size + 1}, host_i32);
            auto decode_cu_seqlens    = torch::arange(0, py_attn_inputs.sequence_lengths.size(0) + 1, 1, host_i32);
            py_attn_inputs.decode_cu_seqlens        = decode_cu_seqlens;
            py_attn_inputs.decode_cu_seqlens_device = tensorHoldHostAndToCuda(decode_cu_seqlens);
        }
    }

    // NOTE: to_device_i32/tensorHoldHostAndToCuda return *deferred-copy* tensors:
    // their storage is only filled by the fusedCopy() flush right before forward.
    // CUDA arithmetic on them here would read uninitialized memory, so "+1" must
    // happen on the host side for host-resident inputs.
    auto plus_1_to_device = [&](const torch::Tensor& t) -> torch::Tensor {
        if (t.defined() && t.is_cuda()) {
            return t.to(torch::kInt32) + 1;
        }
        auto host_plus_1 = normalize_i32(t) + 1;
        return to_device_i32(host_plus_1);
    };
    if (py_attn_inputs.is_target_verify && inputs.sequence_lengths_plus_1.defined()) {
        py_attn_inputs.sequence_lengths_plus_1_device = to_device_i32(inputs.sequence_lengths_plus_1);
    } else if (py_attn_inputs.is_target_verify) {
        py_attn_inputs.sequence_lengths_plus_1_device = plus_1_to_device(inputs.prefix_lengths);
    } else {
        py_attn_inputs.sequence_lengths_plus_1_device = plus_1_to_device(inputs.sequence_lengths);
    }

    if (sequence_parallel_padding_enabled_ && !py_attn_inputs.valid_token_mask.defined()) {
        py_attn_inputs.valid_token_mask =
            torch::arange(token_count, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA))
            < logical_tokens;
        if (inputs.is_fake_stream)
            py_attn_inputs.valid_token_mask.zero_();
    }
    return py_attn_inputs;
}

static void calculatePaddingOffsetDeviceAware(torch_ext::PyAttentionInputs& py_attn_inputs) {
    if (!py_attn_inputs.input_lengths.defined() || !py_attn_inputs.input_lengths.is_cuda()) {
        calculatePaddingOffset(py_attn_inputs);
        return;
    }
    if (py_attn_inputs.padding_offset.defined() && py_attn_inputs.padding_offset.is_cuda()) {
        return;
    }

    const auto cuda_i32           = torch::TensorOptions(torch::kInt32).device(torch::kCUDA);
    py_attn_inputs.padding_offset = torch::empty({py_attn_inputs.total_tokens}, cuda_i32);
    if (py_attn_inputs.total_tokens == 0) {
        return;
    }
#if USING_CUDA
    invokeBuildAttentionInputMetadata(py_attn_inputs.input_lengths_device,
                                      py_attn_inputs.prefix_lengths_device,
                                      py_attn_inputs.cu_seqlens_device,
                                      py_attn_inputs.cu_kv_seqlens_device,
                                      py_attn_inputs.padding_offset,
                                      c10::cuda::getCurrentCUDAStream().stream());
#else
    RTP_LLM_FAIL("device padding_offset requires CUDA");
#endif
}

// Helper function to setup KV cache for attention inputs
torch_ext::AttentionInputsByTag
PyWrappedModel::setupKVCacheForAttentionInputs(torch_ext::PyAttentionInputs& py_attn_inputs,
                                               const GptModelInputs&         inputs) {
    RTP_LLM_PROFILE_SCOPE("py_model.setupKVCacheForAttentionInputs");
    if (!inputs.kv_cache_kernel_block_id.defined()) {
        return {};
    }
    RTP_LLM_CHECK_WITH_INFO(inputs.kv_cache_kernel_block_id.dim() == 2 || inputs.kv_cache_kernel_block_id.dim() == 3,
                            "kv_cache_kernel_block_id must be [batch, blocks] or [group, batch, blocks]");

    const auto publication_block_table = [&](const torch::Tensor& table) {
        auto rows = table;
        if (sequence_parallel_padding_enabled_ && inputs.sp_logical_requests > 0) {
            RTP_LLM_CHECK_WITH_INFO(inputs.sp_logical_requests <= table.size(0),
                                    "SP publication requires all logical block-table rows");
            // Compute consumes the padded table, while prepareWriteCacheParams
            // retains the original request metadata. Never publish dummy rows.
            rows = table.narrow(0, 0, inputs.sp_logical_requests);
        }
        return rows.is_cuda() ? rows.cpu() : rows;
    };

    if (inputs.kv_cache_kernel_block_id.dim() == 2) {
        RTP_LLM_CHECK_WITH_INFO(!kv_cache_layer_layout_.has_value()
                                    || kv_cache_layer_layout_->topology().hasSingleGlobalGroup(),
                                "2-D KV block tables require a single global cache group");
        if (!inputs.kv_cache_group_tags.empty()) {
            RTP_LLM_CHECK_WITH_INFO(inputs.kv_cache_group_tags.size() == 1
                                        && !inputs.kv_cache_group_tags.front().empty()
                                        && kv_cache_layer_layout_.has_value(),
                                    "tagged 2-D KV block tables require one known cache group");
            kv_cache_layer_layout_->topology().group(inputs.kv_cache_group_tags.front());
        }
        py_attn_inputs.kv_cache_kernel_block_id = inputs.kv_cache_kernel_block_id;
        py_attn_inputs.kv_cache_kernel_block_id_device =
            tensorHoldHostAndToCuda(py_attn_inputs.kv_cache_kernel_block_id);
        if (inputs.kv_cache_block_id.defined()) {
            RTP_LLM_CHECK_WITH_INFO(inputs.kv_cache_block_id.dim() == 2,
                                    "kv_cache_block_id must be 2-D when kernel block table is 2-D");
            py_attn_inputs.kv_cache_block_id        = inputs.kv_cache_block_id;
            py_attn_inputs.kv_cache_block_id_device = tensorHoldHostAndToCuda(py_attn_inputs.kv_cache_block_id);
            if (py_attn_inputs.cache_store_inputs.has_value()) {
                // Async writer reads via raw host pointers; MTP device-state
                // paths may carry CUDA block tables here.
                py_attn_inputs.cache_store_inputs->host_kv_cache_offset =
                    publication_block_table(py_attn_inputs.kv_cache_block_id);
            }
        }
        return {};
    }

    const size_t group_count = static_cast<size_t>(inputs.kv_cache_kernel_block_id.size(0));
    RTP_LLM_CHECK_WITH_INFO(kv_cache_layer_layout_.has_value(),
                            "tagged attention inputs require the current model cache layout");
    const auto& group_tags   = inputs.kv_cache_group_tags;
    const auto& model_groups = kv_cache_layer_layout_->topology().groups();
    RTP_LLM_CHECK_WITH_INFO(group_count > 0 && group_tags.size() == group_count && !model_groups.empty(),
                            "KV block table group count=%zu must match payload tags; model groups must not be empty",
                            group_count);
    std::unordered_set<std::string> seen_tags;
    for (const auto& tag : group_tags) {
        RTP_LLM_CHECK_WITH_INFO(
            !tag.empty() && seen_tags.insert(tag).second, "empty or duplicate KV payload group tag=%s", tag.c_str());
    }
    for (const auto& group : model_groups) {
        RTP_LLM_CHECK_WITH_INFO(
            seen_tags.count(group.tag) != 0, "KV payload is missing model cache group tag=%s", group.tag.c_str());
    }
    RTP_LLM_CHECK_WITH_INFO(!inputs.kv_cache_group_types.defined()
                                || (inputs.kv_cache_group_types.dim() == 1
                                    && inputs.kv_cache_group_types.size(0) == static_cast<int64_t>(group_count)),
                            "KV group types must have one entry per payload group (groups=%zu)",
                            group_count);
    RTP_LLM_CHECK_WITH_INFO(!inputs.kv_cache_block_id.defined()
                                || (inputs.kv_cache_block_id.dim() == 3
                                    && inputs.kv_cache_block_id.size(0) == static_cast<int64_t>(group_count)
                                    && inputs.kv_cache_block_id.size(1) == inputs.kv_cache_kernel_block_id.size(1)),
                            "physical KV block table must match kernel table group and batch dimensions");

    /** Match payload rows by tag: a PP stage's local group order can differ from the incoming plan. */
    torch_ext::AttentionInputsByTag by_tag;
    for (const auto& group : model_groups) {
        const auto payload_row  = std::find(group_tags.begin(), group_tags.end(), group.tag) - group_tags.begin();
        auto       group_inputs = py_attn_inputs;
        group_inputs.kv_cache_kernel_block_id        = inputs.kv_cache_kernel_block_id[payload_row];
        group_inputs.kv_cache_kernel_block_id_device = tensorHoldHostAndToCuda(group_inputs.kv_cache_kernel_block_id);
        if (inputs.kv_cache_block_id.defined()) {
            group_inputs.kv_cache_block_id = inputs.kv_cache_block_id[payload_row];
            if (group_inputs.cache_store_inputs.has_value()) {
                group_inputs.cache_store_inputs->host_kv_cache_offset =
                    publication_block_table(group_inputs.kv_cache_block_id);
            }
            if (group.kernelBlocksPerKvBlock() == 1) {
                // Attention needs the cyclic execution IDs; CacheStore keeps the publication table above.
                group_inputs.kv_cache_block_id        = group_inputs.kv_cache_kernel_block_id;
                group_inputs.kv_cache_block_id_device = group_inputs.kv_cache_kernel_block_id_device;
            } else {
                group_inputs.kv_cache_block_id_device = tensorHoldHostAndToCuda(group_inputs.kv_cache_block_id);
            }
        }
        by_tag.emplace(group.tag, std::move(group_inputs));
    }

    // The model topology, including placeholders, determines the input shape;
    // extra rows in a shared target payload do not make a single-group model tagged.
    py_attn_inputs = by_tag.at(model_groups.front().tag);
    if (model_groups.size() == 1) {
        return {};
    }
    return by_tag;
}

// Helper function to build BertEmbeddingInputs from GptModelInputs
torch_ext::BertEmbeddingInputs PyWrappedModel::buildBertEmbeddingInputs(const GptModelInputs& inputs,
                                                                        const torch::Tensor&  prepared_position_ids) {
    RTP_LLM_PROFILE_SCOPE("py_model.buildBertEmbeddingInputs");
    torch_ext::BertEmbeddingInputs bert_embedding_inputs;

    // Convert combo_position_ids from Buffer to torch::Tensor
    if (inputs.combo_position_ids.defined()) {
        // forward() already submits this transfer on the current stream. Reuse
        // that tensor instead of issuing a second blocking .cuda() transfer.
        // Other callers retain the host owner until asynchronous use completes.
        bert_embedding_inputs.combo_position_ids = prepared_position_ids.defined() ?
                                                       prepared_position_ids :
                                                       tensorHoldHostAndToCuda(inputs.combo_position_ids);
    }

    // Convert combo_tokens_type_ids from Buffer to torch::Tensor
    if (inputs.combo_tokens_type_ids.defined()) {
        bert_embedding_inputs.combo_tokens_type_ids = inputs.combo_tokens_type_ids.cuda();
    }

    // Get position_encoding from model weights (no clone needed for weights)
    if (weights_.position_encoding) {
        bert_embedding_inputs.position_encoding = weights_.position_encoding->kernel;
    }

    // Get token_type_embedding from model weights (no clone needed for weights)
    if (weights_.token_type_embedding) {
        bert_embedding_inputs.token_type_embedding = weights_.token_type_embedding->kernel;
    }

    // Set input_embedding_scalar
    bert_embedding_inputs.input_embedding_scalar = description_.input_embedding_scalar;
    return bert_embedding_inputs;
}

// Helper function to call forwardPostLayers with common parameters
GptModelOutputs PyWrappedModel::callForwardPostLayers(torch::Tensor         hidden_states,
                                                      const GptModelInputs& inputs,
                                                      bool                  skip_final_layernorm,
                                                      size_t                num_valid_tokens) {
    RTP_LLM_PROFILE_SCOPE("py_model.callForwardPostLayers");
    size_t num_input_tokens = num_valid_tokens != -1 ? num_valid_tokens : inputs.combo_tokens.size(0);
    return forwardPostLayers(hidden_states,
                             inputs.input_lengths.size(0) != inputs.sequence_lengths.size(0),
                             inputs.need_all_logits,
                             inputs.lm_output_indexes,
                             num_input_tokens,
                             inputs,
                             torch::Tensor(),
                             skip_final_layernorm);
}

torch::Tensor PyWrappedModel::customOutputIndexes(const GptModelInputs& inputs) {
    const auto  decode_batch_size  = inputs.sequence_lengths.size(0);
    const auto  context_batch_size = inputs.input_lengths.size(0) - decode_batch_size;
    const auto& indexes            = inputs.custom_output_indexes;
    TORCH_CHECK(indexes.defined() && indexes.dim() == 1 && indexes.size(0) <= context_batch_size,
                "custom output indexes must contain at most one row per context sequence");
    // Device-input staging is optional. Retain CPU indexes for async H2D;
    // already-staged CUDA indexes require no host retention or additional copy.
    buffer_holder_.hold_host(indexes);
    return indexes.to(torch::kCUDA, /*non_blocking=*/true);
}

void PyWrappedModel::initializeCustomOutput() {
    const auto handler = py_model_.attr("custom_output_handler");
    TORCH_CHECK(py::cast<std::vector<std::string>>(handler.attr("extend_forward_args")())
                    == std::vector<std::string>{"selected_hidden_states"},
                "custom output handler must request selected_hidden_states");
    TORCH_CHECK(weights_.lm_head, "custom output requires a model with lm_head");
    // CP does not retain arbitrary token rows for postprocessing.
    TORCH_CHECK(!device_props_.enable_prefill_cp, "custom output does not support context parallel yet");
    custom_output_enabled_ = true;
    RTP_LLM_LOG_INFO("custom output initialized");
}

torch::Tensor PyWrappedModel::runCustomOutput(const torch::Tensor& rows) {
    py::gil_scoped_acquire gil;
    auto                   output = py_model_.attr("custom_output_handler")
                      .attr("extend_forward")(py::arg("selected_hidden_states") = rows)
                      .cast<torch::Tensor>();
    TORCH_CHECK(output.defined() && (output.dim() == 1 || output.dim() == 2) && output.numel() > 0,
                "custom output must be a nonempty [batch] or [batch, width] tensor");
    TORCH_CHECK(output.size(0) == rows.size(0), "custom output must return one row per selected context sequence");
    TORCH_CHECK(output.device() == rows.device(), "custom output must remain on the input CUDA device");
    const auto dtype = output.scalar_type();
    TORCH_CHECK(dtype == torch::kFloat32 || dtype == torch::kFloat16 || dtype == torch::kBFloat16
                    || dtype == torch::kInt32,
                "custom output dtype must be float32, float16, bfloat16 or int32 for RPC serialization");
    return output;
}

std::optional<PyCacheStoreInputs> PyWrappedModel::prepareWriteCacheParams(const GptModelInputs& inputs) {
    RTP_LLM_PROFILE_SCOPE("py_model.prepareWriteCacheParams");
    if (inputs.warmup || inputs.skip_lm_head) {
        return std::nullopt;
    }
    if (!inputs.pd_separation || !inputs.request_id.defined() || inputs.request_id.numel() == 0) {
        return std::nullopt;
    }

    PyCacheStoreInputs cache_store_inputs;
    // runtimeWriteCacheStore reads these via raw host pointers on the async
    // writer thread; MTP device-state paths hand in CUDA tensors, so lift them
    // to host here (sync copy, prefill-frequency only).
    const auto to_host = [](const torch::Tensor& t) { return (t.defined() && t.is_cuda()) ? t.cpu() : t; };
    cache_store_inputs.input_lengths_host    = to_host(inputs.input_lengths);
    cache_store_inputs.prefix_lengths_host   = to_host(inputs.prefix_lengths);
    cache_store_inputs.publish_start_tokens  = to_host(inputs.cache_store_publish_start_tokens);
    cache_store_inputs.host_kv_cache_offset  = to_host(inputs.kv_cache_block_id);
    cache_store_inputs.request_id            = inputs.request_id;
    cache_store_inputs.request_pd_separation = inputs.request_pd_separation;
    cache_store_inputs.cache_keys            = inputs.cache_keys;
    return cache_store_inputs;
}

std::string PyWrappedModel::waitCacheStorePublication() {
    RTP_LLM_PROFILE_SCOPE("py_model.waitCacheStorePublication");
    if (!track_cache_store_completion_) {
        return {};
    }
    try {
        cache_store_async_writer_->waitStoreCompletions();
        return {};
    } catch (const std::exception& e) {
        RTP_LLM_LOG_ERROR("DSpARK cache-store publication failed: %s", e.what());
        return e.what();
    } catch (...) {
        RTP_LLM_LOG_ERROR("DSpARK cache-store publication failed with an unknown exception");
        return "unknown cache-store publication failure";
    }
}

torch_ext::PyEmbeddingInputs PyWrappedModel::buildPyEmbeddingInputs(const GptModelInputs& inputs) {
    torch_ext::PyEmbeddingInputs embedding_inputs;
    if (inputs.combo_tokens_type_ids.defined()) {
        embedding_inputs.combo_tokens_type_ids = inputs.combo_tokens_type_ids.cuda();
    }
    if (inputs.text_tokens_mask.defined()) {
        embedding_inputs.text_tokens_mask = inputs.text_tokens_mask.cuda();
    }
    return embedding_inputs;
}

torch_ext::PyMultimodalInputs PyWrappedModel::buildPyMultimodalInputs(const GptModelInputs& inputs) {
    torch_ext::PyMultimodalInputs multimodal_input;
    if (inputs.multimodal_features && !inputs.multimodal_features.value().empty()) {
        std::vector<torch::Tensor> multimodal_features;
        for (const auto& feature : inputs.multimodal_features.value()) {
            multimodal_features.emplace_back(feature.cuda());
        }
        multimodal_input.multimodal_features = multimodal_features;
    }
    if (inputs.mm_extra_input && !inputs.mm_extra_input.value().empty()) {
        std::vector<torch::Tensor> mm_extra_input;
        for (const auto& embed : inputs.mm_extra_input.value()) {
            mm_extra_input.emplace_back(embed.cuda());
        }
        multimodal_input.mm_extra_input = mm_extra_input;
    }
    if (inputs.mm_features_locs.defined()) {
        multimodal_input.mm_features_locs = inputs.mm_features_locs.cuda();
    }
    if (inputs.mm_features_spans.defined()) {
        buffer_holder_.hold_host(inputs.mm_features_spans);
        multimodal_input.mm_features_spans = inputs.mm_features_spans.to(torch::kCUDA, /*non_blocking=*/true);
    }
    return multimodal_input;
}

void PyWrappedModel::prepareAttentionInputs(const GptModelInputs& inputs) {
    prepareAttentionInputs(inputs, false);
}

void PyWrappedModel::prepareAttentionInputs(const GptModelInputs& inputs,
                                            bool                  skip_forward_event_sync,
                                            const GptModelInputs* publication_inputs) {
    RTP_LLM_PROFILE_SCOPE("py_model.prepareAttentionInputs");
    PreparedAttentionInputsGuard prepared_guard(prepared_attention_inputs_);
    d2d_copies_.clear();
    if (pinned_check_remaining_ > 0) {
        --pinned_check_remaining_;
    }

    torch_ext::PyAttentionInputs attention_inputs;
    {
        RTP_LLM_PROFILE_SCOPE("py_model.prepareAttentionInputs(build)");
        attention_inputs = buildPyAttentionInputs(inputs);
        // Fake/warmup streams alias reserved block zero and have no live cache rows.
        attention_inputs.is_s_padded = inputs.warmup || inputs.is_fake_stream;
    }
    if (!inputs.warmup && inputs.pd_separation) {
        // Compute may already carry SP padding. Publication must retain the
        // original request rows, without a dummy request or padded prefix.
        attention_inputs.cache_store_inputs =
            prepareWriteCacheParams(publication_inputs ? *publication_inputs : inputs);
        if (attention_inputs.cache_store_inputs.has_value()) {
            attention_inputs.cache_store_writer = cache_store_async_writer_;
        }
    }
    {
        RTP_LLM_PROFILE_SCOPE("py_model.prepareAttentionInputs(padding_offset)");
        calculatePaddingOffsetDeviceAware(attention_inputs);
        attention_inputs.padding_offset = tensorHoldHostAndToCuda(attention_inputs.padding_offset);
    }
    torch_ext::AttentionInputsByTag attention_inputs_by_tag;
    {
        RTP_LLM_PROFILE_SCOPE("py_model.prepareAttentionInputs(setup_kv_cache)");
        attention_inputs_by_tag = setupKVCacheForAttentionInputs(attention_inputs, inputs);
    }

    // CRITICAL ORDERING: flush queued H2D copies BEFORE graph_runner_->prepareAttentionInputs.
    // The graph runner internally launches strided D2D copies that READ from these freshly
    // allocated CUDA tensors (e.g. the per-tag kv_cache_kernel_block_id_device); without
    // flushing first, the D2D copies see uninitialized device memory and pollute the capture
    // buffer, which the QKV+RoPE+KVCache kernel then dereferences as block-id pointers →
    // WARP_ILLEGAL_ADDRESS. (Pre-d318b63ea forward() did fusedCopy before
    // graph_runner->forward; the async-prepare extraction split those steps and broke the
    // implicit ordering.)
    {
        RTP_LLM_PROFILE_SCOPE("py_model.prepareAttentionInputs(fused_h2d)");
        fusedCopy(d2d_copies_);
    }

    graph_state_                         = CudaGraphState();
    generation_prefill_cuda_graph_state_ = CudaGraphState();
    auto empty                           = torch::Tensor();
    // buildPyAttentionInputs() has already copied combo_position_ids to the
    // device.  Keep the top-level PyModelInputs field consistent with the
    // nested attention field: CudaGraphRunner validates and copies the
    // top-level tensor during replay preparation.
    auto       py_model_inputs = PyModelInputs({empty,
                                                empty,
                                                attention_inputs.combo_position_ids,
                                                torch_ext::PyEmbeddingInputs(),
                                                torch_ext::PyMultimodalInputs(),
                                                std::move(attention_inputs),
                                                std::move(attention_inputs_by_tag),
                                                torch_ext::BertEmbeddingInputs()});
    auto*      runner          = selectGraphRunner(py_model_inputs.attention_inputs);
    auto&      state           = selectGraphState(py_model_inputs.attention_inputs);
    const bool can_prepare_graph =
        enable_cuda_graph_ && runner != nullptr && runner->canRun(py_model_inputs, state, CudaGraphCheckMode::PREPARE);
    if (can_prepare_graph) {
        RTP_LLM_PROFILE_SCOPE("py_model.prepareAttentionInputs(cuda_graph_prepare)");
        runner->prepareAttentionInputs(py_model_inputs, state, skip_forward_event_sync);
    }
    // Publish the prepared metadata handles without copying the tagged container.
    attention_inputs_        = std::move(py_model_inputs.attention_inputs);
    attention_inputs_by_tag_ = std::move(py_model_inputs.attention_inputs_by_tag);
    prepared_guard.commit();
}

void PyWrappedModel::updateKVCacheKernelBlockId(const GptModelInputs& inputs) {
    if (!prepared_attention_inputs_.load(std::memory_order_acquire))
        return;
    RTP_LLM_PROFILE_SCOPE("py_model.updateKVCacheKernelBlockId");
    if (!inputs.kv_cache_kernel_block_id.defined() || !prepared_attention_inputs_.load(std::memory_order_acquire)) {
        return;
    }

    d2d_copies_.clear();
    attention_inputs_by_tag_ = setupKVCacheForAttentionInputs(attention_inputs_, inputs);
    fusedCopy(d2d_copies_);

    if (enable_cuda_graph_) {
        auto  empty           = torch::Tensor();
        auto  py_model_inputs = PyModelInputs({empty,
                                               empty,
                                               attention_inputs_.combo_position_ids,
                                               torch_ext::PyEmbeddingInputs(),
                                               torch_ext::PyMultimodalInputs(),
                                               attention_inputs_,
                                               attention_inputs_by_tag_,
                                               torch_ext::BertEmbeddingInputs()});
        auto* runner          = selectGraphRunner(attention_inputs_);
        auto& state           = selectGraphState(attention_inputs_);
        if (runner != nullptr && runner->canRun(py_model_inputs, state, CudaGraphCheckMode::PREPARE)) {
            runner->updateKVCacheKernelBlockId(py_model_inputs, state);
        }
    }
}

void PyWrappedModel::finalizeLinearReplay(const GptModelInputs& inputs) {
    const auto* replay = inputs.targetLinearReplay();
    if (!replay) {
        return;
    }
    const auto batch = replay->slot_ids.numel();
    RTP_LLM_CHECK_WITH_INFO(batch > 0 && inputs.combo_tokens.numel() % batch == 0,
                            "LINEAR replay metadata requires a uniform target verification batch");
    py::gil_scoped_acquire gil;
    py_model_.attr("finalize_linear_replay")(*replay, inputs.combo_tokens.numel() / batch);
}

PyModelInputs PyWrappedModel::buildPyModelInputs(const GptModelInputs& inputs) {
    const auto to_cuda = [this](const torch::Tensor& tensor) {
        if (tensor.is_cuda()) {
            return tensor;
        }
        buffer_holder_.hold_host(tensor);
        return tensor.to(torch::kCUDA, /*non_blocking=*/true);
    };
    auto token_ids     = to_cuda(inputs.combo_tokens);
    auto input_hiddens = inputs.last_hidden_states.defined() ? inputs.last_hidden_states : torch::empty({0});
    auto combo_position_ids =
        inputs.combo_position_ids.defined() ? to_cuda(inputs.combo_position_ids) : torch::empty({0});
    auto embedding_inputs  = buildPyEmbeddingInputs(inputs);
    auto multimodal_inputs = buildPyMultimodalInputs(inputs);
    // Reuse the queued position copy for the BERT metadata as well.
    auto bert_inputs                 = inputs;
    bert_inputs.combo_position_ids   = combo_position_ids;
    auto bert_embedding_inputs       = buildBertEmbeddingInputs(bert_inputs);
    auto py_model_inputs             = PyModelInputs({token_ids,
                                                      input_hiddens,
                                                      combo_position_ids,
                                                      embedding_inputs,
                                                      multimodal_inputs,
                                                      attention_inputs_,
                                                      attention_inputs_by_tag_,
                                                      bert_embedding_inputs});
    py_model_inputs.pp_intermediates = inputs.pp_intermediates;
    return py_model_inputs;
}

GptModelOutputs PyWrappedModel::forward(const GptModelInputs& raw_inputs) {
    mtp_prefill_cp_params_.reset();
    // Reset the per-forward prepared flag even if input preparation throws.
    struct PreparedFlagGuard {
        std::atomic<bool>& flag;
        ~PreparedFlagGuard() {
            flag.store(false, std::memory_order_release);
        }
    } flag_guard{prepared_attention_inputs_};

    RTP_LLM_PROFILE_SCOPE("py_model.forward");

    if (hidden_state_capture_publisher_) {
        hidden_state_capture_publisher_->beginForward();
    }
    const bool capture_hidden_states = shouldCaptureHiddenStates(raw_inputs);
    const bool has_context_request   = raw_inputs.input_lengths.size(0) != raw_inputs.sequence_lengths.size(0);
    // Graph buckets already satisfy SP alignment. Keep the source unpadded
    // until selection; eager-only paths retain their existing input layout.
    const bool defer_sp_padding = enable_cuda_graph_ && !capture_hidden_states
                                  && !(device_props_.enable_prefill_cp && has_context_request)
                                  && (!has_context_request || raw_inputs.is_target_verify
                                      || raw_inputs.is_mtp_draft_update || generation_prefill_graph_runner_);
    auto        padded_inputs = sequence_parallel_padding_enabled_ ?
                                    (defer_sp_padding ? raw_inputs : padSequenceParallelInputs(raw_inputs)) :
                                    GptModelInputs{};
    const auto& inputs        = sequence_parallel_padding_enabled_ ? padded_inputs : raw_inputs;
    const auto& post_inputs   = sequence_parallel_padding_enabled_ ? raw_inputs : inputs;
    if (sequence_parallel_padding_enabled_ && !defer_sp_padding) {
        // Async prepare used raw dimensions; eager consumes the padded layout.
        prepared_attention_inputs_.store(false, std::memory_order_release);
    }
    holdInputsHostBuffers(inputs);
    const auto capture_input_ids = capture_hidden_states ? post_inputs.combo_tokens : torch::Tensor();

    // Packed hidden-state collection is eager, not a generation-prefill graph attempt.
    GenerationPrefillCudaGraphStatus generation_prefill_cuda_graph_status =
        !capture_hidden_states && owns_generation_prefill_cuda_graph_ && !is_prefill_cuda_graph_mode_
                && has_context_request ?
            generation_prefill_cuda_graph_init_status_ :
            GenerationPrefillCudaGraphStatus::NOT_REQUESTED;
    const auto with_generation_prefill_cuda_graph_status = [&](GptModelOutputs outputs) {
        outputs.generation_prefill_cuda_graph_status = generation_prefill_cuda_graph_status;
        return outputs;
    };

    try {
        RTP_LLM_LOG_DEBUG("Calling forward method on Python object instance.");

        PyContextParallelParams cp_params;
        if (device_props_.enable_prefill_cp && has_context_request) {
            /** Build rank-local inputs and CP metadata, including configured MTP hidden-state layout. */
            context_parallel_processor_->handleInputs(const_cast<GptModelInputs&>(inputs), cp_params);
            if (use_spec_decoding_) {
                mtp_prefill_cp_params_ = cp_params;
            }
        }

        if (!prepared_attention_inputs_.load(std::memory_order_acquire)) {
            prepareAttentionInputs(inputs, /*skip_forward_event_sync=*/true, &post_inputs);
        }
        if (device_props_.enable_prefill_cp && has_context_request) {
            attention_inputs_.context_parallel_info = cp_params;
            if (attention_inputs_.cache_store_inputs.has_value()) {
                attention_inputs_.cache_store_inputs->input_lengths_host = cp_params.prefill_actual_input_lengths_cpu;
            }
            for (auto& [tag, tagged_inputs] : attention_inputs_by_tag_) {
                tagged_inputs.context_parallel_info = cp_params;
                if (tagged_inputs.cache_store_inputs.has_value()) {
                    tagged_inputs.cache_store_inputs->input_lengths_host = cp_params.prefill_actual_input_lengths_cpu;
                }
            }
        }
        auto py_model_inputs                  = buildPyModelInputs(inputs);
        py_model_inputs.capture_hidden_states = capture_hidden_states;
        PyModelOutputs py_model_outputs;

        // Cast the Python object to PyModelOutputs and extract hidden states
        auto*      graph_runner = selectGraphRunner(py_model_inputs.attention_inputs);
        auto&      graph_state  = selectGraphState(py_model_inputs.attention_inputs);
        const bool is_generation_prefill_runner =
            graph_runner != nullptr && graph_runner == generation_prefill_graph_runner_.get();
        if (!capture_hidden_states && owns_generation_prefill_cuda_graph_ && has_context_request
            && generation_prefill_graph_runner_ != nullptr && !is_generation_prefill_runner) {
            generation_prefill_cuda_graph_status = GenerationPrefillCudaGraphStatus::MIXED_PREFILL_DECODE_NOT_SUPPORTED;
        }
        const bool can_run_graph = !capture_hidden_states && enable_cuda_graph_ && graph_runner != nullptr
                                   && graph_runner->canRun(py_model_inputs, graph_state);
        if (!capture_hidden_states && is_generation_prefill_runner && !can_run_graph) {
            generation_prefill_cuda_graph_status =
                graph_state.generation_prefill_status == GenerationPrefillCudaGraphStatus::NOT_REQUESTED ?
                    GenerationPrefillCudaGraphStatus::GRAPH_INPUT_SHAPE_MISMATCH :
                    graph_state.generation_prefill_status;
        }
        if (!can_run_graph && sequence_parallel_padding_enabled_ && inputs.sp_logical_tokens == 0) {
            padded_inputs = padSequenceParallelInputs(inputs);
            holdInputsHostBuffers(inputs);
            prepareAttentionInputs(inputs, /*skip_forward_event_sync=*/true, &post_inputs);
            py_model_inputs = buildPyModelInputs(inputs);
        }
        py_model_inputs.capture_hidden_states  = capture_hidden_states;
        py_model_inputs.need_all_logits        = inputs.need_all_logits;
        py_model_inputs.need_all_hidden_states = inputs.need_all_hidden_states;
        if (inputs.engram_token_windows.defined()) {
            buffer_holder_.hold_host(inputs.engram_token_windows);
            py_model_inputs.engram_token_windows = inputs.engram_token_windows.to(torch::kCUDA, /*non_blocking=*/true);
        }
        const bool                has_cache_store_work = !inputs.warmup && !inputs.skip_lm_head && inputs.pd_separation;
        CacheStoreWriteCycleGuard cache_store_write_cycle(
            cache_store_async_writer_, has_cache_store_work, track_cache_store_completion_);
        if (can_run_graph) {
            py::gil_scoped_acquire gil;
            RTP_LLM_PROFILE_SCOPE("py_model.forward(cuda_graph)");
            RTP_LLM_LOG_DEBUG(
                "[PyWrappedModel] using CUDA graph forward, is_target_verify=%d, is_prefill=%d, graph_bs=%d",
                py_model_inputs.attention_inputs.is_target_verify,
                py_model_inputs.attention_inputs.is_prefill,
                graph_state.current_real_graph_bs);
            py_model_inputs.attention_inputs.is_s_padded = true;
            py_model_outputs                             = graph_runner->forward(py_model_inputs, graph_state);
            if (is_generation_prefill_runner) {
                generation_prefill_cuda_graph_status = GenerationPrefillCudaGraphStatus::REPLAYED;
            }
            RTP_LLM_LOG_DEBUG("[PyWrappedModel] CUDA graph forward completed");
        } else {
            py::gil_scoped_acquire gil;
            RTP_LLM_PROFILE_SCOPE("py_model.forward(normal)");
            RTP_LLM_LOG_DEBUG("[PyWrappedModel] using normal forward, is_target_verify=%d, is_prefill=%d",
                              py_model_inputs.attention_inputs.is_target_verify,
                              py_model_inputs.attention_inputs.is_prefill);
            held_attn_pyobj_ = py_model_.attr("prepare_fmha_impl")(py_model_inputs, false);
            auto outputs     = py_forward_method_(py_model_inputs, held_attn_pyobj_);
            py_model_outputs = outputs.cast<PyModelOutputs>();
        }

        if (py_model_outputs.mtp_target_hidden_states.defined()) {
            auto& mtp_hidden = py_model_outputs.mtp_target_hidden_states;
            if (sequence_parallel_padding_enabled_) {
                mtp_hidden = mtp_hidden.narrow(0, 0, raw_inputs.combo_tokens.numel());
            }
            // Graph replay reuses its output storage. Each forward result owns
            // the recurrent features consumed by the speculative executor.
            mtp_hidden = mtp_hidden.clone();
        }
        finalizeLinearReplay(post_inputs);
        cache_store_write_cycle.finish();

        RTP_LLM_LOG_DEBUG("Python object instance forward method called successfully.");
        auto attach_model_outputs = [&py_model_outputs](GptModelOutputs outputs) {
            if (py_model_outputs.mtp_target_hidden_states.defined()) {
                outputs.mtp_target_hidden_states = py_model_outputs.mtp_target_hidden_states;
            }
            outputs.pp_intermediates = py_model_outputs.pp_intermediates;
            return outputs;
        };
        if (!py_model_outputs.pp_intermediates.empty()) {
            /** Non-tail stages may return only boundary tensors; leave their layout intact for the next stage. */
            return with_generation_prefill_cuda_graph_status(attach_model_outputs(GptModelOutputs{}));
        }
        /** Capture consumes the packed Python output directly; ordinary outputs retain their owned clone. */
        auto hidden_states =
            capture_hidden_states ? py_model_outputs.hidden_states : py_model_outputs.hidden_states.clone();
        if (sequence_parallel_padding_enabled_) {
            hidden_states = hidden_states.narrow(0, 0, raw_inputs.combo_tokens.numel());
        }
        if (dspark_model_role_ != DSparkModelRole::NONE) {
            if (dspark_model_role_ == DSparkModelRole::PROPOSE) {
                // Python returns normalized [B*gamma, hidden_dim]. Reuse the
                // regular C++ lm_head and TP logits gather for every proposal
                // row; the speculative executor owns only Markov sampling.
                return with_generation_prefill_cuda_graph_status(
                    attach_model_outputs(callForwardPostLayers(hidden_states, post_inputs, true)));
            }
            // Commit only updates the draft KV cache and has no logits
            // consumer. Preserve its row-aligned hidden output for the common
            // CUDA graph contract without running lm_head.
            GptModelOutputs outputs;
            outputs.hidden_states     = hidden_states;
            outputs.all_hidden_states = hidden_states;
            return with_generation_prefill_cuda_graph_status(attach_model_outputs(std::move(outputs)));
        }
        if (capture_hidden_states) {
            const auto& token_ids              = py_model_inputs.input_ids;
            auto        capture_input_lengths  = inputs.input_lengths;
            int64_t     capture_expected_rows  = capture_input_ids.numel();
            std::string capture_layout_context = "captured hidden-state";

            // Every TP rank must agree before any rank publishes or consumes a
            // fallback. This is intentionally the single capture-layout
            // collective for this forward: CP validates its rank-local output
            // here and restores it only after the status has been reduced;
            // non-CP uses the same mechanism instead of letting a non-owner
            // silently return a zero fallback while TP0 publishes.
            {
                RTP_LLM_PROFILE_SCOPE("py_model.capture_layout_status");
                auto local_layout_error = hidden_state_capture_publisher_->validatePackedLayout(
                    hidden_states, token_ids.numel(), token_ids.device(), "captured hidden-state");
                auto layout_status =
                    torch::tensor({local_layout_error.empty() ? int32_t{1} : int32_t{0}},
                                  torch::TensorOptions().dtype(torch::kInt32).device(token_ids.device()));
                if (device_props_.tp_size > 1) {
                    layout_status = execAllReduce({layout_status, ReduceOp::Min, ParallelMode::TP}).buffer;
                }
                if (!local_layout_error.empty() || layout_status.item<int32_t>() == 0) {
                    if (local_layout_error.empty()) {
                        local_layout_error = "captured hidden-state layout contract violation on another TP rank";
                    }
                    hidden_state_capture_publisher_->rejectLayout(std::move(local_layout_error));
                    auto fallback_last_hidden = hidden_state_capture_publisher_->makeFallback(
                        hidden_states, capture_input_ids.numel(), token_ids.device());
                    return with_generation_prefill_cuda_graph_status(attach_model_outputs(
                        callForwardPostLayers(fallback_last_hidden, post_inputs, true, fallback_last_hidden.size(0))));
                }
            }

            if (device_props_.enable_prefill_cp && has_context_request) {
                RTP_LLM_PROFILE_SCOPE("py_model.capture_cp_restore");
                capture_expected_rows = static_cast<int64_t>(
                    context_parallel_processor_->handleOutputs(hidden_states, post_inputs, cp_params));
                capture_layout_context = "context parallel restored captured hidden-state";
                capture_input_lengths  = cp_params.prefill_actual_input_lengths_cpu;
            }
            auto last_hidden_states = hidden_state_capture_publisher_->publish(hidden_states,
                                                                               capture_input_ids,
                                                                               capture_input_lengths,
                                                                               inputs.request_id,
                                                                               capture_expected_rows,
                                                                               capture_layout_context,
                                                                               token_ids.device());
            return with_generation_prefill_cuda_graph_status(attach_model_outputs(
                callForwardPostLayers(last_hidden_states, post_inputs, true, last_hidden_states.size(0))));
        }
        if (device_props_.enable_prefill_cp && has_context_request) {
            if (!inputs.need_all_logits && !inputs.need_all_hidden_states) {
                context_parallel_processor_->handleOutputsLastHidden(hidden_states, inputs, cp_params);
                return with_generation_prefill_cuda_graph_status(
                    attach_model_outputs(forwardPostLayersLastHidden(hidden_states, post_inputs)));
            }
            size_t num_valid_tokens = context_parallel_processor_->handleOutputs(hidden_states, post_inputs, cp_params);
            return with_generation_prefill_cuda_graph_status(
                attach_model_outputs(callForwardPostLayers(hidden_states, post_inputs, true, num_valid_tokens)));
        }
        return with_generation_prefill_cuda_graph_status(
            attach_model_outputs(callForwardPostLayers(hidden_states, post_inputs, true)));

    } catch (const py::error_already_set& e) {
        RTP_LLM_LOG_ERROR("Python error during forward call on Python instance: %s", e.what());
        throw std::runtime_error(std::string("pybind11 error during forward call on Python instance: ") + e.what());
    } catch (const std::exception& e) {
        RTP_LLM_LOG_ERROR("C++ error during forward call on Python instance: %s", e.what());
        throw std::runtime_error(std::string("C++ error during forward call on Python instance: ") + e.what());
    } catch (...) {
        RTP_LLM_LOG_ERROR("An unknown error occurred during forward call on Python instance.");
        throw std::runtime_error("An unknown error occurred during forward call on Python instance.");
    }
}

// --- Methods absorbed from GptModel ---

torch::Tensor PyWrappedModel::computeLogits(const torch::Tensor& hidden) {
    RTP_LLM_PROFILE_SCOPE("py_model.computeLogits");
    const auto&   head   = *weights_.lm_head;
    const auto&   kernel = head.kernel;
    torch::Tensor logits;
    {
        RTP_LLM_PROFILE_SCOPE("py_model.computeLogits(lm_head_mm)");
        if (kernel.scalar_type() == torch::kFloat8_e4m3fn) {
#if USING_CUDA && defined(ENABLE_FP8)
            TORCH_CHECK(head.scales.defined() && head.scales.numel() == 1
                            && head.scales.scalar_type() == torch::kFloat32 && head.scales.device() == hidden.device()
                            && kernel.device() == hidden.device(),
                        "FP8 LM head requires its FP32 weight scale on the hidden device");
            auto rows = hidden.to(torch::kBFloat16).contiguous();
            if (rows.size(0) == 0) {
                logits = torch::empty({0, kernel.size(0)}, rows.options().dtype(torch::kFloat32));
            } else if (kernel.size(0) % 16 != 0 || kernel.size(1) % 16 != 0) {
                auto weight = (kernel.to(torch::kFloat32) * head.scales).to(torch::kBFloat16);
                logits      = torch::mm(rows, weight.t()).to(torch::kFloat32);
            } else {
                // Quantize only live rows; zero padding must not change their amax domain.
                const auto padded_rows = ((rows.size(0) + 15) / 16) * 16;
                auto       quantized   = torch::empty({padded_rows, rows.size(1)}, kernel.options());
                auto       scale       = torch::zeros({1}, rows.options().dtype(torch::kFloat32));
                per_tensor_quant_fp8(rows, quantized, scale, false);
                logits = at::_scaled_mm(quantized, kernel.t(), scale, head.scales, {}, {}, torch::kBFloat16)
                             .narrow(0, 0, rows.size(0))
                             .to(torch::kFloat32);
            }
#else
            TORCH_CHECK(false, "FP8 LM head requires a CUDA build with FP8 support");
#endif
        } else {
            TORCH_CHECK(kernel.scalar_type() == torch::kBFloat16 || kernel.scalar_type() == torch::kHalf
                            || kernel.scalar_type() == torch::kFloat32 || kernel.scalar_type() == torch::kFloat64,
                        "native LM head requires floating-point weights");
#if USING_CUDA
            if (kernel.scalar_type() == torch::kBFloat16) {
                logits = torch_ext::cublas_gemm_bf16_bf16_fp32(hidden.to(torch::kBFloat16), kernel);
            } else
#endif
            {
                logits = torch::mm(hidden.to(kernel.dtype()), kernel.t()).to(torch::kFloat32);
            }
        }
    }
    return logits;
}

torch::Tensor PyWrappedModel::tpSyncEmbeddingOrLogits(const torch::Tensor& input) {
    RTP_LLM_PROFILE_SCOPE("py_model.tpSyncEmbeddingOrLogits");
    const int64_t tp_size = device_props_.tp_size;
    if (tp_size == 1) {
        return input;
    }
    auto gathered = torch::empty({tp_size, input.size(0), input.size(1)}, input.options());
    auto send     = input.is_contiguous() ? input : input.contiguous();
    execAllGather({{gathered.view({tp_size * input.size(0), input.size(1)})}, ParallelMode::TP, {send}, false});
    auto output = gathered.transpose(0, 1).contiguous().flatten(1);
    cudaCheckLastError();
    return output;
}

GptModelOutputs PyWrappedModel::forwardPostLayers(torch::Tensor         hidden,
                                                  const bool            has_context_request,
                                                  const bool            need_all_logits,
                                                  const torch::Tensor&  lm_output_indexes,
                                                  size_t                token_num,
                                                  const GptModelInputs& inputs,
                                                  torch::Tensor         merged_eagle3_hidden,
                                                  bool                  skip_final_layernorm) {
    if (weights_.final_layernorm && !skip_final_layernorm) {
        RTP_LLM_PROFILE_SCOPE("py_model.forwardPostLayers(final_layernorm)");
        const auto& norm_w = *weights_.final_layernorm;
        const auto  eps    = description_.layernorm_eps;
        if (description_.norm_type == NormType::rmsnorm) {
            auto variance = hidden.to(torch::kFloat32).pow(2).mean(-1, /*keepdim=*/true);
            hidden        = hidden * torch::rsqrt(variance + eps);
            if (norm_w.gamma.defined()) {
                hidden = hidden * norm_w.gamma;
            }
        } else {
            auto normalized_shape = std::vector<int64_t>{hidden.size(-1)};
            auto beta             = norm_w.beta.defined() ? norm_w.beta : torch::Tensor();
            hidden                = torch::layer_norm(hidden, normalized_shape, norm_w.gamma, beta, eps);
        }
    }
    printTorchTensorData(hidden, "final_hidden");

    if (inputs.skip_lm_head) {
        return {torch::Tensor(), torch::Tensor(), hidden};
    }

    const auto& lm_head = weights_.lm_head;

    if (lm_head) {
        RTP_LLM_PROFILE_SCOPE("py_model.forwardPostLayers(lm_head)");
        if (description_.output_vocab_size > 0) {
            const auto gather_world_size = static_cast<size_t>(device_props_.tp_size);
            RTP_LLM_CHECK_WITH_INFO(description_.output_vocab_padded_size >= description_.output_vocab_size,
                                    "invalid output vocabulary layout: K=%zu, P=%zu",
                                    description_.output_vocab_size,
                                    description_.output_vocab_padded_size);
            RTP_LLM_CHECK_WITH_INFO(description_.output_vocab_padded_size % gather_world_size == 0,
                                    "output vocabulary padded size %zu is not divisible by gather world size %zu",
                                    description_.output_vocab_padded_size,
                                    gather_world_size);
            const auto expected_local_rows = description_.output_vocab_padded_size / gather_world_size;
            RTP_LLM_CHECK_WITH_INFO(lm_head->kernel.dim() == 2
                                        && static_cast<size_t>(lm_head->kernel.size(0)) == expected_local_rows,
                                    "output vocabulary LM head rows mismatch: expected %zu, got %ld",
                                    expected_local_rows,
                                    lm_head->kernel.dim() > 0 ? lm_head->kernel.size(0) : -1);
        }
        printTorchTensorData(lm_output_indexes, "lm_output_indexes");

        buffer_holder_.hold_host(lm_output_indexes);
        auto lm_output_indexes_device = lm_output_indexes.to(torch::kCUDA, /*non_blocking=*/true);

        torch::Tensor last_hidden;
        if (has_context_request && !need_all_logits) {
            RTP_LLM_PROFILE_SCOPE("py_model.forwardPostLayers(index_select_last_hidden)");
            last_hidden = torch::index_select(hidden, 0, lm_output_indexes_device.to(torch::kLong));
        } else {
            last_hidden = hidden;
        }

        printTorchTensorData(last_hidden, "last_hidden");

        auto logits = computeLogits(last_hidden);

        GptModelOutputs outputs;
        // Internal warmup/system-prefix requests have no selected token and do not score.
        if (custom_output_enabled_ && has_context_request && inputs.custom_output_indexes.defined()
            && inputs.custom_output_indexes.numel() > 0) {
            try {
                auto context_rows = torch::index_select(hidden, 0, customOutputIndexes(inputs));
                // C++ RMSNorm may promote hidden to FP32; heads use the model
                // activation dtype. Cast only selected rows, leaving LM logits unchanged.
                outputs.custom_output = runCustomOutput(context_rows.to(dataTypeToTorchType(description_.data_type)));
            } catch (const std::exception& error) {
                outputs.custom_output_error = error.what();
                RTP_LLM_LOG_ERROR("custom output processor failed: %s", outputs.custom_output_error.c_str());
            }
        }
        printTorchTensorData(logits, "logits");
        if (device_props_.tp_size > 1) {
            RTP_LLM_PROFILE_SCOPE("py_model.forwardPostLayers(tp_sync_logits)");
            logits = tpSyncEmbeddingOrLogits(logits);
        }
        if (description_.output_vocab_size > 0) {
            RTP_LLM_CHECK_WITH_INFO(logits.dim() == 2
                                        && static_cast<size_t>(logits.size(1)) == description_.output_vocab_padded_size,
                                    "output vocabulary gathered width mismatch: expected %zu, got %ld",
                                    description_.output_vocab_padded_size,
                                    logits.dim() > 1 ? logits.size(1) : -1);
            logits = logits.narrow(1, 0, description_.output_vocab_size).contiguous();
        }
        if (check_nan_) {
            RTP_LLM_CHECK_WITH_INFO(!torch::isnan(last_hidden).any().item<bool>(), "NAN detected in last_hidden");
            RTP_LLM_CHECK_WITH_INFO(!torch::isnan(logits).any().item<bool>(), "NAN detected in logits");
        }
        outputs.logits            = logits;
        outputs.hidden_states     = last_hidden;
        outputs.all_hidden_states = hidden;
        if (need_all_logits) {
            RTP_LLM_PROFILE_SCOPE("py_model.forwardPostLayers(need_all_logits_index)");
            const auto output_indexes = lm_output_indexes_device.to(torch::kLong);
            outputs.all_logits        = logits;
            outputs.logits            = torch::index_select(logits, 0, output_indexes);
            /** Prompt scoring expands LM-head rows, but hidden_states must retain the same selected rows as logits. */
            outputs.hidden_states = torch::index_select(hidden, 0, output_indexes);
        } else if (merged_eagle3_hidden.defined()) {
            outputs.all_hidden_states = merged_eagle3_hidden;
        }
        return outputs;
    } else {
        return {torch::Tensor(), torch::Tensor(), hidden};
    }
}

GptModelOutputs PyWrappedModel::forwardPostLayersLastHidden(torch::Tensor hidden, const GptModelInputs& inputs) {
    // `hidden` is already the lm_output_indexes-selected, post-final-layernorm rows
    // ([num_lm, hidden_size]) gathered by handleOutputsLastHidden. Mirror the CP
    // exit's existing tail: skip the final layernorm (the CP path passes
    // skip_final_layernorm=true) and the lm_output_indexes index_select (already
    // applied during the gather), then run lm_head and TP-sync the logits.
    if (inputs.skip_lm_head) {
        return {torch::Tensor(), torch::Tensor(), hidden};
    }

    const auto& lm_head = weights_.lm_head;
    if (!lm_head) {
        return {torch::Tensor(), torch::Tensor(), hidden};
    }
    printTorchTensorData(hidden, "last_hidden");

    torch::Tensor last_hidden = hidden;
    auto          logits      = computeLogits(last_hidden);
    printTorchTensorData(logits, "logits");
    if (device_props_.tp_size > 1) {
        RTP_LLM_PROFILE_SCOPE("py_model.forwardPostLayersLastHidden(tp_sync_logits)");
        logits = tpSyncEmbeddingOrLogits(logits);
    }
    if (check_nan_) {
        RTP_LLM_CHECK_WITH_INFO(!torch::isnan(last_hidden).any().item<bool>(), "NAN detected in last_hidden");
        RTP_LLM_CHECK_WITH_INFO(!torch::isnan(logits).any().item<bool>(), "NAN detected in logits");
    }
    // 3rd field (all_hidden_states) is the small [num_lm, hidden_size] — the whole
    // point of this path is to never materialize the full [seq, hidden] sequence.
    return {logits, last_hidden, last_hidden, torch::Tensor(), torch::Tensor()};
}

PPIntermediateTensors PyWrappedModel::makePPWarmUpInputTensors(const GptModelInputs& inputs, bool enable_cp) {
    /** Keep request inputs global; only fabricated upstream activations use the local token count. */
    int64_t    local_token_num   = inputs.combo_tokens.numel();
    const auto decode_batch_size = inputs.sequence_lengths.size(0);
    const auto batch_size        = inputs.input_lengths.size(0);
    if (enable_cp && batch_size != decode_batch_size) {
        /** Match CP handleInputs by padding each prefill sequence before splitting. */
        const auto* input_lengths = inputs.input_lengths.data_ptr<int32_t>();
        local_token_num           = decode_batch_size;
        for (int64_t i = decode_batch_size; i < batch_size; ++i) {
            local_token_num += makeZigzagTokenLayout(input_lengths[i], device_props_.tp_size).token_count_per_rank;
        }
    }

    auto hidden_template =
        torch::zeros({local_token_num, hidden_size_},
                     torch::TensorOptions().dtype(dataTypeToTorchType(description_.data_type)).device(torch::kCUDA));

    py::gil_scoped_acquire gil;
    try {
        auto tensors = py_model_.attr("make_empty_intermediate_tensors")(hidden_template)
                           .cast<std::map<std::string, torch::Tensor>>();
        RTP_LLM_CHECK_WITH_INFO(!tensors.empty(), "Python model returned no PP warmup intermediate tensors");
        for (const auto& [name, tensor] : tensors) {
            RTP_LLM_CHECK_WITH_INFO(tensor.defined(), "PP warmup intermediate tensor [%s] is undefined", name.c_str());
            RTP_LLM_CHECK_WITH_INFO(
                tensor.is_cuda(), "PP warmup intermediate tensor [%s] must be on CUDA", name.c_str());
        }
        return PPIntermediateTensors{std::move(tensors)};
    } catch (const py::error_already_set& e) {
        RTP_LLM_LOG_ERROR("Python model failed to construct PP warmup intermediate tensors:\n%s", e.what());
        throw;
    }
}

void PyWrappedModel::holdInputsHostBuffers(const GptModelInputs& inputs) {
    buffer_holder_.hold_host(inputs.combo_tokens);
    buffer_holder_.hold_host(inputs.input_lengths);
    buffer_holder_.hold_host(inputs.sequence_lengths);
    buffer_holder_.hold_host(inputs.lm_output_indexes);
    buffer_holder_.hold_host(inputs.prefix_lengths);
    buffer_holder_.hold_host(inputs.cache_store_publish_start_tokens);

    buffer_holder_.hold_host(inputs.combo_position_ids);
    buffer_holder_.hold_host(inputs.combo_tokens_type_ids);

    buffer_holder_.hold_host(inputs.last_hidden_states);

    buffer_holder_.hold_host(inputs.attention_mask);
    buffer_holder_.hold_host(inputs.kv_cache_block_id);
    buffer_holder_.hold_host(inputs.kv_cache_group_types);
    buffer_holder_.hold_host(inputs.kv_cache_update_mapping);
    if (inputs.multimodal_features.has_value()) {
        for (auto& mm_feature : inputs.multimodal_features.value()) {
            buffer_holder_.hold_host(mm_feature);
        }
    }

    buffer_holder_.hold_host(inputs.text_tokens_mask);
    buffer_holder_.hold_host(inputs.mm_features_locs);
    buffer_holder_.hold_host(inputs.engram_token_windows);
    buffer_holder_.hold_host(inputs.mm_features_spans);

    if (inputs.input_embeddings.has_value()) {
        for (auto& input_embedding : inputs.input_embeddings.value()) {
            buffer_holder_.hold_host(input_embedding);
        }
    }
    buffer_holder_.hold_host(inputs.input_embeddings_locs);

    buffer_holder_.hold_host(inputs.request_id);
    buffer_holder_.hold_host(inputs.request_pd_separation);
    buffer_holder_.hold_host(inputs.cache_keys);
}

}  // namespace rtp_llm
