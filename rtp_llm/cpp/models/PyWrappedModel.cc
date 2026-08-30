#include "rtp_llm/cpp/models/PyWrappedModel.h"
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/cpp/utils/DebugUtils.h"
#include "rtp_llm/cpp/utils/utils.h"
#include "rtp_llm/cpp/model_utils/AttentionConfig.h"
#include <cstdint>
#include <stdexcept>
#include <mutex>
#include <unordered_set>
#include <vector>
#include "rtp_llm/cpp/pybind/PyUtils.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include "rtp_llm/cpp/utils/DevicePerfWrapper.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#if USING_CUDA
#include <c10/cuda/CUDAStream.h>
#include "rtp_llm/models_py/bindings/cuda/Bf16GemmOp.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/attention_input_metadata.h"
#endif

using namespace std;

namespace rtp_llm {

namespace {

// Pairs init() with waitAllDone() so an exception between them still returns the
// writer to IDLE. Without this a single failed forward leaves it RUNNING, and every
// later init() trips its IDLE precondition -- the instance can never publish again.
class CacheStoreWriteCycleGuard {
public:
    CacheStoreWriteCycleGuard(const std::shared_ptr<CacheStoreAsyncWriter>& writer, bool has_work):
        writer_(writer), active_(has_work) {
        if (active_) {
            writer_->init();
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

    // Normal path: let a drain failure propagate to the caller.
    void finish() {
        if (!active_) {
            return;
        }
        active_ = false;
        writer_->waitAllDone();
    }

    CacheStoreWriteCycleGuard(const CacheStoreWriteCycleGuard&)            = delete;
    CacheStoreWriteCycleGuard& operator=(const CacheStoreWriteCycleGuard&) = delete;

private:
    std::shared_ptr<CacheStoreAsyncWriter> writer_;
    bool                                   active_{false};
};

}  // namespace

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

PyWrappedModel::~PyWrappedModel() {
    try {
        py::gil_scoped_acquire gil;
        held_attn_pyobj_   = py::object();
        py_forward_method_ = py::object();
        // Always release py_model_ since it's always initialized now
        py_model_.release();
        if (graph_runner_ != nullptr) {
            delete graph_runner_;
            graph_runner_ = nullptr;
        }
        RTP_LLM_LOG_INFO("PyWrappedModel destroyed, Python object instance released.");
    } catch (const py::error_already_set& e) {
        RTP_LLM_LOG_ERROR("Python error during PyWrappedModel destruction: %s", e.what());
    } catch (const std::exception& e) {
        RTP_LLM_LOG_ERROR("C++ error during PyWrappedModel destruction: %s", e.what());
    }
}

// Helper function to build PyAttentionInputs from GptModelInputs
torch_ext::PyAttentionInputs PyWrappedModel::buildPyAttentionInputs(const GptModelInputs& inputs) {
    RTP_LLM_PROFILE_SCOPE("py_model.buildPyAttentionInputs");
    DevicePerfWrapper            wrapper(enable_device_perf_, "py model buildPyAttentionInputs");
    torch_ext::PyAttentionInputs py_attn_inputs;

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
    int    batch_size               = py_attn_inputs.input_lengths.size(0);
    size_t context_batch_size       = py_attn_inputs.prefix_lengths.size(0);
    size_t decode_batch_size        = py_attn_inputs.sequence_lengths.size(0);
    py_attn_inputs.dtype            = dataTypeToTorchType(description_.data_type);
    py_attn_inputs.is_prefill       = !decode_batch_size;
    py_attn_inputs.is_target_verify = inputs.is_target_verify;
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
        py_attn_inputs.total_tokens = inputs.combo_tokens.defined() ? static_cast<int>(inputs.combo_tokens.numel()) : 0;
        // Must match cu_kv_seqlens_device's definition (input_lengths +
        // prefix_lengths): the CUDA graph padding fill copies this scalar into
        // the cu_kv_seqlens tail, and a prefix-less value makes the array
        // non-monotonic whenever prefix reuse / target verify is active.
        // prefix_lengths here is a source tensor (never a deferred H2D copy),
        // so summing it is safe; item() adds one stream sync on this path.
        int64_t prefix_sum = 0;
        if (py_attn_inputs.prefix_lengths.defined() && py_attn_inputs.prefix_lengths.numel() > 0) {
            prefix_sum = py_attn_inputs.prefix_lengths.sum().item<int64_t>();
        }
        py_attn_inputs.context_total_kv_length = py_attn_inputs.total_tokens + static_cast<int>(prefix_sum);
        py_attn_inputs.cu_seqlens              = torch::empty({0}, host_i32);
        py_attn_inputs.cu_seqlens_device       = torch::empty({batch_size + 1}, cuda_i32);
        py_attn_inputs.cu_kv_seqlens_device    = torch::empty({batch_size + 1}, cuda_i32);
        py_attn_inputs.padding_offset          = torch::empty({py_attn_inputs.total_tokens}, cuda_i32);
#if USING_CUDA
        invokeBuildAttentionInputMetadata(py_attn_inputs.input_lengths_device,
                                          py_attn_inputs.prefix_lengths_device,
                                          py_attn_inputs.cu_seqlens_device,
                                          py_attn_inputs.cu_kv_seqlens_device,
                                          py_attn_inputs.padding_offset,
                                          c10::cuda::getCurrentCUDAStream().stream());
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
                                               const GptModelInputs&         inputs,
                                               const std::vector<size_t>&    input_idx_by_tag) {
    RTP_LLM_PROFILE_SCOPE("py_model.setupKVCacheForAttentionInputs");
    DevicePerfWrapper wrapper(enable_device_perf_, "py model setupKVCacheForAttentionInputs");
    if (!inputs.kv_cache_kernel_block_id.defined()) {
        return {};
    }
    RTP_LLM_CHECK_WITH_INFO(inputs.kv_cache_kernel_block_id.dim() == 2 || inputs.kv_cache_kernel_block_id.dim() == 3,
                            "kv_cache_kernel_block_id must be [batch, blocks] or [group, batch, blocks]");

    if (inputs.kv_cache_kernel_block_id.dim() == 2) {
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
                py_attn_inputs.cache_store_inputs->host_kv_cache_offset = py_attn_inputs.kv_cache_block_id.is_cuda() ?
                                                                              py_attn_inputs.kv_cache_block_id.cpu() :
                                                                              py_attn_inputs.kv_cache_block_id;
            }
        }
        return {};
    }

    const size_t group_count = static_cast<size_t>(inputs.kv_cache_kernel_block_id.size(0));
    RTP_LLM_CHECK_WITH_INFO(kv_cache_layer_layout_.has_value(),
                            "tagged attention inputs require the current model cache layout");
    // Boundary adapter (C++ -> Python): dim 0 of the block tables is an
    // adapter-local group_ordinal in canonical sorted-tag order. The producing
    // gatherer derives the same order from its own CacheConfig, so no ordering
    // travels with the tensors and reordering the topology records cannot move a
    // group. The ordinal never leaves this function.
    const auto& group_tags = kv_cache_boundary_group_tags_;
    RTP_LLM_CHECK_WITH_INFO(input_idx_by_tag.size() == group_count,
                            "validated cache tag mapping length=%zu does not match group count=%zu",
                            input_idx_by_tag.size(),
                            group_count);
    RTP_LLM_CHECK_WITH_INFO(group_tags.size() == group_count,
                            "KV block table group count=%zu does not match cache tag count=%zu",
                            group_count,
                            group_tags.size());
    RTP_LLM_CHECK_WITH_INFO(!inputs.kv_cache_block_id.defined() || inputs.kv_cache_block_id.dim() == 3,
                            "physical kv_cache_block_id must be 3-D for tagged inputs");
    RTP_LLM_CHECK_WITH_INFO(!inputs.kv_cache_block_id.defined()
                                || static_cast<size_t>(inputs.kv_cache_block_id.size(0)) == group_count,
                            "physical kv_cache_block_id group count=%ld does not match kernel block table count=%zu",
                            inputs.kv_cache_block_id.defined() ? inputs.kv_cache_block_id.size(0) : -1,
                            group_count);
    torch_ext::AttentionInputsByTag by_tag;
    for (size_t group_ordinal = 0; group_ordinal < group_count; ++group_ordinal) {
        const auto& tag                              = group_tags[group_ordinal];
        const auto  input_idx                        = input_idx_by_tag[group_ordinal];
        auto        group_inputs                     = py_attn_inputs;
        group_inputs.kv_cache_kernel_block_id        = inputs.kv_cache_kernel_block_id[input_idx];
        group_inputs.kv_cache_kernel_block_id_device = tensorHoldHostAndToCuda(group_inputs.kv_cache_kernel_block_id);
        if (inputs.kv_cache_block_id.defined()) {
            group_inputs.kv_cache_block_id        = inputs.kv_cache_block_id[input_idx];
            group_inputs.kv_cache_block_id_device = tensorHoldHostAndToCuda(group_inputs.kv_cache_block_id);
            if (group_inputs.cache_store_inputs.has_value()) {
                group_inputs.cache_store_inputs->host_kv_cache_offset = group_inputs.kv_cache_block_id.is_cuda() ?
                                                                            group_inputs.kv_cache_block_id.cpu() :
                                                                            group_inputs.kv_cache_block_id;
            }
        }
        const auto [it, inserted] = by_tag.emplace(tag, std::move(group_inputs));
        (void)it;
        RTP_LLM_CHECK_WITH_INFO(inserted, "duplicate attention input tag=%s", tag.c_str());
    }

    // A single global group keeps the direct fast path. Multiple groups are
    // exposed only through the outer tag mapping, with the lowest tag mirrored
    // into the direct field so the mirror does not depend on record order.
    py_attn_inputs = by_tag.at(group_tags.front());
    if (group_count == 1) {
        return {};
    }
    return by_tag;
}

std::vector<size_t> PyWrappedModel::validateTaggedCacheBoundary(const GptModelInputs& inputs) const {
    if (!inputs.kv_cache_kernel_block_id.defined() || inputs.kv_cache_kernel_block_id.dim() != 3) {
        return {};
    }
    const auto group_count = static_cast<size_t>(inputs.kv_cache_kernel_block_id.size(0));
    RTP_LLM_CHECK_WITH_INFO(!kv_cache_boundary_group_tags_.empty(),
                            "tagged KV block tables require non-empty model cache tags");
    // tpSyncModelInputs broadcasts tensor payloads and group types, but not
    // std::string tags. A non-root rank therefore reconstructs the documented
    // canonical row order from its identical local CacheConfig. Root and any
    // explicitly tagged input retain exact-set validation and row permutation.
    const bool reconstruct_non_root_tags = inputs.kv_cache_group_tags.empty() && device_props_.tp_rank > 0;
    RTP_LLM_CHECK_WITH_INFO(reconstruct_non_root_tags
                                || inputs.kv_cache_group_tags.size() == kv_cache_boundary_group_tags_.size(),
                            "model input cache tags must exactly match this model's cache tag set");
    std::unordered_set<std::string> input_tags;
    input_tags.reserve(inputs.kv_cache_group_tags.size());
    for (const auto& tag : inputs.kv_cache_group_tags) {
        RTP_LLM_CHECK_WITH_INFO(!tag.empty() && input_tags.insert(tag).second,
                                "model input cache tags must be non-empty and unique");
    }
    if (!reconstruct_non_root_tags) {
        for (const auto& tag : kv_cache_boundary_group_tags_) {
            RTP_LLM_CHECK_WITH_INFO(input_tags.find(tag) != input_tags.end(),
                                    "model input cache tags contain unknown or missing tags");
        }
    }
    RTP_LLM_CHECK_WITH_INFO(group_count == kv_cache_boundary_group_tags_.size(),
                            "kernel KV block-table group count=%zu does not match cache tag count=%zu",
                            group_count,
                            kv_cache_boundary_group_tags_.size());
    RTP_LLM_CHECK_WITH_INFO(!inputs.kv_cache_block_id.defined()
                                || (inputs.kv_cache_block_id.dim() == 3
                                    && static_cast<size_t>(inputs.kv_cache_block_id.size(0)) == group_count),
                            "physical KV block table must have the same group dimension as the kernel block table");
    RTP_LLM_CHECK_WITH_INFO(inputs.kv_cache_group_types.defined() && inputs.kv_cache_group_types.device().is_cpu()
                                && inputs.kv_cache_group_types.scalar_type() == torch::kInt32
                                && inputs.kv_cache_group_types.dim() == 1 && inputs.kv_cache_group_types.is_contiguous()
                                && static_cast<size_t>(inputs.kv_cache_group_types.numel()) == group_count,
                            "cache group-type payload length must match cache tags");
    RTP_LLM_CHECK_WITH_INFO(kv_cache_layer_layout_.has_value(),
                            "tagged KV block tables require the current model cache layout");

    std::vector<size_t> input_idx_by_tag;
    input_idx_by_tag.reserve(kv_cache_boundary_group_tags_.size());
    const auto* input_types = inputs.kv_cache_group_types.data_ptr<int32_t>();
    for (const auto& tag : kv_cache_boundary_group_tags_) {
        size_t input_idx = input_idx_by_tag.size();
        if (!reconstruct_non_root_tags) {
            const auto input_it = std::find(inputs.kv_cache_group_tags.begin(), inputs.kv_cache_group_tags.end(), tag);
            RTP_LLM_CHECK_WITH_INFO(
                input_it != inputs.kv_cache_group_tags.end(), "validated cache tag=%s has no input row", tag.c_str());
            input_idx = static_cast<size_t>(std::distance(inputs.kv_cache_group_tags.begin(), input_it));
        }
        RTP_LLM_CHECK_WITH_INFO(cache_manager_ != nullptr, "KV cache layout requires a cache manager");
        const auto& cache_config  = mtp_cache_config_index_.has_value() ?
                                        cache_manager_->getMTPModuleCacheConfig(*mtp_cache_config_index_) :
                                        cache_manager_->cacheConfig();
        const auto  expected_type = cache_config.group(tag).policy.group_type;
        RTP_LLM_CHECK_WITH_INFO(input_types[input_idx] == static_cast<int32_t>(expected_type),
                                "cache group type mismatch for tag=%s: input=%d expected=%d",
                                tag.c_str(),
                                input_types[input_idx],
                                static_cast<int32_t>(expected_type));
        input_idx_by_tag.push_back(input_idx);
    }
    return input_idx_by_tag;
}

// Helper function to build BertEmbeddingInputs from GptModelInputs
torch_ext::BertEmbeddingInputs PyWrappedModel::buildBertEmbeddingInputs(const GptModelInputs& inputs) {
    RTP_LLM_PROFILE_SCOPE("py_model.buildBertEmbeddingInputs");
    DevicePerfWrapper              wrapper(enable_device_perf_, "py model buildBertEmbeddingInputs");
    torch_ext::BertEmbeddingInputs bert_embedding_inputs;

    // Convert combo_position_ids from Buffer to torch::Tensor
    if (inputs.combo_position_ids.defined()) {
        bert_embedding_inputs.combo_position_ids = inputs.combo_position_ids.cuda();
    }

    // Convert combo_tokens_type_ids from Buffer to torch::Tensor
    if (inputs.combo_tokens_type_ids.defined()) {
        {
            DevicePerfWrapper wrapper(enable_device_perf_, "py model combo_tokens.cuda()");
            bert_embedding_inputs.combo_tokens_type_ids = inputs.combo_tokens_type_ids.cuda();
        }
    }

    // Get position_encoding from model weights (no clone needed for weights)
    if (weights_.position_encoding) {
        DevicePerfWrapper wrapper(enable_device_perf_, "py model weights_.position_encoding->kernel");
        bert_embedding_inputs.position_encoding = weights_.position_encoding->kernel;
    }

    // Get token_type_embedding from model weights (no clone needed for weights)
    if (weights_.token_type_embedding) {
        DevicePerfWrapper wrapper(enable_device_perf_, "py model weights_.token_type_embedding->kernel");
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
                             false,
                             num_input_tokens,
                             inputs,
                             torch::Tensor(),
                             skip_final_layernorm);
}

std::optional<PyCacheStoreInputs> PyWrappedModel::prepareWriteCacheParams(const GptModelInputs& inputs) {
    RTP_LLM_PROFILE_SCOPE("py_model.prepareWriteCacheParams");
    if (inputs.warmup) {
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
    cache_store_inputs.host_kv_cache_offset  = to_host(inputs.kv_cache_block_id);
    cache_store_inputs.request_id            = inputs.request_id;
    cache_store_inputs.request_pd_separation = inputs.request_pd_separation;
    cache_store_inputs.cache_keys            = inputs.cache_keys;
    return cache_store_inputs;
}

GptModelOutputs PyWrappedModel::forwardMicroBatched(const GptModelInputs& inputs) {
    return forwardMicroBatchedValidated(inputs, validateTaggedCacheBoundary(inputs));
}

GptModelOutputs PyWrappedModel::forwardMicroBatchedValidated(const GptModelInputs&      inputs,
                                                             const std::vector<size_t>& input_idx_by_tag) {
    RTP_LLM_PROFILE_SCOPE("py_model.forwardMicroBatched");

    // Per-launch capacity contract: see fuse_copy_util.h sizing rationale.
    // d2d_copies_ accumulates across ALL micro-batches before the single
    // fusedCopy() flush below. Per micro-batch this adds ~6 copies from
    // buildPyAttentionInputs + padding_offset, plus group_count from
    // setupKVCacheForAttentionInputs. With the planMicroBatches cap of 2
    // micro-batches and hybrid group_count of 4 the worst case is ~20.
    // If new tensorHoldHostAndToCuda call sites land below — or if
    // planMicroBatches starts producing >2 micro-batches — re-check
    // MAX_FUSED_D2D_COPIES.
    d2d_copies_.clear();
    if (pinned_check_remaining_ > 0) {
        --pinned_check_remaining_;
    }

    {
        py::gil_scoped_acquire gil;
        if (device_props_.ffn_as_service) {
            py::object py_forward_method = py_model_.attr("forward_micro_batch");
            py::object py_outputs_obj    = py_forward_method(std::vector<PyModelInputs>{});
            return GptModelOutputs();
        }
    }

    auto micro_batch_plan  = planMicroBatches(inputs);
    auto [split_inputs, _] = splitInputsIntoMicroBatches(inputs, micro_batch_plan);
    std::vector<PyModelInputs> input_list;
    input_list.reserve(split_inputs.size());

    for (size_t i = 0; i < split_inputs.size(); ++i) {
        const bool  is_real_micro_input   = split_inputs[i].kv_cache_kernel_block_id.defined();
        const auto& micro_inputs          = is_real_micro_input ? split_inputs[i] : split_inputs[0];
        auto        py_attn_inputs        = buildPyAttentionInputs(micro_inputs);
        auto        embedding_inputs      = buildPyEmbeddingInputs(micro_inputs);
        auto        multimodal_inputs     = buildPyMultimodalInputs(micro_inputs);
        auto        bert_embedding_inputs = buildBertEmbeddingInputs(micro_inputs);
        if (is_real_micro_input && py_attn_inputs.is_prefill) {
            py_attn_inputs.cache_store_inputs = prepareWriteCacheParams(micro_inputs);
            if (py_attn_inputs.cache_store_inputs.has_value()) {
                py_attn_inputs.cache_store_writer = cache_store_async_writer_;
            }
        }
        torch::Tensor combo_position_ids = micro_inputs.combo_position_ids.defined() ?
                                               tensorHoldHostAndToCuda(micro_inputs.combo_position_ids) :
                                               torch::empty({0});
        calculatePaddingOffsetDeviceAware(py_attn_inputs);
        py_attn_inputs.padding_offset = tensorHoldHostAndToCuda(py_attn_inputs.padding_offset);
        auto attention_inputs_by_tag  = setupKVCacheForAttentionInputs(py_attn_inputs, micro_inputs, input_idx_by_tag);

        torch::Tensor token_ids = micro_inputs.combo_tokens.clone().cuda();
        torch::Tensor input_hiddens =
            inputs.last_hidden_states.defined() ? inputs.last_hidden_states : torch::empty({0});
        input_list.emplace_back(PyModelInputs{token_ids,
                                              input_hiddens,
                                              combo_position_ids,
                                              embedding_inputs,
                                              multimodal_inputs,
                                              py_attn_inputs,
                                              attention_inputs_by_tag,
                                              bert_embedding_inputs});
    }

    const bool                has_cache_store_work = !inputs.warmup && inputs.pd_separation;
    CacheStoreWriteCycleGuard cache_store_write_cycle(cache_store_async_writer_, has_cache_store_work);

    fusedCopy(d2d_copies_);

    std::vector<PyModelOutputs> py_model_outputs;
    {
        py::gil_scoped_acquire gil;
        py::object             py_forward_method = py_model_.attr("forward_micro_batch");
        py::object             py_outputs_obj    = py_forward_method(input_list);
        py_model_outputs                         = py_outputs_obj.cast<std::vector<PyModelOutputs>>();
    }

    RTP_LLM_CHECK_WITH_INFO(py_model_outputs.size() == input_list.size(),
                            "py_model_outputs.size:%d != micro_batch_inputs.size:%d",
                            py_model_outputs.size(),
                            input_list.size());

    cache_store_write_cycle.finish();

    // TODO: merge hidden states in one tensor
    torch::Tensor hidden_states;
    if (!micro_batch_plan.enable) {
        RTP_LLM_CHECK_WITH_INFO(py_model_outputs[0].hidden_states.size(0) == inputs.combo_tokens.size(0),
                                "py_model_outputs[0].hidden_states.size(0):%d != inputs.combo_tokens.size(0):%d",
                                py_model_outputs[0].hidden_states.size(0),
                                inputs.combo_tokens.size(0));
        hidden_states = py_model_outputs[0].hidden_states;
    } else {
        size_t total_tokens = inputs.combo_tokens.size(0);
        size_t hidden_size  = description_.attention_conf.head_num * description_.attention_conf.size_per_head;
        hidden_states =
            torch::empty({(int64_t)total_tokens, (int64_t)hidden_size},
                         torch::TensorOptions(dataTypeToTorchType(description_.data_type)).device(torch::kCUDA));
        int offset = 0;
        for (int i = 0; i < py_model_outputs.size(); i++) {
            RTP_LLM_CHECK_WITH_INFO(
                offset + py_model_outputs[i].hidden_states.size(0) <= (int)total_tokens,
                "offset + py_model_outputs[i].hidden_states.size(0):%d > inputs.combo_tokens->shape()[0]:%d",
                offset + py_model_outputs[i].hidden_states.size(0),
                total_tokens);
            auto slice_size = py_model_outputs[i].hidden_states.size(0);
            hidden_states.slice(0, offset, offset + slice_size).copy_(py_model_outputs[i].hidden_states);
            offset += slice_size;
        }
        RTP_LLM_CHECK_WITH_INFO(offset == (int)total_tokens,
                                "total out hidden size:%d != inputs.combo_tokens->shape()[0]:%d",
                                offset,
                                total_tokens);
    }

    RTP_LLM_LOG_DEBUG("Python object instance forward method called successfully.");

    return callForwardPostLayers(hidden_states, inputs, false);
}

torch_ext::PyEmbeddingInputs PyWrappedModel::buildPyEmbeddingInputs(const GptModelInputs& inputs) {
    DevicePerfWrapper            wrapper(enable_device_perf_, "py model buildPyEmbeddingInputs");
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
    DevicePerfWrapper             wrapper(enable_device_perf_, "py model buildPyMultimodalInputs");
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
    return multimodal_input;
}

void PyWrappedModel::prepareAttentionInputs(const GptModelInputs& inputs) {
    prepareAttentionInputs(inputs, false);
}

void PyWrappedModel::prepareAttentionInputs(const GptModelInputs& inputs, bool skip_forward_event_sync) {
    prepareAttentionInputsValidated(inputs, skip_forward_event_sync, validateTaggedCacheBoundary(inputs));
}

void PyWrappedModel::prepareAttentionInputsValidated(const GptModelInputs&      inputs,
                                                     bool                       skip_forward_event_sync,
                                                     const std::vector<size_t>& input_idx_by_tag) {
    RTP_LLM_PROFILE_SCOPE("py_model.prepareAttentionInputs");
    d2d_copies_.clear();
    if (pinned_check_remaining_ > 0) {
        --pinned_check_remaining_;
    }

    DevicePerfWrapper            wrapper(enable_device_perf_, "py model prepareAttentionInputs");
    torch_ext::PyAttentionInputs attention_inputs;
    {
        RTP_LLM_PROFILE_SCOPE("py_model.prepareAttentionInputs(build)");
        attention_inputs = buildPyAttentionInputs(inputs);
    }
    if (!inputs.warmup && inputs.pd_separation) {
        attention_inputs.cache_store_inputs = prepareWriteCacheParams(inputs);
        if (attention_inputs.cache_store_inputs.has_value()) {
            attention_inputs.cache_store_writer = cache_store_async_writer_;
        }
    }
    {
        RTP_LLM_PROFILE_SCOPE("py_model.prepareAttentionInputs(padding_offset)");
        calculatePaddingOffsetDeviceAware(attention_inputs);
        attention_inputs.padding_offset = tensorHoldHostAndToCuda(attention_inputs.padding_offset);
    }
    {
        RTP_LLM_PROFILE_SCOPE("py_model.prepareAttentionInputs(setup_kv_cache)");
        attention_inputs_by_tag_ = setupKVCacheForAttentionInputs(attention_inputs, inputs, input_idx_by_tag);
    }
    attention_inputs_ = std::move(attention_inputs);
    prepared_attention_inputs_.store(true, std::memory_order_release);

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

    graph_state_         = CudaGraphState();
    auto empty           = torch::Tensor();
    auto py_model_inputs = PyModelInputs({empty,
                                          empty,
                                          empty,
                                          torch_ext::PyEmbeddingInputs(),
                                          torch_ext::PyMultimodalInputs(),
                                          attention_inputs_,
                                          attention_inputs_by_tag_,
                                          torch_ext::BertEmbeddingInputs()});
    if (enable_cuda_graph_ && graph_runner_->canRun(py_model_inputs, graph_state_)) {
        RTP_LLM_PROFILE_SCOPE("py_model.prepareAttentionInputs(cuda_graph_prepare)");
        graph_runner_->prepareAttentionInputs(py_model_inputs, graph_state_, skip_forward_event_sync);
    }
}

void PyWrappedModel::updateKVCacheKernelBlockId(const GptModelInputs& inputs) {
    RTP_LLM_PROFILE_SCOPE("py_model.updateKVCacheKernelBlockId");
    if (!inputs.kv_cache_kernel_block_id.defined() || !prepared_attention_inputs_.load(std::memory_order_acquire)) {
        return;
    }
    const auto input_idx_by_tag = validateTaggedCacheBoundary(inputs);

    d2d_copies_.clear();
    attention_inputs_by_tag_ = setupKVCacheForAttentionInputs(attention_inputs_, inputs, input_idx_by_tag);
    fusedCopy(d2d_copies_);

    if (enable_cuda_graph_) {
        auto empty           = torch::Tensor();
        auto py_model_inputs = PyModelInputs({empty,
                                              empty,
                                              empty,
                                              torch_ext::PyEmbeddingInputs(),
                                              torch_ext::PyMultimodalInputs(),
                                              attention_inputs_,
                                              attention_inputs_by_tag_,
                                              torch_ext::BertEmbeddingInputs()});
        if (graph_runner_->canRun(py_model_inputs, graph_state_)) {
            graph_runner_->updateKVCacheKernelBlockId(py_model_inputs, graph_state_);
        }
    }
}

GptModelOutputs PyWrappedModel::forward(const GptModelInputs& inputs) {
    RTP_LLM_PROFILE_SCOPE("py_model.forward");
    DevicePerfWrapper wrapper(enable_device_perf_, "py model forward");
    const auto        input_idx_by_tag = validateTaggedCacheBoundary(inputs);
    holdInputsHostBuffers(inputs);

    // RAII guard: ensure prepared_attention_inputs_ is always reset to false on scope exit,
    // even if forward() throws. Without this, an exception after async prepareAttentionInputs
    // would leave the flag true, causing the next forward() to use stale attention_inputs_.
    struct PreparedFlagGuard {
        std::atomic<bool>& flag;
        ~PreparedFlagGuard() {
            flag.store(false, std::memory_order_release);
        }
    } flag_guard{prepared_attention_inputs_};

    try {
        RTP_LLM_LOG_DEBUG("Calling forward method on Python object instance.");

        if (int(device_props_.enable_layer_micro_batch)) {
            return forwardMicroBatchedValidated(inputs, input_idx_by_tag);
        }
        PyContextParallelParams cp_params;
        const bool              has_context_request = inputs.input_lengths.size(0) != inputs.sequence_lengths.size(0);
        if (device_props_.enable_prefill_cp && has_context_request) {
            // CP accepts pure-prefill batches without MTP/speculative hidden states;
            // handleInputs enforces both constraints before mutating the batch.
            context_parallel_processor_->handleInputs(const_cast<GptModelInputs&>(inputs), cp_params);
        }

        torch::Tensor token_ids;
        if (inputs.combo_tokens.device().is_cuda()) {
            token_ids = inputs.combo_tokens;
        } else {
            buffer_holder_.hold_host(inputs.combo_tokens);
            token_ids = inputs.combo_tokens.to(torch::kCUDA, /*non_blocking=*/true);
        }

        torch::Tensor input_hiddens =
            inputs.last_hidden_states.defined() ? inputs.last_hidden_states : torch::empty({0});

        torch::Tensor combo_position_ids = torch::empty({0});
        if (inputs.combo_position_ids.defined()) {
            if (inputs.combo_position_ids.device().is_cuda()) {
                combo_position_ids = inputs.combo_position_ids;
            } else {
                buffer_holder_.hold_host(inputs.combo_position_ids);
                combo_position_ids = inputs.combo_position_ids.to(torch::kCUDA, /*non_blocking=*/true);
            }
        }

        auto embedding_inputs      = buildPyEmbeddingInputs(inputs);
        auto multimodal_inputs     = buildPyMultimodalInputs(inputs);
        auto bert_embedding_inputs = buildBertEmbeddingInputs(inputs);
        if (!prepared_attention_inputs_.load(std::memory_order_acquire)) {
            prepareAttentionInputsValidated(inputs, /*skip_forward_event_sync=*/true, input_idx_by_tag);
        }
        if (device_props_.enable_prefill_cp && has_context_request) {
            attention_inputs_.context_parallel_info = cp_params;
            for (auto& [tag, tagged_inputs] : attention_inputs_by_tag_) {
                tagged_inputs.context_parallel_info = cp_params;
            }
        }

        if (device_props_.enable_prefill_cp && has_context_request
            && attention_inputs_.cache_store_inputs.has_value()) {
            // ContextParallelProcessor rewrites input_lengths to the rank-local
            // chunk; cache-store planning must keep the full pre-sharding lengths.
            attention_inputs_.cache_store_inputs->input_lengths_host = cp_params.prefill_actual_input_lengths_cpu;
        }
        const bool                has_cache_store_work = !inputs.warmup && inputs.pd_separation;
        CacheStoreWriteCycleGuard cache_store_write_cycle(cache_store_async_writer_, has_cache_store_work);

        auto           py_model_inputs = PyModelInputs({token_ids,
                                                        input_hiddens,
                                                        combo_position_ids,
                                                        embedding_inputs,
                                                        multimodal_inputs,
                                                        attention_inputs_,
                                                        attention_inputs_by_tag_,
                                                        bert_embedding_inputs});
        PyModelOutputs py_model_outputs;
        torch::Tensor  hidden_states;

        // Cast the Python object to PyModelOutputs and extract hidden states
        if (enable_cuda_graph_ && graph_runner_->canRun(py_model_inputs, graph_state_)) {
            py::gil_scoped_acquire gil;
            RTP_LLM_PROFILE_SCOPE("py_model.forward(cuda_graph)");
            DevicePerfWrapper wrapper(enable_device_perf_, "cuda graph python forward");
            RTP_LLM_LOG_DEBUG(
                "[PyWrappedModel] using CUDA graph forward, is_target_verify=%d, is_prefill=%d, graph_bs=%d",
                py_model_inputs.attention_inputs.is_target_verify,
                py_model_inputs.attention_inputs.is_prefill,
                graph_state_.current_real_graph_bs);
            py_model_inputs.attention_inputs.is_s_padded = true;
            py_model_outputs                             = graph_runner_->forward(py_model_inputs, graph_state_);
            RTP_LLM_LOG_DEBUG("[PyWrappedModel] CUDA graph forward completed");
            hidden_states = py_model_outputs.hidden_states.clone();
        } else {
            py::gil_scoped_acquire gil;
            RTP_LLM_PROFILE_SCOPE("py_model.forward(normal)");
            DevicePerfWrapper wrapper(enable_device_perf_, "normal forward");
            RTP_LLM_LOG_DEBUG("[PyWrappedModel] using normal forward, is_target_verify=%d, is_prefill=%d",
                              py_model_inputs.attention_inputs.is_target_verify,
                              py_model_inputs.attention_inputs.is_prefill);
            held_attn_pyobj_ = py_model_.attr("prepare_fmha_impl")(py_model_inputs, false);
            auto outputs     = py_forward_method_(py_model_inputs, held_attn_pyobj_);
            py_model_outputs = outputs.cast<PyModelOutputs>();
            hidden_states    = py_model_outputs.hidden_states.clone();
        }

        cache_store_write_cycle.finish();

        RTP_LLM_LOG_DEBUG("Python object instance forward method called successfully.");
        if (dspark_model_role_ != DSparkModelRole::NONE) {
            if (dspark_model_role_ == DSparkModelRole::PROPOSE) {
                // Python returns normalized [B*gamma, hidden_dim]. Reuse the
                // regular C++ lm_head and TP logits gather for every proposal
                // row; the speculative executor owns only Markov sampling.
                return callForwardPostLayers(hidden_states, inputs, true);
            }
            // Commit only updates the draft KV cache and has no logits
            // consumer. Preserve its row-aligned hidden output for the common
            // CUDA graph contract without running lm_head.
            GptModelOutputs outputs;
            outputs.hidden_states     = hidden_states;
            outputs.all_hidden_states = hidden_states;
            return outputs;
        }
        if (device_props_.enable_prefill_cp && has_context_request) {
            if (!inputs.need_all_logits && !inputs.need_all_hidden_states) {
                context_parallel_processor_->handleOutputsLastHidden(hidden_states, inputs, cp_params);
                return forwardPostLayersLastHidden(hidden_states, inputs);
            }
            size_t num_valid_tokens = context_parallel_processor_->handleOutputs(hidden_states, inputs, cp_params);
            return callForwardPostLayers(hidden_states, inputs, true, num_valid_tokens);
        }
        return callForwardPostLayers(hidden_states, inputs, true);

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

static torch::Tensor
sliceKvCacheBlockIdByBatch(const torch::Tensor& kv_cache_block_id, size_t batch_offset, size_t batch_size) {
    if (!kv_cache_block_id.defined()) {
        return torch::Tensor();
    }
    if (kv_cache_block_id.dim() == 2) {
        return kv_cache_block_id.narrow(0, batch_offset, batch_size);
    }
    if (kv_cache_block_id.dim() == 3) {
        // [group, batch, max_blocks] → narrow on dim 1
        return kv_cache_block_id.narrow(1, batch_offset, batch_size).contiguous();
    }
    return kv_cache_block_id;
}

torch::Tensor PyWrappedModel::tpSyncEmbeddingOrLogits(const torch::Tensor& input) {
    RTP_LLM_PROFILE_SCOPE("py_model.tpSyncEmbeddingOrLogits");
    const auto tp_size     = device_props_.tp_size;
    const auto tp_rank     = device_props_.tp_rank;
    const auto rows        = input.size(0);
    const auto cols        = input.size(1);
    const auto local_numel = input.numel();
    auto       all_data    = torch::empty({rows, cols * (int64_t)tp_size}, input.options());
    // Copy local data into the correct rank position
    auto all_data_flat = all_data.reshape({rows * cols * (int64_t)tp_size});
    auto input_flat    = input.reshape({local_numel});
    all_data_flat.slice(0, local_numel * tp_rank, local_numel * (tp_rank + 1)).copy_(input_flat);
    execAllGather({{all_data}});
    cudaCheckLastError();
    // Transpose [tp_size, batch, hidden] -> [batch, tp_size, hidden] -> [batch, hidden * tp_size]
    auto transposed = all_data.reshape({(int64_t)tp_size, rows, cols})
                          .permute({1, 0, 2})
                          .contiguous()
                          .reshape({rows, cols * (int64_t)tp_size});
    cudaCheckLastError();
    return transposed;
}

GptModelOutputs PyWrappedModel::forwardPostLayers(torch::Tensor         hidden,
                                                  const bool            has_context_request,
                                                  const bool            need_all_logits,
                                                  const torch::Tensor&  lm_output_indexes,
                                                  bool                  enable_sp,
                                                  size_t                token_num,
                                                  const GptModelInputs& inputs,
                                                  torch::Tensor         merged_eagle3_hidden,
                                                  bool                  skip_final_layernorm) {
    DevicePerfWrapper wrapper(enable_device_perf_, "forwardPostLayers");
    if (enable_sp && device_props_.tp_size > 1) {
        RTP_LLM_PROFILE_SCOPE("py_model.forwardPostLayers(sp_all_gather)");
        auto ag_tensor =
            torch::empty({(int64_t)(hidden.size(0) * device_props_.tp_size), hidden.size(1)}, hidden.options());
        size_t m                 = ag_tensor.size(0);
        int    m_split           = device_props_.m_split;
        size_t overlap_comm_type = device_props_.overlap_comm_type;
        if (overlap_comm_type == 1 && m_split > 0) {
            size_t token_idx    = 0;
            size_t ag_token_idx = 0;
            size_t m_chunk      = m / m_split;
            if (m > 128) {
                m_chunk = (m / m_split + 127) & ~127;
            }
            while (token_idx < m) {
                const auto micro_batch_tokens    = std::min(m - token_idx, m_chunk);
                const auto ag_micro_batch_tokens = micro_batch_tokens / device_props_.tp_size;
                auto       micro_batch_recv_t    = ag_tensor.narrow(0, token_idx, micro_batch_tokens);
                auto       micro_ag_send_t       = hidden.narrow(0, ag_token_idx, ag_micro_batch_tokens);
                execAllGather({{micro_batch_recv_t}, ParallelMode::TP, {micro_ag_send_t}, false});
                token_idx += micro_batch_tokens;
                ag_token_idx += ag_micro_batch_tokens;
            }
        } else {
            execAllGather({{ag_tensor}, ParallelMode::TP, {hidden}, false});
        }

        size_t pad_mod_num = device_props_.tp_size * max((size_t)1, device_props_.m_split);
        if (token_num % pad_mod_num != 0) {
            hidden = ag_tensor.slice(0, 0, token_num).contiguous();
        } else {
            hidden = ag_tensor;
        }
    }

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

        torch::Tensor logits;
        {
            RTP_LLM_PROFILE_SCOPE("py_model.forwardPostLayers(lm_head_mm)");
#if USING_CUDA
            if (lm_head->kernel.dtype() == torch::kBFloat16) {
                logits = torch_ext::cublas_gemm_bf16_bf16_fp32(last_hidden.to(torch::kBFloat16), lm_head->kernel);
            } else
#endif
            {
                logits = torch::mm(last_hidden.to(lm_head->kernel.dtype()), lm_head->kernel.t()).to(torch::kFloat32);
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
        torch::Tensor softmax_result_t;
        if (need_all_logits) {
            RTP_LLM_PROFILE_SCOPE("py_model.forwardPostLayers(need_all_logits_index)");
            auto last_logits = torch::index_select(logits, 0, lm_output_indexes_device.to(torch::kLong));
            return {last_logits, last_hidden, hidden, logits, softmax_result_t};
        }

        if (merged_eagle3_hidden.defined()) {
            hidden = merged_eagle3_hidden;
        }
        return {logits, last_hidden, hidden, torch::Tensor(), softmax_result_t};
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
    DevicePerfWrapper wrapper(enable_device_perf_, "forwardPostLayersLastHidden");
    const auto&       lm_head = weights_.lm_head;
    if (!lm_head) {
        return {torch::Tensor(), torch::Tensor(), hidden};
    }
    printTorchTensorData(hidden, "last_hidden");

    torch::Tensor last_hidden = hidden;
    torch::Tensor logits;
    {
        RTP_LLM_PROFILE_SCOPE("py_model.forwardPostLayersLastHidden(lm_head_mm)");
#if USING_CUDA
        if (lm_head->kernel.dtype() == torch::kBFloat16) {
            logits = torch_ext::cublas_gemm_bf16_bf16_fp32(last_hidden.to(torch::kBFloat16), lm_head->kernel);
        } else
#endif
        {
            logits = torch::mm(last_hidden.to(lm_head->kernel.dtype()), lm_head->kernel.t()).to(torch::kFloat32);
        }
    }
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

MicroBatchPlan PyWrappedModel::planMicroBatches(const GptModelInputs& inputs) {
    if (!int(device_props_.enable_layer_micro_batch)) {
        RTP_LLM_LOG_DEBUG("micro batch disable when enable_layer_micro_batch is false");
        return {false, {}};
    }

    const auto&  input_lengths      = inputs.input_lengths;
    const auto&  sequence_lengths   = inputs.sequence_lengths;
    const size_t decoder_batch_size = sequence_lengths.size(0);
    const size_t context_batch_size = input_lengths.size(0) - decoder_batch_size;
    // TODO(async): layer micro-batch planning still needs host lengths for
    // split arithmetic. Keep the CPU mirror explicit while model inputs stay CUDA.
    const auto input_lengths_host = input_lengths.is_cuda() ? input_lengths.cpu().pin_memory() : input_lengths;
    const auto input_lengths_ptr  = input_lengths_host.data_ptr<int32_t>();

    if (decoder_batch_size + context_batch_size < 2) {
        RTP_LLM_LOG_DEBUG("micro batch disable when batch size %ld is less than 2",
                          decoder_batch_size + context_batch_size);
        return {false, {}};
    }

    if (context_batch_size && decoder_batch_size) {
        if (layer_num_ == 1) {
            size_t total_token_num = decoder_batch_size;
            for (size_t i = 0; i < context_batch_size; i++) {
                total_token_num += input_lengths_ptr[i + decoder_batch_size];
            }
            RTP_LLM_LOG_DEBUG("total_token_num %ld, decode_batch_size %ld, context_batch_size %ld",
                              total_token_num,
                              decoder_batch_size,
                              context_batch_size);
            size_t context_batch_0_size = 0;
            size_t context_batch_1_size = 0;
            size_t decode_batch_0_size  = 0;
            size_t decode_batch_1_size  = 0;
            if (total_token_num > decoder_batch_size * 2) {
                decode_batch_0_size        = decoder_batch_size;
                decode_batch_1_size        = 0;
                size_t acc_token_num       = decoder_batch_size;
                size_t context_split_point = 0;
                for (context_split_point = 0; context_split_point < context_batch_size; context_split_point++) {
                    acc_token_num += input_lengths_ptr[context_split_point + decoder_batch_size];
                    if (acc_token_num * 2 >= total_token_num) {
                        break;
                    }
                }
                context_batch_0_size = context_split_point;
                context_batch_1_size = context_batch_size - context_split_point;
            } else {
                decode_batch_0_size  = total_token_num / 2;
                decode_batch_1_size  = decoder_batch_size - total_token_num / 2;
                context_batch_0_size = 0;
                context_batch_1_size = context_batch_size;
            }
            RTP_LLM_LOG_DEBUG("split [c]%d:[d]%d in micro batch 0 and [c]%d:[d]%d in micro batch 1",
                              context_batch_0_size,
                              decode_batch_0_size,
                              context_batch_1_size,
                              decode_batch_1_size);
            return MicroBatchPlan{
                true, {{context_batch_0_size, decode_batch_0_size}, {context_batch_1_size, decode_batch_1_size}}};
        } else {
            RTP_LLM_LOG_DEBUG("split context in micro batch 0, decode in micro batch 1 disabled!");
            return {false, {}};
        }
    }

    const size_t batch_size_to_split = context_batch_size ? context_batch_size : decoder_batch_size;
    const size_t micro_batch_0_size  = (batch_size_to_split + 1) / 2;
    const size_t micro_batch_1_size  = batch_size_to_split - micro_batch_0_size;

    RTP_LLM_LOG_DEBUG("split micro batch size %ld, %ld", micro_batch_0_size, micro_batch_1_size);
    return context_batch_size ? MicroBatchPlan{true, {{micro_batch_0_size, 0}, {micro_batch_1_size, 0}}} :
                                MicroBatchPlan{true, {{0, micro_batch_0_size}, {0, micro_batch_1_size}}};
}

std::pair<std::vector<GptModelInputs>, std::vector<TokenSliceInfo>>
PyWrappedModel::splitInputsIntoMicroBatches(const GptModelInputs& inputs, const MicroBatchPlan& micro_batch_plan) {
    std::vector<GptModelInputs> micro_batch_inputs;
    std::vector<TokenSliceInfo> token_slice_recipes;
    size_t                      sliced_token_idx       = 0;
    size_t                      sliced_lm_output_index = 0;
    size_t                      sliced_batch_idx       = 0;
    size_t                      decode_batch_idx       = 0;
    size_t                      prefill_batch_idx      = 0;
    // TODO(async): micro-batch token slicing still computes CPU scalar sums.
    // Convert explicitly and keep all sliced GptModelInputs device-resident.
    const auto  input_lengths_host = inputs.input_lengths.defined() && inputs.input_lengths.is_cuda() ?
                                         inputs.input_lengths.cpu().pin_memory() :
                                         inputs.input_lengths;
    const auto* input_lengths_ptr  = input_lengths_host.defined() ? input_lengths_host.data_ptr<int32_t>() : nullptr;

    if (!micro_batch_plan.enable) {
        RTP_LLM_LOG_DEBUG("micro batch disable when enable is false, use fake");
        micro_batch_inputs.push_back(inputs);

        GptModelInputs fake_inputs;
        fake_inputs.kv_cache_block_id = torch::Tensor();
        fake_inputs.combo_tokens      = inputs.combo_tokens.narrow(0, 0, 1);
        fake_inputs.input_lengths     = torch::ones({1}, torch::TensorOptions(torch::kInt32).device(torch::kCUDA));
        fake_inputs.sequence_lengths  = torch::empty({0}, torch::TensorOptions(torch::kInt32).device(torch::kCUDA));
        fake_inputs.prefix_lengths    = torch::zeros({1}, torch::TensorOptions(torch::kInt32).device(torch::kCUDA));
        micro_batch_inputs.push_back(fake_inputs);
    } else {
        for (size_t i = 0; i < micro_batch_plan.batch_infos.size(); ++i) {
            const auto& p_micro_batch_size = micro_batch_plan.batch_infos[i].prefill_num;
            const auto& d_micro_batch_size = micro_batch_plan.batch_infos[i].decoder_num;
            RTP_LLM_LOG_DEBUG(
                "micro batch index %ld, prefill size %ld, decode size %ld", i, p_micro_batch_size, d_micro_batch_size);

            if (d_micro_batch_size && p_micro_batch_size) {
                GptModelInputs micro_model_inputs = inputs;
                size_t         total_batch_size   = d_micro_batch_size + p_micro_batch_size;
                RTP_LLM_LOG_DEBUG("d and p slice from %ld %ld %ld %ld",
                                  sliced_token_idx,
                                  sliced_batch_idx,
                                  decode_batch_idx,
                                  prefill_batch_idx);
                micro_model_inputs.input_lengths = inputs.input_lengths.narrow(0, sliced_batch_idx, total_batch_size);
                micro_model_inputs.sequence_lengths =
                    inputs.sequence_lengths.narrow(0, decode_batch_idx, d_micro_batch_size);
                micro_model_inputs.kv_cache_block_id =
                    sliceKvCacheBlockIdByBatch(inputs.kv_cache_block_id, sliced_batch_idx, total_batch_size);
                micro_model_inputs.kv_cache_kernel_block_id =
                    sliceKvCacheBlockIdByBatch(inputs.kv_cache_kernel_block_id, sliced_batch_idx, total_batch_size);
                micro_model_inputs.prefix_lengths =
                    inputs.prefix_lengths.narrow(0, prefill_batch_idx, p_micro_batch_size);
                micro_model_inputs.attention_mask =
                    inputs.attention_mask.defined() ?
                        inputs.attention_mask.narrow(0, sliced_batch_idx, total_batch_size) :
                        torch::Tensor();
                int32_t slice_token_num = std::accumulate(input_lengths_ptr + sliced_batch_idx + d_micro_batch_size,
                                                          input_lengths_ptr + sliced_batch_idx + total_batch_size,
                                                          0)
                                          + d_micro_batch_size;
                int32_t slice_lm_output_num = total_batch_size;
                micro_model_inputs.lm_output_indexes =
                    inputs.lm_output_indexes.narrow(0, sliced_lm_output_index, slice_lm_output_num);
                micro_model_inputs.combo_tokens = inputs.combo_tokens.narrow(0, sliced_token_idx, slice_token_num);
                micro_model_inputs.request_id   = inputs.request_id.defined() ?
                                                      inputs.request_id.narrow(0, prefill_batch_idx, p_micro_batch_size) :
                                                      torch::Tensor();
                micro_model_inputs.request_pd_separation =
                    inputs.request_pd_separation.defined() ?
                        inputs.request_pd_separation.narrow(0, prefill_batch_idx, p_micro_batch_size) :
                        torch::Tensor();
                micro_model_inputs.cache_keys = inputs.cache_keys.defined() ?
                                                    inputs.cache_keys.narrow(0, prefill_batch_idx, p_micro_batch_size) :
                                                    torch::Tensor();

                token_slice_recipes.emplace_back(TokenSliceInfo{sliced_token_idx, (size_t)slice_token_num});

                micro_batch_inputs.push_back(micro_model_inputs);

                sliced_lm_output_index += slice_lm_output_num;
                sliced_token_idx += slice_token_num;
                sliced_batch_idx += total_batch_size;
                prefill_batch_idx += p_micro_batch_size;
                decode_batch_idx += d_micro_batch_size;
                RTP_LLM_LOG_DEBUG(
                    "micro batch %ld sliced context and decode, batch idx %ld, token idx %ld, prefill batch idx %d, decode batch idx %d",
                    i,
                    sliced_batch_idx,
                    sliced_token_idx,
                    prefill_batch_idx,
                    decode_batch_idx);
            } else if (d_micro_batch_size) {
                GptModelInputs micro_model_inputs = inputs;
                RTP_LLM_LOG_DEBUG("d slice from %ld %ld %ld", sliced_token_idx, sliced_batch_idx, decode_batch_idx);
                micro_model_inputs.combo_tokens  = inputs.combo_tokens.narrow(0, sliced_token_idx, d_micro_batch_size);
                micro_model_inputs.input_lengths = inputs.input_lengths.narrow(0, sliced_batch_idx, d_micro_batch_size);
                micro_model_inputs.sequence_lengths =
                    inputs.sequence_lengths.narrow(0, decode_batch_idx, d_micro_batch_size);
                micro_model_inputs.attention_mask =
                    inputs.attention_mask.defined() ?
                        inputs.attention_mask.narrow(0, sliced_batch_idx, d_micro_batch_size) :
                        torch::Tensor();
                micro_model_inputs.kv_cache_block_id =
                    sliceKvCacheBlockIdByBatch(inputs.kv_cache_block_id, sliced_batch_idx, d_micro_batch_size);
                micro_model_inputs.kv_cache_kernel_block_id =
                    sliceKvCacheBlockIdByBatch(inputs.kv_cache_kernel_block_id, sliced_batch_idx, d_micro_batch_size);
                micro_model_inputs.prefix_lengths =
                    torch::empty({0}, torch::TensorOptions(torch::kInt32).device(torch::kCUDA));
                micro_model_inputs.lm_output_indexes =
                    inputs.lm_output_indexes.narrow(0, sliced_batch_idx, d_micro_batch_size);

                token_slice_recipes.emplace_back(TokenSliceInfo{sliced_token_idx, d_micro_batch_size});

                micro_batch_inputs.push_back(micro_model_inputs);

                sliced_token_idx += d_micro_batch_size;
                sliced_batch_idx += d_micro_batch_size;
                decode_batch_idx += d_micro_batch_size;
                sliced_lm_output_index += d_micro_batch_size;
                RTP_LLM_LOG_DEBUG("micro batch %ld sliced decode, batch idx %ld, token idx %ld",
                                  i,
                                  sliced_batch_idx,
                                  sliced_token_idx);
            } else {
                GptModelInputs micro_model_inputs = inputs;
                RTP_LLM_LOG_DEBUG("p slice from %ld %ld %ld", sliced_token_idx, sliced_batch_idx, prefill_batch_idx);
                micro_model_inputs.input_lengths = inputs.input_lengths.narrow(0, sliced_batch_idx, p_micro_batch_size);
                micro_model_inputs.kv_cache_block_id =
                    sliceKvCacheBlockIdByBatch(inputs.kv_cache_block_id, sliced_batch_idx, p_micro_batch_size);
                micro_model_inputs.kv_cache_kernel_block_id =
                    sliceKvCacheBlockIdByBatch(inputs.kv_cache_kernel_block_id, sliced_batch_idx, p_micro_batch_size);
                micro_model_inputs.prefix_lengths =
                    inputs.prefix_lengths.narrow(0, prefill_batch_idx, p_micro_batch_size);
                micro_model_inputs.attention_mask =
                    inputs.attention_mask.defined() ?
                        inputs.attention_mask.narrow(0, sliced_batch_idx, p_micro_batch_size) :
                        torch::Tensor();
                micro_model_inputs.sequence_lengths =
                    torch::empty({0}, torch::TensorOptions(torch::kInt32).device(torch::kCUDA));
                int32_t slice_token_num = std::accumulate(
                    input_lengths_ptr + sliced_batch_idx, input_lengths_ptr + sliced_batch_idx + p_micro_batch_size, 0);
                int32_t slice_lm_output_num = p_micro_batch_size;
                micro_model_inputs.lm_output_indexes =
                    inputs.lm_output_indexes.narrow(0, sliced_lm_output_index, slice_lm_output_num);
                micro_model_inputs.combo_tokens = inputs.combo_tokens.narrow(0, sliced_token_idx, slice_token_num);
                micro_model_inputs.request_id   = inputs.request_id.defined() ?
                                                      inputs.request_id.narrow(0, prefill_batch_idx, p_micro_batch_size) :
                                                      torch::Tensor();
                micro_model_inputs.request_pd_separation =
                    inputs.request_pd_separation.defined() ?
                        inputs.request_pd_separation.narrow(0, prefill_batch_idx, p_micro_batch_size) :
                        torch::Tensor();
                micro_model_inputs.cache_keys = inputs.cache_keys.defined() ?
                                                    inputs.cache_keys.narrow(0, prefill_batch_idx, p_micro_batch_size) :
                                                    torch::Tensor();

                token_slice_recipes.emplace_back(TokenSliceInfo{sliced_token_idx, (size_t)slice_token_num});

                micro_batch_inputs.push_back(micro_model_inputs);
                sliced_lm_output_index += slice_lm_output_num;
                sliced_token_idx += slice_token_num;
                sliced_batch_idx += p_micro_batch_size;
                prefill_batch_idx += p_micro_batch_size;
                RTP_LLM_LOG_DEBUG("micro batch %ld sliced context, batch idx %ld, token idx %ld",
                                  i,
                                  sliced_batch_idx,
                                  sliced_token_idx);
            }
        }
    }
    return {micro_batch_inputs, token_slice_recipes};
}

void PyWrappedModel::holdInputsHostBuffers(const GptModelInputs& inputs) {
    buffer_holder_.hold_host(inputs.combo_tokens);
    buffer_holder_.hold_host(inputs.input_lengths);
    buffer_holder_.hold_host(inputs.sequence_lengths);
    buffer_holder_.hold_host(inputs.lm_output_indexes);
    buffer_holder_.hold_host(inputs.prefix_lengths);

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
