#include "rtp_llm/cpp/cuda_graph/cuda_graph_runner.h"

#include <algorithm>
#include <cstring>
#include "rtp_llm/cpp/cuda_graph/cuda_graph_device_shims.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
using namespace torch_ext;
namespace rtp_llm {

// clang-format off
// CUDA Graph Mode Configuration Table:
// +--------------------------------+-----------------------------+--------------------------------------+--------------+
// | Model Type                     | is_prefill_cuda_graph_mode_ | num_tokens_per_bs_                   | 是否已经支持   |
// +--------------------------------+-----------------------------+--------------------------------------+--------------+
// | Draft Model (prefill)          | true                        | gen_num_per_cycle + 1                | yes          |
// | Target Model (score, prefill)  | false                       | gen_num_per_cycle + 1                | yes          |
// | Draft Model (decode)           | false                       | 1                                    | yes          |
// | Embedding Model (prefill)      | true                        | max_seq_len                          | yes          |
// | Normal Model (decode)          | false                       | 1                                    | yes          |
// +--------------------------------+-----------------------------+--------------------------------------+--------------+
// Notes:
// - Speculative sampling: model_id == 0 (target), model_id == 1 (draft)
// - Target model with spec sampling processes multiple tokens per batch for verification phase
// clang-format on

// Helper function for optimized tensor copy using async operations with current CUDA stream
void optimizedCopyAsync(const torch::Tensor& src, torch::Tensor& dst, size_t size) {
    if (!src.defined() || src.numel() <= 0 || !dst.defined() || dst.numel() <= 0) {
        return;
    }

    void* stream = reinterpret_cast<void*>(cuda_graph::graphGetCurrentStream().stream());
    if (src.is_cuda() && dst.is_cuda()) {
        cuda_graph::graphMemcpyAsync(dst.data_ptr(), src.data_ptr(), size, cuda_graph::GraphMemcpyKind::D2D, stream);
    } else if (!src.is_cuda() && !dst.is_cuda()) {
        std::memcpy(dst.data_ptr(), src.data_ptr(), size);
    } else if (src.is_cuda() && !dst.is_cuda()) {
        cuda_graph::graphMemcpyAsync(dst.data_ptr(), src.data_ptr(), size, cuda_graph::GraphMemcpyKind::D2H, stream);
    } else {
        cuda_graph::graphMemcpyAsync(dst.data_ptr(), src.data_ptr(), size, cuda_graph::GraphMemcpyKind::H2D, stream);
    }
}

void CudaGraphRunner::clearTensorAsync(torch::Tensor& tensor) {
    if (!tensor.defined() || tensor.numel() <= 0) {
        return;
    }
    if (tensor.is_cuda()) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
        check_cuda_value(cudaMemsetAsync(tensor.data_ptr(), 0, tensor.nbytes(), stream));
    } else {
        memset(tensor.data_ptr(), 0, tensor.nbytes());
    }
}

// column dimension
void CudaGraphRunner::copySmallerIntoLarger(const torch::Tensor& source_tensor, torch::Tensor& target_tensor) {
    if (!source_tensor.defined() || source_tensor.numel() <= 0) {
        return;
    }
    if (!target_tensor.defined() || target_tensor.numel() <= 0) {
        return;
    }
    if (source_tensor.dim() != target_tensor.dim()) {
        throw std::runtime_error(
            "Error: Source and target tensors must have the same number of dimensions. source.dim()="
            + std::to_string(source_tensor.dim()) + ", target.dim()=" + std::to_string(target_tensor.dim()) + ".");
    }
    for (int i = 0; i < source_tensor.dim(); ++i) {
        if (source_tensor.size(i) > target_tensor.size(i)) {
            throw std::runtime_error(
                "Error: Target tensor dimension " + std::to_string(i) + " (" + std::to_string(target_tensor.size(i))
                + ") is smaller than source tensor dimension " + std::to_string(i) + " ("
                + std::to_string(source_tensor.size(i)) + "). This violates the function's guarantee.");
        }
    }

    // Fast path: if trailing dimensions match and tensors are contiguous, use direct memcpy
    if (source_tensor.is_contiguous() && target_tensor.is_contiguous()) {
        bool trailing_dims_match = true;
        for (int i = 1; i < source_tensor.dim(); ++i) {
            if (source_tensor.size(i) != target_tensor.size(i)) {
                trailing_dims_match = false;
                break;
            }
        }
        if (trailing_dims_match) {
            optimizedCopyAsync(source_tensor, target_tensor, source_tensor.numel() * source_tensor.element_size());
            return;
        }
    }

    // Fallback: use slice + copy for non-contiguous or mismatched trailing dims
    torch::Tensor target_slice = target_tensor;

    for (int i = 0; i < source_tensor.dim(); ++i) {
        target_slice = target_slice.slice(i, 0, source_tensor.size(i));
    }

    target_slice.copy_(source_tensor);
}

void CudaGraphRunner::prepareInputs(const PyModelInputs& inputs, CudaGraphState& state) {
    RTP_LLM_PROFILE_SCOPE("cuda_graph.prepareInputs");
    // 1. non spec cuda graph:
    // is_prefill_cuda_graph_mode_ is set true only when use embedding model
    // 2. spec cuda graph:
    // 2.1 spec hold target model and draft model. when the user prompt first comes in, the target model
    // adn draft model will do real "prefill forward". And for this phase, we don't support cuda graph
    // 2.2 after real "prefill forward", it is consisted of three parts:
    // 2.2.1 target model score(verfiy)
    // 2.2.2 draft model do first forward (input is from 2.2.1)
    // 2.2.3 draft model do auto-agressive forward
    // for now we only support 2.2.1 and 2.2.3 in deocode cuda graph, and 2.2.2 will be support in prefill cuda graph.

    // should wait last forward done before prepare inputs
    forward_event_.synchronize();

    // Get the appropriate graph instance based on mode
    const size_t graph_idx =
        is_prefill_cuda_graph_mode_ ? state.current_real_graph_seq_len : state.current_real_graph_bs;
    auto& py_model_inputs_ = graph_instances_[graph_idx].mem_hold_.py_model_inputs_;
    auto  attn_pyobj       = graph_instances_[graph_idx].mem_hold_.attn_pyobj_;

    // Clear kv_cache block ids to prevent cache block pollution (use cudaMemsetAsync to avoid kernel launches)
    clearTensorAsync(py_model_inputs_.attention_inputs.kv_cache_kernel_block_id_device);
    if (is_target_verify_) {
        // For target verify, fill_params() reads from the HOST block table to compute
        // page metadata. Stale entries in padding rows would cause append_paged_kv_cache
        // to write garbage K/V into pages belonging to other requests.
        clearTensorAsync(py_model_inputs_.attention_inputs.kv_cache_kernel_block_id_host);
    }
    if (py_model_inputs_.attention_inputs.kv_cache_block_id_device.defined()) {
        clearTensorAsync(py_model_inputs_.attention_inputs.kv_cache_block_id_device);
    }
    if (py_model_inputs_.attention_inputs.kv_cache_block_id_host.defined()) {
        clearTensorAsync(py_model_inputs_.attention_inputs.kv_cache_block_id_host);
    }

    // Common copies: input_ids, input_hiddens, attention lengths, kv_cache blocks
    int token_num = is_prefill_cuda_graph_mode_ ? state.current_seq_len : inputs.input_ids.size(0);
    optimizedCopyAsync(inputs.input_ids, py_model_inputs_.input_ids, token_num * sizeof(int));
    optimizedCopyAsync(inputs.input_hiddens,
                       py_model_inputs_.input_hiddens,
                       inputs.input_hiddens.numel() * inputs.input_hiddens.element_size());
    if (!is_prefill_cuda_graph_mode_ && token_num < py_model_inputs_.input_ids.size(0)) {
        // Decode replay always executes the full captured token shape. When the
        // current request is served by a larger captured graph (common for
        // target-verify MTP), untouched tail token slots would otherwise retain
        // input_ids / hidden states from the previous replay and get consumed as
        // fake padding rows, leading to cross-request corruption under concurrency.
        py_model_inputs_.input_ids.slice(0, token_num, py_model_inputs_.input_ids.size(0)).fill_(0);
        py_model_inputs_.input_hiddens.slice(0, token_num, py_model_inputs_.input_hiddens.size(0)).fill_(0);
    }
    optimizedCopyAsync(inputs.attention_inputs.prefix_lengths,
                       py_model_inputs_.attention_inputs.prefix_lengths,
                       state.current_batch_size * sizeof(int));
    optimizedCopyAsync(inputs.attention_inputs.input_lengths,
                       py_model_inputs_.attention_inputs.input_lengths,
                       state.current_batch_size * sizeof(int));
    // Device tensor copies for prepare_cuda_graph Triton kernels (spec-decode & prefill)
    optimizedCopyAsync(inputs.attention_inputs.prefix_lengths_d,
                       py_model_inputs_.attention_inputs.prefix_lengths_d,
                       state.current_batch_size * sizeof(int));
    optimizedCopyAsync(inputs.attention_inputs.input_lengths_d,
                       py_model_inputs_.attention_inputs.input_lengths_d,
                       state.current_batch_size * sizeof(int));
    optimizedCopyAsync(inputs.attention_inputs.cu_seqlens,
                       py_model_inputs_.attention_inputs.cu_seqlens,
                       (state.current_batch_size + 1) * sizeof(int));
    optimizedCopyAsync(inputs.attention_inputs.cu_kv_seqlens,
                       py_model_inputs_.attention_inputs.cu_kv_seqlens,
                       (state.current_batch_size + 1) * sizeof(int));

    if (inputs.attention_inputs.kv_cache_block_id_device.defined()) {
        copySmallerIntoLarger(inputs.attention_inputs.kv_cache_block_id_device,
                              py_model_inputs_.attention_inputs.kv_cache_block_id_device);
    }
    if (inputs.attention_inputs.kv_cache_block_id_host.defined()) {
        copySmallerIntoLarger(inputs.attention_inputs.kv_cache_block_id_host,
                              py_model_inputs_.attention_inputs.kv_cache_block_id_host);
    }
    copySmallerIntoLarger(inputs.attention_inputs.kv_cache_kernel_block_id_device,
                          py_model_inputs_.attention_inputs.kv_cache_kernel_block_id_device);
    copySmallerIntoLarger(inputs.attention_inputs.kv_cache_kernel_block_id_host,
                          py_model_inputs_.attention_inputs.kv_cache_kernel_block_id_host);

    // Mode-specific copies
    if (!is_prefill_cuda_graph_mode_) {
        optimizedCopyAsync(inputs.attention_inputs.sequence_lengths,
                           py_model_inputs_.attention_inputs.sequence_lengths,
                           state.current_batch_size * sizeof(int));
        optimizedCopyAsync(inputs.attention_inputs.sequence_lengths_plus_1_d,
                           py_model_inputs_.attention_inputs.sequence_lengths_plus_1_d,
                           state.current_batch_size * sizeof(int));
        optimizedCopyAsync(inputs.attention_inputs.decode_cu_seqlens_d,
                           py_model_inputs_.attention_inputs.decode_cu_seqlens_d,
                           (state.current_batch_size + 1) * sizeof(int));
        // For target-verify decode graphs: when current_batch_size < graph_batch_size,
        // extra batch entries retain stale data from previous replays. In particular,
        // kv_cache_kernel_block_id_host keeps stale page IDs that point to KV cache pages
        // belonging to other requests. fill_params() generates page metadata from these
        // stale entries, and append_paged_kv_cache() then writes garbage K/V into those
        // pages, corrupting other requests' KV cache. Fix: zero out extra batch metadata
        // so fill_params() generates no entries for padding batches.
        if (is_target_verify_ && state.current_batch_size < max_bs_) {
            py_model_inputs_.attention_inputs.input_lengths.slice(0, state.current_batch_size, max_bs_).fill_(0);
            py_model_inputs_.attention_inputs.prefix_lengths.slice(0, state.current_batch_size, max_bs_).fill_(0);
            py_model_inputs_.attention_inputs.sequence_lengths.slice(0, state.current_batch_size, max_bs_).fill_(0);
            int last_offset = state.current_batch_size * num_tokens_per_bs_;
            py_model_inputs_.attention_inputs.decode_cu_seqlens_d.slice(0, state.current_batch_size + 1, max_bs_ + 1)
                .fill_(last_offset);
        }
        // Normal decode CG: `_prepare_cg_decode_kernel` reads sequence_lengths_plus_1_d
        // for the FULL captured batch size to fill `_cg.seq_lens`. The above copy only
        // refreshes the first current_batch_size entries; padding rows retain whatever
        // stale lengths the previous (larger) replay wrote, causing FlashInfer to read
        // wrong KV ranges for those rows. For target-verify Qwen3-Next, the linear
        // attention kernels use `sequence_lengths_plus_1_d - 1` to locate the current
        // cache block and explicitly assume the value is >= 1. So padding rows must use
        // 1 (pointing at reserved block 0), not 0.
        if (state.current_batch_size < max_bs_) {
            py_model_inputs_.attention_inputs.sequence_lengths_plus_1_d.slice(0, state.current_batch_size, max_bs_)
                .fill_(is_target_verify_ ? 1 : 0);
        }
        auto attn_pyobj = graph_instances_[state.current_real_graph_bs].mem_hold_.attn_pyobj_;
        // decode padding
        attn_pyobj.attr("prepare_cuda_graph")(py_model_inputs_.attention_inputs);

    } else {
        optimizedCopyAsync(inputs.attention_inputs.padding_offset,
                           py_model_inputs_.attention_inputs.padding_offset,
                           state.current_seq_len * sizeof(int));

        if (py_model_inputs_.attention_inputs.prefill_cuda_graph_copy_params) {
            (*(py_model_inputs_.attention_inputs.prefill_cuda_graph_copy_params->cuda_graph_prefill_batch_size
                   .data_ptr<int>())) = state.current_batch_size;
        }

        if (inputs.bert_embedding_inputs.position_encoding.numel() > 0) {
            optimizedCopyAsync(inputs.bert_embedding_inputs.combo_position_ids,
                               py_model_inputs_.bert_embedding_inputs.combo_position_ids,
                               state.current_seq_len * sizeof(int));
            optimizedCopyAsync(inputs.bert_embedding_inputs.combo_tokens_type_ids,
                               py_model_inputs_.bert_embedding_inputs.combo_tokens_type_ids,
                               state.current_seq_len * sizeof(int));
        }

        // Reset unused batch portions to prevent stale data
        if (state.current_batch_size < max_bs_) {
            py_model_inputs_.attention_inputs.prefix_lengths.slice(0, state.current_batch_size, max_bs_).fill_(0);
            py_model_inputs_.attention_inputs.input_lengths.slice(0, state.current_batch_size, max_bs_).fill_(0);
            if (py_model_inputs_.attention_inputs.prefix_lengths_d.defined()) {
                py_model_inputs_.attention_inputs.prefix_lengths_d.slice(0, state.current_batch_size, max_bs_).fill_(0);
            }
            if (py_model_inputs_.attention_inputs.input_lengths_d.defined()) {
                py_model_inputs_.attention_inputs.input_lengths_d.slice(0, state.current_batch_size, max_bs_).fill_(0);
            }
            // Padding rows: sequence_lengths_plus_1_d must be >= 1 for SSM kernels
            // (they compute index = value - 1; 0 would underflow to ~4.3B → OOB).
            py_model_inputs_.attention_inputs.sequence_lengths_plus_1_d.slice(0, state.current_batch_size, max_bs_)
                .fill_(1);
        }

        // Propagate per-batch sequence_lengths_plus_1_d from the actual request so
        // SSM/linear-attention kernels (Qwen3-Next GatedDeltaNet) see the correct
        // state slot for each batch.  PyWrappedModel sets this to
        // sequence_lengths + 1 (or prefix_lengths + 1 for target_verify), so it
        // reflects the true KV-cache position before the current draft tokens.
        if (inputs.attention_inputs.sequence_lengths_plus_1_d.defined()) {
            optimizedCopyAsync(inputs.attention_inputs.sequence_lengths_plus_1_d,
                               py_model_inputs_.attention_inputs.sequence_lengths_plus_1_d,
                               state.current_batch_size * sizeof(int));
        }

        int last_valid = state.current_seq_len;
        py_model_inputs_.attention_inputs.cu_seqlens.slice(0, state.current_batch_size + 1, max_bs_ + 1)
            .fill_(last_valid);
        py_model_inputs_.attention_inputs.cu_kv_seqlens.slice(0, state.current_batch_size + 1, max_bs_ + 1)
            .fill_(last_valid);
    }

    // Refresh device versions of prefix_lengths/input_lengths on the captured
    // attention_inputs so that prepare_cuda_graph (called below, OUTSIDE the
    // captured graph) reads the live request lengths instead of stale/undefined
    // device pointers. These tensors are consumed by Triton kernels
    // `_prepare_cg_spec_decode_kernel` (target verify) and
    // `_prepare_cg_prefill_kernel` (draft prefill) to fill `_cg.seq_lens`,
    // which is then read by the captured graph during replay. Without this
    // refresh, both target_verify and draft_prefill CG paths consume stale
    // device data and corrupt KV cache offsets, producing non-deterministic
    // and garbled outputs under concurrency. The kernels iterate over the
    // full captured batch size, so we copy live data into the pre-allocated
    // captured device buffers (which already have padding rows zeroed at init).
    const int valid_bs = state.current_batch_size;
    if (inputs.attention_inputs.prefix_lengths_d.defined()
        && py_model_inputs_.attention_inputs.prefix_lengths_d.defined()) {
        // Clear padding rows first to ensure no stale data remains, then copy
        // valid entries from the live device tensor. The Triton prepare kernels
        // iterate over the full captured batch size, so padding rows must be
        // valid (zero seq length) to avoid bogus KV cache offsets.
        if (valid_bs < max_bs_) {
            py_model_inputs_.attention_inputs.prefix_lengths_d.slice(0, valid_bs, max_bs_).fill_(0);
        }
        optimizedCopyAsync(inputs.attention_inputs.prefix_lengths_d,
                           py_model_inputs_.attention_inputs.prefix_lengths_d,
                           valid_bs * sizeof(int));
    }
    if (inputs.attention_inputs.input_lengths_d.defined()
        && py_model_inputs_.attention_inputs.input_lengths_d.defined()) {
        if (valid_bs < max_bs_) {
            py_model_inputs_.attention_inputs.input_lengths_d.slice(0, valid_bs, max_bs_).fill_(0);
        }
        optimizedCopyAsync(inputs.attention_inputs.input_lengths_d,
                           py_model_inputs_.attention_inputs.input_lengths_d,
                           valid_bs * sizeof(int));
    }

    attn_pyobj.attr("prepare_cuda_graph")(py_model_inputs_.attention_inputs);

    // Hybrid cache: update per-group block tables (including group 0).
    if (!inputs.attention_inputs.kv_cache_kernel_block_id_device_by_group.empty()
        && !inputs.attention_inputs.kv_cache_kernel_block_id_host_by_group.empty()
        && !py_model_inputs_.attention_inputs.kv_cache_kernel_block_id_device_by_group.empty()
        && !py_model_inputs_.attention_inputs.kv_cache_kernel_block_id_host_by_group.empty()) {
        RTP_LLM_CHECK_WITH_INFO(
            inputs.attention_inputs.kv_cache_kernel_block_id_device_by_group.size()
                == py_model_inputs_.attention_inputs.kv_cache_kernel_block_id_device_by_group.size(),
            "kv_cache_kernel_block_id_device_by_group size mismatch");
        const size_t group = inputs.attention_inputs.kv_cache_kernel_block_id_device_by_group.size();
        RTP_LLM_CHECK_WITH_INFO(inputs.attention_inputs.kv_cache_kernel_block_id_host_by_group.size() == group
                                    && py_model_inputs_.attention_inputs.kv_cache_kernel_block_id_host_by_group.size()
                                           == group,
                                "kv_cache_kernel_block_id_host_by_group size mismatch");
        for (size_t g = 0; g < group; ++g) {
            // Clear per-group block tables before copying real entries.
            // Without this, padding entries retain stale block IDs from previous calls,
            // causing linear attention (GatedDeltaNet) to corrupt SSM/conv states of stale blocks.
            clearTensorAsync(py_model_inputs_.attention_inputs.kv_cache_kernel_block_id_device_by_group[g]);
            clearTensorAsync(py_model_inputs_.attention_inputs.kv_cache_kernel_block_id_host_by_group[g]);
            copySmallerIntoLarger(inputs.attention_inputs.kv_cache_kernel_block_id_device_by_group[g],
                                  py_model_inputs_.attention_inputs.kv_cache_kernel_block_id_device_by_group[g]);
            copySmallerIntoLarger(inputs.attention_inputs.kv_cache_kernel_block_id_host_by_group[g],
                                  py_model_inputs_.attention_inputs.kv_cache_kernel_block_id_host_by_group[g]);
        }
    }
}

PyModelOutputs CudaGraphRunner::forward(const PyModelInputs& inputs, CudaGraphState& state) {
    PyModelOutputs outputs;

    // decode or embedding model only
    RTP_LLM_LOG_DEBUG("Replay Start");
    prepareInputs(inputs, state);
    if (is_prefill_cuda_graph_mode_) {
        {
            RTP_LLM_PROFILE_SCOPE("cuda_graph.forward(replayPrefill)");
            replayPrefill(state.current_real_graph_seq_len);
        }
        outputs.hidden_states =
            graph_instances_[state.current_real_graph_seq_len].mem_hold_.decoder_layer_hidden_states_.slice(
                0, 0, state.current_seq_len);
    } else {
        {
            RTP_LLM_PROFILE_SCOPE("cuda_graph.forward(replayDecode)");
            replayDecode(state.current_real_graph_bs);
        }
        outputs.hidden_states =
            graph_instances_[state.current_real_graph_bs].mem_hold_.decoder_layer_hidden_states_.slice(
                0, 0, state.seq_len_sum);
    }
    // record forward done event
    forward_event_.record(cuda_graph::graphGetCurrentStream());
    RTP_LLM_LOG_DEBUG("Replay End");
    return outputs;
}

bool CudaGraphRunner::tryGetRealGraphPrefillSeqLen(const PyModelInputs& inputs, CudaGraphState& state) {
    state.current_seq_len = inputs.attention_inputs.input_lengths.sum(0).item<int>();
    if (capture_range_.empty()) {
        RTP_LLM_LOG_WARNING("prefill cuda graph: capture_range_ is empty, cannot run");
        return false;
    }
    auto it = std::lower_bound(capture_range_.begin(), capture_range_.end(), state.current_seq_len);
    // No captured graph for seq_len >= current (all captures smaller than requested)
    if (it == capture_range_.end()) {
        RTP_LLM_LOG_WARNING("prefill seq_len %d exceeds max captured %d, fallback to normal run",
                            state.current_seq_len,
                            capture_range_.back());
        return false;
    }
    state.current_real_graph_seq_len = *it;
    state.current_batch_size         = inputs.attention_inputs.input_lengths.size(0);
    return true;
}

bool CudaGraphRunner::tryGetRealGraphDecodeBatchSize(const PyModelInputs& inputs, CudaGraphState& state) {
    int cuda_graph_bs        = inputs.attention_inputs.input_lengths.size(0);
    state.current_batch_size = cuda_graph_bs;
    RTP_LLM_LOG_DEBUG("canRun judge for batch size: %d", cuda_graph_bs);
    if (capture_range_.empty()) {
        RTP_LLM_LOG_WARNING("decode cuda graph: capture_range_ is empty, cannot run");
        return false;
    }
    auto it = std::lower_bound(capture_range_.begin(), capture_range_.end(), state.current_batch_size);
    // No captured graph for batch >= current (all captures smaller)
    if (it == capture_range_.end()) {
        RTP_LLM_LOG_WARNING("decode batch size %d exceeds max captured %d, fallback to normal run",
                            state.current_batch_size,
                            capture_range_.back());
        return false;
    }
    state.current_real_graph_bs = *it;
    RTP_LLM_LOG_DEBUG(
        "batch size used in replay: %d (graph key %d)", state.current_batch_size, state.current_real_graph_bs);

    if (inputs.attention_inputs.is_prefill) {
        state.seq_len_sum = inputs.attention_inputs.input_lengths.sum(0).item<int>();
    } else {
        state.seq_len_sum = cuda_graph_bs;
    }
    RTP_LLM_LOG_DEBUG("can run cuda graph for decode");
    return true;
}

bool CudaGraphRunner::canRun(const PyModelInputs& inputs, CudaGraphState& state) {
    RTP_LLM_PROFILE_SCOPE("cuda_graph.canRun");
    // Check if this is speculative sampling:
    // 1. prefix_lengths is not empty
    // 2. all values in input_lengths are the same
    // this is for 2.2.1
    if (is_target_verify_) {
        if (inputs.attention_inputs.is_target_verify) {
            // Target-verify must also respect captured decode range.
            // Otherwise we may replay an uncaptured graph key.
            return tryGetRealGraphDecodeBatchSize(inputs, state);
        }
        return false;
    }

    if (!enable_cuda_graph_ || (inputs.attention_inputs.is_prefill && !is_prefill_cuda_graph_mode_)) {
        return false;
    }

    if (!inputs.attention_inputs.kv_cache_kernel_block_id_device_by_group.empty()) {
        const size_t group = inputs.attention_inputs.kv_cache_kernel_block_id_device_by_group.size();
        if (kv_cache_group_num_ <= 0) {
            RTP_LLM_LOG_WARNING("Hybrid kv cache detected but kv_cache_group_num_ is not set, fallback to normal run.");
            return false;
        }
        if (group != static_cast<size_t>(kv_cache_group_num_)) {
            RTP_LLM_LOG_WARNING("Hybrid kv cache group size mismatch: inputs=%zu, captured=%d, fallback to normal run.",
                                group,
                                kv_cache_group_num_);
            return false;
        }
    }

    if (is_prefill_cuda_graph_mode_) {
        if (!tryGetRealGraphPrefillSeqLen(inputs, state)) {
            return false;
        }
        // current_real_graph_seq_len is always *it from lower_bound within capture_range_
        RTP_LLM_LOG_DEBUG("prefill cuda graph replay seq_len key %d", state.current_real_graph_seq_len);
    } else {
        if (!tryGetRealGraphDecodeBatchSize(inputs, state)) {
            return false;
        }
    }
    return true;
}

void CudaGraphRunner::initKernelInternalMemory() {
    torch::Tensor cu_seqlens =
        torch::zeros({int(max_bs_ + 1)}, torch::TensorOptions(torch::kInt32).device(torch::kCPU));
    torch::Tensor cu_kv_seqlens =
        torch::zeros({int(max_bs_ + 1)}, torch::TensorOptions(torch::kInt32).device(torch::kCPU));
    auto input_lengths  = capture_mem_hold_.py_model_inputs_.attention_inputs.input_lengths;
    auto prefix_lengths = capture_mem_hold_.py_model_inputs_.attention_inputs.prefix_lengths;

    cu_seqlens.slice(0, 1, max_bs_ + 1) = input_lengths.cumsum(0);
    if (prefix_lengths.defined()) {
        cu_kv_seqlens.slice(0, 1, max_bs_ + 1) = input_lengths.add(prefix_lengths).cumsum(0);
    }
    capture_mem_hold_.py_model_inputs_.attention_inputs.cu_seqlens    = cu_seqlens.pin_memory();
    capture_mem_hold_.py_model_inputs_.attention_inputs.cu_kv_seqlens = cu_kv_seqlens.pin_memory();
}

int CudaGraphRunner::getCurrentRealGraphBs(const CudaGraphState& state) const {
    return state.current_real_graph_bs;
}

void CudaGraphRunner::initCaptureAttentionInputs(PyModelInputs& inputs, int max_bs, int num_tokens_per_bs) {
    inputs.attention_inputs.is_target_verify = is_target_verify_;
    inputs.attention_inputs.is_prefill       = is_prefill_cuda_graph_mode_ || num_tokens_per_bs_ > 1;

    // input_ids [tokens_nums] = [batch_size * num_tokens_per_bs]
    inputs.input_ids = torch::zeros({max_num_token_}, options_cuda_int32_);
    // input_lengths [batch_size, int32] (decode only)
    inputs.attention_inputs.input_lengths = torch::full({int(max_bs_)}, num_tokens_per_bs_, options_cpu_int32_);
    inputs.attention_inputs.input_lengths = inputs.attention_inputs.input_lengths.pin_memory();
    // input_lengths_d: device copy for Triton kernels (spec-decode prepare_cuda_graph)
    inputs.attention_inputs.input_lengths_d = torch::full({int(max_bs_)}, num_tokens_per_bs_, options_cuda_int32_);
    // sequence_lengths [batch_size, int32] (decode only)
    // sequence_length should in pinned memory
    inputs.attention_inputs.sequence_lengths = torch::ones({int(max_bs_)}, options_cpu_int32_);
    inputs.attention_inputs.sequence_lengths.fill_(max_seq_len_ - num_tokens_per_bs - 1);
    inputs.attention_inputs.sequence_lengths = inputs.attention_inputs.sequence_lengths.pin_memory();

    const int64_t max_kv_blocks =
        static_cast<int64_t>(((max_seq_len_ + seq_size_per_block_ - 1) / seq_size_per_block_) + sp_steps_);
    const int64_t max_blocks = max_kv_blocks * seq_size_per_block_ / kernel_seq_size_per_block_;
    // kv_cache_kernel_block_id_device [batch_size, block_num]
    inputs.attention_inputs.kv_cache_kernel_block_id_device =
        torch::zeros({int(max_bs_), max_blocks}, options_cuda_int32_);

    inputs.attention_inputs.kv_cache_kernel_block_id_host =
        torch::zeros({int(max_bs_), max_blocks}, options_cpu_int32_).pin_memory();

    auto layer_num = kv_cache_layer_to_group_.size();
    if (layer_num > 0) {
        auto kv_cache_layer_to_group_capture_ =
            torch::empty({static_cast<int64_t>(layer_num)}, options_cpu_int32_).pin_memory();
        auto* dst = kv_cache_layer_to_group_capture_.data_ptr<int32_t>();
        for (size_t i = 0; i < layer_num; ++i) {
            dst[i] = static_cast<int32_t>(kv_cache_layer_to_group_[i]);
        }

        // [layer_num] int32, pinned host tensor. Keep empty when not provided.
        inputs.attention_inputs.kv_cache_layer_to_group = kv_cache_layer_to_group_capture_;
    }

    // Hybrid cache: per-group block tables.
    inputs.attention_inputs.kv_cache_kernel_block_id_device_by_group.clear();
    inputs.attention_inputs.kv_cache_kernel_block_id_host_by_group.clear();
    if (kv_cache_group_num_ > 1) {
        inputs.attention_inputs.kv_cache_kernel_block_id_device_by_group.reserve(kv_cache_group_num_);
        inputs.attention_inputs.kv_cache_kernel_block_id_host_by_group.reserve(kv_cache_group_num_);
        for (int g = 0; g < kv_cache_group_num_; ++g) {
            inputs.attention_inputs.kv_cache_kernel_block_id_device_by_group.push_back(
                torch::zeros({int(max_bs_), max_blocks}, options_cuda_int32_));
            inputs.attention_inputs.kv_cache_kernel_block_id_host_by_group.push_back(
                torch::zeros({int(max_bs_), max_blocks}, options_cpu_int32_).pin_memory());
        }
    }

    // prefix_lengths [batch_size, int32] (for attention `prepare`)
    if (num_tokens_per_bs_ > 1 && !is_prefill_cuda_graph_mode_) {
        inputs.attention_inputs.prefix_lengths =
            torch::full({int(max_bs_)}, max_seq_len_ - num_tokens_per_bs_, options_cpu_int32_).pin_memory();
        // prefix_lengths_d: device copy for Triton kernels (spec-decode prepare_cuda_graph)
        inputs.attention_inputs.prefix_lengths_d =
            torch::full({int(max_bs_)}, max_seq_len_ - num_tokens_per_bs_, options_cuda_int32_);
    } else if (is_prefill_cuda_graph_mode_) {
        inputs.attention_inputs.prefix_lengths = torch::zeros({int(max_bs_)}, options_cpu_int32_).pin_memory();
        // prefix_lengths_d: device copy for Triton kernels (prefill prepare_cuda_graph)
        inputs.attention_inputs.prefix_lengths_d = torch::zeros({int(max_bs_)}, options_cuda_int32_);
    }
    // padding_offset [max_num_token_, int32] (for attention padding)
    inputs.attention_inputs.padding_offset = torch::zeros({int(max_seq_len_ * max_bs_)}, options_cpu_int32_);
    inputs.attention_inputs.padding_offset = inputs.attention_inputs.padding_offset.pin_memory();
    inputs.attention_inputs.dtype          = model_data_type_;
    inputs.attention_inputs.is_s_padded    = true;
    // Initialize to 1, not 0: SSM/linear-attention kernels (Qwen3-Next GatedDeltaNet)
    // compute `index = value - 1`; a value of 0 wraps to ~4.3B on unsigned indexing → OOB.
    // The initial forward in initCapture() (output-dtype probe) runs with this tensor,
    // so it must be safe before capturePrefill() fills it with the correct prefix_len+1.
    inputs.attention_inputs.sequence_lengths_plus_1_d = torch::ones({int(max_bs_)}, options_cuda_int32_);
    // Pre-allocate stable device buffers for prefix_lengths_d / input_lengths_d.
    // These are consumed (outside the captured graph) by Triton kernels
    // `_prepare_cg_spec_decode_kernel` and `_prepare_cg_prefill_kernel` which
    // iterate over the full captured batch size. `prepareInputs` will copy
    // live request lengths into these buffers and zero padding rows before
    // calling `prepare_cuda_graph`. Without these stable buffers, the kernels
    // would dereference stale/undefined device pointers from capture time and
    // produce wrong KV cache offsets, causing the well-known target_verify /
    // draft_prefill CUDA graph corruption under concurrency.
    inputs.attention_inputs.prefix_lengths_d = torch::zeros({int(max_bs_)}, options_cuda_int32_);
    inputs.attention_inputs.input_lengths_d  = torch::zeros({int(max_bs_)}, options_cuda_int32_);
    // decode_cu_seqlens_d: cumulative *query token* counts per batch (ragged qo_indptr).
    // For MTP target verify, each batch row has num_tokens_per_bs draft tokens, so use
    // [0, step, 2*step, ..., batch*step] instead of [0,1,...,batch] (single-token decode).
    if (is_target_verify_ && num_tokens_per_bs > 1) {
        inputs.attention_inputs.decode_cu_seqlens_d =
            torch::arange(0,
                          static_cast<int>((max_bs_ + 1) * num_tokens_per_bs),
                          num_tokens_per_bs,
                          torch::TensorOptions(torch::kInt32).device(torch::kCUDA));
    } else {
        inputs.attention_inputs.decode_cu_seqlens_d =
            torch::arange(0, max_bs_ + 1, 1, torch::TensorOptions(torch::kInt32).device(torch::kCUDA));
    }
}

void CudaGraphRunner::initCaptureAttentionInputsPost() {
    auto&         inputs                        = capture_mem_hold_.py_model_inputs_;
    torch::Tensor cuda_graph_prefill_batch_size = torch::zeros({1}, options_cpu_int32_).pin_memory();
    // as one batch to capture
    cuda_graph_prefill_batch_size.fill_(1);
    RTP_LLM_CHECK_WITH_INFO(cuda_graph_prefill_batch_size.is_pinned(),
                            "capture_mem_hold_ cuda_graph_prefill_batch_size is not pinned memory");

    // draft model prefill but not embedding model
    if (num_tokens_per_bs_ > 1 && num_tokens_per_bs_ != max_seq_len_) {
        inputs.attention_inputs.prefill_cuda_graph_copy_params =
            PyPrefillCudaGaphCopyParams{cuda_graph_prefill_batch_size, num_tokens_per_bs_, int(max_bs_)};
    } else {
        inputs.attention_inputs.prefill_cuda_graph_copy_params =
            PyPrefillCudaGaphCopyParams{cuda_graph_prefill_batch_size, max_seq_len_, int(max_bs_)};
    }
}

void CudaGraphRunner::setPositionEncoding(torch::Tensor position_encoding) {
    position_encoding_ = position_encoding;
}

void CudaGraphRunner::setTokenTypeEmbedding(torch::Tensor token_type_embedding) {
    token_type_embedding_ = token_type_embedding;
}

void CudaGraphRunner::setInputEmbeddingScalar(float input_embedding_scalar) {
    input_embedding_scalar_ = input_embedding_scalar;
}

void CudaGraphRunner::initCaptureBertEmbeddingInputs(PyModelInputs& inputs, int max_bs, int max_num_token) {
    auto options_cuda_int32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false);
    // Initialize BertEmbeddingInputs for capture
    // combo_position_ids: empty tensor for capture (will be filled during actual forward)
    inputs.bert_embedding_inputs.combo_position_ids = torch::zeros({max_seq_len_ * max_bs}, options_cuda_int32);

    // position_encoding: from weights
    inputs.bert_embedding_inputs.position_encoding = position_encoding_;

    // combo_tokens_type_ids: empty tensor for capture (will be filled during actual forward)
    inputs.bert_embedding_inputs.combo_tokens_type_ids = torch::zeros({max_seq_len_ * max_bs}, options_cuda_int32);

    // token_type_embedding: from weights
    inputs.bert_embedding_inputs.token_type_embedding = token_type_embedding_;

    // input_embedding_scalar: fixed value
    inputs.bert_embedding_inputs.input_embedding_scalar = input_embedding_scalar_;
}

void CudaGraphRunner::logCudaGraphPoolMemory(const char* phase) {
    size_t free_bytes  = 0;
    size_t total_bytes = 0;
    cuda_graph::graphMemGetInfo(&free_bytes, &total_bytes);
    const size_t used_bytes        = total_bytes - free_bytes;
    const size_t pytorch_allocated = cuda_graph::graphAllocatedBytes();
    const size_t pytorch_reserved  = cuda_graph::graphReservedBytes();
    const size_t pool_overhead     = pytorch_reserved > pytorch_allocated ? pytorch_reserved - pytorch_allocated : 0;

    RTP_LLM_LOG_INFO("[CudaGraph Memory][%s] cudaMemGetInfo: used=%zu MiB, free=%zu MiB, total=%zu MiB | "
                     "PyTorch: allocated=%zu MiB, reserved=%zu MiB, pool_overhead=%zu MiB",
                     phase,
                     used_bytes / 1024 / 1024,
                     free_bytes / 1024 / 1024,
                     total_bytes / 1024 / 1024,
                     pytorch_allocated / 1024 / 1024,
                     pytorch_reserved / 1024 / 1024,
                     pool_overhead / 1024 / 1024);
}

void CudaGraphRunner::initCapture() {
    if (enable_cuda_graph_) {
        RTP_LLM_LOG_INFO("CUDA graph capture is enabled");
        shared_graph_pool_ = cuda_graph::graphPoolHandle();
        if (is_prefill_cuda_graph_mode_) {
            RTP_LLM_LOG_INFO("CUDA graph capture for prefill, num_tokens_per_bs_: %d", num_tokens_per_bs_);
        }
        max_num_token_ = max_bs_ * num_tokens_per_bs_;
        if (is_prefill_cuda_graph_mode_) {
            capture_range_ = getPrefillSequenceLengthsToCapture();
        } else {
            capture_range_ = getDecodeBatchSizesToCapture();
        }

        PyModelInputs inputs;
        // input_ids [tokens_nums] = [batch_size * num_tokens_per_bs]
        inputs.input_ids     = torch::zeros({max_num_token_}, options_cuda_int32_);
        inputs.input_hiddens = torch::zeros({max_num_token_, hidden_size_}, options_cuda_float_);
        // Setup attention inputs using the extracted function
        initCaptureAttentionInputs(inputs, max_bs_, num_tokens_per_bs_);

        // Setup BertEmbedding inputs using the extracted function
        initCaptureBertEmbeddingInputs(inputs, max_bs_, max_num_token_);

        torch::Tensor output;
        capture_mem_hold_ = CaptureMemoryHold(output, inputs, is_prefill_cuda_graph_mode_);
        initKernelInternalMemory();

        // get real output data type (params already prepared in attn impl __init__/create_params)
        auto attn_pyobj = py_attn_pyobj_method_(capture_mem_hold_.py_model_inputs_, true);
        RTP_LLM_LOG_INFO("initCapture forward for output datatype start");
        try {
            py_forward_method_(capture_mem_hold_.py_model_inputs_, attn_pyobj);
        } catch (const pybind11::error_already_set& e) {
            RTP_LLM_LOG_ERROR("initCapture forward (output datatype) python exception: %s", e.what());
            throw;
        } catch (const std::exception& e) {
            RTP_LLM_LOG_ERROR("initCapture forward (output datatype) std exception: %s", e.what());
            throw;
        }
        RTP_LLM_LOG_INFO("initCapture forward for output datatype end");
        output = torch::zeros({max_num_token_, hidden_size_}, options_cuda_float_);
        capture_mem_hold_.setHiddenStates(output);
        initCaptureAttentionInputsPost();
        logCudaGraphPoolMemory("before_capture");

        if (is_prefill_cuda_graph_mode_) {
            RTP_LLM_LOG_INFO("initCapture forward post check start for prefill");
            capture_mem_hold_.py_model_inputs_.attention_inputs.cu_seqlens.data_ptr<int>()[1]    = max_num_token_;
            capture_mem_hold_.py_model_inputs_.attention_inputs.cu_kv_seqlens.data_ptr<int>()[1] = max_num_token_;
            capture_mem_hold_.py_model_inputs_.attention_inputs.input_lengths.data_ptr<int>()[0] = max_num_token_;
            PyModelInputs inputs = capture_mem_hold_.py_model_inputs_;
            inputs.attention_inputs.cu_seqlens =
                capture_mem_hold_.py_model_inputs_.attention_inputs.cu_seqlens.slice(0, 0, 2);
            inputs.attention_inputs.cu_kv_seqlens =
                capture_mem_hold_.py_model_inputs_.attention_inputs.cu_kv_seqlens.slice(0, 0, 2);
            inputs.attention_inputs.input_lengths =
                capture_mem_hold_.py_model_inputs_.attention_inputs.input_lengths.slice(0, 0, 1);
            if (capture_mem_hold_.py_model_inputs_.attention_inputs.prefix_lengths.defined()) {
                inputs.attention_inputs.prefix_lengths =
                    capture_mem_hold_.py_model_inputs_.attention_inputs.prefix_lengths.slice(0, 0, 1);
            }
            if (capture_mem_hold_.py_model_inputs_.attention_inputs.prefix_lengths_d.defined()) {
                inputs.attention_inputs.prefix_lengths_d =
                    capture_mem_hold_.py_model_inputs_.attention_inputs.prefix_lengths_d.slice(0, 0, 1);
            }
            inputs.attention_inputs.kv_cache_kernel_block_id_device =
                capture_mem_hold_.py_model_inputs_.attention_inputs.kv_cache_kernel_block_id_device.slice(0, 0, 1);
            inputs.attention_inputs.kv_cache_kernel_block_id_host =
                capture_mem_hold_.py_model_inputs_.attention_inputs.kv_cache_kernel_block_id_host.slice(0, 0, 1);
            py_forward_method_(inputs);
            RTP_LLM_LOG_INFO("initCapture forward post check end for prefill");
            capturePrefill();
        } else {
            captureDecode();
        }
        logCudaGraphPoolMemory("after_capture");
    } else {
        initKernelInternalMemory();
        RTP_LLM_LOG_INFO("CUDA graph capture is not enabled, skipping initialization");
    }
}

void CudaGraphRunner::replayGraph(int key) {
    graph_instances_[key].graph_.replay();
}

void CudaGraphRunner::captureOneGraphInstance(int key, const char* key_type) {
    auto inputs = graph_instances_[key].mem_hold_.py_model_inputs_;

    size_t pre_capture_reserved = cuda_graph::graphReservedBytes();

    // WarmUp twice (params already prepared in attn impl __init__/create_params when instance was created)
    RTP_LLM_LOG_INFO("WarmUp for %s %d start.", key_type, key);
    auto attn_pyobj = graph_instances_[key].mem_hold_.attn_pyobj_;
    try {
        py_forward_method_(inputs, attn_pyobj);
        py_forward_method_(inputs, attn_pyobj);
    } catch (const py::error_already_set& e) {
        RTP_LLM_LOG_ERROR("WarmUp forward failed for %s %d: %s", key_type, key, e.what());
        throw;
    }
    RTP_LLM_LOG_INFO("WarmUp for %s %d successfully.", key_type, key);

    {
        // sync before capture
        cuda_graph::graphDeviceSynchronize();

        CudaGraphStreamLife stream_life(capture_stream_);
        auto&               graph               = graph_instances_[key].graph_;
        std::string         output_dot_filename = "";
        if (enable_cuda_graph_debug_mode_) {
            graph.enable_debug_mode();
            std::string key_type_str = std::string(key_type);
            std::replace(key_type_str.begin(), key_type_str.end(), ' ', '_');
            output_dot_filename = "cuda_graph_tokens" + std::to_string(num_tokens_per_bs_) + "_" + key_type_str + "_"
                                  + std::to_string(key) + "_visualization.dot";
            RTP_LLM_LOG_INFO("CUDA Graph debug mode enabled, output file: %s", output_dot_filename.c_str());
        }
        RTP_LLM_LOG_INFO("Capture for %s %d begin.", key_type, key);
        PyModelOutputs outputs;
        {
            cuda_graph::graphCaptureBegin(graph, shared_graph_pool_);
            CudaGraphCaptureGuard capture_guard;
            try {
                auto py_outputs_obj = py_forward_method_(inputs, attn_pyobj);
                outputs             = py_outputs_obj.cast<PyModelOutputs>();
            } catch (const py::error_already_set& e) {
                RTP_LLM_LOG_ERROR("Capture forward failed for %s %d: %s", key_type, key, e.what());
                // Format the python traceback so the actual offending Python
                // call site is visible (otherwise pybind only surfaces the
                // top-level error message, which for cudaErrorStreamCapture-
                // Unsupported is unhelpful).
                try {
                    py::object  tb_mod    = py::module::import("traceback");
                    py::object  trace_str = tb_mod.attr("format_exception")(e.type(), e.value(), e.trace());
                    std::string full_tb;
                    for (auto item : trace_str) {
                        full_tb += py::str(item).cast<std::string>();
                    }
                    RTP_LLM_LOG_ERROR("Capture forward python traceback:\n%s", full_tb.c_str());
                } catch (...) {
                    RTP_LLM_LOG_ERROR("Capture forward: failed to format python traceback");
                }
                throw;
            }
            graph_instances_[key].mem_hold_.decoder_layer_hidden_states_.copy_(outputs.hidden_states);
            graph.capture_end();
        }

        if (enable_cuda_graph_debug_mode_) {
            RTP_LLM_LOG_INFO("Calling debug_dump to generate: %s", output_dot_filename.c_str());
            graph.debug_dump(output_dot_filename.c_str());
            RTP_LLM_LOG_INFO("debug_dump completed for: %s", output_dot_filename.c_str());
        }

        size_t post_capture_reserved = cuda_graph::graphReservedBytes();
        size_t graph_pool_delta =
            post_capture_reserved > pre_capture_reserved ? post_capture_reserved - pre_capture_reserved : 0;
        RTP_LLM_LOG_INFO("[CudaGraph Memory] captured %s %d: pool_delta=%zu MiB, total_reserved=%zu MiB",
                         key_type,
                         key,
                         graph_pool_delta / 1024 / 1024,
                         post_capture_reserved / 1024 / 1024);
    }
}

void CudaGraphRunner::replayAndSyncCheck(int key, const char* key_type) {
    RTP_LLM_LOG_INFO("replay start check for %s %d", key_type, key);
    replayGraph(key);
    cuda_graph::graphDeviceSynchronize();
    RTP_LLM_LOG_INFO("replay end check for %s %d", key_type, key);
}

void CudaGraphRunner::prepareCaptureInputs(PyModelInputs& inputs, int batch_size, int seq_len_or_tokens) {
    // Common slice operations for input_ids and padding_offset
    inputs.attention_inputs.is_prefill       = is_prefill_cuda_graph_mode_ || num_tokens_per_bs_ > 1;
    inputs.attention_inputs.is_target_verify = is_target_verify_;
    inputs.input_ids     = capture_mem_hold_.py_model_inputs_.input_ids.slice(0, 0, seq_len_or_tokens);
    inputs.input_hiddens = capture_mem_hold_.py_model_inputs_.input_hiddens.slice(0, 0, seq_len_or_tokens);
    inputs.attention_inputs.input_lengths =
        capture_mem_hold_.py_model_inputs_.attention_inputs.input_lengths.slice(0, 0, batch_size);
    if (capture_mem_hold_.py_model_inputs_.attention_inputs.input_lengths_d.defined()) {
        inputs.attention_inputs.input_lengths_d =
            capture_mem_hold_.py_model_inputs_.attention_inputs.input_lengths_d.slice(0, 0, batch_size);
    }
    inputs.attention_inputs.padding_offset =
        capture_mem_hold_.py_model_inputs_.attention_inputs.padding_offset.slice(0, 0, seq_len_or_tokens);

    // Common slice operations for attention inputs
    if (capture_mem_hold_.py_model_inputs_.attention_inputs.prefix_lengths.defined()) {
        inputs.attention_inputs.prefix_lengths =
            capture_mem_hold_.py_model_inputs_.attention_inputs.prefix_lengths.slice(0, 0, batch_size);
    }
    if (capture_mem_hold_.py_model_inputs_.attention_inputs.prefix_lengths_d.defined()) {
        inputs.attention_inputs.prefix_lengths_d =
            capture_mem_hold_.py_model_inputs_.attention_inputs.prefix_lengths_d.slice(0, 0, batch_size);
    }
    inputs.attention_inputs.sequence_lengths =
        capture_mem_hold_.py_model_inputs_.attention_inputs.sequence_lengths.slice(0, 0, batch_size);

    inputs.attention_inputs.kv_cache_kernel_block_id_device =
        capture_mem_hold_.py_model_inputs_.attention_inputs.kv_cache_kernel_block_id_device.slice(0, 0, batch_size);
    inputs.attention_inputs.kv_cache_kernel_block_id_host =
        capture_mem_hold_.py_model_inputs_.attention_inputs.kv_cache_kernel_block_id_host.slice(0, 0, batch_size);
    inputs.attention_inputs.kv_cache_block_id_device =
        capture_mem_hold_.py_model_inputs_.attention_inputs.kv_cache_block_id_device.defined() ?
            capture_mem_hold_.py_model_inputs_.attention_inputs.kv_cache_block_id_device.slice(0, 0, batch_size) :
            torch::Tensor();
    inputs.attention_inputs.kv_cache_block_id_host =
        capture_mem_hold_.py_model_inputs_.attention_inputs.kv_cache_block_id_host.defined() ?
            capture_mem_hold_.py_model_inputs_.attention_inputs.kv_cache_block_id_host.slice(0, 0, batch_size) :
            torch::Tensor();
    inputs.attention_inputs.cu_seqlens =
        capture_mem_hold_.py_model_inputs_.attention_inputs.cu_seqlens.slice(0, 0, batch_size + 1);
    inputs.attention_inputs.cu_kv_seqlens =
        capture_mem_hold_.py_model_inputs_.attention_inputs.cu_kv_seqlens.slice(0, 0, batch_size + 1);
    inputs.attention_inputs.decode_cu_seqlens_d =
        capture_mem_hold_.py_model_inputs_.attention_inputs.decode_cu_seqlens_d.slice(0, 0, batch_size + 1);
    inputs.attention_inputs.sequence_lengths_plus_1_d =
        capture_mem_hold_.py_model_inputs_.attention_inputs.sequence_lengths_plus_1_d.slice(0, 0, batch_size);
    // Propagate the stable captured device buffers for prefix_lengths_d /
    // input_lengths_d to each per-instance py_model_inputs_. These are read by
    // `prepare_cuda_graph` (called outside the captured graph in prepareInputs)
    // to fill `_cg.seq_lens` for target_verify and draft_prefill paths.
    if (capture_mem_hold_.py_model_inputs_.attention_inputs.prefix_lengths_d.defined()) {
        inputs.attention_inputs.prefix_lengths_d = capture_mem_hold_.py_model_inputs_.attention_inputs.prefix_lengths_d;
    }
    if (capture_mem_hold_.py_model_inputs_.attention_inputs.input_lengths_d.defined()) {
        inputs.attention_inputs.input_lengths_d = capture_mem_hold_.py_model_inputs_.attention_inputs.input_lengths_d;
    }

    const auto& cap_attn = capture_mem_hold_.py_model_inputs_.attention_inputs;
    inputs.attention_inputs.kv_cache_kernel_block_id_device_by_group.clear();
    inputs.attention_inputs.kv_cache_kernel_block_id_host_by_group.clear();
    if (!cap_attn.kv_cache_kernel_block_id_device_by_group.empty()
        && !cap_attn.kv_cache_kernel_block_id_host_by_group.empty()) {
        const size_t group = cap_attn.kv_cache_kernel_block_id_device_by_group.size();
        inputs.attention_inputs.kv_cache_kernel_block_id_device_by_group.reserve(group);
        inputs.attention_inputs.kv_cache_kernel_block_id_host_by_group.reserve(group);
        for (size_t g = 0; g < group; ++g) {
            inputs.attention_inputs.kv_cache_kernel_block_id_device_by_group.push_back(
                cap_attn.kv_cache_kernel_block_id_device_by_group[g].slice(0, 0, batch_size));
            inputs.attention_inputs.kv_cache_kernel_block_id_host_by_group.push_back(
                cap_attn.kv_cache_kernel_block_id_host_by_group[g].slice(0, 0, batch_size));
        }
    }

    // Common direct assignments (no slice needed)
    inputs.attention_inputs.dtype = capture_mem_hold_.py_model_inputs_.attention_inputs.dtype;
    inputs.attention_inputs.kv_cache_layer_to_group =
        capture_mem_hold_.py_model_inputs_.attention_inputs.kv_cache_layer_to_group;
    inputs.bert_embedding_inputs        = capture_mem_hold_.py_model_inputs_.bert_embedding_inputs;
    inputs.attention_inputs.is_s_padded = true;
}

CaptureMemoryHold CudaGraphRunner::createCaptureMemoryHold(PyModelInputs& inputs, int tokens_count) {
    // only when prefill or target model score phase, the num_tokens_per_bs_ > 1
    return CaptureMemoryHold(capture_mem_hold_.decoder_layer_hidden_states_.slice(0, 0, tokens_count),
                             inputs,
                             is_prefill_cuda_graph_mode_ || num_tokens_per_bs_ > 1);
}

CudaGraphRunner* CudaGraphRunner::createForPrefill(py::object py_instance, GraphParams params) {
    params.enable_cuda_graph = true;
    if (params.num_tokens_per_bs == 0) {
        params.num_tokens_per_bs = params.max_seq_len;
    }
    CudaGraphRunner* runner = new CudaGraphRunner(params, std::move(py_instance));
    runner->initCapture();
    return runner;
}

CudaGraphRunner* CudaGraphRunner::createForDecode(py::object py_instance, GraphParams params) {
    params.enable_cuda_graph = true;
    if (params.num_tokens_per_bs == 0) {
        params.num_tokens_per_bs = 1;
    }
    CudaGraphRunner* runner = new CudaGraphRunner(params, std::move(py_instance));
    runner->initCapture();
    return runner;
}

}  // namespace rtp_llm
