#include "rtp_llm/cpp/cuda_graph/cuda_graph_runner.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include "rtp_llm/cpp/cuda_graph/cuda_graph_device_shims.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#include "torch/csrc/autograd/generated/variable_factories.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
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
// clang-format on

// Helper function for optimized tensor copy using async operations with current CUDA stream
void optimizedCopyAsync(const torch::Tensor& src, torch::Tensor& dst, size_t size) {
    if (!src.defined() || !dst.defined() || src.numel() <= 0) {
        return;
    }

    RTP_LLM_PROFILE_SCOPE("optimizedCopyAsync");

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

    const size_t graph_idx =
        is_prefill_cuda_graph_mode_ ? state.current_real_graph_seq_len : state.current_real_graph_bs;
    auto& py_model_inputs_ = graph_instances_[graph_idx].mem_hold_.py_model_inputs_;
    auto  attn_pyobj       = graph_instances_[graph_idx].mem_hold_.attn_pyobj_;

    // Per-launch capacity contract: see fuse_copy_util.h sizing rationale.
    // Worst case here is ~8 contiguous + (1 + group_count) strided copies,
    // batched into one launch each. If new copies are added below — or if the
    // hybrid KV-cache group_count grows materially — re-check MAX_FUSED_*_COPIES.
    FusedD2DCopyParams     d2d_copies;
    FusedStridedCopyParams strided_d2d_copies;

    auto tryAddD2DCopy = [&d2d_copies](const torch::Tensor& src, torch::Tensor& dst, size_t bytes) {
        if (src.defined() && src.numel() > 0) {
            d2d_copies.add(src.data_ptr(), dst.data_ptr(), bytes);
        }
    };

    // Collect a strided 2D D2D copy: copies src[0..rows, 0..cols] into dst[0..rows, 0..cols]
    // where src and dst may have different column strides (copySmallerIntoLarger semantics).
    // For 1D tensors, falls back to a contiguous D2D copy to avoid silent data loss.
    auto tryAddStridedD2DCopy = [&strided_d2d_copies, &d2d_copies](const torch::Tensor& src, torch::Tensor& dst) {
        if (!src.defined() || src.numel() <= 0)
            return;
        if (src.dim() < 2) {
            d2d_copies.add(src.data_ptr(), dst.data_ptr(), src.numel() * src.element_size());
            return;
        }
        strided_d2d_copies.add(src.data_ptr(),
                               dst.data_ptr(),
                               src.size(0),
                               src.size(1) * src.element_size(),
                               src.stride(0) * src.element_size(),
                               dst.stride(0) * dst.element_size());
    };

    // H2H strided 2D copy via row-by-row memcpy (cannot use GPU kernel for host memory).
    // For 1D tensors, falls back to a contiguous memcpy.
    auto stridedCopyHost = [](const torch::Tensor& src, torch::Tensor& dst) {
        if (!src.defined() || src.numel() <= 0)
            return;
        RTP_LLM_PROFILE_SCOPE("stridedCopyHost");
        if (src.dim() < 2) {
            memcpy(dst.data_ptr(), src.data_ptr(), src.numel() * src.element_size());
            return;
        }
        const size_t nrows      = src.size(0);
        const size_t row_bytes  = src.size(1) * src.element_size();
        const size_t src_stride = src.stride(0) * src.element_size();
        const size_t dst_stride = dst.stride(0) * dst.element_size();
        const char*  src_ptr    = reinterpret_cast<const char*>(src.data_ptr());
        char*        dst_ptr    = reinterpret_cast<char*>(dst.data_ptr());
        for (size_t r = 0; r < nrows; ++r) {
            memcpy(dst_ptr + r * dst_stride, src_ptr + r * src_stride, row_bytes);
        }
    };

    // clear kv_cache_kernel_block_id_device, otherwise it will cause the cache block pollution
    py_model_inputs_.attention_inputs.kv_cache_kernel_block_id_device.fill_(0);
    py_model_inputs_.attention_inputs.kv_cache_kernel_block_id_host.fill_(0);

    // NOTE: kv_cache_block_id_{host,device} are physical block IDs dedicated for cache store
    // (see OpDefs.h). They are NOT consumed by any GPU attention kernel during CUDA graph replay;
    // attention kernels only use kv_cache_kernel_block_id_{host,device}. Cache store operations
    // run outside the CUDA graph and read from the original (non-graph) inputs directly.

    // Common device copy
    int token_num = isPrefillCudaGraph() ? state.current_seq_len : inputs.input_ids.size(0);

    tryAddD2DCopy(inputs.input_ids, py_model_inputs_.input_ids, token_num * sizeof(int));
    tryAddD2DCopy(inputs.input_hiddens,
                  py_model_inputs_.input_hiddens,
                  inputs.input_hiddens.numel() * inputs.input_hiddens.element_size());

    const int captured_token_num =
        isPrefillCudaGraph() ? state.current_real_graph_seq_len : state.current_real_graph_bs * num_tokens_per_bs_;
    if (token_num < captured_token_num) {
        const size_t padded_tokens = captured_token_num - token_num;
        d2d_copies.add(zero_input_ids_.data_ptr(),
                       static_cast<char*>(py_model_inputs_.input_ids.data_ptr())
                           + token_num * py_model_inputs_.input_ids.element_size(),
                       padded_tokens * py_model_inputs_.input_ids.element_size());
        d2d_copies.add(zero_input_hiddens_.data_ptr(),
                       static_cast<char*>(py_model_inputs_.input_hiddens.data_ptr())
                           + token_num * hidden_size_ * py_model_inputs_.input_hiddens.element_size(),
                       padded_tokens * hidden_size_ * py_model_inputs_.input_hiddens.element_size());
    }

    tryAddD2DCopy(inputs.attention_inputs.cu_seqlens,
                  py_model_inputs_.attention_inputs.cu_seqlens,
                  (state.current_batch_size + 1) * sizeof(int));
    tryAddD2DCopy(inputs.attention_inputs.cu_kv_seqlens,
                  py_model_inputs_.attention_inputs.cu_kv_seqlens,
                  (state.current_batch_size + 1) * sizeof(int));
    tryAddD2DCopy(inputs.attention_inputs.input_lengths_d,
                  py_model_inputs_.attention_inputs.input_lengths_d,
                  state.current_batch_size * sizeof(int));

    const bool has_hybrid_cache = !inputs.attention_inputs.kv_cache_kernel_block_id_device_by_group.empty()
                                  && !inputs.attention_inputs.kv_cache_kernel_block_id_host_by_group.empty()
                                  && !py_model_inputs_.attention_inputs.kv_cache_kernel_block_id_device_by_group.empty()
                                  && !py_model_inputs_.attention_inputs.kv_cache_kernel_block_id_host_by_group.empty();
    if (!has_hybrid_cache) {
        tryAddStridedD2DCopy(inputs.attention_inputs.kv_cache_kernel_block_id_device,
                             py_model_inputs_.attention_inputs.kv_cache_kernel_block_id_device);
    }

    if (!isPrefillCudaGraph()) {
        tryAddD2DCopy(inputs.attention_inputs.prefix_lengths_d,
                      py_model_inputs_.attention_inputs.prefix_lengths_d,
                      state.current_batch_size * sizeof(int));
        tryAddD2DCopy(inputs.attention_inputs.sequence_lengths_plus_1_d,
                      py_model_inputs_.attention_inputs.sequence_lengths_plus_1_d,
                      state.current_batch_size * sizeof(int));
        tryAddD2DCopy(inputs.attention_inputs.decode_cu_seqlens_d,
                      py_model_inputs_.attention_inputs.decode_cu_seqlens_d,
                      (state.current_batch_size + 1) * sizeof(int));
    } else {
        tryAddD2DCopy(inputs.attention_inputs.prefix_lengths_d,
                      py_model_inputs_.attention_inputs.prefix_lengths_d,
                      state.current_batch_size * sizeof(int));
        tryAddD2DCopy(inputs.bert_embedding_inputs.combo_position_ids,
                      py_model_inputs_.bert_embedding_inputs.combo_position_ids,
                      state.current_seq_len * sizeof(int));
        tryAddD2DCopy(inputs.bert_embedding_inputs.combo_tokens_type_ids,
                      py_model_inputs_.bert_embedding_inputs.combo_tokens_type_ids,
                      state.current_seq_len * sizeof(int));
        if (token_num < captured_token_num) {
            const size_t padded_bytes = (captured_token_num - token_num) * sizeof(int);
            if (py_model_inputs_.bert_embedding_inputs.combo_position_ids.defined()) {
                d2d_copies.add(zero_input_ids_.data_ptr(),
                               py_model_inputs_.bert_embedding_inputs.combo_position_ids.data_ptr<int>() + token_num,
                               padded_bytes);
            }
            if (py_model_inputs_.bert_embedding_inputs.combo_tokens_type_ids.defined()) {
                d2d_copies.add(zero_input_ids_.data_ptr(),
                               py_model_inputs_.bert_embedding_inputs.combo_tokens_type_ids.data_ptr<int>() + token_num,
                               padded_bytes);
            }
        }
    }

    // Hybrid cache: collect per-group D2D strided copies
    size_t hybrid_cache_group = 0;

    if (has_hybrid_cache) {
        RTP_LLM_CHECK_WITH_INFO(
            inputs.attention_inputs.kv_cache_kernel_block_id_device_by_group.size()
                == py_model_inputs_.attention_inputs.kv_cache_kernel_block_id_device_by_group.size(),
            "kv_cache_kernel_block_id_device_by_group size mismatch");
        hybrid_cache_group = inputs.attention_inputs.kv_cache_kernel_block_id_device_by_group.size();
        RTP_LLM_CHECK_WITH_INFO(inputs.attention_inputs.kv_cache_kernel_block_id_host_by_group.size()
                                        == hybrid_cache_group
                                    && py_model_inputs_.attention_inputs.kv_cache_kernel_block_id_host_by_group.size()
                                           == hybrid_cache_group,
                                "kv_cache_kernel_block_id_host_by_group size mismatch");
        for (size_t g = 0; g < hybrid_cache_group; ++g) {
            py_model_inputs_.attention_inputs.kv_cache_kernel_block_id_device_by_group[g].fill_(0);
            py_model_inputs_.attention_inputs.kv_cache_kernel_block_id_host_by_group[g].fill_(0);
            tryAddStridedD2DCopy(inputs.attention_inputs.kv_cache_kernel_block_id_device_by_group[g],
                                 py_model_inputs_.attention_inputs.kv_cache_kernel_block_id_device_by_group[g]);
        }
    }

    // Launch ALL D2D copies (contiguous + strided) in two fused kernels
    fusedCopy(d2d_copies);
    fusedStridedCopy(strided_d2d_copies);

    // NOTE: we do H2H after D2D copies to let GPU finish the D2D copies as soon as possible,
    // so that the GPU can start the kernel launch as soon as possible.

    // H2H copies (common to both modes)
    optimizedCopyAsync(inputs.attention_inputs.cu_seqlens_host,
                       py_model_inputs_.attention_inputs.cu_seqlens_host,
                       (state.current_batch_size + 1) * sizeof(int));

    optimizedCopyAsync(inputs.attention_inputs.input_lengths,
                       py_model_inputs_.attention_inputs.input_lengths,
                       state.current_batch_size * sizeof(int));

    optimizedCopyAsync(inputs.attention_inputs.prefix_lengths,
                       py_model_inputs_.attention_inputs.prefix_lengths,
                       state.current_batch_size * sizeof(int));

    if (!has_hybrid_cache) {
        stridedCopyHost(inputs.attention_inputs.kv_cache_kernel_block_id_host,
                        py_model_inputs_.attention_inputs.kv_cache_kernel_block_id_host);
    }

    optimizedCopyAsync(inputs.attention_inputs.kv_cache_layer_to_group,
                       py_model_inputs_.attention_inputs.kv_cache_layer_to_group,
                       inputs.attention_inputs.kv_cache_layer_to_group.numel() * sizeof(int32_t));

    if (!isPrefillCudaGraph()) {
        optimizedCopyAsync(inputs.attention_inputs.sequence_lengths,
                           py_model_inputs_.attention_inputs.sequence_lengths,
                           state.current_batch_size * sizeof(int));
        const int cu_seqlens_size = py_model_inputs_.attention_inputs.cu_seqlens.numel();
        if (state.current_batch_size + 1 < cu_seqlens_size) {
            py_model_inputs_.attention_inputs.cu_seqlens.slice(0, state.current_batch_size + 1, cu_seqlens_size)
                .fill_(token_num);
            py_model_inputs_.attention_inputs.cu_seqlens_host.slice(0, state.current_batch_size + 1, cu_seqlens_size)
                .fill_(token_num);
        }
    } else {
        if (isGenerationPrefillCudaGraph()) {
            const int   fixed_seq_len  = std::min(state.current_real_graph_seq_len, max_seq_len_);
            auto*       offsets        = prefill_padding_offset_host_.data_ptr<int32_t>();
            const auto* input_lengths  = inputs.attention_inputs.input_lengths.data_ptr<int32_t>();
            int         token_index    = 0;
            int         cumulative_gap = 0;
            for (int batch_index = 0; batch_index < state.current_batch_size; ++batch_index) {
                const int input_length = input_lengths[batch_index];
                std::fill_n(offsets + token_index, input_length, cumulative_gap);
                token_index += input_length;
                cumulative_gap += fixed_seq_len - input_length;
            }
            std::fill(offsets + token_index, offsets + state.current_real_graph_seq_len, 0);
            optimizedCopyAsync(prefill_padding_offset_host_,
                               py_model_inputs_.attention_inputs.padding_offset,
                               state.current_real_graph_seq_len * sizeof(int));
        } else {
            optimizedCopyAsync(inputs.attention_inputs.padding_offset,
                               py_model_inputs_.attention_inputs.padding_offset,
                               state.current_seq_len * sizeof(int));
            if (state.current_seq_len < py_model_inputs_.attention_inputs.padding_offset.numel()) {
                py_model_inputs_.attention_inputs.padding_offset
                    .slice(0, state.current_seq_len, py_model_inputs_.attention_inputs.padding_offset.numel())
                    .fill_(0);
            }
        }

        if (py_model_inputs_.attention_inputs.prefill_cuda_graph_copy_params) {
            auto* batch_size_ptr = py_model_inputs_.attention_inputs.prefill_cuda_graph_copy_params
                                       ->cuda_graph_prefill_batch_size.data_ptr<int>();
            *batch_size_ptr = state.current_batch_size;
        }
    }

    // Hybrid cache: H2H strided copies for per-group block tables
    if (has_hybrid_cache) {
        for (size_t g = 0; g < hybrid_cache_group; ++g) {
            stridedCopyHost(inputs.attention_inputs.kv_cache_kernel_block_id_host_by_group[g],
                            py_model_inputs_.attention_inputs.kv_cache_kernel_block_id_host_by_group[g]);
        }
    }

    // Reset unused batch portions to prevent stale data (prefill only)
    if (isPrefillCudaGraph()) {
        if (state.current_batch_size < max_bs_) {
            py_model_inputs_.attention_inputs.prefix_lengths.slice(0, state.current_batch_size, max_bs_).fill_(0);
            py_model_inputs_.attention_inputs.input_lengths.slice(0, state.current_batch_size, max_bs_).fill_(0);
            py_model_inputs_.attention_inputs.prefix_lengths_d.slice(0, state.current_batch_size, max_bs_).fill_(0);
            py_model_inputs_.attention_inputs.input_lengths_d.slice(0, state.current_batch_size, max_bs_).fill_(0);
        }
        py_model_inputs_.attention_inputs.sequence_lengths.fill_(0);

        int         prefix_sum     = 0;
        const auto* prefix_lengths = inputs.attention_inputs.prefix_lengths.data_ptr<int32_t>();
        for (int b = 0; b < state.current_batch_size; ++b) {
            prefix_sum += prefix_lengths[b];
        }
        const int last_valid_q  = state.current_seq_len;
        const int last_valid_kv = last_valid_q + prefix_sum;
        py_model_inputs_.attention_inputs.cu_seqlens_host.slice(0, state.current_batch_size + 1, max_bs_ + 1)
            .fill_(last_valid_q);
        py_model_inputs_.attention_inputs.cu_seqlens.slice(0, state.current_batch_size + 1, max_bs_ + 1)
            .fill_(last_valid_q);
        py_model_inputs_.attention_inputs.cu_kv_seqlens.slice(0, state.current_batch_size + 1, max_bs_ + 1)
            .fill_(last_valid_kv);
    }

    // launch prepare_cuda_graph when attention inputs are ready
    {
        RTP_LLM_PROFILE_SCOPE("cuda_graph.prepareInputs(prepare_cuda_graph)");
        attn_pyobj.attr("prepare_cuda_graph")(py_model_inputs_.attention_inputs);
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
    state.current_seq_len = inputs.input_ids.size(0);
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
    if (!isGenerationPrefillCudaGraph()) {
        return true;
    }
    if (state.current_batch_size <= 0 || static_cast<size_t>(state.current_batch_size) > max_bs_) {
        RTP_LLM_LOG_WARNING("prefill batch size %d exceeds fixed metadata capacity %zu, fallback to normal run",
                            state.current_batch_size,
                            max_bs_);
        return false;
    }
    if (inputs.attention_inputs.input_lengths.is_cuda()) {
        RTP_LLM_LOG_WARNING("prefill input_lengths must be host-resident for Graph eligibility checks");
        return false;
    }

    const auto* input_lengths = inputs.attention_inputs.input_lengths.data_ptr<int32_t>();
    const bool  has_prefix    = inputs.attention_inputs.prefix_lengths.defined()
                            && inputs.attention_inputs.prefix_lengths.numel() >= state.current_batch_size;
    if (!has_prefix || inputs.attention_inputs.prefix_lengths.is_cuda()) {
        RTP_LLM_LOG_WARNING("prefill prefix_lengths must be host-resident and cover the full batch");
        return false;
    }
    const auto* prefix_lengths = inputs.attention_inputs.prefix_lengths.data_ptr<int32_t>();
    int64_t     packed_tokens  = 0;
    for (int b = 0; b < state.current_batch_size; ++b) {
        const int input_len  = input_lengths[b];
        const int prefix_len = prefix_lengths[b];
        if (input_len <= 0 || prefix_len < 0 || input_len + prefix_len > max_seq_len_) {
            RTP_LLM_LOG_WARNING(
                "prefill sequence %d has unsupported input/prefix lengths %d/%d, fallback to normal run",
                b,
                input_len,
                prefix_len);
            return false;
        }
        packed_tokens += input_len;
    }
    if (packed_tokens != state.current_seq_len) {
        RTP_LLM_LOG_WARNING("prefill packed token mismatch: input_ids=%d, input_lengths=%ld; fallback to normal run",
                            state.current_seq_len,
                            packed_tokens);
        return false;
    }
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

    if (!enable_cuda_graph_ || inputs.attention_inputs.is_prefill != isPrefillCudaGraph()) {
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
        torch::zeros({int(max_bs_ + 1)}, torch::TensorOptions(torch::kInt32).device(torch::kCPU)).pin_memory();
    torch::Tensor cu_kv_seqlens =
        torch::zeros({int(max_bs_ + 1)}, torch::TensorOptions(torch::kInt32).device(torch::kCPU).pinned_memory(true));
    auto input_lengths  = capture_mem_hold_.py_model_inputs_.attention_inputs.input_lengths;
    auto prefix_lengths = capture_mem_hold_.py_model_inputs_.attention_inputs.prefix_lengths;

    cu_seqlens.slice(0, 1, max_bs_ + 1) = input_lengths.cumsum(0);
    if (prefix_lengths.defined() && prefix_lengths.size(0) > 0) {
        cu_kv_seqlens.slice(0, 1, max_bs_ + 1) = input_lengths.add(prefix_lengths).cumsum(0);
    }
    capture_mem_hold_.py_model_inputs_.attention_inputs.cu_seqlens_host = cu_seqlens;
    capture_mem_hold_.py_model_inputs_.attention_inputs.cu_seqlens      = cu_seqlens.cuda();
    capture_mem_hold_.py_model_inputs_.attention_inputs.cu_kv_seqlens   = cu_kv_seqlens.cuda();
}

int CudaGraphRunner::getCurrentRealGraphBs(const CudaGraphState& state) const {
    return isPrefillCudaGraph() ? state.current_real_graph_seq_len : state.current_real_graph_bs;
}

void CudaGraphRunner::initCaptureAttentionInputs(PyModelInputs& inputs, int max_bs, int num_tokens_per_bs) {
    inputs.attention_inputs.is_target_verify = is_target_verify_;
    inputs.attention_inputs.is_prefill       = is_prefill_cuda_graph_mode_ || num_tokens_per_bs_ > 1;
    inputs.attention_inputs.is_cuda_graph    = true;

    // input_ids [tokens_nums] = [batch_size * num_tokens_per_bs]
    inputs.input_ids = torch::zeros({max_num_token_}, options_cuda_int32_);
    // input_lengths [batch_size, int32] (decode only)
    inputs.attention_inputs.input_lengths   = torch::full({int(max_bs_)}, num_tokens_per_bs_, options_cpu_int32_);
    inputs.attention_inputs.input_lengths   = inputs.attention_inputs.input_lengths.pin_memory();
    inputs.attention_inputs.input_lengths_d = inputs.attention_inputs.input_lengths.cuda();
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
    inputs.attention_inputs.kv_cache_block_id_device = torch::zeros({int(max_bs_), max_kv_blocks}, options_cuda_int32_);
    inputs.attention_inputs.kv_cache_block_id_host =
        torch::zeros({int(max_bs_), max_kv_blocks}, options_cpu_int32_).pin_memory();

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
        // FMHA captures the legacy table before per-layer selection, so keep it aliased to full-attention group 0.
        inputs.attention_inputs.kv_cache_kernel_block_id_device =
            inputs.attention_inputs.kv_cache_kernel_block_id_device_by_group.front();
        inputs.attention_inputs.kv_cache_kernel_block_id_host =
            inputs.attention_inputs.kv_cache_kernel_block_id_host_by_group.front();
    }

    // prefix_lengths [batch_size, int32] (for attention `prepare`)
    if (num_tokens_per_bs_ > 1 && !is_prefill_cuda_graph_mode_) {
        inputs.attention_inputs.prefix_lengths =
            torch::full({int(max_bs_)}, max_seq_len_ - num_tokens_per_bs_, options_cpu_int32_).pin_memory();
        inputs.attention_inputs.prefix_lengths_d = inputs.attention_inputs.prefix_lengths.cuda();
    } else if (is_prefill_cuda_graph_mode_) {
        // ROCm needs prefix>0 here for AiterPrefillImplPaged.support(); CUDA keeps prefix=0.
#if USING_ROCM
        const int prefix_init = isMtpDraftPrefillCudaGraph() ? max_seq_len_ : 0;
#else
        const int prefix_init = 0;
#endif
        inputs.attention_inputs.prefix_lengths =
            torch::full({int(max_bs_)}, prefix_init, options_cpu_int32_).pin_memory();
        inputs.attention_inputs.prefix_lengths_d = inputs.attention_inputs.prefix_lengths.cuda();
    } else {
        // Decode CUDA graph mode: prefix_lengths should be empty tensor
        inputs.attention_inputs.prefix_lengths = torch::empty({0}, options_cpu_int32_).pin_memory();
    }
    // padding_offset [max_num_token_, int32] (for attention padding)
    inputs.attention_inputs.padding_offset            = torch::zeros({int(max_seq_len_ * max_bs_)}, options_cpu_int32_);
    inputs.attention_inputs.padding_offset            = inputs.attention_inputs.padding_offset.pin_memory();
    inputs.attention_inputs.dtype                     = model_data_type_;
    inputs.attention_inputs.is_s_padded               = true;
    inputs.attention_inputs.sequence_lengths_plus_1_d = torch::zeros({int(max_bs_)}, options_cuda_int32_);
    inputs.attention_inputs.decode_cu_seqlens_d =
        torch::arange(0, max_bs_ + 1, 1, torch::TensorOptions(torch::kInt32).device(torch::kCUDA));
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
    if (!enable_cuda_graph_) {
        initKernelInternalMemory();
        RTP_LLM_LOG_INFO("CUDA graph capture is not enabled, skipping initialization");
        return;
    }

    if (lazy_capture_) {
        RTP_LLM_LOG_INFO("CUDA graph lazy capture enabled; initializing storage without eager capture");
        initLazyStorage();
        return;
    }

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
    const int max_captured_tokens =
        capture_range_.empty() ?
            0 :
            (is_prefill_cuda_graph_mode_ ? capture_range_.back() : capture_range_.back() * num_tokens_per_bs_);
    zero_input_ids_     = torch::zeros({max_captured_tokens}, options_cuda_int32_);
    zero_input_hiddens_ = torch::zeros({max_captured_tokens, hidden_size_}, options_cuda_float_);

    PyModelInputs inputs;
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
    py_forward_method_(capture_mem_hold_.py_model_inputs_, attn_pyobj);
    RTP_LLM_LOG_INFO("initCapture forward for output datatype end");
    output = torch::zeros({max_num_token_, hidden_size_}, options_cuda_float_);
    capture_mem_hold_.setHiddenStates(output);
    initCaptureAttentionInputsPost();
    logCudaGraphPoolMemory("before_capture");

    if (is_prefill_cuda_graph_mode_) {
        RTP_LLM_CHECK_WITH_INFO(isEmbeddingStylePrefillCudaGraph() || isMtpDraftPrefillCudaGraph(),
                                "prefill cuda graph: expected embedding-style or MTP draft layout");
        capturePrefill();
    } else {
        captureDecode();
    }
    logCudaGraphPoolMemory("after_capture");
}

void CudaGraphRunner::initLazyStorage() {
    if (lazy_storage_ready_) {
        return;
    }
    RTP_LLM_LOG_INFO("CUDA graph lazy storage init start (%s)", isPrefillCudaGraph() ? "prefill" : "decode");
    shared_graph_pool_ = cuda_graph::graphPoolHandle();
    if (isPrefillCudaGraph()) {
        capture_range_ = getPrefillSequenceLengthsToCapture();
    } else {
        capture_range_ = getDecodeBatchSizesToCapture();
    }
    const int max_captured_tokens =
        capture_range_.empty() ?
            0 :
            (isPrefillCudaGraph() ? capture_range_.back() : capture_range_.back() * num_tokens_per_bs_);
    max_num_token_      = max_captured_tokens;
    zero_input_ids_     = torch::zeros({max_captured_tokens}, options_cuda_int32_);
    zero_input_hiddens_ = torch::zeros({max_captured_tokens, hidden_size_}, options_cuda_float_);
    if (isGenerationPrefillCudaGraph()) {
        prefill_padding_offset_host_ = torch::zeros({max_captured_tokens}, options_cpu_int32_).pin_memory();
    }

    PyModelInputs inputs;
    inputs.input_hiddens = torch::zeros({max_num_token_, hidden_size_}, options_cuda_float_);
    initCaptureAttentionInputs(inputs, max_bs_, num_tokens_per_bs_);
    initCaptureBertEmbeddingInputs(inputs, max_bs_, max_num_token_);

    torch::Tensor output;
    capture_mem_hold_ = CaptureMemoryHold(output, inputs, is_prefill_cuda_graph_mode_);
    initKernelInternalMemory();

    // Lazy path skips the eager discovery forward: hidden-state storage always uses model_data_type_.
    output = torch::zeros({max_num_token_, hidden_size_}, options_cuda_float_);
    capture_mem_hold_.setHiddenStates(output);
    initCaptureAttentionInputsPost();

    if (isPrefillCudaGraph()) {
        RTP_LLM_CHECK_WITH_INFO(isEmbeddingStylePrefillCudaGraph() || isMtpDraftPrefillCudaGraph()
                                    || isGenerationPrefillCudaGraph(),
                                "prefill cuda graph: unsupported graph mode");
    }

    // Pre-register buckets; graph memory for each eligible bucket is created on demand.
    for (int key : capture_range_) {
        const bool exceeds_mori_capacity =
            isGenerationPrefillCudaGraph() && mori_max_tokens_ > 0 && key > mori_max_tokens_;
        if (exceeds_mori_capacity) {
            bucket_states_.emplace(key, BucketState::Disabled);
            RTP_LLM_LOG_WARNING(
                "prefill graph bucket %d exceeds Mori capacity %d and is disabled", key, mori_max_tokens_);
            continue;
        }
        graph_instances_.try_emplace(key, enable_cuda_graph_debug_mode_);
        bucket_states_.emplace(key, BucketState::Uncaptured);
    }
    lazy_storage_ready_ = true;
    logCudaGraphPoolMemory("after_lazy_storage_init");
    RTP_LLM_LOG_INFO("CUDA graph lazy storage init done: %zu buckets (%s)",
                     capture_range_.size(),
                     is_prefill_cuda_graph_mode_ ? "prefill" : "decode");
}

void CudaGraphRunner::resetSharedCaptureStorage() {
    auto& attn = capture_mem_hold_.py_model_inputs_.attention_inputs;

    capture_mem_hold_.py_model_inputs_.input_ids.zero_();
    capture_mem_hold_.py_model_inputs_.input_hiddens.zero_();

    attn.kv_cache_kernel_block_id_device.zero_();
    attn.kv_cache_kernel_block_id_host.zero_();
    if (attn.kv_cache_block_id_device.defined()) {
        attn.kv_cache_block_id_device.zero_();
    }
    if (attn.kv_cache_block_id_host.defined()) {
        attn.kv_cache_block_id_host.zero_();
    }
    for (auto& table : attn.kv_cache_kernel_block_id_device_by_group) {
        table.zero_();
    }
    for (auto& table : attn.kv_cache_kernel_block_id_host_by_group) {
        table.zero_();
    }

    attn.padding_offset.zero_();
    attn.sequence_lengths.fill_(max_seq_len_ - num_tokens_per_bs_ - 1);
    attn.sequence_lengths_plus_1_d.zero_();
    attn.input_lengths.fill_(num_tokens_per_bs_);
    attn.input_lengths_d.copy_(attn.input_lengths);

    // Mirror initKernelInternalMemory(): cumulative lengths derived from the reset input_lengths.
    attn.cu_seqlens_host.zero_();
    attn.cu_seqlens_host.slice(0, 1, max_bs_ + 1) = attn.input_lengths.cumsum(0);
    attn.cu_seqlens.copy_(attn.cu_seqlens_host);
    attn.cu_kv_seqlens.zero_();
    if (attn.prefix_lengths.defined() && attn.prefix_lengths.size(0) > 0) {
        attn.cu_kv_seqlens.slice(0, 1, max_bs_ + 1) =
            attn.input_lengths.add(attn.prefix_lengths).cumsum(0).to(attn.cu_kv_seqlens.device());
    }
}

void CudaGraphRunner::buildBucketInstance(int key) {
    resetSharedCaptureStorage();
    graph_instances_.try_emplace(key, enable_cuda_graph_debug_mode_);
    PyModelInputs inputs;

    if (is_prefill_cuda_graph_mode_) {
        const int seq_len = key;
        // for attention, it always run the max_bs, so when we run `forward`, the real batch size is not sure
        // we will transfer a `batch size tensor(int)` for `copy kernel`.
        prepareCaptureInputs(inputs, max_bs_, seq_len);
        if (isEmbeddingStylePrefillCudaGraph()) {
            inputs.attention_inputs.prefix_lengths.fill_(0);
            inputs.attention_inputs.cu_seqlens_host[0] = 0;
            inputs.attention_inputs.cu_seqlens_host[1] = seq_len;
            inputs.attention_inputs.cu_seqlens.copy_(inputs.attention_inputs.cu_seqlens_host, false);
            inputs.attention_inputs.input_lengths[0] = seq_len;
        } else {
            const int tokens_per_sequence = isGenerationPrefillCudaGraph() ? max_seq_len_ : num_tokens_per_bs_;
            const int active_bs = std::min<int>(max_bs_, (seq_len + tokens_per_sequence - 1) / tokens_per_sequence);
            RTP_LLM_CHECK_WITH_INFO(active_bs > 0 && seq_len <= active_bs * tokens_per_sequence,
                                    "prefill bucket %d exceeds fixed metadata capacity",
                                    seq_len);
            const int prefix_len = isGenerationPrefillCudaGraph() ? 0 : max_seq_len_;

            inputs.attention_inputs.input_lengths.fill_(0);
            inputs.attention_inputs.prefix_lengths.fill_(prefix_len);
            auto*     input_lengths  = inputs.attention_inputs.input_lengths.data_ptr<int32_t>();
            auto*     prefix_lengths = inputs.attention_inputs.prefix_lengths.data_ptr<int32_t>();
            const int base_tokens    = seq_len / active_bs;
            const int extra_tokens   = seq_len % active_bs;
            for (int b = 0; b < active_bs; ++b) {
                input_lengths[b] = base_tokens + (b < extra_tokens ? 1 : 0);
            }

            auto  cu_kv_seqlens_host = torch::zeros({int(max_bs_ + 1)}, options_cpu_int32_).pin_memory();
            auto* cu_q               = inputs.attention_inputs.cu_seqlens_host.data_ptr<int32_t>();
            auto* cu_kv              = cu_kv_seqlens_host.data_ptr<int32_t>();
            cu_q[0]                  = 0;
            cu_kv[0]                 = 0;
            for (int b = 0; b < max_bs_; ++b) {
                cu_q[b + 1]  = cu_q[b] + input_lengths[b];
                cu_kv[b + 1] = cu_kv[b] + input_lengths[b] + prefix_lengths[b];
            }

            inputs.attention_inputs.input_lengths_d.copy_(inputs.attention_inputs.input_lengths);
            inputs.attention_inputs.prefix_lengths_d.copy_(inputs.attention_inputs.prefix_lengths);
            inputs.attention_inputs.cu_seqlens.copy_(inputs.attention_inputs.cu_seqlens_host);
            inputs.attention_inputs.cu_kv_seqlens.copy_(cu_kv_seqlens_host);
        }

        inputs.attention_inputs.context_total_kv_length = seq_len;
        inputs.attention_inputs.prefill_cuda_graph_copy_params =
            capture_mem_hold_.py_model_inputs_.attention_inputs.prefill_cuda_graph_copy_params;
        if (inputs.bert_embedding_inputs.position_encoding.numel() > 0) {
            inputs.bert_embedding_inputs.combo_position_ids =
                inputs.bert_embedding_inputs.combo_position_ids.slice(0, 0, seq_len);
            inputs.bert_embedding_inputs.combo_tokens_type_ids =
                inputs.bert_embedding_inputs.combo_tokens_type_ids.slice(0, 0, seq_len);
        }
        graph_instances_[seq_len].mem_hold_ = createCaptureMemoryHold(inputs, seq_len);
        graph_instances_[seq_len].mem_hold_.attn_pyobj_ =
            py_attn_pyobj_method_(graph_instances_[seq_len].mem_hold_.py_model_inputs_, true);
        graph_instances_[seq_len].mem_hold_.decoder_layer_hidden_states_ =
            graph_instances_[seq_len].mem_hold_.decoder_layer_hidden_states_.slice(0, 0, seq_len);
    } else {
        const int bs = key;
        prepareCaptureInputs(inputs, bs, bs * num_tokens_per_bs_);

        // calculate context_total_kv_length
        int max_input_len  = inputs.attention_inputs.input_lengths.max().item<int>();
        int max_prefix_len = 0;
        if (inputs.attention_inputs.prefix_lengths.defined() && inputs.attention_inputs.prefix_lengths.size(0) > 0) {
            max_prefix_len = inputs.attention_inputs.prefix_lengths.max().item<int>();
        }
        inputs.attention_inputs.context_total_kv_length = bs * (max_input_len + max_prefix_len);

        graph_instances_[bs].mem_hold_ = createCaptureMemoryHold(inputs, bs * num_tokens_per_bs_);
        graph_instances_[bs].mem_hold_.attn_pyobj_ =
            py_attn_pyobj_method_(graph_instances_[bs].mem_hold_.py_model_inputs_, true);
    }
}

bool CudaGraphRunner::captureBucketLazy(int key) {
    const char* key_type       = is_prefill_cuda_graph_mode_ ? "seq len" : "batch size";
    bool        local_prepared = true;
    try {
        buildBucketInstance(key);
    } catch (const std::exception& e) {
        local_prepared = false;
        RTP_LLM_LOG_ERROR("lazy capture preparation failed for %s %d: %s", key_type, key, e.what());
    } catch (...) {
        local_prepared = false;
        RTP_LLM_LOG_ERROR("lazy capture preparation failed for %s %d", key_type, key);
    }
    if (!synchronizeCaptureSuccess(local_prepared)) {
        RTP_LLM_LOG_ERROR(
            "lazy capture preparation failed for %s %d; bucket will fall back to eager forever", key_type, key);
        return false;
    }

    bool finish_attempted = false;
    try {
        captureOneGraphInstance(key, key_type);
        finish_attempted = true;
        cuda_graph::finish_capture_session();
        replayAndSyncCheck(key, key_type);
        RTP_LLM_LOG_INFO("lazy capture success for %s: %d", key_type, key);
        return true;
    } catch (...) {
        if (!finish_attempted) {
            try {
                cuda_graph::finish_capture_session();
            } catch (const std::exception& cleanup_error) {
                RTP_LLM_LOG_ERROR(
                    "failed to finalize capture session for %s %d: %s", key_type, key, cleanup_error.what());
            } catch (...) {
                RTP_LLM_LOG_ERROR("failed to finalize capture session for %s %d", key_type, key);
            }
        }
        RTP_LLM_LOG_ERROR(
            "lazy capture failed after collective execution began for %s %d; aborting request", key_type, key);
        throw;
    }
}

bool CudaGraphRunner::synchronizeCaptureSuccess(bool local_success) {
    try {
        py::object dist = py::module_::import("torch.distributed");
        if (!dist.attr("is_available")().cast<bool>() || !dist.attr("is_initialized")().cast<bool>()) {
            return local_success;
        }

        auto status = torch::tensor({local_success ? 1 : 0}, options_cuda_int32_);
        dist.attr("all_reduce")(status, dist.attr("ReduceOp").attr("MIN"));
        const bool global_success = status.item<int>() != 0;
        if (local_success && !global_success) {
            RTP_LLM_LOG_WARNING("lazy capture failed on another rank; disabling this bucket on all ranks");
        }
        return global_success;
    } catch (const std::exception& e) {
        RTP_LLM_LOG_ERROR("failed to synchronize lazy capture result: %s", e.what());
        return false;
    } catch (...) {
        RTP_LLM_LOG_ERROR("failed to synchronize lazy capture result");
        return false;
    }
}

GraphRunDecision CudaGraphRunner::plan(const PyModelInputs& inputs, CudaGraphState& state) {
    state.lazy_capture_key = -1;
    if (!lazy_capture_) {
        return canRun(inputs, state) ? GraphRunDecision::Replay : GraphRunDecision::Eager;
    }

    if (!canRun(inputs, state)) {
        return GraphRunDecision::Eager;
    }

    const int key = is_prefill_cuda_graph_mode_ ? state.current_real_graph_seq_len : state.current_real_graph_bs;
    std::lock_guard<std::mutex> lock(bucket_states_mutex_);
    auto                        it = bucket_states_.find(key);
    if (it == bucket_states_.end()) {
        RTP_LLM_LOG_WARNING("lazy plan: selected key %d has no bucket state, fallback to eager", key);
        return GraphRunDecision::Eager;
    }

    switch (it->second) {
        case BucketState::Ready:
            return GraphRunDecision::Replay;
        case BucketState::Uncaptured:
            state.lazy_capture_key = key;
            RTP_LLM_LOG_INFO("lazy plan: %s bucket %d uncaptured, serve eager then capture",
                             is_prefill_cuda_graph_mode_ ? "prefill" : "decode",
                             key);
            return GraphRunDecision::CaptureAfterEager;
        case BucketState::Capturing:
        case BucketState::Failed:
        case BucketState::Disabled:
        default:
            return GraphRunDecision::Eager;
    }
}

bool CudaGraphRunner::captureCurrentBucket(const CudaGraphState& state) {
    if (!lazy_capture_ || state.lazy_capture_key < 0) {
        return false;
    }
    const int key = state.lazy_capture_key;
    {
        std::lock_guard<std::mutex> lock(bucket_states_mutex_);
        auto                        it = bucket_states_.find(key);
        if (it == bucket_states_.end() || it->second != BucketState::Uncaptured) {
            return false;
        }
        it->second = BucketState::Capturing;
    }

    py::gil_scoped_acquire gil;
    const bool             local_success  = captureBucketLazy(key);
    const bool             global_success = synchronizeCaptureSuccess(local_success);
    {
        std::lock_guard<std::mutex> lock(bucket_states_mutex_);
        bucket_states_[key] = global_success ? BucketState::Ready : BucketState::Failed;
    }
    return global_success;
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
            cuda_graph::GraphNcclCaptureContext capture_ctx;
            CudaGraphCaptureGuard               capture_guard(&capture_ctx);
            try {
                auto py_outputs_obj = py_forward_method_(inputs, attn_pyobj);
                outputs             = py_outputs_obj.cast<PyModelOutputs>();
                graph_instances_[key].mem_hold_.decoder_layer_hidden_states_.copy_(outputs.hidden_states);
            } catch (...) {
                try {
                    graph.capture_end();
                } catch (const std::exception& cleanup_error) {
                    RTP_LLM_LOG_ERROR(
                        "Failed to end invalid capture for %s %d: %s", key_type, key, cleanup_error.what());
                } catch (...) {
                    RTP_LLM_LOG_ERROR("Failed to end invalid capture for %s %d", key_type, key);
                }
                throw;
            }
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
    inputs.attention_inputs.is_cuda_graph    = true;
    inputs.input_ids     = capture_mem_hold_.py_model_inputs_.input_ids.slice(0, 0, seq_len_or_tokens);
    inputs.input_hiddens = capture_mem_hold_.py_model_inputs_.input_hiddens.slice(0, 0, seq_len_or_tokens);
    inputs.attention_inputs.input_lengths =
        capture_mem_hold_.py_model_inputs_.attention_inputs.input_lengths.slice(0, 0, batch_size);
    inputs.attention_inputs.input_lengths_d =
        capture_mem_hold_.py_model_inputs_.attention_inputs.input_lengths_d.slice(0, 0, batch_size);
    inputs.attention_inputs.padding_offset =
        capture_mem_hold_.py_model_inputs_.attention_inputs.padding_offset.slice(0, 0, seq_len_or_tokens);

    // Common slice operations for attention inputs
    if (capture_mem_hold_.py_model_inputs_.attention_inputs.prefix_lengths.defined()) {
        if (capture_mem_hold_.py_model_inputs_.attention_inputs.prefix_lengths.size(0) > 0) {
            inputs.attention_inputs.prefix_lengths =
                capture_mem_hold_.py_model_inputs_.attention_inputs.prefix_lengths.slice(0, 0, batch_size);
            inputs.attention_inputs.prefix_lengths_d =
                capture_mem_hold_.py_model_inputs_.attention_inputs.prefix_lengths_d.slice(0, 0, batch_size);
        } else {
            // For decode CUDA graph mode: prefix_lengths is empty tensor
            inputs.attention_inputs.prefix_lengths = capture_mem_hold_.py_model_inputs_.attention_inputs.prefix_lengths;
        }
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
    inputs.attention_inputs.cu_seqlens_host =
        capture_mem_hold_.py_model_inputs_.attention_inputs.cu_seqlens_host.slice(0, 0, batch_size + 1);
    inputs.attention_inputs.cu_seqlens =
        capture_mem_hold_.py_model_inputs_.attention_inputs.cu_seqlens.slice(0, 0, batch_size + 1);
    inputs.attention_inputs.cu_kv_seqlens =
        capture_mem_hold_.py_model_inputs_.attention_inputs.cu_kv_seqlens.slice(0, 0, batch_size + 1);
    inputs.attention_inputs.decode_cu_seqlens_d =
        capture_mem_hold_.py_model_inputs_.attention_inputs.decode_cu_seqlens_d.slice(0, 0, batch_size + 1);
    inputs.attention_inputs.sequence_lengths_plus_1_d =
        capture_mem_hold_.py_model_inputs_.attention_inputs.sequence_lengths_plus_1_d.slice(0, 0, batch_size);

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
    params.enable_cuda_graph          = true;
    params.is_prefill_cuda_graph_mode = true;
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
