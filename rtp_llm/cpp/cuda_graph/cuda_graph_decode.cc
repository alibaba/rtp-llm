#include "rtp_llm/cpp/cuda_graph/cuda_graph_runner.h"
#include "rtp_llm/cpp/cuda_graph/cuda_graph_capture_range.h"
#if USING_CUDA
#include <cuda_runtime.h>
#endif

namespace rtp_llm {
int CudaGraphRunner::decodeGraphKey(int batch_size, int tokens_per_request) const {
    RTP_LLM_CHECK_WITH_INFO(batch_size > 0 && batch_size <= static_cast<int>(max_bs_) && tokens_per_request > 0
                                && tokens_per_request <= num_tokens_per_bs_,
                            "invalid adaptive graph dimensions: batch=%d tokens_per_request=%d",
                            batch_size,
                            tokens_per_request);
    return batch_size + (num_tokens_per_bs_ - tokens_per_request) * (static_cast<int>(max_bs_) + 1);
}

void CudaGraphRunner::replayDecode(int bs) {
    replayGraph(bs);
}

std::vector<int> CudaGraphRunner::getDecodeBatchSizesToCapture() {
    if (adaptive_verification_) {
        // Ignore sparse decode BS buckets only for adaptive target verification.
        return configuredAdaptiveVerifyCaptureRows(static_cast<int>(max_bs_), num_tokens_per_bs_);
    }
    // If decode_capture_batch_sizes_ is provided from Python, use it directly
    if (!decode_capture_batch_sizes_.empty()) {
        RTP_LLM_LOG_INFO("Using decode capture batch sizes from Python: %zu sizes", decode_capture_batch_sizes_.size());
        // Sort in ascending order (from small to large)
        std::sort(decode_capture_batch_sizes_.begin(), decode_capture_batch_sizes_.end());
        return decode_capture_batch_sizes_;
    }

    // Otherwise, use default logic
    std::vector<int> capture_bs;
    int              max_generate_batch_size = max_bs_;
    RTP_LLM_LOG_INFO("max_generate_batch_size for cuda graph: %d", max_generate_batch_size);
    // Add key batch sizes up to 32
    for (int i : {1, 2, 3, 4, 5, 6, 7, 8, 16, 24, 32}) {
        if (i <= max_generate_batch_size) {
            capture_bs.push_back(i);
        }
    }
    // Add range from 48 to max_generate_batch_size, stepping by 16
    for (int i = 48; i <= max_generate_batch_size; i += 16) {
        capture_bs.push_back(i);
    }
    if (capture_bs[capture_bs.size() - 1] != max_generate_batch_size) {
        capture_bs.push_back(max_generate_batch_size);
    }
    return capture_bs;
}

void CudaGraphRunner::captureDecodeOneBatchSize(int bs, bool needs_distributed_warmup) {
    captureOneGraphInstance(bs, "batch size", needs_distributed_warmup);
}

void CudaGraphRunner::prepareUniformDecodeCapture(int key, int batch_size, bool native_target_layout) {
    const int     rows = batch_size * num_tokens_per_bs_;
    PyModelInputs inputs;
    prepareCaptureInputs(inputs, batch_size, rows);
    inputs.attention_inputs.uniform_target_verify_layout = native_target_layout;
    const int max_input_len                              = inputs.attention_inputs.input_lengths.max().item<int>();
    int       max_prefix_len                             = 0;
    if (inputs.attention_inputs.prefix_lengths.defined() && inputs.attention_inputs.prefix_lengths.numel() > 0) {
        max_prefix_len = inputs.attention_inputs.prefix_lengths.max().item<int>();
    }
    inputs.attention_inputs.context_total_kv_length = batch_size * (max_input_len + max_prefix_len);
    auto& memory                                    = graph_instances_[key].mem_hold_;
    memory                                          = createCaptureMemoryHold(inputs, rows);
    memory.attn_pyobj_                              = py_attn_pyobj_method_(memory.py_model_inputs_, true);
}

void CudaGraphRunner::captureDecode() {
    RTP_LLM_LOG_INFO("Capture Decode Start");
    RTP_LLM_CHECK_WITH_INFO(!adaptive_verification_ || (is_target_verify_ && num_tokens_per_bs_ > 1),
                            "adaptive verification graphs require a multi-token target model");
    if (adaptive_verification_) {
        if (adaptive_native_full_width_) {
            // prepareCaptureInputs returns views. Capture the uniform graph
            // before any adaptive bucket rewrites their lengths and offsets.
            uniform_verify_graph_key_ = max_num_token_ + 1;
            graph_instances_.try_emplace(uniform_verify_graph_key_, enable_cuda_graph_debug_mode_);
            prepareUniformDecodeCapture(uniform_verify_graph_key_, static_cast<int>(max_bs_), true);
            captureOneGraphInstance(uniform_verify_graph_key_, "native full width", /*needs_distributed_warmup=*/true);
            replayAndSyncCheck(uniform_verify_graph_key_, "native full width");
            profileAdaptiveGraph(uniform_verify_graph_key_, max_num_token_);
            RTP_LLM_LOG_INFO(
                "capture native adaptive full width rows=%d key=%d", max_num_token_, uniform_verify_graph_key_);
        }
        // Largest first retains shared-pool reuse and the single distributed warmup.
        for (int rows : capture_range_) {
            graph_instances_.try_emplace(rows, enable_cuda_graph_debug_mode_);
        }
        for (auto it = capture_range_.rbegin(); it != capture_range_.rend(); ++it) {
            const auto    shape   = adaptiveVerifyGraphShape(static_cast<int>(max_bs_), num_tokens_per_bs_, *it);
            const auto    lengths = adaptiveVerifyCaptureLengths(shape);
            const int     bs      = shape.batch_capacity;
            PyModelInputs inputs;
            prepareCaptureInputs(inputs, bs, shape.rows);
            auto lengths_cpu = torch::tensor(lengths, options_cpu_int32_);
            auto offsets     = torch::zeros({bs + 1}, options_cpu_int32_);
            offsets.slice(0, 1).copy_(lengths_cpu.cumsum(0));
            inputs.attention_inputs.input_lengths.copy_(lengths_cpu);
            inputs.attention_inputs.prefix_lengths.copy_(max_seq_len_ - lengths_cpu);
            inputs.attention_inputs.cu_seqlens_host.copy_(offsets);
            inputs.attention_inputs.cu_seqlens.copy_(offsets);
            inputs.attention_inputs.cu_kv_seqlens.copy_(torch::arange(bs + 1, options_cpu_int32_) * max_seq_len_);
            inputs.attention_inputs.total_tokens            = shape.rows;
            inputs.attention_inputs.context_total_kv_length = static_cast<int64_t>(bs) * max_seq_len_;
            graph_instances_[shape.rows].mem_hold_          = createCaptureMemoryHold(inputs, shape.rows);
            graph_instances_[shape.rows].mem_hold_.attn_pyobj_ =
                py_attn_pyobj_method_(graph_instances_[shape.rows].mem_hold_.py_model_inputs_, true);
            captureDecodeOneBatchSize(shape.rows, uniform_verify_graph_key_ == 0 && it == capture_range_.rbegin());
            replayAndSyncCheck(shape.rows, "adaptive verify rows");
            if (uniform_verify_graph_key_ == 0 || shape.rows != max_num_token_) {
                profileAdaptiveGraph(shape.rows, shape.rows);
            }
            RTP_LLM_LOG_INFO("capture adaptive verify rows=%d batch_capacity=%d", shape.rows, bs);
        }
        RTP_LLM_LOG_INFO("Capture adaptive Decode End: %zu sparse token-bucket graphs", capture_range_.size());
        return;
    }
    for (int bs : capture_range_) {
        graph_instances_.try_emplace(decodeGraphKey(bs, num_tokens_per_bs_), enable_cuda_graph_debug_mode_);
    }
    const int capture_range_size = capture_range_.size();
    for (int i = capture_range_size - 1; i >= 0; --i) {
        const int bs  = capture_range_[i];
        const int key = decodeGraphKey(bs, num_tokens_per_bs_);
        prepareUniformDecodeCapture(key, bs);
        try {
            // The largest graph is captured first. Fold the one distributed
            // Warmup forward into its first existing warmup execution so
            // multi-rank MegaMoE still synchronizes before its first JIT/
            // kernel launch without restoring a standalone full forward.
            captureDecodeOneBatchSize(key, i == capture_range_size - 1);
            replayAndSyncCheck(key, "batch size");
            profileAdaptiveGraph(key, bs);
            RTP_LLM_LOG_INFO(
                "capture success for batch size: %d, tokens per request: %d, key: %d", bs, num_tokens_per_bs_, key);
        } catch (const std::exception& e) {
            RTP_LLM_LOG_ERROR("CUDA graph capture failed for decode batch size %d: %s", bs, e.what());
#if USING_CUDA
            cudaGetLastError();
            auto sync_err = cudaDeviceSynchronize();
            if (sync_err != cudaSuccess) {
                RTP_LLM_LOG_ERROR("cudaDeviceSynchronize after capture failure returned: %s",
                                  cudaGetErrorString(sync_err));
                cudaGetLastError();
            }
#endif
            throw;
        }
    }
    RTP_LLM_LOG_INFO("Capture Decode End");
}
}  // namespace rtp_llm
