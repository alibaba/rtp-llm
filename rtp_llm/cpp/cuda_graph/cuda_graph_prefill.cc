#include "rtp_llm/cpp/cuda_graph/cuda_graph_runner.h"
#include "rtp_llm/cpp/cuda_graph/cuda_graph_device_shims.h"
#include <optional>

namespace rtp_llm {
void CudaGraphRunner::capturePrefill() {
    RTP_LLM_LOG_INFO("Capture Prefill Start");
    // Pre-initialize all graph instances with keep_graph based on debug mode
    for (int seq_len : capture_range_) {
        graph_instances_.try_emplace(seq_len, enable_cuda_graph_debug_mode_);
    }
    int capture_range_size = capture_range_.size();
    for (int i = capture_range_size - 1; i >= 0; i--) {
        int seq_len = capture_range_[i];
        RTP_LLM_LOG_INFO("capture range for seq len: %d", seq_len);
        PyModelInputs inputs;
        // for attention, it always run the max_bs, so when we run `forward`, the real batch size is not sure
        // we will transfer a `batch size tensor(int)` for `copy kernel`.
        // Prepare common inputs using shared function
        prepareCaptureInputs(inputs, max_bs_, seq_len);
        // Generative prefill uses Bmax real slots plus one fixed sentinel slot.
        // Capture the conservative layout [0, ..., 0, Tg] so FMHA/RoPE fix
        // batch size and max_q_len at their profile capacity while every KV
        // write lands in runner-owned scratch blocks.
        if (isGenerativePrefillCudaGraph()) {
            RTP_LLM_CHECK_WITH_INFO(max_bs_ == static_cast<size_t>(prefill_cuda_graph_max_requests_ + 1),
                                    "prefill CUDA graph backend capacity mismatch: max_bs=%zu Bmax=%d",
                                    max_bs_,
                                    prefill_cuda_graph_max_requests_);
            inputs.attention_inputs.input_lengths.zero_();
            inputs.attention_inputs.input_lengths[prefill_cuda_graph_max_requests_] = seq_len;
            inputs.attention_inputs.prefix_lengths.zero_();
            inputs.attention_inputs.cu_seqlens.zero_();
            inputs.attention_inputs.cu_seqlens[max_bs_] = seq_len;
            inputs.attention_inputs.padding_offset.fill_(prefill_cuda_graph_max_requests_ * seq_len);
            inputs.attention_inputs.input_lengths_device.copy_(inputs.attention_inputs.input_lengths, false);
            inputs.attention_inputs.prefix_lengths_device.zero_();
            inputs.attention_inputs.cu_seqlens_device.copy_(inputs.attention_inputs.cu_seqlens, false);
            inputs.attention_inputs.cu_kv_seqlens_device.copy_(inputs.attention_inputs.cu_seqlens, false);

            auto install_scratch_row = [&](PyAttentionInputs& attn_inputs, size_t group_id) {
                auto& host_ids   = prefill_scratch_kernel_block_ids_host_.at(group_id);
                auto& device_ids = prefill_scratch_kernel_block_ids_device_.at(group_id);
                RTP_LLM_CHECK_WITH_INFO(host_ids.numel() <= attn_inputs.kv_cache_kernel_block_id.size(1),
                                        "prefill scratch block count exceeds captured table width");
                attn_inputs.kv_cache_kernel_block_id.zero_();
                attn_inputs.kv_cache_kernel_block_id_device.zero_();
                attn_inputs.kv_cache_kernel_block_id[prefill_cuda_graph_max_requests_]
                    .slice(0, 0, host_ids.numel())
                    .copy_(host_ids);
                attn_inputs.kv_cache_kernel_block_id_device[prefill_cuda_graph_max_requests_]
                    .slice(0, 0, device_ids.numel())
                    .copy_(device_ids);
            };
            if (inputs.attention_inputs_by_tag.empty()) {
                install_scratch_row(inputs.attention_inputs, 0);
            } else {
                size_t group_id = 0;
                for (const auto& tag : kv_cache_group_tags_) {
                    install_scratch_row(inputs.attention_inputs_by_tag.at(tag), group_id++);
                }
            }
            // Prefill-specific settings, one the first seq is valid, the post ones are all empty
        } else if (isEmbeddingStylePrefillCudaGraph()) {
            // embedding model, without kv cache
            inputs.attention_inputs.prefix_lengths.fill_(0);
            inputs.attention_inputs.prefix_lengths_device.fill_(0);
            // Must set cu_seqlens/cu_kv_seqlens/input_lengths to match actual seq_len,
            // otherwise FlashInfer plans for max_seq_len tokens but q/k/v only have seq_len tokens
            inputs.attention_inputs.input_lengths.fill_(0);
            inputs.attention_inputs.input_lengths[0] = seq_len;
            inputs.attention_inputs.input_lengths_device.copy_(inputs.attention_inputs.input_lengths, false);
            inputs.attention_inputs.cu_seqlens.fill_(seq_len);
            inputs.attention_inputs.cu_seqlens[0] = 0;
            inputs.attention_inputs.cu_seqlens_device.copy_(inputs.attention_inputs.cu_seqlens, false);
            inputs.attention_inputs.cu_kv_seqlens_device.copy_(inputs.attention_inputs.cu_seqlens, false);
        } else {
            // Draft model prefill: distribute seq_len tokens across batches (max num_tokens_per_bs_ each).
            // All max_bs_ batches get the largest legal prefix so
            // prefix_len + q_len never exceeds max_seq_len_.
            int active_bs  = (seq_len + num_tokens_per_bs_ - 1) / num_tokens_per_bs_;
            int prefix_len = max_seq_len_ > num_tokens_per_bs_ ? max_seq_len_ - num_tokens_per_bs_ : 0;

            // All batches get prefix_len to maximize buffer allocation during capture.
            // Active batches get real input tokens, inactive batches get 0 input tokens.
            inputs.attention_inputs.input_lengths.fill_(0);
            inputs.attention_inputs.prefix_lengths.fill_(prefix_len);
            auto& input_lengths = inputs.attention_inputs.input_lengths;
            for (int b = 0; b < active_bs; b++) {
                int tokens       = (b < active_bs - 1) ? num_tokens_per_bs_ : (seq_len - b * num_tokens_per_bs_);
                input_lengths[b] = tokens;
            }

            // Build cu_seqlens and cu_kv_seqlens as cumulative sums
            auto cu_seqlens_host    = inputs.attention_inputs.cu_seqlens;
            auto cu_kv_seqlens_host = inputs.attention_inputs.cu_kv_seqlens_device.cpu();
            auto prefix_lengths     = inputs.attention_inputs.prefix_lengths;

            cu_seqlens_host[0]    = 0;
            cu_kv_seqlens_host[0] = 0;
            for (int b = 0; b < max_bs_; b++) {
                cu_seqlens_host[b + 1] = cu_seqlens_host[b].item<int>() + input_lengths[b].item<int>();
                cu_kv_seqlens_host[b + 1] =
                    cu_kv_seqlens_host[b].item<int>() + input_lengths[b].item<int>() + prefix_lengths[b].item<int>();
            }

            inputs.attention_inputs.cu_seqlens_device.copy_(cu_seqlens_host);
            inputs.attention_inputs.cu_kv_seqlens_device.copy_(cu_kv_seqlens_host);
            inputs.attention_inputs.input_lengths_device.copy_(input_lengths);
            inputs.attention_inputs.prefix_lengths_device.copy_(prefix_lengths);
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
        // Prefill reshapes common metadata after prepareCaptureInputs synchronized the tag map.
        refreshTaggedAttentionInputs(inputs);
        const int output_capacity           = isGenerativePrefillCudaGraph() ? seq_len : max_bs_ * num_tokens_per_bs_;
        graph_instances_[seq_len].mem_hold_ = createCaptureMemoryHold(inputs, output_capacity);
        graph_instances_[seq_len].mem_hold_.attn_pyobj_ =
            prepareFmhaImpl(graph_instances_[seq_len].mem_hold_.py_model_inputs_, true);
        // HC-shaped MTP draft prefill keeps its output at fixed graph capacity.
        // Other paths produce the real flattened seq_len and must keep their
        // metadata shapes aligned.
        if (!usesFixedCapacityMtpDraftPrefillCudaGraph()) {
            graph_instances_[seq_len].mem_hold_.decoder_layer_hidden_states_ =
                graph_instances_[seq_len].mem_hold_.decoder_layer_hidden_states_.slice(0, 0, seq_len);
        }
        torch::Tensor eager_selfcheck_output;
        if (isGenerativePrefillCudaGraph()) {
            auto eager_outputs = py_forward_method_(graph_instances_[seq_len].mem_hold_.py_model_inputs_,
                                                    graph_instances_[seq_len].mem_hold_.attn_pyobj_)
                                     .cast<PyModelOutputs>();
            eager_selfcheck_output = eager_outputs.hidden_states.clone();
        }
        capturePrefillOneSeqLen(seq_len);
        cuda_graph::finish_capture_session();
        replayAndSyncCheck(seq_len, "seq len");
        // A captured graph is not safe to destroy until its first launch has
        // completed. Keep the dirty guard armed across replay/synchronization;
        // the factory will fail closed and retain graph-owned storage if this
        // phase throws. A numerical self-check failure happens after the stream
        // is drained and remains recoverable.
        capture_session_may_be_dirty_.store(false, std::memory_order_release);
        if (isGenerativePrefillCudaGraph()) {
            const auto& graph_output = graph_instances_[seq_len].mem_hold_.decoder_layer_hidden_states_;
            RTP_LLM_CHECK_WITH_INFO(torch::allclose(graph_output, eager_selfcheck_output, 1e-3, 1e-3),
                                    "prefill CUDA graph startup self-check failed for bucket=%d",
                                    seq_len);
        }
        RTP_LLM_LOG_INFO("capture success for seq_len: %d", seq_len);
    }
    RTP_LLM_LOG_INFO("Capture Prefill End");
}

std::vector<int> CudaGraphRunner::getPrefillSequenceLengthsToCapture() {
    // MTP draft prefill: capture at multiples of num_tokens_per_bs_
    if (isMtpDraftPrefillCudaGraph()) {
        std::vector<int> result;
        for (int i = 1; i <= max_bs_; ++i) {
            result.push_back(i * num_tokens_per_bs_);
        }
        RTP_LLM_LOG_INFO(
            "Draft model prefill: capture seq_lens at %d intervals, %zu total (max_bs=%d, num_tokens_per_bs=%d)",
            num_tokens_per_bs_,
            result.size(),
            max_bs_,
            num_tokens_per_bs_);
        return result;
    }

    // Embedding model prefill: use Python-provided capture seq_lens
    RTP_LLM_CHECK_WITH_INFO(!prefill_capture_seq_lens_.empty(),
                            "prefill_capture_seq_lens_ must be provided from Python and cannot be empty");

    RTP_LLM_LOG_INFO("Using prefill capture sequence lengths from Python: %zu lengths",
                     prefill_capture_seq_lens_.size());

    // Sort and remove duplicates
    std::vector<int> result = prefill_capture_seq_lens_;
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    // A generative-prefill bucket is the total token capacity of one graph and
    // is intentionally bounded by the model sequence limit. Embedding prefill
    // predates that role and captures a flattened batch, so its legal capacity
    // is max_bs * tokens_per_batch rather than one request's max_seq_len.
    const int64_t capture_token_limit = isGenerativePrefillCudaGraph() ?
                                            static_cast<int64_t>(max_seq_len_) :
                                            static_cast<int64_t>(max_bs_) * num_tokens_per_bs_;
    RTP_LLM_CHECK_WITH_INFO(result.front() > 0 && result.back() <= capture_token_limit,
                            "prefill CUDA graph buckets must be in [1, capture_token_limit=%ld], got min=%d max=%d",
                            capture_token_limit,
                            result.front(),
                            result.back());

    RTP_LLM_LOG_INFO(
        "Total sequence lengths to capture: %zu (min: %d, max: %d)", result.size(), result.front(), result.back());
    return result;
}

void CudaGraphRunner::capturePrefillOneSeqLen(int seq_len) {
    try {
        captureOneGraphInstance(seq_len, "seq len");
    } catch (const std::exception& e) {
        RTP_LLM_LOG_ERROR("Exception in capturePrefillOneSeqLen for seq_len %d: %s", seq_len, e.what());
        throw;
    } catch (...) {
        RTP_LLM_LOG_ERROR("Unknown exception in capturePrefillOneSeqLen for seq_len %d", seq_len);
        throw;
    }
}

void CudaGraphRunner::replayPrefill(int seq_len) {
    replayGraph(seq_len);
}
}  // namespace rtp_llm
