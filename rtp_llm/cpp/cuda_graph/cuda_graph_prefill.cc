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
        // Prefill-specific settings, one the first seq is valid, the post ones are all empty
        if (isEmbeddingStylePrefillCudaGraph()) {
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
        } else if (isGenerativePrefillCudaGraph()) {
            // Exact-shape single-request prompt graph. Capture against the
            // longest legal prefix so the paged-attention and recurrent cache
            // paths allocate their maximum metadata, while only seq_len query
            // rows execute. Replay replaces lengths and block tables.
            const int prefix_len = std::max(max_seq_len_ - seq_len, 0);
            inputs.attention_inputs.input_lengths.fill_(0);
            inputs.attention_inputs.prefix_lengths.fill_(0);
            inputs.attention_inputs.input_lengths[0]  = seq_len;
            inputs.attention_inputs.prefix_lengths[0] = prefix_len;

            auto cu_seqlens_host    = inputs.attention_inputs.cu_seqlens;
            auto cu_kv_seqlens_host = inputs.attention_inputs.cu_kv_seqlens_device.cpu();
            cu_seqlens_host.fill_(seq_len);
            cu_kv_seqlens_host.fill_(prefix_len + seq_len);
            cu_seqlens_host[0]    = 0;
            cu_kv_seqlens_host[0] = 0;
            inputs.attention_inputs.cu_seqlens_device.copy_(cu_seqlens_host);
            inputs.attention_inputs.cu_kv_seqlens_device.copy_(cu_kv_seqlens_host);
            inputs.attention_inputs.input_lengths_device.copy_(inputs.attention_inputs.input_lengths);
            inputs.attention_inputs.prefix_lengths_device.copy_(inputs.attention_inputs.prefix_lengths);
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

        inputs.attention_inputs.context_total_kv_length =
            isGenerativePrefillCudaGraph() ? max_seq_len_ : seq_len;
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
        // Every output owned by a graph key must use the same geometry as the
        // model forward. Generic HC-shaped MTP keeps fixed max capacity;
        // embedding and phase-specific DSpARK graphs produce the real
        // flattened token count. This applies to hidden states and graph-native
        // proposal ids alike -- mixing max-capacity proposal storage with a
        // B*width model output makes non-max graph keys impossible to capture.
        const int output_tokens = usesFixedCapacityMtpDraftPrefillCudaGraph() ?
                                      static_cast<int>(max_bs_) * num_tokens_per_bs_ :
                                      seq_len;
        graph_instances_[seq_len].mem_hold_ = createCaptureMemoryHold(inputs, output_tokens);
        graph_instances_[seq_len].mem_hold_.attn_pyobj_ =
            py_attn_pyobj_method_(graph_instances_[seq_len].mem_hold_.py_model_inputs_, true);
        cacheAttentionMetadataCapability(graph_instances_[seq_len].mem_hold_);
        capturePrefillOneSeqLen(seq_len);
        cuda_graph::finish_capture_session();
        replayAndSyncCheck(seq_len, "seq len");
        RTP_LLM_LOG_INFO("capture success for seq_len: %d", seq_len);
    }
    RTP_LLM_LOG_INFO("Capture Prefill End");
}

std::vector<int> CudaGraphRunner::getPrefillSequenceLengthsToCapture() {
    if (isGenerativePrefillCudaGraph()) {
        std::vector<int> result;
        result.reserve(prefill_capture_seq_lens_.size());
        for (const int seq_len : prefill_capture_seq_lens_) {
            if (seq_len > 0 && seq_len <= num_tokens_per_bs_) {
                result.push_back(seq_len);
            }
        }
        std::sort(result.begin(), result.end());
        result.erase(std::unique(result.begin(), result.end()), result.end());
        RTP_LLM_CHECK_WITH_INFO(!result.empty(),
                                "generative prefill graph needs at least one exact capture key <= %d",
                                num_tokens_per_bs_);
        RTP_LLM_LOG_INFO("Generative prompt prefill: capture %zu exact token keys (max=%d, batch=1)",
                         result.size(),
                         result.back());
        return result;
    }

    // MTP draft prefill: capture at multiples of num_tokens_per_bs_
    if (isMtpDraftPrefillCudaGraph()) {
        // DSpARK proposal/commit execute every captured B*width row. In
        // particular, commit performs KV projection/RoPE/scatter without an
        // attention-side active-row mask. Rounding B up to a sparse decode
        // bucket would therefore make the padded rows scatter through cache
        // metadata that belongs to no stream. Preserve the validated DSpARK
        // contract: one exact fixed-width graph for every serving batch.
        if (dspark_call_phase_ != DSparkCallPhase::NONE) {
            std::vector<int> result;
            result.reserve(max_bs_);
            for (int batch_size = 1; batch_size <= static_cast<int>(max_bs_); ++batch_size) {
                result.push_back(batch_size * num_tokens_per_bs_);
            }
            RTP_LLM_LOG_INFO(
                "DSpARK phase %d: capture %zu exact-batch fixed-width prefill token keys "
                "(max_bs=%zu, query_width=%d)",
                static_cast<int>(dspark_call_phase_),
                result.size(),
                max_bs_,
                num_tokens_per_bs_);
            return result;
        }

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
