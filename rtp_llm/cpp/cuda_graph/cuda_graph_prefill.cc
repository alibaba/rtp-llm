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
        const bool draft_prefill_graph_mode  = isMtpDraftPrefillCudaGraph();
        const bool draft_prefill_full_capacity = usesFixedCapacityMtpDraftPrefillCudaGraph();
        const int  active_bs = draft_prefill_graph_mode ? (seq_len + num_tokens_per_bs_ - 1) / num_tokens_per_bs_ :
                                                          static_cast<int>(max_bs_);
        const int  capture_batch_size =
            (draft_prefill_graph_mode && !draft_prefill_full_capacity) ? active_bs : static_cast<int>(max_bs_);
        // for attention, it always run the max_bs, so when we run `forward`, the real batch size is not sure
        // we will transfer a `batch size tensor(int)` for `copy kernel`.
        // Prepare common inputs using shared function
        prepareCaptureInputs(inputs, capture_batch_size, seq_len);
        // Generation prefill uses Bmax real slots plus one fixed sentinel slot.
        // Capture the conservative layout [0, ..., 0, Tg] so FMHA/RoPE fix
        // batch size and max_q_len at their profile capacity. All sentinel KV
        // writes alias allocator-reserved block 0; its output is discarded.
        if (isGenerationPrefillCudaGraph()) {
            RTP_LLM_CHECK_WITH_INFO(max_bs_ == static_cast<size_t>(generation_prefill_cuda_graph_max_requests_ + 1),
                                    "generation prefill CUDA graph backend capacity mismatch: max_bs=%zu Bmax=%d",
                                    max_bs_,
                                    generation_prefill_cuda_graph_max_requests_);
            inputs.attention_inputs.input_lengths.zero_();
            inputs.attention_inputs.input_lengths[generation_prefill_cuda_graph_max_requests_] = seq_len;
            inputs.attention_inputs.prefix_lengths.zero_();
            inputs.attention_inputs.cu_seqlens.zero_();
            inputs.attention_inputs.cu_seqlens[max_bs_] = seq_len;
            inputs.attention_inputs.padding_offset.fill_(generation_prefill_cuda_graph_max_requests_ * seq_len);
            inputs.attention_inputs.input_lengths_device.copy_(inputs.attention_inputs.input_lengths, false);
            inputs.attention_inputs.prefix_lengths_device.zero_();
            inputs.attention_inputs.cu_seqlens_device.copy_(inputs.attention_inputs.cu_seqlens, false);
            inputs.attention_inputs.cu_kv_seqlens_device.copy_(inputs.attention_inputs.cu_seqlens, false);

            auto install_dummy_rows = [](PyAttentionInputs& attn_inputs) {
                attn_inputs.kv_cache_kernel_block_id.zero_();
                attn_inputs.kv_cache_kernel_block_id_device.zero_();
            };
            if (inputs.attention_inputs_by_tag.empty()) {
                install_dummy_rows(inputs.attention_inputs);
            } else {
                for (const auto& tag : kv_cache_group_tags_) {
                    install_dummy_rows(inputs.attention_inputs_by_tag.at(tag));
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
            // Compact draft prefill captures only active requests so attention sees the same
            // token and batch shape as eager. Fixed-capacity models explicitly keep max_bs_.
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
            for (int b = 0; b < capture_batch_size; b++) {
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
        if (draft_prefill_graph_mode && !draft_prefill_full_capacity) {
            // Each compact draft-prefill graph has an exact token and batch shape.
            // Running the max-batch copy path would expand Q beyond cu_seqlens[-1].
            inputs.attention_inputs.prefill_cuda_graph_copy_params.reset();
        } else {
            inputs.attention_inputs.prefill_cuda_graph_copy_params =
                capture_mem_hold_.py_model_inputs_.attention_inputs.prefill_cuda_graph_copy_params;
        }
        if (isGenerationPrefillCudaGraph()) {
            if (inputs.bert_embedding_inputs.combo_position_ids.defined()) {
                inputs.bert_embedding_inputs.combo_position_ids =
                    inputs.bert_embedding_inputs.combo_position_ids.slice(0, 0, seq_len);
            }
            if (inputs.bert_embedding_inputs.combo_tokens_type_ids.defined()) {
                inputs.bert_embedding_inputs.combo_tokens_type_ids =
                    inputs.bert_embedding_inputs.combo_tokens_type_ids.slice(0, 0, seq_len);
            }
        } else if (inputs.bert_embedding_inputs.position_encoding.numel() > 0) {
            inputs.bert_embedding_inputs.combo_position_ids =
                inputs.bert_embedding_inputs.combo_position_ids.slice(0, 0, seq_len);
            inputs.bert_embedding_inputs.combo_tokens_type_ids =
                inputs.bert_embedding_inputs.combo_tokens_type_ids.slice(0, 0, seq_len);
        }
        // Prefill reshapes common metadata after prepareCaptureInputs synchronized the tag map.
        refreshTaggedAttentionInputs(inputs);
        const int output_capacity           = isGenerationPrefillCudaGraph() ? seq_len : max_bs_ * num_tokens_per_bs_;
        graph_instances_[seq_len].mem_hold_ = createCaptureMemoryHold(inputs, output_capacity);
        graph_instances_[seq_len].mem_hold_.attn_pyobj_ =
            prepareFmhaImpl(graph_instances_[seq_len].mem_hold_.py_model_inputs_, true);
        // HC-shaped MTP draft prefill keeps its output at fixed graph capacity.
        // Other paths produce the real flattened seq_len and must keep their
        // metadata shapes aligned.
        if (!draft_prefill_full_capacity) {
            graph_instances_[seq_len].mem_hold_.decoder_layer_hidden_states_ =
                graph_instances_[seq_len].mem_hold_.decoder_layer_hidden_states_.slice(0, 0, seq_len);
        }
        capturePrefillOneSeqLen(seq_len);
        cuda_graph::finish_capture_session();
        replayAndSyncCheck(seq_len, "seq len");
        // Generation-prefill capture uses a discarded sentinel sequence whose
        // logical KV pages intentionally alias reserved block 0. For prompts
        // longer than one block, concurrent writes make that dummy output an
        // invalid numerical oracle. Capture/replay exceptions still fail fast;
        // real-request graph numerics and KV writes are covered by GPU tests.
        // A captured graph is not safe to destroy until its first launch has
        // completed. Keep the dirty guard armed across replay/synchronization;
        // the factory will fail closed and retain graph-owned storage if this
        // phase throws. Once the stream is drained, normal cleanup is safe.
        capture_session_may_be_dirty_.store(false, std::memory_order_release);
        RTP_LLM_LOG_INFO("capture success for seq_len: %d", seq_len);
    }
    RTP_LLM_LOG_INFO("Capture Prefill End");
}

std::vector<int> CudaGraphRunner::getPrefillSequenceLengthsToCapture() {
    // Reuse decode buckets instead of retaining one FlashInfer workspace for
    // every draft-prefill batch size. Compact draft models replay exact
    // buckets only; non-bucket batches fall back to eager execution. Models
    // that explicitly keep full token capacity may pad to the next bucket.
    // Always include max_bs_ so the configured concurrency is capturable.
    if (isMtpDraftPrefillCudaGraph()) {
        std::vector<int> capture_batch_sizes;
        capture_batch_sizes.reserve(decode_capture_batch_sizes_.size() + 1);
        for (const int batch_size : decode_capture_batch_sizes_) {
            if (batch_size > 0 && batch_size <= max_bs_) {
                capture_batch_sizes.push_back(batch_size);
            }
        }
        capture_batch_sizes.push_back(max_bs_);
        std::sort(capture_batch_sizes.begin(), capture_batch_sizes.end());
        capture_batch_sizes.erase(std::unique(capture_batch_sizes.begin(), capture_batch_sizes.end()),
                                  capture_batch_sizes.end());

        std::vector<int> result;
        result.reserve(capture_batch_sizes.size());
        for (const int batch_size : capture_batch_sizes) {
            result.push_back(batch_size * num_tokens_per_bs_);
        }
        RTP_LLM_LOG_INFO("Draft model prefill: capture %zu batch buckets (max_bs=%d, num_tokens_per_bs=%d)",
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
    if (isGenerationPrefillCudaGraph()) {
        // Generation-prefill buckets describe total token capacity. Do not
        // impose this new limit on legacy embedding/MTP capture configurations.
        RTP_LLM_CHECK_WITH_INFO(result.front() > 0 && result.back() <= max_seq_len_,
                                "generation prefill CUDA graph buckets must be in [1, max_seq_len=%d], "
                                "got min=%d max=%d",
                                max_seq_len_,
                                result.front(),
                                result.back());
    }

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
