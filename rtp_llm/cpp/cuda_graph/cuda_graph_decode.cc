#include "rtp_llm/cpp/cuda_graph/cuda_graph_runner.h"
#include "rtp_llm/cpp/cuda_graph/cuda_graph_device_shims.h"

namespace rtp_llm {
void CudaGraphRunner::replayDecode(int bs) {
    replayGraph(bs);
}

void CudaGraphRunner::captureDraftLoop(int bs) {
#if USING_CUDA && ((TORCH_VERSION_MAJOR > 2) || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 8))
    if (draft_decode_steps_ == 0) {
        return;
    }
    auto& instance = graph_instances_.at(bs);
    auto& hold     = instance.mem_hold_;
    auto& inputs   = hold.py_model_inputs_;
    RTP_LLM_CHECK_WITH_INFO(hold.device_attention_preparation_ && num_tokens_per_bs_ == 1,
                            "draft loop requires graph-owned device attention preparation");
    PyModelOutputs outputs(hold.decoder_layer_hidden_states_, hold.mtp_target_hidden_states_);
    const auto&    next_hidden =
        outputs.mtp_target_hidden_states.defined() ? outputs.mtp_target_hidden_states : outputs.hidden_states;
    RTP_LLM_CHECK_WITH_INFO(inputs.input_hiddens.sizes() == next_hidden.sizes(),
                            "draft loop hidden-state ABI mismatch");
    instance.draft_token_ids_ = torch::empty({bs, draft_decode_steps_}, options_cuda_int32_);
    // Initialize head/sampler kernels before capture. The model forward has
    // already been warmed and replayed; sampling does not mutate its caches.
    draft_sampler_(inputs, outputs);
    draft_sampler_(inputs, outputs);
    cuda_graph::graphDeviceSynchronize();

    CudaGraphStreamLife stream_life(capture_stream_);
    instance.draft_loop_graph_ = std::make_unique<at::cuda::CUDAGraph>();
    auto& loop                 = *instance.draft_loop_graph_;
    capture_session_may_be_dirty_.store(true, std::memory_order_release);
    // Do not share this pool with the child: the child's intermediate storage
    // remains live while the head and inter-step operations execute.
    loop.capture_begin();
    {
        cuda_graph::GraphNcclCaptureContext capture_ctx;
        CudaGraphCaptureGuard               capture_guard(&capture_ctx);
        for (int step = 0; step < draft_decode_steps_; ++step) {
            cudaStreamCaptureStatus status;
            cudaGraph_t             parent;
            const cudaGraphNode_t*  dependencies;
            size_t                  count;
            const auto              stream = capture_stream_.stream();
#if CUDA_VERSION >= 13000
            AT_CUDA_CHECK(cudaStreamGetCaptureInfo(stream, &status, nullptr, &parent, &dependencies, nullptr, &count));
#else
            AT_CUDA_CHECK(cudaStreamGetCaptureInfo_v2(stream, &status, nullptr, &parent, &dependencies, &count));
#endif
            RTP_LLM_CHECK_WITH_INFO(status == cudaStreamCaptureStatusActive, "draft loop capture is not active");
            cudaGraphNode_t child;
            AT_CUDA_CHECK(
                cudaGraphAddChildGraphNode(&child, parent, dependencies, count, instance.graph_.raw_cuda_graph()));
#if CUDA_VERSION >= 13000
            AT_CUDA_CHECK(
                cudaStreamUpdateCaptureDependencies(stream, &child, nullptr, 1, cudaStreamSetCaptureDependencies));
#else
            AT_CUDA_CHECK(cudaStreamUpdateCaptureDependencies(stream, &child, 1, cudaStreamSetCaptureDependencies));
#endif
            auto tokens = draft_sampler_(inputs, outputs).reshape({bs});
            instance.draft_token_ids_.select(1, step).copy_(tokens);
            if (step + 1 < draft_decode_steps_) {
                inputs.input_ids.copy_(tokens);
                inputs.input_hiddens.copy_(next_hidden);
                inputs.attention_inputs.sequence_lengths_plus_1_device.add_(1);
                if (inputs.combo_position_ids.defined() && inputs.combo_position_ids.numel() > 0) {
                    inputs.combo_position_ids.add_(1);
                }
            }
        }
        // Preserve the base geometry for subsequent preparation/inspection.
        inputs.attention_inputs.sequence_lengths_plus_1_device.sub_(draft_decode_steps_ - 1);
        if (inputs.combo_position_ids.defined() && inputs.combo_position_ids.numel() > 0) {
            inputs.combo_position_ids.sub_(draft_decode_steps_ - 1);
        }
        loop.capture_end();
    }
    cuda_graph::finish_capture_session();
    loop.replay();
    cuda_graph::graphDeviceSynchronize();
    capture_session_may_be_dirty_.store(false, std::memory_order_release);
    RTP_LLM_LOG_INFO("Captured complete MTP draft loop: batch=%d steps=%d", bs, draft_decode_steps_);
#endif
}

std::vector<int> CudaGraphRunner::getDecodeBatchSizesToCapture() {
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
    // Side-cache models cannot replay a padded batch. Capture every small
    // batch so mixed low-concurrency traffic does not fall back to eager for
    // sizes 2..7. Keep the existing sparse buckets above eight to bound graph
    // memory; deployments needing larger exact batches can set an explicit
    // decode_capture_config.
    if (exact_batch_only_) {
        for (int i = 2; i <= std::min(7, max_generate_batch_size); ++i) {
            capture_bs.push_back(i);
        }
    }
    // Add key batch sizes up to 32
    for (int i : {1, 2, 4, 8, 16, 24, 32}) {
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
    std::sort(capture_bs.begin(), capture_bs.end());
    capture_bs.erase(std::unique(capture_bs.begin(), capture_bs.end()), capture_bs.end());
    return capture_bs;
}

void CudaGraphRunner::captureDecodeOneBatchSize(int bs) {
    captureOneGraphInstance(bs, "batch size");
}

void CudaGraphRunner::captureDecode() {
    RTP_LLM_LOG_INFO("Capture Decode Start");
    // Pre-initialize all graph instances with keep_graph based on debug mode
    for (int bs : capture_range_) {
        graph_instances_.try_emplace(bs, enable_cuda_graph_debug_mode_ || draft_decode_steps_ > 0);
    }
    int capture_range_size = capture_range_.size();
    for (int i = capture_range_size - 1; i >= 0; i--) {
        int           bs = capture_range_[i];
        PyModelInputs inputs;
        // Prepare common inputs using shared function
        prepareCaptureInputs(inputs, bs, bs * num_tokens_per_bs_);

        // calculate context_total_kv_length
        int max_input_len  = inputs.attention_inputs.input_lengths.max().item<int>();
        int max_prefix_len = 0;
        if (inputs.attention_inputs.prefix_lengths.defined() && inputs.attention_inputs.prefix_lengths.numel() > 0) {
            max_prefix_len = inputs.attention_inputs.prefix_lengths.max().item<int>();
        }
        inputs.attention_inputs.context_total_kv_length = bs * (max_input_len + max_prefix_len);
        // capture-specific metadata above was written after prepareCaptureInputs synchronized the tag map.
        refreshTaggedAttentionInputs(inputs);

        graph_instances_[bs].mem_hold_ = createCaptureMemoryHold(inputs, bs * num_tokens_per_bs_);
        graph_instances_[bs].mem_hold_.attn_pyobj_ =
            prepareFmhaImpl(graph_instances_[bs].mem_hold_.py_model_inputs_, true);
        graph_instances_[bs].mem_hold_.device_attention_preparation_ =
            supportsDeviceAttentionPreparation(graph_instances_[bs].mem_hold_.attn_pyobj_);
        captureDecodeOneBatchSize(bs);
        cuda_graph::finish_capture_session();
        replayAndSyncCheck(bs, "batch size");
        // Retain graph-owned buffers until the first replay is known complete.
        // A launch/synchronization failure must take the same fail-closed path
        // as a partially captured graph instead of deleting in-flight storage.
        capture_session_may_be_dirty_.store(false, std::memory_order_release);
        captureDraftLoop(bs);
        RTP_LLM_LOG_INFO("capture success for batch size: %d (role=%d, device_attention_preparation=%d)",
                         bs,
                         static_cast<int>(role_),
                         static_cast<int>(graph_instances_[bs].mem_hold_.device_attention_preparation_));
    }
    RTP_LLM_LOG_INFO("Capture Decode End");
}
}  // namespace rtp_llm
