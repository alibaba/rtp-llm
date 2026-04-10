#include "rtp_llm/cpp/cuda_graph/cuda_graph_prefill_runner.h"

#include "rtp_llm/cpp/cuda_graph/cuda_graph_py_model_inputs.h"

#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"

using namespace torch_ext;
namespace rtp_llm {

CudaGraphPrefillRunner::CudaGraphPrefillRunner(GraphParams graph_params, py::object py_instance):
    CudaGraphRunnerBase(std::move(py_instance), graph_params, graph_params.max_context_batch_size, true) {
    RTP_LLM_CHECK_WITH_INFO(graph_params_.is_prefill_cuda_graph_mode,
                            "CudaGraphPrefillRunner requires is_prefill_cuda_graph_mode");
}

void CudaGraphPrefillRunner::capturePrefill() {
    RTP_LLM_LOG_INFO("Capture Prefill Start");
    for (int seq_len : prefill_capture_dispatcher_.captureRange()) {
        graph_instances_.try_emplace(seq_len, graph_params_.enable_cuda_graph_debug_mode);
    }
    const auto& range              = prefill_capture_dispatcher_.captureRange();
    int         capture_range_size = static_cast<int>(range.size());
    for (int i = capture_range_size - 1; i >= 0; i--) {
        int seq_len = range[static_cast<size_t>(i)];
        RTP_LLM_LOG_INFO("capture range for seq len: %d", seq_len);
        PyModelInputs inputs;
        prepareCaptureInputs(inputs, max_bs_, seq_len);
        patchPrefillCaptureInputs(inputs, seq_len);
        setupPrefillCaptureMemoryHold(seq_len, inputs);
        capturePrefillOneSeqLen(seq_len);
        replayAndSyncCheck(seq_len, "seq len");
        RTP_LLM_LOG_INFO("capture success for seq_len: %d", seq_len);
    }
    RTP_LLM_LOG_INFO("Capture Prefill End");
}

void CudaGraphPrefillRunner::capturePrefillOneSeqLen(int seq_len) {
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

void CudaGraphPrefillRunner::replayPrefill(int seq_len) {
    replayGraph(seq_len);
}

void CudaGraphPrefillRunner::prepareInputs(const PyModelInputs& inputs, BatchDescriptor& batch_descriptor) {
    RTP_LLM_PROFILE_SCOPE("cuda_graph.prepareInputs.prefill");
    forward_event_.synchronize();
    PyModelInputs& cap = graph_instances_[batch_descriptor.current_real_graph_seq_len].mem_hold_.py_model_inputs_;
    cuda_graph::CudaGraphCapturePyModelInputs::copyRuntimePyModelIntoCaptureBuffers(
        inputs, cap, batch_descriptor, true, py::none());
}

PyModelOutputs CudaGraphPrefillRunner::forward(const PyModelInputs& inputs, BatchDescriptor& batch_descriptor) {
    PyModelOutputs outputs;
    RTP_LLM_LOG_DEBUG("Replay Start (prefill)");
    prepareInputs(inputs, batch_descriptor);
    {
        RTP_LLM_PROFILE_SCOPE("cuda_graph.forward(replayPrefill)");
        replayPrefill(batch_descriptor.current_real_graph_seq_len);
    }
    outputs.hidden_states =
        graph_instances_[batch_descriptor.current_real_graph_seq_len].mem_hold_.all_layers_output_.slice(
            0, 0, batch_descriptor.current_seq_len);
    forward_event_.record(cuda_graph::graphGetCurrentStream());
    RTP_LLM_LOG_DEBUG("Replay End (prefill)");
    return outputs;
}

bool CudaGraphPrefillRunner::canRun(const PyModelInputs& inputs, BatchDescriptor& batch_descriptor) {
    RTP_LLM_PROFILE_SCOPE("cuda_graph.canRun.prefill");
    if (graph_params_.is_target_verify) {
        return false;
    }

    if (!graph_params_.enable_cuda_graph
        || (inputs.attention_inputs.is_prefill && !graph_params_.is_prefill_cuda_graph_mode)) {
        return false;
    }

    if (!inputs.attention_inputs.kv_cache_kernel_block_id_device_by_group.empty()) {
        const size_t group = inputs.attention_inputs.kv_cache_kernel_block_id_device_by_group.size();
        if (graph_params_.kv_cache_group_num <= 0) {
            RTP_LLM_LOG_WARNING("Hybrid kv cache detected but kv_cache_group_num is not set, fallback to normal run.");
            return false;
        }
        if (group != static_cast<size_t>(graph_params_.kv_cache_group_num)) {
            RTP_LLM_LOG_WARNING("Hybrid kv cache group size mismatch: inputs=%zu, captured=%d, fallback to normal run.",
                                group,
                                graph_params_.kv_cache_group_num);
            return false;
        }
    }

    if (!prefill_capture_dispatcher_.tryGetRealGraphPrefillSeqLen(inputs, batch_descriptor)) {
        return false;
    }
    RTP_LLM_LOG_DEBUG("prefill cuda graph replay seq_len key %d", batch_descriptor.current_real_graph_seq_len);
    return true;
}

int CudaGraphPrefillRunner::getCurrentRealGraphSize(const BatchDescriptor& batch_descriptor) const {
    return batch_descriptor.current_real_graph_seq_len;
}

void CudaGraphPrefillRunner::initCapture() {
    if (graph_params_.enable_cuda_graph) {
        RTP_LLM_LOG_INFO("CUDA graph capture for embedding, num_tokens_per_bs: %d", graph_params_.num_tokens_per_bs);
        initCapturePreamble();
        RTP_LLM_LOG_INFO("initCapture forward post check start for prefill");
        capture_py_model_inputs_.patchForPrefillProbeForward();
        py_forward_method_(capture_py_model_inputs_.sliceForPrefillProbeForward());
        RTP_LLM_LOG_INFO("initCapture forward post check end for prefill");
        capturePrefill();
        logCudaGraphPoolMemory("after_capture");
    } else {
        cuda_graph::CudaGraphCapturePyModelInputs::fillCuSeqlensForCapture(
            capture_py_model_inputs_.memoryHold().py_model_inputs_, max_bs_);
        RTP_LLM_LOG_INFO("CUDA graph capture is not enabled, skipping initialization");
    }
}

void CudaGraphPrefillRunner::buildCaptureDispatcher() {
    prefill_capture_dispatcher_.build(graph_params_);
}

void CudaGraphPrefillRunner::patchPrefillCaptureInputs(PyModelInputs& inputs, int seq_len) {
    inputs.attention_inputs.cu_seqlens.data_ptr<int>()[1]    = seq_len;
    inputs.attention_inputs.cu_kv_seqlens.data_ptr<int>()[1] = seq_len;
    inputs.attention_inputs.input_lengths.data_ptr<int>()[0] = seq_len;
    inputs.attention_inputs.context_total_kv_length          = seq_len;
    inputs.attention_inputs.prefill_cuda_graph_copy_params =
        capture_py_model_inputs_.memoryHold().py_model_inputs_.attention_inputs.prefill_cuda_graph_copy_params;
    if (inputs.bert_embedding_inputs.position_encoding.numel() > 0) {
        inputs.bert_embedding_inputs.combo_position_ids =
            inputs.bert_embedding_inputs.combo_position_ids.slice(0, 0, seq_len);
        inputs.bert_embedding_inputs.combo_tokens_type_ids =
            inputs.bert_embedding_inputs.combo_tokens_type_ids.slice(0, 0, seq_len);
    }
}

void CudaGraphPrefillRunner::setupPrefillCaptureMemoryHold(int seq_len, PyModelInputs& inputs) {
    graph_instances_[seq_len].mem_hold_ =
        createCaptureMemoryHold(inputs, max_bs_ * graph_params_.num_tokens_per_bs);
    graph_instances_[seq_len].mem_hold_.attn_pyobj_ =
        py_attn_pyobj_method_(graph_instances_[seq_len].mem_hold_.py_model_inputs_, true);
    graph_instances_[seq_len].mem_hold_.all_layers_output_ =
        graph_instances_[seq_len].mem_hold_.all_layers_output_.slice(0, 0, seq_len);
}

}  // namespace rtp_llm
