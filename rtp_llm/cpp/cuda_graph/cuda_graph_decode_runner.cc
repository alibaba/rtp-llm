#include "rtp_llm/cpp/cuda_graph/cuda_graph_decode_runner.h"

#include "rtp_llm/cpp/cuda_graph/cuda_graph_py_model_inputs.h"

#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"

using namespace torch_ext;
namespace rtp_llm {

CudaGraphDecodeRunner::CudaGraphDecodeRunner(GraphParams graph_params, py::object py_instance):
    CudaGraphRunnerBase(std::move(py_instance), graph_params, graph_params.concurrency_limit, false) {
    RTP_LLM_CHECK_WITH_INFO(!graph_params_.is_prefill_cuda_graph_mode,
                            "CudaGraphDecodeRunner requires !is_prefill_cuda_graph_mode");
}

void CudaGraphDecodeRunner::replayDecode(int bs) {
    replayGraph(bs);
}

void CudaGraphDecodeRunner::captureDecodeOneBatchSize(int bs) {
    captureOneGraphInstance(bs, "batch size");
}

void CudaGraphDecodeRunner::captureDecode() {
    RTP_LLM_LOG_INFO("Capture Decode Start");
    for (int bs : decode_capture_dispatcher_.captureRange()) {
        graph_instances_.try_emplace(bs, graph_params_.enable_cuda_graph_debug_mode);
    }
    const auto& range              = decode_capture_dispatcher_.captureRange();
    int         capture_range_size = static_cast<int>(range.size());
    for (int i = capture_range_size - 1; i >= 0; i--) {
        int           bs = range[static_cast<size_t>(i)];
        PyModelInputs inputs;
        prepareCaptureInputs(inputs, bs, bs * graph_params_.num_tokens_per_bs);

        int max_input_len  = inputs.attention_inputs.input_lengths.max().item<int>();
        int max_prefix_len = 0;
        if (inputs.attention_inputs.prefix_lengths.defined()) {
            max_prefix_len = inputs.attention_inputs.prefix_lengths.max().item<int>();
        }
        inputs.attention_inputs.context_total_kv_length = bs * (max_input_len + max_prefix_len);

        graph_instances_[bs].mem_hold_ = createCaptureMemoryHold(inputs, bs * graph_params_.num_tokens_per_bs);
        graph_instances_[bs].mem_hold_.attn_pyobj_ =
            py_attn_pyobj_method_(graph_instances_[bs].mem_hold_.py_model_inputs_, true);
        captureDecodeOneBatchSize(bs);
        replayAndSyncCheck(bs, "batch size");
        RTP_LLM_LOG_INFO("capture success for batch size: %d", bs);
    }
    RTP_LLM_LOG_INFO("Capture Decode End");
}

void CudaGraphDecodeRunner::prepareInputs(const PyModelInputs& inputs, BatchDescriptor& batch_descriptor) {
    RTP_LLM_PROFILE_SCOPE("cuda_graph.prepareInputs.decode");
    forward_event_.synchronize();
    PyModelInputs& cap         = graph_instances_[batch_descriptor.current_real_graph_bs].mem_hold_.py_model_inputs_;
    py::object     decode_attn = graph_instances_[batch_descriptor.current_real_graph_bs].mem_hold_.attn_pyobj_;
    cuda_graph::CudaGraphCapturePyModelInputs::copyRuntimePyModelIntoCaptureBuffers(
        inputs, cap, batch_descriptor, false, decode_attn);
}

PyModelOutputs CudaGraphDecodeRunner::forward(const PyModelInputs& inputs, BatchDescriptor& batch_descriptor) {
    PyModelOutputs outputs;
    RTP_LLM_LOG_DEBUG("Replay Start (decode)");
    prepareInputs(inputs, batch_descriptor);
    {
        RTP_LLM_PROFILE_SCOPE("cuda_graph.forward(replayDecode)");
        replayDecode(batch_descriptor.current_real_graph_bs);
    }
    outputs.hidden_states = graph_instances_[batch_descriptor.current_real_graph_bs].mem_hold_.all_layers_output_.slice(
        0, 0, batch_descriptor.seq_len_sum);
    forward_event_.record(cuda_graph::graphGetCurrentStream());
    RTP_LLM_LOG_DEBUG("Replay End (decode)");
    return outputs;
}

bool CudaGraphDecodeRunner::canRun(const PyModelInputs& inputs, BatchDescriptor& batch_descriptor) {
    RTP_LLM_PROFILE_SCOPE("cuda_graph.canRun.decode");
    if (graph_params_.is_target_verify) {
        if (inputs.attention_inputs.is_target_verify) {
            return decode_capture_dispatcher_.tryGetRealGraphDecodeBatchSize(inputs, batch_descriptor);
        }
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

    if (!decode_capture_dispatcher_.tryGetRealGraphDecodeBatchSize(inputs, batch_descriptor)) {
        return false;
    }
    return true;
}

int CudaGraphDecodeRunner::getCurrentRealGraphSize(const BatchDescriptor& batch_descriptor) const {
    return batch_descriptor.current_real_graph_bs;
}

void CudaGraphDecodeRunner::initCapture() {
    if (graph_params_.enable_cuda_graph) {
        initCapturePreamble();
        captureDecode();
        logCudaGraphPoolMemory("after_capture");
    } else {
        cuda_graph::CudaGraphCapturePyModelInputs::fillCuSeqlensForCapture(
            capture_py_model_inputs_.memoryHold().py_model_inputs_, max_bs_);
        RTP_LLM_LOG_INFO("CUDA graph capture is not enabled, skipping initialization");
    }
}

void CudaGraphDecodeRunner::buildCaptureDispatcher() {
    decode_capture_dispatcher_.build(graph_params_, max_bs_);
}

}  // namespace rtp_llm
