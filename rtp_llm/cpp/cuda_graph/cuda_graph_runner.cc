#include "rtp_llm/cpp/cuda_graph/cuda_graph_runner.h"

#include "rtp_llm/cpp/cuda_graph/cuda_graph_py_model_inputs.h"

#include <algorithm>
#include <string>

#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"

using namespace torch_ext;
namespace rtp_llm {

// CUDA graph execution is split into CudaGraphPrefillRunner vs CudaGraphDecodeRunner (no runtime
// is_prefill_cuda_graph_mode branching in forward/canRun). Shared capture/replay lives in
// CudaGraphRunnerShared. Legacy configuration reference:
// +--------------------------------+-----------------------------+--------------------------------------+
// | Model Type                     | is_prefill_cuda_graph_mode  | num_tokens_per_bs                    |
// +--------------------------------+-----------------------------+--------------------------------------+
// | Embedding Model (prefill)      | true                        | max_seq_len                          |
// | Normal / spec target (decode)  | false                       | 1 or gen_num_per_cycle + 1           |
// +--------------------------------+-----------------------------+--------------------------------------+

// --- CudaGraphRunnerShared ---

CudaGraphRunnerShared::CudaGraphRunnerShared(CudaGraphRunnerBase& owner,
                                             GraphParams          graph_params,
                                             size_t               max_bs,
                                             bool                 is_prefill_capture):
    owner_(owner),
    graph_params_(std::move(graph_params)),
    capture_stream_(cuda_graph::graphGetStreamFromPool(true)),
    max_bs_(max_bs),
    is_prefill_capture_(is_prefill_capture) {
    py::gil_scoped_acquire gil;
    if (!owner_.py_instance_ || owner_.py_instance_.is_none()) {
        throw std::runtime_error("CudaGraphRunner: Python instance is null or none.");
    }
    if (graph_params_.kernel_tokens_per_block <= 0) {
        throw std::runtime_error("CudaGraphRunner: kernel_tokens_per_block must be > 0.");
    }
    py_attn_pyobj_method_ = owner_.py_instance_.attr("prepare_fmha_impl");
    py_forward_method_    = owner_.py_instance_.attr("forward");
    RTP_LLM_LOG_INFO("Initialize CUDA graph runner (%s) with parameters below: \n \
            enable_cuda_graph: %d, max_bs_: %zu, enable_cuda_graph_debug_mode: %d, max_seq_len: %d, kernel_tokens_per_block: %d, \
            hidden_size: %zu, num_tokens_per_bs: %d, is_target_verify: %d",
                     is_prefill_capture_ ? "prefill" : "decode",
                     graph_params_.enable_cuda_graph,
                     max_bs_,
                     graph_params_.enable_cuda_graph_debug_mode,
                     graph_params_.max_seq_len,
                     graph_params_.kernel_tokens_per_block,
                     graph_params_.hidden_size,
                     graph_params_.num_tokens_per_bs,
                     graph_params_.is_target_verify);
}

void CudaGraphRunnerShared::initCapturePreamble() {
    RTP_LLM_LOG_INFO("CUDA graph capture is enabled");
    shared_graph_pool_ = cuda_graph::graphPoolHandle();
    max_num_token_     = max_bs_ * graph_params_.num_tokens_per_bs;
    capture_dispatcher_.build(graph_params_, max_bs_, is_prefill_capture_);

    capture_py_model_inputs_ = cuda_graph::CudaGraphCapturePyModelInputs(
        graph_params_, max_bs_, max_num_token_, position_encoding_, token_type_embedding_, input_embedding_scalar_);
    capture_py_model_inputs_.buildCaptureMemoryHold();

    auto attn_pyobj = py_attn_pyobj_method_(capture_py_model_inputs_.memoryHold().py_model_inputs_, true);
    RTP_LLM_LOG_INFO("initCapture forward for output datatype start");
    py_forward_method_(capture_py_model_inputs_.memoryHold().py_model_inputs_, attn_pyobj);
    RTP_LLM_LOG_INFO("initCapture forward for output datatype end");
    capture_py_model_inputs_.allocateHiddenStatesAndPrefillCopyParams();
    logCudaGraphPoolMemory("before_capture");
}

void CudaGraphRunnerShared::logCudaGraphPoolMemory(const char* phase) {
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

void CudaGraphRunnerShared::captureOneGraphInstance(int key, const char* key_type) {
    auto inputs = graph_instances_[key].mem_hold_.py_model_inputs_;

    size_t pre_capture_reserved = cuda_graph::graphReservedBytes();

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
        cuda_graph::graphDeviceSynchronize();

        CudaGraphStreamGuard stream_guard(capture_stream_);
        auto&                graph               = graph_instances_[key].graph_;
        std::string          output_dot_filename = "";
        if (graph_params_.enable_cuda_graph_debug_mode) {
            graph.enable_debug_mode();
            std::string key_type_str = std::string(key_type);
            std::replace(key_type_str.begin(), key_type_str.end(), ' ', '_');
            output_dot_filename = "cuda_graph_tokens" + std::to_string(graph_params_.num_tokens_per_bs) + "_"
                                  + key_type_str + "_" + std::to_string(key) + "_visualization.dot";
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
                throw;
            }
            graph_instances_[key].mem_hold_.all_layers_output_.copy_(outputs.hidden_states);
            graph.capture_end();
        }

        if (graph_params_.enable_cuda_graph_debug_mode) {
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

void CudaGraphRunnerShared::replayGraph(int key) {
    graph_instances_[key].graph_.replay();
}

void CudaGraphRunnerShared::replayAndSyncCheck(int key, const char* key_type) {
    RTP_LLM_LOG_INFO("replay start check for %s %d", key_type, key);
    replayGraph(key);
    cuda_graph::graphDeviceSynchronize();
    RTP_LLM_LOG_INFO("replay end check for %s %d", key_type, key);
}

void CudaGraphRunnerShared::prepareCaptureInputs(PyModelInputs& inputs, int batch_size, int seq_len_or_tokens) {
    cuda_graph::CudaGraphCapturePyModelInputs::sliceTemplatePyModelInputsForCapture(
        inputs,
        capture_py_model_inputs_.memoryHold().py_model_inputs_,
        batch_size,
        seq_len_or_tokens,
        is_prefill_capture_,
        graph_params_.num_tokens_per_bs,
        graph_params_.is_target_verify);
}

CaptureMemoryHold CudaGraphRunnerShared::createCaptureMemoryHold(PyModelInputs& inputs, int tokens_count) {
    return CaptureMemoryHold(capture_py_model_inputs_.memoryHold().all_layers_output_.slice(0, 0, tokens_count),
                             inputs);
}

// --- CudaGraphPrefillRunner ---

CudaGraphPrefillRunner::CudaGraphPrefillRunner(GraphParams graph_params, py::object py_instance):
    CudaGraphRunnerBase(std::move(py_instance)),
    CudaGraphRunnerShared(*this, graph_params, graph_params.max_context_batch_size, true) {
    RTP_LLM_CHECK_WITH_INFO(graph_params_.is_prefill_cuda_graph_mode,
                            "CudaGraphPrefillRunner requires is_prefill_cuda_graph_mode");
}

void CudaGraphPrefillRunner::setPositionEncoding(torch::Tensor position_encoding) {
    position_encoding_ = std::move(position_encoding);
}

void CudaGraphPrefillRunner::setTokenTypeEmbedding(torch::Tensor token_type_embedding) {
    token_type_embedding_ = std::move(token_type_embedding);
}

void CudaGraphPrefillRunner::setInputEmbeddingScalar(float input_embedding_scalar) {
    input_embedding_scalar_ = input_embedding_scalar;
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

    if (!capture_dispatcher_.tryGetRealGraphPrefillSeqLen(inputs, batch_descriptor)) {
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

// --- CudaGraphDecodeRunner ---

CudaGraphDecodeRunner::CudaGraphDecodeRunner(GraphParams graph_params, py::object py_instance):
    CudaGraphRunnerBase(std::move(py_instance)),
    CudaGraphRunnerShared(*this, graph_params, graph_params.concurrency_limit, false) {
    RTP_LLM_CHECK_WITH_INFO(!graph_params_.is_prefill_cuda_graph_mode,
                            "CudaGraphDecodeRunner requires !is_prefill_cuda_graph_mode");
}

void CudaGraphDecodeRunner::setPositionEncoding(torch::Tensor position_encoding) {
    position_encoding_ = std::move(position_encoding);
}

void CudaGraphDecodeRunner::setTokenTypeEmbedding(torch::Tensor token_type_embedding) {
    token_type_embedding_ = std::move(token_type_embedding);
}

void CudaGraphDecodeRunner::setInputEmbeddingScalar(float input_embedding_scalar) {
    input_embedding_scalar_ = input_embedding_scalar;
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
            return capture_dispatcher_.tryGetRealGraphDecodeBatchSize(inputs, batch_descriptor);
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

    if (!capture_dispatcher_.tryGetRealGraphDecodeBatchSize(inputs, batch_descriptor)) {
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

CudaGraphRunnerBase* CudaGraphRunner::createForPrefill(py::object py_instance, GraphParams params) {
    params.enable_cuda_graph          = true;
    params.is_prefill_cuda_graph_mode = true;
    if (params.num_tokens_per_bs == 0) {
        params.num_tokens_per_bs = params.max_seq_len;
    }
    return new CudaGraphPrefillRunner(std::move(params), std::move(py_instance));
}

CudaGraphRunnerBase* CudaGraphRunner::createForDecode(py::object py_instance, GraphParams params) {
    params.enable_cuda_graph          = true;
    params.is_prefill_cuda_graph_mode = false;
    if (params.num_tokens_per_bs == 0) {
        params.num_tokens_per_bs = 1;
    }
    return new CudaGraphDecodeRunner(std::move(params), std::move(py_instance));
}

}  // namespace rtp_llm
