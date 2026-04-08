#include "rtp_llm/cpp/cuda_graph/cuda_graph_runner.h"

#include "rtp_llm/cpp/cuda_graph/cuda_graph_py_model_inputs.h"

#include <algorithm>
#include <string>

#include "rtp_llm/cpp/utils/ProfilingScope.h"

using namespace torch_ext;
namespace rtp_llm {

// clang-format off
// CUDA Graph Mode Configuration Table:
// +--------------------------------+-----------------------------+--------------------------------------+--------------+
// | Model Type                     | is_prefill_cuda_graph_mode  | num_tokens_per_bs                    | 是否已经支持   |
// +--------------------------------+-----------------------------+--------------------------------------+--------------+
// | Draft Model (prefill)          | true                        | gen_num_per_cycle + 1                | no           |
// | Target Model (score, prefill)  | false                       | gen_num_per_cycle + 1                | yes          |
// | Draft Model (decode)           | false                       | 1                                    | yes          |
// | Embedding Model (prefill)      | true                        | max_seq_len                          | yes          |
// | Normal Model (decode)          | false                       | 1                                    | yes          |
// +--------------------------------+-----------------------------+--------------------------------------+--------------+
// Notes:
// - Speculative sampling: model_id == 0 (target), model_id == 1 (draft)
// - Target model with spec sampling processes multiple tokens per batch for verification phase
// clang-format on

void CudaGraphRunner::prepareInputs(const PyModelInputs& inputs, BatchDescriptor& batch_descriptor) {
    RTP_LLM_PROFILE_SCOPE("cuda_graph.prepareInputs");
    // 1. Non-spec CUDA graph:
    //    - is_prefill_cuda_graph_mode (GraphParams) is true only when using the embedding model.
    // 2. Spec CUDA graph:
    //    2.1 Spec holds target and draft models. On the first user prompt, target and draft run a real
    //        "prefill forward"; CUDA graph is not used in that phase.
    //    2.2 After that prefill, behavior splits into:
    //        2.2.1 Target model score (verify).
    //        2.2.2 Draft model first forward (input from 2.2.1).
    //        2.2.3 Draft model autoregressive forward.
    //    Currently only 2.2.1 and 2.2.3 use the decode CUDA graph; 2.2.2 is intended for prefill CUDA graph later.

    forward_event_.synchronize();
    PyModelInputs& cap         = graph_params_.is_prefill_cuda_graph_mode ?
                                     graph_instances_[batch_descriptor.current_real_graph_seq_len].mem_hold_.py_model_inputs_ :
                                     graph_instances_[batch_descriptor.current_real_graph_bs].mem_hold_.py_model_inputs_;
    py::object     decode_attn = py::none();
    if (!graph_params_.is_prefill_cuda_graph_mode) {
        decode_attn = graph_instances_[batch_descriptor.current_real_graph_bs].mem_hold_.attn_pyobj_;
    }
    cuda_graph::copyRuntimePyModelIntoCaptureBuffers(
        inputs, cap, batch_descriptor, graph_params_.is_prefill_cuda_graph_mode, decode_attn);
}

PyModelOutputs CudaGraphRunner::forward(const PyModelInputs& inputs, BatchDescriptor& batch_descriptor) {
    PyModelOutputs outputs;

    // decode or embedding model only
    RTP_LLM_LOG_DEBUG("Replay Start");
    prepareInputs(inputs, batch_descriptor);
    if (graph_params_.is_prefill_cuda_graph_mode) {
        {
            RTP_LLM_PROFILE_SCOPE("cuda_graph.forward(replayPrefill)");
            replayPrefill(batch_descriptor.current_real_graph_seq_len);
        }
        outputs.hidden_states =
            graph_instances_[batch_descriptor.current_real_graph_seq_len].mem_hold_.all_layers_output_.slice(
                0, 0, batch_descriptor.current_seq_len);
    } else {
        {
            RTP_LLM_PROFILE_SCOPE("cuda_graph.forward(replayDecode)");
            replayDecode(batch_descriptor.current_real_graph_bs);
        }
        outputs.hidden_states =
            graph_instances_[batch_descriptor.current_real_graph_bs].mem_hold_.all_layers_output_.slice(
                0, 0, batch_descriptor.seq_len_sum);
    }
    // record forward done event
    forward_event_.record(cuda_graph::graphGetCurrentStream());
    RTP_LLM_LOG_DEBUG("Replay End");
    return outputs;
}

bool CudaGraphRunner::tryGetRealGraphPrefillSeqLen(const PyModelInputs& inputs, BatchDescriptor& batch_descriptor) {
    batch_descriptor.current_seq_len = inputs.attention_inputs.input_lengths.sum(0).item<int>();
    if (capture_range_.empty()) {
        RTP_LLM_LOG_WARNING("prefill cuda graph: capture_range_ is empty, cannot run");
        return false;
    }
    auto it = std::lower_bound(capture_range_.begin(), capture_range_.end(), batch_descriptor.current_seq_len);
    // No captured graph for seq_len >= current (all captures smaller than requested)
    if (it == capture_range_.end()) {
        RTP_LLM_LOG_WARNING("prefill seq_len %d exceeds max captured %d, fallback to normal run",
                            batch_descriptor.current_seq_len,
                            capture_range_.back());
        return false;
    }
    batch_descriptor.current_real_graph_seq_len = *it;
    batch_descriptor.current_batch_size         = inputs.attention_inputs.input_lengths.size(0);
    return true;
}

bool CudaGraphRunner::tryGetRealGraphDecodeBatchSize(const PyModelInputs& inputs, BatchDescriptor& batch_descriptor) {
    int cuda_graph_bs                   = inputs.attention_inputs.input_lengths.size(0);
    batch_descriptor.current_batch_size = cuda_graph_bs;
    RTP_LLM_LOG_DEBUG("canRun judge for batch size: %d", cuda_graph_bs);
    if (capture_range_.empty()) {
        RTP_LLM_LOG_WARNING("decode cuda graph: capture_range_ is empty, cannot run");
        return false;
    }
    auto it = std::lower_bound(capture_range_.begin(), capture_range_.end(), batch_descriptor.current_batch_size);
    // No captured graph for batch >= current (all captures smaller)
    if (it == capture_range_.end()) {
        RTP_LLM_LOG_WARNING("decode batch size %d exceeds max captured %d, fallback to normal run",
                            batch_descriptor.current_batch_size,
                            capture_range_.back());
        return false;
    }
    batch_descriptor.current_real_graph_bs = *it;
    RTP_LLM_LOG_DEBUG("batch size used in replay: %d (graph key %d)",
                      batch_descriptor.current_batch_size,
                      batch_descriptor.current_real_graph_bs);

    if (inputs.attention_inputs.is_prefill) {
        batch_descriptor.seq_len_sum = inputs.attention_inputs.input_lengths.sum(0).item<int>();
    } else {
        batch_descriptor.seq_len_sum = cuda_graph_bs;
    }
    RTP_LLM_LOG_DEBUG("can run cuda graph for decode");
    return true;
}

bool CudaGraphRunner::canRun(const PyModelInputs& inputs, BatchDescriptor& batch_descriptor) {
    RTP_LLM_PROFILE_SCOPE("cuda_graph.canRun");
    // Check if this is speculative sampling:
    // 1. prefix_lengths is not empty
    // 2. all values in input_lengths are the same
    // this is for 2.2.1
    if (graph_params_.is_target_verify) {
        if (inputs.attention_inputs.is_target_verify) {
            // Target-verify must also respect captured decode range.
            // Otherwise we may replay an uncaptured graph key.
            return tryGetRealGraphDecodeBatchSize(inputs, batch_descriptor);
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

    if (graph_params_.is_prefill_cuda_graph_mode) {
        if (!tryGetRealGraphPrefillSeqLen(inputs, batch_descriptor)) {
            return false;
        }
        // current_real_graph_seq_len is always *it from lower_bound within capture_range_
        RTP_LLM_LOG_DEBUG("prefill cuda graph replay seq_len key %d", batch_descriptor.current_real_graph_seq_len);
    } else {
        if (!tryGetRealGraphDecodeBatchSize(inputs, batch_descriptor)) {
            return false;
        }
    }
    return true;
}

int CudaGraphRunner::getCurrentRealGraphBs(const BatchDescriptor& batch_descriptor) const {
    return batch_descriptor.current_real_graph_bs;
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
    if (graph_params_.enable_cuda_graph) {
        RTP_LLM_LOG_INFO("CUDA graph capture is enabled");
        shared_graph_pool_ = cuda_graph::graphPoolHandle();
        if (graph_params_.is_prefill_cuda_graph_mode) {
            RTP_LLM_LOG_INFO("CUDA graph capture for embedding, num_tokens_per_bs: %d",
                             graph_params_.num_tokens_per_bs);
        }
        max_num_token_ = max_bs_ * graph_params_.num_tokens_per_bs;
        if (graph_params_.is_prefill_cuda_graph_mode) {
            capture_range_ = getPrefillSequenceLengthsToCapture();
        } else {
            capture_range_ = getDecodeBatchSizesToCapture();
        }

        cuda_graph::CudaGraphCapturePyModelInputs capture_py_inputs(graph_params_,
                                                                    max_bs_,
                                                                    max_num_token_,
                                                                    options_cuda_int32_,
                                                                    options_cpu_int32_,
                                                                    options_cuda_float_,
                                                                    position_encoding_,
                                                                    token_type_embedding_,
                                                                    input_embedding_scalar_);
        capture_mem_hold_ = capture_py_inputs.makeCaptureMemoryHold();

        // get real output data type (params already prepared in attn impl __init__/create_params)
        auto attn_pyobj = py_attn_pyobj_method_(capture_mem_hold_.py_model_inputs_, true);
        RTP_LLM_LOG_INFO("initCapture forward for output datatype start");
        py_forward_method_(capture_mem_hold_.py_model_inputs_, attn_pyobj);
        RTP_LLM_LOG_INFO("initCapture forward for output datatype end");
        capture_py_inputs.allocateHiddenStatesAndPrefillCopyParams(capture_mem_hold_);
        logCudaGraphPoolMemory("before_capture");

        if (graph_params_.is_prefill_cuda_graph_mode) {
            RTP_LLM_LOG_INFO("initCapture forward post check start for prefill");
            capture_py_inputs.patchForPrefillProbeForward(capture_mem_hold_);
            py_forward_method_(capture_py_inputs.sliceForPrefillProbeForward(capture_mem_hold_));
            RTP_LLM_LOG_INFO("initCapture forward post check end for prefill");
            capturePrefill();
        } else {
            captureDecode();
        }
        logCudaGraphPoolMemory("after_capture");
    } else {
        cuda_graph::CudaGraphCapturePyModelInputs::fillCuSeqlensForCapture(capture_mem_hold_.py_model_inputs_, max_bs_);
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

void CudaGraphRunner::replayAndSyncCheck(int key, const char* key_type) {
    RTP_LLM_LOG_INFO("replay start check for %s %d", key_type, key);
    replayGraph(key);
    cuda_graph::graphDeviceSynchronize();
    RTP_LLM_LOG_INFO("replay end check for %s %d", key_type, key);
}

void CudaGraphRunner::prepareCaptureInputs(PyModelInputs& inputs, int batch_size, int seq_len_or_tokens) {
    cuda_graph::sliceTemplatePyModelInputsForCapture(inputs,
                                                     capture_mem_hold_.py_model_inputs_,
                                                     batch_size,
                                                     seq_len_or_tokens,
                                                     graph_params_.is_prefill_cuda_graph_mode,
                                                     graph_params_.num_tokens_per_bs,
                                                     graph_params_.is_target_verify);
}

CaptureMemoryHold CudaGraphRunner::createCaptureMemoryHold(PyModelInputs& inputs, int tokens_count) {
    return CaptureMemoryHold(capture_mem_hold_.all_layers_output_.slice(0, 0, tokens_count), inputs);
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
