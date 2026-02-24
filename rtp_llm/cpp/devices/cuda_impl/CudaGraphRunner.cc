#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>

#include <cuda_runtime_api.h>
#include <torch/torch.h>
#include "ATen/core/TensorBody.h"
#include "ATen/ops/zeros.h"
#include "rtp_llm/cpp/devices/cuda_impl/CudaGraphRunner.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/devices/cuda_impl/CudaFlashInfer.h"
#include "rtp_llm/cpp/cuda/cuda_host_utils.h"
#include "rtp_llm/models_py/bindings/OpDefsUtils.h"
#include "torch/csrc/autograd/generated/variable_factories.h"
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
// - Target model with spec sampling processes multiple tokens per batch for verification phase
// clang-format on

// Helper function for optimized tensor copy using async operations with current CUDA stream
void optimizedCopyAsync(const torch::Tensor& src, torch::Tensor& dst, size_t size) {
    if (!src.defined() || src.numel() <= 0) {
        return;
    }
    // Get current CUDA stream from PyTorch
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (src.is_cuda() && dst.is_cuda()) {
        check_cuda_value(cudaMemcpyAsync(dst.data_ptr(), src.data_ptr(), size, cudaMemcpyDeviceToDevice, stream));
    } else if (!src.is_cuda() && !dst.is_cuda()) {
        memcpy(dst.data_ptr(), src.data_ptr(), size);
    } else if (src.is_cuda() && !dst.is_cuda()) {
        check_cuda_value(cudaMemcpyAsync(dst.data_ptr(), src.data_ptr(), size, cudaMemcpyDeviceToHost, stream));
    } else {
        check_cuda_value(cudaMemcpyAsync(dst.data_ptr(), src.data_ptr(), size, cudaMemcpyHostToDevice, stream));
    }
}

// column dimension
void CudaGraphRunner::copySmallerIntoLarger(const torch::Tensor& source_tensor, torch::Tensor& target_tensor) {
    if (source_tensor.dim() != target_tensor.dim()) {
        throw std::runtime_error("Error: Source and target tensors must have the same number of dimensions.");
    }

    for (int i = 0; i < source_tensor.dim(); ++i) {
        if (source_tensor.size(i) > target_tensor.size(i)) {
            std::string error_msg =
                "Error: Target tensor dimension " + std::to_string(i) + " (" + std::to_string(target_tensor.size(i))
                + ")" + " is smaller than source tensor dimension " + std::to_string(i) + " ("
                + std::to_string(source_tensor.size(i)) + "). " + "This violates the function's guarantee.";
            throw std::runtime_error(error_msg);
        }
    }

    torch::Tensor target_slice = target_tensor;

    for (int i = 0; i < source_tensor.dim(); ++i) {
        target_slice = target_slice.slice(i, 0, source_tensor.size(i));
    }

    target_slice.copy_(source_tensor);
}

void CudaGraphRunner::prepareInputs(PyModelInputs& inputs) {
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

    // Get the appropriate graph instance based on mode
    const size_t graph_idx =
        is_prefill_cuda_graph_mode_ ? state_.current_real_graph_seq_len : state_.current_real_graph_bs;
    auto& py_model_inputs_ = graph_instances_[graph_idx].mem_hold_.py_model_inputs_;
    auto  attn_pyobj       = graph_instances_[graph_idx].mem_hold_.attn_pyobj_;

    // Clear kv_cache_block_id_device to prevent cache block pollution
    py_model_inputs_.attention_inputs.kv_cache_block_id_device.fill_(0);
    py_model_inputs_.attention_inputs.kv_cache_block_id_host.fill_(0);

    // Common async copies for both modes
    optimizedCopyAsync(inputs.attention_inputs.prefix_lengths,
                       py_model_inputs_.attention_inputs.prefix_lengths,
                       state_.current_batch_size * sizeof(int));

    optimizedCopyAsync(inputs.attention_inputs.input_lengths,
                       py_model_inputs_.attention_inputs.input_lengths,
                       state_.current_batch_size * sizeof(int));

    optimizedCopyAsync(inputs.attention_inputs.cu_seqlens,
                       py_model_inputs_.attention_inputs.cu_seqlens,
                       (state_.current_batch_size + 1) * sizeof(int));

    optimizedCopyAsync(inputs.attention_inputs.cu_kv_seqlens,
                       py_model_inputs_.attention_inputs.cu_kv_seqlens,
                       (state_.current_batch_size + 1) * sizeof(int));

    optimizedCopyAsync(inputs.input_ids, py_model_inputs_.input_ids, inputs.input_ids.size(0) * sizeof(int));

    optimizedCopyAsync(inputs.input_hiddens,
                       py_model_inputs_.input_hiddens,
                       inputs.input_hiddens.numel() * inputs.input_hiddens.element_size());

    copySmallerIntoLarger(inputs.attention_inputs.kv_cache_block_id_device,
                          py_model_inputs_.attention_inputs.kv_cache_block_id_device);

    copySmallerIntoLarger(inputs.attention_inputs.kv_cache_block_id_host,
                          py_model_inputs_.attention_inputs.kv_cache_block_id_host);

    // Mode-specific operations
    if (!is_prefill_cuda_graph_mode_) {
        // Decode mode specific copies
        optimizedCopyAsync(inputs.attention_inputs.sequence_lengths,
                           py_model_inputs_.attention_inputs.sequence_lengths,
                           state_.current_batch_size * sizeof(int));

        optimizedCopyAsync(inputs.attention_inputs.sequence_lengths_plus_1_d,
                           py_model_inputs_.attention_inputs.sequence_lengths_plus_1_d,
                           state_.current_batch_size * sizeof(int));

        optimizedCopyAsync(inputs.attention_inputs.decode_cu_seqlens_d,
                           py_model_inputs_.attention_inputs.decode_cu_seqlens_d,
                           (state_.current_batch_size + 1) * sizeof(int));
        attn_pyobj.attr("prepare_cuda_graph")(inputs.attention_inputs);
    } else {
        // Prefill mode specific copies
        optimizedCopyAsync(inputs.attention_inputs.padding_offset,
                           py_model_inputs_.attention_inputs.padding_offset,
                           state_.current_seq_len * sizeof(int));

        // Set prefill batch size for cuda graph
        if (py_model_inputs_.attention_inputs.prefill_cuda_graph_copy_params) {
            (*(py_model_inputs_.attention_inputs.prefill_cuda_graph_copy_params->cuda_graph_prefill_batch_size
                   .data_ptr<int>())) = state_.current_batch_size;
        }

        // Copy BERT embedding inputs if needed
        if (inputs.bert_embedding_inputs.position_encoding.numel() > 0) {
            optimizedCopyAsync(inputs.bert_embedding_inputs.combo_position_ids,
                               py_model_inputs_.bert_embedding_inputs.combo_position_ids,
                               state_.current_seq_len * sizeof(int));

            optimizedCopyAsync(inputs.bert_embedding_inputs.combo_tokens_type_ids,
                               py_model_inputs_.bert_embedding_inputs.combo_tokens_type_ids,
                               state_.current_seq_len * sizeof(int));
        }
        // Reset unused batch portions in captured tensors to prevent stale data
        // Only reset the portion from current_batch_size to the end (unused batches)
        if (state_.current_batch_size < max_bs_) {
            auto prefix_len_slice =
                py_model_inputs_.attention_inputs.prefix_lengths.slice(0, state_.current_batch_size, max_bs_);
            prefix_len_slice.fill_(0);

            auto input_len_slice =
                py_model_inputs_.attention_inputs.input_lengths.slice(0, state_.current_batch_size, max_bs_);
            input_len_slice.fill_(0);
        }

        int last_valid = py_model_inputs_.attention_inputs.cu_seqlens.data_ptr<int>()[state_.current_batch_size];
        py_model_inputs_.attention_inputs.cu_seqlens.slice(0, state_.current_batch_size + 1, max_bs_ + 1)
            .fill_(last_valid);
        attn_pyobj.attr("prepare_cuda_graph")(py_model_inputs_.attention_inputs);
    }
}

PyModelOutputs CudaGraphRunner::forward(PyModelInputs& inputs) {
    PyModelOutputs outputs;
    auto           stream = at::cuda::getCurrentCUDAStream();

    // decode or embedding model only
    RTP_LLM_LOG_DEBUG("Replay Start");
    prepareInputs(inputs);
    if (is_prefill_cuda_graph_mode_) {
        replayPrefill(state_.current_real_graph_seq_len);
        outputs.hidden_states =
            graph_instances_[state_.current_real_graph_seq_len].mem_hold_.decoder_layer_hidden_states_.slice(
                0, 0, state_.current_seq_len);
    } else {
        replayDecode(state_.current_real_graph_bs);
        outputs.hidden_states =
            graph_instances_[state_.current_real_graph_bs].mem_hold_.decoder_layer_hidden_states_.slice(
                0, 0, state_.seq_len_sum);
    }

    // record forward done event
    forward_event_.record(stream);
    RTP_LLM_LOG_DEBUG("Replay End");

    return outputs;
}

void CudaGraphRunner::tryGetRealGraphPrefillSeqLen(PyModelInputs& inputs) {
    state_.current_seq_len = inputs.attention_inputs.input_lengths.sum(0).item<int>();
    auto it                = std::lower_bound(capture_range_.begin(), capture_range_.end(), state_.current_seq_len);
    state_.current_real_graph_seq_len = *it;
    state_.current_batch_size         = inputs.attention_inputs.input_lengths.size(0);
}

void CudaGraphRunner::tryGetRealGraphDecodeBatchSize(PyModelInputs& inputs) {
    int cuda_graph_bs         = inputs.attention_inputs.input_lengths.size(0);
    state_.current_batch_size = cuda_graph_bs;
    RTP_LLM_LOG_DEBUG("canRun judge for batch size: %d", cuda_graph_bs);
    auto it = std::lower_bound(capture_range_.begin(), capture_range_.end(), state_.current_batch_size);
    state_.current_real_graph_bs = *it;
    RTP_LLM_CHECK_WITH_INFO(it != capture_range_.end(), "batch size used in replay: %d", state_.current_real_graph_bs);

    if (inputs.attention_inputs.is_prefill) {
        state_.seq_len_sum = inputs.attention_inputs.input_lengths.sum(0).item<int>();
    } else {
        state_.seq_len_sum = cuda_graph_bs;
    }

    RTP_LLM_LOG_DEBUG("can run cuda graph for decode");
}

bool CudaGraphRunner::canRun(PyModelInputs& inputs) {
    // Check if this is speculative sampling:
    // 1. prefix_lengths is not empty
    // 2. all values in input_lengths are the same
    // this is for 2.2.1
    if (!is_prefill_cuda_graph_mode_ && inputs.attention_inputs.prefix_lengths.defined()
        && inputs.attention_inputs.prefix_lengths.numel() > 0
        && inputs.attention_inputs.prefix_lengths.data_ptr<int>()[0] > 0) {
        // Check if all input_lengths are the same (input_lengths is pin memory)
        auto input_lengths_cpu = inputs.attention_inputs.input_lengths;
        int  valid_value       = num_tokens_per_bs_;
        bool all_same          = true;
        for (int i = 0; i < input_lengths_cpu.size(0); i++) {
            if (input_lengths_cpu[i].item<int>() != valid_value) {
                all_same = false;
                break;
            }
        }
        if (all_same && num_tokens_per_bs_ > 1) {
            tryGetRealGraphDecodeBatchSize(inputs);
            return true;
        }
    }

    if (!enable_cuda_graph_ || (inputs.attention_inputs.is_prefill && !is_prefill_cuda_graph_mode_)) {
        return false;
    }

    if (is_prefill_cuda_graph_mode_) {
        tryGetRealGraphPrefillSeqLen(inputs);
        if (state_.current_seq_len > max_perfill_cuda_graph_len_) {
            return false;
        }
        RTP_LLM_CHECK_WITH_INFO(std::any_of(capture_range_.begin(),
                                            capture_range_.end(),
                                            [&](int x) { return x == state_.current_real_graph_seq_len; }),
                                "seqlen used in replay: %d",
                                state_.current_real_graph_seq_len);
        RTP_LLM_LOG_INFO("can run cuda graph for prefill, current_real_graph_seq_len: %d",
                         state_.current_real_graph_seq_len);
    } else {
        tryGetRealGraphDecodeBatchSize(inputs);
    }
    return true;
}

void CudaGraphRunner::initKernelInternalMemory() {
    torch::Tensor cu_seqlens =
        torch::zeros({int(max_bs_ + 1)}, torch::TensorOptions(torch::kInt32).device(torch::kCPU));
    torch::Tensor cu_kv_seqlens =
        torch::zeros({int(max_bs_ + 1)}, torch::TensorOptions(torch::kInt32).device(torch::kCPU));
    auto input_lengths  = capture_mem_hold_.py_model_inputs_.attention_inputs.input_lengths;
    auto prefix_lengths = capture_mem_hold_.py_model_inputs_.attention_inputs.prefix_lengths;

    cu_seqlens.slice(0, 1, max_bs_ + 1)    = input_lengths.cumsum(0);
    cu_kv_seqlens.slice(0, 1, max_bs_ + 1) = input_lengths.add(prefix_lengths).cumsum(0);

    capture_mem_hold_.py_model_inputs_.attention_inputs.cu_seqlens    = cu_seqlens.pin_memory();
    capture_mem_hold_.py_model_inputs_.attention_inputs.cu_kv_seqlens = cu_kv_seqlens.pin_memory();
}

int CudaGraphRunner::getCurrentRealGraphBs() {
    return state_.current_real_graph_bs;
}

void CudaGraphRunner::initCaptureAttentionInputs(PyModelInputs& inputs, int max_bs, int num_tokens_per_bs) {
    // input_ids [tokens_nums] = [batch_size * num_tokens_per_bs]
    inputs.input_ids = torch::zeros({max_num_token_}, options_cuda_int32_);
    // input_lengths [batch_size, int32] (decode only)
    inputs.attention_inputs.input_lengths = torch::full({int(max_bs_)}, num_tokens_per_bs_, options_cpu_int32_);
    inputs.attention_inputs.input_lengths = inputs.attention_inputs.input_lengths.pin_memory();
    // sequence_lengths [batch_size, int32] (decode only)
    // sequence_length should in pinned memory
    inputs.attention_inputs.sequence_lengths = torch::ones({int(max_bs_)}, options_cpu_int32_);
    inputs.attention_inputs.sequence_lengths = inputs.attention_inputs.sequence_lengths.pin_memory();
    // kv_cache_block_id_device [batch_size, block_num]
    inputs.attention_inputs.kv_cache_block_id_device = torch::zeros(
        {int(max_bs_), ((max_seq_len_ + seq_size_per_block_ - 1) / seq_size_per_block_ + 1)}, options_cuda_int32_);
    // prefix_lengths [batch_size, int32] (for attention `prepare`)
    // for draft model prefill and target model verify
    if (num_tokens_per_bs_ > 1) {
        inputs.attention_inputs.prefix_lengths =
            torch::full({int(max_bs_)}, max_seq_len_ + num_tokens_per_bs_, options_cpu_int32_).pin_memory();
    } else {
        inputs.attention_inputs.prefix_lengths = torch::zeros({int(max_bs_)}, options_cpu_int32_).pin_memory();
    }

    inputs.attention_inputs.kv_cache_block_id_host = torch::zeros(
        {int(max_bs_), ((max_seq_len_ + seq_size_per_block_ - 1) / seq_size_per_block_ + 1)}, options_cpu_int32_);
    // padding_offset [max_num_token_, int32] (for attention padding)
    inputs.attention_inputs.padding_offset            = torch::zeros({int(max_seq_len_ * max_bs_)}, options_cpu_int32_);
    inputs.attention_inputs.padding_offset            = inputs.attention_inputs.padding_offset.pin_memory();
    inputs.attention_inputs.dtype                     = model_data_type_;
    inputs.attention_inputs.is_s_padded               = true;
    inputs.attention_inputs.sequence_lengths_plus_1_d = torch::zeros({int(max_bs_)}, options_cuda_int32_);
    inputs.attention_inputs.decode_cu_seqlens_d       = torch::zeros({int(max_bs_)}, options_cuda_int32_);
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

void CudaGraphRunner::setMaxPrefillCudaGraphLen(int max_prefill_cuda_graph_len) {
    max_perfill_cuda_graph_len_ = max_prefill_cuda_graph_len;
    RTP_LLM_LOG_INFO("Set max_perfill_cuda_graph_len_ to %d", max_perfill_cuda_graph_len_);
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

void CudaGraphRunner::initCapture() {
    if (enable_cuda_graph_) {
        RTP_LLM_LOG_INFO("CUDA graph capture is enabled");
        if (is_prefill_cuda_graph_mode_) {
            RTP_LLM_LOG_INFO("CUDA graph capture for prefill, num_tokens_per_bs_: %d", num_tokens_per_bs_);
        }
        max_num_token_ = max_bs_ * num_tokens_per_bs_;
        // Capture
        at::cuda::CUDAGraph graph;
        if (is_prefill_cuda_graph_mode_) {
            capture_range_ = getPrefillSequenceLengthsToCapture();
            // Set max_perfill_cuda_graph_len_ to the maximum value from capture_range_
            if (!capture_range_.empty()) {
                int max_seq_len             = *std::max_element(capture_range_.begin(), capture_range_.end());
                max_perfill_cuda_graph_len_ = max_seq_len;
                RTP_LLM_LOG_INFO("Set max_perfill_cuda_graph_len_ to %d (max from capture_range_)",
                                 max_perfill_cuda_graph_len_);
            }
        } else {
            capture_range_ = getDecodeBatchSizesToCapture();
        }

        PyModelInputs inputs;
        // input_ids [tokens_nums] = [batch_size * num_tokens_per_bs]
        inputs.input_ids     = torch::zeros({max_num_token_}, options_cuda_int32_);
        inputs.input_hiddens = torch::zeros({max_num_token_, hidden_size_}, options_cuda_float_);
        // input_lengths [batch_size, int32] (decode only)
        // Setup attention inputs using the extracted function
        initCaptureAttentionInputs(inputs, max_bs_, num_tokens_per_bs_);

        // Setup BertEmbedding inputs using the extracted function
        initCaptureBertEmbeddingInputs(inputs, max_bs_, max_num_token_);

        torch::Tensor output;
        capture_mem_hold_ = CaptureMemoryHold(output, inputs, is_prefill_cuda_graph_mode_);
        initKernelInternalMemory();
        // get real output data type
        auto attn_pyobj = py_attn_pyobj_method_(capture_mem_hold_.py_model_inputs_, true);
        attn_pyobj.attr("prepare_cuda_graph")(capture_mem_hold_.py_model_inputs_.attention_inputs);
        RTP_LLM_LOG_INFO("initCapture forward for output datatype start");
        py_forward_method_(capture_mem_hold_.py_model_inputs_, attn_pyobj);
        RTP_LLM_LOG_INFO("initCapture forward for output datatype end");
        output = torch::zeros({max_num_token_, hidden_size_}, options_cuda_float_);
        capture_mem_hold_.setHiddenStates(output);
        initCaptureAttentionInputsPost();

        if (is_prefill_cuda_graph_mode_) {
            RTP_LLM_LOG_INFO("initCapture forward post check start for prefill");
            capture_mem_hold_.py_model_inputs_.attention_inputs.cu_seqlens.data_ptr<int>()[1]    = max_num_token_;
            capture_mem_hold_.py_model_inputs_.attention_inputs.cu_kv_seqlens.data_ptr<int>()[1] = max_num_token_;
            capture_mem_hold_.py_model_inputs_.attention_inputs.input_lengths.data_ptr<int>()[0] = max_num_token_;
            PyModelInputs inputs = capture_mem_hold_.py_model_inputs_;
            inputs.attention_inputs.cu_seqlens =
                capture_mem_hold_.py_model_inputs_.attention_inputs.cu_seqlens.slice(0, 0, 2);
            inputs.attention_inputs.cu_kv_seqlens =
                capture_mem_hold_.py_model_inputs_.attention_inputs.cu_kv_seqlens.slice(0, 0, 2);
            inputs.attention_inputs.input_lengths =
                capture_mem_hold_.py_model_inputs_.attention_inputs.input_lengths.slice(0, 0, 1);
            inputs.attention_inputs.kv_cache_block_id_device =
                capture_mem_hold_.py_model_inputs_.attention_inputs.kv_cache_block_id_device.slice(0, 0, 1);
            inputs.attention_inputs.kv_cache_block_id_host =
                capture_mem_hold_.py_model_inputs_.attention_inputs.kv_cache_block_id_host.slice(0, 0, 1);
            py_forward_method_(inputs);
            RTP_LLM_LOG_INFO("initCapture forward post check end for prefill");
            capturePrefill();
        } else {
            captureDecode();
        }
    } else {
        initKernelInternalMemory();
        RTP_LLM_LOG_INFO("CUDA graph capture is not enabled, skipping initialization");
    }
}

void CudaGraphRunner::replayGraph(int key) {
    graph_instances_[key].graph_.replay();
}

void CudaGraphRunner::captureOneGraphInstance(int key, const char* key_type) {
    auto inputs = graph_instances_[key].mem_hold_.py_model_inputs_;
    // WarmUp twice
    RTP_LLM_LOG_INFO("WarmUp for %s %d start.", key_type, key);
    auto attn_pyobj = graph_instances_[key].mem_hold_.attn_pyobj_;
    attn_pyobj.attr("prepare_cuda_graph")(inputs.attention_inputs);
    py_forward_method_(inputs, attn_pyobj);
    py_forward_method_(inputs, attn_pyobj);
    RTP_LLM_LOG_INFO("WarmUp for %s %d successfully.", key_type, key);

    {
        // sync before capture
        check_cuda_value(cudaDeviceSynchronize());

        CudaGraphStreamLife  stream_life(capture_stream_);
        at::cuda::CUDAGraph& graph               = graph_instances_[key].graph_;
        std::string          output_dot_filename = "";
        if (enable_cuda_graph_debug_mode_) {
            graph.enable_debug_mode();
            // Generate unique filename with num_tokens_per_bs, key_type and key
            // Replace spaces with underscores for better filename compatibility
            std::string key_type_str = std::string(key_type);
            std::replace(key_type_str.begin(), key_type_str.end(), ' ', '_');
            output_dot_filename = "cuda_graph_tokens" + std::to_string(num_tokens_per_bs_) + "_" + key_type_str + "_"
                                  + std::to_string(key) + "_visualization.dot";
            RTP_LLM_LOG_INFO("CUDA Graph debug mode enabled, output file: %s", output_dot_filename.c_str());
        }
        RTP_LLM_LOG_INFO("Capture for %s %d begin.", key_type, key);
        PyModelOutputs outputs;
        {
            graph.capture_begin();
            CudaGraphCaptureGuard capture_guard;
            auto                  py_outputs_obj = py_forward_method_(inputs, attn_pyobj);
            outputs                              = py_outputs_obj.cast<PyModelOutputs>();
            graph_instances_[key].mem_hold_.decoder_layer_hidden_states_.copy_(outputs.hidden_states);
            graph.capture_end();
        }
        if (enable_cuda_graph_debug_mode_) {
            graph.debug_dump(output_dot_filename.c_str());
        }
    }
}

void CudaGraphRunner::replayAndSyncCheck(int key, const char* key_type) {
    RTP_LLM_LOG_INFO("replay start check for %s %d", key_type, key);
    replayGraph(key);
    check_cuda_value(cudaDeviceSynchronize());
    RTP_LLM_LOG_INFO("replay end check for %s %d", key_type, key);
}

void CudaGraphRunner::prepareCaptureInputs(PyModelInputs& inputs, int batch_size, int seq_len_or_tokens) {
    // Common slice operations for input_ids and padding_offset
    // only for target model score phase, the `num_tokens_per_bs_` > 1, but this will run decode cuda graph.
    inputs.attention_inputs.is_prefill = is_prefill_cuda_graph_mode_ || num_tokens_per_bs_ > 1;
    inputs.input_ids                   = capture_mem_hold_.py_model_inputs_.input_ids.slice(0, 0, seq_len_or_tokens);
    inputs.input_hiddens = capture_mem_hold_.py_model_inputs_.input_hiddens.slice(0, 0, seq_len_or_tokens);
    inputs.attention_inputs.input_lengths =
        capture_mem_hold_.py_model_inputs_.attention_inputs.input_lengths.slice(0, 0, batch_size);
    inputs.attention_inputs.padding_offset =
        capture_mem_hold_.py_model_inputs_.attention_inputs.padding_offset.slice(0, 0, seq_len_or_tokens);

    // Common slice operations for attention inputs
    inputs.attention_inputs.prefix_lengths =
        capture_mem_hold_.py_model_inputs_.attention_inputs.prefix_lengths.slice(0, 0, batch_size);
    inputs.attention_inputs.sequence_lengths =
        capture_mem_hold_.py_model_inputs_.attention_inputs.sequence_lengths.slice(0, 0, batch_size);
    inputs.attention_inputs.kv_cache_block_id_device =
        capture_mem_hold_.py_model_inputs_.attention_inputs.kv_cache_block_id_device.slice(0, 0, batch_size);
    inputs.attention_inputs.kv_cache_block_id_host =
        capture_mem_hold_.py_model_inputs_.attention_inputs.kv_cache_block_id_host.slice(0, 0, batch_size);
    inputs.attention_inputs.cu_seqlens =
        capture_mem_hold_.py_model_inputs_.attention_inputs.cu_seqlens.slice(0, 0, batch_size + 1);
    inputs.attention_inputs.cu_kv_seqlens =
        capture_mem_hold_.py_model_inputs_.attention_inputs.cu_kv_seqlens.slice(0, 0, batch_size + 1);
    inputs.attention_inputs.decode_cu_seqlens_d =
        capture_mem_hold_.py_model_inputs_.attention_inputs.decode_cu_seqlens_d.slice(0, 0, batch_size + 1);
    inputs.attention_inputs.sequence_lengths_plus_1_d =
        capture_mem_hold_.py_model_inputs_.attention_inputs.sequence_lengths_plus_1_d.slice(0, 0, batch_size);

    // Common direct assignments (no slice needed)
    inputs.attention_inputs.dtype       = capture_mem_hold_.py_model_inputs_.attention_inputs.dtype;
    inputs.bert_embedding_inputs        = capture_mem_hold_.py_model_inputs_.bert_embedding_inputs;
    inputs.attention_inputs.is_s_padded = true;
}

CaptureMemoryHold CudaGraphRunner::createCaptureMemoryHold(PyModelInputs& inputs, int tokens_count) {
    // only when prefill or target model score phase, the num_tokens_per_bs_ > 1
    return CaptureMemoryHold(capture_mem_hold_.decoder_layer_hidden_states_.slice(0, 0, tokens_count),
                             inputs,
                             is_prefill_cuda_graph_mode_ || num_tokens_per_bs_ > 1);
}

}  // namespace rtp_llm
