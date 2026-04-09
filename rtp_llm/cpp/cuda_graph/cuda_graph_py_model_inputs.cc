#include "rtp_llm/cpp/cuda_graph/cuda_graph_py_model_inputs.h"

#include "rtp_llm/cpp/cuda_graph/cuda_graph_device_shims.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

#include <torch/torch.h>

#include <cstring>
#include <stdexcept>
#include <string>

using namespace torch_ext;

namespace rtp_llm {
namespace cuda_graph {

void CudaGraphCapturePyModelInputs::optimizedCopyAsync(const at::Tensor& src, at::Tensor& dst, size_t size) {
    if (!src.defined() || src.numel() <= 0) {
        return;
    }

    void* stream = reinterpret_cast<void*>(graphGetCurrentStream().stream());
    if (src.is_cuda() && dst.is_cuda()) {
        graphMemcpyAsync(dst.data_ptr(), src.data_ptr(), size, GraphMemcpyKind::D2D, stream);
    } else if (!src.is_cuda() && !dst.is_cuda()) {
        std::memcpy(dst.data_ptr(), src.data_ptr(), size);
    } else if (src.is_cuda() && !dst.is_cuda()) {
        graphMemcpyAsync(dst.data_ptr(), src.data_ptr(), size, GraphMemcpyKind::D2H, stream);
    } else {
        graphMemcpyAsync(dst.data_ptr(), src.data_ptr(), size, GraphMemcpyKind::H2D, stream);
    }
}

void CudaGraphCapturePyModelInputs::copySmallerIntoLarger(const at::Tensor& source_tensor, at::Tensor& target_tensor) {
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

    at::Tensor target_slice = target_tensor;

    for (int i = 0; i < source_tensor.dim(); ++i) {
        target_slice = target_slice.slice(i, 0, source_tensor.size(i));
    }

    target_slice.copy_(source_tensor);
}

void CudaGraphCapturePyModelInputs::initCaptureAttentionTensors(PyModelInputs&                      inputs,
                                                                const CaptureAttentionLayoutConfig& cfg) {
    inputs.attention_inputs.is_target_verify = cfg.is_target_verify;
    inputs.attention_inputs.is_prefill       = cfg.is_prefill_cuda_graph_mode || cfg.member_num_tokens_per_bs > 1;

    inputs.attention_inputs.input_lengths =
        torch::full({static_cast<int64_t>(cfg.max_bs)}, cfg.member_num_tokens_per_bs, cfg.options_cpu_int32);
    inputs.attention_inputs.input_lengths = inputs.attention_inputs.input_lengths.pin_memory();

    inputs.attention_inputs.sequence_lengths = torch::ones({static_cast<int64_t>(cfg.max_bs)}, cfg.options_cpu_int32);
    inputs.attention_inputs.sequence_lengths.fill_(cfg.max_seq_len - cfg.passed_num_tokens_per_bs - 1);
    inputs.attention_inputs.sequence_lengths = inputs.attention_inputs.sequence_lengths.pin_memory();

    const int64_t max_kv_blocks =
        static_cast<int64_t>(((cfg.max_seq_len + cfg.seq_size_per_block - 1) / cfg.seq_size_per_block) + cfg.sp_steps);
    const int64_t max_blocks = max_kv_blocks * cfg.seq_size_per_block / cfg.kernel_seq_size_per_block;

    inputs.attention_inputs.kv_cache_kernel_block_id_device =
        torch::zeros({static_cast<int64_t>(cfg.max_bs), max_blocks}, cfg.options_cuda_int32);

    inputs.attention_inputs.kv_cache_kernel_block_id_host =
        torch::zeros({static_cast<int64_t>(cfg.max_bs), max_blocks}, cfg.options_cpu_int32).pin_memory();
    inputs.attention_inputs.kv_cache_block_id_device = inputs.attention_inputs.kv_cache_kernel_block_id_device;
    inputs.attention_inputs.kv_cache_block_id_host   = inputs.attention_inputs.kv_cache_kernel_block_id_host;

    const auto layer_num = cfg.kv_cache_layer_to_group.size();
    if (layer_num > 0) {
        auto kv_cache_layer_to_group_capture =
            torch::empty({static_cast<int64_t>(layer_num)}, cfg.options_cpu_int32).pin_memory();
        auto* dst = kv_cache_layer_to_group_capture.data_ptr<int32_t>();
        for (size_t i = 0; i < layer_num; ++i) {
            dst[i] = static_cast<int32_t>(cfg.kv_cache_layer_to_group[i]);
        }
        inputs.attention_inputs.kv_cache_layer_to_group = kv_cache_layer_to_group_capture;
    }

    inputs.attention_inputs.kv_cache_kernel_block_id_device_by_group.clear();
    inputs.attention_inputs.kv_cache_kernel_block_id_host_by_group.clear();
    if (cfg.kv_cache_group_num > 1) {
        inputs.attention_inputs.kv_cache_kernel_block_id_device_by_group.reserve(
            static_cast<size_t>(cfg.kv_cache_group_num));
        inputs.attention_inputs.kv_cache_kernel_block_id_host_by_group.reserve(
            static_cast<size_t>(cfg.kv_cache_group_num));
        for (int g = 0; g < cfg.kv_cache_group_num; ++g) {
            inputs.attention_inputs.kv_cache_kernel_block_id_device_by_group.push_back(
                torch::zeros({static_cast<int64_t>(cfg.max_bs), max_blocks}, cfg.options_cuda_int32));
            inputs.attention_inputs.kv_cache_kernel_block_id_host_by_group.push_back(
                torch::zeros({static_cast<int64_t>(cfg.max_bs), max_blocks}, cfg.options_cpu_int32).pin_memory());
        }
    }

    if (cfg.member_num_tokens_per_bs > 1 && !cfg.is_prefill_cuda_graph_mode) {
        inputs.attention_inputs.prefix_lengths = torch::full({static_cast<int64_t>(cfg.max_bs)},
                                                             cfg.max_seq_len - cfg.member_num_tokens_per_bs,
                                                             cfg.options_cpu_int32)
                                                     .pin_memory();
    } else if (cfg.is_prefill_cuda_graph_mode) {
        inputs.attention_inputs.prefix_lengths =
            torch::zeros({static_cast<int64_t>(cfg.max_bs)}, cfg.options_cpu_int32).pin_memory();
    }

    inputs.attention_inputs.padding_offset =
        torch::zeros({cfg.max_seq_len * static_cast<int64_t>(cfg.max_bs)}, cfg.options_cpu_int32);
    inputs.attention_inputs.padding_offset = inputs.attention_inputs.padding_offset.pin_memory();
    inputs.attention_inputs.dtype          = cfg.model_data_type;
    inputs.attention_inputs.is_s_padded    = true;
    inputs.attention_inputs.sequence_lengths_plus_1_d =
        torch::zeros({static_cast<int64_t>(cfg.max_bs)}, cfg.options_cuda_int32);
    inputs.attention_inputs.decode_cu_seqlens_d = torch::arange(
        0, static_cast<int64_t>(cfg.max_bs) + 1, 1, torch::TensorOptions(torch::kInt32).device(torch::kCUDA));
}

CaptureMaxPyModelInputsConfig CudaGraphCapturePyModelInputs::makeMaxCapturePyModelInputsConfig() const {
    CaptureMaxPyModelInputsConfig cfg;
    cfg.hidden_size            = static_cast<int64_t>(graph_params_.hidden_size);
    cfg.options_cuda_float     = options_cuda_float_;
    cfg.bert_combo_flat_len    = graph_params_.max_seq_len * static_cast<int>(max_bs_);
    cfg.position_encoding      = position_encoding_;
    cfg.token_type_embedding   = token_type_embedding_;
    cfg.input_embedding_scalar = input_embedding_scalar_;

    auto& a                      = cfg.attention;
    a.is_target_verify           = graph_params_.is_target_verify;
    a.is_prefill_cuda_graph_mode = graph_params_.is_prefill_cuda_graph_mode;
    a.member_num_tokens_per_bs   = graph_params_.num_tokens_per_bs;
    a.passed_num_tokens_per_bs   = graph_params_.num_tokens_per_bs;
    a.max_num_token              = max_num_token_;
    a.max_bs                     = max_bs_;
    a.max_seq_len                = graph_params_.max_seq_len;
    a.seq_size_per_block         = graph_params_.tokens_per_block;
    a.kernel_seq_size_per_block  = graph_params_.kernel_tokens_per_block;
    a.sp_steps                   = graph_params_.sp_steps;
    a.model_data_type            = graph_params_.model_data_type;
    a.kv_cache_layer_to_group    = graph_params_.kv_cache_layer_to_group;
    a.kv_cache_group_num         = graph_params_.kv_cache_group_num;
    a.options_cuda_int32         = options_cuda_int32_;
    a.options_cpu_int32          = options_cpu_int32_;
    return cfg;
}

void CudaGraphCapturePyModelInputs::initMaxCapturePyModelInputs(PyModelInputs& inputs) const {
    const CaptureMaxPyModelInputsConfig cfg = makeMaxCapturePyModelInputsConfig();
    const auto&                         a   = cfg.attention;
    inputs.input_ids                        = torch::zeros({a.max_num_token}, a.options_cuda_int32);
    inputs.input_hiddens                    = torch::zeros({a.max_num_token, cfg.hidden_size}, cfg.options_cuda_float);
    initCaptureAttentionTensors(inputs, a);
    inputs.bert_embedding_inputs.combo_position_ids     = torch::zeros({cfg.bert_combo_flat_len}, a.options_cuda_int32);
    inputs.bert_embedding_inputs.position_encoding      = cfg.position_encoding;
    inputs.bert_embedding_inputs.combo_tokens_type_ids  = torch::zeros({cfg.bert_combo_flat_len}, a.options_cuda_int32);
    inputs.bert_embedding_inputs.token_type_embedding   = cfg.token_type_embedding;
    inputs.bert_embedding_inputs.input_embedding_scalar = cfg.input_embedding_scalar;
}

void CudaGraphCapturePyModelInputs::initPrefillCudaGraphCopyParams(PyModelInputs& inputs) const {
    torch::Tensor cuda_graph_prefill_batch_size = torch::zeros({1}, options_cpu_int32_).pin_memory();
    cuda_graph_prefill_batch_size.fill_(1);
    RTP_LLM_CHECK_WITH_INFO(cuda_graph_prefill_batch_size.is_pinned(),
                            "prefill_cuda_graph_copy_params cuda_graph_prefill_batch_size is not pinned memory");
    inputs.attention_inputs.prefill_cuda_graph_copy_params = PyPrefillCudaGaphCopyParams{
        cuda_graph_prefill_batch_size, graph_params_.max_seq_len, static_cast<int>(max_bs_)};
}

CudaGraphCapturePyModelInputs::CudaGraphCapturePyModelInputs(const GraphParams& graph_params,
                                                             size_t             max_bs,
                                                             int                max_num_token,
                                                             const at::Tensor&  position_encoding,
                                                             const at::Tensor&  token_type_embedding,
                                                             float              input_embedding_scalar):
    graph_params_(graph_params),
    max_bs_(max_bs),
    max_num_token_(max_num_token),
    position_encoding_(position_encoding),
    token_type_embedding_(token_type_embedding),
    input_embedding_scalar_(input_embedding_scalar) {
    options_cuda_int32_ = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false);
    options_cpu_int32_  = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU).requires_grad(false);
    options_cuda_float_ =
        torch::TensorOptions().dtype(graph_params_.model_data_type).device(torch::kCUDA).requires_grad(false);
}

void CudaGraphCapturePyModelInputs::fillCuSeqlensForCapture(PyModelInputs& py_inputs, size_t max_bs) {
    torch::Tensor cu_seqlens = torch::zeros({int(max_bs + 1)}, torch::TensorOptions(torch::kInt32).device(torch::kCPU));
    torch::Tensor cu_kv_seqlens =
        torch::zeros({int(max_bs + 1)}, torch::TensorOptions(torch::kInt32).device(torch::kCPU));
    auto input_lengths  = py_inputs.attention_inputs.input_lengths;
    auto prefix_lengths = py_inputs.attention_inputs.prefix_lengths;

    cu_seqlens.slice(0, 1, max_bs + 1) = input_lengths.cumsum(0);
    if (prefix_lengths.defined()) {
        cu_kv_seqlens.slice(0, 1, max_bs + 1) = input_lengths.add(prefix_lengths).cumsum(0);
    }
    py_inputs.attention_inputs.cu_seqlens    = cu_seqlens.pin_memory();
    py_inputs.attention_inputs.cu_kv_seqlens = cu_kv_seqlens.pin_memory();
}

void CudaGraphCapturePyModelInputs::buildCaptureMemoryHold() {
    PyModelInputs inputs;
    initMaxCapturePyModelInputs(inputs);
    fillCuSeqlensForCapture(inputs, max_bs_);
    at::Tensor output;
    capture_mem_hold_ = CaptureMemoryHold(std::move(output), inputs);
}

void CudaGraphCapturePyModelInputs::allocateHiddenStatesAndPrefillCopyParams() {
    at::Tensor output =
        torch::zeros({max_num_token_, static_cast<int64_t>(graph_params_.hidden_size)}, options_cuda_float_);
    capture_mem_hold_.setHiddenStates(std::move(output));
    initPrefillCudaGraphCopyParams(capture_mem_hold_.py_model_inputs_);
}

void CudaGraphCapturePyModelInputs::patchForPrefillProbeForward() {
    capture_mem_hold_.py_model_inputs_.attention_inputs.cu_seqlens.data_ptr<int>()[1]    = max_num_token_;
    capture_mem_hold_.py_model_inputs_.attention_inputs.cu_kv_seqlens.data_ptr<int>()[1] = max_num_token_;
    capture_mem_hold_.py_model_inputs_.attention_inputs.input_lengths.data_ptr<int>()[0] = max_num_token_;
}

PyModelInputs CudaGraphCapturePyModelInputs::sliceForPrefillProbeForward() const {
    PyModelInputs sliced                                    = capture_mem_hold_.py_model_inputs_;
    const auto&   cap                                       = capture_mem_hold_.py_model_inputs_.attention_inputs;
    sliced.attention_inputs.cu_seqlens                      = cap.cu_seqlens.slice(0, 0, 2);
    sliced.attention_inputs.cu_kv_seqlens                   = cap.cu_kv_seqlens.slice(0, 0, 2);
    sliced.attention_inputs.input_lengths                   = cap.input_lengths.slice(0, 0, 1);
    sliced.attention_inputs.kv_cache_kernel_block_id_device = cap.kv_cache_kernel_block_id_device.slice(0, 0, 1);
    sliced.attention_inputs.kv_cache_kernel_block_id_host   = cap.kv_cache_kernel_block_id_host.slice(0, 0, 1);
    return sliced;
}

void CudaGraphCapturePyModelInputs::sliceTemplatePyModelInputsForCapture(PyModelInputs&       inputs,
                                                                         const PyModelInputs& cap_template,
                                                                         int                  batch_size,
                                                                         int                  seq_len_or_tokens,
                                                                         bool is_prefill_cuda_graph_mode,
                                                                         int  member_num_tokens_per_bs,
                                                                         bool is_target_verify) {
    inputs.attention_inputs.is_prefill       = is_prefill_cuda_graph_mode || member_num_tokens_per_bs > 1;
    inputs.attention_inputs.is_target_verify = is_target_verify;
    inputs.input_ids                         = cap_template.input_ids.slice(0, 0, seq_len_or_tokens);
    inputs.input_hiddens                     = cap_template.input_hiddens.slice(0, 0, seq_len_or_tokens);
    inputs.attention_inputs.input_lengths    = cap_template.attention_inputs.input_lengths.slice(0, 0, batch_size);
    inputs.attention_inputs.padding_offset =
        cap_template.attention_inputs.padding_offset.slice(0, 0, seq_len_or_tokens);

    if (cap_template.attention_inputs.prefix_lengths.defined()) {
        inputs.attention_inputs.prefix_lengths = cap_template.attention_inputs.prefix_lengths.slice(0, 0, batch_size);
    }
    inputs.attention_inputs.sequence_lengths = cap_template.attention_inputs.sequence_lengths.slice(0, 0, batch_size);

    inputs.attention_inputs.kv_cache_kernel_block_id_device =
        cap_template.attention_inputs.kv_cache_kernel_block_id_device.slice(0, 0, batch_size);
    inputs.attention_inputs.kv_cache_kernel_block_id_host =
        cap_template.attention_inputs.kv_cache_kernel_block_id_host.slice(0, 0, batch_size);
    inputs.attention_inputs.kv_cache_block_id_device = inputs.attention_inputs.kv_cache_kernel_block_id_device;
    inputs.attention_inputs.kv_cache_block_id_host   = inputs.attention_inputs.kv_cache_kernel_block_id_host;
    inputs.attention_inputs.cu_seqlens    = cap_template.attention_inputs.cu_seqlens.slice(0, 0, batch_size + 1);
    inputs.attention_inputs.cu_kv_seqlens = cap_template.attention_inputs.cu_kv_seqlens.slice(0, 0, batch_size + 1);
    inputs.attention_inputs.decode_cu_seqlens_d =
        cap_template.attention_inputs.decode_cu_seqlens_d.slice(0, 0, batch_size + 1);
    inputs.attention_inputs.sequence_lengths_plus_1_d =
        cap_template.attention_inputs.sequence_lengths_plus_1_d.slice(0, 0, batch_size);

    const auto& cap_attn = cap_template.attention_inputs;
    inputs.attention_inputs.kv_cache_kernel_block_id_device_by_group.clear();
    inputs.attention_inputs.kv_cache_kernel_block_id_host_by_group.clear();
    if (!cap_attn.kv_cache_kernel_block_id_device_by_group.empty()
        && !cap_attn.kv_cache_kernel_block_id_host_by_group.empty()) {
        const size_t group = cap_attn.kv_cache_kernel_block_id_device_by_group.size();
        inputs.attention_inputs.kv_cache_kernel_block_id_device_by_group.reserve(group);
        inputs.attention_inputs.kv_cache_kernel_block_id_host_by_group.reserve(group);
        for (size_t g = 0; g < group; ++g) {
            inputs.attention_inputs.kv_cache_kernel_block_id_device_by_group.push_back(
                cap_attn.kv_cache_kernel_block_id_device_by_group[g].slice(0, 0, batch_size));
            inputs.attention_inputs.kv_cache_kernel_block_id_host_by_group.push_back(
                cap_attn.kv_cache_kernel_block_id_host_by_group[g].slice(0, 0, batch_size));
        }
    }

    inputs.attention_inputs.dtype                   = cap_template.attention_inputs.dtype;
    inputs.attention_inputs.kv_cache_layer_to_group = cap_template.attention_inputs.kv_cache_layer_to_group;
    inputs.bert_embedding_inputs                    = cap_template.bert_embedding_inputs;
    inputs.attention_inputs.is_s_padded             = true;
}

void CudaGraphCapturePyModelInputs::copyRuntimePyModelIntoCaptureBuffers(const PyModelInputs&   runtime,
                                                                         PyModelInputs&         cap,
                                                                         const BatchDescriptor& batch_descriptor,
                                                                         bool             is_prefill_cuda_graph_mode,
                                                                         pybind11::object decode_attn_pyobj) {
    if (!is_prefill_cuda_graph_mode) {
        cap.attention_inputs.kv_cache_kernel_block_id_device.fill_(0);

        CudaGraphCapturePyModelInputs::optimizedCopyAsync(runtime.attention_inputs.prefix_lengths,
                                                          cap.attention_inputs.prefix_lengths,
                                                          static_cast<size_t>(batch_descriptor.current_batch_size)
                                                              * sizeof(int));
        CudaGraphCapturePyModelInputs::optimizedCopyAsync(
            runtime.input_ids, cap.input_ids, runtime.input_ids.size(0) * sizeof(int));

        CudaGraphCapturePyModelInputs::optimizedCopyAsync(runtime.input_hiddens,
                                                          cap.input_hiddens,
                                                          runtime.input_hiddens.numel()
                                                              * runtime.input_hiddens.element_size());

        CudaGraphCapturePyModelInputs::optimizedCopyAsync(runtime.attention_inputs.sequence_lengths,
                                                          cap.attention_inputs.sequence_lengths,
                                                          static_cast<size_t>(batch_descriptor.current_batch_size)
                                                              * sizeof(int));

        CudaGraphCapturePyModelInputs::copySmallerIntoLarger(runtime.attention_inputs.kv_cache_kernel_block_id_device,
                                                             cap.attention_inputs.kv_cache_kernel_block_id_device);
        CudaGraphCapturePyModelInputs::copySmallerIntoLarger(runtime.attention_inputs.kv_cache_kernel_block_id_host,
                                                             cap.attention_inputs.kv_cache_kernel_block_id_host);
        CudaGraphCapturePyModelInputs::optimizedCopyAsync(runtime.attention_inputs.sequence_lengths_plus_1_d,
                                                          cap.attention_inputs.sequence_lengths_plus_1_d,
                                                          static_cast<size_t>(batch_descriptor.current_batch_size)
                                                              * sizeof(int));
        CudaGraphCapturePyModelInputs::optimizedCopyAsync(runtime.attention_inputs.decode_cu_seqlens_d,
                                                          cap.attention_inputs.decode_cu_seqlens_d,
                                                          static_cast<size_t>(batch_descriptor.current_batch_size + 1)
                                                              * sizeof(int));
        decode_attn_pyobj.attr("prepare_cuda_graph")(cap.attention_inputs);
    } else {
        cap.attention_inputs.kv_cache_kernel_block_id_device.fill_(0);

        CudaGraphCapturePyModelInputs::optimizedCopyAsync(
            runtime.input_ids, cap.input_ids, static_cast<size_t>(batch_descriptor.current_seq_len) * sizeof(int));

        CudaGraphCapturePyModelInputs::optimizedCopyAsync(runtime.attention_inputs.padding_offset,
                                                          cap.attention_inputs.padding_offset,
                                                          static_cast<size_t>(batch_descriptor.current_seq_len)
                                                              * sizeof(int));

        if (cap.attention_inputs.prefill_cuda_graph_copy_params) {
            (*(cap.attention_inputs.prefill_cuda_graph_copy_params->cuda_graph_prefill_batch_size.data_ptr<int>())) =
                batch_descriptor.current_batch_size;
        }

        if (runtime.bert_embedding_inputs.position_encoding.numel() > 0) {
            CudaGraphCapturePyModelInputs::optimizedCopyAsync(runtime.bert_embedding_inputs.combo_position_ids,
                                                              cap.bert_embedding_inputs.combo_position_ids,
                                                              static_cast<size_t>(batch_descriptor.current_seq_len)
                                                                  * sizeof(int));

            CudaGraphCapturePyModelInputs::optimizedCopyAsync(runtime.bert_embedding_inputs.combo_tokens_type_ids,
                                                              cap.bert_embedding_inputs.combo_tokens_type_ids,
                                                              static_cast<size_t>(batch_descriptor.current_seq_len)
                                                                  * sizeof(int));
        }
    }

    CudaGraphCapturePyModelInputs::optimizedCopyAsync(runtime.attention_inputs.input_lengths,
                                                      cap.attention_inputs.input_lengths,
                                                      static_cast<size_t>(batch_descriptor.current_batch_size)
                                                          * sizeof(int));

    CudaGraphCapturePyModelInputs::optimizedCopyAsync(runtime.attention_inputs.cu_seqlens,
                                                      cap.attention_inputs.cu_seqlens,
                                                      static_cast<size_t>(batch_descriptor.current_batch_size + 1)
                                                          * sizeof(int));

    CudaGraphCapturePyModelInputs::optimizedCopyAsync(runtime.attention_inputs.cu_kv_seqlens,
                                                      cap.attention_inputs.cu_kv_seqlens,
                                                      static_cast<size_t>(batch_descriptor.current_batch_size + 1)
                                                          * sizeof(int));

    if (!runtime.attention_inputs.kv_cache_kernel_block_id_device_by_group.empty()
        && !runtime.attention_inputs.kv_cache_kernel_block_id_host_by_group.empty()
        && !cap.attention_inputs.kv_cache_kernel_block_id_device_by_group.empty()
        && !cap.attention_inputs.kv_cache_kernel_block_id_host_by_group.empty()) {
        RTP_LLM_CHECK_WITH_INFO(runtime.attention_inputs.kv_cache_kernel_block_id_device_by_group.size()
                                    == cap.attention_inputs.kv_cache_kernel_block_id_device_by_group.size(),
                                "kv_cache_kernel_block_id_device_by_group size mismatch");
        const size_t group = runtime.attention_inputs.kv_cache_kernel_block_id_device_by_group.size();
        RTP_LLM_CHECK_WITH_INFO(runtime.attention_inputs.kv_cache_kernel_block_id_host_by_group.size() == group
                                    && cap.attention_inputs.kv_cache_kernel_block_id_host_by_group.size() == group,
                                "kv_cache_kernel_block_id_host_by_group size mismatch");
        for (size_t g = 0; g < group; ++g) {
            CudaGraphCapturePyModelInputs::copySmallerIntoLarger(
                runtime.attention_inputs.kv_cache_kernel_block_id_device_by_group[g],
                cap.attention_inputs.kv_cache_kernel_block_id_device_by_group[g]);
            CudaGraphCapturePyModelInputs::copySmallerIntoLarger(
                runtime.attention_inputs.kv_cache_kernel_block_id_host_by_group[g],
                cap.attention_inputs.kv_cache_kernel_block_id_host_by_group[g]);
        }
    }

    CudaGraphCapturePyModelInputs::optimizedCopyAsync(
        runtime.attention_inputs.kv_cache_layer_to_group,
        cap.attention_inputs.kv_cache_layer_to_group,
        static_cast<size_t>(runtime.attention_inputs.kv_cache_layer_to_group.numel()) * sizeof(int32_t));
}

}  // namespace cuda_graph
}  // namespace rtp_llm
