#include "rtp_llm/models_py/bindings/cuda/TRTAttnOp.h"
#include "rtp_llm/cpp/devices/DeviceFactory.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/core/Dispatch.h"
#include "rtp_llm/cpp/kernels/decoder_masked_multihead_attention/decoder_masked_multihead_attention.h"
#include "rtp_llm/cpp/kernels/unfused_attention_kernels.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"

using namespace torch_ext;

namespace rtp_llm {

TRTPrefillOp::TRTPrefillOp(const GptInitParameter& gpt_init_parameter): FMHACudaBase(gpt_init_parameter) {}

bool TRTPrefillOp::support(torch_ext::PyAttentionInputs attn_inputs) {
    return fmha_config_.enable_paged_trt_fmha && attn_configs_.kv_cache_dtype != KvCacheDataType::INT8;
}

ParamsBasePtr TRTPrefillOp::prepare(torch_ext::PyAttentionInputs attn_inputs) {
    static_scale_        = torch::ones({1}, torch::TensorOptions(torch::kFloat32).device(torch::kCUDA));
    int       batch_size = attn_inputs.input_lengths.size(0);
    BufferPtr kv_cache_block_id_host, kv_cache_block_id_device;
    if (attn_inputs.kv_cache_block_id_host.defined() && attn_inputs.kv_cache_block_id_host.numel() > 0) {
        kv_cache_block_id_host   = torchTensor2Buffer(attn_inputs.kv_cache_block_id_host);
        kv_cache_block_id_device = torchTensor2Buffer(attn_inputs.kv_cache_block_id_device);
    }

    auto          cu_seqlens    = attn_inputs.cu_seqlens;
    torch::Tensor cu_kv_seqlens = cu_seqlens;
    TRTAttnPtr    attn_params;
    auto          params = device_->prepareTrtAttn(
        attn_configs_, attn_inputs.kv_block_offset, kv_cache_block_id_device, attn_inputs.input_lengths.size(0));
    if (params) {
        attn_params = TRTAttnPtr(params, (TRTAttn*)params.get());
    } else {
        attn_params = std::make_shared<TRTAttn>();
    }
    attn_params->attn_type     = torchDTypeToDataType(attn_inputs.dtype);
    attn_params->cu_seqlens    = cu_seqlens;
    attn_params->cu_kv_seqlens = cu_kv_seqlens;
    attn_params->max_seq_len   = attn_inputs.input_lengths.max().item<int32_t>();
    attn_params->input_lengths = attn_inputs.input_lengths;
    // not support has_alibi_slopes
    DataType attn_dtype = attn_configs_.kv_cache_dtype == KvCacheDataType::FP8 ?
                              DataType::TYPE_FP8_E4M3 :
                              torchDTypeToDataType(attn_inputs.dtype);
    cufmha_runner_      = device_->selectCuFMHARunner(attn_configs_, attn_dtype, false);
    return ParamsBasePtr(attn_params);
}

// template<typename T>
// void maskedAttentionWrapper(const AttentionConfigs& attn_configs,
//                             const torch::Tensor&    input,
//                             const torch::Tensor&    output,
//                             const TRTAttnPtr&       params,
//                             cudaStream_t            stream) {
//     size_t batch_size        = params->sequence_lengths.size(0);
//     size_t step              = params->max_seq_len + 1;
//     size_t local_head_num    = attn_configs.head_num;
//     size_t local_head_num_kv = attn_configs.kv_head_num;
//     size_t size_per_head     = attn_configs.size_per_head;

//     const T* qkv_buf_ptr  = (T*)input.data_ptr();
//     void*    attn_out_ptr = output.data_ptr();

//     tensorrt_llm::common::QuantMode kv_cache_quant_mode =
//         tensorrt_llm::common::QuantMode::fromDescription(false, false, false, false, false, false, false, false);

//     fusedQKV_masked_attention_dispatch<T, KVBlockArray>(
//         qkv_buf_ptr,
//         nullptr,  // bias_ptr,
//         nullptr,  // relative_attention_bias
//         nullptr,  // cache_indir
//         reinterpret_cast<T*>(attn_out_ptr),
//         nullptr,  // finished
//         params->sequence_lengths.data_ptr<int>(),
//         batch_size,
//         1,  // beam_width
//         local_head_num,
//         local_head_num_kv,
//         size_per_head,
//         attn_configs.rope_config,
//         attn_configs.use_logn_attn,
//         nullptr,  // params.common.position_ids ? params.common.position_ids->data<int>() : nullptr,
//         step,
//         nullptr,  // prefix_prompt_lengths
//         0,        // max_prefix_prompt_length
//         true,     // count_prefix_length
//         params->input_lengths.data_ptr<int>(),
//         step,
//         attn_configs.q_scaling,
//         0,        // relative_attention_bias_stride,
//         nullptr,  // linear_bias_slopes,
//         nullptr,  // masked_tokens,
//         nullptr,  // query_weight_scale_out
//         nullptr,  // attention_output_orig_quant_scale
//         0,        // int8_mode,
//         kv_cache_quant_mode,
//         false,    // use_multi_block_mode,
//         0,        // (int)max_seq_len_tile,
//         nullptr,  // reinterpret_cast<T*>(partial_out),
//         nullptr,  // partial_sum,
//         nullptr,  // partial_max,
//         nullptr,  // block_counter,
//         attn_configs.softmax_extra_scale,
//         params->kv_block_array,
//         stream);
//     check_cuda_error();
// }

torch::Tensor TRTPrefillOp::forward(const torch::Tensor&              input,
                                    std::optional<torch_ext::KVCache> kv_cache,
                                    const TRTAttnPtr&                 params) {
    // DISPATCH_CUDA_FUNCTION_DATA_TYPE(torchDTypeToDataType(input.dtype()),
    //                                  invoke_debug_kernel2,
    //                                  input.data_ptr(),
    //                                  0,
    //                                  0,
    //                                  220,
    //                                  10,
    //                                  2304,
    //                                  1,
    //                                  device_->getStream());
    KVBlockArray kv_block_array;
    if (kv_cache.has_value()) {
        kv_block_array                 = params->kv_block_array;
        kv_block_array.mPrimaryPoolPtr = kv_cache.value().k_cache_base.data_ptr();
        if (kv_cache.value().k_scale_base.defined() && kv_cache.value().k_scale_base.numel() > 0) {
            kv_block_array.scale = kv_cache.value().k_scale_base.data_ptr();
        }
    }

    const int local_head_num = attn_configs_.head_num;
    const int size_per_head  = attn_configs_.size_per_head;
    const int token_num      = input.size(0);
    const int batch_size     = params->input_lengths.size(0);
    const int max_token_num =
        device_->initParams().fifo_scheduler_config.max_context_batch_size * device_->initParams().max_seq_len;
    torch::TensorOptions options = torch::TensorOptions(input.dtype()).device(input.device());

    static torch::Tensor static_output = torch::zeros({max_token_num, local_head_num * size_per_head}, options);
    torch::Tensor        output        = static_output.slice(0, 0, token_num);
    torch::Tensor        tiled_counter = torch::zeros({1}, torch::TensorOptions(torch::kUInt32).device(input.device()));
    bool                 use_fp8_fmha  = kv_block_array.cache_type == KvCacheDataType::FP8;
    float*               attention_output_orig_quant_scale = use_fp8_fmha ? static_scale_.data_ptr<float>() : nullptr;
    if (kv_cache.has_value() && kv_block_array.cache_type == KvCacheDataType::BASE) {
        // TODO@miji: fix params
        cufmha_runner_->runTrtV2FmhaPaged(input.data_ptr(),
                                          params->cu_seqlens.data_ptr(),
                                          params->cu_kv_seqlens.data_ptr(),
                                          output.data_ptr(),
                                          reinterpret_cast<uint32_t*>(tiled_counter.data_ptr()),
                                          attention_output_orig_quant_scale,
                                          batch_size,  // batch_size,
                                          params->max_seq_len,
                                          params->max_seq_len,  // seq_len_with_prefix,
                                          token_num,
                                          token_num,  // token_num_kv,
                                          kv_block_array);
    } else {
        torch::Tensor tmp_fmha_input, tmp_fmha_output;
        void*         fmha_input_ptr  = input.data_ptr();
        void*         fmha_output_ptr = output.data_ptr();
        RTP_LLM_CHECK_WITH_INFO(fmha_input_ptr, "fmha_input_ptr must be provided for trt v2 fmha");

        if (use_fp8_fmha) {
            tmp_fmha_input  = input.to(torch::kFloat8_e4m3fn);
            tmp_fmha_output = output.to(torch::kFloat8_e4m3fn);
            fmha_input_ptr  = tmp_fmha_input.data_ptr();
            fmha_output_ptr = tmp_fmha_output.data_ptr();
        }
        RTP_LLM_CHECK_WITH_INFO(fmha_output_ptr, "fmha_output_ptr must be provided for trt v2 fmha");

        cufmha_runner_->runTrtV2Fmha(fmha_input_ptr,
                                     params->cu_seqlens.data_ptr(),
                                     fmha_output_ptr,
                                     reinterpret_cast<uint32_t*>(tiled_counter.data_ptr()),
                                     attention_output_orig_quant_scale,
                                     batch_size,
                                     params->max_seq_len,
                                     token_num,
                                     kv_block_array);
        if (use_fp8_fmha) {
            output = tmp_fmha_output.to(output.dtype());
        }
    }
    // DISPATCH_CUDA_FUNCTION_DATA_TYPE(torchDTypeToDataType(output.dtype()),
    //                                  invoke_debug_kernel2,
    //                                  output.data_ptr(),
    //                                  0,
    //                                  0,
    //                                  30,
    //                                  10,
    //                                  768,
    //                                  2,
    //                                  device_->getStream());
    // DISPATCH_CUDA_FUNCTION_DATA_TYPE(torchDTypeToDataType(output.dtype()),
    //                                  invoke_debug_kernel2,
    //                                  output.data_ptr(),
    //                                  208,
    //                                  0,
    //                                  10,
    //                                  10,
    //                                  768,
    //                                  3,
    //                                  device_->getStream());
    return output;
}

void registerTRTAttnOp(const py::module& m) {
    pybind11::class_<TRTPrefillOp>(m, "TRTAttnOp")
        .def(pybind11::init<GptInitParameter>(), py::arg("gpt_init_parameter"))
        .def("support", &TRTPrefillOp::support, py::arg("attn_inputs"))
        .def("prepare", &TRTPrefillOp::prepare, py::arg("attn_inputs"))
        .def("forward", &TRTPrefillOp::forward, py::arg("input"), py::arg("kv_cache"), py::arg("params"));
}

}  // namespace rtp_llm
