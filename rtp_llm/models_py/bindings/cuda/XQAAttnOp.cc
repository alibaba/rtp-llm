
#ifdef USING_CUDA12
#include "rtp_llm/models_py/bindings/cuda/XQAAttnOp.h"
#include "rtp_llm/cpp/cuda/cufmha/TrtV2FmhaRunner.h"
#include <ATen/cuda/CUDAContext.h>

namespace rtp_llm {

XQAAttnOp::XQAAttnOp(const AttentionConfigs& attn_configs): attn_configs_(attn_configs) {}
XQASpecAttnOp::XQASpecAttnOp(const AttentionConfigs& attn_configs): attn_configs_(attn_configs) {}

bool XQAAttnOp::support(torch_ext::PyAttentionInputs attn_inputs) {
    if (attn_inputs.is_target_verify) {
        return false;
    }
    return attn_configs_.kv_cache_dtype != KvCacheDataType::INT8 && get_sm() >= tensorrt_llm::kernels::kSM_90
           && supportXqa(DataType::TYPE_BF16,
                         DataType::TYPE_BF16,
                         DataType::TYPE_FP8_E4M3,
                         attn_configs_.head_num / attn_configs_.kv_head_num,
                         attn_configs_.size_per_head,
                         attn_configs_.kernel_tokens_per_block);
}

ParamsBasePtr XQAAttnOp::prepare(torch_ext::PyAttentionInputs attn_inputs) {
    XQAParamsPtr params     = std::make_shared<XQAParams>();
    int          batch_size = attn_inputs.sequence_lengths.size(0);
    RTP_LLM_CHECK_WITH_INFO(attn_inputs.kv_cache_kernel_block_id_host.defined()
                                && attn_inputs.kv_cache_kernel_block_id_device.defined(),
                            "decode should have kv cache block id.");

    // 使用独立的工具函数准备 TRT attention 参数
    auto run_stream   = at::cuda::getCurrentCUDAStream(at::cuda::current_device()).stream();
    bool use_fp8_fmha = attn_configs_.kv_cache_dtype == KvCacheDataType::FP8;
    auto trt_params   = prepareTrtAttnParams(
        attn_configs_, attn_inputs.kv_cache_kernel_block_id_device, batch_size, use_fp8_fmha, run_stream, false);

    params->kv_block_array            = ((TRTAttn*)trt_params.get())->kv_block_array;
    params->kv_cache_offset           = ((TRTAttn*)trt_params.get())->kv_cache_offset.clone();
    params->batch_size                = batch_size;
    params->max_seq_len               = attn_inputs.sequence_lengths.max().item<int32_t>();
    params->sequence_lengths          = attn_inputs.sequence_lengths;
    params->kv_block_array.cache_type = attn_configs_.kv_cache_dtype;

    return ParamsBasePtr(params);
}

bool XQASpecAttnOp::support(torch_ext::PyAttentionInputs attn_inputs) {
    if (!attn_inputs.is_target_verify || !attn_inputs.decode_cu_seqlens_d.defined()
        || attn_inputs.decode_cu_seqlens_d.numel() <= 1 || attn_configs_.kv_cache_dtype != KvCacheDataType::FP8
        || get_sm() != tensorrt_llm::kernels::kSM_90) {
        return false;
    }
    const auto input_type = attn_configs_.dtype == torch::kBFloat16 ? DataType::TYPE_BF16 : DataType::TYPE_FP16;
    const auto kv_type    = DataType::TYPE_FP8_E4M3;
    return supportXqa(input_type,
                      input_type,
                      kv_type,
                      attn_configs_.head_num / attn_configs_.kv_head_num,
                      attn_configs_.size_per_head,
                      attn_configs_.kernel_tokens_per_block);
}

ParamsBasePtr XQASpecAttnOp::prepare(torch_ext::PyAttentionInputs attn_inputs) {
    XQAParamsPtr params     = std::make_shared<XQAParams>();
    int          batch_size = attn_inputs.sequence_lengths.size(0);
    RTP_LLM_CHECK_WITH_INFO(attn_inputs.kv_cache_kernel_block_id_host.defined()
                                && attn_inputs.kv_cache_kernel_block_id_device.defined(),
                            "decode should have kv cache block id.");

    auto run_stream   = at::cuda::getCurrentCUDAStream(at::cuda::current_device()).stream();
    bool use_fp8_fmha = attn_configs_.kv_cache_dtype == KvCacheDataType::FP8;
    auto trt_params   = prepareTrtAttnParams(
        attn_configs_, attn_inputs.kv_cache_kernel_block_id_device, batch_size, use_fp8_fmha, run_stream, false);
    params->kv_block_array  = ((TRTAttn*)trt_params.get())->kv_block_array;
    params->kv_cache_offset = ((TRTAttn*)trt_params.get())->kv_cache_offset.clone();
    params->batch_size      = batch_size;
    params->sequence_lengths =
        (attn_inputs.sequence_lengths + attn_inputs.input_lengths)
            .to(torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA), /*non_blocking=*/true);
    params->q_cu_seqlens = attn_inputs.decode_cu_seqlens_d;
    params->max_q_len    = static_cast<size_t>(
        (attn_inputs.decode_cu_seqlens_d.slice(0, 1) - attn_inputs.decode_cu_seqlens_d.slice(0, 0, -1))
            .max()
            .item<int32_t>());
    params->max_seq_len =
        attn_inputs.input_lengths.max().item<int32_t>() + attn_inputs.prefix_lengths.max().item<int32_t>();
    params->kv_block_array.cache_type = attn_configs_.kv_cache_dtype;

    return ParamsBasePtr(params);
}

torch::Tensor XQAAttnOp::forward(const torch::Tensor&                   input,
                                 std::optional<torch_ext::LayerKVCache> kv_cache,
                                 const XQAParamsPtr&                    params) {
    const int            batch_size        = input.size(0);
    const int            local_head_num    = attn_configs_.head_num;
    const int            local_head_num_kv = attn_configs_.kv_head_num;
    const int            size_per_head     = attn_configs_.size_per_head;
    torch::TensorOptions options           = torch::TensorOptions(input.dtype()).device(input.device());

    torch::Tensor output = torch::empty({batch_size, local_head_num * size_per_head}, options);

    KVBlockArray kv_block_array;
    if (kv_cache.has_value()) {
        kv_block_array                 = params->kv_block_array;
        kv_block_array.mPrimaryPoolPtr = kv_cache.value().kv_cache_base.data_ptr();
        if (kv_cache.value().kv_scale_base.defined() && kv_cache.value().kv_scale_base.numel() > 0) {
            kv_block_array.scale = kv_cache.value().kv_scale_base.data_ptr();
        }
    }

    RTP_LLM_CHECK_WITH_INFO(kv_cache.has_value(), "decode should have kv cache.");

    runXqa(input.data_ptr(),
           input.dtype() == torch::kBFloat16,
           output.data_ptr(),
           local_head_num,
           local_head_num_kv,
           size_per_head,
           params->batch_size,
           static_cast<size_t>(kv_block_array.mMaxBlocksPerSeq),
           params->max_seq_len + 1,
           attn_configs_.kernel_tokens_per_block,
           kv_cache.value().kv_cache_base.data_ptr(),  // params->kv_block_array.mPrimaryPoolPtr,
           reinterpret_cast<int32_t*>((KVCacheIndex*)(params->kv_cache_offset.data_ptr())),
           kv_block_array.cache_type == KvCacheDataType::FP8,
           reinterpret_cast<uint32_t*>(params->sequence_lengths.data_ptr()));
    return output;
}

torch::Tensor XQASpecAttnOp::forward(const torch::Tensor&                   input,
                                     std::optional<torch_ext::LayerKVCache> kv_cache,
                                     const XQAParamsPtr&                    params) {
    const int            batch_size        = params->batch_size;
    const int            local_head_num    = attn_configs_.head_num;
    const int            local_head_num_kv = attn_configs_.kv_head_num;
    const int            size_per_head     = attn_configs_.size_per_head;
    torch::TensorOptions options           = torch::TensorOptions(input.dtype()).device(input.device());
    torch::Tensor        output =
        torch::empty({batch_size, static_cast<int64_t>(params->max_q_len), local_head_num, size_per_head}, options);

    KVBlockArray kv_block_array;
    if (kv_cache.has_value()) {
        kv_block_array                 = params->kv_block_array;
        kv_block_array.mPrimaryPoolPtr = kv_cache.value().kv_cache_base.data_ptr();
        if (kv_cache.value().kv_scale_base.defined() && kv_cache.value().kv_scale_base.numel() > 0) {
            kv_block_array.scale = kv_cache.value().kv_scale_base.data_ptr();
        }
    }

    RTP_LLM_CHECK_WITH_INFO(kv_cache.has_value(), "spec decode should have kv cache.");

    torch::Tensor xqa_input = input.contiguous();
    runXqa(xqa_input.data_ptr(),
           input.dtype() == torch::kBFloat16,
           output.data_ptr(),
           local_head_num,
           local_head_num_kv,
           size_per_head,
           params->batch_size,
           static_cast<size_t>(kv_block_array.mMaxBlocksPerSeq),
           params->max_seq_len,
           attn_configs_.kernel_tokens_per_block,
           kv_block_array.mPrimaryPoolPtr,
           reinterpret_cast<int32_t*>((KVCacheIndex*)(params->kv_cache_offset.data_ptr())),
           kv_block_array.cache_type == KvCacheDataType::FP8,
           reinterpret_cast<uint32_t*>(params->sequence_lengths.data_ptr()),
           nullptr,
           params->max_q_len,
           params->q_cu_seqlens.data_ptr());
    return output;
}

void registerXQAAttnOp(const py::module& m) {
    pybind11::class_<XQAParams, std::shared_ptr<XQAParams>, rtp_llm::ParamsBase>(m, "XQAParams")
        .def(pybind11::init<>())
        .def(
            "__cpp_ptr__",
            [](XQAParams& self) { return reinterpret_cast<uintptr_t>(&self); },
            "Get C++ object pointer address")
        .def_readwrite("kv_cache_offset", &XQAParams::kv_cache_offset);
    pybind11::class_<XQAAttnOp>(m, "XQAAttnOp")
        .def(pybind11::init<const AttentionConfigs&>(), py::arg("attn_configs"))
        .def("support", &XQAAttnOp::support, py::arg("attn_inputs").noconvert())
        .def("prepare", &XQAAttnOp::prepare, py::arg("attn_inputs"))
        .def("forward", &XQAAttnOp::forward, py::arg("input"), py::arg("kv_cache"), py::arg("params"));
    pybind11::class_<XQASpecAttnOp>(m, "XQASpecAttnOp")
        .def(pybind11::init<const AttentionConfigs&>(), py::arg("attn_configs"))
        .def("support", &XQASpecAttnOp::support, py::arg("attn_inputs").noconvert())
        .def("prepare", &XQASpecAttnOp::prepare, py::arg("attn_inputs"))
        .def("forward", &XQASpecAttnOp::forward, py::arg("input"), py::arg("kv_cache"), py::arg("params"));
}

}  // namespace rtp_llm
#endif
