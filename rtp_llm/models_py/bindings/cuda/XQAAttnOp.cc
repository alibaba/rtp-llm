#ifdef USING_CUDA12

#include "rtp_llm/models_py/bindings/cuda/XQAAttnOp.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/devices/cuda_impl/CudaDevice.h"
#include "rtp_llm/cpp/kernels/unfused_attention_kernels.h"
#include "rtp_llm/cpp/cuda/Dispatch.h"
#include "rtp_llm/cpp/devices/DeviceFactory.h"

namespace rtp_llm {

XQAAttnOp::XQAAttnOp(const GptInitParameter& gpt_init_parameter): FMHACudaBase(gpt_init_parameter) {}

bool XQAAttnOp::support(torch_ext::PyAttentionInputs attn_inputs) {
    return fmha_config_.enable_xqa && attn_configs_.kv_cache_dtype == KvCacheDataType::FP8;
}

XQAParamsPtr XQAAttnOp::prepare(torch_ext::PyAttentionInputs attn_inputs) {
    XQAParamsPtr params     = std::make_shared<XQAParams>();
    int          batch_size = attn_inputs.sequence_lengths.size(0);
    BufferPtr    kv_cache_block_id_host, kv_cache_block_id_device;
    if (attn_inputs.kv_cache_block_id_host.size(0)) {
        kv_cache_block_id_host   = torchTensor2Buffer(attn_inputs.kv_cache_block_id_host);
        kv_cache_block_id_device = torchTensor2Buffer(attn_inputs.kv_cache_block_id_device);
    }
    // not support has_alibi_slopes

    auto trt_params =
        device_->prepareTrtAttn(attn_configs_, attn_inputs.kv_block_offset, kv_cache_block_id_device, batch_size);
    params->kv_block_array            = ((TRTAttn*)trt_params.get())->kv_block_array;
    params->batch_size                = batch_size;
    params->max_seq_len               = attn_inputs.sequence_lengths.max().item<int32_t>();
    params->sequence_lengths          = attn_inputs.sequence_lengths.cuda();
    params->kv_block_array.cache_type = attn_configs_.kv_cache_dtype;
    return params;
}

torch::Tensor
XQAAttnOp::forward(const torch::Tensor& input, std::optional<torch_ext::KVCache> kv_cache, const XQAParamsPtr& params) {
    const int            batch_size        = input.size(0);
    const int            local_head_num    = attn_configs_.head_num;
    const int            local_head_num_kv = attn_configs_.kv_head_num;
    const int            size_per_head     = attn_configs_.size_per_head;
    torch::TensorOptions options           = torch::TensorOptions(input.dtype()).device(input.device());

    torch::Tensor output = torch::empty({batch_size, local_head_num * size_per_head}, options);
    RTP_LLM_CHECK_WITH_INFO(kv_cache.has_value(), "decode should have kv cache.");

    runXqa(input.data_ptr(),
           true,
           output.data_ptr(),
           local_head_num,
           local_head_num_kv,
           size_per_head,
           params->batch_size,
           params->max_seq_len + 1,
           attn_configs_.tokens_per_block,
           kv_cache.value().k_cache_base.data_ptr(),  // params->kv_block_array.mPrimaryPoolPtr,
           reinterpret_cast<int32_t*>(const_cast<KVCacheIndex*>(params->kv_block_array.data)),
           true,
           reinterpret_cast<uint32_t*>(params->sequence_lengths.data_ptr()),
           device_);
    return output;
}

void registerXQAAttnOp(const py::module& m) {
    pybind11::class_<XQAParams, std::shared_ptr<XQAParams>>(m, "XQAParams").def(pybind11::init<>());
    pybind11::class_<XQAAttnOp>(m, "XQAAttnOp")
        .def(pybind11::init<GptInitParameter>(), py::arg("gpt_init_parameter"))
        .def("support", &XQAAttnOp::support, py::arg("attn_inputs"))
        .def("prepare", &XQAAttnOp::prepare, py::arg("attn_inputs"))
        .def("forward", &XQAAttnOp::forward, py::arg("input"), py::arg("kv_cache"), py::arg("params"));
}

}  // namespace rtp_llm
#endif
