
#ifdef USING_CUDA12
#include "rtp_llm/models_py/bindings/cuda/XQAAttnOp.h"
#include "rtp_llm/cpp/cuda/cufmha/TrtV2FmhaRunner.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"

namespace rtp_llm {

XQAAttnOp::XQAAttnOp(const AttentionConfigs& attn_configs):
    attn_configs_(attn_configs),
    device_(dynamic_cast<CudaDevice*>(DeviceFactory::getDefaultDevice())) {}

bool XQAAttnOp::support(torch_ext::PyAttentionInputs attn_inputs) {
    return attn_configs_.kv_cache_dtype != KvCacheDataType::INT8
           && get_sm() >= tensorrt_llm::kernels::kSM_90
           && supportXqa(DataType::TYPE_BF16,
                         DataType::TYPE_BF16,
                         DataType::TYPE_FP8_E4M3,
                         attn_configs_.head_num / attn_configs_.kv_head_num,
                         attn_configs_.size_per_head,
                         attn_configs_.tokens_per_block);
}

ParamsBasePtr XQAAttnOp::prepare(torch_ext::PyAttentionInputs attn_inputs) {
    XQAParamsPtr params     = std::make_shared<XQAParams>();
    int          batch_size = attn_inputs.sequence_lengths.size(0);
    BufferPtr    kv_cache_block_id_host, kv_cache_block_id_device;
    RTP_LLM_CHECK_WITH_INFO(attn_inputs.kv_cache_block_id_host.defined()
                                && attn_inputs.kv_cache_block_id_device.defined(),
                            "decode should have kv cache block id.");
    kv_cache_block_id_host   = torchTensor2Buffer(attn_inputs.kv_cache_block_id_host);
    kv_cache_block_id_device = torchTensor2Buffer(attn_inputs.kv_cache_block_id_device);

    // 使用独立的工具函数准备 TRT attention 参数
    auto run_stream   = at::cuda::getCurrentCUDAStream(at::cuda::current_device()).stream();
    bool use_fp8_fmha = attn_configs_.kv_cache_dtype == KvCacheDataType::FP8;
    auto trt_params   = prepareTrtAttnParams(attn_configs_,
                                           attn_inputs.kv_block_offset,
                                           kv_cache_block_id_device,
                                           batch_size,
                                           use_fp8_fmha,
                                           run_stream,
                                           false);

    params->kv_block_array            = ((TRTAttn*)trt_params.get())->kv_block_array;
    params->kv_cache_offset           = ((TRTAttn*)trt_params.get())->kv_cache_offset.clone();
    params->batch_size                = batch_size;
    params->max_seq_len               = attn_inputs.sequence_lengths.max().item<int32_t>();
    params->sequence_lengths          = attn_inputs.sequence_lengths;
    params->kv_block_array.cache_type = attn_configs_.kv_cache_dtype;

    return ParamsBasePtr(params);
}

torch::Tensor
XQAAttnOp::forward(const torch::Tensor& input, std::optional<torch_ext::KVCache> kv_cache, const XQAParamsPtr& params) {
    const int            batch_size        = input.size(0);
    const int            local_head_num    = attn_configs_.head_num;
    const int            local_head_num_kv = attn_configs_.kv_head_num;
    const int            size_per_head     = attn_configs_.size_per_head;
    torch::TensorOptions options           = torch::TensorOptions(input.dtype()).device(input.device());

    torch::Tensor output = torch::empty({batch_size, local_head_num * size_per_head}, options);

    KVBlockArray kv_block_array;
    if (kv_cache.has_value()) {
        kv_block_array                 = params->kv_block_array;
        kv_block_array.mPrimaryPoolPtr = kv_cache.value().k_cache_base.data_ptr();
        if (kv_cache.value().k_scale_base.defined() && kv_cache.value().k_scale_base.numel() > 0) {
            kv_block_array.scale = kv_cache.value().k_scale_base.data_ptr();
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
           attn_configs_.tokens_per_block,
           kv_cache.value().k_cache_base.data_ptr(),  // params->kv_block_array.mPrimaryPoolPtr,
           reinterpret_cast<int32_t*>((KVCacheIndex*)(params->kv_cache_offset.data_ptr())),
           kv_block_array.cache_type == KvCacheDataType::FP8,
           reinterpret_cast<uint32_t*>(params->sequence_lengths.data_ptr()));
    return output;
}

void registerXQAAttnOp(const py::module& m) {
    pybind11::class_<XQAParams, std::shared_ptr<XQAParams>, rtp_llm::ParamsBase>(m, "XQAParams")
        .def(pybind11::init<>());
    pybind11::class_<XQAAttnOp>(m, "XQAAttnOp")
        .def(pybind11::init<const AttentionConfigs&>(),
             py::arg("attn_configs"))
        .def("support", &XQAAttnOp::support, py::arg("attn_inputs").noconvert())
        .def("prepare", &XQAAttnOp::prepare, py::arg("attn_inputs"))
        .def("forward", &XQAAttnOp::forward, py::arg("input"), py::arg("kv_cache"), py::arg("params"));
}

}  // namespace rtp_llm
#endif
