
#ifdef USING_CUDA12
#include "rtp_llm/models_py/bindings/cuda/XQAAttnOp.h"
#include "rtp_llm/models_py/bindings/common/kernels/kv_cache_kernels.h"
#include "rtp_llm/models_py/bindings/cuda/cuda_host_utils.h"
#include <ATen/cuda/CUDAContext.h>

namespace rtp_llm {

namespace {

size_t maxSeqLenWithoutDeviceSync(const AttentionConfigs& attn_configs, const torch::Tensor& sequence_lengths) {
    if (!sequence_lengths.defined() || sequence_lengths.numel() == 0) {
        return 0;
    }
    if (sequence_lengths.is_cuda()) {
        return static_cast<size_t>(attn_configs.max_seq_len);
    }
    return static_cast<size_t>(sequence_lengths.max().item<int32_t>());
}

void updateKvCacheOffsetTensor(const torch::Tensor& kv_cache_offset, const torch::Tensor& block_ids) {
    RTP_LLM_CHECK_WITH_INFO(block_ids.defined() && block_ids.is_cuda(),
                            "XQAAttnOp expects CUDA kv_cache_kernel_block_id_device");
    RTP_LLM_CHECK_WITH_INFO(block_ids.scalar_type() == torch::kInt32,
                            "XQAAttnOp expects int32 kv_cache_kernel_block_id_device");
    RTP_LLM_CHECK_WITH_INFO(block_ids.dim() == 2, "XQAAttnOp expects 2-D kv_cache block table");
    RTP_LLM_CHECK_WITH_INFO(kv_cache_offset.defined() && kv_cache_offset.is_cuda(),
                            "XQAAttnOp expects existing CUDA kv_cache_offset");
    RTP_LLM_CHECK_WITH_INFO(kv_cache_offset.scalar_type() == torch::kInt32, "XQAAttnOp expects int32 kv_cache_offset");
    RTP_LLM_CHECK_WITH_INFO(kv_cache_offset.dim() == 4 && kv_cache_offset.size(1) == 1 && kv_cache_offset.size(2) == 2,
                            "XQAAttnOp expects kv_cache_offset shape [batch, 1, 2, blocks]");

    const int batch_size           = static_cast<int>(block_ids.size(0));
    const int max_blocks_per_batch = static_cast<int>(block_ids.size(1));
    RTP_LLM_CHECK_WITH_INFO(kv_cache_offset.size(0) == batch_size && kv_cache_offset.size(3) == max_blocks_per_batch,
                            "XQAAttnOp shape mismatch: offset [%ld, %ld] vs block table [%d, %d]",
                            kv_cache_offset.size(0),
                            kv_cache_offset.size(3),
                            batch_size,
                            max_blocks_per_batch);

    auto run_stream = at::cuda::getCurrentCUDAStream(at::cuda::current_device()).stream();
    invokeConvertOffsetToBlockArrayData(
        kv_cache_offset.data_ptr<int32_t>(), block_ids.data_ptr<int>(), batch_size, max_blocks_per_batch, run_stream);
}

}  // namespace

XQAAttnOp::XQAAttnOp(const AttentionConfigs& attn_configs): attn_configs_(attn_configs) {}

bool XQAAttnOp::support(torch_ext::PyAttentionInputs attn_inputs) {
    return get_sm() >= 90
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
    RTP_LLM_CHECK_WITH_INFO(attn_inputs.kv_cache_kernel_block_id_device.defined(),
                            "decode should have kv cache block id.");

    const auto& block_ids = attn_inputs.kv_cache_kernel_block_id_device;
    RTP_LLM_CHECK_WITH_INFO(block_ids.size(0) == batch_size,
                            "XQA kv blocks batch size expected [%d] but got [%d]",
                            batch_size,
                            static_cast<int>(block_ids.size(0)));
    const size_t max_blocks_per_seq = block_ids.size(1);
    const bool   use_fp8_cache      = attn_configs_.kv_cache_dtype == KvCacheDataType::FP8;
    const int    element_size       = use_fp8_cache ? 1 : 2;

    params->kv_cache_offset           = torch::empty({batch_size, 1, 2, static_cast<int64_t>(max_blocks_per_seq)},
                                           torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    params->kv_block_array            = KVBlockArray(batch_size,
                                          max_blocks_per_seq,
                                          attn_configs_.kernel_tokens_per_block,
                                          attn_configs_.kv_head_num * attn_configs_.size_per_head * element_size,
                                          0,
                                          0,
                                          nullptr,
                                          nullptr,
                                          reinterpret_cast<KVCacheIndex*>(params->kv_cache_offset.data_ptr<int>()));
    params->kv_block_array.cache_type = use_fp8_cache ? KvCacheDataType::FP8 : KvCacheDataType::BASE;
    params->kv_block_array.mScaleBytesPerBlock =
        attn_configs_.kernel_tokens_per_block * attn_configs_.kv_head_num * sizeof(float);

    auto run_stream = at::cuda::getCurrentCUDAStream(at::cuda::current_device()).stream();
    invokeConvertOffsetToBlockArrayData(
        params->kv_cache_offset.data_ptr<int>(), block_ids.data_ptr<int>(), batch_size, max_blocks_per_seq, run_stream);
    check_cuda_error();

    params->batch_size       = batch_size;
    params->max_seq_len      = maxSeqLenWithoutDeviceSync(attn_configs_, attn_inputs.sequence_lengths);
    params->sequence_lengths = attn_inputs.sequence_lengths;

    return ParamsBasePtr(params);
}

void XQAAttnOp::update(const XQAParamsPtr& params, torch_ext::PyAttentionInputs attn_inputs) {
    RTP_LLM_CHECK_WITH_INFO(params != nullptr, "XQAAttnOp::update received null params");

    const auto& block_ids = attn_inputs.kv_cache_kernel_block_id_device;
    updateKvCacheOffsetTensor(params->kv_cache_offset, block_ids);

    params->batch_size       = static_cast<size_t>(block_ids.size(0));
    params->sequence_lengths = attn_inputs.sequence_lengths;
}

void XQAAttnOp::updateKvCacheOffset(const torch::Tensor& kv_cache_offset,
                                    const torch::Tensor& kv_cache_block_id_device) {
    updateKvCacheOffsetTensor(kv_cache_offset, kv_cache_block_id_device);
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
        .def("update", &XQAAttnOp::update, py::arg("params"), py::arg("attn_inputs"))
        .def("update_kv_cache_offset",
             &XQAAttnOp::updateKvCacheOffset,
             py::arg("kv_cache_offset"),
             py::arg("kv_cache_block_id_device"))
        .def("forward", &XQAAttnOp::forward, py::arg("input"), py::arg("kv_cache"), py::arg("params"));
}

}  // namespace rtp_llm
#endif
