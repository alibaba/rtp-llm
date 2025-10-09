#include "rtp_llm/models_py/bindings/cuda/FusedRopeKVCacheOp.h"
#include "rtp_llm/cpp/kernels/unfused_attention_kernels.h"
#include "rtp_llm/cpp/core/Dispatch.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/devices/cuda_impl/CudaFlashInfer.h"
#include "rtp_llm/cpp/core/BufferHelper.h"
namespace rtp_llm {

FusedRopeKVCachePrefillOp::FusedRopeKVCachePrefillOp(const GptInitParameter& gpt_init_parameter):
    FMHACudaBase(gpt_init_parameter) {}

TRTAttnPtr FusedRopeKVCachePrefillOp::prepare(torch_ext::PyAttentionInputs attn_inputs) {
    int       batch_size = attn_inputs.input_lengths.size(0);
    BufferPtr kv_cache_block_id_host, kv_cache_block_id_device;
    if (attn_inputs.kv_cache_block_id_host.defined() && attn_inputs.kv_cache_block_id_host.numel() > 0) {
        kv_cache_block_id_host   = torchTensor2Buffer(attn_inputs.kv_cache_block_id_host);
        kv_cache_block_id_device = torchTensor2Buffer(attn_inputs.kv_cache_block_id_device);
    }
    // not support has_alibi_slopes
    auto          cu_seqlens    = attn_inputs.cu_seqlens;
    torch::Tensor cu_kv_seqlens = cu_seqlens;
    TRTAttnPtr    attn_params;
    auto          params =
        device_->prepareTrtAttn(attn_configs_, attn_inputs.kv_block_offset, kv_cache_block_id_device, batch_size);
    if (params) {
        attn_params = TRTAttnPtr(params, (TRTAttn*)params.get());
    } else {
        attn_params = std::make_shared<TRTAttn>();
    }
    attn_params->attn_type                 = torchDTypeToDataType(attn_inputs.dtype);
    attn_params->cu_seqlens                = cu_seqlens;
    attn_params->cu_kv_seqlens             = cu_kv_seqlens;
    attn_params->max_seq_len               = attn_inputs.input_lengths.max().item<int32_t>();
    attn_params->kv_block_array.cache_type = attn_configs_.kv_cache_dtype;
    attn_params->padding_offset            = attn_inputs.padding_offset;
    return attn_params;
}

torch::Tensor FusedRopeKVCachePrefillOp::forward(const torch::Tensor&              qkv,
                                                 FMHAType                          fmha_type,
                                                 std::optional<torch_ext::KVCache> kv_cache,
                                                 const TRTAttnPtr&                 params) {
    // bool store_cache = params.common.kv_cache.has_value();
    const int     local_head_num        = attn_configs_.head_num;
    const int     local_head_num_kv     = attn_configs_.kv_head_num;
    const int     size_per_head         = attn_configs_.size_per_head;
    const int     token_num             = qkv.size(0);
    const int     batch_size            = params->cu_seqlens.size(0) - 1;
    torch::Tensor q_no_transpose_output = torch::empty({token_num, local_head_num, size_per_head},
                                                       torch::TensorOptions(qkv.dtype()).device(qkv.device()));
    torch::Tensor q_output              = torch::empty({token_num, local_head_num, size_per_head},
                                          torch::TensorOptions(qkv.dtype()).device(qkv.device()));
    torch::Tensor k_output              = torch::empty({token_num, local_head_num_kv, size_per_head},
                                          torch::TensorOptions(qkv.dtype()).device(qkv.device()));
    torch::Tensor v_output              = torch::empty({token_num, local_head_num_kv, size_per_head},
                                          torch::TensorOptions(qkv.dtype()).device(qkv.device()));

    torch::Tensor qkv_fp8 = torch::empty({token_num, (local_head_num + 2 * local_head_num_kv), size_per_head},
                                         torch::TensorOptions(torch::kFloat8_e4m3fn).device(qkv.device()));

    PrefixPromptBatchWeightsParam prefix_prompt_param;
    if (kv_cache.has_value()) {
        auto kv_block_array            = params->kv_block_array;
        kv_block_array.mPrimaryPoolPtr = kv_cache.value().k_cache_base.data_ptr();
        if (kv_cache.value().k_scale_base.defined() && kv_cache.value().k_scale_base.numel()) {
            kv_block_array.scale = kv_cache.value().k_scale_base.data_ptr();
        }
        prefix_prompt_param.kv_block_array = kv_block_array;
        // if (attn_inputs.prefix_lengths.size(0)) {
        //     prefix_prompt_param.d_prefix_prompt_lengths  = attn_inputs.prefix_lengths.data_ptr<int>();
        //     prefix_prompt_param.max_prefix_prompt_length = attn_inputs.prefix_lengths.max().item<int>();
        //     prefix_prompt_param.count_length             = 1;
        // }
    }
    // not support fp8 now

    bool store_qkv =
        fmha_type != FMHAType::PAGED_TRT_V2 && fmha_type != FMHAType::NONE && fmha_type != FMHAType::FLASH_INFER;
    bool store_q_no_transpose = fmha_type == FMHAType::FLASH_INFER;
    bool store_q              = fmha_type == FMHAType::PAGED_TRT_V2 || fmha_type == FMHAType::NONE;
    bool store_kv             = fmha_type == FMHAType::NONE;
    bool store_cache          = kv_cache.has_value();
    // bool use_qkv_fp8 =
    //     fmha_type == FMHAType::TRT_V2 && prefix_prompt_param.kv_block_array.cache_type == KvCacheDataType::FP8;

    // tmp not use qkv fp8 buffer
    bool use_qkv_fp8 = false;
    // embedding cuda graph, we use padded mode, so the padding offset is always zero and in this case the
    // `padding_offset` should be nullptr.
    int* padding_offset = nullptr;
    if (params->padding_offset.defined()) {
        padding_offset = params->padding_offset.data_ptr<int>();
    }
    DISPATCH_CUDA_FUNCTION_DATA_TYPE(
        torchDTypeToDataType(qkv.dtype()),
        invokeAddFusedQKVBiasTranspose,
        q_no_transpose_output.data_ptr(),
        q_output.data_ptr(),
        k_output.data_ptr(),
        v_output.data_ptr(),
        &prefix_prompt_param,
        qkv.data_ptr(),
        use_qkv_fp8 ? qkv_fp8.data_ptr() : nullptr,
        nullptr,  // params.common.position_ids ? params.common.position_ids->dataWithOffset<int>(decoder_batch_size *
                  // params.configs.rope_config.index_factor): nullptr,
        nullptr,  // params.configs.fuse_qkv_add_bias && params.weights.qkv_weight->bias ?
                  // params.weights.qkv_weight->bias->data() : nullptr,
        padding_offset,
        params->cu_seqlens.data_ptr<int>(),
        batch_size,
        params->max_seq_len,  // seq_len
        token_num,
        local_head_num,
        local_head_num_kv,
        size_per_head,
        attn_configs_.rope_config,
        attn_configs_.use_logn_attn,
        nullptr,  // scale_out_ptr,
        0,        // int8_mode,
        fmha_type == FMHAType::PAGED_TRT_V2,
        store_qkv,
        store_q_no_transpose,
        store_q,
        store_kv,
        store_cache,
        device_->getStream());

    if (use_qkv_fp8) {
        return qkv_fp8;
    } else if (fmha_type == FMHAType::PAGED_TRT_V2) {
        return q_output;
    } else if (fmha_type == FMHAType::FLASH_INFER) {
        return q_no_transpose_output;
    } else {
        return qkv;
    }
}

FusedRopeKVCacheDecodeOp::FusedRopeKVCacheDecodeOp(const GptInitParameter& gpt_init_parameter):
    FMHACudaBase(gpt_init_parameter) {}

TRTAttnPtr FusedRopeKVCacheDecodeOp::prepare(torch_ext::PyAttentionInputs attn_inputs) {
    int       batch_size = attn_inputs.sequence_lengths.size(0);
    BufferPtr kv_cache_block_id_host, kv_cache_block_id_device;
    if (attn_inputs.kv_cache_block_id_host.defined() && attn_inputs.kv_cache_block_id_host.numel() > 0) {
        kv_cache_block_id_host   = torchTensor2Buffer(attn_inputs.kv_cache_block_id_host);
        kv_cache_block_id_device = torchTensor2Buffer(attn_inputs.kv_cache_block_id_device);
    }
    // not support has_alibi_slopes
    attn_inputs.cu_seqlens.slice(0, 1, batch_size + 1) = attn_inputs.input_lengths.cumsum(0);
    auto          cu_seqlens                           = attn_inputs.cu_seqlens;
    torch::Tensor cu_kv_seqlens                        = cu_seqlens;
    TRTAttnPtr    attn_params;
    auto          params = device_->prepareTrtAttn(
        attn_configs_, attn_inputs.kv_block_offset, kv_cache_block_id_device, attn_inputs.sequence_lengths.size(0));
    RTP_LLM_CHECK_WITH_INFO(params != nullptr, "TRTAttnPtr Build Failed");
    attn_params                            = TRTAttnPtr(params, (TRTAttn*)params.get());
    attn_params->decode_plan               = true;
    attn_params->attn_type                 = torchDTypeToDataType(attn_inputs.dtype);
    attn_params->cu_seqlens                = cu_seqlens;
    attn_params->cu_kv_seqlens             = cu_kv_seqlens;
    attn_params->sequence_lengths          = attn_inputs.sequence_lengths;
    attn_params->kv_block_array.cache_type = attn_configs_.kv_cache_dtype;
    return attn_params;
}

torch::Tensor FusedRopeKVCacheDecodeOp::forward(const torch::Tensor&              qkv,
                                                FMHAType                          fmha_type,
                                                std::optional<torch_ext::KVCache> kv_cache,
                                                const TRTAttnPtr&                 params) {
    RTP_LLM_CHECK_WITH_INFO(kv_cache.has_value(), "decode should have kv cache.");
    auto kv_block_array            = params->kv_block_array;
    kv_block_array.mPrimaryPoolPtr = kv_cache.value().k_cache_base.data_ptr();
    if (kv_cache.value().k_scale_base.defined() && kv_cache.value().k_scale_base.numel()) {
        kv_block_array.scale = kv_cache.value().k_scale_base.data_ptr();
    }

    const int     local_head_num    = attn_configs_.head_num;
    const int     local_head_num_kv = attn_configs_.kv_head_num;
    const int     size_per_head     = attn_configs_.size_per_head;
    const int     token_num         = qkv.size(0);
    const int     batch_size        = params->sequence_lengths.size(0);
    torch::Tensor q_output          = torch::empty({token_num, local_head_num, size_per_head},
                                          torch::TensorOptions(qkv.dtype()).device(qkv.device()));

    static torch::Tensor cos_sin_cache = getRopeCosSin(attn_configs_.rope_config.style,
                                                       attn_configs_.rope_config.dim,
                                                       attn_configs_.rope_config.base,
                                                       attn_configs_.rope_config.scale,
                                                       device_->initParams().max_seq_len);

    RTP_LLM_CHECK_WITH_INFO(params->sequence_lengths.is_pinned(), "sequence_lengths is not pinned memory");
    DISPATCH_CUDA_FUNCTION_DATA_TYPE(torchDTypeToDataType(qkv.dtype()),
                                     invokeDecodeAddFusedQKVBiasTranspose,
                                     q_output.data_ptr(),
                                     nullptr,  // k_buf
                                     nullptr,  // v_buf
                                     kv_block_array,
                                     qkv.data_ptr(),
                                     params->sequence_lengths.data_ptr<int>(),
                                     nullptr,  // params.configs.fuse_qkv_add_bias && params.weights.qkv_weight->bias ?
                                               // params.weights.qkv_weight->bias->data() : nullptr,
                                     cos_sin_cache.defined() ? cos_sin_cache.data_ptr<float>() : nullptr,
                                     batch_size,
                                     local_head_num,
                                     local_head_num_kv,
                                     size_per_head,
                                     attn_configs_.rope_config,
                                     attn_configs_.use_logn_attn,
                                     true,   // store_q,
                                     false,  // store_kv,
                                     true,   // store_cache,
                                     device_->getStream());
    return q_output;
}

void registerFusedRopeKVCacheOp(const py::module& m) {
    pybind11::class_<KVBlockArray>(m, "KVBlockArray").def(pybind11::init<>());
    pybind11::class_<TRTAttn, std::shared_ptr<TRTAttn>, rtp_llm::ParamsBase>(m, "TRTAttn").def(pybind11::init<>());
    pybind11::class_<FusedRopeKVCachePrefillOp>(m, "FusedRopeKVCachePrefillOp")
        .def(pybind11::init<GptInitParameter>(), py::arg("gpt_init_parameter"))
        .def("prepare", &FusedRopeKVCachePrefillOp::prepare, py::arg("attn_inputs"))
        .def("forward",
             &FusedRopeKVCachePrefillOp::forward,
             py::arg("qkv"),
             py::arg("fmha_type"),
             py::arg("kv_cache"),
             py::arg("params"));

    pybind11::class_<FusedRopeKVCacheDecodeOp>(m, "FusedRopeKVCacheDecodeOp")
        .def(pybind11::init<GptInitParameter>(), py::arg("gpt_init_parameter"))
        .def("prepare", &FusedRopeKVCacheDecodeOp::prepare, py::arg("attn_inputs"))
        .def("forward",
             &FusedRopeKVCacheDecodeOp::forward,
             py::arg("qkv"),
             py::arg("fmha_type"),
             py::arg("kv_cache"),
             py::arg("params"));
}

}  // namespace rtp_llm
