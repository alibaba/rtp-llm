#include "rtp_llm/models_py/bindings/cuda/FusedRopeKVCacheOp.h"
#include "rtp_llm/models_py/bindings/common/kernels/unfused_attention_kernels.h"
#include "rtp_llm/cpp/core/Dispatch.h"
#include "rtp_llm/cpp/core/torch_utils/TypeConvert.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/cuda/ops/CudaFlashInfer.h"
#include "rtp_llm/cpp/cuda/cufmha/TrtV2FmhaRunner.h"
#include "rtp_llm/cpp/model_utils/RopeCache.h"
#include "rtp_llm/models_py/bindings/common/Torch_ext.h"

#include <iostream>

namespace rtp_llm {
namespace {

void invokeFusedQKVBiasTransposeHelper(const torch::Tensor&                   qkv,
                                       std::optional<torch_ext::LayerKVCache> kv_cache,
                                       const TRTAttnPtr&                      params,
                                       const AttentionConfigs&                attn_configs,
                                       size_t                                 max_seq_len,
                                       void*                                  q_no_transpose_output,
                                       void*                                  q_output,
                                       void*                                  qkv_fp8_output,
                                       int                                    token_num,
                                       int                                    batch_size,
                                       int                                    local_head_num,
                                       int                                    local_head_num_kv,
                                       int                                    size_per_head,
                                       bool                                   use_paged_attention,
                                       bool                                   store_qkv,
                                       bool                                   store_q_no_transpose,
                                       bool                                   store_q,
                                       bool                                   store_kv,
                                       bool                                   store_cache) {
    PrefixPromptBatchWeightsParam prefix_prompt_param;
    if (kv_cache.has_value()) {
        auto kv_block_array            = params->kv_block_array;
        kv_block_array.mPrimaryPoolPtr = kv_cache.value().kv_cache_base.data_ptr();
        if (kv_cache.value().kv_scale_base.defined() && kv_cache.value().kv_scale_base.numel()) {
            kv_block_array.scale = kv_cache.value().kv_scale_base.data_ptr();
        }
        prefix_prompt_param.kv_block_array = kv_block_array;
        if (params->max_prefix_length > 0) {
            prefix_prompt_param.d_prefix_prompt_lengths  = params->prefix_lengths.data_ptr<int>();
            prefix_prompt_param.max_prefix_prompt_length = params->max_prefix_length;
            prefix_prompt_param.count_length             = 1;
        }
    }

    int* padding_offset = nullptr;
    if (params->padding_offset.defined()) {
        padding_offset = params->padding_offset.data_ptr<int>();
    }
    auto       rope_cache = getRopeCacheOnce(attn_configs.rope_config, attn_configs.max_seq_len);
    StreamType stream     = GET_CURRENT_STREAM();

    int* position_ids = nullptr;
    if (params->cp_position_ids.defined()) {
        position_ids = params->cp_position_ids.data_ptr<int>();
        store_cache  = false;
    }

    DISPATCH_CUDA_FUNCTION_DATA_TYPE(
        torchDTypeToDataType(qkv.dtype()),
        invokeAddFusedQKVBiasTranspose,
        q_no_transpose_output,
        q_output,
        nullptr,  // k_output.data_ptr(),
        nullptr,  // v_output.data_ptr(),
        &prefix_prompt_param,
        qkv.data_ptr(),
        qkv_fp8_output,
        position_ids,  // position_ids
        nullptr,       // qkv_bias
        padding_offset,
        params->cu_seqlens.data_ptr<int>(),
        rope_cache.used,
        checkRopeCache(attn_configs.rope_config, rope_cache) ? rope_cache.data.data_ptr<float>() : nullptr,
        batch_size,
        params->max_seq_len,
        token_num,
        local_head_num,
        local_head_num_kv,
        size_per_head,
        attn_configs.rope_config,
        attn_configs.use_logn_attn,
        nullptr,  // scale_out_ptr,
        0,        // int8_mode,
        use_paged_attention,
        store_qkv,
        store_q_no_transpose,
        store_q,
        store_kv,
        store_cache,
        stream);
}
}  // namespace

FusedRopeKVCachePrefillOpBase::FusedRopeKVCachePrefillOpBase(const AttentionConfigs& attn_configs,
                                                             size_t                  max_seq_len,
                                                             bool                    use_fp8_fmha):
    attn_configs_(attn_configs), max_seq_len_(max_seq_len), use_fp8_fmha_(use_fp8_fmha) {}

TRTAttnPtr FusedRopeKVCachePrefillOpBase::prepare(torch_ext::PyAttentionInputs attn_inputs) {
    int           batch_size = attn_inputs.input_lengths.size(0);
    torch::Tensor kv_cache_kernel_block_id_device;
    if (attn_inputs.kv_cache_kernel_block_id_host.defined() && attn_inputs.kv_cache_kernel_block_id_host.numel() > 0) {
        kv_cache_kernel_block_id_device = attn_inputs.kv_cache_kernel_block_id_device;
    }

    TRTAttnPtr attn_params;
    auto       params = prepareTrtAttnParams(
        attn_configs_, kv_cache_kernel_block_id_device, batch_size, use_fp8_fmha_, GET_CURRENT_STREAM());
    if (params) {
        attn_params = TRTAttnPtr(params, (TRTAttn*)params.get());
    } else {
        attn_params = std::make_shared<TRTAttn>();
    }
    attn_params->attn_type                 = torchDTypeToDataType(attn_inputs.dtype);
    attn_params->cu_seqlens                = attn_inputs.cu_seqlens;
    attn_params->cu_kv_seqlens             = attn_inputs.cu_kv_seqlens;
    attn_params->max_seq_len               = attn_inputs.input_lengths.max().item<int32_t>();
    attn_params->max_prefix_length         = attn_inputs.prefix_lengths.max().item<int32_t>();
    attn_params->prefix_lengths            = attn_inputs.prefix_lengths;
    attn_params->kv_block_array.cache_type = attn_configs_.kv_cache_dtype;
    attn_params->padding_offset            = attn_inputs.padding_offset;

    if (attn_inputs.context_parallel_info.has_value()
        && attn_inputs.context_parallel_info->prefill_shuffle_indices.defined()) {
        attn_params->cp_position_ids = attn_inputs.context_parallel_info->prefill_shuffle_indices;
    }
    return attn_params;
}

FusedRopeKVCachePrefillOpQOut::FusedRopeKVCachePrefillOpQOut(const AttentionConfigs& attn_configs,
                                                             size_t                  max_seq_len,
                                                             bool                    use_fp8_fmha):
    FusedRopeKVCachePrefillOpBase(attn_configs, max_seq_len, use_fp8_fmha) {}

torch::Tensor FusedRopeKVCachePrefillOpQOut::forward(const torch::Tensor&                   qkv,
                                                     std::optional<torch_ext::LayerKVCache> kv_cache,
                                                     const TRTAttnPtr&                      params) {
    const int     local_head_num    = attn_configs_.head_num;
    const int     local_head_num_kv = attn_configs_.kv_head_num;
    const int     size_per_head     = attn_configs_.size_per_head;
    const int     token_num         = qkv.size(0);
    const int     batch_size        = params->cu_seqlens.size(0) - 1;
    torch::Tensor q_output          = torch::empty({token_num, local_head_num, size_per_head},
                                          torch::TensorOptions(qkv.dtype()).device(qkv.device()));

    bool use_paged_attention = kv_cache.has_value() && params->max_prefix_length > 0;

    invokeFusedQKVBiasTransposeHelper(qkv,
                                      kv_cache,
                                      params,
                                      attn_configs_,
                                      max_seq_len_,
                                      q_output.data_ptr(),  // q_no_transpose_output
                                      nullptr,              // q_output
                                      nullptr,              // qkv_fp8_output
                                      token_num,
                                      batch_size,
                                      local_head_num,
                                      local_head_num_kv,
                                      size_per_head,
                                      use_paged_attention,
                                      false,                  // store_qkv
                                      true,                   // store_q_no_transpose
                                      false,                  // store_q
                                      false,                  // store_kv
                                      kv_cache.has_value());  // store_cache

    return q_output;
}

FusedRopeKVCachePrefillOpQKVOut::FusedRopeKVCachePrefillOpQKVOut(const AttentionConfigs& attn_configs,
                                                                 size_t                  max_seq_len,
                                                                 bool                    use_fp8_fmha):
    FusedRopeKVCachePrefillOpBase(attn_configs, max_seq_len, use_fp8_fmha) {}

torch::Tensor FusedRopeKVCachePrefillOpQKVOut::forward(const torch::Tensor&                   qkv,
                                                       std::optional<torch_ext::LayerKVCache> kv_cache,
                                                       const TRTAttnPtr&                      params) {
    const int local_head_num    = attn_configs_.head_num;
    const int local_head_num_kv = attn_configs_.kv_head_num;
    const int size_per_head     = attn_configs_.size_per_head;
    const int token_num         = qkv.size(0);
    const int batch_size        = params->cu_seqlens.size(0) - 1;

    invokeFusedQKVBiasTransposeHelper(qkv,
                                      kv_cache,
                                      params,
                                      attn_configs_,
                                      max_seq_len_,
                                      nullptr,  // q_no_transpose_output
                                      nullptr,  // q_output
                                      nullptr,  // qkv_fp8_output
                                      token_num,
                                      batch_size,
                                      local_head_num,
                                      local_head_num_kv,
                                      size_per_head,
                                      false,                  // use_paged_attention
                                      true,                   // store_qkv,
                                      false,                  // store_q_no_transpose
                                      false,                  // store_q
                                      false,                  // store_kv,
                                      kv_cache.has_value());  // store_cache
    return qkv;
}

FusedRopeKVCacheDecodeOp::FusedRopeKVCacheDecodeOp(const AttentionConfigs& attn_configs,
                                                   size_t                  max_seq_len,
                                                   bool                    use_fp8_fmha):
    attn_configs_(attn_configs), max_seq_len_(max_seq_len), use_fp8_fmha_(use_fp8_fmha) {}

TRTAttnPtr FusedRopeKVCacheDecodeOp::prepare(torch_ext::PyAttentionInputs attn_inputs) {
    int           batch_size = attn_inputs.sequence_lengths.size(0);
    torch::Tensor kv_cache_kernel_block_id_device;
    if (attn_inputs.kv_cache_kernel_block_id_host.defined() && attn_inputs.kv_cache_kernel_block_id_host.numel() > 0) {
        kv_cache_kernel_block_id_device = attn_inputs.kv_cache_kernel_block_id_device;
    }

    TRTAttnPtr attn_params;
    auto       params = prepareTrtAttnParams(
        attn_configs_, kv_cache_kernel_block_id_device, batch_size, use_fp8_fmha_, GET_CURRENT_STREAM());
    RTP_LLM_CHECK_WITH_INFO(params != nullptr, "TRTAttnPtr Build Failed");
    attn_params                            = TRTAttnPtr(params, (TRTAttn*)params.get());
    attn_params->decode_plan               = true;
    attn_params->attn_type                 = torchDTypeToDataType(attn_inputs.dtype);
    attn_params->cu_seqlens                = attn_inputs.cu_seqlens;
    attn_params->cu_kv_seqlens             = attn_inputs.cu_kv_seqlens;
    attn_params->sequence_lengths          = attn_inputs.sequence_lengths;
    attn_params->cp_position_ids           = attn_inputs.position_ids;
    attn_params->kv_block_array.cache_type = attn_configs_.kv_cache_dtype;
    return attn_params;
}

torch::Tensor FusedRopeKVCacheDecodeOp::forward(const torch::Tensor&                   qkv,
                                                std::optional<torch_ext::LayerKVCache> kv_cache,
                                                const TRTAttnPtr&                      params) {
    RTP_LLM_CHECK_WITH_INFO(kv_cache.has_value(), "decode should have kv cache.");
    auto kv_block_array            = params->kv_block_array;
    kv_block_array.mPrimaryPoolPtr = kv_cache.value().kv_cache_base.data_ptr();
    if (kv_cache.value().kv_scale_base.defined() && kv_cache.value().kv_scale_base.numel()) {
        kv_block_array.scale = kv_cache.value().kv_scale_base.data_ptr();
    }

    const int     local_head_num    = attn_configs_.head_num;
    const int     local_head_num_kv = attn_configs_.kv_head_num;
    const int     size_per_head     = attn_configs_.size_per_head;
    const int     token_num         = qkv.size(0);
    const int     batch_size        = params->sequence_lengths.size(0);
    torch::Tensor q_output          = torch::empty({token_num, local_head_num, size_per_head},
                                          torch::TensorOptions(qkv.dtype()).device(qkv.device()));

    auto rope_cache = getRopeCacheOnce(attn_configs_.rope_config, attn_configs_.max_seq_len);

    RTP_LLM_CHECK_WITH_INFO(params->sequence_lengths.is_pinned(), "sequence_lengths is not pinned memory");
    StreamType stream = GET_CURRENT_STREAM();
    DISPATCH_CUDA_FUNCTION_DATA_TYPE(
        torchDTypeToDataType(qkv.dtype()),
        invokeDecodeAddFusedQKVBiasTranspose,
        q_output.data_ptr(),
        nullptr,  // k_buf
        nullptr,  // v_buf
        kv_block_array,
        qkv.data_ptr(),
        params->sequence_lengths.data_ptr<int>(),
        nullptr,  // qkv_bias
        rope_cache.used,
        checkRopeCache(attn_configs_.rope_config, rope_cache) ? rope_cache.data.data_ptr<float>() : nullptr,
        batch_size,
        local_head_num,
        local_head_num_kv,
        size_per_head,
        attn_configs_.rope_config,
        attn_configs_.use_logn_attn,
        true,   // store_q,
        false,  // store_kv,
        true,   // store_cache,
        stream);
    return q_output;
}

void registerFusedRopeKVCacheOp(const py::module& m) {
    pybind11::class_<KVBlockArray>(m, "KVBlockArray")
        .def(pybind11::init<>())
        .def(
            "__cpp_ptr__",
            [](KVBlockArray& self) { return reinterpret_cast<uintptr_t>(&self); },
            "Get C++ object pointer address");
    pybind11::class_<FusedRopeKVCachePrefillOpQKVOut>(m, "FusedRopeKVCachePrefillOpQKVOut")
        .def(pybind11::init<const AttentionConfigs&, size_t, bool>(),
             py::arg("attn_configs"),
             py::arg("max_seq_len")  = 0,
             py::arg("use_fp8_fmha") = false)
        .def("prepare", &FusedRopeKVCachePrefillOpQKVOut::prepare, py::arg("attn_inputs"))
        .def("forward",
             &FusedRopeKVCachePrefillOpQKVOut::forward,
             py::arg("qkv"),
             py::arg("kv_cache"),
             py::arg("params"));

    pybind11::class_<FusedRopeKVCachePrefillOpQOut>(m, "FusedRopeKVCachePrefillOpQOut")
        .def(pybind11::init<const AttentionConfigs&, size_t, bool>(),
             py::arg("attn_configs"),
             py::arg("max_seq_len")  = 0,
             py::arg("use_fp8_fmha") = false)
        .def("prepare", &FusedRopeKVCachePrefillOpQOut::prepare, py::arg("attn_inputs"))
        .def(
            "forward", &FusedRopeKVCachePrefillOpQOut::forward, py::arg("qkv"), py::arg("kv_cache"), py::arg("params"));

    pybind11::class_<FusedRopeKVCacheDecodeOp>(m, "FusedRopeKVCacheDecodeOp")
        .def(pybind11::init<const AttentionConfigs&, size_t, bool>(),
             py::arg("attn_configs"),
             py::arg("max_seq_len")  = 0,
             py::arg("use_fp8_fmha") = false)
        .def("prepare", &FusedRopeKVCacheDecodeOp::prepare, py::arg("attn_inputs"))
        .def("forward", &FusedRopeKVCacheDecodeOp::forward, py::arg("qkv"), py::arg("kv_cache"), py::arg("params"));
}

}  // namespace rtp_llm
