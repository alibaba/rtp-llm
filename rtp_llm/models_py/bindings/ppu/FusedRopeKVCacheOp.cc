#include "rtp_llm/models_py/bindings/ppu/FusedRopeKVCacheOp.h"
#include "rtp_llm/models_py/bindings/ppu/kernels/fused_rope_kvcache_kernel.h"
#include "rtp_llm/models_py/bindings/core/Dispatch.h"
#include "rtp_llm/models_py/bindings/core/torch_utils/TypeConvert.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/models_py/bindings/common/kernels/kv_cache_kernels.h"
#include "rtp_llm/cpp/model_utils/RopeCache.h"
#include "rtp_llm/models_py/bindings/common/Torch_ext.h"

#include <iostream>

namespace rtp_llm {
namespace {

PpuFusedRopeParamsPtr preparePpuFusedRopeParams(const AttentionConfigs& configs,
                                                const torch::Tensor&    kv_cache_block_id,
                                                int                     batch_size,
                                                bool                    use_fp8_fmha) {
#ifndef ENABLE_FP8
    (void)use_fp8_fmha;
#endif
    auto params = std::make_shared<PpuFusedRopeParams>();
    if (!kv_cache_block_id.defined() || batch_size == 0) {
        return params;
    }

    int             element_size = 2;
    KvCacheDataType cache_type   = KvCacheDataType::BASE;
#ifdef ENABLE_FP8
    if (use_fp8_fmha) {
        cache_type   = KvCacheDataType::FP8;
        element_size = 1;
    } else
#endif
        if (configs.kv_cache_dtype == KvCacheDataType::FP8) {
        cache_type   = KvCacheDataType::FP8;
        element_size = 1;
    }

    RTP_LLM_CHECK_WITH_INFO(kv_cache_block_id.size(0) == batch_size,
                            "attention kv blocks batch size expected [%d] but got [%d]",
                            batch_size,
                            static_cast<int>(kv_cache_block_id.size(0)));

    const auto max_blocks_per_batch = kv_cache_block_id.size(1);
    params->kv_cache_offset =
        torch::empty({batch_size, 1, 2, max_blocks_per_batch}, kv_cache_block_id.options().dtype(torch::kInt32));
    params->kv_block_array                     = KVBlockArray(batch_size,
                                          max_blocks_per_batch,
                                          configs.kernel_tokens_per_block,
                                          configs.kv_head_num * configs.size_per_head * element_size,
                                          0,
                                          0,
                                          nullptr,
                                          nullptr,
                                          reinterpret_cast<KVCacheIndex*>(params->kv_cache_offset.data_ptr<int>()));
    params->kv_block_array.cache_type          = cache_type;
    params->kv_block_array.mScaleBytesPerBlock = configs.kernel_tokens_per_block * configs.kv_head_num * sizeof(float);

    invokeConvertOffsetToBlockArrayData(params->kv_cache_offset.data_ptr<int>(),
                                        kv_cache_block_id.data_ptr<int>(),
                                        batch_size,
                                        max_blocks_per_batch,
                                        GET_CURRENT_STREAM());
    return params;
}

void invokeFusedQKVBiasTransposeHelper(const torch::Tensor&                   qkv,
                                       std::optional<torch_ext::LayerKVCache> kv_cache,
                                       const PpuFusedRopeParamsPtr&           params,
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
    if (params->position_ids.defined()) {
        position_ids = params->position_ids.data_ptr<int>();
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

PpuFusedRopeParamsPtr FusedRopeKVCachePrefillOpBase::prepare(torch_ext::PyAttentionInputs attn_inputs) {
    int           batch_size = attn_inputs.input_lengths.size(0);
    torch::Tensor kv_cache_kernel_block_id_device;
    if (attn_inputs.kv_cache_kernel_block_id.defined() && attn_inputs.kv_cache_kernel_block_id.numel() > 0) {
        kv_cache_kernel_block_id_device = attn_inputs.kv_cache_kernel_block_id_device;
    }

    auto attn_params =
        preparePpuFusedRopeParams(attn_configs_, kv_cache_kernel_block_id_device, batch_size, use_fp8_fmha_);
    attn_params->cu_seqlens                = attn_inputs.cu_seqlens_device;
    attn_params->cu_kv_seqlens             = attn_inputs.cu_kv_seqlens_device;
    attn_params->max_seq_len               = attn_inputs.input_lengths.max().item<int32_t>();
    attn_params->max_prefix_length         = attn_inputs.prefix_lengths.max().item<int32_t>();
    attn_params->prefix_lengths            = attn_inputs.prefix_lengths;
    attn_params->kv_block_array.cache_type = attn_configs_.kv_cache_dtype;
    attn_params->padding_offset            = attn_inputs.padding_offset;

    if (attn_inputs.context_parallel_info.has_value()
        && attn_inputs.context_parallel_info->prefill_shuffle_indices.defined()) {
        auto cp_pos = attn_inputs.context_parallel_info->prefill_shuffle_indices;
        if (attn_params->max_prefix_length > 0) {
            auto device = cp_pos.device();
            auto per_token_prefix =
                at::repeat_interleave(attn_inputs.prefix_lengths.to(device), attn_inputs.input_lengths.to(device));
            cp_pos = cp_pos + per_token_prefix;
        }
        attn_params->position_ids = cp_pos;
    } else if (attn_inputs.combo_position_ids.defined()) {
        attn_params->position_ids = attn_inputs.combo_position_ids;
    }

    return attn_params;
}

FusedRopeKVCachePrefillOpQOut::FusedRopeKVCachePrefillOpQOut(const AttentionConfigs& attn_configs,
                                                             size_t                  max_seq_len,
                                                             bool                    use_fp8_fmha):
    FusedRopeKVCachePrefillOpBase(attn_configs, max_seq_len, use_fp8_fmha) {}

torch::Tensor FusedRopeKVCachePrefillOpQOut::forward(const torch::Tensor&                   qkv,
                                                     std::optional<torch_ext::LayerKVCache> kv_cache,
                                                     const PpuFusedRopeParamsPtr&           params) {
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
                                                       const PpuFusedRopeParamsPtr&           params) {
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

PpuFusedRopeParamsPtr FusedRopeKVCacheDecodeOp::prepare(torch_ext::PyAttentionInputs attn_inputs) {
    int           batch_size = attn_inputs.sequence_lengths.size(0);
    torch::Tensor kv_cache_kernel_block_id_device;
    if (attn_inputs.kv_cache_kernel_block_id.defined() && attn_inputs.kv_cache_kernel_block_id.numel() > 0) {
        kv_cache_kernel_block_id_device = attn_inputs.kv_cache_kernel_block_id_device;
    }

    RTP_LLM_CHECK_WITH_INFO(kv_cache_kernel_block_id_device.defined(), "decode requires kv cache block ids");
    auto attn_params =
        preparePpuFusedRopeParams(attn_configs_, kv_cache_kernel_block_id_device, batch_size, use_fp8_fmha_);
    attn_params->cu_seqlens                = attn_inputs.cu_seqlens_device;
    attn_params->cu_kv_seqlens             = attn_inputs.cu_kv_seqlens_device;
    attn_params->sequence_lengths          = attn_inputs.sequence_lengths;
    attn_params->position_ids              = attn_inputs.combo_position_ids;
    attn_params->kv_block_array.cache_type = attn_configs_.kv_cache_dtype;
    return attn_params;
}

torch::Tensor FusedRopeKVCacheDecodeOp::forward(const torch::Tensor&                   qkv,
                                                std::optional<torch_ext::LayerKVCache> kv_cache,
                                                const PpuFusedRopeParamsPtr&           params) {
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

    RTP_LLM_CHECK_WITH_INFO(params->sequence_lengths.is_cuda() || params->sequence_lengths.is_pinned(),
                            "sequence_lengths must be CUDA or pinned host memory");
    const int* position_ids_ptr = (params->position_ids.defined() && params->position_ids.numel() > 0) ?
                                      static_cast<const int*>(params->position_ids.data_ptr()) :
                                      static_cast<const int*>(params->sequence_lengths.data_ptr());
    StreamType stream           = GET_CURRENT_STREAM();
    DISPATCH_CUDA_FUNCTION_DATA_TYPE(
        torchDTypeToDataType(qkv.dtype()),
        invokeDecodeAddFusedQKVBiasTranspose,
        q_output.data_ptr(),
        nullptr,  // k_buf
        nullptr,  // v_buf
        kv_block_array,
        qkv.data_ptr(),
        position_ids_ptr,
        static_cast<const int*>(params->sequence_lengths.data_ptr()),
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
    pybind11::class_<PpuFusedRopeParams, std::shared_ptr<PpuFusedRopeParams>, ParamsBase>(m, "PpuFusedRopeParams")
        .def(pybind11::init<>())
        .def_readwrite("kv_cache_offset", &PpuFusedRopeParams::kv_cache_offset);
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
