#include "rtp_llm/models_py/bindings/cuda/FusedRopeKVCacheOp.h"
#include "rtp_llm/cpp/kernels/unfused_attention_kernels.h"
#include "rtp_llm/cpp/core/Dispatch.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/devices/cuda_impl/CudaFlashInfer.h"
#include "rtp_llm/cpp/devices/utils/RopeCache.h"
#include "rtp_llm/cpp/core/BufferHelper.h"

namespace rtp_llm {

namespace {
// Helper function to prepare prefix prompt param and padding offset, then invoke the kernel
void invokeFusedQKVBiasTransposeHelper(const torch::Tensor&              qkv,
                                       std::optional<torch_ext::KVCache> kv_cache,
                                       const TRTAttnPtr&                 params,
                                       const AttentionConfigs&           attn_configs,
                                       CudaDevice*                       device,
                                       void*                             q_no_transpose_output,
                                       void*                             q_output,
                                       void*                             qkv_fp8_output,
                                       int                               token_num,
                                       int                               batch_size,
                                       int                               local_head_num,
                                       int                               local_head_num_kv,
                                       int                               size_per_head,
                                       bool                              use_paged_attention,
                                       bool                              store_qkv,
                                       bool                              store_q_no_transpose,
                                       bool                              store_q,
                                       bool                              store_kv,
                                       bool                              store_cache) {
    PrefixPromptBatchWeightsParam prefix_prompt_param;
    if (kv_cache.has_value()) {
        auto kv_block_array            = params->kv_block_array;
        kv_block_array.mPrimaryPoolPtr = kv_cache.value().k_cache_base.data_ptr();
        if (kv_cache.value().k_scale_base.defined() && kv_cache.value().k_scale_base.numel()) {
            kv_block_array.scale = kv_cache.value().k_scale_base.data_ptr();
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
    auto rope_cache = getRopeCacheOnce(attn_configs.rope_config, device->initParams().max_seq_len);
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
        nullptr,  // position_ids
        nullptr,  // qkv_bias
        padding_offset,
        params->cu_seqlens.data_ptr<int>(),
        params->cu_seqlens_without_prefix.data_ptr<int>(),
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
        device->getStream());
}
}  // namespace

FusedRopeKVCachePrefillOpBase::FusedRopeKVCachePrefillOpBase(const GptInitParameter& gpt_init_parameter):
    FMHACudaBase(gpt_init_parameter) {}

TRTAttnPtr FusedRopeKVCachePrefillOpBase::prepare(torch_ext::PyAttentionInputs attn_inputs) {
    int       batch_size = attn_inputs.input_lengths.size(0);
    BufferPtr kv_cache_block_id_host, kv_cache_block_id_device;
    if (attn_inputs.kv_cache_block_id_host.defined() && attn_inputs.kv_cache_block_id_host.numel() > 0) {
        kv_cache_block_id_host   = torchTensor2Buffer(attn_inputs.kv_cache_block_id_host);
        kv_cache_block_id_device = torchTensor2Buffer(attn_inputs.kv_cache_block_id_device);
    }
    // not support has_alibi_slopes
    TRTAttnPtr attn_params;
    auto       params =
        device_->prepareTrtAttn(attn_configs_, attn_inputs.kv_block_offset, kv_cache_block_id_device, batch_size);
    if (params) {
        attn_params = TRTAttnPtr(params, (TRTAttn*)params.get());
    } else {
        attn_params = std::make_shared<TRTAttn>();
    }
    attn_params->attn_type                 = torchDTypeToDataType(attn_inputs.dtype);
    attn_params->cu_seqlens                = attn_inputs.cu_seqlens;
    attn_params->cu_kv_seqlens             = attn_inputs.cu_kv_seqlens;
    attn_params->cu_seqlens_without_prefix = attn_inputs.cu_seqlens_without_prefix;
    attn_params->max_seq_len               = attn_inputs.input_lengths.max().item<int32_t>();
    attn_params->max_prefix_length         = attn_inputs.prefix_lengths.max().item<int32_t>();
    attn_params->prefix_lengths            = attn_inputs.prefix_lengths;
    attn_params->kv_block_array.cache_type = attn_configs_.kv_cache_dtype;
    attn_params->padding_offset            = attn_inputs.padding_offset;
    return attn_params;
}

FusedRopeKVCachePrefillOpQOut::FusedRopeKVCachePrefillOpQOut(const GptInitParameter& gpt_init_parameter):
    FusedRopeKVCachePrefillOpBase(gpt_init_parameter) {}

torch::Tensor FusedRopeKVCachePrefillOpQOut::forward(const torch::Tensor&              qkv,
                                                     FMHAType                          fmha_type,
                                                     std::optional<torch_ext::KVCache> kv_cache,
                                                     const TRTAttnPtr&                 params) {
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
                                      device_,
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

FusedRopeKVCachePrefillOpQKVOut::FusedRopeKVCachePrefillOpQKVOut(const GptInitParameter& gpt_init_parameter):
    FusedRopeKVCachePrefillOpBase(gpt_init_parameter) {}

torch::Tensor FusedRopeKVCachePrefillOpQKVOut::forward(const torch::Tensor&              qkv,
                                                       FMHAType                          fmha_type,
                                                       std::optional<torch_ext::KVCache> kv_cache,
                                                       const TRTAttnPtr&                 params) {
    const int local_head_num    = attn_configs_.head_num;
    const int local_head_num_kv = attn_configs_.kv_head_num;
    const int size_per_head     = attn_configs_.size_per_head;
    const int token_num         = qkv.size(0);
    const int batch_size        = params->cu_seqlens.size(0) - 1;

    invokeFusedQKVBiasTransposeHelper(qkv,
                                      kv_cache,
                                      params,
                                      attn_configs_,
                                      device_,
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

    TRTAttnPtr attn_params;
    auto       params = device_->prepareTrtAttn(
        attn_configs_, attn_inputs.kv_block_offset, kv_cache_block_id_device, attn_inputs.sequence_lengths.size(0));
    RTP_LLM_CHECK_WITH_INFO(params != nullptr, "TRTAttnPtr Build Failed");
    attn_params                            = TRTAttnPtr(params, (TRTAttn*)params.get());
    attn_params->decode_plan               = true;
    attn_params->attn_type                 = torchDTypeToDataType(attn_inputs.dtype);
    attn_params->cu_seqlens                = attn_inputs.cu_seqlens;
    attn_params->cu_kv_seqlens             = attn_inputs.cu_kv_seqlens;
    attn_params->cu_seqlens_without_prefix = attn_inputs.cu_seqlens_without_prefix;
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

    auto rope_cache = getRopeCacheOnce(attn_configs_.rope_config, device_->initParams().max_seq_len);

    RTP_LLM_CHECK_WITH_INFO(params->sequence_lengths.is_pinned(), "sequence_lengths is not pinned memory");
    DISPATCH_CUDA_FUNCTION_DATA_TYPE(
        torchDTypeToDataType(qkv.dtype()),
        invokeDecodeAddFusedQKVBiasTranspose,
        q_output.data_ptr(),
        nullptr,  // k_buf
        nullptr,  // v_buf
        kv_block_array,
        qkv.data_ptr(),
        params->sequence_lengths.data_ptr<int>(),
        nullptr,  // params.configs.fuse_qkv_add_bias && params.weights.qkv_weight->bias ?
                  // params.weights.qkv_weight->bias->data() : nullptr,
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
        device_->getStream());
    return q_output;
}

void registerFusedRopeKVCacheOp(const py::module& m) {
    pybind11::class_<KVBlockArray>(m, "KVBlockArray").def(pybind11::init<>());
    pybind11::class_<TRTAttn, std::shared_ptr<TRTAttn>, rtp_llm::ParamsBase>(m, "TRTAttn").def(pybind11::init<>());

    pybind11::class_<FusedRopeKVCachePrefillOpQKVOut>(m, "FusedRopeKVCachePrefillOpQKVOut")
        .def(pybind11::init<GptInitParameter>(), py::arg("gpt_init_parameter"))
        .def("prepare", &FusedRopeKVCachePrefillOpQKVOut::prepare, py::arg("attn_inputs"))
        .def("forward",
             &FusedRopeKVCachePrefillOpQKVOut::forward,
             py::arg("qkv"),
             py::arg("fmha_type"),
             py::arg("kv_cache"),
             py::arg("params"));

    pybind11::class_<FusedRopeKVCachePrefillOpQOut>(m, "FusedRopeKVCachePrefillOpQOut")
        .def(pybind11::init<GptInitParameter>(), py::arg("gpt_init_parameter"))
        .def("prepare", &FusedRopeKVCachePrefillOpQOut::prepare, py::arg("attn_inputs"))
        .def("forward",
             &FusedRopeKVCachePrefillOpQOut::forward,
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
