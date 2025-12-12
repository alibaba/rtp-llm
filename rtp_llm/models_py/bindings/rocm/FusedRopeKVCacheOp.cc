#include "rtp_llm/models_py/bindings/rocm/FusedRopeKVCacheOp.h"
#include "rtp_llm/cpp/kernels/unfused_attention_kernels.h"
#include "rtp_llm/cpp/core/Dispatch.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include <stdexcept>
#include <string>
#include "rtp_llm/cpp/model_utils/RopeConfig.h"
#include "rtp_llm/cpp/kernels/kv_cache/kv_cache_utils.h"
#include "rtp_llm/cpp/devices/utils/RopeCache.h"
#include "rtp_llm/cpp/devices/utils/DebugUtils.h"

namespace rtp_llm {

FusedRopeKVCachePrefillOpBase::FusedRopeKVCachePrefillOpBase(const AttentionConfigs& attn_configs):
    attn_configs_(attn_configs),
    device_(dynamic_cast<ROCmDevice*>(DeviceFactory::getDefaultDevice())) {}

FusedRopeKVCachePrefillOpAsm::FusedRopeKVCachePrefillOpAsm(const AttentionConfigs& attn_configs):
    FusedRopeKVCachePrefillOpBase(attn_configs) {}

FusedRopeKVCachePrefillOpNonAsm::FusedRopeKVCachePrefillOpNonAsm(const AttentionConfigs& attn_configs):
    FusedRopeKVCachePrefillOpBase(attn_configs) {}

CKAttnPtr FusedRopeKVCachePrefillOpBase::prepare(torch_ext::PyAttentionInputs attn_inputs) {
    int       batch_size = attn_inputs.input_lengths.size(0);
    BufferPtr kv_cache_block_id_host, kv_cache_block_id_device;
    if (attn_inputs.kv_cache_block_id_host.defined() && attn_inputs.kv_cache_block_id_host.numel() > 0) {
        kv_cache_block_id_host   = torchTensor2Buffer(attn_inputs.kv_cache_block_id_host);
        kv_cache_block_id_device = torchTensor2Buffer(attn_inputs.kv_cache_block_id_device);
    }
    // 计算累积序列长度
    torch::Tensor cu_seqlens = torch::zeros({batch_size + 1}, torch::TensorOptions(torch::kInt32).device(torch::kCPU));
    cu_seqlens.slice(0, 1, batch_size + 1) = attn_inputs.input_lengths.cumsum(0);
    cu_seqlens                             = cu_seqlens.cuda();

    // 计算包含前缀的累积序列长度
    torch::Tensor kv_lengths = attn_inputs.input_lengths.clone();
    bool          has_prefix = attn_inputs.prefix_lengths.defined() && attn_inputs.prefix_lengths.numel() > 0;
    if (has_prefix) {
        kv_lengths = kv_lengths + attn_inputs.prefix_lengths;
    }
    torch::Tensor cu_kv_seqlens =
        torch::zeros({batch_size + 1}, torch::TensorOptions(torch::kInt32).device(torch::kCPU));
    cu_kv_seqlens.slice(0, 1, batch_size + 1) = kv_lengths.cumsum(0);
    cu_kv_seqlens                             = cu_kv_seqlens.cuda();

    bool use_fmha_fp8 = false;
    use_fmha_fp8 = attn_configs_.kv_cache_dtype == KvCacheDataType::FP8;
    CKAttnPtr attn_params;
    auto      params = device_->PrepareCKAttn(
        attn_configs_, attn_inputs.kv_block_offset, kv_cache_block_id_device, attn_inputs.input_lengths.size(0), use_fmha_fp8);
    if (params) {
        attn_params = CKAttnPtr(params, (CKAttn*)params.get());
    } else {
        attn_params = std::make_shared<CKAttn>();
    }
    attn_params->attn_type      = torchDTypeToDataType(attn_inputs.dtype);
    attn_params->cu_seqlens     = cu_seqlens;
    attn_params->cu_kv_seqlens  = cu_kv_seqlens;
    attn_params->attn_type      = torchDTypeToDataType(attn_inputs.dtype);
    attn_params->max_seq_len    = attn_inputs.input_lengths.max().item<int32_t>();
    attn_params->padding_offset = attn_inputs.padding_offset;
    // 处理 prefix_lengths：确保在 CUDA 上且连续
    if (has_prefix) {
        torch::Tensor prefix_lengths = attn_inputs.prefix_lengths;
        if (!prefix_lengths.is_cuda()) {
            prefix_lengths = prefix_lengths.to(torch::kCUDA, /*non_blocking=*/false, /*copy=*/true);
        }
        attn_params->prefix_lengths = prefix_lengths.contiguous();
    } else {
        attn_params->prefix_lengths = attn_inputs.prefix_lengths;
    }
    attn_params->kv_block_array.cache_type = attn_configs_.kv_cache_dtype;
    return attn_params;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> FusedRopeKVCachePrefillOpBase::forward(
    const torch::Tensor& qkv, FMHAType fmha_type, std::optional<torch_ext::KVCache> kv_cache, const CKAttnPtr& params) {
    const int local_head_num    = attn_configs_.head_num;
    const int local_head_num_kv = attn_configs_.kv_head_num;
    const int size_per_head     = attn_configs_.size_per_head;
    const int token_num         = qkv.size(0);
    const int batch_size        = params->cu_seqlens.size(0) - 1;
    const int seq_len           = params->max_seq_len;

    // 计算包含 prefix 的序列长度
    int max_prefix_length = 0;
    if (params->prefix_lengths.size(0)) {
        max_prefix_length = params->prefix_lengths.max().item<int>();
    }
    const int seq_len_with_prefix = seq_len + max_prefix_length;

    torch::Tensor q_output = torch::zeros({batch_size, local_head_num, seq_len, size_per_head},
                                          torch::TensorOptions(qkv.dtype()).device(qkv.device()));
    torch::Tensor k_output = torch::zeros({batch_size, local_head_num_kv, seq_len_with_prefix, size_per_head},
                                          torch::TensorOptions(qkv.dtype()).device(qkv.device()));
    torch::Tensor v_output = torch::zeros({batch_size, local_head_num_kv, seq_len_with_prefix, size_per_head},
                                          torch::TensorOptions(qkv.dtype()).device(qkv.device()));

    PrefixPromptBatchWeightsParam prefix_prompt_param{};
    bool use_fmha_fp8 = false;
    if (kv_cache.has_value()) {
        // 验证KV cache指针有效性
        if (!kv_cache.value().k_cache_base.defined() || kv_cache.value().k_cache_base.numel() == 0) {
            throw std::runtime_error("FusedRopeKVCachePrefillOp: k_cache_base is not defined or empty");
        }

        auto  kv_block_array = params->kv_block_array;
        void* k_cache_ptr    = kv_cache.value().k_cache_base.data_ptr();
        if (k_cache_ptr == nullptr) {
            throw std::runtime_error("FusedRopeKVCachePrefillOp: k_cache_base data pointer is null");
        }

        kv_block_array.mPrimaryPoolPtr = k_cache_ptr;
        if (kv_cache.value().k_scale_base.defined() && kv_cache.value().k_scale_base.numel() > 0) {
            void* scale_ptr = kv_cache.value().k_scale_base.data_ptr();
            if (scale_ptr != nullptr) {
                kv_block_array.scale = scale_ptr;
            }
        }
        prefix_prompt_param.kv_block_array = kv_block_array;
        use_fmha_fp8 = kv_block_array.cache_type == KvCacheDataType::FP8;
    }

    // 设置 prefix_lengths 参数
    if (max_prefix_length > 0) {
        int* prefix_lengths_ptr = params->prefix_lengths.data_ptr<int>();
        if (prefix_lengths_ptr == nullptr) {
            throw std::runtime_error("FusedRopeKVCachePrefillOp: prefix_lengths data pointer is null");
        }

        prefix_prompt_param.d_prefix_prompt_lengths  = prefix_lengths_ptr;
        prefix_prompt_param.max_prefix_prompt_length = max_prefix_length;
        prefix_prompt_param.count_length             = 1;
    }
    if (prefix_prompt_param.max_prefix_prompt_length > 0) {
        float* scale_out_ptr = nullptr;
        int    int8_mode     = 0;
        // Always use aiter_pa for ROCm
        hipStream_t stream_ = GET_CURRENT_STREAM();
        if (use_asm()) {
            DISPATCH_CUDA_FUNCTION_DATA_TYPE(torchDTypeToDataType(qkv.dtype()),
                                             invokeLoadPrefixKVCacheAiter,
                                             q_output.data_ptr(),
                                             k_output.data_ptr(),
                                             v_output.data_ptr(),
                                             &prefix_prompt_param,
                                             batch_size,
                                             seq_len,
                                             local_head_num,
                                             local_head_num_kv,
                                             size_per_head,
                                             scale_out_ptr,
                                             int8_mode,
                                             stream_);
        } else {
            DISPATCH_CUDA_FUNCTION_DATA_TYPE(torchDTypeToDataType(qkv.dtype()),
                                             invokeLoadPrefixKVCacheAiterV1,
                                             q_output.data_ptr(),
                                             k_output.data_ptr(),
                                             v_output.data_ptr(),
                                             &prefix_prompt_param,
                                             batch_size,
                                             seq_len,
                                             local_head_num,
                                             local_head_num_kv,
                                             size_per_head,
                                             scale_out_ptr,
                                             int8_mode,
                                             stream_);
        }
    }
    if (qkv.dtype().toScalarType() == torch::kFloat16) {
        // TODO: FP8 FMHA currently does not support FP16 output.
        //       Please run with BF16 activation instead (set environment variable ACT_TYPE=bf16)
        use_fmha_fp8 = false;
    }
    bool store_qkv   = true;  // 存储回原始 QKV
    bool store_q     = true;   // 存储到独立 Q 缓冲区
    bool store_kv    = true;   // 存储到独立 K、V 缓冲区
    bool store_cache = kv_cache.has_value();

    // int8
    float* scale_out_ptr = nullptr;
    int    int8_mode     = 0;
    // Always use aiter_pa for ROCm
    hipStream_t stream_ = GET_CURRENT_STREAM();
    // 添加 FP8 缓冲区支持
    torch::Tensor qkv_buf_fp8;
    if (use_fmha_fp8) {
        qkv_buf_fp8 = torch::empty(qkv.sizes(), torch::TensorOptions(torch::kFloat8_e4m3fn).device(qkv.device()));
    }

    if (use_asm()) {
        DISPATCH_CUDA_FUNCTION_DATA_TYPE(torchDTypeToDataType(qkv.dtype()),
                                         invokeAddFusedQKVBiasTransposePrefill,
                                         q_output.data_ptr(),
                                         k_output.data_ptr(),
                                         v_output.data_ptr(),
                                         &prefix_prompt_param,
                                         qkv.data_ptr(),
                                         use_fmha_fp8 && qkv_buf_fp8.defined() ? qkv_buf_fp8.data_ptr() : nullptr,
                                         nullptr,  // position_ids - 需要根据实际需求传入，暂时保持 nullptr
                                         nullptr,  // qkv_bias - 需要根据实际需求传入，暂时保持 nullptr
                                         params->padding_offset.data_ptr<int>(),
                                         params->cu_seqlens.data_ptr<int>(),
                                         batch_size,
                                         seq_len,
                                         token_num,
                                         local_head_num,
                                         local_head_num_kv,
                                         size_per_head,
                                         attn_configs_.rope_config,
                                         attn_configs_.use_logn_attn,
                                         scale_out_ptr,
                                         int8_mode,
                                         false,        // use_paged_fmha
                                         store_qkv,    // store_qkv
                                         store_q,      // store_q
                                         store_kv,     // store_kv
                                         store_cache,  // store_cache
                                         nullptr,
                                         stream_  // 必须作为最后一个参数
        );
    } else {
        DISPATCH_CUDA_FUNCTION_DATA_TYPE(torchDTypeToDataType(qkv.dtype()),
                                         invokeAddFusedQKVBiasTransposePrefillV1,
                                         q_output.data_ptr(),
                                         k_output.data_ptr(),
                                         v_output.data_ptr(),
                                         &prefix_prompt_param,
                                         qkv.data_ptr(),
                                         nullptr,
                                         nullptr,
                                         nullptr,
                                         nullptr,
                                         params->cu_seqlens.data_ptr<int>(),
                                         batch_size,
                                         seq_len,
                                         token_num,
                                         local_head_num,
                                         local_head_num_kv,
                                         size_per_head,
                                         attn_configs_.rope_config,
                                         attn_configs_.use_logn_attn,
                                         nullptr,
                                         0,
                                         false,        // use_paged_fmha
                                         store_qkv,    // store_qkv
                                         store_q,      // store_q
                                         store_kv,     // store_kv
                                         store_cache,  // store_cache
                                         nullptr,
                                         stream_  // 必须作为最后一个参数
        );
    }
    if (use_fmha_fp8) {
        return std::make_tuple(qkv_buf_fp8, torch::Tensor(), torch::Tensor());
    }
    if (prefix_prompt_param.max_prefix_prompt_length <= 0) {
        return std::make_tuple(qkv, torch::Tensor(), torch::Tensor());
    }
    // local_head_num, seq_len * batch_size, size_per_head
    torch::Tensor q_contiguous = torch::zeros({local_head_num, seq_len * batch_size, size_per_head},
                                              torch::TensorOptions(qkv.dtype()).device(qkv.device()));
    torch::Tensor k_contiguous = torch::zeros({local_head_num_kv, seq_len_with_prefix * batch_size, size_per_head},
                                              torch::TensorOptions(qkv.dtype()).device(qkv.device()));
    torch::Tensor v_contiguous = torch::zeros({local_head_num_kv, seq_len_with_prefix * batch_size, size_per_head},
                                              torch::TensorOptions(qkv.dtype()).device(qkv.device()));
    DISPATCH_CUDA_FUNCTION_DATA_TYPE(torchDTypeToDataType(qkv.dtype()),
                                     invokeGatherSequencesCombined,
                                     q_contiguous.data_ptr(),
                                     k_contiguous.data_ptr(),
                                     v_contiguous.data_ptr(),
                                     q_output.data_ptr(),
                                     k_output.data_ptr(),
                                     v_output.data_ptr(),
                                     params->cu_seqlens.data_ptr<int>(),
                                     params->cu_kv_seqlens.data_ptr<int>(),
                                     batch_size,
                                     seq_len,
                                     seq_len_with_prefix,
                                     local_head_num,
                                     local_head_num_kv,
                                     size_per_head,
                                     stream_);

    return std::make_tuple(q_contiguous, k_contiguous, v_contiguous);
}

FusedRopeKVCacheDecodeOpBase::FusedRopeKVCacheDecodeOpBase(const AttentionConfigs& attn_configs):
    attn_configs_(attn_configs),
    device_(dynamic_cast<ROCmDevice*>(DeviceFactory::getDefaultDevice())) {}

FusedRopeKVCacheDecodeOpAsm::FusedRopeKVCacheDecodeOpAsm(const AttentionConfigs& attn_configs):
    FusedRopeKVCacheDecodeOpBase(attn_configs) {}

FusedRopeKVCacheDecodeOpNonAsm::FusedRopeKVCacheDecodeOpNonAsm(const AttentionConfigs& attn_configs):
    FusedRopeKVCacheDecodeOpBase(attn_configs) {}

CKAttnPtr FusedRopeKVCacheDecodeOpBase::prepare(torch_ext::PyAttentionInputs attn_inputs) {
    int       batch_size = attn_inputs.sequence_lengths.size(0);
    BufferPtr kv_cache_block_id_host, kv_cache_block_id_device;
    if (attn_inputs.kv_cache_block_id_host.defined() && attn_inputs.kv_cache_block_id_host.numel() > 0) {

        kv_cache_block_id_host   = torchTensor2Buffer(attn_inputs.kv_cache_block_id_host);
        kv_cache_block_id_device = torchTensor2Buffer(attn_inputs.kv_cache_block_id_device);
    }
    // not support has_alibi_slopes
    attn_inputs.cu_seqlens.slice(0, 1, batch_size + 1) = attn_inputs.input_lengths.cumsum(0);
    auto cu_seqlens                                    = attn_inputs.cu_seqlens;

    // 计算包含前缀的累积序列长度
    torch::Tensor kv_lengths = attn_inputs.input_lengths;
    if (attn_inputs.prefix_lengths.defined() && attn_inputs.prefix_lengths.numel() > 0) {
        kv_lengths = kv_lengths + attn_inputs.prefix_lengths;
    }

    torch::Tensor cu_kv_seqlens =
        torch::zeros({batch_size + 1}, torch::TensorOptions(torch::kInt32).device(torch::kCPU));
    cu_kv_seqlens.slice(0, 1, batch_size + 1) = kv_lengths.cumsum(0);
    cu_kv_seqlens                             = cu_kv_seqlens.cuda();

    CKAttnPtr attn_params;
    bool use_fmha_fp8 = false;
    use_fmha_fp8 = attn_configs_.kv_cache_dtype == KvCacheDataType::FP8;

    auto params = device_->PrepareCKAttn(
        attn_configs_, attn_inputs.kv_block_offset, kv_cache_block_id_device, attn_inputs.sequence_lengths.size(0), use_fmha_fp8);
    if (!params) {
        throw std::runtime_error("FusedRopeKVCacheDecodeOp::prepare: PrepareCKAttn failed. "
                                 "kv_block_offset="
                                 + std::to_string(attn_inputs.kv_block_offset) + ", kv_cache_block_id_size="
                                 + std::to_string(attn_inputs.kv_cache_block_id_host.size(0)));
    }

    attn_params                            = CKAttnPtr(params, (CKAttn*)params.get());
    attn_params->decode_plan               = true;
    attn_params->cu_seqlens                = cu_seqlens;
    attn_params->cu_kv_seqlens             = cu_kv_seqlens;
    attn_params->sequence_lengths          = attn_inputs.sequence_lengths;
    attn_params->kv_block_array.cache_type = attn_configs_.kv_cache_dtype;
    attn_params->input_lengths             = attn_inputs.input_lengths;
    attn_params->prefix_lengths            = attn_inputs.prefix_lengths;
    attn_params->padding_offset            = attn_inputs.padding_offset;

    if (attn_inputs.kv_cache_block_id_device.defined() && attn_inputs.kv_cache_block_id_device.numel() > 0) {
        attn_params->kv_cache_block_id_device = attn_inputs.kv_cache_block_id_device;
    }

    return attn_params;
}

torch::Tensor FusedRopeKVCacheDecodeOpBase::forward(const torch::Tensor&              qkv,
                                                FMHAType                          fmha_type,
                                                std::optional<torch_ext::KVCache> kv_cache,
                                                const CKAttnPtr&                  params) {
    // Check that kv_cache is provided
    // (CUDA version uses RTP_LLM_CHECK_WITH_INFO, use assert or similar if not available)
    assert(kv_cache.has_value() && "decode should have kv cache.");

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

    PrefixPromptBatchWeightsParam prefix_prompt_param;
    prefix_prompt_param.kv_block_array = kv_block_array;

    // 设置 prefix_lengths 参数
    int max_prefix_length = 0;
    if (params->prefix_lengths.defined() && params->prefix_lengths.size(0) > 0) {
        max_prefix_length = params->prefix_lengths.max().item<int>();

        int* prefix_lengths_ptr = params->prefix_lengths.data_ptr<int>();
        if (prefix_lengths_ptr == nullptr) {
            throw std::runtime_error("FusedRopeKVCacheDecodeOp: prefix_lengths data pointer is null");
        }

        prefix_prompt_param.d_prefix_prompt_lengths  = prefix_lengths_ptr;
        prefix_prompt_param.max_prefix_prompt_length = max_prefix_length;
        prefix_prompt_param.count_length             = 1;
    }

    size_t seq_len     = 1;
    bool   store_qkv   = false;
    bool   store_q     = true;
    bool   store_kv    = false;
    bool   store_cache = kv_cache.has_value();

    // Always use aiter_pa for ROCm
    hipStream_t stream_ = GET_CURRENT_STREAM();
    if (use_asm()) {
        DISPATCH_CUDA_FUNCTION_DATA_TYPE(
            torchDTypeToDataType(qkv.dtype()),
            invokeAddFusedQKVBiasTransposeDecode,
            q_output.data_ptr(),
            nullptr,
            nullptr,
            &prefix_prompt_param,
            params->input_lengths.data_ptr<int>(),
            qkv.data_ptr(),
            nullptr,
            /*params.common.position_ids*/ nullptr,
            /*qkv_bias*/ nullptr,  //                params.configs.fuse_qkv_add_bias &&
                                   //                params.weights.qkv_weight->bias?
                                   //                params.weights.qkv_weight->bias->data(): nullptr,???
            params->padding_offset.data_ptr<int>(),
            params->cu_seqlens.data_ptr<int>(),
            params->sequence_lengths.data_ptr<int>(),
            batch_size,
            seq_len,
            token_num,
            local_head_num,
            local_head_num_kv,
            size_per_head,
            attn_configs_.rope_config,
            attn_configs_.use_logn_attn,
            nullptr,
            0,
            false,
            store_qkv,
            store_q,
            store_kv,
            store_cache,
            nullptr,
            stream_);
    } else {
        DISPATCH_CUDA_FUNCTION_DATA_TYPE(
            torchDTypeToDataType(qkv.dtype()),
            invokeAddFusedQKVBiasTransposeDecodeV1,
            q_output.data_ptr(),
            nullptr,
            nullptr,
            &prefix_prompt_param,
            params->input_lengths.data_ptr<int>(),
            qkv.data_ptr(),
            nullptr,
            /*params.common.position_ids*/ nullptr,
            /*qkv_bias*/ nullptr,  //                params.configs.fuse_qkv_add_bias &&
                                   //                params.weights.qkv_weight->bias?
                                   //                params.weights.qkv_weight->bias->data(): nullptr,???
            params->padding_offset.data_ptr<int>(),
            params->cu_seqlens.data_ptr<int>(),
            params->sequence_lengths.data_ptr<int>(),
            batch_size,
            seq_len,
            token_num,
            local_head_num,
            local_head_num_kv,
            size_per_head,
            attn_configs_.rope_config,
            attn_configs_.use_logn_attn,
            nullptr,
            0,
            false,
            store_qkv,
            store_q,
            store_kv,
            store_cache,
            nullptr,
            stream_);
    }

    return q_output;
}

void registerFusedRopeKVCacheOp(const py::module& m) {
    pybind11::class_<KVBlockArray>(m, "KVBlockArray").def(pybind11::init<>());
    pybind11::class_<CKAttn, std::shared_ptr<CKAttn>>(m, "CKAttn").def(pybind11::init<>());
    
    // Prefill ASM
    pybind11::class_<FusedRopeKVCachePrefillOpAsm>(m, "FusedRopeKVCachePrefillOpAsm")
        .def(pybind11::init<const AttentionConfigs&>(),
             py::arg("attn_configs"))
        .def("prepare", &FusedRopeKVCachePrefillOpAsm::prepare, py::arg("attn_inputs"))
        .def("forward",
             &FusedRopeKVCachePrefillOpAsm::forward,
             py::arg("qkv"),
             py::arg("fmha_type"),
             py::arg("kv_cache"),
             py::arg("params"));
    
    // Prefill Non-ASM
    pybind11::class_<FusedRopeKVCachePrefillOpNonAsm>(m, "FusedRopeKVCachePrefillOpNonAsm")
        .def(pybind11::init<const AttentionConfigs&>(),
             py::arg("attn_configs"))
        .def("prepare", &FusedRopeKVCachePrefillOpNonAsm::prepare, py::arg("attn_inputs"))
        .def("forward",
             &FusedRopeKVCachePrefillOpNonAsm::forward,
             py::arg("qkv"),
             py::arg("fmha_type"),
             py::arg("kv_cache"),
             py::arg("params"));
    
    // Decode ASM
    pybind11::class_<FusedRopeKVCacheDecodeOpAsm>(m, "FusedRopeKVCacheDecodeOpAsm")
        .def(pybind11::init<const AttentionConfigs&>(),
             py::arg("attn_configs"))
        .def("prepare", &FusedRopeKVCacheDecodeOpAsm::prepare, py::arg("attn_inputs"))
        .def("forward",
             &FusedRopeKVCacheDecodeOpAsm::forward,
             py::arg("qkv"),
             py::arg("fmha_type"),
             py::arg("kv_cache"),
             py::arg("params"));
    
    // Decode Non-ASM
    pybind11::class_<FusedRopeKVCacheDecodeOpNonAsm>(m, "FusedRopeKVCacheDecodeOpNonAsm")
        .def(pybind11::init<const AttentionConfigs&>(),
             py::arg("attn_configs"))
        .def("prepare", &FusedRopeKVCacheDecodeOpNonAsm::prepare, py::arg("attn_inputs"))
        .def("forward",
             &FusedRopeKVCacheDecodeOpNonAsm::forward,
             py::arg("qkv"),
             py::arg("fmha_type"),
             py::arg("kv_cache"),
             py::arg("params"));
}
}  // namespace rtp_llm
