#include "rtp_llm/models_py/bindings/rocm/FusedRopeKVCacheOp.h"
#include "rtp_llm/cpp/kernels/unfused_attention_kernels.h"
#include "rtp_llm/cpp/core/Dispatch.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include <stdexcept>
#include <string>
#include "rtp_llm/cpp/model_utils/RopeConfig.h"
#include "rtp_llm/cpp/kernels/kv_cache/kv_cache_utils.h"
#include "rtp_llm/cpp/kernels/kv_cache_kernels.h"
#include "rtp_llm/cpp/devices/utils/RopeCache.h"
#include "rtp_llm/cpp/devices/utils/DebugUtils.h"

namespace rtp_llm {

void updateKvCacheOffset(CKAttn& params, const torch::Tensor& kv_cache_block_id_device) {
    if (!params.kv_cache_offset || !kv_cache_block_id_device.defined() || kv_cache_block_id_device.numel() == 0) {
        return;
    }
    const int   batch_size        = kv_cache_block_id_device.size(0);
    const int   max_blocks_per_bs = kv_cache_block_id_device.size(1);
    hipStream_t stream            = GET_CURRENT_STREAM();
    invokeConvertOffsetToBlockArrayData(params.kv_cache_offset->data<int>(),
                                        kv_cache_block_id_device.data_ptr<int>(),
                                        batch_size,
                                        max_blocks_per_bs,
                                        stream);
}

FusedRopeKVCachePrefillOpBase::FusedRopeKVCachePrefillOpBase(const AttentionConfigs& attn_configs):
    attn_configs_(attn_configs), device_(dynamic_cast<ROCmDevice*>(DeviceFactory::getDefaultDevice())) {}

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

    bool has_prefix = attn_inputs.prefix_lengths.defined() && attn_inputs.prefix_lengths.numel() > 0;

    bool use_fmha_fp8 = false;
    use_fmha_fp8      = attn_configs_.kv_cache_dtype == KvCacheDataType::FP8;
    CKAttnPtr attn_params;
    auto      params = device_->PrepareCKAttn(
        attn_configs_, kv_cache_block_id_device, attn_inputs.input_lengths.size(0), use_fmha_fp8);
    if (params) {
        attn_params = CKAttnPtr(params, (CKAttn*)params.get());
    } else {
        attn_params = std::make_shared<CKAttn>();
    }
    attn_params->attn_type      = torchDTypeToDataType(attn_inputs.dtype);
    attn_params->cu_seqlens     = attn_inputs.cu_seqlens;
    attn_params->cu_kv_seqlens  = attn_inputs.cu_kv_seqlens;
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
    const torch::Tensor& qkv, std::optional<torch_ext::KVCache> kv_cache, const CKAttnPtr& params) {
    const int local_head_num    = attn_configs_.head_num;
    const int local_head_num_kv = attn_configs_.kv_head_num;
    const int size_per_head     = attn_configs_.size_per_head;
    const int token_num         = qkv.size(0);
    const int batch_size        = params->cu_seqlens.size(0) - 1;
    const int seq_len           = params->max_seq_len;

    int max_prefix_length = 0;
    if (params->prefix_lengths.size(0)) {
        max_prefix_length = params->prefix_lengths.max().item<int>();
    }
    const int seq_len_with_prefix = seq_len + max_prefix_length;

    PrefixPromptBatchWeightsParam prefix_prompt_param{};
    bool                          use_fmha_fp8 = false;
    if (kv_cache.has_value()) {
        if (!kv_cache.value().kv_cache_base.defined() || kv_cache.value().kv_cache_base.numel() == 0) {
            throw std::runtime_error("FusedRopeKVCachePrefillOp: kv_cache_base is not defined or empty");
        }

        auto  kv_block_array = params->kv_block_array;
        void* k_cache_ptr    = kv_cache.value().kv_cache_base.data_ptr();
        if (k_cache_ptr == nullptr) {
            throw std::runtime_error("FusedRopeKVCachePrefillOp: kv_cache_base data pointer is null");
        }

        kv_block_array.mPrimaryPoolPtr = k_cache_ptr;
        if (kv_cache.value().kv_scale_base.defined() && kv_cache.value().kv_scale_base.numel() > 0) {
            void* scale_ptr = kv_cache.value().kv_scale_base.data_ptr();
            if (scale_ptr != nullptr) {
                kv_block_array.scale = scale_ptr;
            }
        }
        prefix_prompt_param.kv_block_array = kv_block_array;
        use_fmha_fp8                       = kv_block_array.cache_type == KvCacheDataType::FP8;
    }

    if (max_prefix_length > 0) {
        int* prefix_lengths_ptr = params->prefix_lengths.data_ptr<int>();
        if (prefix_lengths_ptr == nullptr) {
            throw std::runtime_error("FusedRopeKVCachePrefillOp: prefix_lengths data pointer is null");
        }

        prefix_prompt_param.d_prefix_prompt_lengths  = prefix_lengths_ptr;
        prefix_prompt_param.max_prefix_prompt_length = max_prefix_length;
        prefix_prompt_param.count_length             = 1;
    }

    // Unified: Q is always packed [token_num, head_num, size_per_head].
    // K/V are written directly to paged cache; no separate output tensors needed.
    torch::Tensor q_output = torch::zeros({token_num, local_head_num, size_per_head},
                                          torch::TensorOptions(qkv.dtype()).device(qkv.device()));

    if (qkv.dtype().toScalarType() == torch::kFloat16) {
        use_fmha_fp8 = false;
    }

    // Unified flags: always store packed Q + cache, never store padded QKV or linear KV.
    bool store_qkv   = false;
    bool store_q     = true;
    bool store_kv    = false;
    bool store_cache = kv_cache.has_value();

    float*      scale_out_ptr = nullptr;
    int         int8_mode     = 0;
    hipStream_t stream_       = GET_CURRENT_STREAM();

    torch::Tensor qkv_buf_fp8;
    if (use_fmha_fp8) {
        qkv_buf_fp8 = torch::empty(qkv.sizes(), torch::TensorOptions(torch::kFloat8_e4m3fn).device(qkv.device()));
    }

    if (use_asm()) {
        DISPATCH_CUDA_FUNCTION_DATA_TYPE(torchDTypeToDataType(qkv.dtype()),
                                         invokeAddFusedQKVBiasTransposePrefill,
                                         q_output.data_ptr(),
                                         nullptr,
                                         nullptr,
                                         &prefix_prompt_param,
                                         qkv.data_ptr(),
                                         use_fmha_fp8 && qkv_buf_fp8.defined() ? qkv_buf_fp8.data_ptr() : nullptr,
                                         nullptr,
                                         nullptr,
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
                                         true,  // use_paged_fmha: always true for ASM
                                         store_qkv,
                                         store_q,
                                         store_kv,
                                         store_cache,
                                         nullptr,
                                         stream_);
    } else {
        // V1 kernel does not support use_paged_fmha=true, fall back to legacy path
        // with temporary K/V output buffers (only used internally by the kernel).
        torch::Tensor k_output_legacy =
            torch::zeros({batch_size, local_head_num_kv, seq_len_with_prefix, size_per_head},
                         torch::TensorOptions(qkv.dtype()).device(qkv.device()));
        torch::Tensor v_output_legacy =
            torch::zeros({batch_size, local_head_num_kv, seq_len_with_prefix, size_per_head},
                         torch::TensorOptions(qkv.dtype()).device(qkv.device()));

        DISPATCH_CUDA_FUNCTION_DATA_TYPE(torchDTypeToDataType(qkv.dtype()),
                                         invokeAddFusedQKVBiasTransposePrefillV1,
                                         q_output.data_ptr(),
                                         k_output_legacy.data_ptr(),
                                         v_output_legacy.data_ptr(),
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
                                         false,  // V1 does not support paged fmha
                                         true,   // store_qkv: V1 legacy needs this
                                         true,   // store_q
                                         true,   // store_kv: V1 legacy needs this
                                         store_cache,
                                         nullptr,
                                         stream_);
    }
    if (use_fmha_fp8) {
        return std::make_tuple(qkv_buf_fp8, torch::Tensor(), torch::Tensor());
    }
    // Unified: always return packed Q
    return std::make_tuple(q_output, torch::Tensor(), torch::Tensor());
}

FusedRopeKVCacheDecodeOpBase::FusedRopeKVCacheDecodeOpBase(const AttentionConfigs& attn_configs):
    attn_configs_(attn_configs), device_(dynamic_cast<ROCmDevice*>(DeviceFactory::getDefaultDevice())) {}

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

    CKAttnPtr attn_params;
    bool      use_fmha_fp8 = false;
    use_fmha_fp8           = attn_configs_.kv_cache_dtype == KvCacheDataType::FP8;

    auto params = device_->PrepareCKAttn(
        attn_configs_, kv_cache_block_id_device, attn_inputs.sequence_lengths.size(0), use_fmha_fp8);
    if (!params) {
        throw std::runtime_error("FusedRopeKVCacheDecodeOp::prepare: PrepareCKAttn failed. "
                                 "kv_cache_block_id_size="
                                 + std::to_string(attn_inputs.kv_cache_block_id_host.size(0)));
    }

    attn_params                            = CKAttnPtr(params, (CKAttn*)params.get());
    attn_params->decode_plan               = true;
    attn_params->attn_type                 = torchDTypeToDataType(attn_inputs.dtype);
    attn_params->cu_seqlens                = attn_inputs.cu_seqlens;
    attn_params->cu_kv_seqlens             = attn_inputs.cu_kv_seqlens;
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
                                                    std::optional<torch_ext::KVCache> kv_cache,
                                                    const CKAttnPtr&                  params) {

    RTP_LLM_CHECK_WITH_INFO(kv_cache.has_value(), "FusedRopeKVCacheDecodeOp: kv_cache is not initialized.");
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
    pybind11::class_<CKAttn, std::shared_ptr<CKAttn>>(m, "CKAttn")
        .def(pybind11::init<>())
        .def("update_kv_cache_offset", &updateKvCacheOffset, py::arg("kv_cache_block_id_device"));

    // Prefill ASM
    pybind11::class_<FusedRopeKVCachePrefillOpAsm>(m, "FusedRopeKVCachePrefillOpAsm")
        .def(pybind11::init<const AttentionConfigs&>(), py::arg("attn_configs"))
        .def("prepare", &FusedRopeKVCachePrefillOpAsm::prepare, py::arg("attn_inputs"))
        .def("forward", &FusedRopeKVCachePrefillOpAsm::forward, py::arg("qkv"), py::arg("kv_cache"), py::arg("params"));

    // Prefill Non-ASM
    pybind11::class_<FusedRopeKVCachePrefillOpNonAsm>(m, "FusedRopeKVCachePrefillOpNonAsm")
        .def(pybind11::init<const AttentionConfigs&>(), py::arg("attn_configs"))
        .def("prepare", &FusedRopeKVCachePrefillOpNonAsm::prepare, py::arg("attn_inputs"))
        .def("forward",
             &FusedRopeKVCachePrefillOpNonAsm::forward,
             py::arg("qkv"),
             py::arg("kv_cache"),
             py::arg("params"));

    // Decode ASM
    pybind11::class_<FusedRopeKVCacheDecodeOpAsm>(m, "FusedRopeKVCacheDecodeOpAsm")
        .def(pybind11::init<const AttentionConfigs&>(), py::arg("attn_configs"))
        .def("prepare", &FusedRopeKVCacheDecodeOpAsm::prepare, py::arg("attn_inputs"))
        .def("forward", &FusedRopeKVCacheDecodeOpAsm::forward, py::arg("qkv"), py::arg("kv_cache"), py::arg("params"));

    // Decode Non-ASM
    pybind11::class_<FusedRopeKVCacheDecodeOpNonAsm>(m, "FusedRopeKVCacheDecodeOpNonAsm")
        .def(pybind11::init<const AttentionConfigs&>(), py::arg("attn_configs"))
        .def("prepare", &FusedRopeKVCacheDecodeOpNonAsm::prepare, py::arg("attn_inputs"))
        .def("forward",
             &FusedRopeKVCacheDecodeOpNonAsm::forward,
             py::arg("qkv"),
             py::arg("kv_cache"),
             py::arg("params"));
}
}  // namespace rtp_llm
