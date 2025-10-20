#include "rtp_llm/models_py/bindings/rocm/FusedRopeKVCacheOp.h"
#include "rtp_llm/cpp/kernels/unfused_attention_kernels.h"
#include "rtp_llm/cpp/core/Dispatch.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include <stdexcept>
#include "rtp_llm/cpp/model_utils/RopeConfig.h"
#include "rtp_llm/cpp/kernels/kv_cache/kv_cache_utils.h"

namespace rtp_llm {

FusedRopeKVCachePrefillOp::FusedRopeKVCachePrefillOp(const GptInitParameter& gpt_init_parameter):
    FMHARocmBase(gpt_init_parameter) {}

CKAttnPtr FusedRopeKVCachePrefillOp::prepare(torch_ext::PyAttentionInputs attn_inputs) {
    int       batch_size = attn_inputs.input_lengths.size(0);
    BufferPtr kv_cache_block_id_host, kv_cache_block_id_device;
    if (attn_inputs.kv_cache_block_id_host.size(0)) {
        kv_cache_block_id_host   = torchTensor2Buffer(attn_inputs.kv_cache_block_id_host);
        kv_cache_block_id_device = torchTensor2Buffer(attn_inputs.kv_cache_block_id_device);
    }
    // not support has_alibi_slopes

    torch::Tensor cu_seqlens = torch::zeros({batch_size + 1}, torch::TensorOptions(torch::kInt32).device(torch::kCPU));
    cu_seqlens.slice(0, 1, batch_size + 1) = attn_inputs.input_lengths.cumsum(0);
    cu_seqlens                             = cu_seqlens.cuda();
    torch::Tensor cu_kv_seqlens            = cu_seqlens;
    CKAttnPtr     attn_params;
    auto          params = device_->PrepareCKAttn(
        attn_configs_, attn_inputs.kv_block_offset, kv_cache_block_id_device, attn_inputs.input_lengths.size(0));
    attn_params                = CKAttnPtr(params, (CKAttn*)params.get());
    attn_params->attn_type     = torchDTypeToDataType(attn_inputs.dtype);
    attn_params->cu_seqlens    = cu_seqlens;
    attn_params->cu_kv_seqlens = cu_kv_seqlens;
    attn_params->max_seq_len   = attn_inputs.input_lengths.max().item<int32_t>();
    return attn_params;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> FusedRopeKVCachePrefillOp::forward(
    const torch::Tensor& qkv, FMHAType fmha_type, std::optional<torch_ext::KVCache> kv_cache, const CKAttnPtr& params) {
    // bool store_cache = params.common.kv_cache.has_value();
    const int local_head_num    = attn_configs_.head_num;
    const int local_head_num_kv = attn_configs_.kv_head_num;
    const int size_per_head     = attn_configs_.size_per_head;
    const int token_num         = qkv.size(0);
    const int batch_size        = params->cu_seqlens.size(0) - 1;  // 修复：应该减1，与CUDA版本一致
    const int seq_len           = params->max_seq_len;

    const int seq_len_with_prefix = seq_len;  // 如果有 prefix 支持，这里应该是 seq_len + max_prefix_length

    torch::Tensor q_output = torch::empty({batch_size, local_head_num, seq_len, size_per_head},
                                          torch::TensorOptions(qkv.dtype()).device(qkv.device()));
    torch::Tensor k_output = torch::empty({batch_size, local_head_num_kv, seq_len_with_prefix, size_per_head},
                                          torch::TensorOptions(qkv.dtype()).device(qkv.device()));
    torch::Tensor v_output = torch::empty({batch_size, local_head_num_kv, seq_len_with_prefix, size_per_head},
                                          torch::TensorOptions(qkv.dtype()).device(qkv.device()));

    PrefixPromptBatchWeightsParam prefix_prompt_param;
    if (kv_cache.has_value()) {
        auto kv_block_array            = params->kv_block_array;
        kv_block_array.mPrimaryPoolPtr = kv_cache.value().k_cache_base.data_ptr();
        if (kv_cache.value().k_scale_base.defined() && kv_cache.value().k_scale_base.numel()) {
            kv_block_array.scale = kv_cache.value().k_scale_base.data_ptr();
        }
        prefix_prompt_param.kv_block_array = kv_block_array;
        // if (params.prefix_lengths.size(0)) {
        //      prefix_prompt_param.d_prefix_prompt_lengths  = params.prefix_lengths.data_ptr<int>();
        //      prefix_prompt_param.max_prefix_prompt_length = params.prefix_lengths.max().item<int>();
        //      prefix_prompt_param.count_length             = 1;
        // }
    }

    bool store_qkv   = false;  // 不存储回原始 QKV
    bool store_q     = true;   // 存储到独立 Q 缓冲区
    bool store_kv    = true;   // 存储到独立 K、V 缓冲区
    bool store_cache = kv_cache.has_value();
    if (hw_kernel_config_.use_aiter_pa) {
        hipStream_t stream_ = device_->getStream();
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
                                         device_->getStream()  // 必须作为最后一个参数
        );
    } else {
        DISPATCH_CUDA_FUNCTION_DATA_TYPE(
            torchDTypeToDataType(qkv.dtype()),
            invokeAddFusedQKVBiasTranspose,
            nullptr,
            q_output.data_ptr(),
            k_output.data_ptr(),
            v_output.data_ptr(),
            &prefix_prompt_param,
            qkv.data_ptr(),
            nullptr,  // qkv_buf_fp8 != nullptr ? qkv_buf_fp8->data() : nullptr,
            nullptr,  // params.common.position_ids ? params.common.position_ids->dataWithOffset<int>(decoder_batch_size
                      // * params.configs.rope_config.index_factor): nullptr,
            nullptr,  // params.configs.fuse_qkv_add_bias && params.weights.qkv_weight->bias ?
                      // params.weights.qkv_weight->bias->data() : nullptr,
            nullptr,  // params.common.padding_offset->data<int>(),
            params->cu_seqlens.data_ptr<int>(),
            batch_size,
            seq_len,  // 使用 seq_len 而不是 params->max_seq_len
            token_num,
            local_head_num,
            local_head_num_kv,
            size_per_head,
            attn_configs_.rope_config,
            attn_configs_.use_logn_attn,
            nullptr,  // scale_out_ptr,
            0,        // int8_mode,
            false,
            store_qkv,
            false,  // store_q_no_transpose
            store_q,
            store_kv,
            store_cache,
            device_->getStream());
        check_cuda_error();
    }

    // 根据 cu_seqlens 对每个 batch 超出实际长度的部分进行0填充
    // cu_seqlens 格式: [0, len1, len1+len2, len1+len2+len3, ...]
    // 每个 batch 的实际长度为: cu_seqlens[i+1] - cu_seqlens[i]
    auto cu_seqlens_cpu  = params->cu_seqlens.cpu();
    auto cu_seqlens_data = cu_seqlens_cpu.data_ptr<int>();

    for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        int actual_len = cu_seqlens_data[batch_idx + 1] - cu_seqlens_data[batch_idx];

        // 对 Q tensor 进行填充：只处理超出实际长度的部分
        if (actual_len < seq_len) {
            // q_output: {batch_size, head_num, seq_len, head_dim}
            // 将 [batch_idx, :, actual_len:, :] 设置为0
            q_output
                .index({batch_idx,
                        torch::indexing::Slice(),
                        torch::indexing::Slice(actual_len, torch::indexing::None),
                        torch::indexing::Slice()})
                .zero_();
        }

        // 对 K 和 V tensor 进行填充
        if (actual_len < seq_len_with_prefix) {
            // k_output, v_output: {batch_size, head_num_kv, seq_len_with_prefix, head_dim}
            // 将 [batch_idx, :, actual_len:, :] 设置为0
            k_output
                .index({batch_idx,
                        torch::indexing::Slice(),
                        torch::indexing::Slice(actual_len, torch::indexing::None),
                        torch::indexing::Slice()})
                .zero_();
            v_output
                .index({batch_idx,
                        torch::indexing::Slice(),
                        torch::indexing::Slice(actual_len, torch::indexing::None),
                        torch::indexing::Slice()})
                .zero_();
        }
    }

    // 返回4维格式的 Q、K、V tensor
    return std::make_tuple(q_output, k_output, v_output);
}

FusedRopeKVCacheDecodeOp::FusedRopeKVCacheDecodeOp(const GptInitParameter& gpt_init_parameter):
    FMHARocmBase(gpt_init_parameter) {}

CKAttnPtr FusedRopeKVCacheDecodeOp::prepare(torch_ext::PyAttentionInputs attn_inputs) {
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
    CKAttnPtr     attn_params;
    auto          params = device_->PrepareCKAttn(
        attn_configs_, attn_inputs.kv_block_offset, kv_cache_block_id_device, attn_inputs.sequence_lengths.size(0));

    attn_params                            = CKAttnPtr(params, (CKAttn*)params.get());
    attn_params->decode_plan               = true;
    attn_params->attn_type                 = torchDTypeToDataType(attn_inputs.dtype);
    attn_params->cu_seqlens                = cu_seqlens;
    attn_params->cu_kv_seqlens             = cu_kv_seqlens;
    attn_params->sequence_lengths          = attn_inputs.sequence_lengths;
    attn_params->kv_block_array.cache_type = attn_configs_.kv_cache_dtype;
    attn_params->input_lengths             = attn_inputs.input_lengths;

    if (attn_inputs.kv_cache_block_id_device.defined() && attn_inputs.kv_cache_block_id_device.numel() > 0) {
        attn_params->kv_cache_block_id_device = attn_inputs.kv_cache_block_id_device;
    }

    return attn_params;
}

torch::Tensor FusedRopeKVCacheDecodeOp::forward(const torch::Tensor&              qkv,
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

    // Create offset_kv_block_array similar to ROCmAttentionOp
    PrefixPromptBatchWeightsParam prefix_prompt_param;
    auto                          offset_kv_block_array = OffsetIndexedKVBlockArray(
        kv_block_array,
        (rtp_llm::KVBlockArrayForContextFMHA::DataType*)params->kv_cache_block_id_device.data_ptr(),
        kv_cache.value().k_cache_base.size(0) * layer_num_);
    prefix_prompt_param.offset_kv_block_array = offset_kv_block_array;

    // prefix_prompt_param.kv_block_array = kv_block_array;
    // if (params->prefix_lengths.defined() && params->prefix_lengths.numel() > 0) {
    //     prefix_prompt_param.d_prefix_prompt_lengths =params->prefix_lengths.data_ptr<int>();
    //     prefix_prompt_param.max_prefix_prompt_length =params->prefix_lengths.max().item<int>();
    //     prefix_prompt_param.count_length = 1;
    // }
    RTP_LLM_CHECK_WITH_INFO(params->sequence_lengths.is_pinned(), "sequence_lengths is not pinned memory");
    size_t seq_len     = 1;
    bool   store_qkv   = false;
    bool   store_q     = true;
    bool   store_kv    = false;
    bool   store_cache = kv_cache.has_value();

    if (hw_kernel_config_.use_aiter_pa) {
        // Use the offset_kv_block_array for AITER_PA path
        if (params->prefix_lengths.defined() && params->prefix_lengths.numel() > 0) {
            prefix_prompt_param.d_prefix_prompt_lengths  = params->prefix_lengths.data_ptr<int>();
            prefix_prompt_param.max_prefix_prompt_length = params->prefix_lengths.max().item<int>();
            prefix_prompt_param.count_length             = 1;
        }

        size_t seq_len = 1;

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
            /*params.common.padding_offset->data<int>(),*/ nullptr,
            /*params.common.cu_seqlens->data<int>(),*/ nullptr,
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
            device_->getStream());
    } else {
        assert(false && "not implemented");
    }

    return q_output;
}

void registerFusedRopeKVCacheOp(const py::module& m) {
    pybind11::class_<KVBlockArray>(m, "KVBlockArray").def(pybind11::init<>());
    pybind11::class_<CKAttn, std::shared_ptr<CKAttn>>(m, "CKAttn").def(pybind11::init<>());
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