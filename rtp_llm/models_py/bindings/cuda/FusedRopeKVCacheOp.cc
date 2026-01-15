#include "rtp_llm/models_py/bindings/cuda/FusedRopeKVCacheOp.h"
#include "rtp_llm/cpp/kernels/unfused_attention_kernels.h"
#include "rtp_llm/cpp/core/Dispatch.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/devices/cuda_impl/CudaFlashInfer.h"
#include "rtp_llm/cpp/devices/utils/RopeCache.h"
#include "rtp_llm/cpp/core/BufferHelper.h"
#include "rtp_llm/models_py/bindings/common/Torch_ext.h"

namespace rtp_llm {

FusedRopeKVCachePrefillOp::FusedRopeKVCachePrefillOp(const AttentionConfigs& attn_configs):
    attn_configs_(attn_configs), device_(dynamic_cast<CudaDevice*>(DeviceFactory::getDefaultDevice())) {}

TRTAttnPtr FusedRopeKVCachePrefillOp::prepare(torch_ext::PyAttentionInputs attn_inputs) {
    int       batch_size = attn_inputs.input_lengths.size(0);
    BufferPtr kv_cache_block_id_host, kv_cache_block_id_device;
    if (attn_inputs.kv_cache_block_id_host.defined() && attn_inputs.kv_cache_block_id_host.numel() > 0) {
        kv_cache_block_id_host   = torchTensor2Buffer(attn_inputs.kv_cache_block_id_host);
        kv_cache_block_id_device = torchTensor2Buffer(attn_inputs.kv_cache_block_id_device);
    }
    TRTAttnPtr attn_params;
    // TODO: should not use device to do that, we will change it later
    auto params = device_->prepareTrtAttn(attn_configs_, kv_cache_block_id_device, batch_size);
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
    return attn_params;
}

torch::Tensor FusedRopeKVCachePrefillOp::forward(const torch::Tensor&              qkv,
                                                 FMHAType                          fmha_type,
                                                 std::optional<torch_ext::KVCache> kv_cache,
                                                 const TRTAttnPtr&                 params) {
    // bool store_cache = params.common.kv_cache.has_value();
    const int local_head_num    = attn_configs_.head_num;
    const int local_head_num_kv = attn_configs_.kv_head_num;
    const int size_per_head     = attn_configs_.size_per_head;
    const int token_num         = qkv.size(0);
    const int batch_size        = params->cu_seqlens.size(0) - 1;

    // // Debug: Print QKV input before kernel execution
    // if (fmha_type == FMHAType::PAGED_TRT_V2 && token_num > 2 && local_head_num > 0) {
    //     std::cout << "\n=== FusedRopeKVCache Input QKV Debug ===" << std::endl;
    //     std::cout << "qkv input shape: " << qkv.sizes() << " [token_num, (head_num + 2*kv_head_num) * size_per_head]"
    //     << std::endl; std::cout << "fmha_type: PAGED_TRT_V2" << std::endl; std::cout << "local_head_num: " <<
    //     local_head_num << ", local_head_num_kv: " << local_head_num_kv
    //               << ", size_per_head: " << size_per_head << std::endl;

    //     // Print for first 10 tokens and first 4 heads
    //     int print_tokens = std::min(10, token_num);
    //     int print_heads = std::min(4, local_head_num);
    //     int print_dim = std::min(16, size_per_head);

    //     for (int tok = 0; tok < print_tokens; tok++) {
    //         std::cout << "\n--- Token " << tok << " (Input QKV) ---" << std::endl;
    //         for (int h = 0; h < print_heads; h++) {
    //             // Extract Q from qkv input: [token_num, (head_num + 2*kv_head_num) * size_per_head]
    //             // Q for head h is at [tok, h*size_per_head : (h+1)*size_per_head]
    //             auto qkv_token = qkv.select(0, tok);  // [(head_num + 2*kv_head_num) * size_per_head]
    //             auto qkv_q = qkv_token.slice(0, h * size_per_head, (h + 1) * size_per_head)
    //                                   .slice(0, 0, print_dim);  // [print_dim]

    //             std::cout << "  Head " << h << " Q (first " << print_dim << "): " << qkv_q << std::endl;
    //         }
    //     }
    //     std::cout << "=== End Input QKV Debug ===\n" << std::endl;
    // }
    torch::Tensor q_no_transpose_output = torch::empty({token_num, local_head_num, size_per_head},
                                                       torch::TensorOptions(qkv.dtype()).device(qkv.device()));
    torch::Tensor q_output              = torch::empty({local_head_num, token_num, size_per_head},
                                          torch::TensorOptions(qkv.dtype()).device(qkv.device()));

    torch::Tensor qkv_fp8 = torch::empty({token_num, (local_head_num + 2 * local_head_num_kv), size_per_head},
                                         torch::TensorOptions(torch::kFloat8_e4m3fn).device(qkv.device()));

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
    // not support fp8 now
    if (fmha_type == FMHAType::TRT_V2 && params->max_prefix_length > 0 && kv_cache.has_value()
        && prefix_prompt_param.kv_block_array.cache_type == KvCacheDataType::BASE) {
        fmha_type = FMHAType::PAGED_TRT_V2;
    }

    bool store_qkv =
        fmha_type != FMHAType::PAGED_TRT_V2 && fmha_type != FMHAType::NONE && fmha_type != FMHAType::FLASH_INFER;
    bool store_q_no_transpose = fmha_type == FMHAType::FLASH_INFER;
    store_q_no_transpose      = true;
    bool store_q              = fmha_type == FMHAType::PAGED_TRT_V2 || fmha_type == FMHAType::NONE;
    store_q                   = true;
    bool store_kv             = fmha_type == FMHAType::NONE;
    bool store_cache          = kv_cache.has_value();
    // bool use_qkv_fp8 =
    //     fmha_type == FMHAType::TRT_V2 && prefix_prompt_param.kv_block_array.cache_type == KvCacheDataType::FP8;

    int* padding_offset = nullptr;
    if (params->padding_offset.defined()) {
        padding_offset = params->padding_offset.data_ptr<int>();
    }
    // tmp not use qkv fp8 buffer
    bool use_qkv_fp8 = false;

    auto       rope_cache = getRopeCacheOnce(attn_configs_.rope_config, device_->initParams().max_seq_len);
    StreamType stream     = GET_CURRENT_STREAM();

    // // Force disable RoPE for debugging
    // RopeConfig disabled_rope_config = attn_configs_.rope_config;
    // disabled_rope_config.style = RopeStyle::No;  // Disable RoPE
    // std::cout << "\n!!! RoPE is DISABLED for debugging (RopeStyle::No) !!!\n" << std::endl;

    DISPATCH_CUDA_FUNCTION_DATA_TYPE(
        torchDTypeToDataType(qkv.dtype()),
        invokeAddFusedQKVBiasTranspose,
        q_no_transpose_output.data_ptr(),
        q_output.data_ptr(),
        nullptr,  // k_output.data_ptr(),
        nullptr,  // v_output.data_ptr(),
        &prefix_prompt_param,
        qkv.data_ptr(),
        use_qkv_fp8 ? qkv_fp8.data_ptr() : nullptr,
        nullptr,  // params.common.position_ids ? params.common.position_ids->dataWithOffset<int>(decoder_batch_size *
                  // params.configs.rope_config.index_factor): nullptr,
        nullptr,  // params.configs.fuse_qkv_add_bias && params.weights.qkv_weight->bias ?
                  // params.weights.qkv_weight->bias->data() : nullptr,
        padding_offset,
        params->cu_seqlens.data_ptr<int>(),
        rope_cache.used,
        checkRopeCache(attn_configs_.rope_config, rope_cache) ? rope_cache.data.data_ptr<float>() : nullptr,
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
        stream);

    // Debug: Print q_output after kernel execution (with RoPE applied)
    // if (fmha_type == FMHAType::PAGED_TRT_V2 && token_num > 2 && local_head_num > 0) {
    //     std::cout << "q_output " << q_output << std::endl;
    //     std::cout << "q_no_transpose_output " << q_no_transpose_output << std::endl;
    //     std::cout << "\n=== FusedRopeKVCache Output Q Debug (After RoPE) ===" << std::endl;
    //     std::cout << "q_output shape: " << q_output.sizes() << " [head_num, token_num, size_per_head]" << std::endl;
    //     std::cout << "q_no_transpose_output shape: " << q_no_transpose_output.sizes() << " [token_num, head_num,
    //     size_per_head]" << std::endl;

    //     // Print for first 10 tokens and first 4 heads
    //     int print_tokens = std::min(10, token_num);
    //     int print_heads = std::min(4, local_head_num);
    //     int print_dim = std::min(16, size_per_head);

    //     for (int tok = 0; tok < print_tokens; tok++) {
    //         std::cout << "\n--- Token " << tok << " (Output Q after RoPE) ---" << std::endl;
    //         for (int h = 0; h < print_heads; h++) {
    //             // Extract Q from q_output: [head_num, token_num, size_per_head]
    //             auto q_out = q_output.select(0, h).select(0, tok).slice(0, 0, print_dim);

    //             // Extract Q from q_no_transpose_output: [token_num, head_num, size_per_head]
    //             auto q_no_trans = q_no_transpose_output.select(0, tok).select(0, h).slice(0, 0, print_dim);

    //             std::cout << "  Head " << h << ":" << std::endl;
    //             std::cout << "    q_output [head, token, dim] (first " << print_dim << "): " << q_out << std::endl;
    //             std::cout << "    q_no_transpose [token, head, dim] (first " << print_dim << "): " << q_no_trans <<
    //             std::endl;
    //         }
    //     }
    //     std::cout << "=== End Output Q Debug ===\n" << std::endl;
    // }

    if (use_qkv_fp8) {
        // [token_num, (local_head_num + 2 * local_head_num_kv), size_per_head]
        return qkv_fp8;
    } else if (fmha_type == FMHAType::PAGED_TRT_V2) {
        // [local_head_num, token_num, size_per_head]
        // return q_output;
        return q_no_transpose_output;
    } else if (fmha_type == FMHAType::FLASH_INFER) {
        // [token_num, local_head_num, size_per_head]
        return q_no_transpose_output;
    } else {
        // [token_num, hidden_size]
        return qkv;
    }
}

FusedRopeKVCacheDecodeOp::FusedRopeKVCacheDecodeOp(const AttentionConfigs& attn_configs):
    attn_configs_(attn_configs), device_(dynamic_cast<CudaDevice*>(DeviceFactory::getDefaultDevice())) {}

TRTAttnPtr FusedRopeKVCacheDecodeOp::prepare(torch_ext::PyAttentionInputs attn_inputs) {
    int       batch_size = attn_inputs.sequence_lengths.size(0);
    BufferPtr kv_cache_block_id_host, kv_cache_block_id_device;
    if (attn_inputs.kv_cache_block_id_host.defined() && attn_inputs.kv_cache_block_id_host.numel() > 0) {
        kv_cache_block_id_host   = torchTensor2Buffer(attn_inputs.kv_cache_block_id_host);
        kv_cache_block_id_device = torchTensor2Buffer(attn_inputs.kv_cache_block_id_device);
    }

    TRTAttnPtr attn_params;
    auto       params =
        device_->prepareTrtAttn(attn_configs_, kv_cache_block_id_device, attn_inputs.sequence_lengths.size(0));
    RTP_LLM_CHECK_WITH_INFO(params != nullptr, "TRTAttnPtr Build Failed");
    attn_params                            = TRTAttnPtr(params, (TRTAttn*)params.get());
    attn_params->decode_plan               = true;
    attn_params->attn_type                 = torchDTypeToDataType(attn_inputs.dtype);
    attn_params->cu_seqlens                = attn_inputs.cu_seqlens;
    attn_params->cu_kv_seqlens             = attn_inputs.cu_kv_seqlens;
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

    auto rope_cache = getRopeCacheOnce(attn_configs_.rope_config, device_->initParams().max_seq_len);

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
        stream);
    return q_output;
}

void registerFusedRopeKVCacheOp(const py::module& m) {
    pybind11::class_<KVBlockArray>(m, "KVBlockArray").def(pybind11::init<>());
    pybind11::class_<TRTAttn, std::shared_ptr<TRTAttn>, rtp_llm::ParamsBase>(m, "TRTAttn").def(pybind11::init<>());
    pybind11::class_<FusedRopeKVCachePrefillOp>(m, "FusedRopeKVCachePrefillOp")
        .def(pybind11::init<const AttentionConfigs&>(), py::arg("attn_configs"))
        .def("prepare", &FusedRopeKVCachePrefillOp::prepare, py::arg("attn_inputs"))
        .def("forward",
             &FusedRopeKVCachePrefillOp::forward,
             py::arg("qkv"),
             py::arg("fmha_type"),
             py::arg("kv_cache"),
             py::arg("params"));

    pybind11::class_<FusedRopeKVCacheDecodeOp>(m, "FusedRopeKVCacheDecodeOp")
        .def(pybind11::init<const AttentionConfigs&>(), py::arg("attn_configs"))
        .def("prepare", &FusedRopeKVCacheDecodeOp::prepare, py::arg("attn_inputs"))
        .def("forward",
             &FusedRopeKVCacheDecodeOp::forward,
             py::arg("qkv"),
             py::arg("fmha_type"),
             py::arg("kv_cache"),
             py::arg("params"));
}

}  // namespace rtp_llm
