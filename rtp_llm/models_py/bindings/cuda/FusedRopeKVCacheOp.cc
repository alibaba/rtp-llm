#include "rtp_llm/models_py/bindings/cuda/FusedRopeKVCacheOp.h"
#include "rtp_llm/cpp/kernels/unfused_attention_kernels.h"
#include "rtp_llm/cpp/core/Dispatch.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/devices/cuda_impl/CudaFlashInfer.h"
#include "rtp_llm/cpp/devices/utils/RopeCache.h"
#include "rtp_llm/cpp/core/BufferHelper.h"
#include "rtp_llm/models_py/bindings/common/Torch_ext.h"
#include <iostream>

namespace rtp_llm {

FusedRopeKVCachePrefillOp::FusedRopeKVCachePrefillOp(const AttentionConfigs& attn_configs):
    attn_configs_(attn_configs), device_(dynamic_cast<CudaDevice*>(DeviceFactory::getDefaultDevice())) {}

TRTAttnPtr FusedRopeKVCachePrefillOp::prepare(torch_ext::PyAttentionInputs attn_inputs) {
    int       batch_size = attn_inputs.input_lengths.size(0);
    BufferPtr kv_cache_block_id_host, kv_cache_block_id_device;
    if (attn_inputs.kv_cache_block_id_host.defined() && attn_inputs.kv_cache_block_id_host.numel() > 0) {
        kv_cache_block_id_host   = torchTensor2Buffer(attn_inputs.kv_cache_block_id_host);
        kv_cache_block_id_device = torchTensor2Buffer(attn_inputs.kv_cache_block_id_device);
        // Save for later use in forward
        // kv_cache_block_id_host_ = attn_inputs.kv_cache_block_id_host;
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

    bool store_qkv = fmha_type != FMHAType::PY_FLASHINFER_PREFILL_PAGED && fmha_type != FMHAType::PAGED_TRT_V2
                     && fmha_type != FMHAType::NONE && fmha_type != FMHAType::FLASH_INFER;
    bool store_q_no_transpose = fmha_type == FMHAType::FLASH_INFER || fmha_type == FMHAType::PAGED_TRT_V2
                                || fmha_type == FMHAType::PY_FLASHINFER_PREFILL_PAGED;

    bool store_q = fmha_type == FMHAType::NONE;

    bool store_kv    = fmha_type == FMHAType::NONE;
    bool store_cache = kv_cache.has_value();
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

    // // ========== DEBUG: Print key flags and fmha_type ==========
    // std::cout << "\n[FusedRopeKVCachePrefillOp] fmha_type=" << static_cast<int>(fmha_type)
    //           << " | store_qkv=" << store_qkv
    //           << " | store_q_no_transpose=" << store_q_no_transpose
    //           << " | store_q=" << store_q
    //           << " | store_kv=" << store_kv
    //           << " | store_cache=" << store_cache
    //           << " | use_qkv_fp8=" << use_qkv_fp8
    //           << " | use_paged=" << (fmha_type == FMHAType::PAGED_TRT_V2 || fmha_type ==
    //           FMHAType::PY_FLASHINFER_PREFILL_PAGED)
    //           << std::endl;

    // // ========== DEBUG: Print QKV input statistics BEFORE kernel ==========
    // std::cout << "\n  📊 QKV Input Statistics (BEFORE kernel):" << std::endl;
    // std::cout << "    qkv shape: [" << qkv.size(0) << ", " << qkv.size(1) << "]" << std::endl;
    // std::cout << "    qkv max: " << qkv.max().item<float>() << std::endl;
    // std::cout << "    qkv min: " << qkv.min().item<float>() << std::endl;
    // std::cout << "    qkv mean: " << qkv.mean().item<float>() << std::endl;
    // std::cout << "    qkv std: " << qkv.std().item<float>() << std::endl;

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
        fmha_type == FMHAType::PAGED_TRT_V2 || fmha_type == FMHAType::PY_FLASHINFER_PREFILL_PAGED,
        store_qkv,
        store_q_no_transpose,
        store_q,
        store_kv,
        store_cache,
        stream);

    // std::cout << "  ✅ invokeAddFusedQKVBiasTranspose completed" << std::endl;

    // // ========== DEBUG: Print Q, K, V output statistics AFTER kernel ==========
    // std::cout << "\n  📊 Q/K/V Output Statistics (AFTER kernel):" << std::endl;

    // if (store_q_no_transpose) {
    //     std::cout << "    q_no_transpose_output shape: [" << q_no_transpose_output.size(0) << ", "
    //               << q_no_transpose_output.size(1) << ", " << q_no_transpose_output.size(2) << "]" << std::endl;
    //     std::cout << "    Q (no transpose) max: " << q_no_transpose_output.max().item<float>() << std::endl;
    //     std::cout << "    Q (no transpose) min: " << q_no_transpose_output.min().item<float>() << std::endl;
    //     std::cout << "    Q (no transpose) mean: " << q_no_transpose_output.mean().item<float>() << std::endl;
    //     std::cout << "    Q (no transpose) std: " << q_no_transpose_output.std().item<float>() << std::endl;
    // }

    // if (store_q) {
    //     std::cout << "    q_output shape: [" << q_output.size(0) << ", "
    //               << q_output.size(1) << ", " << q_output.size(2) << "]" << std::endl;
    //     std::cout << "    Q (transposed) max: " << q_output.max().item<float>() << std::endl;
    //     std::cout << "    Q (transposed) min: " << q_output.min().item<float>() << std::endl;
    //     std::cout << "    Q (transposed) mean: " << q_output.mean().item<float>() << std::endl;
    //     std::cout << "    Q (transposed) std: " << q_output.std().item<float>() << std::endl;
    // }

    // if (store_cache && kv_cache.has_value()) {
    //     auto kv_cache_tensor = kv_cache.value().kv_cache_base;
    //     std::cout << "    KV cache tensor shape: [";
    //     for (int64_t i = 0; i < kv_cache_tensor.dim(); ++i) {
    //         if (i > 0) std::cout << ", ";
    //         std::cout << kv_cache_tensor.size(i);
    //     }
    //     std::cout << "]" << std::endl;

    //     // Check used blocks if we have block indices
    //     if (kv_cache_block_id_host_.defined() && kv_cache_block_id_host_.numel() > 0) {
    //         std::cout << "    📋 Checking used blocks from kv_cache_block_id_host_..." << std::endl;

    //         // Get unique block IDs from kv_cache_block_id_host_ (this contains the actual block indices used)
    //         auto block_ids_flat = kv_cache_block_id_host_.flatten();
    //         auto valid_mask = block_ids_flat >= 0;  // Filter out -1 (invalid blocks)
    //         auto valid_block_ids = block_ids_flat.masked_select(valid_mask);

    //         if (valid_block_ids.numel() > 0) {
    //             std::cout << "      Valid block IDs count: " << valid_block_ids.numel() << std::endl;
    //             std::cout << "      Valid block IDs (all): " << valid_block_ids << std::endl;

    //             // Move valid_block_ids to GPU for index_select
    //             auto valid_block_ids_gpu = valid_block_ids.to(kv_cache_tensor.device());
    //             // Index the used blocks
    //             auto used_kv_blocks = kv_cache_tensor.index_select(0, valid_block_ids_gpu);
    //             std::cout << "      Used KV blocks shape: [";
    //             for (int64_t i = 0; i < used_kv_blocks.dim(); ++i) {
    //                 if (i > 0) std::cout << ", ";
    //                 std::cout << used_kv_blocks.size(i);
    //             }
    //             std::cout << "]" << std::endl;
    //             std::cout << "      Used KV cache max: " << used_kv_blocks.max().item<float>() << std::endl;
    //             std::cout << "      Used KV cache min: " << used_kv_blocks.min().item<float>() << std::endl;
    //             std::cout << "      Used KV cache mean: " << used_kv_blocks.mean().item<float>() << std::endl;
    //             std::cout << "      Used KV cache std: " << used_kv_blocks.std().item<float>() << std::endl;

    //             // Check for NaN and Inf in used blocks only
    //             bool has_nan = torch::isnan(used_kv_blocks).any().item<bool>();
    //             bool has_inf = torch::isinf(used_kv_blocks).any().item<bool>();
    //             if (has_nan || has_inf) {
    //                 std::cout << "      ⚠️  WARNING: Used KV blocks contain NaN: " << has_nan << ", Inf: " << has_inf
    //                 << std::endl; if (has_nan) {
    //                     int64_t nan_count = torch::isnan(used_kv_blocks).sum().item<int64_t>();
    //                     std::cout << "        NaN count: " << nan_count << " / " << used_kv_blocks.numel() <<
    //                     std::endl;
    //                 }
    //                 if (has_inf) {
    //                     int64_t inf_count = torch::isinf(used_kv_blocks).sum().item<int64_t>();
    //                     std::cout << "        Inf count: " << inf_count << " / " << used_kv_blocks.numel() <<
    //                     std::endl;
    //                 }
    //             } else {
    //                 std::cout << "      ✅ No NaN or Inf in used KV blocks" << std::endl;
    //             }
    //         }
    //     } else {
    //         std::cout << "    KV cache max: " << kv_cache_tensor.max().item<float>() << std::endl;
    //         std::cout << "    KV cache min: " << kv_cache_tensor.min().item<float>() << std::endl;
    //         std::cout << "    KV cache mean: " << kv_cache_tensor.mean().item<float>() << std::endl;
    //         std::cout << "    KV cache std: " << kv_cache_tensor.std().item<float>() << std::endl;
    //     }
    // }
    // std::cout << "  ========================================================\n" << std::endl;

    if (use_qkv_fp8) {
        // [token_num, (local_head_num + 2 * local_head_num_kv), size_per_head]
        return qkv_fp8;
    } else if (fmha_type == FMHAType::PAGED_TRT_V2 || fmha_type == FMHAType::FLASH_INFER
               || fmha_type == FMHAType::PY_FLASHINFER_PREFILL_PAGED) {
        // // Print last 10 tokens, first 10 dimensions each
        // std::cout << "\n  🔍 Q Output sample (last 10 tokens, first 10 dims each):" << std::endl;
        // int num_tokens_to_print = std::min(10, static_cast<int>(q_no_transpose_output.size(0)));
        // int start_idx = q_no_transpose_output.size(0) - num_tokens_to_print;
        // for (int i = start_idx; i < q_no_transpose_output.size(0); ++i) {
        //     std::cout << "    Token " << i << ": [";
        //     auto token_data = q_no_transpose_output[i].flatten();
        //     for (int j = 0; j < std::min(10, static_cast<int>(token_data.size(0))); ++j) {
        //         if (j > 0) std::cout << ", ";
        //         std::cout << token_data[j].item<float>();
        //     }
        //     std::cout << "]" << std::endl;
        // }

        // // Check for NaN and Inf
        // bool has_nan = torch::isnan(q_no_transpose_output).any().item<bool>();
        // bool has_inf = torch::isinf(q_no_transpose_output).any().item<bool>();
        // if (has_nan || has_inf) {
        //     std::cout << "\n  ⚠️  WARNING: Q output contains NaN: " << has_nan << ", Inf: " << has_inf << std::endl;
        //     if (has_nan) {
        //         int64_t nan_count = torch::isnan(q_no_transpose_output).sum().item<int64_t>();
        //         std::cout << "    NaN count: " << nan_count << " / " << q_no_transpose_output.numel() << std::endl;
        //     }
        //     if (has_inf) {
        //         int64_t inf_count = torch::isinf(q_no_transpose_output).sum().item<int64_t>();
        //         std::cout << "    Inf count: " << inf_count << " / " << q_no_transpose_output.numel() << std::endl;
        //     }
        // } else {
        //     std::cout << "\n  ✅ No NaN or Inf in Q output" << std::endl;
        // }
        // std::cout << "  ========================================================\n" << std::endl;

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
    pybind11::class_<KVBlockArray>(m, "KVBlockArray")
        .def(pybind11::init<>())
        .def(
            "__cpp_ptr__",
            [](KVBlockArray& self) { return reinterpret_cast<uintptr_t>(&self); },
            "Get C++ object pointer address");
    pybind11::class_<TRTAttn, std::shared_ptr<TRTAttn>, rtp_llm::ParamsBase>(m, "TRTAttn")
        .def(pybind11::init<>())
        .def_readwrite("kv_cache_offset", &TRTAttn::kv_cache_offset)
        .def(
            "__cpp_ptr__",
            [](TRTAttn& self) { return reinterpret_cast<uintptr_t>(&self); },
            "Get C++ object pointer address");
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
