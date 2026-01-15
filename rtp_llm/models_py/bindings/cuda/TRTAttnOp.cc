#include "rtp_llm/models_py/bindings/cuda/TRTAttnOp.h"
#include "rtp_llm/cpp/cuda/cufmha/TrtV2FmhaRunner.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/models_py/bindings/common/Torch_ext.h"
#include "rtp_llm/cpp/devices/DeviceFactory.h"
#include "rtp_llm/cpp/devices/cuda_impl/CudaDevice.h"
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

using namespace torch_ext;

namespace rtp_llm {

TRTPrefillOpBase::TRTPrefillOpBase(const AttentionConfigs& attn_configs): attn_configs_(attn_configs) {}

bool TRTPrefillOpBase::support(torch_ext::PyAttentionInputs attn_inputs) {
    // FMHAConfig check will be done in Python layer
    return attn_configs_.kv_cache_dtype != KvCacheDataType::INT8;
}

ParamsBasePtr TRTPrefillOpBase::prepare(torch_ext::PyAttentionInputs attn_inputs) {
    static_scale_        = torch::ones({1}, torch::TensorOptions(torch::kFloat32).device(torch::kCUDA));
    int       batch_size = attn_inputs.input_lengths.size(0);
    BufferPtr kv_cache_block_id_host, kv_cache_block_id_device;
    if (attn_inputs.kv_cache_block_id_host.defined() && attn_inputs.kv_cache_block_id_host.numel() > 0) {
        kv_cache_block_id_host   = torchTensor2Buffer(attn_inputs.kv_cache_block_id_host);
        kv_cache_block_id_device = torchTensor2Buffer(attn_inputs.kv_cache_block_id_device);
    }

    TRTAttnPtr attn_params;
    auto       run_stream   = GET_CURRENT_STREAM();
    bool       use_fp8_fmha = attn_configs_.kv_cache_dtype == KvCacheDataType::FP8;
    auto       params       = prepareTrtAttnParams(attn_configs_,
                                       kv_cache_block_id_device,
                                       attn_inputs.input_lengths.size(0),
                                       use_fp8_fmha,
                                       run_stream,
                                       false);  // enable_paged_trt_fmha check is done in Python layer
    if (params) {
        attn_params = TRTAttnPtr(params, (TRTAttn*)params.get());
    } else {
        attn_params = std::make_shared<TRTAttn>();
    }
    attn_params->attn_type               = torchDTypeToDataType(attn_inputs.dtype);
    attn_params->cu_seqlens              = attn_inputs.cu_seqlens;
    attn_params->cu_kv_seqlens           = attn_inputs.cu_kv_seqlens;
    attn_params->max_seq_len             = attn_inputs.input_lengths.max().item<int32_t>();
    attn_params->max_prefix_length       = attn_inputs.prefix_lengths.max().item<int32_t>();
    attn_params->context_total_kv_length = attn_inputs.context_total_kv_length;
    attn_params->input_lengths           = attn_inputs.input_lengths;

    // 创建 TRT V2 FMHA Runner
    DataType attn_dtype = use_fp8_fmha ? DataType::TYPE_FP8_E4M3 : torchDTypeToDataType(attn_inputs.dtype);

    if (!trt_v2_runner_) {
        auto runner_config = TrtV2FmhaRunnerConfig::fromAttentionConfigs(attn_configs_);
        trt_v2_runner_ =
            std::make_shared<TrtV2FmhaRunner>(runner_config, attn_dtype, attn_inputs.is_s_padded, run_stream);
    }

    return ParamsBasePtr(attn_params);
}

bool TRTPagedPrefillOp::support(torch_ext::PyAttentionInputs attn_inputs) {
    std::cout << "\n=== TRTPagedPrefillOp::support Debug Info ===" << std::endl;

    // Print input_lengths
    if (attn_inputs.input_lengths.defined()) {
        std::cout << "input_lengths shape: " << attn_inputs.input_lengths.sizes() << std::endl;
        std::cout << "input_lengths values: " << attn_inputs.input_lengths << std::endl;
    } else {
        std::cout << "input_lengths: undefined" << std::endl;
    }

    // Print prefix_lengths
    if (attn_inputs.prefix_lengths.defined()) {
        std::cout << "prefix_lengths shape: " << attn_inputs.prefix_lengths.sizes() << std::endl;
        std::cout << "prefix_lengths values: " << attn_inputs.prefix_lengths << std::endl;
    } else {
        std::cout << "prefix_lengths: undefined" << std::endl;
    }

    // Print kv_cache_block_id_device (use host version to avoid GPU sync)
    if (attn_inputs.kv_cache_block_id_host.defined()) {
        std::cout << "kv_cache_block_id_host shape: " << attn_inputs.kv_cache_block_id_host.sizes() << std::endl;
        std::cout << "kv_cache_block_id_host dtype: " << attn_inputs.kv_cache_block_id_host.dtype() << std::endl;
        // Print first few values from host tensor
        if (attn_inputs.kv_cache_block_id_host.numel() > 0) {
            int print_size = std::min(20, (int)attn_inputs.kv_cache_block_id_host.numel());
            std::cout << "kv_cache_block_id_host (first " << print_size
                      << " values): " << attn_inputs.kv_cache_block_id_host.flatten().slice(0, 0, print_size)
                      << std::endl;
        }
    } else if (attn_inputs.kv_cache_block_id_device.defined()) {
        // Fallback: use device tensor but copy to CPU first
        std::cout << "kv_cache_block_id_device shape: " << attn_inputs.kv_cache_block_id_device.sizes() << std::endl;
        std::cout << "kv_cache_block_id_device dtype: " << attn_inputs.kv_cache_block_id_device.dtype() << std::endl;
        std::cout << "kv_cache_block_id_device device: " << attn_inputs.kv_cache_block_id_device.device() << std::endl;
        // Copy to CPU for printing
        if (attn_inputs.kv_cache_block_id_device.numel() > 0) {
            auto block_id_cpu = attn_inputs.kv_cache_block_id_device.cpu();
            int  print_size   = std::min(20, (int)block_id_cpu.numel());
            std::cout << "kv_cache_block_id (first " << print_size
                      << " values, copied to CPU): " << block_id_cpu.flatten().slice(0, 0, print_size) << std::endl;
        }
    } else {
        std::cout << "kv_cache_block_id: undefined (both host and device)" << std::endl;
    }

    // Print other info
    std::cout << "is_s_padded: " << attn_inputs.is_s_padded << std::endl;
    std::cout << "dtype: " << attn_inputs.dtype << std::endl;
    std::cout << "kv_cache_dtype (config): " << static_cast<int>(attn_configs_.kv_cache_dtype) << std::endl;

    std::cout << "=== End Support Debug Info ===\n" << std::endl;

    bool has_prefix =
        attn_inputs.prefix_lengths.defined() && torch::any(attn_inputs.prefix_lengths.reshape({-1})).item<bool>();

    // FMHAConfig check is done in Python layer
    if (!has_prefix || attn_configs_.kv_cache_dtype == KvCacheDataType::INT8) {
        return false;
    }

    // 创建 runner 并检查是否支持
    DataType attn_dtype = attn_configs_.kv_cache_dtype == KvCacheDataType::FP8 ?
                              DataType::TYPE_FP8_E4M3 :
                              torchDTypeToDataType(attn_inputs.dtype);

    auto run_stream   = GET_CURRENT_STREAM();
    bool use_fp8_fmha = attn_configs_.kv_cache_dtype == KvCacheDataType::FP8;
    if (!trt_v2_runner_) {
        auto runner_config = TrtV2FmhaRunnerConfig::fromAttentionConfigs(attn_configs_);
        trt_v2_runner_.reset(new TrtV2FmhaRunner(runner_config, attn_dtype, attn_inputs.is_s_padded, run_stream));
    }

    return trt_v2_runner_->trtV2PagedFmhaSupported();
}

torch::Tensor TRTPagedPrefillOp::forward(const torch::Tensor&              input,
                                         std::optional<torch_ext::KVCache> kv_cache,
                                         const TRTAttnPtr&                 params) {
    // std::cout << "\n=== TRTPagedPrefillOp::forward Input Info ===" << std::endl;

    // // Print input tensor info
    // std::cout << "Input shape: " << input.sizes() << std::endl;
    // std::cout << "Input dtype: " << input.dtype() << std::endl;
    // std::cout << "Input device: " << input.device() << std::endl;

    // // Print params info
    // if (params->input_lengths.defined()) {
    //     std::cout << "params->input_lengths: " << params->input_lengths << std::endl;
    // }
    // if (params->prefix_lengths.defined()) {
    //     std::cout << "params->prefix_lengths: " << params->prefix_lengths << std::endl;
    // }
    // std::cout << "params->max_seq_len: " << params->max_seq_len << std::endl;
    // std::cout << "params->max_prefix_length: " << params->max_prefix_length << std::endl;
    // if (params->cu_seqlens.defined()) {
    //     std::cout << "params->cu_seqlens: " << params->cu_seqlens << std::endl;
    // }
    // if (params->cu_kv_seqlens.defined()) {
    //     std::cout << "params->cu_kv_seqlens: " << params->cu_kv_seqlens << std::endl;
    // }
    // std::cout << "params->context_total_kv_length: " << params->context_total_kv_length << std::endl;

    // // Print KV cache info
    // if (kv_cache.has_value()) {
    //     std::cout << "kv_cache_base shape: " << kv_cache.value().kv_cache_base.sizes() << std::endl;
    //     std::cout << "kv_cache_base dtype: " << kv_cache.value().kv_cache_base.dtype() << std::endl;
    // }

    // std::cout << "=== End Forward Input Info ===\n" << std::endl;

    KVBlockArray kv_block_array;
    if (!kv_cache.has_value()) {
        throw std::runtime_error("kv_cache must be provided for trt v2 fmha paged");
    }
    kv_block_array                 = params->kv_block_array;
    kv_block_array.mPrimaryPoolPtr = kv_cache.value().kv_cache_base.data_ptr();
    if (kv_cache.value().kv_scale_base.defined() && kv_cache.value().kv_scale_base.numel() > 0) {
        kv_block_array.scale = kv_cache.value().kv_scale_base.data_ptr();
    }

    // // Print KVBlockArray info
    // std::cout << "\n=== KVBlockArray Info ===" << std::endl;
    // std::cout << "mMaxSeqs: " << kv_block_array.mMaxSeqs << std::endl;
    // std::cout << "mMaxBlocksPerSeq: " << kv_block_array.mMaxBlocksPerSeq << std::endl;
    // std::cout << "mTokensPerBlock: " << kv_block_array.mTokensPerBlock << std::endl;
    // std::cout << "mTokensPerBlockLog2: " << kv_block_array.mTokensPerBlockLog2 << std::endl;
    // std::cout << "mBytesPerBlock: " << kv_block_array.mBytesPerBlock << std::endl;
    // std::cout << "cache_type: " << static_cast<int>(kv_block_array.cache_type) << std::endl;
    // std::cout << "=== End KVBlockArray Info ===\n" << std::endl;

    const int local_head_num = attn_configs_.head_num;
    const int size_per_head  = attn_configs_.size_per_head;

    // TRT kernel expects Q in [token, head, dim] layout
    // input shape: [token_num, head_num, head_dim]
    const int            token_num  = input.size(0);  // Changed from size(1) to size(0)
    const int            batch_size = params->input_lengths.size(0);
    torch::TensorOptions options    = torch::TensorOptions(input.dtype()).device(input.device());

    // // Debug: Print token 2, head 0 for Q, K, V and compute manual attention
    // if (token_num > 2 && kv_cache.has_value()) {
    //     std::cout << "\n=== Debug: Token 2, Head 0 ===" << std::endl;
    //     std::cout << "Input (Q) shape: " << input.sizes() << std::endl;

    //     // Q: [head_num, token_num, head_dim]
    //     // Select head 0, then token 2
    //     auto q_token2_head0 = input.select(0, 0).select(0, 2);  // [head_dim]
    //     std::cout << "Q (token 2, head 0, first 16 values): " << q_token2_head0.slice(0, 0, 16) << std::endl;

    //     // For K and V in paged KV cache
    //     auto kv_cache_tensor = kv_cache.value().kv_cache_base;
    //     std::cout << "KV Cache shape: " << kv_cache_tensor.sizes() << std::endl;
    //     std::cout << "tokens_per_block: " << kv_block_array.mTokensPerBlock << std::endl;

    //     // Calculate which block contains token 2
    //     int token_idx = 2;
    //     int block_idx = token_idx / kv_block_array.mTokensPerBlock;
    //     int token_in_block = token_idx % kv_block_array.mTokensPerBlock;
    //     std::cout << "Token 2 -> block " << block_idx << ", position " << token_in_block << std::endl;

    //     try {
    //         // Extract K and V for token 2, head 0
    //         // KV cache layout: [num_blocks, 2, num_kv_heads, tokens_per_block, head_dim]
    //         if (kv_cache_tensor.dim() == 5 && kv_cache_tensor.size(1) == 2) {
    //             auto k_token2_head0 = kv_cache_tensor.select(0, block_idx)  // select block
    //                                                   .select(0, 0)          // select K
    //                                                   .select(0, 0)          // select head 0
    //                                                   .select(0, token_in_block);
    //             auto v_token2_head0 = kv_cache_tensor.select(0, block_idx)
    //                                                   .select(0, 1)          // select V
    //                                                   .select(0, 0)
    //                                                   .select(0, token_in_block);
    //             std::cout << "K (token 2, head 0, first 16 values): " << k_token2_head0.slice(0, 0, 16) << std::endl;
    //             std::cout << "V (token 2, head 0, first 16 values): " << v_token2_head0.slice(0, 0, 16) << std::endl;

    //             // Manual attention computation for token 2, head 0
    //             std::cout << "\n--- Manual Attention Computation ---" << std::endl;

    //             // Get prefix_lengths and input_lengths for the first batch
    //             int prefix_len = 0;
    //             int input_len = 0;
    //             if (params->prefix_lengths.defined() && params->prefix_lengths.numel() > 0) {
    //                 prefix_len = params->prefix_lengths[0].item<int>();
    //             }
    //             if (params->input_lengths.defined() && params->input_lengths.numel() > 0) {
    //                 input_len = params->input_lengths[0].item<int>();
    //             }
    //             std::cout << "prefix_len (batch 0): " << prefix_len << std::endl;
    //             std::cout << "input_len (batch 0): " << input_len << std::endl;

    //             // Calculate total sequence length up to token 2
    //             // Total: prefix_length + (token_idx + 1)
    //             int total_kv_length = prefix_len + token_idx + 1;  // prefix + tokens 0,1,2
    //             std::cout << "Total KV length to attend: " << total_kv_length << " (prefix: "
    //                       << prefix_len << " + current: " << (token_idx + 1) << ")" << std::endl;

    //             // Note: In paged KV cache, logical block index should be mapped to physical block index
    //             // via kv_cache_block_id. For this debug code, we assume sequential allocation.
    //             // In production, the kernel uses kv_block_array.data to get the actual mapping.
    //             std::cout << "Note: Using simplified assumption (logical_block_id == physical_block_id)" <<
    //             std::endl; std::cout << "      In production, physical block ID should be read from
    //             kv_cache_block_id_device" << std::endl;

    //             // Collect all K and V for head 0 using block IDs
    //             std::vector<torch::Tensor> k_list, v_list;

    //             // KV cache layout: [num_blocks, 2, num_kv_heads, tokens_per_block, head_dim]
    //             int tokens_per_block = kv_block_array.mTokensPerBlock;

    //             std::cout << "Extracting K,V from paged cache:" << std::endl;

    //             for (int pos = 0; pos < total_kv_length; pos++) {
    //                 // Calculate logical block and position within block
    //                 int logical_blk_idx = pos / tokens_per_block;
    //                 int pos_in_blk = pos % tokens_per_block;

    //                 // TODO: Should use kv_cache_block_id to map logical -> physical block
    //                 // For now, assume sequential allocation (physical_blk = logical_blk)
    //                 int physical_blk_idx = logical_blk_idx;  // Simplified assumption

    //                 if (pos < 5 || pos >= total_kv_length - 2) {
    //                     std::cout << "  pos " << pos << " -> logical_block " << logical_blk_idx
    //                               << ", physical_block " << physical_blk_idx
    //                               << ", pos_in_block " << pos_in_blk << std::endl;
    //                 } else if (pos == 5) {
    //                     std::cout << "  ..." << std::endl;
    //                 }

    //                 // Extract K and V for this position, head 0
    //                 auto k_pos = kv_cache_tensor.select(0, physical_blk_idx)  // select physical block
    //                                             .select(0, 0)  // K
    //                                             .select(0, 0)  // head 0
    //                                             .select(0, pos_in_blk);  // [head_dim]
    //                 auto v_pos = kv_cache_tensor.select(0, physical_blk_idx)
    //                                             .select(0, 1)  // V
    //                                             .select(0, 0)  // head 0
    //                                             .select(0, pos_in_blk);  // [head_dim]

    //                 k_list.push_back(k_pos);
    //                 v_list.push_back(v_pos);
    //             }

    //             // Stack to [total_kv_length, head_dim]
    //             auto k_head0_all = torch::stack(k_list, 0);  // [total_kv_length, head_dim]
    //             auto v_head0_all = torch::stack(v_list, 0);  // [total_kv_length, head_dim]

    //             std::cout << "K shape for attention: " << k_head0_all.sizes() << std::endl;
    //             std::cout << "V shape for attention: " << v_head0_all.sizes() << std::endl;

    //             // Q for token 2: [head_dim] -> reshape to [1, head_dim] for matmul
    //             auto q_for_attn = q_token2_head0.unsqueeze(0);  // [1, head_dim]

    //             // Compute attention scores: Q @ K^T / sqrt(d)
    //             float scale = 1.0f / std::sqrt(static_cast<float>(size_per_head));
    //             auto scores = torch::matmul(q_for_attn, k_head0_all.transpose(0, 1)) * scale;  // [1,
    //             total_kv_length] std::cout << "Attention scores shape: " << scores.sizes() << std::endl; std::cout <<
    //             "Attention scores (first 10): " << scores.slice(1, 0, std::min(10, (int)scores.size(1))) <<
    //             std::endl;

    //             // Apply softmax
    //             auto attn_weights = torch::softmax(scores, -1);  // [1, total_kv_length]
    //             std::cout << "Attention weights (first 10): " << attn_weights.slice(1, 0, std::min(10,
    //             (int)attn_weights.size(1))) << std::endl;

    //             // Compute output: weights @ V
    //             auto manual_output = torch::matmul(attn_weights, v_head0_all);  // [1, head_dim]
    //             manual_output = manual_output.squeeze(0);  // [head_dim]
    //             std::cout << "Manual attention output (first 16 values): " << manual_output.slice(0, 0, 16) <<
    //             std::endl; std::cout << "--- End Manual Attention ---\n" << std::endl;

    //         } else {
    //             std::cout << "KV cache layout not recognized, manual extraction needed" << std::endl;
    //             std::cout << "Cache dimensions: " << kv_cache_tensor.dim() << std::endl;
    //         }
    //     } catch (const std::exception& e) {
    //         std::cout << "Error in debug computation: " << e.what() << std::endl;
    //     }
    //     std::cout << "=== End Debug ===\n" << std::endl;
    // }

    torch::Tensor output        = torch::zeros({token_num, local_head_num * size_per_head}, options);
    torch::Tensor tiled_counter = torch::zeros({1}, torch::TensorOptions(torch::kUInt32).device(input.device()));
    bool          use_fp8_fmha  = kv_block_array.cache_type == KvCacheDataType::FP8;
    float*        attention_output_orig_quant_scale = use_fp8_fmha ? static_scale_.data_ptr<float>() : nullptr;

    trt_v2_runner_->runTrtV2FmhaPaged(input.data_ptr(),
                                      params->cu_seqlens.data_ptr(),
                                      params->cu_kv_seqlens.data_ptr(),
                                      output.data_ptr(),
                                      reinterpret_cast<uint32_t*>(tiled_counter.data_ptr()),
                                      attention_output_orig_quant_scale,
                                      batch_size,  // batch_size,
                                      params->max_seq_len,
                                      params->max_seq_len + params->max_prefix_length,  // seq_len_with_prefix,
                                      token_num,
                                      params->context_total_kv_length,  // token_num_kv,
                                      kv_block_array);

    // Debug: Print output for token 2, head 0
    // if (token_num > 2) {
    //     std::cout << "\n=== Debug: Output Token 2, Head 0 ===" << std::endl;
    //     std::cout << "Output shape: " << output.sizes() << std::endl;

    //     // Output: [token_num, head_num * head_dim]
    //     // Extract token 2, head 0's data: [0:head_dim]
    //     auto output_token2 = output.select(0, 2);  // [head_num * head_dim]
    //     auto output_token2_head0 = output_token2.slice(0, 0, size_per_head);  // [head_dim]
    //     std::cout << "Output (token 2, head 0, first 16 values): " << output_token2_head0.slice(0, 0, 16) <<
    //     std::endl; std::cout << "=== End Debug Output ===\n" << std::endl;

    //     // Stop service after printing all debug info
    //     std::cout << "\n!!! Debug complete - stopping service !!!\n" << std::endl;
    //     std::cout.flush();  // Ensure all output is written before throwing exception
    //     throw std::runtime_error("Debug complete: all Q, K, V, manual attention, and output printed. Service stopped
    //     intentionally.");
    // }

    return output;
}

bool TRTNormalPrefillOp::support(torch_ext::PyAttentionInputs attn_inputs) {
    bool has_prefix =
        attn_inputs.prefix_lengths.defined() && torch::any(attn_inputs.prefix_lengths.reshape({-1})).item<bool>();

    // FMHAConfig check is done in Python layer
    if (has_prefix || attn_configs_.kv_cache_dtype == KvCacheDataType::INT8) {
        return false;
    }

    auto     run_stream   = GET_CURRENT_STREAM();
    bool     use_fp8_fmha = attn_configs_.kv_cache_dtype == KvCacheDataType::FP8;
    DataType attn_dtype   = use_fp8_fmha ? DataType::TYPE_FP8_E4M3 : torchDTypeToDataType(attn_inputs.dtype);

    if (!trt_v2_runner_) {
        auto runner_config = TrtV2FmhaRunnerConfig::fromAttentionConfigs(attn_configs_);
        trt_v2_runner_.reset(new TrtV2FmhaRunner(runner_config, attn_dtype, attn_inputs.is_s_padded, run_stream));
    }

    return trt_v2_runner_->trtV2FmhaSupported();
}

torch::Tensor TRTNormalPrefillOp::forward(const torch::Tensor&              input,
                                          std::optional<torch_ext::KVCache> kv_cache,
                                          const TRTAttnPtr&                 params) {
    KVBlockArray kv_block_array;
    if (kv_cache.has_value()) {
        kv_block_array                 = params->kv_block_array;
        kv_block_array.mPrimaryPoolPtr = kv_cache.value().kv_cache_base.data_ptr();
        if (kv_cache.value().kv_scale_base.defined() && kv_cache.value().kv_scale_base.numel() > 0) {
            kv_block_array.scale = kv_cache.value().kv_scale_base.data_ptr();
        }
    }

    const int local_head_num = attn_configs_.head_num;
    const int size_per_head  = attn_configs_.size_per_head;
    const int token_num      = input.size(0);
    const int batch_size     = params->input_lengths.size(0);
    auto*     device         = dynamic_cast<CudaDevice*>(DeviceFactory::getDefaultDevice());
    const int max_token_num  = device->initParams().runtime_config.fifo_scheduler_config.max_context_batch_size
                              * device->initParams().max_seq_len;
    torch::TensorOptions options = torch::TensorOptions(input.dtype()).device(input.device());

    static torch::Tensor static_output = torch::zeros({max_token_num, local_head_num * size_per_head}, options);
    torch::Tensor        output        = static_output.slice(0, 0, token_num);
    torch::Tensor        tiled_counter = torch::zeros({1}, torch::TensorOptions(torch::kUInt32).device(input.device()));
    bool                 use_fp8_fmha  = kv_block_array.cache_type == KvCacheDataType::FP8;
    float*               attention_output_orig_quant_scale = use_fp8_fmha ? static_scale_.data_ptr<float>() : nullptr;

    torch::Tensor tmp_fmha_input, tmp_fmha_output;
    void*         fmha_input_ptr  = input.data_ptr();
    void*         fmha_output_ptr = output.data_ptr();
    RTP_LLM_CHECK_WITH_INFO(fmha_input_ptr, "fmha_input_ptr must be provided for trt v2 fmha");

    if (use_fp8_fmha) {
        tmp_fmha_input  = input.to(torch::kFloat8_e4m3fn);
        tmp_fmha_output = output.to(torch::kFloat8_e4m3fn);
        fmha_input_ptr  = tmp_fmha_input.data_ptr();
        fmha_output_ptr = tmp_fmha_output.data_ptr();
    }
    RTP_LLM_CHECK_WITH_INFO(fmha_output_ptr, "fmha_output_ptr must be provided for trt v2 fmha");

    trt_v2_runner_->runTrtV2Fmha(fmha_input_ptr,
                                 params->cu_seqlens.data_ptr(),
                                 fmha_output_ptr,
                                 reinterpret_cast<uint32_t*>(tiled_counter.data_ptr()),
                                 attention_output_orig_quant_scale,
                                 batch_size,
                                 params->max_seq_len,
                                 token_num,
                                 kv_block_array);
    if (use_fp8_fmha) {
        output = tmp_fmha_output.to(output.dtype());
    }
    // reserve for debug
    // DISPATCH_CUDA_FUNCTION_DATA_TYPE(torchDTypeToDataType(output.dtype()),
    //                                  invoke_debug_kernel2,
    //                                  output.data_ptr(),
    //                                  0,
    //                                  0,
    //                                  30,
    //                                  10,
    //                                  output.sizes()[1],
    //                                  2,
    //                                  device_->getStream());
    return output;
}

void registerTRTAttnOp(const py::module& m) {
    pybind11::class_<TRTPagedPrefillOp>(m, "TRTPagedAttnOp")
        .def(pybind11::init<const AttentionConfigs&>(), py::arg("attn_configs"))
        .def("support", &TRTPagedPrefillOp::support, py::arg("attn_inputs"))
        .def("prepare", &TRTPagedPrefillOp::prepare, py::arg("attn_inputs"))
        .def("forward", &TRTPagedPrefillOp::forward, py::arg("input"), py::arg("kv_cache"), py::arg("params"));

    pybind11::class_<TRTNormalPrefillOp>(m, "TRTAttnOp")
        .def(pybind11::init<const AttentionConfigs&>(), py::arg("attn_configs"))
        .def("support", &TRTNormalPrefillOp::support, py::arg("attn_inputs"))
        .def("prepare", &TRTNormalPrefillOp::prepare, py::arg("attn_inputs"))
        .def("forward", &TRTNormalPrefillOp::forward, py::arg("input"), py::arg("kv_cache"), py::arg("params"));
}

}  // namespace rtp_llm