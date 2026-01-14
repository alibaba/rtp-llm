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
    KVBlockArray kv_block_array;
    if (!kv_cache.has_value()) {
        throw std::runtime_error("kv_cache must be provided for trt v2 fmha paged");
    }
    kv_block_array                 = params->kv_block_array;
    kv_block_array.mPrimaryPoolPtr = kv_cache.value().kv_cache_base.data_ptr();
    if (kv_cache.value().kv_scale_base.defined() && kv_cache.value().kv_scale_base.numel() > 0) {
        kv_block_array.scale = kv_cache.value().kv_scale_base.data_ptr();
    }

    const int local_head_num = attn_configs_.head_num;
    const int size_per_head  = attn_configs_.size_per_head;

    // TRT kernel expects Q in [token, head, dim] layout
    // input shape: [token_num, head_num, head_dim]
    const int            token_num  = input.size(0);  // Changed from size(1) to size(0)
    const int            batch_size = params->input_lengths.size(0);
    torch::TensorOptions options    = torch::TensorOptions(input.dtype()).device(input.device());

    torch::Tensor output        = torch::zeros({token_num, local_head_num * size_per_head}, options);
    torch::Tensor tiled_counter = torch::zeros({1}, torch::TensorOptions(torch::kUInt32).device(input.device()));
    bool          use_fp8_fmha  = kv_block_array.cache_type == KvCacheDataType::FP8;
    float*        attention_output_orig_quant_scale = use_fp8_fmha ? static_scale_.data_ptr<float>() : nullptr;

    std::cout << "\n[TRTPagedPrefillOp::forward] Parameters:" << std::endl;
    std::cout << "  batch_size: " << batch_size << std::endl;
    std::cout << "  max_seq_len (new tokens): " << params->max_seq_len << std::endl;
    std::cout << "  max_prefix_length: " << params->max_prefix_length << std::endl;
    std::cout << "  seq_len_with_prefix: " << (params->max_seq_len + params->max_prefix_length) << std::endl;
    std::cout << "  token_num (total new tokens): " << token_num << std::endl;
    std::cout << "  context_total_kv_length (total KV): " << params->context_total_kv_length << std::endl;
    std::cout << "  input shape (Q): [" << input.size(0) << ", " << input.size(1) << ", " << input.size(2) << "]"
              << std::endl;

    // Print all runTrtV2FmhaPaged parameters
    std::cout << "\n[DEBUG] runTrtV2FmhaPaged call parameters:" << std::endl;
    std::cout << "  input shape: [" << input.size(0) << ", " << input.size(1) << ", " << input.size(2) << "]"
              << std::endl;

    // Print input tensor sample values (按 token x head 组织)
    // Input shape: [token_num, head_num, head_dim]
    auto input_cpu    = input.to(torch::kCPU).to(torch::kFloat);
    int  print_tokens = std::min(4, (int)input.size(0));  // input.size(0) is token_num
    int  print_heads  = std::min(4, (int)input.size(1));  // input.size(1) is head_num
    int  print_dims   = std::min(8, (int)input.size(2));  // input.size(2) is head_dim
    std::cout << "\n  [C++ Q TENSOR] input (first " << print_tokens << " tokens, first " << print_heads
              << " heads):" << std::endl;
    for (int t = 0; t < print_tokens; t++) {
        std::cout << "    Token " << t << ":" << std::endl;
        for (int h = 0; h < print_heads; h++) {
            std::cout << "      Q_Head[" << h << "]: [";
            for (int d = 0; d < print_dims; d++) {
                std::cout << input_cpu[t][h][d].item<float>();
                if (d < print_dims - 1)
                    std::cout << ", ";
            }
            std::cout << ", ...]" << std::endl;
        }
    }

    std::cout << "  output shape: [" << output.size(0) << ", " << output.size(1) << "]" << std::endl;

    std::cout << "  batch_size: " << batch_size << std::endl;
    std::cout << "  max_seq_len: " << params->max_seq_len << std::endl;
    std::cout << "  max_past_kv_len (seq_len_with_prefix): " << (params->max_seq_len + params->max_prefix_length)
              << std::endl;
    std::cout << "  token_num: " << token_num << std::endl;
    std::cout << "  token_num_kv (context_total_kv_length): " << params->context_total_kv_length << std::endl;

    // Print cu_seqlens content (complete)
    std::cout << "  cu_seqlens (shape=[" << params->cu_seqlens.numel() << "]):" << std::endl;
    std::cout << "    ";
    auto cu_seqlens_cpu = params->cu_seqlens.to(torch::kCPU);
    for (int i = 0; i < params->cu_seqlens.numel(); i++) {
        std::cout << cu_seqlens_cpu[i].item<int>() << " ";
        if ((i + 1) % 10 == 0 && i + 1 < params->cu_seqlens.numel())
            std::cout << std::endl << "    ";
    }
    std::cout << std::endl;

    // Print cu_kv_seqlens content (complete)
    std::cout << "  cu_kv_seqlens (shape=[" << params->cu_kv_seqlens.numel() << "]):" << std::endl;
    std::cout << "    ";
    auto cu_kv_seqlens_cpu = params->cu_kv_seqlens.to(torch::kCPU);
    for (int i = 0; i < params->cu_kv_seqlens.numel(); i++) {
        std::cout << cu_kv_seqlens_cpu[i].item<int>() << " ";
        if ((i + 1) % 10 == 0 && i + 1 < params->cu_kv_seqlens.numel())
            std::cout << std::endl << "    ";
    }
    std::cout << std::endl;

    // Print KVBlockArray details
    std::cout << "  kv_block_array details:" << std::endl;
    std::cout << "    mMaxSeqs: " << kv_block_array.mMaxSeqs << std::endl;
    std::cout << "    mMaxBlocksPerSeq: " << kv_block_array.mMaxBlocksPerSeq << std::endl;
    std::cout << "    mTokensPerBlock: " << kv_block_array.mTokensPerBlock << std::endl;
    std::cout << "    mTokensPerBlockLog2: " << kv_block_array.mTokensPerBlockLog2 << std::endl;
    std::cout << "    mBytesPerBlock: " << kv_block_array.mBytesPerBlock << std::endl;
    std::cout << "    mPrimaryPoolPtr: " << kv_block_array.mPrimaryPoolPtr << std::endl;
    std::cout << "    mSecondaryPoolPtr: " << kv_block_array.mSecondaryPoolPtr << std::endl;
    std::cout << "    cache_type: " << static_cast<int>(kv_block_array.cache_type) << std::endl;

    // Print block array data content (first few elements)
    if (kv_block_array.data != nullptr && kv_block_array.mPrimaryPoolPtr != nullptr) {
        // Layout: [batch, 2, max_blocks_per_seq] where 2 = [K_blocks, V_blocks]
        int total_blocks = kv_block_array.mMaxSeqs * kv_block_array.mMaxBlocksPerSeq * 2;  // *2 for K and V
        int print_count  = std::min(40, total_blocks);
        std::vector<int32_t> block_data_host(print_count);
        cudaMemcpy(block_data_host.data(), kv_block_array.data, print_count * sizeof(int32_t), cudaMemcpyDeviceToHost);

        std::cout << "    block array indices (first " << print_count
                  << " values, layout=[batch, 2, max_blocks]):" << std::endl;
        std::cout << "      ";
        for (int i = 0; i < print_count; i++) {
            std::cout << block_data_host[i] << " ";
            if ((i + 1) % 10 == 0 && i + 1 < print_count)
                std::cout << std::endl << "      ";
        }
        std::cout << std::endl;

        // Now read actual KV cache data using these block indices
        // KV cache layout: each block contains [tokens_per_block, head_num, head_dim] data
        int tokens_per_block = kv_block_array.mTokensPerBlock;
        int head_num         = attn_configs_.kv_head_num;
        int head_dim         = size_per_head;

        // Print K cache for first batch, first block
        if (block_data_host[0] >= 0) {  // Valid block index
            int k_block_idx = block_data_host[0];
            int v_block_idx = block_data_host[kv_block_array.mMaxBlocksPerSeq];  // V blocks start after all K blocks

            std::cout << "    KV cache actual data (batch=0, first block):" << std::endl;
            std::cout << "      K block_idx=" << k_block_idx << ", V block_idx=" << v_block_idx << std::endl;

            // Calculate K block data address
            // Each block size = tokens_per_block * head_num * head_dim * sizeof(dtype)
            size_t elements_per_block = tokens_per_block * head_num * head_dim;

            // Read ALL K data for this block (assuming fp16)
            std::vector<uint16_t> k_data_host(tokens_per_block * head_num * head_dim);
            uint16_t*             k_block_ptr =
                reinterpret_cast<uint16_t*>(kv_block_array.mPrimaryPoolPtr) + k_block_idx * elements_per_block;
            cudaMemcpy(k_data_host.data(),
                       k_block_ptr,
                       tokens_per_block * head_num * head_dim * sizeof(uint16_t),
                       cudaMemcpyDeviceToHost);

            // TRT KV cache layout: [numHeads, tokensPerBlock, hiddenSizePerHead]
            std::cout << "      K data (ALL " << tokens_per_block << " tokens, ALL " << head_num << " heads, ALL "
                      << head_dim << " dims):" << std::endl;
            std::cout << "      Layout in memory: [numHeads=" << head_num << ", tokensPerBlock=" << tokens_per_block
                      << ", hiddenSizePerHead=" << head_dim << "]" << std::endl;
            // Print in memory order: head first, then token
            for (int h = 0; h < head_num; h++) {
                for (int t = 0; t < tokens_per_block; t++) {
                    std::cout << "        head[" << h << "] token[" << t << "]: [";
                    for (int d = 0; d < head_dim; d++) {
                        // Index: headIdx * tokensPerBlock * dimsPerHead + tokenIdx * dimsPerHead + channelIdx
                        int idx = h * tokens_per_block * head_dim + t * head_dim + d;
                        // Convert fp16 to float for display
                        __half val_half  = *reinterpret_cast<__half*>(&k_data_host[idx]);
                        float  val_float = __half2float(val_half);
                        std::cout << val_float;
                        if (d < head_dim - 1)
                            std::cout << ", ";
                    }
                    std::cout << "]" << std::endl;
                }
            }

            // Read ALL V data for this block
            std::vector<uint16_t> v_data_host(tokens_per_block * head_num * head_dim);
            uint16_t*             v_block_ptr =
                reinterpret_cast<uint16_t*>(kv_block_array.mPrimaryPoolPtr) + v_block_idx * elements_per_block;
            cudaMemcpy(v_data_host.data(),
                       v_block_ptr,
                       tokens_per_block * head_num * head_dim * sizeof(uint16_t),
                       cudaMemcpyDeviceToHost);

            // TRT KV cache layout: [numHeads, tokensPerBlock, hiddenSizePerHead]
            std::cout << "      V data (ALL " << tokens_per_block << " tokens, ALL " << head_num << " heads, ALL "
                      << head_dim << " dims):" << std::endl;
            std::cout << "      Layout in memory: [numHeads=" << head_num << ", tokensPerBlock=" << tokens_per_block
                      << ", hiddenSizePerHead=" << head_dim << "]" << std::endl;
            // Print in memory order: head first, then token
            for (int h = 0; h < head_num; h++) {
                for (int t = 0; t < tokens_per_block; t++) {
                    std::cout << "        head[" << h << "] token[" << t << "]: [";
                    for (int d = 0; d < head_dim; d++) {
                        // Index: headIdx * tokensPerBlock * dimsPerHead + tokenIdx * dimsPerHead + channelIdx
                        int    idx       = h * tokens_per_block * head_dim + t * head_dim + d;
                        __half val_half  = *reinterpret_cast<__half*>(&v_data_host[idx]);
                        float  val_float = __half2float(val_half);
                        std::cout << val_float;
                        if (d < head_dim - 1)
                            std::cout << ", ";
                    }
                    std::cout << "]" << std::endl;
                }
            }
        }
    } else {
        std::cout << "    data or mPrimaryPoolPtr: nullptr" << std::endl;
    }
    std::cout << std::flush;

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

    // Print output tensor sample values
    std::cout << "\n[DEBUG] After runTrtV2FmhaPaged:" << std::endl;
    auto output_cpu       = output.to(torch::kCPU).to(torch::kFloat);
    int  out_print_tokens = std::min(3, (int)output.size(0));
    int  out_print_dims   = std::min(8, (int)output.size(1));
    std::cout << "  output sample values (first " << out_print_tokens << " tokens, first " << out_print_dims
              << " dims):" << std::endl;
    for (int t = 0; t < out_print_tokens; t++) {
        std::cout << "    token[" << t << "]: ";
        for (int d = 0; d < out_print_dims; d++) {
            std::cout << output_cpu[t][d].item<float>() << " ";
        }
        std::cout << "..." << std::endl;
    }
    std::cout << std::flush;

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