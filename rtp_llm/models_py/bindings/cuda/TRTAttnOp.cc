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

    // ========== DEBUG: Print TRT Attention parameters ==========
    std::cout << "\n[TRTPrefillOpBase.prepare] TRT Attention parameters:" << std::endl;
    std::cout << "  Batch size: " << batch_size << std::endl;

    std::cout << "  input_lengths shape: [" << attn_inputs.input_lengths.size(0)
              << "], device: " << attn_inputs.input_lengths.device() << std::endl;
    std::cout << "  input_lengths: " << attn_inputs.input_lengths << std::endl;

    if (attn_inputs.sequence_lengths.defined()) {
        std::cout << "  sequence_lengths shape: [" << attn_inputs.sequence_lengths.size(0) << "]" << std::endl;
        std::cout << "  sequence_lengths: " << attn_inputs.sequence_lengths << std::endl;
    }

    if (attn_inputs.prefix_lengths.defined()) {
        std::cout << "  prefix_lengths shape: [" << attn_inputs.prefix_lengths.size(0) << "]" << std::endl;
        std::cout << "  prefix_lengths: " << attn_inputs.prefix_lengths << std::endl;
    }

    std::cout << "  cu_seqlens shape: [" << attn_inputs.cu_seqlens.size(0) << "]" << std::endl;
    std::cout << "  cu_seqlens: " << attn_inputs.cu_seqlens << std::endl;

    std::cout << "  cu_kv_seqlens shape: [" << attn_inputs.cu_kv_seqlens.size(0) << "]" << std::endl;
    std::cout << "  cu_kv_seqlens: " << attn_inputs.cu_kv_seqlens << std::endl;

    if (attn_inputs.kv_cache_block_id_host.defined() && attn_inputs.kv_cache_block_id_host.numel() > 0) {
        std::cout << "  kv_cache_block_id_host shape: [";
        for (int64_t i = 0; i < attn_inputs.kv_cache_block_id_host.dim(); ++i) {
            if (i > 0)
                std::cout << ", ";
            std::cout << attn_inputs.kv_cache_block_id_host.size(i);
        }
        std::cout << "]" << std::endl;
        std::cout << "  kv_cache_block_id_host (all): " << attn_inputs.kv_cache_block_id_host.flatten() << std::endl;
    }

    std::cout << "  head_num (Q): " << attn_configs_.head_num << std::endl;
    std::cout << "  kv_head_num (KV): " << attn_configs_.kv_head_num << std::endl;
    std::cout << "  head_dim: " << attn_configs_.size_per_head << std::endl;
    std::cout << "  kv_cache_dtype: " << static_cast<int>(attn_configs_.kv_cache_dtype) << std::endl;
    std::cout << "  ========================================================\n" << std::endl;

    // Save kv_cache_block_id_host for later use in forward
    if (attn_inputs.kv_cache_block_id_host.defined() && attn_inputs.kv_cache_block_id_host.numel() > 0) {
        kv_cache_block_id_host_ = attn_inputs.kv_cache_block_id_host;
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
    const int            token_num  = input.size(0);
    const int            batch_size = params->input_lengths.size(0);
    torch::TensorOptions options    = torch::TensorOptions(input.dtype()).device(input.device());

    torch::Tensor output        = torch::zeros({token_num, local_head_num * size_per_head}, options);
    torch::Tensor tiled_counter = torch::zeros({1}, torch::TensorOptions(torch::kUInt32).device(input.device()));
    bool          use_fp8_fmha  = kv_block_array.cache_type == KvCacheDataType::FP8;
    float*        attention_output_orig_quant_scale = use_fp8_fmha ? static_scale_.data_ptr<float>() : nullptr;

    // ========== DEBUG: Print input statistics ==========
    std::cout << "\n[TRTPagedPrefillOp.forward] Running TRT Paged Attention:" << std::endl;
    std::cout << "  input (Q) shape: [" << input.size(0) << ", " << input.size(1) << ", " << input.size(2) << "]"
              << std::endl;
    std::cout << "  batch_size: " << batch_size << ", token_num: " << token_num << std::endl;
    std::cout << "  kv_cache shape (inferred): [num_blocks, 2, block_size, " << attn_configs_.kv_head_num << ", "
              << size_per_head << "]" << std::endl;

    // Print Q statistics
    std::cout << "\n  📊 Q Statistics:" << std::endl;
    std::cout << "    Q max: " << input.max().item<float>() << std::endl;
    std::cout << "    Q min: " << input.min().item<float>() << std::endl;
    std::cout << "    Q mean: " << input.mean().item<float>() << std::endl;
    std::cout << "    Q std: " << input.std().item<float>() << std::endl;

    // Print KV cache block array info
    std::cout << "\n  📦 KV Block Array Info:" << std::endl;
    std::cout << "    max_blocks_per_seq: " << kv_block_array.mMaxBlocksPerSeq << std::endl;
    std::cout << "    max_attention_window: " << kv_block_array.mMaxAttentionWindow << std::endl;
    std::cout << "    tokens_per_block: " << kv_block_array.mTokensPerBlock << std::endl;

    // Print kv_cache_block_id_host_ information
    if (kv_cache_block_id_host_.defined() && kv_cache_block_id_host_.numel() > 0) {
        std::cout << "\n  📋 KV Cache Block ID (from prepare):" << std::endl;
        std::cout << "    kv_cache_block_id_host_ shape: [";
        for (int64_t i = 0; i < kv_cache_block_id_host_.dim(); ++i) {
            if (i > 0)
                std::cout << ", ";
            std::cout << kv_cache_block_id_host_.size(i);
        }
        std::cout << "]" << std::endl;
        std::cout << "    kv_cache_block_id_host_ (all): " << kv_cache_block_id_host_.flatten() << std::endl;
    }

    // Print statistics for used blocks only
    if (kv_cache.has_value() && kv_cache.value().kv_cache_base.defined() && kv_cache_block_id_host_.defined()
        && kv_cache_block_id_host_.numel() > 0) {
        auto kv_cache_tensor = kv_cache.value().kv_cache_base;
        std::cout << "\n  📦 KV Cache Statistics (used blocks only):" << std::endl;
        std::cout << "    Full KV cache tensor shape: [";
        for (int64_t i = 0; i < kv_cache_tensor.dim(); ++i) {
            if (i > 0)
                std::cout << ", ";
            std::cout << kv_cache_tensor.size(i);
        }
        std::cout << "]" << std::endl;

        // Flatten block IDs and filter out invalid indices
        auto block_ids_flat  = kv_cache_block_id_host_.flatten();
        auto valid_mask      = block_ids_flat >= 0;  // Filter out -1 (invalid blocks)
        auto valid_block_ids = block_ids_flat.masked_select(valid_mask);

        std::cout << "    Valid block IDs count: " << valid_block_ids.numel() << std::endl;
        std::cout << "    Valid block IDs (all): " << valid_block_ids << std::endl;

        if (valid_block_ids.numel() > 0) {
            // Move valid_block_ids to GPU for index_select
            auto valid_block_ids_gpu = valid_block_ids.to(kv_cache_tensor.device());
            // Index the used blocks: kv_cache_tensor shape is typically [num_blocks, 2, tokens_per_block, num_heads,
            // head_dim]
            auto used_kv_blocks = kv_cache_tensor.index_select(0, valid_block_ids_gpu);
            std::cout << "    Used KV blocks shape: [";
            for (int64_t i = 0; i < used_kv_blocks.dim(); ++i) {
                if (i > 0)
                    std::cout << ", ";
                std::cout << used_kv_blocks.size(i);
            }
            std::cout << "]" << std::endl;
            std::cout << "    KV max: " << used_kv_blocks.max().item<float>() << std::endl;
            std::cout << "    KV min: " << used_kv_blocks.min().item<float>() << std::endl;
            std::cout << "    KV mean: " << used_kv_blocks.mean().item<float>() << std::endl;
            std::cout << "    KV std: " << used_kv_blocks.std().item<float>() << std::endl;
        }
    }
    std::cout << "  ========================================================\n" << std::endl;

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

    // Print last 10 tokens, first 10 dimensions each
    std::cout << "\n  🔍 TRT Output sample (last 10 tokens, first 10 dims each):" << std::endl;
    int num_tokens_to_print = std::min(10, static_cast<int>(output.size(0)));
    int start_idx           = output.size(0) - num_tokens_to_print;
    for (int i = start_idx; i < output.size(0); ++i) {
        std::cout << "    Token " << i << ": [";
        auto token_data = output[i].flatten();
        for (int j = 0; j < std::min(10, static_cast<int>(token_data.size(0))); ++j) {
            if (j > 0)
                std::cout << ", ";
            std::cout << token_data[j].item<float>();
        }
        std::cout << "]" << std::endl;
    }

    // Check for NaN and Inf
    bool has_nan = torch::isnan(output).any().item<bool>();
    bool has_inf = torch::isinf(output).any().item<bool>();
    if (has_nan || has_inf) {
        std::cout << "\n  ⚠️  WARNING: Output contains NaN: " << has_nan << ", Inf: " << has_inf << std::endl;
        if (has_nan) {
            int64_t nan_count = torch::isnan(output).sum().item<int64_t>();
            std::cout << "    NaN count: " << nan_count << " / " << output.numel() << std::endl;
        }
        if (has_inf) {
            int64_t inf_count = torch::isinf(output).sum().item<int64_t>();
            std::cout << "    Inf count: " << inf_count << " / " << output.numel() << std::endl;
        }
    } else {
        std::cout << "\n  ✅ No NaN or Inf in output" << std::endl;
    }
    std::cout << "  ========================================================\n" << std::endl;

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
    std::cout << "TRTNormalPrefillOp forward successfully" << std::endl;
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