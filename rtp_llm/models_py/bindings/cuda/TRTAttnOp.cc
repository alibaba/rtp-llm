#include "rtp_llm/models_py/bindings/cuda/TRTAttnOp.h"
#include "rtp_llm/cpp/cuda/cufmha/TrtV2FmhaRunner.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"

using namespace torch_ext;

namespace rtp_llm {

TRTPrefillOpBase::TRTPrefillOpBase(const GptInitParameter& gpt_init_parameter): FMHACudaBase(gpt_init_parameter) {}

ParamsBasePtr TRTPrefillOpBase::prepare(torch_ext::PyAttentionInputs attn_inputs) {
    static_scale_        = torch::ones({1}, torch::TensorOptions(torch::kFloat32).device(torch::kCUDA));
    int       batch_size = attn_inputs.input_lengths.size(0);
    BufferPtr kv_cache_block_id_host, kv_cache_block_id_device;
    if (attn_inputs.kv_cache_block_id_host.defined() && attn_inputs.kv_cache_block_id_host.numel() > 0) {
        kv_cache_block_id_host   = torchTensor2Buffer(attn_inputs.kv_cache_block_id_host);
        kv_cache_block_id_device = torchTensor2Buffer(attn_inputs.kv_cache_block_id_device);
    }

    TRTAttnPtr attn_params;
    auto       run_stream   = at::cuda::getCurrentCUDAStream(at::cuda::current_device()).stream();
    bool       use_fp8_fmha = attn_configs_.kv_cache_dtype == KvCacheDataType::FP8;
    auto       params       = prepareTrtAttnParams(attn_configs_,
                                       attn_inputs.kv_block_offset,
                                       kv_cache_block_id_device,
                                       attn_inputs.input_lengths.size(0),
                                       use_fp8_fmha,
                                       run_stream,
                                       fmha_config_.enable_paged_trt_fmha);
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

    if (!has_prefix || !fmha_config_.enable_trt_fmha || !fmha_config_.enable_paged_trt_fmha
        || attn_configs_.kv_cache_dtype == KvCacheDataType::INT8) {
        return false;
    }

    // 创建 runner 并检查是否支持
    DataType attn_dtype = attn_configs_.kv_cache_dtype == KvCacheDataType::FP8 ?
                              DataType::TYPE_FP8_E4M3 :
                              torchDTypeToDataType(attn_inputs.dtype);

    auto run_stream   = at::cuda::getCurrentCUDAStream(at::cuda::current_device()).stream();
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
    kv_block_array.mPrimaryPoolPtr = kv_cache.value().k_cache_base.data_ptr();
    if (kv_cache.value().k_scale_base.defined() && kv_cache.value().k_scale_base.numel() > 0) {
        kv_block_array.scale = kv_cache.value().k_scale_base.data_ptr();
    }

    const int            local_head_num = attn_configs_.head_num;
    const int            size_per_head  = attn_configs_.size_per_head;
    const int            token_num      = input.size(1);
    const int            batch_size     = params->input_lengths.size(0);
    torch::TensorOptions options        = torch::TensorOptions(input.dtype()).device(input.device());

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
    return output;
}

bool TRTNormalPrefillOp::support(torch_ext::PyAttentionInputs attn_inputs) {
    bool has_prefix =
        attn_inputs.prefix_lengths.defined() && torch::any(attn_inputs.prefix_lengths.reshape({-1})).item<bool>();

    if (has_prefix || !fmha_config_.enable_trt_fmha || attn_configs_.kv_cache_dtype == KvCacheDataType::INT8) {
        return false;
    }

    auto     run_stream   = at::cuda::getCurrentCUDAStream(at::cuda::current_device()).stream();
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
        kv_block_array.mPrimaryPoolPtr = kv_cache.value().k_cache_base.data_ptr();
        if (kv_cache.value().k_scale_base.defined() && kv_cache.value().k_scale_base.numel() > 0) {
            kv_block_array.scale = kv_cache.value().k_scale_base.data_ptr();
        }
    }

    const int local_head_num = attn_configs_.head_num;
    const int size_per_head  = attn_configs_.size_per_head;
    const int token_num      = input.size(0);
    const int batch_size     = params->input_lengths.size(0);
    const int max_token_num =
        device_->initParams().fifo_scheduler_config.max_context_batch_size * device_->initParams().max_seq_len;
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
        .def(pybind11::init<GptInitParameter>(), py::arg("gpt_init_parameter"))
        .def("support", &TRTPagedPrefillOp::support, py::arg("attn_inputs"))
        .def("prepare", &TRTPagedPrefillOp::prepare, py::arg("attn_inputs"))
        .def("forward", &TRTPagedPrefillOp::forward, py::arg("input"), py::arg("kv_cache"), py::arg("params"));

    pybind11::class_<TRTNormalPrefillOp>(m, "TRTAttnOp")
        .def(pybind11::init<GptInitParameter>(), py::arg("gpt_init_parameter"))
        .def("support", &TRTNormalPrefillOp::support, py::arg("attn_inputs"))
        .def("prepare", &TRTNormalPrefillOp::prepare, py::arg("attn_inputs"))
        .def("forward", &TRTNormalPrefillOp::forward, py::arg("input"), py::arg("kv_cache"), py::arg("params"));
}

}  // namespace rtp_llm
