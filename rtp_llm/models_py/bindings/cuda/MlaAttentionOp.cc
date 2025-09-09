#include "rtp_llm/models_py/bindings/cuda/MlaAttentionOp.h"
#include "rtp_llm/cpp/devices/cuda_impl/CudaDevice.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "3rdparty/flashinfer/flashinfer.h"
#include "rtp_llm/cpp/devices/cuda_impl/CudaFlashInfer.h"
#include "rtp_llm/cpp/devices/OpData.h"
#include "rtp_llm/cpp/devices/DeviceFactory.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "flashmla/flashmla.h"
#include "rtp_llm/cpp/kernels/mla_kernels/mla_merge_transpose_kernel.h"
#include "rtp_llm/cpp/kernels/kv_cache/kv_cache_utils.h"
#include <torch/nn/functional/linear.h>
#include "rtp_llm/cpp/core/Dispatch.h"
#include "rtp_llm/models_py/bindings/common/Torch_ext.h"

using namespace torch_ext;
using namespace std;

namespace rtp_llm {

MlaAttentionCudaBase::MlaAttentionCudaBase(const GptInitParameter& gpt_init_parameter):
    FMHACudaBase(gpt_init_parameter) {
    use_mla_ = gpt_init_parameter.use_mla_;
}

bool MlaAttentionCudaBase::support(torch_ext::PyAttentionInputs attn_inputs) {
    return use_mla_;
}

FlashInferAttnParamsPtr MlaAttentionCudaBase::prepare(torch_ext::PyAttentionInputs attn_inputs) {
    if (attn_inputs.mla_inputs.has_value()) {
        context_batch_size_ = attn_inputs.mla_inputs.value().context_batch_size;
        decoder_batch_size_ = attn_inputs.mla_inputs.value().decoder_batch_size;
        context_token_num_  = attn_inputs.mla_inputs.value().context_token_num;
        // max_prefix_length_ attr need to opt
        max_prefix_length_   = attn_inputs.mla_inputs.value().max_prefix_length;
        max_context_seq_len_ = attn_inputs.mla_inputs.value().max_context_seq_len;
    }
    is_prefill_                     = attn_inputs.is_prefill;
    cu_seqlens_                     = attn_inputs.cu_seqlens;
    auto      prefix_lengths_host   = torchTensor2Buffer(attn_inputs.prefix_lengths);
    auto      sequence_lengths_host = torchTensor2Buffer(attn_inputs.sequence_lengths);
    auto      input_lengths_host    = torchTensor2Buffer(attn_inputs.input_lengths);
    BufferPtr kv_cache_block_id_host, kv_cache_block_id_device;
    if (attn_inputs.kv_cache_block_id_host.size(0)) {
        kv_cache_block_id_host   = torchTensor2Buffer(attn_inputs.kv_cache_block_id_host);
        kv_cache_block_id_device = torchTensor2Buffer(attn_inputs.kv_cache_block_id_device);
    }
    attn_inputs.dtype = torch::kFloat16;
    DataType dtype    = torchDTypeToDataType(attn_inputs.dtype);
    if (!is_prefill_) {
        prefix_lengths_host = nullptr;
    }
    if (attn_configs_.kv_cache_dtype == KvCacheDataType::FP8) {
        dtype = DataType::TYPE_FP8_E4M3;
    }
    auto                    params = FlashInferAttnParams::prepare(device_,
                                                attn_configs_,
                                                prefix_lengths_host,
                                                sequence_lengths_host,
                                                input_lengths_host,
                                                kv_cache_block_id_host,
                                                kv_cache_block_id_device,
                                                dtype,  // torchDTypeToDataType(attn_inputs.dtype),
                                                false);
    FlashInferAttnParamsPtr attn_params(params, (FlashInferAttnParams*)params.get());
    // DataType                attn_dtype = attn_configs_.kv_cache_dtype == KvCacheDataType::FP8 ?
    //                                          DataType::TYPE_FP8_E4M3 :
    //                                          torchDTypeToDataType(attn_inputs.dtype);
    // cufmha_runner_                     = device_->selectCuFMHARunner(attn_configs_, attn_dtype, false);
    return attn_params;
}

MlaContextAttentionOp::MlaContextAttentionOp(const GptInitParameter& gpt_init_parameter):
    MlaAttentionCudaBase(gpt_init_parameter) {}

bool MlaContextAttentionOp::support(torch_ext::PyAttentionInputs attn_inputs) {
    return (use_mla_ && attn_inputs.is_prefill);
}

torch::Tensor MlaContextAttentionOp::forward(torch::Tensor&                 q,
                                             torch::Tensor&                 kv_a,
                                             torch::Tensor&                 k_rope,
                                             const int64_t                  kv_offset,
                                             const FlashInferAttnParamsPtr& params,
                                             const torch::Tensor&           k_nope_weight,
                                             const torch::Tensor&           v_weight) {
    RTP_LLM_CHECK_WITH_INFO(params != nullptr, "MlaContextAttentionOp op should have params");
    q = q.slice(0, decoder_batch_size_, context_token_num_, 1);

    torch::Tensor        k_nope  = torch::nn::functional::linear(kv_a, k_nope_weight.transpose(0, 1), {});
    torch::Tensor        v       = torch::nn::functional::linear(kv_a, v_weight.transpose(0, 1), {});
    torch::TensorOptions options = torch::TensorOptions(q.dtype()).device(q.device());

    const auto d_type    = q.dtype();
    DataType   data_type = torchDTypeToDataType(d_type);
    auto const token_num = q.size(0);

    StreamType stream = GET_CURRENT_STREAM();

    auto nope_rope_dim = attn_configs_.nope_head_dim + attn_configs_.rope_head_dim;
    auto qkv =
        torch::zeros({(int64_t)(token_num), (int64_t)attn_configs_.head_num * (int64_t)nope_rope_dim * 3}, options);
    DISPATCH_CUDA_FUNCTION_DATA_TYPE(data_type,
                                     invokeMlaQKVMerge,
                                     (q.data_ptr()),
                                     (k_nope.data_ptr()),
                                     (k_rope.data_ptr()),
                                     (v.data_ptr()),
                                     (qkv.data_ptr()),
                                     (int)token_num,
                                     (int)attn_configs_.head_num,
                                     (int)attn_configs_.nope_head_dim,
                                     (int)attn_configs_.rope_head_dim,
                                     (int)attn_configs_.v_head_dim,
                                     stream);

    k_nope.reset();
    v.reset();
    const int local_head_num = attn_configs_.head_num;
    const int size_per_head  = attn_configs_.size_per_head;

    torch::Tensor output = torch::zeros({(int64_t)token_num, (int64_t)local_head_num * size_per_head}, options);

    void* fmha_input_ptr  = qkv.data_ptr();
    void* fmha_output_ptr = output.data_ptr();

    auto     tiled_counter_ptr = torch::zeros({1}, torch::TensorOptions(torch::kUInt32).device(qkv.device()));
    DataType attn_dtype        = attn_configs_.kv_cache_dtype == KvCacheDataType::FP8 ? DataType::TYPE_FP8_E4M3 :
                                                                                        torchDTypeToDataType(qkv.dtype());
    cufmha_runner_             = device_->selectCuFMHARunner(attn_configs_, attn_dtype, false);
    KVBlockArray kv_block_array;
    cufmha_runner_->runTrtV2Fmha(fmha_input_ptr,
                                 cu_seqlens_.data_ptr(),
                                 fmha_output_ptr,
                                 reinterpret_cast<uint32_t*>(tiled_counter_ptr.data_ptr()),
                                 nullptr,
                                 context_batch_size_,
                                 max_context_seq_len_,
                                 token_num,
                                 kv_block_array);
    auto qkv_output_reshaped = output.reshape({(int64_t)token_num, (int64_t)local_head_num, (int64_t)size_per_head});
    auto sliced_buffer       = qkv_output_reshaped.slice(-1, 0, attn_configs_.v_head_dim);
    return sliced_buffer;
}

MlaAbsorbAttentionOp::MlaAbsorbAttentionOp(const GptInitParameter& gpt_init_parameter):
    MlaAttentionCudaBase(gpt_init_parameter) {}

bool MlaAbsorbAttentionOp::support(torch_ext::PyAttentionInputs attn_inputs) {
    return (use_mla_ && !attn_inputs.is_prefill);
}

torch::Tensor MlaAbsorbAttentionOp::forward(torch::Tensor&                    q,
                                            torch::Tensor&                    fused_q_input_t,
                                            std::optional<torch_ext::KVCache> kv_cache,
                                            const FlashInferAttnParamsPtr&    params,
                                            const torch::Tensor&              kc_weight,
                                            const torch::Tensor&              vc_weight) {
    auto qkv_output_t = torch::empty({(int64_t)(context_token_num_ + decoder_batch_size_),
                                      (int64_t)attn_configs_.head_num,
                                      (int64_t)attn_configs_.v_head_dim},
                                     torch::TensorOptions(q.dtype()).device(q.device()));

    RTP_LLM_CHECK_WITH_INFO(params != nullptr, "MlaAbsorbAttentionOp should have params");
    q             = q.slice(0, 0, decoder_batch_size_, 1);  // split on dim=0
    auto absorb_q = fused_q_input_t.slice(-1, 0, attn_configs_.kv_lora_rank);
    // shape: [head_num, nope_head_dim, kv_lora_rank]
    auto q_input_reshape = q.reshape({(int64_t)q.size(0),
                                      (int64_t)attn_configs_.head_num,
                                      (int64_t)(attn_configs_.nope_head_dim + attn_configs_.rope_head_dim)});

    auto q_nope_t = torch::from_blob(
        q_input_reshape.data_ptr(),
        {(int64_t)q.size(0), (int64_t)attn_configs_.head_num, (int64_t)attn_configs_.nope_head_dim},
        q_input_reshape.strides(),
        torch::dtype(q_input_reshape.scalar_type()).device(q_input_reshape.device()).requires_grad(false));
    torch::bmm_out(absorb_q.transpose_(0, 1), q_nope_t.transpose(0, 1), kc_weight);

    torch::Tensor output =
        torch::empty({(int64_t)q.size(0), (int64_t)attn_configs_.head_num, (int64_t)attn_configs_.kv_lora_rank},
                     torch::TensorOptions(q.dtype()).device(q.device()));

    torch::Tensor generate_qkv_output;
    if (is_prefill_) {
        generate_qkv_output = qkv_output_t.slice(0, decoder_batch_size_, context_token_num_ + decoder_batch_size_, 1);
    } else {
        generate_qkv_output = qkv_output_t.slice(0, 0, decoder_batch_size_, 1);
    }
    const auto& ckv_cache_shape = kv_cache.value().k_cache_base.sizes();
    RTP_LLM_LOG_DEBUG("mla_ops_type %d", params->mla_ops_type);
    if (params->mla_ops_type == MlaOpsType::FLASH_MLA) {
        if (is_prefill_) {
            fused_q_input_t =
                fused_q_input_t.reshape({(int64_t)context_batch_size_,
                                         (int64_t)(q.size(0) / context_batch_size_),
                                         (int64_t)attn_configs_.head_num,
                                         (int64_t)(attn_configs_.kv_lora_rank + attn_configs_.rope_head_dim)});
        } else {
            fused_q_input_t =
                fused_q_input_t.reshape({(int64_t)q.size(0),
                                         1,
                                         (int64_t)attn_configs_.head_num,
                                         (int64_t)(attn_configs_.kv_lora_rank + attn_configs_.rope_head_dim)});
        }

        auto ckv_cache_reshape_t = kv_cache.value().k_cache_base.reshape({
            ckv_cache_shape[0],
            ckv_cache_shape[1],
            1,
            ckv_cache_shape[2],
        });

        RTP_LLM_LOG_TRACE("kv_lora_rank = %zu", attn_configs_.kv_lora_rank);
        const float softmax_scale = attn_configs_.softmax_extra_scale / sqrtf(attn_configs_.size_per_head * 1.0f);

        RTP_LLM_LOG_TRACE("softmax_scale = %f", softmax_scale);
        const auto& tile_scheduler_metadata = params->flash_mla_plan[0];
        const auto& num_splits              = params->flash_mla_plan[1];

        output =
            mha_fwd_unified_kvcache_mla(fused_q_input_t,
                                        ckv_cache_reshape_t,
                                        attn_configs_.kv_lora_rank,
                                        params->kvlen_d,
                                        params->kv_cache_block_id_d,
                                        softmax_scale,
                                        /* is_causal = */ true,
                                        tile_scheduler_metadata,
                                        num_splits)[0]
                .reshape({(int64_t)q.size(0), (int64_t)attn_configs_.head_num, (int64_t)attn_configs_.kv_lora_rank});
    } else {
        auto q_rope = fused_q_input_t.slice(
            -1, attn_configs_.kv_lora_rank, attn_configs_.kv_lora_rank + attn_configs_.rope_head_dim);

        auto ckv_nope_cache = torch::from_blob(
            kv_cache.value().k_cache_base.data_ptr(),
            {(int64_t)ckv_cache_shape[0], (int64_t)ckv_cache_shape[1], (int64_t)attn_configs_.kv_lora_rank},
            kv_cache.value().k_cache_base.strides(),
            torch::dtype(kv_cache.value().k_cache_base.scalar_type())
                .device(kv_cache.value().k_cache_base.device())
                .requires_grad(false));

        auto k_rope_cache = torch::from_blob(
            static_cast<char*>(kv_cache.value().k_cache_base.data_ptr())
                + attn_configs_.kv_lora_rank * kv_cache.value().k_cache_base.element_size(),
            {(int64_t)ckv_cache_shape[0], (int64_t)ckv_cache_shape[1], (int64_t)attn_configs_.rope_head_dim},
            kv_cache.value().k_cache_base.strides(),
            torch::dtype(kv_cache.value().k_cache_base.scalar_type())
                .device(kv_cache.value().k_cache_base.device())
                .requires_grad(false));

        BatchMLAPagedAttentionRun(params->float_workspace_d,
                                  params->int_workspace_d,
                                  params->plan,
                                  fused_q_input_t.slice(-1, 0, attn_configs_.kv_lora_rank),
                                  q_rope,
                                  ckv_nope_cache,
                                  k_rope_cache,
                                  params->page_indice_d,
                                  output,
                                  std::nullopt,
                                  1,
                                  attn_configs_.head_num,
                                  attn_configs_.tokens_per_block,
                                  (1.0f / sqrtf(attn_configs_.size_per_head * 1.0f))
                                      * attn_configs_.softmax_extra_scale,
                                  (int64_t)device_->getStream());
    }
    torch::bmm_out(generate_qkv_output.transpose_(0, 1), output.transpose(0, 1), vc_weight);
    return qkv_output_t;
}

void registerMlaAttentionOp(const py::module& m) {
    pybind11::class_<MlaAttentionCudaBase>(m, "MlaAttentionCudaBase")
        .def(pybind11::init<GptInitParameter>(), py::arg("gpt_init_parameter"))
        .def("support", &MlaAttentionCudaBase::support, py::arg("attn_inputs"))
        .def("prepare", &MlaAttentionCudaBase::prepare, py::arg("attn_inputs"));
    pybind11::class_<MlaContextAttentionOp>(m, "MlaContextAttentionOp")
        .def(pybind11::init<GptInitParameter>(), py::arg("gpt_init_parameter"))
        .def("support", &MlaContextAttentionOp::support, py::arg("attn_inputs"))
        .def("prepare", &MlaAttentionCudaBase::prepare, py::arg("attn_inputs"))
        .def("forward",
             &MlaContextAttentionOp::forward,
             py::arg("q"),
             py::arg("kv_a"),
             py::arg("k_rope"),
             py::arg("kv_offset"),
             py::arg("params"),
             py::arg("k_nope_weight"),
             py::arg("v_weight"));
    pybind11::class_<MlaAbsorbAttentionOp>(m, "MlaAbsorbAttentionOp")
        .def(pybind11::init<GptInitParameter>(), py::arg("gpt_init_parameter"))
        .def("support", &MlaAbsorbAttentionOp::support, py::arg("attn_inputs"))
        .def("prepare", &MlaAttentionCudaBase::prepare, py::arg("attn_inputs"))
        .def("forward",
             &MlaAbsorbAttentionOp::forward,
             py::arg("q"),
             py::arg("fused_q_input_t"),
             py::arg("kv_cache"),
             py::arg("params"),
             py::arg("kc_weight"),
             py::arg("vc_weight"));
}

}  // namespace rtp_llm
