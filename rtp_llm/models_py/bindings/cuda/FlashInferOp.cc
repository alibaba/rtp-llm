#include "rtp_llm/models_py/bindings/cuda/FlashInferOp.h"
#include "rtp_llm/cpp/devices/cuda_impl/CudaDevice.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "3rdparty/flashinfer/flashinfer.h"
#include "rtp_llm/cpp/devices/cuda_impl/CudaFlashInfer.h"
#include "rtp_llm/cpp/devices/OpData.h"
#include "rtp_llm/cpp/devices/DeviceFactory.h"
#include "rtp_llm/cpp/utils/Logger.h"

using namespace torch_ext;

namespace rtp_llm {

FlashInferPrefillOp::FlashInferPrefillOp(const GptInitParameter& gpt_init_parameter):
    FMHACudaBase(gpt_init_parameter) {}

bool FlashInferPrefillOp::support(torch_ext::PyAttentionInputs attn_inputs) {
    if (fmha_config_.disable_flash_infer || attn_configs_.kv_cache_dtype != KvCacheDataType::BASE) {
        return false;
    }
    auto     prefix_lengths_host   = torchTensor2Buffer(attn_inputs.prefix_lengths);
    auto     sequence_lengths_host = torchTensor2Buffer(attn_inputs.sequence_lengths);
    auto     input_lengths_host    = torchTensor2Buffer(attn_inputs.input_lengths);
    DataType dtype                 = torchDTypeToDataType(attn_inputs.dtype);
    if (attn_configs_.kv_cache_dtype == KvCacheDataType::FP8) {
        dtype = DataType::TYPE_FP8_E4M3;
    }
    return FlashInferAttnParams::checkPrefill(
        device_, attn_configs_, prefix_lengths_host, input_lengths_host, dtype, false);
}

ParamsBasePtr FlashInferPrefillOp::prepare(torch_ext::PyAttentionInputs attn_inputs) {
    auto      prefix_lengths_host   = torchTensor2Buffer(attn_inputs.prefix_lengths);
    auto      sequence_lengths_host = torchTensor2Buffer(attn_inputs.sequence_lengths);
    auto      input_lengths_host    = torchTensor2Buffer(attn_inputs.input_lengths);
    BufferPtr kv_cache_block_id_host, kv_cache_block_id_device;
    if (attn_inputs.kv_cache_block_id_host.size(0)) {
        kv_cache_block_id_host   = torchTensor2Buffer(attn_inputs.kv_cache_block_id_host);
        kv_cache_block_id_device = torchTensor2Buffer(attn_inputs.kv_cache_block_id_device);
    }
    DataType dtype = torchDTypeToDataType(attn_inputs.dtype);
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
    RTP_LLM_CHECK_WITH_INFO(!attn_params->decode_plan, "flash infer params should gen prefill plan");
    return ParamsBasePtr(attn_params);
}

torch::Tensor FlashInferPrefillOp::forward(const torch::Tensor&              q,
                                           std::optional<torch_ext::KVCache> kv_cache,
                                           const FlashInferAttnParamsPtr&    params) {
    RTP_LLM_CHECK_WITH_INFO(params != nullptr, "flash infer op should have params");

    const int     local_head_num = attn_configs_.head_num;
    const int     size_per_head  = attn_configs_.size_per_head;
    const int     bs             = q.size(0);
    torch::Tensor output =
        torch::empty({bs, local_head_num * size_per_head}, torch::TensorOptions(q.dtype()).device(q.device()));
    auto softmax_scale = (1.0f / sqrtf(size_per_head * 1.0f)) * attn_configs_.softmax_extra_scale;
    RTP_LLM_LOG_DEBUG("prefill flashinfer");
    torch::Tensor k_cache, v_cache;
    if (kv_cache.has_value()) {
        k_cache = kv_cache.value().k_cache_base.select(1, 0);
        v_cache = kv_cache.value().k_cache_base.select(1, 1);
    }
    BatchPrefillWithPagedKVCacheRun(params->float_workspace_d,         // float_workspace_buffer
                                    params->int_workspace_d,           // int_workspace_buffer
                                    params->plan,                      // plan_info_vec
                                    q,                                 // q
                                    k_cache,                           // paged_k_cache
                                    v_cache,                           // paged_v_cache
                                    params->qo_indptr_d,               // qo_indptr
                                    params->page_indptr_d,             // paged_kv_indptr
                                    params->page_indice_d,             // paged_kv_indices
                                    params->paged_kv_last_page_len_d,  // paged_kv_last_page_len
                                    output,
                                    std::nullopt,  // maybe_lse
                                    1,             // mask_mode_code,
                                    1,             // layout
                                    -1,            // window_left
                                    std::nullopt,  // maybe_custom_mask
                                    std::nullopt,  // maybe_mask_indptr
                                    std::nullopt,  // maybe_alibi_slopes
                                    0,             // logits_soft_cap
                                    softmax_scale,
                                    attn_configs_.rope_config.scale,
                                    attn_configs_.rope_config.base,
                                    (int64_t)device_->getStream());
    return output;
}

FlashInferDecodeOp::FlashInferDecodeOp(const GptInitParameter& gpt_init_parameter): FMHACudaBase(gpt_init_parameter) {}

bool FlashInferDecodeOp::support(torch_ext::PyAttentionInputs attn_inputs) {
    if (fmha_config_.disable_flash_infer || attn_configs_.kv_cache_dtype != KvCacheDataType::BASE) {
        return false;
    }
    // FIXME: FlashInferDecodeOp causes crash in this case, temporarily bypassing it here
    if (attn_configs_.head_num / attn_configs_.kv_head_num == 12) {
        return false;
    }
    return FlashInferAttnParams::checkDecode(device_, attn_configs_, torchDTypeToDataType(attn_inputs.dtype));
}

ParamsBasePtr FlashInferDecodeOp::prepare(torch_ext::PyAttentionInputs attn_inputs) {
    auto      sequence_lengths_host = torchTensor2Buffer(attn_inputs.sequence_lengths);
    auto      input_lengths_host    = torchTensor2Buffer(attn_inputs.input_lengths);
    BufferPtr kv_cache_block_id_host, kv_cache_block_id_device;
    if (attn_inputs.kv_cache_block_id_host.size(0)) {
        kv_cache_block_id_host   = torchTensor2Buffer(attn_inputs.kv_cache_block_id_host);
        kv_cache_block_id_device = torchTensor2Buffer(attn_inputs.kv_cache_block_id_device);
    }
    auto                    params = FlashInferAttnParams::prepare(device_,
                                                attn_configs_,
                                                nullptr,
                                                sequence_lengths_host,
                                                input_lengths_host,
                                                kv_cache_block_id_host,
                                                kv_cache_block_id_device,
                                                torchDTypeToDataType(attn_inputs.dtype),
                                                false);
    FlashInferAttnParamsPtr attn_params(params, (FlashInferAttnParams*)params.get());
    RTP_LLM_CHECK_WITH_INFO(attn_params->decode_plan, "flash infer params should gen decode plan");
    return ParamsBasePtr(attn_params);
}

torch::Tensor FlashInferDecodeOp::forward(const torch::Tensor&              q,
                                          std::optional<torch_ext::KVCache> kv_cache,
                                          const FlashInferAttnParamsPtr&    params) {
    RTP_LLM_CHECK_WITH_INFO(params != nullptr, "flash infer op should have params");
    const int     local_head_num = attn_configs_.head_num;
    const int     size_per_head  = attn_configs_.size_per_head;
    const int     bs             = q.size(0);
    torch::Tensor output =
        torch::empty({bs, local_head_num * size_per_head}, torch::TensorOptions(q.dtype()).device(q.device()));
    auto softmax_scale = (1.0f / sqrtf(size_per_head * 1.0f)) * attn_configs_.softmax_extra_scale;
    RTP_LLM_LOG_DEBUG("decode flashinfer");
    torch::Tensor k_cache, v_cache;
    if (kv_cache.has_value()) {
        k_cache = kv_cache.value().k_cache_base.select(1, 0);
        v_cache = kv_cache.value().k_cache_base.select(1, 1);
    }

    BatchDecodeWithPagedKVCacheRun(params->float_workspace_d,         // float_workspace_buffer
                                   params->int_workspace_d,           // int_workspace_buffer
                                   params->plan,                      // plan_info_vec
                                   q,                                 // q
                                   k_cache,                           // paged_k_cache
                                   v_cache,                           // paged_v_cache
                                   params->page_indptr_d,             // paged_kv_indptr
                                   params->page_indice_d,             // paged_kv_indices
                                   params->paged_kv_last_page_len_d,  // paged_kv_last_page_len
                                   output,
                                   std::nullopt,  // maybe_lse
                                   1,             // kv_layout_code
                                   -1,            // window_left
                                   std::nullopt,  // maybe_alibi_slopes
                                   0,             // logits_soft_cap
                                   softmax_scale,
                                   0,
                                   0,
                                   (int64_t)device_->getStream());
    return output;
}

void registerFlashInferOp(const py::module& m) {
    pybind11::class_<FlashInferAttnParams, std::shared_ptr<FlashInferAttnParams>, rtp_llm::ParamsBase>(
        m, "FlashInferAttnParams")
        .def(pybind11::init<>());
    pybind11::class_<FlashInferPrefillOp>(m, "FlashInferPrefillOp")
        .def(pybind11::init<GptInitParameter>(), py::arg("gpt_init_parameter"))
        .def("support", &FlashInferPrefillOp::support, py::arg("attn_inputs"))
        .def("prepare", &FlashInferPrefillOp::prepare, py::arg("attn_inputs"))
        .def("forward", &FlashInferPrefillOp::forward, py::arg("q"), py::arg("kv_cache"), py::arg("params"));
    pybind11::class_<FlashInferDecodeOp>(m, "FlashInferDecodeOp")
        .def(pybind11::init<GptInitParameter>(), py::arg("gpt_init_parameter"))
        .def("support", &FlashInferDecodeOp::support, py::arg("attn_inputs"))
        .def("prepare", &FlashInferDecodeOp::prepare, py::arg("attn_inputs"))
        .def("forward", &FlashInferDecodeOp::forward, py::arg("q"), py::arg("kv_cache"), py::arg("params"));
}

}  // namespace rtp_llm
