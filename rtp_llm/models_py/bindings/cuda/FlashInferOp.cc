#include "rtp_llm/models_py/bindings/cuda/FlashInferOp.h"
#include "rtp_llm/models_py/bindings/core/torch_utils/TypeConvert.h"
#include "3rdparty/flashinfer/flashinfer.h"
#include "rtp_llm/models_py/bindings/cuda/ops/CudaFlashInfer.h"
#include "rtp_llm/models_py/bindings/core/OpData.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/models_py/bindings/common/Torch_ext.h"

using namespace torch_ext;

namespace rtp_llm {

FlashInferPrefillOp::FlashInferPrefillOp(const AttentionConfigs& attn_configs,
                                         MlaOpsType              mla_ops_type,
                                         bool                    enable_cuda_graph):
    attn_configs_(attn_configs), mla_ops_type_(mla_ops_type), enable_cuda_graph_(enable_cuda_graph) {}

bool FlashInferPrefillOp::support(torch_ext::PyAttentionInputs attn_inputs) {
    // TODO: if (fmha_config_.disable_flash_infer || attn_configs_.kv_cache_dtype == KvCacheDataType::INT8
    // || attn_inputs.prefix_lengths.max().item<int32_t>() > 0) {

    if (attn_configs_.kv_cache_dtype != KvCacheDataType::BASE) {
        return false;
    }
    DataType dtype = torchDTypeToDataType(attn_inputs.dtype);
    if (attn_configs_.kv_cache_dtype == KvCacheDataType::FP8) {
        dtype = DataType::TYPE_FP8_E4M3;
    }
    return FlashInferAttnParams::checkPrefill(
        attn_configs_, attn_inputs.prefix_lengths, attn_inputs.input_lengths, dtype, false);
}

ParamsBasePtr FlashInferPrefillOp::prepare(torch_ext::PyAttentionInputs attn_inputs) {
    DataType dtype = torchDTypeToDataType(attn_inputs.dtype);
    if (attn_configs_.kv_cache_dtype == KvCacheDataType::FP8) {
        dtype = DataType::TYPE_FP8_E4M3;
    }
    auto params = FlashInferAttnParams::prepare(attn_configs_,
                                                attn_inputs.prefix_lengths,
                                                attn_inputs.sequence_lengths,
                                                attn_inputs.input_lengths,
                                                attn_inputs.kv_cache_kernel_block_id_host,
                                                attn_inputs.kv_cache_kernel_block_id_device,
                                                dtype,
                                                mla_ops_type_,
                                                enable_cuda_graph_,
                                                false);
    RTP_LLM_CHECK_WITH_INFO(params != nullptr, "unsupported or empty FlashInfer prefill plan");
    FlashInferAttnParamsPtr attn_params(params, (FlashInferAttnParams*)params.get());
    RTP_LLM_CHECK_WITH_INFO(!attn_params->decode_plan, "flash infer params should gen prefill plan");
    return ParamsBasePtr(attn_params);
}

torch::Tensor FlashInferPrefillOp::forward(const torch::Tensor&                   q,
                                           std::optional<torch_ext::LayerKVCache> kv_cache,
                                           const FlashInferAttnParamsPtr&         params) {
    RTP_LLM_CHECK_WITH_INFO(params != nullptr, "flash infer op should have params");

    const int local_head_num = attn_configs_.head_num;
    const int size_per_head  = attn_configs_.size_per_head;
    const int bs             = q.size(0);
    const int kv_heads       = attn_configs_.kv_head_num;
    RTP_LLM_CHECK_WITH_INFO(q.is_cuda() && (q.dim() == 2 || q.dim() == 3) && size_per_head > 0 && bs > 0,
                            "FlashInfer prefill expects CUDA token-major Q or packed QKV");
    RTP_LLM_CHECK_WITH_INFO(!params->decode_plan && params->qo_indptr_h.defined() && params->qo_indptr_h.numel() > 0
                                && params->qo_indptr_h.data_ptr<int>()[params->qo_indptr_h.numel() - 1] == bs,
                            "FlashInfer prefill query row count must match its plan");
    const auto heads = q.numel() / (static_cast<int64_t>(bs) * size_per_head);
    RTP_LLM_CHECK_WITH_INFO(bs > 0 && q.numel() == static_cast<int64_t>(bs) * heads * size_per_head
                                && (heads == local_head_num || heads == local_head_num + 2 * kv_heads),
                            "FlashInfer prefill Q/QKV shape does not match its attention head geometry");
    const auto token_heads = q.reshape({bs, heads, size_per_head});
    const auto query       = token_heads.narrow(1, 0, local_head_num);
    RTP_LLM_CHECK_WITH_INFO(params->ragged_kv == !kv_cache.has_value(),
                            "FlashInfer prefill KV storage must match its ragged/paged plan");
    torch::Tensor output =
        torch::empty({bs, local_head_num * size_per_head}, torch::TensorOptions(q.dtype()).device(q.device()));
    auto       softmax_scale = (1.0f / sqrtf(size_per_head * 1.0f)) * attn_configs_.softmax_extra_scale;
    StreamType stream        = GET_CURRENT_STREAM();
    if (params->ragged_kv) {
        RTP_LLM_CHECK_WITH_INFO(heads == local_head_num + 2 * kv_heads,
                                "cacheless FlashInfer prefill requires packed QKV including current K and V");
        const auto key   = token_heads.narrow(1, local_head_num, kv_heads);
        const auto value = token_heads.narrow(1, local_head_num + kv_heads, kv_heads);
        BatchPrefillWithRaggedKVCacheRun(params->float_workspace_d,
                                         params->int_workspace_d,
                                         params->plan,
                                         query,
                                         key,
                                         value,
                                         params->qo_indptr_d,
                                         params->page_indptr_d,
                                         output,
                                         std::nullopt,
                                         1,  // causal
                                         0,  // NHD
                                         -1,
                                         std::nullopt,
                                         std::nullopt,
                                         std::nullopt,
                                         0,
                                         softmax_scale,
                                         attn_configs_.rope_config.scale,
                                         attn_configs_.rope_config.base,
                                         (int64_t)stream);
        return output;
    }
    RTP_LLM_LOG_DEBUG("prefill flashinfer");
    torch::Tensor k_cache, v_cache;
    if (kv_cache.has_value()) {
        k_cache = kv_cache.value().kv_cache_base.select(1, 0);
        v_cache = kv_cache.value().kv_cache_base.select(1, 1);
    }
    BatchPrefillWithPagedKVCacheRun(params->float_workspace_d,         // float_workspace_buffer
                                    params->int_workspace_d,           // int_workspace_buffer
                                    params->plan,                      // plan_info_vec
                                    query,                             // q
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
                                    (int64_t)stream);
    return output;
}

FlashInferDecodeOp::FlashInferDecodeOp(const AttentionConfigs& attn_configs,
                                       MlaOpsType              mla_ops_type,
                                       bool                    enable_cuda_graph):
    attn_configs_(attn_configs), mla_ops_type_(mla_ops_type), enable_cuda_graph_(enable_cuda_graph) {}

bool FlashInferDecodeOp::support(torch_ext::PyAttentionInputs attn_inputs) {
    if (attn_configs_.kv_cache_dtype != KvCacheDataType::BASE) {
        return false;
    }
    // FIXME: FlashInferDecodeOp causes crash in this case, temporarily bypassing it here
    if (attn_configs_.head_num / attn_configs_.kv_head_num == 12) {
        return false;
    }
    return FlashInferAttnParams::checkDecode(attn_configs_, torchDTypeToDataType(attn_inputs.dtype));
}

ParamsBasePtr FlashInferDecodeOp::prepare(torch_ext::PyAttentionInputs attn_inputs) {
    auto params = FlashInferAttnParams::prepare(attn_configs_,
                                                torch::Tensor(),
                                                attn_inputs.sequence_lengths,
                                                attn_inputs.input_lengths,
                                                attn_inputs.kv_cache_kernel_block_id_host,
                                                attn_inputs.kv_cache_kernel_block_id_device,
                                                torchDTypeToDataType(attn_inputs.dtype),
                                                mla_ops_type_,
                                                enable_cuda_graph_,
                                                false);
    RTP_LLM_CHECK_WITH_INFO(params != nullptr, "unsupported or empty FlashInfer decode plan");
    FlashInferAttnParamsPtr attn_params(params, (FlashInferAttnParams*)params.get());
    RTP_LLM_CHECK_WITH_INFO(attn_params->decode_plan, "flash infer params should gen decode plan");
    return ParamsBasePtr(attn_params);
}

torch::Tensor FlashInferDecodeOp::forward(const torch::Tensor&                   q,
                                          std::optional<torch_ext::LayerKVCache> kv_cache,
                                          const FlashInferAttnParamsPtr&         params) {
    RTP_LLM_CHECK_WITH_INFO(params != nullptr, "flash infer op should have params");
    RTP_LLM_CHECK_WITH_INFO(kv_cache.has_value(), "FlashInfer decode requires allocated KV storage");
    const int     local_head_num = attn_configs_.head_num;
    const int     size_per_head  = attn_configs_.size_per_head;
    const int     bs             = q.size(0);
    torch::Tensor output =
        torch::empty({bs, local_head_num * size_per_head}, torch::TensorOptions(q.dtype()).device(q.device()));
    auto softmax_scale = (1.0f / sqrtf(size_per_head * 1.0f)) * attn_configs_.softmax_extra_scale;
    RTP_LLM_LOG_DEBUG("decode flashinfer");
    torch::Tensor k_cache, v_cache;
    if (kv_cache.has_value()) {
        k_cache = kv_cache.value().kv_cache_base.select(1, 0);
        v_cache = kv_cache.value().kv_cache_base.select(1, 1);
    }
    StreamType stream = GET_CURRENT_STREAM();
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
                                   (int64_t)stream);
    return output;
}

void registerFlashInferOp(const py::module& m) {
    pybind11::class_<FlashInferAttnParams, std::shared_ptr<FlashInferAttnParams>, rtp_llm::ParamsBase>(
        m, "FlashInferAttnParams")
        .def(pybind11::init<>());
    pybind11::class_<FlashInferPrefillOp>(m, "FlashInferPrefillOp")
        .def(pybind11::init<const AttentionConfigs&, MlaOpsType, bool>(),
             py::arg("attn_configs"),
             py::arg("mla_ops_type")      = MlaOpsType::AUTO,
             py::arg("enable_cuda_graph") = false)
        .def("support", &FlashInferPrefillOp::support, py::arg("attn_inputs"))
        .def("prepare", &FlashInferPrefillOp::prepare, py::arg("attn_inputs"))
        .def("forward", &FlashInferPrefillOp::forward, py::arg("q"), py::arg("kv_cache"), py::arg("params"));
    pybind11::class_<FlashInferDecodeOp>(m, "FlashInferDecodeOp")
        .def(pybind11::init<const AttentionConfigs&, MlaOpsType, bool>(),
             py::arg("attn_configs"),
             py::arg("mla_ops_type")      = MlaOpsType::AUTO,
             py::arg("enable_cuda_graph") = false)
        .def("support", &FlashInferDecodeOp::support, py::arg("attn_inputs"))
        .def("prepare", &FlashInferDecodeOp::prepare, py::arg("attn_inputs"))
        .def("forward", &FlashInferDecodeOp::forward, py::arg("q"), py::arg("kv_cache"), py::arg("params"));
}

}  // namespace rtp_llm
