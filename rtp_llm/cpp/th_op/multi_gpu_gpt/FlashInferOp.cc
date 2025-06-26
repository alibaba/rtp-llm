#include "rtp_llm/cpp/th_op/multi_gpu_gpt/FlashInferOp.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "3rdparty/flashinfer/flashinfer.h"
#include "rtp_llm/cpp/devices/cuda_impl/CudaFlashInfer.h"
#include "rtp_llm/cpp/devices/OpData.h"

namespace rtp_llm {

FlashInferOp::FlashInferOp(const GptInitParameter& gpt_init_parameter): configs(gpt_init_parameter), rope_config(gpt_init_parameter.getRopeConfig()) {}

void FlashInferOp::forward(torch::Tensor input, torch::Tensor output, torch::Tensor k_cache, torch::Tensor v_cache, py::object attn_params) {
    // std::unique_ptr<FlashInferAttnParams> params;
    assert(!attn_params.is_none());
    // const FlashInferAttnParams& params = attn_params.cast<FlashInferAttnParams>();
    const AttentionCommonInputs& attn_param = attn_params.cast<AttentionCommonInputs>();
    FlashInferAttnParams* params;
    if (attn_param.prefill_flash_infer_attn) {
        params = (FlashInferAttnParams*)attn_param.prefill_flash_infer_attn.get();
    } else {
        params = (FlashInferAttnParams*)attn_param.decode_flash_infer_attn.get();
    }
    assert(params);

    const int local_head_num = configs.head_num_;
    const int local_head_num_kv = configs.head_num_kv_;
    const int size_per_head = configs.size_per_head_;

    const int bs = input.size(0);
    const std::vector<int64_t> strides = {(local_head_num + 2 * local_head_num_kv) * size_per_head, size_per_head, 1};
    // const auto cuda_option = torch::dtype(input.dtype()).device(torch::DeviceType::CUDA).requires_grad(false);
    std::vector<torch::Tensor> qkv = input.split_with_sizes({local_head_num * size_per_head, local_head_num_kv * size_per_head, local_head_num_kv * size_per_head}, -1);
    auto q = qkv[0].reshape({bs, local_head_num, size_per_head});
    auto append_k = qkv[1].reshape({bs, local_head_num_kv, size_per_head});
    auto append_v = qkv[2].reshape({bs, local_head_num_kv, size_per_head});
    cudaStream_t stream = 0;

    if (rope_config.style == RopeStyle::Base) {
        apply_rope_pos_ids(q,
                           append_k,
                           q,
                           append_k,
                           params->positions_d,
                           (int64_t)rope_config.dim,
                           false,
                           (double)rope_config.scale,
                           (double)rope_config.base,
                           (int64_t)stream);
        check_cuda_error();
    }

    // Note: skip_append_kv_cache is only used for unit test
    bool skip_append_kv_cache = false;
    if (!skip_append_kv_cache) {
        if (append_k.type() != k_cache.type()) {
            append_k = append_k.to(k_cache.type());
            append_v = append_v.to(k_cache.type());
        }
        append_paged_kv_cache(append_k,
                              append_v,
                              params->batch_indice_d,
                              params->positions_d,
                              k_cache,
                              v_cache,
                              params->page_indice_d,
                              params->page_indptr_d,
                              params->paged_kv_last_page_len_d,
                              1,
                              (int64_t)stream);
    }

    // moe_insertion_callback();

    check_cuda_error();

    auto softmax_scale = (1.0f / sqrtf(size_per_head * 1.0f)) * configs.softmax_extra_scale_;

    if (params->decode_plan) {
        RTP_LLM_LOG_DEBUG("decode flashinfer");
        BatchDecodeWithPagedKVCacheRun(
                params->float_workspace_d, // float_workspace_buffer
                params->int_workspace_d, // int_workspace_buffer
                params->plan, // plan_info_vec
                q, // q
                k_cache, // paged_k_cache
                v_cache, // paged_v_cache
                params->page_indptr_d, // paged_kv_indptr
                params->page_indice_d, // paged_kv_indices
                params->paged_kv_last_page_len_d, // paged_kv_last_page_len
                output,
                std::nullopt, // maybe_lse
                1, // kv_layout_code
                -1, // window_left
                std::nullopt, // maybe_alibi_slopes
                0, // logits_soft_cap
                softmax_scale,
                0,
                0,
                (int64_t)stream);
    } else {
        RTP_LLM_LOG_DEBUG("prefill flashinfer");
        BatchPrefillWithPagedKVCacheRun(
                params->float_workspace_d, // float_workspace_buffer
                params->int_workspace_d,  // int_workspace_buffer
                params->plan, // plan_info_vec
                q, // q
                k_cache, // paged_k_cache
                v_cache, // paged_v_cache
                params->qo_indptr_d, // qo_indptr
                params->page_indptr_d, // paged_kv_indptr
                params->page_indice_d, // paged_kv_indices
                params->paged_kv_last_page_len_d, // paged_kv_last_page_len
                output,
                std::nullopt, // maybe_lse
                1, // mask_mode_code,
                1, // layout
                -1, // window_left
                std::nullopt, // maybe_custom_mask
                std::nullopt, // maybe_mask_indptr
                std::nullopt, // maybe_alibi_slopes
                0, // logits_soft_cap
                softmax_scale,
                rope_config.scale,
                rope_config.base,
                (int64_t)stream);
    }
    // if (params.configs.kv_cache_dtype == KvCacheDataType::FP8) {
    //     const auto &scale = params.weights.static_scale_reciprocal_weight;
    //     RTP_LLM_CHECK_WITH_INFO(scale != nullptr, "static_scale_reciprocal_weight is not set");
    //     auto scale_t = Buffer2torchTensor(scale->kernel, false);
    //     auto fp8_out = Buffer2torchTensor(params.output, false);
    //     fp8_out.copy_((scale_t * out).to(torch::kFloat8_e4m3fn));
    // }
}

void register_attn_params(pybind11::module& m) {
    pybind11::class_<AttentionCommonInputs>(m, "AttentionCommonInputs")
        .def(pybind11::init<>());
}


void registerFlashInferOp(const py::module& m) {
    pybind11::class_<FlashInferOp>(m, "FlashInferOp")
        .def(pybind11::init<GptInitParameter>(), py::arg("gpt_init_parameter"))
        .def("forward",
             &FlashInferOp::forward,
             py::arg("input"),
             py::arg("output"),
             py::arg("k_cache"),
             py::arg("v_cache"),
             py::arg("attn_params"));
}

}
