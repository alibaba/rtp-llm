#pragma once

#include <memory>
#include <unordered_map>
#include "rtp_llm/cpp/devices/cuda_impl/CudaFlashInfer.h"
#include "rtp_llm/cpp/config/GptInitParameter.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"
#include "rtp_llm/models_py/bindings/cuda/FMHACudaBase.h"

namespace rtp_llm {

class MlaAttentionCudaBase: public FMHACudaBase {
public:
    MlaAttentionCudaBase(const GptInitParameter& gpt_init_parameter);
    bool                    support(torch_ext::PyAttentionInputs attn_inputs);
    FlashInferAttnParamsPtr prepare(torch_ext::PyAttentionInputs attn_inputs);

protected:
    bool                    is_prefill_          = true;
    size_t                  max_prefix_length_   = 0;
    size_t                  context_batch_size_  = 1;
    size_t                  decoder_batch_size_  = 0;
    size_t                  context_token_num_   = 0;
    size_t                  max_context_seq_len_ = 0;
    torch::Tensor           cu_seqlens_;
    std::shared_ptr<cufmha> cufmha_runner_;
    bool                    use_mla_ = true;
};

class MlaContextAttentionOp: public MlaAttentionCudaBase {
public:
    MlaContextAttentionOp(const GptInitParameter& gpt_init_parameter);
    bool support(torch_ext::PyAttentionInputs attn_inputs);

    torch::Tensor forward(torch::Tensor&                 q,
                          torch::Tensor&                 kv_a,
                          torch::Tensor&                 k_rope,
                          const int64_t                  kv_offset,
                          const FlashInferAttnParamsPtr& params,
                          const torch::Tensor&           k_nope_weight,
                          const torch::Tensor&           v_weight);

private:
    std::unique_ptr<tensorrt_llm::kernels::FusedMHARunnerV2> trtv2_fmha_runner_;
};

class MlaAbsorbAttentionOp: public MlaAttentionCudaBase {
public:
    MlaAbsorbAttentionOp(const GptInitParameter& gpt_init_parameter);
    bool support(torch_ext::PyAttentionInputs attn_inputs);

    torch::Tensor forward(torch::Tensor&                    q,
                          torch::Tensor&                    fused_q_input_t,
                          std::optional<torch_ext::KVCache> kv_cache,
                          const FlashInferAttnParamsPtr&    params,
                          const torch::Tensor&              kc_weight,
                          const torch::Tensor&              vc_weight);
};

void registerMlaAttentionOp(const py::module& m);

}  // namespace rtp_llm
