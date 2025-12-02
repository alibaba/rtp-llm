#pragma once

#include <memory>
#include "rtp_llm/cpp/config/GptInitParameter.h"
#include "rtp_llm/cpp/cuda/cufmha/TrtV2FmhaRunner.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"
#include "rtp_llm/models_py/bindings/cuda/FMHACudaBase.h"

namespace rtp_llm {

class TrtV2FmhaRunner;
struct TRTAttn;
using TRTAttnPtr = std::shared_ptr<TRTAttn>;

class TRTPrefillOpBase: public FMHACudaBase {
public:
    TRTPrefillOpBase(const GptInitParameter& gpt_init_parameter);
    virtual bool support(torch_ext::PyAttentionInputs attn_inputs) = 0;

    virtual ParamsBasePtr prepare(torch_ext::PyAttentionInputs attn_inputs);

    virtual torch::Tensor
    forward(const torch::Tensor& input, std::optional<torch_ext::KVCache> kv_cache, const TRTAttnPtr& params) = 0;

protected:
    std::shared_ptr<TrtV2FmhaRunner> trt_v2_runner_;
    torch::Tensor                    static_scale_;
};

class TRTPagedPrefillOp: public TRTPrefillOpBase {
public:
    using TRTPrefillOpBase::TRTPrefillOpBase;
    bool support(torch_ext::PyAttentionInputs attn_inputs) override;
    torch::Tensor
    forward(const torch::Tensor& input, std::optional<torch_ext::KVCache> kv_cache, const TRTAttnPtr& params) override;
};

class TRTNormalPrefillOp: public TRTPrefillOpBase {
public:
    using TRTPrefillOpBase::TRTPrefillOpBase;
    bool support(torch_ext::PyAttentionInputs attn_inputs) override;
    torch::Tensor
    forward(const torch::Tensor& input, std::optional<torch_ext::KVCache> kv_cache, const TRTAttnPtr& params) override;
};

void registerTRTAttnOp(const py::module& m);

}  // namespace rtp_llm
