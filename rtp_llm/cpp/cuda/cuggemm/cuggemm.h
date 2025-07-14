#pragma once

#include "trt_plugins/GroupGemmPlugin/GroupGemmPlugin.h"
#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace trt_plugins = tensorrt_llm::plugins;
namespace rtp_llm {

class cuggemm {

public:
    cuggemm()  = default;
    ~cuggemm() = default;

    void init(cudaStream_t stream) {
        stream_      = stream;
        half_runner_ = std::make_unique<trt_plugins::GroupGemmPlugin<half>>();
        bf16_runner_ = std::make_unique<trt_plugins::GroupGemmPlugin<__nv_bfloat16>>();
        fp32_runner_ = std::make_unique<trt_plugins::GroupGemmPlugin<float>>();
    }

    void setup(DataType dtype) {
        dtype_ = dtype;
    }

    void groupGemm(void**      A,
                   void**      B,
                   void**      C,
                   const int*  m,
                   const int*  n,
                   const int*  k,
                   const float alpha,
                   const float beta,
                   const int   count);

private:
    std::unique_ptr<trt_plugins::GroupGemmPlugin<half>>          half_runner_;
    std::unique_ptr<trt_plugins::GroupGemmPlugin<__nv_bfloat16>> bf16_runner_;
    std::unique_ptr<trt_plugins::GroupGemmPlugin<float>>         fp32_runner_;
    DataType                                                     dtype_;
    cudaStream_t                                                 stream_;
};

}  // namespace rtp_llm