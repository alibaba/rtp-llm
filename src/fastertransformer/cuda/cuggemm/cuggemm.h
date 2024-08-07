#pragma once

#include "src/fastertransformer/cutlass/interface.h"
#include "src/fastertransformer/core/Types.h"


namespace fastertransformer{

class cuggemm {

public:
    cuggemm() = default;
    ~cuggemm() = default;

    void init(cudaStream_t stream) {
        stream_ = stream;
        half_runner_ = std::make_unique<CutlassGroupGemmRunner<half>>();
        bf16_runner_ = std::make_unique<CutlassGroupGemmRunner<__nv_bfloat16>>();
        fp32_runner_ = std::make_unique<CutlassGroupGemmRunner<float>>();
    }

    void setup(DataType dtype) {
        dtype_ = dtype;
    }

    void groupGemm(void**                           A,
                   void**                           B,
                   void**                           C,
                   const int*                       m,
                   const int*                       n,
                   const int*                       k,
                   const float                      alpha,
                   const float                      beta,
                   const int                        count);

private:
    std::unique_ptr<CutlassGroupGemmRunner<half>> half_runner_;
    std::unique_ptr<CutlassGroupGemmRunner<__nv_bfloat16>> bf16_runner_;
    std::unique_ptr<CutlassGroupGemmRunner<float>> fp32_runner_;
    DataType dtype_;
    cudaStream_t stream_;
};

} // namespace fastertransformer