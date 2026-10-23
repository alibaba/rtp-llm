#pragma once

#include <torch/extension.h>
#include <memory>

namespace torch_ext {

at::Tensor cublas_gemm_bf16_bf16_fp32(const at::Tensor& input, const at::Tensor& weight);

class V41CompactCpProjection {
public:
    explicit V41CompactCpProjection(int64_t device);
    ~V41CompactCpProjection();
    static bool supported(int64_t device);
    at::Tensor  forward(const at::Tensor& input, const at::Tensor& weight) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace torch_ext
