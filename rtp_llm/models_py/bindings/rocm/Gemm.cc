#include "rtp_llm/models_py/bindings/rocm/Gemm.h"
#include "rtp_llm/cpp/rocm/hip_host_utils.h"
#include <hipblas/hipblas.h>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include "rtp_llm/cpp/core/torch_utils/TypeConvert.h"

namespace rtp_llm {

void gemm(at::Tensor& output, at::Tensor& input, at::Tensor& weight) {
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    auto device = input.device();
    CHECK_EQ(weight.device(), device);
    CHECK_DIM(2, input);
    CHECK_DIM(2, weight);
    CHECK_EQ(input.size(1), weight.size(0));
    CHECK_EQ(input.size(0), output.size(0));
    CHECK_EQ(weight.size(1), output.size(1));

    // Use torch::mm for gemm on ROCm
    output.copy_(torch::mm(input, weight));
}

}  // namespace rtp_llm
