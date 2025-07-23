#include "rtp_llm/models_py/bindings/cuda/Torch_ext.h"
#include "rtp_llm/cpp/kernels/gpt_kernels.h"
#include <cuda_bf16.h>
#include <cuda_device_runtime_api.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <iostream>
#include <type_traits>
#include <vector>
#include <ATen/cuda/CUDAContext.h>
using namespace std;
namespace th = torch;
using namespace rtp_llm;

namespace torch_ext {

void embedding(at::Tensor& output, at::Tensor& input, at::Tensor& weight) {
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    auto device = input.device();
    CHECK_EQ(weight.device(), device);
    CHECK_DIM(1, input);   // input: (tokens)
    CHECK_DIM(2, weight);  // weight: (hidden_size, hidden_size)
    const int tokens      = input.size(0);
    const int hidden_size = weight.size(1);
    CHECK_EQ(output.size(0), tokens);
    CHECK_EQ(output.size(1), hidden_size);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(at::cuda::current_device()).stream();
    DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(weight.scalar_type(), c_type, [&] {
        const int vecSize = sizeof(float4) / sizeof(c_type);
        // std::cout << "vecSize: " << vecSize << ", weight shape: " << weight.sizes() << std::endl;
        // std::cout << "tokens: " << tokens << std::endl;
        // std::cout << "hidden_size: " << hidden_size << std::endl;
        // std::cout << "input: " << input << std::endl;
        if (hidden_size % vecSize == 0) {
            invokeEmebeddingLookupVec(static_cast<c_type*>(output.data_ptr()),
                                      static_cast<const c_type*>(weight.data_ptr()),
                                      1.0,
                                      static_cast<const c_type*>(nullptr),  // postition_table
                                      static_cast<const c_type*>(nullptr),  // token_type_table
                                      static_cast<const int*>(input.data_ptr()),
                                      static_cast<const int*>(nullptr),  // position_ids
                                      static_cast<const int*>(nullptr),  // token_types
                                      static_cast<const int*>(nullptr),  // mask
                                      tokens,
                                      hidden_size,
                                      stream);
        } else {
            invokeEmebeddingLookup(static_cast<c_type*>(output.data_ptr()),
                                   static_cast<const c_type*>(weight.data_ptr()),
                                   1.0,
                                   static_cast<const c_type*>(nullptr),  // postition_table
                                   static_cast<const c_type*>(nullptr),  // token_type_table
                                   static_cast<const int*>(input.data_ptr()),
                                   static_cast<const int*>(nullptr),  // position_ids
                                   static_cast<const int*>(nullptr),  // token_types
                                   static_cast<const int*>(nullptr),  // mask
                                   tokens,
                                   hidden_size,
                                   stream);
        }
        return true;
    });
}
}  // namespace torch_ext