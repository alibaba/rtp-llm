
#include "rtp_llm/cpp/kernels/gpt_kernels.h"
#include "rtp_llm/models_py/bindings/common/Torch_ext.h"
#include <cstdint>
#include <iostream>
#include <type_traits>
#include <vector>
#if USING_CUDA
#include <ATen/cuda/CUDAContext.h>
#endif
#if USING_ROCM
#include <hip/hip_runtime.h>
#include "rtp_llm/cpp/devices/rocm_impl/ROCmDevice.h"
#endif
using namespace std;
namespace th = torch;
using namespace rtp_llm;
namespace rtp_llm {

void embedding(at::Tensor& output, at::Tensor& input, at::Tensor& weight)
{
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    auto device = input.device();
    CHECK_EQ(weight.device(), device);
    CHECK_DIM(1, input);   // input: (tokens)
    CHECK_DIM(2, weight);  // weight: (hidden_size, hidden_size)
    const int tokens = input.size(0);
    const int hidden_size = weight.size(1);
    CHECK_EQ(output.size(0), tokens);
    CHECK_EQ(output.size(1), hidden_size);
#if USING_ROCM
    hipStream_t stream = at::hip::getCurrentHIPStream().stream();
#endif
#if USING_CUDA
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(at::cuda::current_device()).stream();
#endif
    DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(weight.scalar_type(), c_type, [&] {

        const int vecSize = sizeof(float4) / sizeof(c_type);
        if (hidden_size % vecSize == 0) {
            invokeEmebeddingLookupVec(static_cast<c_type*>(output.data_ptr()),
                static_cast<const c_type*>(weight.data_ptr()),
                1.0,
                static_cast<const c_type*>(nullptr), // postition_table
                static_cast<const c_type*>(nullptr), // token_type_table
                static_cast<const int*>(input.data_ptr()),
                static_cast<const int*>(nullptr), // position_ids
                static_cast<const int*>(nullptr), // token_types
                static_cast<const int*>(nullptr), // mask
                tokens, hidden_size, stream);
        } else {
            invokeEmebeddingLookup(static_cast<c_type*>(output.data_ptr()),
                static_cast<const c_type*>(weight.data_ptr()),
                1.0,
                static_cast<const c_type*>(nullptr), // postition_table
                static_cast<const c_type*>(nullptr), // token_type_table
                static_cast<const int*>(input.data_ptr()),
                static_cast<const int*>(nullptr), // position_ids
                static_cast<const int*>(nullptr), // token_types
                static_cast<const int*>(nullptr), // mask
                tokens, hidden_size, stream);
        }
        return true;
    });
}
}
