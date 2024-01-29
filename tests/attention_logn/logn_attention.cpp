#include "src/fastertransformer/kernels/decoder_masked_multihead_attention.h"
#include "src/fastertransformer/kernels/decoder_masked_multihead_attention_utils.h"

#include "src/fastertransformer/cuda/cuda_fp8_utils.h"
#include "src/fastertransformer/cuda/cuda_type_utils.cuh"
#include "torch/csrc/cuda/Stream.h"
#include "torch/extension.h"
#include <ATen/cuda/CUDAContext.h>

namespace th = torch;
namespace unittest {
__global__ void logn_attention_kernel(
    at::PackedTensorAccessor32<float,4,at::RestrictPtrTraits> input,
    at::PackedTensorAccessor32<float,4,at::RestrictPtrTraits> output,
    const int logn_seq_len) {
    extern __shared__ __align__(sizeof(float2)) float shared_memory[];
    const int batch_idx = blockIdx.x;
    const int seq_idx = blockIdx.y / input.size(2);
    const int headnum_idx = blockIdx.y % input.size(2);
    const int headsize_idx = threadIdx.x * 2;
    const bool work = (batch_idx < input.size(0)) && (seq_idx < input.size(1)) &&
                           (headnum_idx < input.size(2)) && (headsize_idx < input.size(3));
    float2 query_vec;
    if (work) {
        query_vec.x = input[batch_idx][seq_idx][headnum_idx][headsize_idx];
        query_vec.y = input[batch_idx][seq_idx][headnum_idx][headsize_idx+1];
        fastertransformer::logn_attention(query_vec, seq_idx, logn_seq_len);
        output[batch_idx][seq_idx][headnum_idx][headsize_idx] = query_vec.x;
        output[batch_idx][seq_idx][headnum_idx][headsize_idx+1] = query_vec.y;
    }
}
class LognAttentionOp: public torch::jit::CustomClassHolder {
public:
    LognAttentionOp(int64_t logn_seq_len):logn_seq_len(logn_seq_len){};
    torch::Tensor forward(torch::Tensor input);
private:
    int64_t logn_seq_len;
};
torch::Tensor LognAttentionOp::forward(torch::Tensor input) {
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    dim3   block((input.size(3) + 31) / 32 * 32);
    dim3   grid(input.size(0), input.size(1)*input.size(2));
    torch::Tensor output = torch::zeros_like(input);
    logn_attention_kernel<<<grid, block, 0, stream>>>(
        input.packed_accessor32<float,4,at::RestrictPtrTraits>(), 
        output.packed_accessor32<float,4,at::RestrictPtrTraits>(),
        logn_seq_len);
    return output;
}
}  // namespace unittest

static auto LognAttentionTHS =
    torch::jit::class_<unittest::LognAttentionOp>("unittest", "LognAttentionOp")
        .def(torch::jit::init<int64_t>())
        .def("forward", &unittest::LognAttentionOp::forward);
