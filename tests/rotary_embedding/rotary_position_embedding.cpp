
#include "src/fastertransformer/kernels/rotary_position_embedding.h"
#include "torch/csrc/cuda/Stream.h"
#include "torch/extension.h"
#include <ATen/cuda/CUDAContext.h>

using namespace fastertransformer;
namespace unittest {


template<RotaryEmbeddingStyle style>
__global__ void KernelWrapper(
    at::PackedTensorAccessor32<float,4,at::RestrictPtrTraits> input,
    const int dim, const float base, const float scalar,
    int max_position_embeddings,
    int max_logn_seq_len) {

    extern __shared__ __align__(sizeof(float2)) char smem[];  // align on largest vector type
    const int batch_idx = blockIdx.x / input.size(1);
    const int seq_idx = blockIdx.x % input.size(1);
    const int headnum_idx = blockIdx.y;
    const int headsize_idx = threadIdx.x;
    const bool work = (batch_idx < input.size(0)) && (seq_idx < input.size(1)) &&
                           (headnum_idx < input.size(2)) && (headsize_idx * 2 < input.size(3));

    if (work) {
        float2 x;
        x = *reinterpret_cast<float2*>(&input[batch_idx][seq_idx][headnum_idx][headsize_idx * 2]);
        if constexpr (style == RotaryEmbeddingStyle::LinearScalar) {
            fastertransformer::Rope<float, float2, style>::impl(x, (float*)smem, headsize_idx, seq_idx, dim, base, scalar);
        } else if constexpr (style == RotaryEmbeddingStyle::NTKScalar) {
            fastertransformer::Rope<float, float2, style>::impl(x, (float*)smem, headsize_idx, seq_idx, dim, base, scalar, 
                                                        input.size(1), max_position_embeddings);
        } else if constexpr (style == RotaryEmbeddingStyle::QWenNTKScalar) {
            fastertransformer::Rope<float, float2, style>::impl(x, (float*)smem, headsize_idx, seq_idx, dim, base, scalar, 
                                                        input.size(1), max_logn_seq_len);
        }
        
        *reinterpret_cast<float2*>(&input[batch_idx][seq_idx][headnum_idx][headsize_idx * 2]) = x;

    }
}

class RotaryPositionEmbeddingOp: public torch::jit::CustomClassHolder {
public:
    RotaryPositionEmbeddingOp(int64_t dim, int64_t max_position_embeddings,
                              double base, double scalar, int64_t max_logn_seq_len,
                              int64_t style):
        dim(dim),max_position_embeddings(max_position_embeddings),
        base(base),scalar(scalar),style(style),
        max_logn_seq_len(max_logn_seq_len) {};

    torch::Tensor forward(torch::Tensor input);

private:
    int64_t dim;
    double base;
    double scalar = 1.0;
    int64_t max_position_embeddings = 2048;
    int64_t style = 0;
    int64_t max_logn_seq_len = 2048;
};

torch::Tensor RotaryPositionEmbeddingOp::forward(torch::Tensor input) {
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    dim3   block((input.size(3) + 31) / 32 * 32);
    dim3   grid(input.size(0) * input.size(1), input.size(2));

    size_t smem_size = dim * sizeof(float);

    cudaEvent_t start, stop;
    float elapsedTime;

    cudaEventCreate(&start);
    cudaEventRecord(start,0);

    switch (style)
    {
    case 0:
        break;
    
    case 1:
        KernelWrapper<RotaryEmbeddingStyle::LinearScalar><<<grid, block, smem_size, stream>>>(
            input.packed_accessor32<float,4,at::RestrictPtrTraits>(), 
            dim, base, scalar, max_position_embeddings, max_logn_seq_len);
        break;
    
    case 2:
        KernelWrapper<RotaryEmbeddingStyle::NTKScalar><<<grid, block, smem_size, stream>>>(
            input.packed_accessor32<float,4,at::RestrictPtrTraits>(), 
            dim, base, scalar, max_position_embeddings, max_logn_seq_len);
        break;
    
    case 3:
        KernelWrapper<RotaryEmbeddingStyle::QWenNTKScalar><<<grid, block, smem_size, stream>>>(
            input.packed_accessor32<float,4,at::RestrictPtrTraits>(), 
            dim, base, scalar, max_position_embeddings, max_logn_seq_len);
        break;
    
    default:
        break;
    }

    cudaEventCreate(&stop);
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTime, start,stop);
    printf("Input shape : [%d, %d, %d, %d]\n" ,input.size(0), input.size(1), input.size(2), input.size(3));
    printf("Elapsed time : %f ms\n" ,elapsedTime);

    
    torch::Tensor output = input.detach().clone();
    return output;
}

}  // namespace unittest

static auto RotaryPositionEmbeddingTHS =
    torch::jit::class_<unittest::RotaryPositionEmbeddingOp>("unittest", "RotaryPositionEmbeddingOp")
        .def(torch::jit::init<int64_t, int64_t, int64_t, double, int64_t, int64_t>())
        .def("forward", &unittest::RotaryPositionEmbeddingOp::forward);
