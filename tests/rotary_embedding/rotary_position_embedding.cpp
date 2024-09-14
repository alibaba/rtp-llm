
#include "src/fastertransformer/kernels/rotary_position_embedding.h"
#include "torch/csrc/cuda/Stream.h"
#include "torch/extension.h"
#include <ATen/cuda/CUDAContext.h>

using namespace fastertransformer;
namespace unittest {

using _4DTensor = at::PackedTensorAccessor32<float,4,at::RestrictPtrTraits>;
__global__ void
KernelWrapper(_4DTensor input, RopeStyle style, int dim, int base, float scale, int max_pos, float mscale, int offset) {
    extern __shared__ __align__(sizeof(float2)) char smem[];  // align on largest vector type
    const int batch_idx = blockIdx.x / input.size(1);
    const int seq_idx = blockIdx.x % input.size(1);
    const int headnum_idx = blockIdx.y;
    const int headsize_idx = threadIdx.x;
    const bool work = (batch_idx < input.size(0)) && (seq_idx < input.size(1)) &&
                           (headnum_idx < input.size(2)) && (headsize_idx * 2 < input.size(3));

    RopeConfig rope_config;
    rope_config.style   = style;
    rope_config.dim     = dim;
    rope_config.base    = base;
    rope_config.scale   = scale;
    rope_config.max_pos = max_pos;
    rope_config.mscale  = mscale;
    rope_config.offset  = offset;

    if (style == RopeStyle::Yarn) {
        rope_config.factor1 = 1;
        rope_config.factor2 = 32;
    } else if (style == RopeStyle::Llama3) {
        rope_config.factor1 = 1;
        rope_config.factor2 = 4;
    }
    if (work) {
        float2 x;
        x = *reinterpret_cast<float2*>(&input[batch_idx][seq_idx][headnum_idx][headsize_idx * 2]);
        FT_ROPE_SWITCH(rope_config.style, ROPE_STYLE, [&]{
            apply_rope<float, float2, ROPE_STYLE>(rope_config, x, (float*)smem, headsize_idx, seq_idx, input.size(1));
        });
        *reinterpret_cast<float2*>(&input[batch_idx][seq_idx][headnum_idx][headsize_idx * 2]) = x;
    }
}

class RotaryPositionEmbeddingOp: public torch::jit::CustomClassHolder {
public:
    RotaryPositionEmbeddingOp(int64_t dim, int64_t max_pos, int64_t base, double scale, int64_t style, double mscale):
        dim(dim), max_pos(max_pos), base(base), scale(scale), style(style), mscale(mscale) {}

    torch::Tensor forward(torch::Tensor input, int64_t offset);

private:
    int64_t dim;
    int64_t base;
    double scale = 1.0;
    int64_t max_pos = 2048;
    int64_t style = 0;
    double mscale = 1.0;
};

torch::Tensor RotaryPositionEmbeddingOp::forward(torch::Tensor input, int64_t offset) {
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    dim3   block((input.size(3) + 31) / 32 * 32);
    dim3   grid(input.size(0) * input.size(1), input.size(2));

    size_t smem_size = dim * sizeof(float);

    KernelWrapper<<<grid, block, smem_size, stream>>>(
            input.packed_accessor32<float, 4 ,at::RestrictPtrTraits>(),
            RopeStyle(style), dim, base, scale, max_pos, mscale, offset);

    torch::Tensor output = input.detach().clone();
    return output;
}

}  // namespace unittest

static auto RotaryPositionEmbeddingTHS =
torch::jit::class_<unittest::RotaryPositionEmbeddingOp>("unittest", "RotaryPositionEmbeddingOp")
.def(torch::jit::init<int64_t, int64_t, int64_t, double, int64_t, double>())
.def("forward", &unittest::RotaryPositionEmbeddingOp::forward);
