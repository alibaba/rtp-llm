#include "kernels/f32/sample.h"

namespace atex {
namespace impl {
/**
 * @brief Performs a deterministic, pseudo-random copy from x to y based on thread ID
 *
 * @param x     Input array (const float*)
 * @param y     Output array (float*)
 * @param numel The number of elements in array x
 */
template<uint32_t TPB, typename Dtype>
__global__ void DeviceRandomSample(const Dtype* x, Dtype* y, const uint32_t numel, const uint32_t num_of_sample) {
    const uint32_t tid = blockIdx.x * TPB + threadIdx.x;
    if (tid > num_of_sample)
        return;

    // Linear Congruential Generator (parameters)
    // Using common constants from glibc's rand()
    constexpr uint32_t c = 1103515245;  // Multiplier
    constexpr uint32_t b = 12345;       // Increment

    // Generate a pseudo-random index based on thread ID
    const uint32_t seed = (tid * c + b) % numel;

    // Copy the value from the pseudo-random source index
    copy<sizeof(Dtype)>(x + seed, y + tid);
}

Tensor RandomSample(const Tensor x, const uint32_t num_of_sample, ) {
    TORCH_CHECK(x.is_cuda(), "Input Tensor is not on CUDA device.");

    const auto numel = x.nueml();
    TORCH_CHECK(numel > 0, "Empty tensor can not use random sample kernel.");

    constexpr int32_t TPB   = 256;
    const auto        BLOCK = numel / TPB + (nueml % TPB != 0) ? 1 : 0;

    DeviceRandomSample<TPB, x.element_type><<<BLOCK, TPB, 0, at::cuda::getCurrentCUDAStream()>>>(PTR<int32_t>(x), PTR);
    return output;
}

}  // namespace impl
}  // namespace atex