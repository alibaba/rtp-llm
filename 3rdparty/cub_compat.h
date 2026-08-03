// Compat shim for CUDA 13 CUB.
//
// CCCL shipped with CUDA 13 dropped cub::Max / cub::Min / cub::Sum (and moved
// cub::CountingInputIterator / cub::TransformInputIterator to thrust::) so any
// rtp_llm first-party .cu file that still uses those names would fail to
// compile against CUDA 13.  Include this header AFTER `<cub/cub.cuh>` to pull
// aliases back into namespace cub for CUDA 13 only — on CUDA 12 the header is
// a no-op.
//
// Intentionally scoped to symbols that first-party code references and CUDA 13
// removed. CUDA 13 CCCL still provides cub::ArgMax and cub::KeyValuePair, so
// they are not backfilled. The legacy bundled 3rdparty/cub branch does not use
// this shim. See the matching CUDA 13 call sites in:
//   rtp_llm/models_py/bindings/common/kernels/moe/moe_routing_kernels.cu
//   3rdparty/trt_beam_search/topkLastDim.cu
// FlashInfer is patched in its external repository and cannot include this
// header; keep its cub::Max / cub::Min definitions structurally equivalent to
// 3rdparty/flashinfer/0011-cuda13-cub-compat.patch.

#ifndef RTP_LLM_3RDPARTY_CUB_COMPAT_H_
#define RTP_LLM_3RDPARTY_CUB_COMPAT_H_

#if defined(__CUDACC_VER_MAJOR__) && __CUDACC_VER_MAJOR__ >= 13

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

namespace cub {

// These functors are semantically equivalent to cuda::maximum<> /
// cuda::minimum<> / cuda::std::plus<>. Keep the handwritten definitions
// structurally aligned with the FlashInfer patch referenced above.
struct Max {
    template <typename T>
    __host__ __device__ __forceinline__ T operator()(const T& a, const T& b) const {
        return (a < b) ? b : a;
    }
};
struct Min {
    template <typename T>
    __host__ __device__ __forceinline__ T operator()(const T& a, const T& b) const {
        return (b < a) ? b : a;
    }
};
struct Sum {
    template <typename T>
    __host__ __device__ __forceinline__ T operator()(const T& a, const T& b) const {
        return a + b;
    }
};

// Iterators: old cub types redirected to thrust equivalents.  thrust's
// transform_iterator deduces its value type from the unary functor's return,
// so the leading ValueT template arg from the cub signature is unused here.
template <typename T>
using CountingInputIterator = ::thrust::counting_iterator<T>;

template <typename ValueT, typename ConversionOp, typename InputIteratorT>
using TransformInputIterator = ::thrust::transform_iterator<ConversionOp, InputIteratorT>;

}  // namespace cub

#endif  // __CUDACC_VER_MAJOR__ >= 13

#endif  // RTP_LLM_3RDPARTY_CUB_COMPAT_H_
