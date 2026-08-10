// Compat shim for CUDA 13 CUB.
//
// CCCL shipped with CUDA 13 dropped cub::Max / cub::Min / cub::Sum (and moved
// cub::CountingInputIterator / cub::TransformInputIterator to thrust::) so any
// rtp_llm first-party .cu file that still uses those names would fail to
// compile against CUDA 13.  Include this header AFTER `<cub/cub.cuh>` to pull
// aliases back into namespace cub for CUDA 13 only — on CUDA 12 the header is
// a no-op.
//
// Intentionally scoped to the symbols we actually reference in rtp_llm
// first-party code — see the matching call sites in:
//   rtp_llm/models_py/bindings/common/kernels/moe/moe_routing_kernels.cu
//   3rdparty/trt_beam_search/topkLastDim.cu
//   3rdparty/trt_fused_multihead_attention/common.cuh

#ifndef RTP_LLM_3RDPARTY_CUB_COMPAT_H_
#define RTP_LLM_3RDPARTY_CUB_COMPAT_H_

#if defined(__CUDACC_VER_MAJOR__) && __CUDACC_VER_MAJOR__ >= 13

#include <cuda/functional>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

namespace cub {

// Reduction functors: cub::Max / cub::Min / cub::Sum were the stateless
// functor structs taking two operands and returning the comparator/sum.
// cuda::maximum<> / cuda::minimum<> / cuda::std::plus<> match the semantics.
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
