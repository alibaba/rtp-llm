#include "trt_plugins/GroupGemmPlugin/GroupGemmPlugin.h"
#include <cuda_fp16.h>
namespace tensorrt_llm::plugins {
template<typename T>
GroupGemmPlugin<T>::GroupGemmPlugin() {
    group_gemm_runner_ = std::make_unique<rtp_llm::CutlassGroupGemmRunner<T>>();
}
template<typename T>
void GroupGemmPlugin<T>::gemm(T**          A,
                              T**          B,
                              T**          C,
                              const int*   m,
                              const int*   n,
                              const int*   k,
                              const float  alpha,
                              const float  beta,
                              const int    count,
                              cudaStream_t stream) {
    group_gemm_runner_->gemm(A, B, C, m, n, k, alpha, beta, count, stream);
}
template class GroupGemmPlugin<half>;
template class GroupGemmPlugin<float>;
#ifdef ENABLE_BF16
template class GroupGemmPlugin<__nv_bfloat16>;
#endif
}  // namespace tensorrt_llm::plugins