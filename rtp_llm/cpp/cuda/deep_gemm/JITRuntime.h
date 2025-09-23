#include <cuda.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "rtp_llm/cpp/cuda/deep_gemm/JIT.h"

namespace rtp_llm {

#ifdef ENABLE_FP8
typedef void (*runDeepGemmFunc)(
    __nv_bfloat16*, __nv_fp8_e4m3*, float*, __nv_fp8_e4m3*, float*, int*, uint32_t, cudaStream_t, uint32_t, uint32_t);
#endif

class KernelParams: public KernelParamsBase {
public:
    uint32_t     n;
    uint32_t     k;
    uint32_t     bm;
    uint32_t     bn;
    uint32_t     bk;
    uint32_t     num_groups;
    uint32_t     num_stages;
    uint32_t     num_tma_multicast;
    DeepGemmType gemm_type;
    bool         swap_ab;

    std::string           getShortParamsStr() const;
    std::string           getParamsStr() const;
    std::string           getKernelStr() const;
    std::vector<uint32_t> getKey() const;
    std::string           getCommandStr(const std::string& hdrs_path) const;

    KernelParams() = default;
    KernelParams(uint32_t     n,
                 uint32_t     k,
                 uint32_t     bm,
                 uint32_t     bn,
                 uint32_t     bk,
                 uint32_t     num_groups,
                 uint32_t     num_stages,
                 uint32_t     num_tma_multicast,
                 DeepGemmType gemm_type,
                 bool         swap_ab):
        n(n),
        k(k),
        bm(bm),
        bn(bn),
        bk(bk),
        num_groups(num_groups),
        num_stages(num_stages),
        num_tma_multicast(num_tma_multicast),
        gemm_type(gemm_type),
        swap_ab(swap_ab) {
        interleave          = true;
        hdrs_relative_path  = "/cpp/cuda/deep_gemm";
        cache_relative_path = "/deep_gemm_runtime/";
    }
};

#ifdef ENABLE_FP8
void runKernel(__nv_bfloat16* output,
               __nv_fp8_e4m3* lhs,
               float*         lhs_scale,
               __nv_fp8_e4m3* rhs,
               float*         rhs_scale,
               int*           grouped_layout,
               uint32_t       m,
               uint32_t       n,
               uint32_t       k,
               uint32_t       bm,
               uint32_t       bn,
               uint32_t       bk,
               uint32_t       num_groups,
               uint32_t       num_stages,
               uint32_t       num_tma_multicast,
               DeepGemmType   gemm_type,
               cudaStream_t   stream,
               uint32_t       num_sms,
               uint32_t       smem_size,
               bool           swap_ab);
#endif

}  // namespace rtp_llm
