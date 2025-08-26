#include <cuda.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <mutex>
#include <fcntl.h>
#include <unistd.h>
#include <vector>
#include <cstdlib>
#include <dlfcn.h>
#include <iostream>
#include <string>
#include <unordered_map>
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/deep_gemm/utils.h"

namespace rtp_llm {

#ifdef ENABLE_FP8
typedef void (*runDeepGemmFunc)(
    __nv_bfloat16*, __nv_fp8_e4m3*, float*, __nv_fp8_e4m3*, float*, int*, uint32_t, cudaStream_t, uint32_t, uint32_t);

class JITRuntimeMap {
private:
    std::unordered_map<std::vector<uint32_t>, runDeepGemmFunc, VectorHasher> map_;

public:
    void insert(const std::vector<uint32_t>& key, const runDeepGemmFunc value) {
        map_[key] = value;
    }

    runDeepGemmFunc find(const std::vector<uint32_t> key) {
        if (map_.find(key) != map_.end()) {
            return map_[key];
        }
        return nullptr;
    }
};

struct KernelParams {
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
};

class JIT {
private:
    static JITRuntimeMap   jit_kernels_;
    static std::string     getParamsStr(KernelParams& params);
    static std::string     getKernelStr(KernelParams& params);
    static bool            loadFromCache(KernelParams& params);
    static void            compileAndSave(KernelParams& params);
    static runDeepGemmFunc searchKernel(KernelParams& params);

public:
    static void runKernel(__nv_bfloat16* output,
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
};
#endif

}  // namespace rtp_llm
