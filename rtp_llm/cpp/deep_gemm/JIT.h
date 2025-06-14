#include <cuda.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

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

class JITFilelock {
    int fd = -1;
    std::string filepath;
public:
    explicit JITFilelock(const std::string& path, int timeout_sec) : filepath(path) {
        fd = open(filepath.c_str(), O_CREAT | O_RDWR, 0666);
        if (fd == -1) {
            RTP_LLM_FAIL("Cannot open filelock:" + filepath);
        }

        struct flock fl {};
        fl.l_type = F_WRLCK;
        
        time_t start = time(nullptr);
        bool lock_success = false;
        while (time(nullptr) - start < timeout_sec) {
            if (fcntl(fd, F_SETLK, &fl) == 0) {
                lock_success = true;
                break;
            }
            sleep(1);
        }
        
        if (!lock_success) {
            RTP_LLM_FAIL("Cannot get filelock " + filepath + " in " + std::to_string(timeout_sec) + " secs");
        }
    }

    ~JITFilelock() {
        if (fd != -1) {
            struct flock fl {};
            fl.l_type = F_UNLCK;
            fcntl(fd, F_SETLK, &fl);
            close(fd);
        }
    }

    JITFilelock(const JITFilelock&) = delete;
    JITFilelock& operator=(const JITFilelock&) = delete;
};

#ifdef ENABLE_FP8
typedef void (*runDeepGemmFunc)(__nv_bfloat16*,
                                __nv_fp8_e4m3*,
                                float*,
                                __nv_fp8_e4m3*,
                                float*,
                                int*,
                                uint32_t,
                                cudaStream_t,
                                uint32_t,
                                uint32_t);

struct VectorHasher {
    int operator()(const std::vector<uint32_t> &V) const {
        int hash = V.size();
        for(auto &i : V) {
            hash ^= i + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
        return hash;
    }
};

class JIT
{
private:
    static std::unordered_map<std::vector<uint32_t>, runDeepGemmFunc, VectorHasher> jit_kernels_;
    static uint64_t generateInputKey(uint64_t n, uint64_t k, uint64_t group_num, DeepGemmType gemm_type);
    static void systemCall(std::string command);
public:
    static std::string getKernelStr(uint32_t n, uint32_t k, uint32_t bm, uint32_t bn, uint32_t bk, uint32_t num_groups, uint32_t num_stages, uint32_t num_tma_multicast, DeepGemmType gemm_type, bool swap_ab);

    static runDeepGemmFunc compileAndLoadKernel(uint32_t n, uint32_t k, uint32_t bm, uint32_t bn, uint32_t bk, uint32_t num_groups, uint32_t num_stages, uint32_t num_tma_multicast, DeepGemmType gemm_type, bool swap_ab);

    static void runKernel(__nv_bfloat16*         output,
                          __nv_fp8_e4m3*         lhs,
                          float*                 lhs_scale,
                          __nv_fp8_e4m3*         rhs,
                          float*                 rhs_scale,
                          int*                   grouped_layout,
                          uint32_t               m,
                          uint32_t               n,
                          uint32_t               k,
                          uint32_t               bm,
                          uint32_t               bn,
                          uint32_t               bk,
                          uint32_t               num_groups,
                          uint32_t               num_stages,
                          uint32_t               num_tma_multicast,
                          DeepGemmType           gemm_type,
                          cudaStream_t           stream,
                          uint32_t               num_sms,
                          uint32_t               smem_size,
                          bool                   swap_ab);
};
#endif

} // namespace rtp_llm
