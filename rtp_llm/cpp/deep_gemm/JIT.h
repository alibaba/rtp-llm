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
#include <filesystem>
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/deep_gemm/utils.h"

namespace rtp_llm {

struct KernelSoStatus {
    enum class StatusCode {
        NotFound,
        Found,
        Timeout
    };
    StatusCode  status;
    std::string path;
    KernelSoStatus(StatusCode status = StatusCode::NotFound, std::string path = ""): status(status), path(path) {}
};

struct VectorHasher {
    int operator()(const std::vector<uint32_t>& V) const {
        int hash = V.size();
        for (auto& i : V) {
            hash ^= i + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
        return hash;
    }
};

std::string getFilesHash(std::string& jit_hdrs_path, std::string rel_path, bool interleave = true);
void        collectFiles(const std::filesystem::path& dir_path, std::vector<std::filesystem::path>& files);

std::string getDeepGemmTypeStr(DeepGemmType type);

std::string getJITPath();
std::string getRemoteJITDir();
std::string generateKernelName();

void* loadKernel(const std::string& path, const std::string& func_name);
void* loadKernelWithTimeout(const std::string& path, const std::string& func_name, const int timeout_sec = 0);

KernelSoStatus searchKernelSo(const std::filesystem::path& directory);
KernelSoStatus searchKernelSoWithTimeout(const std::filesystem::path& directory, const int timeout_sec = 0);

std::string compileAndSaveKernel(const std::string& dir_path,
                                 const std::string& jit_hdrs_path,
                                 const std::string& cu_file_content,
                                 const std::string& nvcc_command,
                                 bool               interleave);

void copyFileToRemote(const std::filesystem::path& src_path, const std::filesystem::path& dst_path);
int  copyFileToRemoteWithTimeout(const std::filesystem::path& src_path,
                                 const std::filesystem::path& dst_path,
                                 const int                    timeout_sec);

template<typename T>
class JITRuntimeMap {
private:
    std::unordered_map<std::vector<uint32_t>, T, VectorHasher> map_;

public:
    void insert(const std::vector<uint32_t>& key, const T value) {
        map_[key] = value;
    }

    T find(const std::vector<uint32_t> key) {
        if (map_.find(key) != map_.end()) {
            return map_[key];
        }
        return nullptr;
    }
};

#ifdef ENABLE_FP8
typedef void (*runDeepGemmFunc)(
    __nv_bfloat16*, __nv_fp8_e4m3*, float*, __nv_fp8_e4m3*, float*, int*, uint32_t, cudaStream_t, uint32_t, uint32_t);
#endif

class KernelParamsBase {
public:
    bool                          interleave;
    std::string                   hdrs_relative_path;
    std::string                   cache_relative_path;
    virtual std::string           getShortParamsStr() const                         = 0;
    virtual std::string           getParamsStr() const                              = 0;
    virtual std::string           getKernelStr() const                              = 0;
    virtual std::vector<uint32_t> getKey() const                                    = 0;
    virtual std::string           getCommandStr(const std::string& hdrs_path) const = 0;
};

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
        hdrs_relative_path  = "/cpp/deep_gemm";
        cache_relative_path = "/deep_gemm_runtime/";
    }
};

template<typename ParamsType, typename FuncType>
class JIT {
public:
    JITRuntimeMap<FuncType> jit_kernels_;

    std::string jit_hdrs_path_ = "";

    void* searchKernel(const std::string& local_dir_path,
                       const std::string& remote_dir_path,
                       const std::string& remote_jit_dir,
                       const std::string& func_name,
                       const std::string& kernel_content,
                       const std::string& compile_command,
                       bool               interleave = true) {
        KernelSoStatus remote_kernel_so_status, local_kernel_so_status;
        std::string    so_path;

        static bool use_remote_jit = std::filesystem::exists(remote_jit_dir);
        if (use_remote_jit) {
            remote_kernel_so_status = searchKernelSoWithTimeout(remote_dir_path, 30);
            if (remote_kernel_so_status.status == KernelSoStatus::StatusCode::Timeout) {
                RTP_LLM_LOG_INFO("Remote jit cache search timeout, use local");
                use_remote_jit = false;
            } else if (remote_kernel_so_status.status == KernelSoStatus::StatusCode::Found) {
                RTP_LLM_LOG_INFO("Found kernel in remote jit cache, use remote %s",
                                 remote_kernel_so_status.path.c_str());
                so_path = remote_kernel_so_status.path;
            }
        }

        if (remote_kernel_so_status.status == KernelSoStatus::StatusCode::NotFound) {
            local_kernel_so_status = searchKernelSoWithTimeout(local_dir_path);
            if (local_kernel_so_status.status == KernelSoStatus::StatusCode::Found) {
                RTP_LLM_LOG_INFO("Found kernel in local jit cache, use local %s", local_kernel_so_status.path.c_str());
                so_path = local_kernel_so_status.path;
            }
        }

        if (so_path.empty()) {
            so_path = compileAndSaveKernel(local_dir_path, jit_hdrs_path_, kernel_content, compile_command, interleave);
            if (use_remote_jit) {
                std::filesystem::path path = so_path;
                if (copyFileToRemoteWithTimeout(so_path, remote_dir_path + "/" + path.filename().string(), 30) == -1) {
                    RTP_LLM_LOG_INFO("Failed to copy kernel to remote, use local");
                    use_remote_jit = false;
                }
            }
        }

        void* so_kernel = loadKernelWithTimeout(so_path, func_name, 30);
        if (!so_kernel) {
            RTP_LLM_FAIL("Failed to load kernel from %s", so_path.c_str());
        }

        return so_kernel;
    }

    FuncType getKernelPtr(const ParamsType& params) {
        auto kernel_key = params.getKey();

        // find in runtime cache first
        auto kernel = jit_kernels_.find(kernel_key);
        if (kernel) {
            return kernel;
        }

        RTP_LLM_LOG_INFO("Start load deepgemm kernel for %u %u %u %u",
                         params.n,
                         params.k,
                         params.num_groups,
                         uint32_t(params.gemm_type));

        if (jit_hdrs_path_.empty()) {
            jit_hdrs_path_ = getJITPath();
        }

        static const std::string remote_jit_dir   = getRemoteJITDir();
        const std::string        short_params_str = params.getShortParamsStr();
        const std::string        params_str       = params.getParamsStr();
        const std::string        func_name        = "runDeepGemm_" + params_str;
        const std::string        file_hash        = getFilesHash(jit_hdrs_path_, params.hdrs_relative_path);
        const std::string        local_dir_path =
            std::string("." + params.cache_relative_path + file_hash + "/" + short_params_str + "/" + params_str);
        const std::string remote_dir_path = std::string(remote_jit_dir + params.cache_relative_path + file_hash + "/"
                                                        + short_params_str + "/" + params_str);

        auto so_kernel = searchKernel(local_dir_path,
                                      remote_dir_path,
                                      remote_jit_dir,
                                      func_name,
                                      params.getKernelStr(),
                                      params.getCommandStr(jit_hdrs_path_),
                                      params.interleave);
        jit_kernels_.insert(kernel_key, (reinterpret_cast<FuncType>(so_kernel)));

        RTP_LLM_LOG_INFO("Finish loading deepgemm kernel for %u %u %u %u",
                         params.n,
                         params.k,
                         params.num_groups,
                         uint32_t(params.gemm_type));

        return reinterpret_cast<FuncType>(so_kernel);
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
