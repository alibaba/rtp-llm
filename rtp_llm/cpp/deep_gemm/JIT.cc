#include <cuda.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdlib>
#include <vector>
#include <future>
#include <openssl/sha.h>

#include "rtp_llm/cpp/deep_gemm/JIT.h"

using namespace std;

namespace rtp_llm {

string KernelParams::getShortParamsStr() const {
    return to_string(n) + "_" + to_string(k) + "_" + to_string(num_groups) + "_" + to_string(uint32_t(gemm_type));
}

string KernelParams::getParamsStr() const {
    return to_string(n) + "_" + to_string(k) + "_" + to_string(bm) + "_" + to_string(bn) + "_" + to_string(bk) + "_"
           + to_string(num_groups) + "_" + to_string(num_stages) + "_" + to_string(num_tma_multicast) + "_"
           + to_string(uint32_t(gemm_type)) + "_" + to_string(uint32_t(swap_ab));
}

string KernelParams::getKernelStr() const {
    const string template_str = to_string(n) + ", " + to_string(k) + ", " + to_string(bm) + ", " + to_string(bn) + ", "
                                + to_string(bk) + ", " + to_string(num_groups) + ", " + to_string(num_stages) + ", "
                                + to_string(num_tma_multicast) + ", " + getDeepGemmTypeStr(gemm_type);
    const string func_name_str = getParamsStr();
    string       code          = R"delimiter(
#include <cuda.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#ifdef ENABLE_FP8
#include "rtp_llm/cpp/deep_gemm/include/fp8_gemm.cuh"

extern "C"{
void runDeepGemm_)delimiter"
                  + func_name_str + R"delimiter((__nv_bfloat16*         output,
                             __nv_fp8_e4m3*         lhs,
                             float*                 lhs_scale,
                             __nv_fp8_e4m3*         rhs,
                             float*                 rhs_scale,
                             int*                   grouped_layout,
                             uint32_t               m,
                             cudaStream_t           stream,
                             uint32_t               num_sms,
                             uint32_t               smem_size) {
)delimiter";
    if (!swap_ab) {
        code += R"delimiter(
    using gemm_runner = deep_gemm::Gemm<)delimiter"
                + template_str + R"delimiter(>;
    auto tma_a_desc = gemm_runner::template make_2d_tma_a_desc<__nv_fp8_e4m3>(lhs, m);
    auto tma_b_desc = gemm_runner::template make_2d_tma_b_desc<__nv_fp8_e4m3>(rhs);
    auto tma_scales_a_desc = gemm_runner::template make_2d_tma_scales_a_desc<float>(lhs_scale, m);
    auto tma_d_desc = gemm_runner::template make_2d_tma_d_desc<__nv_bfloat16>(output, m);
    gemm_runner::run(output, rhs_scale, grouped_layout, m, tma_a_desc, tma_b_desc, tma_scales_a_desc, tma_d_desc, stream, num_sms, smem_size);
    return;
}
}
#endif
)delimiter";
    } else {
        if (gemm_type != DeepGemmType::Normal && gemm_type != DeepGemmType::GroupedMasked) {
            RTP_LLM_FAIL("deep_gemm::GemmSwapAB does not support gemm type other than Normal and GroupedMasked yet");
        }
        code += R"delimiter(
    using gemm_runner = deep_gemm::GemmSwapAB<)delimiter"
                + template_str + R"delimiter(>;
    auto tma_a_desc = gemm_runner::template make_2d_tma_a_desc<__nv_fp8_e4m3>(rhs);
    auto tma_b_desc = gemm_runner::template make_2d_tma_b_desc<__nv_fp8_e4m3>(lhs, m);
    auto tma_scales_b_desc = gemm_runner::template make_2d_tma_scales_b_desc<float>(lhs_scale, m);
    auto tma_d_desc = gemm_runner::template make_2d_tma_d_desc<__nv_bfloat16>(output, m);
    gemm_runner::run(output, rhs_scale, grouped_layout, m, tma_a_desc, tma_b_desc, tma_scales_b_desc, tma_d_desc, stream, num_sms, smem_size);
    return;
}
}
#endif
)delimiter";
    }
    return code;
}

vector<uint32_t> KernelParams::getKey() const {
    return {n, k, bm, bn, bk, num_groups, num_stages, num_tma_multicast, uint32_t(gemm_type), uint32_t(swap_ab)};
}

string KernelParams::getCommandStr(const string& hdrs_path) const {
    return " -std=c++17 -shared -O3 --expt-relaxed-constexpr --expt-extended-lambda -gencode=arch=compute_90a,code=sm_90a --compiler-options=-fPIC,-O3,-Wno-deprecated-declarations,-Wno-abi -diag-suppress 177 -DENABLE_FP8 -I"
           + hdrs_path + "/../ -I" + hdrs_path + "/cpp/deep_gemm/cutlass_hdr/cutlass/include";
}

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
               bool           swap_ab) {
    KernelParams now_params(n, k, bm, bn, bk, num_groups, num_stages, num_tma_multicast, gemm_type, swap_ab);
    static JIT<KernelParams, runDeepGemmFunc> jit;
    auto                                      kernel_ptr = jit.getKernelPtr(now_params);
    if (kernel_ptr) {
        kernel_ptr(output, lhs, lhs_scale, rhs, rhs_scale, grouped_layout, m, stream, num_sms, smem_size);
        return;
    } else {
        RTP_LLM_FAIL("Failed to get kernel ptr for %u %u %u %u", n, k, num_groups, uint32_t(gemm_type));
    }
}
#endif

std::string getFilesHash(std::string& jit_hdrs_path, std::string rel_path, bool interleave) {
    std::vector<std::filesystem::path> files = {
        jit_hdrs_path + "/cpp/deep_gemm/utils.h",
        jit_hdrs_path + "/cpp/deep_gemm/JIT.h",
        jit_hdrs_path + "/cpp/deep_gemm/JIT.cc",
    };
    RTP_LLM_LOG_INFO("files: %s", files[0].c_str());
    if (interleave) {
        files.push_back(jit_hdrs_path + "/interleave_ffma.py");
    } else {
        // for not cuda compile
        files.push_back(jit_hdrs_path + rel_path + "/JIT.h");
        files.push_back(jit_hdrs_path + rel_path + "/JIT.cc");
    }
    collectFiles(std::filesystem::path(jit_hdrs_path + rel_path + "/cutlass_hdr"), files);
    collectFiles(std::filesystem::path(jit_hdrs_path + rel_path + "/deepgemm_hdr"), files);
    collectFiles(std::filesystem::path(jit_hdrs_path + rel_path + "/include"), files);

    sort(files.begin(), files.end());

    SHA256_CTX sha256;
    if (SHA256_Init(&sha256) != 1) {
        RTP_LLM_FAIL("Failed to initialize SHA256 context");
    }

    for (const auto& file : files) {
        std::string filename = file.string();

        std::ifstream ifs(file, std::ios::binary);
        char          buffer[4096];
        while (ifs.read(buffer, sizeof(buffer)) || ifs.gcount() > 0) {
            SHA256_Update(&sha256, buffer, ifs.gcount());
        }
    }

    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256_Final(hash, &sha256);

    std::ostringstream oss;
    for (int i = 0; i < SHA256_DIGEST_LENGTH; ++i) {
        oss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(hash[i]);
    }

    return oss.str();
}

void collectFiles(const std::filesystem::path& dir_path, std::vector<std::filesystem::path>& files) {
    if (!std::filesystem::exists(dir_path)) {
        return;
    }
    for (const auto& entry : std::filesystem::recursive_directory_iterator(dir_path)) {
        if (entry.is_regular_file()) {
            files.emplace_back(entry.path());
        }
    }
}

std::string getDeepGemmTypeStr(DeepGemmType type) {
    switch (type) {
        case DeepGemmType::Normal:
            return std::string("deep_gemm::GemmType::Normal");
        case DeepGemmType::GroupedContiguous:
            return std::string("deep_gemm::GemmType::GroupedContiguous");
        case DeepGemmType::GroupedMasked:
            return std::string("deep_gemm::GemmType::GroupedMasked");
        default:
            return "";
    }
}

std::string getJITPath() {
    const std::string package = "rtp_llm";
    std::stringstream cmd;
    cmd << "/opt/conda310/bin/python -c \"from importlib.resources import files; path=files('" << package
        << "');\nif hasattr(path, '_paths'):\n\tpath=next(iter(path._paths))\nprint(str(path))\"";

    FILE*       pipe = popen(cmd.str().c_str(), "r");
    char        buffer[128];
    std::string result;
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        result += buffer;
    }
    pclose(pipe);

    if (!result.empty() && result.back() == '\n') {
        result.pop_back();
    }

    if (result.rfind("\n") != std::string::npos) {
        result = result.substr(result.rfind("\n") + 1);
    }

    return result;
}

std::string getRemoteJITDir() {
    char const* remote_jit_dir_env = getenv("REMOTE_JIT_DIR");
    if (remote_jit_dir_env) {
        return std::string(remote_jit_dir_env);
    }
    return std::string("");
}

std::string generateKernelName() {
    auto now       = std::chrono::system_clock::now();
    auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();

    std::ostringstream oss;
    oss << getpid() << "_" << std::this_thread::get_id() << "_" << timestamp;
    return oss.str();
}

void* loadKernel(const std::string& path, const std::string& func_name) {
    void* lib = dlopen(path.c_str(), RTLD_LAZY);
    if (!lib) {
        RTP_LLM_LOG_INFO(dlerror());
        return nullptr;
    }

    auto kernel = (void*)dlsym(lib, func_name.c_str());
    if (!kernel) {
        RTP_LLM_LOG_INFO(dlerror());
        return nullptr;
    }
    return kernel;
}

void* loadKernelWithTimeout(const std::string& path, const std::string& func_name, const int timeout_sec) {
    if (timeout_sec <= 0) {
        return loadKernel(path, func_name);
    }

    auto future = std::async(std::launch::async, loadKernel, path, func_name);
    auto status = future.wait_for(std::chrono::seconds(timeout_sec));
    if (status == std::future_status::timeout) {
        return nullptr;
    } else {
        return future.get();
    }
}

KernelSoStatus searchKernelSo(const std::filesystem::path& directory) {
    if (!std::filesystem::exists(directory)) {
        return KernelSoStatus(KernelSoStatus::StatusCode::NotFound, "");
    }

    const std::string&                 format = ".so";
    std::vector<std::filesystem::path> matched_files;

    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        if (entry.is_regular_file()) {
            const std::string filename = entry.path().filename().string();

            if (filename.size() >= format.size() + 1 && filename.substr(filename.size() - format.size()) == format) {
                matched_files.push_back(entry.path());
            }
        }
    }

    if (matched_files.empty()) {
        return KernelSoStatus(KernelSoStatus::StatusCode::NotFound, "");
    }

    return KernelSoStatus(KernelSoStatus::StatusCode::Found, matched_files[0]);
}

KernelSoStatus searchKernelSoWithTimeout(const std::filesystem::path& directory, const int timeout_sec) {
    if (timeout_sec <= 0) {
        return searchKernelSo(directory);
    }
    auto future = std::async(std::launch::async, searchKernelSo, directory);
    auto status = future.wait_for(std::chrono::seconds(timeout_sec));
    if (status == std::future_status::timeout) {
        return KernelSoStatus(KernelSoStatus::StatusCode::Timeout, "");
    } else {
        return future.get();
    }
}

std::string compileAndSaveKernel(const std::string& dir_path,
                                 const std::string& jit_hdrs_path,
                                 const std::string& cu_file_content,
                                 const std::string& nvcc_command,
                                 bool               interleave) {
    RTP_LLM_LOG_INFO("Start compile kernel %s", dir_path.c_str());

    if (!std::filesystem::exists(dir_path)) {
        std::filesystem::create_directories(dir_path);
    }

    const std::string pid_and_timestamp_str = generateKernelName();
    const std::string cu_filename           = dir_path + "/" + pid_and_timestamp_str + ".cu";
    const std::string so_filename           = dir_path + "/" + pid_and_timestamp_str + ".so.temp";
    const std::string so_filename_final     = dir_path + "/" + pid_and_timestamp_str + ".so";

    std::ofstream cu_file(cu_filename.c_str());
    cu_file << cu_file_content;
    cu_file.close();

    int         result;
    std::string command;

    command = "/usr/local/cuda/bin/nvcc " + cu_filename + " -o " + so_filename + " " + nvcc_command;
    result  = system(command.c_str());
    if (result != 0) {
        RTP_LLM_FAIL("Failed to run command: \"" + command + "\" with error status: " + std::to_string(result));
    }

    if (interleave) {
        command = "/opt/conda310/bin/python " + jit_hdrs_path + "/cpp/deep_gemm/interleave_ffma.py --so " + so_filename;
        result  = system(command.c_str());
        if (result != 0) {
            RTP_LLM_FAIL("Failed to run command: \"" + command + "\" with error status: " + std::to_string(result));
        }
    }

    std::filesystem::rename(so_filename, so_filename_final);
    RTP_LLM_LOG_INFO("Finish compile kernel %s", so_filename_final.c_str());

    return so_filename_final;
}

void copyFileToRemote(const std::filesystem::path& src_path, const std::filesystem::path& dst_path) {
    std::string temp_path = dst_path.string() + ".temp";
    if (!std::filesystem::exists(dst_path.parent_path())) {
        std::filesystem::create_directories(dst_path.parent_path());
    }
    std::filesystem::copy(src_path, temp_path);
    std::filesystem::rename(temp_path, dst_path);
}

int copyFileToRemoteWithTimeout(const std::filesystem::path& src_path,
                                const std::filesystem::path& dst_path,
                                const int                    timeout_sec) {
    if (timeout_sec <= 0) {
        copyFileToRemote(src_path, dst_path);
        return 0;
    }

    auto future = std::async(std::launch::async, copyFileToRemote, src_path, dst_path);
    auto status = future.wait_for(std::chrono::seconds(timeout_sec));
    if (status == std::future_status::timeout) {
        return -1;
    } else {
        future.get();
        return 0;
    }
}

}  // namespace rtp_llm
