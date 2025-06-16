#include <cuda.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <vector>
#include <openssl/sha.h>
#include <filesystem>
#include "rtp_llm/cpp/deep_gemm/JIT.h"

using namespace std;

namespace rtp_llm {

#ifdef ENABLE_FP8

unordered_map<vector<uint32_t>, runDeepGemmFunc, VectorHasher> JIT::jit_kernels_ = {};

string getDeepGemmTypeStr(DeepGemmType type) {
    switch (type)
    {
    case DeepGemmType::Normal:
        return string("deep_gemm::GemmType::Normal");
        break;
    case DeepGemmType::GroupedContiguous:
        return string("deep_gemm::GemmType::GroupedContiguous");
        break;
    case DeepGemmType::GroupedMasked:
        return string("deep_gemm::GemmType::GroupedMasked");
        break;
    default:
        return "";
        break;
    }
}

string getPath() {
    const string package = "rtp_llm";
    stringstream cmd;
    cmd << "/opt/conda310/bin/python -c \"from importlib.resources import files; path=files('"
        << package << "');\nif hasattr(path, '_paths'):\n\tpath=next(iter(path._paths))\nprint(str(path))\"";

    FILE* pipe = popen(cmd.str().c_str(), "r");
    char buffer[128];
    string result;
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        result += buffer;
    }
    pclose(pipe);

    if (!result.empty() && result.back() == '\n') {
        result.pop_back();
    }
   
    if (result.rfind("\n") != string::npos) {
        result = result.substr(result.rfind("\n") + 1);
    }

    return result;
}

void collectFiles(const filesystem::path& dir_path, vector<filesystem::path>& files) {
    for (const auto& entry: filesystem::recursive_directory_iterator(dir_path)) {
        if (entry.is_regular_file()) {
            files.emplace_back(entry.path());
        }
    }
}

string getFilesHash(filesystem::path path) {
    vector<filesystem::path> files = {
        path.string() + "/cpp/deep_gemm/deep_gemm_template.h",
        path.string() + "/cpp/deep_gemm/utils.h",
        path.string() + "/cpp/deep_gemm/interleave_ffma.py",
        path.string() + "/cpp/deep_gemm/JIT.h",
        path.string() + "/cpp/deep_gemm/JIT.cc"
    };
    collectFiles(filesystem::path(path.string() + "/cpp/deep_gemm/include"), files);
    collectFiles(filesystem::path(path.string() + "/cpp/deep_gemm/cutlass_hdr"), files);

    sort(files.begin(), files.end());

    SHA256_CTX sha256;
    SHA256_Init(&sha256);

    for (const auto& file : files) {
        std::string filename = file.string();
        SHA256_Update(&sha256, filename.c_str(), filename.size());

        ifstream ifs(file, std::ios::binary);
        char buffer[4096];
        while (ifs.read(buffer, sizeof(buffer)) || ifs.gcount() > 0) {
            SHA256_Update(&sha256, buffer, ifs.gcount());
        }
    }

    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256_Final(hash, &sha256);

    ostringstream oss;
    for (int i = 0; i < SHA256_DIGEST_LENGTH; ++i) {
        oss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(hash[i]);
    }

    return oss.str();
}


string JIT::getKernelStr(uint32_t n, 
                         uint32_t k, 
                         uint32_t bm, 
                         uint32_t bn, 
                         uint32_t bk, 
                         uint32_t num_groups, 
                         uint32_t num_stages, 
                         uint32_t num_tma_multicast,
                         DeepGemmType gemm_type,
                         bool     swap_ab) {
    const string template_str = to_string(n) + ", " + to_string(k) + ", " + to_string(bm) + ", " + to_string(bn) + ", " + to_string(bk) + ", " + to_string(num_groups) + ", " + to_string(num_stages) + ", " + to_string(num_tma_multicast) + ", " + getDeepGemmTypeStr(gemm_type);
    const string func_name_str = to_string(n) + "_" + to_string(k) + "_" + to_string(bm) + "_" + to_string(bn) + "_" + to_string(bk) + "_" + to_string(num_groups) + "_" + to_string(num_stages) + "_" + to_string(num_tma_multicast) + "_" + to_string(uint32_t(gemm_type)) + "_" + to_string(uint32_t(swap_ab));
    string code = R"delimiter(
#include <cuda.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#ifdef ENABLE_FP8
#include "rtp_llm/cpp/deep_gemm/utils.h"
#include "rtp_llm/cpp/deep_gemm/include/fp8_gemm.cuh"

using namespace rtp_llm;
extern "C"{
void runDeepGemm_)delimiter" + func_name_str + R"delimiter((__nv_bfloat16*         output,
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
    using gemm_runner = deep_gemm::Gemm<)delimiter" + template_str + R"delimiter(>;
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
    using gemm_runner = deep_gemm::GemmSwapAB<)delimiter" + template_str + R"delimiter(>;
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

runDeepGemmFunc JIT::compileAndLoadKernel(uint32_t n, 
                                          uint32_t k, 
                                          uint32_t bm, 
                                          uint32_t bn, 
                                          uint32_t bk, 
                                          uint32_t num_groups, 
                                          uint32_t num_stages, 
                                          uint32_t num_tma_multicast,
                                          DeepGemmType gemm_type,
                                          bool     swap_ab) {
    auto kernel_key = vector<uint32_t>{n, k, bm, bn, bk, num_groups, num_stages, num_tma_multicast, uint32_t(gemm_type), uint32_t(swap_ab)};

    const string params_str = "_" + to_string(n) + "_" + to_string(k) + "_" + to_string(bm) + "_" + to_string(bn) + "_" + to_string(bk) + "_" + to_string(num_groups) + "_" + to_string(num_stages) + "_" + to_string(num_tma_multicast) + "_" + to_string(uint32_t(gemm_type)) + "_" + to_string(uint32_t(swap_ab));
    if (jit_kernels_.find(kernel_key) != jit_kernels_.end()) {
        return jit_kernels_[kernel_key];
    }
    
    static const string hdrs_path = getPath();
    static const string file_hash = getFilesHash(hdrs_path);
    filesystem::path dir_path = string("./deep_gemm_runtime/" + file_hash);
    filesystem::create_directories(dir_path);

    JITFilelock lock(string(dir_path) + "/deepgemm" + params_str + "_lock", 30);

    string cu_filename = dir_path.string() + "/deepgemm" + params_str + ".cu";
    string so_name = dir_path.string() + "/libdeepgemm" + params_str + ".so";

    if (!filesystem::exists(so_name)) {
        RTP_LLM_LOG_INFO("JIT compilation %s begin", cu_filename);
        ofstream cu_file(cu_filename.c_str());
        cu_file << getKernelStr(n, k, bm, bn, bk, num_groups, num_stages, num_tma_multicast, gemm_type, swap_ab);
        cu_file.close();

        string command = "/usr/local/cuda/bin/nvcc " + cu_filename + " -o " + so_name + " -std=c++17 -shared -O3 --expt-relaxed-constexpr --expt-extended-lambda -gencode=arch=compute_90a,code=sm_90a --compiler-options=-fPIC,-O3,-Wno-deprecated-declarations,-Wno-abi -DENABLE_FP8 -I" + hdrs_path + "/../ -I" + hdrs_path + "/cpp/deep_gemm/cutlass_hdr/cutlass/include";

        int result = system(command.c_str());
        if (result != 0) {
            RTP_LLM_FAIL("Compilation error for template %u %u %u %u", n, k, num_groups, gemm_type);
        }

        command = "/opt/conda310/bin/python " + hdrs_path + "/cpp/deep_gemm/interleave_ffma.py --so ./" + so_name;
        result = system(command.c_str());
        if (result != 0) {
            RTP_LLM_FAIL("Failed to do interleave ffma");
        }
        RTP_LLM_LOG_INFO("JIT compilation finished");
    }

    void* lib = dlopen(so_name.c_str(), RTLD_LAZY);
    if (!lib) {
        RTP_LLM_FAIL("Failed to load library: " + so_name + ", error: " + dlerror());
    }

    string func_name = "runDeepGemm" + params_str;
    runDeepGemmFunc kernel = (runDeepGemmFunc)dlsym(lib, func_name.c_str());
    if (!kernel) {
        RTP_LLM_FAIL("Failed to find function: " + func_name + ", error: " + dlerror());
    }
    RTP_LLM_LOG_INFO("JIT load %s finished", so_name);

    jit_kernels_[kernel_key] = kernel;
    return kernel;
}

void JIT::runKernel(__nv_bfloat16*         output,
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
                    bool                   swap_ab) {
    auto kernel = compileAndLoadKernel(n, k, bm, bn, bk, num_groups, num_stages, num_tma_multicast, gemm_type, swap_ab);
    kernel(output, lhs, lhs_scale, rhs, rhs_scale, grouped_layout, m, stream, num_sms, smem_size);
}
#endif

} // namespace rtp_llm
