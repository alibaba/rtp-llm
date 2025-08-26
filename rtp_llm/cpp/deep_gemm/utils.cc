#include "rtp_llm/cpp/deep_gemm/utils.h"
#include <future>
#include <openssl/sha.h>

namespace rtp_llm {

const std::string jit_hdrs_path  = getJITPath();
const std::string remote_jit_dir = getRemoteJITDir();
bool              use_remote_jit = std::filesystem::exists(remote_jit_dir);

std::string getFilesHash(std::filesystem::path path, bool interleave) {
    std::vector<std::filesystem::path> files = {path.string() + "/JIT.h", path.string() + "/JIT.cc"};
    files.push_back(jit_hdrs_path + "/cpp/deep_gemm/utils.h");
    files.push_back(path.string() + "/cpp/deep_gemm/utils.cc");
    if (interleave) {
        files.push_back(path.string() + "/interleave_ffma.py");
    }
    collectFiles(std::filesystem::path(path.string() + "/cutlass_hdr"), files);
    collectFiles(std::filesystem::path(path.string() + "/deepgemm_hdr"), files);
    collectFiles(std::filesystem::path(path.string() + "/include"), files);

    sort(files.begin(), files.end());

    SHA256_CTX sha256;
    SHA256_Init(&sha256);

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

std::string generateKernelName() {
    auto now       = std::chrono::system_clock::now();
    auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();

    std::ostringstream oss;
    oss << getpid() << "_" << std::this_thread::get_id() << "_" << timestamp;
    return oss.str();
}

std::string getRemoteJITDir() {
    char const* remote_jit_dir_env = getenv("REMOTE_JIT_DIR");
    if (remote_jit_dir_env) {
        return std::string(remote_jit_dir_env);
    }
    return std::string("");
}

void* loadKernel(const std::string& path, const std::string& params_str) {
    void* lib = dlopen(path.c_str(), RTLD_LAZY);
    if (!lib) {
        RTP_LLM_LOG_INFO(dlerror());
        return nullptr;
    }

    std::string func_name = "runDeepGemm_" + params_str;
    auto        kernel    = (void*)dlsym(lib, func_name.c_str());
    if (!kernel) {
        RTP_LLM_LOG_INFO(dlerror());
        return nullptr;
    }
    return kernel;
}

void* searchAndLoadKernel(const std::filesystem::path& directory, const std::string& params_str) {
    if (!std::filesystem::exists(directory)) {
        return nullptr;
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
        return nullptr;
    }

    return loadKernel(matched_files[0], params_str);
}

void* findCachedKernel(const std::filesystem::path& remote_dir_path,
                       const std::filesystem::path& local_dir_path,
                       const std::string&           params_str) {
    void* remote_cache = nullptr;
    if (use_remote_jit) {
        remote_cache = searchAndLoadKernel(remote_dir_path, params_str);
        auto future  = std::async(std::launch::async, searchAndLoadKernel, remote_dir_path, params_str);
        auto status  = future.wait_for(std::chrono::seconds(30));
        if (status == std::future_status::timeout) {
            use_remote_jit = false;
        } else {
            remote_cache = future.get();
        }
    }
    if (!remote_cache) {
        return searchAndLoadKernel(local_dir_path, params_str);
    }

    return remote_cache;
}

std::string compileAndSaveKernel(const std::filesystem::path& local_dir_path,
                                 const std::filesystem::path& remote_dir_path,
                                 const std::string&           cu_file_content,
                                 const std::string&           nvcc_command,
                                 bool                         interleave) {
    if (!std::filesystem::exists(local_dir_path)) {
        std::filesystem::create_directories(local_dir_path);
    }

    const std::string pid_and_timestamp_str = generateKernelName();
    const std::string cu_filename           = local_dir_path.string() + "/" + pid_and_timestamp_str + ".cu";
    const std::string so_filename           = local_dir_path.string() + "/" + pid_and_timestamp_str + ".so.temp";
    const std::string so_filename_final     = local_dir_path.string() + "/" + pid_and_timestamp_str + ".so";
    const std::string remote_filename       = remote_dir_path.string() + "/" + pid_and_timestamp_str + ".so.temp";
    const std::string remote_filename_final = remote_dir_path.string() + "/" + pid_and_timestamp_str + ".so";
    RTP_LLM_LOG_INFO("JIT compilation " + cu_filename + " begin");

    std::ofstream cu_file(cu_filename.c_str());
    cu_file << cu_file_content;
    cu_file.close();

    int         result;
    std::string command;

    result = system(("/usr/local/cuda/bin/nvcc " + cu_filename + " -o " + so_filename + nvcc_command).c_str());
    if (result != 0) {
        return std::string("");
    }

    if (interleave) {
        command = "/opt/conda310/bin/python " + jit_hdrs_path + "/cpp/deep_gemm/interleave_ffma.py --so " + so_filename;
        result  = system(command.c_str());
        if (result != 0) {
            return std::string("");
        }
    }

    std::filesystem::rename(so_filename, so_filename_final);

    if (use_remote_jit && !std::filesystem::exists(remote_filename)) {
        if (!std::filesystem::exists(remote_dir_path)) {
            std::filesystem::create_directories(remote_dir_path);
        }
        auto future = std::async(std::launch::async, [&] {
            try {
                std::filesystem::copy(so_filename_final, remote_filename);
                std::filesystem::rename(remote_filename, remote_filename_final);
                return true;
            } catch (const std::exception& e) {
                return false;
            }
        });

        auto status = future.wait_for(std::chrono::seconds(30));
        if (status == std::future_status::timeout || !future.get()) {
            use_remote_jit = false;
        }
    }
    return so_filename_final;
}

}  // namespace rtp_llm
