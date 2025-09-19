#include "rtp_llm/cpp/cuda/deep_gemm/JIT.h"
#include <future>
#include <openssl/sha.h>

using namespace std;

namespace rtp_llm {

string getFilesHash(string& jit_hdrs_path, string rel_path, bool interleave) {
    vector<filesystem::path> files = {
        jit_hdrs_path + "/cpp/cuda/deep_gemm/utils.h",
        jit_hdrs_path + "/cpp/cuda/deep_gemm/JIT.h",
        jit_hdrs_path + "/cpp/cuda/deep_gemm/JIT.cc",
    };
    files.push_back(jit_hdrs_path + rel_path + "/JITRuntime.h");
    files.push_back(jit_hdrs_path + rel_path + "/JITRuntime.cc");

    if (interleave) {
        files.push_back(jit_hdrs_path + "/interleave_ffma.py");
    }
    collectFiles(filesystem::path(jit_hdrs_path + rel_path + "/cutlass_hdr"), files);
    collectFiles(filesystem::path(jit_hdrs_path + rel_path + "/deepgemm_hdr"), files);
    collectFiles(filesystem::path(jit_hdrs_path + rel_path + "/include"), files);

    sort(files.begin(), files.end());

    SHA256_CTX sha256;
    if (SHA256_Init(&sha256) != 1) {
        RTP_LLM_FAIL("Failed to initialize SHA256 context");
    }

    for (const auto& file : files) {
        string filename = file.string();

        ifstream ifs(file, ios::binary);
        char     buffer[4096];
        while (ifs.read(buffer, sizeof(buffer)) || ifs.gcount() > 0) {
            SHA256_Update(&sha256, buffer, ifs.gcount());
        }
    }

    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256_Final(hash, &sha256);

    ostringstream oss;
    for (int i = 0; i < SHA256_DIGEST_LENGTH; ++i) {
        oss << hex << setw(2) << setfill('0') << static_cast<int>(hash[i]);
    }

    return oss.str();
}

void collectFiles(const filesystem::path& dir_path, vector<filesystem::path>& files) {
    if (!filesystem::exists(dir_path)) {
        return;
    }
    for (const auto& entry : filesystem::recursive_directory_iterator(dir_path)) {
        if (entry.is_regular_file()) {
            files.emplace_back(entry.path());
        }
    }
}

string getDeepGemmTypeStr(DeepGemmType type) {
    switch (type) {
        case DeepGemmType::Normal:
            return string("deep_gemm::GemmType::Normal");
        case DeepGemmType::GroupedContiguous:
            return string("deep_gemm::GemmType::GroupedContiguous");
        case DeepGemmType::GroupedMasked:
            return string("deep_gemm::GemmType::GroupedMasked");
        default:
            return "";
    }
}

string getJITPath() {
    const string package = "rtp_llm";
    stringstream cmd;
    cmd << "/opt/conda310/bin/python -c \"from importlib.resources import files; path=files('" << package
        << "');\nif hasattr(path, '_paths'):\n\tpath=next(iter(path._paths))\nprint(str(path))\"";

    FILE*  pipe = popen(cmd.str().c_str(), "r");
    char   buffer[128];
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

string getRemoteJITDir() {
    char const* remote_jit_dir_env = getenv("REMOTE_JIT_DIR");
    if (remote_jit_dir_env) {
        return string(remote_jit_dir_env);
    }
    return string("");
}

string generateKernelName() {
    auto now       = chrono::system_clock::now();
    auto timestamp = chrono::duration_cast<chrono::milliseconds>(now.time_since_epoch()).count();

    ostringstream oss;
    oss << getpid() << "_" << this_thread::get_id() << "_" << timestamp;
    return oss.str();
}

void* loadKernel(const string& path, const string& func_name) {
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

void* loadKernelWithTimeout(const string& path, const string& func_name, const int timeout_sec) {
    if (timeout_sec <= 0) {
        return loadKernel(path, func_name);
    }

    auto future = async(launch::async, loadKernel, path, func_name);
    auto status = future.wait_for(chrono::seconds(timeout_sec));
    if (status == future_status::timeout) {
        return nullptr;
    } else {
        return future.get();
    }
}

KernelSoStatus searchKernelSo(const filesystem::path& directory) {
    if (!filesystem::exists(directory)) {
        return KernelSoStatus(KernelSoStatus::StatusCode::NotFound, "");
    }

    const string&            format = ".so";
    vector<filesystem::path> matched_files;

    for (const auto& entry : filesystem::directory_iterator(directory)) {
        if (entry.is_regular_file()) {
            const string filename = entry.path().filename().string();

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

KernelSoStatus searchKernelSoWithTimeout(const filesystem::path& directory, const int timeout_sec) {
    if (timeout_sec <= 0) {
        return searchKernelSo(directory);
    }
    auto future = async(launch::async, searchKernelSo, directory);
    auto status = future.wait_for(chrono::seconds(timeout_sec));
    if (status == future_status::timeout) {
        return KernelSoStatus(KernelSoStatus::StatusCode::Timeout, "");
    } else {
        return future.get();
    }
}

string compileAndSaveKernel(const string& dir_path,
                            const string& jit_hdrs_path,
                            const string& cu_file_content,
                            const string& nvcc_command,
                            bool          interleave) {
    RTP_LLM_LOG_INFO("Start compile kernel %s", dir_path.c_str());

    if (!filesystem::exists(dir_path)) {
        filesystem::create_directories(dir_path);
    }

    const string pid_and_timestamp_str = generateKernelName();
    const string cu_filename           = dir_path + "/" + pid_and_timestamp_str + ".cu";
    const string so_filename           = dir_path + "/" + pid_and_timestamp_str + ".so.temp";
    const string so_filename_final     = dir_path + "/" + pid_and_timestamp_str + ".so";

    ofstream cu_file(cu_filename.c_str());
    cu_file << cu_file_content;
    cu_file.close();

    int    result;
    string command;

    command = "/usr/local/cuda/bin/nvcc " + cu_filename + " -o " + so_filename + " " + nvcc_command;
    result  = system(command.c_str());
    if (result != 0) {
        RTP_LLM_FAIL("Failed to run command: \"" + command + "\" with error status: " + to_string(result));
    }

    if (interleave) {
        command = "/opt/conda310/bin/python " + jit_hdrs_path + "/cpp/cuda/deep_gemm/interleave_ffma.py --so " + so_filename;
        result  = system(command.c_str());
        if (result != 0) {
            RTP_LLM_FAIL("Failed to run command: \"" + command + "\" with error status: " + to_string(result));
        }
    }

    filesystem::rename(so_filename, so_filename_final);
    RTP_LLM_LOG_INFO("Finish compile kernel %s", so_filename_final.c_str());

    return so_filename_final;
}

void copyFileToRemote(const filesystem::path& src_path, const filesystem::path& dst_path) {
    string temp_path = dst_path.string() + ".temp";
    if (!filesystem::exists(dst_path.parent_path())) {
        filesystem::create_directories(dst_path.parent_path());
    }
    filesystem::copy(src_path, temp_path);
    filesystem::rename(temp_path, dst_path);
}

int copyFileToRemoteWithTimeout(const filesystem::path& src_path,
                                const filesystem::path& dst_path,
                                const int               timeout_sec) {
    if (timeout_sec <= 0) {
        copyFileToRemote(src_path, dst_path);
        return 0;
    }

    auto future = async(launch::async, copyFileToRemote, src_path, dst_path);
    auto status = future.wait_for(chrono::seconds(timeout_sec));
    if (status == future_status::timeout) {
        return -1;
    } else {
        future.get();
        return 0;
    }
}

}  // namespace rtp_llm