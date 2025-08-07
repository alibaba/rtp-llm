#pragma once
#include <mutex>
#include <fcntl.h>
#include <unistd.h>
#include <vector>
#include <cstdlib>
#include <dlfcn.h>
#include <iostream>
#include <string>
#include <filesystem>
#include <algorithm>
#include <random>
#include <chrono>

namespace rtp_llm {

enum class DeepGemmType {
    Normal,
    GroupedContiguous,
    GroupedMasked
};

struct KernelPathCacheStatus {
    bool                  find;
    std::filesystem::path path;
    KernelPathCacheStatus(bool find = false, std::filesystem::path path = std::string("")): find(find), path(path) {}
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

inline std::string getDeepGemmTypeStr(DeepGemmType type) {
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

inline std::string getJITPath() {
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

inline void collectFiles(const std::filesystem::path& dir_path, std::vector<std::filesystem::path>& files) {
    for (const auto& entry : std::filesystem::recursive_directory_iterator(dir_path)) {
        if (entry.is_regular_file()) {
            files.emplace_back(entry.path());
        }
    }
}

inline std::string generateKernelName() {
    auto now       = std::chrono::system_clock::now();
    auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();

    std::ostringstream oss;
    oss << getpid() << "_" << timestamp;
    return oss.str();
}

inline KernelPathCacheStatus findMatchingFiles(const std::filesystem::path& directory, const std::string& format) {
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
        return KernelPathCacheStatus(false, std::string(""));
    }

    std::random_device              rd;
    std::mt19937                    gen(rd());
    std::uniform_int_distribution<> dis(0, matched_files.size() - 1);

    return KernelPathCacheStatus(true, matched_files[dis(gen)]);
}

}  // namespace rtp_llm
