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
#include <thread>
#include <stdexcept>
#include <sstream>
#include "rtp_llm/cpp/utils/AssertUtils.h"

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

std::string getFilesHash(std::filesystem::path path, bool interleave = true);

std::string getDeepGemmTypeStr(DeepGemmType type);

std::string getJITPath();

void collectFiles(const std::filesystem::path& dir_path, std::vector<std::filesystem::path>& files);

std::string generateKernelName();

std::string getRemoteJITDir();

void* loadKernel(const std::string& path, const std::string& params_str);

void* searchAndLoadKernel(const std::filesystem::path& directory, const std::string& params_str);

void* findCachedKernel(const std::filesystem::path& remote_dir_path,
                       const std::filesystem::path& local_dir_path,
                       const std::string&           params_str);

std::string compileAndSaveKernel(const std::filesystem::path& local_dir_path,
                                 const std::filesystem::path& remote_dir_path,
                                 const std::string&           cu_file_content,
                                 const std::string&           nvcc_command,
                                 bool                         interleave = true);

extern const std::string jit_hdrs_path;
extern const std::string remote_jit_dir;

}  // namespace rtp_llm
