/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "rtp_llm/cpp/cuda/cuda_utils.h"
#include "rtp_llm/cpp/utils/StackTrace.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include <cuda.h>
#include <ATen/cuda/CUDAGraph.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGraphsUtils.cuh>
#include <mutex>
#include <unordered_map>
#include <stdio.h>
#include <stdlib.h>

namespace rtp_llm {

static const char* _cudaGetErrorEnum(cudaError_t error) {
    return cudaGetErrorString(error);
}

static const char* _cudaGetErrorEnum(cublasStatus_t error) {
    switch (error) {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";

        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "CUBLAS_STATUS_NOT_SUPPORTED";

        case CUBLAS_STATUS_LICENSE_ERROR:
            return "CUBLAS_STATUS_LICENSE_ERROR";
    }
    return "<unknown>";
}

static const char* _cudaGetErrorEnum(CUresult error) {
    const char* error_name  = nullptr;
    CUresult    name_result = cuGetErrorName(error, &error_name);
    if (name_result != CUDA_SUCCESS) {
        RTP_LLM_LOG_WARNING("Failed to get error name, %d", int(error));
        return "Unknown CUDA error (failed to get error name)";
    }
    return error_name;
}

template<typename T>
void check(T result, const char* const file, int const line) {
    if (result) {
        std::string error_str = std::string("[FT][ERROR] CUDA runtime error: ") + (_cudaGetErrorEnum(result)) + " "
                                + file + ":" + std::to_string(line);
        printStackTrace();
        RTP_LLM_LOG_ERROR(error_str);
        fflush(stdout);
        fflush(stderr);
        throw std::runtime_error(error_str);
    }
}

template void check<cudaError_t>(cudaError_t result, const char* const file, int const line);
template void check<cublasStatus_t>(cublasStatus_t result, const char* const file, int const line);
template void check<CUresult>(CUresult result, const char* const file, int const line);

void syncAndCheckInDebug(const char* const file, int const line) {
    if (rtp_llm::Logger::getEngineLogger().isDebugMode()) {
        if (at::cuda::currentStreamCaptureStatus() == at::cuda::CaptureStatus::None) {
            cudaDeviceSynchronize();
        }
        cudaError_t result = cudaGetLastError();
        check(result, file, line);
        RTP_LLM_LOG_DEBUG(rtp_llm::fmtstr("run syncAndCheckInDebug at %s:%d", file, line));
    }
}

int get_sm() {
    static int sm = []() {
        int device;
        check_cuda_value(cudaGetDevice(&device));
        cudaDeviceProp deviceProp;
        check_cuda_value(cudaGetDeviceProperties(&deviceProp, device));
        return deviceProp.major * 10 + deviceProp.minor;
    }();
    return sm;
}

bool is_sm70() {
    static bool IS_SM70 = []() { return get_sm() == 70; }();
    return IS_SM70;
}

bool is_sm8x() {
    static bool IS_SM8X = []() { return (get_sm() >= 80) && (get_sm() <= 89); }();
    return IS_SM8X;
}

bool is_sm90() {
    static bool IS_SM90 = []() { return get_sm() == 90; }();
    return IS_SM90;
}

bool is_sm100() {
    static bool IS_SM100 = []() { return get_sm() == 100; }();
    return IS_SM100;
}

float timing_function(const std::function<void(cudaStream_t)>& operation,
                      int64_t                                  timing_iterations,
                      cudaStream_t                             stream) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaDeviceSynchronize();
    cudaEventRecord(start, stream);

    for (int64_t iter = 0; iter < timing_iterations; ++iter) {
        operation(stream);
    }

    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    float total_time_ms = 0;
    cudaEventElapsedTime(&total_time_ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return total_time_ms / float(timing_iterations);
}

int getDevice() {
    // Note: Device ID could change if cudaSetDevice is called, but in many applications it's fixed
    // Use thread_local static to cache per thread
    static thread_local int device_id = []() {
        int current_dev_id = 0;
        check_cuda_value(cudaGetDevice(&current_dev_id));
        return current_dev_id;
    }();
    return device_id;
}

int getDeviceCount() {
    // Device count is hardware fixed and never changes during program execution
    static int device_count = []() {
        int count = 0;
        check_cuda_value(cudaGetDeviceCount(&count));
        return count;
    }();
    return device_count;
}

int currentDeviceId() {
    // Use the same caching strategy as getDevice()
    return getDevice();
}

void priorityRange(int* low_priority, int* high_priority, int device_id) {
    static std::vector<std::pair<int, int>> cache(getDeviceCount());
    static std::vector<std::once_flag>      flags(getDeviceCount());
    if (device_id < 0) {
        device_id = currentDeviceId();
    }
    RTP_LLM_CHECK_WITH_INFO(0 <= device_id && device_id < getDeviceCount(), "invalid CUDA device ID");
    auto init = [&]() {
        int ori_dev = currentDeviceId();
        if (device_id != ori_dev) {
            check_cuda_value(cudaSetDevice(device_id));
        }
        int min_pri, max_pri;
        check_cuda_value(cudaDeviceGetStreamPriorityRange(&min_pri, &max_pri));
        if (device_id != ori_dev) {
            check_cuda_value(cudaSetDevice(ori_dev));
        }
        cache[device_id] = std::make_pair(min_pri, max_pri);
    };
    std::call_once(flags[device_id], init);
    *low_priority  = cache[device_id].first;
    *high_priority = cache[device_id].second;
}

std::string getDriverVersion() {
    // Driver version is system-wide and doesn't change during program execution
    static std::string driver_version = []() {
        nvmlReturn_t result;
        nvmlDevice_t device;
        size_t       device_count = getDeviceCount();
        if (device_count == 0) {
            throw std::runtime_error("no cuda device");
        }

        result = nvmlInit();
        if (NVML_SUCCESS != result) {
            throw std::runtime_error("Failed to initialize NVML, Error code: " + std::to_string(result));
        }

        char pci_bus_id[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
        check_cuda_value(cudaDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), 0));
        result = nvmlDeviceGetHandleByPciBusId(pci_bus_id, &device);
        if (NVML_SUCCESS != result) {
            throw std::runtime_error("Failed to call nvmlDeviceGetHandleByIndex() API, Error code:"
                                     + std::to_string(result));
        }

        char driverVersion[NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE];
        result = nvmlSystemGetDriverVersion(driverVersion, NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE);
        if (NVML_SUCCESS != result) {
            throw std::runtime_error("Failed to call nvmlSystemGetDriverVersion() API, Error code: "
                                     + std::to_string(result));
        }
        result = nvmlShutdown();
        if (NVML_SUCCESS != result) {
            RTP_LLM_LOG_INFO("Failed to shutdown NVML, Error code: %s", std::to_string(result).c_str());
        }
        return std::string(driverVersion);
    }();
    return driver_version;
}

int getCudaVersion() {
    // CUDA version is system-wide and doesn't change during program execution
    static int cuda_version = []() {
        int cuda_driver_version;
        check_cuda_value(cudaDriverGetVersion(&cuda_driver_version));
        return cuda_driver_version;
    }();
    return cuda_version;
}

bool checkP2PAvailable(const std::vector<size_t>& tp_ranks, size_t rank) {
    // check P2P access
    for (size_t i = 0; i < tp_ranks.size(); i++) {
        size_t peer_rank = tp_ranks[i];
        if (peer_rank == rank) {
            continue;
        }
        int peer_access_available = 0;
        check_cuda_value(cudaDeviceCanAccessPeer(&peer_access_available, rank, i));
        if (peer_access_available == 0) {
            return false;
        }
    }
    return true;
}

bool checkAllNVLinks(std::vector<size_t> device_ids) {
    nvmlReturn_t result;
    nvmlDevice_t deviceHandles[2];

    result = nvmlInit();
    if (NVML_SUCCESS != result) {
        throw std::runtime_error("Failed to initialize NVML, Error code: " + std::to_string(result));
    }

    for (size_t i = 0; i < device_ids.size(); i++) {
        for (size_t j = i + 1; j < device_ids.size(); j++) {
            size_t device_id1 = device_ids[i];
            size_t device_id2 = device_ids[j];

            char pci_bus_id1[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
            check_cuda_value(cudaDeviceGetPCIBusId(pci_bus_id1, sizeof(pci_bus_id1), device_id1));
            result = nvmlDeviceGetHandleByPciBusId(pci_bus_id1, &deviceHandles[0]);
            if (NVML_SUCCESS != result) {
                throw std::runtime_error("Failed to get handle for device " + std::to_string(device_id1)
                                         + ", Error code: " + std::to_string(result));
            }

            char pci_bus_id2[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
            check_cuda_value(cudaDeviceGetPCIBusId(pci_bus_id2, sizeof(pci_bus_id2), device_id2));
            result = nvmlDeviceGetHandleByPciBusId(pci_bus_id2, &deviceHandles[1]);
            if (NVML_SUCCESS != result) {
                throw std::runtime_error("Failed to get handle for device " + std::to_string(device_id2)
                                         + ", Error code: " + std::to_string(result));
            }

            nvmlGpuP2PStatus_t isActive;
            result = nvmlDeviceGetP2PStatus(deviceHandles[0], deviceHandles[1], NVML_P2P_CAPS_INDEX_NVLINK, &isActive);
            if (NVML_SUCCESS != result) {
                throw std::runtime_error("Failed to call nvmlDeviceGetP2PStatus() API, Error code: "
                                         + std::to_string(result));
            }
            if (isActive != NVML_P2P_STATUS_OK) {
                RTP_LLM_LOG_INFO("GPU %d and GPU %d are not connected via NVLink", device_id1, device_id2);
                return false;
            }
        }
    }
    result = nvmlShutdown();
    if (NVML_SUCCESS != result) {
        RTP_LLM_LOG_INFO("Failed to shutdown NVML, Error code: %s", std::to_string(result).c_str());
    }
    RTP_LLM_LOG_INFO("All GPUs are connected via NVLink");
    return true;
}

bool checkOnSameNumaNodes(std::vector<size_t> device_ids) {
    nvmlReturn_t result;
    nvmlDevice_t deviceHandles[2];

    result = nvmlInit();
    if (NVML_SUCCESS != result) {
        throw std::runtime_error("Failed to initialize NVML, Error code: " + std::to_string(result));
    }

    for (size_t i = 0; i < device_ids.size(); i++) {
        for (size_t j = i + 1; j < device_ids.size(); j++) {
            size_t device_id1 = device_ids[i];
            size_t device_id2 = device_ids[j];

            char pci_bus_id1[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
            check_cuda_value(cudaDeviceGetPCIBusId(pci_bus_id1, sizeof(pci_bus_id1), device_id1));
            result = nvmlDeviceGetHandleByPciBusId(pci_bus_id1, &deviceHandles[0]);
            if (NVML_SUCCESS != result) {
                throw std::runtime_error("Failed to get handle for device " + std::to_string(device_id1)
                                         + ", Error code: " + std::to_string(result));
            }

            char pci_bus_id2[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
            check_cuda_value(cudaDeviceGetPCIBusId(pci_bus_id2, sizeof(pci_bus_id2), device_id2));
            result = nvmlDeviceGetHandleByPciBusId(pci_bus_id2, &deviceHandles[1]);
            if (NVML_SUCCESS != result) {
                throw std::runtime_error("Failed to get handle for device " + std::to_string(device_id2)
                                         + ", Error code: " + std::to_string(result));
            }

            nvmlGpuTopologyLevel_t topo;
            result = nvmlDeviceGetTopologyCommonAncestor(deviceHandles[0], deviceHandles[1], &topo);
            if (NVML_SUCCESS != result) {
                throw std::runtime_error("Failed to call nvmlDeviceGetTopologyCommonAncestor() API, Error code: "
                                         + std::to_string(result));
            }
            if (topo == NVML_TOPOLOGY_SYSTEM) {
                RTP_LLM_LOG_INFO("GPU %d and GPU %d are not on same numa node", device_id1, device_id2);
                return false;
            }
        }
    }
    result = nvmlShutdown();
    if (NVML_SUCCESS != result) {
        RTP_LLM_LOG_INFO("Failed to shutdown NVML, Error code: %s", std::to_string(result).c_str());
    }
    RTP_LLM_LOG_INFO("All GPUs are on same numa node");
    return true;
}

int getVisibleDeviceNum() {
    return getDeviceCount();
}

std::tuple<size_t, size_t> getDeviceMemoryInfo(bool const useUvm) {
    if (useUvm) {
        size_t freeSysMem, totalSysMem;
#ifndef _WIN32  // Linux
        struct sysinfo info;
        sysinfo(&info);
        totalSysMem = info.totalram * info.mem_unit;
        freeSysMem  = info.freeram * info.mem_unit;
#else   // Windows
        MEMORYSTATUSEX memInfo;
        memInfo.dwLength = sizeof(memInfo);
        GlobalMemoryStatusEx(&memInfo);
        totalSysMem = memInfo.ullTotalPhys;
        freeSysMem  = memInfo.ullAvailPhys;
#endif  // WIN32

        RTP_LLM_LOG_INFO("Using UVM based system memory for KV cache, total memory %0.2f GB, available memory %0.2f GB",
                         ((double)totalSysMem / 1e9),
                         ((double)freeSysMem / 1e9));
        return {freeSysMem, totalSysMem};
    } else {
        size_t free, total;
        check_cuda_value(cudaMemGetInfo(&free, &total));
        RTP_LLM_LOG_DEBUG("Using GPU memory for KV cache, total memory %0.2f GB, available memory %0.2f GB",
                          ((double)total / 1e9),
                          ((double)free / 1e9));
        return {free, total};
    }
}

bool shared_mem_sufficient(int smem_size) {
    // Cache device properties per device since they're hardware fixed
    static std::unordered_map<int, cudaDeviceProp> device_props_cache;
    static std::mutex                              cache_mutex;

    int device_idx = getDevice();

    std::lock_guard<std::mutex> lock(cache_mutex);
    auto                        it = device_props_cache.find(device_idx);
    if (it == device_props_cache.end()) {
        cudaDeviceProp properties;
        check_cuda_value(cudaGetDeviceProperties(&properties, device_idx));
        device_props_cache[device_idx] = properties;
        return int(properties.sharedMemPerMultiprocessor) >= smem_size;
    } else {
        return int(it->second.sharedMemPerMultiprocessor) >= smem_size;
    }
}

bool should_print() {
    int tp_rank = ParallelInfo::globalParallelInfo().getWorldRank();
    if (tp_rank != 0) {
        return false;
    }

    return rtp_llm::Logger::getEngineLogger().isTraceMode();
}

// Device property utility functions with static caching
int getMultiProcessorCount(int device_id) {
    static std::unordered_map<int, int> mp_count_cache;
    static std::mutex                   cache_mutex;

    if (device_id < 0) {
        device_id = getDevice();
    }

    std::lock_guard<std::mutex> lock(cache_mutex);
    auto                        it = mp_count_cache.find(device_id);
    if (it == mp_count_cache.end()) {
        int mp_count;
        check_cuda_value(cudaDeviceGetAttribute(&mp_count, cudaDevAttrMultiProcessorCount, device_id));
        mp_count_cache[device_id] = mp_count;
        return mp_count;
    }
    return it->second;
}

int getMaxSharedMemoryPerMultiprocessor(int device_id) {
    static std::unordered_map<int, int> max_smem_cache;
    static std::mutex                   cache_mutex;

    if (device_id < 0) {
        device_id = getDevice();
    }

    std::lock_guard<std::mutex> lock(cache_mutex);
    auto                        it = max_smem_cache.find(device_id);
    if (it == max_smem_cache.end()) {
        int max_smem;
        check_cuda_value(cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerMultiprocessor, device_id));
        max_smem_cache[device_id] = max_smem;
        return max_smem;
    }
    return it->second;
}

int getMaxSharedMemoryPerBlockOptin(int device_id) {
    static std::unordered_map<int, int> max_smem_block_cache;
    static std::mutex                   cache_mutex;

    if (device_id < 0) {
        device_id = getDevice();
    }

    std::lock_guard<std::mutex> lock(cache_mutex);
    auto                        it = max_smem_block_cache.find(device_id);
    if (it == max_smem_block_cache.end()) {
        int max_smem_block;
        check_cuda_value(cudaDeviceGetAttribute(&max_smem_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device_id));
        max_smem_block_cache[device_id] = max_smem_block;
        return max_smem_block;
    }
    return it->second;
}

int getMaxThreadsPerMultiprocessor(int device_id) {
    static std::unordered_map<int, int> max_threads_cache;
    static std::mutex                   cache_mutex;

    if (device_id < 0) {
        device_id = getDevice();
    }

    std::lock_guard<std::mutex> lock(cache_mutex);
    auto                        it = max_threads_cache.find(device_id);
    if (it == max_threads_cache.end()) {
        int max_threads;
        check_cuda_value(cudaDeviceGetAttribute(&max_threads, cudaDevAttrMaxThreadsPerMultiProcessor, device_id));
        max_threads_cache[device_id] = max_threads;
        return max_threads;
    }
    return it->second;
}

int getMaxBlocksPerMultiprocessor(int device_id) {
    static std::unordered_map<int, int> max_blocks_cache;
    static std::mutex                   cache_mutex;

    if (device_id < 0) {
        device_id = getDevice();
    }

    std::lock_guard<std::mutex> lock(cache_mutex);
    auto                        it = max_blocks_cache.find(device_id);
    if (it == max_blocks_cache.end()) {
        int max_blocks;
        check_cuda_value(cudaDeviceGetAttribute(&max_blocks, cudaDevAttrMaxBlocksPerMultiprocessor, device_id));
        max_blocks_cache[device_id] = max_blocks;
        return max_blocks;
    }
    return it->second;
}

int getComputeCapabilityMajor(int device_id) {
    static std::unordered_map<int, int> cc_major_cache;
    static std::mutex                   cache_mutex;

    if (device_id < 0) {
        device_id = getDevice();
    }

    std::lock_guard<std::mutex> lock(cache_mutex);
    auto                        it = cc_major_cache.find(device_id);
    if (it == cc_major_cache.end()) {
        int cc_major;
        check_cuda_value(cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, device_id));
        cc_major_cache[device_id] = cc_major;
        return cc_major;
    }
    return it->second;
}

int getComputeCapabilityMinor(int device_id) {
    static std::unordered_map<int, int> cc_minor_cache;
    static std::mutex                   cache_mutex;

    if (device_id < 0) {
        device_id = getDevice();
    }

    std::lock_guard<std::mutex> lock(cache_mutex);
    auto                        it = cc_minor_cache.find(device_id);
    if (it == cc_minor_cache.end()) {
        int cc_minor;
        check_cuda_value(cudaDeviceGetAttribute(&cc_minor, cudaDevAttrComputeCapabilityMinor, device_id));
        cc_minor_cache[device_id] = cc_minor;
        return cc_minor;
    }
    return it->second;
}

std::pair<int, int> getComputeCapability(int device_id) {
    // Return both major and minor in one call to avoid double caching overhead
    return std::make_pair(getComputeCapabilityMajor(device_id), getComputeCapabilityMinor(device_id));
}

/*
  b = batch_szie
  s = seq_len
  h = head_num
  d = hidden_per_head/hidden
 */
template<typename T>
void print_bshd(const int   layer_id,
                const char* name,
                const T*    ptr,
                int         batch_size,
                int         seq_len,
                int         num_heads,
                int         hidden_size_per_head,
                int         total_num_heads,
                int         head_offset,
                bool        is_device_ptr) {
    if (!should_print()) {
        return;
    }
    if (total_num_heads == 0) {
        total_num_heads = num_heads;
    }
    T* cpu_ptr = nullptr;
    if (is_device_ptr) {
        cudaDeviceSynchronize();
        auto size = batch_size * seq_len * total_num_heads * hidden_size_per_head * sizeof(T);
        cpu_ptr   = reinterpret_cast<T*>(malloc(size));
        check_cuda_value(cudaMemcpy(cpu_ptr, ptr, size, cudaMemcpyDeviceToHost));
        cudaDeviceSynchronize();
    } else {
        cpu_ptr = const_cast<T*>(ptr);
    }
    printf("layer_id: %d %s [%d %d %d %d]\n", layer_id, name, batch_size, seq_len, num_heads, hidden_size_per_head);
    fflush(stdout);
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < seq_len; j++) {
            auto print_func = [&](int head_start, int head_end) {
                auto md_array_ptr = (T(*)[seq_len][total_num_heads][hidden_size_per_head])cpu_ptr;
                for (int k = head_start; k < head_end; k++) {
                    printf("b_%d s_%d h_%d ", i, j, k);
                    fflush(stdout);
                    int kk = k + head_offset;
                    for (int d = 0; d < 4; d++) {
                        printf("%f ", float(md_array_ptr[i][j][kk][d]));
                    }
                    printf(" ...... ");
                    for (int d = std::max(0, hidden_size_per_head - 4); d < hidden_size_per_head; d++) {
                        printf("%f ", float(md_array_ptr[i][j][kk][d]));
                    }
                    printf("\n");
                    fflush(stdout);
                }
            };
            print_func(0, std::min(num_heads, 4));
            print_func(std::max(0, num_heads - 4), num_heads);
        }
    }
    fflush(stdout);
}

template<typename T>
void print_bhsd(const int   layer_id,
                const char* name,
                const T*    ptr,
                int         batch_size,
                int         num_heads,
                int         seq_len,
                int         hidden_size_per_head,
                bool        is_device_ptr) {
    if (!should_print()) {
        return;
    }
    T* cpu_ptr = nullptr;
    if (is_device_ptr) {
        cudaDeviceSynchronize();
        auto size = batch_size * seq_len * num_heads * hidden_size_per_head * sizeof(T);
        cpu_ptr   = reinterpret_cast<T*>(malloc(size));
        check_cuda_value(cudaMemcpy(cpu_ptr, ptr, size, cudaMemcpyDeviceToHost));
        cudaDeviceSynchronize();
    } else {
        cpu_ptr = const_cast<T*>(ptr);
    }
    printf("layer_id: %d %s [%d %d %d %d]\n", layer_id, name, batch_size, num_heads, seq_len, hidden_size_per_head);
    fflush(stdout);
    auto md_array_ptr = (T(*)[num_heads][seq_len][hidden_size_per_head])cpu_ptr;
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < seq_len; j++) {
            for (int k = 0; k < std::min(num_heads, 4); k++) {
                printf("b_%d s_%d h_%d %f %f %f %f\n",
                       i,
                       j,
                       k,
                       float(md_array_ptr[i][k][j][0]),
                       float(md_array_ptr[i][k][j][1]),
                       float(md_array_ptr[i][k][j][2]),
                       float(md_array_ptr[i][k][j][3]));
                fflush(stdout);
            }
        }
    }
}

template<typename T>
void print_bhss(const int   layer_id,
                const char* name,
                const T*    ptr,
                int         batch_size,
                int         num_heads,
                int         seq_len,
                int         seq_len2,
                bool        is_device_ptr) {
    if (!should_print()) {
        return;
    }
    T* cpu_ptr = nullptr;
    if (is_device_ptr) {
        cudaDeviceSynchronize();
        uint64_t size = (uint64_t)batch_size * (uint64_t)num_heads * (uint64_t)seq_len * (uint64_t)seq_len2 * sizeof(T);
        cpu_ptr       = reinterpret_cast<T*>(malloc(size));
        check_cuda_value(cudaMemcpy(cpu_ptr, ptr, size, cudaMemcpyDeviceToHost));
        cudaDeviceSynchronize();
    } else {
        cpu_ptr = const_cast<T*>(ptr);
    }
    printf("layer_id: %d %s [%d %d %d %d]\n", layer_id, name, batch_size, num_heads, seq_len, seq_len2);
    fflush(stdout);
    auto md_array_ptr = (T(*)[num_heads][seq_len][seq_len2])cpu_ptr;
    for (int i = 0; i < batch_size; i++) {
        for (int head = 0; head < std::min(num_heads, 4); head++) {
            for (int j1 = 0; j1 < seq_len; j1++) {
                for (int j2 = 0; j2 < seq_len2; j2++) {
                    printf("b_%d h_%d s_%d_%d %f \n", i, head, j1, j2, float(md_array_ptr[i][head][j1][j2]));
                    fflush(stdout);
                }
            }
        }
    }
}

template<typename T>
void print_bsd(const int   layer_id,
               const char* name,
               const T*    ptr,
               int         batch_size,
               int         seq_len,
               int         hidden_size,
               int         start,
               int         end,
               bool        is_device_ptr) {
    if (!should_print()) {
        return;
    }
    T* cpu_ptr = nullptr;
    if (is_device_ptr) {
        cudaDeviceSynchronize();
        auto size = batch_size * hidden_size * seq_len * sizeof(T);
        cpu_ptr   = reinterpret_cast<T*>(malloc(size));
        check_cuda_value(cudaMemcpy(cpu_ptr, ptr, size, cudaMemcpyDeviceToHost));
        cudaDeviceSynchronize();
    } else {
        cpu_ptr = const_cast<T*>(ptr);
    }
    printf("layer_id: %d %s [%d %d %d]\n", layer_id, name, batch_size, seq_len, hidden_size);
    fflush(stdout);
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < seq_len; j++) {
            printf("b_%d s_%d ", i, j);
            fflush(stdout);
            double sum1       = 0;
            double sum2       = 0;
            auto   print_func = [&](int k_start, int k_end) {
                auto md_array_ptr = (T(*)[seq_len][hidden_size])cpu_ptr;
                for (int k = k_start; k < k_end && k < hidden_size; k++) {
                    printf("k = %d, value = %f ", k, float(md_array_ptr[i][j][k]));
                    fflush(stdout);
                    sum1 += float(md_array_ptr[i][j][k]);
                    sum2 += float(md_array_ptr[i][j][k]) * float(md_array_ptr[i][j][k]);
                }
            };
            print_func(start, end);
            printf("......");
            print_func(std::max(0, hidden_size - (end - start)), hidden_size);
            printf("\n");
            printf("sum1 = %f, square sum2 = %lf\n", sum1, sum2);
            fflush(stdout);
        }
    }
    fflush(stdout);
}

template<typename T>
void print_kv_cache(const int   layer_id,
                    const char* name,
                    const T*    ptr,
                    int         dim1,
                    int         dim2,
                    int         dim3,
                    int         dim4,
                    int         dim5,
                    int         dim6,
                    bool        print_all,
                    bool        is_device_ptr) {
    if (!should_print()) {
        return;
    }

    std::cout.setf(std::ios::fixed);

    T* cpu_ptr = nullptr;
    if (is_device_ptr) {
        cpu_ptr = reinterpret_cast<T*>(malloc(dim1 * dim2 * dim3 * dim4 * dim5 * dim6 * sizeof(T)));
        check_cuda_value(
            cudaMemcpy(cpu_ptr, ptr, dim1 * dim2 * dim3 * dim4 * dim5 * dim6 * sizeof(T), cudaMemcpyDeviceToHost));
    } else {
        cpu_ptr = const_cast<T*>(ptr);
    }

    printf("layer_id: %d %s [%d %d %d %d %d %d]\n", layer_id, name, dim1, dim2, dim3, dim4, dim5, dim6);
    fflush(stdout);
    for (int i = 0; i < dim1; i++) {
        for (int j = 0; j < dim2; j++) {
            for (int k = 0; k < dim3; k++) {
                for (int x = 0; x < dim4; x++) {
                    for (int y = 0; y < dim5; y++) {
                        printf("i = %d, j = %d, k = %d, x = %d, y = %d\n", i, j, k, x, y);
                        fflush(stdout);
                        if (print_all) {
                            for (int z = 0; z < dim6; z++) {
                                std::cout << float(*(cpu_ptr + i * dim2 * dim3 * dim4 * dim5 * dim6
                                                     + j * dim3 * dim4 * dim5 * dim6 + k * dim4 * dim5 * dim6
                                                     + x * dim5 * dim6 + y * dim6 + z))
                                          << ", ";
                            }
                        }
                        printf("\n");
                    }
                    printf("\n");
                }
                printf("\n\n");
            }
            printf("\n\n");
        }
        printf("\n\n");
    }
    printf("\n\n");
    fflush(stdout);
}

template<typename T>
void print_bsd_sum_and_square(const int   layer_id,
                              const char* name,
                              const T*    ptr,
                              int         batch_size,
                              int         seq_len,
                              int         hidden_size,
                              int         start,
                              int         end,
                              bool        is_device_ptr) {
    if (!should_print()) {
        return;
    }

    auto tp_rank = ParallelInfo::globalParallelInfo().getWorldRank();
    if (tp_rank != 0 && tp_rank != 1) {
        return;
    }

    T* cpu_ptr = nullptr;
    if (is_device_ptr) {
        cudaDeviceSynchronize();
        auto size = batch_size * hidden_size * seq_len * sizeof(T);
        cpu_ptr   = reinterpret_cast<T*>(malloc(size));
        check_cuda_value(cudaMemcpy(cpu_ptr, ptr, size, cudaMemcpyDeviceToHost));
        cudaDeviceSynchronize();
    } else {
        cpu_ptr = const_cast<T*>(ptr);
    }
    auto md_array_ptr = (T(*)[seq_len][hidden_size])cpu_ptr;
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < seq_len; j++) {
            double sum1 = 0;
            double sum2 = 0;
            for (int k = start; k < end; k++) {
                printf("layer_id: %d %s [%d %d %d %d], rank = %d, k = %d, value = %f ",
                       layer_id,
                       name,
                       batch_size,
                       seq_len,
                       hidden_size,
                       j,
                       tp_rank,
                       k,
                       float(md_array_ptr[i][j][k]));
                sum1 += float(md_array_ptr[i][j][k]);
                sum2 += float(md_array_ptr[i][j][k]) * float(md_array_ptr[i][j][k]);
            }
            printf("\nlayer_id: %d %s [%d %d %d %d], rank = %d, sum1 = %f, square sum2 = %lf\n",
                   layer_id,
                   name,
                   batch_size,
                   seq_len,
                   hidden_size,
                   j,
                   tp_rank,
                   sum1,
                   sum2);
        }
    }
    fflush(stdout);
}

#define DECLARE_PRINT_TYPE(CPP_TYPE)                                                                                   \
    template void print_bhsd<CPP_TYPE>(const int       layer_id,                                                       \
                                       const char*     name,                                                           \
                                       const CPP_TYPE* ptr,                                                            \
                                       int             batch_size,                                                     \
                                       int             num_heads,                                                      \
                                       int             seq_len,                                                        \
                                       int             hidden_size_per_head,                                           \
                                       bool            is_device_ptr);                                                            \
    template void print_bshd<CPP_TYPE>(const int       layer_id,                                                       \
                                       const char*     name,                                                           \
                                       const CPP_TYPE* ptr,                                                            \
                                       int             batch_size,                                                     \
                                       int             seq_len,                                                        \
                                       int             num_heads,                                                      \
                                       int             hidden_size_per_head,                                           \
                                       int             total_num_heads,                                                \
                                       int             heads_offset,                                                   \
                                       bool            is_device_ptr);                                                            \
                                                                                                                       \
    template void print_bhss<CPP_TYPE>(const int       layer_id,                                                       \
                                       const char*     name,                                                           \
                                       const CPP_TYPE* ptr,                                                            \
                                       int             batch_size,                                                     \
                                       int             num_heads,                                                      \
                                       int             seq_len,                                                        \
                                       int             seq_len2,                                                       \
                                       bool            is_device_ptr);                                                            \
    template void print_bsd<CPP_TYPE>(const int       layer_id,                                                        \
                                      const char*     name,                                                            \
                                      const CPP_TYPE* ptr,                                                             \
                                      int             batch_size,                                                      \
                                      int             seq_len,                                                         \
                                      int             hidden_size,                                                     \
                                      int             start,                                                           \
                                      int             end,                                                             \
                                      bool            is_device_ptr);                                                             \
    template void print_bsd_sum_and_square<CPP_TYPE>(const int       layer_id,                                         \
                                                     const char*     name,                                             \
                                                     const CPP_TYPE* ptr,                                              \
                                                     int             batch_size,                                       \
                                                     int             seq_len,                                          \
                                                     int             hidden_size,                                      \
                                                     int             start,                                            \
                                                     int             end,                                              \
                                                     bool            is_device_ptr);                                              \
    template void print_kv_cache<CPP_TYPE>(const int       layer_id,                                                   \
                                           const char*     name,                                                       \
                                           const CPP_TYPE* ptr,                                                        \
                                           int             dim1,                                                       \
                                           int             dim2,                                                       \
                                           int             dim3,                                                       \
                                           int             dim4,                                                       \
                                           int             dim5,                                                       \
                                           int             dim6,                                                       \
                                           bool            print_all,                                                  \
                                           bool            is_device_ptr);

DECLARE_PRINT_TYPE(float);
DECLARE_PRINT_TYPE(half);
DECLARE_PRINT_TYPE(__nv_bfloat16);
DECLARE_PRINT_TYPE(int8_t);
DECLARE_PRINT_TYPE(uint8_t);
DECLARE_PRINT_TYPE(int);
DECLARE_PRINT_TYPE(int64_t);
#ifdef ENABLE_FP8
DECLARE_PRINT_TYPE(__nv_fp8_e4m3);
#endif

}  // namespace rtp_llm
