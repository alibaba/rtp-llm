#include "hip_host_utils.h"
#include "rtp_llm/cpp/devices/OpData.h"
#include "rtp_llm/cpp/config/StaticConfig.h"
namespace rtp_llm {
namespace rocm {

bool CaptureCheck::in_hip_graph_capture = false;

static const char* _hipGetErrorEnum(hipError_t error) {
    return hipGetErrorString(error);
}

static const char* _hipGetErrorEnum(hipblasStatus_t error) {
    switch (error) {
        case HIPBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case HIPBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case HIPBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case HIPBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case HIPBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case HIPBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case HIPBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case HIPBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";

        case HIPBLAS_STATUS_NOT_SUPPORTED:
            return "CUBLAS_STATUS_NOT_SUPPORTED";

        case HIPBLAS_STATUS_UNKNOWN:
            return "CUBLAS_STATUS_LICENSE_ERROR";

        case HIPBLAS_STATUS_HANDLE_IS_NULLPTR:
            return "HIPBLAS_STATUS_HANDLE_IS_NULLPTR";

        case HIPBLAS_STATUS_INVALID_ENUM:
            return "HIPBLAS_STATUS_INVALID_ENUM";
    }
    return "<unknown>";
}

void throwRocmError(const char* const file, int const line, std::string const& info) {
    auto error_msg =
        std::string("[ROCm][ERROR] ") + info + " Assertion fail: " + file + ":" + std::to_string(line) + " \n";
    std::printf("%s", error_msg.c_str());
    fflush(stdout);
    fflush(stderr);
    if (rtp_llm::StaticConfig::user_ft_core_dump_on_exception) {
        abort();
    }
    throw RTP_EXCEPTION(error_msg);
}

template<typename T>
void check(T result, const char* const file, int const line) {
    if (result) {
        throwRocmError(file, line, _hipGetErrorEnum(result));
    }
}

void syncAndCheckInDebug(const char* const file, int const line) {
    if (rtp_llm::Logger::getEngineLogger().isDebugMode()) {
        if (!CaptureCheck::in_hip_graph_capture) {
            ROCM_CHECK(hipDeviceSynchronize());
        }
        check(hipGetLastError(), file, line);
        RTP_LLM_LOG_DEBUG(rtp_llm::fmtstr("run syncAndCheckInDebug at %s:%d", file, line));
    }
}

template void check<hipblasStatus_t>(hipblasStatus_t result, const char* const file, int const line);
template void check<hipError_t>(hipError_t result, const char* const file, int const line);

int get_sm() {
    static int sm = []() {
        int device{-1};
        ROCM_CHECK(hipGetDevice(&device));
        int sm_major = 0;
        int sm_minor = 0;
        ROCM_CHECK(hipDeviceGetAttribute(&sm_major, hipDeviceAttributeComputeCapabilityMajor, device));
        ROCM_CHECK(hipDeviceGetAttribute(&sm_minor, hipDeviceAttributeComputeCapabilityMinor, device));
        return sm_major * 10 + sm_minor;
    }();
    return sm;
}

int getDevice() {
    static int device_id = []() {
        int current_dev_id = 0;
        ROCM_CHECK(hipGetDevice(&current_dev_id));
        return current_dev_id;
    }();
    return device_id;
}

int getDeviceCount() {
    static int device_count = []() {
        int count = 0;
        ROCM_CHECK(hipGetDeviceCount(&count));
        return count;
    }();
    return device_count;
}

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
        ROCM_CHECK(hipDeviceGetAttribute(&mp_count, hipDeviceAttributeMultiprocessorCount, device_id));
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
        ROCM_CHECK(hipDeviceGetAttribute(&max_smem, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device_id));
        max_smem_cache[device_id] = max_smem;
        return max_smem;
    }
    return it->second;
}
}  // namespace rocm
}  // namespace rtp_llm
