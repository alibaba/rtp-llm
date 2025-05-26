#include "hip_utils.h"
#include "rtp_llm/cpp/th_op/GlobalConfig.h"
namespace rtp_llm {
namespace rocm {

static const char* _hipGetErrorEnum(hipError_t error)
{
    return hipGetErrorString(error);
}

static const char* _hipGetErrorEnum(hipblasStatus_t error)
{
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
    auto error_msg = std::string("[ROCm][ERROR] ") + info + " Assertion fail: " + file + ":" + std::to_string(line) + " \n";
    std::printf("%s", error_msg.c_str());
    fflush(stdout);
    fflush(stderr);
    if (GlobalConfig::get().profiling_debug_logging_config.ft_core_dump_on_exception) {
        abort();
    }
    throw FT_EXCEPTION(error_msg);
}

template<typename T>
void check(T result, const char* const file, int const line)
{
    if (result) {
        throwRocmError(file, line, _hipGetErrorEnum(result));
    }
}

void syncAndCheckInDebug(const char* const file, int const line) {
    if (rtp_llm::Logger::getEngineLogger().isDebugMode()) {
        ROCM_CHECK(hipDeviceSynchronize());
        check(hipGetLastError(), file, line);
        RTP_LLM_LOG_DEBUG(rtp_llm::fmtstr("run syncAndCheckInDebug at %s:%d", file, line));
    }
}

template void check<hipblasStatus_t>(hipblasStatus_t result, const char* const file, int const line);
template void check<hipError_t>(hipError_t result, const char* const file, int const line);


int get_sm()
{
    int device{-1};
    ROCM_CHECK(hipGetDevice(&device));
    int sm_major = 0;
    int sm_minor = 0;
    ROCM_CHECK(hipDeviceGetAttribute(&sm_major, hipDeviceAttributeComputeCapabilityMajor, device));
    ROCM_CHECK(hipDeviceGetAttribute(&sm_minor, hipDeviceAttributeComputeCapabilityMinor, device));
    return sm_major * 10 + sm_minor;
}

int getDevice()
{
    int current_dev_id = 0;
    ROCM_CHECK(hipGetDevice(&current_dev_id));
    return current_dev_id;
}

int getDeviceCount()
{
    int count = 0;
    ROCM_CHECK(hipGetDeviceCount(&count));
    return count;
}

}  // namespace rocm
}  // namespace rtp_llm
