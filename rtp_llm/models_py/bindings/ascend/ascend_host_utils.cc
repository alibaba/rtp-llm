#include "ascend_host_utils.h"

#if USING_ASCEND
#include <acl/acl.h>
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace ascend {

static bool ascend_graph_capture_enabled = false;

template<typename T>
void check(T result, const char* const file, int const line) {
    if (result != ACL_SUCCESS) {
        RTP_LLM_LOG_ERROR("Ascend error at %s:%d, error code: %d", file, line,
                          static_cast<int>(result));
        throw std::runtime_error("Ascend runtime error");
    }
}

template void check<aclError>(aclError result, const char* const file, int const line);

void syncAndCheckInDebug(const char* const file, int const line) {
    aclError err = aclrtSynchronizeDevice();
    if (err != ACL_SUCCESS) {
        RTP_LLM_LOG_ERROR("Ascend sync error at %s:%d, error code: %d", file, line,
                          static_cast<int>(err));
        throw std::runtime_error("Ascend sync error");
    }
}

int getDevice() {
    int32_t device_id = 0;
    aclrtGetDevice(&device_id);
    return device_id;
}

int getDeviceCount() {
    uint32_t count = 0;
    aclrtGetDeviceCount(&count);
    return static_cast<int>(count);
}

int currentDeviceId() {
    return getDevice();
}

std::tuple<size_t, size_t> getDeviceMemoryInfo(bool const useUvm) {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    aclrtGetMemInfo(ACL_HBM_MEM, &free_bytes, &total_bytes);
    return {total_bytes - free_bytes, free_bytes};
}

int getMultiProcessorCount(int device_id) {
    return 0;
}

void setAscendGraphCaptureEnabled(bool enabled) {
    ascend_graph_capture_enabled = enabled;
}

bool isAscendGraphCaptureEnabled() {
    return ascend_graph_capture_enabled;
}

}  // namespace ascend
}  // namespace rtp_llm

#endif  // USING_ASCEND
