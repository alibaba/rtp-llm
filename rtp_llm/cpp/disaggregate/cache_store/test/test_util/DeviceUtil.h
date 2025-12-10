#ifndef RTP_LLM_CPP_DISAGGREGATE_CACHE_STORE_TEST_TEST_UTIL_DEVICEUTIL_H_
#define RTP_LLM_CPP_DISAGGREGATE_CACHE_STORE_TEST_TEST_UTIL_DEVICEUTIL_H_

#include <vector>

#include "rtp_llm/cpp/devices/DeviceBase.h"
#include "rtp_llm/cpp/devices/DeviceFactory.h"

namespace rtp_llm {

class DeviceUtil {
public:
    DeviceUtil(const DeviceResourceConfig device_resource_config = DeviceResourceConfig());
    ~DeviceUtil();

    void* mallocCPU(size_t size);
    void  freeCPU(void* ptr);
    void* mallocGPU(size_t size);
    void  freeGPU(void* ptr);
    void  memsetCPU(void*, int value, size_t len);
    bool  memsetGPU(void*, int value, size_t len);
    bool  memcopy(void* dst, bool dst_gpu, const void* src, bool src_gpu, size_t size);

public:
    rtp_llm::DeviceBase*                          device_;
    std::unordered_map<void*, rtp_llm::BufferPtr> buffer_map_;

private:
    std::mutex mutex_;
};

}  // namespace rtp_llm

#endif  // RTP_LLM_CPP_DISAGGREGATE_CACHE_STORE_TEST_TEST_UTIL_DEVICEUTIL_H_
