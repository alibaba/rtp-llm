#include <vector>

#include "rtp_llm/models_py/bindings/core/ExecOps.h"

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
    std::unordered_map<void*, torch::Tensor> buffer_map_;

private:
    std::mutex mutex_;
};

}  // namespace rtp_llm
