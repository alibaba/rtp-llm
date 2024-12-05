#include <vector>

#include "src/fastertransformer/devices/DeviceBase.h"
#include "src/fastertransformer/devices/DeviceFactory.h"

namespace ft = fastertransformer;

namespace rtp_llm {

class DeviceUtil {
public:
    DeviceUtil();
    ~DeviceUtil();

    void* mallocCPU(size_t size);
    void  freeCPU(void* ptr);
    void* mallocGPU(size_t size);
    void  freeGPU(void* ptr);
    void  memsetCPU(void*, int value, size_t len);
    bool  memsetGPU(void*, int value, size_t len);
    bool  memcopy(void* dst, bool dst_gpu, const void* src, bool src_gpu, size_t size);

public:
    ft::DeviceBase*                          device_;
    std::unordered_map<void*, ft::BufferPtr> buffer_map_;
private:
    std::mutex mutex_;
};

}  // namespace rtp_llm
