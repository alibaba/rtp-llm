#pragma once

#include <vector>
#include <cstddef>
#include <cstdint>

namespace rtp_llm {
namespace transfer {
namespace tcp {

/// @brief 待拷贝的 buffer 信息
struct CopyTask {
    void*  src_ptr;  // 源地址
    size_t size;     // 拷贝大小
    char*  dst_ptr;  // 目标地址，必须由调用者预先分配
    // Defaults preserve legacy direct CopyTask callers; cache-store callers
    // must populate these fields from the originating BlockInfo.
    bool    is_cuda      = true;
    int32_t device_index = -1;
    int32_t device_type  = -1;  // c10::DeviceType; -1 resolves via is_cuda
};

/// @brief GPU 拷贝工具类，封装批量拷贝逻辑
/// CPU buffers use host copies; CUDA/ROCm buffers use execNoBlockCopy.
/// Other device kinds are rejected before any copy, never treated as host memory.
class CudaCopyUtil {
public:
    CudaCopyUtil()  = default;
    ~CudaCopyUtil() = default;

    // 禁止拷贝
    CudaCopyUtil(const CudaCopyUtil&)            = delete;
    CudaCopyUtil& operator=(const CudaCopyUtil&) = delete;

    /// @brief 批量执行 GPU -> CPU 拷贝
    /// @param tasks 待拷贝任务列表（dst_ptr 必须由调用者预先分配）
    /// @return 成功返回 true，若 device 未初始化或 dst_ptr 为 nullptr 则返回 false
    bool batchCopyToHost(std::vector<CopyTask>& tasks);

    /// @brief 批量执行 CPU -> GPU 拷贝
    /// @param tasks 待拷贝任务列表（dst_ptr 必须由调用者预先分配）
    /// @return 成功返回 true，若 device 未初始化或 dst_ptr 为 nullptr 则返回 false
    bool batchCopyToDevice(std::vector<CopyTask>& tasks);
};

}  // namespace tcp
}  // namespace transfer
}  // namespace rtp_llm
