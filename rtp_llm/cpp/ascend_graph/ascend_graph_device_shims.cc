// Ascend NPU graph device shim implementation.
//
// All graph-runtime functions are funneled through this TU so the rest of the
// AscendGraphRunner can stay device-agnostic at the C++ level.

#include "rtp_llm/cpp/ascend_graph/ascend_graph_device_shims.h"

#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace ascend_graph {

namespace {
#if USING_ASCEND
inline aclrtMemcpyKind toAclKind(GraphMemcpyKind kind) {
    switch (kind) {
        case GraphMemcpyKind::D2D: return ACL_MEMCPY_DEVICE_TO_DEVICE;
        case GraphMemcpyKind::D2H: return ACL_MEMCPY_DEVICE_TO_HOST;
        case GraphMemcpyKind::H2D: return ACL_MEMCPY_HOST_TO_DEVICE;
    }
    return ACL_MEMCPY_DEVICE_TO_DEVICE;
}

inline aclmdlRICaptureMode toAclMode(GraphCaptureMode mode) {
    switch (mode) {
        case GraphCaptureMode::Global:      return ACL_MODEL_RI_CAPTURE_MODE_GLOBAL;
        case GraphCaptureMode::ThreadLocal: return ACL_MODEL_RI_CAPTURE_MODE_THREAD_LOCAL;
        case GraphCaptureMode::Relaxed:     return ACL_MODEL_RI_CAPTURE_MODE_RELAXED;
    }
    return ACL_MODEL_RI_CAPTURE_MODE_THREAD_LOCAL;
}
#endif

#if USING_ASCEND
bool g_ascend_graph_capture_enabled = false;
#else
bool g_ascend_graph_capture_enabled_unused = false;
#endif
}  // namespace

#if USING_ASCEND
GraphStream graphGetStreamFromPool(bool is_high_priority) {
    return c10_npu::getStreamFromPool(is_high_priority, c10_npu::current_device());
}

GraphStream graphGetCurrentStream() {
    return c10_npu::getCurrentNPUStream(c10_npu::current_device());
}

void graphSetCurrentStream(const GraphStream& stream) {
    c10_npu::setCurrentNPUStream(stream);
}

torch::Event makeGraphEvent() {
    return torch::Event(c10::DeviceType::PrivateUse1);
}

void graphMemcpyAsync(void* dst, const void* src, size_t size, GraphMemcpyKind kind, const GraphStream& stream) {
    auto ret = aclrtMemcpyAsync(dst, size, src, size, toAclKind(kind), stream.stream());
    if (ret != ACL_SUCCESS) {
        RTP_LLM_LOG_ERROR("aclrtMemcpyAsync failed: ret=%d, size=%zu", static_cast<int>(ret), size);
        throw std::runtime_error("aclrtMemcpyAsync failed");
    }
}

void graphDeviceSynchronize() {
    auto ret = aclrtSynchronizeDevice();
    if (ret != ACL_SUCCESS) {
        RTP_LLM_LOG_ERROR("aclrtSynchronizeDevice failed: ret=%d", static_cast<int>(ret));
        throw std::runtime_error("aclrtSynchronizeDevice failed");
    }
}

void graphStreamSynchronize(const GraphStream& stream) {
    auto ret = aclrtSynchronizeStream(stream.stream());
    if (ret != ACL_SUCCESS) {
        RTP_LLM_LOG_ERROR("aclrtSynchronizeStream failed: ret=%d", static_cast<int>(ret));
        throw std::runtime_error("aclrtSynchronizeStream failed");
    }
}

void graphMemGetInfo(size_t* free_bytes, size_t* total_bytes) {
    size_t free_hbm = 0, total_hbm = 0;
    auto   ret = aclrtGetMemInfo(ACL_HBM_MEM, &free_hbm, &total_hbm);
    if (ret != ACL_SUCCESS) {
        if (free_bytes) *free_bytes = 0;
        if (total_bytes) *total_bytes = 0;
        return;
    }
    if (free_bytes) *free_bytes = free_hbm;
    if (total_bytes) *total_bytes = total_hbm;
}

void graphCaptureBegin(c10_npu::NPUGraph& graph, GraphPoolHandle /*pool*/, GraphCaptureMode mode) {
    // NPUGraph::capture_begin auto-creates a mempool when pool={0,0}; sharing
    // across graphs via explicit mempool id is currently unused.
    graph.capture_begin({0, 0}, toAclMode(mode));
}

void graphCaptureEnd(c10_npu::NPUGraph& graph) {
    graph.capture_end();
}

void graphReplay(c10_npu::NPUGraph& graph) {
    graph.replay();
}
#else
GraphStream graphGetStreamFromPool(bool) { return GraphStream{}; }
GraphStream graphGetCurrentStream() { return GraphStream{}; }
void        graphSetCurrentStream(const GraphStream&) {}
torch::Event makeGraphEvent() { return torch::Event(c10::DeviceType::CPU); }
void graphMemcpyAsync(void*, const void*, size_t, GraphMemcpyKind, const GraphStream&) {}
void graphDeviceSynchronize() {}
void graphStreamSynchronize(const GraphStream&) {}
void graphMemGetInfo(size_t* free_bytes, size_t* total_bytes) {
    if (free_bytes) *free_bytes = 0;
    if (total_bytes) *total_bytes = 0;
}
#endif  // USING_ASCEND

void setGraphCaptureEnabled(bool enabled) {
#if USING_ASCEND
    g_ascend_graph_capture_enabled = enabled;
#else
    (void)enabled;
#endif
}

bool isGraphCaptureEnabled() {
#if USING_ASCEND
    return g_ascend_graph_capture_enabled;
#else
    return false;
#endif
}

}  // namespace ascend_graph
}  // namespace rtp_llm
