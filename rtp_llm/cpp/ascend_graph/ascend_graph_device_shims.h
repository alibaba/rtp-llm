// Ascend NPU graph device shim.
//
// Isolates the Ascend-specific stream/event/memcpy/graph capture APIs used by
// AscendGraphRunner, independent of the legacy `rtp_llm::cuda_graph` shim so
// changes here never affect CUDA/ROCm paths.

#pragma once

#include <cstddef>
#include <cstdint>
#include <pybind11/pybind11.h>
#include <torch/torch.h>

#if USING_ASCEND
#include <acl/acl.h>
// CANN's acl_sk.h (pulled in via NPUGraph.h) contains a nameless
// "typedef struct {...};" which is a benign -Wpedantic warning under gcc;
// -Werror would turn it fatal (also downgraded in the BUILD copts).
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wpedantic"
#include <torch_npu/csrc/core/npu/NPUGraph.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/core/npu/NPUFunctions.h>
#pragma GCC diagnostic pop
#endif

namespace py = pybind11;

namespace rtp_llm {
namespace ascend_graph {

// Stream abstraction. On Ascend we keep the underlying aclrtStream via NPUStream.
#if USING_ASCEND
using GraphStream = c10_npu::NPUStream;
#else
struct GraphStream {};  // dummy for non-Ascend platforms (never instantiated)
#endif

// Mempool handle: Ascend's NPUGraph uses {0,0} to auto-create a mempool and
// does not support CUDA-style explicit mempool sharing across graphs.
struct GraphPoolHandle {};

// Capture mode used by aclrt/aclmdlRICaptureMode.
enum class GraphCaptureMode {
    Global,
    ThreadLocal,
    Relaxed,
};

// Stream helpers.
GraphStream graphGetStreamFromPool(bool is_high_priority);
GraphStream graphGetCurrentStream();
void        graphSetCurrentStream(const GraphStream& stream);

// Event helper (returns torch::Event on PrivateUse1 device).
torch::Event makeGraphEvent();

// Async memcpy wrapper over aclrtMemcpyAsync.
enum class GraphMemcpyKind {
    D2D,
    D2H,
    H2D,
};
void graphMemcpyAsync(void* dst, const void* src, size_t size, GraphMemcpyKind kind, const GraphStream& stream);

// Synchronize current NPU device.
void graphDeviceSynchronize();

// Synchronize a specific stream.
void graphStreamSynchronize(const GraphStream& stream);

// Memory info in bytes (HBM).
void graphMemGetInfo(size_t* free_bytes, size_t* total_bytes);

// NPUGraph capture begin/end wrappers. No-op when USING_ASCEND is undefined.
#if USING_ASCEND
void graphCaptureBegin(c10_npu::NPUGraph& graph, GraphPoolHandle pool, GraphCaptureMode mode);
void graphCaptureEnd(c10_npu::NPUGraph& graph);
void graphReplay(c10_npu::NPUGraph& graph);
#endif

// In-graph capture flag (used by Python kernels to detect capture mode).
void setGraphCaptureEnabled(bool enabled);
bool isGraphCaptureEnabled();

}  // namespace ascend_graph
}  // namespace rtp_llm
