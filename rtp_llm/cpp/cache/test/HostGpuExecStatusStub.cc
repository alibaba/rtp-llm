#include "rtp_llm/cpp/cache/test/HostGpuExecStatusStub.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"

// Host-side provider for the one ExecOps symbol cache_core references, so tests that exercise
// MemoryEvaluationHelper without the CUDA op libraries can link.
//
// The status is test-writable rather than hardcoded zeros: that is what lets a test assert the
// *sampling* overload of getKVCacheMemorySize (the one production actually calls) agrees with the
// injected overload. With a fixed all-zero status the two would agree trivially -- both would abort
// on a zero-byte device -- and a field the sampling overload forgot to forward would go unnoticed.
namespace rtp_llm {
namespace test {

MemoryStatus& stubbedDeviceMemoryStatus() {
    static MemoryStatus status{};
    return status;
}

}  // namespace test

ExecStatus getGpuExecStatus() {
    ExecStatus status;
    status.device_memory_status = test::stubbedDeviceMemoryStatus();
    return status;
}

}  // namespace rtp_llm
