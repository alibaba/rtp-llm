#pragma once

#include "rtp_llm/models_py/bindings/core/DeviceData.h"

namespace rtp_llm {
namespace test {

// The MemoryStatus that the stubbed getGpuExecStatus() reports. Writable so a test can drive the
// sampling overload of MemoryEvaluationHelper::getKVCacheMemorySize -- the overload production
// actually calls -- and compare it against the injected overload fed the same status. Defaults to a
// zeroed status, which is what every case that only uses the injected overload sees.
MemoryStatus& stubbedDeviceMemoryStatus();

}  // namespace test
}  // namespace rtp_llm
