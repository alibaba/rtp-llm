#include "rtp_llm/models_py/bindings/cuda/test/StagedMemcpyProfiler.h"

#include <chrono>

namespace {
thread_local size_t                                           active_payload_bytes = 0;
thread_local rtp_llm::crc_copy_benchmark::StagedMemcpyProfile profile;
}  // namespace

extern "C" void* __real_memcpy(void* destination, const void* source, size_t bytes);

// grpc also defines __wrap_memcpy. The diagnostic link explicitly aliases that
// entry to this unique symbol; __real_memcpy keeps the ordinary libc version.
extern "C" void* stagedProfileMemcpy(void* destination, const void* source, size_t bytes) {
    if (!active_payload_bytes || bytes != active_payload_bytes)
        return __real_memcpy(destination, source, bytes);
    const auto start   = std::chrono::steady_clock::now();
    void*      result  = __real_memcpy(destination, source, bytes);
    const auto elapsed = std::chrono::steady_clock::now() - start;
    ++profile.calls;
    profile.bytes += bytes;
    profile.nanoseconds += std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count();
    return result;
}

namespace rtp_llm::crc_copy_benchmark {

void beginStagedMemcpyProfile(size_t payload_bytes) {
    profile              = {};
    active_payload_bytes = payload_bytes;
}

StagedMemcpyProfile endStagedMemcpyProfile() {
    active_payload_bytes = 0;
    return profile;
}

}  // namespace rtp_llm::crc_copy_benchmark
