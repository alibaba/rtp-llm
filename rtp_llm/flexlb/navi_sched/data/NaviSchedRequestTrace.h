#pragma once

#include <cerrno>
#include <cstdint>
#include <cstdlib>

namespace navi_sched {

inline uint64_t requestTraceBatchModulus()
{
    static const uint64_t modulus = []() -> uint64_t {
        const auto value = std::getenv("NAVI_SCHED_REQUEST_TRACE_BATCH_MODULUS");
        if (!value || !*value || *value == '-') {
            return 0;
        }
        const auto savedErrno = errno;
        errno = 0;
        char *end = nullptr;
        const auto parsed = std::strtoull(value, &end, 10);
        const bool valid = errno == 0 && end != value && *end == '\0';
        errno = savedErrno;
        return valid ? parsed : 0;
    }();
    return modulus;
}

inline bool traceRequestBatch(int64_t batchId)
{
    const auto modulus = requestTraceBatchModulus();
    return modulus > 0 && batchId > 0 && static_cast<uint64_t>(batchId) % modulus == 0;
}

}
