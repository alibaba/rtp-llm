#pragma once

#include <cstddef>
#include <cstdint>

#ifdef __CUDACC__
#define K3_TRACE_HD __host__ __device__
#else
#define K3_TRACE_HD
#endif

namespace deep_gemm {

// ABI 2. Rows are (source rank, source token, top-k slot). Only the owner
// of a routed expert writes that row. Reset expert_ids to -1 before each call.
struct K3MegaMoETrace {
    float*    fc1;
    uint16_t* rounded;
    float*    activation;
    uint8_t*  fp8;
    uint8_t*  scales;
    int32_t*  expert_ids;
    int32_t*  overflow;

    K3_TRACE_HD static constexpr size_t bytes(size_t slots, size_t width) {
        return slots * (17 * width + width / 32 + sizeof(int32_t)) + sizeof(int32_t);
    }

    K3_TRACE_HD K3MegaMoETrace(void* raw, size_t slots, size_t width) {
        auto p = reinterpret_cast<uint8_t*>(raw);
        fc1    = reinterpret_cast<float*>(p);
        p += slots * width * 2 * sizeof(float);
        rounded = reinterpret_cast<uint16_t*>(p);
        p += slots * width * 2 * sizeof(uint16_t);
        activation = reinterpret_cast<float*>(p);
        p += slots * width * sizeof(float);
        fp8 = p;
        p += slots * width;
        scales = p;
        p += slots * width / 32;
        expert_ids = reinterpret_cast<int32_t*>(p);
        overflow   = expert_ids + slots;
    }
};

// Independent ABI 1 for fused BF16 shared experts. Rows are rank-local tokens.
// The full activation consumed by FC2 is retained in shared_l2_acts by the
// original kernel; this side buffer preserves FC1 before its registers expire.
struct K3SharedMoETrace {
    float*    fc1;
    uint16_t* rounded;
    int32_t*  valid;
    int32_t*  overflow;

    K3_TRACE_HD static constexpr size_t bytes(size_t tokens, size_t width) {
        return tokens * (12 * width + sizeof(int32_t)) + sizeof(int32_t);
    }

    K3_TRACE_HD K3SharedMoETrace(void* raw, size_t tokens, size_t width) {
        auto p = reinterpret_cast<uint8_t*>(raw);
        fc1    = reinterpret_cast<float*>(p);
        p += tokens * 2 * width * sizeof(float);
        rounded = reinterpret_cast<uint16_t*>(p);
        p += tokens * 2 * width * sizeof(uint16_t);
        valid    = reinterpret_cast<int32_t*>(p);
        overflow = valid + tokens;
    }
};

}  // namespace deep_gemm

#undef K3_TRACE_HD
