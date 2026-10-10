#pragma once

#include <cstddef>
#include <cstdint>

namespace rtp_llm {

enum class LinearCheckpointDType : uint8_t {
    INT8,
    BF16,
    INT16
};

inline size_t linearCheckpointPackedBytes(size_t heads, size_t keys, size_t values, LinearCheckpointDType dtype) {
    const size_t channels = heads * keys;
    if (dtype == LinearCheckpointDType::BF16) {
        return channels * values * sizeof(uint16_t);
    }
    const size_t element_bytes = dtype == LinearCheckpointDType::INT16 ? sizeof(int16_t) : sizeof(int8_t);
    return channels * (values * element_bytes + sizeof(float));
}

// Conv history is transported unchanged as ordinary tiles in the same batch.
// INT8 and INT16 append one FP32 scale per [H,K] row after the packed values.
struct LinearCheckpointCopyTile {
    float*                state     = nullptr;
    void*                 host      = nullptr;
    int                   heads     = 0;
    int                   value_dim = 0;
    int                   key_dim   = 0;
    LinearCheckpointDType dtype     = LinearCheckpointDType::INT8;
};

}  // namespace rtp_llm
