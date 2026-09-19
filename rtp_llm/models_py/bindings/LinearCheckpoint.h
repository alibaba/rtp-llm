#pragma once

#include <cstdint>

namespace rtp_llm {

enum class LinearCheckpointDType : uint8_t {
    INT8,
    BF16
};

// Conv history is transported unchanged as ordinary tiles in the same batch.
// INT8 appends one FP32 scale per [H,K] row after the packed values.
struct LinearCheckpointCopyTile {
    float*                state     = nullptr;
    void*                 host      = nullptr;
    int                   heads     = 0;
    int                   value_dim = 0;
    int                   key_dim   = 0;
    LinearCheckpointDType dtype     = LinearCheckpointDType::INT8;
};

}  // namespace rtp_llm
