#pragma once

#include "rtp_llm/models_py/bindings/LinearCheckpoint.h"
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace rtp_llm {

struct LinearCheckpointDeviceTile {
    float*                state;
    int8_t*               packed;
    int                   heads;
    int                   value_dim;
    int                   key_dim;
    LinearCheckpointDType dtype;
};

void invokeLinearCheckpointCopy(
    const LinearCheckpointDeviceTile* tiles, int count, int max_channels, bool unpack, cudaStream_t stream);

}  // namespace rtp_llm
