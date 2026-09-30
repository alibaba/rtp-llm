#pragma once

#include <cstddef>
#include <cuda_runtime.h>

namespace rtp_llm {

struct CopyTile {
    void*  device;
    size_t staging_offset;
    size_t bytes;
};

constexpr int kCopyTileThreads = 256;

// One CUDA block copies each tile between its device pointer and staging + staging_offset.
// Descriptors reside on the device. The caller owns stream ordering, allocation, and bounds validation;
// nonempty tiles must have valid, nonoverlapping source/destination spans. No allocation or synchronization occurs
// here. scatter=false gathers into staging; scatter=true scatters out of staging.
cudaError_t launchCopyTiles(const CopyTile* tiles,
                            size_t          count,
                            void*           staging,
                            bool            scatter,
                            cudaStream_t    stream,
                            int             threads = kCopyTileThreads);

}  // namespace rtp_llm
