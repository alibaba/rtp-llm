#pragma once

#include <cstddef>

namespace rtp_llm {

// Compatibility options apply only to the legacy copy strategies.
struct DeviceHostCopyOptions {
    size_t staged_sm_min_tile_count{16};
    size_t staged_sm_min_bytes{64 * 1024};
    bool   staged_sm_copy_enabled{true};
    bool   cuda_batch_copy_enabled{true};
};

}  // namespace rtp_llm
