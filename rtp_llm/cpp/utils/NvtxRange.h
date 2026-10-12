#pragma once

#include <cstdlib>
#include <cstring>

#if USING_CUDA
#include <cuda.h>
#if CUDA_VERSION >= 12090
#include <nvtx3/nvToolsExt.h>
#else
#include <nvToolsExt.h>
#endif
#endif

namespace rtp_llm {

// Diagnostic annotations only; no tensor, stream or synchronization changes.
class ScopedModelNvtxRange {
public:
    explicit ScopedModelNvtxRange(const char* name) {
#if USING_CUDA
        static const bool enabled = [] {
            const char* value = std::getenv("GEMMA4_PROFILE_RANGES");
            return value && std::strcmp(value, "1") == 0;
        }();
        active_ = enabled;
        if (active_) {
            nvtxRangePushA(name);
        }
#else
        (void)name;
#endif
    }

    ~ScopedModelNvtxRange() {
#if USING_CUDA
        if (active_) {
            nvtxRangePop();
        }
#endif
    }

    ScopedModelNvtxRange(const ScopedModelNvtxRange&)            = delete;
    ScopedModelNvtxRange& operator=(const ScopedModelNvtxRange&) = delete;

private:
#if USING_CUDA
    bool active_ = false;
#endif
};

}  // namespace rtp_llm
