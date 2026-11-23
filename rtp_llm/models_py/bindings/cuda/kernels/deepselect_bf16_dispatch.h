#pragma once

#include <cstdint>

namespace torch_ext {

struct DraftBf16Dispatch {
    bool    supported;
    int64_t min_batch;

    constexpr bool supportsBatch(int64_t batch) const {
        return supported && batch >= min_batch;
    }
};

// The SM90 image is currently emitted by the CUDA13 x86 build only.
// Raw kernel calls support smaller batches on Blackwell; this threshold
// controls the sampler's BF16 GEMM/selector dispatch.
constexpr DraftBf16Dispatch draftBf16Dispatch(int major, int minor, bool has_sm90_image) {
    if (major == 9 && minor == 0) {
        return {has_sm90_image, 1};
    }
    return {major == 10 && (minor == 0 || minor == 3), 16};
}

}  // namespace torch_ext
