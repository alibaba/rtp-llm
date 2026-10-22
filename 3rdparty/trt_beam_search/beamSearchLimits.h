#pragma once

namespace tensorrt_llm {
namespace kernels {

// Keep request admission and kernel dispatch consistent with the compiled cases.
#ifdef FAST_BUILD
inline constexpr int kMaxSupportedBeamWidth = 16;
#else
inline constexpr int kMaxSupportedBeamWidth = 4096;
#endif

inline constexpr bool isSupportedBeamWidth(int width) {
    return width > 0 && width <= kMaxSupportedBeamWidth;
}

}  // namespace kernels
}  // namespace tensorrt_llm
