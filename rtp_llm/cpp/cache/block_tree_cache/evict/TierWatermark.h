#pragma once

#include <cmath>
#include <cstddef>

namespace rtp_llm {

struct TierWatermark {
    double low_ratio{0.0};
    double high_ratio{0.0};

    bool reachedHigh(std::size_t capacity, std::size_t used) const {
        return capacity != 0 && used >= static_cast<std::size_t>(std::ceil(capacity * high_ratio));
    }
};

inline constexpr double kDefaultDeviceLowWatermarkRatio  = 0.82;
inline constexpr double kDefaultDeviceHighWatermarkRatio = 0.90;
inline constexpr double kDefaultMemoryLowWatermarkRatio  = 0.90;
inline constexpr double kDefaultMemoryHighWatermarkRatio = 0.94;
inline constexpr double kDefaultDiskLowWatermarkRatio    = 0.92;
inline constexpr double kDefaultDiskHighWatermarkRatio   = 0.97;

}  // namespace rtp_llm
