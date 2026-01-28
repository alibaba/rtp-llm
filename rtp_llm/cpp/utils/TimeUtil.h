#pragma once

#include "autil/TimeUtility.h"

namespace rtp_llm {

inline int64_t currentTimeUs() {
    return autil::TimeUtility::currentTimeInMicroSeconds();
}

inline int64_t currentTimeMs() {
    return autil::TimeUtility::currentTimeInMilliSeconds();
}

}  // namespace rtp_llm
