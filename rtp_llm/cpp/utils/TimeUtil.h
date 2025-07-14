#pragma once

#include "autil/TimeUtility.h"

namespace rtp_llm {

inline int64_t currentTimeUs() {
    return autil::TimeUtility::currentTimeInMicroSeconds();
}

}  // namespace rtp_llm
