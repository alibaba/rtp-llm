#pragma once
#include "rtp_llm/cpp/core/FusedCopyTypes.h"
#include "rtp_llm/cpp/runtime/DeviceTypes.h"

namespace rtp_llm {

void invokeFusedCopy(const FusedD2DCopyParams& params, DeviceStream stream);

void invokeFusedStridedCopy(const FusedStridedCopyParams& params, DeviceStream stream);

}  // namespace rtp_llm
