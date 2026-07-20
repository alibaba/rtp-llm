#pragma once

#include "rtp_llm/cpp/cache/KVCacheGroup.h"

namespace rtp_llm {

// Compatibility name for the BlockTreeCache integration. The target declarative
// KVCacheGroup remains the request-allocation authority.
using DeviceKVCacheGroup    = KVCacheGroup;
using DeviceKVCacheGroupPtr = KVCacheGroupPtr;

}  // namespace rtp_llm
