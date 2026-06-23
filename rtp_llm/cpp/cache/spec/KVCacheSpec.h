#pragma once

// This header includes all KVCacheSpec related classes
// Split into separate files for better modularity

#include "rtp_llm/cpp/cache/spec/KVCacheSpecBase.h"
#include "rtp_llm/cpp/cache/spec/MHAKVCacheSpec.h"
#include "rtp_llm/cpp/cache/spec/MLAKVCacheSpec.h"
#include "rtp_llm/cpp/cache/spec/LinearKVCacheSpec.h"
#include "rtp_llm/cpp/cache/spec/OpaqueKVCacheSpec.h"

namespace rtp_llm {
// All KVCacheSpec classes are now available through individual headers
// This file serves as a convenience header to include all of them
}  // namespace rtp_llm
