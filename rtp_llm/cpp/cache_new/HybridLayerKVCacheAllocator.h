#pragma once

#include <memory>
#include <map>
#include "rtp_llm/cpp/cache_new/KVCacheAllocator.h"

namespace rtp_llm {

class HybridLayerKVCacheAllocator : public KVCacheAllocator {

};

using HybridLayerKVCacheAllocatorPtr = std::shared_ptr<HybridLayerKVCacheAllocatorPtr>;

}  // namespace rtp_llm
