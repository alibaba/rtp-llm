#pragma once

#include <memory>
#include <map>
#include "rtp_llm/cpp/cache_new/KVCacheAllocator.h"

namespace rtp_llm {

class HybridPoolKVCacheAllocator : public KVCacheAllocator {

};

using HybridPoolKVCacheAllocatorPtr = std::shared_ptr<HybridPoolKVCacheAllocator>;

}  // namespace rtp_llm
