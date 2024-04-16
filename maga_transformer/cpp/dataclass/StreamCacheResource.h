#pragma once

#include "maga_transformer/cpp/cache/CacheManager.h"
#include <memory>

namespace rtp_llm {

class GenerateStream;

class StreamCacheResource {
public:
    StreamCacheResource(GenerateStream* stream): stream_(stream) {}
    ~StreamCacheResource() {
        releaseResource();
    }
    bool initKVBlock();
    bool incrKVBlock();
    // TODO(xinfei.sxf) flash attention must suppor prefix prompt
    int    tryReleaseKVBlock(size_t nums);
    void   releaseResource();
    int    initalKVCacheCount() const;
    int    nextNeedBlockNums() const;
    size_t maxBlockSize() const;

    const BatchKVCacheBlockAddr& kvCache() const;
    void                         setKVCache(const BatchKVCacheBlockAddr& kv_cache_block_addr);
    void                         setCacheManager(std::shared_ptr<CacheManager> cache_manager);
    void                         setReuseCache(bool reuse_cache);

private:
    BatchKVCacheBlockAddr         kv_cache_block_addr_;
    GenerateStream*               stream_;
    std::shared_ptr<CacheManager> cache_manager_;
    // TODO(xinfei.sxf) set gen_num_per_circle_
    int  gen_num_per_circle_ = 1;
    int  seq_size_per_block_;
    bool reuse_cache_ = false;
};

}  // namespace rtp_llm
