#pragma once

#include "maga_transformer/cpp/system_prompt/SystemPrompt.h"
#include "maga_transformer/cpp/cache/CacheManager.h"
#include <memory>

namespace rtp_llm {

class GenerateStream;

struct ResourceContext {
    std::shared_ptr<CacheManager>   cache_manager;
    std::shared_ptr<SystemPrompt>   system_prompt;
    bool                            reuse_cache{false};
};

class BatchKVCacheBlockAddr {
public:
    BatchKVCacheBlockAddr() {}
    void clear() {
        k_ptr.clear();
        v_ptr.clear();
        k_scale_ptr.clear();
        v_scale_ptr.clear();
    }
    void pushBack(const KVCacheBlockAddr& addr) {
        k_ptr.push_back(addr.k_ptr);
        v_ptr.push_back(addr.v_ptr);
        if (!addr.k_scale_ptr.empty()) {
            k_scale_ptr.push_back(addr.k_scale_ptr);
            v_scale_ptr.push_back(addr.v_scale_ptr);
        }
    }
    void append(size_t index, const KVCacheBlockAddr& addr) {
        assert(k_ptr.size() > index);
        auto append_func = [](auto& dst_vec, auto& src_vec) {
            dst_vec.insert(dst_vec.end(), src_vec.begin(), src_vec.end());
        };
        for (auto layer_id = 0; layer_id < k_ptr[index].size(); layer_id++) {
            append_func(k_ptr[index][layer_id], addr.k_ptr[layer_id]);
            append_func(v_ptr[index][layer_id], addr.v_ptr[layer_id]);
            if (!addr.k_scale_ptr.empty()) {
                append_func(k_scale_ptr[index][layer_id], addr.k_scale_ptr[layer_id]);
                append_func(v_scale_ptr[index][layer_id], addr.v_scale_ptr[layer_id]);
            }
        }
    }

    std::string debugString() const {
        std::stringstream debug_string, k_ptr_string, v_ptr_string, k_scale_ptr_string, v_scale_ptr_string;
        for (int i = 0; i < k_ptr.size(); i++) {
            k_ptr_string << "batch: " << i << " ";
            v_ptr_string << "batch: " << i << " ";
            for (int j = 0; j < k_ptr[0].size(); ++j) {
                k_ptr_string << "layer:" << j << ";";
                v_ptr_string << "layer:" << j << ";";
                for (auto &v: k_ptr[i][j]) {
                    k_ptr_string << (int64_t)v << ", ";
                }
                for (auto &v: v_ptr[i][j]) {
                    v_ptr_string << (int64_t)v << ", ";
                }
            }
        }

        if (!k_scale_ptr.empty()) {
            for (int i = 0; i < k_scale_ptr.size(); i++) {
                k_scale_ptr_string << "batch: " << i << " ";
                v_scale_ptr_string << "batch: " << i << " ";
                for (int j = 0; j < k_scale_ptr[0].size(); ++j) {
                    k_scale_ptr_string << "layer:" << j << ";";
                    v_scale_ptr_string << "layer:" << j << ";";
                    for (auto &v: k_scale_ptr[i][j]) {
                        k_scale_ptr_string << (int64_t)v << ", ";
                    }
                    for (auto &v: v_scale_ptr[i][j]) {
                        v_scale_ptr_string << (int64_t)v << ", ";
                    }
                }
            }
        }

        debug_string << "BatchKVCacheBlockAddr {"
                     << "k_ptr: " << k_ptr_string.str()
                     << "v_ptr: " << v_ptr_string.str()
                     << "k_scale_ptr: " << k_scale_ptr_string.str()
                     << "v_scale_ptr: " << v_scale_ptr_string.str()
                     << "}";
        return debug_string.str();
    }
    
public:
    // [batch_size, layer_num, max_block_per_seq]
    std::vector<std::vector<std::vector<void*>>> k_ptr;
    std::vector<std::vector<std::vector<void*>>> v_ptr;

    std::vector<std::vector<std::vector<void*>>> k_scale_ptr;
    std::vector<std::vector<std::vector<void*>>> v_scale_ptr;
};

class StreamCacheResource {
public:
    StreamCacheResource(GenerateStream* stream, const ResourceContext& resource_context): stream_(stream), resource_context_(resource_context) {}
    ~StreamCacheResource() {
        releaseResource();
    }
    bool initKVBlock();
    bool incrKVBlock();
    // TODO(xinfei.sxf) flash attention must suppor prefix prompt
    int     tryReleaseKVBlock(size_t nums);
    void    setNeedReleaseResource(bool need_release_resource);
    void    releaseResource();
    int     needKVCacheBlockNums() const;
    int     maxBlockSize() const;

    const BatchKVCacheBlockAddr& kvCache() const;
    void                         setKVCache(const BatchKVCacheBlockAddr& kv_cache_block_addr);

    const ResourceContext& resourceContext() const {
        return resource_context_;
    }

    int seqSizePerBlock() const {
        return resource_context_.cache_manager->cacheConfig().seq_size_per_block;
    }

private:
    BatchKVCacheBlockAddr           kv_cache_block_addr_;
    GenerateStream*                 stream_;
    ResourceContext                 resource_context_;
    // TODO(xinfei.sxf) set gen_num_per_circle_
    int                             gen_num_per_circle_    = 1;
    int                             seq_size_per_block_    = 0;
    bool                            need_release_resource_ = true;
};

}  // namespace rtp_llm
