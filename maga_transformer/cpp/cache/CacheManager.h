#pragma once

#include <algorithm>
#include <cassert>
#include <iostream>
#include <list>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "maga_transformer/cpp/cache/BlockCache.h"
#include "maga_transformer/cpp/cache/BlockRefCounter.h"
#include "maga_transformer/cpp/cache/CacheConfig.h"
#include "src/fastertransformer/core/Buffer.h"
#include "src/fastertransformer/core/Types.h"
#include "src/fastertransformer/devices/DeviceBase.h"
#include "src/fastertransformer/devices/DeviceFactory.h"

namespace ft = fastertransformer;
namespace rtp_llm {

struct SeqPosition {
    int index;
    int offset;
};

class CacheManager;

class KVCacheBlockAddr {
public:
    void clear() {
        k_ptr.clear();
        v_ptr.clear();
        k_scale_ptr.clear();
        v_scale_ptr.clear();
    }

    KVCacheBlockAddr clone(std::shared_ptr<CacheManager>& cache_manager);

public:
    // [layer_num, max_block_per_seq]
    std::vector<std::vector<void*>> k_ptr;
    std::vector<std::vector<void*>> v_ptr;

    std::vector<std::vector<void*>> k_scale_ptr;
    std::vector<std::vector<void*>> v_scale_ptr;
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
        std::stringstream debug_string, k_ptr_string, v_ptr_string;
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
        debug_string << "BatchKVCacheBlockAddr {"
                     << "k_ptr: " << k_ptr_string.str()
                     << "v_ptr: " << v_ptr_string.str()
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

struct KVCacheBuffer {
    ft::BufferPtr k_blocks;
    ft::BufferPtr v_blocks;
    ft::BufferPtr k_scale;
    ft::BufferPtr v_scale;
};

class CacheManager {
public:
    CacheManager(const CacheConfig& config, ft::DeviceBase* device);
    ~CacheManager();

    const CacheConfig&     cacheConfig() const;
    const BlockRefCounter& blockRefCounter() const;
    const BlockCache&      blockCache() const;
    size_t                 freeBlockNums() const;
    size_t                 cacheItemNum() const;
    const KVCacheBuffer&   kvCacheBuffer() const;

    std::tuple<bool, KVCacheBlockAddr>      malloc(int nums = 1);
    std::tuple<bool, KVCacheBlockAddr, int> mallocWithCache(int want_block_nums, const std::vector<int>& token_ids);
    void                                    reserveBlocks(int nums);
    void                                    incrBlockRefCounter(const std::vector<void*> pointers);

    void free(const std::vector<void*> pointer);
    void free(const std::vector<KVCacheBlockAddr>& resource);
    void freeWithCache(const std::vector<void *> pointer, const std::vector<int>& token_ids);
    void insertResidentCache(const std::vector<void *> pointer, const std::vector<int>& token_ids);
    void insertResidentCache(const std::vector<int>& block_indices, const std::vector<int>& token_ids);

    void setKVBlockValue(int index, ft::BufferPtr& k_value, ft::BufferPtr& v_value);
    void blockCopy(int src_block_index, int dest_block_index);
    void copyKvCacheFromSeqIdxs(const std::vector<int>& block_indice_list, const std::vector<int>& src_index, const std::vector<int>& tgt_index);
    SeqPosition getSeqPosition(const std::vector<int>& block_indice_list, int idx);
    void        copyKvCacheFromSeqPosition(const SeqPosition& src_seq_position, const SeqPosition& dst_seq_position);

private:
    void                                    initFreeBlock(const CacheConfig& config);
    void                                    initKvCache(const CacheConfig& config);
    std::tuple<bool, std::vector<int>>      mallocIndex(int nums = 1);
    std::tuple<bool, std::vector<int>>      mallocImpl(int nums);
    std::tuple<bool, std::vector<int>, int> mallocWithCacheImpl(int want_block_nums, const std::vector<int>& token_ids);
    void                                    maybeFreeBlockFromCache(int nums);
    void                                    free(const std::vector<std::vector<int>>& indices);
    void                                    free(const std::vector<int>& indice);
    void freeWithCache(const std::vector<std::vector<int>>& block_indices, const std::vector<int>& token_ids);
    void insertIntoCache(const std::vector<std::vector<int>>& block_indices,
                         const std::vector<int>&              token_ids,
                         bool                                 is_resident);
    KVCacheBlockAddr convertIndexToAddr(const std::vector<int>& block_indices) const;
    std::vector<int> convertAddrToIndex(const std::vector<void*>& pointers) const;

private:
    CacheConfig     config_;
    int             seq_size_per_block_;
    std::set<int>   free_blocks_index_;
    BlockRefCounter block_ref_counter_;
    BlockCache      block_cache_;
    int             block_nums_;
    KVCacheBuffer   kv_cache_;
    ft::DeviceBase* device_;
};

typedef std::shared_ptr<CacheManager> CacheManagerPtr;

}  // namespace rtp_llm
