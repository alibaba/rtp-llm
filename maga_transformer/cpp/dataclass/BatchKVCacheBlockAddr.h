#pragma once

#include "maga_transformer/cpp/cache/KVCacheBlockAddr.h"
#include "src/fastertransformer/utils/assert_utils.h"
#include <memory>

namespace rtp_llm {

class BatchKVCacheBlockAddr {
public:
    BatchKVCacheBlockAddr() {}
    int                         batchSize() const;
    int                         blockSize(int batch_id) const;
    void                        resize(size_t batch_size);
    void                        resize(size_t batch_id, int reserver_blocks);
    void                        pushBack(const KVCacheBlockAddr& addr);
    void                        append(size_t batch_id, const KVCacheBlockAddr& addr);
    void                        appendClone(const KVCacheBlockAddr& addr, std::shared_ptr<CacheManager>& cache_manager);
    void                        append(const std::vector<KVCacheBlockAddr>& resource);
    int                         maxBlockSize() const;
    const std::vector<int>&     blocks(int batch_id) const;
    void                        clear();

    std::string                 debugString() const;

public:
    // [batch_size, max_block_per_seq]
    std::vector<std::vector<int>> batch_offset;
};

}  // namespace rtp_llm
