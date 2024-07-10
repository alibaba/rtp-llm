#pragma once

#include "maga_transformer/cpp/cache/KVCacheBlockAddr.h"
#include "src/fastertransformer/utils/assert_utils.h"
#include <memory>

namespace rtp_llm {

class BatchKVCacheBlockAddr {
public:
    BatchKVCacheBlockAddr() {}
    void        clear();
    void        pushBack(const KVCacheBlockAddr& addr);
    void        resize(size_t batch_id, int reserver_blocks);
    void        append(size_t batch_id, const KVCacheBlockAddr& addr);
    std::string debugString() const;

public:
    // [batch_size, max_block_per_seq]
    std::vector<std::vector<int>> batch_offset;
};

}  // namespace rtp_llm
