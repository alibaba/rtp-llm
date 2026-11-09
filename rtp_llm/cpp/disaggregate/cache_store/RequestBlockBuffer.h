#pragma once

#include <torch/all.h>
#include "rtp_llm/cpp/cache/BlockInfo.h"

#include <shared_mutex>
#include <unordered_map>
#include <memory>
#include <vector>
#include <functional>

namespace rtp_llm {

// 关联一块内存/显存
class BlockBuffer {
public:
    BlockBuffer(const std::string&           key_,
                const std::shared_ptr<void>& addr_,
                uint32_t                     len_,
                bool                         gpu_mem_,
                bool                         adopted_,
                IndexerCacheLayout           layout = {}):
        key(key_), addr(addr_), len(len_), gpu_mem(gpu_mem_), adopted(adopted_), indexer_layout(layout) {}
    BlockBuffer(const BlockBuffer& rhs):
        key(rhs.key),
        addr(rhs.addr),
        len(rhs.len),
        gpu_mem(rhs.gpu_mem),
        adopted(rhs.adopted),
        indexer_layout(rhs.indexer_layout) {}

    BlockInfo blockInfo() const {
        BlockInfo result;
        result.addr           = addr.get();
        result.size_bytes     = len;
        result.is_cuda        = gpu_mem;
        result.indexer_layout = indexer_layout;
        return result;
    }

    std::string           key;
    std::shared_ptr<void> addr;
    uint32_t              len{0};
    bool                  gpu_mem{true};
    bool                  adopted{true};
    IndexerCacheLayout    indexer_layout;
};

class MemoryUtil;
std::shared_ptr<BlockBuffer> legacyIndexerBlock(const std::shared_ptr<BlockBuffer>& block,
                                                const std::shared_ptr<MemoryUtil>&  memory_util = nullptr);

//  request 关联的 block buffer
class RequestBlockBuffer {
public:
    RequestBlockBuffer(const std::string& requestid, const std::string& request_key = "");
    RequestBlockBuffer(const std::string& requestid, std::shared_ptr<torch::Event> event);

    ~RequestBlockBuffer();

public:
    const std::string&  getRequestId() const;
    const std::string&  getRequestKey() const;
    const torch::Event* getEvent() const;

    std::unordered_map<std::string, std::shared_ptr<BlockBuffer>> getBlocks() const;
    std::shared_ptr<BlockBuffer>                                  getBlock(const std::string& id) const;
    size_t                                                        getBlocksCount() const;
    size_t                                                        getBlocksSize() const;

    void addBlock(const std::shared_ptr<BlockBuffer>& block);
    void addBlock(const std::string&           key,
                  const std::shared_ptr<void>& addr,
                  uint32_t                     len,
                  bool                         gpu_mem,
                  bool                         adopted,
                  IndexerCacheLayout           layout = {});
    void addBlocks(const std::vector<std::shared_ptr<BlockBuffer>>& blocks);

    bool isValid() const;

    // change with true callback, dtor with false callback
    typedef std::function<void(bool ok, const std::vector<std::shared_ptr<BlockBuffer>>&)> WatchFunc;
    bool setWatchFunc(WatchFunc&& watch_func);
    void notifyRequestDone();

    std::string debugInfo() const;

private:
    void triggerWatchFunc(bool ok, const std::vector<std::shared_ptr<BlockBuffer>>&);

private:
    std::string requestid_;
    std::string request_key_;

    std::shared_ptr<torch::Event> event_;

    mutable std::shared_mutex                                     blocks_mutex_;
    std::unordered_map<std::string, std::shared_ptr<BlockBuffer>> blocks_;
    size_t                                                        blocks_size_ = 0;

    mutable std::shared_mutex watch_func_mutex_;
    std::vector<WatchFunc>    watch_funcs_;
};

}  // namespace rtp_llm
