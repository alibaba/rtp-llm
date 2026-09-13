#pragma once

#include <torch/all.h>

#include <atomic>
#include <shared_mutex>
#include <unordered_map>
#include <memory>
#include <vector>
#include <functional>

namespace rtp_llm {

// 关联一块内存/显存
class BlockBuffer {
public:
    BlockBuffer(
        const std::string& key_, const std::shared_ptr<void>& addr_, uint32_t len_, bool gpu_mem_, bool adopted_):
        key(key_), addr(addr_), len(len_), gpu_mem(gpu_mem_), adopted(adopted_) {}
    BlockBuffer(const BlockBuffer& rhs):
        key(rhs.key), addr(rhs.addr), len(rhs.len), gpu_mem(rhs.gpu_mem), adopted(rhs.adopted) {}

    std::string           key;
    std::shared_ptr<void> addr;
    uint32_t              len{0};
    bool                  gpu_mem{true};
    bool                  adopted{true};
};

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
    // Iterates every block while holding the shared lock. The callback must not
    // call back into this RequestBlockBuffer (no addBlock/getBlocks) to avoid
    // recursive locking. Use instead of getBlocks() on read-only hot paths:
    // getBlocks() deep-copies the whole map (one node allocation and string copy
    // per block), which dominates cache-store load latency for long contexts.
    void forEachBlock(const std::function<void(const std::string&                                key,
                                                const std::shared_ptr<BlockBuffer>& block)>& callback) const;
    // Moves every block node from src into this buffer without reallocating or
    // copying keys (std::unordered_map::merge). Used by the RDMA combined-load
    // path, which previously copied and re-inserted every block of every layer
    // buffer. src is left empty; both sizes are updated under a deadlock-free
    // dual lock.
    void mergeBlocksFrom(RequestBlockBuffer& src);
    std::shared_ptr<BlockBuffer>                                  getBlock(const std::string& id) const;
    size_t                                                        getBlocksCount() const;
    size_t                                                        getBlocksSize() const;

    void addBlock(const std::shared_ptr<BlockBuffer>& block);
    void addBlock(const std::string& key, const std::shared_ptr<void>& addr, uint32_t len, bool gpu_mem, bool adopted);
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
    // Set (release) once the first watch func is registered; read (relaxed) on
    // the per-block add path so block construction skips the watch-func lock
    // and the one-element argument vector when no watch func exists. The load
    // path adds thousands of blocks before any watch func is registered.
    std::atomic<bool>          has_watch_funcs_{false};
};

}  // namespace rtp_llm
