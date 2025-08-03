#pragma once

#include "rtp_llm/cpp/core/Buffer.h"
#include "rtp_llm/cpp/core/Event.h"

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

    rtp_llm::Buffer toDeviceBuffer();

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
    RequestBlockBuffer(const std::string& requestid, rtp_llm::DeviceEventPtr event);

    ~RequestBlockBuffer();

public:
    const std::string&          getRequestId() const;
    const std::string&          getRequestKey() const;
    const rtp_llm::DeviceEvent* getEvent() const;

    std::unordered_map<std::string, std::shared_ptr<BlockBuffer>> getBlocks() const;
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

    rtp_llm::DeviceEventPtr event_;

    mutable std::shared_mutex                                     blocks_mutex_;
    std::unordered_map<std::string, std::shared_ptr<BlockBuffer>> blocks_;
    size_t                                                        blocks_size_ = 0;

    mutable std::shared_mutex watch_func_mutex_;
    std::vector<WatchFunc>    watch_funcs_;
};

}  // namespace rtp_llm
