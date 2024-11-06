#pragma once

#include <shared_mutex>
#include <unordered_map>

namespace rtp_llm {

// 关联一块内存/显存
class BlockBuffer {
public:
    BlockBuffer(
        const std::string& key_, const std::shared_ptr<void>& addr_, uint32_t len_, bool gpu_mem_, bool adopted_):
        key(key_), addr(addr_), len(len_), gpu_mem(gpu_mem_), adopted(adopted_) {}

    std::string           key;
    std::shared_ptr<void> addr;
    uint32_t              len{0};
    bool                  gpu_mem{true};
    bool                  adopted{true};
};

//  request 关联的 block buffer
class RequestBlockBuffer {
public:
    RequestBlockBuffer(const std::string& requestid);
    RequestBlockBuffer(const std::string& requestid, const std::shared_ptr<void>& event);

public:
    const std::string&           getRequestId() const;
    const std::shared_ptr<void>& getEvent() const;

    std::unordered_map<std::string, std::shared_ptr<BlockBuffer>> getBlocks() const;
    std::shared_ptr<BlockBuffer>                                  getBlock(const std::string& id) const;
    size_t                                                        getBlocksCount() const;

    void addBlock(const std::shared_ptr<BlockBuffer>& block);
    void addBlock(const std::string& key, const std::shared_ptr<void>& addr, uint32_t len, bool gpu_mem, bool adopted);

    bool isValid() const;

private:
    std::string           requestid_;
    std::shared_ptr<void> event_;

    mutable std::shared_mutex                                     blocks_mutex_;
    std::unordered_map<std::string, std::shared_ptr<BlockBuffer>> blocks_;
};

}  // namespace rtp_llm
