#pragma once

#include <cstdint>
#include <map>
#include <shared_mutex>

namespace rtp_llm::threefs {

class ThreeFSMempool final {
public:
    ThreeFSMempool(void* buffer, size_t total_size, size_t align_size):
        buffer_(static_cast<uint8_t*>(buffer)), total_size_(total_size), align_size_(align_size) {}
    ~ThreeFSMempool();

public:
    bool  init();
    void* alloc(size_t size);
    void  free(void* ptr);

    size_t allocatedSize() const;
    size_t freeSize() const;
    int    allocatedBlockCount() const;

    // for debug
    void printStatus() const;

private:
    int         freeBlockCount() const;
    std::string toString() const;

private:
    uint8_t* buffer_;
    size_t   total_size_;
    size_t   align_size_;

    // 空闲块管理: 起始地址 -> 块大小
    std::map<uint8_t*, size_t> free_blocks_;
    mutable std::shared_mutex  free_blocks_mutex_;

    // 已分配块管理: 起始地址 -> 块大小
    std::map<uint8_t*, size_t> allocated_blocks_;
    mutable std::shared_mutex  allocated_blocks_mutex_;
};

}  // namespace rtp_llm::threefs