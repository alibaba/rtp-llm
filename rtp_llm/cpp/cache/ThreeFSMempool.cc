#include "rtp_llm/cpp/cache/ThreeFSMempool.h"

#include <algorithm>
#include <numeric>
#include <vector>

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm::threefs {

ThreeFSMempool::~ThreeFSMempool() {
    buffer_     = nullptr;
    total_size_ = 0;
}

bool ThreeFSMempool::init() {
    if (buffer_ == nullptr || total_size_ <= 0) {
        RTP_LLM_LOG_WARNING("3fs mempool init failed, buffer or total size is invalid, buffer: %p, total size: %zu",
                            buffer_,
                            total_size_);
        return false;
    }

    {
        // 初始整个buffer为空闲块
        std::unique_lock<std::shared_mutex> lock(free_blocks_mutex_);
        free_blocks_[static_cast<uint8_t*>(buffer_)] = total_size_;
        RTP_LLM_LOG_DEBUG("init free size: %zu, addr: %p", total_size_, buffer_);
    }
    return true;
}

void* ThreeFSMempool::alloc(size_t size) {
    if (size == 0) {
        RTP_LLM_LOG_WARNING("alloc failed, alloc size is 0");
        return nullptr;
    }

    // 向上对齐
    if (align_size_ > 0 && size % align_size_ != 0) {
        size = (size / align_size_ + 1) * align_size_;
    }

    uint8_t* block_addr = nullptr;
    size_t   block_size = 0;
    {
        std::unique_lock<std::shared_mutex> lock(free_blocks_mutex_);

        // 寻找最佳匹配
        for (auto it = free_blocks_.begin(); it != free_blocks_.end(); ++it) {
            if (it->second >= size && (block_addr == nullptr || it->second < block_size)) {
                block_addr = it->first;
                block_size = it->second;
            }
        }

        // 分割剩余空间
        if (block_addr != nullptr && block_size > 0) {
            free_blocks_.erase(block_addr);
            if (block_size > size) {
                uint8_t* new_free_start      = block_addr + size;
                size_t   new_free_size       = block_size - size;
                free_blocks_[new_free_start] = new_free_size;
            }
        }
    }

    if (block_addr == nullptr || block_size == 0) {
        const auto status = toString();
        RTP_LLM_LOG_WARNING("alloc failed, no free block, alloc size: %zu, mempool: [%s]", size, status.c_str());
        return nullptr;
    }

    {
        // 记录分配信息
        std::unique_lock<std::shared_mutex> lock(allocated_blocks_mutex_);
        allocated_blocks_[block_addr] = size;
    }

    RTP_LLM_LOG_DEBUG("alloc size: %zu, addr: %p", size, block_addr);
    return static_cast<void*>(block_addr);
}

void ThreeFSMempool::free(void* ptr) {
    if (!ptr) {
        return;
    }

    RTP_LLM_LOG_DEBUG("free addr: %p", ptr);
    uint8_t* block_start = static_cast<uint8_t*>(ptr);
    size_t   block_size  = 0;
    {
        std::unique_lock<std::shared_mutex> lock(allocated_blocks_mutex_);
        if (auto alloc_it = allocated_blocks_.find(block_start); alloc_it != allocated_blocks_.end()) {
            block_size = alloc_it->second;
            allocated_blocks_.erase(alloc_it);
        } else {
            RTP_LLM_LOG_WARNING("free memory but pointer is not allocated, ptr: %p", block_start);
            return;
        }
    }

    uint8_t* merge_start = block_start;
    size_t   merge_size  = block_size;

    {
        std::unique_lock<std::shared_mutex> lock(free_blocks_mutex_);
        // 向前合并：查找结束地址等于当前块起始地址的空闲块
        for (auto it = free_blocks_.begin(); it != free_blocks_.end(); ++it) {
            uint8_t* prev_end = it->first + it->second;
            if (prev_end == block_start) {
                merge_start = it->first;
                merge_size += it->second;
                free_blocks_.erase(it);
                break;  // 最多只有一个前相邻空闲块
            }
        }

        // 向后合并：查找起始地址等于当前块结束地址的空闲块
        uint8_t* block_end = block_start + block_size;
        if (auto next_it = free_blocks_.find(block_end); next_it != free_blocks_.end()) {
            merge_size += next_it->second;
            free_blocks_.erase(next_it);
        }

        // 添加合并后的空闲块
        free_blocks_[merge_start] = merge_size;
    }
}

size_t ThreeFSMempool::allocatedSize() const {
    std::shared_lock<std::shared_mutex> lock(allocated_blocks_mutex_);
    return std::accumulate(allocated_blocks_.begin(), allocated_blocks_.end(), 0ULL, [](size_t sum, const auto& block) {
        return sum + block.second;
    });
}

size_t ThreeFSMempool::freeSize() const {
    std::shared_lock<std::shared_mutex> lock(free_blocks_mutex_);
    return std::accumulate(free_blocks_.begin(), free_blocks_.end(), 0ULL, [](size_t sum, const auto& block) {
        return sum + block.second;
    });
}

int ThreeFSMempool::allocatedBlockCount() const {
    std::shared_lock<std::shared_mutex> lock(allocated_blocks_mutex_);
    return static_cast<int>(allocated_blocks_.size());
}

int ThreeFSMempool::freeBlockCount() const {
    std::shared_lock<std::shared_mutex> lock(free_blocks_mutex_);
    return static_cast<int>(free_blocks_.size());
}

std::string ThreeFSMempool::toString() const {
    std::ostringstream oss;
    oss << "total size: " << total_size_ << ", allocated size: " << allocatedSize() << ", free size: " << freeSize()
        << ", align size: " << align_size_ << ", allocated block count: " << allocatedBlockCount()
        << ", free block count: " << freeBlockCount();
    return oss.str();
}

void ThreeFSMempool::printStatus() const {
    printf("\n---------- Memory Pool Status ----------\n");
    printf("Total Size:       %zu bytes\n", total_size_);

    // 计算已用空间
    size_t used_size = 0;
    {
        std::shared_lock<std::shared_mutex> lock(allocated_blocks_mutex_);
        for (const auto& alloc_block : allocated_blocks_) {
            used_size += alloc_block.second;
        }
    }

    // 计算空闲空间
    size_t free_size      = 0;
    size_t max_free_block = 0;

    {
        std::shared_lock<std::shared_mutex> lock(free_blocks_mutex_);
        for (const auto& free_block : free_blocks_) {
            free_size += free_block.second;
            if (free_block.second > max_free_block) {
                max_free_block = free_block.second;
            }
        }
    }

    // 计算碎片率
    double fragmentation = 0.0;
    if (free_size > 0) {
        fragmentation = 100.0 * (1.0 - static_cast<double>(max_free_block) / free_size);
    }

    size_t allocated_blocks_size;
    {
        std::shared_lock<std::shared_mutex> lock(allocated_blocks_mutex_);
        allocated_blocks_size = allocated_blocks_.size();
    }

    size_t free_blocks_size;
    {
        std::shared_lock<std::shared_mutex> lock(free_blocks_mutex_);
        free_blocks_size = free_blocks_.size();
    }

    printf("Used Space:       %zu bytes\n", used_size);
    printf("Free Space:       %zu bytes\n", free_size);
    printf("Largest Free:     %zu bytes\n", max_free_block);
    printf("Fragmentation:    %.1f%%\n", fragmentation);
    printf("Allocated Blocks: %zu\n", allocated_blocks_size);
    printf("Free Blocks:      %zu\n", free_blocks_size);

    // 打印内存布局
    std::vector<std::pair<uint8_t*, size_t>> segments;
    for (const auto& alloc_block : allocated_blocks_) {
        segments.push_back({alloc_block.first, alloc_block.second});
    }
    {
        std::shared_lock<std::shared_mutex> lock(free_blocks_mutex_);
        for (const auto& free_block : free_blocks_) {
            segments.push_back(free_block);
        }
    }

    std::sort(segments.begin(), segments.end());

    printf("\nMemory Layout:\n");
    for (const auto& seg : segments) {
        printf("[%s] Addr: %p Size: %zu bytes\n",
               allocated_blocks_.count(seg.first) ? "USED" : "FREE",
               static_cast<void*>(seg.first),
               seg.second);
    }
    printf("----------------------------------------\n");
}

}  // namespace rtp_llm::threefs