#include "rtp_llm/cpp/cache/connector/p2p/transfer/barex_rdma/RdmaMemoryManager.h"

#include "aios/network/accl-barex/include/accl/barex/xsimple_mempool.h"
#include <optional>
#include "autil/EnvUtil.h"

#include "rtp_llm/cpp/utils/Logger.h"

#include <algorithm>
#include <cstdint>

namespace rtp_llm {
namespace transfer {
namespace barex_rdma {

RdmaMemoryManager::RdmaMemoryManager() {
    // 初始化 RdmaMempoolFactory（如果尚未初始化）
    arpc::RdmaMempoolFactory::Init();
    rdma_mempool_ = arpc::RdmaMempoolFactory::GetMempool();
    if (!rdma_mempool_) {
        RTP_LLM_LOG_ERROR("rdma memory manager init failed, get rdma mempool failed");
    }

#ifdef ACCL_USE_EIC  // eic current only support less than 64MB
    max_reg_mem_size_ = 64L * 1024 * 1024;
#endif
    max_reg_mem_size_ = autil::EnvUtil::getEnv("CACHE_STORE_MAX_REG_MEM_SIZE", max_reg_mem_size_);
    min_reg_mem_size_ = autil::EnvUtil::getEnv("CACHE_STORE_MIN_REG_MEM_SIZE", min_reg_mem_size_);
    RTP_LLM_LOG_INFO(
        "rdma memory manager max reg mem size is %lu, min reg mem size is %lu", max_reg_mem_size_, min_reg_mem_size_);
}

RdmaMemoryManager::~RdmaMemoryManager() {
    // 自动注销所有已注册的 MR
    if (!rdma_mempool_) {
        return;
    }

    std::unique_lock<std::shared_mutex> lock(mr_mutex_);
    size_t                              mr_count = mr_info_list_.size();
    for (auto& mr_info : mr_info_list_) {
        bool is_gpu = (mr_info.mem_info_.d_type == ::accl::barex::GPU);
        rdma_mempool_->DeregUserMr(mr_info.begin_, is_gpu ? arpc::RdmaMempool::GPU : arpc::RdmaMempool::CPU);
        RTP_LLM_LOG_INFO("auto dereg user mr in dtor, base: %p, buf: %p", mr_info.base_, mr_info.begin_);
    }
    mr_info_list_.clear();
    if (mr_count > 0) {
        RTP_LLM_LOG_INFO("rdma memory manager dtor, deregistered %lu MRs", mr_count);
    }

    arpc::RdmaMempoolFactory::Shutdown();
}

bool RdmaMemoryManager::regUserMr(const BlockInfo& block_info, uint64_t aligned_size) {
    if (block_info.addr == nullptr || block_info.size_bytes == 0) {
        RTP_LLM_LOG_WARNING("reg user mr failed, block_info addr is nullptr or size is 0");
        return false;
    }

    RTP_LLM_LOG_INFO("reg user mr, addr: %p, size: %lu, aligned_size: %lu",
                     block_info.addr, block_info.size_bytes, aligned_size);

    auto ret = regUserMrInternal(block_info.addr, block_info.size_bytes, block_info.is_cuda, aligned_size);
    if (!ret) {
        RTP_LLM_LOG_ERROR("reg user mr failed, addr: %p, size: %lu", block_info.addr, block_info.size_bytes);
        return false;
    }
    return true;
}

bool RdmaMemoryManager::deregUserMr(const BlockInfo& block_info) {
    if (block_info.addr == nullptr) {
        RTP_LLM_LOG_WARNING("dereg user mr failed, block_info addr is nullptr");
        return false;
    }
    return deregUserMrInternal(block_info.addr, block_info.is_cuda);
}

std::shared_ptr<RemoteBuffer> RdmaMemoryManager::findMemoryMr(const BlockInfo& block_info) {
    if (block_info.addr == nullptr || block_info.size_bytes == 0) {
        return nullptr;
    }

    auto mr = findMemoryMrInternal(block_info.addr, block_info.size_bytes, block_info.is_cuda);
    if (!mr) {
        return nullptr;
    }

    return std::make_shared<RemoteBuffer>(reinterpret_cast<int64_t>(block_info.addr),
                                          static_cast<uint32_t>(block_info.size_bytes),
                                          mr->nic_rkeys_);
}

bool RdmaMemoryManager::regUserMrInternal(void* buf, uint64_t size, bool gpu, uint64_t aligned_size) {
    if (!rdma_mempool_) {
        RTP_LLM_LOG_WARNING("reg user mr failed, rdma mempool is nullptr");
        return false;
    }
    if (aligned_size == 0) {
        aligned_size = min_reg_mem_size_;
    }
    if (size % aligned_size != 0) {
        RTP_LLM_LOG_WARNING("reg user mr failed, size %lu not aligned to %lu", size, aligned_size);
        return false;
    } else if (size < aligned_size || max_reg_mem_size_ < aligned_size) {
        RTP_LLM_LOG_WARNING(
            "reg user mr failed, size %lu, max reg size %lu, align size %lu", size, max_reg_mem_size_, aligned_size);
        return false;
    }

    auto get_align_size = [](uint64_t size, uint64_t aligned_size) {
        return size % aligned_size == 0 ? size : size - size % aligned_size;
    };

    uint64_t max_reg_mem_size = get_align_size(max_reg_mem_size_, aligned_size);
    if (max_reg_mem_size == 0) {
        RTP_LLM_LOG_WARNING(
            "reg user mr failed, max reg mem size is %lu after align %lu", max_reg_mem_size, aligned_size);
        return false;
    }

    void*                reg_buf     = buf;
    uint64_t             to_reg_size = size;
    std::vector<mr_info> mr_infos;
    while (to_reg_size > 0) {
        uint64_t            reg_size = std::min(to_reg_size, max_reg_mem_size);
        accl::barex::memp_t mr_mem;
        bool                success = false;
        do {
            reg_size = get_align_size(reg_size, aligned_size);  // make sure reg_size is aligned
            if (reg_size < min_reg_mem_size_) {
                RTP_LLM_LOG_WARNING("reg user mr failed, after align reg size %lu is too small, align size %lu",
                                    reg_size,
                                    aligned_size);
                break;
            }
            if (rdma_mempool_->RegUserMr(
                    mr_mem, reg_buf, reg_size, gpu ? arpc::RdmaMempool::GPU : arpc::RdmaMempool::CPU)) {
                success = true;
                break;
            }
            RTP_LLM_LOG_INFO("reg user mr backward, buf: %p, size: %lu, dec to %lu", reg_buf, reg_size, reg_size / 2);
            reg_size = reg_size / 2;  // dec to half
        } while (reg_size >= min_reg_mem_size_);
        if (!success) {
            RTP_LLM_LOG_WARNING("reg user mr failed, buf: %p. size: %lu not work", reg_buf, reg_size);
            break;
        }
        auto nic_rkeys = std::make_shared<std::map<uint32_t, uint32_t>>();
        for (auto& [nicid, ibv_mr] : mr_mem.mrs) {
            nic_rkeys->insert({nicid, ibv_mr->rkey});
        }
        mr_infos.push_back(mr_info(buf, reg_buf, (int8_t*)reg_buf + reg_size, mr_mem, nic_rkeys));
        reg_buf = (void*)((uint64_t)reg_buf + reg_size);
        to_reg_size -= reg_size;
        RTP_LLM_LOG_INFO(
            "reg user mr success, base: %p, buf: %p, size: %lu, align size %lu", buf, reg_buf, reg_size, aligned_size);
    }

    if (to_reg_size > 0) {
        // reg user mr failed, call dereg user mr
        for (auto& mr_info : mr_infos) {
            rdma_mempool_->DeregUserMr(mr_info.begin_, gpu ? arpc::RdmaMempool::GPU : arpc::RdmaMempool::CPU);
        }
        return false;
    }

    RTP_LLM_LOG_INFO("reg user mr success, base: %p, size: %lu, mr count %lu, align size %lu",
                     buf,
                     size,
                     mr_infos.size(),
                     aligned_size);

    std::unique_lock<std::shared_mutex> lock(mr_mutex_);
    mr_info_list_.insert(mr_info_list_.end(), mr_infos.begin(), mr_infos.end());
    std::sort(mr_info_list_.begin(), mr_info_list_.end(), [](const mr_info& a, const mr_info& b) {
        return a.begin_ < b.begin_;
    });
    return true;
}

bool RdmaMemoryManager::deregUserMrInternal(void* buf, bool gpu) {
    if (!rdma_mempool_) {
        RTP_LLM_LOG_WARNING("dereg user mr failed, rdma mempool is nullptr");
        return false;
    }

    std::unique_lock<std::shared_mutex> lock(mr_mutex_);
    for (auto iter = mr_info_list_.begin(); iter != mr_info_list_.end();) {
        if (buf == iter->base_) {
            rdma_mempool_->DeregUserMr(iter->begin_, gpu ? arpc::RdmaMempool::GPU : arpc::RdmaMempool::CPU);
            RTP_LLM_LOG_INFO("dereg user mr success, base: %p, buf: %p", buf, iter->begin_);
            iter = mr_info_list_.erase(iter);
        } else {
            iter++;
        }
    }
    return true;
}

bool RdmaMemoryManager::findMemoryMr(::accl::barex::memp_t& mem_info, void* buf, uint64_t size, bool gpu) {
    auto mr = findMemoryMrInternal(buf, size, gpu);
    if (!mr) {
        return false;
    }

    mem_info         = mr->mem_info_;
    mem_info.buf     = (char*)buf;
    mem_info.buf_len = size;
    return true;
}

std::optional<RdmaMemoryManager::mr_info>
RdmaMemoryManager::findMemoryMrInternal(void* buf, uint64_t size, bool gpu) const {
    ::accl::barex::device_type          d_type = gpu ? ::accl::barex::GPU : ::accl::barex::CPU;
    std::shared_lock<std::shared_mutex> lock(mr_mutex_);

    auto it = std::upper_bound(mr_info_list_.begin(), mr_info_list_.end(), buf, [](void* buf, const mr_info& info) {
        return buf < info.begin_;
    });

    if (it == mr_info_list_.begin()) {
        RTP_LLM_LOG_WARNING("find memory mr from mr_info_list failed, buf: %p, gpu: %d", buf, gpu);
        return std::nullopt;
    }

    --it;
    const uintptr_t buf_end = reinterpret_cast<uintptr_t>(buf) + size;
    if (buf < it->begin_ || buf_end > reinterpret_cast<uintptr_t>(it->end_) || it->mem_info_.d_type != d_type) {
        RTP_LLM_LOG_WARNING("find memory mr from mr_info_list failed, buf: %p, size: %lu, begin: %p, end: %p, gpu: %d",
                            buf,
                            size,
                            it->begin_,
                            it->end_,
                            gpu);
        return std::nullopt;
    }

    // 在持锁期间按值拷贝，调用方拿到的副本生命周期独立于 mr_info_list_，并发 reg/dereg 安全。
    return *it;
}

}  // namespace barex_rdma
}  // namespace transfer
}  // namespace rtp_llm
