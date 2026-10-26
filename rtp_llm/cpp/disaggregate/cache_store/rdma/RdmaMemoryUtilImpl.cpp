#include "rtp_llm/cpp/disaggregate/cache_store/rdma/RdmaMemoryUtilImpl.h"
#include "aios/network/accl-barex/include/accl/barex/xsimple_mempool.h"
#include "autil/EnvUtil.h"

#include "rtp_llm/cpp/utils/Logger.h"

#include <algorithm>

namespace rtp_llm {

RdmaMemoryUtilImpl::RdmaMemoryUtilImpl() {
#ifdef ACCL_USE_EIC  // eic current only support less than 64MB
    max_reg_mem_size_ = 64L * 1024 * 1024;
#endif

    arpc::RdmaMempoolFactory::Init();
    rdma_mempool_     = arpc::RdmaMempoolFactory::GetMempool();
    max_reg_mem_size_ = autil::EnvUtil::getEnv("CACHE_STORE_MAX_REG_MEM_SIZE", max_reg_mem_size_);
    min_reg_mem_size_ = autil::EnvUtil::getEnv("CACHE_STORE_MIN_REG_MEM_SIZE", min_reg_mem_size_);
    RTP_LLM_LOG_INFO(
        "rdma memory util max reg mem size is %lu, min reg mem size is %lu", max_reg_mem_size_, min_reg_mem_size_);
}

RdmaMemoryUtilImpl::~RdmaMemoryUtilImpl() {
    arpc::RdmaMempoolFactory::Shutdown();
}

bool RdmaMemoryUtilImpl::isRdmaMode() {
    return true;
}

bool RdmaMemoryUtilImpl::regUserMr(void* buf, uint64_t size, bool gpu, uint64_t aligned_size) {
    if (!rdma_mempool_) {
        RTP_LLM_LOG_WARNING("reg user mr failed, rdma mempool is nullptr");
        return false;
    }
    if (aligned_size == 0) {
        aligned_size = min_reg_mem_size_;
    } else if (size % aligned_size != 0) {
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
        mr_infos.push_back(mr_info(buf, reg_buf, (int8_t*)reg_buf + reg_size, mr_mem));
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

    // Log registered rkeys
    for (auto& info : mr_infos) {
        std::string rkey_str;
        for (auto& kv : info.mem_info_.mrs) {
            if (!rkey_str.empty()) rkey_str += ",";
            rkey_str += std::to_string(kv.first) + ":" + std::to_string(kv.second->rkey);
        }
        RTP_LLM_LOG_INFO("[RKEY_TRACE] blockpool register: base=%p begin=%p len=%lu gpu=%d rkeys=[%s]",
                         info.base_, info.begin_,
                         (unsigned long)((char*)info.end_ - (char*)info.begin_),
                         info.mem_info_.d_type == ::accl::barex::GPU ? 1 : 0,
                         rkey_str.c_str());
    }

    return true;
}

bool RdmaMemoryUtilImpl::deregUserMr(void* buf, bool gpu) {
    if (!rdma_mempool_) {
        RTP_LLM_LOG_WARNING("dereg user mr failed, rdma mempool is nullptr");
        return false;
    }

    std::unique_lock<std::shared_mutex> lock(mr_mutex_);
    for (auto iter = mr_info_list_.begin(); iter != mr_info_list_.end();) {
        if (buf == iter->base_) {
            rdma_mempool_->DeregUserMr(iter->begin_, gpu ? arpc::RdmaMempool::GPU : arpc::RdmaMempool::CPU);
            RTP_LLM_LOG_DEBUG("dereg user mr success, base: %p, buf: %p", buf, iter->begin_);
            iter = mr_info_list_.erase(iter);
        } else {
            iter++;
        }
    }
    return true;
}

bool RdmaMemoryUtilImpl::isMemoryMr(void* ptr, size_t len, bool gpu, bool adopted) {
    accl::barex::memp_t mem;
    if (!findMemoryMr((void*)&mem, ptr, len, gpu, adopted)) {
        return false;
    }
    return true;
}

bool RdmaMemoryUtilImpl::findMemoryMr(void* out, void* buf, uint64_t size, bool gpu, bool adopted) {
    ::accl::barex::memp_t*     mem_out = (accl::barex::memp_t*)out;
    ::accl::barex::device_type d_type  = gpu ? ::accl::barex::GPU : ::accl::barex::CPU;
    if (!adopted) {
        if (!rdma_mempool_) {
            RTP_LLM_LOG_WARNING("find memory mr failed, rdma mempool is nullptr");
            return false;
        }

        if (rdma_mempool_->GetXMempool()->FindBufferMr(*mem_out, buf, d_type, 0, adopted)
            != ::accl::barex::BAREX_SUCCESS) {
            RTP_LLM_LOG_WARNING("find memory mr from xmempool failed, gpu: %d, adopted: %d", gpu, adopted);
            return false;
        }
        return true;
    }

    std::shared_lock<std::shared_mutex> lock(mr_mutex_);
    auto it = std::upper_bound(mr_info_list_.begin(), mr_info_list_.end(), buf, [](void* buf, const mr_info& info) {
        return buf < info.begin_;
    });

    if (it == mr_info_list_.begin()) {
        RTP_LLM_LOG_WARNING(
            "find memory mr from mr_info_list failed, buf: %p, gpu: %d, adopted: %d", buf, gpu, adopted);
        return false;
    }

    // get real iter
    it--;
    if (buf < it->begin_ || ((int64_t)buf + size) > (int64_t)(it->end_) || it->mem_info_.d_type != d_type) {
        RTP_LLM_LOG_WARNING(
            "find memory mr from mr_info_list failed, buf: %p, size: %lu, begin: %p, end: %p, gpu: %d, adopted: %d",
            buf,
            size,
            it->begin_,
            it->end_,
            gpu,
            adopted);
        return false;
    }

    *mem_out           = it->mem_info_;
    (*mem_out).buf     = (char*)buf;
    (*mem_out).buf_len = size;
    return true;
}

}  // namespace rtp_llm
