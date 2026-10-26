#pragma once

#include <cstddef>
#include <memory>
#include <shared_mutex>

#include "aios/network/rdma/accl/RdmaMempool.h"
#include "aios/network/rdma/accl/RdmaMempoolFactory.h"

#include "rtp_llm/cpp/disaggregate/cache_store/MemoryUtil.h"

namespace rtp_llm {

class RdmaMemoryUtilImpl: public MemoryUtil {
public:
    RdmaMemoryUtilImpl();
    virtual ~RdmaMemoryUtilImpl();

    bool regUserMr(void* buf, uint64_t size, bool gpu, uint64_t aligned_size) override;
    bool deregUserMr(void* buf, bool gpu) override;
    bool isMemoryMr(void* ptr, uint64_t size, bool gpu, bool adopted) override;
    bool findMemoryMr(void* mem_info, void* buf, uint64_t size, bool gpu, bool adopted) override;

    bool isRdmaMode() override;

private:
    std::shared_ptr<arpc::RdmaMempool> rdma_mempool_;

    struct mr_info {
        void*               base_{nullptr};
        void*               begin_{nullptr};
        void*               end_{nullptr};
        accl::barex::memp_t mem_info_;
        mr_info(void* base, void* begin, void* end, accl::barex::memp_t mem_info):
            base_(base), begin_(begin), end_(end), mem_info_(mem_info) {}
    };
    std::shared_mutex    mr_mutex_;
    std::vector<mr_info> mr_info_list_;
    uint64_t             max_reg_mem_size_ = 2L * 1024 * 1024 * 1024;  // default 2GB
    uint64_t             min_reg_mem_size_ = 1L * 1024;         // 1KB
};

}  // namespace rtp_llm
