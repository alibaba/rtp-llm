#pragma once

#include "rtp_llm/cpp/cache/connector/p2p/transfer/barex_rdma/RdmaInterface.h"
#include "aios/network/rdma/accl/RdmaMempool.h"
#include "aios/network/rdma/accl/RdmaMempoolFactory.h"
#include "aios/network/accl-barex/include/accl/barex/xsimple_mempool.h"
#include <memory>
#include <optional>
#include <shared_mutex>
#include <vector>

namespace rtp_llm {
namespace transfer {
namespace barex_rdma {

/// @brief RDMA 内存管理器，负责注册、注销和查找 RDMA Memory Region
class RdmaMemoryManager: public IRdmaMemoryManager {
public:
    RdmaMemoryManager();
    ~RdmaMemoryManager();

public:
    bool regUserMr(const BlockInfo& block_info, uint64_t aligned_size = 0) override;
    bool deregUserMr(const BlockInfo& block_info) override;
    std::shared_ptr<RemoteBuffer> findMemoryMr(const BlockInfo& block_info) override;

    /// @brief 通过 memp_t 填充 MR 信息（供连接层使用）
    bool findMemoryMr(::accl::barex::memp_t& mem_info, void* buf, uint64_t size, bool gpu);

    /// @brief 获取底层 XSimpleMempool 供 RDMA transport 初始化使用
    std::shared_ptr<::accl::barex::XSimpleMempool> getXMempool() {
        if (!rdma_mempool_) {
            return nullptr;
        }
        return rdma_mempool_->GetXMempool();
    }

private:
    struct mr_info {
        void*                                         base_{nullptr};
        void*                                         begin_{nullptr};
        void*                                         end_{nullptr};
        accl::barex::memp_t                           mem_info_;
        std::shared_ptr<std::map<uint32_t, uint32_t>> nic_rkeys_;
        mr_info(void*                                                base,
                void*                                                begin,
                void*                                                end,
                accl::barex::memp_t                                  mem_info,
                const std::shared_ptr<std::map<uint32_t, uint32_t>>& nic_rkeys):
            base_(base), begin_(begin), end_(end), mem_info_(mem_info), nic_rkeys_(nic_rkeys) {}
    };

private:
    std::optional<mr_info> findMemoryMrInternal(void* buf, uint64_t size, bool gpu) const;
    bool regUserMrInternal(void* buf, uint64_t size, bool gpu, uint64_t aligned_size = 0);
    bool deregUserMrInternal(void* buf, bool gpu);

    void setMaxRegMemSize(uint64_t max_size) { max_reg_mem_size_ = max_size; }
    void setMinRegMemSize(uint64_t min_size) { min_reg_mem_size_ = min_size; }
    uint64_t getMaxRegMemSize() const { return max_reg_mem_size_; }
    uint64_t getMinRegMemSize() const { return min_reg_mem_size_; }

private:
    std::shared_ptr<arpc::RdmaMempool> rdma_mempool_;
    mutable std::shared_mutex          mr_mutex_;
    std::vector<mr_info>               mr_info_list_;
    uint64_t                           max_reg_mem_size_ = std::numeric_limits<uint64_t>::max();  // unlimited size
    uint64_t                           min_reg_mem_size_ = 1L * 1024;                             // 1KB
};

}  // namespace barex_rdma
}  // namespace transfer
}  // namespace rtp_llm
