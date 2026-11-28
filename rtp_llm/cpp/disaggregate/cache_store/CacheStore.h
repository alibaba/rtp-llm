#pragma once
#include "rtp_llm/cpp/disaggregate/cache_store/RequestBlockBuffer.h"
#include "rtp_llm/cpp/disaggregate/cache_store/CommonDefine.h"
#include "rtp_llm/cpp/disaggregate/cache_store/MemoryUtil.h"
#include "rtp_llm/cpp/disaggregate/cache_store/LoadContext.h"
#include "rtp_llm/cpp/disaggregate/cache_store/RemoteStoreTask.h"
#include "rtp_llm/cpp/disaggregate/cache_store/CacheStoreMetricsCollector.h"


#include <memory>
#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

namespace rtp_llm {

class CacheStore: public std::enable_shared_from_this<CacheStore> {

public:
    CacheStore() {};
    virtual ~CacheStore() {};

    virtual void store(const std::shared_ptr<RequestBlockBuffer>& request_block_buffer,
                       CacheStoreStoreDoneCallback                callback) = 0;

    virtual void load(const std::shared_ptr<RequestBlockBuffer>& request_block_buffer,
                      CacheStoreLoadDoneCallback                 callback,
                      const std::string&                         ip,
                      uint32_t                                   port,
                      uint32_t                                   rdma_port,
                      uint32_t                                   timeout_ms      = 1000,
                      int                                        partition_count = 1,
                      int                                        partition_id    = 0) = 0;

    virtual std::shared_ptr<LoadContext>
    loadBuffers(const std::vector<std::shared_ptr<RequestBlockBuffer>>& request_block_buffers,
                const std::string&                                      ip,
                uint32_t                                                port,
                uint32_t                                                rdma_port,
                int64_t                                                 timeout_ms,
                LoadContext::CheckCancelFunc                            check_cancel_func,
                int                                                     partition_count = 1,
                int                                                     partition_id    = 0) = 0;

    virtual std::shared_ptr<StoreContext>
    storeBuffers(const std::vector<std::shared_ptr<RequestBlockBuffer>>& request_block_buffers, int64_t timeout_ms) = 0;

    virtual std::shared_ptr<RemoteStoreTask>
                 submitRemoteStoreTask(const std::shared_ptr<RemoteStoreRequest>& request,
                                       const std::shared_ptr<CacheStoreRemoteStoreMetricsCollector>& collector,
                                       RemoteStoreTask::CheckCancelFunc           check_cancel_func) = 0;
    virtual void releaseRemoteStoreTask(const std::shared_ptr<RemoteStoreTask>& task)      = 0;

    virtual bool                         regUserBuffers(const std::vector<std::shared_ptr<BlockBuffer>>& buffers) = 0;
    virtual std::shared_ptr<BlockBuffer> findUserBuffer(const std::string& buffer_key)                            = 0;

    virtual const std::shared_ptr<MemoryUtil>& getMemoryUtil() const = 0;

    using BlockLeaseProvider = std::function<std::shared_ptr<void>(const std::vector<int32_t>&, const std::string&)>;

    // Configure once during initialization, before publishing or loading blocks.
    void setBlockLeaseProvider(BlockLeaseProvider provider) {
        block_lease_provider_ = std::move(provider);
    }

    std::shared_ptr<void> retainTransferBlocks(const std::vector<int32_t>& blocks, const std::string& tag) const {
        return block_lease_provider_(blocks, tag);
    }

    // Called after all pools have been registered, before serving requests.
    virtual void finishRegistration() {}
    virtual void debugInfo() = 0;

private:
    BlockLeaseProvider block_lease_provider_;
};

}  // namespace rtp_llm