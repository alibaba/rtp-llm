#pragma once
#include "rtp_llm/cpp/disaggregate/cache_store/RequestBlockBuffer.h"
#include "rtp_llm/cpp/disaggregate/cache_store/CommonDefine.h"
#include "rtp_llm/cpp/disaggregate/cache_store/MemoryUtil.h"
#include "rtp_llm/cpp/disaggregate/cache_store/LoadContext.h"
#include "rtp_llm/cpp/disaggregate/cache_store/RemoteStoreTask.h"
#include "rtp_llm/cpp/disaggregate/cache_store/CacheStoreMetricsCollector.h"

#include <memory>

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
                 submitRemoteStoreTask(const std::shared_ptr<RemoteStoreRequest>&                    request,
                                       const std::shared_ptr<CacheStoreRemoteStoreMetricsCollector>& collector,
                                       RemoteStoreTask::CheckCancelFunc                              check_cancel_func) = 0;
    virtual void releaseRemoteStoreTask(const std::shared_ptr<RemoteStoreTask>& task)      = 0;

    virtual bool                         regUserBuffers(const std::vector<std::shared_ptr<BlockBuffer>>& buffers) = 0;
    virtual std::shared_ptr<BlockBuffer> findUserBuffer(const std::string& buffer_key)                            = 0;

    virtual const std::shared_ptr<MemoryUtil>& getMemoryUtil() const = 0;

    // Global in-flight transfer count (store/load tasks + remote store tasks).
    // Used by DrainManager to decide drain completion before sleep.
    virtual size_t activeTransferCount() const {
        return 0;
    }

    // Every level uses begin/drain and resume to protect MR/KV release. Level 3
    // additionally follows: transport-only teardown ->
    // external MR deregistration -> MR-owner teardown -> CUDA checkpoint/restore
    // -> MR-owner rebuild -> transport rebuild (still paused) -> external MR
    // registration -> resume. teardownForCheckpoint must not destroy the
    // MemoryUtil/MR owner.
    virtual bool beginCheckpointDrain() {
        return true;
    }
    virtual bool resumeAfterCheckpoint() {
        return true;
    }
    virtual bool teardownForCheckpoint() {
        return true;
    }
    virtual bool teardownMemoryOwnerAfterMrDereg() {
        return true;
    }
    virtual bool rebuildMemoryOwnerBeforeMrReg() {
        return true;
    }
    virtual bool rebuildAfterRestore() {
        return true;
    }
    virtual bool isAvailable() const {
        return true;
    }

    virtual void debugInfo() = 0;
};

}  // namespace rtp_llm
