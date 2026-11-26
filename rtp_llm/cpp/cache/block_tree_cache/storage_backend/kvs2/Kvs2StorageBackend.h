#pragma once

#include <atomic>
#include <list>
#include <shared_mutex>

#include "kvs/client.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/StorageBackend.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/FrozenTask.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTaskRegistry.h"

class FunctionRequestPB;
class RemoteOperationRequestPB;
class RemoteOperationResponsePB;

namespace rtp_llm {

class BroadcastManager;
struct CacheConfig;

class Kvs2StorageBackend final: public StorageBackend {
public:
    Kvs2StorageBackend(const CacheConfig&                               cache_config,
                       kvs::ClientOptions                               options,
                       std::string                                      tenant,
                       std::string                                      model_name,
                       int64_t                                          tp_rank,
                       int64_t                                          tp_size,
                       int64_t                                          world_rank,
                       std::shared_ptr<BroadcastManager>                 broadcast,
                       std::shared_ptr<BlockTreeCacheMetricsReporter> metrics);
    ~Kvs2StorageBackend() override;

    void execute(const RemoteOperationRequestPB& request, RemoteOperationResponsePB& response);
    void setCompletionCleanup(std::shared_ptr<TransferCompletionCleanup> cleanup);
    std::vector<std::shared_ptr<FrozenTask>> frozenTasks() const;
    TransferTaskRegistry::State queryCompletion(const std::string& trace);
    TransferTaskRegistry::ForgetResult forgetCompletion(const std::string& trace);
    void cleanupExpiredMissing();

protected:
    bool initImpl() override;
    StorageMatchResult matchImpl(const StorageRequest& request) override;
    void readImpl(const StorageRequest& request, const std::shared_ptr<StorageBackendMatchMeta>& meta) override;
    void writeImpl(const StorageRequest& request) override;
    void shutdownImpl() noexcept override;

private:
    kvs::ClientOptions optionsFor(TransferTask::Clock::time_point deadline) const;
    std::string key(CacheKeyType key) const;
    FunctionRequestPB payloadRequest(bool write);
    std::vector<bool> dispatch(const StorageRequest& request, FunctionRequestPB& payload);
    void executePayload(const RemoteOperationRequestPB& request,
                        TransferTask::Clock::time_point deadline,
                        RemoteOperationResponsePB& response);

    const CacheConfig&                         cache_config_;
    const kvs::ClientOptions                   options_;
    const std::string                          tenant_;
    const std::string                          model_name_;
    const int64_t                              tp_rank_;
    const int64_t                              tp_size_;
    const std::shared_ptr<BroadcastManager>    broadcast_;
    kvs::BlockLayout                           layout_;
    std::string                                layout_id_;
    std::shared_mutex                          runtime_mutex_;
    std::unique_ptr<kvs::Client>               runtime_client_;
    TransferTaskRegistry                       registry_;
    std::atomic<uint64_t>                      sequence_{0};
    std::shared_ptr<TransferCompletionCleanup> cleanup_;
    mutable std::mutex                        frozen_mutex_;
    mutable std::list<std::shared_ptr<FrozenTask>> frozen_;
};

}  // namespace rtp_llm
