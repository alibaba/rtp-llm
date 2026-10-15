#pragma once

#include <memory>

#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/StorageBackend.h"
#include "rtp_llm/cpp/config/ConfigModules.h"

class RemoteOperationRequestPB;
class RemoteOperationResponsePB;

namespace kmonitor {
class MetricsReporter;
}

namespace rtp_llm {

class BroadcastManager;
namespace kvcm {
class ClientWrapper;
}

// KVCM adapter for the BlockTree remote tier. Metadata operations are owned by
// rank 0; payload copies are broadcast to the per-rank adapters so every
// process transfers its own GPU blocks through its locally registered client.
class KVCMStorageBackend final: public StorageBackend {
public:
    KVCMStorageBackend(const CacheConfig&                            cache_config,
                       const KVCacheConfig&                          kv_cache_config,
                       const RuntimeConfig&                          runtime_config,
                       const ParallelismConfig&                      parallelism_config,
                       const SpeculativeExecutionConfig&             sp_config,
                       std::shared_ptr<BroadcastManager>             broadcast_manager,
                       bool                                          gdr_enabled,
                       std::shared_ptr<kmonitor::MetricsReporter>    metrics_reporter = nullptr,
                       std::shared_ptr<kvcm::ClientWrapper>          client_wrapper = nullptr);
    ~KVCMStorageBackend() override;

    // Returns whether the request was handled and response is valid. Transfer
    // success, failure, and timeout are carried by response.transfer_status().
    bool execute(const RemoteOperationRequestPB& request, RemoteOperationResponsePB& response);

protected:
    bool               initImpl() override;
    StorageMatchResult matchImpl(const StorageRequest& request) override;
    void readImpl(const StorageRequest& request, const std::shared_ptr<StorageBackendMatchMeta>& match_meta) override;
    void writeImpl(const StorageRequest& request) override;
    void shutdownImpl() noexcept override;
    void onQuarantineChanged(uint64_t generation, size_t task_count, size_t block_count) noexcept override;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace rtp_llm
