#pragma once

#include <memory>

#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/StorageBackend.h"
#include "rtp_llm/cpp/config/ConfigModules.h"

class RemoteOperationRequestPB;
class RemoteOperationResponsePB;

namespace rtp_llm {

class BroadcastManager;

// KVCM adapter for the BlockTree remote tier. Metadata operations are owned by
// rank 0; payload copies are broadcast to the per-rank adapters so every
// process transfers its own GPU blocks through its locally registered client.
class KVCMStorageBackend final: public StorageBackend {
public:
    KVCMStorageBackend(const CacheConfig&                cache_config,
                       const KVCacheConfig&              kv_cache_config,
                       const RuntimeConfig&              runtime_config,
                       const ParallelismConfig&          parallelism_config,
                       const SpeculativeExecutionConfig& sp_config,
                       std::shared_ptr<BroadcastManager> broadcast_manager);
    ~KVCMStorageBackend() override;

    bool execute(const RemoteOperationRequestPB& request, RemoteOperationResponsePB& response);

protected:
    bool               initImpl() override;
    StorageMatchResult matchImpl(const StorageRequest& request) override;
    void readImpl(const StorageRequest& request, const std::shared_ptr<StorageBackendMatchMeta>& match_meta) override;
    void writeImpl(const StorageRequest& request) override;
    void shutdownImpl() noexcept override;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace rtp_llm
