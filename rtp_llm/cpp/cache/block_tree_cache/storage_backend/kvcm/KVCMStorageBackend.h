#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/StorageBackend.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTaskRegistry.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/FrozenTask.h"
#include "rtp_llm/cpp/config/ConfigModules.h"

class RemoteOperationRequestPB;
class RemoteOperationResponsePB;

namespace kmonitor {
class MetricsReporter;
}

namespace rtp_llm {

class BroadcastManager;
class TransferCompletionCleanup;
namespace kvcm {
class ClientWrapper;
}

// KVCM adapter for the BlockTree remote tier. Metadata operations are owned by
// rank 0; payload copies are broadcast to the per-rank adapters so every
// process transfers its own GPU blocks through its locally registered client.
class KVCMStorageBackend: public StorageBackend {
public:
    KVCMStorageBackend(const CacheConfig&                         cache_config,
                       const KVCacheConfig&                       kv_cache_config,
                       const RuntimeConfig&                       runtime_config,
                       const ParallelismConfig&                   parallelism_config,
                       const SpeculativeExecutionConfig&          sp_config,
                       std::shared_ptr<BroadcastManager>          broadcast_manager,
                       bool                                       gdr_enabled,
                       std::shared_ptr<kmonitor::MetricsReporter> metrics_reporter = nullptr,
                       std::shared_ptr<kvcm::ClientWrapper>       client_wrapper   = nullptr);
    ~KVCMStorageBackend() override;
    BlockTreeQueueSizes queueSizes() const override;

    struct ControlConfig {
        std::string              instance_group;
        std::string              instance_id;
        std::string              register_instance_request;
        bool                     enable_vipserver = false;
        std::string              vipserver_domain;
        std::vector<std::string> addresses;
        bool                     merge_full_specs = false;
    };
    // Immutable snapshot of the effective SDK registration, available after init.
    std::optional<ControlConfig> controlConfig() const;

    // Returns whether the request was handled and response is valid. Transfer
    // success, failure, and timeout are carried by response.transfer_status().
    bool execute(const RemoteOperationRequestPB& request, RemoteOperationResponsePB& response);
    // Local READ completion evidence; this accessor does not issue a query RPC.
    const TransferTaskRegistry& taskRegistry() const;

    // Recovery query atomically prevents a previously unseen READ from starting.
    TransferTaskRegistry::State queryReadCompletion(const std::string& trace_id) const;
    // Removes only completed READ evidence. Running executions remain queryable.
    TransferTaskRegistry::ForgetResult forgetReadCompletion(const std::string& trace_id);
    // Configure before serving READ requests; cleanup does not own their buffers.
    void setCompletionCleanup(std::shared_ptr<TransferCompletionCleanup> cleanup);
    // Retained non-protected READ targets; enumeration removes only RELEASED
    // records. It never queries completion or releases frozen references.
    std::vector<std::shared_ptr<FrozenTask>> frozenTasks() const;

    // Optional EOS publication; true means handled/skipped, not committed.
    bool
    captureCompletedDecode(const BatchKVCacheResource& resource, size_t computed_end, bool finished = true) override;

protected:
    bool               initImpl() override;
    StorageMatchResult matchImpl(const StorageRequest& request) override;
    void readImpl(const StorageRequest& request, const std::shared_ptr<StorageBackendMatchMeta>& match_meta) override;
    bool usesAsyncPayload(const StorageRequest& request, bool is_write) const noexcept override;
    bool readAsyncImpl(const StorageRequest&                           request,
                       const std::shared_ptr<StorageBackendMatchMeta>& match_meta,
                       ReadCompletion                                  done) override;
    void writeImpl(const StorageRequest& request) override;
    bool startWriteImpl(const StorageRequest& request, WriteCompletion done) override;
    void shutdownImpl() noexcept override;
    void onRemoteCacheSkipped(const char* stage) noexcept override;
    bool protectUnknownCompletion() const noexcept override;
    std::shared_ptr<void> completionResourceOwner() const override;
    bool                  hasUnresolvedExternalIo() const noexcept override;
    void onQuarantined(bool is_write, StorageQuarantineReason reason, size_t block_count) noexcept override;
    void onQuarantineChanged(uint64_t                          generation,
                             size_t                            task_count,
                             size_t                            block_count,
                             const StorageQuarantineBreakdown& breakdown) noexcept override;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace rtp_llm
