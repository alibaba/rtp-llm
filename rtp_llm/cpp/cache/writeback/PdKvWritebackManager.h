#pragma once

#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/status/status.h"
#include "kmonitor/client/MetricsReporter.h"
#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/cache/writeback/PdKvWritebackTopology.h"
#include "rtp_llm/cpp/cache/writeback/PdKvWritebackManifest.h"
#include "rtp_llm/cpp/cache/writeback/PdKvWritebackTransfer.h"
#include "rtp_llm/cpp/config/ConfigModules.h"

namespace rtp_llm {

enum class PdKvWritebackLaunchStatus {
    Started,
    Skipped,
    Failed,
};

enum class PdKvWritebackReceiveStage {
    Prepare,
    Commit,
    Abort,
};

struct PdKvWritebackCompatibility {
    int32_t              seq_size_per_block = 0;
    int32_t              layer_count        = 0;
    int32_t              group_count        = 0;
    int32_t              partition_count    = 1;
    std::vector<int32_t> layer_to_group_id;
    std::vector<int32_t> group_types;
};

struct PdKvWritebackLaunchRequest {
    PdKvWritebackManifest      manifest;
    PdKvWritebackCompatibility source;
    PdKvWritebackCompatibility destination;
    std::vector<std::string>   source_prefill_grpc_addrs;
    std::vector<std::string>   decode_worker_addrs;
    std::vector<std::string>   prefill_worker_addrs;
    int64_t                    deadline_ms   = 0;
    int32_t                    local_tp_rank = 0;
    PdKvWritebackReceiveStage  receive_stage = PdKvWritebackReceiveStage::Prepare;
    KVCacheResourcePtr         held_resource;
};

struct PdKvWritebackLaunchResult {
    PdKvWritebackLaunchStatus status = PdKvWritebackLaunchStatus::Skipped;
    std::string               reason;
};

class PdKvWritebackLauncher {
public:
    virtual ~PdKvWritebackLauncher()                                                                    = default;
    virtual PdKvWritebackLaunchResult launchFromDecode(const PdKvWritebackLaunchRequest& request) const = 0;
};

class PdKvWritebackCacheWriter {
public:
    virtual ~PdKvWritebackCacheWriter()                                                              = default;
    virtual absl::Status mallocWritebackBlocks(const BatchKVCacheResourcePtr& batch_kv_cache_resource,
                                               size_t                         block_count)                                   = 0;
    virtual void         commitWritebackBlocks(const BatchKVCacheResourcePtr& batch_kv_cache_resource,
                                               const CacheKeysType&           cache_keys,
                                               bool                           is_resident)                                     = 0;
    virtual void         freeWritebackBlocks(const BatchKVCacheResourcePtr& batch_kv_cache_resource) = 0;
};

class PdKvWritebackRpcClient {
public:
    virtual ~PdKvWritebackRpcClient()                                                     = default;
    virtual absl::Status requestPrefillReceive(const PdKvWritebackLaunchRequest& request,
                                               const PdKvWritebackTopologyPlan&  topology) = 0;
    virtual absl::Status requestPrefillCommit(const PdKvWritebackLaunchRequest& request,
                                              const PdKvWritebackTopologyPlan&  topology)  = 0;
    virtual absl::Status requestPrefillAbort(const PdKvWritebackLaunchRequest& request,
                                             const PdKvWritebackTopologyPlan&  topology)   = 0;
    virtual absl::Status requestDecodeSend(const PdKvWritebackLaunchRequest& request,
                                           const PdKvWritebackTopologyPlan&  topology)     = 0;
};

absl::Status validatePdKvWritebackCompatibility(const PdKvWritebackCompatibility& source,
                                                const PdKvWritebackCompatibility& destination);

class PdKvWritebackManager: public PdKvWritebackLauncher {
public:
    PdKvWritebackManager(const PDSepConfig& pd_config, PdKvWritebackCacheWriter* cache_writer);
    PdKvWritebackManager(const PDSepConfig&           pd_config,
                         PdKvWritebackCacheWriter*    cache_writer,
                         PdKvWritebackTransferClient* transfer_client);
    PdKvWritebackManager(const PDSepConfig&                           pd_config,
                         PdKvWritebackCacheWriter*                    cache_writer,
                         std::shared_ptr<PdKvWritebackTransferClient> transfer_client,
                         std::shared_ptr<PdKvWritebackRpcClient>      rpc_client);
    PdKvWritebackManager(const PDSepConfig&                           pd_config,
                         PdKvWritebackCacheWriter*                    cache_writer,
                         std::shared_ptr<PdKvWritebackTransferClient> transfer_client,
                         std::shared_ptr<PdKvWritebackRpcClient>      rpc_client,
                         std::vector<std::string>                     decode_worker_grpc_addrs,
                         kmonitor::MetricsReporterPtr                 metrics_reporter);

    PdKvWritebackLaunchResult launchFromDecode(const PdKvWritebackLaunchRequest& request) const override;
    absl::Status              receiveOnPrefill(const PdKvWritebackLaunchRequest& request,
                                               const BatchKVCacheResourcePtr&    destination_resource);
    absl::Status              prepareReceiveOnPrefill(const PdKvWritebackLaunchRequest& request,
                                                      const BatchKVCacheResourcePtr&    destination_resource);
    absl::Status              commitReceiveOnPrefill(const PdKvWritebackLaunchRequest& request);
    absl::Status              abortReceiveOnPrefill(const PdKvWritebackLaunchRequest& request);
    absl::Status              sendOnDecode(const PdKvWritebackLaunchRequest& request,
                                           const BatchKVCacheResourcePtr&    source_resource) const;
    void                      waitForWritebackTasksForTest() const;
    size_t                    trackedWritebackTaskCountForTest() const;
    size_t                    completedWritebackTaskCountForTest() const;

private:
    void                      pruneCompletedWritebackTasksLocked() const;
    PdKvWritebackTransferPlan buildTransferPlan(const PdKvWritebackLaunchRequest& request,
                                                const BatchKVCacheResourcePtr&    destination_resource) const;

private:
    struct PendingReceive {
        PdKvWritebackLaunchRequest request;
        BatchKVCacheResourcePtr    destination_resource;
    };

private:
    PDSepConfig                                     pd_config_;
    PdKvWritebackCacheWriter*                       cache_writer_    = nullptr;
    PdKvWritebackTransferClient*                    transfer_client_ = nullptr;
    PdKvWritebackRpcClient*                         rpc_client_      = nullptr;
    std::shared_ptr<PdKvWritebackTransferClient>    owned_transfer_client_;
    std::shared_ptr<PdKvWritebackRpcClient>         owned_rpc_client_;
    std::vector<std::string>                        decode_worker_grpc_addrs_;
    kmonitor::MetricsReporterPtr                    metrics_reporter_;
    mutable std::mutex                              writeback_tasks_mutex_;
    mutable std::vector<std::future<void>>          writeback_tasks_;
    std::mutex                                      pending_receives_mutex_;
    std::unordered_map<std::string, PendingReceive> pending_receives_;
};

using PdKvWritebackManagerPtr  = std::shared_ptr<PdKvWritebackManager>;
using PdKvWritebackLauncherPtr = std::shared_ptr<PdKvWritebackLauncher>;

}  // namespace rtp_llm
