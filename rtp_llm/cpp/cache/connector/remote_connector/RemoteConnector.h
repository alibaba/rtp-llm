#pragma once

#include <thread>
#include <functional>
#include <map>
#include <unordered_map>
#include <string>
#include <atomic>
#include <memory>
#include <future>

#include "autil/LockFreeThreadPool.h"
#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/connector/KVCacheConnector.h"
#include "rtp_llm/cpp/cache/connector/remote_connector/ClientWrapper.h"
#include "rtp_llm/cpp/cache/connector/remote_connector/GroupPolicy.h"
#include "rtp_llm/cpp/model_rpc/TpBroadcastManager.h"
#include "kmonitor/client/MetricsReporter.h"

namespace rtp_llm {

class KVCacheAllocator;
class DeviceBase;
class RemoteAsyncMatchContext;
class RemoteConnectorAsyncContext;

struct RemoteConnectorMeta: public KVCacheConnector::Meta {
    RemoteConnectorMeta(const std::string& unique_id_, const std::string& trace_id_):
        unique_id(unique_id_), trace_id(trace_id_) {}
    RemoteConnectorMeta(const std::string&          unique_id_,
                        const std::string&          trace_id_,
                        const std::vector<int64_t>& tokens_):
        unique_id(unique_id_), trace_id(trace_id_), tokens(tokens_) {}
    RemoteConnectorMeta(const std::string& unique_id_, const std::string& trace_id_, std::vector<int64_t>&& tokens_):
        unique_id(unique_id_), trace_id(trace_id_), tokens(std::move(tokens_)) {}

    std::pair<int, int> blockRange() const override {
        return {};
    }

    std::string          unique_id;  // lora_name; if no lora, empty string
    std::string          trace_id;
    std::vector<int64_t> tokens;  // only for write
};

class RemoteConnector: public KVCacheConnector {
public:
    RemoteConnector(const CacheConfig&                 cache_config,
                    const KVCacheConfig&               kv_cache_config,
                    const RuntimeConfig&               runtime_config,
                    const ParallelismConfig&           parallelism_config,
                    const SpeculativeExecutionConfig&  sp_config,
                    DeviceBase*                        device,
                    void*                              register_buffer_addr,
                    size_t                             register_buffer_size,
                    std::shared_ptr<KVCacheAllocator>  allocator,
                    RemoteConnectorGroupMode           group_mode,
                    const std::vector<int32_t>&        full_group_ids,
                    const std::vector<int32_t>&        other_group_ids                 = {},
                    const kmonitor::MetricsReporterPtr metrics_reporter                = nullptr,
                    uint32_t                           linear_attention_write_interval = 1,  // for linear attention
                    size_t                             sink_size            = 0,  // for slide window attention
                    size_t                             sw_size              = 0,  // for slide window attention
                    const std::map<std::string, std::string>& lora_info_map = {});
    ~RemoteConnector() override;

    bool init();

    // for rank_0:
    std::shared_ptr<AsyncMatchContext> asyncMatch(const std::shared_ptr<KVCacheResource>& resource,
                                                  const std::shared_ptr<Meta>&            meta) override;
    std::shared_ptr<AsyncContext>      asyncRead(const std::shared_ptr<KVCacheResource>&   resource,
                                                 const std::shared_ptr<Meta>&              meta,
                                                 const std::shared_ptr<AsyncMatchContext>& match_context) override;
    std::shared_ptr<AsyncContext>      asyncWrite(const std::shared_ptr<KVCacheResource>&        resource,
                                                  const std::shared_ptr<KVCacheConnector::Meta>& meta) override;
    std::shared_ptr<AsyncContext>      asyncWriteByLayer(int                                     layer_id,
                                                         const std::shared_ptr<KVCacheResource>& resource,
                                                         const std::shared_ptr<Meta>&            meta) override;

    // for all rank:
    bool copyCache(const RemoteBroadcastTpRequestPB& request, RemoteBroadcastTpResponsePB& response);

private:
    // for rank_0:
    using ActualUriGather = std::vector<std::vector<kv_cache_manager::LocationSpecUnit*>>;
    void asyncMatchTask(const std::shared_ptr<KVCacheResource>&         resource,
                        const std::shared_ptr<RemoteConnectorMeta>&     meta,
                        const std::shared_ptr<RemoteAsyncMatchContext>& async_context);
    void asyncReadTask(const std::shared_ptr<KVCacheResource>&             resource,
                       const std::shared_ptr<Meta>&                        meta,
                       const std::shared_ptr<RemoteConnectorAsyncContext>& async_context,
                       const std::shared_ptr<RemoteAsyncMatchContext>&     match_context);
    void asyncWriteTask(const std::shared_ptr<KVCacheResource>&             resource,
                        const std::shared_ptr<RemoteConnectorMeta>&         meta,
                        const std::shared_ptr<RemoteConnectorAsyncContext>& async_context);
    bool genReadRequest(size_t                                   tp_size,
                        const kv_cache_manager::Locations&       locations,
                        size_t                                   block_idx,
                        const kv_cache_manager::BlockMaskOffset& block_mask,
                        const std::string&                       trace_id,
                        const std::shared_ptr<KVCacheResource>&  resource,
                        std::vector<BroadcastTpRequestPB>&       requests,
                        size_t&                                  new_reuse_block_num) const;
    bool genWriteRequest(size_t                                  tp_size,
                         const kv_cache_manager::Locations&      locations,
                         const kv_cache_manager::BlockMask&      block_mask,
                         const std::string&                      trace_id,
                         const std::shared_ptr<KVCacheResource>& resource,
                         std::vector<BroadcastTpRequestPB>&      requests,
                         ActualUriGather&                        actual_uri_gather) const;
    // for all_rank:
    bool Read(const std::string&                 trace_id,
              const std::vector<int32_t>&        group_ids,
              const std::vector<int32_t>&        block_ids,
              const kv_cache_manager::UriStrVec& uri_str_vec);
    bool Write(const std::string&                 trace_id,
               const std::vector<int32_t>&        group_ids,
               const std::vector<int32_t>&        block_ids,
               const kv_cache_manager::UriStrVec& uri_str_vec,
               kv_cache_manager::UriStrVec&       out_uri_str_vec);

private:
    remote_connector::ClientWrapper::ConfigMap genClientConfig();
    std::pair<std::shared_ptr<RemoteConnectorConfig::LocationSpecInfoMap>,
              std::shared_ptr<RemoteConnectorConfig::LocationSpecGroups>>
         genLocationSpecInfoMapAndGroups(int64_t tp_size);
    void printInfo() const;

private:
    struct InitParams {
        const CacheConfig&                 cache_config;
        const KVCacheConfig&               kv_cache_config;
        const RuntimeConfig&               runtime_config;
        const ParallelismConfig&           parallelism_config;
        const SpeculativeExecutionConfig&  sp_config;
        DeviceBase*                        device;
        void*                              register_buffer_addr;
        size_t                             register_buffer_size;
        std::map<std::string, std::string> lora_info_map;
        std::vector<std::string>           tp_addrs;
    };

    // use thread_pool to simulate async operation
    std::unique_ptr<autil::LockFreeThreadPool>       thread_pool_;
    std::shared_ptr<remote_connector::ClientWrapper> client_wrapper_;
    std::shared_ptr<TpBroadcastManager>              broadcaster_;
    int                                              get_broadcast_timeout_ = 2000;
    int                                              put_broadcast_timeout_ = 2000;
    std::shared_ptr<InitParams>                      init_params_;

    std::unique_ptr<remote_connector::GroupPolicy> group_policy_;
    const kmonitor::MetricsReporterPtr             metrics_reporter_;
};

class RemoteConnectorState {
public:
    virtual ~RemoteConnectorState() = default;

protected:
    friend class RemoteConnector;

    enum class State {
        RCS_INIT  = 0,
        RCS_START = 1,
        RCS_ERROR = 2,

        RCS_READ_MATCH     = 10,
        RCS_READ_BROADCAST = 11,

        RCS_WRITE_START     = 20,
        RCS_WRITE_BROADCAST = 21,
        RCS_WRITE_FINISH    = 22,

        RCS_SUCCESS          = 30,  // if State >= RCS_SUCCESS, we consider it success
        RCS_READ_MATCH_ERROR = 31,
        RCS_THREADPOOL_FULL  = 32,
    };

    bool doneImpl() const;

    bool successImpl() const;

    inline void setState(State state) {
        state_.store(state, std::memory_order_release);
    }
    inline State state() const {
        return state_.load(std::memory_order_relaxed);
    }

    std::atomic<State> state_ = State::RCS_INIT;
};

class RemoteAsyncMatchContext: public KVCacheConnector::AsyncMatchContext, public RemoteConnectorState {
public:
    explicit RemoteAsyncMatchContext(size_t prev_reuse_blocks_num): prev_reuse_blocks_num_(prev_reuse_blocks_num) {}
    ~RemoteAsyncMatchContext() override = default;

    bool   done() const override;
    bool   success() const override;
    size_t matchedBlockCount() const override {
        return matched_block_count_;
    }

    KVCacheConnector::ConnectorType connectorType() const override {
        return KVCacheConnector::ConnectorType::Remote;
    }

private:
    friend class RemoteConnector;
    // inline void set_
    inline void set_locations(kv_cache_manager::Locations&& locations) {
        *locations_ptr_ = std::move(locations);
    }
    inline const auto& locations_ptr() const {
        return locations_ptr_;
    }
    inline void set_trace_id(const std::string& trace_id) {
        trace_id_ = trace_id;
    }
    inline const auto& trace_id() const {
        return trace_id_;
    }
    inline size_t prev_reuse_blocks_num() const {
        return prev_reuse_blocks_num_;
    }
    inline void set_matched_block_count(size_t matched_block_count) {
        matched_block_count_ = matched_block_count;
    }

    size_t                                       prev_reuse_blocks_num_ = 0;
    size_t                                       matched_block_count_   = 0;
    std::shared_ptr<kv_cache_manager::Locations> locations_ptr_ = std::make_shared<kv_cache_manager::Locations>();
    std::string                                  trace_id_;
};

class RemoteConnectorAsyncContext: public AsyncContext, public RemoteConnectorState {
public:
    ~RemoteConnectorAsyncContext() override = default;

public:
    bool done() const override;
    bool success() const override;
};

}  // namespace rtp_llm