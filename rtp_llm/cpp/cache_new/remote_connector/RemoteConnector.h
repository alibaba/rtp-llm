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
#include "rtp_llm/cpp/cache_new/BatchKVCacheResource.h"
#include "rtp_llm/cpp/cache_new/CacheConfig.h"
#include "rtp_llm/cpp/cache_new/KVCacheConnector.h"
#include "rtp_llm/cpp/cache_new/remote_connector/ClientWrapper.h"
#include "rtp_llm/cpp/cache_new/remote_connector/GroupPolicy.h"
#include "rtp_llm/cpp/model_rpc/TpBroadcastManager.h"
#include "kmonitor/client/MetricsReporter.h"

namespace rtp_llm {

class KVCacheAllocator;
class DeviceBase;
class GptInitParameter;
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

    std::string          unique_id;  // lora_name; if no lora, empty string
    std::string          trace_id;
    std::vector<int64_t> tokens;  // only for write
};

class RemoteConnector: public KVCacheConnector {
public:
    RemoteConnector(const CacheConfig&                 cache_config,
                    const GptInitParameter&            model_parameter,
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

    bool init() override;

    // for rank_0:
    std::shared_ptr<AsyncContext> asyncRead(const std::shared_ptr<KVCacheResourceV1>&      resource,
                                            const std::shared_ptr<KVCacheConnector::Meta>& meta) override;
    std::shared_ptr<AsyncContext> asyncWrite(const std::shared_ptr<KVCacheResourceV1>&      resource,
                                             const std::shared_ptr<KVCacheConnector::Meta>& meta) override;
    std::shared_ptr<AsyncContext> asyncWriteByLayer(int                                       layer_id,
                                                    const std::shared_ptr<KVCacheResourceV1>& resource,
                                                    const std::shared_ptr<Meta>&              meta) override;

    // for all rank:
    bool copyCache(const RemoteCopyCacheRequestPB& request, RemoteCopyCacheResponsePB& response);

private:
    // for rank_0:
    using ActualUriGather = std::vector<std::vector<kv_cache_manager::LocationSpecUnit*>>;
    void asyncReadTask(const std::shared_ptr<KVCacheResourceV1>&           resource,
                       const std::shared_ptr<RemoteConnectorMeta>&         meta,
                       const std::shared_ptr<RemoteConnectorAsyncContext>& async_context);
    void asyncWriteTask(const std::shared_ptr<KVCacheResourceV1>&           resource,
                        const std::shared_ptr<RemoteConnectorMeta>&         meta,
                        const std::shared_ptr<RemoteConnectorAsyncContext>& async_context);
    bool genReadRequest(size_t                                    tp_size,
                        const kv_cache_manager::Locations&        locations,
                        const kv_cache_manager::BlockMaskOffset&  block_mask,
                        const std::string&                        trace_id,
                        const std::shared_ptr<KVCacheResourceV1>& resource,
                        std::vector<CopyCacheRequestPB>&          requests,
                        size_t&                                   new_reuse_block_num) const;
    bool genWriteRequest(size_t                                    tp_size,
                         const kv_cache_manager::Locations&        locations,
                         const kv_cache_manager::BlockMask&        block_mask,
                         const std::string&                        trace_id,
                         const std::shared_ptr<KVCacheResourceV1>& resource,
                         std::vector<CopyCacheRequestPB>&          requests,
                         ActualUriGather&                          actual_uri_gather) const;
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
        const GptInitParameter&            model_parameter;
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

class RemoteConnectorAsyncContext: public AsyncContext {
public:
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

    ~RemoteConnectorAsyncContext() override;

public:
    void  waitDone() override;
    void  cancel() override;
    bool  done() const override;
    bool  success() const override;
    State state() const {
        return state_.load(std::memory_order_relaxed);
    }
    inline size_t remote_reuse_block_num() const {
        return remote_reuse_block_num_;
    }

private:
    friend class RemoteConnector;
    RemoteConnectorAsyncContext(std::shared_future<void> future);
    inline void setState(State state) {
        state_.store(state, std::memory_order_release);
    }
    inline void set_remote_reuse_block_num(size_t remote_reuse_block_num) {
        remote_reuse_block_num_ = remote_reuse_block_num;
    }

    std::atomic<State>       state_ = State::RCS_INIT;
    std::shared_future<void> future_;
    size_t                   remote_reuse_block_num_ = 0;
};

}  // namespace rtp_llm