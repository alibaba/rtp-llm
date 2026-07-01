#pragma once

#include <atomic>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "autil/ThreadPool.h"
#include "kmonitor/client/MetricsReporter.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/KVCacheResource.h"
#include "rtp_llm/cpp/cache/connector/KVCacheConnector.h"
#include "rtp_llm/cpp/cache/connector/kvs_connector/KVSCacheStore.h"
#include "rtp_llm/cpp/cache/connector/kvs_connector/KVSClient.h"
#include "rtp_llm/cpp/config/ConfigModules.h"

namespace rtp_llm {

class KVCacheAllocator;

namespace kvs {

struct KVSConnectorSharedState {
    CacheConfig                       cache_config;
    KVCacheConfig                     kv_cache_config;
    RuntimeConfig                     runtime_config;
    ParallelismConfig                 parallelism_config;
    std::shared_ptr<KVCacheAllocator> allocator;
    kmonitor::MetricsReporterPtr      metrics_reporter;
    std::shared_ptr<KVSClient>        client;
    std::string                       deployment_id;
};

class KVSConnectorState {
public:
    enum class State {
        INIT            = 0,
        START           = 1,
        SUCCESS         = 2,
        ERROR           = 3,
        THREADPOOL_FULL = 4,
        CLIENT_ERROR    = 5,
    };

    bool  done() const;
    bool  success() const;
    void  set(State state);
    State state() const;

private:
    std::atomic<State> state_{State::INIT};
};

class KVSAsyncMatchContext: public AsyncMatchContext {
public:
    KVSAsyncMatchContext(size_t prev_reuse_blocks_num, std::shared_ptr<KVSClient> client):
        prev_reuse_blocks_num_(prev_reuse_blocks_num), client_(std::move(client)) {}
    ~KVSAsyncMatchContext() override;

    bool   done() const override;
    bool   success() const override;
    void   waitDone() override {}
    size_t matchedBlockCount() const override;

private:
    friend class KVSConnector;

    size_t                        prev_reuse_blocks_num_ = 0;
    size_t                        matched_block_count_   = 0;
    std::vector<KVSBlockObjects>  matched_blocks_;
    std::optional<KVSReadSession> read_session_;
    std::shared_ptr<KVSClient>    client_;
    KVSConnectorState             state_;
};

class KVSAsyncContext: public AsyncContext {
public:
    bool done() const override;
    bool success() const override;
    void waitDone() override {}

private:
    friend class KVSConnector;
    KVSConnectorState state_;
};

class KVSConnector: public KVCacheConnector {
public:
    KVSConnector(const CacheConfig&                 cache_config,
                 const KVCacheConfig&               kv_cache_config,
                 const RuntimeConfig&               runtime_config,
                 const ParallelismConfig&           parallelism_config,
                 std::shared_ptr<KVCacheAllocator>  allocator,
                 const kmonitor::MetricsReporterPtr metrics_reporter = nullptr,
                 std::shared_ptr<KVSClient>         client           = nullptr);
    ~KVSConnector() override;

    bool init();

    std::shared_ptr<AsyncMatchContext> asyncMatch(const std::shared_ptr<KVCacheResource>& resource,
                                                  const std::shared_ptr<Meta>&            meta) override;
    std::shared_ptr<AsyncContext>      asyncRead(const std::shared_ptr<KVCacheResource>&   resource,
                                                 const std::shared_ptr<Meta>&              meta,
                                                 const std::shared_ptr<AsyncMatchContext>& match_context,
                                                 int                                       start_read_block_index,
                                                 int                                       read_block_num) override;
    std::shared_ptr<AsyncContext>      asyncWrite(const std::shared_ptr<KVCacheResource>& resource,
                                                  const std::shared_ptr<Meta>&            meta) override;
    std::shared_ptr<AsyncContext>
    asyncWriteByLayer(int layer_id, const std::shared_ptr<KVCacheConnectorLayerContext>& layer_context) override;

private:
    static void asyncMatchTask(const std::shared_ptr<KVSConnectorSharedState>& state,
                               const std::shared_ptr<KVCacheResource>&         resource,
                               const std::shared_ptr<Meta>&                    meta,
                               const std::shared_ptr<KVSAsyncMatchContext>&    async_context);
    static void asyncReadTask(const std::shared_ptr<KVSConnectorSharedState>& state,
                              const std::shared_ptr<KVCacheResource>&         resource,
                              const std::shared_ptr<Meta>&                    meta,
                              int                                             start_read_block_index,
                              int                                             read_block_num,
                              const std::shared_ptr<KVSAsyncContext>&         async_context,
                              const std::shared_ptr<KVSAsyncMatchContext>&    match_context);
    static void asyncWriteTask(const std::shared_ptr<KVSConnectorSharedState>& state,
                               const std::shared_ptr<KVCacheResource>&         resource,
                               const std::shared_ptr<Meta>&                    meta,
                               const std::shared_ptr<KVSAsyncContext>&         async_context);

    static std::vector<KVSBlockObjects> buildBlockObjects(const std::shared_ptr<KVSConnectorSharedState>& state,
                                                          const std::shared_ptr<KVCacheResource>&         resource,
                                                          const std::vector<CacheKeyType>&                keys);
    static std::string
    buildObjectKey(const std::shared_ptr<KVSConnectorSharedState>& state, CacheKeyType cache_key, int32_t group_id);
    static std::string buildDeploymentId(const std::shared_ptr<KVSConnectorSharedState>& state);

private:
    std::shared_ptr<KVSConnectorSharedState> state_;
    std::unique_ptr<autil::ThreadPool>       thread_pool_;
};

}  // namespace kvs
}  // namespace rtp_llm
