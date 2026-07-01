#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "kmonitor/client/MetricsReporter.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/KVCacheResource.h"
#include "rtp_llm/cpp/cache/connector/kvs_connector/KVSClient.h"
#include "rtp_llm/cpp/config/ConfigModules.h"

namespace rtp_llm {

class KVCacheAllocator;
class Meta;

namespace kvs {

struct KVSBlockObject {
    size_t                 block_index = 0;
    int32_t                group_id    = 0;
    std::string            object_key;
    std::vector<BlockInfo> iovs;
};

struct KVSBlockObjects {
    size_t                      block_index = 0;
    std::vector<KVSBlockObject> objects;
};

struct KVSMatchSession {
    size_t prev_reuse_blocks = 0;
    size_t matched_blocks    = 0;

    std::vector<KVSBlockObjects>  matched_block_objects;
    std::optional<KVSReadSession> read_session;
};

struct KVSCacheStoreConfig {
    CacheConfig       cache_config;
    KVCacheConfig     kv_cache_config;
    RuntimeConfig     runtime_config;
    ParallelismConfig parallelism_config;
    std::string       deployment_id;
};

class KVSCacheStore {
public:
    KVSCacheStore(KVSCacheStoreConfig                 config,
                  std::shared_ptr<KVCacheAllocator>  allocator,
                  std::shared_ptr<KVSClient>         client,
                  kmonitor::MetricsReporterPtr       metrics_reporter);

    bool init();

    std::optional<KVSMatchSession> match(const std::shared_ptr<KVCacheResource>& resource,
                                         const std::shared_ptr<Meta>&            meta);

    bool read(KVSMatchSession&                         session,
              const std::shared_ptr<KVCacheResource>& resource,
              int                                      start_read_block_index,
              int                                      read_block_num);

    bool write(const std::shared_ptr<KVCacheResource>& resource, const std::shared_ptr<Meta>& meta);

    void close(KVSMatchSession& session);

private:
    std::vector<KVSBlockObjects>
    buildBlockObjects(const std::shared_ptr<KVCacheResource>& resource, const std::vector<CacheKeyType>& keys) const;

    std::string buildObjectKey(CacheKeyType cache_key, int32_t group_id) const;
    std::string buildDeploymentId() const;

private:
    KVSCacheStoreConfig                config_;
    std::shared_ptr<KVCacheAllocator>  allocator_;
    std::shared_ptr<KVSClient>         client_;
    kmonitor::MetricsReporterPtr       metrics_reporter_;
};

}  // namespace kvs
}  // namespace rtp_llm
