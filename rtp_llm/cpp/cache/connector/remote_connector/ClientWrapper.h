#pragma once

#include <vector>
#include <map>
#include <memory>
#include <string>
#include <shared_mutex>
#include <mutex>
#include <atomic>
#include "ClientFactory.h"
#include "RemoteConnectorConfig.h"

namespace rtp_llm {
namespace remote_connector {
class Subscriber;

class ClientWrapper: public std::enable_shared_from_this<ClientWrapper> {
public:
    using ConfigMap = std::map<std::string, RemoteConnectorConfigPtr>;
    virtual ~ClientWrapper();
    bool init(const ConfigMap& config_str_map, const kv_cache_manager::InitParams& init_params);
    // for meta client
    std::pair<bool, kv_cache_manager::Locations> match(const std::string&                      unique_id,
                                                       const std::string&                      trace_id,
                                                       kv_cache_manager::QueryType             query_type,
                                                       const std::vector<int64_t>&             keys,
                                                       const kv_cache_manager::BlockMask&      block_mask,
                                                       const kv_cache_manager::ForwardContext& forward_context);

    std::pair<bool, kv_cache_manager::WriteLocation>
    getWriteLocation(const std::string&              unique_id,
                     const std::string&              trace_id,
                     const std::vector<int64_t>&     keys,
                     const std::vector<int64_t>&     tokens,
                     const std::vector<std::string>& location_spec_group_names,
                     int64_t                         write_timeout_seconds);

    bool finishWrite(const std::string&                 unique_id,
                     const std::string&                 trace_id,
                     const std::string&                 write_session_id,
                     const kv_cache_manager::BlockMask& block_mask,
                     const kv_cache_manager::Locations& locations);

    // for transfer client
    bool loadKvCaches(const kv_cache_manager::UriStrVec&                          uri_str_vec,
                      kv_cache_manager::BlockBuffers&                             block_buffers,
                      const std::shared_ptr<kv_cache_manager::TransferTraceInfo>& trace_info = nullptr);

    std::pair<bool, kv_cache_manager::UriStrVec>
    saveKvCaches(const kv_cache_manager::UriStrVec&                          uri_str_vec,
                 const kv_cache_manager::BlockBuffers&                       block_buffers,
                 const std::shared_ptr<kv_cache_manager::TransferTraceInfo>& trace_info = nullptr);

private:
    using MetaClientMap = std::map<std::string, std::shared_ptr<kv_cache_manager::MetaClient>>;
    bool initMetaClient(const std::string& unique_id, RemoteConnectorConfigPtr config);
    // reinit if address_snapshot_ change
    bool
    reinit(const std::string& unique_id, ConfigMap::iterator& config_iter, MetaClientMap::iterator& meta_client_iter);
    bool tryReinit(const std::string& unique_id);
    bool checkError(kv_cache_manager::ClientErrorCode ec);
    void reinitAllMetaClients();

    kv_cache_manager::InitParams init_params_;
    // keys of config_map_/meta_client_map_ will not change after init
    ConfigMap                config_map_;
    MetaClientMap            meta_client_map_;
    std::vector<std::string> address_snapshot_;
    std::shared_mutex        reinit_mutex_;

    // for re-registration
    std::atomic_bool  rr_other_working_ = false;
    std::shared_mutex rr_mutex_;

    // when slaver reaches 3, need reinitAllMetaClients
    std::atomic<int> grpc_error_count_{0};

    static std::unique_ptr<ClientFactory>                    client_factory_;
    static std::unique_ptr<kv_cache_manager::TransferClient> transfer_client_;
    static std::unique_ptr<remote_connector::Subscriber>     subscriber_;
};

}  // namespace remote_connector
}  // namespace rtp_llm