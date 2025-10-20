#pragma once

#include <vector>
#include <map>
#include <memory>
#include <string>
#include <shared_mutex>
#include "kvcm_client/meta_client.h"
#include "kvcm_client/transfer_client.h"
#include "KVCMClientWrapperConfig.h"

namespace rtp_llm {
class KVCMSubscriber;
class KVCMClientWrapper {
public:
    ~KVCMClientWrapper();
    bool init(const std::map<std::string, std::string>& config_str_map,
              const kv_cache_manager::InitParams&       init_params);
    // for meta client
    std::pair<bool, kv_cache_manager::LocationsMap> match(const std::string&                      unique_id,
                                                          const std::string&                      trace_id,
                                                          kv_cache_manager::QueryType             query_type,
                                                          const std::vector<int64_t>&             keys,
                                                          const kv_cache_manager::BlockMask&      block_mask,
                                                          const kv_cache_manager::ForwardContext& forward_context);

    std::pair<bool, kv_cache_manager::WriteLocation>
    getWriteLocation(const std::string&                      unique_id,
                     const std::string&                      trace_id,
                     const std::vector<int64_t>&             keys,
                     const std::vector<std::string>&         location_spec_names,
                     int64_t                                 write_timeout_seconds,
                     const kv_cache_manager::ForwardContext& forward_context);

    bool finishWrite(const std::string&                    unique_id,
                     const std::string&                    trace_id,
                     const std::string&                    write_session_id,
                     const kv_cache_manager::BlockMask&    block_mask,
                     const kv_cache_manager::LocationsMap& locations_map);

    // for transfer client
    bool loadKvCaches(const kv_cache_manager::Locations& locations, kv_cache_manager::BlockBuffers& block_buffers);

    std::pair<bool, kv_cache_manager::Locations> saveKvCaches(const kv_cache_manager::Locations&    locations,
                                                              const kv_cache_manager::BlockBuffers& block_buffers);

private:
    bool initMetaClient(const std::string& unique_id, const std::string& config_str);
    bool reinit(const std::string& unique_id);
    bool tryReinit(const std::string& unique_id);

private:
    kv_cache_manager::InitParams                                         init_params_;
    std::shared_mutex                                                    vipserver_mutex_;
    std::map<std::string, std::shared_ptr<KVCMClientWrapperConfig>>      config_map_;
    std::map<std::string, std::unique_ptr<kv_cache_manager::MetaClient>> meta_client_map_;
    static std::unique_ptr<kv_cache_manager::TransferClient>             transfer_client_;
    static std::unique_ptr<KVCMSubscriber>                               subscriber_;
    std::vector<std::string>                                             address_snapshot_;
};
}  // namespace rtp_llm