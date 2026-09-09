#include "ClientFactory.h"

namespace rtp_llm {
namespace remote_connector {

std::unique_ptr<kv_cache_manager::MetaClient>
ClientFactory::CreateMetaClient(const std::string& config, const kv_cache_manager::InitParams& init_params) const {
    return kv_cache_manager::MetaClient::Create(config, init_params);
}

std::unique_ptr<kv_cache_manager::TransferClient>
ClientFactory::CreateTransferClient(const std::string& config, const kv_cache_manager::InitParams& init_params) const {
    return kv_cache_manager::TransferClient::Create(config, init_params);
}

std::unique_ptr<kv_cache_manager::TransferClient>
ClientFactory::CreateTransferClient(const std::string&                                config,
                                    const kv_cache_manager::InitParams&                init_params,
                                    const kv_cache_manager::SharedMemoryRegistration& shared_memory_registration) const {
    return kv_cache_manager::TransferClient::Create(config, init_params, shared_memory_registration);
}

}  // namespace remote_connector
}  // namespace rtp_llm
