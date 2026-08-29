#include "rtp_llm/cpp/cache/connector/remote_connector/ClientFactory.h"

#include "rtp_llm/cpp/cache/connector/remote_connector/DirectSubscriber.h"
#include "rtp_llm/cpp/cache/connector/remote_connector/VIPServerSubscriber.h"

namespace rtp_llm {
namespace remote_connector {

std::unique_ptr<kv_cache_manager::MetaClient>
ClientFactory::createMetaClient(const std::string& config, const kv_cache_manager::InitParams& init_params) const {
    return kv_cache_manager::MetaClient::Create(config, init_params);
}

std::unique_ptr<kv_cache_manager::TransferClient>
ClientFactory::createTransferClient(const std::string& config, const kv_cache_manager::InitParams& init_params) const {
    return kv_cache_manager::TransferClient::Create(config, init_params);
}

std::unique_ptr<Subscriber> ClientFactory::createSubscriber(bool enable_vipserver) const {
    if (enable_vipserver) {
        return std::make_unique<VIPServerSubscriber>();
    }
    return std::make_unique<DirectSubscriber>();
}

}  // namespace remote_connector
}  // namespace rtp_llm
