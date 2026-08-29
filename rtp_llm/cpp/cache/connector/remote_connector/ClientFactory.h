#pragma once

#include <memory>
#include <string>

#include "kvcm_client/meta_client.h"
#include "kvcm_client/transfer_client.h"

namespace rtp_llm {
namespace remote_connector {

class Subscriber;

class ClientFactory {
public:
    virtual ~ClientFactory() = default;

    virtual std::unique_ptr<kv_cache_manager::MetaClient>
    createMetaClient(const std::string& config, const kv_cache_manager::InitParams& init_params) const;
    virtual std::unique_ptr<kv_cache_manager::TransferClient>
    createTransferClient(const std::string& config, const kv_cache_manager::InitParams& init_params) const;
    virtual std::unique_ptr<Subscriber> createSubscriber(bool enable_vipserver) const;
};

}  // namespace remote_connector
}  // namespace rtp_llm
