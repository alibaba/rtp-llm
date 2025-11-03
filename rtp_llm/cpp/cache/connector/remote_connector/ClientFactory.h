#pragma once

#include <memory>
#include "kvcm_client/meta_client.h"
#include "kvcm_client/transfer_client.h"

namespace rtp_llm {
namespace remote_connector {

class ClientFactory {
public:
    virtual ~ClientFactory() = default;
    // for mock
    virtual std::unique_ptr<kv_cache_manager::MetaClient>
    CreateMetaClient(const std::string& config, const kv_cache_manager::InitParams& init_params) const;
    virtual std::unique_ptr<kv_cache_manager::TransferClient>
    CreateTransferClient(const std::string& config, const kv_cache_manager::InitParams& init_params) const;
};

}  // namespace remote_connector
}  // namespace rtp_llm