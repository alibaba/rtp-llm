#pragma once

#include <memory>
#include <string>

#include "kvcm_client/meta_client.h"
#include "kvcm_client/transfer_client.h"

namespace rtp_llm {
namespace kvcm {

class Subscriber;

class ClientFactory {
public:
    virtual ~ClientFactory() = default;

    virtual std::unique_ptr<kv_cache_manager::MetaClient>
    createMetaClient(const std::string& config, const kv_cache_manager::InitParams& init_params) const;
    virtual std::unique_ptr<kv_cache_manager::TransferClient>
    createTransferClient(const std::string& config, const kv_cache_manager::InitParams& init_params) const;
    // The SDK exposes external-memory initialization as a distinct API. A
    // custom factory that supports GDR must override this method so the
    // registrations cannot be silently discarded by a legacy override.
    virtual std::unique_ptr<kv_cache_manager::TransferClient>
    createTransferClientWithMemory(const std::string&                                config,
                                   const kv_cache_manager::InitParams&                init_params,
                                   const kv_cache_manager::ClientMemoryRegistrations& memory_registrations) const;
    virtual std::unique_ptr<Subscriber> createSubscriber(bool enable_vipserver) const;
};

}  // namespace kvcm
}  // namespace rtp_llm
