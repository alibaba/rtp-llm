#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "kvcm_client/meta_client.h"
#include "kvcm_client/transfer_client.h"

namespace rtp_llm::kvcm {

// Owns the registered SDK client and executes payload I/O for both device
// buffers and fd-backed Host memory. The engine supplies registration spans.
class SdkTransferClient {
public:
    SdkTransferClient() = default;
    explicit SdkTransferClient(std::unique_ptr<kv_cache_manager::TransferClient> client);

    static std::unique_ptr<kv_cache_manager::TransferClient>
    registerClient(const std::string& config,
                   const kv_cache_manager::InitParams& params,
                   const kv_cache_manager::ClientMemoryRegistrations* memory = nullptr);

    explicit operator bool() const { return client_ != nullptr; }

    kv_cache_manager::ClientErrorCode
    load(const kv_cache_manager::UriStrVec& uris,
         const kv_cache_manager::BlockBuffers& buffers,
         int64_t deadline_ms = 0,
         const std::shared_ptr<kv_cache_manager::TransferTraceInfo>& trace = {});

    std::pair<kv_cache_manager::ClientErrorCode, kv_cache_manager::UriStrVec>
    save(const kv_cache_manager::UriStrVec& uris,
         const kv_cache_manager::BlockBuffers& buffers,
         int64_t deadline_ms = 0,
         const std::shared_ptr<kv_cache_manager::TransferTraceInfo>& trace = {});

private:
    std::unique_ptr<kv_cache_manager::TransferClient> client_;
};

std::unique_ptr<kv_cache_manager::MetaClient>
registerSdkMetadata(const std::string& config, const kv_cache_manager::InitParams& params);

std::pair<kv_cache_manager::ClientErrorCode, kv_cache_manager::WriteLocation>
startSdkWrite(kv_cache_manager::MetaClient& client,
              const std::string& trace,
              const std::vector<int64_t>& keys,
              const std::vector<int64_t>& tokens,
              const std::vector<std::string>& spec_groups,
              int64_t timeout_seconds,
              int32_t replicas);

kv_cache_manager::ClientErrorCode
finishSdkWrite(kv_cache_manager::MetaClient& client,
               const std::string& trace,
               const std::string& session,
               const kv_cache_manager::BlockMask& mask,
               const kv_cache_manager::Locations& locations);

std::pair<kv_cache_manager::ClientErrorCode, kv_cache_manager::BackendLocations>
querySdkLocations(kv_cache_manager::MetaClient& client,
                  const std::string& trace,
                  const std::vector<int64_t>& keys,
                  const std::vector<int64_t>& tokens,
                  const kv_cache_manager::BlockMask& mask,
                  const std::vector<std::string>& specs,
                  kv_cache_manager::StorageType backend);

}  // namespace rtp_llm::kvcm
