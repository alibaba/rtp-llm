#include "SdkClient.h"

#include "kvcm_client/TransferClientCompat.h"

namespace rtp_llm::kvcm {
namespace sdk = kv_cache_manager;

SdkTransferClient::SdkTransferClient(std::unique_ptr<sdk::TransferClient> client): client_(std::move(client)) {}

std::unique_ptr<sdk::TransferClient>
SdkTransferClient::registerClient(const std::string& config,
                                  const sdk::InitParams& params,
                                  const sdk::ClientMemoryRegistrations* memory) {
    if (memory) {
        return sdk::TransferClient::Create(config, params, *memory);
    }
    return sdk::TransferClient::Create(config, params);
}

sdk::ClientErrorCode SdkTransferClient::load(const sdk::UriStrVec& uris,
                                            const sdk::BlockBuffers& buffers,
                                            int64_t deadline_ms,
                                            const std::shared_ptr<sdk::TransferTraceInfo>& trace) {
    return loadSdkKvCaches(*client_, uris, buffers, deadline_ms, trace);
}

std::pair<sdk::ClientErrorCode, sdk::UriStrVec>
SdkTransferClient::save(const sdk::UriStrVec& uris,
                        const sdk::BlockBuffers& buffers,
                        int64_t deadline_ms,
                        const std::shared_ptr<sdk::TransferTraceInfo>& trace) {
    return saveSdkKvCaches(*client_, uris, buffers, deadline_ms, trace);
}

std::unique_ptr<sdk::MetaClient> registerSdkMetadata(const std::string& config, const sdk::InitParams& params) {
    return sdk::MetaClient::Create(config, params);
}

std::pair<sdk::ClientErrorCode, sdk::WriteLocation>
startSdkWrite(sdk::MetaClient& client,
              const std::string& trace,
              const std::vector<int64_t>& keys,
              const std::vector<int64_t>& tokens,
              const std::vector<std::string>& spec_groups,
              int64_t timeout_seconds,
              int32_t replicas) {
    return client.StartWrite(trace, keys, tokens, spec_groups, timeout_seconds, replicas);
}

sdk::ClientErrorCode finishSdkWrite(sdk::MetaClient& client,
                                     const std::string& trace,
                                     const std::string& session,
                                     const sdk::BlockMask& mask,
                                     const sdk::Locations& locations) {
    return client.FinishWrite(trace, session, mask, locations);
}

std::pair<sdk::ClientErrorCode, sdk::BackendLocations>
querySdkLocations(sdk::MetaClient& client,
                  const std::string& trace,
                  const std::vector<int64_t>& keys,
                  const std::vector<int64_t>& tokens,
                  const sdk::BlockMask& mask,
                  const std::vector<std::string>& specs,
                  sdk::StorageType backend) {
    return client.GetCacheLocationsByBackend(
        trace, keys, tokens, mask, specs, backend, sdk::BackendSelectStrategy::LSS_WEIGHTED_RANDOM);
}

}  // namespace rtp_llm::kvcm
