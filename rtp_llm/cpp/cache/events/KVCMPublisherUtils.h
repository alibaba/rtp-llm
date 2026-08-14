#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace rtp_llm::detail {

// Keep these values aligned with kv_cache_manager.proto.meta.ErrorCode. This
// client intentionally does not depend on KVCM's generated protobuf target.
// New protocol codes must also be added to the single mapping table in
// KVCMPublisherUtils.cc. An unknown future code remains a parsed failure so
// response advisories are not discarded during a rolling upgrade.
enum class KVCMResponseCode : int32_t {
    UNRECOGNIZED              = -1,
    UNSPECIFIED               = 0,
    OK                        = 1,
    UNSUPPORTED               = 2,
    INTERNAL_ERROR            = 3,
    SERVICE_NOT_READY         = 4,
    INVALID_ARGUMENT          = 5,
    DUPLICATE_ENTITY          = 6,
    REACH_MAX_ENTITY_CAPACITY = 7,
    INSTANCE_NOT_EXIST        = 8,
    SERVER_NOT_LEADER         = 9,
    NODE_NOT_REGISTERED       = 10,
    SNAPSHOT_IN_PROGRESS      = 11,
    SNAPSHOT_RATE_LIMITED     = 13,
    SNAPSHOT_REQUIRED         = 14,
    IO_ERROR                  = 20,
    UNKNOWN_ERROR             = 100,
    ERROR_MAX                 = 65535,
};
static_assert(static_cast<int32_t>(KVCMResponseCode::UNRECOGNIZED) == -1);
static_assert(static_cast<int32_t>(KVCMResponseCode::UNSPECIFIED) == 0);
static_assert(static_cast<int32_t>(KVCMResponseCode::OK) == 1);
static_assert(static_cast<int32_t>(KVCMResponseCode::UNSUPPORTED) == 2);
static_assert(static_cast<int32_t>(KVCMResponseCode::INTERNAL_ERROR) == 3);
static_assert(static_cast<int32_t>(KVCMResponseCode::SERVICE_NOT_READY) == 4);
static_assert(static_cast<int32_t>(KVCMResponseCode::INVALID_ARGUMENT) == 5);
static_assert(static_cast<int32_t>(KVCMResponseCode::DUPLICATE_ENTITY) == 6);
static_assert(static_cast<int32_t>(KVCMResponseCode::REACH_MAX_ENTITY_CAPACITY) == 7);
static_assert(static_cast<int32_t>(KVCMResponseCode::INSTANCE_NOT_EXIST) == 8);
static_assert(static_cast<int32_t>(KVCMResponseCode::SERVER_NOT_LEADER) == 9);
static_assert(static_cast<int32_t>(KVCMResponseCode::NODE_NOT_REGISTERED) == 10);
static_assert(static_cast<int32_t>(KVCMResponseCode::SNAPSHOT_IN_PROGRESS) == 11);
static_assert(static_cast<int32_t>(KVCMResponseCode::SNAPSHOT_RATE_LIMITED) == 13);
static_assert(static_cast<int32_t>(KVCMResponseCode::SNAPSHOT_REQUIRED) == 14);
static_assert(static_cast<int32_t>(KVCMResponseCode::IO_ERROR) == 20);
static_assert(static_cast<int32_t>(KVCMResponseCode::UNKNOWN_ERROR) == 100);
static_assert(static_cast<int32_t>(KVCMResponseCode::ERROR_MAX) == 65535);

struct KVCMResponseInfo {
    bool                          parsed                = false;
    bool                          has_unrecognized_code = false;
    KVCMResponseCode              header_code           = KVCMResponseCode::UNRECOGNIZED;
    std::vector<KVCMResponseCode> item_results;
    std::string                   committed_snapshot_version;
    uint64_t                      retry_after_ms    = 0;
    bool                          snapshot_required = false;

    bool             ok() const noexcept;
    KVCMResponseCode firstFailure() const noexcept;
    bool             hasCode(KVCMResponseCode code) const noexcept;
    bool             hasPermanentFailure() const noexcept;
    bool             requiresRegistration() const noexcept;
    bool             requestsSnapshot() const noexcept;
};

std::string      normalizeKVCacheEventEndpoint(std::string endpoint);
bool             isValidKVCacheEventEndpoint(std::string_view endpoint) noexcept;
bool             isValidKVCacheEventHostIpPort(std::string_view host_ip_port) noexcept;
bool             isValidKVCacheEventIdentity(std::string_view identity) noexcept;
bool             isValidSnapshotVersionToken(std::string_view token) noexcept;
KVCMResponseInfo parseKVCMResponse(const std::string& response) noexcept;

}  // namespace rtp_llm::detail
