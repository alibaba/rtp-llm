#pragma once
#include "rtp_llm/cpp/disaggregate/cache_store/remote_mock/RemoteMockState.h"

namespace rtp_llm {
// Canonical decimal codec rm1:{r|w}:epoch:session:generation:layout:rank:group:
// pool:index:operation:record_index:offset:length:checksum. Reads: index=source index, operation=0,
// checksum=CRC32c over the whole storage record. Writes: index=slot, operation=sequence,
// record_index=unique frozen record identity within this operation (including sequence ordinal),
// checksum=CRC32c over this exact range, so each sliced transfer can be checked before ACK.
struct RemoteMockKey {
    bool              write = false;
    RemoteMockSession session;
    uint64_t          layout_hash  = 0;
    uint32_t          rank         = 0;
    uint32_t          group        = 0;
    uint32_t          pool         = 0;
    uint32_t          index        = 0;
    uint64_t          operation    = 0;
    uint64_t          record_index = 0;  // Must be zero for read keys.
    uint64_t          offset       = 0;
    uint64_t          length       = 0;
    uint32_t          checksum     = 0;
};
constexpr size_t              kRemoteMockMaxKeyBytes = 384;
std::string                   encodeRemoteMockKey(const RemoteMockKey& key);
absl::StatusOr<RemoteMockKey> decodeRemoteMockKey(const std::string& key);
uint32_t                      remoteMockCrc32c(const uint8_t* bytes, size_t length);
absl::Status                  validateRemoteMockLayout(const RemoteMockLayout& layout, const RemoteMockLimits& limits);
absl::Status
initializeRemoteMockRecord(const RemoteMockLayout& layout, uint8_t* bytes, size_t capacity, uint8_t payload_fill = 0);
absl::Status validateRemoteMockRecord(const RemoteMockLayout& layout, const uint8_t* bytes, size_t length);
// Base payload/envelope charge, plus caller-measured/conservative simultaneous metadata.
// The default preserves core-only callers; network services must add their owned representations.
absl::StatusOr<uint64_t> remoteMockStagingCharge(uint64_t                bytes,
                                                 uint32_t                blocks,
                                                 const RemoteMockLimits& limits,
                                                 uint64_t                additional_metadata_bytes = 0);
}  // namespace rtp_llm
