#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace rtp_llm {
struct RemoteMockLimits {
    uint64_t total_bytes           = 768ULL << 20;
    uint64_t source_bytes          = 512ULL << 20;
    uint64_t write_slot_bytes      = 16ULL << 20;
    uint64_t read_staging_bytes    = 64ULL << 20;
    uint64_t write_staging_bytes   = 64ULL << 20;
    uint32_t write_slots           = 8;
    uint32_t max_rpc_bytes         = 16U << 20;
    uint32_t max_blocks_per_rpc    = 256;
    uint32_t max_read_operations   = 64;
    uint32_t max_ranks             = 128;
    uint32_t max_groups_per_rank   = 256;
    uint32_t max_sources_per_group = 65536;
    uint32_t max_members_per_group = 4096;
    uint32_t replay_window         = 4096;
    uint32_t max_delay_ms          = 60000;
};
struct RemoteMockSession {
    uint64_t server_epoch = 0;
    uint64_t session_id   = 0;
    uint64_t generation   = 0;
    bool     operator==(const RemoteMockSession& rhs) const;
};
struct RemoteMockLayout {
    uint32_t              group              = 0;
    uint32_t              pool               = 0;
    uint64_t              payload_bytes      = 0;
    uint64_t              storage_bytes      = 0;
    uint64_t              stride_bytes       = 0;
    bool                  crc_enabled        = false;
    uint32_t              crc_format_version = 1;
    uint64_t              crc_footer_offset  = 0;
    uint32_t              source_count       = 64;
    uint32_t              alignment          = 16;
    std::string           dtype              = "fp16";
    std::vector<uint64_t> member_offsets;
    uint32_t              tokens_per_block = 1;
    uint32_t              cp_world_size    = 1;
    uint32_t              cp_rank          = 0;
    uint64_t              attention_window = 0;  // Zero is FULL; positive is SWA.
};
struct RemoteMockManifest {
    RemoteMockSession             session;
    uint32_t                      rank           = 0;
    uint32_t                      expected_ranks = 1;
    uint64_t                      layout_hash    = 0;
    uint32_t                      read_delay_ms  = 0;
    uint32_t                      write_delay_ms = 0;
    std::vector<RemoteMockLayout> groups;
};
struct RemoteMockSourceDescriptor {
    uint32_t rank            = 0;
    uint32_t group           = 0;
    uint32_t pool            = 0;
    uint32_t source_count    = 0;
    uint64_t layout_hash     = 0;
    uint32_t record_checksum = 0;  // All initialized zero-payload records in this group have this CRC.
};
struct RemoteMockSessionInfo {
    RemoteMockSession session;
    uint32_t          expected_ranks   = 0;
    uint32_t          registered_ranks = 0;
    uint32_t          read_delay_ms    = 0;
    uint32_t          write_delay_ms   = 0;
    bool              ready            = false;
};
struct RemoteMockWriteOperation {
    RemoteMockSession session;
    uint64_t          layout_hash = 0;
    uint32_t          rank        = 0;
    uint32_t          slot        = 0;  // Global server slot, negotiated through limits; never reused while leased.
    uint64_t          sequence    = 0;  // Positive, per rank; bounded out-of-order replay window.
};

// Only State can construct leases. The source view cannot mutate the backing storage.
class RemoteMockSourceLease {
public:
    const uint8_t* data() const {
        return data_;
    }
    size_t size() const {
        return size_;
    }
    uint32_t recordChecksum() const {
        return checksum_;
    }

private:
    friend class RemoteMockState;
    std::shared_ptr<void> owner_;
    const uint8_t*        data_     = nullptr;
    size_t                size_     = 0;
    uint32_t              checksum_ = 0;
};
class RemoteMockReadLease {
public:
    ~RemoteMockReadLease();
    RemoteMockReadLease(const RemoteMockReadLease&)            = delete;
    RemoteMockReadLease& operator=(const RemoteMockReadLease&) = delete;

private:
    friend class RemoteMockState;
    RemoteMockReadLease() = default;
    std::function<void()> release_;
};
class RemoteMockWriteLease {
public:
    ~RemoteMockWriteLease();
    RemoteMockWriteLease(const RemoteMockWriteLease&)            = delete;
    RemoteMockWriteLease& operator=(const RemoteMockWriteLease&) = delete;
    uint8_t*              data() const {
        return data_;
    }
    size_t size() const {
        return size_;
    }

private:
    friend class RemoteMockState;
    RemoteMockWriteLease()      = default;
    uint8_t*              data_ = nullptr;
    size_t                size_ = 0;
    std::function<void()> release_;
};

// CPU core only: no transport, Torch, GroupSet, polling or sleeps. Service reserves
// one read/write lease per accepted RPC and retains it through actual I/O completion.
class RemoteMockState {
public:
    // Explicit epoch supports deterministic testing; zero generates a process-instance epoch.
    explicit RemoteMockState(RemoteMockLimits limits = {}, uint64_t server_epoch = 0);
    ~RemoteMockState();
    absl::StatusOr<RemoteMockSession>                            registerRank(const RemoteMockManifest& manifest);
    absl::Status                                                 ready(const RemoteMockSession& session) const;
    absl::Status                                                 close(const RemoteMockSession& session);
    absl::StatusOr<std::string>                                  sourceKey(const RemoteMockSession& session,
                                                                           uint32_t                 rank,
                                                                           uint32_t                 group,
                                                                           uint32_t                 index,
                                                                           uint64_t                 offset,
                                                                           uint64_t                 length) const;
    absl::StatusOr<std::shared_ptr<const RemoteMockSourceLease>> resolve(const std::string& key, uint64_t length) const;
    absl::StatusOr<std::shared_ptr<RemoteMockReadLease>>         reserveRead(const RemoteMockSession& session,
                                                                             uint64_t                 bytes,
                                                                             uint32_t                 blocks,
                                                                             uint64_t additional_metadata_bytes = 0);
    absl::StatusOr<std::shared_ptr<RemoteMockWriteLease>>        reserveWrite(const RemoteMockWriteOperation& operation,
                                                                              uint64_t                        bytes,
                                                                              uint32_t                        blocks = 1,
                                                                              uint64_t additional_metadata_bytes = 0);
    // For each target slice, before reverse blockRead. checksum is a slice CRC for writes;
    // the adapter computes it over the exact frozen Host bytes (including any footer).
    absl::Status
    validateWriteKey(const RemoteMockWriteOperation& operation, const std::string& key, uint64_t length) const;
    absl::StatusOr<RemoteMockSessionInfo>                   sessionInfo() const;
    absl::StatusOr<std::vector<RemoteMockSourceDescriptor>> describeRank(const RemoteMockSession& session,
                                                                         uint32_t                 rank) const;
    uint64_t                                                serverEpoch() const {
        return server_epoch_;
    }
    uint64_t lastGeneration() const;
    uint64_t allocatedBytes()
        const;  // Owned source/write backing plus conservative metadata reservation, including closed leases.
    const RemoteMockLimits& limits() const {
        return limits_;
    }

private:
    struct Data;
    absl::Status           checkReady(const RemoteMockSession& session) const;  // mutex_ held
    const RemoteMockLimits limits_;
    const uint64_t         server_epoch_;
    mutable std::mutex     mutex_;
    uint64_t               last_generation_ = 0;
    std::shared_ptr<Data>  active_;
    std::weak_ptr<Data>    retiring_;
};
}  // namespace rtp_llm
