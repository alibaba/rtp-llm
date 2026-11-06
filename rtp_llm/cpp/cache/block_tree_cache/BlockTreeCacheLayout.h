#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/CacheConfig.h"

namespace rtp_llm {

// Layout of one packed HOST/DISK record for one GroupSet block on this rank.
// The payload combines one block from each member pool, including all of that
// member's layer buffers (KV and any scales). Factory computes this plan once
// and shares it with GroupSet, pool creation and host/disk capacity budgeting.
// All sizes below are bytes per record, not the size of the whole pool.
//
// payload_bytes: Packed bytes before CRC; 0 derives logical sizes from topology, otherwise uses exact physical sizes.
// storage_bytes: Record bytes (payload or alignUp(payload_bytes + 4, 16) with CRC); 0 uses the resolved payload size.
// pool_stride: Distance between HOST/DISK slots, aligned to 4096 for direct disk I/O or 16 otherwise.
// crc_enabled: Whether transfers encode and verify a CRC32C checksum for each record.
//
// Example: member A contributes 2048 bytes and member B contributes 2048 bytes.
// Packing their buffers produces a 4096-byte payload (diagrams not to scale).
//
// Field order: {payload_bytes, storage_bytes, pool_stride, crc_enabled}.
// CRC off: BackingLayout{4096, 4096, 4096, false}
//   0                          2048                       4096
//   +--------------------------+--------------------------+ next record
//   |     member A: 2048 B     |     member B: 2048 B     |
//   +--------------------------+--------------------------+
//   payload_bytes = storage_bytes = pool_stride = 4096
//
// CRC on, HOST only or buffered disk I/O: BackingLayout{4096, 4112, 4112, true}
//   0                     4096          4108        4112
//   +---------------------+-------------+-----------+
//   |   payload: 4096 B   |  pad: 12 B  |  CRC: 4 B | next record
//   +---------------------+-------------+-----------+
//   |<-- payload_bytes -->|
//   |<-------- storage_bytes = pool_stride -------->|
//
// CRC on, direct disk I/O (both HOST/DISK slots): BackingLayout{4096, 4112, 8192, true}
//   0                     4096          4108        4112                    8192
//   +---------------------+-------------+-----------+-----------------------+
//   |   payload: 4096 B   |  pad: 12 B  |  CRC: 4 B |  pool padding: 4080 B | next record
//   +---------------------+-------------+-----------+-----------------------+
//   |<-- payload_bytes -->|
//   |<--------------- storage_bytes --------------->|
//   |<---------------------------- pool_stride ---------------------------->|
// CRC32C covers only the payload. The 4-byte checksum is stored at the END of
// the 16-byte-aligned encoded record; pool padding lies outside that record.
struct BackingLayout {
    size_t payload_bytes{0};
    size_t storage_bytes{0};
    size_t pool_stride{0};
    bool   crc_enabled{false};
};

bool          buildBlockTreeGroupMembers(const CacheConfig& config, std::vector<std::vector<std::string>>& members);
BackingLayout computeBlockTreeBackingLayout(const CacheConfig&              config,
                                            const std::vector<std::string>& members,
                                            bool                            enable_crc,
                                            size_t                          alignment);

// Shared by pre-allocation budgeting and the selected transfer implementation.
size_t asyncTransferBatchLimit(size_t configured_batch, size_t staging_blocks, bool has_device_disk_route);
size_t crcTransferWorkspaceCount(size_t workers, size_t slots_per_loop, bool async);
// Explicit GPU allocations for this rank's CRC services. No pools or CUDA
// resources are created, and no capacity-dependent KV allocations are counted.
size_t blockTreeTransferDeviceMemory(const CacheConfig& config, const KVCacheConfig& options);

}  // namespace rtp_llm
