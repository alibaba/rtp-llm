#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

#include "absl/status/statusor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSet.h"

namespace rtp_llm {

// Immutable DISK records held outside the BlockTree with MOCK_SOURCE refs.
// Creation writes each rank's local disk file before returning a source. It
// requires direct I/O and rejects known tmpfs/ramfs backing. Initialization uses
// one temporary aligned record per GroupSet, with no Host-pool reservations or
// retained record buffers. Readers must retain a
// shared source until their I/O completes; destruction releases its disk refs.
// The bounded ring can hit storage-controller caches even with O_DIRECT. Writes
// complete through DiskBlockPool, without implying fdatasync durability.
class MockDiskCacheSource {
public:
    static absl::StatusOr<std::shared_ptr<MockDiskCacheSource>>
    create(const std::vector<GroupSetPtr>& group_sets, size_t source_blocks, uint64_t seed);

    ~MockDiskCacheSource();
    MockDiskCacheSource(const MockDiskCacheSource&)            = delete;
    MockDiskCacheSource& operator=(const MockDiskCacheSource&) = delete;

    // Selection wraps without overflow, including SIZE_MAX. Unknown group IDs
    // throw std::out_of_range, matching MockMemoryCacheSource.
    BlockIdxType blockFor(size_t group_set_id, size_t logical_block) const;
    size_t       sourceBlockCount(size_t group_set_id) const;

private:
    MockDiskCacheSource() = default;
    struct Reservation {
        GroupSetPtr group;
        BlockIdList blocks;
        size_t      offset{0};
    };
    std::unordered_map<size_t, Reservation> reservations_;
};

}  // namespace rtp_llm
