#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

#include "absl/status/statusor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSet.h"

namespace rtp_llm {

uint32_t mockMemoryCacheCrc32c(const uint8_t* data, size_t bytes);

// Immutable, cache-lifetime HOST records; these blocks never enter the BlockTree.
class MockMemoryCacheSource {
public:
    static absl::StatusOr<std::shared_ptr<MockMemoryCacheSource>>
    create(const std::vector<GroupSetPtr>& group_sets, size_t source_blocks, uint64_t seed);

    ~MockMemoryCacheSource();
    MockMemoryCacheSource(const MockMemoryCacheSource&)            = delete;
    MockMemoryCacheSource& operator=(const MockMemoryCacheSource&) = delete;

    BlockIdxType blockFor(size_t group_set_id, size_t logical_block) const;
    size_t       sourceBlockCount(size_t group_set_id) const;

private:
    MockMemoryCacheSource() = default;
    struct Reservation {
        GroupSetPtr group;
        BlockIdList blocks;
        size_t      offset{0};
    };
    std::unordered_map<size_t, Reservation> reservations_;
};

}  // namespace rtp_llm
