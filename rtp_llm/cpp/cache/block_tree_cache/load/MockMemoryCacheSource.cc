#include "rtp_llm/cpp/cache/block_tree_cache/load/MockMemoryCacheSource.h"

#include <cstring>
#include <exception>
#include <string>
#include <unordered_set>
#include <utility>

#include "rtp_llm/models_py/bindings/CrcBlockCopy.h"
#include "rtp_llm/cpp/utils/Crc32c.h"

namespace rtp_llm {
namespace {

uint64_t mixSeed(uint64_t value) {
    // SplitMix64 gives stable offsets independently of library hash implementations.
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

}  // namespace

uint32_t mockMemoryCacheCrc32c(const uint8_t* data, size_t bytes) {
    return cpuCrc32c(data, bytes);
}

absl::StatusOr<std::shared_ptr<MockMemoryCacheSource>>
MockMemoryCacheSource::create(const std::vector<GroupSetPtr>& group_sets, size_t source_blocks, uint64_t seed) {
    if (group_sets.empty() || source_blocks == 0) {
        return absl::InvalidArgumentError("mock Host source requires GroupSets and a positive block count");
    }
    auto source = std::shared_ptr<MockMemoryCacheSource>(new MockMemoryCacheSource());
    try {
        std::unordered_set<size_t>               group_ids;
        std::unordered_set<const HostBlockPool*> host_pools;
        for (const auto& group : group_sets) {
            if (!group || !group->hostPool() || !group_ids.insert(group->groupSetId()).second
                || !host_pools.insert(group->hostPool().get()).second) {
                return absl::InvalidArgumentError("mock Host source requires distinct GroupSets with Host pools");
            }
            const auto   host = group->hostPool();
            const size_t storage =
                group->crcEnabled() ? CrcCopyWorkspace::encodedBytes(group->payloadBytes()) : group->payloadBytes();
            if (group->payloadBytes() == 0 || host->payloadBytes() != group->payloadBytes()
                || group->storageBytes() != storage || storage > host->strideBytes()) {
                return absl::InvalidArgumentError("mock Host source record does not fit its Host pool layout");
            }
        }
        source->reservations_.reserve(group_sets.size());
        for (const auto& group : group_sets) {
            // Install ownership before allocating so every subsequent failure is
            // rolled back by the source destructor, including later GroupSets.
            auto& reservation = source->reservations_[group->groupSetId()];
            reservation.group = group;
            auto blocks       = group->allocateBlocks(source_blocks, Tier::HOST, BlockTreeRefType::MOCK_SOURCE);
            if (!blocks) {
                return absl::ResourceExhaustedError("insufficient Host capacity for mock source GroupSet "
                                                    + std::to_string(group->groupSetId()));
            }
            reservation.blocks = std::move(*blocks);
            reservation.offset = mixSeed(seed ^ mixSeed(group->groupSetId())) % source_blocks;
            const auto host    = group->hostPool();
            for (const auto block : reservation.blocks) {
                std::memset(host->blockBuffer(block).addr, 0, host->strideBytes());
            }
            if (group->crcEnabled()) {
                const uint32_t crc = mockMemoryCacheCrc32c(
                    static_cast<const uint8_t*>(host->blockBuffer(reservation.blocks.front()).addr),
                    group->payloadBytes());
                const size_t footer = CrcCopyWorkspace::encodedBytes(group->payloadBytes()) - sizeof(crc);
                for (const auto block : reservation.blocks) {
                    auto* record = static_cast<uint8_t*>(host->blockBuffer(block).addr);
                    std::memcpy(record + footer, &crc, sizeof(crc));
                }
            }
        }
    } catch (const std::exception& error) {
        return absl::InternalError(std::string("mock Host source initialization failed: ") + error.what());
    }
    return source;
}

MockMemoryCacheSource::~MockMemoryCacheSource() {
    for (const auto& entry : reservations_) {
        const auto& reservation = entry.second;
        reservation.group->releaseBlocks(Tier::HOST, reservation.blocks, BlockTreeRefType::MOCK_SOURCE);
    }
}

BlockIdxType MockMemoryCacheSource::blockFor(size_t group_set_id, size_t logical_block) const {
    const auto&  reservation = reservations_.at(group_set_id);
    const size_t count       = reservation.blocks.size();
    const size_t index       = logical_block % count;
    // Add the offset modulo count without overflowing a large logical index.
    const size_t until_wrap = count - reservation.offset;
    return reservation.blocks[index >= until_wrap ? index - until_wrap : index + reservation.offset];
}

size_t MockMemoryCacheSource::sourceBlockCount(size_t group_set_id) const {
    return reservations_.at(group_set_id).blocks.size();
}

}  // namespace rtp_llm
