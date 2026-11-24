#include "rtp_llm/cpp/cache/block_tree_cache/load/MockDiskCacheSource.h"

#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <linux/magic.h>
#include <sys/vfs.h>
#include <unordered_set>
#include <utility>

#include "rtp_llm/cpp/cache/block_tree_cache/load/MockMemoryCacheSource.h"
#include "rtp_llm/models_py/bindings/CrcBlockCopy.h"

namespace rtp_llm {
namespace {

constexpr size_t kDirectIoAlignment = 4096;

uint64_t mixSeed(uint64_t value) {
    // SplitMix64 matches the memory source's stable, library-independent rings.
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

}  // namespace

absl::StatusOr<std::shared_ptr<MockDiskCacheSource>>
MockDiskCacheSource::create(const std::vector<GroupSetPtr>& group_sets, size_t source_blocks, uint64_t seed) {
    if (group_sets.empty() || source_blocks == 0) {
        return absl::InvalidArgumentError("mock Disk source requires GroupSets and a positive block count");
    }
    auto source = std::shared_ptr<MockDiskCacheSource>(new MockDiskCacheSource());
    try {
        std::unordered_set<size_t>               group_ids;
        std::unordered_set<const DiskBlockPool*> disk_pools;
        for (const auto& group : group_sets) {
            if (!group || !group->diskPool() || !group_ids.insert(group->groupSetId()).second
                || !disk_pools.insert(group->diskPool().get()).second) {
                return absl::InvalidArgumentError("mock Disk source requires distinct GroupSets with Disk pools");
            }
            const auto   disk = group->diskPool();
            const size_t storage =
                group->crcEnabled() ? CrcCopyWorkspace::encodedBytes(group->payloadBytes()) : group->payloadBytes();
            if (disk->bufferedIO() || disk->strideBytes() % kDirectIoAlignment != 0) {
                return absl::InvalidArgumentError(
                    "mock Disk source requires aligned direct I/O; buffered I/O is unsupported");
            }
            struct statfs filesystem{};
            if (::statfs(disk->filePath().c_str(), &filesystem) != 0) {
                return absl::InternalError("cannot inspect mock Disk source filesystem: " + disk->filePath());
            }
            // Some RAM filesystems accept O_DIRECT while ignoring its storage
            // semantics. DiskMountGuard handles ownership, not filesystem type.
            if (filesystem.f_type == TMPFS_MAGIC || filesystem.f_type == RAMFS_MAGIC) {
                return absl::InvalidArgumentError("mock Disk source does not support tmpfs/ramfs backing");
            }
            if (group->payloadBytes() == 0 || disk->payloadBytes() != group->payloadBytes()
                || group->storageBytes() != storage || storage > disk->strideBytes()) {
                return absl::InvalidArgumentError("mock Disk source record does not fit its Disk pool layout");
            }
        }
        source->reservations_.reserve(group_sets.size());
        for (const auto& group : group_sets) {
            // Establish rollback ownership before allocation, including failures
            // in later GroupSets or after only part of this group's writes.
            auto& reservation = source->reservations_[group->groupSetId()];
            reservation.group = group;
            auto blocks       = group->allocateBlocks(source_blocks, Tier::DISK, BlockTreeRefType::MOCK_SOURCE);
            if (!blocks) {
                return absl::ResourceExhaustedError("insufficient Disk capacity for mock source GroupSet "
                                                    + std::to_string(group->groupSetId()));
            }
            reservation.blocks    = std::move(*blocks);
            reservation.offset    = mixSeed(seed ^ mixSeed(group->groupSetId())) % source_blocks;
            const auto disk       = group->diskPool();
            void*      allocation = nullptr;
            if (::posix_memalign(&allocation, kDirectIoAlignment, disk->strideBytes()) != 0) {
                return absl::ResourceExhaustedError("failed to allocate aligned mock Disk source scratch record");
            }
            // Ordinary aligned CPU memory is sufficient for synchronous pwrite;
            // do not consume Host/staging pool capacity or retain it in source.
            const std::unique_ptr<void, decltype(&std::free)> scratch(allocation, &std::free);
            std::memset(scratch.get(), 0, disk->strideBytes());
            if (group->crcEnabled()) {
                const uint32_t crc =
                    mockMemoryCacheCrc32c(static_cast<const uint8_t*>(scratch.get()), group->payloadBytes());
                const size_t footer = group->storageBytes() - sizeof(crc);
                std::memcpy(static_cast<uint8_t*>(scratch.get()) + footer, &crc, sizeof(crc));
            }
            for (const auto block : reservation.blocks) {
                if (disk->write(block, scratch.get(), disk->strideBytes()) != BlockIOStatus::OK) {
                    return absl::InternalError("mock Disk source write failed for GroupSet "
                                               + std::to_string(group->groupSetId()) + " block "
                                               + std::to_string(block));
                }
            }
        }
    } catch (const std::exception& error) {
        return absl::InternalError(std::string("mock Disk source initialization failed: ") + error.what());
    }
    return source;
}

MockDiskCacheSource::~MockDiskCacheSource() {
    for (const auto& entry : reservations_) {
        const auto& reservation = entry.second;
        reservation.group->releaseBlocks(Tier::DISK, reservation.blocks, BlockTreeRefType::MOCK_SOURCE);
    }
}

BlockIdxType MockDiskCacheSource::blockFor(size_t group_set_id, size_t logical_block) const {
    const auto&  reservation = reservations_.at(group_set_id);
    const size_t count       = reservation.blocks.size();
    const size_t index       = logical_block % count;
    const size_t until_wrap  = count - reservation.offset;
    return reservation.blocks[index >= until_wrap ? index - until_wrap : index + reservation.offset];
}

size_t MockDiskCacheSource::sourceBlockCount(size_t group_set_id) const {
    return reservations_.at(group_set_id).blocks.size();
}

}  // namespace rtp_llm
