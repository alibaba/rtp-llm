#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/IBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/host/DiskBlockIO.h"
#include "rtp_llm/cpp/cache/block_tree_cache/host/DiskMountGuard.h"
#include "rtp_llm/cpp/cache/CacheBlockKind.h"

namespace rtp_llm {

enum class BlockIOStatus {
    OK,
    INVALID_BLOCK,
    INVALID_SIZE,
    ALIGNMENT_ERROR,
    IO_ERROR,
    PARTIAL_FAILURE,
};

struct DiskBlockPoolConfig: public BlockPoolConfigBase {
    std::string work_dir;
    int64_t     local_rank{0};
    int64_t     world_rank{0};
    size_t      disk_size_bytes{0};
    size_t      payload_bytes{0};
    size_t      stride_bytes{0};
    bool        buffered_io{true};
    std::shared_ptr<DiskMountGuard> mount_guard;
    CacheBlockKind                  pool_kind{CacheBlockKind::COMPLETE};
};

// DiskBlockPool backs every block with a fixed-stride slice of a single preallocated
// disk file, driving reads/writes through a DiskBlockIO (a PosixDiskBlockIO by
// default, or an injected implementation for tests). All lifecycle behavior
// (malloc/free/incRef/decRef/metrics) is inherited unchanged from IBlockPool; this
// class only owns the disk backing file and maps block indices to file offsets via
// blockOffset(). Unlike the incRef/decRef refcount, read()/write() only check that
// the physical block index is valid, so the same rank-0 index can be used on workers.
class DiskBlockPool: public IBlockPool {
public:
    explicit DiskBlockPool(std::shared_ptr<const DiskBlockPoolConfig> config,
                           std::unique_ptr<DiskBlockIO>               io = nullptr);
    ~DiskBlockPool() override;

    // Validates the config's payload/stride/work_dir invariants, builds the backing
    // file path from work_dir/world_rank/local_rank, preallocates it to
    // physical_block_count * stride_bytes (this includes block 0's reserved slot),
    // and marks the pool initialized. Always returns true; invariant violations are
    // enforced via RTP_LLM_CHECK.
    bool init();

    // Single-block read/write. Only validBlock(block) is checked (not allocation/refCount);
    // block 0 and any out-of-range block return INVALID_BLOCK. bytes greater
    // than strideBytes() returns INVALID_SIZE.
    BlockIOStatus read(BlockIdxType block, void* dst, size_t bytes);
    BlockIOStatus write(BlockIdxType block, const void* src, size_t bytes);

    // Batch read/write. Every block is validated with validBlock() and bytes_per_block is
    // validated against strideBytes() before any I/O is issued; on success the
    // underlying DiskBlockIO batch call is driven in blocks[] order.
    BlockIOStatus read(const BlockIdList& blocks, const std::vector<void*>& dsts, size_t bytes_per_block);
    BlockIOStatus write(const BlockIdList& blocks, const std::vector<const void*>& srcs, size_t bytes_per_block);

    size_t             payloadBytes() const;
    size_t             strideBytes() const;
    size_t             blockSizeBytes() const override;
    size_t             readBytes() const;
    size_t             writeBytes() const;
    uint64_t           blockOffset(BlockIdxType block) const;
    const std::string& filePath() const;

    std::string debugString() const override;

private:
    // Computes/validates physical_block_count from disk_size_bytes/stride_bytes
    // BEFORE the IBlockPool base constructor runs (it requires physical_block_count
    // > 1 and seeds free blocks immediately from it). Returns a copy of config with
    // physical_block_count normalized to the computed value.
    static std::shared_ptr<const DiskBlockPoolConfig>
    normalizeConfig(const std::shared_ptr<const DiskBlockPoolConfig>& config);

    const DiskBlockPoolConfig& config() const;

    static BlockIOStatus mapStatus(DiskBlockIOStatus status);

    std::unique_ptr<DiskBlockIO> io_;
    std::string                  file_path_;
    std::atomic<size_t>          read_bytes_{0};
    std::atomic<size_t>          write_bytes_{0};
};

}  // namespace rtp_llm
