#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/cache/AsyncContext.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/PerRankBlockTransferEngine.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/DiskBlockIO.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/DiskBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/HostBlockPool.h"

namespace rtp_llm::block_transfer_engine_test {

GroupBase makeTestGroupBase(CacheGroupPolicy policy                = defaultCacheGroupPolicy(CacheGroupType::FULL),
                            std::vector<int> layer_ids             = {0},
                            size_t           kv_block_stride_bytes = 16,
                            size_t           kv_scale_stride_bytes = 0,
                            uint32_t         block_num             = 128,
                            size_t           seq_size_per_block    = 1);

std::shared_ptr<const CacheTopology> makeTestTopology(std::vector<GroupBase> groups);

GroupSetPtr makeTestGroupSet(size_t                               group_set_id,
                             std::shared_ptr<const CacheTopology> topology,
                             std::vector<size_t>                  group_ids,
                             std::vector<DeviceBlockPoolPtr>      device_pools,
                             std::shared_ptr<HostBlockPool>       host_pool = nullptr,
                             BlockTreeDiskBlockPoolPtr            disk_pool = nullptr);

DeviceBlockPoolPtr makeTestDevicePool(const std::vector<std::pair<size_t, size_t>>& layer_bytes,
                                      size_t                                        usable_count,
                                      const std::string&                            pool_name);

std::shared_ptr<HostBlockPool> makeHostPool(size_t payload_bytes, size_t usable_count, bool enable_pinned);

class TempDirGuard {
public:
    explicit TempDirGuard(const char* name);
    ~TempDirGuard();

    TempDirGuard(const TempDirGuard&)            = delete;
    TempDirGuard& operator=(const TempDirGuard&) = delete;

    std::string path;
};

std::shared_ptr<BlockTreeDiskBlockPool> makeDiskPool(size_t                       payload_bytes,
                                                     size_t                       usable_count,
                                                     const std::string&           work_dir,
                                                     std::unique_ptr<DiskBlockIO> io = nullptr,
                                                     const std::string& pool_name    = "per_rank_transfer_engine_disk",
                                                     bool               buffered_io  = true);

class StatusDiskBlockIO: public DiskBlockIO {
public:
    explicit StatusDiskBlockIO(DiskBlockIOStatus status);

    DiskBlockIOStatus openAndPreallocate(const std::string&, size_t, bool) override;
    DiskBlockIOStatus read(uint64_t, void*, size_t) override;
    DiskBlockIOStatus write(uint64_t, const void*, size_t) override;
    DiskBlockIOStatus read(const std::vector<DiskRead>&) override;
    DiskBlockIOStatus write(const std::vector<DiskWrite>&) override;
    void              close() override;
    std::string       debugString() const override;

    void setStatus(DiskBlockIOStatus status);

private:
    DiskBlockIOStatus status_;
};

// Deterministically enforces O_DIRECT alignment.
class DirectAlignmentDiskBlockIO: public DiskBlockIO {
public:
    static constexpr size_t kAlignment = 4096;

    DiskBlockIOStatus openAndPreallocate(const std::string&, size_t bytes, bool buffered_io) override;
    DiskBlockIOStatus read(uint64_t offset, void* dst, size_t bytes) override;
    DiskBlockIOStatus write(uint64_t offset, const void* src, size_t bytes) override;
    DiskBlockIOStatus read(const std::vector<DiskRead>& reads) override;
    DiskBlockIOStatus write(const std::vector<DiskWrite>& writes) override;
    void              close() override;
    std::string       debugString() const override;

    size_t lastReadBytes() const;
    size_t lastWriteBytes() const;
    bool   bufferedIo() const;

private:
    static bool aligned(uint64_t offset, const void* buffer, size_t bytes);

    std::vector<char> data_;
    size_t            last_read_bytes_{0};
    size_t            last_write_bytes_{0};
    bool              buffered_io_{true};
};

BlockIdxType poolMalloc(IBlockPool& pool);
void         releasePoolBlock(IBlockPool& pool, BlockIdxType block);

TransferDescriptor makeDescriptor(Tier                             source_tier,
                                  Tier                             target_tier,
                                  const std::vector<BlockIdxType>& device_blocks,
                                  BlockIdxType                     host_block   = NULL_BLOCK_IDX,
                                  BlockIdxType                     disk_block   = NULL_BLOCK_IDX,
                                  size_t                           group_set_id = 0);

bool submitSucceeded(const std::shared_ptr<PerRankBlockTransferEngine>& engine, const TransferDescriptor& desc);

void expectStatus(const std::shared_ptr<PerRankBlockTransferEngine>& engine,
                  const TransferDescriptor&                          desc,
                  TransferStatus                                     expected);

}  // namespace rtp_llm::block_transfer_engine_test
