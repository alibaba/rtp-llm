#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/transfer/PerRankBlockTransferEngine.h"
#include "rtp_llm/cpp/cache/block_tree_cache/host/DiskBlockIO.h"
#include "rtp_llm/cpp/cache/block_tree_cache/host/DiskBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/host/HostBlockPool.h"

namespace rtp_llm::block_transfer_engine_test {

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
                                                     std::unique_ptr<DiskBlockIO> io        = nullptr,
                                                     const std::string& pool_name    = "per_rank_transfer_engine_disk");

BlockIdxType poolMalloc(IBlockPool& pool);

Component makeSchemaComponent(int                        component_id,
                              int                        component_group_id,
                              const std::string&         tag,
                              const std::vector<size_t>& layer_bytes,
                              const std::vector<int>&    model_layer_ids = {});

std::shared_ptr<const std::vector<Component>> makeComponentRegistry(std::vector<Component> components);

TransferDescriptor makeDescriptor(Tier                             source_tier,
                                  Tier                             target_tier,
                                  const std::vector<BlockIdxType>& device_blocks,
                                  BlockIdxType                     host_block = NULL_BLOCK_IDX,
                                  BlockIdxType                     disk_block = NULL_BLOCK_IDX,
                                  int                              group_id   = 0);

void expectStatus(const std::shared_ptr<PerRankBlockTransferEngine>& engine,
                  const TransferDescriptor&                          desc,
                  TransferStatus                                     expected);

}  // namespace rtp_llm::block_transfer_engine_test
