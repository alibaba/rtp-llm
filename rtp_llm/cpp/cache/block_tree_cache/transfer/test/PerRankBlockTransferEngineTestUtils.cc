#include "rtp_llm/cpp/cache/block_tree_cache/transfer/test/PerRankBlockTransferEngineTestUtils.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

#include <dirent.h>
#include <unistd.h>

#include "rtp_llm/cpp/cache/MHAKVCacheSpec.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/FullGroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/LinearGroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/SWAGroupSet.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm::block_transfer_engine_test {
namespace {

std::string makeTempDir(const char* name) {
    std::string       path = std::string("/tmp/") + name + "_XXXXXX";
    std::vector<char> writable(path.begin(), path.end());
    writable.push_back('\0');
    char* result = ::mkdtemp(writable.data());
    RTP_LLM_CHECK(result != nullptr);
    return result;
}

void removeTempDir(const std::string& path) {
    DIR* dir = ::opendir(path.c_str());
    if (dir != nullptr) {
        while (auto* entry = ::readdir(dir)) {
            const std::string name = entry->d_name;
            if (name != "." && name != "..") {
                std::remove((path + "/" + name).c_str());
            }
        }
        ::closedir(dir);
    }
    ::rmdir(path.c_str());
}

}  // namespace

GroupBase makeTestGroupBase(CacheGroupPolicy policy,
                            std::vector<int> layer_ids,
                            size_t           kv_block_stride_bytes,
                            size_t           kv_scale_stride_bytes,
                            uint32_t         block_num,
                            size_t           seq_size_per_block) {
    GroupBase group;
    group.spec                      = std::make_shared<MHAKVCacheSpec>();
    group.policy                    = policy;
    group.layer_ids                 = std::move(layer_ids);
    group.block_num                 = block_num;
    group.local_kv_head_num         = 1;
    group.seq_size_per_block        = seq_size_per_block;
    group.kernel_seq_size_per_block = seq_size_per_block;
    group.kv_block_stride_bytes     = kv_block_stride_bytes;
    group.kv_scale_stride_bytes     = kv_scale_stride_bytes;
    return group;
}

std::shared_ptr<const CacheTopology> makeTestTopology(std::vector<GroupBase> groups) {
    RTP_LLM_CHECK(!groups.empty());
    for (const auto& group : groups) {
        RTP_LLM_CHECK(group.spec != nullptr);
        RTP_LLM_CHECK(!group.layer_ids.empty());
        for (int layer_id : group.layer_ids) {
            RTP_LLM_CHECK(layer_id >= 0);
        }
    }
    return test::makeIndexedTestTopology(std::move(groups));
}

GroupSetPtr makeTestGroupSet(size_t                               group_set_id,
                             std::shared_ptr<const CacheTopology> topology,
                             std::vector<size_t>                  group_ids,
                             std::vector<DeviceBlockPoolPtr>      device_pools) {
    RTP_LLM_CHECK(topology != nullptr);
    RTP_LLM_CHECK(!group_ids.empty());
    const auto& first = topology->groupById(group_ids.front());

    GroupSetPtr group_set;
    switch (first.policy.group_type) {
        case CacheGroupType::FULL:
            group_set = std::make_shared<FullGroupSet>();
            break;
        case CacheGroupType::SWA:
            group_set = std::make_shared<SWAGroupSet>(static_cast<size_t>(first.policy.sliding_window_size),
                                                      first.seq_size_per_block);
            break;
        case CacheGroupType::LINEAR:
            group_set = std::make_shared<LinearGroupSet>();
            break;
    }
    RTP_LLM_CHECK(group_set != nullptr);
    group_set->initialize(group_set_id, std::move(topology), std::move(group_ids), std::move(device_pools));
    return group_set;
}

DeviceBlockPoolPtr makeTestDevicePool(const std::vector<std::pair<size_t, size_t>>& layer_bytes,
                                      size_t                                        usable_count,
                                      const std::string&                            pool_name) {
    RTP_LLM_CHECK(!layer_bytes.empty());
    const size_t physical_block_count = usable_count + 1;
    auto         config               = std::make_shared<DeviceBlockPoolConfig>();
    config->pool_type                 = BlockPoolType::DEVICE;
    config->pool_name                 = pool_name;
    config->physical_block_count      = physical_block_count;
    config->use_cuda_malloc_backing   = false;

    size_t offset = 0;
    for (const auto& [kv_bytes, scale_bytes] : layer_bytes) {
        MemoryLayoutConfig layout;
        layout.layer_num                = 1;
        layout.block_num                = static_cast<uint32_t>(physical_block_count);
        layout.dtype                    = TYPE_INT8;
        layout.kv_cache_offset_bytes    = offset;
        layout.kv_block_stride_bytes    = kv_bytes;
        layout.kv_block_pool_size_bytes = physical_block_count * kv_bytes;
        layout.block_stride_bytes       = kv_bytes + scale_bytes;
        layout.total_size_bytes         = layout.kv_block_pool_size_bytes;
        offset += layout.kv_block_pool_size_bytes;
        if (scale_bytes > 0) {
            layout.enable_kv_scale          = true;
            layout.kv_scale_offset_bytes    = offset;
            layout.kv_scale_stride_bytes    = scale_bytes;
            layout.kv_scale_pool_size_bytes = physical_block_count * scale_bytes;
            layout.total_size_bytes += layout.kv_scale_pool_size_bytes;
            offset += layout.kv_scale_pool_size_bytes;
        }
        layout.local_head_num_kv          = 1;
        layout.seq_size_per_block         = 1;
        layout.kernel_blocks_per_kv_block = 1;
        config->memory_layouts.push_back(layout);
    }
    config->total_size_bytes = offset;

    auto pool = std::make_shared<DeviceBlockPool>(std::move(config));
    RTP_LLM_CHECK(pool->init());
    return pool;
}

std::shared_ptr<HostBlockPool> makeHostPool(size_t payload_bytes, size_t usable_count, bool enable_pinned) {
    auto config                  = std::make_shared<HostBlockPoolConfig>();
    config->pool_type            = BlockPoolType::HOST;
    config->pool_name            = "per_rank_transfer_engine_host";
    config->physical_block_count = usable_count + 1;
    config->payload_bytes        = payload_bytes;
    config->stride_bytes         = ((payload_bytes + 4095) / 4096) * 4096;
    config->enable_pinned        = enable_pinned;
    config->alignment            = 4096;

    auto pool = std::make_shared<HostBlockPool>(config);
    RTP_LLM_CHECK(pool->init());
    return pool;
}

TempDirGuard::TempDirGuard(const char* name): path(makeTempDir(name)) {}

TempDirGuard::~TempDirGuard() {
    removeTempDir(path);
}

std::shared_ptr<BlockTreeDiskBlockPool> makeDiskPool(size_t                       payload_bytes,
                                                     size_t                       usable_count,
                                                     const std::string&           work_dir,
                                                     std::unique_ptr<DiskBlockIO> io,
                                                     const std::string&           pool_name) {
    const size_t stride_bytes = ((payload_bytes + 4095) / 4096) * 4096;

    auto config             = std::make_shared<BlockTreeDiskBlockPoolConfig>();
    config->pool_type       = BlockPoolType::DISK;
    config->pool_name       = pool_name;
    config->work_dir        = work_dir;
    config->local_rank      = 0;
    config->world_rank      = 0;
    config->disk_size_bytes = stride_bytes * (usable_count + 1);
    config->payload_bytes   = payload_bytes;
    config->stride_bytes    = stride_bytes;
    config->buffered_io     = true;

    auto pool = std::make_shared<BlockTreeDiskBlockPool>(config, std::move(io));
    RTP_LLM_CHECK(pool->init());
    return pool;
}

BlockIdxType poolMalloc(IBlockPool& pool) {
    auto block = pool.malloc();
    return block.has_value() ? *block : NULL_BLOCK_IDX;
}

TransferDescriptor makeDescriptor(Tier                             source_tier,
                                  Tier                             target_tier,
                                  const std::vector<BlockIdxType>& device_blocks,
                                  BlockIdxType                     host_block,
                                  BlockIdxType                     disk_block,
                                  size_t                           group_set_id) {
    if (source_tier == Tier::DEVICE && target_tier == Tier::HOST) {
        return TransferDescriptor::deviceToHost(group_set_id, device_blocks, host_block);
    }
    if (source_tier == Tier::HOST && target_tier == Tier::DEVICE) {
        return TransferDescriptor::hostToDevice(group_set_id, host_block, device_blocks);
    }
    if (source_tier == Tier::HOST && target_tier == Tier::DISK) {
        return TransferDescriptor::hostToDisk(group_set_id, host_block, disk_block);
    }
    if (source_tier == Tier::DISK && target_tier == Tier::HOST) {
        return TransferDescriptor::diskToHost(group_set_id, disk_block, host_block);
    }

    TransferDescriptor desc;
    desc.group_set_id  = group_set_id;
    desc.source_tier   = source_tier;
    desc.target_tier   = target_tier;
    desc.device_blocks = device_blocks;
    desc.host_block    = host_block;
    desc.disk_block    = disk_block;
    return desc;
}

bool submitSucceeded(const std::shared_ptr<PerRankBlockTransferEngine>& engine, const TransferDescriptor& desc) {
    auto context = engine->submit(desc);
    context->waitDone();
    return context->success();
}

void expectStatus(const std::shared_ptr<PerRankBlockTransferEngine>& engine,
                  const TransferDescriptor&                          desc,
                  TransferStatus                                     expected) {
    auto context = engine->submit(desc);
    ASSERT_NE(context, nullptr);
    EXPECT_TRUE(context->done());
    context->waitDone();
    EXPECT_EQ(context->success(), expected == TransferStatus::OK);

    const ErrorInfo error_info = context->errorInfo();
    if (expected == TransferStatus::OK) {
        EXPECT_TRUE(error_info.ok());
    } else {
        EXPECT_FALSE(error_info.ok());
        EXPECT_FALSE(error_info.ToString().empty());
        if (expected == TransferStatus::INVALID_ARGS) {
            EXPECT_EQ(error_info.code(), ErrorCode::INVALID_PARAMS);
        } else {
            EXPECT_EQ(error_info.code(), ErrorCode::EXECUTION_EXCEPTION);
        }
    }
}

}  // namespace rtp_llm::block_transfer_engine_test
