#include "rtp_llm/cpp/cache/block_tree_cache/test/PerRankBlockTransferEngineTestUtils.h"

#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

#include <dirent.h>
#include <unistd.h>

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

Component makeSchemaComponent(int                        component_id,
                              int                        component_group_id,
                              const std::string&         tag,
                              const std::vector<size_t>& layer_bytes,
                              const std::vector<int>&    model_layer_ids) {
    Component component;
    component.component_id       = component_id;
    component.component_group_id = component_group_id;
    component.type               = CacheGroupType::FULL;
    component.tag                = tag;
    component.layer_bytes        = layer_bytes;
    if (model_layer_ids.empty()) {
        for (size_t layer_idx = 0; layer_idx < layer_bytes.size(); ++layer_idx) {
            component.model_layer_ids.push_back(static_cast<int>(layer_idx));
        }
    } else {
        component.model_layer_ids = model_layer_ids;
    }
    return component;
}

std::shared_ptr<const std::vector<Component>> makeComponentRegistry(std::vector<Component> components) {
    return std::make_shared<const std::vector<Component>>(std::move(components));
}

TransferDescriptor makeDescriptor(Tier                             source_tier,
                                  Tier                             target_tier,
                                  const std::vector<BlockIdxType>& device_blocks,
                                  BlockIdxType                     host_block,
                                  BlockIdxType                     disk_block,
                                  int                              group_id) {
    if (source_tier == Tier::DEVICE && target_tier == Tier::HOST) {
        return TransferDescriptor::deviceToHost(group_id, device_blocks, host_block);
    }
    if (source_tier == Tier::HOST && target_tier == Tier::DEVICE) {
        return TransferDescriptor::hostToDevice(group_id, host_block, device_blocks);
    }
    if (source_tier == Tier::HOST && target_tier == Tier::DISK) {
        return TransferDescriptor::hostToDisk(group_id, host_block, disk_block);
    }
    if (source_tier == Tier::DISK && target_tier == Tier::HOST) {
        return TransferDescriptor::diskToHost(group_id, disk_block, host_block);
    }

    TransferDescriptor desc;
    desc.component_group_id = group_id;
    desc.source_tier        = source_tier;
    desc.target_tier        = target_tier;
    desc.device_blocks      = device_blocks;
    desc.host_block         = host_block;
    desc.disk_block         = disk_block;
    return desc;
}

void expectStatus(const std::shared_ptr<PerRankBlockTransferEngine>& engine,
                  const TransferDescriptor&                          desc,
                  TransferStatus                                     expected) {
    auto handle = engine->submit(desc);
    EXPECT_TRUE(handle.valid());
    EXPECT_TRUE(handle.done());
    EXPECT_EQ(handle.status(), expected);
    EXPECT_EQ(handle.ok(), expected == TransferStatus::OK);
}

}  // namespace rtp_llm::block_transfer_engine_test
