#include <gtest/gtest.h>
#include <memory>
#include <vector>

#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/CPSlotMapper.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/cache/test/BlockPoolTestHelper.h"
#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace test {

static CacheConfig makeTestConfig(int block_num = 20, int seq_size_per_block = 4) {
    return makeSimpleMhaCacheConfig(
        /*layer_num=*/2,
        block_num,
        /*tokens_per_block=*/static_cast<size_t>(seq_size_per_block),
        rtp_llm::DataType::TYPE_FP16,
        /*local_head_num_kv=*/1,
        /*size_per_head=*/16);
}

static CompleteTokenIdsPtr makeTokenIds(rtp_llm::DeviceBase* device, int batch_size, int seq_len, int block_size) {
    auto ids = std::make_shared<CompleteTokenIds>(device, batch_size, batch_size, seq_len + 100, block_size);
    auto buf =
        device->allocateBuffer({rtp_llm::DataType::TYPE_INT32, {(size_t)seq_len}, rtp_llm::AllocationType::HOST}, {});
    auto* ptr = buf->data<int32_t>();
    for (int i = 0; i < seq_len; ++i)
        ptr[i] = i + 1;
    auto gi             = std::make_shared<GenerateInput>();
    gi->input_ids       = buf;
    gi->generate_config = std::make_shared<GenerateConfig>();
    ids->init(gi);
    return ids;
}

static BatchKVCacheResourcePtr makeResource(int batch_size, int layer_num) {
    auto res = std::make_shared<BatchKVCacheResource>();
    res->resetBatchSize(batch_size);
    std::vector<int> layer_to_group_id(layer_num, 0);
    res->initGroups(/*group_nums=*/1, layer_num, layer_to_group_id);
    return res;
}

class KVCacheManagerCPSlotMapperTest: public ::testing::Test {
protected:
    void SetUp() override {
        rtp_llm::initLogger();
        device_ = createDevice();
        ASSERT_NE(device_, nullptr);
    }
    rtp_llm::DeviceBase* device_ = nullptr;
};

// When kv_cache_sharded is false (default), cpSlotMapper() should return nullptr.
TEST_F(KVCacheManagerCPSlotMapperTest, NoCPSharding_ReturnsNullMapper) {
    auto              config = makeTestConfig();
    ParallelismConfig par;
    par.tp_rank                            = 0;
    par.tp_size                            = 2;
    par.prefill_cp_config.kv_cache_sharded = false;

    auto mgr = std::make_shared<KVCacheManager>(config, device_, false, nullptr, KVCacheConfig{}, par);
    ASSERT_TRUE(mgr->init());

    EXPECT_EQ(mgr->cpSlotMapper(), nullptr);
}

// When tp_size == 1, cpSlotMapper() should return nullptr even if kv_cache_sharded is true.
TEST_F(KVCacheManagerCPSlotMapperTest, SingleRank_ReturnsNullMapper) {
    auto              config = makeTestConfig();
    ParallelismConfig par;
    par.tp_rank                            = 0;
    par.tp_size                            = 1;
    par.prefill_cp_config.kv_cache_sharded = true;

    auto mgr = std::make_shared<KVCacheManager>(config, device_, false, nullptr, KVCacheConfig{}, par);
    ASSERT_TRUE(mgr->init());

    EXPECT_EQ(mgr->cpSlotMapper(), nullptr);
}

// When kv_cache_sharded is true and tp_size > 1, cpSlotMapper() should return a valid mapper.
TEST_F(KVCacheManagerCPSlotMapperTest, CPShardingEnabled_ReturnsValidMapper) {
    const int seq_size_per_block = 4;
    auto      config             = makeTestConfig(/*block_num=*/20, seq_size_per_block);

    ParallelismConfig par;
    par.tp_rank                            = 1;
    par.tp_size                            = 2;
    par.prefill_cp_config.kv_cache_sharded = true;

    auto mgr = std::make_shared<KVCacheManager>(config, device_, false, nullptr, KVCacheConfig{}, par);
    ASSERT_TRUE(mgr->init());

    auto mapper = mgr->cpSlotMapper();
    ASSERT_NE(mapper, nullptr);
    EXPECT_TRUE(mapper->isSharded());
    EXPECT_EQ(mapper->cpRank(), 1);
    EXPECT_EQ(mapper->cpSize(), 2);
    EXPECT_EQ(mapper->blockSize(), seq_size_per_block);
    EXPECT_EQ(mapper->virtualBlockSize(), seq_size_per_block * 2);
}

// malloc() should auto-inject cpSlotMapper when caller does not provide one.
// With CP sharding (cp_size=2, block_size=4), virtual_block_size=8.
// A sequence of 16 tokens needs ceil(16/8)=2 physical blocks per batch (not 4).
TEST_F(KVCacheManagerCPSlotMapperTest, MallocAutoInjectReducesBlockCount) {
    const int seq_size_per_block = 4;
    auto      config             = makeTestConfig(/*block_num=*/20, seq_size_per_block);

    ParallelismConfig par;
    par.tp_rank                            = 0;
    par.tp_size                            = 2;
    par.prefill_cp_config.kv_cache_sharded = true;

    auto mgr = std::make_shared<KVCacheManager>(config, device_, false, nullptr, KVCacheConfig{}, par);
    ASSERT_TRUE(mgr->init());

    const int seq_len   = 16;
    auto      resource  = makeResource(1, config.layer_num);
    auto      token_ids = makeTokenIds(device_, 1, seq_len, seq_size_per_block);

    MallocInfo info{resource, token_ids};
    // cp_slot_mapper left as nullptr -- should be auto-injected
    auto result = mgr->malloc(info);
    ASSERT_TRUE(result.success);

    // virtual_block_size = 4 * 2 = 8
    // effectiveSeqLenForAlloc(16) = ceil(16/8) * 4 = 8 tokens worth => ceil(8/4) = 2 blocks
    EXPECT_EQ(resource->blocksNum(0, 0), 2);
}

// Without CP sharding, the same seq_len should allocate more blocks.
TEST_F(KVCacheManagerCPSlotMapperTest, MallocWithoutCPAllocatesFullBlocks) {
    const int seq_size_per_block = 4;
    auto      config             = makeTestConfig(/*block_num=*/20, seq_size_per_block);

    ParallelismConfig par;
    par.tp_rank                            = 0;
    par.tp_size                            = 2;
    par.prefill_cp_config.kv_cache_sharded = false;

    auto mgr = std::make_shared<KVCacheManager>(config, device_, false, nullptr, KVCacheConfig{}, par);
    ASSERT_TRUE(mgr->init());

    const int seq_len   = 16;
    auto      resource  = makeResource(1, config.layer_num);
    auto      token_ids = makeTokenIds(device_, 1, seq_len, seq_size_per_block);

    MallocInfo info{resource, token_ids};
    auto       result = mgr->malloc(info);
    ASSERT_TRUE(result.success);

    // Without CP: ceil(16/4) = 4 blocks
    EXPECT_EQ(resource->blocksNum(0, 0), 4);
}

// Caller-provided cp_slot_mapper should override the auto-injected one.
TEST_F(KVCacheManagerCPSlotMapperTest, MallocExplicitMapperOverridesAutoInject) {
    const int seq_size_per_block = 4;
    auto      config             = makeTestConfig(/*block_num=*/30, seq_size_per_block);

    // Manager has cp_size=2, but we'll pass a mapper with cp_size=4.
    ParallelismConfig par;
    par.tp_rank                            = 0;
    par.tp_size                            = 2;
    par.prefill_cp_config.kv_cache_sharded = true;

    auto mgr = std::make_shared<KVCacheManager>(config, device_, false, nullptr, KVCacheConfig{}, par);
    ASSERT_TRUE(mgr->init());

    const int seq_len   = 64;
    auto      resource  = makeResource(1, config.layer_num);
    auto      token_ids = makeTokenIds(device_, 1, seq_len, seq_size_per_block);

    auto explicit_mapper = std::make_shared<CPSlotMapper>(0, 4, seq_size_per_block);
    // virtual_block_size = 4 * 4 = 16
    // effectiveSeqLenForAlloc(64) = ceil(64/16)*4 = 16 tokens => ceil(16/4) = 4 blocks

    MallocInfo info{resource, token_ids};
    info.cp_slot_mapper = explicit_mapper;
    auto result         = mgr->malloc(info);
    ASSERT_TRUE(result.success);

    EXPECT_EQ(resource->blocksNum(0, 0), 4);
}

// insertIntoCache() should also auto-inject the mapper.
TEST_F(KVCacheManagerCPSlotMapperTest, InsertAutoInjectsMapper) {
    const int seq_size_per_block = 4;
    auto      config             = makeTestConfig(/*block_num=*/20, seq_size_per_block);

    ParallelismConfig par;
    par.tp_rank                            = 0;
    par.tp_size                            = 2;
    par.prefill_cp_config.kv_cache_sharded = true;

    KVCacheConfig kv_cfg;
    kv_cfg.reuse_cache         = true;
    kv_cfg.enable_device_cache = true;

    auto mgr = std::make_shared<KVCacheManager>(config, device_, false, nullptr, kv_cfg, par);
    ASSERT_TRUE(mgr->init());
    // virtual_block_size = 4 * 2 = 8
    // effectiveSeqLenForAlloc(16) = ceil(16/8) * 4 = 8 tokens worth => ceil(8/4) = 2 blocks

    const int seq_len   = 16;
    auto      resource  = makeResource(1, config.layer_num);
    auto      token_ids = makeTokenIds(device_, 1, seq_len, seq_size_per_block);

    MallocInfo malloc_info{resource, token_ids};
    malloc_info.reuse_cache         = true;
    malloc_info.enable_device_cache = true;
    auto result                     = mgr->malloc(malloc_info);
    ASSERT_TRUE(result.success);

    // Insert into cache (cp_slot_mapper is auto-injected).
    // This should not crash and should use sharded insert logic.
    InsertInfo insert_info{resource, token_ids, /*is_resident=*/false};
    EXPECT_NO_THROW(mgr->insertIntoCache(insert_info));

    // Now try to malloc again with the same token_ids -- should get reuse hit.
    auto       resource2 = makeResource(1, config.layer_num);
    MallocInfo malloc_info2{resource2, token_ids};
    malloc_info2.reuse_cache         = true;
    malloc_info2.enable_device_cache = true;
    auto result2                     = mgr->malloc(malloc_info2);
    ASSERT_TRUE(result2.success);
    EXPECT_GT(result2.reuse_len, 0);
}

}  // namespace test
}  // namespace rtp_llm
