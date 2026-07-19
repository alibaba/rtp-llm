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

static CompleteTokenIdsPtr makeTokenIds(int batch_size, int seq_len, int block_size) {
    auto  ids       = std::make_shared<CompleteTokenIds>(batch_size, batch_size, seq_len + 100, block_size);
    auto  input_ids = torch::empty({(int64_t)seq_len}, torch::kInt32);
    auto* ptr       = input_ids.data_ptr<int32_t>();
    for (int i = 0; i < seq_len; ++i)
        ptr[i] = i + 1;
    auto gi             = std::make_shared<GenerateInput>();
    gi->input_ids       = input_ids;
    gi->generate_config = std::make_shared<GenerateConfig>();
    ids->init(gi);
    return ids;
}

static BatchKVCacheResourcePtr makeResource(int batch_size, int layer_num) {
    auto res = std::make_shared<BatchKVCacheResource>();
    res->resetBatchSize(batch_size);
    std::vector<std::vector<int>> layer_group_ids(static_cast<size_t>(layer_num), std::vector<int>{0});
    res->initGroups(/*group_nums=*/1, layer_num, layer_group_ids);
    return res;
}

class KVCacheManagerCPSlotMapperTest: public ::testing::Test {
protected:
    void SetUp() override {
        rtp_llm::initLogger();
        createDevice();
    }
};

// When kv_cache_sharded is false (default), cpSlotMapper() should return nullptr.
TEST_F(KVCacheManagerCPSlotMapperTest, NoCPSharding_ReturnsNullMapper) {
    auto              config = makeTestConfig();
    ParallelismConfig par;
    par.tp_rank                            = 0;
    par.tp_size                            = 2;
    par.prefill_cp_config.kv_cache_sharded = false;

    // warmup=true skips allocateAndSync (which would NCCL all-gather across the
    // tp_size process group; in single-process UT there are no peers).  cp_slot_mapper_
    // is constructed regardless of warmup, so cpSlotMapper() check is unaffected.
    auto mgr = std::make_shared<KVCacheManager>(config, /*warmup=*/true, nullptr, KVCacheConfig{}, par);
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

    // warmup=true skips allocateAndSync (which would NCCL all-gather across the
    // tp_size process group; in single-process UT there are no peers).  cp_slot_mapper_
    // is constructed regardless of warmup, so cpSlotMapper() check is unaffected.
    auto mgr = std::make_shared<KVCacheManager>(config, /*warmup=*/true, nullptr, KVCacheConfig{}, par);
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

    // warmup=true skips allocateAndSync (which would NCCL all-gather across the
    // tp_size process group; in single-process UT there are no peers).  cp_slot_mapper_
    // is constructed regardless of warmup, so cpSlotMapper() check is unaffected.
    auto mgr = std::make_shared<KVCacheManager>(config, /*warmup=*/true, nullptr, KVCacheConfig{}, par);
    ASSERT_TRUE(mgr->init());

    auto mapper = mgr->cpSlotMapper();
    ASSERT_NE(mapper, nullptr);
    EXPECT_TRUE(mapper->isSharded());
    EXPECT_EQ(mapper->cpRank(), 1);
    EXPECT_EQ(mapper->cpSize(), 2);
    EXPECT_EQ(mapper->blockSize(), seq_size_per_block);
    EXPECT_EQ(mapper->virtualBlockSize(), seq_size_per_block * 2);
}

TEST_F(KVCacheManagerCPSlotMapperTest, CPShardingEnabled_CacheInfoReportsVirtualBlockSize) {
    const int seq_size_per_block = 4;
    auto      config             = makeTestConfig(/*block_num=*/20, seq_size_per_block);

    ParallelismConfig par;
    par.tp_rank                            = 0;
    par.tp_size                            = 4;
    par.prefill_cp_config.kv_cache_sharded = true;

    auto mgr = std::make_shared<KVCacheManager>(config, /*warmup=*/true, nullptr, KVCacheConfig{}, par);
    ASSERT_TRUE(mgr->init());

    auto info = mgr->getKVCacheInfo(/*latest_version=*/-1, /*need_cache_keys=*/false);
    EXPECT_EQ(info.block_size, static_cast<size_t>(seq_size_per_block * par.tp_size));
}

// Partial tails may be allocated as live KV blocks before they become cacheable
// full blocks. CP invariants must therefore be based on logical sequence length,
// not cacheKeys().size().
TEST_F(KVCacheManagerCPSlotMapperTest, CPShardedMallocAllowsPartialTailWithoutCacheKey) {
    const int seq_size_per_block = 4;
    auto      config             = makeTestConfig(/*block_num=*/20, seq_size_per_block);

    ParallelismConfig par;

    auto mgr = std::make_shared<KVCacheManager>(config, /*warmup=*/false, nullptr, KVCacheConfig{}, par);
    ASSERT_TRUE(mgr->init());

    auto resource  = makeResource(1, config.layer_num);
    auto token_ids = makeTokenIds(1, /*seq_len=*/1, seq_size_per_block);

    MallocInfo info{resource, token_ids};
    auto       cp_mapper = std::make_shared<CPSlotMapper>(0, 2, seq_size_per_block);
    mgr->cp_slot_mapper_ = cp_mapper;
    mgr->allocator_->setCPSlotMapper(cp_mapper);

    auto result = mgr->malloc(info);
    ASSERT_TRUE(result.success);
    EXPECT_EQ(resource->blocksNum(0, 0), 1);

    token_ids->setSeqLength(2);
    result = mgr->malloc(info);
    ASSERT_TRUE(result.success);
    EXPECT_EQ(resource->blocksNum(0, 0), 1);
    EXPECT_EQ(resource->cacheKeys(0).size(), 0);
}

TEST_F(KVCacheManagerCPSlotMapperTest, ManagerInjectsSameMapperIntoAllocator) {
    const int seq_size_per_block = 4;
    auto      config             = makeTestConfig(/*block_num=*/20, seq_size_per_block);

    ParallelismConfig par;
    par.tp_rank                            = 0;
    par.tp_size                            = 2;
    par.prefill_cp_config.kv_cache_sharded = true;

    auto mgr = std::make_shared<KVCacheManager>(config, /*warmup=*/true, nullptr, KVCacheConfig{}, par);
    ASSERT_TRUE(mgr->init());

    const auto manager_mapper   = mgr->cpSlotMapper();
    const auto allocator_mapper = mgr->allocator_->cpSlotMapper();
    ASSERT_NE(manager_mapper, nullptr);
    EXPECT_EQ(allocator_mapper, manager_mapper);
    EXPECT_EQ(manager_mapper->cpRank(), par.tp_rank);
    EXPECT_EQ(manager_mapper->cpSize(), par.tp_size);
    EXPECT_EQ(manager_mapper->blockSize(), seq_size_per_block);
}

}  // namespace test
}  // namespace rtp_llm
