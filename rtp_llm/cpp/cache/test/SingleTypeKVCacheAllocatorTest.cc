#include <gtest/gtest.h>
#include <algorithm>
#include <cerrno>
#include <cstring>
#include <memory>
#include <vector>
#include <set>
#include <optional>
#include <torch/torch.h>
#include <dirent.h>
#include <unistd.h>
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/cache/SingleTypeKVCacheAllocator.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/cache/SingleConfigCreator.h"
#include "rtp_llm/cpp/cache/CPSlotMapper.h"
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/config/MTPModelConfigHelper.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"
#include "rtp_llm/cpp/cache/test/BlockPoolTestHelper.h"
#include "rtp_llm/cpp/cache/test/BlockTreeCacheAllocatorTestHelper.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"

namespace rtp_llm {
namespace test {

using TestSingleTypeKVCacheAllocator = BlockTreeCacheTestAllocator<SingleTypeKVCacheAllocator>;
using PendingLoadBackItem = LoadBackTicket::PendingLoadBackItem;

class CountingSingleTypeCopyEngine: public CopyEngine {
public:
    CountingSingleTypeCopyEngine(const std::vector<ComponentGroupPtr>& groups,
                                 const std::vector<Component>&         components):
        CopyEngine(groups, std::make_shared<const std::vector<Component>>(components)) {}

    TransferHandle submit(const TransferDescriptor&) override {
        ++submit_count_;
        return TransferHandle::completed(CopyStatus::OK);
    }

    size_t submitCount() const {
        return submit_count_;
    }

private:
    size_t submit_count_{0};
};

class ScopedSingleTypeDiskDirectory {
public:
    ScopedSingleTypeDiskDirectory() {
        std::string       pattern = "/tmp/rtp_llm_target_single_loadback_XXXXXX";
        std::vector<char> writable(pattern.begin(), pattern.end());
        writable.push_back('\0');
        char* result = ::mkdtemp(writable.data());
        EXPECT_NE(result, nullptr);
        if (result != nullptr) {
            path_ = result;
        }
    }

    ~ScopedSingleTypeDiskDirectory() {
        if (path_.empty()) {
            return;
        }
        const std::string work_dir = path_ + "/rtp_llm_disk_kv";
        if (DIR* dir = ::opendir(work_dir.c_str()); dir != nullptr) {
            while (auto* entry = ::readdir(dir)) {
                const std::string name = entry->d_name;
                if (name != "." && name != "..") {
                    EXPECT_EQ(::unlink((work_dir + "/" + name).c_str()), 0);
                }
            }
            EXPECT_EQ(::closedir(dir), 0);
        } else {
            EXPECT_EQ(errno, ENOENT) << std::strerror(errno);
        }
        if (::rmdir(work_dir.c_str()) != 0) {
            EXPECT_EQ(errno, ENOENT) << std::strerror(errno);
        }
        EXPECT_EQ(::rmdir(path_.c_str()), 0);
    }

    const std::string& path() const {
        return path_;
    }

private:
    std::string path_;
};

KVCacheConfig makeSingleTypeTieredConfig(Tier source_tier, const std::string& disk_path) {
    KVCacheConfig config;
    config.enable_memory_cache        = true;
    config.enable_tiered_memory_cache = true;
    config.memory_cache_size_mb       = 1;
    config.enable_memory_cache_disk   = source_tier == Tier::DISK;
    config.memory_cache_disk_size_mb  = source_tier == Tier::DISK ? 1 : 0;
    config.memory_cache_disk_paths    = disk_path;
    return config;
}

BlockIdxType seedSingleTypeLowerTier(BlockTreeCache& cache, Tier source_tier, CacheKeyType key) {
    const auto& group        = cache.componentGroups().front();
    const auto  source_block = group->allocateSingleBlock(source_tier);
    EXPECT_NE(source_block, NULL_BLOCK_IDX);
    if (isNullBlockIdx(source_block)) {
        return source_block;
    }
    std::vector<std::vector<GroupSlot>> slots(1, std::vector<GroupSlot>(1));
    if (source_tier == Tier::HOST) {
        slots[0][0].host_block = source_block;
    } else {
        slots[0][0].disk_slot = source_block;
    }
    EXPECT_NE(cache.tree()->insertNode(nullptr, CacheKeysType{key}, slots).leaf, nullptr);
    return source_block;
}

CacheConfig createSingleTypeTestConfig(int layer_num = 4, int block_num = 10, int seq_size_per_block = 8) {
    return makeSimpleMhaCacheConfig(/*layer_num=*/layer_num,
                                    /*block_num=*/block_num,
                                    /*tokens_per_block=*/static_cast<size_t>(seq_size_per_block),
                                    rtp_llm::DataType::TYPE_FP16,
                                    /*local_head_num_kv=*/8,
                                    /*size_per_head=*/128);
}

static rtp_llm::ModelConfig makeTestModelConfig(uint32_t num_layers) {
    rtp_llm::ModelConfig m;
    m.num_layers                   = static_cast<int>(num_layers);
    m.max_seq_len                  = 128;
    m.hidden_size                  = 1;
    m.vocab_size                   = 1;
    m.data_type                    = rtp_llm::DataType::TYPE_FP16;
    m.attn_config.use_mla          = false;
    m.attn_config.tokens_per_block = 4;
    m.attn_config.kv_head_num      = 2;
    m.attn_config.size_per_head    = 1;
    m.attn_config.kv_cache_dtype   = KvCacheDataType::FP8;
    m.attn_config.kv_lora_rank     = 0;
    m.attn_config.rope_head_dim    = 0;
    m.attn_config.head_num         = 2;
    setDefaultKvCacheSpec(m);
    return m;
}

static rtp_llm::CacheConfig makeMtpCacheConfigByCreateSpConfig(uint32_t main_layers,
                                                               int      mtp_module_num,
                                                               uint32_t block_num,
                                                               uint32_t mtp_module_layers = 1) {
    auto score_model_config   = makeTestModelConfig(main_layers);
    auto propose_model_config = makeTestModelConfig(mtp_module_layers);

    rtp_llm::ParallelismConfig parallelism_config;
    parallelism_config.tp_size = 1;

    rtp_llm::RuntimeConfig runtime_config;

    rtp_llm::KVCacheConfig kv_cache_config;
    kv_cache_config.test_block_num = static_cast<int>(block_num);

    rtp_llm::SpeculativeExecutionConfig sp_config;
    sp_config.type              = SP_TYPE_MTP;
    sp_config.gen_num_per_cycle = mtp_module_num;

    return rtp_llm::CacheConfigCreator::createSpConfig(score_model_config,
                                                       propose_model_config,
                                                       parallelism_config,
                                                       runtime_config,
                                                       kv_cache_config,
                                                       sp_config,
                                                       /*warm_up_result=*/std::nullopt,
                                                       /*is_mtp=*/true,
                                                       /*is_eagle=*/false);
}

CompleteTokenIdsPtr createCompleteTokenIds(int batch_size, int seq_length, int seq_size_per_block = 8) {
    auto complete_token_ids =
        std::make_shared<CompleteTokenIds>(batch_size, batch_size, seq_length + 100, seq_size_per_block);

    auto  input_ids  = torch::empty({(int64_t)seq_length}, torch::kInt32);
    auto* token_data = input_ids.data_ptr<int32_t>();
    for (int i = 0; i < seq_length; ++i) {
        token_data[i] = i + 1;
    }

    auto generate_input             = std::make_shared<GenerateInput>();
    generate_input->input_ids       = input_ids;
    generate_input->generate_config = std::make_shared<GenerateConfig>();

    complete_token_ids->init(generate_input);
    return complete_token_ids;
}

BatchKVCacheResourcePtr
createBatchKVCacheResource(int batch_size, const CacheConfig& config, int block_num_per_batch = 0) {
    auto resource = std::make_shared<BatchKVCacheResource>();
    resource->resetBatchSize(batch_size);
    resource->initGroups(config.topologyPtr());
    for (int i = 0; i < batch_size; ++i) {
        resource->setBatchBlocks(i, 0, std::vector<int>(block_num_per_batch));
        resource->setBatchCacheKeys(i, CacheKeysType(block_num_per_batch, static_cast<CacheKeyType>(i * 100)));
    }
    return resource;
}

static int estimateBatchPeakForSingleSequence(const KVCacheAllocator&        allocator,
                                              const BatchKVCacheResourcePtr& batch_resource,
                                              int                            seq_len,
                                              int                            remaining_tokens,
                                              int                            reserve_step,
                                              bool                           enable_reuse_cache) {
    return allocator.estimateBatchPeakNeedBlocks(batch_resource,
                                                 seq_len,
                                                 /*common_seq_len=*/seq_len,
                                                 remaining_tokens,
                                                 reserve_step,
                                                 enable_reuse_cache,
                                                 /*target_batch_size=*/1);
}

class SingleTypeKVCacheAllocatorTest: public ::testing::Test {
protected:
    void SetUp() override {
        createDevice();
    }

    void TearDown() override {
        allocator_.reset();
    }

    // DeviceBlockPool is DEVICE-only. Stage bytes through runtime copy helpers
    // rather than dereferencing its addresses from the host.
    static void writeDeviceBytes(void* dst_device, const std::vector<uint8_t>& host) {
        auto host_tensor = torch::from_blob(const_cast<uint8_t*>(host.data()),
                                            {static_cast<int64_t>(host.size())},
                                            torch::TensorOptions(torch::kUInt8))
                               .clone();
        auto device_tensor = torch::from_blob(
            dst_device, {static_cast<int64_t>(host.size())}, torch::TensorOptions(torch::kUInt8).device(torch::kCUDA));
        CopyParams copy_params{device_tensor, host_tensor};
        runtimeCopy(copy_params);
        runtimeSyncAndCheck();
    }

    static std::vector<uint8_t> readDeviceBytes(const void* src_device, size_t bytes) {
        auto        device_tensor = torch::from_blob(const_cast<void*>(src_device),
                                                     {static_cast<int64_t>(bytes)},
                                              torch::TensorOptions(torch::kUInt8).device(torch::kCUDA));
        auto        host_tensor   = device_tensor.cpu();
        const auto* data          = host_tensor.data_ptr<uint8_t>();
        return std::vector<uint8_t>(data, data + bytes);
    }

    std::shared_ptr<TestSingleTypeKVCacheAllocator> allocator_;
};

// Test init
TEST_F(SingleTypeKVCacheAllocatorTest, ConstructorAndInit) {
    auto config = createSingleTypeTestConfig();
    allocator_  = std::make_shared<TestSingleTypeKVCacheAllocator>(config);
    ASSERT_NE(allocator_, nullptr);

    bool init_result = allocator_->init();
    EXPECT_TRUE(init_result);

    EXPECT_EQ(allocator_->totalBlocksNum(), config.block_num - 1);
    EXPECT_EQ(allocator_->freeBlocksNum(), config.block_num - 1);  // reserve 1 block
}

TEST_F(SingleTypeKVCacheAllocatorTest, InitRejectsLinearGroupBeforeCreatingBlockPool) {
    auto config = makeSimpleLinearCacheConfig(
        /*layer_num=*/2, /*block_num=*/4, /*tokens_per_block=*/4, rtp_llm::DataType::TYPE_FP16);
    allocator_ = std::make_shared<TestSingleTypeKVCacheAllocator>(config);

    EXPECT_THROW(allocator_->init(), std::runtime_error);
    EXPECT_EQ(allocator_->getDeviceBlockPool(), nullptr);
}

TEST_F(SingleTypeKVCacheAllocatorTest, InitWithDifferentLayerNum) {
    auto config = createSingleTypeTestConfig(8, 20, 16);
    allocator_  = std::make_shared<TestSingleTypeKVCacheAllocator>(config);

    bool init_result = allocator_->init();
    EXPECT_TRUE(init_result);

    EXPECT_EQ(allocator_->totalBlocksNum(), 20 - 1);
}

TEST_F(SingleTypeKVCacheAllocatorTest, GetNeedBlocksComputesCommonAndExtra) {
    auto config = createSingleTypeTestConfig(/*layer_num=*/4, /*block_num=*/10, /*seq_size_per_block=*/4);
    allocator_  = std::make_shared<TestSingleTypeKVCacheAllocator>(config);
    ASSERT_TRUE(allocator_->init());

    const int batch_size = 3;
    auto      batch_res  = createBatchKVCacheResource(batch_size, config);
    auto      token_ids  = createCompleteTokenIds(batch_size, /*seq_length=*/17, /*seq_size_per_block=*/4);
    token_ids->setReserveStep(3);

    // common_len = floor(17/4)*4 = 16 => 4 common blocks
    // total per-batch blocks = ceil((17+3)/4) = 5 => extra per-batch = 1
    // total = common + batch * extra = 4 + 3*1 = 7
    MallocInfo malloc_info{batch_res, token_ids};
    EXPECT_EQ(allocator_->getNeedBlocks(malloc_info), 7);
}

// Test malloc
TEST_F(SingleTypeKVCacheAllocatorTest, MallocSingleBatch) {
    auto config = createSingleTypeTestConfig();
    allocator_  = std::make_shared<TestSingleTypeKVCacheAllocator>(config);
    allocator_->init();

    int  seq_length         = 16;
    auto batch_resource     = createBatchKVCacheResource(1, config);
    auto complete_token_ids = createCompleteTokenIds(1, seq_length);

    MallocInfo malloc_info{batch_resource, complete_token_ids};
    auto       result = allocator_->malloc(malloc_info);

    EXPECT_TRUE(result.success);
    EXPECT_EQ(batch_resource->blocksNum(0, 0), 2);
    EXPECT_LT(allocator_->freeBlocksNum(), config.block_num);

    seq_length         = 160;
    complete_token_ids = createCompleteTokenIds(1, seq_length);
    MallocInfo malloc_info2{batch_resource, complete_token_ids};
    auto       result2 = allocator_->malloc(malloc_info2);
    EXPECT_FALSE(result2.success);
}

TEST_F(SingleTypeKVCacheAllocatorTest, ReserveBlocksOnlyAppliedToInitMalloc) {
    auto config = createSingleTypeTestConfig(/*layer_num=*/4, /*block_num=*/10, /*seq_size_per_block=*/1);
    allocator_  = std::make_shared<TestSingleTypeKVCacheAllocator>(config);
    ASSERT_TRUE(allocator_->init());

    allocator_->setReserveBlocksNum(2);

    const size_t free_before = allocator_->freeBlocksNum();
    ASSERT_EQ(free_before, 9u);

    // Init malloc requesting 8 blocks should fail: 9 < 8 + 2 reserved.
    {
        auto batch_resource     = createBatchKVCacheResource(/*batch_size=*/1, config);
        auto complete_token_ids = createCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/8, /*seq_size_per_block=*/1);

        MallocInfo malloc_info{batch_resource, complete_token_ids};
        auto       result = allocator_->malloc(malloc_info);
        EXPECT_FALSE(result.success);
        EXPECT_EQ(batch_resource->curBlocksNum(), 0);
        EXPECT_EQ(allocator_->freeBlocksNum(), free_before);
    }

    // Init malloc requesting 7 blocks should succeed: 9 >= 7 + 2 reserved.
    auto       batch_resource_ok = createBatchKVCacheResource(/*batch_size=*/1, config);
    auto       token_ids_7       = createCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/7, /*seq_size_per_block=*/1);
    MallocInfo info_7{batch_resource_ok, token_ids_7};
    auto       r1 = allocator_->malloc(info_7);
    ASSERT_TRUE(r1.success);
    EXPECT_EQ(batch_resource_ok->curBlocksNum(), 7);

    // Incr malloc is allowed to consume the reserved blocks.
    auto       token_ids_9 = createCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/9, /*seq_size_per_block=*/1);
    MallocInfo info_9{batch_resource_ok, token_ids_9};
    auto       r2 = allocator_->malloc(info_9);
    EXPECT_TRUE(r2.success);
    EXPECT_EQ(batch_resource_ok->curBlocksNum(), 9);
}

TEST_F(SingleTypeKVCacheAllocatorTest, ReserveBlocksCheckHappensAfterReuseReferenceInInitMallocForCommonLen) {
    auto config = createSingleTypeTestConfig(/*layer_num=*/4, /*block_num=*/10, /*seq_size_per_block=*/4);
    allocator_  = std::make_shared<TestSingleTypeKVCacheAllocator>(config);
    ASSERT_TRUE(allocator_->init());

    allocator_->setReserveBlocksNum(2);
    ASSERT_EQ(allocator_->freeBlocksNum(), 9);

    // set system property with 4 blocks: cache keys {100, 101, 102, 103}.
    {
        auto seed_resource = createBatchKVCacheResource(/*batch_size=*/1, config);
        seed_resource->setBatchCacheKeys(0, CacheKeysType{100, 101, 102, 103});
        auto seed_token_ids = createCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/16, /*seq_size_per_block=*/4);

        MallocInfo seed_malloc_info{seed_resource, seed_token_ids};
        auto       seed_malloc_result = allocator_->malloc(seed_malloc_info);
        ASSERT_TRUE(seed_malloc_result.success);
        ASSERT_EQ(seed_resource->curBlocksNum(), 4);

        InsertInfo seed_insert_info{seed_resource, seed_token_ids, /*is_resident=*/true};
        allocator_->insertIntoCache(seed_insert_info);
    }

    // reuse 4 block, allocate 1 new block
    {
        auto batch_resource = createBatchKVCacheResource(/*batch_size=*/1, config);
        batch_resource->setBatchCacheKeys(0, CacheKeysType{100, 101, 102, 103, 200});  // match_keys -> {100}

        auto       token_ids = createCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/20, /*seq_size_per_block=*/4);
        MallocInfo malloc_info{batch_resource, token_ids};
        malloc_info.enable_device_cache = true;

        auto result = allocator_->malloc(malloc_info);
        EXPECT_TRUE(result.success);
        const size_t reuse_blocks = batch_resource->cacheResource(0).reuseBlockNum();
        EXPECT_EQ(reuse_blocks * static_cast<size_t>(config.seq_size_per_block), static_cast<size_t>(result.reuse_len));
        EXPECT_EQ(batch_resource->curBlocksNum(), 5);
        EXPECT_EQ(allocator_->freeBlocksNum(), 4);
        FreeInfo free_info{batch_resource, token_ids};
        allocator_->free(free_info);
        EXPECT_EQ(allocator_->freeBlocksNum(), 5);
    }

    // reuse 4 blocks but allocate 5 new blocks, exceed reserved blocks
    {
        auto batch_resource = createBatchKVCacheResource(/*batch_size=*/1, config);
        batch_resource->setBatchCacheKeys(0, CacheKeysType{100, 101, 102, 103, 300, 301, 302, 303});

        auto       token_ids = createCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/32, /*seq_size_per_block=*/4);
        MallocInfo malloc_info{batch_resource, token_ids};
        malloc_info.enable_device_cache = false;

        auto result = allocator_->malloc(malloc_info);
        EXPECT_FALSE(result.success);
        EXPECT_EQ(batch_resource->curBlocksNum(), 0);

        EXPECT_EQ(allocator_->freeBlocksNum(), 5);
    }
}

TEST_F(SingleTypeKVCacheAllocatorTest, MallocMultipleBatches) {
    auto config = createSingleTypeTestConfig();
    allocator_  = std::make_shared<TestSingleTypeKVCacheAllocator>(config);
    allocator_->init();

    int  batch_size         = 3;
    int  seq_length         = 17;
    auto batch_resource     = createBatchKVCacheResource(batch_size, config);
    auto complete_token_ids = createCompleteTokenIds(batch_size, seq_length);

    MallocInfo malloc_info{batch_resource, complete_token_ids};

    auto result = allocator_->malloc(malloc_info);

    EXPECT_TRUE(result.success);
    for (int i = 0; i < batch_size; ++i) {
        EXPECT_EQ(batch_resource->blocksNum(i, 0), 3);
    }
    EXPECT_EQ(allocator_->freeBlocksNum(), config.block_num - 6);  // 2 shared + 3 batches * 1 blocks + 1 reserved
}

// TEST_F(SingleTypeKVCacheAllocatorTest, MallocWithInsufficientBlocks) {
//     auto config = createSingleTypeTestConfig(4, 5, 8);
//     allocator_ = std::make_shared<TestSingleTypeKVCacheAllocator>(config);
//     allocator_->init();

//     int batch_size = 3;
//     int seq_length = 16;  // 3 batches * 2 blocks
//     auto batch_resource = createBatchKVCacheResource(batch_size, config);
//     auto complete_token_ids = createCompleteTokenIds(batch_size, seq_length);

//     MallocInfo malloc_info{batch_resource, complete_token_ids};
//     auto result = allocator_->malloc(malloc_info);

//     EXPECT_LE(allocator_->freeBlocksNum(), 5);
// }

// Test free
TEST_F(SingleTypeKVCacheAllocatorTest, FreeSingleBatch) {
    auto config = createSingleTypeTestConfig();
    allocator_  = std::make_shared<TestSingleTypeKVCacheAllocator>(config);
    allocator_->init();

    int  seq_length         = 16;
    auto batch_resource     = createBatchKVCacheResource(1, config);
    auto complete_token_ids = createCompleteTokenIds(1, seq_length);

    MallocInfo malloc_info{batch_resource, complete_token_ids};
    allocator_->malloc(malloc_info);

    size_t free_before = allocator_->freeBlocksNum();

    FreeInfo free_info{batch_resource, complete_token_ids};
    allocator_->free(free_info);
    EXPECT_GT(allocator_->freeBlocksNum(), free_before);
}

TEST_F(SingleTypeKVCacheAllocatorTest, FreeMultipleBatches) {
    auto config = createSingleTypeTestConfig();
    allocator_  = std::make_shared<TestSingleTypeKVCacheAllocator>(config);
    allocator_->init();

    int  batch_size         = 3;
    int  seq_length         = 16;
    auto batch_resource     = createBatchKVCacheResource(batch_size, config);
    auto complete_token_ids = createCompleteTokenIds(batch_size, seq_length);

    MallocInfo malloc_info{batch_resource, complete_token_ids};
    allocator_->malloc(malloc_info);

    FreeInfo free_info{batch_resource, complete_token_ids};
    allocator_->free(free_info);
    EXPECT_EQ(allocator_->freeBlocksNum(), config.block_num - 1);  // reserve 1 block
}

// Test malloc free cycle
TEST_F(SingleTypeKVCacheAllocatorTest, MallocFreeCycle) {
    auto config = createSingleTypeTestConfig();
    allocator_  = std::make_shared<TestSingleTypeKVCacheAllocator>(config);
    allocator_->init();

    for (int i = 0; i < 5; ++i) {
        int  seq_length         = 16;
        auto batch_resource     = createBatchKVCacheResource(1, config);
        auto complete_token_ids = createCompleteTokenIds(1, seq_length);

        MallocInfo malloc_info{batch_resource, complete_token_ids};
        auto       malloc_result = allocator_->malloc(malloc_info);
        EXPECT_TRUE(malloc_result.success);

        FreeInfo free_info{batch_resource, complete_token_ids};
        allocator_->free(free_info);

        EXPECT_EQ(allocator_->freeBlocksNum(), config.block_num - 1);  // reserve 1 block
    }
}

// Test insert into cache
TEST_F(SingleTypeKVCacheAllocatorTest, InsertIntoCache) {
    auto config = createSingleTypeTestConfig();
    allocator_  = std::make_shared<TestSingleTypeKVCacheAllocator>(config);
    allocator_->init();

    int  seq_length         = 16;
    auto batch_resource     = createBatchKVCacheResource(1, config);
    auto complete_token_ids = createCompleteTokenIds(1, seq_length);

    MallocInfo malloc_info{batch_resource, complete_token_ids};
    allocator_->malloc(malloc_info);

    InsertInfo insert_info{batch_resource, complete_token_ids, false};
    allocator_->insertIntoCache(insert_info);
}

TEST_F(SingleTypeKVCacheAllocatorTest, InsertIntoCacheAsResident) {
    auto config = createSingleTypeTestConfig();
    allocator_  = std::make_shared<TestSingleTypeKVCacheAllocator>(config);
    allocator_->init();

    int  seq_length         = 16;
    auto batch_resource     = createBatchKVCacheResource(1, config);
    auto complete_token_ids = createCompleteTokenIds(1, seq_length);

    MallocInfo malloc_info{batch_resource, complete_token_ids};
    allocator_->malloc(malloc_info);

    InsertInfo insert_info{batch_resource, complete_token_ids, true};
    allocator_->insertIntoCache(insert_info);
}

TEST_F(SingleTypeKVCacheAllocatorTest, OrdinaryAllocationEvictsOnlyAfterTreeEntryLosesRequestHold) {
    const auto config = createSingleTypeTestConfig(/*layer_num=*/2, /*block_num=*/4, /*seq_size_per_block=*/4);
    allocator_        = std::make_shared<TestSingleTypeKVCacheAllocator>(config);
    ASSERT_TRUE(allocator_->init());

    auto seed = createBatchKVCacheResource(/*batch_size=*/1, config);
    seed->setBatchCacheKeys(0, CacheKeysType{100});
    auto       seed_tokens = createCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/4, /*seq_size_per_block=*/4);
    MallocInfo seed_malloc{seed, seed_tokens};
    seed_malloc.enable_device_cache = false;
    ASSERT_TRUE(allocator_->malloc(seed_malloc).success);
    ASSERT_EQ(seed->blocksNum(0, 0), 1);
    const BlockIdxType seed_block = seed->blocks(0, 0).front();
    allocator_->insertIntoCache(InsertInfo{seed, seed_tokens, /*is_resident=*/false});

    const auto& device_pool = allocator_->blockTreeCacheOwner()->componentGroups().front()->devicePools().front();
    ASSERT_NE(device_pool, nullptr);
    EXPECT_EQ(device_pool->refCount(seed_block), 2u);

    auto pressure = createBatchKVCacheResource(/*batch_size=*/1, config);
    pressure->setBatchCacheKeys(0, CacheKeysType{200, 201, 202});
    auto       pressure_tokens = createCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/12, /*seq_size_per_block=*/4);
    MallocInfo pressure_malloc{pressure, pressure_tokens};
    pressure_malloc.enable_device_cache = false;
    EXPECT_FALSE(allocator_->malloc(pressure_malloc).success);
    EXPECT_TRUE(device_pool->isAllocated(seed_block));
    EXPECT_EQ(pressure->curBlocksNum(), 0);

    allocator_->free(FreeInfo{seed, seed_tokens});
    EXPECT_EQ(device_pool->refCount(seed_block), 1u);
    EXPECT_TRUE(allocator_->malloc(pressure_malloc).success);
    EXPECT_EQ(allocator_->blockTreeCacheOwner()->tree()->findNode(CacheKeysType{100}).matched_blocks, 0u);
    EXPECT_NE(std::find(pressure->blocks(0, 0).begin(), pressure->blocks(0, 0).end(), seed_block),
              pressure->blocks(0, 0).end())
        << "the freed numeric id may be immediately reused by the pressure request";

    allocator_->free(FreeInfo{pressure, pressure_tokens});
}

TEST_F(SingleTypeKVCacheAllocatorTest, InsertIntoCachePublishesOnlyBatchZero) {
    const auto config = createSingleTypeTestConfig(/*layer_num=*/2, /*block_num=*/8, /*seq_size_per_block=*/4);
    allocator_        = std::make_shared<TestSingleTypeKVCacheAllocator>(config);
    ASSERT_TRUE(allocator_->init());

    auto block_pool = allocator_->getDeviceBlockPool();
    ASSERT_NE(block_pool, nullptr);
    const auto blocks = block_pool->malloc(2).value();
    ASSERT_EQ(blocks.size(), 2u);
    block_pool->incRef(blocks);

    auto resource = createBatchKVCacheResource(/*batch_size=*/2, config);
    resource->setBatchBlocks(0, 0, BlockIndicesType{blocks[0]});
    resource->setBatchBlocks(1, 0, BlockIndicesType{blocks[1]});
    resource->setBatchCacheKeys(0, CacheKeysType{100});
    resource->setBatchCacheKeys(1, CacheKeysType{200});
    allocator_->insertIntoCache(InsertInfo{resource, nullptr, /*is_resident=*/false});

    const auto& tag              = config.tagForGroup(0);
    auto        batch_zero_match = allocator_->blockTreeCacheOwner()->match(CacheKeysType{100});
    ASSERT_EQ(batch_zero_match.matched_blocks, 1u);
    ASSERT_EQ(batch_zero_match.group_block_indices.at(tag), (BlockIndicesType{blocks[0]}));
    allocator_->blockTreeCacheOwner()->releaseMatchedBlocks(batch_zero_match.matched_block_sets);

    auto batch_one_match = allocator_->blockTreeCacheOwner()->match(CacheKeysType{200});
    EXPECT_EQ(batch_one_match.matched_blocks, 0u);
    EXPECT_TRUE(batch_one_match.group_block_indices.empty());
    allocator_->blockTreeCacheOwner()->releaseMatchedBlocks(batch_one_match.matched_block_sets);

    block_pool->decRef(blocks);
}

TEST_F(SingleTypeKVCacheAllocatorTest, CPInsertAndAllocatorMatchShareLastRankCanonicalKeys) {
    const auto config = createSingleTypeTestConfig(/*layer_num=*/2, /*block_num=*/12, /*seq_size_per_block=*/4);
    allocator_        = std::make_shared<TestSingleTypeKVCacheAllocator>(config);
    ASSERT_TRUE(allocator_->init());
    allocator_->setCPSlotMapper(std::make_shared<CPSlotMapper>(/*cp_rank=*/0, /*cp_size=*/2, /*block_size=*/4));

    auto block_pool = allocator_->getDeviceBlockPool();
    ASSERT_NE(block_pool, nullptr);
    const auto seed_blocks = block_pool->malloc(2).value();
    ASSERT_EQ(seed_blocks.size(), 2u);
    block_pool->incRef(seed_blocks);

    auto seed = createBatchKVCacheResource(/*batch_size=*/1, config);
    seed->setBatchBlocks(0, 0, seed_blocks);
    seed->setBatchCacheKeys(0, CacheKeysType{100, 101, 102, 103});
    allocator_->insertIntoCache(InsertInfo{seed, nullptr, /*is_resident=*/false});

    auto noncanonical_match = allocator_->blockTreeCacheOwner()->match(CacheKeysType{100, 102});
    EXPECT_EQ(noncanonical_match.matched_blocks, 0u);
    allocator_->blockTreeCacheOwner()->releaseMatchedBlocks(noncanonical_match.matched_block_sets);

    const auto& tag             = config.tagForGroup(0);
    auto        canonical_match = allocator_->blockTreeCacheOwner()->match(CacheKeysType{101, 103});
    ASSERT_EQ(canonical_match.matched_blocks, 2u);
    EXPECT_EQ(canonical_match.group_block_indices.at(tag), seed_blocks);
    allocator_->blockTreeCacheOwner()->releaseMatchedBlocks(canonical_match.matched_block_sets);

    auto hit = createBatchKVCacheResource(/*batch_size=*/1, config);
    hit->setBatchCacheKeys(0, CacheKeysType{100, 101, 102, 103, 200, 201});
    auto       hit_tokens = createCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/24, /*seq_size_per_block=*/4);
    MallocInfo hit_info{hit, hit_tokens};
    auto       hit_result = allocator_->malloc(hit_info);
    ASSERT_TRUE(hit_result.success);
    EXPECT_EQ(hit_result.reuse_len, 16);
    ASSERT_GE(hit->blocksNum(0, 0), 2);
    EXPECT_EQ(hit->blocks(0, 0)[0], seed_blocks[0]);
    EXPECT_EQ(hit->blocks(0, 0)[1], seed_blocks[1]);

    allocator_->free(FreeInfo{hit, hit_tokens});
    block_pool->decRef(seed_blocks);
}

TEST_F(SingleTypeKVCacheAllocatorTest, EarlyCommonMallocFailureAbortsTicketBeforeRequestTargetFree) {
    for (const Tier source_tier : {Tier::HOST, Tier::DISK}) {
        SCOPED_TRACE(source_tier == Tier::HOST ? "host" : "disk");
        ScopedSingleTypeDiskDirectory disk_directory;
        const auto config = createSingleTypeTestConfig(/*layer_num=*/2, /*block_num=*/3, /*seq_size_per_block=*/4);
        allocator_        = std::make_shared<TestSingleTypeKVCacheAllocator>(config);
        allocator_->setBlockTreeCacheConfigForTest(makeSingleTypeTieredConfig(source_tier, disk_directory.path()));
        ASSERT_TRUE(allocator_->init());

        const auto& cache = allocator_->blockTreeCacheOwner();
        ASSERT_NE(cache, nullptr);
        const BlockIdxType source_block = seedSingleTypeLowerTier(*cache, source_tier, /*key=*/100);
        ASSERT_NE(source_block, NULL_BLOCK_IDX);
        const auto&  group             = cache->componentGroups().front();
        const size_t source_ref_before = source_tier == Tier::HOST ? group->hostPool()->refCount(source_block) :
                                                                     group->diskPool()->refCount(source_block);
        const size_t free_before       = allocator_->freeBlocksNum();
        const auto   snapshot_before   = cache->getKeySnapshot(/*limit=*/16);

        auto                     registry                 = cache->load_back_ticket_registry_;
        auto                     original_abort           = registry->abort_callback_;
        size_t                   abort_count              = 0;
        size_t                   free_blocks_during_abort = free_before;
        std::vector<std::string> events;
        registry->abort_callback_ = [&](const LoadBackTicket& ticket) {
            ++abort_count;
            const auto& items = ticket.items();
            events.push_back("ticket_abort_begin");
            EXPECT_FALSE(items.empty());
            EXPECT_LT(allocator_->freeBlocksNum(), free_before);
            original_abort(ticket);
            free_blocks_during_abort = allocator_->freeBlocksNum();
            EXPECT_LT(free_blocks_during_abort, free_before);
            const size_t source_ref_after_abort = source_tier == Tier::HOST ?
                                                      group->hostPool()->refCount(source_block) :
                                                      group->diskPool()->refCount(source_block);
            EXPECT_EQ(source_ref_after_abort, source_ref_before);
            events.push_back("source_protection_released");
        };

        auto resource = createBatchKVCacheResource(/*batch_size=*/1, config);
        resource->setBatchCacheKeys(0, CacheKeysType{100, 200, 300});
        auto       token_ids = createCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/9, /*seq_size_per_block=*/4);
        const auto result    = allocator_->malloc(MallocInfo{resource, token_ids});
        if (allocator_->freeBlocksNum() == free_before) {
            events.push_back("request_targets_freed");
        }

        EXPECT_FALSE(result.success);
        EXPECT_EQ(result.async_context, nullptr);
        EXPECT_EQ(abort_count, 1u);
        EXPECT_LT(free_blocks_during_abort, free_before);
        EXPECT_EQ(
            events,
            (std::vector<std::string>{"ticket_abort_begin", "source_protection_released", "request_targets_freed"}));
        EXPECT_EQ(resource->curBlocksNum(), 0);
        EXPECT_EQ(allocator_->freeBlocksNum(), free_before);
        const auto snapshot_after = cache->getKeySnapshot(/*limit=*/16);
        EXPECT_EQ(snapshot_after.version, snapshot_before.version);
        EXPECT_EQ(snapshot_after.keys, snapshot_before.keys);

        registry->abort_callback_ = std::move(original_abort);
    }
}

TEST_F(SingleTypeKVCacheAllocatorTest, LowerTierHitFollowedByOuterIncrFailureNeverCommits) {
    for (const Tier source_tier : {Tier::HOST, Tier::DISK}) {
        SCOPED_TRACE(source_tier == Tier::HOST ? "host" : "disk");
        ScopedSingleTypeDiskDirectory disk_directory;
        const auto config = createSingleTypeTestConfig(/*layer_num=*/2, /*block_num=*/5, /*seq_size_per_block=*/8);
        allocator_        = std::make_shared<TestSingleTypeKVCacheAllocator>(config);
        allocator_->setBlockTreeCacheConfigForTest(makeSingleTypeTieredConfig(source_tier, disk_directory.path()));
        ASSERT_TRUE(allocator_->init());

        const auto& cache = allocator_->blockTreeCacheOwner();
        ASSERT_NE(cache, nullptr);
        auto copy_engine =
            std::make_shared<CountingSingleTypeCopyEngine>(cache->componentGroups(), cache->components());
        cache->copy_engine_ = copy_engine;

        const BlockIdxType source_block = seedSingleTypeLowerTier(*cache, source_tier, /*key=*/100);
        ASSERT_NE(source_block, NULL_BLOCK_IDX);
        const auto&  group             = cache->componentGroups().front();
        const size_t source_ref_before = source_tier == Tier::HOST ? group->hostPool()->refCount(source_block) :
                                                                     group->diskPool()->refCount(source_block);
        const size_t free_before       = allocator_->freeBlocksNum();
        const auto   snapshot_before   = cache->getKeySnapshot(/*limit=*/16);

        auto   registry            = cache->load_back_ticket_registry_;
        auto   original_commit     = registry->commit_callback_;
        auto   original_abort      = registry->abort_callback_;
        size_t commit_count        = 0;
        size_t abort_count         = 0;
        registry->commit_callback_ = [&](const LoadBackTicket& ticket) {
            ++commit_count;
            return original_commit(ticket);
        };
        registry->abort_callback_ = [&](const LoadBackTicket& ticket) {
            ++abort_count;
            original_abort(ticket);
        };

        auto resource = createBatchKVCacheResource(/*batch_size=*/3, config);
        resource->setBatchCacheKeys(0, CacheKeysType{100, 200, 300});
        auto       token_ids = createCompleteTokenIds(/*batch_size=*/3, /*seq_length=*/17, /*seq_size_per_block=*/8);
        MallocInfo info{resource, token_ids};
        const auto result = allocator_->malloc(info);

        EXPECT_FALSE(result.success);
        EXPECT_EQ(result.async_context, nullptr);
        EXPECT_EQ(commit_count, 0u);
        EXPECT_EQ(abort_count, 1u);
        EXPECT_EQ(copy_engine->submitCount(), 0u);
        EXPECT_EQ(resource->curBlocksNum(), 0);
        EXPECT_EQ(allocator_->freeBlocksNum(), free_before);

        cache->waitForPendingTasks();
        const auto snapshot_after = cache->getKeySnapshot(/*limit=*/16);
        EXPECT_EQ(snapshot_after.version, snapshot_before.version);
        EXPECT_EQ(snapshot_after.keys, snapshot_before.keys);
        const auto find = cache->tree()->findNode(CacheKeysType{100});
        ASSERT_NE(find.matched_node, nullptr);
        const auto& slot = find.matched_node->group_slots[0];
        EXPECT_EQ(slot.transfer_state, SlotTransferState::IDLE);
        if (source_tier == Tier::HOST) {
            EXPECT_EQ(slot.host_block, source_block);
            EXPECT_EQ(group->hostPool()->refCount(source_block), source_ref_before);
        } else {
            EXPECT_EQ(slot.disk_slot, source_block);
            EXPECT_EQ(group->diskPool()->refCount(source_block), source_ref_before);
        }
        registry->commit_callback_ = std::move(original_commit);
        registry->abort_callback_  = std::move(original_abort);
    }
}

TEST_F(SingleTypeKVCacheAllocatorTest, SuccessfulOuterAllocationCommitsLoadBackExactlyOnce) {
    ScopedSingleTypeDiskDirectory disk_directory;
    const auto config  = createSingleTypeTestConfig(/*layer_num=*/2, /*block_num=*/16, /*seq_size_per_block=*/4);
    allocator_         = std::make_shared<TestSingleTypeKVCacheAllocator>(config);
    auto tiered_config = makeSingleTypeTieredConfig(Tier::HOST, disk_directory.path());
    tiered_config.device_cache_min_free_blocks = 15;
    allocator_->setBlockTreeCacheConfigForTest(std::move(tiered_config));
    ASSERT_TRUE(allocator_->init());

    const auto& cache = allocator_->blockTreeCacheOwner();
    ASSERT_NE(cache, nullptr);
    auto copy_engine    = std::make_shared<CountingSingleTypeCopyEngine>(cache->componentGroups(), cache->components());
    cache->copy_engine_ = copy_engine;
    ASSERT_NE(seedSingleTypeLowerTier(*cache, Tier::HOST, /*key=*/100), NULL_BLOCK_IDX);

    auto   registry            = cache->load_back_ticket_registry_;
    auto   original_commit     = registry->commit_callback_;
    size_t commit_count        = 0;
    registry->commit_callback_ = [&](const LoadBackTicket& ticket) {
        ++commit_count;
        return original_commit(ticket);
    };

    auto resource = createBatchKVCacheResource(/*batch_size=*/2, config);
    resource->setBatchCacheKeys(0, CacheKeysType{100, 200, 300});
    auto       token_ids = createCompleteTokenIds(/*batch_size=*/2, /*seq_length=*/9, /*seq_size_per_block=*/4);
    const auto result    = allocator_->malloc(MallocInfo{resource, token_ids});
    ASSERT_TRUE(result.success);
    ASSERT_NE(result.async_context, nullptr);
    EXPECT_EQ(commit_count, 1u);
    result.async_context->waitDone();
    EXPECT_TRUE(result.async_context->success()) << result.async_context->errorInfo().ToString();
    EXPECT_EQ(commit_count, 1u);
    EXPECT_GT(copy_engine->submitCount(), 0u);

    const auto find = cache->tree()->findNode(CacheKeysType{100});
    ASSERT_NE(find.matched_node, nullptr);
    const auto& slot = find.matched_node->group_slots.front();
    ASSERT_EQ(slot.device_blocks.size(), 1u);
    const BlockIdxType published_target = slot.device_blocks.front();
    const auto&        device_pool      = cache->componentGroups().front()->devicePools().front();
    ASSERT_NE(device_pool, nullptr);
    ASSERT_FALSE(resource->blocks(0, 0).empty());
    ASSERT_FALSE(resource->blocks(1, 0).empty());
    EXPECT_EQ(resource->blocks(0, 0).front(), published_target);
    EXPECT_EQ(resource->blocks(1, 0).front(), published_target);
    // Two request holders (one per batch) plus the published tree holder.
    EXPECT_EQ(device_pool->refCount(published_target), 3u);
    EXPECT_EQ(cache->getStats().device_heap_total_size, 0u);
    const auto before_watermark_retry = cache->getKeySnapshot(/*limit=*/16);
    cache->onBlocksReleased();
    cache->waitForPendingTasks();
    EXPECT_EQ(cache->getKeySnapshot(/*limit=*/16).version, before_watermark_retry.version);
    EXPECT_EQ(find.matched_node->group_slots.front().device_blocks, (BlockIndicesType{published_target}));
    EXPECT_EQ(cache->evictForTag(config.tagForGroup(0), 1), 0);

    allocator_->free(FreeInfo{resource, token_ids});
    registry->commit_callback_ = std::move(original_commit);
}

TEST_F(SingleTypeKVCacheAllocatorTest, PrefixReuseDisabledSkipsMatchAndInsert) {
    auto config   = createSingleTypeTestConfig(/*layer_num=*/4, /*block_num=*/12, /*seq_size_per_block=*/4);
    auto policies = config.groupPoliciesSnapshot();
    ASSERT_EQ(policies.size(), 1u);
    policies[0].enable_prefix_reuse = false;
    config.setGroupPolicies(policies);

    allocator_ = std::make_shared<TestSingleTypeKVCacheAllocator>(config);
    ASSERT_TRUE(allocator_->init());
    ASSERT_NE(allocator_->blockTreeCacheOwner(), nullptr);
    EXPECT_TRUE(allocator_->blockTreeCacheOwner()->componentGroups().empty());

    auto hit_resource = createBatchKVCacheResource(/*batch_size=*/1, config);
    hit_resource->setBatchCacheKeys(0, CacheKeysType{100, 101, 102, 103, 200});
    auto hit_tokens = createCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/20, /*seq_size_per_block=*/4);

    MallocInfo hit_malloc_info{hit_resource, hit_tokens};
    hit_malloc_info.enable_device_cache = true;
    auto hit_result                     = allocator_->malloc(hit_malloc_info);
    ASSERT_TRUE(hit_result.success);
    EXPECT_EQ(hit_result.reuse_len, 0);
    EXPECT_EQ(hit_resource->cacheResource(0).reuseBlockNum(), 0u);

    auto insert_resource = createBatchKVCacheResource(/*batch_size=*/1, config);
    insert_resource->setBatchCacheKeys(0, CacheKeysType{300, 301});
    auto insert_tokens = createCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/8, /*seq_size_per_block=*/4);

    MallocInfo insert_malloc_info{insert_resource, insert_tokens};
    ASSERT_TRUE(allocator_->malloc(insert_malloc_info).success);
    allocator_->insertIntoCache(InsertInfo{insert_resource, insert_tokens, /*is_resident=*/false});
    EXPECT_TRUE(allocator_->blockTreeCacheOwner()->getKeySnapshot(/*limit=*/16).keys.empty());
}

// Test convert index to addr
TEST_F(SingleTypeKVCacheAllocatorTest, ConvertIndexToAddr) {
    auto config = createSingleTypeTestConfig();
    allocator_  = std::make_shared<TestSingleTypeKVCacheAllocator>(config);
    allocator_->init();

    for (int layer_id = 0; layer_id < config.layer_num; ++layer_id) {
        for (int block_id = 0; block_id < 3; ++block_id) {
            auto addr_info = allocator_->convertIndexToAddr(layer_id, block_id);
            EXPECT_NE(addr_info.kv_addr, nullptr);
        }
    }
}

TEST_F(SingleTypeKVCacheAllocatorTest, ConvertToGlobalLayerIdSingleNoMtp) {
    auto config = createSingleTypeTestConfig(/*layer_num=*/4, /*block_num=*/10, /*seq_size_per_block=*/8);
    allocator_  = std::make_shared<TestSingleTypeKVCacheAllocator>(config);

    EXPECT_EQ(allocator_->convertToGlobalLayerId(/*model_id=*/0, /*local_layer_id=*/0), 0u);
    EXPECT_EQ(allocator_->convertToGlobalLayerId(/*model_id=*/0, /*local_layer_id=*/3), 3u);
    EXPECT_EQ(allocator_->convertToGlobalLayerId(/*model_id=*/0, /*local_layer_id=*/-1),
              std::numeric_limits<uint32_t>::max());
    EXPECT_EQ(allocator_->convertToGlobalLayerId(/*model_id=*/0, /*local_layer_id=*/4),
              std::numeric_limits<uint32_t>::max());

    // no mtp sub-model
    EXPECT_EQ(allocator_->convertToGlobalLayerId(/*model_id=*/1, /*local_layer_id=*/0),
              std::numeric_limits<uint32_t>::max());
}

TEST_F(SingleTypeKVCacheAllocatorTest, ConvertToGlobalLayerIdSingleWithMtp) {
    auto config = makeMtpCacheConfigByCreateSpConfig(
        /*main_layers=*/2, /*mtp_module_num=*/2, /*block_num=*/8, /*mtp_module_layers=*/3);
    allocator_ = std::make_shared<TestSingleTypeKVCacheAllocator>(config);

    // main model: global == local
    EXPECT_EQ(allocator_->convertToGlobalLayerId(/*model_id=*/0, /*local_layer_id=*/0), 0u);
    EXPECT_EQ(allocator_->convertToGlobalLayerId(/*model_id=*/0, /*local_layer_id=*/1), 1u);
    EXPECT_EQ(allocator_->convertToGlobalLayerId(/*model_id=*/0, /*local_layer_id=*/2),
              std::numeric_limits<uint32_t>::max());

    // Global ids follow main_layers + module_index * module_layers + local_layer.
    EXPECT_EQ(allocator_->convertToGlobalLayerId(/*model_id=*/1, /*local_layer_id=*/0), 2u);
    EXPECT_EQ(allocator_->convertToGlobalLayerId(/*model_id=*/1, /*local_layer_id=*/1), 3u);
    EXPECT_EQ(allocator_->convertToGlobalLayerId(/*model_id=*/1, /*local_layer_id=*/2), 4u);
    EXPECT_EQ(allocator_->convertToGlobalLayerId(/*model_id=*/2, /*local_layer_id=*/0), 5u);
    EXPECT_EQ(allocator_->convertToGlobalLayerId(/*model_id=*/2, /*local_layer_id=*/1), 6u);
    EXPECT_EQ(allocator_->convertToGlobalLayerId(/*model_id=*/2, /*local_layer_id=*/2), 7u);
    EXPECT_EQ(allocator_->convertToGlobalLayerId(/*model_id=*/1, /*local_layer_id=*/3),
              std::numeric_limits<uint32_t>::max());
    EXPECT_EQ(allocator_->convertToGlobalLayerId(/*model_id=*/3, /*local_layer_id=*/0),
              std::numeric_limits<uint32_t>::max());
}

TEST_F(SingleTypeKVCacheAllocatorTest, MtpGlobalLayerIdRejectsInvalidModuleAndLocalIds) {
    constexpr auto invalid = std::numeric_limits<uint32_t>::max();
    EXPECT_EQ(CacheConfig::mtpGlobalLayerId(/*main=*/2, /*module=*/0, /*module_layers=*/3, /*local=*/0), 2u);
    EXPECT_EQ(CacheConfig::mtpGlobalLayerId(/*main=*/2, /*module=*/0, /*module_layers=*/3, /*local=*/2), 4u);
    EXPECT_EQ(CacheConfig::mtpGlobalLayerId(/*main=*/2, /*module=*/1, /*module_layers=*/3, /*local=*/0), 5u);
    EXPECT_EQ(CacheConfig::mtpGlobalLayerId(/*main=*/2, /*module=*/1, /*module_layers=*/3, /*local=*/2), 7u);
    EXPECT_EQ(CacheConfig::mtpGlobalLayerId(/*main=*/2, /*module=*/-1, /*module_layers=*/3, /*local=*/0), invalid);
    EXPECT_EQ(CacheConfig::mtpGlobalLayerId(/*main=*/2, /*module=*/0, /*module_layers=*/3, /*local=*/-1), invalid);
    EXPECT_EQ(CacheConfig::mtpGlobalLayerId(/*main=*/2, /*module=*/0, /*module_layers=*/3, /*local=*/3), invalid);
    EXPECT_EQ(CacheConfig::mtpGlobalLayerId(/*main=*/2, /*module=*/0, /*module_layers=*/0, /*local=*/0), invalid);
}

TEST_F(SingleTypeKVCacheAllocatorTest, SingleLayerMtpConfigSlicesDescriptorAndAttentionType) {
    auto config                                            = makeTestModelConfig(/*num_layers=*/2);
    config.kv_cache_spec_descs[0][0].tag                   = "layer0";
    config.kv_cache_spec_descs[1][0].tag                   = "layer1";
    config.hybrid_attention_config.enable_hybrid_attention = true;
    config.hybrid_attention_config.hybrid_attention_types  = {HybridAttentionType::LINEAR,
                                                              HybridAttentionType::SLIDING_WINDOW};

    const auto single_layer = makeSingleLayerMTPModelConfig(config, /*source_layer=*/1);

    ASSERT_EQ(single_layer.num_layers, 1);
    ASSERT_EQ(single_layer.kv_cache_spec_descs.size(), 1u);
    ASSERT_EQ(single_layer.kv_cache_spec_descs[0].size(), 1u);
    EXPECT_EQ(single_layer.kv_cache_spec_descs[0][0].tag, "layer1");
    ASSERT_EQ(single_layer.hybrid_attention_config.hybrid_attention_types.size(), 1u);
    EXPECT_EQ(single_layer.hybrid_attention_config.hybrid_attention_types[0], HybridAttentionType::SLIDING_WINDOW);
}

TEST_F(SingleTypeKVCacheAllocatorTest, SingleLayerMtpConfigSupportsDescriptorDrivenIndependentPools) {
    auto config                                                      = makeTestModelConfig(/*num_layers=*/2);
    config.hybrid_attention_config.enable_hybrid_attention           = true;
    config.hybrid_attention_config.enable_independent_kv_cache_pools = true;
    config.hybrid_attention_config.hybrid_attention_types            = {};
    auto second_desc                                                 = config.kv_cache_spec_descs[1][0];
    second_desc.tag                                                  = "layer1_state";
    config.kv_cache_spec_descs[1].push_back(second_desc);

    const auto single_layer = makeSingleLayerMTPModelConfig(config, /*source_layer=*/1);

    ASSERT_EQ(single_layer.num_layers, 1);
    ASSERT_EQ(single_layer.kv_cache_spec_descs.size(), 1u);
    ASSERT_EQ(single_layer.kv_cache_spec_descs[0].size(), 2u);
    EXPECT_EQ(single_layer.kv_cache_spec_descs[0][1].tag, "layer1_state");
    EXPECT_TRUE(single_layer.hybrid_attention_config.hybrid_attention_types.empty());
}

TEST_F(SingleTypeKVCacheAllocatorTest, SingleLayerMtpConfigRejectsLegacyHybridWithoutAttentionTypes) {
    auto config                                            = makeTestModelConfig(/*num_layers=*/2);
    config.hybrid_attention_config.enable_hybrid_attention = true;
    config.hybrid_attention_config.hybrid_attention_types  = {};

    EXPECT_THROW(makeSingleLayerMTPModelConfig(config, /*source_layer=*/0), std::runtime_error);
}

TEST_F(SingleTypeKVCacheAllocatorTest, ActiveMtpCacheLayoutValidationOnlyChecksModule0) {
    auto config                                            = makeTestModelConfig(/*num_layers=*/2);
    config.hybrid_attention_config.enable_hybrid_attention = true;
    config.hybrid_attention_config.hybrid_attention_types  = {HybridAttentionType::NONE, HybridAttentionType::NONE};

    auto module0 = makeSingleLayerMTPModelConfig(config, 0);
    EXPECT_NO_THROW(validateActiveMTPCacheLayout(module0));

    auto invalid_module0                = module0;
    invalid_module0.kv_cache_spec_descs = {};
    EXPECT_THROW(validateActiveMTPCacheLayout(invalid_module0), std::runtime_error);

    config.kv_cache_spec_descs.resize(1);
    config.hybrid_attention_config.hybrid_attention_types.resize(1);
    EXPECT_NO_THROW(buildMTPModuleConfigPlan(config, /*weight_count=*/2, /*gen_num_per_cycle=*/2, SP_TYPE_MTP));
}

TEST_F(SingleTypeKVCacheAllocatorTest, MtpModuleConfigPlanKeepsWeightsAndCopiesActiveCacheLayout) {
    auto config                                 = makeTestModelConfig(/*num_layers=*/2);
    config.kv_cache_spec_descs[0][0].tag        = "active";
    config.kv_cache_spec_descs[1][0].tag        = "inactive-heterogeneous";
    config.kv_cache_spec_descs[1][0].cache_type = KVCacheSpecType::MultiHeadLatentAttention;

    const auto multi_weight_plan =
        buildMTPModuleConfigPlan(config, /*weight_count=*/2, /*gen_num_per_cycle=*/2, SP_TYPE_MTP);
    ASSERT_EQ(multi_weight_plan.source_layer_indices, (std::vector<size_t>{0, 1}));
    ASSERT_EQ(multi_weight_plan.module_configs.size(), 2u);
    for (const auto& module_config : multi_weight_plan.module_configs) {
        EXPECT_EQ(module_config.num_layers, 1);
        ASSERT_EQ(module_config.kv_cache_spec_descs.size(), 1u);
        ASSERT_EQ(module_config.kv_cache_spec_descs[0].size(), config.kv_cache_spec_descs[0].size());
        EXPECT_EQ(module_config.kv_cache_spec_descs[0][0].tag, "active");
        EXPECT_EQ(module_config.kv_cache_spec_descs[0][0].cache_type, config.kv_cache_spec_descs[0][0].cache_type);
    }

    const auto reused_weight_plan =
        buildMTPModuleConfigPlan(config, /*weight_count=*/1, /*gen_num_per_cycle=*/3, SP_TYPE_MTP);
    EXPECT_EQ(reused_weight_plan.source_layer_indices, (std::vector<size_t>{0, 0, 0}));
    ASSERT_EQ(reused_weight_plan.module_configs.size(), 3u);
    for (const auto& module_config : reused_weight_plan.module_configs) {
        EXPECT_EQ(module_config.num_layers, 1);
        ASSERT_EQ(module_config.kv_cache_spec_descs.size(), 1u);
        ASSERT_EQ(module_config.kv_cache_spec_descs[0].size(), config.kv_cache_spec_descs[0].size());
        EXPECT_EQ(module_config.kv_cache_spec_descs[0][0].tag, config.kv_cache_spec_descs[0][0].tag);
        EXPECT_EQ(module_config.kv_cache_spec_descs[0][0].cache_type, config.kv_cache_spec_descs[0][0].cache_type);
    }
}

// Test convert index to buffer
TEST_F(SingleTypeKVCacheAllocatorTest, ConvertIndexToBuffer) {
    auto config = createSingleTypeTestConfig();
    allocator_  = std::make_shared<TestSingleTypeKVCacheAllocator>(config);
    allocator_->init();

    auto buffer_info = allocator_->convertIndexToBuffer(0, 0);
    ASSERT_EQ(buffer_info.size(), 1u);
    EXPECT_NE(buffer_info[0].addr, nullptr);
}

// Test layer cache base
TEST_F(SingleTypeKVCacheAllocatorTest, LayerCacheBase) {
    auto config = createSingleTypeTestConfig();
    allocator_  = std::make_shared<TestSingleTypeKVCacheAllocator>(config);
    allocator_->init();

    auto layout = allocator_->allLayerCacheBase();
    ASSERT_EQ(layout.groups().size(), 1u);
    EXPECT_EQ(layout.topology().layerGroupIdsSnapshot(), (std::vector<std::vector<int>>(4, std::vector<int>{0})));
    EXPECT_EQ(layout.topology().groupTypesSnapshot(), std::vector<CacheGroupType>{CacheGroupType::FULL});
    EXPECT_EQ(layout.topology().groupTagsSnapshot(), std::vector<std::string>{"default"});
    const auto& default_layout = layout.group("default");
    EXPECT_EQ(default_layout.size(), config.layer_num);
    EXPECT_EQ(default_layout.activeLayerCount(), config.layer_num);
    for (size_t i = 0; i < default_layout.size(); ++i) {
        ASSERT_TRUE(default_layout.hasLayer(i));
        EXPECT_GT(default_layout.at(i).kv_addr.nbytes(), 0u);
        EXPECT_EQ(layout.group(0).at(i).kv_addr.data_ptr(), default_layout.at(i).kv_addr.data_ptr());
    }
}

TEST_F(SingleTypeKVCacheAllocatorTest, ManagerLayoutsPreserveSingleTypeGroupTensorsForMainAndMtp) {
    auto config = makeMtpCacheConfigByCreateSpConfig(
        /*main_layers=*/2, /*mtp_module_num=*/2, /*block_num=*/8, /*mtp_module_layers=*/3);
    auto manager = std::make_shared<KVCacheManager>(config);
    ASSERT_TRUE(manager->init());

    const auto all_layout  = manager->allLayerCacheBase();
    const auto main_layout = manager->getMainModelCacheLayerLayout();
    ASSERT_EQ(all_layout.group("default").size(), 8u);

    auto verify_layout = [](const GroupedCacheLayerLayout& local_layout,
                            const GroupedCacheLayerLayout& all,
                            size_t                         global_begin) {
        ASSERT_EQ(local_layout.topology().groupTypesSnapshot(), std::vector<CacheGroupType>{CacheGroupType::FULL});
        ASSERT_EQ(local_layout.topology().groupTagsSnapshot(), std::vector<std::string>{"default"});
        const auto& local_group = local_layout.group("default");
        const auto& all_group   = all.group("default");
        for (size_t local_layer = 0; local_layer < local_group.size(); ++local_layer) {
            const size_t global_layer = global_begin + local_layer;
            ASSERT_TRUE(local_group.hasLayer(local_layer));
            EXPECT_EQ(local_group.at(local_layer).kv_addr.data_ptr(), all_group.at(global_layer).kv_addr.data_ptr());
            ASSERT_TRUE(local_group.at(local_layer).kv_scale_addr.defined());
        }

        torch_ext::KVCache kv_cache(local_layout);

        const auto by_tag        = kv_cache.getLayerCache(/*idx=*/0, "default");
        const auto by_sole_group = kv_cache.getLayerCache(/*idx=*/0);
        EXPECT_TRUE(by_tag.kv_cache_base.defined());
        EXPECT_TRUE(by_sole_group.kv_cache_base.defined());
        EXPECT_EQ(by_tag.kv_cache_base.data_ptr(), local_group.at(0).kv_addr.data_ptr());
        EXPECT_EQ(by_sole_group.kv_cache_base.data_ptr(), local_group.at(0).kv_addr.data_ptr());
    };

    verify_layout(main_layout, all_layout, /*global_begin=*/0);
    verify_layout(manager->getMTPModuleCacheLayerLayout(0), all_layout, /*global_begin=*/2);
    verify_layout(manager->getMTPModuleCacheLayerLayout(1), all_layout, /*global_begin=*/5);
    EXPECT_THROW(manager->getMTPModuleCacheLayerLayout(-1), std::runtime_error);
    EXPECT_THROW(manager->getMTPModuleCacheLayerLayout(2), std::runtime_error);
}

// Test block copy
TEST_F(SingleTypeKVCacheAllocatorTest, BlockCopySingle) {
    auto config = createSingleTypeTestConfig();
    allocator_  = std::make_shared<TestSingleTypeKVCacheAllocator>(config);
    ASSERT_TRUE(allocator_->init());

    const auto& pool = allocator_->getDeviceBlockPool();
    ASSERT_NE(pool, nullptr);
    const auto allocated = pool->malloc(2);
    ASSERT_TRUE(allocated.has_value());
    ASSERT_EQ(allocated->size(), 2u);
    pool->incRef(*allocated);
    const int src_block = (*allocated)[0];
    const int dst_block = (*allocated)[1];

    auto&  spec         = config.specForGroup(0);
    size_t k_block_size = spec->k_block_size();
    size_t v_block_size = spec->v_block_size();
    size_t block_size   = k_block_size + v_block_size;

    for (int layer_id = 0; layer_id < config.layer_num; ++layer_id) {
        auto src_addr = allocator_->convertIndexToAddr(layer_id, src_block);
        ASSERT_NE(src_addr.kv_addr, nullptr) << "KV addr is null for layer " << layer_id << ", block " << src_block;

        auto dst_addr = allocator_->convertIndexToAddr(layer_id, dst_block);
        ASSERT_NE(dst_addr.kv_addr, nullptr) << "KV addr is null for layer " << layer_id << ", block " << dst_block;

        std::vector<uint8_t> pattern(block_size);
        for (size_t i = 0; i < k_block_size; ++i) {
            pattern[i] = static_cast<uint8_t>((layer_id * 100 + src_block * 10 + i) % 256);
        }
        for (size_t i = 0; i < v_block_size; ++i) {
            pattern[k_block_size + i] = static_cast<uint8_t>((layer_id * 100 + src_block * 10 + i + 128) % 256);
        }
        writeDeviceBytes(src_addr.kv_addr, pattern);
    }

    EXPECT_NO_THROW(allocator_->blockCopy(src_block, dst_block));

    for (int layer_id = 0; layer_id < config.layer_num; ++layer_id) {
        auto src_addr = allocator_->convertIndexToAddr(layer_id, src_block);
        auto dst_addr = allocator_->convertIndexToAddr(layer_id, dst_block);

        EXPECT_EQ(readDeviceBytes(dst_addr.kv_addr, block_size), readDeviceBytes(src_addr.kv_addr, block_size))
            << "cache mismatch at layer " << layer_id;
    }
    pool->decRef(*allocated);
}

TEST_F(SingleTypeKVCacheAllocatorTest, BlockBatchCopyVector) {
    auto config = createSingleTypeTestConfig();
    allocator_  = std::make_shared<TestSingleTypeKVCacheAllocator>(config);
    ASSERT_TRUE(allocator_->init());

    const auto& pool = allocator_->getDeviceBlockPool();
    ASSERT_NE(pool, nullptr);
    const auto allocated = pool->malloc(6);
    ASSERT_TRUE(allocated.has_value());
    ASSERT_EQ(allocated->size(), 6u);
    pool->incRef(*allocated);

    std::vector<BlockIdPair> copy_mapping = {
        {(*allocated)[0], (*allocated)[1]}, {(*allocated)[2], (*allocated)[3]}, {(*allocated)[4], (*allocated)[5]}};

    auto&  spec         = config.specForGroup(0);
    size_t k_block_size = spec->k_block_size();
    size_t v_block_size = spec->v_block_size();
    size_t block_size   = k_block_size + v_block_size;

    for (const auto& pair : copy_mapping) {
        for (int layer_id = 0; layer_id < config.layer_num; ++layer_id) {
            auto src_addr = allocator_->convertIndexToAddr(layer_id, pair.src);
            ASSERT_NE(src_addr.kv_addr, nullptr);

            std::vector<uint8_t> pattern(block_size);
            for (size_t i = 0; i < k_block_size; ++i) {
                pattern[i] = static_cast<uint8_t>((layer_id * 100 + pair.src * 10 + i) % 256);
            }
            for (size_t i = 0; i < v_block_size; ++i) {
                pattern[k_block_size + i] = static_cast<uint8_t>((layer_id * 100 + pair.src * 10 + i + 128) % 256);
            }
            writeDeviceBytes(src_addr.kv_addr, pattern);
        }
    }

    EXPECT_NO_THROW(allocator_->blockBatchCopy(copy_mapping));

    // Verify data correctness for each block
    for (const auto& pair : copy_mapping) {
        for (int layer_id = 0; layer_id < config.layer_num; ++layer_id) {
            auto src_addr = allocator_->convertIndexToAddr(layer_id, pair.src);
            auto dst_addr = allocator_->convertIndexToAddr(layer_id, pair.dst);

            EXPECT_EQ(readDeviceBytes(dst_addr.kv_addr, block_size), readDeviceBytes(src_addr.kv_addr, block_size))
                << "cache mismatch at block pair (" << pair.src << "->" << pair.dst << "), layer " << layer_id;
        }
    }
    pool->decRef(*allocated);
}

TEST_F(SingleTypeKVCacheAllocatorTest, BlockBatchCopyEmpty) {
    auto config = createSingleTypeTestConfig();
    allocator_  = std::make_shared<TestSingleTypeKVCacheAllocator>(config);
    allocator_->init();

    std::vector<BlockIdPair> empty_mapping;

    EXPECT_NO_THROW(allocator_->blockBatchCopy(empty_mapping));
}

TEST_F(SingleTypeKVCacheAllocatorTest, BlockBatchCopyCopiesCompleteSparseIndexerStride) {
    auto model_config                         = makeTestModelConfig(/*num_layers=*/1);
    model_config.attn_config.is_sparse        = true;
    model_config.attn_config.indexer_head_dim = 256;

    ParallelismConfig parallelism_config;
    parallelism_config.tp_size = 1;
    auto config                = SingleConfigCreator::createSingleConfig(model_config, parallelism_config);
    config.block_num           = 5;

    ASSERT_TRUE(config.is_sparse);
    ASSERT_GT(config.kv_scale_stride_bytes, 0u);
    ASSERT_EQ(config.kv_scale_stride_bytes, config.kvScaleStrideBytesForGroup(0));

    allocator_ = std::make_shared<TestSingleTypeKVCacheAllocator>(config);
    ASSERT_TRUE(allocator_->init());

    const auto& pool = allocator_->getDeviceBlockPool();
    ASSERT_NE(pool, nullptr);
    const auto allocated = pool->malloc(4);
    ASSERT_TRUE(allocated.has_value());
    ASSERT_EQ(allocated->size(), 4u);
    pool->incRef(*allocated);

    const auto stride   = config.kv_scale_stride_bytes;
    auto       snapshot = [&]() {
        std::vector<std::vector<uint8_t>> blocks(allocated->size());
        for (size_t index = 0; index < allocated->size(); ++index) {
            auto addr = allocator_->convertIndexToAddr(/*layer_id=*/0, (*allocated)[index]);
            EXPECT_NE(addr.kv_scale_addr, nullptr);
            blocks[index] = readDeviceBytes(addr.kv_scale_addr, stride);
        }
        return blocks;
    };
    auto verify = [&](const std::vector<std::vector<uint8_t>>& expected) {
        for (size_t index = 0; index < allocated->size(); ++index) {
            auto addr = allocator_->convertIndexToAddr(/*layer_id=*/0, (*allocated)[index]);
            EXPECT_EQ(readDeviceBytes(addr.kv_scale_addr, stride), expected[index])
                << "sparse indexer mismatch at block " << (*allocated)[index];
        }
    };

    for (size_t index = 0; index < allocated->size(); ++index) {
        const BlockIdxType block = (*allocated)[index];
        auto               addr  = allocator_->convertIndexToAddr(/*layer_id=*/0, block);
        ASSERT_NE(addr.kv_scale_addr, nullptr);
        std::vector<uint8_t> pattern(stride);
        for (size_t offset = 0; offset < stride; ++offset) {
            pattern[offset] = static_cast<uint8_t>((block * 67 + offset) % 251);
        }
        writeDeviceBytes(addr.kv_scale_addr, pattern);
    }

    const auto initial = snapshot();
    EXPECT_NO_THROW(allocator_->blockBatchCopy(std::vector<BlockIdPair>{}));
    verify(initial);

    EXPECT_NO_THROW(allocator_->blockBatchCopy(std::vector<BlockIdPair>{{(*allocated)[0], (*allocated)[1]}}));
    auto after_single = initial;
    after_single[1]   = initial[0];
    verify(after_single);

    const int last_block = allocated->back();
    EXPECT_NO_THROW(allocator_->blockBatchCopy(std::vector<BlockIdPair>{{(*allocated)[1], last_block}}));
    auto after_last   = after_single;
    after_last.back() = after_single[1];
    verify(after_last);
    pool->decRef(*allocated);
}

TEST_F(SingleTypeKVCacheAllocatorTest, BlockBatchCopyPointers) {
    auto config = createSingleTypeTestConfig();
    allocator_  = std::make_shared<TestSingleTypeKVCacheAllocator>(config);
    ASSERT_TRUE(allocator_->init());

    const auto& pool = allocator_->getDeviceBlockPool();
    ASSERT_NE(pool, nullptr);
    const auto allocated = pool->malloc(4);
    ASSERT_TRUE(allocated.has_value());
    ASSERT_EQ(allocated->size(), 4u);
    pool->incRef(*allocated);

    BlockIdPair pairs[] = {{(*allocated)[0], (*allocated)[1]}, {(*allocated)[2], (*allocated)[3]}};

    auto&  spec         = config.specForGroup(0);
    size_t k_block_size = spec->k_block_size();
    size_t v_block_size = spec->v_block_size();
    size_t block_size   = k_block_size + v_block_size;

    for (const auto& pair : pairs) {
        for (int layer_id = 0; layer_id < config.layer_num; ++layer_id) {
            auto                 src_addr = allocator_->convertIndexToAddr(layer_id, pair.src);
            std::vector<uint8_t> pattern(block_size);
            for (size_t i = 0; i < k_block_size; ++i) {
                pattern[i] = static_cast<uint8_t>((layer_id * 50 + pair.src * 20 + i) % 256);
            }
            for (size_t i = 0; i < v_block_size; ++i) {
                pattern[k_block_size + i] = static_cast<uint8_t>((layer_id * 50 + pair.src * 20 + i + 64) % 256);
            }
            writeDeviceBytes(src_addr.kv_addr, pattern);
        }
    }

    EXPECT_NO_THROW(allocator_->blockBatchCopy(pairs, pairs + 2));

    for (const auto& pair : pairs) {
        for (int layer_id = 0; layer_id < config.layer_num; ++layer_id) {
            auto src_addr = allocator_->convertIndexToAddr(layer_id, pair.src);
            auto dst_addr = allocator_->convertIndexToAddr(layer_id, pair.dst);

            EXPECT_EQ(readDeviceBytes(dst_addr.kv_addr, block_size), readDeviceBytes(src_addr.kv_addr, block_size))
                << "cache mismatch for block pair (" << pair.src << "->" << pair.dst << "), layer " << layer_id;
        }
    }
    pool->decRef(*allocated);
}

TEST_F(SingleTypeKVCacheAllocatorTest, BlockBatchCopyBuffer) {
    auto config = createSingleTypeTestConfig();
    allocator_  = std::make_shared<TestSingleTypeKVCacheAllocator>(config);
    ASSERT_TRUE(allocator_->init());

    const auto& pool = allocator_->getDeviceBlockPool();
    ASSERT_NE(pool, nullptr);
    const auto allocated = pool->malloc(6);
    ASSERT_TRUE(allocated.has_value());
    ASSERT_EQ(allocated->size(), 6u);
    pool->incRef(*allocated);

    std::vector<int32_t> data(allocated->begin(), allocated->end());
    auto                 tensor = torch::from_blob(data.data(), {3, 2}, torch::kInt32).clone();

    auto&  spec         = config.specForGroup(0);
    size_t k_block_size = spec->k_block_size();
    size_t v_block_size = spec->v_block_size();
    size_t block_size   = k_block_size + v_block_size;

    for (size_t i = 0; i < data.size(); i += 2) {
        int src_block = data[i];
        for (int layer_id = 0; layer_id < config.layer_num; ++layer_id) {
            auto                 src_addr = allocator_->convertIndexToAddr(layer_id, src_block);
            std::vector<uint8_t> pattern(block_size);
            for (size_t j = 0; j < k_block_size; ++j) {
                pattern[j] = static_cast<uint8_t>((layer_id * 70 + src_block * 15 + j) % 256);
            }
            for (size_t j = 0; j < v_block_size; ++j) {
                pattern[k_block_size + j] = static_cast<uint8_t>((layer_id * 70 + src_block * 15 + j + 96) % 256);
            }
            writeDeviceBytes(src_addr.kv_addr, pattern);
        }
    }

    EXPECT_NO_THROW(allocator_->blockBatchCopy(tensor));

    for (size_t i = 0; i < data.size(); i += 2) {
        int src_block = data[i];
        int dst_block = data[i + 1];
        for (int layer_id = 0; layer_id < config.layer_num; ++layer_id) {
            auto src_addr = allocator_->convertIndexToAddr(layer_id, src_block);
            auto dst_addr = allocator_->convertIndexToAddr(layer_id, dst_block);

            EXPECT_EQ(readDeviceBytes(dst_addr.kv_addr, block_size), readDeviceBytes(src_addr.kv_addr, block_size))
                << "cache mismatch for block pair (" << src_block << "->" << dst_block << "), layer " << layer_id;
        }
    }
    pool->decRef(*allocated);
}

// Test getter methods
TEST_F(SingleTypeKVCacheAllocatorTest, FreeBlocksNums) {
    auto config = createSingleTypeTestConfig();
    allocator_  = std::make_shared<TestSingleTypeKVCacheAllocator>(config);
    allocator_->init();

    EXPECT_EQ(allocator_->freeBlocksNum(), config.block_num - 1);  // reserve 1 block
}

TEST_F(SingleTypeKVCacheAllocatorTest, IncrKVCacheRefReferencesMatchedBlocksOnly) {
    auto config = createSingleTypeTestConfig(/*layer_num=*/4, /*block_num=*/10, /*seq_size_per_block=*/8);
    allocator_  = std::make_shared<TestSingleTypeKVCacheAllocator>(config, AllocationType::HOST);
    ASSERT_TRUE(allocator_->init());

    auto block_pool = allocator_->getDeviceBlockPool();
    ASSERT_NE(block_pool, nullptr);

    const size_t total_free_before = allocator_->freeBlocksNum();
    auto         blocks            = block_pool->malloc(4).value();
    ASSERT_EQ(blocks.size(), 4);
    block_pool->incRef(blocks);
    EXPECT_EQ(allocator_->freeBlocksNum(), total_free_before - 4);

    KVCacheResource resource;
    resource.initGroups(config.topologyPtr());

    resource.cacheKeys() = CacheKeysType{100, 101, 102, 103};
    resource.mutableBlockIds(0).assign(BlockIndicesType{blocks[0], blocks[1], 0, blocks[2]});
    resource.setDeviceReuseBlockNum(3);

    // Reference keys: 101(pos1)->blocks[1], 102(pos2)->0(ignored), 103(pos3)->blocks[2]
    auto ref_resource = allocator_->incrKVCacheRef(resource, CacheKeysType{101, 999, 102, 103});
    ASSERT_NE(ref_resource, nullptr);
    // Validate: incrKVCacheRef propagates reuseBlockNum to returned resource.
    EXPECT_EQ(ref_resource->reuseBlockNum(), resource.reuseBlockNum());

    block_pool->decRef(blocks);
    EXPECT_EQ(allocator_->freeBlocksNum(), total_free_before - 2);  // blocks[1] & blocks[2] are still referenced
    // incrKVCacheRef returns a resource with a custom deleter that calls decrKVCacheRef().
    // Release it to drop ref-counts and unblock the pending frees.
    ref_resource.reset();
    EXPECT_EQ(allocator_->freeBlocksNum(), total_free_before);
}

TEST_F(SingleTypeKVCacheAllocatorTest, IncrKVCacheRefPreservesConnectorDummyTail) {
    auto config = createSingleTypeTestConfig(/*layer_num=*/4, /*block_num=*/10, /*seq_size_per_block=*/8);
    allocator_  = std::make_shared<TestSingleTypeKVCacheAllocator>(config, AllocationType::HOST);
    ASSERT_TRUE(allocator_->init());

    auto block_pool = allocator_->getDeviceBlockPool();
    ASSERT_NE(block_pool, nullptr);

    const size_t total_free_before = allocator_->freeBlocksNum();
    auto         blocks            = block_pool->malloc(2).value();
    ASSERT_EQ(blocks.size(), 2);
    block_pool->incRef(blocks);

    KVCacheResource resource;
    resource.initGroups(config.topologyPtr());
    resource.cacheKeys() = CacheKeysType{101, 103, 999};
    resource.rebuildLinearBlockDependencies();
    resource.setLastBlockAligned(false);
    resource.mutableBlockIds(0).assign(BlockIndicesType{blocks[0], blocks[1]});

    auto ref_resource = allocator_->incrKVCacheRef(resource, CacheKeysType{101, 103, 999}, /*is_connector=*/true);
    ASSERT_NE(ref_resource, nullptr);
    EXPECT_FALSE(ref_resource->lastBlockAligned());
    EXPECT_EQ(ref_resource->cacheKeys(), (CacheKeysType{101, 103, 999}));
    EXPECT_EQ(ref_resource->blocks(0), (BlockIndicesType{blocks[0], blocks[1], NULL_BLOCK_IDX}));

    block_pool->decRef(blocks);
    EXPECT_EQ(allocator_->freeBlocksNum(), total_free_before - 2);

    ref_resource.reset();
    EXPECT_EQ(allocator_->freeBlocksNum(), total_free_before);
}

TEST_F(SingleTypeKVCacheAllocatorTest, IncrKVCacheRefEmptyInputNoEffect) {
    auto config = createSingleTypeTestConfig(/*layer_num=*/4, /*block_num=*/10, /*seq_size_per_block=*/8);
    allocator_  = std::make_shared<TestSingleTypeKVCacheAllocator>(config, AllocationType::HOST);
    ASSERT_TRUE(allocator_->init());

    auto block_pool = allocator_->getDeviceBlockPool();
    ASSERT_NE(block_pool, nullptr);

    const size_t total_free_before = allocator_->freeBlocksNum();
    auto         blocks            = block_pool->malloc(2).value();
    ASSERT_EQ(blocks.size(), 2);
    block_pool->incRef(blocks);
    EXPECT_EQ(allocator_->freeBlocksNum(), total_free_before - 2);

    KVCacheResource resource;
    resource.initGroups(config.topologyPtr());
    resource.cacheKeys() = CacheKeysType{100, 101};
    resource.mutableBlockIds(0).assign(BlockIndicesType{blocks[0], blocks[1]});

    auto ref_resource = allocator_->incrKVCacheRef(resource, CacheKeysType{});
    ASSERT_EQ(ref_resource, nullptr);

    block_pool->decRef(blocks);
    EXPECT_EQ(allocator_->freeBlocksNum(), total_free_before);
}

TEST_F(SingleTypeKVCacheAllocatorTest, TotalBlocksNums) {
    auto config = createSingleTypeTestConfig(4, 20);
    allocator_  = std::make_shared<TestSingleTypeKVCacheAllocator>(config);
    allocator_->init();

    EXPECT_EQ(allocator_->totalBlocksNum(), 20 - 1);
}

TEST_F(SingleTypeKVCacheAllocatorTest, MaxSeqLen) {
    auto config = createSingleTypeTestConfig(4, 10, 8);
    allocator_  = std::make_shared<TestSingleTypeKVCacheAllocator>(config);
    allocator_->init();

    EXPECT_EQ(allocator_->maxAvailableTokensNum(), (10 - 1) * 8);  // block_num * seq_size_per_block
}

TEST_F(SingleTypeKVCacheAllocatorTest, CapacityAndNeedBlocksUseCPVirtualBlockSize) {
    auto config = createSingleTypeTestConfig(/*layer_num=*/4, /*block_num=*/10, /*seq_size_per_block=*/8);
    allocator_  = std::make_shared<TestSingleTypeKVCacheAllocator>(config);
    ASSERT_TRUE(allocator_->init());

    allocator_->setCPSlotMapper(std::make_shared<CPSlotMapper>(/*cp_rank=*/0, /*cp_size=*/2, /*block_size=*/8));

    EXPECT_EQ(allocator_->maxAvailableTokensNum(), (10u - 1u) * 16u);
    EXPECT_EQ(allocator_->availableTokensNum(), (10u - 1u) * 16u);

    auto batch_resource = createBatchKVCacheResource(/*batch_size=*/1, config);
    EXPECT_EQ(allocator_->singleBatchNeedBlocks(batch_resource, /*seq_len=*/65, /*reserve_step=*/0), 5);
}

// Test boundary conditions

TEST_F(SingleTypeKVCacheAllocatorTest, MallocWithZeroSeqLength) {
    auto config = createSingleTypeTestConfig();
    allocator_  = std::make_shared<TestSingleTypeKVCacheAllocator>(config);
    allocator_->init();

    auto batch_resource     = createBatchKVCacheResource(1, config);
    auto complete_token_ids = createCompleteTokenIds(1, 0);

    MallocInfo malloc_info{batch_resource, complete_token_ids};
    auto       result = allocator_->malloc(malloc_info);
    // not crash
    EXPECT_TRUE(result.success || !result.success);
}

TEST_F(SingleTypeKVCacheAllocatorTest, FreeEmptyBatchResource) {
    auto config = createSingleTypeTestConfig();
    allocator_  = std::make_shared<TestSingleTypeKVCacheAllocator>(config);
    allocator_->init();

    auto batch_resource     = createBatchKVCacheResource(0, config);
    auto complete_token_ids = createCompleteTokenIds(0, 0);

    FreeInfo free_info{batch_resource, complete_token_ids};
    allocator_->free(free_info);
}

TEST_F(SingleTypeKVCacheAllocatorTest, InitMallocRollbackWhenInitMallocForCommonLenFails) {
    auto config = createSingleTypeTestConfig(/*layer_num=*/4, /*block_num=*/6, /*seq_size_per_block=*/4);
    allocator_  = std::make_shared<TestSingleTypeKVCacheAllocator>(config, AllocationType::HOST);
    ASSERT_TRUE(allocator_->init());

    // System-prompt residency is represented by retained request ownership. Hold
    // four physical blocks so the failing allocation cannot reclaim them.
    auto block_pool = allocator_->getDeviceBlockPool();
    ASSERT_NE(block_pool, nullptr);
    auto resident_request_holds = block_pool->malloc(4).value();
    ASSERT_EQ(resident_request_holds.size(), 4u);
    block_pool->incRef(resident_request_holds);
    ASSERT_EQ(allocator_->freeBlocksNum(), 1u);

    auto batch_resource = createBatchKVCacheResource(/*batch_size=*/2, config);
    batch_resource->setBatchCacheKeys(0, CacheKeysType{100, 101});  // match_keys -> {100}
    batch_resource->setBatchCacheKeys(1, CacheKeysType{200, 201});

    auto token_ids = createCompleteTokenIds(/*batch_size=*/2, /*seq_length=*/13, /*seq_size_per_block=*/4);

    const size_t free_before_fail = allocator_->freeBlocksNum();
    ASSERT_EQ(free_before_fail, 1u);

    MallocInfo malloc_info{batch_resource, token_ids};
    malloc_info.enable_device_cache = true;
    auto result                     = allocator_->malloc(malloc_info);
    EXPECT_FALSE(result.success);

    // KVCacheAllocator::initMalloc should call free() to rollback any referenced/allocated blocks.
    EXPECT_EQ(batch_resource->curBlocksNum(), 0);
    EXPECT_EQ(batch_resource->blocksNum(0, 0), 0);
    EXPECT_EQ(batch_resource->blocksNum(1, 0), 0);

    EXPECT_EQ(allocator_->freeBlocksNum(), free_before_fail);

    block_pool->decRef(resident_request_holds);
    EXPECT_EQ(allocator_->freeBlocksNum(), 5u);
}

// Test rollback logic in incrMalloc
TEST_F(SingleTypeKVCacheAllocatorTest, IncrMallocRollback) {
    // Create a config with limited blocks to trigger rollback
    auto config = createSingleTypeTestConfig(4, 8, 4);  // 8 blocks, 4 seq per block
    allocator_  = std::make_shared<TestSingleTypeKVCacheAllocator>(config);
    allocator_->init();

    size_t initial_free_blocks = allocator_->freeBlocksNum();
    EXPECT_EQ(initial_free_blocks, 7);

    int  batch_size         = 3;
    auto batch_resource     = createBatchKVCacheResource(batch_size, config);
    auto complete_token_ids = createCompleteTokenIds(batch_size, 4, 4);  // 4 seq length = 1 block per batch

    // First, do a common allocation for all batches (1 block each)
    MallocInfo common_malloc_info{batch_resource, complete_token_ids};
    auto       common_result = allocator_->initMallocForCommonLen(common_malloc_info);
    EXPECT_TRUE(common_result.success);

    // 1 block allocated and shared by all batches
    size_t after_common_free_blocks = allocator_->freeBlocksNum();
    EXPECT_EQ(after_common_free_blocks, 6);

    // Verify each batch has 1 block
    for (int i = 0; i < batch_size; ++i) {
        EXPECT_EQ(batch_resource->blocksNum(i, 0), 1);
    }

    // update complete_token_ids to 16 tokens
    complete_token_ids->setSeqLength(16);
    MallocInfo incr_malloc_info{batch_resource, complete_token_ids};

    auto incr_result = allocator_->incrMalloc(incr_malloc_info);
    EXPECT_FALSE(incr_result.success);

    size_t after_rollback_free_blocks = allocator_->freeBlocksNum();
    EXPECT_EQ(after_rollback_free_blocks, 6);

    for (int i = 0; i < batch_size; ++i) {
        EXPECT_EQ(batch_resource->blocksNum(i, 0), 1);
    }

    // Verify that no extra blocks were allocated and left unfreed
    // If rollback didn't work properly, we might have partially allocated blocks
}

TEST_F(SingleTypeKVCacheAllocatorTest, InitMallocRollbackWhenIncrMallocFails) {
    auto config = createSingleTypeTestConfig(/*layer_num=*/4, /*block_num=*/5, /*seq_size_per_block=*/8);
    allocator_  = std::make_shared<TestSingleTypeKVCacheAllocator>(config, AllocationType::HOST);
    ASSERT_TRUE(allocator_->init());

    const size_t free_before = allocator_->freeBlocksNum();
    ASSERT_EQ(free_before, 4u);

    auto batch_resource     = createBatchKVCacheResource(/*batch_size=*/3, config);
    auto complete_token_ids = createCompleteTokenIds(/*batch_size=*/3, /*seq_length=*/17, /*seq_size_per_block=*/8);

    MallocInfo malloc_info{batch_resource, complete_token_ids};
    malloc_info.enable_device_cache = false;
    auto result                     = allocator_->malloc(malloc_info);
    EXPECT_FALSE(result.success);

    // KVCacheAllocator::initMalloc should call free() to clear shared blocks after incrMalloc fails.
    EXPECT_EQ(batch_resource->curBlocksNum(), 0);
    for (int i = 0; i < batch_resource->batchSize(); ++i) {
        EXPECT_EQ(batch_resource->blocksNum(i, 0), 0);
    }

    EXPECT_EQ(allocator_->freeBlocksNum(), free_before);
}

// ==================== Stress tests ====================

TEST_F(SingleTypeKVCacheAllocatorTest, MixedOperations) {
    auto config = createSingleTypeTestConfig(4, 30);
    allocator_  = std::make_shared<TestSingleTypeKVCacheAllocator>(config);
    allocator_->init();

    std::vector<BatchKVCacheResourcePtr> resources;
    std::vector<CompleteTokenIdsPtr>     token_ids_list;

    for (int i = 0; i < 5; ++i) {
        auto batch_resource     = createBatchKVCacheResource(2, config);
        auto complete_token_ids = createCompleteTokenIds(2, 16);

        MallocInfo malloc_info{batch_resource, complete_token_ids};
        auto       result = allocator_->malloc(malloc_info);
        EXPECT_TRUE(result.success);

        resources.push_back(batch_resource);
        token_ids_list.push_back(complete_token_ids);
    }

    for (int i = 0; i < 3; ++i) {
        FreeInfo free_info{resources[i], token_ids_list[i]};
        allocator_->free(free_info);
    }

    for (int i = 0; i < 2; ++i) {
        auto batch_resource     = createBatchKVCacheResource(1, config);
        auto complete_token_ids = createCompleteTokenIds(1, 16);

        MallocInfo malloc_info{batch_resource, complete_token_ids};
        auto       result = allocator_->malloc(malloc_info);
        EXPECT_TRUE(result.success);
    }

    EXPECT_GT(allocator_->freeBlocksNum(), 0);
}

TEST_F(SingleTypeKVCacheAllocatorTest, EstimatePeakNeedBlocks) {
    // seq_size_per_block=4, block_num=10
    auto config = createSingleTypeTestConfig(/*layer_num=*/1, /*block_num=*/10, /*seq_size_per_block=*/4);
    allocator_  = std::make_shared<SingleTypeKVCacheAllocator>(config);
    ASSERT_TRUE(allocator_->init());

    // New resource (no blocks allocated): ceil((8+100)/4) - 0 = 27
    auto new_res = createBatchKVCacheResource(1, config);
    EXPECT_EQ(estimateBatchPeakForSingleSequence(*allocator_, new_res, 8, 100, 0, /*enable_reuse_cache=*/false), 27);

    // With reserve_step=3: ceil((8+100+3)/4) - 0 = 28
    EXPECT_EQ(estimateBatchPeakForSingleSequence(*allocator_, new_res, 8, 100, 3, /*enable_reuse_cache=*/false), 28);

    // Allocate blocks for seq_len=8 (2 blocks)
    auto       token_ids = createCompleteTokenIds(1, /*seq_length=*/8, /*seq_size_per_block=*/4);
    MallocInfo mi{new_res, token_ids};
    auto       result = allocator_->malloc(mi);
    ASSERT_TRUE(result.success);
    ASSERT_EQ(new_res->blocksNum(0, 0), 2);

    // After malloc: ceil((8+0)/4) - 2 = 0
    EXPECT_EQ(estimateBatchPeakForSingleSequence(*allocator_, new_res, 8, 0, 0, /*enable_reuse_cache=*/false), 0);

    // remaining=4: ceil((8+4)/4) - 2 = 1
    EXPECT_EQ(estimateBatchPeakForSingleSequence(*allocator_, new_res, 8, 4, 0, /*enable_reuse_cache=*/false), 1);

    // remaining=4, reserve=4: ceil((8+4+4)/4) - 2 = 2
    EXPECT_EQ(estimateBatchPeakForSingleSequence(*allocator_, new_res, 8, 4, 4, /*enable_reuse_cache=*/false), 2);
}

TEST_F(SingleTypeKVCacheAllocatorTest, EstimateBatchPeakNeedBlocksAccountsForNonEmptyTargetWidth) {
    auto config = createSingleTypeTestConfig(/*layer_num=*/1, /*block_num=*/16, /*seq_size_per_block=*/4);
    allocator_  = std::make_shared<SingleTypeKVCacheAllocator>(config);
    ASSERT_TRUE(allocator_->init());

    auto resource = createBatchKVCacheResource(/*batch_size=*/2, config);
    resource->setBatchBlocks(/*batch_id=*/0, /*group_id=*/0, {1, 2, 3});
    resource->setBatchBlocks(/*batch_id=*/1, /*group_id=*/0, {1, 2, 4});

    // Two common blocks are shared. Each current batch owns one private tail block.
    EXPECT_EQ(allocator_->estimateBatchPeakNeedBlocks(resource,
                                                      /*seq_len=*/9,
                                                      /*common_seq_len=*/8,
                                                      /*remaining_tokens=*/0,
                                                      /*reserve_step=*/0,
                                                      /*enable_reuse_cache=*/false,
                                                      /*target_batch_size=*/2),
              0);

    // Expanding the partial tail from two sequences to four needs two physical copies.
    EXPECT_EQ(allocator_->estimateBatchPeakNeedBlocks(resource,
                                                      /*seq_len=*/9,
                                                      /*common_seq_len=*/8,
                                                      /*remaining_tokens=*/0,
                                                      /*reserve_step=*/0,
                                                      /*enable_reuse_cache=*/false,
                                                      /*target_batch_size=*/4),
              2);

    // Existing resources need one additional block for each current batch.
    EXPECT_EQ(allocator_->estimateBatchPeakNeedBlocks(resource,
                                                      /*seq_len=*/9,
                                                      /*common_seq_len=*/8,
                                                      /*remaining_tokens=*/4,
                                                      /*reserve_step=*/0,
                                                      /*enable_reuse_cache=*/false,
                                                      /*target_batch_size=*/2),
              2);

    // Charge four future blocks plus two copies of the current partial tail.
    EXPECT_EQ(allocator_->estimateBatchPeakNeedBlocks(resource,
                                                      /*seq_len=*/9,
                                                      /*common_seq_len=*/8,
                                                      /*remaining_tokens=*/4,
                                                      /*reserve_step=*/0,
                                                      /*enable_reuse_cache=*/false,
                                                      /*target_batch_size=*/4),
              6);

    // An aligned tail remains shared while the batch expands.
    EXPECT_EQ(allocator_->estimateBatchPeakNeedBlocks(resource,
                                                      /*seq_len=*/12,
                                                      /*common_seq_len=*/8,
                                                      /*remaining_tokens=*/0,
                                                      /*reserve_step=*/0,
                                                      /*enable_reuse_cache=*/false,
                                                      /*target_batch_size=*/4),
              0);

    auto empty_resource = createBatchKVCacheResource(/*batch_size=*/1, config);
    // Empty resource: two prompt blocks are shared and one future block is private per target batch.
    EXPECT_EQ(allocator_->estimateBatchPeakNeedBlocks(empty_resource,
                                                      /*seq_len=*/8,
                                                      /*common_seq_len=*/8,
                                                      /*remaining_tokens=*/3,
                                                      /*reserve_step=*/0,
                                                      /*enable_reuse_cache=*/false,
                                                      /*target_batch_size=*/4),
              6);
}

TEST_F(SingleTypeKVCacheAllocatorTest, EstimateBatchPeakCoversPartialTailCopiesAtExactCapacity) {
    auto config = createSingleTypeTestConfig(/*layer_num=*/1, /*block_num=*/6, /*seq_size_per_block=*/4);
    allocator_  = std::make_shared<SingleTypeKVCacheAllocator>(config);
    ASSERT_TRUE(allocator_->init());

    auto resource  = createBatchKVCacheResource(/*batch_size=*/1, config);
    auto token_ids = createCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/5, /*seq_size_per_block=*/4);
    ASSERT_TRUE(allocator_->malloc(MallocInfo{resource, token_ids}).success);
    ASSERT_EQ(allocator_->freeBlocksNum(), 3);

    // The two future KV steps fit in the current tail, but a delayed 1-to-4 beam expansion copies it three times.
    EXPECT_EQ(allocator_->estimateBatchPeakNeedBlocks(resource,
                                                      /*seq_len=*/5,
                                                      /*common_seq_len=*/4,
                                                      /*remaining_tokens=*/2,
                                                      /*reserve_step=*/0,
                                                      /*enable_reuse_cache=*/false,
                                                      /*target_batch_size=*/4),
              3);

    std::vector<TaggedBlockIdPair> block_update_mapping;
    ASSERT_TRUE(allocator_->updateKVBlock(
        resource, /*block_src_batch=*/{0, 0, 0, 0}, /*copy_last_block=*/true, block_update_mapping));
    EXPECT_EQ(resource->batchSize(), 4);
    EXPECT_EQ(block_update_mapping.size(), 3);
    EXPECT_EQ(allocator_->freeBlocksNum(), 0);
}

}  // namespace test
}  // namespace rtp_llm

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
