#include <gtest/gtest.h>
#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <set>
#include <dirent.h>
#include <unistd.h>
#include <utility>
#include <vector>
#include <torch/torch.h>
#include "rtp_llm/cpp/cache/AsyncContext.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/cache/allocator/SingleTypeKVCacheAllocator.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCacheFactory.h"
#include "rtp_llm/cpp/cache/block_tree_cache/FullComponentGroup.h"
#include "rtp_llm/cpp/cache/block_tree_cache/host/HostBlockPool.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/config_creator/CacheConfigCreator.h"
#include "rtp_llm/cpp/cache/CPSlotMapper.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/cpp/cache/test/BlockPoolTestHelper.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"

namespace rtp_llm {
namespace test {

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
    m.attn_config.kv_cache_dtype   = KvCacheDataType::INT8;
    m.attn_config.kv_lora_rank     = 0;
    m.attn_config.rope_head_dim    = 0;
    m.attn_config.head_num         = 2;
    setDefaultKvCacheSpec(m);
    return m;
}

static rtp_llm::CacheConfig
makeMtpCacheConfigByCreateSpConfig(uint32_t main_layers, int mtp_module_num, uint32_t block_num) {
    auto score_model_config   = makeTestModelConfig(main_layers);
    auto propose_model_config = makeTestModelConfig(/*num_layers=*/1);

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

BatchKVCacheResourcePtr createBatchKVCacheResource(int batch_size, int layer_num, int block_num_per_batch = 0) {
    auto resource = std::make_shared<BatchKVCacheResource>();
    resource->resetBatchSize(batch_size);
    for (int i = 0; i < batch_size; ++i) {
        std::vector<std::vector<int>> layer_group_ids(static_cast<size_t>(layer_num), std::vector<int>{0});
        resource->initBatchGroups(i, 1, layer_num, layer_group_ids);
        resource->setBatchBlocks(i, 0, std::vector<int>(block_num_per_batch));
        resource->setBatchCacheKeys(i, CacheKeysType(block_num_per_batch, static_cast<CacheKeyType>(i * 100)));
    }
    return resource;
}

class SingleTypeKVCacheAllocatorTest: public ::testing::Test {
protected:
    void SetUp() override {
        createDevice();
    }

    void TearDown() override {
        // allocator_ holds a raw pointer into block_tree_cache_; drop it first.
        allocator_.reset();
        block_tree_cache_.reset();
    }

    // After init(), the DeviceKVCacheGroup lives inside BlockTreeCache, not the allocator, so
    // the allocator needs the BlockTreeCache injected before any malloc/free/convert/getNeedBlocks
    // call (otherwise fullGroup() aborts). Mirror KVCacheManager's wiring: init() builds the
    // device pool, then create + inject the BlockTreeCache.
    bool initWithBlockTreeCache(const CacheConfig& config, AllocationType allocation_type = AllocationType::DEVICE) {
        allocator_ = std::make_shared<SingleTypeKVCacheAllocator>(config, allocation_type);
        if (!allocator_->init()) {
            return false;
        }

        auto block_pool = allocator_->getDeviceBlockPool();
        if (!block_pool) {
            return false;
        }
        const size_t                                 total_before  = allocator_->totalBlocksNum();
        const size_t                                 free_before   = allocator_->freeBlocksNum();
        const size_t                                 active_before = allocator_->activeTreeCachedBlocksNum();
        std::vector<std::pair<BlockIdxType, size_t>> allocated_refs_before;
        for (BlockIdxType block = 0; block < static_cast<BlockIdxType>(config.block_num); ++block) {
            if (block_pool->isAllocated(block)) {
                allocated_refs_before.emplace_back(block, block_pool->refCount(block));
            }
        }

        KVCacheConfig kv_cache_config;  // device-only: memory/disk tiers disabled
        block_tree_cache_ = createBlockTreeCache(config, kv_cache_config, allocator_);
        if (!block_tree_cache_) {
            return false;
        }
        allocator_->setBlockTreeCache(block_tree_cache_.get());

        bool unchanged = true;
        EXPECT_EQ(allocator_->blockTreeCache(), block_tree_cache_.get());
        EXPECT_EQ(allocator_->totalBlocksNum(), total_before);
        EXPECT_EQ(allocator_->freeBlocksNum(), free_before);
        EXPECT_EQ(allocator_->activeTreeCachedBlocksNum(), active_before);
        unchanged = allocator_->blockTreeCache() == block_tree_cache_.get()
                    && allocator_->totalBlocksNum() == total_before && allocator_->freeBlocksNum() == free_before
                    && allocator_->activeTreeCachedBlocksNum() == active_before;
        for (const auto& [block, ref_count] : allocated_refs_before) {
            const bool still_allocated = block_pool->isAllocated(block);
            EXPECT_TRUE(still_allocated);
            unchanged = unchanged && still_allocated;
            if (still_allocated) {
                EXPECT_EQ(block_pool->refCount(block), ref_count);
                unchanged = unchanged && block_pool->refCount(block) == ref_count;
            }
        }
        return unchanged;
    }

    bool initWithTieredBlockTreeCache(const CacheConfig&   config,
                                      const KVCacheConfig& kv_cache_config,
                                      AllocationType       allocation_type = AllocationType::DEVICE) {
        allocator_ = std::make_shared<SingleTypeKVCacheAllocator>(config, allocation_type);
        if (!allocator_->init()) {
            return false;
        }
        block_tree_cache_ = createBlockTreeCache(config, kv_cache_config, allocator_);
        if (!block_tree_cache_) {
            return false;
        }
        allocator_->setBlockTreeCache(block_tree_cache_.get());
        return true;
    }

    std::shared_ptr<SingleTypeKVCacheAllocator> allocator_;
    BlockTreeCachePtr                           block_tree_cache_;

    // DeviceBlockPool is DEVICE-only: cache bytes live in device memory and cannot be
    // dereferenced from the host. Stage bytes through the backend-neutral runtimeCopy /
    // runtimeSyncAndCheck (works for CUDA/ROCm/... — device is not necessarily CUDA)
    // instead of a CUDA-specific cudaMemcpy.
    static void writeDeviceBytes(void* dst_device, const std::vector<uint8_t>& host) {
        auto host_t = torch::from_blob(const_cast<uint8_t*>(host.data()),
                                       {static_cast<int64_t>(host.size())},
                                       torch::TensorOptions(torch::kUInt8))
                          .clone();
        auto dev_t = torch::from_blob(
            dst_device, {static_cast<int64_t>(host.size())}, torch::TensorOptions(torch::kUInt8).device(torch::kCUDA));
        CopyParams cp{dev_t, host_t};
        runtimeCopy(cp);
        runtimeSyncAndCheck();
    }
    static std::vector<uint8_t> readDeviceBytes(const void* src_device, size_t n) {
        auto        dev_t  = torch::from_blob(const_cast<void*>(src_device),
                                              {static_cast<int64_t>(n)},
                                      torch::TensorOptions(torch::kUInt8).device(torch::kCUDA));
        auto        host_t = dev_t.cpu();
        const auto* ptr    = host_t.data_ptr<uint8_t>();
        return std::vector<uint8_t>(ptr, ptr + n);
    }
};

class SingleTypePreflightTestPeer: public SingleTypeKVCacheAllocator {
public:
    using SingleTypeKVCacheAllocator::SingleTypeKVCacheAllocator;
    using SingleTypeKVCacheAllocator::preflightLoadBackMappings;
};

struct SingleTypePreflightEnvironment {
    BlockTreeCachePtr              cache;
    std::shared_ptr<HostBlockPool> host_pool;
    BlockIdxType                   source_block{NULL_BLOCK_IDX};
};

static SingleTypePreflightEnvironment makeSingleTypePreflightEnvironment(const DeviceBlockPoolPtr& device_pool) {
    SingleTypePreflightEnvironment environment;

    auto host_config                  = std::make_shared<HostBlockPoolConfig>();
    host_config->pool_type            = BlockPoolType::HOST;
    host_config->pool_name            = "single_type_preflight_host";
    host_config->physical_block_count = 3;
    host_config->payload_bytes        = 2;
    host_config->stride_bytes         = 4096;
    host_config->enable_pinned        = false;
    host_config->alignment            = 4096;
    environment.host_pool             = std::make_shared<HostBlockPool>(host_config);
    if (!environment.host_pool->init()) {
        return environment;
    }

    auto primary                = std::make_shared<FullComponentGroup>();
    primary->component_group_id = 0;
    primary->setDevicePools({device_pool, device_pool});
    primary->setHostPool(environment.host_pool);

    auto components = makeUnitLayerComponents(2);
    if (!primary->finalizeLayout({0, 1}, components)) {
        return environment;
    }

    BlockTreeCacheConfig cache_config;
    cache_config.enable_device_cache = true;
    cache_config.enable_memory_cache = true;
    cache_config.enable_load_back    = true;
    environment.cache =
        std::make_shared<BlockTreeCache>(std::make_unique<BlockTree>(1),
                                         std::vector<ComponentGroupPtr>{primary},
                                         std::move(components),
                                         std::move(cache_config),
                                         nullptr,
                                         nullptr,
                                         std::vector<DeviceKVCacheGroupPtr>(3),
                                         std::vector<BlockTreeCache::PerTagMapping>{{0, 1}, {0, 0}, {-1, -1}});
    if (!environment.cache->init()) {
        environment.cache.reset();
        return environment;
    }

    environment.source_block = primary->allocateSingleBlock(Tier::HOST);
    if (isNullBlockIdx(environment.source_block)) {
        environment.cache.reset();
        return environment;
    }
    std::vector<std::vector<GroupSlot>> slots(1, std::vector<GroupSlot>(1));
    slots[0][0].host_block = environment.source_block;
    if (environment.cache->tree()->insertNode(nullptr, {100}, slots).leaf == nullptr) {
        environment.cache.reset();
    }
    return environment;
}

static std::shared_ptr<LoadBackTicket> captureSingleTypePreflightTicket(BlockTreeCache& cache) {
    BlockTreeMatchResult result = cache.match({100});
    EXPECT_NE(result.load_back_ticket, nullptr);
    if (result.load_back_ticket == nullptr) {
        return nullptr;
    }
    EXPECT_EQ(result.load_back_ticket->items().size(), 1u);
    return std::move(result.load_back_ticket);
}

class ScopedSingleTypeDiskCacheDirectory {
public:
    ScopedSingleTypeDiskCacheDirectory() {
        std::string       pattern = "/tmp/rtp_llm_single_type_load_back_XXXXXX";
        std::vector<char> writable(pattern.begin(), pattern.end());
        writable.push_back('\0');
        char* result = ::mkdtemp(writable.data());
        EXPECT_NE(result, nullptr);
        if (result != nullptr) {
            path_ = result;
        }
    }

    ~ScopedSingleTypeDiskCacheDirectory() {
        if (path_.empty()) {
            return;
        }
        const std::string work_dir = path_ + "/rtp_llm_disk_kv";
        if (DIR* dir = ::opendir(work_dir.c_str()); dir != nullptr) {
            while (true) {
                errno       = 0;
                auto* entry = ::readdir(dir);
                if (entry == nullptr) {
                    const int error = errno;
                    if (error != 0) {
                        reportUnexpectedFailure("readdir", work_dir, error);
                    }
                    break;
                }
                const std::string name = entry->d_name;
                if (name != "." && name != "..") {
                    const std::string entry_path = work_dir + "/" + name;
                    if (::unlink(entry_path.c_str()) != 0) {
                        const int error = errno;
                        if (error != ENOENT) {
                            reportUnexpectedFailure("unlink", entry_path, error);
                        }
                    }
                }
            }
            if (::closedir(dir) != 0) {
                reportUnexpectedFailure("closedir", work_dir, errno);
            }
        } else {
            const int error = errno;
            if (error != ENOENT) {
                reportUnexpectedFailure("opendir", work_dir, error);
            }
        }
        removeDirectory(work_dir);
        removeDirectory(path_);
        expectAbsent(work_dir);
        expectAbsent(path_);
    }

    std::string string() const {
        return path_;
    }

private:
    static void reportUnexpectedFailure(const char* operation, const std::string& path, int error) {
        ADD_FAILURE() << operation << " failed for " << path << ": errno=" << error << " (" << std::strerror(error)
                      << ")";
    }

    static void removeDirectory(const std::string& path) {
        if (::rmdir(path.c_str()) != 0) {
            const int error = errno;
            if (error != ENOENT) {
                reportUnexpectedFailure("rmdir", path, error);
            }
        }
    }

    static void expectAbsent(const std::string& path) {
        errno = 0;
        if (::access(path.c_str(), F_OK) == 0) {
            ADD_FAILURE() << "cleanup left path behind: " << path;
            return;
        }
        const int error = errno;
        if (error != ENOENT) {
            reportUnexpectedFailure("access", path, error);
        }
    }

    std::string path_;
};

static KVCacheConfig makeSingleTypeTieredConfig(Tier source_tier, const std::string& disk_path) {
    KVCacheConfig config;
    config.enable_memory_cache        = true;
    config.enable_tiered_memory_cache = true;
    config.memory_cache_size_mb       = 1;
    config.enable_memory_cache_disk   = source_tier == Tier::DISK;
    config.memory_cache_disk_size_mb  = source_tier == Tier::DISK ? 1 : 0;
    config.memory_cache_disk_paths    = disk_path;
    return config;
}

static BlockIdxType seedSingleTypeLowerTier(BlockTreeCache& cache, Tier source_tier, CacheKeyType key) {
    const ComponentGroupPtr& group        = cache.componentGroups().front();
    const BlockIdxType       source_block = group->allocateSingleBlock(source_tier);
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
    EXPECT_NE(cache.tree()->insertNode(nullptr, {key}, slots).leaf, nullptr);
    return source_block;
}

TEST_F(SingleTypeKVCacheAllocatorTest, RealHostAndDiskTicketsLoadBackIntoRequestOwnedTarget) {
    for (Tier source_tier : {Tier::HOST, Tier::DISK}) {
        SCOPED_TRACE(source_tier == Tier::HOST ? "HOST" : "DISK");
        ScopedSingleTypeDiskCacheDirectory disk_directory;

        const CacheConfig config =
            createSingleTypeTestConfig(/*layer_num=*/4, /*block_num=*/8, /*seq_size_per_block=*/4);
        const KVCacheConfig tiered_config = makeSingleTypeTieredConfig(source_tier, disk_directory.string());
        ASSERT_TRUE(initWithTieredBlockTreeCache(config, tiered_config));
        ASSERT_NE(seedSingleTypeLowerTier(*block_tree_cache_, source_tier, /*key=*/100), NULL_BLOCK_IDX);

        BatchKVCacheResourcePtr resource = createBatchKVCacheResource(/*batch_size=*/1, config.layer_num);
        resource->setBatchCacheKeys(0, CacheKeysType{100, 200});
        CompleteTokenIdsPtr token_ids =
            createCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/8, /*seq_size_per_block=*/4);

        MallocInfo malloc_info{resource, token_ids};
        malloc_info.enable_device_cache = true;
        MallocResult result             = allocator_->malloc(malloc_info);
        ASSERT_TRUE(result.success);
        EXPECT_EQ(result.reuse_len, 4);
        ASSERT_NE(result.async_context, nullptr);
        result.async_context->waitDone();
        ASSERT_TRUE(result.async_context->success()) << result.async_context->errorInfo().ToString();
        EXPECT_EQ(resource->cacheResource(0).deviceReuseBlockNum(), 1u);

        BlockTreeFindResult find = block_tree_cache_->tree()->findNode({100});
        ASSERT_NE(find.matched_node, nullptr);
        ASSERT_EQ(resource->blocks(0, 0).size(), 2u);
        ASSERT_EQ(find.matched_node->group_slots[0].device_blocks.size(), 1u);
        EXPECT_EQ(find.matched_node->group_slots[0].device_blocks.front(), resource->blocks(0, 0).front());

        allocator_->free(FreeInfo{resource, token_ids});
        result.async_context.reset();
        allocator_.reset();
        block_tree_cache_.reset();
    }
}

TEST_F(SingleTypeKVCacheAllocatorTest, RealHostTicketAbortsBeforeCommitWhenTargetAllocationFails) {
    ScopedSingleTypeDiskCacheDirectory disk_directory;
    const CacheConfig config = createSingleTypeTestConfig(/*layer_num=*/4, /*block_num=*/2, /*seq_size_per_block=*/4);
    ASSERT_TRUE(initWithTieredBlockTreeCache(config, makeSingleTypeTieredConfig(Tier::HOST, disk_directory.string())));

    const BlockIdxType source_block = seedSingleTypeLowerTier(*block_tree_cache_, Tier::HOST, /*key=*/100);
    ASSERT_NE(source_block, NULL_BLOCK_IDX);
    const ComponentGroupPtr& group              = block_tree_cache_->componentGroups().front();
    const size_t             source_ref_before  = group->hostPool()->refCount(source_block);
    const size_t             device_free_before = allocator_->freeBlocksNum();

    BatchKVCacheResourcePtr resource = createBatchKVCacheResource(/*batch_size=*/1, config.layer_num);
    resource->setBatchCacheKeys(0, CacheKeysType{100, 200});
    CompleteTokenIdsPtr token_ids =
        createCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/8, /*seq_size_per_block=*/4);

    MallocInfo malloc_info{resource, token_ids};
    malloc_info.enable_device_cache = true;
    const MallocResult result       = allocator_->malloc(malloc_info);
    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.async_context, nullptr);
    EXPECT_EQ(resource->curBlocksNum(), 0);
    EXPECT_EQ(allocator_->freeBlocksNum(), device_free_before);
    EXPECT_EQ(group->hostPool()->refCount(source_block), source_ref_before);

    BlockTreeFindResult find = block_tree_cache_->tree()->findNode({100});
    ASSERT_NE(find.matched_node, nullptr);
    EXPECT_EQ(find.matched_node->group_slots[0].host_block, source_block);
    EXPECT_EQ(find.matched_node->group_slots[0].transfer_state, SlotTransferState::IDLE);
}

TEST_F(SingleTypeKVCacheAllocatorTest, NonContiguousDeviceHostDevicePreservesIdentityAndPayload) {
    const CacheConfig config = createSingleTypeTestConfig(/*layer_num=*/4, /*block_num=*/8, /*seq_size_per_block=*/4);
    ASSERT_TRUE(initWithTieredBlockTreeCache(config, makeSingleTypeTieredConfig(Tier::HOST, "")));

    const ComponentGroupPtr& group    = block_tree_cache_->componentGroups().front();
    GroupBlockSet            resident = group->allocateBlocks(Tier::DEVICE, 2);
    ASSERT_EQ(resident.per_node.size(), 2u);
    ASSERT_EQ(resident.per_node[0].size(), 1u);
    ASSERT_EQ(resident.per_node[1].size(), 1u);
    const BlockIdxType first_device = resident.per_node[0][0];
    const BlockIdxType last_device  = resident.per_node[1][0];
    const BlockIdxType host_source  = group->allocateSingleBlock(Tier::HOST);
    ASSERT_NE(host_source, NULL_BLOCK_IDX);

    const size_t device_payload_bytes = config.specForGroup(0)->k_block_size() + config.specForGroup(0)->v_block_size();
    writeDeviceBytes(allocator_->convertIndexToAddr(/*layer_id=*/0, first_device).kv_addr,
                     std::vector<uint8_t>(device_payload_bytes, 0x11));
    writeDeviceBytes(allocator_->convertIndexToAddr(/*layer_id=*/0, last_device).kv_addr,
                     std::vector<uint8_t>(device_payload_bytes, 0x33));
    std::memset(group->hostPool()->blockBuffer(host_source).addr, 0x22, group->hostPool()->payloadBytes());

    std::vector<std::vector<GroupSlot>> slots(3, std::vector<GroupSlot>(1));
    slots[0][0].device_blocks = {first_device};
    slots[1][0].host_block    = host_source;
    slots[2][0].device_blocks = {last_device};
    ASSERT_NE(block_tree_cache_->tree()->insertNode(nullptr, {100, 200, 300}, slots).leaf, nullptr);

    auto make_resource = [&]() {
        BatchKVCacheResourcePtr resource = createBatchKVCacheResource(/*batch_size=*/1, config.layer_num);
        resource->setBatchCacheKeys(0, CacheKeysType{100, 200, 300, 400});
        return resource;
    };
    CompleteTokenIdsPtr token_ids =
        createCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/16, /*seq_size_per_block=*/4);

    const size_t free_before_rejection = allocator_->freeBlocksNum();
    allocator_->setReserveBlockNum(allocator_->freeBlocksNum());
    BatchKVCacheResourcePtr rejected_resource = make_resource();
    MallocInfo              rejected_info{rejected_resource, token_ids};
    rejected_info.enable_device_cache  = true;
    const MallocResult rejected_result = allocator_->malloc(rejected_info);
    EXPECT_FALSE(rejected_result.success);
    EXPECT_EQ(rejected_resource->curBlocksNum(), 0);
    EXPECT_EQ(rejected_resource->cacheResource(0).deviceReuseBlockNum(), 0u);
    EXPECT_EQ(allocator_->freeBlocksNum(), free_before_rejection);
    EXPECT_EQ(group->devicePools()[0]->refCount(first_device), 1u);
    EXPECT_EQ(group->devicePools()[0]->refCount(last_device), 1u);
    EXPECT_EQ(group->hostPool()->refCount(host_source), 1u);

    allocator_->setReserveBlockNum(0);
    BatchKVCacheResourcePtr resource = make_resource();
    MallocInfo              malloc_info{resource, token_ids};
    malloc_info.enable_device_cache = true;
    MallocResult result             = allocator_->malloc(malloc_info);
    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.reuse_len, 12);
    EXPECT_EQ(resource->cacheResource(0).deviceReuseBlockNum(), 3u);
    ASSERT_NE(result.async_context, nullptr);
    result.async_context->waitDone();
    ASSERT_TRUE(result.async_context->success()) << result.async_context->errorInfo().ToString();

    ASSERT_EQ(resource->blocks(0, 0).size(), 4u);
    EXPECT_EQ(resource->blocks(0, 0)[0], first_device);
    EXPECT_EQ(resource->blocks(0, 0)[2], last_device);
    EXPECT_NE(resource->blocks(0, 0)[1], first_device);
    EXPECT_NE(resource->blocks(0, 0)[1], last_device);
    EXPECT_EQ(readDeviceBytes(allocator_->convertIndexToAddr(0, first_device).kv_addr, device_payload_bytes),
              std::vector<uint8_t>(device_payload_bytes, 0x11));
    EXPECT_EQ(
        readDeviceBytes(allocator_->convertIndexToAddr(0, resource->blocks(0, 0)[1]).kv_addr, device_payload_bytes),
        std::vector<uint8_t>(device_payload_bytes, 0x22));
    EXPECT_EQ(readDeviceBytes(allocator_->convertIndexToAddr(0, last_device).kv_addr, device_payload_bytes),
              std::vector<uint8_t>(device_payload_bytes, 0x33));
    EXPECT_EQ(group->devicePools()[0]->refCount(first_device), 2u);
    EXPECT_EQ(group->devicePools()[0]->refCount(last_device), 2u);
    EXPECT_FALSE(group->hostPool()->isAllocated(host_source));

    const BlockIdxType loaded_host_target = resource->blocks(0, 0)[1];
    allocator_->free(FreeInfo{resource, token_ids});
    EXPECT_EQ(group->devicePools()[0]->refCount(first_device), 1u);
    EXPECT_EQ(group->devicePools()[0]->refCount(loaded_host_target), 1u);
    EXPECT_EQ(group->devicePools()[0]->refCount(last_device), 1u);
    while (block_tree_cache_->reclaimBlocks(1, Tier::DEVICE) > 0) {}
    block_tree_cache_->waitForPendingTasks();
    EXPECT_FALSE(group->devicePools()[0]->isAllocated(first_device));
    EXPECT_FALSE(group->devicePools()[0]->isAllocated(loaded_host_target));
    EXPECT_FALSE(group->devicePools()[0]->isAllocated(last_device));
    EXPECT_EQ(block_tree_cache_->getStats().tree_node_count, 0u);
    EXPECT_EQ(allocator_->freeBlocksNum(), allocator_->totalBlocksNum());
}

TEST_F(SingleTypeKVCacheAllocatorTest, ProtectedPreflightRejectsMalformedRealTicketWithoutMutation) {
    const CacheConfig config = createSingleTypeTestConfig(/*layer_num=*/4, /*block_num=*/8, /*seq_size_per_block=*/4);
    auto              allocator = std::make_shared<SingleTypePreflightTestPeer>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    SingleTypePreflightEnvironment environment = makeSingleTypePreflightEnvironment(allocator->getDeviceBlockPool());
    ASSERT_NE(environment.cache, nullptr);
    ASSERT_NE(environment.host_pool, nullptr);
    ASSERT_NE(environment.source_block, NULL_BLOCK_IDX);
    allocator->setBlockTreeCache(environment.cache.get());

    const size_t        source_ref_baseline  = environment.host_pool->refCount(environment.source_block);
    const size_t        allocator_free       = allocator->freeBlocksNum();
    const size_t        tree_node_count      = environment.cache->getStats().tree_node_count;
    const CopyEnginePtr copy_engine_identity = environment.cache->copyEngine();

    EXPECT_TRUE(allocator->preflightLoadBackMappings(nullptr));
    auto empty_ticket_registry = std::make_shared<LoadBackTicketRegistry>(LoadBackTicketRegistry::CommitCallback{},
                                                                          LoadBackTicketRegistry::AbortCallback{});
    std::shared_ptr<LoadBackTicket> empty_ticket = empty_ticket_registry->createTicket({});
    ASSERT_NE(empty_ticket, nullptr);
    EXPECT_TRUE(allocator->preflightLoadBackMappings(empty_ticket));

    {
        std::shared_ptr<LoadBackTicket> ticket = captureSingleTypePreflightTicket(*environment.cache);
        ASSERT_NE(ticket, nullptr);
        ASSERT_EQ(ticket->items().size(), 1u);
        EXPECT_EQ(ticket->items().front().device_group_ids, (std::vector<int>{1, 0}));
        EXPECT_EQ(environment.host_pool->refCount(environment.source_block), source_ref_baseline + 1);
        EXPECT_TRUE(allocator->preflightLoadBackMappings(ticket));
        EXPECT_EQ(ticket->items().front().device_group_ids, (std::vector<int>{1, 0}));
        EXPECT_TRUE(ticket->items().front().target_device_blocks.empty());
        EXPECT_EQ(allocator->freeBlocksNum(), allocator_free);
        EXPECT_EQ(environment.host_pool->refCount(environment.source_block), source_ref_baseline + 1);
        ticket.reset();
        EXPECT_EQ(environment.host_pool->refCount(environment.source_block), source_ref_baseline);
    }

    struct MappingCase {
        const char*      name;
        std::vector<int> device_group_ids;
    };
    const std::vector<MappingCase> malformed = {
        {"empty", {}},
        {"short", {1}},
        {"long", {1, 0, 2}},
        {"out_of_range_gid", {1, 3}},
        {"duplicated_gid_replacement", {1, 1}},
        {"permutation", {0, 1}},
        {"wrong_but_valid_gid", {1, 2}},
    };
    for (const MappingCase& mapping_case : malformed) {
        SCOPED_TRACE(mapping_case.name);
        std::shared_ptr<LoadBackTicket> ticket = captureSingleTypePreflightTicket(*environment.cache);
        ASSERT_NE(ticket, nullptr);
        ASSERT_EQ(ticket->items().size(), 1u);
        EXPECT_EQ(ticket->items().front().device_group_ids, (std::vector<int>{1, 0}));
        EXPECT_EQ(environment.host_pool->refCount(environment.source_block), source_ref_baseline + 1);

        ticket->items().front().device_group_ids = mapping_case.device_group_ids;
        EXPECT_FALSE(allocator->preflightLoadBackMappings(ticket));
        EXPECT_EQ(ticket->items().front().device_group_ids, mapping_case.device_group_ids);
        EXPECT_TRUE(ticket->items().front().target_device_blocks.empty());
        EXPECT_EQ(allocator->freeBlocksNum(), allocator_free);
        EXPECT_EQ(environment.host_pool->refCount(environment.source_block), source_ref_baseline + 1);
        EXPECT_EQ(environment.cache->getStats().tree_node_count, tree_node_count);
        EXPECT_EQ(environment.cache->copyEngine(), copy_engine_identity);
        BlockTreeFindResult find = environment.cache->tree()->findNode({100});
        ASSERT_NE(find.matched_node, nullptr);
        EXPECT_EQ(find.matched_node->group_slots[0].transfer_state, SlotTransferState::IDLE);

        ticket.reset();
        EXPECT_EQ(environment.host_pool->refCount(environment.source_block), source_ref_baseline);
        ticket.reset();
        EXPECT_EQ(environment.host_pool->refCount(environment.source_block), source_ref_baseline)
            << "ticket reset must release source protection exactly once";
    }

    allocator->setBlockTreeCache(nullptr);
}

// Test init
TEST_F(SingleTypeKVCacheAllocatorTest, ConstructorAndInit) {
    auto config = createSingleTypeTestConfig();
    allocator_  = std::make_shared<SingleTypeKVCacheAllocator>(config);
    ASSERT_NE(allocator_, nullptr);

    bool init_result = allocator_->init();
    EXPECT_TRUE(init_result);

    EXPECT_EQ(allocator_->totalBlocksNum(), config.block_num - 1);
    EXPECT_EQ(allocator_->freeBlocksNum(), config.block_num - 1);  // reserve 1 block
}

TEST_F(SingleTypeKVCacheAllocatorTest, InitWithDifferentLayerNum) {
    auto config = createSingleTypeTestConfig(8, 20, 16);
    allocator_  = std::make_shared<SingleTypeKVCacheAllocator>(config);

    bool init_result = allocator_->init();
    EXPECT_TRUE(init_result);

    EXPECT_EQ(allocator_->totalBlocksNum(), 20 - 1);
}

TEST_F(SingleTypeKVCacheAllocatorTest, GetNeedBlocksComputesCommonAndExtra) {
    auto config = createSingleTypeTestConfig(/*layer_num=*/4, /*block_num=*/10, /*seq_size_per_block=*/4);
    ASSERT_TRUE(initWithBlockTreeCache(config));

    const int batch_size = 3;
    auto      batch_res  = createBatchKVCacheResource(batch_size, config.layer_num);
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
    ASSERT_TRUE(initWithBlockTreeCache(config));

    int  seq_length         = 16;
    auto batch_resource     = createBatchKVCacheResource(1, config.layer_num);
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
    ASSERT_TRUE(initWithBlockTreeCache(config));

    allocator_->setReserveBlockNum(2);

    const size_t free_before = allocator_->freeBlocksNum();
    ASSERT_EQ(free_before, 9u);

    // Init malloc requesting 8 blocks should fail: 9 < 8 + 2 reserved.
    {
        auto batch_resource     = createBatchKVCacheResource(/*batch_size=*/1, config.layer_num);
        auto complete_token_ids = createCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/8, /*seq_size_per_block=*/1);

        MallocInfo malloc_info{batch_resource, complete_token_ids};
        auto       result = allocator_->malloc(malloc_info);
        EXPECT_FALSE(result.success);
        EXPECT_EQ(batch_resource->curBlocksNum(), 0);
        EXPECT_EQ(allocator_->freeBlocksNum(), free_before);
    }

    // Init malloc requesting 7 blocks should succeed: 9 >= 7 + 2 reserved.
    auto       batch_resource_ok = createBatchKVCacheResource(/*batch_size=*/1, config.layer_num);
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
    ASSERT_TRUE(initWithBlockTreeCache(config));
    ASSERT_EQ(allocator_->blockTreeCache(), block_tree_cache_.get());

    auto pool = allocator_->getDeviceBlockPool();
    ASSERT_NE(pool, nullptr);
    allocator_->setReserveBlockNum(2);
    ASSERT_EQ(allocator_->freeBlocksNum(), 9);
    ASSERT_EQ(allocator_->activeTreeCachedBlocksNum(), 0u);

    auto seed_resource = createBatchKVCacheResource(/*batch_size=*/1, config.layer_num);
    seed_resource->setBatchCacheKeys(0, CacheKeysType{100, 101, 102, 103});
    auto seed_token_ids = createCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/16, /*seq_size_per_block=*/4);

    MallocInfo seed_malloc_info{seed_resource, seed_token_ids};
    seed_malloc_info.enable_device_cache = false;
    auto seed_malloc_result              = allocator_->malloc(seed_malloc_info);
    ASSERT_TRUE(seed_malloc_result.success);
    EXPECT_EQ(seed_malloc_result.async_context, nullptr);
    ASSERT_EQ(seed_resource->curBlocksNum(), 4);
    const auto seed_blocks = seed_resource->blocks(/*batch_id=*/0);
    ASSERT_EQ(seed_blocks.size(), 4u);
    for (BlockIdxType block : seed_blocks) {
        EXPECT_EQ(pool->refCount(block), 1u);
    }

    InsertInfo seed_insert_info{seed_resource, seed_token_ids, /*is_resident=*/true};
    allocator_->insertIntoCache(seed_insert_info);
    for (BlockIdxType block : seed_blocks) {
        EXPECT_EQ(pool->refCount(block), 2u);
    }
    FreeInfo seed_free_info{seed_resource, seed_token_ids};
    allocator_->free(seed_free_info);
    EXPECT_EQ(seed_resource->curBlocksNum(), 0);
    for (BlockIdxType block : seed_blocks) {
        EXPECT_EQ(pool->refCount(block), 1u);
    }

    const size_t cache_only_free = allocator_->freeBlocksNum();
    const size_t cache_only_heap = block_tree_cache_->getStats().device_heap_total_size;
    auto         initial_match   = block_tree_cache_->match(CacheKeysType{100, 101, 102, 103});
    ASSERT_EQ(initial_match.matched_blocks, 4u);
    EXPECT_EQ(initial_match.async_context, nullptr);
    block_tree_cache_->releaseMatchedBlocks(initial_match.matched_block_sets);
    for (BlockIdxType block : seed_blocks) {
        EXPECT_EQ(pool->refCount(block), 1u);
    }

    // Reuse the four cached blocks and allocate one request-only block.
    {
        auto batch_resource = createBatchKVCacheResource(/*batch_size=*/1, config.layer_num);
        batch_resource->setBatchCacheKeys(0, CacheKeysType{100, 101, 102, 103, 200});

        auto       token_ids = createCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/20, /*seq_size_per_block=*/4);
        MallocInfo malloc_info{batch_resource, token_ids};
        malloc_info.enable_device_cache = true;

        auto result = allocator_->malloc(malloc_info);
        ASSERT_TRUE(result.success);
        EXPECT_EQ(result.async_context, nullptr);
        const size_t reuse_blocks = batch_resource->cacheResource(0).reuseBlockNum();
        ASSERT_EQ(reuse_blocks, 4u);
        EXPECT_EQ(result.reuse_len, 16);
        EXPECT_EQ(batch_resource->curBlocksNum(), 5);
        const auto& request_blocks = batch_resource->blocks(/*batch_id=*/0);
        ASSERT_EQ(request_blocks.size(), 5u);
        EXPECT_TRUE(std::equal(seed_blocks.begin(), seed_blocks.end(), request_blocks.begin()));
        for (BlockIdxType block : seed_blocks) {
            EXPECT_EQ(pool->refCount(block), 2u);
        }
        FreeInfo free_info{batch_resource, token_ids};
        allocator_->free(free_info);
        EXPECT_EQ(batch_resource->curBlocksNum(), 0);
        EXPECT_EQ(allocator_->freeBlocksNum(), cache_only_free);
        EXPECT_EQ(block_tree_cache_->getStats().device_heap_total_size, cache_only_heap);
        EXPECT_EQ(allocator_->activeTreeCachedBlocksNum(), 0u);
        for (BlockIdxType block : seed_blocks) {
            EXPECT_EQ(pool->refCount(block), 1u);
        }
    }

    // Reuse four cached blocks, but reject five new blocks because reserve headroom must remain.
    {
        auto batch_resource = createBatchKVCacheResource(/*batch_size=*/1, config.layer_num);
        batch_resource->setBatchCacheKeys(0, CacheKeysType{100, 101, 102, 103, 300, 301, 302, 303, 304});

        auto       token_ids = createCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/36, /*seq_size_per_block=*/4);
        MallocInfo malloc_info{batch_resource, token_ids};
        malloc_info.enable_device_cache = true;

        auto result = allocator_->malloc(malloc_info);
        EXPECT_FALSE(result.success);
        EXPECT_EQ(result.async_context, nullptr);
        EXPECT_EQ(batch_resource->curBlocksNum(), 0);
        EXPECT_EQ(allocator_->freeBlocksNum(), cache_only_free);
        EXPECT_EQ(block_tree_cache_->getStats().device_heap_total_size, cache_only_heap);
        EXPECT_EQ(allocator_->activeTreeCachedBlocksNum(), 0u);
        for (BlockIdxType block : seed_blocks) {
            EXPECT_EQ(pool->refCount(block), 1u);
        }
    }

    auto final_match = block_tree_cache_->match(CacheKeysType{100, 101, 102, 103});
    ASSERT_EQ(final_match.matched_blocks, 4u);
    EXPECT_EQ(final_match.async_context, nullptr);
    block_tree_cache_->releaseMatchedBlocks(final_match.matched_block_sets);
    for (BlockIdxType block : seed_blocks) {
        EXPECT_EQ(pool->refCount(block), 1u);
    }
}

TEST_F(SingleTypeKVCacheAllocatorTest, MallocMultipleBatches) {
    auto config = createSingleTypeTestConfig();
    ASSERT_TRUE(initWithBlockTreeCache(config));

    int  batch_size         = 3;
    int  seq_length         = 17;
    auto batch_resource     = createBatchKVCacheResource(batch_size, config.layer_num);
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
//     allocator_ = std::make_shared<SingleTypeKVCacheAllocator>(config);
//     allocator_->init();

//     int batch_size = 3;
//     int seq_length = 16;  // 3 batches * 2 blocks
//     auto batch_resource = createBatchKVCacheResource(batch_size, config.layer_num);
//     auto complete_token_ids = createCompleteTokenIds(batch_size, seq_length);

//     MallocInfo malloc_info{batch_resource, complete_token_ids};
//     auto result = allocator_->malloc(malloc_info);

//     EXPECT_LE(allocator_->freeBlocksNum(), 5);
// }

// Test free
TEST_F(SingleTypeKVCacheAllocatorTest, FreeSingleBatch) {
    auto config = createSingleTypeTestConfig();
    ASSERT_TRUE(initWithBlockTreeCache(config));

    int  seq_length         = 16;
    auto batch_resource     = createBatchKVCacheResource(1, config.layer_num);
    auto complete_token_ids = createCompleteTokenIds(1, seq_length);

    MallocInfo malloc_info{batch_resource, complete_token_ids};
    allocator_->malloc(malloc_info);

    size_t free_before = allocator_->freeBlocksNum();

    FreeInfo free_info{batch_resource, complete_token_ids};
    allocator_->free(free_info);
    EXPECT_GT(allocator_->freeBlocksNum(), free_before);
}

// C007-T02: allocator-direct release must refresh real tree candidacy without a
// manager-side notification or test-only callback.
TEST_F(SingleTypeKVCacheAllocatorTest, DirectFreeMakesCacheOnlyBlockReclaimable) {
    auto config = createSingleTypeTestConfig(/*layer_num=*/4, /*block_num=*/5, /*seq_size_per_block=*/4);
    ASSERT_TRUE(initWithBlockTreeCache(config));

    auto pool = allocator_->getDeviceBlockPool();
    ASSERT_NE(pool, nullptr);
    const size_t free_before = allocator_->freeBlocksNum();

    auto resource = createBatchKVCacheResource(/*batch_size=*/1, config.layer_num);
    resource->setBatchCacheKeys(/*batch_id=*/0, CacheKeysType{100});
    auto tokens = createCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/4, /*seq_size_per_block=*/4);

    MallocInfo malloc_info{resource, tokens};
    malloc_info.reuse_cache         = true;
    malloc_info.enable_device_cache = false;
    ASSERT_TRUE(allocator_->malloc(malloc_info).success);
    ASSERT_EQ(resource->blocksNum(/*batch_id=*/0, /*group_id=*/0), 1);
    const auto block = resource->blocks(/*batch_id=*/0, /*group_id=*/0)[0];
    EXPECT_EQ(pool->refCount(block), 1u);

    allocator_->insertIntoCache(InsertInfo{resource, tokens, /*is_resident=*/false});
    EXPECT_EQ(pool->refCount(block), 2u);
    EXPECT_EQ(block_tree_cache_->getStats().device_heap_total_size, 0u);
    EXPECT_EQ(allocator_->activeTreeCachedBlocksNum(), 1u);
    EXPECT_EQ(allocator_->freeBlocksNum(), free_before - 1);

    allocator_->free(FreeInfo{resource, tokens});
    EXPECT_EQ(resource->curBlocksNum(), 0);
    EXPECT_TRUE(pool->isAllocated(block));
    EXPECT_EQ(pool->refCount(block), 1u);
    EXPECT_EQ(block_tree_cache_->getStats().device_heap_total_size, 1u);
    EXPECT_EQ(allocator_->activeTreeCachedBlocksNum(), 0u);
    EXPECT_EQ(allocator_->freeBlocksNum(), free_before - 1);

    EXPECT_EQ(block_tree_cache_->reclaimBlocks(/*num_blocks=*/1, Tier::DEVICE), 1);
    EXPECT_FALSE(pool->isAllocated(block));
    EXPECT_EQ(block_tree_cache_->getStats().device_heap_total_size, 0u);
    EXPECT_EQ(allocator_->activeTreeCachedBlocksNum(), 0u);
    EXPECT_EQ(allocator_->freeBlocksNum(), free_before);
}

TEST_F(SingleTypeKVCacheAllocatorTest, ZeroBlockFreeWithoutBlockTreeCacheIsNoOp) {
    auto config = createSingleTypeTestConfig(/*layer_num=*/4, /*block_num=*/5, /*seq_size_per_block=*/4);
    allocator_  = std::make_shared<SingleTypeKVCacheAllocator>(config);
    ASSERT_TRUE(allocator_->init());
    ASSERT_EQ(allocator_->blockTreeCache(), nullptr);

    const size_t free_before = allocator_->freeBlocksNum();
    auto         resource    = createBatchKVCacheResource(/*batch_size=*/1, config.layer_num);
    auto         tokens      = createCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/0, /*seq_size_per_block=*/4);

    allocator_->free(FreeInfo{resource, tokens});
    EXPECT_EQ(resource->curBlocksNum(), 0);
    EXPECT_EQ(allocator_->freeBlocksNum(), free_before);
}

TEST_F(SingleTypeKVCacheAllocatorTest, FreeMultipleBatches) {
    auto config = createSingleTypeTestConfig();
    ASSERT_TRUE(initWithBlockTreeCache(config));

    int  batch_size         = 3;
    int  seq_length         = 16;
    auto batch_resource     = createBatchKVCacheResource(batch_size, config.layer_num);
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
    ASSERT_TRUE(initWithBlockTreeCache(config));

    for (int i = 0; i < 5; ++i) {
        int  seq_length         = 16;
        auto batch_resource     = createBatchKVCacheResource(1, config.layer_num);
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
    ASSERT_TRUE(initWithBlockTreeCache(config));

    int  seq_length         = 16;
    auto batch_resource     = createBatchKVCacheResource(1, config.layer_num);
    auto complete_token_ids = createCompleteTokenIds(1, seq_length);

    MallocInfo malloc_info{batch_resource, complete_token_ids};
    allocator_->malloc(malloc_info);

    InsertInfo insert_info{batch_resource, complete_token_ids, false};
    allocator_->insertIntoCache(insert_info);
}

TEST_F(SingleTypeKVCacheAllocatorTest, InsertIntoCacheAsResident) {
    auto config = createSingleTypeTestConfig();
    ASSERT_TRUE(initWithBlockTreeCache(config));

    int  seq_length         = 16;
    auto batch_resource     = createBatchKVCacheResource(1, config.layer_num);
    auto complete_token_ids = createCompleteTokenIds(1, seq_length);

    MallocInfo malloc_info{batch_resource, complete_token_ids};
    allocator_->malloc(malloc_info);

    InsertInfo insert_info{batch_resource, complete_token_ids, true};
    allocator_->insertIntoCache(insert_info);
}

// Test convert index to addr
TEST_F(SingleTypeKVCacheAllocatorTest, ConvertIndexToAddr) {
    auto config = createSingleTypeTestConfig();
    ASSERT_TRUE(initWithBlockTreeCache(config));

    for (int layer_id = 0; layer_id < config.layer_num; ++layer_id) {
        for (int block_id = 0; block_id < 3; ++block_id) {
            auto addr_info = allocator_->convertIndexToAddr(layer_id, block_id);
            EXPECT_NE(addr_info.kv_addr, nullptr);
        }
    }
}

TEST_F(SingleTypeKVCacheAllocatorTest, ConvertToGlobalLayerIdSingleNoMtp) {
    auto config = createSingleTypeTestConfig(/*layer_num=*/4, /*block_num=*/10, /*seq_size_per_block=*/8);
    allocator_  = std::make_shared<SingleTypeKVCacheAllocator>(config);

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
    auto config = makeMtpCacheConfigByCreateSpConfig(/*main_layers=*/2, /*mtp_module_num=*/2, /*block_num=*/8);
    allocator_  = std::make_shared<SingleTypeKVCacheAllocator>(config);

    // main model: global == local
    EXPECT_EQ(allocator_->convertToGlobalLayerId(/*model_id=*/0, /*local_layer_id=*/0), 0u);
    EXPECT_EQ(allocator_->convertToGlobalLayerId(/*model_id=*/0, /*local_layer_id=*/1), 1u);
    EXPECT_EQ(allocator_->convertToGlobalLayerId(/*model_id=*/0, /*local_layer_id=*/2),
              std::numeric_limits<uint32_t>::max());

    // mtp sub-models map via sub_cfg->global_layer_ids[0]
    EXPECT_EQ(allocator_->convertToGlobalLayerId(/*model_id=*/1, /*local_layer_id=*/0), 2u);
    EXPECT_EQ(allocator_->convertToGlobalLayerId(/*model_id=*/2, /*local_layer_id=*/0), 3u);
    EXPECT_EQ(allocator_->convertToGlobalLayerId(/*model_id=*/1, /*local_layer_id=*/1),
              std::numeric_limits<uint32_t>::max());
    EXPECT_EQ(allocator_->convertToGlobalLayerId(/*model_id=*/3, /*local_layer_id=*/0),
              std::numeric_limits<uint32_t>::max());
}

// Test convert index to buffer
TEST_F(SingleTypeKVCacheAllocatorTest, ConvertIndexToBuffer) {
    auto config = createSingleTypeTestConfig();
    ASSERT_TRUE(initWithBlockTreeCache(config));

    auto buffer_info = allocator_->convertIndexToBuffer(0, 0);
    ASSERT_EQ(buffer_info.size(), 1u);
    EXPECT_NE(buffer_info[0].addr, nullptr);
}

// Test layer cache base
TEST_F(SingleTypeKVCacheAllocatorTest, LayerCacheBase) {
    auto config = createSingleTypeTestConfig();
    ASSERT_TRUE(initWithBlockTreeCache(config));

    auto layout = allocator_->allLayerCacheBase();
    EXPECT_EQ(layout.layers_to_kv_buffer_ptrs.size(), config.layer_num);
    EXPECT_EQ(layout.layers_to_scale_buffer_ptrs.size(), config.layer_num);
    EXPECT_EQ((std::vector<std::vector<int>>(4, std::vector<int>{0})), layout.layer_to_group_ids);

    for (size_t i = 0; i < layout.layers_to_kv_buffer_ptrs.size(); ++i) {
        EXPECT_TRUE(layout.layers_to_kv_buffer_ptrs[i].defined());
        EXPECT_GT(layout.layers_to_kv_buffer_ptrs[i].nbytes(), 0u);
    }
}

// Test block copy
TEST_F(SingleTypeKVCacheAllocatorTest, BlockCopySingle) {
    auto config = createSingleTypeTestConfig();
    ASSERT_TRUE(initWithBlockTreeCache(config));

    int src_block = 0;
    int dst_block = 1;

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

        auto src_bytes = readDeviceBytes(src_addr.kv_addr, block_size);
        auto dst_bytes = readDeviceBytes(dst_addr.kv_addr, block_size);
        EXPECT_EQ(src_bytes, dst_bytes) << "cache mismatch at layer " << layer_id;
    }
}

TEST_F(SingleTypeKVCacheAllocatorTest, BlockBatchCopyVector) {
    auto config = createSingleTypeTestConfig();
    ASSERT_TRUE(initWithBlockTreeCache(config));

    std::vector<BlockIdPair> copy_mapping;
    copy_mapping.push_back({0, 1});
    copy_mapping.push_back({2, 3});
    copy_mapping.push_back({4, 5});

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

            auto src_bytes = readDeviceBytes(src_addr.kv_addr, block_size);
            auto dst_bytes = readDeviceBytes(dst_addr.kv_addr, block_size);
            EXPECT_EQ(src_bytes, dst_bytes)
                << "cache mismatch at block pair (" << pair.src << "->" << pair.dst << "), layer " << layer_id;
        }
    }
}

TEST_F(SingleTypeKVCacheAllocatorTest, BlockBatchCopyEmpty) {
    auto config = createSingleTypeTestConfig();
    ASSERT_TRUE(initWithBlockTreeCache(config));

    std::vector<BlockIdPair> empty_mapping;

    EXPECT_NO_THROW(allocator_->blockBatchCopy(empty_mapping));
}

TEST_F(SingleTypeKVCacheAllocatorTest, BlockBatchCopyPointers) {
    auto config = createSingleTypeTestConfig();
    ASSERT_TRUE(initWithBlockTreeCache(config));

    BlockIdPair pairs[] = {{0, 1}, {2, 3}};

    auto&  spec         = config.specForGroup(0);
    size_t k_block_size = spec->k_block_size();
    size_t v_block_size = spec->v_block_size();
    size_t block_size   = k_block_size + v_block_size;

    for (const auto& pair : pairs) {
        for (int layer_id = 0; layer_id < config.layer_num; ++layer_id) {
            auto src_addr = allocator_->convertIndexToAddr(layer_id, pair.src);
            ASSERT_NE(src_addr.kv_addr, nullptr);

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

            auto src_bytes = readDeviceBytes(src_addr.kv_addr, block_size);
            auto dst_bytes = readDeviceBytes(dst_addr.kv_addr, block_size);
            EXPECT_EQ(src_bytes, dst_bytes)
                << "cache mismatch for block pair (" << pair.src << "->" << pair.dst << "), layer " << layer_id;
        }
    }
}

TEST_F(SingleTypeKVCacheAllocatorTest, BlockBatchCopyBuffer) {
    auto config = createSingleTypeTestConfig();
    ASSERT_TRUE(initWithBlockTreeCache(config));

    std::vector<int32_t> data   = {0, 1, 2, 3, 4, 5};  // 3 pairs: (0->1, 2->3, 4->5)
    auto                 tensor = torch::from_blob(data.data(), {3, 2}, torch::kInt32).clone();

    auto&  spec         = config.specForGroup(0);
    size_t k_block_size = spec->k_block_size();
    size_t v_block_size = spec->v_block_size();
    size_t block_size   = k_block_size + v_block_size;

    for (size_t i = 0; i < data.size(); i += 2) {
        int src_block = data[i];
        for (int layer_id = 0; layer_id < config.layer_num; ++layer_id) {
            auto src_addr = allocator_->convertIndexToAddr(layer_id, src_block);
            ASSERT_NE(src_addr.kv_addr, nullptr);

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

            auto src_bytes = readDeviceBytes(src_addr.kv_addr, block_size);
            auto dst_bytes = readDeviceBytes(dst_addr.kv_addr, block_size);
            EXPECT_EQ(src_bytes, dst_bytes)
                << "cache mismatch for block pair (" << src_block << "->" << dst_block << "), layer " << layer_id;
        }
    }
}

// Test getter methods
TEST_F(SingleTypeKVCacheAllocatorTest, FreeBlocksNums) {
    auto config = createSingleTypeTestConfig();
    ASSERT_TRUE(initWithBlockTreeCache(config));

    EXPECT_EQ(allocator_->freeBlocksNum(), config.block_num - 1);  // reserve 1 block
}

TEST_F(SingleTypeKVCacheAllocatorTest, IncrKVCacheRefReferencesMatchedBlocksOnly) {
    auto config = createSingleTypeTestConfig(/*layer_num=*/4, /*block_num=*/10, /*seq_size_per_block=*/8);
    ASSERT_TRUE(initWithBlockTreeCache(config));

    auto block_pool = allocator_->getDeviceBlockPool();
    ASSERT_NE(block_pool, nullptr);

    const size_t total_free_before = allocator_->freeBlocksNum();
    auto         blocks_opt        = block_pool->malloc(4);
    ASSERT_TRUE(blocks_opt.has_value());
    auto blocks = *blocks_opt;
    ASSERT_EQ(blocks.size(), 4);
    // Single-count pool: malloc() reserves capacity with refCount 0. Add one ref to replicate
    // the legacy malloc(int) auto request-ref that requestFree()/decRef() later drops.
    block_pool->incRef(blocks);
    EXPECT_EQ(allocator_->freeBlocksNum(), total_free_before - 4);

    KVCacheResource resource;
    resource.initGroups(1, config.layer_all_num, config.layerGroupIdsSnapshot());

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
    ASSERT_TRUE(initWithBlockTreeCache(config));

    auto block_pool = allocator_->getDeviceBlockPool();
    ASSERT_NE(block_pool, nullptr);

    const size_t total_free_before = allocator_->freeBlocksNum();
    auto         blocks_opt        = block_pool->malloc(2);
    ASSERT_TRUE(blocks_opt.has_value());
    auto blocks = *blocks_opt;
    ASSERT_EQ(blocks.size(), 2);
    // Single-count pool: replicate the legacy malloc(int) auto request-ref.
    block_pool->incRef(blocks);

    KVCacheResource resource;
    resource.initGroups(1, config.layer_all_num, config.layerGroupIdsSnapshot());
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
    ASSERT_TRUE(initWithBlockTreeCache(config));

    auto block_pool = allocator_->getDeviceBlockPool();
    ASSERT_NE(block_pool, nullptr);

    const size_t total_free_before = allocator_->freeBlocksNum();
    auto         blocks_opt        = block_pool->malloc(2);
    ASSERT_TRUE(blocks_opt.has_value());
    auto blocks = *blocks_opt;
    ASSERT_EQ(blocks.size(), 2);
    // Single-count pool: replicate the legacy malloc(int) auto request-ref.
    block_pool->incRef(blocks);
    EXPECT_EQ(allocator_->freeBlocksNum(), total_free_before - 2);

    KVCacheResource resource;
    resource.initGroups(1, config.layer_all_num, config.layerGroupIdsSnapshot());
    resource.cacheKeys() = CacheKeysType{100, 101};
    resource.mutableBlockIds(0).assign(BlockIndicesType{blocks[0], blocks[1]});

    auto ref_resource = allocator_->incrKVCacheRef(resource, CacheKeysType{});
    ASSERT_EQ(ref_resource, nullptr);

    block_pool->decRef(blocks);
    EXPECT_EQ(allocator_->freeBlocksNum(), total_free_before);
}

TEST_F(SingleTypeKVCacheAllocatorTest, TotalBlocksNums) {
    auto config = createSingleTypeTestConfig(4, 20);
    ASSERT_TRUE(initWithBlockTreeCache(config));

    EXPECT_EQ(allocator_->totalBlocksNum(), 20 - 1);
}

TEST_F(SingleTypeKVCacheAllocatorTest, MaxSeqLen) {
    auto config = createSingleTypeTestConfig(4, 10, 8);
    ASSERT_TRUE(initWithBlockTreeCache(config));

    EXPECT_EQ(allocator_->maxAvailableTokensNum(), (10 - 1) * 8);  // block_num * seq_size_per_block
}

TEST_F(SingleTypeKVCacheAllocatorTest, CapacityAndNeedBlocksUseCPVirtualBlockSize) {
    auto config = createSingleTypeTestConfig(/*layer_num=*/4, /*block_num=*/10, /*seq_size_per_block=*/8);
    ASSERT_TRUE(initWithBlockTreeCache(config));

    allocator_->setCPSlotMapper(std::make_shared<CPSlotMapper>(/*cp_rank=*/0, /*cp_size=*/2, /*block_size=*/8));

    EXPECT_EQ(allocator_->maxAvailableTokensNum(), (10u - 1u) * 16u);
    EXPECT_EQ(allocator_->availableTokensNum(), (10u - 1u) * 16u);

    auto batch_resource = createBatchKVCacheResource(/*batch_size=*/1, config.layer_num);
    EXPECT_EQ(allocator_->singleBatchNeedBlocks(batch_resource, /*seq_len=*/65, /*reserve_step=*/0), 5);
}

// Test boundary conditions

TEST_F(SingleTypeKVCacheAllocatorTest, MallocWithZeroSeqLength) {
    auto config = createSingleTypeTestConfig();
    ASSERT_TRUE(initWithBlockTreeCache(config));

    auto batch_resource     = createBatchKVCacheResource(1, config.layer_num);
    auto complete_token_ids = createCompleteTokenIds(1, 0);

    MallocInfo malloc_info{batch_resource, complete_token_ids};
    auto       result = allocator_->malloc(malloc_info);
    // not crash
    EXPECT_TRUE(result.success || !result.success);
}

TEST_F(SingleTypeKVCacheAllocatorTest, FreeEmptyBatchResource) {
    auto config = createSingleTypeTestConfig();
    ASSERT_TRUE(initWithBlockTreeCache(config));

    auto batch_resource     = createBatchKVCacheResource(0, config.layer_num);
    auto complete_token_ids = createCompleteTokenIds(0, 0);

    FreeInfo free_info{batch_resource, complete_token_ids};
    allocator_->free(free_info);
}

TEST_F(SingleTypeKVCacheAllocatorTest, InitMallocRollbackWhenInitMallocForCommonLenFails) {
    auto config = createSingleTypeTestConfig(/*layer_num=*/4, /*block_num=*/6, /*seq_size_per_block=*/4);
    ASSERT_TRUE(initWithBlockTreeCache(config));
    ASSERT_EQ(allocator_->blockTreeCache(), block_tree_cache_.get());

    auto pool = allocator_->getDeviceBlockPool();
    ASSERT_NE(pool, nullptr);
    const size_t initial_free = allocator_->freeBlocksNum();
    ASSERT_EQ(initial_free, 5u);
    ASSERT_EQ(allocator_->activeTreeCachedBlocksNum(), 0u);

    // Keep one non-keyed pool holder alive so initMalloc succeeds for the shared
    // common prefix and the first batch increment, then fails on the second increment.
    auto pressure_blocks_opt = pool->malloc(1);
    ASSERT_TRUE(pressure_blocks_opt.has_value());
    ASSERT_EQ(pressure_blocks_opt->size(), 1u);
    const BlockIdxType pressure_block = pressure_blocks_opt->front();
    pool->incRef(*pressure_blocks_opt);
    ASSERT_EQ(pool->refCount(pressure_block), 1u);
    const size_t free_before_fail = allocator_->freeBlocksNum();
    ASSERT_EQ(free_before_fail, 4u);
    ASSERT_EQ(allocator_->activeTreeCachedBlocksNum(), 0u);

    auto batch_resource = createBatchKVCacheResource(/*batch_size=*/2, config.layer_num);
    auto token_ids      = createCompleteTokenIds(/*batch_size=*/2, /*seq_length=*/13, /*seq_size_per_block=*/4);

    MallocInfo malloc_info{batch_resource, token_ids};
    malloc_info.enable_device_cache = false;
    auto result                     = allocator_->malloc(malloc_info);
    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.async_context, nullptr);

    // initMalloc must remove the common references and the first batch's partial increment.
    EXPECT_EQ(batch_resource->curBlocksNum(), 0);
    EXPECT_EQ(batch_resource->blocksNum(0, 0), 0);
    EXPECT_EQ(batch_resource->blocksNum(1, 0), 0);
    EXPECT_EQ(allocator_->freeBlocksNum(), free_before_fail);
    EXPECT_EQ(allocator_->activeTreeCachedBlocksNum(), 0u);
    EXPECT_EQ(pool->refCount(pressure_block), 1u);

    pool->decRef(*pressure_blocks_opt);
    EXPECT_FALSE(pool->isAllocated(pressure_block));
    EXPECT_EQ(allocator_->freeBlocksNum(), initial_free);
    EXPECT_EQ(allocator_->activeTreeCachedBlocksNum(), 0u);
}

// Test rollback logic in incrMalloc
TEST_F(SingleTypeKVCacheAllocatorTest, IncrMallocRollback) {
    // Create a config with limited blocks to trigger rollback
    auto config = createSingleTypeTestConfig(4, 8, 4);  // 8 blocks, 4 seq per block
    ASSERT_TRUE(initWithBlockTreeCache(config));

    size_t initial_free_blocks = allocator_->freeBlocksNum();
    EXPECT_EQ(initial_free_blocks, 7);

    int  batch_size         = 3;
    auto batch_resource     = createBatchKVCacheResource(batch_size, config.layer_num);
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
    ASSERT_TRUE(initWithBlockTreeCache(config));

    const size_t free_before = allocator_->freeBlocksNum();
    ASSERT_EQ(free_before, 4u);

    auto batch_resource     = createBatchKVCacheResource(/*batch_size=*/3, config.layer_num);
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

// C005-T01: populated SingleType growth remains synchronous at an exact block boundary.
TEST_F(SingleTypeKVCacheAllocatorTest, PopulatedIncrementIsSynchronousAndRestoresCapacity) {
    auto config = createSingleTypeTestConfig(/*layer_num=*/4, /*block_num=*/10, /*seq_size_per_block=*/4);
    ASSERT_TRUE(initWithBlockTreeCache(config));

    auto pool            = allocator_->getDeviceBlockPool();
    auto batch_resource  = createBatchKVCacheResource(/*batch_size=*/1, config.layer_num);
    auto complete_tokens = createCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/4, /*seq_size_per_block=*/4);
    batch_resource->setBatchCacheKeys(/*batch_id=*/0, CacheKeysType{100, 101});
    const size_t free_before = allocator_->freeBlocksNum();
    ASSERT_EQ(allocator_->activeTreeCachedBlocksNum(), 0u);

    MallocInfo init_info{batch_resource, complete_tokens};
    init_info.reuse_cache         = true;
    init_info.enable_device_cache = true;
    auto init_result              = allocator_->malloc(init_info);
    ASSERT_TRUE(init_result.success);
    EXPECT_EQ(init_result.reuse_len, 0);
    EXPECT_EQ(init_result.async_context, nullptr);
    ASSERT_EQ(batch_resource->blocksNum(/*batch_id=*/0), 1);
    const auto initial_block = batch_resource->blocks(/*batch_id=*/0)[0];
    EXPECT_EQ(pool->refCount(initial_block), 1u);
    EXPECT_EQ(allocator_->freeBlocksNum(), free_before - 1);
    EXPECT_EQ(allocator_->activeTreeCachedBlocksNum(), 0u);

    complete_tokens->setSeqLength(8);
    MallocInfo incr_info{batch_resource, complete_tokens};
    incr_info.reuse_cache         = true;
    incr_info.enable_device_cache = true;
    auto incr_result              = allocator_->malloc(incr_info);
    ASSERT_TRUE(incr_result.success);
    EXPECT_EQ(incr_result.reuse_len, 0);
    EXPECT_EQ(incr_result.async_context, nullptr);
    ASSERT_EQ(batch_resource->blocksNum(/*batch_id=*/0), 2);
    const auto appended_block = batch_resource->blocks(/*batch_id=*/0)[1];
    EXPECT_NE(appended_block, initial_block);
    EXPECT_EQ(pool->refCount(initial_block), 1u);
    EXPECT_EQ(pool->refCount(appended_block), 1u);
    EXPECT_EQ(allocator_->freeBlocksNum(), free_before - 2);
    EXPECT_EQ(allocator_->activeTreeCachedBlocksNum(), 0u);

    FreeInfo free_info{batch_resource, complete_tokens};
    allocator_->free(free_info);
    EXPECT_EQ(batch_resource->curBlocksNum(), 0);
    EXPECT_FALSE(pool->isAllocated(initial_block));
    EXPECT_FALSE(pool->isAllocated(appended_block));
    EXPECT_EQ(allocator_->freeBlocksNum(), free_before);
    EXPECT_EQ(allocator_->activeTreeCachedBlocksNum(), 0u);
}
// ==================== Stress tests ====================

TEST_F(SingleTypeKVCacheAllocatorTest, MixedOperations) {
    auto config = createSingleTypeTestConfig(4, 30);
    ASSERT_TRUE(initWithBlockTreeCache(config));

    std::vector<BatchKVCacheResourcePtr> resources;
    std::vector<CompleteTokenIdsPtr>     token_ids_list;

    for (int i = 0; i < 5; ++i) {
        auto batch_resource     = createBatchKVCacheResource(2, config.layer_num);
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
        auto batch_resource     = createBatchKVCacheResource(1, config.layer_num);
        auto complete_token_ids = createCompleteTokenIds(1, 16);

        MallocInfo malloc_info{batch_resource, complete_token_ids};
        auto       result = allocator_->malloc(malloc_info);
        EXPECT_TRUE(result.success);
    }

    EXPECT_GT(allocator_->freeBlocksNum(), 0);
}

}  // namespace test
}  // namespace rtp_llm

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
