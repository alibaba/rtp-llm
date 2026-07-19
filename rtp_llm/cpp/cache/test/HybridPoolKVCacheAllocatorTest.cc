#include <gtest/gtest.h>

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <dirent.h>
#include <unistd.h>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/utils/AssertUtils.h"

#include "rtp_llm/cpp/cache/AsyncContext.h"
#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/spec/CacheGroupType.h"
#include "rtp_llm/cpp/cache/CPSlotMapper.h"
#include "rtp_llm/cpp/cache/config_creator/CacheConfigCreator.h"
#include "rtp_llm/cpp/cache/config_creator/HybridPoolConfigCreator.h"
#include "rtp_llm/cpp/cache/allocator/HybridPoolKVCacheAllocator.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCacheFactory.h"
#include "rtp_llm/cpp/cache/block_tree_cache/FullComponentGroup.h"
#include "rtp_llm/cpp/cache/block_tree_cache/host/HostBlockPool.h"
#include "rtp_llm/cpp/cache/spec/LinearKVCacheSpec.h"
#include "rtp_llm/cpp/cache/spec/MHAKVCacheSpec.h"
#include "rtp_llm/cpp/cache/test/BlockPoolTestHelper.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/config/ModelConfig.h"
#include "rtp_llm/cpp/disaggregate/cache_store/CacheStore.h"
#include "rtp_llm/cpp/disaggregate/cache_store/MemoryUtil.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace test {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Build a tiny multi-pool config with two groups: gid=0 LINEAR(layers 0,1)
// and gid=1 FULL(layers 2,3). Each group has its own per-group block budget,
// so HybridPoolKVCacheAllocator creates two independent BlockPools.
static CacheConfig makeTinyMultiPoolHybridConfig(uint32_t       linear_block_num = 6,
                                                 uint32_t       full_block_num   = 8,
                                                 CacheGroupType second_type      = CacheGroupType::FULL) {
    CacheConfig config;
    config.dtype                     = rtp_llm::DataType::TYPE_FP16;
    config.layer_num                 = 4;
    config.layer_all_num             = 4;
    config.block_num                 = std::max(linear_block_num, full_block_num);
    config.seq_size_per_block        = 4;
    config.kernel_seq_size_per_block = 4;
    config.linear_step               = 2;
    config.group_layer_num           = 2;

    auto linear_spec                = std::make_shared<LinearKVCacheSpec>();
    linear_spec->type               = KVCacheSpecType::LinearAttention;
    linear_spec->dtype              = config.dtype;
    linear_spec->local_num_k_heads  = 1;
    linear_spec->local_num_v_heads  = 1;
    linear_spec->head_k_dim         = 1;
    linear_spec->head_v_dim         = 1;
    linear_spec->conv_kernel_dim    = 2;
    linear_spec->local_head_num_kv  = 1;
    linear_spec->seq_size_per_block = static_cast<uint32_t>(config.seq_size_per_block);

    auto full_spec                = std::make_shared<MHAKVCacheSpec>();
    full_spec->type               = KVCacheSpecType::MultiHeadAttention;
    full_spec->dtype              = config.dtype;
    full_spec->local_head_num_kv  = 1;
    full_spec->size_per_head      = 1;
    full_spec->seq_size_per_block = static_cast<uint32_t>(config.seq_size_per_block);

    config.use_independent_block_pools = true;
    config.fromGroupedSpecs({linear_spec, full_spec},
                            {{0, 1}, {2, 3}},
                            {CacheGroupType::LINEAR, second_type},
                            {"linear", second_type == CacheGroupType::SWA ? "swa" : "full"});

    // Same tokens per block for both groups.
    config.group_seq_size_per_block = {config.seq_size_per_block, config.seq_size_per_block};

    config.kv_block_stride_bytes = std::max(full_spec->block_size_bytes(), linear_spec->block_size_bytes());
    config.kv_block_size_bytes   = static_cast<size_t>(config.group_layer_num) * config.kv_block_stride_bytes;
    config.kv_scale_stride_bytes = 0;
    config.kv_scale_size_bytes   = 0;
    config.block_size_bytes      = config.kv_block_size_bytes + config.kv_scale_size_bytes;
    config.layer_to_block_stride_bytes.assign(static_cast<size_t>(config.layer_all_num),
                                              static_cast<int>(config.kv_block_stride_bytes));
    const auto linear_stride = linear_spec->block_size_bytes();
    const auto full_stride   = full_spec->block_size_bytes();
    config.setGroupBlockLayout({linear_block_num, full_block_num}, {linear_stride, full_stride}, {0, 0});
    return config;
}

static CacheConfig makeTinySwaMultiPoolHybridConfig(uint32_t linear_block_num = 6, uint32_t swa_block_num = 8) {
    return makeTinyMultiPoolHybridConfig(linear_block_num, swa_block_num, CacheGroupType::SWA);
}

// Put FULL before LINEAR so one FULL primary reclaim plan deterministically
// cascades the matching LINEAR slot at the same tree node.
static CacheConfig makeTinyReclaimCascadeConfig(uint32_t full_block_num = 8, uint32_t linear_block_num = 8) {
    auto config      = makeTinyMultiPoolHybridConfig(linear_block_num, full_block_num);
    auto linear_spec = config.specForGroup(/*gid=*/0);
    auto full_spec   = config.specForGroup(/*gid=*/1);

    config.fromGroupedSpecs(
        {full_spec, linear_spec}, {{2, 3}, {0, 1}}, {CacheGroupType::FULL, CacheGroupType::LINEAR}, {"full", "linear"});
    config.group_seq_size_per_block = {config.seq_size_per_block, config.seq_size_per_block};
    config.setGroupBlockLayout(
        {full_block_num, linear_block_num}, {full_spec->block_size_bytes(), linear_spec->block_size_bytes()}, {0, 0});
    return config;
}

static ModelConfig makeTinyDSV4ModelConfig() {
    ModelConfig mc;
    mc.num_layers                                                = 5;
    mc.hidden_size                                               = 32;
    mc.attn_config.head_num                                      = 4;
    mc.attn_config.kv_head_num                                   = 1;
    mc.attn_config.size_per_head                                 = 8;
    mc.attn_config.rope_head_dim                                 = 4;
    mc.attn_config.sliding_window                                = 128;
    mc.attn_config.indexer_head_dim                              = 8;
    mc.attn_config.indexer_head_num                              = 2;
    mc.attn_config.indexer_topk                                  = 16;
    mc.attn_config.o_groups                                      = 2;
    mc.attn_config.o_lora_rank                                   = 16;
    mc.attn_config.layer_compress_ratios                         = {4, 128, 4, 128, 0};
    mc.hybrid_attention_config.enable_hybrid_attention           = true;
    mc.hybrid_attention_config.enable_independent_kv_cache_pools = true;
    setDsv4KvCacheSpecs(mc);
    return mc;
}

static ModelConfig makeProModelConfig() {
    ModelConfig mc;
    mc.num_layers                   = 61;
    mc.hidden_size                  = 7168;
    mc.attn_config.head_num         = 128;
    mc.attn_config.kv_head_num      = 1;
    mc.attn_config.size_per_head    = 512;
    mc.attn_config.rope_head_dim    = 64;
    mc.attn_config.sliding_window   = 128;
    mc.attn_config.indexer_head_dim = 128;
    mc.attn_config.indexer_head_num = 64;
    mc.attn_config.indexer_topk     = 1024;
    mc.attn_config.o_groups         = 16;
    mc.attn_config.o_lora_rank      = 1024;
    std::vector<int> ratios;
    ratios.push_back(128);
    ratios.push_back(128);
    for (int i = 2; i < 61; i++) {
        ratios.push_back((i % 2 == 0) ? 4 : 128);
    }
    ratios.push_back(0);
    mc.attn_config.layer_compress_ratios = ratios;
    setDsv4KvCacheSpecs(mc);
    return mc;
}

// Build a DSV4 7-pool CacheConfig (uses use_independent_block_pools=true).
static CacheConfig makeDSV4HybridPoolConfig(uint32_t block_num = 200) {
    auto mc                                                      = makeProModelConfig();
    mc.hybrid_attention_config.enable_hybrid_attention           = true;
    mc.hybrid_attention_config.enable_independent_kv_cache_pools = true;
    ParallelismConfig pc;
    KVCacheConfig     kv_cache_config;
    kv_cache_config.seq_size_per_block        = 128;
    kv_cache_config.kernel_seq_size_per_block = 128;
    auto config      = CacheConfigCreator::createBasicConfig(mc, pc, kv_cache_config, false, 0);
    config.block_num = block_num;
    return config;
}

static void setExplicitBlocksForGroup(CacheConfig& config, size_t group_id, uint32_t block_num) {
    ASSERT_LT(group_id, static_cast<size_t>(config.groupNums()));
    std::vector<CacheGroupPolicy> policies;
    policies.reserve(static_cast<size_t>(config.groupNums()));
    for (size_t gid = 0; gid < static_cast<size_t>(config.groupNums()); ++gid) {
        policies.push_back(config.policyForGroup(gid));
    }
    policies[group_id].explicit_block_num = block_num;
    config.setGroupPolicies(policies);
}

static size_t firstExplicitIndependentGroup(const CacheConfig& config) {
    for (size_t gid = 0; gid < static_cast<size_t>(config.groupNums()); ++gid) {
        const auto policy = config.policyForGroup(gid);
        if (policy.evict_policy == CacheEvictPolicy::INDEPENDENT && policy.explicit_block_num > 0) {
            return gid;
        }
    }
    ADD_FAILURE() << "missing explicit independent cache group";
    return 0;
}

static CompleteTokenIdsPtr makeCompleteTokenIds(int batch_size, int seq_length, int seq_size_per_block) {
    auto  cti        = std::make_shared<CompleteTokenIds>(batch_size, batch_size, seq_length + 64, seq_size_per_block);
    auto  input_ids  = torch::empty({(int64_t)seq_length}, torch::kInt32);
    auto* token_data = input_ids.data_ptr<int32_t>();
    for (int i = 0; i < seq_length; ++i) {
        token_data[i] = i + 1;
    }
    auto gi             = std::make_shared<GenerateInput>();
    gi->input_ids       = input_ids;
    gi->generate_config = std::make_shared<GenerateConfig>();
    cti->init(gi);
    return cti;
}

static BatchKVCacheResourcePtr makeBatchResource(int batch_size, const CacheConfig& config) {
    auto res = std::make_shared<BatchKVCacheResource>();
    res->resetBatchSize(batch_size);
    res->initGroups(config.groupNums(),
                    static_cast<int>(config.layer_all_num),
                    config.layerGroupIdsSnapshot(),
                    config.kernelBlocksPerKvBlock(),
                    config.groupTypesSnapshot());
    return res;
}

static std::vector<uint32_t> groupBlockNumsSnapshot(const CacheConfig& config) {
    std::vector<uint32_t> block_nums;
    block_nums.reserve(static_cast<size_t>(config.groupNums()));
    for (size_t gid = 0; gid < static_cast<size_t>(config.groupNums()); ++gid) {
        block_nums.push_back(config.blockNumForGroup(gid));
    }
    return block_nums;
}

static void setGroupBlockNums(CacheConfig& config, const std::vector<uint32_t>& block_nums) {
    std::vector<size_t> kv_strides;
    std::vector<size_t> scale_strides;
    kv_strides.reserve(static_cast<size_t>(config.groupNums()));
    scale_strides.reserve(static_cast<size_t>(config.groupNums()));
    for (size_t gid = 0; gid < static_cast<size_t>(config.groupNums()); ++gid) {
        kv_strides.push_back(config.kvBlockStrideBytesForGroup(gid));
        scale_strides.push_back(config.kvScaleStrideBytesForGroup(gid));
    }
    config.setGroupBlockLayout(block_nums, kv_strides, scale_strides);
}

static size_t validBlockCount(const BlockIndicesType& blocks) {
    return static_cast<size_t>(
        std::count_if(blocks.begin(), blocks.end(), [](BlockIdxType block) { return !isNullBlockIdx(block); }));
}

// Create a pre-init HybridPoolKVCacheAllocator; tests inject BlockTreeCache after init.
static HybridPoolKVCacheAllocatorPtr makeAllocator(const CacheConfig& config, RoleType role_type = RoleType::PDFUSION) {
    auto allocator =
        std::make_shared<HybridPoolKVCacheAllocator>(config, AllocationType::DEVICE, nullptr, 0, role_type);
    return allocator;
}

class RecordingMemoryUtil: public MemoryUtil {
public:
    bool regUserMr(void*, uint64_t, bool gpu, uint64_t) override {
        reg_gpu_flags.push_back(gpu);
        return true;
    }

    bool deregUserMr(void*, bool gpu) override {
        dereg_gpu_flags.push_back(gpu);
        return true;
    }

    bool isMemoryMr(void*, uint64_t, bool, bool) override {
        return false;
    }

    bool findMemoryMr(void*, void*, uint64_t, bool, bool) override {
        return false;
    }

    bool isRdmaMode() override {
        return true;
    }

    std::vector<bool> reg_gpu_flags;
    std::vector<bool> dereg_gpu_flags;
};

class RecordingCacheStore: public CacheStore {
public:
    explicit RecordingCacheStore(std::shared_ptr<MemoryUtil> memory_util): memory_util_(std::move(memory_util)) {}

    void store(const std::shared_ptr<RequestBlockBuffer>&, CacheStoreStoreDoneCallback callback) override {
        if (callback) {
            callback(false, CacheStoreErrorCode::InvalidParams);
        }
    }

    void load(const std::shared_ptr<RequestBlockBuffer>&,
              CacheStoreLoadDoneCallback callback,
              const std::string&,
              uint32_t,
              uint32_t,
              uint32_t,
              int,
              int) override {
        if (callback) {
            callback(false, CacheStoreErrorCode::InvalidParams);
        }
    }

    std::shared_ptr<LoadContext> loadBuffers(const std::vector<std::shared_ptr<RequestBlockBuffer>>&,
                                             const std::string&,
                                             uint32_t,
                                             uint32_t,
                                             int64_t,
                                             LoadContext::CheckCancelFunc,
                                             int,
                                             int) override {
        return nullptr;
    }

    std::shared_ptr<StoreContext> storeBuffers(const std::vector<std::shared_ptr<RequestBlockBuffer>>&,
                                               int64_t) override {
        return nullptr;
    }

    std::shared_ptr<RemoteStoreTask>
    submitRemoteStoreTask(const std::shared_ptr<RemoteStoreRequest>&,
                          const std::shared_ptr<CacheStoreRemoteStoreMetricsCollector>&,
                          RemoteStoreTask::CheckCancelFunc) override {
        return nullptr;
    }

    void releaseRemoteStoreTask(const std::shared_ptr<RemoteStoreTask>&) override {}

    bool regUserBuffers(const std::vector<std::shared_ptr<BlockBuffer>>&) override {
        return true;
    }

    std::shared_ptr<BlockBuffer> findUserBuffer(const std::string&) override {
        return nullptr;
    }

    const std::shared_ptr<MemoryUtil>& getMemoryUtil() const override {
        return memory_util_;
    }

    void debugInfo() override {}

private:
    std::shared_ptr<MemoryUtil> memory_util_;
};

struct RealCacheSeed {
    std::vector<BlockIndicesType> group_blocks;
};

static RealCacheSeed seedRealCachePath(const HybridPoolKVCacheAllocatorPtr& allocator,
                                       const CacheConfig&                   config,
                                       const CacheKeysType&                 cache_keys,
                                       int                                  seq_length) {
    RealCacheSeed seed;
    auto          resource = makeBatchResource(/*batch_size=*/1, config);
    resource->setBatchCacheKeys(/*batch_id=*/0, cache_keys);
    auto tokens = makeCompleteTokenIds(/*batch_size=*/1, seq_length, static_cast<int>(config.seq_size_per_block));

    MallocInfo malloc_info{resource, tokens};
    malloc_info.reuse_cache         = false;
    malloc_info.enable_device_cache = false;
    const auto result               = allocator->malloc(malloc_info);
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.async_context, nullptr);
    if (!result.success) {
        return seed;
    }

    seed.group_blocks.reserve(static_cast<size_t>(resource->groupNums()));
    for (int gid = 0; gid < resource->groupNums(); ++gid) {
        seed.group_blocks.push_back(resource->blocks(/*batch_id=*/0, gid));
    }

    allocator->insertIntoCache(InsertInfo{resource, tokens, /*is_resident=*/false});
    allocator->free(FreeInfo{resource, tokens});
    EXPECT_EQ(resource->curBlocksNum(), 0u);
    return seed;
}

// Single-count DeviceBlockPool exposes only free/total block counts (the legacy per-pool
// request/connector/blockCache ref-count columns and legacy aggregate availability are gone; the
// three-way holder split is not recoverable at the pool). Each independent pool still
// reports its own free/total, which is what these rollback checks rely on.
struct PoolCounters {
    size_t free_blocks;
    size_t total_blocks;
};

static std::vector<PoolCounters> snapshotPoolCounters(const HybridPoolKVCacheAllocatorPtr& allocator) {
    std::vector<PoolCounters> counters;
    counters.reserve(allocator->groupBlockPools().size());
    for (const auto& pool : allocator->groupBlockPools()) {
        counters.push_back({pool->freeBlocksNum(), pool->totalBlocksNum()});
    }
    return counters;
}

static void expectPoolCountersEq(const HybridPoolKVCacheAllocatorPtr& allocator,
                                 const std::vector<PoolCounters>&     expected) {
    ASSERT_EQ(allocator->groupBlockPools().size(), expected.size());
    for (size_t gid = 0; gid < expected.size(); ++gid) {
        const auto& pool = allocator->groupBlockPools()[gid];
        EXPECT_EQ(pool->freeBlocksNum(), expected[gid].free_blocks) << "gid=" << gid;
        EXPECT_EQ(pool->totalBlocksNum(), expected[gid].total_blocks) << "gid=" << gid;
    }
}

static size_t freePlusDeviceReclaimCandidates(const HybridPoolKVCacheAllocatorPtr& allocator,
                                              const std::shared_ptr<BlockTreeCache>& cache) {
    return allocator->freeBlocksNum() + cache->getStats().device_heap_total_size;
}

class HybridPoolKVCacheAllocatorTest: public ::testing::Test {
protected:
    void SetUp() override {
        rtp_llm::initLogger();
        createDevice();
    }

    // In the refactored design the DeviceKVCacheGroups live inside BlockTreeCache, not the
    // allocator. makeAllocator() returns a pre-init allocator; after init() builds the per-group
    // device pools, tests must inject a BlockTreeCache (mirrors KVCacheManager wiring) before any
    // group access (convertIndexToAddr / malloc / reserve / reuse). The fixture owns the
    // BlockTreeCache so it outlives the allocator's raw pointer.
    bool injectBlockTreeCache(const HybridPoolKVCacheAllocatorPtr& allocator,
                              const CacheConfig&                   config,
                              const KVCacheConfig&                 kv_cache_config) {
        auto btc = createBlockTreeCache(config, kv_cache_config, allocator);
        if (!btc) {
            return false;
        }
        allocator->setBlockTreeCache(btc.get());
        block_tree_caches_.push_back(std::move(btc));
        return true;
    }

    bool injectBlockTreeCache(const HybridPoolKVCacheAllocatorPtr& allocator, const CacheConfig& config) {
        KVCacheConfig kv_cache_config;  // device-only: memory/disk tiers disabled
        return injectBlockTreeCache(allocator, config, kv_cache_config);
    }

    std::vector<BlockTreeCachePtr> block_tree_caches_;
};

class HybridPoolPreflightTestPeer: public HybridPoolKVCacheAllocator {
public:
    using HybridPoolKVCacheAllocator::HybridPoolKVCacheAllocator;
    using HybridKVCacheAllocator::preflightLoadBackMappings;
};

struct HybridPoolPreflightEnvironment {
    BlockTreeCachePtr              cache;
    std::shared_ptr<HostBlockPool> host_pool;
    BlockIdxType                   source_block{NULL_BLOCK_IDX};
};

static HybridPoolPreflightEnvironment
makeHybridPoolPreflightEnvironment(const std::vector<DeviceBlockPoolPtr>& device_pools) {
    HybridPoolPreflightEnvironment environment;
    if (device_pools.size() < 2) {
        return environment;
    }

    auto host_config                  = std::make_shared<HostBlockPoolConfig>();
    host_config->pool_type            = BlockPoolType::HOST;
    host_config->pool_name            = "hybrid_pool_preflight_host";
    host_config->physical_block_count = 3;
    host_config->payload_bytes        = 1;
    host_config->stride_bytes         = 4096;
    host_config->enable_pinned        = false;
    host_config->alignment            = 4096;
    environment.host_pool             = std::make_shared<HostBlockPool>(host_config);
    if (!environment.host_pool->init()) {
        return environment;
    }

    auto primary                = std::make_shared<FullComponentGroup>();
    primary->component_group_id = 0;
    primary->setDevicePools({device_pools[0], device_pools[1]});
    primary->setHostPool(environment.host_pool);

    BlockTreeCacheConfig cache_config;
    cache_config.enable_device_cache = true;
    cache_config.enable_memory_cache = true;
    cache_config.enable_load_back    = true;
    environment.cache =
        std::make_shared<BlockTreeCache>(std::make_unique<BlockTree>(1),
                                         std::vector<ComponentGroupPtr>{primary},
                                         std::vector<Component>{},
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

static std::shared_ptr<LoadBackTicket> captureHybridPoolPreflightTicket(BlockTreeCache& cache) {
    BlockTreeMatchResult result = cache.match({100});
    EXPECT_NE(result.load_back_ticket, nullptr);
    if (result.load_back_ticket == nullptr) {
        return nullptr;
    }
    EXPECT_EQ(result.load_back_ticket->items().size(), 1u);
    return std::move(result.load_back_ticket);
}

class ScopedHybridPoolDiskCacheDirectory {
public:
    ScopedHybridPoolDiskCacheDirectory() {
        std::string       pattern = "/tmp/rtp_llm_hybrid_pool_load_back_XXXXXX";
        std::vector<char> writable(pattern.begin(), pattern.end());
        writable.push_back('\0');
        char* result = ::mkdtemp(writable.data());
        EXPECT_NE(result, nullptr);
        if (result != nullptr) {
            path_ = result;
        }
    }

    ~ScopedHybridPoolDiskCacheDirectory() {
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

static KVCacheConfig makeHybridPoolTieredConfig(Tier source_tier, const std::string& disk_path) {
    KVCacheConfig config;
    config.enable_memory_cache        = true;
    config.enable_tiered_memory_cache = true;
    config.memory_cache_size_mb       = 1;
    config.enable_memory_cache_disk   = source_tier == Tier::DISK;
    config.memory_cache_disk_size_mb  = source_tier == Tier::DISK ? 1 : 0;
    config.memory_cache_disk_paths    = disk_path;
    return config;
}

static void seedHybridPoolLowerTier(BlockTreeCache& cache, Tier source_tier, CacheKeyType key) {
    const std::vector<ComponentGroupPtr>& groups = cache.componentGroups();
    std::vector<std::vector<GroupSlot>>   slots(1, std::vector<GroupSlot>(groups.size()));
    for (size_t group_id = 0; group_id < groups.size(); ++group_id) {
        const BlockIdxType source_block = groups[group_id]->allocateSingleBlock(source_tier);
        ASSERT_NE(source_block, NULL_BLOCK_IDX) << "component_group_id=" << group_id;
        if (source_tier == Tier::HOST) {
            slots[0][group_id].host_block = source_block;
        } else {
            slots[0][group_id].disk_slot = source_block;
        }
    }
    ASSERT_NE(cache.tree()->insertNode(nullptr, {key}, slots).leaf, nullptr);
}

TEST_F(HybridPoolKVCacheAllocatorTest, RealHostAndDiskTicketsLoadBackIntoRequestOwnedTargets) {
    for (CacheGroupType second_type : {CacheGroupType::FULL, CacheGroupType::SWA}) {
        SCOPED_TRACE(second_type == CacheGroupType::FULL ? "LINEAR+FULL" : "LINEAR+SWA");
        for (Tier source_tier : {Tier::HOST, Tier::DISK}) {
            SCOPED_TRACE(source_tier == Tier::HOST ? "HOST" : "DISK");
            ScopedHybridPoolDiskCacheDirectory disk_directory;

            const CacheConfig config    = second_type == CacheGroupType::FULL ? makeTinyMultiPoolHybridConfig() :
                                                                                makeTinySwaMultiPoolHybridConfig();
            auto              allocator = makeAllocator(config);
            ASSERT_TRUE(allocator->init());
            ASSERT_TRUE(injectBlockTreeCache(
                allocator, config, makeHybridPoolTieredConfig(source_tier, disk_directory.string())));
            BlockTreeCachePtr cache = block_tree_caches_.back();
            seedHybridPoolLowerTier(*cache, source_tier, /*key=*/100);

            BatchKVCacheResourcePtr resource = makeBatchResource(/*batch_size=*/1, config);
            resource->setBatchCacheKeys(0, CacheKeysType{100, 200});
            CompleteTokenIdsPtr token_ids = makeCompleteTokenIds(
                /*batch_size=*/1, /*seq_length=*/8, /*seq_size_per_block=*/static_cast<int>(config.seq_size_per_block));

            MallocInfo malloc_info{resource, token_ids};
            malloc_info.enable_device_cache = true;
            MallocResult result             = allocator->malloc(malloc_info);
            ASSERT_TRUE(result.success);
            EXPECT_EQ(result.reuse_len, static_cast<int>(config.seq_size_per_block));
            ASSERT_NE(result.async_context, nullptr);
            result.async_context->waitDone();
            ASSERT_TRUE(result.async_context->success()) << result.async_context->errorInfo().ToString();
            EXPECT_EQ(resource->cacheResource(0).deviceReuseBlockNum(), 1u);

            BlockTreeFindResult find = cache->tree()->findNode({100});
            ASSERT_NE(find.matched_node, nullptr);
            ASSERT_EQ(config.groupNums(), 2);
            ASSERT_EQ(find.matched_node->group_slots.size(), 2u);
            for (int gid = 0; gid < config.groupNums(); ++gid) {
                ASSERT_EQ(resource->blocks(0, gid).size(), 2u) << "gid=" << gid;
                const GroupSlot& slot = find.matched_node->group_slots[static_cast<size_t>(gid)];
                ASSERT_EQ(slot.device_blocks.size(), 1u) << "gid=" << gid;
                EXPECT_EQ(slot.device_blocks.front(), resource->blocks(0, gid).front()) << "gid=" << gid;
            }

            allocator->free(FreeInfo{resource, token_ids});
            result.async_context.reset();
            allocator.reset();
            cache.reset();
            block_tree_caches_.clear();
        }
    }
}

TEST_F(HybridPoolKVCacheAllocatorTest, LowerTierLinearTargetIsChargedBeforePerPoolReserveAdmission) {
    // Pool capacities exclude reserved block zero: LINEAR has 3 available blocks,
    // FULL has 6. reserve=6 distributes 2 to LINEAR and 4 to FULL. Current FULL-only
    // accounting needs one LINEAR plus two FULL blocks and exactly admits both pools.
    // The lower-tier-only LINEAR target is a second physical LINEAR allocation, so
    // all-target accounting must reject that pool (2 + 2 > 3).
    const CacheConfig config    = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/4, /*full_block_num=*/7);
    auto              allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());
    ASSERT_TRUE(injectBlockTreeCache(allocator, config, makeHybridPoolTieredConfig(Tier::HOST, /*disk_path=*/"")));
    BlockTreeCachePtr cache = block_tree_caches_.back();
    seedHybridPoolLowerTier(*cache, Tier::HOST, /*key=*/100);

    BlockTreeFindResult source_find = cache->tree()->findNode({100});
    ASSERT_NE(source_find.matched_node, nullptr);
    ASSERT_EQ(source_find.matched_node->group_slots.size(), cache->componentGroups().size());
    std::vector<size_t> source_ref_baselines;
    source_ref_baselines.reserve(cache->componentGroups().size());
    for (size_t group_id = 0; group_id < cache->componentGroups().size(); ++group_id) {
        const auto& group = cache->componentGroups()[group_id];
        ASSERT_NE(group->hostPool(), nullptr) << "component_group_id=" << group_id;
        const BlockIdxType source_block = source_find.matched_node->group_slots[group_id].host_block;
        ASSERT_NE(source_block, NULL_BLOCK_IDX) << "component_group_id=" << group_id;
        source_ref_baselines.push_back(group->hostPool()->refCount(source_block));
    }
    const auto linear_group = std::find_if(
        cache->componentGroups().begin(), cache->componentGroups().end(), [](const ComponentGroupPtr& group) {
            return group != nullptr && group->group_type == CacheGroupType::LINEAR;
        });
    ASSERT_NE(linear_group, cache->componentGroups().end());
    ASSERT_GE((*linear_group)->component_group_id, 0);
    const GroupSlot& linear_source_slot =
        source_find.matched_node->group_slots[static_cast<size_t>((*linear_group)->component_group_id)];
    ASSERT_NE(linear_source_slot.host_block, NULL_BLOCK_IDX);
    EXPECT_TRUE(linear_source_slot.device_blocks.empty())
        << "the pending LINEAR target must be lower-tier-only before allocator admission";

    const std::vector<PoolCounters> counters_before = snapshotPoolCounters(allocator);
    ASSERT_EQ(counters_before.size(), 2u);
    ASSERT_EQ(counters_before[0].free_blocks, 3u);
    ASSERT_EQ(counters_before[1].free_blocks, 6u);
    allocator->setReserveBlockNum(6);
    const std::vector<KVCachePoolMetricsSnapshot> reserve_snapshots = allocator->poolMetricsSnapshots();
    ASSERT_EQ(reserve_snapshots.size(), 2u);
    EXPECT_EQ(reserve_snapshots[0].free_blocks, 3u);
    EXPECT_EQ(reserve_snapshots[1].free_blocks, 6u);
    EXPECT_EQ(reserve_snapshots[0].reserve_blocks, 2u);
    EXPECT_EQ(reserve_snapshots[1].reserve_blocks, 4u);
    EXPECT_EQ(reserve_snapshots[0].free_blocks - 1u, 2u);
    EXPECT_EQ(reserve_snapshots[1].free_blocks - 1u, 5u);

    BatchKVCacheResourcePtr resource = makeBatchResource(/*batch_size=*/1, config);
    resource->setBatchCacheKeys(0, CacheKeysType{100, 200});
    CompleteTokenIdsPtr token_ids = makeCompleteTokenIds(
        /*batch_size=*/1, /*seq_length=*/8, /*seq_size_per_block=*/static_cast<int>(config.seq_size_per_block));

    MallocInfo malloc_info{resource, token_ids};
    malloc_info.enable_device_cache = true;
    malloc_info.verbose             = false;
    MallocResult result             = allocator->malloc(malloc_info);

    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.async_context, nullptr);
    EXPECT_EQ(resource->curBlocksNum(), 0u);
    for (int gid = 0; gid < config.groupNums(); ++gid) {
        EXPECT_EQ(resource->blocksNum(0, gid), 0u) << "gid=" << gid;
    }
    EXPECT_EQ(resource->cacheResource(0).deviceReuseBlockNum(), 0u);
    EXPECT_EQ(allocator->activeTreeCachedBlocksNum(), 0u);
    expectPoolCountersEq(allocator, counters_before);

    source_find = cache->tree()->findNode({100});
    ASSERT_NE(source_find.matched_node, nullptr);
    for (size_t group_id = 0; group_id < cache->componentGroups().size(); ++group_id) {
        const auto&        group        = cache->componentGroups()[group_id];
        const BlockIdxType source_block = source_find.matched_node->group_slots[group_id].host_block;
        EXPECT_EQ(group->hostPool()->refCount(source_block), source_ref_baselines[group_id])
            << "component_group_id=" << group_id;
        EXPECT_EQ(source_find.matched_node->group_slots[group_id].transfer_state, SlotTransferState::IDLE)
            << "component_group_id=" << group_id;
    }
}

TEST_F(HybridPoolKVCacheAllocatorTest, ReserveRejectDoesNotEvictUnrelatedDeviceEntry) {
    const CacheConfig config    = makeTinySwaMultiPoolHybridConfig(/*linear_block_num=*/4, /*swa_block_num=*/4);
    auto              allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());
    ASSERT_TRUE(injectBlockTreeCache(allocator, config, makeHybridPoolTieredConfig(Tier::HOST, /*disk_path=*/"")));
    BlockTreeCachePtr cache = block_tree_caches_.back();
    ASSERT_EQ(config.typeForGroup(0), CacheGroupType::LINEAR);
    ASSERT_EQ(config.typeForGroup(1), CacheGroupType::SWA);

    BatchKVCacheResourcePtr cached_resource = makeBatchResource(/*batch_size=*/1, config);
    cached_resource->setBatchCacheKeys(0, CacheKeysType{900});
    CompleteTokenIdsPtr cached_tokens = makeCompleteTokenIds(
        /*batch_size=*/1, /*seq_length=*/4, /*seq_size_per_block=*/static_cast<int>(config.seq_size_per_block));
    MallocInfo cached_malloc{cached_resource, cached_tokens};
    cached_malloc.enable_device_cache = false;
    cached_malloc.reuse_cache         = false;
    ASSERT_TRUE(allocator->malloc(cached_malloc).success);
    ASSERT_EQ(cached_resource->blocksNum(0, /*gid=*/0), 1u);
    ASSERT_EQ(cached_resource->blocksNum(0, /*gid=*/1), 1u);
    const BlockIdxType cached_linear_block = cached_resource->blocks(0, /*gid=*/0).front();
    const BlockIdxType cached_swa_block    = cached_resource->blocks(0, /*gid=*/1).front();

    allocator->insertIntoCache(InsertInfo{cached_resource, cached_tokens, /*is_resident=*/false});
    allocator->free(FreeInfo{cached_resource, cached_tokens});
    ASSERT_EQ(cached_resource->curBlocksNum(), 0u);

    BlockTreeFindResult cached_find = cache->tree()->findNode({900});
    ASSERT_NE(cached_find.matched_node, nullptr);
    ASSERT_EQ(cached_find.matched_node->group_slots.size(), 2u);
    ASSERT_EQ(cached_find.matched_node->group_slots[0].device_blocks, (BlockIndicesType{cached_linear_block}));
    ASSERT_EQ(cached_find.matched_node->group_slots[1].device_blocks, (BlockIndicesType{cached_swa_block}));

    const auto& pools = allocator->groupBlockPools();
    ASSERT_EQ(pools.size(), 2u);
    EXPECT_EQ(pools[0]->refCount(cached_linear_block), 1u);
    EXPECT_EQ(pools[1]->refCount(cached_swa_block), 1u);

    BlockTreeMatchResult cached_match_before = cache->match({900});
    ASSERT_EQ(cached_match_before.matched_blocks, 1u);
    EXPECT_EQ(cached_match_before.group_block_indices.at(0), (BlockIndicesType{cached_linear_block}));
    EXPECT_EQ(cached_match_before.group_block_indices.at(1), (BlockIndicesType{cached_swa_block}));
    cache->releaseMatchedBlocks(cached_match_before.matched_block_sets);
    cached_match_before.matched_block_sets.clear();
    EXPECT_EQ(pools[0]->refCount(cached_linear_block), 1u);
    EXPECT_EQ(pools[1]->refCount(cached_swa_block), 1u);

    BatchKVCacheResourcePtr occupying_resource = makeBatchResource(/*batch_size=*/1, config);
    occupying_resource->setBatchCacheKeys(0, CacheKeysType{700, 701});
    CompleteTokenIdsPtr occupying_tokens = makeCompleteTokenIds(
        /*batch_size=*/1, /*seq_length=*/8, /*seq_size_per_block=*/static_cast<int>(config.seq_size_per_block));
    MallocInfo occupying_malloc{occupying_resource, occupying_tokens};
    occupying_malloc.enable_device_cache = false;
    occupying_malloc.reuse_cache         = false;
    ASSERT_TRUE(allocator->malloc(occupying_malloc).success);
    ASSERT_EQ(occupying_resource->blocksNum(0, /*gid=*/0), 2u);
    ASSERT_EQ(occupying_resource->blocksNum(0, /*gid=*/1), 2u);
    const BlockIndicesType occupying_linear_blocks = occupying_resource->blocks(0, /*gid=*/0);
    const BlockIndicesType occupying_swa_blocks    = occupying_resource->blocks(0, /*gid=*/1);

    const std::vector<PoolCounters> counters_before = snapshotPoolCounters(allocator);
    ASSERT_EQ(counters_before.size(), 2u);
    ASSERT_EQ(counters_before[0].free_blocks, 0u);
    ASSERT_EQ(counters_before[1].free_blocks, 0u);
    for (const BlockIdxType block : occupying_linear_blocks) {
        EXPECT_EQ(pools[0]->refCount(block), 1u);
    }
    for (const BlockIdxType block : occupying_swa_blocks) {
        EXPECT_EQ(pools[1]->refCount(block), 1u);
    }

    allocator->setReserveBlockNum(2);
    const std::vector<KVCachePoolMetricsSnapshot> reserve_snapshots = allocator->poolMetricsSnapshots();
    ASSERT_EQ(reserve_snapshots.size(), 2u);
    EXPECT_EQ(reserve_snapshots[0].free_blocks, 0u);
    EXPECT_EQ(reserve_snapshots[1].free_blocks, 0u);
    EXPECT_EQ(reserve_snapshots[0].active_tree_cached_blocks, 0u);
    EXPECT_EQ(reserve_snapshots[1].active_tree_cached_blocks, 0u);
    EXPECT_EQ(reserve_snapshots[0].reserve_blocks, 0u);
    EXPECT_EQ(reserve_snapshots[1].reserve_blocks, 0u);
    const size_t reclaim_candidates_before = cache->getStats().device_heap_total_size;
    EXPECT_EQ(reclaim_candidates_before, 2u);

    seedHybridPoolLowerTier(*cache, Tier::HOST, /*key=*/100);
    BlockTreeFindResult source_find = cache->tree()->findNode({100});
    ASSERT_NE(source_find.matched_node, nullptr);
    ASSERT_EQ(source_find.matched_node->group_slots.size(), cache->componentGroups().size());
    std::vector<size_t> source_ref_baselines;
    source_ref_baselines.reserve(cache->componentGroups().size());
    for (size_t group_id = 0; group_id < cache->componentGroups().size(); ++group_id) {
        const auto& group = cache->componentGroups()[group_id];
        ASSERT_NE(group->hostPool(), nullptr) << "component_group_id=" << group_id;
        const GroupSlot& source_slot = source_find.matched_node->group_slots[group_id];
        ASSERT_NE(source_slot.host_block, NULL_BLOCK_IDX) << "component_group_id=" << group_id;
        EXPECT_TRUE(source_slot.device_blocks.empty()) << "component_group_id=" << group_id;
        source_ref_baselines.push_back(group->hostPool()->refCount(source_slot.host_block));
    }
    const size_t tree_nodes_before = cache->getStats().tree_node_count;

    BatchKVCacheResourcePtr rejected_resource = makeBatchResource(/*batch_size=*/1, config);
    rejected_resource->setBatchCacheKeys(0, CacheKeysType{100, 200});
    CompleteTokenIdsPtr rejected_tokens = makeCompleteTokenIds(
        /*batch_size=*/1, /*seq_length=*/8, /*seq_size_per_block=*/static_cast<int>(config.seq_size_per_block));
    MallocInfo rejected_malloc{rejected_resource, rejected_tokens};
    rejected_malloc.enable_device_cache = true;
    rejected_malloc.reuse_cache         = true;
    rejected_malloc.verbose             = false;
    MallocResult result                 = allocator->malloc(rejected_malloc);

    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.async_context, nullptr);
    EXPECT_EQ(rejected_resource->curBlocksNum(), 0u);
    EXPECT_EQ(rejected_resource->blocksNum(0, /*gid=*/0), 0u);
    EXPECT_EQ(rejected_resource->blocksNum(0, /*gid=*/1), 0u);
    EXPECT_EQ(rejected_resource->cacheResource(0).deviceReuseBlockNum(), 0u);
    expectPoolCountersEq(allocator, counters_before);
    EXPECT_EQ(cache->getStats().device_heap_total_size, reclaim_candidates_before);
    EXPECT_EQ(cache->getStats().tree_node_count, tree_nodes_before);

    EXPECT_EQ(occupying_resource->blocks(0, /*gid=*/0), occupying_linear_blocks);
    EXPECT_EQ(occupying_resource->blocks(0, /*gid=*/1), occupying_swa_blocks);
    for (const BlockIdxType block : occupying_linear_blocks) {
        EXPECT_EQ(pools[0]->refCount(block), 1u);
    }
    for (const BlockIdxType block : occupying_swa_blocks) {
        EXPECT_EQ(pools[1]->refCount(block), 1u);
    }

    cached_find = cache->tree()->findNode({900});
    ASSERT_NE(cached_find.matched_node, nullptr);
    ASSERT_EQ(cached_find.matched_node->group_slots[0].device_blocks, (BlockIndicesType{cached_linear_block}));
    ASSERT_EQ(cached_find.matched_node->group_slots[1].device_blocks, (BlockIndicesType{cached_swa_block}));
    BlockTreeMatchResult cached_match_after = cache->match({900});
    ASSERT_EQ(cached_match_after.matched_blocks, 1u);
    EXPECT_EQ(cached_match_after.group_block_indices.at(0), (BlockIndicesType{cached_linear_block}));
    EXPECT_EQ(cached_match_after.group_block_indices.at(1), (BlockIndicesType{cached_swa_block}));
    cache->releaseMatchedBlocks(cached_match_after.matched_block_sets);
    cached_match_after.matched_block_sets.clear();
    EXPECT_EQ(pools[0]->refCount(cached_linear_block), 1u);
    EXPECT_EQ(pools[1]->refCount(cached_swa_block), 1u);

    source_find = cache->tree()->findNode({100});
    ASSERT_NE(source_find.matched_node, nullptr);
    for (size_t group_id = 0; group_id < cache->componentGroups().size(); ++group_id) {
        const auto&      group       = cache->componentGroups()[group_id];
        const GroupSlot& source_slot = source_find.matched_node->group_slots[group_id];
        EXPECT_EQ(group->hostPool()->refCount(source_slot.host_block), source_ref_baselines[group_id])
            << "component_group_id=" << group_id;
        EXPECT_TRUE(source_slot.device_blocks.empty()) << "component_group_id=" << group_id;
        EXPECT_EQ(source_slot.transfer_state, SlotTransferState::IDLE) << "component_group_id=" << group_id;
    }

    allocator->free(FreeInfo{occupying_resource, occupying_tokens});
}

TEST_F(HybridPoolKVCacheAllocatorTest, ProtectedPreflightRejectsMalformedRealTicketWithoutMutation) {
    const CacheConfig config    = makeTinyMultiPoolHybridConfig();
    auto              allocator = std::make_shared<HybridPoolPreflightTestPeer>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    HybridPoolPreflightEnvironment environment = makeHybridPoolPreflightEnvironment(allocator->groupBlockPools());
    ASSERT_NE(environment.cache, nullptr);
    ASSERT_NE(environment.host_pool, nullptr);
    ASSERT_NE(environment.source_block, NULL_BLOCK_IDX);
    allocator->setBlockTreeCache(environment.cache.get());

    const size_t                    source_ref_baseline  = environment.host_pool->refCount(environment.source_block);
    const std::vector<PoolCounters> allocator_counters   = snapshotPoolCounters(allocator);
    const size_t                    tree_node_count      = environment.cache->getStats().tree_node_count;
    const CopyEnginePtr             copy_engine_identity = environment.cache->copyEngine();

    EXPECT_TRUE(allocator->preflightLoadBackMappings(nullptr));
    auto empty_ticket_registry = std::make_shared<LoadBackTicketRegistry>(LoadBackTicketRegistry::CommitCallback{},
                                                                          LoadBackTicketRegistry::AbortCallback{});
    std::shared_ptr<LoadBackTicket> empty_ticket = empty_ticket_registry->createTicket({});
    ASSERT_NE(empty_ticket, nullptr);
    EXPECT_TRUE(allocator->preflightLoadBackMappings(empty_ticket));

    {
        std::shared_ptr<LoadBackTicket> ticket = captureHybridPoolPreflightTicket(*environment.cache);
        ASSERT_NE(ticket, nullptr);
        ASSERT_EQ(ticket->items().size(), 1u);
        EXPECT_EQ(ticket->items().front().device_group_ids, (std::vector<int>{1, 0}));
        EXPECT_EQ(environment.host_pool->refCount(environment.source_block), source_ref_baseline + 1);
        EXPECT_TRUE(allocator->preflightLoadBackMappings(ticket));
        EXPECT_EQ(ticket->items().front().device_group_ids, (std::vector<int>{1, 0}));
        EXPECT_TRUE(ticket->items().front().target_device_blocks.empty());
        expectPoolCountersEq(allocator, allocator_counters);
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
        std::shared_ptr<LoadBackTicket> ticket = captureHybridPoolPreflightTicket(*environment.cache);
        ASSERT_NE(ticket, nullptr);
        ASSERT_EQ(ticket->items().size(), 1u);
        EXPECT_EQ(ticket->items().front().device_group_ids, (std::vector<int>{1, 0}));
        EXPECT_EQ(environment.host_pool->refCount(environment.source_block), source_ref_baseline + 1);

        ticket->items().front().device_group_ids = mapping_case.device_group_ids;
        EXPECT_FALSE(allocator->preflightLoadBackMappings(ticket));
        EXPECT_EQ(ticket->items().front().device_group_ids, mapping_case.device_group_ids);
        EXPECT_TRUE(ticket->items().front().target_device_blocks.empty());
        expectPoolCountersEq(allocator, allocator_counters);
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

// ---------------------------------------------------------------------------
// Init / per-group pool creation
// ---------------------------------------------------------------------------

TEST_F(HybridPoolKVCacheAllocatorTest, InitCreatesIndependentBlockPoolPerGroup) {
    auto config    = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/6, /*full_block_num=*/8);
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());
    ASSERT_TRUE(injectBlockTreeCache(allocator, config));

    ASSERT_EQ(allocator->groupBlockPools().size(), 2u);
    EXPECT_NE(allocator->groupBlockPools()[0], allocator->groupBlockPools()[1]);

    // Per-pool totalBlocksNum = group_block_nums[gid] - 1 (block 0 reserved).
    EXPECT_EQ(allocator->groupBlockPools()[0]->totalBlocksNum(), 6u - 1u);
    EXPECT_EQ(allocator->groupBlockPools()[1]->totalBlocksNum(), 8u - 1u);
}

TEST_F(HybridPoolKVCacheAllocatorTest, SwaDefaultRegionGroupPoolUsesGpuBacking) {
    auto config    = makeTinySwaMultiPoolHybridConfig(/*linear_block_num=*/6, /*swa_block_num=*/8);
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());
    ASSERT_TRUE(injectBlockTreeCache(allocator, config));

    ASSERT_EQ(allocator->groupBlockPools().size(), 2u);
    EXPECT_EQ(allocator->groupBlockPools()[0]->where(), MemoryType::MEMORY_GPU);
    EXPECT_EQ(allocator->groupBlockPools()[1]->where(), MemoryType::MEMORY_GPU);
}

TEST_F(HybridPoolKVCacheAllocatorTest, GetBlockPoolReturnsNullptrInHybridPoolMode) {
    // HybridPoolKVCacheAllocator owns one DeviceBlockPool per group and does not
    // expose a single canonical block_pool_; getDeviceBlockPool() must return nullptr.
    auto config    = makeTinyMultiPoolHybridConfig();
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());
    ASSERT_TRUE(injectBlockTreeCache(allocator, config));
    EXPECT_EQ(allocator->getDeviceBlockPool(), nullptr);
}

// ---------------------------------------------------------------------------
// Aggregated counters
// ---------------------------------------------------------------------------

TEST_F(HybridPoolKVCacheAllocatorTest, TotalAndFreeBlocksAggregateAcrossGroups) {
    auto config    = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/6, /*full_block_num=*/8);
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());
    ASSERT_TRUE(injectBlockTreeCache(allocator, config));

    const size_t expected_total = (6u - 1u) + (8u - 1u);
    EXPECT_EQ(allocator->totalBlocksNum(), expected_total);
    EXPECT_EQ(allocator->freeBlocksNum(), expected_total);
}

TEST_F(HybridPoolKVCacheAllocatorTest, TokenAggregatorsUseDifferentCapacityScopes) {
    auto config = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/6, /*full_block_num=*/8);
    // Group 0 (LINEAR): seq_size_per_block=2 -> 5 blocks * 2 = 10
    // Group 1 (FULL):   seq_size_per_block=4 -> 7 blocks * 4 = 28
    config.group_seq_size_per_block = {2, 4};
    auto allocator                  = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    EXPECT_EQ(allocator->maxAvailableTokensNum(), 28u);
    EXPECT_EQ(allocator->availableTokensNum(), 28u);
    EXPECT_EQ(allocator->totalTokensNum(), 28u);
}

TEST_F(HybridPoolKVCacheAllocatorTest, TokenAggregatorsUseCPVirtualBlockSizeForFullGroups) {
    auto config                     = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/6, /*full_block_num=*/8);
    config.group_seq_size_per_block = {100, 4};
    auto allocator                  = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    EXPECT_EQ(allocator->maxAvailableTokensNum(), 7u * 4u);
    EXPECT_EQ(allocator->availableTokensNum(), 7u * 4u);

    allocator->setCPSlotMapper(std::make_shared<CPSlotMapper>(/*cp_rank=*/0, /*cp_size=*/2, /*block_size=*/4));

    EXPECT_EQ(allocator->maxAvailableTokensNum(), 7u * 8u);
    EXPECT_EQ(allocator->availableTokensNum(), 7u * 8u);
}

TEST_F(HybridPoolKVCacheAllocatorTest, TokenAggregatorsFallBackToGlobalSeqSize) {
    auto config = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/6, /*full_block_num=*/6);
    config.group_seq_size_per_block.clear();  // fall back to config.seq_size_per_block
    config.seq_size_per_block = 4;
    auto allocator            = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    EXPECT_EQ(allocator->maxAvailableTokensNum(), 5u * 4u);
    EXPECT_EQ(allocator->availableTokensNum(), 5u * 4u);
}

TEST_F(HybridPoolKVCacheAllocatorTest, ActiveTreeCachedBlocksTrackMultipleReferencesAcrossGroups) {
    auto config    = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/6, /*full_block_num=*/8);
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());
    ASSERT_TRUE(injectBlockTreeCache(allocator, config));

    auto pool0 = allocator->groupBlockPools()[0];
    auto pool1 = allocator->groupBlockPools()[1];

    const size_t free_total_before = allocator->freeBlocksNum();
    auto         g0_blocks         = pool0->malloc(2).value();
    auto         g1_blocks         = pool1->malloc(3).value();
    ASSERT_EQ(g0_blocks.size(), 2u);
    ASSERT_EQ(g1_blocks.size(), 3u);
    // Single-count malloc reserves capacity only (refCount 0); take one holder per block to
    // occupy them (replicates the legacy auto request ref).
    pool0->incRef(g0_blocks);
    pool1->incRef(g1_blocks);

    EXPECT_EQ(allocator->freeBlocksNum(), free_total_before - 5u);
    EXPECT_EQ(pool0->activeTreeCachedBlocksNum(), 0u);
    EXPECT_EQ(pool1->activeTreeCachedBlocksNum(), 0u);
    for (auto b : g0_blocks) {
        EXPECT_EQ(pool0->refCount(b), 1u);
    }
    for (auto b : g1_blocks) {
        EXPECT_EQ(pool1->refCount(b), 1u);
    }

    // Take a SECOND holder on one block per group (what a connector transfer would do in the
    // single-count model: another incRef on the same block).
    pool0->incRef(g0_blocks[0]);
    pool1->incRef(g1_blocks[0]);
    EXPECT_EQ(pool0->refCount(g0_blocks[0]), 2u);
    EXPECT_EQ(pool1->refCount(g1_blocks[0]), 2u);
    EXPECT_EQ(pool0->activeTreeCachedBlocksNum(), 1u);
    EXPECT_EQ(pool1->activeTreeCachedBlocksNum(), 1u);

    // Release the first (request) holder on every block.
    pool0->decRef(g0_blocks);
    pool1->decRef(g1_blocks);

    // The two doubly-held blocks stay allocated (refCount drops to 1); the rest are freed.
    EXPECT_TRUE(pool0->isAllocated(g0_blocks[0]));
    EXPECT_TRUE(pool1->isAllocated(g1_blocks[0]));
    EXPECT_EQ(pool0->refCount(g0_blocks[0]), 1u);
    EXPECT_EQ(pool1->refCount(g1_blocks[0]), 1u);
    EXPECT_EQ(pool0->activeTreeCachedBlocksNum(), 0u);
    EXPECT_EQ(pool1->activeTreeCachedBlocksNum(), 0u);
    EXPECT_EQ(allocator->freeBlocksNum(), free_total_before - 2u);

    // Release the second holder → blocks are fully freed and availability is restored.
    pool0->decRef(g0_blocks[0]);
    pool1->decRef(g1_blocks[0]);
    EXPECT_FALSE(pool0->isAllocated(g0_blocks[0]));
    EXPECT_FALSE(pool1->isAllocated(g1_blocks[0]));
    EXPECT_EQ(allocator->freeBlocksNum(), free_total_before);
}

TEST_F(HybridPoolKVCacheAllocatorTest, RealCacheOnlyRefsAndAvailabilityAggregateAcrossIndependentPools) {
    auto config    = makeTinyReclaimCascadeConfig();
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());
    ASSERT_TRUE(injectBlockTreeCache(allocator, config));

    const auto   counters_before = snapshotPoolCounters(allocator);
    const size_t free_before     = allocator->freeBlocksNum();
    const auto   seed             = seedRealCachePath(allocator, config, CacheKeysType{100}, /*seq_length=*/4);
    ASSERT_EQ(seed.group_blocks.size(), 2u);
    ASSERT_EQ(seed.group_blocks[0].size(), 1u);
    ASSERT_EQ(seed.group_blocks[1].size(), 1u);

    const auto pools = allocator->groupBlockPools();
    ASSERT_EQ(pools.size(), 2u);
    auto cache = block_tree_caches_.back();
    ASSERT_NE(cache, nullptr);
    const auto full_block   = seed.group_blocks[0][0];
    const auto linear_block = seed.group_blocks[1][0];
    ASSERT_EQ(pools[0]->refCount(full_block), 1u);
    ASSERT_EQ(pools[1]->refCount(linear_block), 1u);
    EXPECT_EQ(cache->getStats().device_heap_total_size, 2u);
    EXPECT_EQ(allocator->activeTreeCachedBlocksNum(), 0u);
    EXPECT_EQ(allocator->freeBlocksNum(), free_before - 2u);
    EXPECT_EQ(freePlusDeviceReclaimCandidates(allocator, cache), free_before);

    EXPECT_EQ(cache->reclaimBlocks(/*num_blocks=*/1, Tier::DEVICE), 1);
    EXPECT_FALSE(pools[0]->isAllocated(full_block));
    EXPECT_FALSE(pools[1]->isAllocated(linear_block));
    EXPECT_EQ(cache->getStats().device_heap_total_size, 0u);
    EXPECT_EQ(allocator->activeTreeCachedBlocksNum(), 0u);
    expectPoolCountersEq(allocator, counters_before);
    EXPECT_EQ(allocator->freeBlocksNum(), free_before);
}

// ---------------------------------------------------------------------------
// Address / buffer lookups
// ---------------------------------------------------------------------------

TEST_F(HybridPoolKVCacheAllocatorTest, ConvertIndexToAddrAndBufferDefault) {
    auto config    = makeTinyMultiPoolHybridConfig();
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());
    ASSERT_TRUE(injectBlockTreeCache(allocator, config));

    // Layer in linear group.
    {
        auto addr = allocator->convertIndexToAddr(/*layer_id=*/0, /*block_id=*/1);
        EXPECT_NE(addr.kv_addr, nullptr);
        auto bufs = allocator->convertIndexToBuffer(/*layer_id=*/0, /*block_id=*/1);
        ASSERT_FALSE(bufs.empty());
        EXPECT_NE(bufs[0].addr, nullptr);
    }
    // Layer in full group.
    {
        auto addr = allocator->convertIndexToAddr(/*layer_id=*/3, /*block_id=*/1);
        EXPECT_NE(addr.kv_addr, nullptr);
        auto bufs = allocator->convertIndexToBuffer(/*layer_id=*/3, /*block_id=*/1);
        ASSERT_FALSE(bufs.empty());
        EXPECT_NE(bufs[0].addr, nullptr);
    }
}

TEST_F(HybridPoolKVCacheAllocatorTest, ConvertIndexToBufferPartitionDefault) {
    auto config    = makeTinyMultiPoolHybridConfig();
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());
    ASSERT_TRUE(injectBlockTreeCache(allocator, config));

    auto bufs = allocator->convertIndexToBuffer(
        /*layer_id=*/3, /*block_id=*/1, /*partition_count=*/1, /*partition_id=*/0);
    ASSERT_FALSE(bufs.empty());
    EXPECT_NE(bufs[0].addr, nullptr);
}

TEST_F(HybridPoolKVCacheAllocatorTest, ConvertIndexToAddrAndBufferByGroup) {
    auto config    = makeTinyMultiPoolHybridConfig();
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());
    ASSERT_TRUE(injectBlockTreeCache(allocator, config));

    auto addr_default   = allocator->convertIndexToAddr(/*layer_id=*/0, /*group_id=*/0, /*block_id=*/1);
    auto addr_via_layer = allocator->convertIndexToAddr(/*layer_id=*/0, /*block_id=*/1);
    EXPECT_EQ(addr_default.kv_addr, addr_via_layer.kv_addr);

    auto bufs_default = allocator->convertIndexToBuffer(/*layer_id=*/0, /*group_id=*/0, /*block_id=*/1);
    ASSERT_FALSE(bufs_default.empty());
    EXPECT_NE(bufs_default[0].addr, nullptr);

    auto bufs_partitioned = allocator->convertIndexToBuffer(
        /*layer_id=*/0, /*group_id=*/0, /*block_id=*/1, /*partition_count=*/1, /*partition_id=*/0);
    ASSERT_FALSE(bufs_partitioned.empty());
    EXPECT_NE(bufs_partitioned[0].addr, nullptr);
}

TEST_F(HybridPoolKVCacheAllocatorTest, AllLayerCacheBaseExposesPerLayerAndPerGroupTensors) {
    auto config    = makeTinyMultiPoolHybridConfig();
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());
    ASSERT_TRUE(injectBlockTreeCache(allocator, config));

    auto layout = allocator->allLayerCacheBase();
    ASSERT_EQ(layout.layers_to_kv_buffer_ptrs.size(), static_cast<size_t>(config.layer_all_num));
    for (size_t i = 0; i < layout.layers_to_kv_buffer_ptrs.size(); ++i) {
        EXPECT_TRUE(layout.layers_to_kv_buffer_ptrs[i].defined()) << "layer " << i << " missing kv buffer";
    }
    EXPECT_EQ(layout.layer_to_group_ids, config.layerGroupIdsSnapshot());
    EXPECT_EQ(layout.group_types, config.groupTypesSnapshot());

    ASSERT_EQ(layout.layers_to_kv_buffer_ptrs_by_group.size(), static_cast<size_t>(config.layer_all_num));
    for (size_t i = 0; i < layout.layers_to_kv_buffer_ptrs_by_group.size(); ++i) {
        EXPECT_EQ(layout.layers_to_kv_buffer_ptrs_by_group[i].size(), static_cast<size_t>(config.groupNums()));
    }

    for (size_t i = 0; i < static_cast<size_t>(config.layer_all_num); ++i) {
        ASSERT_FALSE(layout.layer_to_group_ids[i].empty());
        const auto  gid        = static_cast<size_t>(layout.layer_to_group_ids[i].front());
        const auto& by_default = layout.layers_to_kv_buffer_ptrs_by_group[i][gid];
        EXPECT_TRUE(by_default.defined()) << "layer " << i << " primary group tensor undefined";
        EXPECT_EQ(by_default.data_ptr(), layout.layers_to_kv_buffer_ptrs[i].data_ptr());
    }
}

// ---------------------------------------------------------------------------
// regUserMr / getMrCostTimeMs
// ---------------------------------------------------------------------------

TEST_F(HybridPoolKVCacheAllocatorTest, RegUserMrWithoutCacheStoreIsNoOpAndZeroCost) {
    auto config    = makeTinyMultiPoolHybridConfig();
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());
    ASSERT_TRUE(injectBlockTreeCache(allocator, config));

    // No CacheStore is plumbed in: regUserMr should be a benign no-op for every
    // group pool, and the aggregated MR cost remains zero.
    EXPECT_NO_THROW(allocator->regUserMr(/*model_id=*/0, /*cache_store=*/nullptr));
    EXPECT_EQ(allocator->getMrCostTimeMs(), 0);
}

// ---------------------------------------------------------------------------
// Canonical count-returning in-place reclaim
// ---------------------------------------------------------------------------

TEST_F(HybridPoolKVCacheAllocatorTest, CanonicalReclaimZeroRequestPreservesCachedTopology) {
    auto config    = makeTinyReclaimCascadeConfig();
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());
    ASSERT_TRUE(injectBlockTreeCache(allocator, config));

    const auto pools = allocator->groupBlockPools();
    ASSERT_EQ(pools.size(), 2u);
    auto cache = block_tree_caches_.back();
    ASSERT_NE(cache, nullptr);

    auto resource = makeBatchResource(/*batch_size=*/1, config);
    resource->setBatchCacheKeys(/*batch_id=*/0, CacheKeysType{100});
    auto       tokens = makeCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/4, /*seq_size_per_block=*/4);
    MallocInfo malloc_info{resource, tokens};
    malloc_info.reuse_cache         = false;
    malloc_info.enable_device_cache = false;
    ASSERT_TRUE(allocator->malloc(malloc_info).success);
    const auto full_block   = resource->blocks(/*batch_id=*/0, /*gid=*/0)[0];
    const auto linear_block = resource->blocks(/*batch_id=*/0, /*gid=*/1)[0];
    allocator->insertIntoCache(InsertInfo{resource, tokens, /*is_resident=*/false});
    allocator->free(FreeInfo{resource, tokens});

    const auto counters_before        = snapshotPoolCounters(allocator);
    const auto free_before            = allocator->freeBlocksNum();
    const auto free_plus_candidates_before = freePlusDeviceReclaimCandidates(allocator, cache);
    ASSERT_EQ(pools[0]->refCount(full_block), 1u);
    ASSERT_EQ(pools[1]->refCount(linear_block), 1u);
    ASSERT_EQ(cache->getStats().device_heap_total_size, 2u);
    ASSERT_EQ(allocator->activeTreeCachedBlocksNum(), 0u);

    EXPECT_EQ(cache->reclaimBlocks(/*num_blocks=*/0, Tier::DEVICE), 0);
    expectPoolCountersEq(allocator, counters_before);
    EXPECT_EQ(allocator->freeBlocksNum(), free_before);
    EXPECT_EQ(freePlusDeviceReclaimCandidates(allocator, cache), free_plus_candidates_before);
    EXPECT_EQ(pools[0]->refCount(full_block), 1u);
    EXPECT_EQ(pools[1]->refCount(linear_block), 1u);
    EXPECT_EQ(cache->getStats().device_heap_total_size, 2u);
    EXPECT_EQ(allocator->activeTreeCachedBlocksNum(), 0u);

    EXPECT_EQ(cache->reclaimBlocks(/*num_blocks=*/1, Tier::DEVICE), 1);
}

TEST_F(HybridPoolKVCacheAllocatorTest, CanonicalReclaimWithoutCandidatesPreservesEmptyPools) {
    auto config    = makeTinyMultiPoolHybridConfig();
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());
    ASSERT_TRUE(injectBlockTreeCache(allocator, config));

    const auto counters_before        = snapshotPoolCounters(allocator);
    const auto free_before            = allocator->freeBlocksNum();
    auto       cache            = block_tree_caches_.back();
    ASSERT_NE(cache, nullptr);
    const auto free_plus_candidates_before = freePlusDeviceReclaimCandidates(allocator, cache);

    EXPECT_EQ(cache->reclaimBlocks(/*num_blocks=*/4, Tier::DEVICE), 0);
    expectPoolCountersEq(allocator, counters_before);
    EXPECT_EQ(allocator->freeBlocksNum(), free_before);
    EXPECT_EQ(freePlusDeviceReclaimCandidates(allocator, cache), free_plus_candidates_before);
    EXPECT_EQ(cache->getStats().device_heap_total_size, 0u);
    EXPECT_EQ(allocator->activeTreeCachedBlocksNum(), 0u);
}

TEST_F(HybridPoolKVCacheAllocatorTest, CanonicalReclaimOverRequestSeparatesPlanCascadeAndPoolDeltas) {
    auto config    = makeTinyReclaimCascadeConfig();
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());
    ASSERT_TRUE(injectBlockTreeCache(allocator, config));

    const auto pools = allocator->groupBlockPools();
    ASSERT_EQ(pools.size(), 2u);
    auto cache = block_tree_caches_.back();
    ASSERT_NE(cache, nullptr);

    auto resource = makeBatchResource(/*batch_size=*/1, config);
    resource->setBatchCacheKeys(/*batch_id=*/0, CacheKeysType{100});
    auto       tokens = makeCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/4, /*seq_size_per_block=*/4);
    MallocInfo malloc_info{resource, tokens};
    malloc_info.reuse_cache         = false;
    malloc_info.enable_device_cache = false;
    ASSERT_TRUE(allocator->malloc(malloc_info).success);
    const auto full_block   = resource->blocks(/*batch_id=*/0, /*gid=*/0)[0];
    const auto linear_block = resource->blocks(/*batch_id=*/0, /*gid=*/1)[0];
    allocator->insertIntoCache(InsertInfo{resource, tokens, /*is_resident=*/false});
    allocator->free(FreeInfo{resource, tokens});

    const auto pool_counters_before        = snapshotPoolCounters(allocator);
    const auto aggregate_free_before       = allocator->freeBlocksNum();
    const auto free_plus_candidates_before = freePlusDeviceReclaimCandidates(allocator, cache);
    ASSERT_EQ(cache->getStats().device_heap_total_size, 2u);
    ASSERT_EQ(allocator->activeTreeCachedBlocksNum(), 0u);

    // P=1 FULL primary plan, C=1 LINEAR cascade slot, D={1, 1} physical pool blocks.
    EXPECT_EQ(cache->reclaimBlocks(/*num_blocks=*/5, Tier::DEVICE), 1);
    EXPECT_FALSE(pools[0]->isAllocated(full_block));
    EXPECT_FALSE(pools[1]->isAllocated(linear_block));
    EXPECT_EQ(cache->getStats().device_heap_total_size, 0u);
    EXPECT_EQ(allocator->activeTreeCachedBlocksNum(), 0u);
    EXPECT_EQ(pools[0]->freeBlocksNum(), pool_counters_before[0].free_blocks + 1);
    EXPECT_EQ(pools[1]->freeBlocksNum(), pool_counters_before[1].free_blocks + 1);
    EXPECT_EQ(allocator->freeBlocksNum(), aggregate_free_before + 2);
    EXPECT_EQ(freePlusDeviceReclaimCandidates(allocator, cache), free_plus_candidates_before);
}

TEST_F(HybridPoolKVCacheAllocatorTest, CanonicalReclaimBoundedPhasesKeepPlanAndPhysicalCountsIndependent) {
    auto config    = makeTinyReclaimCascadeConfig();
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());
    ASSERT_TRUE(injectBlockTreeCache(allocator, config));

    const auto pools = allocator->groupBlockPools();
    ASSERT_EQ(pools.size(), 2u);
    auto cache = block_tree_caches_.back();
    ASSERT_NE(cache, nullptr);

    auto resource = makeBatchResource(/*batch_size=*/1, config);
    resource->setBatchCacheKeys(/*batch_id=*/0, CacheKeysType{100, 101});
    auto       tokens = makeCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/8, /*seq_size_per_block=*/4);
    MallocInfo malloc_info{resource, tokens};
    malloc_info.reuse_cache         = false;
    malloc_info.enable_device_cache = false;
    ASSERT_TRUE(allocator->malloc(malloc_info).success);
    ASSERT_EQ(validBlockCount(resource->blocks(/*batch_id=*/0, /*gid=*/0)), 2u);
    ASSERT_EQ(validBlockCount(resource->blocks(/*batch_id=*/0, /*gid=*/1)), 2u);
    allocator->insertIntoCache(InsertInfo{resource, tokens, /*is_resident=*/false});
    allocator->free(FreeInfo{resource, tokens});

    const auto counters_before              = snapshotPoolCounters(allocator);
    const auto free_before                  = allocator->freeBlocksNum();
    const auto free_plus_candidates_before = freePlusDeviceReclaimCandidates(allocator, cache);
    ASSERT_EQ(cache->getStats().device_heap_total_size, 3u);
    ASSERT_EQ(allocator->activeTreeCachedBlocksNum(), 0u);

    // First phase: K=1 primary plan, C_first=1 cascade slot, D_first={1, 1}.
    EXPECT_EQ(cache->reclaimBlocks(/*num_blocks=*/1, Tier::DEVICE), 1);
    EXPECT_EQ(pools[0]->freeBlocksNum(), counters_before[0].free_blocks + 1);
    EXPECT_EQ(pools[1]->freeBlocksNum(), counters_before[1].free_blocks + 1);
    EXPECT_EQ(allocator->freeBlocksNum(), free_before + 2);
    EXPECT_EQ(cache->getStats().device_heap_total_size, 2u);
    EXPECT_EQ(allocator->activeTreeCachedBlocksNum(), 0u);
    // Removing the first FULL leaf promotes its parent to a candidate, so both post-phase
    // aggregate availabilities are one above the pre-reclaim snapshot.
    EXPECT_EQ(freePlusDeviceReclaimCandidates(allocator, cache), free_plus_candidates_before + 1);

    // Remaining phase: one more primary plan and one cascade slot release D_rest={1, 1}.
    EXPECT_EQ(cache->reclaimBlocks(/*num_blocks=*/8, Tier::DEVICE), 1);
    EXPECT_EQ(pools[0]->freeBlocksNum(), counters_before[0].free_blocks + 2);
    EXPECT_EQ(pools[1]->freeBlocksNum(), counters_before[1].free_blocks + 2);
    EXPECT_EQ(allocator->freeBlocksNum(), free_before + 4);
    EXPECT_EQ(cache->getStats().device_heap_total_size, 0u);
    EXPECT_EQ(allocator->activeTreeCachedBlocksNum(), 0u);
    EXPECT_EQ(freePlusDeviceReclaimCandidates(allocator, cache), free_plus_candidates_before + 1);
}

// ---------------------------------------------------------------------------
// hasAvailableBlocksForReserve via reserve_block_num
// ---------------------------------------------------------------------------

TEST_F(HybridPoolKVCacheAllocatorTest, ReserveBlocksAreDistributedAcrossGroupsForInitMalloc) {
    // Group 0 (linear) gets 6 blocks (5 free), group 1 (full) gets 4 blocks (3 free).
    // total_available = 8. Set reserve = 4.
    // Expected per-group reserve: floor(4 * 5/8) = 2 for gid=0, floor(4 * 3/8) = 1 for gid=1.
    auto config    = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/6, /*full_block_num=*/4);
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());
    ASSERT_TRUE(injectBlockTreeCache(allocator, config));

    allocator->setReserveBlockNum(4);

    // seq_len=4 -> 1 block per group.
    auto batch_res = makeBatchResource(/*batch_size=*/1, config);
    batch_res->setBatchCacheKeys(0, CacheKeysType{100});
    auto       token_ids = makeCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/4, /*seq_size_per_block=*/4);
    MallocInfo malloc_info{batch_res, token_ids};
    malloc_info.enable_device_cache = false;
    malloc_info.reuse_cache         = false;
    auto result                     = allocator->malloc(malloc_info);
    EXPECT_TRUE(result.success);
}

TEST_F(HybridPoolKVCacheAllocatorTest, ReserveBlocksRejectsWhenGroupCannotMeetItsShare) {
    auto config    = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/6, /*full_block_num=*/4);
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());
    ASSERT_TRUE(injectBlockTreeCache(allocator, config));

    // A reserve large enough to hide most blocks should reject init malloc.
    allocator->setReserveBlockNum(allocator->freeBlocksNum());

    auto batch_res = makeBatchResource(/*batch_size=*/1, config);
    batch_res->setBatchCacheKeys(0, CacheKeysType{100});
    auto       token_ids = makeCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/4, /*seq_size_per_block=*/4);
    MallocInfo malloc_info{batch_res, token_ids};
    malloc_info.enable_device_cache = false;
    malloc_info.reuse_cache         = false;
    malloc_info.verbose             = false;
    auto result                     = allocator->malloc(malloc_info);
    EXPECT_FALSE(result.success);
}

TEST_F(HybridPoolKVCacheAllocatorTest, PoolMetricsSnapshotsReportReserveBlocks) {
    auto config    = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/6, /*full_block_num=*/8);
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());
    ASSERT_TRUE(injectBlockTreeCache(allocator, config));

    constexpr size_t reserve_blocks = 6;
    allocator->setReserveBlockNum(reserve_blocks);

    const auto snapshots = allocator->poolMetricsSnapshots();
    ASSERT_EQ(snapshots.size(), 2u);
    EXPECT_EQ("linear", snapshots[0].pool_name);
    EXPECT_EQ("full", snapshots[1].pool_name);

    const size_t total_reservable_free_blocks = snapshots[0].free_blocks + snapshots[1].free_blocks;
    ASSERT_GT(total_reservable_free_blocks, 0u);
    EXPECT_EQ(reserve_blocks * snapshots[0].free_blocks / total_reservable_free_blocks, snapshots[0].reserve_blocks);
    EXPECT_EQ(reserve_blocks * snapshots[1].free_blocks / total_reservable_free_blocks, snapshots[1].reserve_blocks);
}

TEST_F(HybridPoolKVCacheAllocatorTest, PoolMetricsSnapshotsReportActiveTreeCachedBlocks) {
    auto config    = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/6, /*full_block_num=*/8);
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());
    ASSERT_TRUE(injectBlockTreeCache(allocator, config));

    auto pool0  = allocator->groupBlockPools()[0];
    auto pool1  = allocator->groupBlockPools()[1];
    auto block0 = pool0->malloc();
    auto block1 = pool1->malloc();
    ASSERT_TRUE(block0.has_value());
    ASSERT_TRUE(block1.has_value());
    pool0->incRef(*block0);
    pool0->incRef(*block0);
    pool1->incRef(*block1);
    pool1->incRef(*block1);

    const auto snapshots = allocator->poolMetricsSnapshots();
    ASSERT_EQ(snapshots.size(), 2u);
    EXPECT_EQ(snapshots[0].active_tree_cached_blocks, 1u);
    EXPECT_EQ(snapshots[1].active_tree_cached_blocks, 1u);
    EXPECT_EQ(allocator->activeTreeCachedBlocksNum(), 2u);

    pool0->decRef(*block0);
    pool0->decRef(*block0);
    pool1->decRef(*block1);
    pool1->decRef(*block1);
    EXPECT_EQ(allocator->activeTreeCachedBlocksNum(), 0u);
}

TEST_F(HybridPoolKVCacheAllocatorTest, ReserveBlocksUseCPShardedFullGroupNeed) {
    auto config    = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/20, /*full_block_num=*/6);
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());
    ASSERT_TRUE(injectBlockTreeCache(allocator, config));

    allocator->setReserveBlockNum(1);

    auto batch_res = makeBatchResource(/*batch_size=*/1, config);
    batch_res->setBatchCacheKeys(0, CacheKeysType{100, 101, 102, 103, 104, 105, 106, 107});
    auto token_ids = makeCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/32, /*seq_size_per_block=*/4);
    allocator->setCPSlotMapper(std::make_shared<CPSlotMapper>(/*cp_rank=*/0, /*cp_size=*/2, /*block_size=*/4));

    MallocInfo malloc_info{batch_res, token_ids};
    malloc_info.enable_device_cache = false;
    malloc_info.reuse_cache         = false;

    auto result = allocator->malloc(malloc_info);
    ASSERT_TRUE(result.success);
    EXPECT_EQ(validBlockCount(batch_res->blocks(0, /*gid=*/1)), 4u);

    FreeInfo free_info{batch_res, token_ids};
    allocator->free(free_info);
}

TEST_F(HybridPoolKVCacheAllocatorTest, ReserveCheckIsBypassedWhenMallocInfoLacksContext) {
    // hasAvailableBlocksForReserve returns true when info has no resource/tokens.
    auto config    = makeTinyMultiPoolHybridConfig();
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());
    ASSERT_TRUE(injectBlockTreeCache(allocator, config));

    MallocInfo info{};
    EXPECT_TRUE(allocator->hasAvailableBlocksForReserve(info, /*reserve_blocks=*/9999));
}

TEST_F(HybridPoolKVCacheAllocatorTest, InitMallocRollbackFreesPartiallyAllocatedGroupBlocks) {
    // gid=0 has enough room for the LINEAR tail block; gid=1 cannot satisfy
    // the 3 FULL blocks needed for seq_len=9. initMallocForCommonLen should
    // roll gid=0 back after gid=1 fails.
    auto config    = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/3, /*full_block_num=*/3);
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());
    ASSERT_TRUE(injectBlockTreeCache(allocator, config));

    const auto counters_before = snapshotPoolCounters(allocator);

    auto batch_res = makeBatchResource(/*batch_size=*/1, config);
    batch_res->setBatchCacheKeys(0, CacheKeysType{100, 101, 102});
    auto       token_ids = makeCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/9, /*seq_size_per_block=*/4);
    MallocInfo malloc_info{batch_res, token_ids};
    malloc_info.enable_device_cache = false;
    malloc_info.reuse_cache         = false;
    malloc_info.verbose             = false;

    auto result = allocator->malloc(malloc_info);
    EXPECT_FALSE(result.success);

    EXPECT_EQ(batch_res->curBlocksNum(), 0u);
    EXPECT_EQ(batch_res->blocksNum(0, /*gid=*/0), 0u);
    EXPECT_EQ(batch_res->blocksNum(0, /*gid=*/1), 0u);
    expectPoolCountersEq(allocator, counters_before);
}

TEST_F(HybridPoolKVCacheAllocatorTest, InitMallocRollbackReleasesRealMatchedRefsOnReserveReject) {
    auto config    = makeTinyReclaimCascadeConfig(/*full_block_num=*/4, /*linear_block_num=*/4);
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());
    ASSERT_TRUE(injectBlockTreeCache(allocator, config));

    const auto counters_before_seed = snapshotPoolCounters(allocator);
    const auto seed                 = seedRealCachePath(allocator, config, CacheKeysType{100}, /*seq_length=*/4);
    ASSERT_EQ(seed.group_blocks.size(), 2u);
    ASSERT_EQ(seed.group_blocks[0].size(), 1u);
    ASSERT_EQ(seed.group_blocks[1].size(), 1u);
    const auto full_cached   = seed.group_blocks[0][0];
    const auto linear_cached = seed.group_blocks[1][0];
    ASSERT_FALSE(isNullBlockIdx(linear_cached));
    ASSERT_FALSE(isNullBlockIdx(full_cached));
    const auto pools = allocator->groupBlockPools();
    ASSERT_EQ(pools.size(), 2u);
    auto cache = block_tree_caches_.back();
    ASSERT_NE(cache, nullptr);
    ASSERT_EQ(pools[0]->refCount(full_cached), 1u);
    ASSERT_EQ(pools[1]->refCount(linear_cached), 1u);
    ASSERT_EQ(cache->getStats().device_heap_total_size, 2u);
    ASSERT_EQ(allocator->activeTreeCachedBlocksNum(), 0u);

    const size_t free_before     = allocator->freeBlocksNum();
    const auto   counters_before = snapshotPoolCounters(allocator);
    allocator->setReserveBlockNum(std::max<size_t>(1, free_before * 8));

    auto batch_res = makeBatchResource(/*batch_size=*/1, config);
    batch_res->setBatchCacheKeys(0, CacheKeysType{100, 101, 102});
    auto       token_ids = makeCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/8, /*seq_size_per_block=*/4);
    MallocInfo malloc_info{batch_res, token_ids};
    malloc_info.enable_device_cache = true;
    malloc_info.reuse_cache         = true;
    malloc_info.verbose             = false;

    auto result = allocator->malloc(malloc_info);
    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.async_context, nullptr);

    EXPECT_EQ(batch_res->curBlocksNum(), 0u);
    EXPECT_EQ(batch_res->blocksNum(0, /*gid=*/0), 0u);
    EXPECT_EQ(batch_res->blocksNum(0, /*gid=*/1), 0u);
    EXPECT_EQ(pools[0]->refCount(full_cached), 1u);
    EXPECT_EQ(pools[1]->refCount(linear_cached), 1u);
    EXPECT_EQ(allocator->freeBlocksNum(), free_before);
    expectPoolCountersEq(allocator, counters_before);
    EXPECT_EQ(cache->getStats().device_heap_total_size, 2u);
    EXPECT_EQ(allocator->activeTreeCachedBlocksNum(), 0u);

    allocator->setReserveBlockNum(0);
    auto hit_res = makeBatchResource(/*batch_size=*/1, config);
    hit_res->setBatchCacheKeys(/*batch_id=*/0, CacheKeysType{100, 101});
    auto       hit_tokens = makeCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/8, /*seq_size_per_block=*/4);
    MallocInfo hit_info{hit_res, hit_tokens};
    hit_info.enable_device_cache = true;
    hit_info.reuse_cache         = true;
    const auto hit_result        = allocator->malloc(hit_info);
    ASSERT_TRUE(hit_result.success);
    EXPECT_EQ(hit_result.async_context, nullptr);
    EXPECT_EQ(hit_result.reuse_len, 4);
    ASSERT_EQ(hit_res->blocks(0, /*gid=*/0)[0], full_cached);
    ASSERT_EQ(hit_res->blocks(0, /*gid=*/1)[0], linear_cached);
    EXPECT_EQ(pools[0]->refCount(full_cached), 2u);
    EXPECT_EQ(pools[1]->refCount(linear_cached), 2u);

    allocator->free(FreeInfo{hit_res, hit_tokens});
    EXPECT_EQ(hit_res->curBlocksNum(), 0u);
    EXPECT_EQ(pools[0]->refCount(full_cached), 1u);
    EXPECT_EQ(pools[1]->refCount(linear_cached), 1u);
    expectPoolCountersEq(allocator, counters_before);

    EXPECT_EQ(cache->reclaimBlocks(/*num_blocks=*/1, Tier::DEVICE), 1);
    EXPECT_FALSE(pools[0]->isAllocated(full_cached));
    EXPECT_FALSE(pools[1]->isAllocated(linear_cached));
    expectPoolCountersEq(allocator, counters_before_seed);
}

TEST_F(HybridPoolKVCacheAllocatorTest, IncrMallocRollbackFreesPartiallyAllocatedGroupBlocks) {
    auto config    = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/4, /*full_block_num=*/2);
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());
    ASSERT_TRUE(injectBlockTreeCache(allocator, config));

    auto batch_res = makeBatchResource(/*batch_size=*/1, config);
    batch_res->setBatchCacheKeys(0, CacheKeysType{100, 101, 102});

    auto       token_ids = makeCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/4, /*seq_size_per_block=*/4);
    MallocInfo init_info{batch_res, token_ids};
    init_info.enable_device_cache = false;
    init_info.reuse_cache         = false;
    ASSERT_TRUE(allocator->malloc(init_info).success);

    ASSERT_EQ(batch_res->blocksNum(0, /*gid=*/0), 1u);
    ASSERT_EQ(batch_res->blocksNum(0, /*gid=*/1), 1u);
    const auto linear_block_before = batch_res->blocks(0, /*gid=*/0)[0];
    const auto full_block_before   = batch_res->blocks(0, /*gid=*/1)[0];
    const auto counters_before     = snapshotPoolCounters(allocator);

    // gid=0 can append one real LINEAR tail block. gid=1 has no remaining
    // free blocks and no cache to evict, so FULL allocation fails.
    token_ids->setSeqLength(9);
    MallocInfo incr_info{batch_res, token_ids};
    incr_info.enable_device_cache = false;
    incr_info.reuse_cache         = false;
    auto incr_result              = allocator->malloc(incr_info);
    EXPECT_FALSE(incr_result.success);

    ASSERT_EQ(batch_res->blocksNum(0, /*gid=*/0), 1u);
    ASSERT_EQ(batch_res->blocksNum(0, /*gid=*/1), 1u);
    EXPECT_EQ(batch_res->blocks(0, /*gid=*/0)[0], linear_block_before);
    EXPECT_EQ(batch_res->blocks(0, /*gid=*/1)[0], full_block_before);
    expectPoolCountersEq(allocator, counters_before);
}

// ---------------------------------------------------------------------------
// Full malloc / free cycle
// ---------------------------------------------------------------------------

TEST_F(HybridPoolKVCacheAllocatorTest, MallocAndFreeCycleAcrossPerGroupPools) {
    auto config    = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/8, /*full_block_num=*/8);
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());
    ASSERT_TRUE(injectBlockTreeCache(allocator, config));

    const size_t free_before = allocator->freeBlocksNum();

    auto batch_res = makeBatchResource(/*batch_size=*/1, config);
    batch_res->setBatchCacheKeys(0, CacheKeysType{100, 101, 102});
    auto       token_ids = makeCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/12, /*seq_size_per_block=*/4);
    MallocInfo malloc_info{batch_res, token_ids};
    malloc_info.enable_device_cache = false;
    malloc_info.reuse_cache         = false;
    auto result                     = allocator->malloc(malloc_info);
    ASSERT_TRUE(result.success);
    EXPECT_LT(allocator->freeBlocksNum(), free_before);

    FreeInfo free_info{batch_res, token_ids};
    allocator->free(free_info);
    EXPECT_EQ(allocator->freeBlocksNum(), free_before);
}

// C007-T04: direct HybridPool release refreshes both real component-group
// candidates and permits in-place reclaim with exact per-pool balance.
TEST_F(HybridPoolKVCacheAllocatorTest, DirectFreeMakesEveryPoolCacheOnlyBlockReclaimable) {
    auto config    = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/5, /*full_block_num=*/5);
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());
    ASSERT_TRUE(injectBlockTreeCache(allocator, config));

    const auto pools = allocator->groupBlockPools();
    ASSERT_EQ(pools.size(), 2u);
    auto cache = block_tree_caches_.back();
    ASSERT_NE(cache, nullptr);
    const auto counters_before              = snapshotPoolCounters(allocator);
    const auto free_before                  = allocator->freeBlocksNum();
    const auto free_plus_candidates_before = freePlusDeviceReclaimCandidates(allocator, cache);

    auto resource = makeBatchResource(/*batch_size=*/1, config);
    resource->setBatchCacheKeys(/*batch_id=*/0, CacheKeysType{100});
    auto tokens = makeCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/4, /*seq_size_per_block=*/4);

    MallocInfo malloc_info{resource, tokens};
    malloc_info.reuse_cache         = true;
    malloc_info.enable_device_cache = false;
    ASSERT_TRUE(allocator->malloc(malloc_info).success);
    ASSERT_EQ(resource->blocksNum(/*batch_id=*/0, /*group_id=*/0), 1);
    ASSERT_EQ(resource->blocksNum(/*batch_id=*/0, /*group_id=*/1), 1);
    const auto linear_block = resource->blocks(/*batch_id=*/0, /*group_id=*/0)[0];
    const auto full_block   = resource->blocks(/*batch_id=*/0, /*group_id=*/1)[0];
    EXPECT_EQ(pools[0]->refCount(linear_block), 1u);
    EXPECT_EQ(pools[1]->refCount(full_block), 1u);

    allocator->insertIntoCache(InsertInfo{resource, tokens, /*is_resident=*/false});
    EXPECT_EQ(pools[0]->refCount(linear_block), 2u);
    EXPECT_EQ(pools[1]->refCount(full_block), 2u);
    EXPECT_EQ(cache->getStats().device_heap_total_size, 0u);
    EXPECT_EQ(allocator->activeTreeCachedBlocksNum(), 2u);
    EXPECT_EQ(allocator->freeBlocksNum(), free_before - 2);
    EXPECT_EQ(freePlusDeviceReclaimCandidates(allocator, cache), free_plus_candidates_before - 2);

    allocator->free(FreeInfo{resource, tokens});
    EXPECT_EQ(resource->curBlocksNum(), 0);
    EXPECT_EQ(pools[0]->refCount(linear_block), 1u);
    EXPECT_EQ(pools[1]->refCount(full_block), 1u);
    EXPECT_EQ(cache->getStats().device_heap_total_size, 2u);
    EXPECT_EQ(allocator->activeTreeCachedBlocksNum(), 0u);
    EXPECT_EQ(allocator->freeBlocksNum(), free_before - 2);
    EXPECT_EQ(freePlusDeviceReclaimCandidates(allocator, cache), free_plus_candidates_before);

    EXPECT_EQ(cache->reclaimBlocks(/*num_blocks=*/2, Tier::DEVICE), 2);
    EXPECT_FALSE(pools[0]->isAllocated(linear_block));
    EXPECT_FALSE(pools[1]->isAllocated(full_block));
    EXPECT_EQ(cache->getStats().device_heap_total_size, 0u);
    EXPECT_EQ(allocator->activeTreeCachedBlocksNum(), 0u);
    expectPoolCountersEq(allocator, counters_before);
    EXPECT_EQ(allocator->freeBlocksNum(), free_before);
    EXPECT_EQ(freePlusDeviceReclaimCandidates(allocator, cache), free_plus_candidates_before);
}

TEST_F(HybridPoolKVCacheAllocatorTest, ZeroBlockFreeWithoutBlockTreeCacheIsNoOp) {
    auto config    = makeTinyMultiPoolHybridConfig();
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());
    ASSERT_EQ(allocator->blockTreeCache(), nullptr);

    const auto counters_before = snapshotPoolCounters(allocator);
    const auto free_before     = allocator->freeBlocksNum();
    auto       resource         = makeBatchResource(/*batch_size=*/1, config);
    auto       tokens           = makeCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/0, /*seq_size_per_block=*/4);

    allocator->free(FreeInfo{resource, tokens});
    EXPECT_EQ(resource->curBlocksNum(), 0);
    expectPoolCountersEq(allocator, counters_before);
    EXPECT_EQ(allocator->freeBlocksNum(), free_before);
}

// C005-T01: populated HybridPool growth stays synchronous and charges each
// independent pool at an exact block boundary.
TEST_F(HybridPoolKVCacheAllocatorTest, PopulatedIncrementIsSynchronousAndRestoresPerPoolCapacity) {
    auto config    = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/8, /*full_block_num=*/8);
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());
    ASSERT_TRUE(injectBlockTreeCache(allocator, config));

    const auto pools = allocator->groupBlockPools();
    ASSERT_EQ(pools.size(), 2u);
    const auto counters_before = snapshotPoolCounters(allocator);
    const auto free_before     = allocator->freeBlocksNum();
    ASSERT_EQ(allocator->activeTreeCachedBlocksNum(), 0u);

    auto batch_resource = makeBatchResource(/*batch_size=*/1, config);
    batch_resource->setBatchCacheKeys(/*batch_id=*/0, CacheKeysType{100, 101});
    auto complete_tokens = makeCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/4, /*seq_size_per_block=*/4);

    MallocInfo init_info{batch_resource, complete_tokens};
    init_info.reuse_cache         = true;
    init_info.enable_device_cache = true;
    auto init_result              = allocator->malloc(init_info);
    ASSERT_TRUE(init_result.success);
    EXPECT_EQ(init_result.reuse_len, 0);
    EXPECT_EQ(init_result.async_context, nullptr);
    ASSERT_EQ(batch_resource->blocksNum(/*batch_id=*/0, /*group_id=*/0), 1);
    ASSERT_EQ(batch_resource->blocksNum(/*batch_id=*/0, /*group_id=*/1), 1);
    const auto linear_initial = batch_resource->blocks(/*batch_id=*/0, /*group_id=*/0)[0];
    const auto full_initial   = batch_resource->blocks(/*batch_id=*/0, /*group_id=*/1)[0];
    EXPECT_EQ(pools[0]->refCount(linear_initial), 1u);
    EXPECT_EQ(pools[1]->refCount(full_initial), 1u);
    EXPECT_EQ(pools[0]->freeBlocksNum(), counters_before[0].free_blocks - 1);
    EXPECT_EQ(pools[1]->freeBlocksNum(), counters_before[1].free_blocks - 1);
    EXPECT_EQ(allocator->freeBlocksNum(), free_before - 2);
    EXPECT_EQ(allocator->activeTreeCachedBlocksNum(), 0u);

    complete_tokens->setSeqLength(8);
    MallocInfo incr_info{batch_resource, complete_tokens};
    incr_info.reuse_cache         = true;
    incr_info.enable_device_cache = true;
    auto incr_result              = allocator->malloc(incr_info);
    ASSERT_TRUE(incr_result.success);
    EXPECT_EQ(incr_result.reuse_len, 0);
    EXPECT_EQ(incr_result.async_context, nullptr);
    ASSERT_EQ(batch_resource->blocksNum(/*batch_id=*/0, /*group_id=*/0), 2);
    ASSERT_EQ(batch_resource->blocksNum(/*batch_id=*/0, /*group_id=*/1), 2);
    const auto linear_appended = batch_resource->blocks(/*batch_id=*/0, /*group_id=*/0)[1];
    const auto full_appended   = batch_resource->blocks(/*batch_id=*/0, /*group_id=*/1)[1];
    EXPECT_NE(linear_appended, linear_initial);
    EXPECT_NE(full_appended, full_initial);
    EXPECT_EQ(pools[0]->refCount(linear_initial), 1u);
    EXPECT_EQ(pools[1]->refCount(full_initial), 1u);
    EXPECT_EQ(pools[0]->refCount(linear_appended), 1u);
    EXPECT_EQ(pools[1]->refCount(full_appended), 1u);
    EXPECT_EQ(pools[0]->freeBlocksNum(), counters_before[0].free_blocks - 2);
    EXPECT_EQ(pools[1]->freeBlocksNum(), counters_before[1].free_blocks - 2);
    EXPECT_EQ(allocator->freeBlocksNum(), free_before - 4);
    EXPECT_EQ(allocator->activeTreeCachedBlocksNum(), 0u);

    FreeInfo free_info{batch_resource, complete_tokens};
    allocator->free(free_info);
    EXPECT_EQ(batch_resource->curBlocksNum(), 0);
    EXPECT_FALSE(pools[0]->isAllocated(linear_initial));
    EXPECT_FALSE(pools[1]->isAllocated(full_initial));
    EXPECT_FALSE(pools[0]->isAllocated(linear_appended));
    EXPECT_FALSE(pools[1]->isAllocated(full_appended));
    expectPoolCountersEq(allocator, counters_before);
    EXPECT_EQ(allocator->freeBlocksNum(), free_before);
    EXPECT_EQ(allocator->activeTreeCachedBlocksNum(), 0u);
}

// ---------------------------------------------------------------------------
// DSV4 7-group HybridPool: covers per-tag addressing and SWA tail
// ---------------------------------------------------------------------------

TEST_F(HybridPoolKVCacheAllocatorTest, DSV4InitAndAggregatedCounters) {
    auto config    = makeDSV4HybridPoolConfig(/*block_num=*/200);
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());
    ASSERT_TRUE(injectBlockTreeCache(allocator, config));

    EXPECT_EQ(config.groupNums(), 7);
    ASSERT_EQ(allocator->groupBlockPools().size(), 7u);

    // Sum of per-pool totals must equal aggregated totalBlocksNum.
    size_t expected_total = 0;
    for (const auto& pool : allocator->groupBlockPools()) {
        expected_total += pool->totalBlocksNum();
    }
    EXPECT_EQ(allocator->totalBlocksNum(), expected_total);
    EXPECT_EQ(allocator->freeBlocksNum(), expected_total);
}

TEST_F(HybridPoolKVCacheAllocatorTest, DSV4FixedTagPoolsUseGpuBacking) {
    auto config    = makeDSV4HybridPoolConfig(/*block_num=*/200);
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());
    ASSERT_TRUE(injectBlockTreeCache(allocator, config));

    ASSERT_EQ(allocator->groupBlockPools().size(), 7u);
    for (size_t gid = 0; gid < allocator->groupBlockPools().size(); ++gid) {
        EXPECT_EQ(allocator->groupBlockPools()[gid]->where(), MemoryType::MEMORY_GPU)
            << "gid=" << gid << " tag=" << config.tagForGroup(gid);
    }
}

TEST_F(HybridPoolKVCacheAllocatorTest, DSV4HCAStateReuseEnabledAllocatesTailOnly) {
    auto config        = makeDSV4HybridPoolConfig(/*block_num=*/200);
    config.linear_step = 4;
    auto allocator     = makeAllocator(config);
    ASSERT_TRUE(allocator->init());
    ASSERT_TRUE(injectBlockTreeCache(allocator, config));

    const int hca_state_gid = config.groupIdForTag("hca_state");
    ASSERT_GE(hca_state_gid, 0);
    ASSERT_GT(allocator->groupBlockPools().size(), static_cast<size_t>(hca_state_gid));

    const size_t hca_free_before = allocator->groupBlockPools()[hca_state_gid]->freeBlocksNum();

    auto batch_res = makeBatchResource(/*batch_size=*/1, config);
    batch_res->setBatchCacheKeys(0, CacheKeysType{100, 101, 102, 103, 104, 105, 106, 107, 108, 109});
    auto token_ids = makeCompleteTokenIds(
        /*batch_size=*/1, /*seq_length=*/10 * static_cast<int>(config.seq_size_per_block), config.seq_size_per_block);

    MallocInfo malloc_info{batch_res, token_ids};
    malloc_info.enable_device_cache = false;
    malloc_info.reuse_cache         = true;
    auto result                     = allocator->malloc(malloc_info);
    ASSERT_TRUE(result.success);

    const auto& hca_blocks = batch_res->blocks(0, hca_state_gid);
    ASSERT_EQ(hca_blocks.size(), 10u);
    EXPECT_EQ(validBlockCount(hca_blocks), 1u);
    EXPECT_TRUE(isNullBlockIdx(hca_blocks[8]));
    EXPECT_FALSE(isNullBlockIdx(hca_blocks[9]));
    EXPECT_EQ(hca_free_before - allocator->groupBlockPools()[hca_state_gid]->freeBlocksNum(), 1u);
}

TEST_F(HybridPoolKVCacheAllocatorTest, TokenAggregatorsIgnoreSmallHCAStatePool) {
    auto config = makeDSV4HybridPoolConfig(/*block_num=*/50);

    const int hca_state_gid = config.groupIdForTag("hca_state");
    ASSERT_GE(hca_state_gid, 0);
    auto block_nums           = groupBlockNumsSnapshot(config);
    block_nums[hca_state_gid] = 2;
    setGroupBlockNums(config, block_nums);

    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());
    ASSERT_TRUE(injectBlockTreeCache(allocator, config));
    ASSERT_GT(allocator->groupBlockPools().size(), static_cast<size_t>(hca_state_gid));

    const auto hca_state_tokens =
        allocator->groupBlockPools()[hca_state_gid]->totalBlocksNum() * config.group_seq_size_per_block[hca_state_gid];
    EXPECT_LT(hca_state_tokens, allocator->totalTokensNum());
    EXPECT_EQ(allocator->availableTokensNum(), allocator->maxAvailableTokensNum());
    EXPECT_EQ(allocator->totalTokensNum(), allocator->maxAvailableTokensNum());
}

TEST_F(HybridPoolKVCacheAllocatorTest, DSV4ConfigUsesOnlyPagedGroupsForBlockSize) {
    auto              mc = makeTinyDSV4ModelConfig();
    ParallelismConfig pc;
    KVCacheConfig     kv_cache_config;
    kv_cache_config.seq_size_per_block        = 128;
    kv_cache_config.kernel_seq_size_per_block = 128;
    auto config = CacheConfigCreator::createBasicConfig(mc, pc, kv_cache_config, false, 0);

    ASSERT_EQ(config.groupNums(), 7);
    ASSERT_EQ(config.groupNums(), 7);

    size_t expected_non_full_bytes = 0;
    size_t expected_full_bytes     = 0;
    for (size_t gid = 0; gid < static_cast<size_t>(config.groupNums()); ++gid) {
        const auto type = config.typeForGroup(gid);
        if (type == CacheGroupType::FULL) {
            expected_full_bytes += config.blockSizeBytesForGroup(gid);
        } else {
            expected_non_full_bytes += config.blockSizeBytesForGroup(gid);
        }
    }

    EXPECT_GT(expected_non_full_bytes, 0u);
    EXPECT_GT(expected_full_bytes, 0u);

    EXPECT_EQ(config.block_size_bytes, expected_full_bytes);
}

TEST_F(HybridPoolKVCacheAllocatorTest, DSV4FinalizeBlockNumsUsesHcaStatePoolBlocks) {
    auto         config       = makeDSV4HybridPoolConfig(/*block_num=*/50);
    const size_t explicit_gid = firstExplicitIndependentGroup(config);
    setExplicitBlocksForGroup(config, explicit_gid, 50);

    RuntimeConfig rt;  // unused inside finalizeBlockNums today
    config.finalizeBlockNums(/*global_block_num=*/200, rt);

    for (size_t gid = 0; gid < static_cast<size_t>(config.groupNums()); ++gid) {
        const uint32_t expected = config.policyForGroup(gid).explicit_block_num > 0 ? 50u : 200u;
        EXPECT_EQ(config.blockNumForGroup(gid), expected) << "gid=" << gid;
    }

    const size_t expected_reserve = 50u * config.blockSizeBytesForGroup(explicit_gid);
    EXPECT_EQ(config.explicitly_sized_pool_reserve_bytes, expected_reserve);
}

TEST_F(HybridPoolKVCacheAllocatorTest, DSV4FinalizeBlockNumsUsesGlobalBlocksWhenHcaStateBlocksDisabled) {
    auto config = makeDSV4HybridPoolConfig(/*block_num=*/123);
    setExplicitBlocksForGroup(config, firstExplicitIndependentGroup(config), 0);

    RuntimeConfig rt;
    config.finalizeBlockNums(/*global_block_num=*/123, rt);

    for (size_t gid = 0; gid < static_cast<size_t>(config.groupNums()); ++gid) {
        EXPECT_EQ(config.blockNumForGroup(gid), 123u);
    }
    EXPECT_EQ(config.explicitly_sized_pool_reserve_bytes, 0u);
}

TEST_F(HybridPoolKVCacheAllocatorTest, DSV4GpuHcaStatePoolIncludesFixedReserve) {
    auto         config       = makeDSV4HybridPoolConfig(/*block_num=*/50);
    const size_t explicit_gid = firstExplicitIndependentGroup(config);
    setExplicitBlocksForGroup(config, explicit_gid, 50);

    RuntimeConfig rt;
    config.finalizeBlockNums(/*global_block_num=*/200, rt);

    for (size_t gid = 0; gid < static_cast<size_t>(config.groupNums()); ++gid) {
        const uint32_t expected = config.policyForGroup(gid).explicit_block_num > 0 ? 50u : 200u;
        EXPECT_EQ(config.blockNumForGroup(gid), expected) << "gid=" << gid;
    }
    const size_t expected_reserve = 50u * config.blockSizeBytesForGroup(explicit_gid);
    EXPECT_EQ(config.explicitly_sized_pool_reserve_bytes, expected_reserve);
}

TEST_F(HybridPoolKVCacheAllocatorTest, DSV4StateSwaPoolsWithoutExplicitBlocksUseGlobalBlocks) {
    auto mc                                                      = makeProModelConfig();
    mc.hybrid_attention_config.enable_hybrid_attention           = true;
    mc.hybrid_attention_config.enable_independent_kv_cache_pools = true;
    ParallelismConfig pc;
    KVCacheConfig     kv_cache_config;
    kv_cache_config.seq_size_per_block        = 128;
    kv_cache_config.kernel_seq_size_per_block = 128;
    setDsv4ExplicitPoolBlocks(mc, "hca_state", 0);
    auto config        = CacheConfigCreator::createBasicConfig(mc, pc, kv_cache_config, false, 0);
    config.linear_step = 4;

    RuntimeConfig rt;
    config.finalizeBlockNums(/*global_block_num=*/128, rt);

    for (size_t gid = 0; gid < static_cast<size_t>(config.groupNums()); ++gid) {
        EXPECT_EQ(config.blockNumForGroup(gid), 128u) << "gid=" << gid;
    }
    EXPECT_EQ(config.explicitly_sized_pool_reserve_bytes, 0u);
}

TEST_F(HybridPoolKVCacheAllocatorTest, DSV4ConvertIndexToAddrByTagRoutesToCorrectPool) {
    auto config    = makeDSV4HybridPoolConfig();
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());
    ASSERT_TRUE(injectBlockTreeCache(allocator, config));

    // CSA layer (compress_ratio=4) -- pick the first one.
    int csa_layer = -1;
    for (size_t l = 0; l < config.layer_all_num; ++l) {
        if (config.layerTagToGroupIdSnapshot()[l].count("csa_kv") > 0) {
            csa_layer = static_cast<int>(l);
            break;
        }
    }
    ASSERT_GE(csa_layer, 0);

    // csa_kv tag routes to gid=0; it must produce a non-null kv address that
    // matches the CSA group's pool.
    auto addr_csa = allocator->convertIndexToAddrByTag(csa_layer, "csa_kv", 1);
    EXPECT_NE(addr_csa.kv_addr, nullptr);

    auto addr_swa = allocator->convertIndexToAddrByTag(csa_layer, "swa_kv", 1);
    EXPECT_NE(addr_swa.kv_addr, nullptr);

    // The two tags live in different pools, so their addresses cannot alias.
    EXPECT_NE(addr_csa.kv_addr, addr_swa.kv_addr);

    // Default single-group access is ambiguous for multi-tag layers.
    EXPECT_DEATH((void)allocator->convertIndexToAddr(csa_layer, /*block_id=*/1), "");
}

TEST_F(HybridPoolKVCacheAllocatorTest, DSV4ConvertIndexToBufferByTagAndPartition) {
    auto config    = makeDSV4HybridPoolConfig();
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());
    ASSERT_TRUE(injectBlockTreeCache(allocator, config));

    int csa_layer = -1;
    for (size_t l = 0; l < config.layer_all_num; ++l) {
        if (config.layerTagToGroupIdSnapshot()[l].count("csa_kv") > 0) {
            csa_layer = static_cast<int>(l);
            break;
        }
    }
    ASSERT_GE(csa_layer, 0);

    auto buf = allocator->convertIndexToBufferByTag(csa_layer, "csa_kv", /*block_id=*/1);
    ASSERT_FALSE(buf.empty());
    EXPECT_NE(buf[0].addr, nullptr);

    auto buf_part = allocator->convertIndexToBufferByTag(
        csa_layer, "csa_kv", /*block_id=*/1, /*partition_count=*/1, /*partition_id=*/0);
    ASSERT_FALSE(buf_part.empty());
    EXPECT_NE(buf_part[0].addr, nullptr);
}

TEST_F(HybridPoolKVCacheAllocatorTest, DSV4AllLayerCacheBaseHasPerGroupTensors) {
    auto config    = makeDSV4HybridPoolConfig();
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());
    ASSERT_TRUE(injectBlockTreeCache(allocator, config));

    auto layout = allocator->allLayerCacheBase();
    ASSERT_EQ(layout.layers_to_kv_buffer_ptrs.size(), static_cast<size_t>(config.layer_all_num));
    ASSERT_EQ(layout.layers_to_kv_buffer_ptrs_by_group.size(), static_cast<size_t>(config.layer_all_num));

    for (size_t l = 0; l < static_cast<size_t>(config.layer_all_num); ++l) {
        EXPECT_FALSE(layout.layers_to_kv_buffer_ptrs[l].defined())
            << "multi-tag DSV4 layer should not publish a legacy single-group tensor";
        const auto& swa_t = layout.layers_to_kv_buffer_ptrs_by_group[l][config.groupIdForTag("swa_kv")];
        EXPECT_TRUE(swa_t.defined()) << "layer " << l << " missing SWA_KV tensor";
    }
    EXPECT_EQ(layout.group_tags.size(), 7u);
    EXPECT_EQ(layout.group_types.size(), 7u);
}

// Single-count co-hold invariant: a block co-held by a request holder and a cache holder is
// not freed when the request holder is released; only releasing the last holder frees it.
TEST_F(HybridPoolKVCacheAllocatorTest, RequestReleaseDoesNotFreeCachedBlock) {
    auto config    = makeTinyMultiPoolHybridConfig();
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());
    ASSERT_TRUE(injectBlockTreeCache(allocator, config));
    auto pool  = allocator->groupBlockPools()[0];
    auto block = pool->malloc().value();
    pool->incRef(block);  // cache holder
    pool->incRef(block);  // request holder
    EXPECT_EQ(pool->refCount(block), 2u);
    pool->decRef(block);  // release request holder
    EXPECT_TRUE(pool->isAllocated(block));
    EXPECT_EQ(pool->refCount(block), 1u);
    pool->decRef(block);  // release cache holder
    EXPECT_FALSE(pool->isAllocated(block));
}

TEST_F(HybridPoolKVCacheAllocatorTest, DSV4CPShardedInsertThenReuseSamePrefix) {
    auto config    = makeDSV4HybridPoolConfig(/*block_num=*/64);
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());
    ASSERT_TRUE(injectBlockTreeCache(allocator, config));

    const int spb     = static_cast<int>(config.seq_size_per_block);
    const int seq_len = 10 * spb + 17;

    CacheKeysType full_keys;
    for (int i = 0; i < 10; ++i) {
        full_keys.push_back(1000 + i);
    }
    CacheKeysType request_keys = full_keys;
    request_keys.push_back(2000);  // partial tail key present on the incoming request.

    auto cp_mapper = std::make_shared<CPSlotMapper>(/*cp_rank=*/0, /*cp_size=*/2, spb);
    allocator->setCPSlotMapper(cp_mapper);

    auto seed_res = makeBatchResource(/*batch_size=*/1, config);
    seed_res->setBatchCacheKeys(0, full_keys);
    auto seed_tokens = makeCompleteTokenIds(/*batch_size=*/1, seq_len, spb);

    MallocInfo seed_malloc{seed_res, seed_tokens};
    seed_malloc.reuse_cache         = true;
    seed_malloc.enable_device_cache = false;
    allocator->setCPSlotMapper(cp_mapper);
    ASSERT_TRUE(allocator->malloc(seed_malloc).success);

    InsertInfo insert_info{seed_res, seed_tokens, /*is_resident=*/false};
    allocator->setCPSlotMapper(cp_mapper);
    allocator->insertIntoCache(insert_info);

    FreeInfo seed_free{seed_res, seed_tokens};
    allocator->free(seed_free);

    auto hit_res = makeBatchResource(/*batch_size=*/1, config);
    hit_res->setBatchCacheKeys(0, request_keys);
    auto hit_tokens = makeCompleteTokenIds(/*batch_size=*/1, seq_len, spb);

    MallocInfo hit_malloc{hit_res, hit_tokens};
    hit_malloc.reuse_cache         = true;
    hit_malloc.enable_device_cache = true;
    allocator->setCPSlotMapper(cp_mapper);
    auto result = allocator->malloc(hit_malloc);

    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.reuse_len, 5 * spb * 2);

    FreeInfo hit_free{hit_res, hit_tokens};
    allocator->free(hit_free);
}

TEST_F(HybridPoolKVCacheAllocatorTest, DSV4CPShardedEvictionReclaimsDeviceBlocksInPlace) {
    auto config    = makeDSV4HybridPoolConfig(/*block_num=*/64);
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());
    ASSERT_TRUE(injectBlockTreeCache(allocator, config));

    const int spb     = static_cast<int>(config.seq_size_per_block);
    const int seq_len = 10 * spb + 17;

    CacheKeysType full_keys;
    for (int i = 0; i < 10; ++i) {
        full_keys.push_back(1000 + i);
    }

    auto cp_mapper = std::make_shared<CPSlotMapper>(/*cp_rank=*/0, /*cp_size=*/2, spb);
    allocator->setCPSlotMapper(cp_mapper);

    auto seed_res = makeBatchResource(/*batch_size=*/1, config);
    seed_res->setBatchCacheKeys(0, full_keys);
    auto seed_tokens = makeCompleteTokenIds(/*batch_size=*/1, seq_len, spb);

    MallocInfo seed_malloc{seed_res, seed_tokens};
    seed_malloc.reuse_cache         = true;
    seed_malloc.enable_device_cache = false;
    ASSERT_TRUE(allocator->malloc(seed_malloc).success);

    InsertInfo insert_info{seed_res, seed_tokens, /*is_resident=*/false};
    allocator->insertIntoCache(insert_info);

    FreeInfo seed_free{seed_res, seed_tokens};
    allocator->free(seed_free);

    // Canonical reclaim returns completed primary plans. Cascade group slots and
    // physical free-block progress remain separate observations.
    const size_t free_before = allocator->freeBlocksNum();
    const int reclaimed = allocator->blockTreeCache()->reclaimBlocks(/*num_blocks=*/4, Tier::DEVICE);
    EXPECT_EQ(reclaimed, 4);
    EXPECT_GE(allocator->freeBlocksNum(), free_before + 4);
}

}  // namespace test
}  // namespace rtp_llm

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
