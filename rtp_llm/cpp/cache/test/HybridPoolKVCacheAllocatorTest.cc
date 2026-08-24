#include <gtest/gtest.h>

#include <algorithm>
#include <deque>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "rtp_llm/cpp/utils/AssertUtils.h"

#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/CacheGroupType.h"
#include "rtp_llm/cpp/cache/CPSlotMapper.h"
#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/cache/HybridPoolConfigCreator.h"
#include "rtp_llm/cpp/cache/HybridPoolKVCacheAllocator.h"
#include "rtp_llm/cpp/cache/block_tree_cache/load/LoadAsyncContext.h"
#include "rtp_llm/cpp/cache/KVCacheMetrics.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/StorageBackend.h"
#include "rtp_llm/cpp/cache/block_tree_cache/test/BlockTreeCacheTestUtils.h"
#include "rtp_llm/cpp/cache/LinearKVCacheSpec.h"
#include "rtp_llm/cpp/cache/MHAKVCacheSpec.h"
#include "rtp_llm/cpp/cache/test/BlockPoolTestHelper.h"
#include "rtp_llm/cpp/cache/test/BlockTreeCacheAllocatorTestHelper.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/config/ModelConfig.h"
#include "rtp_llm/cpp/disaggregate/cache_store/CacheStore.h"
#include "rtp_llm/cpp/disaggregate/cache_store/MemoryUtil.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace test {
using block_tree_cache_test::BlockTreeCacheTestPeer;

using TestHybridPoolKVCacheAllocator = BlockTreeCacheTestAllocator<HybridPoolKVCacheAllocator>;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Build a tiny multi-pool config with two groups: group_id=0 LINEAR(layers 0,1)
// and group_id=1 FULL(layers 2,3). Each group has its own per-group block budget,
// so TestHybridPoolKVCacheAllocator creates two independent BlockPools.
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

    auto linear_spec = makeResolvedLinearSpec(config.dtype,
                                              1,
                                              1,
                                              1,
                                              1,
                                              2,
                                              static_cast<uint32_t>(config.seq_size_per_block),
                                              config.dtype,
                                              config.dtype,
                                              "linear");
    auto full_spec = makeResolvedMhaSpec(config.dtype, 1, 1, static_cast<uint32_t>(config.seq_size_per_block), "full");

    config.use_independent_block_pools = true;
    config.fromGroupedSpecs({linear_spec, full_spec},
                            {{0, 1}, {2, 3}},
                            {CacheGroupType::LINEAR, second_type},
                            {"linear", second_type == CacheGroupType::SWA ? "swa" : "full"});

    // Same tokens per block for both groups.
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

static CacheConfig makeTinyFullSwaMultiPoolHybridConfig(uint32_t full_block_num = 12, uint32_t swa_block_num = 12) {
    CacheConfig config;
    config.dtype                       = DataType::TYPE_FP16;
    config.layer_num                   = 4;
    config.layer_all_num               = 4;
    config.block_num                   = std::max(full_block_num, swa_block_num);
    config.seq_size_per_block          = 4;
    config.kernel_seq_size_per_block   = 4;
    config.linear_step                 = 2;
    config.group_layer_num             = 2;
    config.use_independent_block_pools = true;

    auto full_spec                 = makeResolvedMhaSpec(config.dtype, 1, 1, 4, "full");
    auto swa_spec                  = makeResolvedMhaSpec(config.dtype, 1, 1, 4, "swa");
    auto full_policy               = defaultCacheGroupPolicy(CacheGroupType::FULL);
    auto swa_policy                = defaultCacheGroupPolicy(CacheGroupType::SWA);
    swa_policy.enable_prefix_reuse = true;
    swa_policy.sliding_window_size = 8;
    config.fromGroupedSpecs({full_spec, swa_spec},
                            {{0, 1}, {2, 3}},
                            {CacheGroupType::FULL, CacheGroupType::SWA},
                            {"full", "swa"},
                            {full_policy, swa_policy});

    const size_t full_stride     = full_spec->block_size_bytes();
    const size_t swa_stride      = swa_spec->block_size_bytes();
    config.kv_block_stride_bytes = std::max(full_stride, swa_stride);
    config.kv_block_size_bytes   = 2 * config.kv_block_stride_bytes;
    config.kv_scale_stride_bytes = 0;
    config.kv_scale_size_bytes   = 0;
    config.block_size_bytes      = config.kv_block_size_bytes;
    config.layer_to_block_stride_bytes.assign(4, static_cast<int>(config.kv_block_stride_bytes));
    config.setGroupBlockLayout({full_block_num, swa_block_num}, {full_stride, swa_stride}, {0, 0});
    return config;
}

struct MemoryStorageState {
    std::mutex                                                   mutex;
    std::unordered_map<CacheKeyType, std::unordered_set<size_t>> groups_by_key;
};

class InlineStorageBackendExecutor: public StorageBackendExecutor {
public:
    bool start() override {
        return true;
    }
    bool submit(Task task) override {
        task();
        return true;
    }
    void shutdown() noexcept override {}
};

class ManualStorageBackendExecutor: public StorageBackendExecutor {
public:
    bool start() override {
        return true;
    }
    bool submit(Task task) override {
        std::lock_guard<std::mutex> lock(mutex_);
        tasks_.push_back(std::move(task));
        return true;
    }
    void shutdown() noexcept override {
        while (runOne()) {}
    }

    bool runOne() {
        Task task;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (tasks_.empty()) {
                return false;
            }
            task = std::move(tasks_.front());
            tasks_.pop_front();
        }
        task();
        return true;
    }

    size_t pendingCount() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return tasks_.size();
    }

private:
    mutable std::mutex mutex_;
    std::deque<Task>   tasks_;
};

class PolicyMemoryStorageBackend: public StorageBackend {
public:
    explicit PolicyMemoryStorageBackend(
        std::shared_ptr<MemoryStorageState> state,
        std::shared_ptr<StorageBackendExecutor> executor = std::make_shared<InlineStorageBackendExecutor>()):
        StorageBackend(std::move(executor)), state_(std::move(state)) {}
    ~PolicyMemoryStorageBackend() override {
        shutdown();
    }

    size_t matchedKeys() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return matched_keys_;
    }
    std::vector<std::vector<size_t>> writeGroupIds() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return write_group_ids_;
    }
    std::vector<std::vector<size_t>> readGroupIds() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return read_group_ids_;
    }

protected:
    bool initImpl() override {
        return true;
    }
    StorageMatchResult matchImpl(const StorageRequest& request) override {
        size_t matched = request.local_matched_blocks_num;
        {
            std::lock_guard<std::mutex> storage_lock(state_->mutex);
            for (size_t candidate = request.local_matched_blocks_num + 1; candidate <= request.handles.size();
                 ++candidate) {
                bool found = true;
                for (size_t key_index = request.local_matched_blocks_num; found && key_index < candidate; ++key_index) {
                    const auto stored = state_->groups_by_key.find((*request.keys)[key_index]);
                    for (const auto& handle : request.handles[key_index]) {
                        if (isHandleRequired(key_index, candidate, handle.group_id)
                            && (stored == state_->groups_by_key.end() || !stored->second.count(handle.group_id))) {
                            found = false;
                            break;
                        }
                    }
                }
                if (found) {
                    matched = candidate;
                }
            }
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            matched_keys_ = matched;
        }
        return {matched, nullptr};
    }

    void readImpl(const StorageRequest& request, const std::shared_ptr<StorageBackendMatchMeta>&) override {
        std::vector<std::vector<size_t>> group_ids;
        for (const auto& handles : request.handles) {
            group_ids.emplace_back();
            for (const auto& handle : handles) {
                group_ids.back().push_back(handle.group_id);
            }
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            read_group_ids_ = std::move(group_ids);
        }
    }

    void writeImpl(const StorageRequest& request) override {
        std::vector<std::vector<size_t>> group_ids(request.handles.size());
        {
            std::lock_guard<std::mutex> storage_lock(state_->mutex);
            for (size_t key_index = 0; key_index < request.handles.size(); ++key_index) {
                auto& stored = state_->groups_by_key[(*request.keys)[key_index]];
                for (const auto& handle : request.handles[key_index]) {
                    stored.insert(handle.group_id);
                    group_ids[key_index].push_back(handle.group_id);
                }
            }
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            write_group_ids_ = std::move(group_ids);
        }
    }

private:
    std::shared_ptr<MemoryStorageState> state_;
    mutable std::mutex                  mutex_;
    size_t                              matched_keys_{0};
    std::vector<std::vector<size_t>>    write_group_ids_;
    std::vector<std::vector<size_t>>    read_group_ids_;
};

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
    mc.attn_config.tokens_per_block                              = 128;
    mc.attn_config.layer_compress_ratios                         = {4, 128, 4, 128, 0};
    mc.hybrid_attention_config.enable_hybrid_attention           = true;
    mc.hybrid_attention_config.enable_independent_kv_cache_pools = true;
    setDsv4KvCacheSpecs(mc, mc.attn_config.layer_compress_ratios);
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
    mc.attn_config.tokens_per_block = 128;
    std::vector<int> ratios;
    ratios.push_back(128);
    ratios.push_back(128);
    for (int i = 2; i < 61; i++) {
        ratios.push_back((i % 2 == 0) ? 4 : 128);
    }
    ratios.push_back(0);
    mc.attn_config.layer_compress_ratios = ratios;
    setDsv4KvCacheSpecs(mc, mc.attn_config.layer_compress_ratios);
    return mc;
}

// Build a DSV4 7-pool CacheConfig (uses use_independent_block_pools=true).
static CacheConfig makeDSV4HybridPoolConfig(uint32_t block_num = 200) {
    auto mc                                                      = makeProModelConfig();
    mc.hybrid_attention_config.enable_hybrid_attention           = true;
    mc.hybrid_attention_config.enable_independent_kv_cache_pools = true;
    ParallelismConfig pc;
    auto              config = CacheConfigCreator::createBasicConfig(mc, pc, false, 0);
    config.finalizeBlockNums(block_num, RuntimeConfig{});
    return config;
}

static void setExplicitBlocksForGroup(CacheConfig& config, size_t group_id, uint32_t block_num) {
    ASSERT_LT(group_id, static_cast<size_t>(config.groupNums()));
    std::vector<CacheGroupPolicy> policies;
    policies.reserve(static_cast<size_t>(config.groupNums()));
    for (size_t group_id = 0; group_id < static_cast<size_t>(config.groupNums()); ++group_id) {
        policies.push_back(config.policyForGroup(group_id));
    }
    policies[group_id].explicit_block_num     = block_num;
    policies[group_id].charge_to_paged_budget = block_num > 0;
    config.setGroupPolicies(policies);
}

// In a HybridPool config, every explicitly-sized group owns an independent
// pool. Move those pools to pinned host memory and remove their HBM budget
// charge; the source-final policy no longer carries a separate evict mode.
static std::vector<size_t> setPinnedHostPlacementForExplicitIndependentGroups(CacheConfig& config) {
    std::vector<CacheGroupPolicy> policies;
    std::vector<size_t>           pinned_group_ids;
    policies.reserve(static_cast<size_t>(config.groupNums()));
    for (size_t group_id = 0; group_id < static_cast<size_t>(config.groupNums()); ++group_id) {
        auto policy = config.policyForGroup(group_id);
        if (policy.explicit_block_num > 0) {
            policy.memory_placement       = CacheMemoryPlacement::HOST_PINNED;
            policy.charge_to_paged_budget = false;
            pinned_group_ids.push_back(group_id);
        }
        policies.push_back(policy);
    }
    config.setGroupPolicies(policies);
    return pinned_group_ids;
}

static void setGroupReservable(CacheConfig& config, size_t group_id, bool reservable) {
    ASSERT_LT(group_id, static_cast<size_t>(config.groupNums()));
    std::vector<CacheGroupPolicy> policies;
    policies.reserve(static_cast<size_t>(config.groupNums()));
    for (size_t group_id = 0; group_id < static_cast<size_t>(config.groupNums()); ++group_id) {
        policies.push_back(config.policyForGroup(group_id));
    }
    policies[group_id].reservable = reservable;
    config.setGroupPolicies(policies);
}

static size_t firstExplicitGroup(const CacheConfig& config) {
    for (size_t group_id = 0; group_id < static_cast<size_t>(config.groupNums()); ++group_id) {
        const auto policy = config.policyForGroup(group_id);
        if (policy.explicit_block_num > 0) {
            return group_id;
        }
    }
    ADD_FAILURE() << "missing explicit cache group";
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
    res->initGroups(config.topologyPtr());
    return res;
}

static std::vector<uint32_t> groupBlockNumsSnapshot(const CacheConfig& config) {
    std::vector<uint32_t> block_nums;
    block_nums.reserve(static_cast<size_t>(config.groupNums()));
    for (size_t group_id = 0; group_id < static_cast<size_t>(config.groupNums()); ++group_id) {
        block_nums.push_back(config.blockNumForGroup(group_id));
    }
    return block_nums;
}

static void setGroupBlockNums(CacheConfig& config, const std::vector<uint32_t>& block_nums) {
    std::vector<size_t> kv_strides;
    std::vector<size_t> scale_strides;
    kv_strides.reserve(static_cast<size_t>(config.groupNums()));
    scale_strides.reserve(static_cast<size_t>(config.groupNums()));
    for (size_t group_id = 0; group_id < static_cast<size_t>(config.groupNums()); ++group_id) {
        kv_strides.push_back(config.kvBlockStrideBytesForGroup(group_id));
        scale_strides.push_back(config.kvScaleStrideBytesForGroup(group_id));
    }
    config.setGroupBlockLayout(block_nums, kv_strides, scale_strides);
}

static size_t validBlockCount(const BlockIndicesType& blocks) {
    return static_cast<size_t>(
        std::count_if(blocks.begin(), blocks.end(), [](BlockIdxType block) { return !isNullBlockIdx(block); }));
}

static std::shared_ptr<TestHybridPoolKVCacheAllocator>
makeAllocator(const CacheConfig& config, RoleType role_type = RoleType::PDFUSION, int64_t reserve_block_ratio = 0) {
    return std::make_shared<TestHybridPoolKVCacheAllocator>(
        config, AllocationType::DEVICE, nullptr, reserve_block_ratio, role_type);
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
    for (size_t group_id = 0; group_id < expected.size(); ++group_id) {
        const auto& pool = allocator->groupBlockPools()[group_id];
        EXPECT_EQ(pool->freeBlocksNum(), expected[group_id].free_blocks) << "group_id=" << group_id;
        EXPECT_EQ(pool->totalBlocksNum(), expected[group_id].total_blocks) << "group_id=" << group_id;
    }
}

class HybridPoolKVCacheAllocatorTest: public ::testing::Test {
protected:
    void SetUp() override {
        rtp_llm::initLogger();
        createDevice();
    }
};

static void runStorageRoundTrip(const CacheConfig&                      config,
                                const CacheKeysType&                    writer_keys,
                                int                                     writer_seq_len,
                                const CacheKeysType&                    reader_keys,
                                int                                     reader_seq_len,
                                const std::shared_ptr<CPSlotMapper>&    cp_mapper,
                                const std::vector<std::vector<size_t>>& expected_write_groups,
                                const std::vector<std::vector<size_t>>& expected_read_groups) {
    auto          state          = std::make_shared<MemoryStorageState>();
    auto          writer_backend = std::make_shared<PolicyMemoryStorageBackend>(state);
    auto          writer         = makeAllocator(config);
    KVCacheConfig remote_config;
    remote_config.enable_remote_cache = true;
    writer->setBlockTreeCacheConfigForTest(remote_config);
    writer->setStorageBackendForTest(writer_backend);
    ASSERT_TRUE(writer->init());
    writer->setCPSlotMapper(cp_mapper);

    auto writer_resource = makeBatchResource(/*batch_size=*/1, config);
    writer_resource->setBatchCacheKeys(0, writer_keys);
    auto       writer_tokens = makeCompleteTokenIds(/*batch_size=*/1, writer_seq_len, config.seq_size_per_block);
    MallocInfo writer_malloc{writer_resource, writer_tokens};
    writer_malloc.enable_cache_lookup          = false;
    writer_malloc.reuse_cache                  = true;
    writer_malloc.enable_remove_skipped_blocks = false;
    ASSERT_TRUE(writer->malloc(writer_malloc).success);
    writer->insertIntoCache(InsertInfo{writer_resource, writer_tokens, /*is_resident=*/false});
    EXPECT_EQ(writer_backend->writeGroupIds(), expected_write_groups);

    auto reader_backend = std::make_shared<PolicyMemoryStorageBackend>(state);
    auto reader         = makeAllocator(config);
    reader->setBlockTreeCacheConfigForTest(remote_config);
    reader->setStorageBackendForTest(reader_backend);
    ASSERT_TRUE(reader->init());
    reader->setCPSlotMapper(cp_mapper);

    auto reader_resource = makeBatchResource(/*batch_size=*/1, config);
    reader_resource->setBatchCacheKeys(0, reader_keys);
    auto       reader_tokens = makeCompleteTokenIds(/*batch_size=*/1, reader_seq_len, config.seq_size_per_block);
    MallocInfo reader_malloc{reader_resource, reader_tokens};
    reader_malloc.enable_cache_lookup          = true;
    reader_malloc.reuse_cache                  = true;
    reader_malloc.enable_remove_skipped_blocks = false;
    auto result                                = reader->malloc(reader_malloc);
    ASSERT_TRUE(result.success);
    ASSERT_NE(result.async_context, nullptr);
    result.async_context->waitDone();
    EXPECT_TRUE(result.async_context->success());
    EXPECT_EQ(reader_backend->matchedKeys(), 4u);
    EXPECT_EQ(reader_backend->readGroupIds(), expected_read_groups);

    reader->free(FreeInfo{reader_resource, reader_tokens});
    writer->free(FreeInfo{writer_resource, writer_tokens});
}

TEST_F(HybridPoolKVCacheAllocatorTest, StorageRoundTripUsesSparseFullLinearShape) {
    const auto          config = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/12, /*full_block_num=*/12);
    const CacheKeysType stored_keys{100, 101, 102, 103};
    const CacheKeysType request_keys{100, 101, 102, 103, 104, 105};
    runStorageRoundTrip(config,
                        stored_keys,
                        /*writer_seq_len=*/16,
                        request_keys,
                        /*reader_seq_len=*/21,
                        nullptr,
                        {{1}, {0, 1}, {1}, {0, 1}},
                        {{1}, {1}, {1}, {0, 1}});
}

TEST_F(HybridPoolKVCacheAllocatorTest, StorageRoundTripUsesFullSwaWindowShape) {
    const auto          config = makeTinyFullSwaMultiPoolHybridConfig();
    const CacheKeysType stored_keys{200, 201, 202, 203};
    const CacheKeysType request_keys{200, 201, 202, 203, 204, 205};
    runStorageRoundTrip(config,
                        stored_keys,
                        /*writer_seq_len=*/16,
                        request_keys,
                        /*reader_seq_len=*/21,
                        nullptr,
                        {{0}, {0, 1}, {0, 1}, {0, 1}},
                        {{0}, {0}, {0, 1}, {0, 1}});
}

TEST_F(HybridPoolKVCacheAllocatorTest, StorageRoundTripMapsCpCanonicalFullLinearTargets) {
    const auto          config = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/16, /*full_block_num=*/16);
    const CacheKeysType stored_keys{300, 301, 302, 303, 304, 305, 306, 307};
    const CacheKeysType request_keys{300, 301, 302, 303, 304, 305, 306, 307, 308, 309};
    auto                cp_mapper = std::make_shared<CPSlotMapper>(/*cp_rank=*/0, /*cp_size=*/2, /*block_size=*/4);
    runStorageRoundTrip(config,
                        stored_keys,
                        /*writer_seq_len=*/32,
                        request_keys,
                        /*reader_seq_len=*/41,
                        cp_mapper,
                        {{0, 1}, {0, 1}, {0, 1}, {0, 1}},
                        {{1}, {1}, {1}, {0, 1}});
}

// ---------------------------------------------------------------------------
// Init / per-group pool creation
// ---------------------------------------------------------------------------

TEST_F(HybridPoolKVCacheAllocatorTest, InitCreatesIndependentBlockPoolPerGroup) {
    auto config    = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/6, /*full_block_num=*/8);
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    ASSERT_EQ(allocator->groupBlockPools().size(), 2u);
    EXPECT_NE(allocator->groupBlockPools()[0], allocator->groupBlockPools()[1]);

    // Per-pool totalBlocksNum = group_block_nums[group_id] - 1 (block 0 reserved).
    EXPECT_EQ(allocator->groupBlockPools()[0]->totalBlocksNum(), 6u - 1u);
    EXPECT_EQ(allocator->groupBlockPools()[1]->totalBlocksNum(), 8u - 1u);
}

TEST_F(HybridPoolKVCacheAllocatorTest, SwaDefaultRegionGroupPoolUsesGpuBacking) {
    auto config    = makeTinySwaMultiPoolHybridConfig(/*linear_block_num=*/6, /*swa_block_num=*/8);
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    ASSERT_EQ(allocator->groupBlockPools().size(), 2u);
    EXPECT_EQ(allocator->groupBlockPools()[0]->where(), MemoryType::MEMORY_GPU);
    EXPECT_EQ(allocator->groupBlockPools()[1]->where(), MemoryType::MEMORY_GPU);
}

TEST_F(HybridPoolKVCacheAllocatorTest, GetDeviceBlockPoolReturnsNullptrInHybridPoolMode) {
    // HybridPoolKVCacheAllocator owns one DeviceBlockPool per group and does not
    // expose a single canonical pool.
    auto config    = makeTinyMultiPoolHybridConfig();
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());
    EXPECT_EQ(allocator->getDeviceBlockPool(), nullptr);
}

// ---------------------------------------------------------------------------
// Aggregated counters
// ---------------------------------------------------------------------------

TEST_F(HybridPoolKVCacheAllocatorTest, TotalAndFreeBlocksAggregateAcrossGroups) {
    auto config    = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/6, /*full_block_num=*/8);
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    const size_t expected_total = (6u - 1u) + (8u - 1u);
    EXPECT_EQ(allocator->totalBlocksNum(), expected_total);
    EXPECT_EQ(allocator->freeBlocksNum(), expected_total);
}

TEST_F(HybridPoolKVCacheAllocatorTest, TokenAggregatorsUseDifferentCapacityScopes) {
    auto config = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/6, /*full_block_num=*/8);
    // Token capacity aggregators use FULL groups first: 7 blocks * 4 tokens.
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    EXPECT_EQ(allocator->maxAvailableTokensNum(), 28u);
    EXPECT_EQ(allocator->availableTokensNum(), 28u);
    EXPECT_EQ(allocator->totalTokensNum(), 28u);
}

TEST_F(HybridPoolKVCacheAllocatorTest, TokenAggregatorsUseCPVirtualBlockSizeForFullGroups) {
    auto config    = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/6, /*full_block_num=*/8);
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    EXPECT_EQ(allocator->maxAvailableTokensNum(), 7u * 4u);
    EXPECT_EQ(allocator->availableTokensNum(), 7u * 4u);

    allocator->setCPSlotMapper(std::make_shared<CPSlotMapper>(/*cp_rank=*/0, /*cp_size=*/2, /*block_size=*/4));

    EXPECT_EQ(allocator->maxAvailableTokensNum(), 7u * 8u);
    EXPECT_EQ(allocator->availableTokensNum(), 7u * 8u);
}

TEST_F(HybridPoolKVCacheAllocatorTest, TokenAggregatorsFallBackToGlobalSeqSize) {
    auto config               = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/6, /*full_block_num=*/6);
    config.seq_size_per_block = 4;
    auto allocator            = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    EXPECT_EQ(allocator->maxAvailableTokensNum(), 5u * 4u);
    EXPECT_EQ(allocator->availableTokensNum(), 5u * 4u);
}

TEST_F(HybridPoolKVCacheAllocatorTest, IndependentPoolsUseOneBalancedReferenceCount) {
    auto config    = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/6, /*full_block_num=*/8);
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    auto pool0 = allocator->groupBlockPools()[0];
    auto pool1 = allocator->groupBlockPools()[1];

    const size_t free_total_before = allocator->freeBlocksNum();
    auto         g0_blocks         = pool0->malloc(2).value();
    auto         g1_blocks         = pool1->malloc(3).value();
    ASSERT_EQ(g0_blocks.size(), 2u);
    ASSERT_EQ(g1_blocks.size(), 3u);
    pool0->incRef(g0_blocks);
    pool1->incRef(g1_blocks);

    EXPECT_EQ(allocator->freeBlocksNum(), free_total_before - 5u);
    for (const auto block : g0_blocks) {
        EXPECT_EQ(pool0->refCount(block), 1u);
    }
    for (const auto block : g1_blocks) {
        EXPECT_EQ(pool1->refCount(block), 1u);
    }

    // A second holder on the same numeric block id remains pool-local.
    pool0->incRef(g0_blocks[0]);
    pool1->incRef(g1_blocks[0]);
    EXPECT_EQ(pool0->refCount(g0_blocks[0]), 2u);
    EXPECT_EQ(pool1->refCount(g1_blocks[0]), 2u);

    pool0->decRef(g0_blocks);
    pool1->decRef(g1_blocks);
    EXPECT_EQ(allocator->freeBlocksNum(), free_total_before - 2u);

    pool0->decRef(g0_blocks[0]);
    pool1->decRef(g1_blocks[0]);
    EXPECT_EQ(allocator->freeBlocksNum(), free_total_before);
}

// ---------------------------------------------------------------------------
// Address / buffer lookups
// ---------------------------------------------------------------------------

TEST_F(HybridPoolKVCacheAllocatorTest, ConvertIndexToAddrAndBufferDefault) {
    auto config    = makeTinyMultiPoolHybridConfig();
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

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

    auto bufs = allocator->convertIndexToBuffer(
        /*layer_id=*/3, /*block_id=*/1, /*partition_count=*/1, /*partition_id=*/0);
    ASSERT_FALSE(bufs.empty());
    EXPECT_NE(bufs[0].addr, nullptr);
}

TEST_F(HybridPoolKVCacheAllocatorTest, ConvertIndexToAddrAndBufferByGroup) {
    auto config    = makeTinyMultiPoolHybridConfig();
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

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

    auto layout = allocator->allLayerCacheBase();
    EXPECT_EQ(layout.topology().layerGroupIdsSnapshot(), config.layerGroupIdsSnapshot());
    EXPECT_EQ(layout.topology().groupTypesSnapshot(), config.groupTypesSnapshot());
    EXPECT_EQ(layout.groups().size(), static_cast<size_t>(config.groupNums()));
    for (size_t i = 0; i < static_cast<size_t>(config.layer_all_num); ++i) {
        const auto& layer = layout.topology().layer(static_cast<int>(i));
        ASSERT_FALSE(layer.group_tags.empty());
        for (const auto& tag : layer.group_tags) {
            EXPECT_TRUE(layout.group(tag).hasLayer(i)) << "layer " << i << " tag=" << tag;
        }
    }
}

// ---------------------------------------------------------------------------
// regUserMr / getMrCostTimeMs
// ---------------------------------------------------------------------------

TEST_F(HybridPoolKVCacheAllocatorTest, RegUserMrWithoutCacheStoreIsNoOpAndZeroCost) {
    auto config    = makeTinyMultiPoolHybridConfig();
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    // No CacheStore is plumbed in: regUserMr should be a benign no-op for every
    // group pool, and the aggregated MR cost remains zero.
    EXPECT_NO_THROW(allocator->regUserMr(/*model_id=*/0, /*cache_store=*/nullptr));
    EXPECT_EQ(allocator->getMrCostTimeMs(), 0);
}

// ---------------------------------------------------------------------------
// hasAvailableBlocksForReserve via reserve_block_ratio
// ---------------------------------------------------------------------------

TEST_F(HybridPoolKVCacheAllocatorTest, ReserveRatioIsAppliedToEachGroupPoolForInitMalloc) {
    auto config    = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/6, /*full_block_num=*/4);
    auto allocator = makeAllocator(config, RoleType::PDFUSION, /*reserve_block_ratio=*/50);
    ASSERT_TRUE(allocator->init());

    // seq_len=4 -> 1 block per group.
    auto batch_res = makeBatchResource(/*batch_size=*/1, config);
    batch_res->setBatchCacheKeys(0, CacheKeysType{100});
    auto       token_ids = makeCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/4, /*seq_size_per_block=*/4);
    MallocInfo malloc_info{batch_res, token_ids};
    malloc_info.enable_cache_lookup = false;
    malloc_info.reuse_cache         = false;
    auto result                     = allocator->malloc(malloc_info);
    EXPECT_TRUE(result.success);
}

TEST_F(HybridPoolKVCacheAllocatorTest, ReserveBlocksRejectsWhenGroupCannotMeetItsShare) {
    // Force a group whose free_blocks < need + group_reserve_blocks.
    auto config    = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/6, /*full_block_num=*/4);
    auto allocator = makeAllocator(config, RoleType::PDFUSION, /*reserve_block_ratio=*/100);
    ASSERT_TRUE(allocator->init());

    auto batch_res = makeBatchResource(/*batch_size=*/1, config);
    batch_res->setBatchCacheKeys(0, CacheKeysType{100});
    auto       token_ids = makeCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/4, /*seq_size_per_block=*/4);
    MallocInfo malloc_info{batch_res, token_ids};
    malloc_info.enable_cache_lookup = false;
    malloc_info.reuse_cache         = false;
    malloc_info.verbose             = false;
    auto result                     = allocator->malloc(malloc_info);
    EXPECT_FALSE(result.success);
}

TEST_F(HybridPoolKVCacheAllocatorTest, PoolMetricsSnapshotsReportReserveBlocks) {
    auto              config        = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/6, /*full_block_num=*/8);
    constexpr int64_t reserve_ratio = 50;
    auto              allocator     = makeAllocator(config, RoleType::PDFUSION, reserve_ratio);
    ASSERT_TRUE(allocator->init());

    const auto snapshots = allocator->poolMetricsSnapshots();
    ASSERT_EQ(snapshots.size(), 2u);
    EXPECT_EQ("linear", snapshots[0].pool_name);
    EXPECT_EQ("full", snapshots[1].pool_name);
    EXPECT_EQ(snapshots[0].used_blocks, snapshots[0].total_blocks - snapshots[0].free_blocks);
    EXPECT_EQ(snapshots[1].used_blocks, snapshots[1].total_blocks - snapshots[1].free_blocks);

    const size_t total_blocks = snapshots[0].total_blocks + snapshots[1].total_blocks;
    EXPECT_EQ(allocator->reserveBlocksNum() * snapshots[0].total_blocks / total_blocks, snapshots[0].reserve_blocks);
    EXPECT_EQ(allocator->reserveBlocksNum() * snapshots[1].total_blocks / total_blocks, snapshots[1].reserve_blocks);
}

static const CachePoolMetricsSnapshot* findFinalPoolSnapshot(const std::vector<CachePoolMetricsSnapshot>& snapshots,
                                                             const std::string&                           pool_name) {
    const CachePoolMetricsSnapshot* found = nullptr;
    for (const CachePoolMetricsSnapshot& snapshot : snapshots) {
        if (snapshot.pool_name != pool_name) {
            continue;
        }
        if (found != nullptr) {
            ADD_FAILURE() << "duplicate final pool snapshot: " << pool_name;
            return nullptr;
        }
        found = &snapshot;
    }
    if (found == nullptr) {
        ADD_FAILURE() << "missing final pool snapshot: " << pool_name;
    }
    return found;
}

static size_t distinctValidBlockCount(const BlockIndicesType& blocks) {
    std::unordered_set<BlockIdxType> distinct;
    for (BlockIdxType block : blocks) {
        if (!isNullBlockIdx(block)) {
            distinct.insert(block);
        }
    }
    return distinct.size();
}

static void expectSameFinalPoolMetrics(const CachePoolMetricsSnapshot& expected,
                                       const CachePoolMetricsSnapshot& actual,
                                       const std::string&              stage) {
    const std::string context = "pool=" + actual.pool_name + " stage=" + stage;
    EXPECT_EQ(actual.tier, expected.tier) << context;
    EXPECT_EQ(actual.block_size_bytes, expected.block_size_bytes) << context;
    EXPECT_EQ(actual.total_blocks, expected.total_blocks) << context;
    EXPECT_EQ(actual.reserve_blocks, expected.reserve_blocks) << context;
    EXPECT_EQ(actual.free_blocks, expected.free_blocks) << context;
    EXPECT_EQ(actual.used_blocks, expected.used_blocks) << context;
    EXPECT_EQ(actual.available_blocks, expected.available_blocks) << context;
    EXPECT_EQ(actual.active_blocks, expected.active_blocks) << context;
    EXPECT_EQ(actual.request_ref_blocks, expected.request_ref_blocks) << context;
    EXPECT_EQ(actual.block_cache_ref_blocks, expected.block_cache_ref_blocks) << context;
    EXPECT_EQ(actual.load_ref_blocks, expected.load_ref_blocks) << context;
    EXPECT_EQ(actual.eviction_ref_blocks, expected.eviction_ref_blocks) << context;
    EXPECT_EQ(actual.store_ref_blocks, expected.store_ref_blocks) << context;
    EXPECT_FLOAT_EQ(actual.used_ratio, expected.used_ratio) << context;
}

TEST_F(HybridPoolKVCacheAllocatorTest, AllPrefixReuseDisabledPoolMetricsFollowAllocatorLifecycle) {
    auto config   = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/6, /*full_block_num=*/4);
    auto policies = config.groupPoliciesSnapshot();
    ASSERT_EQ(policies.size(), 2u);
    for (CacheGroupPolicy& policy : policies) {
        policy.enable_prefix_reuse = false;
    }
    config.setGroupPolicies(policies);

    auto allocator = makeAllocator(config, RoleType::PDFUSION, /*reserve_block_ratio=*/50);
    ASSERT_TRUE(allocator->init());
    const BlockTreeCachePtr& cache = allocator->blockTreeCacheOwner();
    ASSERT_NE(cache, nullptr);

    auto collect_final = [&]() {
        const std::vector<BlockTreePoolMetricsSnapshot> tree   = cache->poolMetricsSnapshots();
        const std::vector<KVCachePoolMetricsSnapshot>   device = allocator->poolMetricsSnapshots();
        return mergeCachePoolMetricsSnapshots(device, tree);
    };
    auto expect_empty_tree_state = [&](const std::string& stage) {
        EXPECT_TRUE(cache->groupSets().empty()) << stage;
        EXPECT_TRUE(cache->poolMetricsSnapshots().empty()) << stage;
        const CacheStats stats = cache->getStats();
        EXPECT_EQ(stats.tree_node_count, 0u) << stage;
        EXPECT_EQ(stats.device_heap_total_size, 0u) << stage;
        EXPECT_EQ(stats.host_heap_total_size, 0u) << stage;
        EXPECT_EQ(stats.disk_heap_total_size, 0u) << stage;
    };

    // State A: initialized, nothing allocated.
    expect_empty_tree_state("init");
    const std::vector<KVCachePoolMetricsSnapshot> allocator_init = allocator->poolMetricsSnapshots();
    ASSERT_EQ(allocator_init.size(), 2u);
    const std::vector<CachePoolMetricsSnapshot> init_final = collect_final();
    ASSERT_EQ(init_final.size(), allocator_init.size());

    size_t init_reserve_blocks = 0;
    for (const KVCachePoolMetricsSnapshot& source : allocator_init) {
        const CachePoolMetricsSnapshot* snapshot = findFinalPoolSnapshot(init_final, source.pool_name);
        ASSERT_NE(snapshot, nullptr) << source.pool_name;
        EXPECT_EQ(snapshot->tier, tierName(Tier::DEVICE)) << source.pool_name;
        EXPECT_EQ(snapshot->block_size_bytes, source.block_size_bytes) << source.pool_name;
        EXPECT_EQ(snapshot->total_blocks, source.total_blocks) << source.pool_name;
        EXPECT_EQ(snapshot->reserve_blocks, source.reserve_blocks) << source.pool_name;
        EXPECT_GT(snapshot->total_blocks, 0u) << source.pool_name;
        EXPECT_EQ(snapshot->free_blocks, snapshot->total_blocks) << source.pool_name;
        EXPECT_EQ(snapshot->used_blocks, 0u) << source.pool_name;
        EXPECT_EQ(snapshot->available_blocks, snapshot->free_blocks) << source.pool_name;
        EXPECT_EQ(snapshot->active_blocks, 0u) << source.pool_name;
        EXPECT_EQ(snapshot->request_ref_blocks, 0u) << source.pool_name;
        EXPECT_EQ(snapshot->block_cache_ref_blocks, 0u) << source.pool_name;
        EXPECT_EQ(snapshot->eviction_ref_blocks, 0u) << source.pool_name;
        EXPECT_EQ(snapshot->store_ref_blocks, 0u) << source.pool_name;
        EXPECT_FLOAT_EQ(snapshot->used_ratio, 0.0f) << source.pool_name;
        init_reserve_blocks += snapshot->reserve_blocks;
    }
    // A zero quota would silently pass the reserve assertions below.
    EXPECT_GT(init_reserve_blocks, 0u);

    // State B: one real block per pool held by a REQUEST reference.
    auto batch_res = makeBatchResource(/*batch_size=*/1, config);
    batch_res->setBatchCacheKeys(0, CacheKeysType{100});
    auto       token_ids = makeCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/4, /*seq_size_per_block=*/4);
    MallocInfo malloc_info{batch_res, token_ids};
    malloc_info.enable_cache_lookup = false;
    malloc_info.reuse_cache         = false;
    ASSERT_TRUE(allocator->malloc(malloc_info).success);

    expect_empty_tree_state("after_malloc");
    const std::vector<KVCachePoolMetricsSnapshot> allocator_malloc = allocator->poolMetricsSnapshots();
    const std::vector<CachePoolMetricsSnapshot>   malloc_final     = collect_final();
    ASSERT_EQ(allocator_malloc.size(), allocator_init.size());
    ASSERT_EQ(malloc_final.size(), allocator_malloc.size());

    for (const KVCachePoolMetricsSnapshot& source : allocator_malloc) {
        const CachePoolMetricsSnapshot* snapshot = findFinalPoolSnapshot(malloc_final, source.pool_name);
        ASSERT_NE(snapshot, nullptr) << source.pool_name;
        const CachePoolMetricsSnapshot* init_snapshot = findFinalPoolSnapshot(init_final, source.pool_name);
        ASSERT_NE(init_snapshot, nullptr) << source.pool_name;

        const size_t expected_request_blocks =
            distinctValidBlockCount(batch_res->blocks(0, static_cast<int>(source.pool_index)));
        ASSERT_EQ(expected_request_blocks, 1u) << source.pool_name;

        EXPECT_EQ(snapshot->block_size_bytes, init_snapshot->block_size_bytes) << source.pool_name;
        EXPECT_EQ(snapshot->total_blocks, init_snapshot->total_blocks) << source.pool_name;
        EXPECT_EQ(snapshot->reserve_blocks, init_snapshot->reserve_blocks) << source.pool_name;
        EXPECT_EQ(snapshot->used_blocks, snapshot->total_blocks - snapshot->free_blocks) << source.pool_name;
        EXPECT_EQ(snapshot->used_blocks, expected_request_blocks) << source.pool_name;
        EXPECT_EQ(snapshot->request_ref_blocks, expected_request_blocks) << source.pool_name;
        EXPECT_EQ(snapshot->available_blocks, snapshot->total_blocks - expected_request_blocks) << source.pool_name;
        EXPECT_EQ(snapshot->active_blocks, expected_request_blocks) << source.pool_name;
        EXPECT_EQ(snapshot->block_cache_ref_blocks, 0u) << source.pool_name;
        EXPECT_EQ(snapshot->eviction_ref_blocks, 0u) << source.pool_name;
        EXPECT_EQ(snapshot->store_ref_blocks, 0u) << source.pool_name;
        EXPECT_FLOAT_EQ(snapshot->used_ratio,
                        static_cast<float>(100.0 * snapshot->used_blocks / static_cast<double>(snapshot->total_blocks)))
            << source.pool_name;
    }

    // State C: freeing must restore every field of state A.
    FreeInfo free_info{batch_res, token_ids};
    allocator->free(free_info);

    expect_empty_tree_state("after_free");
    const std::vector<CachePoolMetricsSnapshot> free_final = collect_final();
    ASSERT_EQ(free_final.size(), init_final.size());
    for (const CachePoolMetricsSnapshot& expected : init_final) {
        const CachePoolMetricsSnapshot* actual = findFinalPoolSnapshot(free_final, expected.pool_name);
        ASSERT_NE(actual, nullptr) << expected.pool_name;
        expectSameFinalPoolMetrics(expected, *actual, "after_free");
        EXPECT_EQ(actual->free_blocks, actual->total_blocks) << expected.pool_name;
        EXPECT_EQ(actual->available_blocks, actual->total_blocks) << expected.pool_name;
    }
}

// Distinct per-field values keep a wrong field source visible instead of coincidentally equal.
static KVCachePoolMetricsSnapshot
makeMergeAllocatorInput(const std::string& pool_name, size_t seed, size_t total_blocks, size_t free_blocks) {
    KVCachePoolMetricsSnapshot snapshot;
    snapshot.pool_index                = seed;
    snapshot.pool_name                 = pool_name;
    snapshot.block_size_bytes          = 1000 + seed;
    snapshot.total_blocks              = total_blocks;
    snapshot.free_blocks               = free_blocks;
    snapshot.used_blocks               = total_blocks - free_blocks;
    snapshot.active_blocks             = 7 + seed;
    snapshot.reserve_blocks            = 5 + seed;
    snapshot.request_ref_blocks        = 11 + seed;
    snapshot.block_cache_ref_blocks    = 13 + seed;
    snapshot.load_ref_blocks           = 14 + seed;
    snapshot.eviction_ref_blocks       = 15 + seed;
    snapshot.store_ref_blocks          = 16 + seed;
    snapshot.used_ratio = static_cast<float>(100.0 * snapshot.used_blocks / static_cast<double>(snapshot.total_blocks));
    return snapshot;
}

static BlockTreePoolMetricsSnapshot makeMergeTreeInput(Tier               tier,
                                                       const std::string& pool_name,
                                                       size_t             seed,
                                                       size_t             total_blocks,
                                                       size_t             free_blocks,
                                                       size_t             available_blocks) {
    BlockTreePoolMetricsSnapshot snapshot;
    snapshot.tier                      = tier;
    snapshot.pool_name                 = pool_name;
    snapshot.block_size_bytes          = 2000 + seed;
    snapshot.total_blocks              = total_blocks;
    snapshot.free_blocks               = free_blocks;
    snapshot.used_blocks               = total_blocks - free_blocks;
    snapshot.available_blocks          = available_blocks;
    snapshot.active_blocks             = total_blocks - available_blocks;
    snapshot.request_ref_blocks        = 31 + seed;
    snapshot.block_cache_ref_blocks    = 33 + seed;
    snapshot.load_ref_blocks           = 34 + seed;
    snapshot.eviction_ref_blocks       = 35 + seed;
    snapshot.store_ref_blocks          = 36 + seed;
    return snapshot;
}

static void expectMergedRowFromAllocator(const KVCachePoolMetricsSnapshot& source,
                                         size_t                            expected_available,
                                         const CachePoolMetricsSnapshot&   actual,
                                         const std::string&                context) {
    EXPECT_EQ(actual.tier, tierName(Tier::DEVICE)) << context;
    EXPECT_EQ(actual.pool_name, source.pool_name) << context;
    EXPECT_EQ(actual.block_size_bytes, source.block_size_bytes) << context;
    EXPECT_EQ(actual.total_blocks, source.total_blocks) << context;
    EXPECT_EQ(actual.free_blocks, source.free_blocks) << context;
    EXPECT_EQ(actual.used_blocks, source.used_blocks) << context;
    EXPECT_EQ(actual.reserve_blocks, source.reserve_blocks) << context;
    EXPECT_EQ(actual.active_blocks, source.active_blocks) << context;
    EXPECT_EQ(actual.request_ref_blocks, source.request_ref_blocks) << context;
    EXPECT_EQ(actual.block_cache_ref_blocks, source.block_cache_ref_blocks) << context;
    EXPECT_EQ(actual.load_ref_blocks, source.load_ref_blocks) << context;
    EXPECT_EQ(actual.eviction_ref_blocks, source.eviction_ref_blocks) << context;
    EXPECT_EQ(actual.store_ref_blocks, source.store_ref_blocks) << context;
    EXPECT_FLOAT_EQ(actual.used_ratio, source.used_ratio) << context;
    EXPECT_EQ(actual.available_blocks, expected_available) << context;
}

static void expectMergedRowFromTree(const BlockTreePoolMetricsSnapshot& source,
                                    size_t                              expected_available,
                                    const CachePoolMetricsSnapshot&     actual,
                                    const std::string&                  context) {
    EXPECT_EQ(actual.tier, tierName(source.tier)) << context;
    EXPECT_EQ(actual.pool_name, source.pool_name) << context;
    EXPECT_EQ(actual.block_size_bytes, source.block_size_bytes) << context;
    EXPECT_EQ(actual.total_blocks, source.total_blocks) << context;
    EXPECT_EQ(actual.free_blocks, source.free_blocks) << context;
    EXPECT_EQ(actual.used_blocks, source.used_blocks) << context;
    // Reserve is an allocator quota, so tree-only pools keep the default.
    EXPECT_EQ(actual.reserve_blocks, 0u) << context;
    EXPECT_EQ(actual.active_blocks, source.active_blocks) << context;
    EXPECT_EQ(actual.request_ref_blocks, source.request_ref_blocks) << context;
    EXPECT_EQ(actual.block_cache_ref_blocks, source.block_cache_ref_blocks) << context;
    EXPECT_EQ(actual.load_ref_blocks, source.load_ref_blocks) << context;
    EXPECT_EQ(actual.eviction_ref_blocks, source.eviction_ref_blocks) << context;
    EXPECT_EQ(actual.store_ref_blocks, source.store_ref_blocks) << context;
    EXPECT_FLOAT_EQ(actual.used_ratio,
                    static_cast<float>(100.0 * (source.total_blocks - source.free_blocks)
                                       / static_cast<double>(source.total_blocks)))
        << context;
    EXPECT_EQ(actual.available_blocks, expected_available) << context;
}

TEST_F(HybridPoolKVCacheAllocatorTest, MergeCachePoolMetricsSnapshotsPreservesReportContract) {
    struct ExpectedRow {
        bool   from_allocator;
        size_t input_index;
        size_t available_blocks;
    };
    struct MergeCase {
        std::string                               name;
        std::vector<KVCachePoolMetricsSnapshot>   allocator_snapshots;
        std::vector<BlockTreePoolMetricsSnapshot> tree_snapshots;
        std::vector<ExpectedRow>                  expected_rows;
    };

    const std::vector<MergeCase> cases = {
        {"allocator_then_tree_only_append",
         {makeMergeAllocatorInput("linear", 1, /*total=*/100, /*free=*/40),
          makeMergeAllocatorInput("full", 2, /*total=*/80, /*free=*/80)},
         {makeMergeTreeInput(Tier::DEVICE, "linear", 3, /*total=*/111, /*free=*/44, /*available=*/70),
          makeMergeTreeInput(Tier::DEVICE, "tree_only_device", 4, /*total=*/50, /*free=*/20, /*available=*/35),
          makeMergeTreeInput(Tier::HOST, "host_pool", 5, /*total=*/200, /*free=*/150, /*available=*/180),
          makeMergeTreeInput(Tier::DISK, "disk_pool", 6, /*total=*/300, /*free=*/300, /*available=*/300)},
         {// Device rows come entirely from allocator snapshots.
          {true, 0, 92},
          {true, 1, 71},
          // Tree-only pools follow every allocator pool, in tree input order.
          {false, 1, 35},
          {false, 2, 180},
          {false, 3, 300}}},
        {"allocator_duplicate_keeps_first_and_skips_rest",
         {makeMergeAllocatorInput("dup_pool", 1, /*total=*/100, /*free=*/40),
          makeMergeAllocatorInput("dup_pool", 9, /*total=*/500, /*free=*/5)},
         {},
         {{true, 0, 92}}},
        {"same_named_tree_does_not_override_allocator",
         {makeMergeAllocatorInput("dup_tree_pool", 1, /*total=*/100, /*free=*/40)},
         {makeMergeTreeInput(Tier::DEVICE, "dup_tree_pool", 3, /*total=*/100, /*free=*/40, /*available=*/70),
          makeMergeTreeInput(Tier::DEVICE, "dup_tree_pool", 4, /*total=*/100, /*free=*/40, /*available=*/25)},
         {{true, 0, 92}}},
        // Asymmetry with the allocator loop: tree-only duplicates are all emitted.
        {"tree_only_duplicates_are_not_deduplicated",
         {},
         {makeMergeTreeInput(Tier::DEVICE, "orphan_device", 3, /*total=*/100, /*free=*/40, /*available=*/70),
          makeMergeTreeInput(Tier::DEVICE, "orphan_device", 4, /*total=*/100, /*free=*/30, /*available=*/65),
          makeMergeTreeInput(Tier::HOST, "host_dup", 5, /*total=*/200, /*free=*/150, /*available=*/180),
          makeMergeTreeInput(Tier::HOST, "host_dup", 6, /*total=*/200, /*free=*/100, /*available=*/170)},
         {{false, 0, 70}, {false, 1, 65}, {false, 2, 180}, {false, 3, 170}}},
    };

    for (const MergeCase& merge_case : cases) {
        const std::vector<CachePoolMetricsSnapshot> merged =
            mergeCachePoolMetricsSnapshots(merge_case.allocator_snapshots, merge_case.tree_snapshots);
        ASSERT_EQ(merged.size(), merge_case.expected_rows.size()) << "case=" << merge_case.name;
        for (size_t row = 0; row < merge_case.expected_rows.size(); ++row) {
            const ExpectedRow& expected = merge_case.expected_rows[row];
            const std::string  context  = "case=" + merge_case.name + " row=" + std::to_string(row);
            if (expected.from_allocator) {
                ASSERT_LT(expected.input_index, merge_case.allocator_snapshots.size()) << context;
                expectMergedRowFromAllocator(merge_case.allocator_snapshots[expected.input_index],
                                             expected.available_blocks,
                                             merged[row],
                                             context);
            } else {
                ASSERT_LT(expected.input_index, merge_case.tree_snapshots.size()) << context;
                expectMergedRowFromTree(
                    merge_case.tree_snapshots[expected.input_index], expected.available_blocks, merged[row], context);
            }
        }
    }
}

TEST_F(HybridPoolKVCacheAllocatorTest, DeviceCacheMinFreeBlocksAreDistributedByPoolCapacity) {
    CacheConfig config = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/6, /*full_block_num=*/8);
    std::shared_ptr<TestHybridPoolKVCacheAllocator> allocator =
        makeAllocator(config, RoleType::PDFUSION, /*reserve_block_ratio=*/0);
    ASSERT_TRUE(allocator->init());
    allocator->setReserveBlocksNum(6);

    const std::vector<KVCachePoolMetricsSnapshot> snapshots = allocator->poolMetricsSnapshots();
    ASSERT_EQ(snapshots.size(), 2u);
    const size_t total_blocks = snapshots[0].total_blocks + snapshots[1].total_blocks;
    EXPECT_EQ(6u * snapshots[0].total_blocks / total_blocks, snapshots[0].reserve_blocks);
    EXPECT_EQ(6u * snapshots[1].total_blocks / total_blocks, snapshots[1].reserve_blocks);
}

TEST_F(HybridPoolKVCacheAllocatorTest, ReserveRatioAndSnapshotsExcludeNonReservablePool) {
    auto config = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/6, /*full_block_num=*/8);
    setGroupReservable(config, /*linear group_id=*/0, false);

    constexpr int64_t reserve_ratio = 50;
    auto              allocator     = makeAllocator(config, RoleType::PDFUSION, reserve_ratio);
    ASSERT_TRUE(allocator->init());
    ASSERT_EQ(allocator->groupBlockPools().size(), 2u);

    const size_t excluded_free      = allocator->groupBlockPools()[0]->freeBlocksNum();
    const size_t participating_free = allocator->groupBlockPools()[1]->freeBlocksNum();
    ASSERT_GT(excluded_free, 0u);
    ASSERT_GT(participating_free, 0u);
    EXPECT_EQ(allocator->reserveBlocksNum(),
              static_cast<size_t>(reserve_ratio) * participating_free / static_cast<size_t>(100));
    EXPECT_NE(allocator->reserveBlocksNum(),
              static_cast<size_t>(reserve_ratio) * (excluded_free + participating_free) / static_cast<size_t>(100));

    const auto snapshots = allocator->poolMetricsSnapshots();
    ASSERT_EQ(snapshots.size(), 2u);
    EXPECT_EQ(snapshots[0].reserve_blocks, 0u);
    EXPECT_EQ(snapshots[1].reserve_blocks, allocator->reserveBlocksNum());
}

TEST_F(HybridPoolKVCacheAllocatorTest, NonReservableOnlyPoolsHaveDivisionSafeZeroReserveShares) {
    auto config = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/6, /*full_block_num=*/8);
    setGroupReservable(config, /*linear group_id=*/0, false);
    setGroupReservable(config, /*full group_id=*/1, false);

    auto allocator = makeAllocator(config, RoleType::PDFUSION, /*reserve_block_ratio=*/50);
    ASSERT_TRUE(allocator->init());
    EXPECT_EQ(allocator->reserveBlocksNum(), 0u);

    // Non-reservable pools must report zero reserve.
    const auto snapshots = allocator->poolMetricsSnapshots();
    ASSERT_EQ(snapshots.size(), 2u);
    EXPECT_EQ(snapshots[0].reserve_blocks, 0u);
    EXPECT_EQ(snapshots[1].reserve_blocks, 0u);
}

TEST_F(HybridPoolKVCacheAllocatorTest, ReserveBlocksUseCPShardedFullGroupNeed) {
    auto config    = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/20, /*full_block_num=*/6);
    auto allocator = makeAllocator(config, RoleType::PDFUSION, /*reserve_block_ratio=*/1);
    ASSERT_TRUE(allocator->init());

    auto batch_res = makeBatchResource(/*batch_size=*/1, config);
    batch_res->setBatchCacheKeys(0, CacheKeysType{100, 101, 102, 103, 104, 105, 106, 107});
    auto token_ids = makeCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/32, /*seq_size_per_block=*/4);
    allocator->setCPSlotMapper(std::make_shared<CPSlotMapper>(/*cp_rank=*/0, /*cp_size=*/2, /*block_size=*/4));

    MallocInfo malloc_info{batch_res, token_ids};
    malloc_info.enable_cache_lookup = false;
    malloc_info.reuse_cache         = false;

    auto result = allocator->malloc(malloc_info);
    ASSERT_TRUE(result.success);
    EXPECT_EQ(validBlockCount(batch_res->blocks(0, /*group_id=*/1)), 4u);

    FreeInfo free_info{batch_res, token_ids};
    allocator->free(free_info);
}

TEST_F(HybridPoolKVCacheAllocatorTest, PreparedSWALoadReserveRejectsPoolLocalShortfall) {
    auto config    = makeTinySwaMultiPoolHybridConfig(/*linear_block_num=*/8, /*swa_block_num=*/4);
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());
    allocator->setReserveBlocksNum(1);

    const auto pools = allocator->groupBlockPools();
    ASSERT_EQ(pools.size(), 2u);
    auto swa_holds = pools[1]->malloc(pools[1]->freeBlocksNum());
    ASSERT_TRUE(swa_holds.has_value());
    pools[1]->incRef(*swa_holds);
    ASSERT_EQ(pools[1]->freeBlocksNum(), 0u);

    auto batch_res = makeBatchResource(/*batch_size=*/1, config);
    batch_res->mutableBlockIds(0, /*group_id=*/0).assign({NULL_BLOCK_IDX});
    batch_res->mutableBlockIds(0, /*group_id=*/1).assign({NULL_BLOCK_IDX});
    auto       token_ids = makeCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/4, /*seq_size_per_block=*/4);
    MallocInfo malloc_info{batch_res, token_ids};
    malloc_info.reuse_cache = true;

    // The old aggregate predicate saw the free LINEAR pool and admitted this
    // load even though its required SWA target had no physical block.
    ASSERT_GE(allocator->freeBlocksNum(), allocator->reserveBlocksNum() + 1);
    EXPECT_EQ(allocator->preparedReserveStatusForTest(malloc_info, allocator->reserveBlocksNum(), {{}, {0}}),
              MallocStatus::RETRYABLE_RESOURCE_EXHAUSTED);

    pools[1]->decRef(*swa_holds);
}

TEST_F(HybridPoolKVCacheAllocatorTest, PreparedSWALoadCountsHeldDeviceBlocksForPermanentCapacity) {
    auto config    = makeTinyFullSwaMultiPoolHybridConfig(/*full_block_num=*/8, /*swa_block_num=*/3);
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());
    allocator->setReserveBlocksNum(0);

    const auto pools = allocator->groupBlockPools();
    ASSERT_EQ(pools.size(), 2u);
    const size_t swa_total = pools[1]->totalBlocksNum();
    ASSERT_GT(swa_total, 0u);
    auto device_matches = pools[1]->malloc(swa_total);
    ASSERT_TRUE(device_matches.has_value());
    pools[1]->incRef(*device_matches);

    auto resource = makeBatchResource(/*batch_size=*/1, config);
    resource->mutableBlockIds(0, /*group_id=*/1).assign(*device_matches);
    resource->mutableBlockIds(0, /*group_id=*/1).add({NULL_BLOCK_IDX});
    const size_t remote_position = swa_total;
    auto token_ids = makeCompleteTokenIds(
        /*batch_size=*/1, /*seq_length=*/static_cast<int>((swa_total + 1) * 4), /*seq_size_per_block=*/4);
    MallocInfo malloc_info{resource, token_ids};
    malloc_info.reuse_cache = true;
    malloc_info.verbose     = false;

    // The request already owns every physical SWA block through DEVICE
    // matches. Its remote target is an additional block inside the same
    // logical prefix, so retrying can never make the full footprint fit.
    EXPECT_EQ(allocator->preparedReserveStatusForTest(
                  malloc_info, /*reserve_blocks=*/0, {{}, RequiredPositions{remote_position}}),
              MallocStatus::PERMANENT_RESOURCE_EXHAUSTED);

    allocator->free(FreeInfo{resource, token_ids});
}

TEST_F(HybridPoolKVCacheAllocatorTest, PreparedSWALoadIgnoresNonPhysicalDummyInHeldFootprint) {
    auto config    = makeTinyFullSwaMultiPoolHybridConfig(/*full_block_num=*/8, /*swa_block_num=*/3);
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());
    allocator->setReserveBlocksNum(0);

    const auto pools = allocator->groupBlockPools();
    ASSERT_EQ(pools.size(), 2u);
    ASSERT_EQ(pools[1]->totalBlocksNum(), 2u);
    auto device_match = pools[1]->malloc();
    ASSERT_TRUE(device_match.has_value());
    pools[1]->incRef(*device_match);

    auto resource = makeBatchResource(/*batch_size=*/1, config);
    resource->mutableBlockIds(0, /*group_id=*/1).assign({*device_match, 0, NULL_BLOCK_IDX});
    auto token_ids = makeCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/12, /*seq_size_per_block=*/4);
    MallocInfo malloc_info{resource, token_ids};
    malloc_info.reuse_cache = true;
    malloc_info.verbose     = false;

    // One held DEVICE block plus one remote target exactly fits this pool.
    // The dummy block 0 and the NULL target are not physical holds.
    EXPECT_EQ(allocator->preparedReserveStatusForTest(
                  malloc_info, /*reserve_blocks=*/0, {{}, RequiredPositions{2}}),
              MallocStatus::NONE);

    resource->mutableBlockIds(0, /*group_id=*/1).assign({*device_match});
    allocator->free(FreeInfo{resource, token_ids});
}

TEST_F(HybridPoolKVCacheAllocatorTest, PreparedNoReuseDoesNotDoubleCountPartialGroupAllocation) {
    auto config    = makeTinyFullSwaMultiPoolHybridConfig(/*full_block_num=*/2, /*swa_block_num=*/3);
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());
    allocator->setReserveBlocksNum(0);

    const auto pools = allocator->groupBlockPools();
    ASSERT_EQ(pools.size(), 2u);
    ASSERT_EQ(pools[0]->totalBlocksNum(), 1u);
    auto partial_full = pools[0]->malloc();
    ASSERT_TRUE(partial_full.has_value());
    pools[0]->incRef(*partial_full);
    auto swa_pins = pools[1]->malloc(pools[1]->freeBlocksNum());
    ASSERT_TRUE(swa_pins.has_value());
    pools[1]->incRef(*swa_pins);

    auto resource = makeBatchResource(/*batch_size=*/1, config);
    resource->mutableBlockIds(0, /*group_id=*/0).assign({*partial_full});
    auto token_ids = makeCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/4, /*seq_size_per_block=*/4);
    MallocInfo malloc_info{resource, token_ids};
    malloc_info.reuse_cache = false;
    malloc_info.verbose     = false;

    // With reuse disabled each planner already reports the full footprint.
    // The partial group-0 allocation must not be added a second time; group 1
    // is only temporarily unavailable and therefore remains RETRYABLE.
    EXPECT_EQ(allocator->preparedReserveStatusForTest(malloc_info, /*reserve_blocks=*/0, {{}, {}}),
              MallocStatus::RETRYABLE_RESOURCE_EXHAUSTED);

    pools[1]->decRef(*swa_pins);
    allocator->free(FreeInfo{resource, token_ids});
}

TEST_F(HybridPoolKVCacheAllocatorTest, PreparedNoReuseCountsRequiredSparseSWAHole) {
    auto config    = makeTinyFullSwaMultiPoolHybridConfig(/*full_block_num=*/8, /*swa_block_num=*/4);
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());
    allocator->setReserveBlocksNum(0);

    const auto pools = allocator->groupBlockPools();
    ASSERT_EQ(pools.size(), 2u);
    ASSERT_EQ(pools[1]->totalBlocksNum(), 3u);
    auto swa_pin = pools[1]->malloc();
    ASSERT_TRUE(swa_pin.has_value());
    pools[1]->incRef(*swa_pin);
    ASSERT_EQ(pools[1]->freeBlocksNum(), 2u);

    auto resource = makeBatchResource(/*batch_size=*/1, config);
    resource->mutableBlockIds(0, /*group_id=*/1).assign({NULL_BLOCK_IDX, NULL_BLOCK_IDX, NULL_BLOCK_IDX});
    auto token_ids = makeCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/12, /*seq_size_per_block=*/4);
    MallocInfo malloc_info{resource, token_ids};
    malloc_info.reuse_cache = false;
    malloc_info.verbose     = false;

    // The no-reuse SWA policy normally materializes only the two active tail
    // blocks. Remote loading additionally requires sparse position 0, so the
    // two currently free blocks are insufficient for this prepared attempt.
    EXPECT_EQ(allocator->preparedReserveStatusForTest(
                  malloc_info, /*reserve_blocks=*/0, {{}, RequiredPositions{0}}),
              MallocStatus::RETRYABLE_RESOURCE_EXHAUSTED);

    pools[1]->decRef(*swa_pin);
}

TEST_F(HybridPoolKVCacheAllocatorTest, DeferredBackendMatchReportsSWAPoolShortfallAsRetryable) {
    auto config    = makeTinyFullSwaMultiPoolHybridConfig(/*full_block_num=*/8, /*swa_block_num=*/4);
    auto state     = std::make_shared<MemoryStorageState>();
    auto executor  = std::make_shared<ManualStorageBackendExecutor>();
    auto backend   = std::make_shared<PolicyMemoryStorageBackend>(state, executor);
    auto allocator = makeAllocator(config);

    KVCacheConfig remote_config;
    remote_config.enable_remote_cache = true;
    allocator->setBlockTreeCacheConfigForTest(remote_config);
    allocator->setStorageBackendForTest(backend);
    ASSERT_TRUE(allocator->init());
    allocator->setReserveBlocksNum(0);

    constexpr CacheKeyType key = 7001;
    state->groups_by_key[key]  = {0, 1};

    const auto pools = allocator->groupBlockPools();
    ASSERT_EQ(pools.size(), 2u);
    auto swa_holds = pools[1]->malloc(pools[1]->freeBlocksNum());
    ASSERT_TRUE(swa_holds.has_value());
    pools[1]->incRef(*swa_holds);
    ASSERT_EQ(pools[1]->freeBlocksNum(), 0u);
    ASSERT_GT(pools[0]->freeBlocksNum(), 0u);

    auto resource = makeBatchResource(/*batch_size=*/1, config);
    resource->setBatchCacheKeys(0, {key});
    // Keep one completed block reusable; the final block remains writable and
    // therefore is not eligible for backend matching.
    auto       token_ids = makeCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/8, /*seq_size_per_block=*/4);
    MallocInfo malloc_info{resource, token_ids};
    malloc_info.enable_cache_lookup = true;
    malloc_info.reuse_cache         = true;
    malloc_info.verbose             = false;

    auto result = allocator->malloc(malloc_info);
    ASSERT_TRUE(result.success);
    auto load_context = std::dynamic_pointer_cast<LoadAsyncContext>(result.async_context);
    ASSERT_NE(load_context, nullptr);
    EXPECT_TRUE(load_context->needBackendMatch());
    EXPECT_FALSE(load_context->done());
    ASSERT_EQ(executor->pendingCount(), 1u);

    ASSERT_TRUE(executor->runOne());
    EXPECT_TRUE(load_context->done());
    EXPECT_FALSE(load_context->success());
    EXPECT_EQ(load_context->mallocStatus(), MallocStatus::RETRYABLE_RESOURCE_EXHAUSTED);
    EXPECT_EQ(resource->curBlocksNum(), 0u);
    EXPECT_EQ(executor->pendingCount(), 0u);

    pools[1]->decRef(*swa_holds);
    load_context.reset();
    result.async_context.reset();

    auto retry_result = allocator->malloc(malloc_info);
    ASSERT_TRUE(retry_result.success);
    auto retry_context = std::dynamic_pointer_cast<LoadAsyncContext>(retry_result.async_context);
    ASSERT_NE(retry_context, nullptr);
    ASSERT_EQ(executor->pendingCount(), 1u);
    ASSERT_TRUE(executor->runOne());  // match and materialize
    ASSERT_EQ(executor->pendingCount(), 1u);
    ASSERT_TRUE(executor->runOne());  // read
    retry_context->waitDone();
    EXPECT_TRUE(retry_context->success());
    EXPECT_EQ(retry_context->mallocStatus(), MallocStatus::NONE);
    EXPECT_GT(resource->curBlocksNum(), 0u);
    allocator->free(FreeInfo{resource, token_ids});
}

TEST_F(HybridPoolKVCacheAllocatorTest, ReserveCheckIsBypassedWhenMallocInfoLacksContext) {
    // hasAvailableBlocksForReserve returns true when info has no resource/tokens.
    auto config    = makeTinyMultiPoolHybridConfig();
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    MallocInfo info{};
    EXPECT_TRUE(allocator->hasAvailableBlocksForReserve(info, /*reserve_blocks=*/9999));
}

TEST_F(HybridPoolKVCacheAllocatorTest, InitMallocRollbackFreesPartiallyAllocatedGroupBlocks) {
    // group_id=0 has enough room for the LINEAR tail block; group_id=1 cannot satisfy
    // the 3 FULL blocks needed for seq_len=9. initMallocForCommonLen should
    // roll group_id=0 back after group_id=1 fails.
    auto config    = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/3, /*full_block_num=*/3);
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    const auto counters_before = snapshotPoolCounters(allocator);

    auto batch_res = makeBatchResource(/*batch_size=*/1, config);
    batch_res->setBatchCacheKeys(0, CacheKeysType{100, 101, 102});
    auto       token_ids = makeCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/9, /*seq_size_per_block=*/4);
    MallocInfo malloc_info{batch_res, token_ids};
    malloc_info.enable_cache_lookup = false;
    malloc_info.reuse_cache         = false;
    malloc_info.verbose             = false;

    auto result = allocator->malloc(malloc_info);
    EXPECT_FALSE(result.success);
    // A 3-block pool exposes 2 usable blocks (block 0 is the null sentinel), so 3 FULL blocks can
    // never fit: the per-group total-capacity test must report PERMANENT so the scheduler errors
    // the stream out instead of parking it in WAITING forever.
    EXPECT_EQ(result.status, MallocStatus::PERMANENT_RESOURCE_EXHAUSTED);

    EXPECT_EQ(batch_res->curBlocksNum(), 0u);
    EXPECT_EQ(batch_res->blocksNum(0, /*group_id=*/0), 0u);
    EXPECT_EQ(batch_res->blocksNum(0, /*group_id=*/1), 0u);
    expectPoolCountersEq(allocator, counters_before);
}

TEST_F(HybridPoolKVCacheAllocatorTest, InitMallocRollbackReleasesLowerTierBackfillsAndAppendedBlocks) {
    auto config    = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/4, /*full_block_num=*/3);
    auto allocator = makeAllocator(config);

    KVCacheConfig tiered_config;
    tiered_config.enable_host_cache  = true;
    tiered_config.host_cache_size_mb = 1;
    allocator->setBlockTreeCacheConfigForTest(std::move(tiered_config));
    ASSERT_TRUE(allocator->init());

    const auto& cache = allocator->blockTreeCacheOwner();
    ASSERT_NE(cache, nullptr);

    const CacheKeysType                               cached_keys{100, 101};
    std::vector<std::vector<GroupSetResource>>        slots(cached_keys.size(),
                                                     std::vector<GroupSetResource>(cache->groupSets().size()));
    std::vector<std::pair<GroupSetPtr, BlockIdxType>> host_sources;
    for (const GroupSetPtr& group_set : cache->groupSets()) {
        ASSERT_NE(group_set->hostPool(), nullptr);
        for (size_t path_index = 0; path_index < cached_keys.size(); ++path_index) {
            const BlockIdxType source_block = group_set->allocateSingleBlock(Tier::HOST, BlockTreeRefType::CACHE);
            ASSERT_FALSE(isNullBlockIdx(source_block));
            slots[path_index][group_set->groupSetId()].host_block = source_block;
            host_sources.emplace_back(group_set, source_block);
        }
    }
    cache->insert(cached_keys, slots, Tier::HOST);

    const auto counters_before = snapshotPoolCounters(allocator);
    for (const auto& [group_set, source_block] : host_sources) {
        EXPECT_EQ(group_set->hostPool()->treeRefCount(source_block), 1u);
    }

    auto batch_res = makeBatchResource(/*batch_size=*/1, config);
    batch_res->setBatchCacheKeys(0, CacheKeysType{100, 101, 102});
    auto       token_ids = makeCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/9, /*seq_size_per_block=*/4);
    MallocInfo malloc_info{batch_res, token_ids};
    malloc_info.enable_cache_lookup = true;
    malloc_info.reuse_cache         = true;
    malloc_info.verbose             = false;

    const auto result = allocator->malloc(malloc_info);
    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.status, MallocStatus::PERMANENT_RESOURCE_EXHAUSTED);
    EXPECT_EQ(result.async_context, nullptr);
    EXPECT_EQ(batch_res->curBlocksNum(), 0u);
    EXPECT_EQ(batch_res->blocksNum(0, /*group_id=*/0), 0u);
    EXPECT_EQ(batch_res->blocksNum(0, /*group_id=*/1), 0u);
    expectPoolCountersEq(allocator, counters_before);
    for (const auto& [group_set, source_block] : host_sources) {
        EXPECT_EQ(group_set->hostPool()->treeRefCount(source_block), 1u);
    }
}

TEST_F(HybridPoolKVCacheAllocatorTest, InitMallocRollbackReleasesRequestRefsAndPreservesCachedTree) {
    auto config    = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/4, /*full_block_num=*/4);
    auto allocator = makeAllocator(config, RoleType::PDFUSION, /*reserve_block_ratio=*/100);
    ASSERT_TRUE(allocator->init());

    const auto seeded = seedCompleteBlockTreePath(allocator, CacheKeysType{100});
    ASSERT_TRUE(seeded.success);
    const auto linear_cached = seeded.blocks_by_tag.at(config.tagForGroup(/*group_id=*/0)).front();
    const auto full_cached   = seeded.blocks_by_tag.at(config.tagForGroup(/*group_id=*/1)).front();
    ASSERT_FALSE(isNullBlockIdx(linear_cached));
    ASSERT_FALSE(isNullBlockIdx(full_cached));
    const auto pools = allocator->groupBlockPools();
    ASSERT_EQ(pools.size(), 2u);
    ASSERT_EQ(pools[0]->refCount(linear_cached), 1u);
    ASSERT_EQ(pools[1]->refCount(full_cached), 1u);

    const auto counters_before = snapshotPoolCounters(allocator);

    auto batch_res = makeBatchResource(/*batch_size=*/1, config);
    batch_res->setBatchCacheKeys(0, CacheKeysType{100, 101, 102});
    auto       token_ids = makeCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/8, /*seq_size_per_block=*/4);
    MallocInfo malloc_info{batch_res, token_ids};
    malloc_info.enable_cache_lookup = true;
    malloc_info.reuse_cache         = true;
    malloc_info.verbose             = false;

    auto result = allocator->malloc(malloc_info);
    EXPECT_FALSE(result.success);

    EXPECT_EQ(batch_res->curBlocksNum(), 0u);
    EXPECT_EQ(batch_res->blocksNum(0, /*group_id=*/0), 0u);
    EXPECT_EQ(batch_res->blocksNum(0, /*group_id=*/1), 0u);
    expectPoolCountersEq(allocator, counters_before);
    EXPECT_TRUE(pools[0]->isAllocated(linear_cached));
    EXPECT_TRUE(pools[1]->isAllocated(full_cached));
    EXPECT_EQ(pools[0]->refCount(linear_cached), 1u);
    EXPECT_EQ(pools[1]->refCount(full_cached), 1u);
    EXPECT_FALSE(allocator->blockTreeCacheOwner()->tree()->findNode(CacheKeysType{100}).empty());
}

TEST_F(HybridPoolKVCacheAllocatorTest, IncrMallocRollbackFreesPartiallyAllocatedGroupBlocks) {
    auto config    = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/4, /*full_block_num=*/2);
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    auto batch_res = makeBatchResource(/*batch_size=*/1, config);
    batch_res->setBatchCacheKeys(0, CacheKeysType{100, 101, 102});

    auto       token_ids = makeCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/4, /*seq_size_per_block=*/4);
    MallocInfo init_info{batch_res, token_ids};
    init_info.enable_cache_lookup = false;
    init_info.reuse_cache         = false;
    ASSERT_TRUE(allocator->malloc(init_info).success);

    ASSERT_EQ(batch_res->blocksNum(0, /*group_id=*/0), 1u);
    ASSERT_EQ(batch_res->blocksNum(0, /*group_id=*/1), 1u);
    const auto linear_block_before = batch_res->blocks(0, /*group_id=*/0)[0];
    const auto full_block_before   = batch_res->blocks(0, /*group_id=*/1)[0];
    const auto counters_before     = snapshotPoolCounters(allocator);

    // group_id=0 can append one real LINEAR tail block. group_id=1 has no remaining
    // free blocks and no cache to evict, so FULL allocation fails.
    token_ids->setSeqLength(9);
    MallocInfo incr_info{batch_res, token_ids};
    incr_info.enable_cache_lookup = false;
    incr_info.reuse_cache         = false;
    auto incr_result              = allocator->malloc(incr_info);
    EXPECT_FALSE(incr_result.success);

    ASSERT_EQ(batch_res->blocksNum(0, /*group_id=*/0), 1u);
    ASSERT_EQ(batch_res->blocksNum(0, /*group_id=*/1), 1u);
    EXPECT_EQ(batch_res->blocks(0, /*group_id=*/0)[0], linear_block_before);
    EXPECT_EQ(batch_res->blocks(0, /*group_id=*/1)[0], full_block_before);
    expectPoolCountersEq(allocator, counters_before);
}

TEST_F(HybridPoolKVCacheAllocatorTest, IncrMallocRollbackRestoresLinearBackfilledSlots) {
    // Block 0 is reserved by each pool, so FULL needs three configured blocks
    // to provide the two request blocks used by the initial allocation.
    auto config    = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/4, /*full_block_num=*/3);
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    auto batch_res = makeBatchResource(/*batch_size=*/1, config);
    batch_res->setBatchCacheKeys(0, CacheKeysType{100, 101, 102});

    auto       token_ids = makeCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/8, /*seq_size_per_block=*/4);
    MallocInfo init_info{batch_res, token_ids};
    init_info.enable_cache_lookup = false;
    init_info.reuse_cache         = false;
    ASSERT_TRUE(allocator->malloc(init_info).success);
    ASSERT_EQ(batch_res->blocksNum(0, /*group_id=*/0), 2u);
    ASSERT_EQ(batch_res->blocksNum(0, /*group_id=*/1), 2u);

    auto& linear_ids       = batch_res->mutableBlockIds(0, /*group_id=*/0);
    auto  removed_block_id = linear_ids.blocks()[1];
    ASSERT_FALSE(isNullBlockIdx(removed_block_id));
    allocator->groupBlockPools()[0]->decRef(removed_block_id);
    linear_ids.setAt(1, NULL_BLOCK_IDX);
    const auto counters_before = snapshotPoolCounters(allocator);

    // LINEAR first backfills the old sparse tail and appends a new tail block.
    // FULL then fails because its independent pool is exhausted. Rollback must
    // restore both the historical NULL resource and the original logical length.
    token_ids->setSeqLength(9);
    MallocInfo incr_info{batch_res, token_ids};
    incr_info.enable_cache_lookup = false;
    incr_info.reuse_cache         = false;
    EXPECT_FALSE(allocator->malloc(incr_info).success);

    ASSERT_EQ(batch_res->blocksNum(0, /*group_id=*/0), 2u);
    ASSERT_EQ(batch_res->blocksNum(0, /*group_id=*/1), 2u);
    EXPECT_TRUE(isNullBlockIdx(batch_res->blocks(0, /*group_id=*/0)[1]));
    expectPoolCountersEq(allocator, counters_before);
}

// ---------------------------------------------------------------------------
// Full malloc / free cycle
// ---------------------------------------------------------------------------

TEST_F(HybridPoolKVCacheAllocatorTest, MallocAndFreeCycleAcrossPerGroupPools) {
    auto config    = makeTinyMultiPoolHybridConfig(/*linear_block_num=*/8, /*full_block_num=*/8);
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    const size_t free_before = allocator->freeBlocksNum();

    auto batch_res = makeBatchResource(/*batch_size=*/1, config);
    batch_res->setBatchCacheKeys(0, CacheKeysType{100, 101, 102});
    auto       token_ids = makeCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/12, /*seq_size_per_block=*/4);
    MallocInfo malloc_info{batch_res, token_ids};
    malloc_info.enable_cache_lookup = false;
    malloc_info.reuse_cache         = false;
    auto result                     = allocator->malloc(malloc_info);
    ASSERT_TRUE(result.success);
    EXPECT_LT(allocator->freeBlocksNum(), free_before);

    FreeInfo free_info{batch_res, token_ids};
    allocator->free(free_info);
    EXPECT_EQ(allocator->freeBlocksNum(), free_before);
}

// ---------------------------------------------------------------------------
// DSV4 7-group HybridPool: covers per-tag addressing and SWA tail
// ---------------------------------------------------------------------------

TEST_F(HybridPoolKVCacheAllocatorTest, DSV4InitAndAggregatedCounters) {
    auto config    = makeDSV4HybridPoolConfig(/*block_num=*/200);
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

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

    ASSERT_EQ(allocator->groupBlockPools().size(), 7u);
    for (size_t group_id = 0; group_id < allocator->groupBlockPools().size(); ++group_id) {
        EXPECT_EQ(allocator->groupBlockPools()[group_id]->where(), MemoryType::MEMORY_GPU)
            << "group_id=" << group_id << " tag=" << config.tagForGroup(group_id);
    }
}

// memory_placement=HOST_PINNED must move only the opted-in pools off HBM; every other pool of
// the same DSV4 config stays on the device.
TEST_F(HybridPoolKVCacheAllocatorTest, DSV4FixedTagPoolsUsePinnedHostBackingWhenPlacementIsHostPinned) {
    auto       config      = makeDSV4HybridPoolConfig(/*block_num=*/200);
    const auto pinned_gids = setPinnedHostPlacementForExplicitIndependentGroups(config);
    ASSERT_FALSE(pinned_gids.empty());
    ASSERT_LT(pinned_gids.size(), static_cast<size_t>(config.groupNums()));

    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    ASSERT_EQ(allocator->groupBlockPools().size(), 7u);
    const std::unordered_set<size_t> pinned_set(pinned_gids.begin(), pinned_gids.end());
    for (size_t gid = 0; gid < allocator->groupBlockPools().size(); ++gid) {
        const bool expect_pinned = pinned_set.count(gid) > 0;
        EXPECT_EQ(allocator->groupBlockPools()[gid]->where(),
                  expect_pinned ? MemoryType::MEMORY_CPU_PINNED : MemoryType::MEMORY_GPU)
            << "gid=" << gid << " tag=" << config.tagForGroup(gid);
    }
}

TEST_F(HybridPoolKVCacheAllocatorTest, DSV4HCAStateReuseEnabledAllocatesTailOnly) {
    auto config        = makeDSV4HybridPoolConfig(/*block_num=*/200);
    config.linear_step = 4;
    auto allocator     = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    const int hca_state_group_id = config.groupIdForTag("hca_state");
    ASSERT_GE(hca_state_group_id, 0);
    ASSERT_EQ(config.tagForGroup(hca_state_group_id), "hca_state");
    ASSERT_GT(allocator->groupBlockPools().size(), static_cast<size_t>(hca_state_group_id));

    const size_t hca_free_before = allocator->groupBlockPools()[hca_state_group_id]->freeBlocksNum();

    auto batch_res = makeBatchResource(/*batch_size=*/1, config);
    batch_res->setBatchCacheKeys(0, CacheKeysType{100, 101, 102, 103, 104, 105, 106, 107, 108, 109});
    auto token_ids = makeCompleteTokenIds(
        /*batch_size=*/1, /*seq_length=*/10 * static_cast<int>(config.seq_size_per_block), config.seq_size_per_block);

    MallocInfo malloc_info{batch_res, token_ids};
    malloc_info.enable_cache_lookup = false;
    malloc_info.reuse_cache         = true;
    auto result                     = allocator->malloc(malloc_info);
    ASSERT_TRUE(result.success);

    const auto& hca_blocks = batch_res->blocks(0, hca_state_group_id);
    ASSERT_EQ(hca_blocks.size(), 10u);
    EXPECT_EQ(validBlockCount(hca_blocks), 1u);
    EXPECT_TRUE(isNullBlockIdx(hca_blocks[8]));
    EXPECT_FALSE(isNullBlockIdx(hca_blocks[9]));
    EXPECT_EQ(hca_free_before - allocator->groupBlockPools()[hca_state_group_id]->freeBlocksNum(), 1u);
}

TEST_F(HybridPoolKVCacheAllocatorTest, TokenAggregatorsIgnoreSmallHCAStatePool) {
    auto config = makeDSV4HybridPoolConfig(/*block_num=*/50);

    const int hca_state_group_id = config.groupIdForTag("hca_state");
    ASSERT_GE(hca_state_group_id, 0);
    ASSERT_EQ(config.tagForGroup(hca_state_group_id), "hca_state");
    auto block_nums                = groupBlockNumsSnapshot(config);
    block_nums[hca_state_group_id] = 2;
    setGroupBlockNums(config, block_nums);

    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());
    ASSERT_GT(allocator->groupBlockPools().size(), static_cast<size_t>(hca_state_group_id));

    const auto hca_state_tokens =
        allocator->groupBlockPools()[hca_state_group_id]->totalBlocksNum() * config.seq_size_per_block;
    EXPECT_LT(hca_state_tokens, allocator->totalTokensNum());
    EXPECT_EQ(allocator->availableTokensNum(), allocator->maxAvailableTokensNum());
    EXPECT_EQ(allocator->totalTokensNum(), allocator->maxAvailableTokensNum());
}

TEST_F(HybridPoolKVCacheAllocatorTest, DSV4ConfigUsesGroupOwnedBytesForPagedBlockSize) {
    auto              mc = makeTinyDSV4ModelConfig();
    ParallelismConfig pc;
    auto              config = CacheConfigCreator::createBasicConfig(mc, pc, false, 0);

    ASSERT_EQ(config.groupNums(), 7);

    size_t expected_non_paged_bytes = 0;
    size_t expected_paged_bytes     = 0;
    for (size_t group_id = 0; group_id < static_cast<size_t>(config.groupNums()); ++group_id) {
        const auto type = config.typeForGroup(group_id);
        const auto expected_group_bytes =
            config.layerIdsForGroup(group_id).size()
            * (config.kvBlockStrideBytesForGroup(group_id) + config.kvScaleStrideBytesForGroup(group_id));
        EXPECT_EQ(config.blockSizeBytesForGroup(group_id), expected_group_bytes) << "group_id=" << group_id;
        if (!config.usesExplicitIndependentBlocks(group_id)
            && (type == CacheGroupType::FULL || type == CacheGroupType::LINEAR)) {
            expected_paged_bytes += expected_group_bytes;
        } else {
            expected_non_paged_bytes += expected_group_bytes;
        }
    }

    EXPECT_GT(expected_non_paged_bytes, 0u);
    EXPECT_GT(expected_paged_bytes, 0u);

    EXPECT_EQ(config.block_size_bytes, expected_paged_bytes);
}

TEST_F(HybridPoolKVCacheAllocatorTest, ReserveRatioExcludesExplicitIndependentPools) {
    auto config = makeDSV4HybridPoolConfig(/*block_num=*/200);
    ASSERT_LT(firstExplicitGroup(config), static_cast<size_t>(config.groupNums()));

    constexpr int64_t reserve_ratio = 10;
    auto              allocator     = makeAllocator(config, RoleType::PDFUSION, reserve_ratio);
    ASSERT_TRUE(allocator->init());

    size_t reservable_available = 0;
    size_t all_available        = 0;
    for (size_t group_id = 0; group_id < allocator->groupBlockPools().size(); ++group_id) {
        const size_t available = allocator->groupBlockPools()[group_id]->freeBlocksNum();
        all_available += available;
        if (!config.usesExplicitIndependentBlocks(group_id)) {
            reservable_available += available;
        }
    }
    ASSERT_GT(reservable_available, 0u);
    ASSERT_GT(all_available, reservable_available);
    EXPECT_EQ(allocator->reserveBlocksNum(),
              static_cast<size_t>(reserve_ratio) * reservable_available / static_cast<size_t>(100));
    EXPECT_NE(allocator->reserveBlocksNum(),
              static_cast<size_t>(reserve_ratio) * all_available / static_cast<size_t>(100));
}

TEST_F(HybridPoolKVCacheAllocatorTest, DSV4FinalizeBlockNumsUsesHcaStatePoolBlocks) {
    auto         config            = makeDSV4HybridPoolConfig(/*block_num=*/50);
    const size_t explicit_group_id = firstExplicitGroup(config);
    setExplicitBlocksForGroup(config, explicit_group_id, 50);

    RuntimeConfig rt;  // unused inside finalizeBlockNums today
    config.finalizeBlockNums(/*global_block_num=*/200, rt);

    for (size_t group_id = 0; group_id < static_cast<size_t>(config.groupNums()); ++group_id) {
        const uint32_t expected = config.policyForGroup(group_id).explicit_block_num > 0 ? 50u : 200u;
        EXPECT_EQ(config.blockNumForGroup(group_id), expected) << "group_id=" << group_id;
    }

    const size_t expected_reserve = 50u * config.blockSizeBytesForGroup(explicit_group_id);
    EXPECT_EQ(config.explicitly_sized_pool_reserve_bytes, expected_reserve);
}

TEST_F(HybridPoolKVCacheAllocatorTest, DSV4FinalizeBlockNumsUsesGlobalBlocksWhenHcaStateBlocksDisabled) {
    auto config = makeDSV4HybridPoolConfig(/*block_num=*/123);
    setExplicitBlocksForGroup(config, firstExplicitGroup(config), 0);

    RuntimeConfig rt;
    config.finalizeBlockNums(/*global_block_num=*/123, rt);

    for (size_t group_id = 0; group_id < static_cast<size_t>(config.groupNums()); ++group_id) {
        EXPECT_EQ(config.blockNumForGroup(group_id), 123u);
    }
    EXPECT_EQ(config.explicitly_sized_pool_reserve_bytes, 0u);
}

TEST_F(HybridPoolKVCacheAllocatorTest, DSV4GpuHcaStatePoolIncludesFixedReserve) {
    auto         config            = makeDSV4HybridPoolConfig(/*block_num=*/50);
    const size_t explicit_group_id = firstExplicitGroup(config);
    setExplicitBlocksForGroup(config, explicit_group_id, 50);

    RuntimeConfig rt;
    config.finalizeBlockNums(/*global_block_num=*/200, rt);

    for (size_t group_id = 0; group_id < static_cast<size_t>(config.groupNums()); ++group_id) {
        const uint32_t expected = config.policyForGroup(group_id).explicit_block_num > 0 ? 50u : 200u;
        EXPECT_EQ(config.blockNumForGroup(group_id), expected) << "group_id=" << group_id;
    }
    const size_t expected_reserve = 50u * config.blockSizeBytesForGroup(explicit_group_id);
    EXPECT_EQ(config.explicitly_sized_pool_reserve_bytes, expected_reserve);
}

// Mirror image of DSV4GpuHcaStatePoolIncludesFixedReserve: a pinned-host pool keeps its explicit
// block count but must NOT be deducted from the device paged budget, otherwise the KV cache
// silently shrinks by bytes that never live in HBM.
TEST_F(HybridPoolKVCacheAllocatorTest, DSV4PinnedHcaStatePoolExcludesFixedReserve) {
    auto         config       = makeDSV4HybridPoolConfig(/*block_num=*/50);
    const size_t explicit_gid = firstExplicitGroup(config);
    setExplicitBlocksForGroup(config, explicit_gid, 50);
    const auto pinned_gids = setPinnedHostPlacementForExplicitIndependentGroups(config);
    ASSERT_EQ(pinned_gids.size(), 1u);
    ASSERT_EQ(pinned_gids.front(), explicit_gid);

    RuntimeConfig rt;
    config.finalizeBlockNums(/*global_block_num=*/200, rt);

    // Block counts are unaffected by residency: the explicit pool still gets its 50 blocks.
    for (size_t gid = 0; gid < static_cast<size_t>(config.groupNums()); ++gid) {
        const uint32_t expected = config.policyForGroup(gid).explicit_block_num > 0 ? 50u : 200u;
        EXPECT_EQ(config.blockNumForGroup(gid), expected) << "gid=" << gid;
    }
    EXPECT_GT(config.blockSizeBytesForGroup(explicit_gid), 0u);
    EXPECT_EQ(config.explicitly_sized_pool_reserve_bytes, 0u);
}

TEST_F(HybridPoolKVCacheAllocatorTest, DSV4StateSwaPoolsWithoutExplicitBlocksScaleWithLinearStep) {
    auto mc                                                      = makeProModelConfig();
    mc.hybrid_attention_config.enable_hybrid_attention           = true;
    mc.hybrid_attention_config.enable_independent_kv_cache_pools = true;
    ParallelismConfig pc;
    setDsv4ExplicitPoolBlocks(mc, "hca_state", 0);
    auto config        = CacheConfigCreator::createBasicConfig(mc, pc, false, 0);
    config.linear_step = 4;

    RuntimeConfig rt;
    config.finalizeBlockNums(/*global_block_num=*/128, rt);

    for (size_t group_id = 0; group_id < static_cast<size_t>(config.groupNums()); ++group_id) {
        const uint32_t expected = config.typeForGroup(group_id) == CacheGroupType::SWA ? 32u : 128u;
        EXPECT_EQ(config.blockNumForGroup(group_id), expected) << "group_id=" << group_id;
    }
    EXPECT_EQ(config.explicitly_sized_pool_reserve_bytes, 0u);
}

TEST_F(HybridPoolKVCacheAllocatorTest, FinalizeNonExplicitSwaBlocksUsesCeilDivision) {
    auto config        = makeTinySwaMultiPoolHybridConfig();
    config.linear_step = 4;
    RuntimeConfig rt;

    config.finalizeBlockNums(/*global_block_num=*/1, rt);
    EXPECT_EQ(config.blockNumForGroup(/*linear group_id=*/0), 1u);
    EXPECT_EQ(config.blockNumForGroup(/*swa group_id=*/1), 1u);

    config.finalizeBlockNums(/*global_block_num=*/8, rt);
    EXPECT_EQ(config.blockNumForGroup(/*linear group_id=*/0), 8u);
    EXPECT_EQ(config.blockNumForGroup(/*swa group_id=*/1), 2u);

    config.finalizeBlockNums(/*global_block_num=*/9, rt);
    EXPECT_EQ(config.blockNumForGroup(/*linear group_id=*/0), 9u);
    EXPECT_EQ(config.blockNumForGroup(/*swa group_id=*/1), 3u);

    config.linear_step = 1;
    config.finalizeBlockNums(/*global_block_num=*/9, rt);
    EXPECT_EQ(config.blockNumForGroup(/*linear group_id=*/0), 9u);
    EXPECT_EQ(config.blockNumForGroup(/*swa group_id=*/1), 9u);
}

TEST_F(HybridPoolKVCacheAllocatorTest, DSV4ConvertIndexToAddrByTagRoutesToCorrectPool) {
    auto config    = makeDSV4HybridPoolConfig();
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

    // CSA layer (compress_ratio=4) -- pick the first one.
    int csa_layer = -1;
    for (size_t l = 0; l < config.layer_all_num; ++l) {
        if (config.layerTagToGroupIdSnapshot()[l].count("csa_kv") > 0) {
            csa_layer = static_cast<int>(l);
            break;
        }
    }
    ASSERT_GE(csa_layer, 0);

    // csa_kv tag routes to group_id=0; it must produce a non-null kv address that
    // matches the CSA group's pool.
    auto addr_csa = allocator->convertIndexToAddrByTag(csa_layer, "csa_kv", 1);
    EXPECT_NE(addr_csa.kv_addr, nullptr);
    const auto csa_group_id = config.groupIdForTag("csa_kv");
    EXPECT_EQ(addr_csa.kv_addr, allocator->convertIndexToAddr(csa_layer, csa_group_id, 1).kv_addr);

    auto addr_swa = allocator->convertIndexToAddrByTag(csa_layer, "swa_kv", 1);
    EXPECT_NE(addr_swa.kv_addr, nullptr);

    // The two tags live in different pools, so their addresses cannot alias.
    EXPECT_NE(addr_csa.kv_addr, addr_swa.kv_addr);
    EXPECT_THROW((void)allocator->convertIndexToAddrByTag(csa_layer, "missing", 1), std::exception);
    EXPECT_THROW((void)allocator->convertIndexToAddr(csa_layer, config.groupNums(), 1), std::exception);

    // Default single-group access is ambiguous for multi-tag layers.
    EXPECT_THROW((void)allocator->convertIndexToAddr(csa_layer, /*block_id=*/1), std::exception);
}

TEST_F(HybridPoolKVCacheAllocatorTest, DSV4ConvertIndexToBufferByTagAndPartition) {
    auto config    = makeDSV4HybridPoolConfig();
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

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

    auto layout = allocator->allLayerCacheBase();
    for (size_t l = 0; l < static_cast<size_t>(config.layer_all_num); ++l) {
        EXPECT_TRUE(layout.group("swa_kv").hasLayer(l)) << "layer " << l << " missing SWA_KV tensor";
    }
    EXPECT_EQ(layout.groups().size(), 7u);
    EXPECT_EQ(layout.topology().groups().size(), 7u);
}

TEST_F(HybridPoolKVCacheAllocatorTest, DSV4CPShardedInsertThenReuseSamePrefix) {
    auto config    = makeDSV4HybridPoolConfig(/*block_num=*/64);
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

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
    seed_malloc.enable_cache_lookup = false;
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
    hit_malloc.enable_cache_lookup = true;
    allocator->setCPSlotMapper(cp_mapper);
    auto result = allocator->malloc(hit_malloc);

    ASSERT_TRUE(result.success);
    // Ten full keys under cp_size=2 subsample to five canonical virtual blocks.
    // The 17-token tail keeps reuse strictly below the query length, so all
    // five complete virtual blocks are reusable.
    EXPECT_EQ(result.reuse_len, 5 * spb * 2);

    FreeInfo hit_free{hit_res, hit_tokens};
    allocator->free(hit_free);
}

TEST_F(HybridPoolKVCacheAllocatorTest, DSV4CPShardedEvictionCascadesFromFullToLowerPriorityGroups) {
    auto config    = makeDSV4HybridPoolConfig(/*block_num=*/64);
    auto allocator = makeAllocator(config);
    ASSERT_TRUE(allocator->init());

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
    seed_malloc.enable_cache_lookup = false;
    ASSERT_TRUE(allocator->malloc(seed_malloc).success);

    InsertInfo insert_info{seed_res, seed_tokens, /*is_resident=*/false};
    allocator->insertIntoCache(insert_info);

    const std::string target_tag      = "csa_kv";
    const int         target_group_id = config.groupIdForTag(target_tag);
    ASSERT_GE(target_group_id, 0);
    ASSERT_LT(static_cast<size_t>(target_group_id), allocator->groupBlockPools().size());

    FreeInfo seed_free{seed_res, seed_tokens};
    allocator->free(seed_free);

    KVCacheResource canonical_source;
    canonical_source.setCacheKeys(full_keys);
    const auto expected_canonical = canonical_source.localCacheKeys(cp_mapper->cpSize() - 1, cp_mapper->cpSize());
    const auto before             = allocator->blockTreeCacheOwner()->getKeySnapshot(expected_canonical.size() + 1);
    EXPECT_EQ(before.keys, expected_canonical);

    const auto& group_sets      = allocator->blockTreeCacheOwner()->groupSets();
    const auto target_group_set = std::find_if(group_sets.begin(), group_sets.end(), [&](const GroupSetPtr& group_set) {
        return group_set != nullptr
               && std::find(
                      group_set->groupIds().begin(), group_set->groupIds().end(), static_cast<size_t>(target_group_id))
                      != group_set->groupIds().end();
    });
    ASSERT_NE(target_group_set, group_sets.end());

    std::vector<size_t> free_before;
    free_before.reserve(allocator->groupBlockPools().size());
    for (const auto& pool : allocator->groupBlockPools()) {
        free_before.push_back(pool->freeBlocksNum());
    }

    // Trigger reclamation through the production pressure entry: demanding more
    // free blocks than this CP rank's physical pool currently has forces
    // KVCacheGroup::ensureFreeBlocks() to run the allocator-registered eviction
    // callback (group-id eviction + task-pool idle wait) and its free
    // recomputation loop, instead of test code calling the cache eviction API directly.
    KVCacheGroupPtr target_group;
    for (const auto& group : allocator->cacheGroups()) {
        if (group != nullptr && group->group_id() == target_group_id) {
            target_group = group;
        }
    }
    ASSERT_NE(target_group, nullptr);
    const size_t target_free_before = free_before[static_cast<size_t>(target_group_id)];
    const size_t required_blocks    = target_free_before + expected_canonical.size();
    ASSERT_TRUE(target_group->ensureFreeBlocks(static_cast<int>(required_blocks)));

    const size_t target_free_after =
        allocator->groupBlockPools()[static_cast<size_t>(target_group_id)]->freeBlocksNum();
    ASSERT_GT(target_free_after, target_free_before);
    const size_t reclaimed = target_free_after - target_free_before;
    EXPECT_EQ(reclaimed, expected_canonical.size());

    std::unordered_set<size_t> reclaimed_group_ids((*target_group_set)->groupIds().begin(),
                                                   (*target_group_set)->groupIds().end());
    for (const auto& group_set : group_sets) {
        if (group_set->groupType() == CacheGroupType::SWA || group_set->groupType() == CacheGroupType::LINEAR) {
            reclaimed_group_ids.insert(group_set->groupIds().begin(), group_set->groupIds().end());
        }
    }
    for (size_t group_id = 0; group_id < allocator->groupBlockPools().size(); ++group_id) {
        const auto& pool = allocator->groupBlockPools()[group_id];
        if (reclaimed_group_ids.find(group_id) != reclaimed_group_ids.end()) {
            // FULL subtree pruning also releases unreachable descendants, so a
            // cascaded group may reclaim more blocks than the triggering pool.
            EXPECT_GE(pool->freeBlocksNum(), free_before[group_id] + reclaimed) << "group_id=" << group_id;
        } else {
            // Other group types can still own resources below the dropped FULL
            // prefix. Those unreachable descendants are pruned as well.
            EXPECT_GE(pool->freeBlocksNum(), free_before[group_id]) << "group_id=" << group_id;
        }
    }
    EXPECT_EQ(allocator->groupBlockPools()[static_cast<size_t>(target_group_id)]->freeBlocksNum(),
              free_before[static_cast<size_t>(target_group_id)] + static_cast<size_t>(reclaimed));
    const auto after_target_reclaim = allocator->blockTreeCacheOwner()->getKeySnapshot(expected_canonical.size() + 1);
    EXPECT_GT(after_target_reclaim.version, before.version);

    // Reverse cascading includes every group set on a tier leaf. The pressure
    // eviction above therefore prunes the complete cached path; there is
    // nothing left for a second explicit reclaim.
    EXPECT_TRUE(after_target_reclaim.keys.empty());
}

}  // namespace test
}  // namespace rtp_llm

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
