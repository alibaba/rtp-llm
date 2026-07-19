#include <gtest/gtest.h>

#include <algorithm>
#include <condition_variable>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <dirent.h>
#include <unistd.h>
#include <utility>
#include <vector>

#include <torch/torch.h>

#include "rtp_llm/cpp/cache/AsyncContext.h"
#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/cache/allocator/HybridTypeKVCacheAllocator.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCacheFactory.h"
#include "rtp_llm/cpp/cache/block_tree_cache/FullComponentGroup.h"
#include "rtp_llm/cpp/cache/block_tree_cache/host/HostBlockPool.h"
#include "rtp_llm/cpp/cache/config_creator/CacheConfigCreator.h"
#include "rtp_llm/cpp/cache/test/BlockPoolTestHelper.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/config/ModelConfig.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"

namespace rtp_llm::block_tree_cache_test {
class BlockTreeCacheTestPeer {
public:
    static void setCopyEngine(BlockTreeCache& cache, CopyEnginePtr copy_engine) {
        cache.copy_engine_ = std::move(copy_engine);
    }

    static bool submitDeviceDemotion(BlockTreeCache&                  cache,
                                     TreeNode*                        node,
                                     int                              component_group_id,
                                     const std::vector<BlockIdxType>& source_blocks) {
        std::lock_guard<std::mutex> lock(cache.mutex_);
        EvictionMove                move;
        move.node               = node;
        move.component_group_id = component_group_id;
        move.source_tier        = Tier::DEVICE;
        move.target_tier        = Tier::HOST;
        move.source_blocks      = source_blocks;
        return cache.submitEvictionLocked(move);
    }
};
}  // namespace rtp_llm::block_tree_cache_test

namespace rtp_llm {
namespace test {

class FailingHybridCopyEngine: public CopyEngine {
public:
    using CopyEngine::CopyEngine;

    TransferHandle submit(const TransferDescriptor&) override {
        ++submit_count_;
        return TransferHandle::completed(CopyStatus::DEVICE_IO_ERROR);
    }

    size_t submitCount() const {
        return submit_count_;
    }

private:
    size_t submit_count_{0};
};

class PausableHybridCopyEngine: public CopyEngine {
public:
    explicit PausableHybridCopyEngine(CopyEnginePtr delegate):
        CopyEngine(std::vector<ComponentGroupPtr>{}, std::vector<Component>{}), delegate_(std::move(delegate)) {}

    TransferHandle submit(const TransferDescriptor& descriptor) override {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            descriptors_.push_back(descriptor);
            if (!paused_once_ && descriptor.source_tier == Tier::DEVICE && descriptor.target_tier == Tier::HOST) {
                paused_once_ = true;
                paused_      = true;
                cv_.notify_all();
                cv_.wait(lock, [this] { return released_; });
            }
        }
        return delegate_->submit(descriptor);
    }

    void waitUntilPaused() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return paused_; });
    }

    void releaseDemotion() {
        std::lock_guard<std::mutex> lock(mutex_);
        released_ = true;
        cv_.notify_all();
    }

    std::vector<TransferDescriptor> descriptors() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return descriptors_;
    }

private:
    CopyEnginePtr                   delegate_;
    mutable std::mutex              mutex_;
    std::condition_variable         cv_;
    bool                            paused_once_{false};
    bool                            paused_{false};
    bool                            released_{false};
    std::vector<TransferDescriptor> descriptors_;
};

class OwnershipHandoffHybridTypeAllocator: public HybridTypeKVCacheAllocator {
public:
    using HybridTypeKVCacheAllocator::HybridTypeKVCacheAllocator;

    void armReferenceGate(int gid) {
        std::lock_guard<std::mutex> lock(gate_mutex_);
        gated_gid_     = gid;
        gate_armed_    = true;
        gate_entered_  = false;
        allow_forward_ = false;
        forwarded_     = false;
        allow_return_  = false;
    }

    void waitUntilReferenceGate() {
        std::unique_lock<std::mutex> lock(gate_mutex_);
        gate_cv_.wait(lock, [this] { return gate_entered_; });
    }

    void allowReferenceForward() {
        std::lock_guard<std::mutex> lock(gate_mutex_);
        allow_forward_ = true;
        gate_cv_.notify_all();
    }

    void waitUntilReferenceForwarded() {
        std::unique_lock<std::mutex> lock(gate_mutex_);
        gate_cv_.wait(lock, [this] { return forwarded_; });
    }

    void releaseReferenceGate() {
        std::lock_guard<std::mutex> lock(gate_mutex_);
        allow_return_ = true;
        gate_cv_.notify_all();
    }

    void armReleaseGate(int gid, const BlockIndicesType& blocks) {
        std::lock_guard<std::mutex> lock(release_gate_mutex_);
        release_gated_gid_    = gid;
        release_gated_blocks_ = blocks;
        release_gate_armed_   = true;
        release_forwarded_    = false;
        allow_release_return_ = false;
    }

    void waitUntilReleaseForwarded() {
        std::unique_lock<std::mutex> lock(release_gate_mutex_);
        release_gate_cv_.wait(lock, [this] { return release_forwarded_; });
    }

    void releaseReleaseGate() {
        std::lock_guard<std::mutex> lock(release_gate_mutex_);
        allow_release_return_ = true;
        release_gate_cv_.notify_all();
    }

private:
    void referenceBlocksInGroup(int gid, const BlockIndicesType& blocks, bool is_connector = false) const override {
        bool gated = false;
        {
            std::unique_lock<std::mutex> lock(gate_mutex_);
            gated = gate_armed_ && gid == gated_gid_;
            if (gated) {
                gate_armed_   = false;
                gate_entered_ = true;
                gate_cv_.notify_all();
                gate_cv_.wait(lock, [this] { return allow_forward_; });
            }
        }

        (void)is_connector;
        group(gid)->reference(blocks);

        if (gated) {
            std::unique_lock<std::mutex> lock(gate_mutex_);
            forwarded_ = true;
            gate_cv_.notify_all();
            gate_cv_.wait(lock, [this] { return allow_return_; });
        }
    }

    void freeBlocksInGroup(int gid, const BlockIndicesType& blocks, bool is_connector = false) override {
        bool gated = false;
        {
            std::lock_guard<std::mutex> lock(release_gate_mutex_);
            gated = release_gate_armed_ && gid == release_gated_gid_ && blocks == release_gated_blocks_;
            if (gated) {
                release_gate_armed_ = false;
            }
        }

        (void)is_connector;
        group(gid)->free(blocks);

        if (gated) {
            std::unique_lock<std::mutex> lock(release_gate_mutex_);
            release_forwarded_ = true;
            release_gate_cv_.notify_all();
            release_gate_cv_.wait(lock, [this] { return allow_release_return_; });
        }
    }

    mutable std::mutex              gate_mutex_;
    mutable std::condition_variable gate_cv_;
    mutable int                     gated_gid_{-1};
    mutable bool                    gate_armed_{false};
    mutable bool                    gate_entered_{false};
    mutable bool                    allow_forward_{false};
    mutable bool                    forwarded_{false};
    mutable bool                    allow_return_{false};

    std::mutex              release_gate_mutex_;
    std::condition_variable release_gate_cv_;
    int                     release_gated_gid_{-1};
    BlockIndicesType        release_gated_blocks_;
    bool                    release_gate_armed_{false};
    bool                    release_forwarded_{false};
    bool                    allow_release_return_{false};
};

static CacheConfig makeTinyHybridConfig() {
    // 4 layers: [0,1] linear, [2,3] full. gcd(2,2)=2 => group_size=2.
    CacheConfig config;
    config.dtype                     = rtp_llm::DataType::TYPE_FP16;
    config.layer_num                 = 4;
    config.layer_all_num             = 4;
    config.block_num                 = 10;
    config.seq_size_per_block        = 4;
    config.kernel_seq_size_per_block = 2;
    config.linear_step               = 2;
    config.group_layer_num           = 2;

    // Linear spec (small but valid).
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

    // Full spec.
    auto full_spec                = std::make_shared<MHAKVCacheSpec>();
    full_spec->type               = KVCacheSpecType::MultiHeadAttention;
    full_spec->dtype              = config.dtype;
    full_spec->local_head_num_kv  = 1;
    full_spec->size_per_head      = 1;
    full_spec->seq_size_per_block = static_cast<uint32_t>(config.seq_size_per_block);

    // Order matters: linear groups first, then full groups (as in CacheConfigCreator).
    config.fromGroupedSpecs(
        {linear_spec, full_spec}, {{0, 1}, {2, 3}}, {CacheGroupType::LINEAR, CacheGroupType::FULL}, {"linear", "full"});

    // Physical block strides: take max between full and linear.
    config.kv_block_stride_bytes = std::max(full_spec->block_size_bytes(), linear_spec->block_size_bytes());
    config.kv_block_size_bytes   = static_cast<size_t>(config.group_layer_num) * config.kv_block_stride_bytes;

    // No kv scale for fp16.
    config.kv_scale_stride_bytes = 0;
    config.kv_scale_size_bytes   = 0;

    config.block_size_bytes             = config.kv_block_size_bytes + config.kv_scale_size_bytes;
    const size_t per_layer_stride_bytes = config.kv_block_stride_bytes + config.kv_scale_stride_bytes;
    config.layer_to_block_stride_bytes.assign(static_cast<size_t>(config.layer_all_num),
                                              static_cast<int>(per_layer_stride_bytes));

    return config;
}

static CacheConfig makeTinyFullSWAConfig() {
    CacheConfig config;
    config.dtype                     = rtp_llm::DataType::TYPE_FP16;
    config.layer_num                 = 4;
    config.layer_all_num             = 4;
    config.block_num                 = 16;
    config.seq_size_per_block        = 4;
    config.kernel_seq_size_per_block = 4;
    config.group_layer_num           = 2;

    auto make_spec = [&]() {
        auto spec                = std::make_shared<MHAKVCacheSpec>();
        spec->type               = KVCacheSpecType::MultiHeadAttention;
        spec->dtype              = config.dtype;
        spec->local_head_num_kv  = 1;
        spec->size_per_head      = 1;
        spec->seq_size_per_block = static_cast<uint32_t>(config.seq_size_per_block);
        return spec;
    };
    auto full_spec = make_spec();
    auto swa_spec  = make_spec();
    config.fromGroupedSpecs(
        {full_spec, swa_spec}, {{0, 1}, {2, 3}}, {CacheGroupType::FULL, CacheGroupType::SWA}, {"full", "swa"});
    auto policies                   = config.groupPoliciesSnapshot();
    policies[1].sliding_window_size = 2 * static_cast<int>(config.seq_size_per_block);
    config.setGroupPolicies(policies);

    config.kv_block_stride_bytes = full_spec->block_size_bytes();
    config.kv_block_size_bytes   = static_cast<size_t>(config.group_layer_num) * config.kv_block_stride_bytes;
    config.kv_scale_stride_bytes = 0;
    config.kv_scale_size_bytes   = 0;
    config.block_size_bytes      = config.kv_block_size_bytes;
    config.layer_to_block_stride_bytes.assign(static_cast<size_t>(config.layer_all_num),
                                              static_cast<int>(config.kv_block_stride_bytes));
    return config;
}

static ModelConfig makeTinyModelConfig(uint32_t num_layers) {
    ModelConfig cfg;
    cfg.num_layers                   = static_cast<int64_t>(num_layers);
    cfg.max_seq_len                  = 128;
    cfg.hidden_size                  = 64;
    cfg.vocab_size                   = 1024;
    cfg.data_type                    = rtp_llm::DataType::TYPE_FP16;
    cfg.attn_config.head_num         = 2;
    cfg.attn_config.kv_head_num      = 2;
    cfg.attn_config.size_per_head    = 16;
    cfg.attn_config.tokens_per_block = 4;
    cfg.attn_config.use_mla          = false;
    cfg.attn_config.kv_cache_dtype   = KvCacheDataType::BASE;
    return cfg;
}

static CacheConfig makeTinyHybridMtpConfigByCreateSpConfig() {
    auto score_model_cfg   = makeTinyModelConfig(/*num_layers=*/4);
    auto propose_model_cfg = makeTinyModelConfig(/*num_layers=*/1);

    score_model_cfg.hybrid_attention_config.enable_hybrid_attention = true;
    score_model_cfg.hybrid_attention_config.hybrid_attention_types  = {
        HybridAttentionType::LINEAR, HybridAttentionType::LINEAR, HybridAttentionType::NONE, HybridAttentionType::NONE};
    score_model_cfg.linear_attention_config.linear_conv_kernel_dim = 2;
    score_model_cfg.linear_attention_config.linear_key_head_dim    = 8;
    score_model_cfg.linear_attention_config.linear_value_head_dim  = 8;
    score_model_cfg.linear_attention_config.linear_num_key_heads   = 2;
    score_model_cfg.linear_attention_config.linear_num_value_heads = 2;
    setHybridAttentionKvCacheSpecs(score_model_cfg);
    // Propose model must use compatible tags so mergeMTPModule can match groups.
    // Use a single "full" layer to match the score model's "full" group.
    propose_model_cfg.hybrid_attention_config.enable_hybrid_attention = true;
    propose_model_cfg.hybrid_attention_config.hybrid_attention_types  = {HybridAttentionType::NONE};
    setHybridAttentionKvCacheSpecs(propose_model_cfg);

    ParallelismConfig parallelism_cfg;
    parallelism_cfg.tp_size = 1;

    RuntimeConfig runtime_cfg;
    KVCacheConfig kv_cache_cfg;
    kv_cache_cfg.test_block_num = 8;

    SpeculativeExecutionConfig sp_cfg;
    sp_cfg.type              = SP_TYPE_MTP;
    sp_cfg.gen_num_per_cycle = 2;

    return CacheConfigCreator::createSpConfig(score_model_cfg,
                                              propose_model_cfg,
                                              parallelism_cfg,
                                              runtime_cfg,
                                              kv_cache_cfg,
                                              sp_cfg,
                                              /*warm_up_result=*/std::nullopt,
                                              /*is_mtp=*/true,
                                              /*is_eagle=*/false);
}

static CompleteTokenIdsPtr makeCompleteTokenIds(int batch_size, int seq_length, int seq_size_per_block) {
    auto complete_token_ids =
        std::make_shared<CompleteTokenIds>(batch_size, batch_size, seq_length + 64, seq_size_per_block);
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

static BatchKVCacheResourcePtr makeBatchResource(int batch_size, const CacheConfig& config, CacheKeysType keys) {
    auto res = std::make_shared<BatchKVCacheResource>();
    res->resetBatchSize(batch_size);
    res->initGroups(config.groupNums(),
                    static_cast<int>(config.layer_all_num),
                    config.layerGroupIdsSnapshot(),
                    config.kernelBlocksPerKvBlock(),
                    config.groupTypesSnapshot());
    for (int b = 0; b < batch_size; ++b) {
        res->setBatchCacheKeys(b, keys);
    }
    return res;
}

struct RealCacheSeed {
    std::vector<BlockIndicesType> group_blocks;
};

static RealCacheSeed seedRealCachePath(const std::shared_ptr<HybridTypeKVCacheAllocator>& allocator,
                                       const CacheConfig&                                 config,
                                       const CacheKeysType&                               cache_keys,
                                       int                                                seq_length) {
    RealCacheSeed seed;
    auto          resource = makeBatchResource(/*batch_size=*/1, config, cache_keys);
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

static size_t countValidBlocks(const BlockIndicesType& blocks) {
    size_t n = 0;
    for (auto b : blocks) {
        if (!isNullBlockIdx(b)) {
            ++n;
        }
    }
    return n;
}

static CompleteTokenIdsPtr makeIncrementTokenIds(int batch_size, int seq_length, int seq_size_per_block) {
    auto  token_ids  = std::make_shared<CompleteTokenIds>(batch_size, batch_size, seq_length + 64, seq_size_per_block);
    auto  input_ids  = torch::empty({static_cast<int64_t>(seq_length)}, torch::kInt32);
    auto* token_data = input_ids.data_ptr<int32_t>();
    for (int i = 0; i < seq_length; ++i) {
        token_data[i] = i + 1;
    }
    auto input             = std::make_shared<GenerateInput>();
    input->input_ids       = input_ids;
    input->generate_config = std::make_shared<GenerateConfig>();
    token_ids->init(input);
    return token_ids;
}

static BatchKVCacheResourcePtr makeIncrementBatchResource(const CacheConfig& config) {
    auto resource = std::make_shared<BatchKVCacheResource>();
    resource->resetBatchSize(/*batch_size=*/1);
    resource->initGroups(config.groupNums(),
                         static_cast<int>(config.layer_all_num),
                         config.layerGroupIdsSnapshot(),
                         config.kernelBlocksPerKvBlock(),
                         config.groupTypesSnapshot());
    resource->setBatchCacheKeys(/*batch_id=*/0, CacheKeysType{100, 101});
    return resource;
}

class HybridTypeKVCacheAllocatorTest: public ::testing::Test {
protected:
    void SetUp() override {
        rtp_llm::initLogger();
        createDevice();
    }

    void TearDown() override {
        allocator_.reset();
        block_tree_cache_.reset();
    }

    bool injectBlockTreeCache(const std::shared_ptr<HybridTypeKVCacheAllocator>& allocator, const CacheConfig& config) {
        KVCacheConfig kv_cache_config;
        auto          block_tree_cache = createBlockTreeCache(config, kv_cache_config, allocator);
        if (!block_tree_cache) {
            return false;
        }
        allocator->setBlockTreeCache(block_tree_cache.get());
        block_tree_cache_ = std::move(block_tree_cache);
        return true;
    }

    bool initWithBlockTreeCache(const CacheConfig& config) {
        allocator_ = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
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

        KVCacheConfig kv_cache_config;
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

    std::shared_ptr<HybridTypeKVCacheAllocator> allocator_;
    BlockTreeCachePtr                           block_tree_cache_;
};

// C005-T01: populated HybridType growth stays synchronous and charges every
// group target at an exact block boundary.
TEST_F(HybridTypeKVCacheAllocatorTest, PopulatedIncrementIsSynchronousAndRestoresCapacity) {
    auto config    = makeTinyHybridConfig();
    auto allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());
    ASSERT_TRUE(injectBlockTreeCache(allocator, config));

    auto         pool             = allocator->getDeviceBlockPool();
    auto         batch_resource   = makeIncrementBatchResource(config);
    auto         complete_tokens  = makeIncrementTokenIds(/*batch_size=*/1, /*seq_length=*/4, /*seq_size_per_block=*/4);
    const size_t free_before = allocator->freeBlocksNum();
    ASSERT_EQ(allocator->activeTreeCachedBlocksNum(), 0u);

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
    EXPECT_NE(linear_initial, full_initial);
    EXPECT_EQ(pool->refCount(linear_initial), 1u);
    EXPECT_EQ(pool->refCount(full_initial), 1u);
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
    EXPECT_NE(linear_appended, full_appended);
    EXPECT_EQ(pool->refCount(linear_initial), 1u);
    EXPECT_EQ(pool->refCount(full_initial), 1u);
    EXPECT_EQ(pool->refCount(linear_appended), 1u);
    EXPECT_EQ(pool->refCount(full_appended), 1u);
    EXPECT_EQ(allocator->freeBlocksNum(), free_before - 4);
    EXPECT_EQ(allocator->activeTreeCachedBlocksNum(), 0u);

    FreeInfo free_info{batch_resource, complete_tokens};
    allocator->free(free_info);
    EXPECT_EQ(batch_resource->curBlocksNum(), 0);
    EXPECT_FALSE(pool->isAllocated(linear_initial));
    EXPECT_FALSE(pool->isAllocated(full_initial));
    EXPECT_FALSE(pool->isAllocated(linear_appended));
    EXPECT_FALSE(pool->isAllocated(full_appended));
    EXPECT_EQ(allocator->freeBlocksNum(), free_before);
    EXPECT_EQ(allocator->activeTreeCachedBlocksNum(), 0u);
}

class HybridTypePreflightTestPeer: public HybridTypeKVCacheAllocator {
public:
    using HybridTypeKVCacheAllocator::HybridTypeKVCacheAllocator;
    using HybridKVCacheAllocator::preflightLoadBackMappings;
};

struct HybridTypePreflightEnvironment {
    BlockTreeCachePtr              cache;
    std::shared_ptr<HostBlockPool> host_pool;
    BlockIdxType                   source_block{NULL_BLOCK_IDX};
};

static HybridTypePreflightEnvironment makeHybridTypePreflightEnvironment(const DeviceBlockPoolPtr& device_pool) {
    HybridTypePreflightEnvironment environment;

    auto host_config                  = std::make_shared<HostBlockPoolConfig>();
    host_config->pool_type            = BlockPoolType::HOST;
    host_config->pool_name            = "hybrid_type_preflight_host";
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
    primary->setDevicePools({device_pool, device_pool});
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

static std::shared_ptr<LoadBackTicket> captureHybridTypePreflightTicket(BlockTreeCache& cache) {
    BlockTreeMatchResult result = cache.match({100});
    EXPECT_NE(result.load_back_ticket, nullptr);
    if (result.load_back_ticket == nullptr) {
        return nullptr;
    }
    EXPECT_EQ(result.load_back_ticket->items().size(), 1u);
    return std::move(result.load_back_ticket);
}

class ScopedHybridTypeDiskCacheDirectory {
public:
    ScopedHybridTypeDiskCacheDirectory() {
        std::string       pattern = "/tmp/rtp_llm_hybrid_type_load_back_XXXXXX";
        std::vector<char> writable(pattern.begin(), pattern.end());
        writable.push_back('\0');
        char* result = ::mkdtemp(writable.data());
        EXPECT_NE(result, nullptr);
        if (result != nullptr) {
            path_ = result;
        }
    }

    ~ScopedHybridTypeDiskCacheDirectory() {
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

static KVCacheConfig makeHybridTypeTieredConfig(Tier source_tier, const std::string& disk_path) {
    KVCacheConfig config;
    config.enable_memory_cache        = true;
    config.enable_tiered_memory_cache = true;
    config.memory_cache_size_mb       = 1;
    config.enable_memory_cache_disk   = source_tier == Tier::DISK;
    config.memory_cache_disk_size_mb  = source_tier == Tier::DISK ? 1 : 0;
    config.memory_cache_disk_paths    = disk_path;
    return config;
}

static CompleteTokenIdsPtr makeHybridTypeLoadBackTokenIds(int seq_length, int seq_size_per_block) {
    auto token_ids = std::make_shared<CompleteTokenIds>(
        /*batch_size=*/1, /*batch_size=*/1, seq_length + 64, seq_size_per_block);
    torch::Tensor input_ids  = torch::empty({static_cast<int64_t>(seq_length)}, torch::kInt32);
    int32_t*      token_data = input_ids.data_ptr<int32_t>();
    for (int index = 0; index < seq_length; ++index) {
        token_data[index] = index + 1;
    }
    auto input             = std::make_shared<GenerateInput>();
    input->input_ids       = input_ids;
    input->generate_config = std::make_shared<GenerateConfig>();
    token_ids->init(input);
    return token_ids;
}

static BatchKVCacheResourcePtr makeHybridTypeLoadBackResource(const CacheConfig& config) {
    auto resource = std::make_shared<BatchKVCacheResource>();
    resource->resetBatchSize(/*batch_size=*/1);
    resource->initGroups(config.groupNums(),
                         static_cast<int>(config.layer_all_num),
                         config.layerGroupIdsSnapshot(),
                         config.kernelBlocksPerKvBlock(),
                         config.groupTypesSnapshot());
    return resource;
}

static void writeHybridDeviceBytes(void* destination, const std::vector<uint8_t>& bytes) {
    auto host = torch::from_blob(const_cast<uint8_t*>(bytes.data()),
                                 {static_cast<int64_t>(bytes.size())},
                                 torch::TensorOptions(torch::kUInt8))
                    .clone();
    auto device = torch::from_blob(
        destination, {static_cast<int64_t>(bytes.size())}, torch::TensorOptions(torch::kUInt8).device(torch::kCUDA));
    CopyParams params{device, host};
    runtimeCopy(params);
    runtimeSyncAndCheck();
}

static std::vector<uint8_t> readHybridDeviceBytes(const void* source, size_t size) {
    auto        device = torch::from_blob(const_cast<void*>(source),
                                          {static_cast<int64_t>(size)},
                                   torch::TensorOptions(torch::kUInt8).device(torch::kCUDA));
    auto        host   = device.cpu();
    const auto* data   = host.data_ptr<uint8_t>();
    return {data, data + size};
}

static void seedHybridTypeLowerTier(BlockTreeCache& cache, Tier source_tier, CacheKeyType key) {
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

TEST_F(HybridTypeKVCacheAllocatorTest, RealHostAndDiskTicketsLoadBackIntoRequestOwnedTargets) {
    for (Tier source_tier : {Tier::HOST, Tier::DISK}) {
        SCOPED_TRACE(source_tier == Tier::HOST ? "HOST" : "DISK");
        ScopedHybridTypeDiskCacheDirectory disk_directory;

        const CacheConfig config    = makeTinyHybridConfig();
        auto              allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
        ASSERT_TRUE(allocator->init());
        BlockTreeCachePtr cache =
            createBlockTreeCache(config, makeHybridTypeTieredConfig(source_tier, disk_directory.string()), allocator);
        ASSERT_NE(cache, nullptr);
        allocator->setBlockTreeCache(cache.get());
        seedHybridTypeLowerTier(*cache, source_tier, /*key=*/100);

        BatchKVCacheResourcePtr resource = makeHybridTypeLoadBackResource(config);
        resource->setBatchCacheKeys(0, CacheKeysType{100, 200});
        CompleteTokenIdsPtr token_ids = makeHybridTypeLoadBackTokenIds(
            /*seq_length=*/8, /*seq_size_per_block=*/static_cast<int>(config.seq_size_per_block));

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
    }
}

TEST_F(HybridTypeKVCacheAllocatorTest, FullSWAReserveRollbackFinalizesRequestBeforeDevicePlanning) {
    const CacheConfig config    = makeTinyFullSWAConfig();
    auto              allocator = std::make_shared<OwnershipHandoffHybridTypeAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());
    BlockTreeCachePtr cache =
        createBlockTreeCache(config, makeHybridTypeTieredConfig(Tier::HOST, /*disk_path=*/""), allocator);
    ASSERT_NE(cache, nullptr);
    allocator->setBlockTreeCache(cache.get());

    const auto& component_groups = cache->componentGroups();
    const auto  full_it =
        std::find_if(component_groups.begin(), component_groups.end(), [](const ComponentGroupPtr& group) {
            return group->group_type == CacheGroupType::FULL;
        });
    const auto swa_it =
        std::find_if(component_groups.begin(), component_groups.end(), [](const ComponentGroupPtr& group) {
            return group->group_type == CacheGroupType::SWA;
        });
    ASSERT_NE(full_it, component_groups.end());
    ASSERT_NE(swa_it, component_groups.end());
    const ComponentGroupPtr full_group = *full_it;
    const ComponentGroupPtr swa_group  = *swa_it;

    const GroupBlockSet full_resident = full_group->allocateBlocks(Tier::DEVICE, 4);
    const GroupBlockSet swa_resident  = swa_group->allocateBlocks(Tier::DEVICE, 2);
    const GroupBlockSet swa_host      = swa_group->allocateBlocks(Tier::HOST, 2);
    ASSERT_EQ(full_resident.per_node.size(), 4u);
    ASSERT_EQ(swa_resident.per_node.size(), 2u);
    ASSERT_EQ(swa_host.per_node.size(), 2u);

    const size_t payload_bytes = config.specForGroup(0)->block_size_bytes();
    for (size_t path_index = 0; path_index < 4; ++path_index) {
        writeHybridDeviceBytes(allocator->convertIndexToAddr(0, full_resident.per_node[path_index][0]).kv_addr,
                               std::vector<uint8_t>(payload_bytes, static_cast<uint8_t>(0x10 + path_index)));
        if (path_index < 2) {
            writeHybridDeviceBytes(allocator->convertIndexToAddr(2, swa_resident.per_node[path_index][0]).kv_addr,
                                   std::vector<uint8_t>(payload_bytes, static_cast<uint8_t>(0x20 + path_index)));
        } else {
            std::memset(swa_group->hostPool()->blockBuffer(swa_host.per_node[path_index - 2][0]).addr,
                        static_cast<int>(0x30 + path_index),
                        swa_group->hostPool()->payloadBytes());
        }
    }

    std::vector<std::vector<GroupSlot>> slots(4, std::vector<GroupSlot>(component_groups.size()));
    for (size_t path_index = 0; path_index < 4; ++path_index) {
        slots[path_index][static_cast<size_t>(full_group->component_group_id)].device_blocks =
            full_resident.per_node[path_index];
        if (path_index < 2) {
            slots[path_index][static_cast<size_t>(swa_group->component_group_id)].device_blocks =
                swa_resident.per_node[path_index];
        } else {
            slots[path_index][static_cast<size_t>(swa_group->component_group_id)].host_block =
                swa_host.per_node[path_index - 2][0];
        }
    }
    ASSERT_NE(cache->tree()->insertNode(nullptr, {100, 200, 300, 400}, slots).leaf, nullptr);

    BatchKVCacheResourcePtr resource = makeHybridTypeLoadBackResource(config);
    resource->setBatchCacheKeys(0, CacheKeysType{100, 200, 300, 400, 500});
    CompleteTokenIdsPtr token_ids = makeHybridTypeLoadBackTokenIds(
        /*seq_length=*/20, /*seq_size_per_block=*/static_cast<int>(config.seq_size_per_block));
    auto device_pool = allocator->getDeviceBlockPool();
    ASSERT_NE(device_pool, nullptr);
    const size_t free_before = allocator->freeBlocksNum();
    ASSERT_GT(free_before, 4u);

    const BlockIndicesType final_request_release = {
        swa_resident.per_node[0][0],
        swa_resident.per_node[1][0],
    };
    const BlockIndicesType ticket_owned_suffix = {
        full_resident.per_node[2][0],
        full_resident.per_node[3][0],
    };
    allocator->setReserveBlockNum(free_before - 3u);
    allocator->armReferenceGate(/*gid=*/0);
    allocator->armReleaseGate(/*gid=*/1, final_request_release);

    MallocInfo malloc_info{resource, token_ids};
    malloc_info.enable_device_cache = true;
    malloc_info.verbose             = false;
    MallocResult result{false, 0};
    std::thread  malloc_thread([&] { result = allocator->malloc(malloc_info); });
    allocator->waitUntilReferenceGate();
    allocator->allowReferenceForward();
    allocator->waitUntilReferenceForwarded();
    allocator->releaseReferenceGate();
    allocator->waitUntilReleaseForwarded();

    EXPECT_EQ(allocator->activeTreeCachedBlocksNum(), ticket_owned_suffix.size());
    for (BlockIdxType block : final_request_release) {
        EXPECT_EQ(device_pool->refCount(block), 1u);
    }
    for (size_t suffix_index = 0; suffix_index < ticket_owned_suffix.size(); ++suffix_index) {
        const BlockIdxType block = ticket_owned_suffix[suffix_index];
        EXPECT_TRUE(device_pool->isAllocated(block));
        EXPECT_EQ(device_pool->refCount(block), 2u);
        EXPECT_EQ(readHybridDeviceBytes(allocator->convertIndexToAddr(0, block).kv_addr, payload_bytes),
                  std::vector<uint8_t>(payload_bytes, static_cast<uint8_t>(0x12 + suffix_index)));
    }
    EXPECT_GT(allocator->activeTreeCachedBlocksNum(), 0u);
    EXPECT_EQ(cache->evictForGroup(full_group->component_group_id, 1), 0);

    allocator->releaseReleaseGate();
    malloc_thread.join();
    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.reuse_len, 0);
    EXPECT_EQ(result.async_context, nullptr);
    EXPECT_EQ(resource->curBlocksNum(), 0u);
    EXPECT_EQ(resource->cacheResource(0).deviceReuseBlockNum(), 0u);
    EXPECT_EQ(allocator->activeTreeCachedBlocksNum(), 0u);
    EXPECT_EQ(allocator->freeBlocksNum(), free_before);
    EXPECT_EQ(allocator->activeTreeCachedBlocksNum(), 0u);

    EXPECT_EQ(cache->evictForGroup(full_group->component_group_id, 4), 4);
    for (BlockIdxType block : ticket_owned_suffix) {
        EXPECT_FALSE(device_pool->isAllocated(block));
    }
    EXPECT_EQ(allocator->freeBlocksNum(), allocator->totalBlocksNum());

    allocator->setReserveBlockNum(0);
    while (cache->reclaimBlocks(1, Tier::DEVICE) > 0) {}
    while (cache->reclaimBlocks(1, Tier::HOST) > 0) {}
    cache->waitForPendingTasks();
    EXPECT_EQ(cache->getStats().tree_node_count, 0u);
    EXPECT_EQ(allocator->freeBlocksNum(), allocator->totalBlocksNum());
}

TEST_F(HybridTypeKVCacheAllocatorTest, FullSWAAsyncFailureFinalizesPlanningBeforeNormalRequestFree) {
    const CacheConfig config    = makeTinyFullSWAConfig();
    auto              allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());
    BlockTreeCachePtr cache =
        createBlockTreeCache(config, makeHybridTypeTieredConfig(Tier::HOST, /*disk_path=*/""), allocator);
    ASSERT_NE(cache, nullptr);
    allocator->setBlockTreeCache(cache.get());

    const auto& component_groups = cache->componentGroups();
    const auto  full_it =
        std::find_if(component_groups.begin(), component_groups.end(), [](const ComponentGroupPtr& group) {
            return group->group_type == CacheGroupType::FULL;
        });
    const auto swa_it =
        std::find_if(component_groups.begin(), component_groups.end(), [](const ComponentGroupPtr& group) {
            return group->group_type == CacheGroupType::SWA;
        });
    ASSERT_NE(full_it, component_groups.end());
    ASSERT_NE(swa_it, component_groups.end());
    const ComponentGroupPtr full_group = *full_it;
    const ComponentGroupPtr swa_group  = *swa_it;

    const GroupBlockSet full_resident = full_group->allocateBlocks(Tier::DEVICE, 4);
    const GroupBlockSet swa_resident  = swa_group->allocateBlocks(Tier::DEVICE, 2);
    const GroupBlockSet swa_host      = swa_group->allocateBlocks(Tier::HOST, 2);
    ASSERT_EQ(full_resident.per_node.size(), 4u);
    ASSERT_EQ(swa_resident.per_node.size(), 2u);
    ASSERT_EQ(swa_host.per_node.size(), 2u);

    const size_t payload_bytes = config.specForGroup(0)->block_size_bytes();
    for (size_t path_index = 0; path_index < 4; ++path_index) {
        writeHybridDeviceBytes(allocator->convertIndexToAddr(0, full_resident.per_node[path_index][0]).kv_addr,
                               std::vector<uint8_t>(payload_bytes, static_cast<uint8_t>(0x10 + path_index)));
        if (path_index < 2) {
            writeHybridDeviceBytes(allocator->convertIndexToAddr(2, swa_resident.per_node[path_index][0]).kv_addr,
                                   std::vector<uint8_t>(payload_bytes, static_cast<uint8_t>(0x20 + path_index)));
        } else {
            std::memset(swa_group->hostPool()->blockBuffer(swa_host.per_node[path_index - 2][0]).addr,
                        static_cast<int>(0x30 + path_index),
                        swa_group->hostPool()->payloadBytes());
        }
    }

    std::vector<std::vector<GroupSlot>> slots(4, std::vector<GroupSlot>(component_groups.size()));
    for (size_t path_index = 0; path_index < 4; ++path_index) {
        slots[path_index][static_cast<size_t>(full_group->component_group_id)].device_blocks =
            full_resident.per_node[path_index];
        if (path_index < 2) {
            slots[path_index][static_cast<size_t>(swa_group->component_group_id)].device_blocks =
                swa_resident.per_node[path_index];
        } else {
            slots[path_index][static_cast<size_t>(swa_group->component_group_id)].host_block =
                swa_host.per_node[path_index - 2][0];
        }
    }
    ASSERT_NE(cache->tree()->insertNode(nullptr, {100, 200, 300, 400}, slots).leaf, nullptr);

    const auto failing_copy_engine =
        std::make_shared<FailingHybridCopyEngine>(cache->componentGroups(), cache->components());
    block_tree_cache_test::BlockTreeCacheTestPeer::setCopyEngine(*cache, failing_copy_engine);
    BatchKVCacheResourcePtr resource = makeHybridTypeLoadBackResource(config);
    resource->setBatchCacheKeys(0, CacheKeysType{100, 200, 300, 400, 500});
    CompleteTokenIdsPtr token_ids = makeHybridTypeLoadBackTokenIds(
        /*seq_length=*/20, /*seq_size_per_block=*/static_cast<int>(config.seq_size_per_block));
    auto device_pool = allocator->getDeviceBlockPool();
    ASSERT_NE(device_pool, nullptr);
    const size_t free_before = allocator->freeBlocksNum();

    MallocInfo malloc_info{resource, token_ids};
    malloc_info.enable_device_cache = true;
    MallocResult result             = allocator->malloc(malloc_info);
    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.reuse_len, 16);
    ASSERT_NE(result.async_context, nullptr);
    result.async_context->waitDone();
    ASSERT_TRUE(result.async_context->done());
    EXPECT_FALSE(result.async_context->success());
    EXPECT_GT(failing_copy_engine->submitCount(), 0u);

    const BlockIndicesType ticket_owned_suffix = {
        full_resident.per_node[2][0],
        full_resident.per_node[3][0],
    };
    for (size_t suffix_index = 0; suffix_index < ticket_owned_suffix.size(); ++suffix_index) {
        const BlockIdxType block = ticket_owned_suffix[suffix_index];
        EXPECT_TRUE(device_pool->isAllocated(block));
        EXPECT_EQ(device_pool->refCount(block), 2u);
        EXPECT_EQ(readHybridDeviceBytes(allocator->convertIndexToAddr(0, block).kv_addr, payload_bytes),
                  std::vector<uint8_t>(payload_bytes, static_cast<uint8_t>(0x12 + suffix_index)));
    }
    EXPECT_GT(allocator->activeTreeCachedBlocksNum(), 0u);
    EXPECT_EQ(cache->evictForGroup(full_group->component_group_id, 1), 0);

    allocator->free(FreeInfo{resource, token_ids});
    EXPECT_EQ(allocator->activeTreeCachedBlocksNum(), 0u);
    EXPECT_EQ(allocator->freeBlocksNum(), free_before);
    EXPECT_EQ(allocator->activeTreeCachedBlocksNum(), 0u);
    EXPECT_EQ(cache->evictForGroup(full_group->component_group_id, 4), 4);
    for (BlockIdxType block : ticket_owned_suffix) {
        EXPECT_FALSE(device_pool->isAllocated(block));
    }

    const size_t free_after_reclaim   = allocator->freeBlocksNum();
    const size_t active_after_reclaim = allocator->activeTreeCachedBlocksNum();
    result.async_context->waitDone();
    EXPECT_FALSE(result.async_context->success());
    result.async_context.reset();
    EXPECT_EQ(allocator->freeBlocksNum(), free_after_reclaim);
    EXPECT_EQ(allocator->activeTreeCachedBlocksNum(), active_after_reclaim);

    while (cache->reclaimBlocks(1, Tier::DEVICE) > 0) {}
    while (cache->reclaimBlocks(1, Tier::HOST) > 0) {}
    cache->waitForPendingTasks();
    EXPECT_EQ(cache->getStats().tree_node_count, 0u);
    EXPECT_EQ(allocator->freeBlocksNum(), allocator->totalBlocksNum());
}

TEST_F(HybridTypeKVCacheAllocatorTest, FullSWAPartialReadyPreservesResidentSuffixAndChargesLowerTargets) {
    const CacheConfig config    = makeTinyFullSWAConfig();
    auto              allocator = std::make_shared<OwnershipHandoffHybridTypeAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());
    BlockTreeCachePtr cache =
        createBlockTreeCache(config, makeHybridTypeTieredConfig(Tier::HOST, /*disk_path=*/""), allocator);
    ASSERT_NE(cache, nullptr);
    allocator->setBlockTreeCache(cache.get());
    ASSERT_EQ(config.typeForGroup(0), CacheGroupType::FULL);
    ASSERT_EQ(config.typeForGroup(1), CacheGroupType::SWA);

    const auto& component_groups = cache->componentGroups();
    const auto  full_it =
        std::find_if(component_groups.begin(), component_groups.end(), [](const ComponentGroupPtr& group) {
            return group->group_type == CacheGroupType::FULL;
        });
    const auto swa_it =
        std::find_if(component_groups.begin(), component_groups.end(), [](const ComponentGroupPtr& group) {
            return group->group_type == CacheGroupType::SWA;
        });
    ASSERT_NE(full_it, component_groups.end());
    ASSERT_NE(swa_it, component_groups.end());
    const ComponentGroupPtr full_group = *full_it;
    const ComponentGroupPtr swa_group  = *swa_it;

    GroupBlockSet full_resident = full_group->allocateBlocks(Tier::DEVICE, 4);
    GroupBlockSet swa_resident  = swa_group->allocateBlocks(Tier::DEVICE, 2);
    GroupBlockSet swa_host      = swa_group->allocateBlocks(Tier::HOST, 2);
    ASSERT_EQ(full_resident.per_node.size(), 4u);
    ASSERT_EQ(swa_resident.per_node.size(), 2u);
    ASSERT_EQ(swa_host.per_node.size(), 2u);

    const size_t payload_bytes = config.specForGroup(0)->block_size_bytes();
    for (size_t path_index = 0; path_index < 4; ++path_index) {
        writeHybridDeviceBytes(allocator->convertIndexToAddr(0, full_resident.per_node[path_index][0]).kv_addr,
                               std::vector<uint8_t>(payload_bytes, static_cast<uint8_t>(0x10 + path_index)));
        if (path_index < 2) {
            writeHybridDeviceBytes(allocator->convertIndexToAddr(2, swa_resident.per_node[path_index][0]).kv_addr,
                                   std::vector<uint8_t>(payload_bytes, static_cast<uint8_t>(0x20 + path_index)));
        } else {
            std::memset(swa_group->hostPool()->blockBuffer(swa_host.per_node[path_index - 2][0]).addr,
                        static_cast<int>(0x30 + path_index),
                        swa_group->hostPool()->payloadBytes());
        }
    }

    std::vector<std::vector<GroupSlot>> slots(4, std::vector<GroupSlot>(component_groups.size()));
    for (size_t path_index = 0; path_index < 4; ++path_index) {
        slots[path_index][static_cast<size_t>(full_group->component_group_id)].device_blocks =
            full_resident.per_node[path_index];
        if (path_index < 2) {
            slots[path_index][static_cast<size_t>(swa_group->component_group_id)].device_blocks =
                swa_resident.per_node[path_index];
        } else {
            slots[path_index][static_cast<size_t>(swa_group->component_group_id)].host_block =
                swa_host.per_node[path_index - 2][0];
        }
    }
    ASSERT_NE(cache->tree()->insertNode(nullptr, {100, 200, 300, 400}, slots).leaf, nullptr);

    auto make_resource = [&]() {
        BatchKVCacheResourcePtr resource = makeHybridTypeLoadBackResource(config);
        resource->setBatchCacheKeys(0, CacheKeysType{100, 200, 300, 400, 500});
        return resource;
    };
    CompleteTokenIdsPtr token_ids = makeHybridTypeLoadBackTokenIds(
        /*seq_length=*/20, /*seq_size_per_block=*/static_cast<int>(config.seq_size_per_block));
    auto device_pool = allocator->getDeviceBlockPool();
    ASSERT_NE(device_pool, nullptr);
    const size_t free_before = allocator->freeBlocksNum();
    ASSERT_GT(free_before, 4u);

    allocator->setReserveBlockNum(free_before - 3u);
    BatchKVCacheResourcePtr rejected_resource = make_resource();
    MallocInfo              rejected_info{rejected_resource, token_ids};
    rejected_info.enable_device_cache = true;
    rejected_info.verbose             = false;
    allocator->armReferenceGate(/*gid=*/0);
    MallocResult rejected{false, 0};
    std::thread  rejected_thread([&] { rejected = allocator->malloc(rejected_info); });
    allocator->waitUntilReferenceGate();

    const BlockIdxType protected_ready_block = full_resident.per_node.front().front();
    EXPECT_TRUE(device_pool->isAllocated(protected_ready_block));
    EXPECT_EQ(device_pool->refCount(protected_ready_block), 2u);
    EXPECT_EQ(readHybridDeviceBytes(allocator->convertIndexToAddr(0, protected_ready_block).kv_addr, payload_bytes),
              std::vector<uint8_t>(payload_bytes, 0x10));
    EXPECT_EQ(cache->reclaimBlocks(1, Tier::DEVICE), 0);
    EXPECT_TRUE(device_pool->isAllocated(protected_ready_block));

    allocator->allowReferenceForward();
    allocator->waitUntilReferenceForwarded();
    EXPECT_EQ(device_pool->refCount(protected_ready_block), 3u);
    EXPECT_EQ(cache->reclaimBlocks(1, Tier::DEVICE), 0);
    EXPECT_TRUE(device_pool->isAllocated(protected_ready_block));
    allocator->releaseReferenceGate();
    rejected_thread.join();

    EXPECT_FALSE(rejected.success);
    EXPECT_EQ(rejected_resource->curBlocksNum(), 0u);
    EXPECT_EQ(rejected_resource->cacheResource(0).deviceReuseBlockNum(), 0u);
    EXPECT_EQ(allocator->freeBlocksNum(), free_before);
    EXPECT_EQ(allocator->activeTreeCachedBlocksNum(), 0u);
    EXPECT_EQ(allocator->activeTreeCachedBlocksNum(), 0u);
    for (const auto& blocks : full_resident.per_node) {
        EXPECT_EQ(device_pool->refCount(blocks[0]), 1u);
    }
    for (const auto& blocks : swa_resident.per_node) {
        EXPECT_EQ(device_pool->refCount(blocks[0]), 1u);
    }
    for (const auto& blocks : swa_host.per_node) {
        EXPECT_EQ(swa_group->hostPool()->refCount(blocks[0]), 1u);
    }

    allocator->setReserveBlockNum(0);
    CopyEnginePtr real_copy_engine = cache->copyEngine();
    ASSERT_NE(real_copy_engine, nullptr);
    auto failing_copy_engine = std::make_shared<FailingHybridCopyEngine>(cache->componentGroups(), cache->components());
    block_tree_cache_test::BlockTreeCacheTestPeer::setCopyEngine(*cache, failing_copy_engine);
    BatchKVCacheResourcePtr failed_resource = make_resource();
    MallocInfo              failed_info{failed_resource, token_ids};
    failed_info.enable_device_cache = true;
    MallocResult failed_result      = allocator->malloc(failed_info);
    ASSERT_TRUE(failed_result.success);
    ASSERT_NE(failed_result.async_context, nullptr);
    failed_result.async_context->waitDone();
    EXPECT_FALSE(failed_result.async_context->success());
    EXPECT_GT(failing_copy_engine->submitCount(), 0u);
    ASSERT_EQ(failed_resource->blocks(0, 0).size(), 5u);
    ASSERT_EQ(failed_resource->blocks(0, 1).size(), 5u);
    for (size_t path_index = 0; path_index < 4; ++path_index) {
        EXPECT_EQ(failed_resource->blocks(0, 0)[path_index], full_resident.per_node[path_index][0]);
    }
    EXPECT_EQ(failed_resource->blocks(0, 1)[0], swa_resident.per_node[0][0]);
    EXPECT_EQ(failed_resource->blocks(0, 1)[1], swa_resident.per_node[1][0]);
    allocator->free(FreeInfo{failed_resource, token_ids});
    EXPECT_EQ(allocator->freeBlocksNum(), free_before);
    EXPECT_EQ(allocator->activeTreeCachedBlocksNum(), 0u);
    EXPECT_EQ(allocator->activeTreeCachedBlocksNum(), 0u);
    for (const auto& blocks : full_resident.per_node) {
        EXPECT_EQ(device_pool->refCount(blocks[0]), 1u);
    }
    for (const auto& blocks : swa_resident.per_node) {
        EXPECT_EQ(device_pool->refCount(blocks[0]), 1u);
    }
    for (const auto& blocks : swa_host.per_node) {
        EXPECT_EQ(swa_group->hostPool()->refCount(blocks[0]), 1u);
    }
    block_tree_cache_test::BlockTreeCacheTestPeer::setCopyEngine(*cache, real_copy_engine);

    BlockTreeFindResult demoting_path = cache->tree()->findNode({100, 200, 300, 400});
    ASSERT_EQ(demoting_path.path.size(), 4u);
    TreeNode* const        demoting_node           = demoting_path.path[3];
    const int              full_component_group_id = full_group->component_group_id;
    const BlockIndicesType demoting_source         = full_resident.per_node[3];
    auto                   pausable_copy_engine    = std::make_shared<PausableHybridCopyEngine>(real_copy_engine);
    block_tree_cache_test::BlockTreeCacheTestPeer::setCopyEngine(*cache, pausable_copy_engine);
    ASSERT_TRUE(block_tree_cache_test::BlockTreeCacheTestPeer::submitDeviceDemotion(
        *cache, demoting_node, full_component_group_id, demoting_source));
    pausable_copy_engine->waitUntilPaused();
    ASSERT_EQ(demoting_node->group_slots[static_cast<size_t>(full_component_group_id)].transfer_state,
              SlotTransferState::DEMOTING);
    ASSERT_EQ(demoting_node->group_slots[static_cast<size_t>(full_component_group_id)].device_blocks, demoting_source);

    allocator->setReserveBlockNum(free_before - 4u);
    BatchKVCacheResourcePtr resource = make_resource();
    MallocInfo              malloc_info{resource, token_ids};
    malloc_info.enable_device_cache = true;
    malloc_info.verbose             = false;
    MallocResult result             = allocator->malloc(malloc_info);
    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.reuse_len, 16);
    EXPECT_EQ(resource->cacheResource(0).deviceReuseBlockNum(), 4u);
    EXPECT_EQ(allocator->freeBlocksNum(), free_before - 4u);
    EXPECT_GT(allocator->activeTreeCachedBlocksNum(), 0u);
    ASSERT_NE(result.async_context, nullptr);
    result.async_context->waitDone();
    ASSERT_TRUE(result.async_context->success()) << result.async_context->errorInfo().ToString();
    EXPECT_EQ(demoting_node->group_slots[static_cast<size_t>(full_component_group_id)].transfer_state,
              SlotTransferState::DEMOTING);
    EXPECT_EQ(demoting_node->group_slots[static_cast<size_t>(full_component_group_id)].device_blocks, demoting_source);
    EXPECT_EQ(resource->blocks(0, 0)[3], demoting_source.front());
    EXPECT_EQ(device_pool->refCount(demoting_source.front()), 2u);
    EXPECT_EQ(readHybridDeviceBytes(allocator->convertIndexToAddr(0, demoting_source.front()).kv_addr, payload_bytes),
              std::vector<uint8_t>(payload_bytes, 0x13));

    const std::vector<TransferDescriptor> descriptors_while_demoting = pausable_copy_engine->descriptors();
    EXPECT_EQ(std::count_if(descriptors_while_demoting.begin(),
                            descriptors_while_demoting.end(),
                            [](const TransferDescriptor& descriptor) {
                                return descriptor.source_tier == Tier::DEVICE && descriptor.target_tier == Tier::HOST;
                            }),
              1);
    EXPECT_EQ(std::count_if(descriptors_while_demoting.begin(),
                            descriptors_while_demoting.end(),
                            [](const TransferDescriptor& descriptor) {
                                return descriptor.source_tier == Tier::HOST && descriptor.target_tier == Tier::DEVICE;
                            }),
              2);

    pausable_copy_engine->releaseDemotion();
    cache->waitForPendingTasks();
    const GroupSlot& demoted_full_slot = demoting_node->group_slots[static_cast<size_t>(full_component_group_id)];
    EXPECT_EQ(demoted_full_slot.transfer_state, SlotTransferState::IDLE);
    EXPECT_FALSE(demoted_full_slot.has_value(Tier::DEVICE));
    EXPECT_EQ(demoted_full_slot.device_blocks.size(), demoting_source.size());
    EXPECT_TRUE(std::all_of(demoted_full_slot.device_blocks.begin(),
                            demoted_full_slot.device_blocks.end(),
                            [](BlockIdxType block) { return isNullBlockIdx(block); }));
    EXPECT_NE(demoted_full_slot.host_block, NULL_BLOCK_IDX);
    EXPECT_EQ(device_pool->refCount(demoting_source.front()), 1u);
    result.async_context->waitDone();
    EXPECT_TRUE(result.async_context->success());

    ASSERT_EQ(resource->blocks(0, 0).size(), 5u);
    ASSERT_EQ(resource->blocks(0, 1).size(), 5u);
    for (size_t path_index = 0; path_index < 4; ++path_index) {
        EXPECT_EQ(resource->blocks(0, 0)[path_index], full_resident.per_node[path_index][0]);
        EXPECT_EQ(readHybridDeviceBytes(allocator->convertIndexToAddr(0, resource->blocks(0, 0)[path_index]).kv_addr,
                                        payload_bytes),
                  std::vector<uint8_t>(payload_bytes, static_cast<uint8_t>(0x10 + path_index)));
    }
    for (size_t path_index = 0; path_index < 2; ++path_index) {
        EXPECT_EQ(resource->blocks(0, 1)[path_index], swa_resident.per_node[path_index][0]);
        EXPECT_EQ(readHybridDeviceBytes(allocator->convertIndexToAddr(2, resource->blocks(0, 1)[path_index]).kv_addr,
                                        payload_bytes),
                  std::vector<uint8_t>(payload_bytes, static_cast<uint8_t>(0x20 + path_index)));
    }
    for (size_t path_index = 2; path_index < 4; ++path_index) {
        EXPECT_EQ(readHybridDeviceBytes(allocator->convertIndexToAddr(2, resource->blocks(0, 1)[path_index]).kv_addr,
                                        payload_bytes),
                  std::vector<uint8_t>(payload_bytes, static_cast<uint8_t>(0x30 + path_index)));
    }

    BlockTreeFindResult settled = cache->tree()->findNode({100, 200, 300, 400});
    ASSERT_EQ(settled.path.size(), 4u);
    for (size_t path_index = 0; path_index < 4; ++path_index) {
        const GroupSlot& full_slot =
            settled.path[path_index]->group_slots[static_cast<size_t>(full_group->component_group_id)];
        const GroupSlot& swa_slot =
            settled.path[path_index]->group_slots[static_cast<size_t>(swa_group->component_group_id)];
        if (path_index == 3) {
            EXPECT_FALSE(full_slot.has_value(Tier::DEVICE));
            EXPECT_EQ(full_slot.device_blocks.size(), demoting_source.size());
            EXPECT_TRUE(std::all_of(full_slot.device_blocks.begin(),
                                    full_slot.device_blocks.end(),
                                    [](BlockIdxType block) { return isNullBlockIdx(block); }));
            EXPECT_EQ(resource->blocks(0, 0)[path_index], full_resident.per_node[path_index][0]);
        } else {
            EXPECT_EQ(full_slot.device_blocks[0], full_resident.per_node[path_index][0]);
            EXPECT_EQ(device_pool->refCount(full_slot.device_blocks[0]), 2u);
        }
        EXPECT_EQ(swa_slot.device_blocks[0], resource->blocks(0, 1)[path_index]);
        EXPECT_EQ(device_pool->refCount(swa_slot.device_blocks[0]), 2u);
    }
    for (const auto& blocks : swa_host.per_node) {
        EXPECT_FALSE(swa_group->hostPool()->isAllocated(blocks[0]));
    }

    allocator->free(FreeInfo{resource, token_ids});
    EXPECT_EQ(allocator->activeTreeCachedBlocksNum(), 0u);
    allocator->setReserveBlockNum(0);
    while (cache->reclaimBlocks(1, Tier::DEVICE) > 0) {}
    while (cache->reclaimBlocks(1, Tier::HOST) > 0) {}
    cache->waitForPendingTasks();
    EXPECT_EQ(cache->getStats().tree_node_count, 0u);
    EXPECT_EQ(allocator->freeBlocksNum(), allocator->totalBlocksNum());
}

TEST_F(HybridTypeKVCacheAllocatorTest, AllLowerTierTargetsAreChargedBeforeSharedPoolReserveAdmission) {
    const CacheConfig config    = makeTinyHybridConfig();
    auto              allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());
    BlockTreeCachePtr cache =
        createBlockTreeCache(config, makeHybridTypeTieredConfig(Tier::HOST, /*disk_path=*/""), allocator);
    ASSERT_NE(cache, nullptr);
    allocator->setBlockTreeCache(cache.get());
    seedHybridTypeLowerTier(*cache, Tier::HOST, /*key=*/100);

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

    const size_t free_before = allocator->freeBlocksNum();
    ASSERT_GT(free_before, 3u);
    ASSERT_EQ(allocator->activeTreeCachedBlocksNum(), 0u);

    // seq_len=8 needs two FULL slots and two LINEAR slots. A one-block lower-tier
    // match contributes no resident device index. Current FULL-only accounting sees
    // two FULL plus one LINEAR block and exactly admits this reserve boundary, but
    // materialization also allocates the lower-tier-only LINEAR target. Charging all
    // four physical allocations must reject before consuming a reserved block.
    allocator->setReserveBlockNum(free_before - 3u);

    BatchKVCacheResourcePtr resource = makeHybridTypeLoadBackResource(config);
    resource->setBatchCacheKeys(0, CacheKeysType{100, 200});
    CompleteTokenIdsPtr token_ids = makeHybridTypeLoadBackTokenIds(
        /*seq_length=*/8, /*seq_size_per_block=*/static_cast<int>(config.seq_size_per_block));

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
    EXPECT_EQ(allocator->freeBlocksNum(), free_before);
    EXPECT_EQ(allocator->activeTreeCachedBlocksNum(), 0u);

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

TEST_F(HybridTypeKVCacheAllocatorTest, FullySatisfiedLowerTierTargetsStillRespectSharedPoolReserve) {
    const CacheConfig config    = makeTinyHybridConfig();
    auto              allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());
    BlockTreeCachePtr cache =
        createBlockTreeCache(config, makeHybridTypeTieredConfig(Tier::HOST, /*disk_path=*/""), allocator);
    ASSERT_NE(cache, nullptr);
    allocator->setBlockTreeCache(cache.get());
    seedHybridTypeLowerTier(*cache, Tier::HOST, /*key=*/100);

    BlockTreeFindResult source_find = cache->tree()->findNode({100});
    ASSERT_NE(source_find.matched_node, nullptr);
    ASSERT_EQ(source_find.matched_node->group_slots.size(), cache->componentGroups().size());
    std::vector<size_t>       source_ref_baselines;
    std::vector<BlockIdxType> source_blocks;
    source_ref_baselines.reserve(cache->componentGroups().size());
    source_blocks.reserve(cache->componentGroups().size());
    for (size_t group_id = 0; group_id < cache->componentGroups().size(); ++group_id) {
        const auto& group = cache->componentGroups()[group_id];
        ASSERT_NE(group->hostPool(), nullptr) << "component_group_id=" << group_id;
        const GroupSlot& source_slot = source_find.matched_node->group_slots[group_id];
        ASSERT_NE(source_slot.host_block, NULL_BLOCK_IDX) << "component_group_id=" << group_id;
        EXPECT_TRUE(source_slot.device_blocks.empty()) << "component_group_id=" << group_id;
        source_ref_baselines.push_back(group->hostPool()->refCount(source_slot.host_block));
        source_blocks.push_back(source_slot.host_block);
    }

    const size_t free_before = allocator->freeBlocksNum();
    ASSERT_GT(free_before, 1u);
    ASSERT_EQ(allocator->activeTreeCachedBlocksNum(), 0u);

    // seq_len=4 has one physical position in each group. The HOST match can
    // materialize both positions and make ordinary remaining demand zero, but
    // those two pending targets must still be admitted against stable capacity.
    allocator->setReserveBlockNum(free_before - 1u);

    BatchKVCacheResourcePtr resource = makeHybridTypeLoadBackResource(config);
    resource->setBatchCacheKeys(0, CacheKeysType{100, 200});
    CompleteTokenIdsPtr token_ids = makeHybridTypeLoadBackTokenIds(
        /*seq_length=*/4, /*seq_size_per_block=*/static_cast<int>(config.seq_size_per_block));

    MallocInfo malloc_info{resource, token_ids};
    malloc_info.enable_device_cache = true;
    malloc_info.reuse_cache         = true;
    malloc_info.verbose             = false;
    MallocResult result             = allocator->malloc(malloc_info);

    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.async_context, nullptr);
    EXPECT_EQ(resource->curBlocksNum(), 0u);
    for (int gid = 0; gid < config.groupNums(); ++gid) {
        EXPECT_EQ(resource->blocksNum(0, gid), 0u) << "gid=" << gid;
    }
    EXPECT_EQ(resource->cacheResource(0).deviceReuseBlockNum(), 0u);
    EXPECT_EQ(resource->cacheKeys(0), (CacheKeysType{100, 200}));
    EXPECT_EQ(allocator->activeTreeCachedBlocksNum(), 0u);
    EXPECT_EQ(allocator->freeBlocksNum(), free_before);
    EXPECT_EQ(allocator->activeTreeCachedBlocksNum(), 0u);

    source_find = cache->tree()->findNode({100});
    ASSERT_NE(source_find.matched_node, nullptr);
    for (size_t group_id = 0; group_id < cache->componentGroups().size(); ++group_id) {
        const auto&        group        = cache->componentGroups()[group_id];
        const GroupSlot&   source_slot  = source_find.matched_node->group_slots[group_id];
        const BlockIdxType source_block = source_slot.host_block;
        EXPECT_EQ(source_block, source_blocks[group_id]) << "component_group_id=" << group_id;
        EXPECT_EQ(group->hostPool()->refCount(source_block), source_ref_baselines[group_id])
            << "component_group_id=" << group_id;
        EXPECT_TRUE(source_slot.device_blocks.empty()) << "component_group_id=" << group_id;
        EXPECT_EQ(source_slot.transfer_state, SlotTransferState::IDLE) << "component_group_id=" << group_id;
    }
}

TEST_F(HybridTypeKVCacheAllocatorTest, NonReusableFullDemandIsNotHiddenByReusableLowerTierTicket) {
    CacheConfig config   = makeTinyHybridConfig();
    auto        policies = config.groupPoliciesSnapshot();
    ASSERT_EQ(policies.size(), 2u);
    ASSERT_EQ(policies[0].group_type, CacheGroupType::LINEAR);
    ASSERT_EQ(policies[1].group_type, CacheGroupType::FULL);
    policies[1].reuse_policy         = CacheReusePolicy::NON_REUSABLE;
    policies[1].active_tail_blocks   = 1;
    policies[1].validate_tail_blocks = false;
    config.setGroupPolicies(policies);

    auto allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());
    BlockTreeCachePtr cache =
        createBlockTreeCache(config, makeHybridTypeTieredConfig(Tier::HOST, /*disk_path=*/""), allocator);
    ASSERT_NE(cache, nullptr);
    allocator->setBlockTreeCache(cache.get());

    ASSERT_EQ(cache->componentGroups().size(), 1u);
    ASSERT_EQ(cache->componentGroups().front()->group_type, CacheGroupType::LINEAR);
    const DeviceKVCacheGroupPtr reusable_device_group   = cache->deviceKVGroup(/*gid=*/0);
    const DeviceKVCacheGroupPtr non_reusable_full_group = cache->deviceKVGroup(/*gid=*/1);
    ASSERT_NE(reusable_device_group, nullptr);
    ASSERT_NE(non_reusable_full_group, nullptr);
    ASSERT_EQ(reusable_device_group->reusePolicy(), CacheReusePolicy::REUSABLE);
    ASSERT_EQ(non_reusable_full_group->reusePolicy(), CacheReusePolicy::NON_REUSABLE);
    seedHybridTypeLowerTier(*cache, Tier::HOST, /*key=*/100);

    BlockTreeFindResult source_find = cache->tree()->findNode({100});
    ASSERT_NE(source_find.matched_node, nullptr);
    TreeNode* const source_node = source_find.matched_node;
    ASSERT_EQ(source_node->group_slots.size(), 1u);
    const GroupSlot& source_slot = source_node->group_slots.front();
    ASSERT_NE(source_slot.host_block, NULL_BLOCK_IDX);
    ASSERT_TRUE(source_slot.device_blocks.empty());
    ASSERT_EQ(source_slot.transfer_state, SlotTransferState::IDLE);
    const BlockIdxType source_block        = source_slot.host_block;
    const auto&        reusable_group      = cache->componentGroups().front();
    const size_t       source_ref_baseline = reusable_group->hostPool()->refCount(source_block);

    const DeviceBlockPoolPtr device_pool = allocator->getDeviceBlockPool();
    ASSERT_NE(device_pool, nullptr);
    std::vector<std::optional<uint32_t>> device_refs_before;
    device_refs_before.reserve(device_pool->totalBlocksNum());
    for (size_t block = 1; block <= device_pool->totalBlocksNum(); ++block) {
        const BlockIdxType block_id = static_cast<BlockIdxType>(block);
        device_refs_before.push_back(device_pool->isAllocated(block_id) ?
                                         std::optional<uint32_t>(device_pool->refCount(block_id)) :
                                         std::nullopt);
    }
    const size_t               free_before       = allocator->freeBlocksNum();
    const size_t               active_before     = allocator->activeTreeCachedBlocksNum();
    const BlockTreeKeySnapshot tree_keys_before  = cache->getKeySnapshot(/*limit=*/16);
    const size_t               tree_nodes_before = cache->getStats().tree_node_count;
    ASSERT_EQ(free_before, 9u);
    ASSERT_EQ(active_before, 0u);
    ASSERT_EQ(tree_keys_before.keys, (CacheKeysType{100}));
    ASSERT_GT(source_ref_baseline, 0u);

    BatchKVCacheResourcePtr resource = makeHybridTypeLoadBackResource(config);
    resource->setBatchCacheKeys(0, CacheKeysType{100, 200});
    CompleteTokenIdsPtr token_ids = makeHybridTypeLoadBackTokenIds(
        /*seq_length=*/8, /*seq_size_per_block=*/static_cast<int>(config.seq_size_per_block));
    const CacheKeysType           request_keys_before = resource->cacheKeys(0);
    std::vector<BlockIndicesType> request_blocks_before;
    request_blocks_before.reserve(static_cast<size_t>(config.groupNums()));
    for (int gid = 0; gid < config.groupNums(); ++gid) {
        request_blocks_before.push_back(resource->blocks(0, gid));
    }

    // The reusable LINEAR ticket needs one target plus one ordinary block. The
    // separate NON_REUSABLE FULL group needs two ordinary blocks. Reserve=6
    // therefore rejects the true four-block demand, while incorrectly padding
    // FULL with one reused NULL slot would undercount three and exactly admit.
    allocator->setReserveBlockNum(free_before - 3u);
    MallocInfo malloc_info{resource, token_ids};
    malloc_info.enable_device_cache = true;
    malloc_info.reuse_cache         = true;
    malloc_info.verbose             = false;
    MallocResult result             = allocator->malloc(malloc_info);

    ASSERT_FALSE(result.success);
    ASSERT_EQ(result.async_context, nullptr);
    EXPECT_EQ(resource->batchSize(), 1);
    EXPECT_EQ(resource->groupNums(), config.groupNums());
    EXPECT_EQ(resource->cacheKeys(0), request_keys_before);
    EXPECT_EQ(resource->curBlocksNum(), 0u);
    for (int gid = 0; gid < config.groupNums(); ++gid) {
        EXPECT_EQ(resource->blocks(0, gid), request_blocks_before[static_cast<size_t>(gid)]) << "gid=" << gid;
    }
    EXPECT_EQ(resource->cacheResource(0).deviceReuseBlockNum(), 0u);
    EXPECT_EQ(allocator->activeTreeCachedBlocksNum(), 0u);
    EXPECT_EQ(allocator->freeBlocksNum(), free_before);
    EXPECT_EQ(allocator->activeTreeCachedBlocksNum(), active_before);
    for (size_t block = 1; block <= device_pool->totalBlocksNum(); ++block) {
        const BlockIdxType block_id        = static_cast<BlockIdxType>(block);
        const bool         allocated_after = device_pool->isAllocated(block_id);
        EXPECT_EQ(allocated_after, device_refs_before[block - 1].has_value()) << "device_block=" << block;
        if (allocated_after && device_refs_before[block - 1].has_value()) {
            EXPECT_EQ(device_pool->refCount(block_id), *device_refs_before[block - 1]) << "device_block=" << block;
        }
    }

    const BlockTreeKeySnapshot tree_keys_after = cache->getKeySnapshot(/*limit=*/16);
    EXPECT_EQ(tree_keys_after.version, tree_keys_before.version);
    EXPECT_EQ(tree_keys_after.keys, tree_keys_before.keys);
    EXPECT_EQ(cache->getStats().tree_node_count, tree_nodes_before);
    source_find = cache->tree()->findNode({100});
    ASSERT_EQ(source_find.matched_node, source_node);
    ASSERT_EQ(source_node->group_slots.size(), 1u);
    EXPECT_EQ(source_node->group_slots.front().host_block, source_block);
    EXPECT_TRUE(source_node->group_slots.front().device_blocks.empty());
    EXPECT_EQ(source_node->group_slots.front().transfer_state, SlotTransferState::IDLE);
    EXPECT_EQ(reusable_group->hostPool()->refCount(source_block), source_ref_baseline);

    BlockTreeMatchResult match_probe = cache->match({100});
    ASSERT_EQ(match_probe.matched_node, nullptr);
    ASSERT_EQ(match_probe.matched_blocks, 0u);
    ASSERT_NE(match_probe.load_back_ticket, nullptr);
    EXPECT_EQ(match_probe.load_back_ticket->logicalMatchedBlocks(), 1u);
    EXPECT_EQ(match_probe.load_back_blocks, 1u);
    EXPECT_EQ(match_probe.host_load_back_blocks, 1u);
    EXPECT_EQ(match_probe.disk_load_back_blocks, 0u);
    ASSERT_EQ(match_probe.load_back_ticket->items().size(), 1u);
    const PendingLoadBackItem& pending = match_probe.load_back_ticket->items().front();
    EXPECT_EQ(pending.node, source_node);
    EXPECT_EQ(pending.source_tier, Tier::HOST);
    EXPECT_EQ(pending.source_blocks, (std::vector<BlockIdxType>{source_block}));
    EXPECT_EQ(pending.device_group_ids, (std::vector<int>{0}));
    EXPECT_TRUE(pending.target_device_blocks.empty());
    EXPECT_EQ(reusable_group->hostPool()->refCount(source_block), source_ref_baseline + 1u);
    cache->releaseMatchedBlocks(match_probe.matched_block_sets);
    match_probe.matched_block_sets.clear();
    match_probe.load_back_ticket.reset();
    EXPECT_EQ(reusable_group->hostPool()->refCount(source_block), source_ref_baseline);
    EXPECT_EQ(source_node->group_slots.front().transfer_state, SlotTransferState::IDLE);
}

TEST_F(HybridTypeKVCacheAllocatorTest, ProtectedPreflightRejectsMalformedRealTicketWithoutMutation) {
    const CacheConfig config    = makeTinyHybridConfig();
    auto              allocator = std::make_shared<HybridTypePreflightTestPeer>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    HybridTypePreflightEnvironment environment = makeHybridTypePreflightEnvironment(allocator->getDeviceBlockPool());
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
        std::shared_ptr<LoadBackTicket> ticket = captureHybridTypePreflightTicket(*environment.cache);
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
        std::shared_ptr<LoadBackTicket> ticket = captureHybridTypePreflightTicket(*environment.cache);
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

TEST_F(HybridTypeKVCacheAllocatorTest, InitAndAddressLookupSmoke) {
    auto config = makeTinyHybridConfig();
    ASSERT_TRUE(initWithBlockTreeCache(config));
    ASSERT_EQ(allocator_->blockTreeCache(), block_tree_cache_.get());

    EXPECT_EQ(allocator_->seqSizePerBlock(), 4);
    EXPECT_EQ(allocator_->totalBlocksNum(), config.block_num - 1);
    EXPECT_EQ(allocator_->freeBlocksNum(), config.block_num - 1);
    EXPECT_EQ(allocator_->activeTreeCachedBlocksNum(), 0u);

    // Should be able to fetch address for any global layer and non-zero block id.
    auto addr0 = allocator_->convertIndexToAddr(/*layer_id=*/0, /*block_id=*/1);
    auto addr3 = allocator_->convertIndexToAddr(/*layer_id=*/3, /*block_id=*/1);
    EXPECT_NE(addr0.kv_addr, nullptr);
    EXPECT_NE(addr3.kv_addr, nullptr);
}

TEST_F(HybridTypeKVCacheAllocatorTest, ConvertToGlobalLayerIdHybridNoMtp) {
    auto config    = makeTinyHybridConfig();
    auto allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);

    EXPECT_EQ(allocator->convertToGlobalLayerId(/*model_id=*/0, /*local_layer_id=*/0), 0u);
    EXPECT_EQ(allocator->convertToGlobalLayerId(/*model_id=*/0, /*local_layer_id=*/3), 3u);
    EXPECT_EQ(allocator->convertToGlobalLayerId(/*model_id=*/0, /*local_layer_id=*/4),
              std::numeric_limits<uint32_t>::max());

    // no mtp sub-model
    EXPECT_EQ(allocator->convertToGlobalLayerId(/*model_id=*/1, /*local_layer_id=*/0),
              std::numeric_limits<uint32_t>::max());
}

TEST_F(HybridTypeKVCacheAllocatorTest, ConvertToGlobalLayerIdHybridWithMtpSubConfigs) {
    auto config    = makeTinyHybridMtpConfigByCreateSpConfig();
    auto allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);

    EXPECT_EQ(allocator->convertToGlobalLayerId(/*model_id=*/0, /*local_layer_id=*/2), 2u);
    EXPECT_EQ(allocator->convertToGlobalLayerId(/*model_id=*/1, /*local_layer_id=*/0), 4u);
    EXPECT_EQ(allocator->convertToGlobalLayerId(/*model_id=*/2, /*local_layer_id=*/0), 5u);
    EXPECT_EQ(allocator->convertToGlobalLayerId(/*model_id=*/2, /*local_layer_id=*/1),
              std::numeric_limits<uint32_t>::max());
    EXPECT_EQ(allocator->convertToGlobalLayerId(/*model_id=*/3, /*local_layer_id=*/0),
              std::numeric_limits<uint32_t>::max());
}

TEST_F(HybridTypeKVCacheAllocatorTest, GetNeedBlocksUsesGroupGetNeedBlocksAndReuseFlag) {
    auto config = makeTinyHybridConfig();
    ASSERT_TRUE(initWithBlockTreeCache(config));
    ASSERT_EQ(allocator_->blockTreeCache(), block_tree_cache_.get());

    // batch=2, seq_len=12 (3 slots), reserve_step=2
    auto token_ids = makeCompleteTokenIds(/*batch_size=*/2, /*seq_length=*/12, /*seq_size_per_block=*/4);
    token_ids->setReserveStep(2);

    // Reuse disabled: linear group keeps tail and tail-1 for common blocks; reserve_step contributes extra blocks.
    // full group contributes common=3, extra=1.
    {
        auto       batch_res = makeBatchResource(/*batch_size=*/2, config, CacheKeysType{100, 101, 102, 103});
        MallocInfo info{batch_res, token_ids};
        info.enable_device_cache = false;
        info.reuse_cache         = false;
        // common_total = full(3) + linear(2) = 5
        // extra_total  = full(1) + linear(reserve_step-1=1) = 2
        // total = 5 + 2*2 = 9
        EXPECT_EQ(allocator_->getNeedBlocks(info), 9);
    }

    // Reuse enabled but no existing blocks: linear group keeps step hits plus tail/tail-1.
    {
        auto       batch_res = makeBatchResource(/*batch_size=*/2, config, CacheKeysType{100, 101, 102, 103});
        MallocInfo info{batch_res, token_ids};
        info.enable_device_cache = true;
        info.reuse_cache         = true;
        // full: common=3 extra=1
        // linear: common=2, extra=reserve_step-1(=1)
        // common_total = 3 + 2 = 5
        // extra_total  = 1 + 1 = 2
        // total = 5 + 2*2 = 9
        EXPECT_EQ(allocator_->getNeedBlocks(info), 9);
    }
}

TEST_F(HybridTypeKVCacheAllocatorTest, JointReuseUsesFullPrefixAndLinearTailOnly) {
    auto config = makeTinyHybridConfig();
    ASSERT_TRUE(initWithBlockTreeCache(config));
    auto block_pool = allocator_->getDeviceBlockPool();
    ASSERT_NE(block_pool, nullptr);

    const size_t free_before = allocator_->freeBlocksNum();
    const auto   seed = seedRealCachePath(allocator_, config, CacheKeysType{100, 101, 102, 103}, /*seq_length=*/12);
    ASSERT_EQ(seed.group_blocks.size(), 2u);
    ASSERT_EQ(seed.group_blocks[0].size(), 3u);
    ASSERT_EQ(seed.group_blocks[1].size(), 3u);
    ASSERT_TRUE(isNullBlockIdx(seed.group_blocks[0][0]));
    ASSERT_FALSE(isNullBlockIdx(seed.group_blocks[0][1]));
    ASSERT_FALSE(isNullBlockIdx(seed.group_blocks[0][2]));
    for (const auto& group_blocks : seed.group_blocks) {
        for (auto block : group_blocks) {
            if (!isNullBlockIdx(block)) {
                EXPECT_EQ(block_pool->refCount(block), 1u);
            }
        }
    }

    auto batch_res = makeBatchResource(/*batch_size=*/1, config, CacheKeysType{100, 101, 102, 103});
    auto token_ids = makeCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/12, /*seq_size_per_block=*/4);

    MallocInfo info{batch_res, token_ids};
    info.enable_device_cache = true;
    info.reuse_cache         = true;
    auto result              = allocator_->malloc(info);
    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.reuse_len, 12);
    EXPECT_EQ(result.async_context, nullptr);

    // Config order: gid=0 LINEAR, gid=1 FULL. The real joint match reuses the
    // complete FULL prefix and only the latest LINEAR tail; LINEAR tail-1 is
    // allocated as ordinary request state.
    const auto& full_out = batch_res->blocks(/*batch_id=*/0, /*gid=*/1);
    ASSERT_EQ(full_out.size(), 3u);
    EXPECT_EQ(full_out, seed.group_blocks[1]);

    const auto& linear_out = batch_res->blocks(/*batch_id=*/0, /*gid=*/0);
    ASSERT_EQ(linear_out.size(), 3u);
    EXPECT_TRUE(isNullBlockIdx(linear_out[0]));
    EXPECT_FALSE(isNullBlockIdx(linear_out[1]));
    EXPECT_EQ(linear_out[2], seed.group_blocks[0][2]);
    EXPECT_NE(linear_out[1], seed.group_blocks[0][1]);

    allocator_->free(FreeInfo{batch_res, token_ids});
    EXPECT_EQ(batch_res->curBlocksNum(), 0u);
    for (const auto& group_blocks : seed.group_blocks) {
        for (auto block : group_blocks) {
            if (!isNullBlockIdx(block)) {
                EXPECT_EQ(block_pool->refCount(block), 1u);
            }
        }
    }
    EXPECT_GT(allocator_->blockTreeCache()->reclaimBlocks(config.block_num, Tier::DEVICE), 0);
    EXPECT_EQ(allocator_->freeBlocksNum(), free_before);
}

TEST_F(HybridTypeKVCacheAllocatorTest, DisableReuseKeepsLinearTailAndTailMinusOneOnInitMalloc) {
    auto config = makeTinyHybridConfig();
    ASSERT_TRUE(initWithBlockTreeCache(config));
    ASSERT_EQ(allocator_->blockTreeCache(), block_tree_cache_.get());

    auto batch_res = makeBatchResource(/*batch_size=*/1, config, CacheKeysType{100, 101, 102, 103});
    // Disable device cache reuse.

    auto token_ids = makeCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/12, /*seq_size_per_block=*/4);

    MallocInfo info{batch_res, token_ids};
    info.enable_device_cache = false;
    info.reuse_cache         = false;
    auto result              = allocator_->malloc(info);
    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.async_context, nullptr);

    // Linear group should keep tail and tail-1 across common length slots.
    const auto& linear_out = batch_res->blocks(0, /*group_id=*/0);
    ASSERT_EQ(linear_out.size(), 3u);
    EXPECT_TRUE(isNullBlockIdx(linear_out[0]));
    EXPECT_FALSE(isNullBlockIdx(linear_out[1]));
    EXPECT_FALSE(isNullBlockIdx(linear_out[2]));

    FreeInfo free_info{batch_res, token_ids};
    allocator_->free(free_info);
    EXPECT_EQ(batch_res->curBlocksNum(), 0);
}

TEST_F(HybridTypeKVCacheAllocatorTest, DisableDeviceCacheSkipsReuseMatchAndAllocatesLinearTailAndTailMinusOne) {
    auto config      = makeTinyHybridConfig();
    config.block_num = 16;
    ASSERT_TRUE(initWithBlockTreeCache(config));
    auto block_pool = allocator_->getDeviceBlockPool();
    ASSERT_NE(block_pool, nullptr);

    const size_t free_before = allocator_->freeBlocksNum();
    const auto   seed = seedRealCachePath(allocator_, config, CacheKeysType{100, 101, 102, 103}, /*seq_length=*/12);
    ASSERT_EQ(seed.group_blocks.size(), 2u);
    ASSERT_EQ(seed.group_blocks[1].size(), 3u);

    auto batch_res = makeBatchResource(/*batch_size=*/1, config, CacheKeysType{100, 101, 102, 103});
    auto token_ids = makeCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/12, /*seq_size_per_block=*/4);

    MallocInfo info{batch_res, token_ids};
    info.enable_device_cache = false;
    info.reuse_cache         = false;
    auto result              = allocator_->malloc(info);
    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.reuse_len, 0);
    EXPECT_EQ(result.async_context, nullptr);

    const auto& full_out = batch_res->blocks(/*batch_id=*/0, /*gid=*/1);
    ASSERT_EQ(full_out.size(), 3u);
    for (size_t i = 0; i < full_out.size(); ++i) {
        EXPECT_FALSE(isNullBlockIdx(full_out[i]));
        EXPECT_NE(full_out[i], seed.group_blocks[1][i]);
    }

    const auto& linear_out = batch_res->blocks(/*batch_id=*/0, /*gid=*/0);
    ASSERT_EQ(linear_out.size(), 3u);
    EXPECT_TRUE(isNullBlockIdx(linear_out[0]));
    EXPECT_FALSE(isNullBlockIdx(linear_out[1]));
    EXPECT_FALSE(isNullBlockIdx(linear_out[2]));
    EXPECT_EQ(countValidBlocks(linear_out), 2u);

    for (auto block : linear_out) {
        if (!isNullBlockIdx(block)) {
            EXPECT_NE(block, seed.group_blocks[0][1]);
            EXPECT_NE(block, seed.group_blocks[0][2]);
        }
    }

    allocator_->free(FreeInfo{batch_res, token_ids});
    EXPECT_EQ(batch_res->curBlocksNum(), 0u);
    EXPECT_GT(allocator_->blockTreeCache()->reclaimBlocks(config.block_num, Tier::DEVICE), 0);
    EXPECT_EQ(allocator_->freeBlocksNum(), free_before);
}

TEST_F(HybridTypeKVCacheAllocatorTest, IncrDecrKVCacheRefReferencesOnlyMatchedValidBlocksAcrossGroups) {
    auto config = makeTinyHybridConfig();
    ASSERT_TRUE(initWithBlockTreeCache(config));
    auto block_pool = allocator_->getDeviceBlockPool();
    ASSERT_NE(block_pool, nullptr);

    const size_t free_before = allocator_->freeBlocksNum();
    auto         resource    = makeBatchResource(/*batch_size=*/1, config, CacheKeysType{100, 101, 102});
    auto         tokens      = makeCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/12, /*seq_size_per_block=*/4);
    MallocInfo   malloc_info{resource, tokens};
    malloc_info.enable_device_cache = false;
    malloc_info.reuse_cache         = false;
    auto malloc_result              = allocator_->malloc(malloc_info);
    ASSERT_TRUE(malloc_result.success);
    EXPECT_EQ(malloc_result.async_context, nullptr);

    const auto& linear_blocks = resource->blocks(/*batch_id=*/0, /*gid=*/0);
    const auto& full_blocks   = resource->blocks(/*batch_id=*/0, /*gid=*/1);
    ASSERT_EQ(linear_blocks.size(), 3u);
    ASSERT_EQ(full_blocks.size(), 3u);
    ASSERT_TRUE(isNullBlockIdx(linear_blocks[0]));
    ASSERT_FALSE(isNullBlockIdx(linear_blocks[2]));
    for (auto block : linear_blocks) {
        if (!isNullBlockIdx(block)) {
            EXPECT_EQ(block_pool->refCount(block), 1u);
        }
    }
    for (auto block : full_blocks) {
        EXPECT_EQ(block_pool->refCount(block), 1u);
    }

    auto ref = allocator_->incrKVCacheRef(resource->cacheResource(0), CacheKeysType{100, 999, 102});
    ASSERT_NE(ref, nullptr);
    ASSERT_EQ(ref->groupNums(), 2);
    ASSERT_EQ(ref->cacheKeys(), (CacheKeysType{100, 102}));
    ASSERT_EQ(ref->blocks(0).size(), 2u);
    ASSERT_EQ(ref->blocks(1).size(), 2u);
    EXPECT_TRUE(isNullBlockIdx(ref->blocks(0)[0]));
    EXPECT_EQ(ref->blocks(0)[1], linear_blocks[2]);
    EXPECT_EQ(ref->blocks(1)[0], full_blocks[0]);
    EXPECT_EQ(ref->blocks(1)[1], full_blocks[2]);
    EXPECT_EQ(block_pool->refCount(linear_blocks[2]), 2u);
    EXPECT_EQ(block_pool->refCount(full_blocks[0]), 2u);
    EXPECT_EQ(block_pool->refCount(full_blocks[2]), 2u);

    ref.reset();
    EXPECT_EQ(block_pool->refCount(linear_blocks[2]), 1u);
    EXPECT_EQ(block_pool->refCount(full_blocks[0]), 1u);
    EXPECT_EQ(block_pool->refCount(full_blocks[2]), 1u);
    allocator_->free(FreeInfo{resource, tokens});
    EXPECT_EQ(resource->curBlocksNum(), 0u);
    EXPECT_EQ(allocator_->freeBlocksNum(), free_before);
}

TEST_F(HybridTypeKVCacheAllocatorTest, IncrKVCacheRefPreservesConnectorDummyTail) {
    auto config = makeTinyHybridConfig();
    ASSERT_TRUE(initWithBlockTreeCache(config));
    auto block_pool = allocator_->getDeviceBlockPool();
    ASSERT_NE(block_pool, nullptr);

    const size_t free_before = allocator_->freeBlocksNum();
    auto         resource    = makeBatchResource(/*batch_size=*/1, config, CacheKeysType{101, 103, 999});
    auto         tokens      = makeCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/8, /*seq_size_per_block=*/4);
    MallocInfo   malloc_info{resource, tokens};
    malloc_info.enable_device_cache = false;
    malloc_info.reuse_cache         = false;
    ASSERT_TRUE(allocator_->malloc(malloc_info).success);

    auto& source = resource->cacheResource(/*batch_id=*/0);
    source.rebuildLinearBlockDependencies();
    source.setLastBlockAligned(false);
    ASSERT_EQ(source.blocks(/*gid=*/0).size(), 2u);
    ASSERT_EQ(source.blocks(/*gid=*/1).size(), 2u);

    std::vector<BlockIdxType> valid_source_blocks;
    for (int gid = 0; gid < source.groupNums(); ++gid) {
        for (auto block : source.blocks(gid)) {
            if (!isNullBlockIdx(block)) {
                valid_source_blocks.push_back(block);
                EXPECT_EQ(block_pool->refCount(block), 1u);
            }
        }
    }
    ASSERT_EQ(valid_source_blocks.size(), 4u);

    auto ref = allocator_->incrKVCacheRef(source, CacheKeysType{101, 103, 999}, /*is_connector=*/true);
    ASSERT_NE(ref, nullptr);
    EXPECT_FALSE(ref->lastBlockAligned());
    EXPECT_EQ(ref->cacheKeys(), (CacheKeysType{101, 103, 999}));
    ASSERT_EQ(ref->blocks(0).size(), 3u);
    ASSERT_EQ(ref->blocks(1).size(), 3u);
    EXPECT_TRUE(isNullBlockIdx(ref->blocks(0)[2]));
    EXPECT_TRUE(isNullBlockIdx(ref->blocks(1)[2]));

    for (auto block : valid_source_blocks) {
        EXPECT_EQ(block_pool->refCount(block), 2u);
    }

    ref.reset();
    for (auto block : valid_source_blocks) {
        EXPECT_EQ(block_pool->refCount(block), 1u);
    }
    allocator_->free(FreeInfo{resource, tokens});
    EXPECT_EQ(resource->curBlocksNum(), 0u);
    EXPECT_EQ(allocator_->freeBlocksNum(), free_before);
}

TEST_F(HybridTypeKVCacheAllocatorTest, InsertIntoCachePreservesLegacyNonCpAggregateSurface) {
    auto config = makeTinyHybridConfig();
    ASSERT_TRUE(initWithBlockTreeCache(config));
    auto block_pool = allocator_->getDeviceBlockPool();
    ASSERT_NE(block_pool, nullptr);

    const size_t free_before = allocator_->freeBlocksNum();
    const auto   seed        = seedRealCachePath(allocator_, config, CacheKeysType{100, 101, 102}, /*seq_length=*/12);
    ASSERT_EQ(seed.group_blocks.size(), 2u);
    ASSERT_EQ(seed.group_blocks[0].size(), 3u);
    ASSERT_EQ(seed.group_blocks[1].size(), 3u);

    auto match = block_tree_cache_->match(CacheKeysType{100, 101, 102});
    ASSERT_EQ(match.matched_blocks, 3u);
    ASSERT_EQ(match.group_block_indices.at(/*gid=*/1), seed.group_blocks[1]);
    ASSERT_EQ(match.group_block_indices.at(/*gid=*/0).size(), 1u);
    EXPECT_EQ(match.group_block_indices.at(/*gid=*/0).back(), seed.group_blocks[0].back());

    block_tree_cache_->releaseMatchedBlocks(match.matched_block_sets);
    match.matched_block_sets.clear();
    for (const auto& group_blocks : seed.group_blocks) {
        for (auto block : group_blocks) {
            if (!isNullBlockIdx(block)) {
                EXPECT_EQ(block_pool->refCount(block), 1u);
            }
        }
    }
    EXPECT_GT(allocator_->blockTreeCache()->reclaimBlocks(config.block_num, Tier::DEVICE), 0);
    EXPECT_EQ(allocator_->freeBlocksNum(), free_before);
}

TEST_F(HybridTypeKVCacheAllocatorTest, ConvertIndexToBufferAndAllLayerCacheBaseSmoke) {
    auto config = makeTinyHybridConfig();
    ASSERT_TRUE(initWithBlockTreeCache(config));
    ASSERT_EQ(allocator_->blockTreeCache(), block_tree_cache_.get());

    KVCacheAllocator* base = allocator_.get();
    auto              buf0 = base->convertIndexToBuffer(/*layer_id=*/0, /*block_id=*/1);
    ASSERT_FALSE(buf0.empty());
    EXPECT_NE(buf0[0].addr, nullptr);

    auto layout = allocator_->allLayerCacheBase();
    EXPECT_EQ(layout.layers_to_kv_buffer_ptrs.size(), static_cast<size_t>(config.layer_num));
    for (size_t i = 0; i < layout.layers_to_kv_buffer_ptrs.size(); ++i) {
        EXPECT_TRUE(layout.layers_to_kv_buffer_ptrs[i].defined());
    }
}

TEST_F(HybridTypeKVCacheAllocatorTest, IncrMallocRollbackFreesPartiallyAllocatedBlocks) {
    auto config      = makeTinyHybridConfig();
    config.block_num = 6;  // free=5
    ASSERT_TRUE(initWithBlockTreeCache(config));
    auto block_pool = allocator_->getDeviceBlockPool();
    ASSERT_NE(block_pool, nullptr);

    const size_t free_before = allocator_->freeBlocksNum();
    auto         batch_res   = makeBatchResource(/*batch_size=*/1, config, CacheKeysType{100, 101, 102});
    auto         token_ids   = makeCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/4, /*seq_size_per_block=*/4);
    MallocInfo   init_info{batch_res, token_ids};
    init_info.enable_device_cache = false;
    init_info.reuse_cache         = false;
    auto init_result              = allocator_->malloc(init_info);
    ASSERT_TRUE(init_result.success);
    EXPECT_EQ(init_result.async_context, nullptr);
    ASSERT_EQ(batch_res->blocksNum(0, /*gid=*/0), 1);
    ASSERT_EQ(batch_res->blocksNum(0, /*gid=*/1), 1);

    const auto linear_block_before = batch_res->blocks(0, /*gid=*/0)[0];
    const auto full_block_before   = batch_res->blocks(0, /*gid=*/1)[0];

    // Leave exactly 1 free block in pool, so linear allocates 1 and full fails on the next allocation.
    const size_t free_before_incr = block_pool->freeBlocksNum();
    ASSERT_GE(free_before_incr, 1u);
    auto keep_opt = block_pool->malloc(free_before_incr - 1);
    ASSERT_TRUE(keep_opt.has_value());
    auto keep = keep_opt.value();
    // Single-count pool: hold the reserved blocks with a request ref (reserve-only malloc()).
    block_pool->incRef(keep);
    ASSERT_EQ(block_pool->freeBlocksNum(), 1u);

    // Incr to seq_len=9 => 3 slots per group. Linear adds 2 slots but allocates only 1 real block; full needs 2.
    token_ids->setSeqLength(9);
    MallocInfo incr_info{batch_res, token_ids};
    incr_info.enable_device_cache = false;
    incr_info.reuse_cache         = false;
    auto incr_result              = allocator_->malloc(incr_info);
    EXPECT_FALSE(incr_result.success);

    ASSERT_EQ(batch_res->blocksNum(0, /*gid=*/0), 1);
    ASSERT_EQ(batch_res->blocksNum(0, /*gid=*/1), 1);
    EXPECT_EQ(batch_res->blocks(0, /*gid=*/0)[0], linear_block_before);
    EXPECT_EQ(batch_res->blocks(0, /*gid=*/1)[0], full_block_before);

    // Free blocks count should return to 1 (no leaks).
    EXPECT_EQ(block_pool->freeBlocksNum(), 1u);

    block_pool->decRef(keep);
    allocator_->free(FreeInfo{batch_res, token_ids});
    EXPECT_EQ(batch_res->curBlocksNum(), 0u);
    EXPECT_EQ(allocator_->freeBlocksNum(), free_before);
}

}  // namespace test
}  // namespace rtp_llm

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
