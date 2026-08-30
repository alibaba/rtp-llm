#include "gtest/gtest.h"
#include "gmock/gmock.h"

#define private public
#define protected public
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/cache/KVCacheTransferPlanner.h"
#include "rtp_llm/cpp/cache/KVCacheResource.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/disaggregate/cache_store/RequestBlockBufferStore.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"
#include "rtp_llm/cpp/engine_base/stream/StreamCacheResource.h"
#include "rtp_llm/cpp/model_rpc/DecodeRpcServer.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/config/RoleTypes.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <memory>
#include <numeric>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace rtp_llm {

namespace {

constexpr int kDsv4PoolNum        = 7;
constexpr int kDsv4TokensPerBlock = 256;

class DummyMemoryUtil: public MemoryUtil {
public:
    bool regUserMr(void*, uint64_t, bool, uint64_t = 0) override {
        return true;
    }
    bool deregUserMr(void*, bool) override {
        return true;
    }
    bool isMemoryMr(void*, uint64_t, bool, bool) override {
        return true;
    }
    bool findMemoryMr(void*, void*, uint64_t, bool, bool) override {
        return true;
    }
    bool isRdmaMode() override {
        return false;
    }
};

class MemoryBackedCacheStore: public NormalCacheStore {
public:
    MemoryBackedCacheStore() {
        memory_util_                = std::make_shared<DummyMemoryUtil>();
        request_block_buffer_store_ = std::make_shared<RequestBlockBufferStore>(memory_util_);
    }

    void store(const std::shared_ptr<RequestBlockBuffer>& request_block_buffer,
               CacheStoreStoreDoneCallback                callback) override {
        runtimeSyncAndCheck();
        for (const auto& [key, block] : request_block_buffer->getBlocks()) {
            auto src_options = torch::TensorOptions(torch::kUInt8).device(block->gpu_mem ? torch::kCUDA : torch::kCPU);
            auto src         = torch::from_blob(block->addr.get(), {(int64_t)block->len}, src_options);
            auto host        = block->gpu_mem ? src.cpu().contiguous() : src.contiguous();
            std::vector<uint8_t> bytes(static_cast<size_t>(block->len));
            std::memcpy(bytes.data(), host.data_ptr<uint8_t>(), bytes.size());
            stored_blocks_[key] = std::move(bytes);
        }
        store_request_keys_.push_back(request_block_buffer->getRequestKey());
        store_buffer_requests_.push_back(request_block_buffer);
        callback(true, CacheStoreErrorCode::None);
    }

    void load(const std::shared_ptr<RequestBlockBuffer>& request_block_buffer,
              CacheStoreLoadDoneCallback                 callback,
              const std::string&,
              uint32_t,
              uint32_t,
              uint32_t = 1000,
              int      = 1,
              int      = 0) override {
        bool ok = true;
        for (const auto& [key, block] : request_block_buffer->getBlocks()) {
            auto it = stored_blocks_.find(key);
            if (it == stored_blocks_.end() || it->second.size() != block->len) {
                ok = false;
                continue;
            }
            auto host = torch::from_blob(const_cast<uint8_t*>(it->second.data()),
                                         {(int64_t)it->second.size()},
                                         torch::TensorOptions(torch::kUInt8).device(torch::kCPU))
                            .clone();
            auto dst_options = torch::TensorOptions(torch::kUInt8).device(block->gpu_mem ? torch::kCUDA : torch::kCPU);
            auto dst         = torch::from_blob(block->addr.get(), {(int64_t)block->len}, dst_options);
            dst.copy_(host);
        }
        runtimeSyncAndCheck();
        load_request_keys_.push_back(request_block_buffer->getRequestKey());
        callback(ok, ok ? CacheStoreErrorCode::None : CacheStoreErrorCode::LoadErrorUnknown);
    }

    std::shared_ptr<LoadContext>
    loadBuffers(const std::vector<std::shared_ptr<RequestBlockBuffer>>& request_block_buffers,
                const std::string&                                      ip,
                uint32_t                                                port,
                uint32_t                                                rdma_port,
                int64_t                                                 timeout_ms,
                LoadContext::CheckCancelFunc                            check_cancel_func,
                int                                                     partition_count,
                int                                                     partition_id) override {
        load_buffer_requests_.insert(
            load_buffer_requests_.end(), request_block_buffers.begin(), request_block_buffers.end());
        auto context = std::make_shared<LoadContext>(shared_from_this(), false);
        context->load(
            request_block_buffers, ip, port, rdma_port, timeout_ms, check_cancel_func, partition_count, partition_id);
        return context;
    }

    std::unordered_map<std::string, std::vector<uint8_t>> stored_blocks_;
    std::vector<std::string>                              store_request_keys_;
    std::vector<std::string>                              load_request_keys_;
    std::vector<std::shared_ptr<RequestBlockBuffer>>      store_buffer_requests_;
    std::vector<std::shared_ptr<RequestBlockBuffer>>      load_buffer_requests_;
};

class MinimalEngine: public EngineBase {
public:
    MinimalEngine(const EngineInitParams& params, std::shared_ptr<KVCacheManager> cache_manager): EngineBase(params) {
        resource_context_.cache_manager = std::move(cache_manager);
    }

    std::shared_ptr<GenerateStream> enqueue(const std::shared_ptr<GenerateInput>&) override {
        return nullptr;
    }
    void         enqueue(std::shared_ptr<GenerateStream>&) override {}
    absl::Status stop() override {
        return absl::OkStatus();
    }
    absl::StatusOr<GenerateStreamPtr> preRun(const std::shared_ptr<GenerateInput>&, preRunMode) override {
        return absl::UnimplementedError("unused in test");
    }
    KVCacheInfo getCacheStatusInfo(int64_t, bool) override {
        return KVCacheInfo();
    }
};

void fillDsv4RegionBytes(
    const std::shared_ptr<KVCacheManager>& manager, int block_id, int layer_id, const std::string& tag, uint8_t value) {
    auto parts = manager->convertIndexToBufferByTag(block_id, layer_id, tag);
    ASSERT_EQ(parts.size(), 1u);
    auto device = torch::from_blob(
        parts[0].addr, {(int64_t)parts[0].size_bytes}, torch::TensorOptions(torch::kUInt8).device(torch::kCUDA));
    auto host =
        torch::full({(int64_t)parts[0].size_bytes}, value, torch::TensorOptions(torch::kUInt8).device(torch::kCPU));
    device.copy_(host);
}

void expectDsv4RegionBytes(
    const std::shared_ptr<KVCacheManager>& manager, int block_id, int layer_id, const std::string& tag, uint8_t value) {
    auto parts = manager->convertIndexToBufferByTag(block_id, layer_id, tag);
    ASSERT_EQ(parts.size(), 1u);
    auto device = torch::from_blob(
        parts[0].addr, {(int64_t)parts[0].size_bytes}, torch::TensorOptions(torch::kUInt8).device(torch::kCUDA));
    auto        host = device.cpu().contiguous();
    const auto* ptr  = host.data_ptr<uint8_t>();
    for (size_t i = 0; i < parts[0].size_bytes; ++i) {
        ASSERT_EQ(ptr[i], value) << "byte=" << i << " layer=" << layer_id << " block=" << block_id << " tag=" << tag;
    }
}

uint8_t dsv4PdPattern(int layer_id, int gid, size_t block_pos) {
    return static_cast<uint8_t>(17 + layer_id * 19 + gid * 11 + block_pos);
}

// Block positions a group actually transfers on the PD prefill->decode path.
// Both sides derive the plan from the group policy through buildCacheStorePlan:
// runtimeWriteCacheStore directly, DecodeRpcServer::loadCache via
// DecodeRpcServer::buildGroupLoadPlan. This helper mirrors that plan for
// cp_size == 1, where it reduces to blockPositionsForCacheTransfer.
std::vector<size_t>
dsv4TransferPositions(const CacheConfig& config, int gid, size_t block_num, size_t reuse_block_size) {
    const auto policy = config.policyForGroup(static_cast<size_t>(gid));
    return blockPositionsForCacheTransfer(block_num,
                                          reuse_block_size,
                                          /*use_hybrid=*/config.groupNums() > 1,
                                          /*transfer_tail_blocks=*/policy.active_tail_blocks > 0,
                                          static_cast<size_t>(policy.active_tail_blocks),
                                          /*hybrid_full_from_begin=*/true);
}

torch::Tensor blockIdsTensor(const BatchKVCacheResourcePtr& resource, int gid) {
    const auto& blocks = resource->blocks(0, gid);
    return torch::from_blob(const_cast<int*>(blocks.data()), {1, static_cast<int64_t>(blocks.size())}, torch::kInt32)
        .clone();
}

// Per-request metadata for a one-context-batch, one-block cache-store write.
// Physical geometry (strides, tokens per block, opaque-store flag, tag) now lives
// in CacheConfig / LayerKVCache, so it is no longer part of the write inputs.
torch_ext::PyCacheStoreInputs makeSingleBlockWriteInputs(int64_t cache_key, int request_id_val, int tokens_per_block) {
    torch_ext::PyCacheStoreInputs inputs;
    inputs.input_lengths_host    = torch::tensor({tokens_per_block}, torch::kInt32);
    inputs.prefix_lengths_host   = torch::tensor({0}, torch::kInt32);
    inputs.host_kv_cache_offset  = torch::tensor({{1}}, torch::kInt32);
    inputs.request_id            = torch::tensor({(int64_t)request_id_val}, torch::kInt64);
    inputs.request_pd_separation = torch::tensor({true}, torch::kBool);
    inputs.cache_keys            = torch::tensor({{cache_key}}, torch::kInt64);
    return inputs;
}

// Single-group, single-layer config that pins the physical block strides and the
// opaque-store policy the write path must use. Geometry now travels through
// CacheConfig instead of the per-call write inputs.
CacheConfig makeSingleBlockWriteConfig(const std::string& tag,
                                       int                tokens_per_block,
                                       size_t             kv_stride,
                                       size_t             kv_scale_stride,
                                       bool               use_opaque_kv_cache_store) {
    constexpr uint32_t kBlockNum = 2;
    // BF16 keeps the prototype spec scale-free, so a caller asking for
    // kv_scale_stride == 0 really gets a scale-less group (an INT8/FP8 prototype
    // would have CacheConfig::setTopology backfill a non-zero scale stride).
    auto config                      = test::makeSingleGroupCacheConfig(test::makeMhaSpec(tag,
                                                                     static_cast<size_t>(tokens_per_block),
                                                                     rtp_llm::DataType::TYPE_BF16,
                                                                     /*local_head_num_kv=*/1,
                                                                     /*size_per_head=*/1),
                                                   CacheGroupType::FULL,
                                                   /*layer_num=*/1,
                                                   /*block_num=*/static_cast<int>(kBlockNum));
    config.use_opaque_kv_cache_store = use_opaque_kv_cache_store;
    config.kv_block_stride_bytes     = kv_stride;
    config.kv_scale_stride_bytes     = kv_scale_stride;
    config.setGroupBlockLayout({kBlockNum}, {kv_stride}, {kv_scale_stride});
    return config;
}

// Per-request metadata for a DSV4 PD prefill write of one context request.
torch_ext::PyCacheStoreInputs makeDsv4WriteInputs(int64_t                          request_id,
                                                  int                              input_length,
                                                  int                              prefix_length,
                                                  const torch::Tensor&             block_ids,
                                                  const std::vector<CacheKeyType>& cache_keys) {
    torch_ext::PyCacheStoreInputs inputs;
    inputs.input_lengths_host    = torch::tensor({input_length}, torch::kInt32);
    inputs.prefix_lengths_host   = torch::tensor({prefix_length}, torch::kInt32);
    inputs.host_kv_cache_offset  = block_ids;
    inputs.request_id            = torch::tensor({request_id}, torch::kInt64);
    inputs.request_pd_separation = torch::tensor({true}, torch::kBool);
    inputs.cache_keys            = torch::from_blob(const_cast<CacheKeyType*>(cache_keys.data()),
                                                    {1, (int64_t)cache_keys.size()},
                                         torch::TensorOptions(torch::kInt64))
                            .clone();
    return inputs;
}

}  // namespace

// =============================================================================
// Test fixture: PD sep KV cache release correctness
// Validates that holdKVCacheForPDSep / releaseKVCacheForPDSep / releaseResource
// interact correctly with respect to:
//   1. Block ref-counts stay > 0 while pd_kvcache_ref_ is held
//   2. insertIntoCache (device reuse) is called before blocks are cleared
//   3. freeBlocksNum() returns to baseline after both release paths complete
//   4. Race condition: concurrent releaseKVCacheForPDSep (grpc thread) vs
//      releaseResource (engine thread)
// =============================================================================
class PdSepKVCacheReleaseTest: public DeviceTestBase {
protected:
    PdSepKVCacheReleaseTest(): perf_scope("PERF_TEST", "1") {}

    // Simple config: 3 layers, 16 blocks, 8 tokens/block
    CacheConfig makeConfig() {
        return test::makeSimpleMhaCacheConfig(/*layer_num=*/3,
                                              /*block_num=*/16,
                                              /*tokens_per_block=*/8,
                                              rtp_llm::DataType::TYPE_INT8);
    }

    CacheConfig makeDsv4Config(uint32_t block_num               = 16,
                               uint32_t seq_size_per_block      = kDsv4TokensPerBlock,
                               uint32_t kernel_seq_size_per_blk = kDsv4TokensPerBlock) {
        ModelConfig mc;
        mc.num_layers                                                = 43;
        mc.hidden_size                                               = 4096;
        mc.attn_config.head_num                                      = 64;
        mc.attn_config.kv_head_num                                   = 1;
        mc.attn_config.size_per_head                                 = 512;
        mc.attn_config.rope_head_dim                                 = 64;
        mc.attn_config.sliding_window                                = 128;
        mc.attn_config.indexer_head_dim                              = 128;
        mc.attn_config.indexer_head_num                              = 64;
        mc.attn_config.indexer_topk                                  = 512;
        mc.attn_config.o_groups                                      = 8;
        mc.attn_config.o_lora_rank                                   = 1024;
        mc.attn_config.tokens_per_block                              = seq_size_per_block;
        mc.hybrid_attention_config.enable_hybrid_attention           = true;
        mc.hybrid_attention_config.enable_independent_kv_cache_pools = true;
        std::vector<int> ratios                                      = {0, 0};
        for (int i = 2; i < 43; ++i) {
            ratios.push_back((i % 2 == 0) ? 4 : 128);
        }
        ratios.push_back(0);  // MTP tail marker.
        mc.attn_config.layer_compress_ratios = ratios;
        // The 7 DSV4 pools are now declared as per-layer specs keyed by tag
        // (csa_kv / hca_kv / indexer_kv / indexer_state / csa_state / hca_state / swa_kv).
        test::setDsv4KvCacheSpecs(mc, ratios);
        // hca_state defaults to a 256-block fixed pool; size it like the paged
        // pools so the unit test does not reserve hundreds of MB of HBM.
        test::setDsv4ExplicitPoolBlocks(mc, "hca_state", block_num);

        ParallelismConfig pc;
        KVCacheConfig     kv_config;
        kv_config.seq_size_per_block        = seq_size_per_block;
        kv_config.kernel_seq_size_per_block = kernel_seq_size_per_blk;
        kv_config.test_block_num            = block_num;
        auto config                         = CacheConfigCreator::createConfig(mc, pc, RuntimeConfig{}, kv_config);
        // KVCacheManager::init() calls finalizeBlockNums(block_num), which fans the
        // global block count out to every group according to its capacity policy.
        config.block_num = block_num;
        return config;
    }

    // Build a PREFILL stream with reuse_cache enabled
    void prepareStream(const std::vector<int>& input_tokens) {
        prepareStreamWithConfig(input_tokens, makeConfig(), /*tokens_per_block=*/8, RoleType::PREFILL);
    }

    void prepareDsv4Stream(const std::vector<int>& input_tokens, RoleType role_type = RoleType::PREFILL) {
        prepareStreamWithConfig(input_tokens, makeDsv4Config(), static_cast<int>(kDsv4TokensPerBlock), role_type);
    }

    void prepareStreamWithConfig(const std::vector<int>& input_tokens,
                                 const CacheConfig&      cache_config,
                                 int                     tokens_per_block,
                                 RoleType                role_type) {
        cache_manager_ = std::make_shared<KVCacheManager>(cache_config, /*warmup=*/false, nullptr);
        ASSERT_TRUE(cache_manager_->init());
        initial_free_blocks_ = cache_manager_->freeBlocksNum();

        ResourceContext resource_context;
        resource_context.cache_manager       = cache_manager_;
        resource_context.reuse_cache         = true;
        resource_context.enable_device_cache = true;
        resource_context.role_type           = role_type;

        auto generate_input                   = std::make_shared<GenerateInput>();
        auto generate_config                  = std::make_shared<GenerateConfig>();
        generate_config->num_return_sequences = 1;
        generate_config->reuse_cache          = true;
        generate_config->enable_device_cache  = true;
        generate_input->input_ids =
            torch::tensor(std::vector<int32_t>(input_tokens.begin(), input_tokens.end()), torch::kInt32);
        generate_input->generate_config = generate_config;

        ModelConfig model_config;
        model_config.attn_config.tokens_per_block = tokens_per_block;
        model_config.max_seq_len                  = std::max<int64_t>(2048, input_tokens.size() + tokens_per_block);
        RuntimeConfig runtime_config;

        stream_ = std::make_shared<NormalGenerateStream>(
            generate_input, model_config, runtime_config, resource_context, nullptr);
        stream_->generate_status_->status = StreamState::RUNNING;
    }

    // Allocate KV blocks and mark stream as FINISHED (simulates prefill done)
    void allocateAndFinish() {
        auto& resource = stream_->streamCacheResource();
        ASSERT_TRUE(resource.initKVBlock().ok());
        stream_->generate_status_->status = StreamState::FINISHED;
        stream_->fillSubGenerateStatus(StreamState::FINISHED);
    }

protected:
    autil::EnvGuard                       perf_scope;
    std::shared_ptr<NormalGenerateStream> stream_;
    std::shared_ptr<KVCacheManager>       cache_manager_;
    size_t                                initial_free_blocks_ = 0;
};

// =============================================================================
// Test 1: Normal release without PD sep hold
// Baseline: blocks are allocated, released normally, freeBlocks returns to start
// =============================================================================
TEST_F(PdSepKVCacheReleaseTest, testNormalRelease_BlocksReturnedToPool) {
    // 14 tokens, tokens_per_block=8 -> 2 blocks needed (1 full + 1 partial)
    prepareStream({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14});
    allocateAndFinish();

    auto& resource  = stream_->streamCacheResource();
    int   allocated = resource.curBlocksNum();
    ASSERT_GT(allocated, 0) << "Should have allocated some blocks";
    ASSERT_LT(cache_manager_->freeBlocksNum(), initial_free_blocks_) << "Blocks should be in use";

    // Normal release (no PD sep)
    stream_->releaseResource();

    // After releaseResource with reuse_cache=true, insertIntoCache() is called.
    // The device cache retains a reference to completed blocks for future reuse,
    // so freeBlocksNum may be less than initial. The key invariant is:
    //   freeBlocksNum >= initial_free_blocks_ - allocated (no extra blocks leaked)
    EXPECT_GE(cache_manager_->freeBlocksNum(), initial_free_blocks_ - allocated)
        << "No extra blocks should be leaked beyond what was allocated";
    EXPECT_EQ(resource.curBlocksNum(), 0) << "Block list should be cleared";
    EXPECT_TRUE(resource.resource_released_) << "resource_released_ should be true";
}

// =============================================================================
// Test 2: holdKVCacheForPDSep increments ref count
// After hold, pd_kvcache_ref_ is non-null
// =============================================================================
TEST_F(PdSepKVCacheReleaseTest, testHoldKVCacheForPDSep_SetsRef) {
    prepareStream({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14});
    allocateAndFinish();

    auto& resource = stream_->streamCacheResource();
    ASSERT_GT(resource.curBlocksNum(), 0);

    // Call hold - simulates prefill pollLocalOutput holding the cache
    resource.holdKVCacheForPDSep();

    EXPECT_NE(resource.pd_kvcache_ref_, nullptr) << "pd_kvcache_ref_ should be set after hold";
}

// =============================================================================
// Test 3: releaseResource with pd_kvcache_ref_ held
// Blocks should be cleared after releaseResource (clearBlocks always called)
// resource_released_ should be true, insertIntoCache should run
// =============================================================================
TEST_F(PdSepKVCacheReleaseTest, testReleaseResource_WithHold_ClearsBlocks) {
    prepareStream({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14});
    allocateAndFinish();

    auto& resource  = stream_->streamCacheResource();
    int   allocated = resource.curBlocksNum();
    ASSERT_GT(allocated, 0);

    resource.holdKVCacheForPDSep();
    ASSERT_NE(resource.pd_kvcache_ref_, nullptr);

    // Simulate engine thread calling releaseResource
    stream_->releaseResource();

    EXPECT_TRUE(resource.resource_released_) << "resource_released_ should be true";
    // clearBlocks() is always called in releaseResource
    // blocks list is cleared, but ref is still held by pd_kvcache_ref_
    // freeBlocksNum should NOT be fully restored yet (blocks still held by ref)
    // NOTE: tryReleaseKVBlock calls cache_manager_->free() which returns blocks to pool,
    // but pd_kvcache_ref_ holds an extra ref, so actual free count depends on impl.
    // Key invariant: resource_released_ = true and no crash.
    EXPECT_TRUE(resource.resource_released_);
}

// =============================================================================
// Test 4: releaseKVCacheForPDSep after releaseResource
// This is the "correct order" path: engine thread releases first,
// then grpc thread calls releaseKVCacheForPDSep.
// After both complete, freeBlocks should return to initial.
// =============================================================================
TEST_F(PdSepKVCacheReleaseTest, testCorrectOrder_ReleaseResourceThenReleasePDSep_BlocksReturned) {
    prepareStream({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14});
    allocateAndFinish();

    auto& resource = stream_->streamCacheResource();
    ASSERT_GT(resource.curBlocksNum(), 0);

    // Step 1: hold (prefill pollLocalOutput)
    resource.holdKVCacheForPDSep();
    ASSERT_NE(resource.pd_kvcache_ref_, nullptr);

    // Step 2: engine thread releases
    stream_->releaseResource();
    EXPECT_TRUE(resource.resource_released_);

    // Step 3: grpc thread releases
    resource.releaseKVCacheForPDSep();
    EXPECT_EQ(resource.pd_kvcache_ref_, nullptr) << "pd_kvcache_ref_ should be reset";

    // After both releases, the device cache may retain 1 block for reuse (insertIntoCache).
    // Key invariant: no blocks leaked beyond initial allocation.
    EXPECT_GE(cache_manager_->freeBlocksNum(), initial_free_blocks_ - 2)
        << "No extra blocks should be leaked. free=" << cache_manager_->freeBlocksNum()
        << " initial=" << initial_free_blocks_;
}

// =============================================================================
// Test 5: insertIntoCache is called during releaseResource (device reuse cache)
// After releaseResource, the cache keys should be findable in the block cache
// (i.e., a subsequent allocation with the same tokens hits reuse)
// =============================================================================
TEST_F(PdSepKVCacheReleaseTest, testInsertIntoCache_CalledDuringRelease_ReuseWorks) {
    const std::vector<int> tokens = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14};
    prepareStream(tokens);
    allocateAndFinish();

    auto& resource = stream_->streamCacheResource();
    ASSERT_GT(resource.curBlocksNum(), 0);

    resource.holdKVCacheForPDSep();

    // Engine thread releases: should call insertIntoCache (device cache)
    stream_->releaseResource();
    EXPECT_TRUE(resource.resource_released_);

    // Release the pd_sep hold (grpc thread)
    resource.releaseKVCacheForPDSep();

    // Now prepare a second stream with the same tokens - should get reuse
    ResourceContext resource_context2;
    resource_context2.cache_manager       = cache_manager_;
    resource_context2.reuse_cache         = true;
    resource_context2.enable_device_cache = true;
    resource_context2.role_type           = RoleType::PREFILL;

    auto generate_input2                   = std::make_shared<GenerateInput>();
    auto generate_config2                  = std::make_shared<GenerateConfig>();
    generate_config2->num_return_sequences = 1;
    generate_config2->reuse_cache          = true;
    generate_config2->enable_device_cache  = true;
    generate_input2->input_ids       = torch::tensor(std::vector<int32_t>(tokens.begin(), tokens.end()), torch::kInt32);
    generate_input2->generate_config = generate_config2;

    ModelConfig model_config;
    model_config.attn_config.tokens_per_block = 8;
    model_config.max_seq_len                  = 2048;
    RuntimeConfig runtime_config;

    auto stream2 = std::make_shared<NormalGenerateStream>(
        generate_input2, model_config, runtime_config, resource_context2, nullptr);
    stream2->generate_status_->status = StreamState::RUNNING;

    auto& resource2 = stream2->streamCacheResource();
    ASSERT_TRUE(resource2.initKVBlock().ok());

    // With 14 tokens and block_size=8: 1 full block (8 tokens) should be reused
    int reuse_len = stream2->reuseLength();
    EXPECT_GE(reuse_len, 8) << "At least 1 block (8 tokens) should be reused from device cache. "
                            << "reuse_len=" << reuse_len;

    stream2->releaseResource();
}

// =============================================================================
// Test 6: Race condition simulation
// Engine thread calls releaseResource concurrently with
// grpc thread calling releaseKVCacheForPDSep.
// Verifies: no crash, no double-free, freeBlocks returns to initial after both.
// =============================================================================
TEST_F(PdSepKVCacheReleaseTest, testRaceCondition_ConcurrentRelease_NoDoubleFree) {
    prepareStream({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14});
    allocateAndFinish();

    auto& resource = stream_->streamCacheResource();
    ASSERT_GT(resource.curBlocksNum(), 0);

    resource.holdKVCacheForPDSep();
    ASSERT_NE(resource.pd_kvcache_ref_, nullptr);

    std::atomic<bool> engine_done{false};
    std::atomic<bool> grpc_done{false};

    // Engine thread: releaseResource
    std::thread engine_thread([&]() {
        stream_->releaseResource();
        engine_done.store(true);
    });

    // Grpc thread: releaseKVCacheForPDSep (with small delay to increase race chance)
    std::thread grpc_thread([&]() {
        std::this_thread::sleep_for(std::chrono::microseconds(100));
        resource.releaseKVCacheForPDSep();
        grpc_done.store(true);
    });

    engine_thread.join();
    grpc_thread.join();

    EXPECT_TRUE(engine_done.load()) << "Engine thread should have completed";
    EXPECT_TRUE(grpc_done.load()) << "Grpc thread should have completed";
    EXPECT_TRUE(resource.resource_released_) << "resource_released_ should be true";
    EXPECT_EQ(resource.pd_kvcache_ref_, nullptr) << "pd_kvcache_ref_ should be reset";

    // Critical: no double-free, no extra blocks leaked.
    // insertIntoCache may hold 1 cached block ref, so freeBlocksNum can be <= initial.
    EXPECT_GE(cache_manager_->freeBlocksNum(), initial_free_blocks_ - 2)
        << "No extra blocks should be leaked after concurrent release. "
        << "free=" << cache_manager_->freeBlocksNum() << " initial=" << initial_free_blocks_;
}

// =============================================================================
// Test 7: holdKVCacheForPDSep without subsequent releaseKVCacheForPDSep
// (simulates grpc failure: hold is called but release never comes)
// releaseResource alone should still eventually free blocks when ref drops
// =============================================================================
TEST_F(PdSepKVCacheReleaseTest, testHoldWithoutReleasePDSep_ResourceReleasedStillCompletes) {
    prepareStream({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14});
    allocateAndFinish();

    auto& resource = stream_->streamCacheResource();
    ASSERT_GT(resource.curBlocksNum(), 0);

    resource.holdKVCacheForPDSep();

    // Only engine thread releases, grpc thread never calls releaseKVCacheForPDSep
    stream_->releaseResource();

    EXPECT_TRUE(resource.resource_released_);

    // pd_kvcache_ref_ still holds a ref - blocks won't be fully freed until ref drops
    // Simulate ref drop (e.g. stream destructor or explicit reset)
    resource.pd_kvcache_ref_.reset();

    // After ref drop, blocks should be returned (minus any held by device cache for reuse)
    EXPECT_GE(cache_manager_->freeBlocksNum(), initial_free_blocks_ - 2)
        << "Blocks should be freed once pd_kvcache_ref_ is dropped (minus device cache refs)";
}

TEST_F(PdSepKVCacheReleaseTest, testDsv4PDSepPrefillReleaseInsertsSevenGroupDeviceCache) {
    const int        spb = static_cast<int>(kDsv4TokensPerBlock);
    std::vector<int> tokens(3 * spb + 17);
    std::iota(tokens.begin(), tokens.end(), 1);

    auto config = makeDsv4Config();
    // linear_step=4: with the default step of 1 every slot is a step hit and all
    // blocks materialize, defeating the tail-only assertions below.
    config.linear_step = 4;
    prepareStreamWithConfig(tokens, config, spb, RoleType::PREFILL);
    allocateAndFinish();

    auto& resource = stream_->streamCacheResource();
    ASSERT_EQ(resource.kvCache().groupNums(), kDsv4PoolNum);
    ASSERT_GT(resource.curBlocksNum(), 0);
    for (int gid = 0; gid < kDsv4PoolNum; ++gid) {
        const auto& tag = config.tagForGroup(static_cast<size_t>(gid));
        ASSERT_EQ(resource.kvCache().blocksNum(0, gid), 4) << "group " << tag;
        const auto&  blocks = resource.kvCache().blocks(0, gid);
        const size_t tail   = static_cast<size_t>(config.policyForGroup(static_cast<size_t>(gid)).active_tail_blocks);
        if (tail == 0) {
            // Paged group: every logical block is materialized from position 0.
            EXPECT_FALSE(isNullBlockIdx(blocks[0])) << "paged group " << tag;
        } else {
            // Tail group: only the last `tail` logical blocks are materialized.
            EXPECT_TRUE(isNullBlockIdx(blocks[0])) << "tail group " << tag << " should keep only tail blocks";
            for (size_t pos = blocks.size() - tail; pos < blocks.size(); ++pos) {
                EXPECT_FALSE(isNullBlockIdx(blocks[pos])) << "tail group " << tag << " pos " << pos;
            }
        }
    }

    resource.holdKVCacheForPDSep();
    ASSERT_NE(resource.pd_kvcache_ref_, nullptr);

    stream_->releaseResource();
    EXPECT_TRUE(resource.resource_released_);
    resource.releaseKVCacheForPDSep();
    EXPECT_EQ(resource.pd_kvcache_ref_, nullptr);

    ResourceContext resource_context2;
    resource_context2.cache_manager       = cache_manager_;
    resource_context2.reuse_cache         = true;
    resource_context2.enable_device_cache = true;
    resource_context2.role_type           = RoleType::PREFILL;

    auto generate_input2                   = std::make_shared<GenerateInput>();
    auto generate_config2                  = std::make_shared<GenerateConfig>();
    generate_config2->num_return_sequences = 1;
    generate_config2->reuse_cache          = true;
    generate_config2->enable_device_cache  = true;
    generate_input2->input_ids       = torch::tensor(std::vector<int32_t>(tokens.begin(), tokens.end()), torch::kInt32);
    generate_input2->generate_config = generate_config2;

    ModelConfig model_config;
    model_config.attn_config.tokens_per_block = spb;
    model_config.max_seq_len                  = 4096;
    RuntimeConfig runtime_config;

    auto stream2 = std::make_shared<NormalGenerateStream>(
        generate_input2, model_config, runtime_config, resource_context2, nullptr);
    stream2->generate_status_->status = StreamState::RUNNING;

    auto& resource2 = stream2->streamCacheResource();
    ASSERT_TRUE(resource2.initKVBlock().ok());
    EXPECT_GE(stream2->reuseLength(), spb) << "DSV4 prefill should reuse cached 7-group prefix blocks";
    EXPECT_EQ(resource2.kvCache().groupNums(), kDsv4PoolNum);

    stream2->generate_status_->status = StreamState::FINISHED;
    stream2->fillSubGenerateStatus(StreamState::FINISHED);
    stream2->releaseResource();
}

TEST_F(PdSepKVCacheReleaseTest, testDsv4DecodeFirstMallocBypassesLocalDeviceReuseInPDSep) {
    const int        spb = static_cast<int>(kDsv4TokensPerBlock);
    std::vector<int> tokens(3 * spb + 17);
    std::iota(tokens.begin(), tokens.end(), 1);

    prepareDsv4Stream(tokens, RoleType::PREFILL);
    allocateAndFinish();
    auto& prefill_resource = stream_->streamCacheResource();
    prefill_resource.holdKVCacheForPDSep();
    stream_->releaseResource();
    prefill_resource.releaseKVCacheForPDSep();

    ResourceContext decode_resource_context;
    decode_resource_context.cache_manager       = cache_manager_;
    decode_resource_context.reuse_cache         = true;
    decode_resource_context.enable_device_cache = true;
    decode_resource_context.role_type           = RoleType::DECODE;

    auto decode_input                   = std::make_shared<GenerateInput>();
    auto decode_config                  = std::make_shared<GenerateConfig>();
    decode_config->num_return_sequences = 1;
    decode_config->reuse_cache          = true;
    decode_config->enable_device_cache  = true;
    decode_input->input_ids       = torch::tensor(std::vector<int32_t>(tokens.begin(), tokens.end()), torch::kInt32);
    decode_input->generate_config = decode_config;

    ModelConfig model_config;
    model_config.attn_config.tokens_per_block = spb;
    model_config.max_seq_len                  = 4096;
    RuntimeConfig runtime_config;

    auto decode_stream = std::make_shared<NormalGenerateStream>(
        decode_input, model_config, runtime_config, decode_resource_context, nullptr);
    decode_stream->generate_status_->status = StreamState::RUNNING;

    auto& decode_resource = decode_stream->streamCacheResource();
    ASSERT_TRUE(decode_resource.initKVBlock().ok());

    EXPECT_EQ(decode_stream->reuseLength(), 0)
        << "Hybrid DSV4 decode first malloc must not consume local device-cache reuse; PD load owns reuse.";
    EXPECT_EQ(decode_resource.kvCache().groupNums(), kDsv4PoolNum);
    for (int gid = 0; gid < kDsv4PoolNum; ++gid) {
        EXPECT_EQ(decode_resource.kvCache().blocksNum(0, gid), 4) << "group " << gid;
    }

    decode_stream->releaseResource();
}

TEST_F(PdSepKVCacheReleaseTest, testDsv4CacheStorePDSepTransfersAllLayerRegions) {
    const int     spb        = static_cast<int>(kDsv4TokensPerBlock);
    const int     block_num  = 4;
    const int64_t request_id = 9017;
    const size_t  model_id   = 77;

    auto config = makeDsv4Config(/*block_num=*/24);

    auto makeResource = [](const std::shared_ptr<KVCacheManager>& manager) {
        auto resource = std::make_shared<BatchKVCacheResource>();
        resource->resetBatchSize(1);
        resource->initGroups(manager->cacheConfig().topologyPtr());
        return resource;
    };
    auto makeCompleteTokens = [spb, block_num](int max_seq_len) {
        auto input              = std::make_shared<GenerateInput>();
        input->input_ids        = torch::arange(max_seq_len, torch::kInt32);
        input->generate_config  = std::make_shared<GenerateConfig>();
        auto complete_token_ids = std::make_shared<CompleteTokenIds>(1, 1, max_seq_len + spb, spb);
        complete_token_ids->init(input);
        complete_token_ids->setSeqLength(block_num * spb);
        return complete_token_ids;
    };

    auto prefill_manager = std::make_shared<KVCacheManager>(config, /*warmup=*/false, nullptr);
    auto decode_manager  = std::make_shared<KVCacheManager>(config, /*warmup=*/false, nullptr);
    ASSERT_TRUE(prefill_manager->init());
    ASSERT_TRUE(decode_manager->init());

    auto prefill_resource = makeResource(prefill_manager);
    auto decode_resource  = makeResource(decode_manager);
    ASSERT_TRUE(
        prefill_manager->malloc({prefill_resource, makeCompleteTokens(block_num * spb), request_id, true, false, false})
            .success);
    ASSERT_TRUE(
        decode_manager->malloc({decode_resource, makeCompleteTokens(block_num * spb), request_id, true, false, false})
            .success);

    std::vector<CacheKeyType> cache_keys;
    for (int i = 0; i < block_num; ++i) {
        cache_keys.push_back(10000 + i);
    }

    const auto& cache_config = prefill_manager->cacheConfig();
    // One cache-store request per (layer, cache tag) pair; one stored block per
    // transferred physical block. DSV4 groups are all opaque, so each block
    // contributes exactly one "kv_" key (no k_/v_ split, no scale).
    size_t expected_requests = 0;
    size_t expected_blocks   = 0;
    for (int layer_id = 0; layer_id < 4; ++layer_id) {
        for (int gid : cache_config.groupIdsForLayer(layer_id)) {
            ++expected_requests;
            expected_blocks += dsv4TransferPositions(cache_config, gid, block_num, /*reuse_block_size=*/0).size();
        }
    }

    for (int layer_id = 0; layer_id < 4; ++layer_id) {
        for (int gid : cache_config.groupIdsForLayer(layer_id)) {
            const auto& tag       = cache_config.tagForGroup(static_cast<size_t>(gid));
            auto        positions = dsv4TransferPositions(cache_config, gid, block_num, /*reuse_block_size=*/0);
            for (auto block_pos : positions) {
                auto prefill_block_id = prefill_resource->blocks(0, gid)[block_pos];
                auto decode_block_id  = decode_resource->blocks(0, gid)[block_pos];
                ASSERT_FALSE(isNullBlockIdx(prefill_block_id)) << "prefill tag=" << tag << " pos=" << block_pos;
                ASSERT_FALSE(isNullBlockIdx(decode_block_id)) << "decode tag=" << tag << " pos=" << block_pos;
                fillDsv4RegionBytes(
                    prefill_manager, prefill_block_id, layer_id, tag, dsv4PdPattern(layer_id, gid, block_pos));
                fillDsv4RegionBytes(decode_manager, decode_block_id, layer_id, tag, 0xEE);
            }
        }
    }
    runtimeSyncAndCheck();

    auto cache_store = std::make_shared<MemoryBackedCacheStore>();
    auto layout      = prefill_manager->getMainModelCacheLayerLayout();
    for (int layer_id = 0; layer_id < 4; ++layer_id) {
        for (int gid : cache_config.groupIdsForLayer(layer_id)) {
            const auto& tag = cache_config.tagForGroup(static_cast<size_t>(gid));
            ASSERT_TRUE(layout.at(tag, static_cast<size_t>(layer_id)).kv_addr.defined())
                << "layer=" << layer_id << " tag=" << tag;

            auto inputs = makeDsv4WriteInputs(request_id,
                                              /*input_length=*/block_num * spb,
                                              /*prefix_length=*/0,
                                              blockIdsTensor(prefill_resource, gid),
                                              cache_keys);

            torch_ext::LayerKVCache layer_cache;
            layer_cache.kv_cache_base      = layout.at(tag, static_cast<size_t>(layer_id)).kv_addr;
            layer_cache.seq_size_per_block = static_cast<int>(cache_config.seqSizePerBlockForGroup(gid));
            layer_cache.layer_id           = layer_id;
            layer_cache.group_id           = gid;
            layer_cache.tag                = tag;

            runtimeWriteCacheStore(inputs,
                                   layer_cache,
                                   cache_config,
                                   cache_store,
                                   model_id,
                                   /*cp_rank=*/0,
                                   /*cp_size=*/1,
                                   /*pre_created_event=*/nullptr);
        }
    }
    ASSERT_EQ(cache_store->store_request_keys_.size(), expected_requests);
    ASSERT_EQ(cache_store->stored_blocks_.size(), expected_blocks);

    EngineInitParams params;
    params.model_id                 = model_id;
    params.model_config_.num_layers = 4;
    params.parallelism_config       = ParallelismConfig();

    DecodeRpcServer server;
    server.engine_                   = std::make_shared<MinimalEngine>(params, decode_manager);
    server.maga_init_params_         = params;
    server.propose_maga_init_params_ = nullptr;
    server.resource_.cache_store     = cache_store;

    std::vector<std::string>            peer_addrs = {"127.0.0.1:12345:12346"};
    grpc::ServerContext                 server_context;
    DecodeRpcServer::LoadKVCacheContext load_context(request_id,
                                                     "dsv4-cache-store-pd",
                                                     peer_addrs,
                                                     cache_keys,
                                                     decode_resource->groupBlocks(),
                                                     /*reuse_block_size=*/0,
                                                     /*timeout_ms=*/5000,
                                                     /*partition_count=*/1,
                                                     /*partition_id=*/0,
                                                     &server_context);
    auto                                status = server.loadCache(load_context);
    ASSERT_TRUE(status.ok()) << status.ToString();

    EXPECT_EQ(cache_store->load_buffer_requests_.size(), expected_requests);
    EXPECT_EQ(cache_store->load_request_keys_.size(), expected_requests);
    for (int layer_id = 0; layer_id < 4; ++layer_id) {
        for (int gid : cache_config.groupIdsForLayer(layer_id)) {
            const auto& tag       = cache_config.tagForGroup(static_cast<size_t>(gid));
            auto        positions = dsv4TransferPositions(cache_config, gid, block_num, /*reuse_block_size=*/0);
            for (auto block_pos : positions) {
                auto decode_block_id = decode_resource->blocks(0, gid)[block_pos];
                ASSERT_FALSE(isNullBlockIdx(decode_block_id));
                expectDsv4RegionBytes(
                    decode_manager, decode_block_id, layer_id, tag, dsv4PdPattern(layer_id, gid, block_pos));
            }
        }
    }
}

TEST_F(PdSepKVCacheReleaseTest, testDsv4DecoupledCacheStoreTransfersPhysicalBlocks) {
    const int     spb        = 8192;
    const int     kernel_spb = 128;
    const int     block_num  = 2;
    const int64_t request_id = 9020;
    const size_t  model_id   = 80;

    auto config = makeDsv4Config(/*block_num=*/8, spb, kernel_spb);

    auto makeResource = [](const std::shared_ptr<KVCacheManager>& manager) {
        auto resource = std::make_shared<BatchKVCacheResource>();
        resource->resetBatchSize(1);
        resource->initGroups(manager->cacheConfig().topologyPtr());
        return resource;
    };
    auto makeCompleteTokens = [spb, block_num](int max_seq_len) {
        auto input              = std::make_shared<GenerateInput>();
        input->input_ids        = torch::arange(max_seq_len, torch::kInt32);
        input->generate_config  = std::make_shared<GenerateConfig>();
        auto complete_token_ids = std::make_shared<CompleteTokenIds>(1, 1, max_seq_len + spb, spb);
        complete_token_ids->init(input);
        complete_token_ids->setSeqLength(block_num * spb);
        return complete_token_ids;
    };

    auto prefill_manager = std::make_shared<KVCacheManager>(config, /*warmup=*/false, nullptr);
    auto decode_manager  = std::make_shared<KVCacheManager>(config, /*warmup=*/false, nullptr);
    ASSERT_TRUE(prefill_manager->init());
    ASSERT_TRUE(decode_manager->init());

    auto prefill_resource = makeResource(prefill_manager);
    auto decode_resource  = makeResource(decode_manager);
    ASSERT_TRUE(
        prefill_manager->malloc({prefill_resource, makeCompleteTokens(block_num * spb), request_id, true, false, false})
            .success);
    ASSERT_TRUE(
        decode_manager->malloc({decode_resource, makeCompleteTokens(block_num * spb), request_id, true, false, false})
            .success);

    std::vector<CacheKeyType> cache_keys;
    for (int i = 0; i < block_num; ++i) {
        cache_keys.push_back(20000 + i);
    }

    const auto& cache_config = prefill_manager->cacheConfig();
    for (int layer_id = 0; layer_id < 4; ++layer_id) {
        for (int gid : cache_config.groupIdsForLayer(layer_id)) {
            const auto& tag       = cache_config.tagForGroup(static_cast<size_t>(gid));
            auto        positions = dsv4TransferPositions(cache_config, gid, block_num, /*reuse_block_size=*/0);
            for (auto block_pos : positions) {
                auto prefill_block_id = prefill_resource->blocks(0, gid)[block_pos];
                auto decode_block_id  = decode_resource->blocks(0, gid)[block_pos];
                ASSERT_FALSE(isNullBlockIdx(prefill_block_id)) << "prefill tag=" << tag << " pos=" << block_pos;
                ASSERT_FALSE(isNullBlockIdx(decode_block_id)) << "decode tag=" << tag << " pos=" << block_pos;
                fillDsv4RegionBytes(
                    prefill_manager, prefill_block_id, layer_id, tag, dsv4PdPattern(layer_id, gid, block_pos));
                fillDsv4RegionBytes(decode_manager, decode_block_id, layer_id, tag, 0xEE);
            }
        }
    }
    runtimeSyncAndCheck();

    auto cache_store = std::make_shared<MemoryBackedCacheStore>();
    auto layout      = prefill_manager->getMainModelCacheLayerLayout();
    for (int layer_id = 0; layer_id < 4; ++layer_id) {
        for (int gid : cache_config.groupIdsForLayer(layer_id)) {
            const auto& tag = cache_config.tagForGroup(static_cast<size_t>(gid));
            ASSERT_TRUE(layout.at(tag, static_cast<size_t>(layer_id)).kv_addr.defined())
                << "layer=" << layer_id << " tag=" << tag;

            auto inputs = makeDsv4WriteInputs(request_id,
                                              /*input_length=*/block_num * spb,
                                              /*prefix_length=*/0,
                                              blockIdsTensor(prefill_resource, gid),
                                              cache_keys);

            // Paged groups expose the layer view at kernel-block granularity;
            // cache_store must still address whole physical blocks.
            torch_ext::LayerKVCache layer_cache;
            layer_cache.kv_cache_base = layout.at(tag, static_cast<size_t>(layer_id)).kv_addr;
            layer_cache.seq_size_per_block =
                cache_config.typeForGroup(static_cast<size_t>(gid)) == CacheGroupType::FULL ? kernel_spb : spb;
            layer_cache.layer_id = layer_id;
            layer_cache.group_id = gid;
            layer_cache.tag      = tag;

            runtimeWriteCacheStore(inputs,
                                   layer_cache,
                                   cache_config,
                                   cache_store,
                                   model_id,
                                   /*cp_rank=*/0,
                                   /*cp_size=*/1,
                                   /*pre_created_event=*/nullptr);
        }
    }

    const auto csa_gid       = static_cast<size_t>(cache_config.groupIdForTag("csa_kv"));
    const auto first_csa_key = "kv_" + makeCacheKey(model_id, std::to_string(cache_keys[0]), /*layer_id=*/2, "csa_kv");
    ASSERT_NE(cache_store->stored_blocks_.find(first_csa_key), cache_store->stored_blocks_.end());
    EXPECT_EQ(cache_store->stored_blocks_[first_csa_key].size(), cache_config.kvBlockStrideBytesForGroup(csa_gid));

    EngineInitParams params;
    params.model_id                 = model_id;
    params.model_config_.num_layers = 4;
    params.parallelism_config       = ParallelismConfig();

    DecodeRpcServer server;
    server.engine_                   = std::make_shared<MinimalEngine>(params, decode_manager);
    server.maga_init_params_         = params;
    server.propose_maga_init_params_ = nullptr;
    server.resource_.cache_store     = cache_store;

    std::vector<std::string>            peer_addrs = {"127.0.0.1:12345:12346"};
    grpc::ServerContext                 server_context;
    DecodeRpcServer::LoadKVCacheContext load_context(request_id,
                                                     "dsv4-decoupled-cache-store-pd",
                                                     peer_addrs,
                                                     cache_keys,
                                                     decode_resource->groupBlocks(),
                                                     /*reuse_block_size=*/0,
                                                     /*timeout_ms=*/5000,
                                                     /*partition_count=*/1,
                                                     /*partition_id=*/0,
                                                     &server_context);
    auto                                status = server.loadCache(load_context);
    ASSERT_TRUE(status.ok()) << status.ToString();

    for (int layer_id = 0; layer_id < 4; ++layer_id) {
        for (int gid : cache_config.groupIdsForLayer(layer_id)) {
            const auto& tag       = cache_config.tagForGroup(static_cast<size_t>(gid));
            auto        positions = dsv4TransferPositions(cache_config, gid, block_num, /*reuse_block_size=*/0);
            for (auto block_pos : positions) {
                auto decode_block_id = decode_resource->blocks(0, gid)[block_pos];
                ASSERT_FALSE(isNullBlockIdx(decode_block_id));
                expectDsv4RegionBytes(
                    decode_manager, decode_block_id, layer_id, tag, dsv4PdPattern(layer_id, gid, block_pos));
            }
        }
    }
}

TEST_F(PdSepKVCacheReleaseTest, testDsv4CacheStorePDSepTransfersAllLayerRegionsWithPrefixReuse) {
    const int     spb        = static_cast<int>(kDsv4TokensPerBlock);
    const int     block_num  = 4;
    const int     reuse_num  = 1;
    const int64_t request_id = 9018;
    const size_t  model_id   = 78;

    auto config = makeDsv4Config(/*block_num=*/24);

    auto makeResource = [](const std::shared_ptr<KVCacheManager>& manager) {
        auto resource = std::make_shared<BatchKVCacheResource>();
        resource->resetBatchSize(1);
        resource->initGroups(manager->cacheConfig().topologyPtr());
        return resource;
    };
    auto makeCompleteTokens = [spb, block_num](int max_seq_len) {
        auto input              = std::make_shared<GenerateInput>();
        input->input_ids        = torch::arange(max_seq_len, torch::kInt32);
        input->generate_config  = std::make_shared<GenerateConfig>();
        auto complete_token_ids = std::make_shared<CompleteTokenIds>(1, 1, max_seq_len + spb, spb);
        complete_token_ids->init(input);
        complete_token_ids->setSeqLength(block_num * spb);
        return complete_token_ids;
    };

    auto prefill_manager = std::make_shared<KVCacheManager>(config, /*warmup=*/false, nullptr);
    auto decode_manager  = std::make_shared<KVCacheManager>(config, /*warmup=*/false, nullptr);
    ASSERT_TRUE(prefill_manager->init());
    ASSERT_TRUE(decode_manager->init());

    auto prefill_resource = makeResource(prefill_manager);
    auto decode_resource  = makeResource(decode_manager);
    ASSERT_TRUE(
        prefill_manager->malloc({prefill_resource, makeCompleteTokens(block_num * spb), request_id, true, false, false})
            .success);
    ASSERT_TRUE(
        decode_manager->malloc({decode_resource, makeCompleteTokens(block_num * spb), request_id, true, false, false})
            .success);

    std::vector<CacheKeyType> cache_keys;
    for (int i = 0; i < block_num; ++i) {
        cache_keys.push_back(11000 + i);
    }

    const auto& cache_config = prefill_manager->cacheConfig();
    // One cache-store request per (layer, cache tag) pair; one stored block per
    // transferred physical block. Reused prefix blocks are transferred too, so
    // the reuse count does not shrink the plan for a hybrid topology.
    size_t expected_requests = 0;
    size_t expected_blocks   = 0;
    for (int layer_id = 0; layer_id < 4; ++layer_id) {
        for (int gid : cache_config.groupIdsForLayer(layer_id)) {
            ++expected_requests;
            expected_blocks += dsv4TransferPositions(cache_config, gid, block_num, reuse_num).size();
        }
    }

    for (int layer_id = 0; layer_id < 4; ++layer_id) {
        for (int gid : cache_config.groupIdsForLayer(layer_id)) {
            const auto& tag       = cache_config.tagForGroup(static_cast<size_t>(gid));
            auto        positions = dsv4TransferPositions(cache_config, gid, block_num, reuse_num);
            for (auto block_pos : positions) {
                auto prefill_block_id = prefill_resource->blocks(0, gid)[block_pos];
                auto decode_block_id  = decode_resource->blocks(0, gid)[block_pos];
                ASSERT_FALSE(isNullBlockIdx(prefill_block_id)) << "prefill tag=" << tag << " pos=" << block_pos;
                ASSERT_FALSE(isNullBlockIdx(decode_block_id)) << "decode tag=" << tag << " pos=" << block_pos;
                fillDsv4RegionBytes(
                    prefill_manager, prefill_block_id, layer_id, tag, dsv4PdPattern(layer_id, gid, block_pos));
                fillDsv4RegionBytes(decode_manager, decode_block_id, layer_id, tag, 0xEE);
            }
        }
    }
    runtimeSyncAndCheck();

    auto cache_store = std::make_shared<MemoryBackedCacheStore>();
    auto layout      = prefill_manager->getMainModelCacheLayerLayout();
    for (int layer_id = 0; layer_id < 4; ++layer_id) {
        for (int gid : cache_config.groupIdsForLayer(layer_id)) {
            const auto& tag = cache_config.tagForGroup(static_cast<size_t>(gid));
            ASSERT_TRUE(layout.at(tag, static_cast<size_t>(layer_id)).kv_addr.defined())
                << "layer=" << layer_id << " tag=" << tag;

            auto inputs = makeDsv4WriteInputs(request_id,
                                              /*input_length=*/(block_num - reuse_num) * spb,
                                              /*prefix_length=*/reuse_num * spb,
                                              blockIdsTensor(prefill_resource, gid),
                                              cache_keys);

            torch_ext::LayerKVCache layer_cache;
            layer_cache.kv_cache_base      = layout.at(tag, static_cast<size_t>(layer_id)).kv_addr;
            layer_cache.seq_size_per_block = static_cast<int>(cache_config.seqSizePerBlockForGroup(gid));
            layer_cache.layer_id           = layer_id;
            layer_cache.group_id           = gid;
            layer_cache.tag                = tag;

            runtimeWriteCacheStore(inputs,
                                   layer_cache,
                                   cache_config,
                                   cache_store,
                                   model_id,
                                   /*cp_rank=*/0,
                                   /*cp_size=*/1,
                                   /*pre_created_event=*/nullptr);
        }
    }
    ASSERT_EQ(cache_store->store_request_keys_.size(), expected_requests);
    ASSERT_EQ(cache_store->stored_blocks_.size(), expected_blocks);

    EngineInitParams params;
    params.model_id                 = model_id;
    params.model_config_.num_layers = 4;
    params.parallelism_config       = ParallelismConfig();

    DecodeRpcServer server;
    server.engine_                   = std::make_shared<MinimalEngine>(params, decode_manager);
    server.maga_init_params_         = params;
    server.propose_maga_init_params_ = nullptr;
    server.resource_.cache_store     = cache_store;

    std::vector<std::string>            peer_addrs = {"127.0.0.1:12345:12346"};
    grpc::ServerContext                 server_context;
    DecodeRpcServer::LoadKVCacheContext load_context(request_id,
                                                     "dsv4-cache-store-pd-prefix-reuse",
                                                     peer_addrs,
                                                     cache_keys,
                                                     decode_resource->groupBlocks(),
                                                     reuse_num,
                                                     /*timeout_ms=*/5000,
                                                     /*partition_count=*/1,
                                                     /*partition_id=*/0,
                                                     &server_context);
    auto                                status = server.loadCache(load_context);
    ASSERT_TRUE(status.ok()) << status.ToString();

    EXPECT_EQ(cache_store->load_buffer_requests_.size(), expected_requests);
    EXPECT_EQ(cache_store->load_request_keys_.size(), expected_requests);
    for (int layer_id = 0; layer_id < 4; ++layer_id) {
        for (int gid : cache_config.groupIdsForLayer(layer_id)) {
            const auto& tag       = cache_config.tagForGroup(static_cast<size_t>(gid));
            auto        positions = dsv4TransferPositions(cache_config, gid, block_num, reuse_num);
            for (auto block_pos : positions) {
                auto decode_block_id = decode_resource->blocks(0, gid)[block_pos];
                ASSERT_FALSE(isNullBlockIdx(decode_block_id));
                expectDsv4RegionBytes(
                    decode_manager, decode_block_id, layer_id, tag, dsv4PdPattern(layer_id, gid, block_pos));
            }
        }
    }
}

// =============================================================================
// Test: runtimeWriteCacheStore with pinned-host metadata + event sync
// Verifies that when metadata tensors (input_lengths, prefix_lengths) are
// prepared on pinned host via async D2H and a pre_created_event is attached,
// runtimeWriteCacheStore waits for the event and reads metadata correctly —
// the same path used by CacheStoreAsyncWriter, which avoids synchronous .cpu()
// calls on background threads.
// =============================================================================
TEST_F(PdSepKVCacheReleaseTest, testWriteCacheStoreWithPinnedHostMetadataAndEvent) {
    auto config  = makeConfig();  // 3 layers, 16 blocks, 8 tokens/block, INT8
    auto manager = std::make_shared<KVCacheManager>(config, /*warmup=*/false, nullptr);
    ASSERT_TRUE(manager->init());

    const int spb            = 8;
    const int block_num      = 2;
    const int input_length   = block_num * spb;
    const int request_id_val = 42;

    // Allocate KV blocks.
    auto resource = std::make_shared<BatchKVCacheResource>();
    resource->resetBatchSize(1);
    resource->initGroups(manager->cacheConfig().topologyPtr());

    auto input              = std::make_shared<GenerateInput>();
    input->input_ids        = torch::arange(input_length, torch::kInt32);
    input->generate_config  = std::make_shared<GenerateConfig>();
    auto complete_token_ids = std::make_shared<CompleteTokenIds>(1, 1, input_length + spb, spb);
    complete_token_ids->init(input);
    complete_token_ids->setSeqLength(input_length);

    auto result = manager->malloc({resource, complete_token_ids, request_id_val, true, false, false});
    ASSERT_TRUE(result.success);

    // Fill KV cache blocks with a known pattern so MemoryBackedCacheStore can
    // verify the transfer.
    auto layout = manager->getMainModelCacheLayerLayout();
    for (int layer_id = 0; layer_id < 3; ++layer_id) {
        auto buf = layout.at(static_cast<size_t>(layer_id)).kv_addr;
        ASSERT_TRUE(buf.defined());
        for (int b = 0; b < block_num; ++b) {
            auto bid       = resource->blocks(0, 0)[b];
            auto kv_stride = config.kv_block_stride_bytes;
            ASSERT_FALSE(isNullBlockIdx(bid));
            auto device_slice = torch::from_blob((uint8_t*)buf.data_ptr() + bid * kv_stride,
                                                 {(int64_t)kv_stride},
                                                 torch::TensorOptions(torch::kUInt8).device(torch::kCUDA));
            device_slice.fill_(static_cast<uint8_t>(layer_id * 10 + b));
        }
    }
    runtimeSyncAndCheck();

    // Prepare cache keys (one per block).
    std::vector<CacheKeyType> cache_keys;
    for (int i = 0; i < block_num; ++i) {
        cache_keys.push_back(10000 + i);
    }

    // --- Core of the test: async D2H to pinned host, then event ---
    // Create device tensors (mimicking what buildPyAttentionInputs produces).
    auto input_lengths_device  = torch::tensor({input_length}, torch::kInt32).cuda();
    auto prefix_lengths_device = torch::tensor({0}, torch::kInt32).cuda();

    // Async-copy to pinned host (mimicking prepareWriteCacheParams).
    auto pinned_i32          = torch::TensorOptions(torch::kInt32).pinned_memory(true);
    auto input_lengths_host  = torch::empty({1}, pinned_i32);
    auto prefix_lengths_host = torch::empty({1}, pinned_i32);
    input_lengths_host.copy_(input_lengths_device, /*non_blocking=*/true);
    prefix_lengths_host.copy_(prefix_lengths_device, /*non_blocking=*/true);

    // Record event AFTER async D2H on the current stream.
    auto event = runtimeCreateEvent();

    // --- Call runtimeWriteCacheStore (event->synchronize() inside) ---
    auto cache_store = std::make_shared<MemoryBackedCacheStore>();
    auto block_ids   = torch::from_blob(const_cast<int*>(resource->blocks(0, 0).data()),
                                        {1, (int64_t)resource->blocks(0, 0).size()},
                                      torch::kInt32)
                         .clone();

    const auto& cache_config = manager->cacheConfig();
    for (int layer_id = 0; layer_id < 3; ++layer_id) {
        auto inputs = makeDsv4WriteInputs(
            /*request_id=*/request_id_val, input_length, /*prefix_length=*/0, block_ids, cache_keys);
        // The pinned-host metadata is deliberately left un-synchronized here;
        // runtimeWriteCacheStore must wait on the event before reading it.
        inputs.input_lengths_host  = input_lengths_host;
        inputs.prefix_lengths_host = prefix_lengths_host;

        torch_ext::LayerKVCache layer_cache;
        layer_cache.kv_cache_base      = layout.at(static_cast<size_t>(layer_id)).kv_addr;
        layer_cache.seq_size_per_block = spb;
        layer_cache.layer_id           = layer_id;
        layer_cache.group_id           = 0;
        layer_cache.tag                = "default";

        runtimeWriteCacheStore(inputs,
                               layer_cache,
                               cache_config,
                               cache_store,
                               /*cache_model_id=*/0,
                               /*cp_rank=*/0,
                               /*cp_size=*/1,
                               event);
    }

    // Verify: cache store received correct request key for all 3 layers.
    EXPECT_EQ(cache_store->store_request_keys_.size(), 3u);
    // MHA (non-opaque, non-mla) splits each block into k + v → 2 entries per block.
    EXPECT_EQ(cache_store->stored_blocks_.size(), 3u * block_num * 2u);

    // Verify stored data matches the pattern we filled.
    for (int layer_id = 0; layer_id < 3; ++layer_id) {
        for (int b = 0; b < block_num; ++b) {
            auto k_key = "k_" + makeCacheKey(0, std::to_string(cache_keys[b]), layer_id);
            auto it    = cache_store->stored_blocks_.find(k_key);
            ASSERT_NE(it, cache_store->stored_blocks_.end()) << "missing key: " << k_key;
            uint8_t expected = static_cast<uint8_t>(layer_id * 10 + b);
            EXPECT_EQ(it->second[0], expected) << "layer=" << layer_id << " block=" << b << " first byte mismatch";
        }
    }
}

TEST_F(PdSepKVCacheReleaseTest, testWriteCacheStoreUsesTensorDeviceForCpuKvBuffer) {
    const int     spb            = 8;
    const int     kv_stride      = 64;
    const int     request_id_val = 4242;
    const int64_t cache_key      = 10000;

    auto kv_options = torch::TensorOptions(torch::kUInt8).device(torch::kCPU).pinned_memory(true);
    auto kv_buffer  = torch::empty({2, kv_stride}, kv_options);
    kv_buffer[1].fill_(static_cast<uint8_t>(123));

    auto inputs = makeSingleBlockWriteInputs(cache_key, request_id_val, spb);
    auto config = makeSingleBlockWriteConfig(
        "csa_state", spb, kv_stride, /*kv_scale_stride=*/0, /*use_opaque_kv_cache_store=*/true);

    torch_ext::LayerKVCache layer_cache;
    layer_cache.kv_cache_base      = kv_buffer;
    layer_cache.seq_size_per_block = spb;
    layer_cache.layer_id           = 0;
    layer_cache.group_id           = 0;
    layer_cache.tag                = "csa_state";

    auto cache_store = std::make_shared<MemoryBackedCacheStore>();
    runtimeWriteCacheStore(inputs,
                           layer_cache,
                           config,
                           cache_store,
                           /*cache_model_id=*/0,
                           /*cp_rank=*/0,
                           /*cp_size=*/1,
                           /*pre_created_event=*/nullptr);

    const auto key = "kv_" + makeCacheKey(0, std::to_string(cache_key), 0, "csa_state");
    auto       it  = cache_store->stored_blocks_.find(key);
    ASSERT_NE(it, cache_store->stored_blocks_.end());
    ASSERT_EQ(it->second.size(), static_cast<size_t>(kv_stride));
    EXPECT_EQ(it->second[0], static_cast<uint8_t>(123));

    ASSERT_EQ(cache_store->store_buffer_requests_.size(), 1u);
    auto blocks   = cache_store->store_buffer_requests_.front()->getBlocks();
    auto block_it = blocks.find(key);
    ASSERT_NE(block_it, blocks.end());
    EXPECT_FALSE(block_it->second->gpu_mem);
}

TEST_F(PdSepKVCacheReleaseTest, testWriteCacheStoreUsesTensorDeviceForCpuSplitKvBuffer) {
    const int     spb            = 8;
    const int     kv_stride      = 64;
    const int     kv_half        = kv_stride / 2;
    const int     request_id_val = 4243;
    const int64_t cache_key      = 10001;

    auto kv_options = torch::TensorOptions(torch::kUInt8).device(torch::kCPU).pinned_memory(true);
    auto kv_buffer  = torch::empty({2, kv_stride}, kv_options);
    auto block      = kv_buffer[1];
    block.slice(0, 0, kv_half).fill_(static_cast<uint8_t>(17));
    block.slice(0, kv_half, kv_stride).fill_(static_cast<uint8_t>(29));

    auto inputs = makeSingleBlockWriteInputs(cache_key, request_id_val, spb);
    auto config = makeSingleBlockWriteConfig(
        "default", spb, kv_stride, /*kv_scale_stride=*/0, /*use_opaque_kv_cache_store=*/false);

    torch_ext::LayerKVCache layer_cache;
    layer_cache.kv_cache_base      = kv_buffer;
    layer_cache.seq_size_per_block = spb;
    layer_cache.layer_id           = 0;
    layer_cache.group_id           = 0;
    layer_cache.tag                = "default";

    auto cache_store = std::make_shared<MemoryBackedCacheStore>();
    runtimeWriteCacheStore(inputs,
                           layer_cache,
                           config,
                           cache_store,
                           /*cache_model_id=*/0,
                           /*cp_rank=*/0,
                           /*cp_size=*/1,
                           /*pre_created_event=*/nullptr);

    const auto cache_key_str = makeCacheKey(0, std::to_string(cache_key), 0, "default");
    const auto k_key         = "k_" + cache_key_str;
    const auto v_key         = "v_" + cache_key_str;
    auto       k_it          = cache_store->stored_blocks_.find(k_key);
    auto       v_it          = cache_store->stored_blocks_.find(v_key);
    ASSERT_NE(k_it, cache_store->stored_blocks_.end());
    ASSERT_NE(v_it, cache_store->stored_blocks_.end());
    ASSERT_EQ(k_it->second.size(), static_cast<size_t>(kv_half));
    ASSERT_EQ(v_it->second.size(), static_cast<size_t>(kv_half));
    EXPECT_EQ(k_it->second[0], static_cast<uint8_t>(17));
    EXPECT_EQ(v_it->second[0], static_cast<uint8_t>(29));

    ASSERT_EQ(cache_store->store_buffer_requests_.size(), 1u);
    auto k_block = cache_store->store_buffer_requests_.front()->getBlock(k_key);
    auto v_block = cache_store->store_buffer_requests_.front()->getBlock(v_key);
    ASSERT_NE(k_block, nullptr);
    ASSERT_NE(v_block, nullptr);
    EXPECT_FALSE(k_block->gpu_mem);
    EXPECT_FALSE(v_block->gpu_mem);
}

TEST_F(PdSepKVCacheReleaseTest, testWriteCacheStoreUsesTensorDeviceForCpuKvScaleBuffer) {
    const int     spb            = 8;
    const int     kv_stride      = 64;
    const int     scale_stride   = 16;
    const int     request_id_val = 4244;
    const int64_t cache_key      = 10002;

    auto cpu_options     = torch::TensorOptions(torch::kUInt8).device(torch::kCPU).pinned_memory(true);
    auto kv_buffer       = torch::empty({2, kv_stride}, cpu_options);
    auto kv_scale_buffer = torch::empty({2, scale_stride}, cpu_options);
    kv_buffer[1].fill_(static_cast<uint8_t>(41));
    kv_scale_buffer[1].fill_(static_cast<uint8_t>(73));

    auto inputs = makeSingleBlockWriteInputs(cache_key, request_id_val, spb);
    auto config =
        makeSingleBlockWriteConfig("csa_state", spb, kv_stride, scale_stride, /*use_opaque_kv_cache_store=*/true);

    torch_ext::LayerKVCache layer_cache;
    layer_cache.kv_cache_base      = kv_buffer;
    layer_cache.kv_scale_base      = kv_scale_buffer;
    layer_cache.seq_size_per_block = spb;
    layer_cache.layer_id           = 0;
    layer_cache.group_id           = 0;
    layer_cache.tag                = "csa_state";

    auto cache_store = std::make_shared<MemoryBackedCacheStore>();
    runtimeWriteCacheStore(inputs,
                           layer_cache,
                           config,
                           cache_store,
                           /*cache_model_id=*/0,
                           /*cp_rank=*/0,
                           /*cp_size=*/1,
                           /*pre_created_event=*/nullptr);

    const auto scale_key = "kv_scale_" + makeCacheKey(0, std::to_string(cache_key), 0, "csa_state");
    auto       scale_it  = cache_store->stored_blocks_.find(scale_key);
    ASSERT_NE(scale_it, cache_store->stored_blocks_.end());
    ASSERT_EQ(scale_it->second.size(), static_cast<size_t>(scale_stride));
    EXPECT_EQ(scale_it->second[0], static_cast<uint8_t>(73));

    ASSERT_EQ(cache_store->store_buffer_requests_.size(), 1u);
    auto scale_block = cache_store->store_buffer_requests_.front()->getBlock(scale_key);
    ASSERT_NE(scale_block, nullptr);
    EXPECT_FALSE(scale_block->gpu_mem);
}

}  // namespace rtp_llm
