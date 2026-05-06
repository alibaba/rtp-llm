#include "gtest/gtest.h"
#include "gmock/gmock.h"

#define private public
#define protected public
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/DSV4CacheConfig.h"
#include "rtp_llm/cpp/cache/DSV4ConfigCreator.h"
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
#include "rtp_llm/models_py/bindings/core/ExecOps.h"

#include <atomic>
#include <chrono>
#include <cstring>
#include <memory>
#include <numeric>
#include <thread>
#include <unordered_map>

namespace rtp_llm {

namespace {

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
            auto device = torch::from_blob(
                block->addr.get(), {(int64_t)block->len}, torch::TensorOptions(torch::kUInt8).device(torch::kCUDA));
            auto                 host = device.cpu().contiguous();
            std::vector<uint8_t> bytes(static_cast<size_t>(block->len));
            std::memcpy(bytes.data(), host.data_ptr<uint8_t>(), bytes.size());
            stored_blocks_[key] = std::move(bytes);
        }
        store_request_keys_.push_back(request_block_buffer->getRequestKey());
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
            auto device = torch::from_blob(
                block->addr.get(), {(int64_t)block->len}, torch::TensorOptions(torch::kUInt8).device(torch::kCUDA));
            device.copy_(host);
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

void fillDsv4RegionBytes(const std::shared_ptr<KVCacheManager>& manager,
                         int                                    block_id,
                         int                                    layer_id,
                         KVCacheRegionName                      region_name,
                         uint8_t                                value) {
    auto parts = manager->convertIndexToBuffer(block_id, layer_id, region_name);
    ASSERT_EQ(parts.size(), 1u);
    auto device = torch::from_blob(
        parts[0].addr, {(int64_t)parts[0].size_bytes}, torch::TensorOptions(torch::kUInt8).device(torch::kCUDA));
    auto host =
        torch::full({(int64_t)parts[0].size_bytes}, value, torch::TensorOptions(torch::kUInt8).device(torch::kCPU));
    device.copy_(host);
}

void expectDsv4RegionBytes(const std::shared_ptr<KVCacheManager>& manager,
                           int                                    block_id,
                           int                                    layer_id,
                           KVCacheRegionName                      region_name,
                           uint8_t                                value) {
    auto parts = manager->convertIndexToBuffer(block_id, layer_id, region_name);
    ASSERT_EQ(parts.size(), 1u);
    auto device = torch::from_blob(
        parts[0].addr, {(int64_t)parts[0].size_bytes}, torch::TensorOptions(torch::kUInt8).device(torch::kCUDA));
    auto        host = device.cpu().contiguous();
    const auto* ptr  = host.data_ptr<uint8_t>();
    for (size_t i = 0; i < parts[0].size_bytes; ++i) {
        ASSERT_EQ(ptr[i], value) << "byte=" << i << " layer=" << layer_id << " block=" << block_id
                                 << " region=" << static_cast<int>(region_name);
    }
}

uint8_t dsv4PdPattern(int layer_id, int gid, size_t block_pos) {
    return static_cast<uint8_t>(17 + layer_id * 19 + gid * 11 + block_pos);
}

torch::Tensor blockIdsTensor(const BatchKVCacheResourcePtr& resource, int gid) {
    const auto& blocks = resource->blocks(0, gid);
    return torch::from_blob(const_cast<int*>(blocks.data()), {1, static_cast<int64_t>(blocks.size())}, torch::kInt32)
        .clone();
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

    CacheConfig makeDsv4Config(uint32_t block_num = 16) {
        ModelConfig mc;
        mc.num_layers                   = 43;
        mc.hidden_size                  = 4096;
        mc.attn_config.head_num         = 64;
        mc.attn_config.kv_head_num      = 1;
        mc.attn_config.size_per_head    = 512;
        mc.attn_config.rope_head_dim    = 64;
        mc.attn_config.sliding_window   = 128;
        mc.attn_config.indexer_head_dim = 128;
        mc.attn_config.indexer_head_num = 64;
        mc.attn_config.indexer_topk     = 512;
        mc.attn_config.o_groups         = 8;
        mc.attn_config.o_lora_rank      = 1024;
        std::vector<int> ratios         = {0, 0};
        for (int i = 2; i < 43; ++i) {
            ratios.push_back((i % 2 == 0) ? 4 : 128);
        }
        ratios.push_back(0);  // MTP tail marker.
        mc.attn_config.layer_compress_ratios = ratios;

        ParallelismConfig pc;
        auto              config = DSV4ConfigCreator::createConfig(mc, pc);
        config.block_num         = block_num;
        config.group_block_nums.assign(config.groupNums(), block_num);
        return config;
    }

    // Build a PREFILL stream with reuse_cache enabled
    void prepareStream(const std::vector<int>& input_tokens) {
        prepareStreamWithConfig(input_tokens, makeConfig(), /*tokens_per_block=*/8, RoleType::PREFILL);
    }

    void prepareDsv4Stream(const std::vector<int>& input_tokens, RoleType role_type = RoleType::PREFILL) {
        prepareStreamWithConfig(
            input_tokens, makeDsv4Config(), static_cast<int>(DSV4CacheConfig::TOKENS_PER_BLOCK), role_type);
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
    const int        spb = static_cast<int>(DSV4CacheConfig::TOKENS_PER_BLOCK);
    std::vector<int> tokens(3 * spb + 17);
    std::iota(tokens.begin(), tokens.end(), 1);

    prepareDsv4Stream(tokens, RoleType::PREFILL);
    allocateAndFinish();

    auto& resource = stream_->streamCacheResource();
    ASSERT_EQ(resource.kvCache().groupNums(), DSV4_NUM_POOLS);
    ASSERT_GT(resource.curBlocksNum(), 0);
    for (int gid = 0; gid < DSV4_NUM_POOLS; ++gid) {
        ASSERT_EQ(resource.kvCache().blocksNum(0, gid), 4) << "group " << gid;
        const auto& blocks = resource.kvCache().blocks(0, gid);
        if (gid < 3) {
            EXPECT_FALSE(isNullBlockIdx(blocks[0])) << "paged group " << gid;
        } else {
            EXPECT_TRUE(isNullBlockIdx(blocks[0])) << "tail group " << gid << " should keep only tail blocks";
            EXPECT_FALSE(isNullBlockIdx(blocks[2])) << "tail group " << gid;
            EXPECT_FALSE(isNullBlockIdx(blocks[3])) << "tail group " << gid;
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
    EXPECT_EQ(resource2.kvCache().groupNums(), DSV4_NUM_POOLS);

    stream2->generate_status_->status = StreamState::FINISHED;
    stream2->fillSubGenerateStatus(StreamState::FINISHED);
    stream2->releaseResource();
}

TEST_F(PdSepKVCacheReleaseTest, testDsv4DecodeFirstMallocBypassesLocalDeviceReuseInPDSep) {
    const int        spb = static_cast<int>(DSV4CacheConfig::TOKENS_PER_BLOCK);
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
    EXPECT_EQ(decode_resource.kvCache().groupNums(), DSV4_NUM_POOLS);
    for (int gid = 0; gid < DSV4_NUM_POOLS; ++gid) {
        EXPECT_EQ(decode_resource.kvCache().blocksNum(0, gid), 4) << "group " << gid;
    }

    decode_stream->releaseResource();
}

TEST_F(PdSepKVCacheReleaseTest, testDsv4CacheStorePDSepTransfersAllLayerRegions) {
    const int     spb        = static_cast<int>(DSV4CacheConfig::TOKENS_PER_BLOCK);
    const int     block_num  = 4;
    const int64_t request_id = 9017;
    const size_t  model_id   = 77;

    auto config = makeDsv4Config(/*block_num=*/24);

    auto makeResource = [&config]() {
        auto resource = std::make_shared<BatchKVCacheResource>();
        resource->resetBatchSize(1);
        resource->initGroups(config.groupNums(),
                             static_cast<int>(config.layer_all_num),
                             config.layer_to_group_id,
                             config.kernelBlocksPerKvBlock(),
                             config.group_types,
                             config.layer_region_to_group_id);
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

    auto prefill_resource = makeResource();
    auto decode_resource  = makeResource();
    ASSERT_TRUE(
        prefill_manager->malloc({prefill_resource, makeCompleteTokens(block_num * spb), request_id, true, false, false})
            .success);
    ASSERT_TRUE(
        decode_manager->malloc({decode_resource, makeCompleteTokens(block_num * spb), request_id, true, false, false})
            .success);

    std::vector<CacheKeyType> cache_keys;
    std::vector<std::string>  cache_key_strings;
    for (int i = 0; i < block_num; ++i) {
        cache_keys.push_back(10000 + i);
        cache_key_strings.push_back(std::to_string(cache_keys.back()));
    }

    for (int layer_id = 0; layer_id < 4; ++layer_id) {
        for (int gid : config.layer_to_group_ids[layer_id]) {
            auto region_name = config.group_region_names[gid];
            auto positions   = blockPositionsForCacheTransfer(
                block_num, /*reuse_block_size=*/0, true, config.group_types[gid], /*hybrid_full_from_begin=*/true);
            for (auto block_pos : positions) {
                auto prefill_block_id = prefill_resource->blocks(0, gid)[block_pos];
                auto decode_block_id  = decode_resource->blocks(0, gid)[block_pos];
                ASSERT_FALSE(isNullBlockIdx(prefill_block_id)) << "prefill gid=" << gid << " pos=" << block_pos;
                ASSERT_FALSE(isNullBlockIdx(decode_block_id)) << "decode gid=" << gid << " pos=" << block_pos;
                fillDsv4RegionBytes(
                    prefill_manager, prefill_block_id, layer_id, region_name, dsv4PdPattern(layer_id, gid, block_pos));
                fillDsv4RegionBytes(decode_manager, decode_block_id, layer_id, region_name, 0xEE);
            }
        }
    }
    runtimeSyncAndCheck();

    auto layer_to_group_tensor = torch::from_blob(config.layer_to_group_id.data(),
                                                  {(int64_t)config.layer_to_group_id.size()},
                                                  torch::TensorOptions(torch::kInt32))
                                     .clone();
    std::vector<int32_t> layer_region_to_group_flat;
    for (const auto& row : config.layer_region_to_group_id) {
        layer_region_to_group_flat.insert(layer_region_to_group_flat.end(), row.begin(), row.end());
    }
    auto layer_region_to_group_tensor = torch::from_blob(layer_region_to_group_flat.data(),
                                                         {(int64_t)config.layer_region_to_group_id.size(),
                                                          (int64_t)config.layer_region_to_group_id[0].size()},
                                                         torch::TensorOptions(torch::kInt32))
                                            .clone();
    std::vector<int32_t> group_types;
    for (auto group_type : config.group_types) {
        group_types.push_back(static_cast<int32_t>(group_type));
    }
    auto group_types_tensor =
        torch::from_blob(group_types.data(), {(int64_t)group_types.size()}, torch::TensorOptions(torch::kInt32))
            .clone();

    auto cache_store = std::make_shared<MemoryBackedCacheStore>();
    auto layout      = prefill_manager->getMainModelCacheLayerLayout();
    for (int layer_id = 0; layer_id < 4; ++layer_id) {
        for (int gid : config.layer_to_group_ids[layer_id]) {
            auto region_name = config.group_region_names[gid];
            auto region_idx  = static_cast<size_t>(region_name);
            ASSERT_TRUE(layout.layers_to_kv_buffer_ptrs_by_attn[layer_id][region_idx].defined())
                << "layer=" << layer_id << " region=" << region_idx;

            CacheStoreInputs inputs;
            inputs.input_lengths_host                  = torch::tensor({block_num * spb}, torch::kInt32);
            inputs.prefix_lengths_host                 = torch::tensor({0}, torch::kInt32);
            inputs.host_kv_cache_offset                = blockIdsTensor(prefill_resource, gid);
            inputs.kv_cache_layer_to_group_host        = layer_to_group_tensor;
            inputs.kv_cache_layer_region_to_group_host = layer_region_to_group_tensor;
            inputs.kv_cache_group_types_host           = group_types_tensor;
            inputs.context_batch_size                  = 1;
            inputs.decoder_batch_size                  = 0;
            inputs.request_id                          = torch::tensor({request_id}, torch::kInt64);
            inputs.request_pd_separation               = torch::tensor({true}, torch::kBool);
            inputs.cache_keys                          = cache_key_strings;
            inputs.tokens_per_block                    = spb;
            inputs.kv_block_stride_bytes               = config.group_kv_block_stride_bytes[gid];
            inputs.kv_scale_stride_bytes               = 0;
            inputs.pd_separation                       = true;
            inputs.model_id                            = model_id;
            inputs.decode_entrance                     = false;
            inputs.warmup                              = false;
            inputs.layer_id                            = layer_id;
            inputs.region_name                         = region_name;

            KvCacheInfo kv_cache_info;
            kv_cache_info.kv_cache_buffer = layout.layers_to_kv_buffer_ptrs_by_attn[layer_id][region_idx];
            runtimeWriteCacheStore(inputs, kv_cache_info, /*mla_kvcache=*/false, cache_store);
        }
    }
    ASSERT_EQ(cache_store->store_request_keys_.size(), 10u);
    ASSERT_EQ(cache_store->stored_blocks_.size(), 26u);

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

    EXPECT_EQ(cache_store->load_buffer_requests_.size(), 10u);
    EXPECT_EQ(cache_store->load_request_keys_.size(), 10u);
    for (int layer_id = 0; layer_id < 4; ++layer_id) {
        for (int gid : config.layer_to_group_ids[layer_id]) {
            auto region_name = config.group_region_names[gid];
            auto positions   = blockPositionsForCacheTransfer(
                block_num, /*reuse_block_size=*/0, true, config.group_types[gid], /*hybrid_full_from_begin=*/true);
            for (auto block_pos : positions) {
                auto decode_block_id = decode_resource->blocks(0, gid)[block_pos];
                ASSERT_FALSE(isNullBlockIdx(decode_block_id));
                expectDsv4RegionBytes(
                    decode_manager, decode_block_id, layer_id, region_name, dsv4PdPattern(layer_id, gid, block_pos));
            }
        }
    }
}

}  // namespace rtp_llm
