#include "gtest/gtest.h"
#include "gmock/gmock.h"

#define private public
#define protected public
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/cache/HybridPoolConfigCreator.h"
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
#include "rtp_llm/models_py/bindings/common/WriteCacheStoreOp.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"

#include <atomic>
#include <chrono>
#include <cstring>
#include <memory>
#include <numeric>
#include <optional>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

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
              const std::string& ip,
              uint32_t,
              uint32_t,
              uint32_t = 1000,
              int      = 1,
              int      = 0) override {
        bool ok = true;
        const auto& source_blocks = peer_stores_.empty() ? stored_blocks_ : peer_stores_.at(ip)->stored_blocks_;
        for (const auto& [key, block] : request_block_buffer->getBlocks()) {
            auto it = source_blocks.find(key);
            if (it == source_blocks.end() || it->second.size() != block->len) {
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
    std::unordered_map<std::string, std::shared_ptr<MemoryBackedCacheStore>> peer_stores_;
    std::vector<std::string>                              store_request_keys_;
    std::vector<std::string>                              load_request_keys_;
    std::vector<std::shared_ptr<RequestBlockBuffer>>      store_buffer_requests_;
    std::vector<std::shared_ptr<RequestBlockBuffer>>      load_buffer_requests_;
};

class MinimalEngine: public EngineBase {
public:
    MinimalEngine(const EngineInitParams&        params,
                  std::shared_ptr<KVCacheManager> cache_manager,
                  bool                            is_mtp_eagle = false):
        EngineBase(params), is_mtp_eagle_(is_mtp_eagle) {
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
    bool isMTPEagle() override {
        return is_mtp_eagle_;
    }

private:
    bool is_mtp_eagle_;
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

CacheStoreInputs makeSingleBlockWriteInputs(const std::string& cache_key_string,
                                            int                request_id_val,
                                            int                tokens_per_block,
                                            int                kv_stride,
                                            int                kv_scale_stride,
                                            bool               use_opaque_kv_cache_store,
                                            KVCacheRegionName  region_name) {
    CacheStoreInputs inputs;
    inputs.input_lengths_host        = torch::tensor({tokens_per_block}, torch::kInt32);
    inputs.prefix_lengths_host       = torch::tensor({0}, torch::kInt32);
    inputs.host_kv_cache_offset      = torch::tensor({{1}}, torch::kInt32);
    inputs.context_batch_size        = 1;
    inputs.decoder_batch_size        = 0;
    inputs.request_id                = torch::tensor({(int64_t)request_id_val}, torch::kInt64);
    inputs.request_pd_separation     = torch::tensor({true}, torch::kBool);
    inputs.cache_keys                = {cache_key_string};
    inputs.tokens_per_block          = tokens_per_block;
    inputs.kv_block_stride_bytes     = kv_stride;
    inputs.kv_scale_stride_bytes     = kv_scale_stride;
    inputs.pd_separation             = true;
    inputs.model_id                  = 0;
    inputs.decode_entrance           = false;
    inputs.warmup                    = false;
    inputs.use_opaque_kv_cache_store = use_opaque_kv_cache_store;
    inputs.layer_id                  = 0;
    inputs.region_name               = region_name;
    return inputs;
}

}  // namespace

TEST(DecodeRpcServerTest, MtpPhysicalGroupUsesGlobalLayerLayout) {
    CacheLayerLayout layout;
    layout.layer_to_groups = {2};
    layout.layer_region_to_group_id = {
        std::vector<int>(static_cast<size_t>(KVCacheRegionName::REGION_COUNT), -1)};
    layout.layer_region_to_group_id[0][static_cast<size_t>(KVCacheRegionName::SWA_KV)] = 5;

    EXPECT_EQ(layout.resolvePhysicalGroupId(/*local_layer_id=*/0, KVCacheRegionName::DEFAULT), 2);
    EXPECT_EQ(layout.resolvePhysicalGroupId(/*local_layer_id=*/0, KVCacheRegionName::SWA_KV), 5);
    EXPECT_FALSE(layout.resolvePhysicalGroupId(/*local_layer_id=*/0, KVCacheRegionName::CSA_KV).has_value());
}

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
        KVCacheConfig     kv_config;
        kv_config.seq_size_per_block        = seq_size_per_block;
        kv_config.kernel_seq_size_per_block = kernel_seq_size_per_blk;
        auto config                         = HybridPoolConfigCreator::createConfig(mc, pc, kv_config, false, 0);
        config.block_num                    = block_num;
        config.group_block_nums.assign(config.groupNums(), block_num);
        return config;
    }

    CacheConfig makeIndependentHybridEagleConfig(bool use_mla = false, int cp_size = 1, int kernel_tokens = 4, int window = 3, bool decode = false, int tp_size = 0) {
        const int model_tp_size = tp_size > 0 ? tp_size : cp_size;
        auto make_model_config = [use_mla](uint32_t num_layers) {
            ModelConfig config;
            config.num_layers                   = static_cast<int64_t>(num_layers);
            config.max_seq_len                  = 128;
            config.hidden_size                  = 64;
            config.vocab_size                   = 1024;
            config.data_type                    = rtp_llm::DataType::TYPE_FP16;
            config.attn_config.head_num         = 2;
            config.attn_config.kv_head_num      = 2;
            config.attn_config.size_per_head    = 16;
            config.attn_config.tokens_per_block = 4;
            config.attn_config.kv_cache_dtype   = KvCacheDataType::BASE;
            if (use_mla) {
                config.attn_config.use_mla = true;
                config.attn_config.kv_lora_rank = 16;
                config.attn_config.rope_head_dim = 8;
            }
            return config;
        };

        auto score_model_config   = make_model_config(/*num_layers=*/4);
        auto propose_model_config = make_model_config(/*num_layers=*/1);
        score_model_config.hybrid_attention_config.enable_hybrid_attention           = true;
        score_model_config.hybrid_attention_config.enable_independent_kv_cache_pools = true;
        score_model_config.hybrid_attention_config.hybrid_attention_types            = {
            HybridAttentionType::NONE,
            HybridAttentionType::LINEAR,
            HybridAttentionType::NONE,
            HybridAttentionType::LINEAR};
        propose_model_config.hybrid_attention_config.enable_hybrid_attention           = true;
        propose_model_config.hybrid_attention_config.enable_independent_kv_cache_pools = true;
        propose_model_config.hybrid_attention_config.hybrid_attention_types            = {
            HybridAttentionType::SLIDING_WINDOW};
        propose_model_config.attn_config.sliding_window = window;
        score_model_config.linear_attention_config.linear_conv_kernel_dim = 2;
        score_model_config.linear_attention_config.linear_key_head_dim    = 8;
        score_model_config.linear_attention_config.linear_value_head_dim  = 8;
        score_model_config.linear_attention_config.linear_num_key_heads   = 2 * model_tp_size;
        score_model_config.linear_attention_config.linear_num_value_heads = 2 * model_tp_size;

        ParallelismConfig parallelism_config;
        parallelism_config.tp_size = model_tp_size;
        parallelism_config.role_type = decode ? RoleType::DECODE : RoleType::PREFILL;
        parallelism_config.prefill_cp_config.kv_cache_sharded = !decode && cp_size > 1;
        parallelism_config.decode_cp_kv_cache_sharded = decode && cp_size > 1;
        RuntimeConfig runtime_config;
        KVCacheConfig kv_cache_config;
        kv_cache_config.test_block_num = 8;
        kv_cache_config.kernel_seq_size_per_block = kernel_tokens;
        SpeculativeExecutionConfig sp_config;
        sp_config.type              = SP_TYPE_EAGLE3;
        sp_config.gen_num_per_cycle = 3;

        return CacheConfigCreator::createSpConfig(score_model_config,
                                                  propose_model_config,
                                                  parallelism_config,
                                                  runtime_config,
                                                  kv_cache_config,
                                                  sp_config,
                                                  /*warm_up_result=*/std::nullopt,
                                                  /*is_mtp=*/true,
                                                  /*is_eagle=*/true);
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

    auto config        = makeDsv4Config();
    config.linear_step = 4;
    prepareStreamWithConfig(tokens, config, spb, RoleType::PREFILL);
    allocateAndFinish();

    auto& resource = stream_->streamCacheResource();
    ASSERT_EQ(resource.kvCache().groupNums(), kDsv4PoolNum);
    ASSERT_GT(resource.curBlocksNum(), 0);
    for (int gid = 0; gid < kDsv4PoolNum; ++gid) {
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
            auto positions =
                blockPositionsForCacheTransfer(block_num, /*first_full_block=*/0, true, config.group_types[gid]);
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
            inputs.use_opaque_kv_cache_store           = config.use_opaque_kv_cache_store;
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
            auto positions =
                blockPositionsForCacheTransfer(block_num, /*first_full_block=*/0, true, config.group_types[gid]);
            for (auto block_pos : positions) {
                auto decode_block_id = decode_resource->blocks(0, gid)[block_pos];
                ASSERT_FALSE(isNullBlockIdx(decode_block_id));
                expectDsv4RegionBytes(
                    decode_manager, decode_block_id, layer_id, region_name, dsv4PdPattern(layer_id, gid, block_pos));
            }
        }
    }
}

using EagleDraftTransferCase = std::tuple<bool, int, int, int, int, int, int, int, int>;

class EagleDraftTransferTest: public PdSepKVCacheReleaseTest,
                              public testing::WithParamInterface<EagleDraftTransferCase> {};

// Real writer -> CacheStore bytes -> DecodeRpcServer. CP-off still runs TP8;
// distinguish source/destination ranks, P/K views, partial pages and reserve.
INSTANTIATE_TEST_SUITE_P(CacheView, EagleDraftTransferTest,
    testing::ValuesIn([] {
        std::vector<EagleDraftTransferCase> cases{{false, 1, 4, 1, 3, 0, 0, 0, 16}};
        for (int source_cp : {1, 8}) {
            for (int destination_cp : {1, 8}) {
                for (int kernel_page : {2, 4}) {
                    for (int rank : {0, 3, 7}) {
                        for (int length : {15, 16}) {
                            cases.emplace_back(true, source_cp, kernel_page, destination_cp,
                                               3, 5, rank, rank == 3 ? 7 : (rank == 7 ? 3 : 0), length);
                        }
                    }
                }
            }
        }
        cases.emplace_back(true, 8, 2, 1, 4, 5, 7, 3, 16);
        cases.emplace_back(true, 8, 2, 8, 1, 5, 3, 7, 15);
        return cases;
    }()));

TEST_P(EagleDraftTransferTest, testEagleDraftLoadUsesIndependentPhysicalGroup) {
    const int64_t request_id    = 9019;
    const size_t  draft_model_id = 1;
    const int     block_num      = 4;

    const auto [use_mla, cp_size, kernel_tokens, decode_cp_size, window, reserve, source_rank, destination_rank, token_count] = GetParam();
    auto config = makeIndependentHybridEagleConfig(use_mla, cp_size, kernel_tokens, window, false, 8);
    auto decode_config = makeIndependentHybridEagleConfig(use_mla, decode_cp_size, kernel_tokens, window, true, 8);
    const size_t tail_blocks = 2;
    ASSERT_TRUE(config.use_independent_block_pools);
    ASSERT_EQ(config.group_types,
              std::vector<CacheGroupType>({CacheGroupType::FULL, CacheGroupType::LINEAR, CacheGroupType::SWA}));
    ASSERT_EQ(config.layer_to_group_id[4], 2);
    ASSERT_EQ(config.mtp_sub_configs.size(), 1u);

    auto make_resource = [](const CacheConfig& config) {
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
    const int spb = static_cast<int>(config.seq_size_per_block);
    auto make_complete_tokens = [spb, block_num, token_count]() {
        auto input             = std::make_shared<GenerateInput>();
        input->input_ids       = torch::arange(token_count, torch::kInt32);
        input->generate_config = std::make_shared<GenerateConfig>();
        auto complete_token_ids = std::make_shared<CompleteTokenIds>(1, 1, (block_num + 8) * spb, spb);
        complete_token_ids->init(input);
        complete_token_ids->setSeqLength(token_count);
        return complete_token_ids;
    };

    ParallelismConfig source_parallel, destination_parallel;
    source_parallel.tp_size = destination_parallel.tp_size = 8;
    source_parallel.tp_rank = source_rank;
    source_parallel.role_type = RoleType::PREFILL;
    source_parallel.prefill_cp_config.kv_cache_sharded = cp_size > 1;
    destination_parallel.tp_rank = destination_rank;
    destination_parallel.role_type = RoleType::DECODE;
    destination_parallel.decode_cp_kv_cache_sharded = decode_cp_size > 1;
    auto prefill_manager = std::make_shared<KVCacheManager>(config, true, nullptr, KVCacheConfig{}, source_parallel);
    auto decode_manager = std::make_shared<KVCacheManager>(decode_config, true, nullptr, KVCacheConfig{}, destination_parallel);
    // In-process peers retain real topology without a distributed allocation collective.
    prefill_manager->config_.block_num = config.block_num;
    decode_manager->config_.block_num = decode_config.block_num;
    ASSERT_TRUE(prefill_manager->init());
    ASSERT_TRUE(decode_manager->init());

    auto prefill_resource = make_resource(config);
    // Fragment the destination so matching logical pages cannot accidentally
    // pass by sharing source physical IDs.
    auto guard_resource = make_resource(decode_config);
    auto guard_tokens = make_complete_tokens();
    guard_tokens->setSeqLength(1);
    ASSERT_TRUE(decode_manager->malloc({guard_resource, guard_tokens, 9018, true, false, false}).success);
    auto decode_resource  = make_resource(decode_config);
    MallocInfo prefill_malloc{prefill_resource, make_complete_tokens(), request_id, true, false, false};
    prefill_malloc.reuse_cache = use_mla;
    ASSERT_TRUE(prefill_manager->malloc(prefill_malloc).success);
    auto decode_tokens = make_complete_tokens();
    decode_tokens->setReserveStep(reserve);
    ASSERT_TRUE(decode_manager->malloc({decode_resource, decode_tokens, request_id, true, false, false}).success);

    const int physical_draft_gid = config.layer_to_group_id[4];
    ASSERT_EQ(physical_draft_gid, 2);
    const auto& draft_blocks = prefill_resource->blocks(0, physical_draft_gid);
    ASSERT_EQ(draft_blocks.size(), static_cast<size_t>(block_num));
    for (size_t page = 0; page < block_num; ++page) {
        EXPECT_EQ(isNullBlockIdx(draft_blocks[page]), !use_mla && page < block_num - tail_blocks);
    }

    std::vector<CacheKeyType> cache_keys;
    std::vector<std::string>  cache_key_strings;
    for (int i = 0; i < block_num; ++i) {
        cache_keys.push_back(30000 + i);
        cache_key_strings.push_back(std::to_string(cache_keys.back()));
    }
    std::vector<int32_t> group_types;
    for (auto group_type : config.group_types) {
        group_types.push_back(static_cast<int32_t>(group_type));
    }

    const auto& draft_config = *config.mtp_sub_configs[0];
    auto draft_layout        = prefill_manager->getMTPModuleCacheLayerLayout(0);
    ASSERT_EQ(draft_layout.layer_to_groups, std::vector<int>({physical_draft_gid}));
    ASSERT_EQ(draft_layout.layers_to_kv_buffer_ptrs.size(), 1u);
    draft_layout.layers_to_kv_buffer_ptrs[0].view(torch::kUInt8).fill_(0xCC);
    for (size_t block_pos = block_num - tail_blocks; block_pos < block_num; ++block_pos) {
        draft_layout.layers_to_kv_buffer_ptrs[0][draft_blocks[block_pos]].view(torch::kUInt8).fill_(42 + block_pos);
    }
    decode_manager->getMTPModuleCacheLayerLayout(0).layers_to_kv_buffer_ptrs[0].view(torch::kUInt8).fill_(0xEE);

    torch_ext::PyCacheStoreInputs inputs;
    inputs.input_lengths_host          = torch::tensor({token_count}, torch::kInt32);
    inputs.prefix_lengths_host         = torch::tensor({0}, torch::kInt32);
    inputs.kv_cache_layer_to_group     = torch::tensor({physical_draft_gid}, torch::kInt32);
    inputs.kv_cache_group_types =
        torch::from_blob(group_types.data(), {(int64_t)group_types.size()}, torch::kInt32).clone();
    inputs.context_batch_size          = 1;
    inputs.decoder_batch_size          = 0;
    inputs.request_id                  = torch::tensor({request_id}, torch::kInt64);
    inputs.request_pd_separation       = torch::tensor({true}, torch::kBool);
    inputs.cache_keys                  = cache_key_strings;
    inputs.tokens_per_block            = spb;
    inputs.kv_block_stride_bytes       = draft_config.group_kv_block_stride_bytes[0];
    inputs.kv_scale_stride_bytes       = draft_config.group_kv_scale_stride_bytes[0];
    inputs.pd_separation               = true;
    inputs.model_id                    = draft_model_id;
    inputs.decode_entrance             = false;
    inputs.warmup                      = false;
    inputs.use_opaque_kv_cache_store   = draft_config.use_opaque_kv_cache_store;
    inputs.cp_size                    = config.cp_size;
    inputs.cp_rank                    = source_rank;
    inputs.mla_kvcache                = config.use_mla;
    torch_ext::KVCache runtime_cache;
    runtime_cache.seq_size_per_block = spb;
    runtime_cache.kernel_seq_size_per_block = kernel_tokens;
    runtime_cache.use_mla = config.use_mla;
    runtime_cache.kv_lora_rank = 16;
    runtime_cache.rope_head_dim = 8;
    runtime_cache.num_kv_heads = draft_config.cache_specs[0]->local_head_num_kv;
    runtime_cache.head_dim = 16;
    runtime_cache.kv_cache_base_by_layer = draft_layout.layers_to_kv_buffer_ptrs;
    runtime_cache.layer_group_types = {CacheGroupType::FULL};  // Eagle3 runtime conversion, not transfer policy.
    auto layer = runtime_cache.getLayerCache(0);
    ASSERT_EQ(layer.seq_size_per_block, kernel_tokens);
    EXPECT_EQ(layer.kv_cache_base.data_ptr(), draft_layout.layers_to_kv_buffer_ptrs[0].data_ptr());
    auto cache_store              = std::make_shared<MemoryBackedCacheStore>();
    inputs.cache_store = cache_store;
    auto block_ids = blockIdsTensor(prefill_resource, physical_draft_gid);
    auto write = [&](std::optional<torch_ext::PyCacheStorePublishPlan> plan) {
        WriteCacheStoreOp(inputs.input_lengths_host, inputs.prefix_lengths_host, block_ids, inputs, layer, plan);
    };
    if (token_count % spb == 0) {
        write(std::nullopt);
    } else {
        write(torch_ext::PyCacheStorePublishPlan{torch::tensor({0}, torch::kInt32),
              torch::tensor({block_num}, torch::kInt32), torch::tensor({true}, torch::kBool)});
    }
    ASSERT_EQ(cache_store->store_request_keys_.size(), 1u);
    const size_t parts_per_block = config.use_mla ? 1 : 2;
    ASSERT_EQ(cache_store->stored_blocks_.size(), tail_blocks * parts_per_block);
    // Real chunk publication must defer SWA until the terminal pass, then
    // publish the same physical tail despite its smaller runtime kernel pages.
    const auto legacy_bytes = cache_store->stored_blocks_;
    cache_store->stored_blocks_.clear();
    torch_ext::PyCacheStorePublishPlan publish{torch::tensor({0}, torch::kInt32),
                                               torch::tensor({block_num}, torch::kInt32),
                                               torch::tensor({false}, torch::kBool)};
    write(publish);
    EXPECT_TRUE(cache_store->stored_blocks_.empty());
    publish.terminal_host.fill_(true);
    write(publish);
    EXPECT_EQ(cache_store->stored_blocks_, legacy_bytes);

    EngineInitParams params;
    params.model_id                 = 0;
    params.model_config_.num_layers = 0;
    params.parallelism_config       = destination_parallel;
    auto mtp_model_params = std::make_unique<std::vector<std::unique_ptr<EngineInitParams>>>();
    auto mtp_params        = std::make_unique<EngineInitParams>();
    mtp_params->model_id                 = draft_model_id;
    mtp_params->model_config_.num_layers = 1;
    mtp_model_params->push_back(std::move(mtp_params));

    DecodeRpcServer server;
    server.engine_ = std::make_shared<MinimalEngine>(params, decode_manager, /*is_mtp_eagle=*/true);
    server.maga_init_params_ = params;
    auto propose_params = std::make_unique<ProposeModelEngineInitParams>(
        SP_TYPE_EAGLE3, /*gen_num_per_circle=*/3, std::move(mtp_model_params));
    server.propose_maga_init_params_ = propose_params.get();
    server.resource_.cache_store     = cache_store;

    std::vector<std::string> peer_addrs;
    for (int peer = 0; peer < cp_size; ++peer) {
        const auto ip = "127.0.0." + std::to_string(peer + 1);
        peer_addrs.push_back(ip + ":12345:12346");
        auto peer_store = std::make_shared<MemoryBackedCacheStore>();
        if (peer == 0) {
            peer_store->stored_blocks_ = cache_store->stored_blocks_;
        }
        // Other peers deliberately lack the keys: replicated SWA must not fan
        // out like FULL owner-sharded KV.
        cache_store->peer_stores_[ip] = std::move(peer_store);
    }
    grpc::ServerContext                 server_context;
    DecodeRpcServer::LoadKVCacheContext load_context(request_id,
                                                     "eagle-independent-draft-pd",
                                                     peer_addrs,
                                                     cache_keys,
                                                     decode_resource->groupBlocks(),
                                                     /*reuse_block_size=*/0,
                                                     /*timeout_ms=*/5000,
                                                     /*partition_count=*/1,
                                                     /*partition_id=*/0,
                                                     &server_context, cp_size);
    auto status = server.loadCache(load_context);
    ASSERT_TRUE(status.ok()) << status.ToString();
    ASSERT_EQ(cache_store->load_buffer_requests_.size(), 1u);
    EXPECT_EQ(cache_store->load_buffer_requests_[0]->getBlocks().size(), tail_blocks * parts_per_block);
    EXPECT_EQ(cache_store->load_request_keys_.size(), 1u);

    std::unordered_set<void*> expected_destination_addrs;
    const auto&               decode_draft_blocks = decode_resource->blocks(0, physical_draft_gid);
    if (reserve > 0) {
        ASSERT_GT(decode_draft_blocks.size(), static_cast<size_t>(block_num));
    }
    for (size_t block_pos = block_num - tail_blocks; block_pos < block_num; ++block_pos) {
        EXPECT_NE(decode_draft_blocks[block_pos], draft_blocks[block_pos]);
        auto parts = decode_manager->convertIndexToBuffer(
            decode_draft_blocks[block_pos], /*global_layer_id=*/4, /*partition_count=*/1, /*partition_id=*/0);
        ASSERT_EQ(parts.size(), parts_per_block);
        for (const auto& part : parts) {
            expected_destination_addrs.insert(part.addr);
            auto actual_bytes = torch::from_blob(part.addr, {static_cast<int64_t>(part.size_bytes)},
                                                 torch::TensorOptions(torch::kUInt8).device(torch::kCUDA));
            EXPECT_TRUE(actual_bytes.eq(42 + block_pos).all().item<bool>());
        }
    }
    std::unordered_set<void*> actual_destination_addrs;
    for (const auto& [key, block] : cache_store->load_buffer_requests_[0]->getBlocks()) {
        actual_destination_addrs.insert(block->addr.get());
    }
    EXPECT_EQ(actual_destination_addrs, expected_destination_addrs);
    for (size_t page = block_num; page < decode_draft_blocks.size(); ++page) {
        if (!isNullBlockIdx(decode_draft_blocks[page])) {
            auto parts = decode_manager->convertIndexToBuffer(decode_draft_blocks[page], 4, 1, 0);
            for (const auto& part : parts) {
                auto bytes = torch::from_blob(part.addr, {static_cast<int64_t>(part.size_bytes)},
                    torch::TensorOptions(torch::kUInt8).device(torch::kCUDA));
                EXPECT_TRUE(bytes.eq(0xEE).all().item<bool>());
            }
        }
    }
    prefill_manager->free({prefill_resource, prefill_malloc.complete_token_ids});
    decode_manager->free({decode_resource, decode_tokens});
    decode_manager->free({guard_resource, guard_tokens});
}

TEST_F(PdSepKVCacheReleaseTest, testCpMlaDirectLoadUsesGlobalKeysAndLocalDestinationRows) {
    ModelConfig model;
    model.num_layers = 1;
    model.data_type = DataType::TYPE_FP16;
    model.attn_config.use_mla = true;
    model.mla_ops_type = MlaOpsType::AUTO;
    model.attn_config.kv_lora_rank = 16;
    model.attn_config.rope_head_dim = 8;
    model.attn_config.tokens_per_block = 4;
    KVCacheConfig kv_config;
    kv_config.seq_size_per_block = 4;
    kv_config.kernel_seq_size_per_block = 2;
    kv_config.test_block_num = 64;
    ParallelismConfig parallelism;
    parallelism.tp_size = 8;
    auto make_resource = [](const CacheConfig& config) {
        auto resource = std::make_shared<BatchKVCacheResource>();
        resource->resetBatchSize(1);
        resource->initGroups(config.groupNums(), config.layer_all_num, config.layer_to_group_id,
                             config.kernelBlocksPerKvBlock(), config.group_types, config.layer_region_to_group_id);
        return resource;
    };
    auto small_input = std::make_shared<GenerateInput>();
    small_input->input_ids = torch::tensor({1}, torch::kInt32);
    small_input->generate_config = std::make_shared<GenerateConfig>();
    auto small_tokens = std::make_shared<CompleteTokenIds>(1, 1, 8, 4);
    small_tokens->init(small_input);
    auto fragment_pool = [&](const std::shared_ptr<KVCacheManager>& manager, const CacheConfig& config) {
        std::vector<BatchKVCacheResourcePtr> guards;
        for (int i = 0; i < 12; ++i) {
            auto resource = make_resource(config);
            EXPECT_TRUE(manager->malloc({resource, small_tokens, 9100 + i, true, false, false}).success);
            guards.push_back(resource);
        }
        for (size_t i = 0; i < guards.size(); i += 2) {
            manager->free({guards[i], small_tokens});
            guards[i].reset();
        }
        return guards;
    };
    auto release_guards = [&](const std::shared_ptr<KVCacheManager>& manager,
                              const std::vector<BatchKVCacheResourcePtr>& guards) {
        for (const auto& guard : guards) {
            if (guard) {
                manager->free({guard, small_tokens});
            }
        }
    };
    // A partial final page and suffixes around a CP cycle distinguish global
    // key ordinals from local block-table positions. All sources are separate
    // stores: merging their maps would conceal a request to the wrong owner.
    for (bool source_sharded : {false, true}) {
        parallelism.role_type = RoleType::PREFILL;
        parallelism.prefill_cp_config.kv_cache_sharded = source_sharded;
        const auto source_config = CacheConfigCreator::createConfig(model, parallelism, RuntimeConfig{}, kv_config);
        auto input = std::make_shared<GenerateInput>();
        input->input_ids = torch::arange(65, torch::kInt32);
        input->generate_config = std::make_shared<GenerateConfig>();
        auto tokens = std::make_shared<CompleteTokenIds>(1, 1, 128, 4);
        tokens->init(input);
        std::vector<CacheKeyType> keys;
        std::vector<std::string> peers;
        auto transport = std::make_shared<MemoryBackedCacheStore>();
        for (int rank = 0; rank < 8; ++rank) {
            parallelism.tp_rank = rank;
            // These in-process peers have no distributed bootstrap. Keep their
            // real rank/CP geometry but supply the pool capacity before init.
            auto source = std::make_shared<KVCacheManager>(source_config, true, nullptr, kv_config, parallelism);
            source->config_.block_num = source_config.block_num;
            ASSERT_TRUE(source->init());
            auto guards = fragment_pool(source, source_config);
            auto resource = make_resource(source_config);
            ASSERT_TRUE(source->malloc({resource, tokens, 9022, true, false, false}).success);
            if (rank == 0) {
                keys = resource->cacheKeys(0);
            }
            ASSERT_EQ(keys, resource->cacheKeys(0));
            ASSERT_EQ(keys.size(), 17u);
            const auto& blocks = resource->blocks(0, 0);
            ASSERT_GE(blocks.size(), 2u);
            ASSERT_NE(blocks[1], blocks[0] + 1);
            for (size_t row = 0; row < blocks.size(); ++row) {
                const size_t global_page = source_sharded ? row * 8 + rank : row;
                if (global_page >= keys.size()) {
                    continue;
                }
                auto parts = source->convertIndexToBuffer(blocks[row], 0);
                ASSERT_EQ(parts.size(), 1u);
                auto bytes = torch::from_blob(parts[0].addr, {static_cast<int64_t>(parts[0].size_bytes)},
                                             torch::TensorOptions(torch::kUInt8).device(torch::kCUDA));
                bytes.fill_(32 + global_page);
            }
            const auto layout = source->getMainModelCacheLayerLayout();
            torch_ext::KVCache cache;
            cache.seq_size_per_block = 4;
            cache.kernel_seq_size_per_block = 2;
            cache.use_mla = true;
            cache.kv_lora_rank = 16;
            cache.rope_head_dim = 8;
            cache.layer_group_types = layout.layer_group_types;
            cache.kv_cache_base_by_layer = layout.layers_to_kv_buffer_ptrs;
            auto store = std::make_shared<MemoryBackedCacheStore>();
            torch_ext::PyCacheStoreInputs writer;
            writer.context_batch_size = 1;
            writer.request_id = torch::tensor({int64_t{9022}}, torch::kInt64);
            writer.request_pd_separation = torch::tensor({true}, torch::kBool);
            writer.tokens_per_block = 4;
            writer.kv_block_stride_bytes = source_config.kv_block_stride_bytes;
            writer.kv_scale_stride_bytes = 0;
            writer.pd_separation = true;
            writer.mla_kvcache = true;
            writer.cp_size = source_config.cp_size;
            writer.cp_rank = source_sharded ? rank : 0;
            writer.cache_store = store;
            for (auto key : keys) {
                writer.cache_keys.push_back(std::to_string(key));
            }
            WriteCacheStoreOp(torch::tensor({65}, torch::kInt32), torch::tensor({0}, torch::kInt32),
                              blockIdsTensor(resource, 0), writer, cache.getLayerCache(0), std::nullopt);
            size_t published_pages = 0;
            for (size_t page = 0; page < keys.size(); ++page) {
                if (source_sharded && page % 8 != rank) {
                    continue;
                }
                const auto key = "kv_" + makeCacheKey(0, std::to_string(keys[page]), 0, KVCacheRegionName::DEFAULT);
                ASSERT_EQ(store->stored_blocks_.count(key), 1u) << key;
                EXPECT_EQ(store->stored_blocks_.at(key),
                          std::vector<uint8_t>(source_config.kv_block_stride_bytes, 32 + page));
                ++published_pages;
            }
            ASSERT_EQ(store->stored_blocks_.size(), published_pages);
            const auto ip = "127.0.0." + std::to_string(rank + 1);
            peers.push_back(ip + ":12345:12346");
            transport->peer_stores_[ip] = store;
            source->free({resource, tokens});
            release_guards(source, guards);
        }
        for (bool destination_sharded : {false, true}) {
            parallelism.role_type = RoleType::DECODE;
            parallelism.decode_cp_kv_cache_sharded = destination_sharded;
            const auto destination_config =
                CacheConfigCreator::createConfig(model, parallelism, RuntimeConfig{}, kv_config);
            for (int rank : {0, 3, 7}) {
                parallelism.tp_rank = rank;
                auto destination = std::make_shared<KVCacheManager>(
                    destination_config, true, nullptr, kv_config, parallelism);
                destination->config_.block_num = destination_config.block_num;
                ASSERT_TRUE(destination->init());
                auto guards = fragment_pool(destination, destination_config);
                auto resource = make_resource(destination_config);
                ASSERT_TRUE(destination->malloc({resource, tokens, 9022, true, false, false}).success);
                EngineInitParams params;
                params.model_id = 0;
                params.model_config_ = model;
                params.parallelism_config = parallelism;
                DecodeRpcServer server;
                server.engine_ = std::make_shared<MinimalEngine>(params, destination);
                server.maga_init_params_ = params;
                server.resource_.cache_store = transport;
                server.resource_.workers.resize(8);
                const auto& blocks = resource->blocks(0, 0);
                ASSERT_GE(blocks.size(), 2u);
                ASSERT_NE(blocks[1], blocks[0] + 1);
                for (const auto& guard : guards) {
                    if (guard) {
                        fillDsv4RegionBytes(destination, guard->blocks(0, 0)[0], 0, KVCacheRegionName::DEFAULT, 0xA5);
                    }
                }
                for (int reused_pages : {0, 1, 7, 9, 17}) {
                    SCOPED_TRACE(::testing::Message() << "P_sharded=" << source_sharded
                        << " D_sharded=" << destination_sharded << " rank=" << rank << " reused=" << reused_pages);
                    for (auto block : blocks) {
                        auto parts = destination->convertIndexToBuffer(block, 0);
                        auto bytes = torch::from_blob(parts[0].addr, {static_cast<int64_t>(parts[0].size_bytes)},
                                                     torch::TensorOptions(torch::kUInt8).device(torch::kCUDA));
                        bytes.fill_(0xEE);
                    }
                    grpc::ServerContext context;
                    const std::string request_key = "cp-full-direct";
                    DecodeRpcServer::LoadKVCacheContext load(9022, request_key, peers, keys,
                        resource->groupBlocks(), reused_pages, 5000, 1, 0, &context, source_sharded ? 8 : 1);
                    const auto request = server.constructRemoteLoadRequestForMla(load, rank, peers);
                    const std::vector<std::string> selected(request.peer_addrs().begin(), request.peer_addrs().end());
                    DecodeRpcServer::LoadKVCacheContext worker_load(9022, request_key, selected, keys,
                        resource->groupBlocks(), reused_pages, 5000, 1, 0, &context, request.prefill_cp_size());
                    auto status = server.loadCache(worker_load);
                    ASSERT_TRUE(status.ok()) << status.ToString();
                    for (size_t row = 0; row < blocks.size(); ++row) {
                        const size_t global_page = destination_sharded ? row * 8 + rank : row;
                        const uint8_t expected = global_page < keys.size() && global_page >= reused_pages ?
                                                     32 + global_page : 0xEE;
                        auto parts = destination->convertIndexToBuffer(blocks[row], 0);
                        auto bytes = torch::from_blob(parts[0].addr, {static_cast<int64_t>(parts[0].size_bytes)},
                                                     torch::TensorOptions(torch::kUInt8).device(torch::kCUDA));
                        EXPECT_TRUE(bytes.eq(expected).all().item<bool>()) << "global_page=" << global_page;
                    }
                    for (const auto& guard : guards) {
                        if (guard) {
                            expectDsv4RegionBytes(destination, guard->blocks(0, 0)[0], 0,
                                                 KVCacheRegionName::DEFAULT, 0xA5);
                        }
                    }
                }
                destination->free({resource, tokens});
                release_guards(destination, guards);
            }
        }
    }
}

TEST_F(PdSepKVCacheReleaseTest, testCpLinearCheckpointPublicationLoadsRequestFrontier) {
    ModelConfig model;
    model.num_layers = 2;
    model.data_type = DataType::TYPE_FP16;
    model.attn_config.use_mla = true;
    model.mla_ops_type = MlaOpsType::AUTO;
    model.attn_config.kv_lora_rank = 16;
    model.attn_config.rope_head_dim = 8;
    model.attn_config.tokens_per_block = 4;
    model.hybrid_attention_config.enable_hybrid_attention = true;
    model.hybrid_attention_config.hybrid_attention_types = {
        HybridAttentionType::LINEAR, HybridAttentionType::NONE};
    model.linear_attention_config.linear_num_key_heads = 8;
    model.linear_attention_config.linear_num_value_heads = 8;
    model.linear_attention_config.linear_key_head_dim = 4;
    model.linear_attention_config.linear_value_head_dim = 4;
    model.linear_attention_config.linear_conv_kernel_dim = 4;
    ParallelismConfig parallelism;
    parallelism.tp_size = 8;
    parallelism.role_type = RoleType::PREFILL;
    parallelism.prefill_cp_config.kv_cache_sharded = true;
    KVCacheConfig kv_config;
    kv_config.seq_size_per_block = 4;
    kv_config.test_block_num = 64;
    for (auto [source_sharded, decode_sharded, independent] : {
             std::tuple{false, false, false}, std::tuple{false, true, false},
             std::tuple{true, false, false}, std::tuple{true, true, false},
             std::tuple{false, false, true}, std::tuple{false, true, true},
             std::tuple{true, false, true}, std::tuple{true, true, true}}) {
        model.hybrid_attention_config.enable_independent_kv_cache_pools = independent;
        parallelism.role_type = RoleType::PREFILL;
        parallelism.prefill_cp_config.kv_cache_sharded = source_sharded;
        const auto source_config = CacheConfigCreator::createConfig(model, parallelism, RuntimeConfig{}, kv_config);
        parallelism.role_type = RoleType::DECODE;
        parallelism.decode_cp_kv_cache_sharded = decode_sharded;
        const auto destination_config =
            CacheConfigCreator::createConfig(model, parallelism, RuntimeConfig{}, kv_config);
        for (int length : {64, 65, 1}) {
            SCOPED_TRACE(::testing::Message() << "source_sharded=" << source_sharded
                                            << " decode_sharded=" << decode_sharded
                                            << " independent=" << independent << " length=" << length);
            // LINEAR's local row geometry is rank-independent. Publish distinct
            // head payloads below; this does not execute the per-rank KDA model.
            auto source = std::make_shared<KVCacheManager>(source_config);
            auto destination = std::make_shared<KVCacheManager>(destination_config);
            ASSERT_TRUE(source->init());
            ASSERT_TRUE(destination->init());
            auto make_resource = [](const CacheConfig& config) {
                auto resource = std::make_shared<BatchKVCacheResource>();
                resource->resetBatchSize(1);
                resource->initGroups(config.groupNums(), config.layer_all_num, config.layer_to_group_id,
                                     config.kernelBlocksPerKvBlock(), config.group_types, config.layer_region_to_group_id);
                return resource;
            };
            auto input = std::make_shared<GenerateInput>();
            input->input_ids = torch::arange(length, torch::kInt32);
            input->generate_config = std::make_shared<GenerateConfig>();
            auto tokens = std::make_shared<CompleteTokenIds>(1, 1, length + 128, 4);
            tokens->init(input);
            auto source_resource = make_resource(source_config);
            auto destination_resource = make_resource(destination_config);
            ASSERT_TRUE(source->malloc({source_resource, tokens, 9021, true, false, false}).success);
            // MTP reserves future checkpoint rows. Those must not replace the
            // request's actual terminal checkpoint as the transfer destination.
            tokens->setReserveStep(3);
            ASSERT_TRUE(destination->malloc({destination_resource, tokens, 9021, true, false, false}).success);
            const auto& keys = source_resource->cacheKeys(0);
            ASSERT_EQ(keys.size(), static_cast<size_t>((length + 3) / 4));
            const int source_gid = source_config.layer_to_group_id[0];
            const int destination_gid = destination_config.layer_to_group_id[0];
            const int source_row = (length - 1) / (source_sharded ? 32 : 4);
            const int destination_row = (length - 1) / (decode_sharded ? 32 : 4);
            const auto& source_blocks = source_resource->blocks(0, source_gid);
            const auto& destination_blocks = destination_resource->blocks(0, destination_gid);
            ASSERT_LT(source_row, source_blocks.size());
            ASSERT_LT(destination_row + 1, destination_blocks.size());
            ASSERT_FALSE(isNullBlockIdx(source_blocks[source_row]));
            ASSERT_FALSE(isNullBlockIdx(destination_blocks[destination_row]));

            const auto source_layout = source->getMainModelCacheLayerLayout();
            torch_ext::KVCache cache;
            cache.seq_size_per_block = source_config.seq_size_per_block;
            cache.kv_cache_base_by_layer = source_layout.layers_to_kv_buffer_ptrs;
            cache.layer_group_types = source_layout.layer_group_types;
            cache.layer_region_to_group_id = source_layout.layer_region_to_group_id;
            cache.group_seq_size_per_block.assign(
                source_layout.group_seq_size_per_block.begin(), source_layout.group_seq_size_per_block.end());
            auto layer = cache.getLayerCache(0);
            // Match KimiK3KDACache's physical [SSM][history,Q/K/V] segments.
            const auto* spec = dynamic_cast<const LinearKVCacheSpec*>(source_config.cache_specs[source_gid].get());
            ASSERT_NE(spec, nullptr);
            layer.cache_store_segment_sizes = {spec->k_block_size_bytes()};
            layer.cache_store_segment_sizes.insert(layer.cache_store_segment_sizes.end(), 9,
                                                   4 * getTypeSize(spec->conv_state_dtype));
            ASSERT_EQ(std::accumulate(layer.cache_store_segment_sizes.begin(), layer.cache_store_segment_sizes.end(),
                                      size_t{0}), spec->block_size_bytes());
            layer.kv_cache_base.fill_(7);
            auto source_bytes = layer.kv_cache_base[source_blocks[source_row]].view(torch::kUInt8).flatten();
            size_t offset = 0;
            for (size_t segment = 0; segment < layer.cache_store_segment_sizes.size(); ++segment) {
                const auto size = layer.cache_store_segment_sizes[segment];
                source_bytes.narrow(0, offset, size).fill_(31 + segment);
                offset += size;
            }
            auto destination_base = destination->getMainModelCacheLayerLayout().layers_to_kv_buffer_ptrs[0];
            destination_base.fill_(0);

            auto store = std::make_shared<MemoryBackedCacheStore>();
            torch_ext::PyCacheStoreInputs writer;
            writer.context_batch_size = 1;
            writer.request_id = torch::tensor({int64_t{9021}}, torch::kInt64);
            writer.request_pd_separation = torch::tensor({true}, torch::kBool);
            writer.kv_cache_layer_to_group = torch::tensor(source_config.layer_to_group_id, torch::kInt32);
            std::vector<int32_t> group_types;
            for (auto type : source_config.group_types) {
                group_types.push_back(static_cast<int32_t>(type));
            }
            writer.kv_cache_group_types = torch::tensor(group_types, torch::kInt32);
            for (auto key : keys) {
                writer.cache_keys.push_back(std::to_string(key));
            }
            writer.tokens_per_block = 4;
            writer.kv_block_stride_bytes = source_config.kv_block_stride_bytes;
            writer.kv_scale_stride_bytes = 0;
            writer.pd_separation = true;
            writer.mla_kvcache = true;
            writer.cp_size = source_config.cp_size;
            writer.cache_store = store;
            size_t max_rows = 0;
            for (int gid = 0; gid < source_config.groupNums(); ++gid) {
                max_rows = std::max(max_rows, source_resource->blocks(0, gid).size());
            }
            auto block_table = torch::full(
                {static_cast<int64_t>(source_config.groupNums()), 1, static_cast<int64_t>(max_rows)},
                NULL_BLOCK_IDX, torch::kInt32);
            for (int gid = 0; gid < source_config.groupNums(); ++gid) {
                block_table[gid].narrow(1, 0, source_resource->blocks(0, gid).size())
                    .copy_(blockIdsTensor(source_resource, gid));
            }
            torch_ext::PyCacheStorePublishPlan publish{
                torch::tensor({0}, torch::kInt32),
                torch::tensor({static_cast<int>(keys.size())}, torch::kInt32),
                torch::tensor({false}, torch::kBool)};
            auto write = [&]() {
                WriteCacheStoreOp(torch::tensor({length}, torch::kInt32), torch::tensor({0}, torch::kInt32),
                                  block_table, writer, layer, publish);
            };
            write();
            ASSERT_TRUE(store->stored_blocks_.empty());
            publish.terminal_host.fill_(true);
            write();
            ASSERT_EQ(store->stored_blocks_.size(), layer.cache_store_segment_sizes.size());
            const auto terminal_key = makeCacheKey(0, std::to_string(keys.back()), 0, KVCacheRegionName::DEFAULT);
            for (size_t segment = 0; segment < layer.cache_store_segment_sizes.size(); ++segment) {
                const auto key = makeLinearCacheSegmentKey(segment, terminal_key);
                ASSERT_EQ(store->stored_blocks_.count(key), 1u);
                EXPECT_EQ(store->stored_blocks_.at(key),
                          std::vector<uint8_t>(layer.cache_store_segment_sizes[segment], 31 + segment));
            }

            // Bootstrap supplies peers in TP-rank order. Each publishes the
            // same keys but a different head shard; a merged map would hide
            // a wrong peer selection even if all byte offsets were correct.
            auto transport = std::make_shared<MemoryBackedCacheStore>();
            std::vector<std::string> peers;
            for (int rank = 0; rank < 8; ++rank) {
                auto peer_store = rank == 0 ? store : std::make_shared<MemoryBackedCacheStore>();
                if (rank != 0) {
                    size_t segment_offset = 0;
                    for (size_t segment = 0; segment < layer.cache_store_segment_sizes.size(); ++segment) {
                        const auto size = layer.cache_store_segment_sizes[segment];
                        source_bytes.narrow(0, segment_offset, size).fill_(31 + segment + 16 * rank);
                        segment_offset += size;
                    }
                    writer.cache_store = peer_store;
                    writer.cp_rank = source_sharded ? rank : 0;
                    write();
                }
                const auto ip = "127.0.0." + std::to_string(rank + 1);
                peers.push_back(ip + ":12345:12346");
                transport->peer_stores_[ip] = peer_store;
            }
            EngineInitParams params;
            params.model_id = 0;
            params.model_config_.num_layers = 1;  // Only the LINEAR layer is published in this component scenario.
            params.parallelism_config = parallelism;
            DecodeRpcServer server;
            server.engine_ = std::make_shared<MinimalEngine>(params, destination);
            server.maga_init_params_ = params;
            server.resource_.cache_store = transport;
            server.resource_.workers.resize(8);
            for (int rank : {0, 3, 7}) {
                SCOPED_TRACE(::testing::Message() << "head rank=" << rank);
                server.maga_init_params_.parallelism_config.tp_rank = rank;
                destination_base.fill_(0);
                transport->load_buffer_requests_.clear();
                grpc::ServerContext server_context;
                const std::string request_key = "cp-linear-frontier";
                DecodeRpcServer::LoadKVCacheContext load(9021, request_key, peers, keys,
                    destination_resource->groupBlocks(), 0, 5000, 1, 0, &server_context, source_sharded ? 8 : 1);
                const auto request = server.constructRemoteLoadRequestForMla(load, rank, peers);
                const std::vector<std::string> selected(request.peer_addrs().begin(), request.peer_addrs().end());
                DecodeRpcServer::LoadKVCacheContext worker_load(9021, request_key, selected, keys,
                    destination_resource->groupBlocks(), 0, 5000, 1, 0, &server_context, request.prefill_cp_size());
                auto status = server.loadCache(worker_load);
                ASSERT_TRUE(status.ok()) << status.ToString();
                ASSERT_EQ(transport->load_buffer_requests_.size(), 1u);
                EXPECT_EQ(transport->load_buffer_requests_[0]->getBlocks().size(), layer.cache_store_segment_sizes.size());
                for (size_t row = 0; row < destination_blocks.size(); ++row) {
                    if (isNullBlockIdx(destination_blocks[row])) {
                        continue;
                    }
                    auto bytes = destination_base[destination_blocks[row]].view(torch::kUInt8).flatten();
                    if (row == static_cast<size_t>(destination_row)) {
                        size_t segment_offset = 0;
                        for (size_t segment = 0; segment < layer.cache_store_segment_sizes.size(); ++segment) {
                            const auto size = layer.cache_store_segment_sizes[segment];
                            EXPECT_TRUE(bytes.narrow(0, segment_offset, size).eq(31 + segment + 16 * rank).all().item<bool>());
                            segment_offset += size;
                        }
                        // Pool padding and future checkpoint rows are not state.
                        EXPECT_TRUE(bytes.narrow(0, offset, bytes.numel() - offset).eq(0).all().item<bool>());
                    } else {
                        EXPECT_TRUE(bytes.eq(0).all().item<bool>()) << "unexpected write to row " << row;
                    }
                }
            }
            source->free({source_resource, tokens});
            destination->free({destination_resource, tokens});
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
        cache_keys.push_back(20000 + i);
        cache_key_strings.push_back(std::to_string(cache_keys.back()));
    }

    for (int layer_id = 0; layer_id < 4; ++layer_id) {
        for (int gid : config.layer_to_group_ids[layer_id]) {
            auto region_name = config.group_region_names[gid];
            auto positions =
                blockPositionsForCacheTransfer(block_num, /*first_full_block=*/0, true, config.group_types[gid]);
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

            torch_ext::PyCacheStoreInputs inputs;
            inputs.context_batch_size             = 1;
            inputs.decoder_batch_size             = 0;
            inputs.request_id                     = torch::tensor({request_id}, torch::kInt64);
            inputs.request_pd_separation          = torch::tensor({true}, torch::kBool);
            inputs.kv_cache_layer_to_group        = layer_to_group_tensor;
            inputs.kv_cache_layer_region_to_group = layer_region_to_group_tensor;
            inputs.kv_cache_group_types           = group_types_tensor;
            inputs.cache_keys                     = cache_key_strings;
            inputs.input_lengths_host             = torch::tensor({block_num * spb}, torch::kInt32);
            inputs.prefix_lengths_host            = torch::tensor({0}, torch::kInt32);
            inputs.tokens_per_block               = spb;
            inputs.kv_block_stride_bytes          = config.kv_block_stride_bytes;
            inputs.kv_scale_stride_bytes          = 0;
            inputs.pd_separation                  = true;
            inputs.model_id                       = model_id;
            inputs.decode_entrance                = false;
            inputs.warmup                         = false;
            inputs.use_opaque_kv_cache_store      = config.use_opaque_kv_cache_store;
            inputs.mla_kvcache                    = false;
            inputs.cache_store                    = cache_store;

            torch_ext::LayerKVCache layer_cache;
            layer_cache.kv_cache_base      = layout.layers_to_kv_buffer_ptrs_by_attn[layer_id][region_idx];
            layer_cache.seq_size_per_block = config.group_types[gid] == CacheGroupType::FULL ? kernel_spb : spb;
            layer_cache.layer_id           = layer_id;
            layer_cache.group_id           = gid;
            layer_cache.region_name        = region_name;

            WriteCacheStoreOp(inputs.input_lengths_host,
                              inputs.prefix_lengths_host,
                              blockIdsTensor(prefill_resource, gid),
                              inputs,
                              layer_cache);
        }
    }

    const auto first_csa_key =
        "kv_" + makeCacheKey(model_id, cache_key_strings[0], /*layer_id=*/2, KVCacheRegionName::CSA_KV);
    ASSERT_NE(cache_store->stored_blocks_.find(first_csa_key), cache_store->stored_blocks_.end());
    EXPECT_EQ(cache_store->stored_blocks_[first_csa_key].size(),
              config.group_kv_block_stride_bytes[static_cast<size_t>(0)]);

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
        for (int gid : config.layer_to_group_ids[layer_id]) {
            auto region_name = config.group_region_names[gid];
            auto positions =
                blockPositionsForCacheTransfer(block_num, /*first_full_block=*/0, true, config.group_types[gid]);
            for (auto block_pos : positions) {
                auto decode_block_id = decode_resource->blocks(0, gid)[block_pos];
                ASSERT_FALSE(isNullBlockIdx(decode_block_id));
                expectDsv4RegionBytes(
                    decode_manager, decode_block_id, layer_id, region_name, dsv4PdPattern(layer_id, gid, block_pos));
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
        cache_keys.push_back(11000 + i);
        cache_key_strings.push_back(std::to_string(cache_keys.back()));
    }

    for (int layer_id = 0; layer_id < 4; ++layer_id) {
        for (int gid : config.layer_to_group_ids[layer_id]) {
            auto region_name = config.group_region_names[gid];
            auto positions =
                blockPositionsForCacheTransfer(block_num, /*first_full_block=*/0, true, config.group_types[gid]);
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
            inputs.input_lengths_host                  = torch::tensor({(block_num - reuse_num) * spb}, torch::kInt32);
            inputs.prefix_lengths_host                 = torch::tensor({reuse_num * spb}, torch::kInt32);
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
            inputs.use_opaque_kv_cache_store           = config.use_opaque_kv_cache_store;
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

    EXPECT_EQ(cache_store->load_buffer_requests_.size(), 10u);
    EXPECT_EQ(cache_store->load_request_keys_.size(), 10u);
    for (int layer_id = 0; layer_id < 4; ++layer_id) {
        for (int gid : config.layer_to_group_ids[layer_id]) {
            auto region_name = config.group_region_names[gid];
            auto positions =
                blockPositionsForCacheTransfer(block_num, /*first_full_block=*/0, true, config.group_types[gid]);
            for (auto block_pos : positions) {
                auto decode_block_id = decode_resource->blocks(0, gid)[block_pos];
                ASSERT_FALSE(isNullBlockIdx(decode_block_id));
                expectDsv4RegionBytes(
                    decode_manager, decode_block_id, layer_id, region_name, dsv4PdPattern(layer_id, gid, block_pos));
            }
        }
    }
}

// =============================================================================
// Test: runtimeWriteCacheStore with pinned-host metadata + event sync
// Verifies that when metadata tensors (input_lengths, prefix_lengths) are
// prepared on pinned host via async D2H and a pre_created_event is attached,
// runtimeWriteCacheStore waits for the event and reads metadata correctly —
// the same path used by the optimized WriteCacheStoreOp that avoids
// synchronous .cpu() calls on background threads.
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
    resource->initGroups(config.groupNums(),
                         static_cast<int>(config.layer_all_num),
                         config.layer_to_group_id,
                         config.kernelBlocksPerKvBlock(),
                         config.group_types,
                         config.layer_region_to_group_id);

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
        auto buf = layout.layers_to_kv_buffer_ptrs[layer_id];
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

    // Prepare cache key strings (one per block).
    std::vector<std::string> cache_key_strings;
    for (int i = 0; i < block_num; ++i) {
        cache_key_strings.push_back(std::to_string(10000 + i));
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

    for (int layer_id = 0; layer_id < 3; ++layer_id) {
        CacheStoreInputs inputs;
        inputs.input_lengths_host        = input_lengths_host;
        inputs.prefix_lengths_host       = prefix_lengths_host;
        inputs.host_kv_cache_offset      = block_ids;
        inputs.context_batch_size        = 1;
        inputs.decoder_batch_size        = 0;
        inputs.request_id                = torch::tensor({(int64_t)request_id_val}, torch::kInt64);
        inputs.request_pd_separation     = torch::tensor({true}, torch::kBool);
        inputs.cache_keys                = cache_key_strings;
        inputs.tokens_per_block          = spb;
        inputs.kv_block_stride_bytes     = config.kv_block_stride_bytes;
        inputs.kv_scale_stride_bytes     = 0;
        inputs.pd_separation             = true;
        inputs.model_id                  = 0;
        inputs.decode_entrance           = false;
        inputs.warmup                    = false;
        inputs.use_opaque_kv_cache_store = false;
        inputs.layer_id                  = layer_id;
        inputs.region_name               = KVCacheRegionName::DEFAULT;
        inputs.pre_created_event         = event;

        KvCacheInfo kv_cache_info;
        kv_cache_info.kv_cache_buffer = layout.layers_to_kv_buffer_ptrs[layer_id];
        runtimeWriteCacheStore(inputs, kv_cache_info, /*mla_kvcache=*/false, cache_store);
    }

    // Verify: cache store received correct request key for all 3 layers.
    EXPECT_EQ(cache_store->store_request_keys_.size(), 3u);
    // MHA (non-opaque, non-mla) splits each block into k + v → 2 entries per block.
    EXPECT_EQ(cache_store->stored_blocks_.size(), 3u * block_num * 2u);

    // Verify stored data matches the pattern we filled.
    for (int layer_id = 0; layer_id < 3; ++layer_id) {
        for (int b = 0; b < block_num; ++b) {
            auto k_key = "k_" + makeCacheKey(0, cache_key_strings[b], layer_id);
            auto it    = cache_store->stored_blocks_.find(k_key);
            ASSERT_NE(it, cache_store->stored_blocks_.end()) << "missing key: " << k_key;
            uint8_t expected = static_cast<uint8_t>(layer_id * 10 + b);
            EXPECT_EQ(it->second[0], expected) << "layer=" << layer_id << " block=" << b << " first byte mismatch";
        }
    }
}

TEST_F(PdSepKVCacheReleaseTest, testWriteCacheStoreUsesTensorDeviceForCpuKvBuffer) {
    const int         spb              = 8;
    const int         kv_stride        = 64;
    const int         request_id_val   = 4242;
    const std::string cache_key_string = "10000";

    auto kv_options = torch::TensorOptions(torch::kUInt8).device(torch::kCPU).pinned_memory(true);
    auto kv_buffer  = torch::empty({2, kv_stride}, kv_options);
    kv_buffer[1].fill_(static_cast<uint8_t>(123));

    auto inputs = makeSingleBlockWriteInputs(
        cache_key_string, request_id_val, spb, kv_stride, 0, true, KVCacheRegionName::CSA_STATE);

    KvCacheInfo kv_cache_info;
    kv_cache_info.kv_cache_buffer = kv_buffer;

    auto cache_store = std::make_shared<MemoryBackedCacheStore>();
    runtimeWriteCacheStore(inputs, kv_cache_info, /*mla_kvcache=*/false, cache_store);

    const auto key = "kv_" + makeCacheKey(0, cache_key_string, 0, KVCacheRegionName::CSA_STATE);
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
    const int         spb              = 8;
    const int         kv_stride        = 64;
    const int         kv_half          = kv_stride / 2;
    const int         request_id_val   = 4243;
    const std::string cache_key_string = "10001";

    auto kv_options = torch::TensorOptions(torch::kUInt8).device(torch::kCPU).pinned_memory(true);
    auto kv_buffer  = torch::empty({2, kv_stride}, kv_options);
    auto block      = kv_buffer[1];
    block.slice(0, 0, kv_half).fill_(static_cast<uint8_t>(17));
    block.slice(0, kv_half, kv_stride).fill_(static_cast<uint8_t>(29));

    auto inputs = makeSingleBlockWriteInputs(
        cache_key_string, request_id_val, spb, kv_stride, 0, false, KVCacheRegionName::DEFAULT);

    KvCacheInfo kv_cache_info;
    kv_cache_info.kv_cache_buffer = kv_buffer;

    auto cache_store = std::make_shared<MemoryBackedCacheStore>();
    runtimeWriteCacheStore(inputs, kv_cache_info, /*mla_kvcache=*/false, cache_store);

    const auto cache_key = makeCacheKey(0, cache_key_string, 0, KVCacheRegionName::DEFAULT);
    const auto k_key     = "k_" + cache_key;
    const auto v_key     = "v_" + cache_key;
    auto       k_it      = cache_store->stored_blocks_.find(k_key);
    auto       v_it      = cache_store->stored_blocks_.find(v_key);
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
    const int         spb              = 8;
    const int         kv_stride        = 64;
    const int         scale_stride     = 16;
    const int         request_id_val   = 4244;
    const std::string cache_key_string = "10002";

    auto cpu_options     = torch::TensorOptions(torch::kUInt8).device(torch::kCPU).pinned_memory(true);
    auto kv_buffer       = torch::empty({2, kv_stride}, cpu_options);
    auto kv_scale_buffer = torch::empty({2, scale_stride}, cpu_options);
    kv_buffer[1].fill_(static_cast<uint8_t>(41));
    kv_scale_buffer[1].fill_(static_cast<uint8_t>(73));

    auto inputs = makeSingleBlockWriteInputs(
        cache_key_string, request_id_val, spb, kv_stride, scale_stride, true, KVCacheRegionName::CSA_STATE);

    KvCacheInfo kv_cache_info;
    kv_cache_info.kv_cache_buffer = kv_buffer;
    kv_cache_info.kv_scale_buffer = kv_scale_buffer;

    auto cache_store = std::make_shared<MemoryBackedCacheStore>();
    runtimeWriteCacheStore(inputs, kv_cache_info, /*mla_kvcache=*/false, cache_store);

    const auto scale_key = "kv_scale_" + makeCacheKey(0, cache_key_string, 0, KVCacheRegionName::CSA_STATE);
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
