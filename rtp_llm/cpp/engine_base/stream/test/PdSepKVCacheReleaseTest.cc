#include <future>

#include "gtest/gtest.h"
#include "rtp_llm/cpp/cache/test/TestLayoutSpec.h"
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

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
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

class CancellableDecodeLoad final: public DecodeGenerateContext {
public:
    using DecodeGenerateContext::DecodeGenerateContext;
    bool isRequestCancelled() const override {
        if (!cancelled.load()) {
            return false;
        }
        std::call_once(observed_once, [this] { cancel_observed.set_value(); });
        return true;
    }

    std::atomic<bool>          cancelled{false};
    mutable std::once_flag     observed_once;
    mutable std::promise<void> cancel_observed;
};

class DummyMemoryUtil: public MemoryUtil {
public:
    explicit DummyMemoryUtil(bool rdma = false, bool supports_write_stop = true):
        rdma_(rdma), supports_write_stop_(supports_write_stop) {}

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
        return rdma_;
    }
    bool supportsLoadWriteStop() override {
        return supports_write_stop_;
    }

private:
    bool rdma_;
    bool supports_write_stop_;
};

class MemoryBackedCacheStore: public NormalCacheStore {
public:
    explicit MemoryBackedCacheStore(bool rdma = false) {
        memory_util_                = std::make_shared<DummyMemoryUtil>(rdma);
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
        load_buffer_requests_.push_back(request_block_buffer);
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
        callback.destinationWritesStopped();
        callback(ok, ok ? CacheStoreErrorCode::None : CacheStoreErrorCode::LoadErrorUnknown);
    }

    std::unordered_map<std::string, std::vector<uint8_t>> stored_blocks_;
    std::vector<std::string>                              store_request_keys_;
    std::vector<std::string>                              load_request_keys_;
    std::vector<std::shared_ptr<RequestBlockBuffer>>      store_buffer_requests_;
    std::vector<std::shared_ptr<RequestBlockBuffer>>      load_buffer_requests_;
};

class DelayedRdmaCacheStore: public MemoryBackedCacheStore {
public:
    DelayedRdmaCacheStore(): MemoryBackedCacheStore(true) {}

    void load(const std::shared_ptr<RequestBlockBuffer>& buffer,
              CacheStoreLoadDoneCallback                 callback,
              const std::string&,
              uint32_t,
              uint32_t,
              uint32_t,
              int,
              int) override {
        load_buffer_requests_.push_back(buffer);
        ASSERT_FALSE(pending);
        pending = std::move(callback);
        submitted.set_value();
    }

    std::promise<void>         submitted;
    CacheStoreLoadDoneCallback pending;
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
    auto parts = manager->convertIndexToBuffer(layer_id, tag, block_id);
    ASSERT_EQ(parts.size(), 1u);
    auto device = torch::from_blob(
        parts[0].addr, {(int64_t)parts[0].size_bytes}, torch::TensorOptions(torch::kUInt8).device(torch::kCUDA));
    auto host =
        torch::full({(int64_t)parts[0].size_bytes}, value, torch::TensorOptions(torch::kUInt8).device(torch::kCPU));
    device.copy_(host);
}

void expectDsv4RegionBytes(
    const std::shared_ptr<KVCacheManager>& manager, int block_id, int layer_id, const std::string& tag, uint8_t value) {
    auto parts = manager->convertIndexToBuffer(layer_id, tag, block_id);
    ASSERT_EQ(parts.size(), 1u);
    auto device = torch::from_blob(
        parts[0].addr, {(int64_t)parts[0].size_bytes}, torch::TensorOptions(torch::kUInt8).device(torch::kCUDA));
    auto        host = device.cpu().contiguous();
    const auto* ptr  = host.data_ptr<uint8_t>();
    for (size_t i = 0; i < parts[0].size_bytes; ++i) {
        ASSERT_EQ(ptr[i], value) << "byte=" << i << " layer=" << layer_id << " block=" << block_id << " tag=" << tag;
    }
}

uint8_t dsv4PdPattern(const CacheConfig& config, int layer_id, const std::string& tag, size_t block_pos) {
    // Keep the original fixture byte pattern in its published topology order.
    const auto& tags = config.groupTags();
    const auto  gid  = std::find(tags.begin(), tags.end(), tag) - tags.begin();
    return static_cast<uint8_t>(17 + layer_id * 19 + gid * 11 + block_pos);
}

// Block positions a group actually transfers on the PD prefill->decode path.
// Both sides derive the plan from the group policy through buildCacheStorePlan:
// runtimeWriteCacheStore directly, DecodeRpcServer::loadCache via
// DecodeRpcServer::buildGroupLoadPlan. This helper mirrors that plan for
// cp_size == 1, where it reduces to blockPositionsForCacheTransfer.
std::vector<size_t>
dsv4TransferPositions(const CacheConfig& config, const std::string& tag, size_t block_num, size_t reuse_block_size) {
    const auto policy = config.group(tag).policy;
    return blockPositionsForCacheTransfer(block_num,
                                          reuse_block_size,
                                          /*use_hybrid=*/config.groupNums() > 1,
                                          /*transfer_tail_blocks=*/policy.active_tail_blocks > 0,
                                          static_cast<size_t>(policy.active_tail_blocks),
                                          /*hybrid_full_from_begin=*/true);
}

torch::Tensor blockIdsTensor(const BatchKVCacheResourcePtr& resource, const std::string& tag) {
    const auto& blocks = resource->blocks(0, tag);
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
    rtp_llm::test::setGroupBlockLayout(config, {tag}, {kBlockNum}, {kv_stride}, {kv_scale_stride});
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

// Collect holders in the resource's own order, retaining their shared ownership.

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
        mc.attn_config.layer_compress_ratios   = ratios;
        mc.attn_config.tokens_per_block        = seq_size_per_block;
        mc.attn_config.kernel_tokens_per_block = kernel_seq_size_per_blk;
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
        auto config                         = CacheConfigCreator::createWarmupConfig(mc, pc, kv_config, 0);
        // Publish fixture capacity before consumers inspect the group layout.
        config.finalizeBlockNums(block_num, RuntimeConfig{});
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

TEST_F(PdSepKVCacheReleaseTest, DecodePDLoadSnapshotAndPinsSurviveRelease) {
    prepareStreamWithConfig({1, 2, 3, 4, 5, 6, 7, 8, 9}, makeConfig(), 8, RoleType::DECODE);
    auto& resource = stream_->streamCacheResource();
    ASSERT_TRUE(resource.initKVBlock().ok());
    CacheKeysType keys;
    GroupBlockIds groups;
    int64_t       reuse_blocks = -1;
    const auto    select_tail  = [](const GroupBlockIds& rows, size_t, int64_t) {
        const auto& tag = rows.tag_to_index_.begin()->first;
        return PDLoadDestinationBlocks{{tag, {rows.blocks(tag).back()}}};
    };
    auto hold = stream_->holdPDLoadDestination(keys, groups, reuse_blocks, select_tail);
    ASSERT_TRUE(hold);
    EXPECT_EQ(keys, stream_->cacheKeys(0));
    EXPECT_EQ(reuse_blocks, stream_->reuseBlockSize());
    const auto& tag = cache_manager_->cacheConfig().groupTags().front();
    const auto& row = stream_->kvCachePtr()->cacheResource(0).groupBlockIds(tag);
    ASSERT_TRUE(row);
    const auto ids = row->blocks();
    ASSERT_GE(ids.size(), 2u);
    BlockIndicesType expected(ids.size(), NULL_BLOCK_IDX);
    expected.back() = ids.back();
    EXPECT_EQ(groups.blocks(tag), expected);
    EXPECT_EQ(row->blocks(), ids);
    row->swap(0, 1);
    EXPECT_EQ(groups.blocks(tag), expected);
    row->swap(0, 1);

    stream_->releaseResource();
    EXPECT_TRUE(resource.isResourceReleased());
    EXPECT_FALSE(stream_->holdPDLoadDestination(keys, groups, reuse_blocks, select_tail));
    const auto& pool = cache_manager_->coordinator_manager_->groupBlockPools().front();
    EXPECT_EQ(pool->referencedBlocksNum(DeviceBlockRefType::PD_CONNECTOR), 1u);
    EXPECT_TRUE(pool->isAllocated(ids.back()));
    EXPECT_FALSE(pool->isAllocated(ids.front()));
    hold.reset();
    EXPECT_EQ(pool->referencedBlocksNum(DeviceBlockRefType::PD_CONNECTOR), 0u);
}

TEST_F(PdSepKVCacheReleaseTest, MissingGroupIdentityRejectsAsyncLoadBeforeConnecting) {
    auto             manager = std::make_shared<KVCacheManager>(makeConfig(), /*warmup=*/false, nullptr);
    EngineInitParams params;
    DecodeRpcServer  server;
    server.engine_                = std::make_shared<MinimalEngine>(params, manager);
    server.resource_.workers      = {"decode-0"};
    server.resource_.grpc_workers = {"127.0.0.1:1"};
    grpc::ServerContext          server_context;
    DecodeRpcContext             rpc_context{nullptr};
    kmonitor::MetricsReporterPtr reporter;
    DecodeGenerateContext        decode_context(rpc_context, 1000, &server_context, reporter, nullptr);
    decode_context.peer_addrs = {"prefill-0"};
    // An uninitialized resource has no rows/tags. The request-only builders may
    // accept this, but the asynchronous dispatch entry must reject it locally.
    KVCacheResource                     resource;
    DecodeRpcServer::LoadKVCacheContext load_context(
        42, "missing-identity", {"prefill-0"}, {}, resource.groupBlockIds(), 0, 1000, 1, 0, &server_context);
    ASSERT_TRUE(server.resource_.rpc_pool.connection_pool_.empty());
    EXPECT_ANY_THROW(server.loadCacheAsyncForTp(decode_context, load_context));
    EXPECT_TRUE(server.resource_.rpc_pool.connection_pool_.empty());
}

TEST_F(PdSepKVCacheReleaseTest, MalformedGroupIndexRejectsAsyncLoadBeforeConnecting) {
    auto             manager = std::make_shared<KVCacheManager>(makeConfig(), /*warmup=*/false, nullptr);
    EngineInitParams params;
    DecodeRpcServer  server;
    server.engine_                = std::make_shared<MinimalEngine>(params, manager);
    server.resource_.workers      = {"decode-0"};
    server.resource_.grpc_workers = {"127.0.0.1:1"};
    grpc::ServerContext          server_context;
    DecodeRpcContext             rpc_context{nullptr};
    kmonitor::MetricsReporterPtr reporter;
    DecodeGenerateContext        decode_context(rpc_context, 1000, &server_context, reporter, nullptr);
    decode_context.peer_addrs = {"prefill-0"};
    KVCacheResource resource;
    resource.initGroups(manager->cacheConfig().topologyPtr());
    auto                             holder = std::make_shared<BlockIds>();
    const auto                       tag    = manager->cacheConfig().groupTags().front();
    const std::vector<GroupBlockIds> malformed{{{}, {holder}},
                                               {{{tag, 0}, {"extra", 0}}, {holder, holder}},
                                               {{{tag, 1}}, {holder}},
                                               {{{tag, 0}, {"extra", 2}}, {holder, holder}},
                                               {{{tag, 0}}, {nullptr}}};
    for (const auto& blocks : malformed) {
        DecodeRpcServer::LoadKVCacheContext context(
            42, "malformed-identity", {"prefill-0"}, {}, resource.groupBlockIds(), 0, 1000, 1, 0, &server_context);
        // Fault injection proves dispatch revalidates even after constructor validation.
        // Production only receives the const complete object.
        const_cast<GroupBlockIds&>(context.groupBlockIds()) = blocks;
        ASSERT_TRUE(server.resource_.rpc_pool.connection_pool_.empty());
        EXPECT_ANY_THROW(server.loadCacheAsyncForTp(decode_context, context));
        EXPECT_TRUE(server.resource_.rpc_pool.connection_pool_.empty());
    }
}

class DelayedTcpCacheStore: public MemoryBackedCacheStore {
public:
    DelayedTcpCacheStore(): MemoryBackedCacheStore(false) {}

    void load(const std::shared_ptr<RequestBlockBuffer>&,
              CacheStoreLoadDoneCallback callback,
              const std::string&,
              uint32_t,
              uint32_t,
              uint32_t,
              int,
              int) override {
        std::lock_guard<std::mutex> lock(mu);
        pending.push_back(std::move(callback));
        cv.notify_all();
    }

    bool waitPending(size_t expected) {
        std::unique_lock<std::mutex> lock(mu);
        return cv.wait_for(lock, std::chrono::seconds(5), [&] { return pending.size() >= expected; });
    }

    std::mutex                              mu;
    std::condition_variable                 cv;
    std::vector<CacheStoreLoadDoneCallback> pending;
};

std::future<void> observeDestinationFence(const CacheStoreLoadDoneCallback& callback) {
    auto seen   = std::make_shared<std::promise<void>>();
    auto future = seen->get_future();
    callback.setDestinationFence([seen] { seen->set_value(); });
    return future;
}

class PdSepRdmaLateCompletionTest: public PdSepKVCacheReleaseTest, public ::testing::WithParamInterface<bool> {};

TEST_P(PdSepRdmaLateCompletionTest, TimedOutLoadKeepsDecodeBlockUntilTransportSettlement) {
    constexpr int    tokens_per_block = 8;
    constexpr size_t target_blocks    = 9;  // >50% of 16-block pool (exceeds ~30%-40% frozen abort ratio).
    std::vector<int> prompt(target_blocks * tokens_per_block, 1);
    for (size_t i = 0; i < prompt.size(); ++i) {
        prompt[i] = static_cast<int>(i + 1);
    }
    prepareStreamWithConfig(prompt, makeConfig(), tokens_per_block, RoleType::DECODE);
    ASSERT_TRUE(stream_->initKVBlock().ok());
    const auto original_blocks = stream_->kvCachePtr()->blocks(0, "default");
    ASSERT_EQ(original_blocks.size(), target_blocks);
    const auto pool = cache_manager_->coordinator_manager_->groupBlockPools().front();

    auto             store = std::make_shared<DelayedRdmaCacheStore>();
    EngineInitParams params;
    params.model_id                 = 1;
    params.model_config_.num_layers = cache_manager_->cacheConfig().layer_num;
    DecodeRpcServer server;
    if (!GetParam()) {
        server.pd_write_recovery_delay_ = std::chrono::milliseconds(50);
    }
    server.maga_init_params_     = params;
    server.engine_               = std::make_shared<MinimalEngine>(params, cache_manager_);
    server.resource_.cache_store = store;
    server.resource_.workers     = {"decode"};
    grpc::ServerContext          server_context;
    DecodeRpcContext             rpc_context{nullptr};
    kmonitor::MetricsReporterPtr reporter;
    DecodeGenerateContext        context(
        rpc_context, /*timeout_ms=*/1, &server_context, reporter, std::make_shared<RpcServerRuntimeMeta>());
    context.peer_addrs = {"127.0.0.1:12345:12346"};
    context.setStream(stream_);
    const auto result = server.loadCacheForAllRank(context);
    EXPECT_EQ(result.error_info.code(), ErrorCode::CACHE_STORE_LOAD_BUFFER_TIMEOUT);
    EXPECT_FALSE(result.destination_writes_quiesced);
    ASSERT_TRUE(store->pending);
    auto fence = observeDestinationFence(store->pending);
    ASSERT_FALSE(store->load_buffer_requests_.empty());
    store->load_buffer_requests_.clear();

    stream_->reportError(result.error_info.code(), result.error_info.ToString());
    stream_->releaseResource();
    context.markRpcHandlingCompleted();
    EXPECT_EQ(pool->referencedBlocksNum(DeviceBlockRefType::PD_CONNECTOR), target_blocks);
    for (const auto b : original_blocks) {
        EXPECT_FALSE(pool->hasRequestRef(b));
        EXPECT_TRUE(pool->isAllocated(b));
        EXPECT_FALSE(pool->isFrozen(b));
    }

    // Give another request every remaining page, so the allocator cannot mask premature reuse.
    const int other_blocks       = static_cast<int>(initial_free_blocks_ - target_blocks);
    auto      other_input        = std::make_shared<GenerateInput>();
    other_input->generate_config = std::make_shared<GenerateConfig>();
    other_input->input_ids       = torch::arange(other_blocks * tokens_per_block, torch::kInt32);
    auto other_tokens =
        std::make_shared<CompleteTokenIds>(1, 1, other_blocks * tokens_per_block + tokens_per_block, tokens_per_block);
    other_tokens->init(other_input);
    auto other = std::make_shared<BatchKVCacheResource>();
    other->resetBatchSize(1);
    other->initGroups(cache_manager_->cacheConfig().topologyPtr());
    ASSERT_TRUE(cache_manager_->malloc({other, other_tokens, 2, true, false, false}).success);
    const auto& allocated = other->blocks(0, "default");
    EXPECT_EQ(allocated.size(), static_cast<size_t>(other_blocks));
    for (const auto b : original_blocks) {
        EXPECT_EQ(std::find(allocated.begin(), allocated.end(), b), allocated.end());
    }
    EXPECT_EQ(pool->freeBlocksNum(), 0u);
    EXPECT_FALSE(server.waitForDrain(std::chrono::system_clock::now()));

    // Business RPC completion alone (success or RDMA write failure) must NOT release or freeze blocks.
    const bool confirmed = GetParam();
    auto       stop      = store->pending.destinationStopNotifier();
    ASSERT_TRUE(stop);
    store->pending(confirmed, confirmed ? CacheStoreErrorCode::None : CacheStoreErrorCode::LoadRdmaWriteFailed);
    store->pending = {};
    for (const auto b : original_blocks) {
        EXPECT_FALSE(pool->isFrozen(b));
    }
    EXPECT_EQ(pool->referencedBlocksNum(DeviceBlockRefType::PD_CONNECTOR), target_blocks);
    EXPECT_EQ(pool->freeBlocksNum(), 0u);

    if (!confirmed) {
        ASSERT_EQ(fence.wait_for(std::chrono::seconds(5)), std::future_status::ready);
        ASSERT_NO_THROW(fence.get());
        EXPECT_EQ(pool->referencedBlocksNum(DeviceBlockRefType::PD_CONNECTOR), target_blocks);
    } else {
        EXPECT_EQ(fence.wait_for(std::chrono::seconds(0)), std::future_status::timeout);
    }
    stop();
    EXPECT_EQ(pool->referencedBlocksNum(DeviceBlockRefType::PD_CONNECTOR), 0u);
    for (const auto b : original_blocks) {
        EXPECT_FALSE(pool->isFrozen(b));
        EXPECT_FALSE(pool->isAllocated(b));
    }
    EXPECT_EQ(pool->referencedBlocksNum(DeviceBlockRefType::PD_CONNECTOR), 0u);
    EXPECT_EQ(pool->freeBlocksNum(), target_blocks);
    EXPECT_TRUE(server.waitForDrain(std::chrono::system_clock::now()));
    cache_manager_->free({other, other_tokens, 2});
    EXPECT_EQ(pool->freeBlocksNum(), initial_free_blocks_);
}

class PdSepSinglePTcpDirectCancelTest: public PdSepKVCacheReleaseTest, public ::testing::WithParamInterface<bool> {};

TEST_P(PdSepSinglePTcpDirectCancelTest, CancelledSingleWorkerTcpLoadHoldsBlocksUntilEveryLocalCopyStops) {
    constexpr int    tokens_per_block = 8;
    constexpr size_t target_blocks    = 9;  // >50% of 16-block pool (exceeds ~30%-40% frozen abort ratio).
    std::vector<int> prompt(target_blocks * tokens_per_block, 1);
    for (size_t i = 0; i < prompt.size(); ++i) {
        prompt[i] = static_cast<int>(i + 1);
    }
    prepareStreamWithConfig(prompt, makeConfig(), tokens_per_block, RoleType::DECODE);
    ASSERT_TRUE(stream_->initKVBlock().ok());
    const auto original_blocks = stream_->kvCachePtr()->blocks(0, "default");
    ASSERT_EQ(original_blocks.size(), target_blocks);
    const auto pool = cache_manager_->coordinator_manager_->groupBlockPools().front();

    auto             store  = std::make_shared<DelayedTcpCacheStore>();
    const size_t     layers = cache_manager_->cacheConfig().layer_num;
    EngineInitParams params;
    params.model_id                            = 1;
    params.model_config_.num_layers            = layers;
    params.pd_sep_config.load_cache_timeout_ms = 10000;
    params.pd_sep_config.max_rpc_timeout_ms    = 10000;
    DecodeRpcServer server;
    if (!GetParam()) {
        server.pd_write_recovery_delay_ = std::chrono::milliseconds(50);
    }
    server.maga_init_params_     = params;
    server.engine_               = std::make_shared<MinimalEngine>(params, cache_manager_);
    server.resource_.cache_store = store;
    server.resource_.workers     = {"decode"};

    grpc::ServerContext          server_context;
    DecodeRpcContext             rpc_context{nullptr};
    kmonitor::MetricsReporterPtr reporter;
    CancellableDecodeLoad        context(
        rpc_context, /*timeout_ms=*/10000, &server_context, reporter, std::make_shared<RpcServerRuntimeMeta>());
    context.peer_addrs = {"127.0.0.1:12345:0"};
    context.setStream(stream_);

    // Start single-P / single-D TCP direct load, wait until every layer is dispatched, then cancel RPC.
    auto load_future = std::async(std::launch::async, [&] { return server.loadCacheForAllRank(context); });
    ASSERT_TRUE(store->waitPending(layers));
    auto fence = observeDestinationFence(store->pending.back());
    context.cancelled.store(true);
    ASSERT_EQ(load_future.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    const auto result = load_future.get();
    EXPECT_EQ(result.error_info.code(), ErrorCode::CANCELLED);
    EXPECT_FALSE(result.destination_writes_quiesced);
    store->load_buffer_requests_.clear();

    stream_->reportError(result.error_info.code(), result.error_info.ToString());
    stream_->releaseResource();
    context.markRpcHandlingCompleted();
    EXPECT_EQ(pool->referencedBlocksNum(DeviceBlockRefType::PD_CONNECTOR), target_blocks);
    for (const auto b : original_blocks) {
        EXPECT_FALSE(pool->hasRequestRef(b));
        EXPECT_TRUE(pool->isAllocated(b));
        EXPECT_FALSE(pool->isFrozen(b));
    }

    // Allocate all remaining free blocks in a second request; none of the cancelled blocks may be reused.
    const int other_blocks       = static_cast<int>(initial_free_blocks_ - target_blocks);
    auto      other_input        = std::make_shared<GenerateInput>();
    other_input->generate_config = std::make_shared<GenerateConfig>();
    other_input->input_ids       = torch::arange(other_blocks * tokens_per_block, torch::kInt32);
    auto other_tokens =
        std::make_shared<CompleteTokenIds>(1, 1, other_blocks * tokens_per_block + tokens_per_block, tokens_per_block);
    other_tokens->init(other_input);
    auto other = std::make_shared<BatchKVCacheResource>();
    other->resetBatchSize(1);
    other->initGroups(cache_manager_->cacheConfig().topologyPtr());
    ASSERT_TRUE(cache_manager_->malloc({other, other_tokens, 2, true, false, false}).success);
    const auto& allocated = other->blocks(0, "default");
    EXPECT_EQ(allocated.size(), static_cast<size_t>(other_blocks));
    for (const auto b : original_blocks) {
        EXPECT_EQ(std::find(allocated.begin(), allocated.end(), b), allocated.end());
    }
    EXPECT_EQ(pool->freeBlocksNum(), 0u);

    // Complete all but the final layer copy; blocks must remain pinned.
    const bool ok = GetParam();
    for (size_t i = 0; i + 1 < store->pending.size(); ++i) {
        store->pending[i].destinationWritesStopped();
        store->pending[i](true, CacheStoreErrorCode::None);
    }
    EXPECT_EQ(pool->referencedBlocksNum(DeviceBlockRefType::PD_CONNECTOR), target_blocks);
    EXPECT_EQ(pool->freeBlocksNum(), 0u);
    EXPECT_FALSE(server.waitForDrain(std::chrono::system_clock::now()));

    // The last RPC result alone cannot prove that its local GPU copy has stopped.
    store->pending.back()(ok, ok ? CacheStoreErrorCode::None : CacheStoreErrorCode::LoadBufferTimeout);
    EXPECT_EQ(pool->referencedBlocksNum(DeviceBlockRefType::PD_CONNECTOR), target_blocks);
    EXPECT_EQ(pool->freeBlocksNum(), 0u);
    if (!ok) {
        ASSERT_EQ(fence.wait_for(std::chrono::seconds(5)), std::future_status::ready);
        ASSERT_NO_THROW(fence.get());
        EXPECT_EQ(pool->referencedBlocksNum(DeviceBlockRefType::PD_CONNECTOR), target_blocks);
    } else {
        EXPECT_EQ(fence.wait_for(std::chrono::seconds(0)), std::future_status::timeout);
    }
    store->pending.back().destinationWritesStopped();
    EXPECT_EQ(pool->referencedBlocksNum(DeviceBlockRefType::PD_CONNECTOR), 0u);
    store->pending.clear();
    for (const auto b : original_blocks) {
        EXPECT_FALSE(pool->isFrozen(b));
        EXPECT_FALSE(pool->isAllocated(b));
    }
    EXPECT_EQ(pool->referencedBlocksNum(DeviceBlockRefType::PD_CONNECTOR), 0u);
    EXPECT_EQ(pool->freeBlocksNum(), target_blocks);
    EXPECT_TRUE(server.waitForDrain(std::chrono::system_clock::now()));
    cache_manager_->free({other, other_tokens, 2});
    EXPECT_EQ(pool->freeBlocksNum(), initial_free_blocks_);
}

INSTANTIATE_TEST_SUITE_P(SinglePTcpDirect,
                         PdSepSinglePTcpDirectCancelTest,
                         ::testing::Bool(),
                         [](const ::testing::TestParamInfo<bool>& info) {
                             return info.param ? "LateCopySucceeded" : "LateCopyFailed";
                         });

INSTANTIATE_TEST_SUITE_P(LateRdmaCallback,
                         PdSepRdmaLateCompletionTest,
                         ::testing::Bool(),
                         [](const ::testing::TestParamInfo<bool>& info) {
                             return info.param ? "Completed" : "Unknown";
                         });

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
class PdSepLoadPublicationTest: public PdSepKVCacheReleaseTest, public ::testing::WithParamInterface<bool> {};

TEST_P(PdSepLoadPublicationTest, ReuseRequiresSuccessfulLoadEvenAfterWritesStop) {
    constexpr int tokens_per_block = 8;
    prepareStreamWithConfig(
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}, makeConfig(), tokens_per_block, RoleType::DECODE);
    stream_->generate_status_->status = StreamState::WAITING;
    ASSERT_TRUE(stream_->initKVBlock().ok());
    const auto blocks = stream_->kvCachePtr()->blocks(0, "default");
    ASSERT_EQ(blocks.size(), 2u);
    const auto pool = cache_manager_->coordinator_manager_->groupBlockPools().front();

    const size_t     layers = cache_manager_->cacheConfig().layer_num;
    auto             store  = std::make_shared<DelayedTcpCacheStore>();
    EngineInitParams params;
    params.model_id                 = 1;
    params.model_config_.num_layers = layers;
    DecodeRpcServer server;
    server.maga_init_params_     = params;
    server.engine_               = std::make_shared<MinimalEngine>(params, cache_manager_);
    server.resource_.cache_store = store;
    server.resource_.workers     = {"decode"};
    grpc::ServerContext          server_context;
    DecodeRpcContext             rpc_context{nullptr};
    kmonitor::MetricsReporterPtr reporter;
    CancellableDecodeLoad        context(
        rpc_context, 10000, &server_context, reporter, std::make_shared<RpcServerRuntimeMeta>());
    context.peer_addrs = {"127.0.0.1:12345:0"};
    context.setStream(stream_);

    const bool succeeded = GetParam();
    auto       load      = std::async(std::launch::async, [&] { return server.loadCacheForAllRank(context); });
    ASSERT_TRUE(store->waitPending(layers));
    for (size_t layer = 0; layer + 1 < layers; ++layer) {
        store->pending[layer].destinationWritesStopped();
    }
    auto independent_pin = pool->pinForRemotePublication(blocks.front());
    ASSERT_TRUE(independent_pin);
    EXPECT_EQ(pool->referencedBlocksNum(DeviceBlockRefType::PD_CONNECTOR), blocks.size());
    EXPECT_EQ(pool->referencedBlocksNum(DeviceBlockRefType::REMOTE_CONNECTOR), 1u);
    store->pending.back().destinationWritesStopped();
    for (size_t layer = 0; layer < layers; ++layer) {
        const bool copied = succeeded || layer + 1 < layers;
        store->pending[layer](copied, copied ? CacheStoreErrorCode::None : CacheStoreErrorCode::LoadErrorUnknown);
    }
    const auto result = load.get();
    ASSERT_EQ(result.error_info.ok(), succeeded);
    ASSERT_TRUE(result.destination_writes_quiesced);
    EXPECT_EQ(pool->referencedBlocksNum(DeviceBlockRefType::PD_CONNECTOR), 0u);
    EXPECT_TRUE(pool->hasRequestRef(blocks.front()));
    EXPECT_EQ(pool->referencedBlocksNum(DeviceBlockRefType::REMOTE_CONNECTOR), 1u);
    context.markRpcHandlingCompleted();

    // Exercise independent FINISHED cleanup even when only the final layer failed.
    // Physical completion and the stream status alone cannot certify the prefix contents.
    ASSERT_TRUE(stream_->finishWithoutGenerate());
    EXPECT_EQ(pool->referencedBlocksNum(BlockTreeRefType::CACHE) > 0, succeeded);

    auto next_input = std::make_shared<GenerateInput>(*stream_->generateInput());
    next_input->request_id += 1;
    ModelConfig model;
    model.attn_config.tokens_per_block = tokens_per_block;
    model.max_seq_len                  = 2048;
    auto reader =
        std::make_shared<NormalGenerateStream>(next_input, model, RuntimeConfig{}, stream_->resourceContext(), nullptr);
    ASSERT_TRUE(reader->initKVBlock().ok());
    EXPECT_EQ(reader->reuseLength(), succeeded ? tokens_per_block : 0);
    const auto reader_blocks = reader->kvCachePtr()->blocks(0, "default");
    ASSERT_FALSE(reader_blocks.empty());
    EXPECT_EQ(reader_blocks.front() == blocks.front(), succeeded);
    EXPECT_EQ(pool->referencedBlocksNum(DeviceBlockRefType::REMOTE_CONNECTOR), 1u);
    EXPECT_TRUE(pool->hasRequestRef(reader_blocks.front()));
    independent_pin.reset();
    EXPECT_EQ(pool->isAllocated(blocks.front()), succeeded);
    EXPECT_TRUE(pool->hasRequestRef(reader_blocks.front()));
    reader->releaseResource();
}

INSTANTIATE_TEST_SUITE_P(PhysicalWritesStopped,
                         PdSepLoadPublicationTest,
                         ::testing::Bool(),
                         [](const ::testing::TestParamInfo<bool>& p) {
                             return p.param ? "LoadSucceeded" : "LoadFailed";
                         });

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
    prepareStreamWithConfig(tokens, config, spb, RoleType::PREFILL);
    allocateAndFinish();

    auto& resource = stream_->streamCacheResource();
    ASSERT_EQ(resource.kvCache().groupNums(), kDsv4PoolNum);
    ASSERT_GT(resource.curBlocksNum(), 0);
    for (const auto& tag : config.groupTags()) {
        ASSERT_EQ(resource.kvCache().blocksNum(0, tag), 4) << "group " << tag;
        const auto&  blocks = resource.kvCache().blocks(0, tag);
        const auto&  policy = config.group(tag).policy;
        const size_t tail   = policy.enable_prefix_reuse ?
                                  std::min(blocks.size(), static_cast<size_t>(config.reuse_cache_tail_blocks)) :
                                  std::min(blocks.size(), static_cast<size_t>(policy.active_tail_blocks));
        for (size_t pos = 0; pos < blocks.size(); ++pos) {
            const bool expected = policy.active_tail_blocks == 0 || pos >= blocks.size() - tail;
            EXPECT_EQ(!isNullBlockIdx(blocks[pos]), expected) << "group " << tag << " pos " << pos;
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
    for (const auto& tag : cache_manager_->cacheConfig().groupTags()) {
        EXPECT_EQ(decode_resource.kvCache().blocksNum(0, tag), 4) << "group " << tag;
    }

    decode_stream->releaseResource();
}

TEST_F(PdSepKVCacheReleaseTest, testCpShardedCacheStoreTransfersRankMappedPhysicalBlocks) {
    constexpr int     cp_size        = 2;
    constexpr int     spb            = 2;
    constexpr int     logical_blocks = 6;
    constexpr int64_t request_id     = 9016;
    constexpr size_t  model_id       = 76;

    auto config = test::makeSimpleMhaCacheConfig(/*layer_num=*/1, /*block_num=*/16, spb, DataType::TYPE_BF16);
    config.use_opaque_kv_cache_store = true;

    auto makeResource = [](const std::shared_ptr<KVCacheManager>& manager) {
        auto resource = std::make_shared<BatchKVCacheResource>();
        resource->resetBatchSize(1);
        resource->initGroups(manager->cacheConfig().topologyPtr());
        return resource;
    };
    auto makeCompleteTokens = [spb](int seq_len) {
        auto input              = std::make_shared<GenerateInput>();
        input->input_ids        = torch::arange(seq_len, torch::kInt32);
        input->generate_config  = std::make_shared<GenerateConfig>();
        auto complete_token_ids = std::make_shared<CompleteTokenIds>(1, 1, seq_len + spb, spb);
        complete_token_ids->init(input);
        complete_token_ids->setSeqLength(seq_len);
        return complete_token_ids;
    };

    std::vector<std::shared_ptr<KVCacheManager>>       prefill_managers;
    std::vector<std::shared_ptr<BatchKVCacheResource>> prefill_resources;
    for (int cp_rank = 0; cp_rank < cp_size; ++cp_rank) {
        // Each producer owns the rank-local compact table. Passing cp_rank/cp_size
        // to runtimeWriteCacheStore below projects it into the global key space.
        auto manager = std::make_shared<KVCacheManager>(config, /*warmup=*/false, nullptr);
        ASSERT_TRUE(manager->init());
        auto resource = makeResource(manager);
        ASSERT_TRUE(
            manager
                ->malloc({resource, makeCompleteTokens(logical_blocks / cp_size * spb), request_id, true, false, false})
                .success);
        ASSERT_EQ(resource->blocksNum(0, "default"), logical_blocks / cp_size);
        prefill_managers.push_back(std::move(manager));
        prefill_resources.push_back(std::move(resource));
    }

    auto decode_manager = std::make_shared<KVCacheManager>(config, /*warmup=*/false, nullptr);
    ASSERT_TRUE(decode_manager->init());
    auto decode_resource = makeResource(decode_manager);
    ASSERT_TRUE(
        decode_manager
            ->malloc({decode_resource, makeCompleteTokens(logical_blocks * spb), request_id, true, false, false})
            .success);
    ASSERT_EQ(decode_resource->blocksNum(0, "default"), logical_blocks);

    std::vector<CacheKeyType> cache_keys;
    for (int logical_pos = 0; logical_pos < logical_blocks; ++logical_pos) {
        cache_keys.push_back(12000 + logical_pos);
    }

    auto cache_store = std::make_shared<MemoryBackedCacheStore>();
    for (int cp_rank = 0; cp_rank < cp_size; ++cp_rank) {
        const auto& manager  = prefill_managers[cp_rank];
        const auto& resource = prefill_resources[cp_rank];
        const auto& blocks   = resource->blocks(0, "default");
        auto        layout   = manager->getMainModelGroupedCacheLayerLayout();
        auto        kv_base  = layout.at("default", 0).kv_addr;

        for (size_t local_pos = 0; local_pos < blocks.size(); ++local_pos) {
            const int logical_pos = cp_rank + static_cast<int>(local_pos) * cp_size;
            fillDsv4RegionBytes(
                manager, blocks[local_pos], /*layer_id=*/0, "default", static_cast<uint8_t>(40 + logical_pos));
        }

        auto                    inputs = makeDsv4WriteInputs(request_id,
                                          logical_blocks * spb,
                                          /*prefix_length=*/0,
                                          blockIdsTensor(resource, "default"),
                                          cache_keys);
        torch_ext::LayerKVCache layer_cache;
        layer_cache.kv_cache_base      = kv_base;
        layer_cache.seq_size_per_block = spb;
        layer_cache.layer_id           = 0;
        layer_cache.tag                = "default";
        runtimeWriteCacheStore(inputs,
                               layer_cache,
                               manager->cacheConfig(),
                               cache_store,
                               model_id,
                               cp_rank,
                               cp_size,
                               /*pre_created_event=*/nullptr);

        const auto& stored_request = cache_store->store_buffer_requests_.at(static_cast<size_t>(cp_rank));
        ASSERT_EQ(stored_request->getBlocksCount(), static_cast<size_t>(logical_blocks / cp_size));
        for (size_t local_pos = 0; local_pos < blocks.size(); ++local_pos) {
            const int  logical_pos = cp_rank + static_cast<int>(local_pos) * cp_size;
            const auto key =
                "kv_" + makeCacheKey(model_id, std::to_string(cache_keys[logical_pos]), /*layer_id=*/0, "default");
            const auto block = stored_request->getBlock(key);
            ASSERT_NE(block, nullptr) << "cp_rank=" << cp_rank << " logical_pos=" << logical_pos;
            const auto transfer_bytes = manager->cacheConfig().topology().groups()[0].kvBlockStrideBytes();
            const auto expected_address =
                static_cast<uint8_t*>(kv_base.data_ptr()) + static_cast<size_t>(blocks[local_pos]) * transfer_bytes;
            EXPECT_EQ(block->addr.get(), expected_address);
            EXPECT_EQ(block->len, transfer_bytes);
            const auto stored = cache_store->stored_blocks_.find(key);
            ASSERT_NE(stored, cache_store->stored_blocks_.end());
            EXPECT_EQ(stored->second.size(), transfer_bytes);
            EXPECT_TRUE(std::all_of(stored->second.begin(), stored->second.end(), [logical_pos](uint8_t byte) {
                return byte == 40 + logical_pos;
            }));
        }
    }
    ASSERT_EQ(cache_store->stored_blocks_.size(), static_cast<size_t>(logical_blocks));

    for (int logical_pos = 0; logical_pos < logical_blocks; ++logical_pos) {
        const auto decode_block = decode_resource->blocks(0, "default")[static_cast<size_t>(logical_pos)];
        fillDsv4RegionBytes(decode_manager, decode_block, /*layer_id=*/0, "default", 0xEE);
    }

    EngineInitParams params;
    params.model_id                 = model_id;
    params.model_config_.num_layers = 1;
    params.parallelism_config       = ParallelismConfig();

    DecodeRpcServer server;
    server.engine_                   = std::make_shared<MinimalEngine>(params, decode_manager);
    server.maga_init_params_         = params;
    server.propose_maga_init_params_ = nullptr;
    server.resource_.cache_store     = cache_store;

    std::vector<std::string>            peer_addrs = {"127.0.0.1:12345:12346", "127.0.0.1:12347:12348"};
    grpc::ServerContext                 server_context;
    DecodeRpcServer::LoadKVCacheContext load_context(request_id,
                                                     "cp-sharded-cache-store-pd",
                                                     peer_addrs,
                                                     cache_keys,
                                                     decode_resource->cacheResource(0).groupBlockIds(),
                                                     /*reuse_block_size=*/0,
                                                     /*timeout_ms=*/5000,
                                                     /*partition_count=*/1,
                                                     /*partition_id=*/0,
                                                     &server_context,
                                                     /*prefill_cp_size=*/cp_size);
    const auto                          result = server.loadCache(load_context);
    ASSERT_TRUE(result.ok()) << result.error_info.ToString();
    ASSERT_EQ(cache_store->load_buffer_requests_.size(), static_cast<size_t>(cp_size));

    for (int cp_rank = 0; cp_rank < cp_size; ++cp_rank) {
        const auto& loaded_request = cache_store->load_buffer_requests_[static_cast<size_t>(cp_rank)];
        ASSERT_EQ(loaded_request->getBlocksCount(), static_cast<size_t>(logical_blocks / cp_size));
        for (int logical_pos = cp_rank; logical_pos < logical_blocks; logical_pos += cp_size) {
            const auto key =
                "kv_" + makeCacheKey(model_id, std::to_string(cache_keys[logical_pos]), /*layer_id=*/0, "default");
            const auto block = loaded_request->getBlock(key);
            ASSERT_NE(block, nullptr) << "cp_rank=" << cp_rank << " logical_pos=" << logical_pos;
            const auto decode_block = decode_resource->blocks(0, "default")[static_cast<size_t>(logical_pos)];
            const auto expected     = decode_manager->convertIndexToBuffer(0, "default", decode_block);
            ASSERT_EQ(expected.size(), 1u);
            EXPECT_EQ(block->addr.get(), expected[0].addr);
            EXPECT_EQ(block->len, expected[0].size_bytes);
            expectDsv4RegionBytes(
                decode_manager, decode_block, /*layer_id=*/0, "default", static_cast<uint8_t>(40 + logical_pos));
        }
    }
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
        for (const auto& tag : cache_config.topology().layer(layer_id).group_tags) {
            ++expected_requests;
            expected_blocks += dsv4TransferPositions(cache_config, tag, block_num, /*reuse_block_size=*/0).size();
        }
    }

    for (int layer_id = 0; layer_id < 4; ++layer_id) {
        for (const auto& tag : cache_config.topology().layer(layer_id).group_tags) {
            auto positions = dsv4TransferPositions(cache_config, tag, block_num, /*reuse_block_size=*/0);
            for (auto block_pos : positions) {
                auto prefill_block_id = prefill_resource->blocks(0, tag)[block_pos];
                auto decode_block_id  = decode_resource->blocks(0, tag)[block_pos];
                ASSERT_FALSE(isNullBlockIdx(prefill_block_id)) << "prefill tag=" << tag << " pos=" << block_pos;
                ASSERT_FALSE(isNullBlockIdx(decode_block_id)) << "decode tag=" << tag << " pos=" << block_pos;
                fillDsv4RegionBytes(prefill_manager,
                                    prefill_block_id,
                                    layer_id,
                                    tag,
                                    dsv4PdPattern(cache_config, layer_id, tag, block_pos));
                fillDsv4RegionBytes(decode_manager, decode_block_id, layer_id, tag, 0xEE);
            }
        }
    }
    runtimeSyncAndCheck();

    auto cache_store = std::make_shared<MemoryBackedCacheStore>();
    auto layout      = prefill_manager->getMainModelGroupedCacheLayerLayout();
    for (int layer_id = 0; layer_id < 4; ++layer_id) {
        for (const auto& tag : cache_config.topology().layer(layer_id).group_tags) {
            ASSERT_TRUE(layout.at(tag, static_cast<size_t>(layer_id)).kv_addr.defined())
                << "layer=" << layer_id << " tag=" << tag;

            auto inputs = makeDsv4WriteInputs(request_id,
                                              /*input_length=*/block_num * spb,
                                              /*prefix_length=*/0,
                                              blockIdsTensor(prefill_resource, tag),
                                              cache_keys);

            torch_ext::LayerKVCache layer_cache;
            layer_cache.kv_cache_base      = layout.at(tag, static_cast<size_t>(layer_id)).kv_addr;
            layer_cache.seq_size_per_block = static_cast<int>(cache_config.group(tag).seqSizePerBlock());
            layer_cache.layer_id           = layer_id;
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
                                                     decode_resource->cacheResource(0).groupBlockIds(),
                                                     /*reuse_block_size=*/0,
                                                     /*timeout_ms=*/5000,
                                                     /*partition_count=*/1,
                                                     /*partition_id=*/0,
                                                     &server_context);
    const auto                          result = server.loadCache(load_context);
    ASSERT_TRUE(result.ok()) << result.error_info.ToString();

    EXPECT_EQ(cache_store->load_buffer_requests_.size(), expected_requests);
    EXPECT_EQ(cache_store->load_request_keys_.size(), expected_requests);
    for (int layer_id = 0; layer_id < 4; ++layer_id) {
        for (const auto& tag : cache_config.topology().layer(layer_id).group_tags) {
            auto positions = dsv4TransferPositions(cache_config, tag, block_num, /*reuse_block_size=*/0);
            for (auto block_pos : positions) {
                auto decode_block_id = decode_resource->blocks(0, tag)[block_pos];
                ASSERT_FALSE(isNullBlockIdx(decode_block_id));
                expectDsv4RegionBytes(decode_manager,
                                      decode_block_id,
                                      layer_id,
                                      tag,
                                      dsv4PdPattern(cache_config, layer_id, tag, block_pos));
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
        for (const auto& tag : cache_config.topology().layer(layer_id).group_tags) {
            auto positions = dsv4TransferPositions(cache_config, tag, block_num, /*reuse_block_size=*/0);
            for (auto block_pos : positions) {
                auto prefill_block_id = prefill_resource->blocks(0, tag)[block_pos];
                auto decode_block_id  = decode_resource->blocks(0, tag)[block_pos];
                ASSERT_FALSE(isNullBlockIdx(prefill_block_id)) << "prefill tag=" << tag << " pos=" << block_pos;
                ASSERT_FALSE(isNullBlockIdx(decode_block_id)) << "decode tag=" << tag << " pos=" << block_pos;
                fillDsv4RegionBytes(prefill_manager,
                                    prefill_block_id,
                                    layer_id,
                                    tag,
                                    dsv4PdPattern(cache_config, layer_id, tag, block_pos));
                fillDsv4RegionBytes(decode_manager, decode_block_id, layer_id, tag, 0xEE);
            }
        }
    }
    runtimeSyncAndCheck();

    auto cache_store = std::make_shared<MemoryBackedCacheStore>();
    auto layout      = prefill_manager->getMainModelGroupedCacheLayerLayout();
    for (int layer_id = 0; layer_id < 4; ++layer_id) {
        for (const auto& tag : cache_config.topology().layer(layer_id).group_tags) {
            ASSERT_TRUE(layout.at(tag, static_cast<size_t>(layer_id)).kv_addr.defined())
                << "layer=" << layer_id << " tag=" << tag;

            auto inputs = makeDsv4WriteInputs(request_id,
                                              /*input_length=*/block_num * spb,
                                              /*prefix_length=*/0,
                                              blockIdsTensor(prefill_resource, tag),
                                              cache_keys);

            // Paged groups expose the layer view at kernel-block granularity;
            // cache_store must still address whole physical blocks.
            torch_ext::LayerKVCache layer_cache;
            layer_cache.kv_cache_base = layout.at(tag, static_cast<size_t>(layer_id)).kv_addr;
            layer_cache.seq_size_per_block =
                cache_config.group(tag).policy.group_type == CacheGroupType::FULL ? kernel_spb : spb;
            layer_cache.layer_id = layer_id;
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

    const auto first_csa_key = "kv_" + makeCacheKey(model_id, std::to_string(cache_keys[0]), /*layer_id=*/2, "csa_kv");
    ASSERT_NE(cache_store->stored_blocks_.find(first_csa_key), cache_store->stored_blocks_.end());
    EXPECT_EQ(cache_store->stored_blocks_[first_csa_key].size(), cache_config.group("csa_kv").kvBlockStrideBytes());

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
                                                     decode_resource->cacheResource(0).groupBlockIds(),
                                                     /*reuse_block_size=*/0,
                                                     /*timeout_ms=*/5000,
                                                     /*partition_count=*/1,
                                                     /*partition_id=*/0,
                                                     &server_context);
    const auto                          result = server.loadCache(load_context);
    ASSERT_TRUE(result.ok()) << result.error_info.ToString();

    for (int layer_id = 0; layer_id < 4; ++layer_id) {
        for (const auto& tag : cache_config.topology().layer(layer_id).group_tags) {
            auto positions = dsv4TransferPositions(cache_config, tag, block_num, /*reuse_block_size=*/0);
            for (auto block_pos : positions) {
                auto decode_block_id = decode_resource->blocks(0, tag)[block_pos];
                ASSERT_FALSE(isNullBlockIdx(decode_block_id));
                expectDsv4RegionBytes(decode_manager,
                                      decode_block_id,
                                      layer_id,
                                      tag,
                                      dsv4PdPattern(cache_config, layer_id, tag, block_pos));
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
        for (const auto& tag : cache_config.topology().layer(layer_id).group_tags) {
            ++expected_requests;
            expected_blocks += dsv4TransferPositions(cache_config, tag, block_num, reuse_num).size();
        }
    }

    for (int layer_id = 0; layer_id < 4; ++layer_id) {
        for (const auto& tag : cache_config.topology().layer(layer_id).group_tags) {
            auto positions = dsv4TransferPositions(cache_config, tag, block_num, reuse_num);
            for (auto block_pos : positions) {
                auto prefill_block_id = prefill_resource->blocks(0, tag)[block_pos];
                auto decode_block_id  = decode_resource->blocks(0, tag)[block_pos];
                ASSERT_FALSE(isNullBlockIdx(prefill_block_id)) << "prefill tag=" << tag << " pos=" << block_pos;
                ASSERT_FALSE(isNullBlockIdx(decode_block_id)) << "decode tag=" << tag << " pos=" << block_pos;
                fillDsv4RegionBytes(prefill_manager,
                                    prefill_block_id,
                                    layer_id,
                                    tag,
                                    dsv4PdPattern(cache_config, layer_id, tag, block_pos));
                fillDsv4RegionBytes(decode_manager, decode_block_id, layer_id, tag, 0xEE);
            }
        }
    }
    runtimeSyncAndCheck();

    auto cache_store = std::make_shared<MemoryBackedCacheStore>();
    auto layout      = prefill_manager->getMainModelGroupedCacheLayerLayout();
    for (int layer_id = 0; layer_id < 4; ++layer_id) {
        for (const auto& tag : cache_config.topology().layer(layer_id).group_tags) {
            ASSERT_TRUE(layout.at(tag, static_cast<size_t>(layer_id)).kv_addr.defined())
                << "layer=" << layer_id << " tag=" << tag;

            auto inputs = makeDsv4WriteInputs(request_id,
                                              /*input_length=*/(block_num - reuse_num) * spb,
                                              /*prefix_length=*/reuse_num * spb,
                                              blockIdsTensor(prefill_resource, tag),
                                              cache_keys);

            torch_ext::LayerKVCache layer_cache;
            layer_cache.kv_cache_base      = layout.at(tag, static_cast<size_t>(layer_id)).kv_addr;
            layer_cache.seq_size_per_block = static_cast<int>(cache_config.group(tag).seqSizePerBlock());
            layer_cache.layer_id           = layer_id;
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
                                                     decode_resource->cacheResource(0).groupBlockIds(),
                                                     reuse_num,
                                                     /*timeout_ms=*/5000,
                                                     /*partition_count=*/1,
                                                     /*partition_id=*/0,
                                                     &server_context);
    const auto                          result = server.loadCache(load_context);
    ASSERT_TRUE(result.ok()) << result.error_info.ToString();

    EXPECT_EQ(cache_store->load_buffer_requests_.size(), expected_requests);
    EXPECT_EQ(cache_store->load_request_keys_.size(), expected_requests);
    for (int layer_id = 0; layer_id < 4; ++layer_id) {
        for (const auto& tag : cache_config.topology().layer(layer_id).group_tags) {
            auto positions = dsv4TransferPositions(cache_config, tag, block_num, reuse_num);
            for (auto block_pos : positions) {
                auto decode_block_id = decode_resource->blocks(0, tag)[block_pos];
                ASSERT_FALSE(isNullBlockIdx(decode_block_id));
                expectDsv4RegionBytes(decode_manager,
                                      decode_block_id,
                                      layer_id,
                                      tag,
                                      dsv4PdPattern(cache_config, layer_id, tag, block_pos));
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
    auto layout = manager->getMainModelGroupedCacheLayerLayout();
    for (int layer_id = 0; layer_id < 3; ++layer_id) {
        auto buf = layout.at(static_cast<size_t>(layer_id)).kv_addr;
        ASSERT_TRUE(buf.defined());
        for (int b = 0; b < block_num; ++b) {
            auto bid       = resource->blocks(0, "default")[b];
            auto kv_stride = config.topology().groups()[0].kvBlockStrideBytes();
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
    auto block_ids   = torch::from_blob(const_cast<int*>(resource->blocks(0, "default").data()),
                                      {1, (int64_t)resource->blocks(0, "default").size()},
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

TEST_F(PdSepKVCacheReleaseTest, testWriteCacheStorePreservesCpSwaOpaqueByteSlices) {
    struct TestCase {
        int    cp_size;
        int    gen_num_per_cycle;
        size_t entries;
        size_t slice_bytes;
    };
    const TestCase cases[] = {
        {8, 3, 136, 9936},
        {8, 5, 136, 9936},
        {8, 7, 136, 9936},
        {16, 3, 144, 5256},
        {16, 5, 144, 5256},
        {16, 7, 144, 5256},
        {8, 9, 144, 10512},
    };
    ModelConfig model_config;
    model_config.num_layers                          = 1;
    model_config.attn_config.sliding_window          = 128;
    model_config.attn_config.size_per_head           = 512;
    model_config.attn_config.indexer_head_dim        = 128;
    model_config.attn_config.kv_cache_dtype          = KvCacheDataType::FP8;
    model_config.attn_config.layer_compress_ratios   = {0};
    model_config.attn_config.tokens_per_block        = 128;
    model_config.attn_config.kernel_tokens_per_block = 128;
    test::setDsv4KvCacheSpecs(model_config, {0});
    KVCacheConfig kv_config;
    kv_config.seq_size_per_block        = 128;
    kv_config.kernel_seq_size_per_block = 128;

    for (const auto& test_case : cases) {
        SCOPED_TRACE(::testing::Message() << "cp=" << test_case.cp_size << " gen=" << test_case.gen_num_per_cycle);
        ParallelismConfig parallelism;
        parallelism.role_type                          = RoleType::PREFILL;
        parallelism.tp_size                            = test_case.cp_size;
        parallelism.prefill_cp_config.kv_cache_sharded = true;
        const auto prefill_config =
            CacheConfigCreator::createWarmupConfig(model_config, parallelism, kv_config, test_case.gen_num_per_cycle);
        parallelism.role_type                         = RoleType::DECODE;
        parallelism.tp_size                           = 1;
        parallelism.prefill_cp_config.method          = CPRotateMethod::PREFILL_CP;
        parallelism.prefill_cp_config.prefill_cp_size = test_case.cp_size;
        const auto decode_config =
            CacheConfigCreator::createWarmupConfig(model_config, parallelism, kv_config, test_case.gen_num_per_cycle);
        const auto&  swa         = prefill_config.group("swa_kv");
        const size_t slice_bytes = swa.kvBlockStrideBytes();
        const size_t full_bytes  = decode_config.group("swa_kv").kvBlockStrideBytes();
        ASSERT_EQ(swa.kvScaleStrideBytes(), 0u);
        ASSERT_EQ(slice_bytes, test_case.slice_bytes);
        ASSERT_EQ(full_bytes, slice_bytes * test_case.cp_size);

        // Python writes a full striped block, then stores each rank's contiguous
        // byte slice. A slice is not a miniature block with its own scale tail.
        const size_t data_end  = test_case.entries * 576;
        const size_t scale_end = test_case.entries * 584;
        ASSERT_LE(scale_end, full_bytes);
        for (size_t model_id : {0u, 1u}) {  // Main model and MTP key namespaces.
            SCOPED_TRACE(::testing::Message() << "model_id=" << model_id);
            std::vector<uint8_t> expected(full_bytes);
            for (size_t byte = 0; byte < full_bytes; ++byte) {
                if (byte < data_end) {
                    expected[byte] = static_cast<uint8_t>((byte + model_id * 31) % 127);
                } else if (byte < scale_end) {
                    expected[byte] = static_cast<uint8_t>(128 + byte % 64);
                } else {
                    expected[byte] = 0xe7;
                }
            }
            auto reconstructed = std::make_shared<std::vector<uint8_t>>(full_bytes, 0xff);
            for (int cp_rank = 0; cp_rank < test_case.cp_size; ++cp_rank) {
                SCOPED_TRACE(::testing::Message() << "cp_rank=" << cp_rank);
                const size_t offset    = static_cast<size_t>(cp_rank) * slice_bytes;
                auto         kv_buffer = torch::zeros({2, static_cast<int64_t>(slice_bytes)}, torch::kUInt8);
                std::memcpy(kv_buffer.data_ptr<uint8_t>() + slice_bytes, expected.data() + offset, slice_bytes);
                std::vector<CacheKeyType> cache_keys;
                for (int key_index = 0; key_index < test_case.cp_size; ++key_index) {
                    cache_keys.push_back(10000 + key_index);
                }
                auto                    inputs = makeDsv4WriteInputs(4245,
                                                  static_cast<int>(swa.seqSizePerBlock()),
                                                  /*prefix_length=*/0,
                                                  torch::tensor({{1}}, torch::kInt32),
                                                  cache_keys);
                torch_ext::LayerKVCache layer_cache;
                layer_cache.kv_cache_base      = kv_buffer;
                layer_cache.seq_size_per_block = static_cast<int>(swa.seqSizePerBlock());
                layer_cache.layer_id           = 0;
                layer_cache.tag                = "swa_kv";
                auto cache_store               = std::make_shared<MemoryBackedCacheStore>();
                runtimeWriteCacheStore(inputs,
                                       layer_cache,
                                       prefill_config,
                                       cache_store,
                                       model_id,
                                       cp_rank,
                                       test_case.cp_size,
                                       /*pre_created_event=*/nullptr);

                const auto cache_key = makeCacheKey(model_id, std::to_string(cache_keys.back()), 0, "swa_kv");
                const auto key       = "kv_" + cache_key;
                ASSERT_EQ(cache_store->store_buffer_requests_.size(), 1u);
                auto published = cache_store->store_buffer_requests_.front();
                ASSERT_EQ(published->getBlocksCount(), 1u);
                ASSERT_NE(published->getBlock(key), nullptr);
                EXPECT_EQ(published->getBlock("kv_scale_" + cache_key), nullptr);
                ASSERT_EQ(published->getBlock(key)->len, slice_bytes);
                EXPECT_FALSE(published->getBlock(key)->gpu_mem);

                // Decode requests the complete opaque slice at its peer offset.
                auto                  request = std::make_shared<RequestBlockBuffer>("4245");
                std::shared_ptr<void> destination(reconstructed, reconstructed->data() + offset);
                request->addBlock(key, destination, slice_bytes, false, true);
                bool loaded = false;
                cache_store->load(
                    request,
                    [&](bool ok, CacheStoreErrorCode error) {
                        loaded = ok;
                        EXPECT_EQ(error, CacheStoreErrorCode::None);
                    },
                    "127.0.0.1",
                    0,
                    0,
                    1000,
                    /*partition_count=*/1,
                    /*partition_id=*/0);
                ASSERT_TRUE(loaded);
            }
            EXPECT_EQ(*reconstructed, expected);  // Includes global scales and CP8's 64 padding bytes.
        }
    }
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

TEST_F(PdSepKVCacheReleaseTest, RdmaWorkerRejectsUnheldOrMaskedTargetBeforeLoading) {
    prepareStreamWithConfig({1, 2, 3}, makeConfig(), 8, RoleType::DECODE);
    auto             store = std::make_shared<DelayedRdmaCacheStore>();
    EngineInitParams params;
    params.model_config_.num_layers = cache_manager_->cacheConfig().layer_num;
    DecodeRpcServer server;
    server.maga_init_params_     = params;
    server.engine_               = std::make_shared<MinimalEngine>(params, cache_manager_);
    server.resource_.cache_store = store;

    BroadcastLoadRequestPB request;
    request.set_dp_rank(params.parallelism_config.dp_rank);
    request.add_cache_keys(42);
    auto* row = request.add_tagged_group_block_ids();
    row->set_tag("default");
    row->add_block_ids(NULL_BLOCK_IDX);
    row->add_masked_positions(0);

    grpc::ServerContext     context;
    BroadcastLoadResponsePB legacy_reply;
    EXPECT_FALSE(server.RemoteLoad(&context, &request, &legacy_reply).ok());
    EXPECT_FALSE(store->pending);
    for (bool held : {false, true}) {
        PreparePDWriteLeaseRequestPB prepare;
        prepare.set_dp_rank(request.dp_rank());
        prepare.set_timeout_ms(10000);
        ASSERT_TRUE(server.PreparePDWriteLease(&context, &prepare, request.mutable_write_lease()).ok());
        request.set_destination_blocks_held(held);
        BroadcastLoadResponsePB response;
        ASSERT_TRUE(server.RemoteLoad(&context, &request, &response).ok());
        EXPECT_NE(response.error_info().error_code(), ErrorCodePB::NONE_ERROR);
        EXPECT_TRUE(response.destination_writes_quiesced());
        EXPECT_NE(response.error_info().error_message().find(held ? "plan" : "hold"), std::string::npos);
        EXPECT_FALSE(store->pending);
        EXPECT_TRUE(store->load_buffer_requests_.empty());
    }
    store->memory_util_ = std::make_shared<DummyMemoryUtil>(true, false);
    PreparePDWriteLeaseRequestPB unsupported;
    unsupported.set_dp_rank(request.dp_rank());
    unsupported.set_timeout_ms(10000);
    PDWriteLeasePB refused;
    EXPECT_EQ(server.PreparePDWriteLease(&context, &unsupported, &refused).error_code(),
              grpc::StatusCode::FAILED_PRECONDITION);
    EXPECT_EQ(refused.token(), 0u);
    EXPECT_TRUE(store->load_buffer_requests_.empty());
}

TEST_F(PdSepKVCacheReleaseTest, WorkerSealArmsInflightRecoveryAndRequiresTransportStop) {
    prepareStreamWithConfig({1, 2, 3}, makeConfig(), 8, RoleType::DECODE);
    ASSERT_TRUE(stream_->initKVBlock().ok());
    CacheKeysType keys;
    GroupBlockIds rows;
    int64_t       reuse = 0;
    auto hold = stream_->holdPDLoadDestination(keys, rows, reuse, [](const GroupBlockIds& rows, size_t, int64_t) {
        return PDLoadDestinationBlocks{{"default", rows.blocks("default")}};
    });
    ASSERT_TRUE(hold);
    ASSERT_EQ(rows.blocks("default").size(), 1u);
    auto             store = std::make_shared<DelayedRdmaCacheStore>();
    EngineInitParams params;
    params.model_config_.num_layers = cache_manager_->cacheConfig().layer_num;
    DecodeRpcServer server;
    server.pd_write_recovery_delay_ = std::chrono::milliseconds(40);
    server.maga_init_params_        = params;
    server.engine_                  = std::make_shared<MinimalEngine>(params, cache_manager_);
    server.resource_.cache_store    = store;
    grpc::ServerContext          context;
    PreparePDWriteLeaseRequestPB prepare;
    prepare.set_request_key("remote-worker-stop");
    prepare.set_dp_rank(params.parallelism_config.dp_rank);
    prepare.set_timeout_ms(10000);
    BroadcastLoadRequestPB request;
    ASSERT_TRUE(server.PreparePDWriteLease(&context, &prepare, request.mutable_write_lease()).ok());
    request.set_request_key(prepare.request_key());
    request.set_dp_rank(prepare.dp_rank());
    request.set_partition_count(1);
    request.set_timeout_ms(10000);
    request.set_destination_blocks_held(true);
    request.add_peer_addrs("127.0.0.1:12345:12346");
    request.add_cache_keys(42);
    auto* row = request.add_tagged_group_block_ids();
    row->set_tag("default");
    row->add_block_ids(rows.blocks("default").front());
    BroadcastLoadResponsePB response;
    auto                    submitted = store->submitted.get_future();
    auto outcome = std::async(std::launch::async, [&] { return server.RemoteLoad(&context, &request, &response); });
    ASSERT_EQ(submitted.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    ASSERT_TRUE(store->pending);
    SealPDWriteLeaseResponsePB sealed;
    ASSERT_TRUE(server.SealPDWriteLease(&context, &request.write_lease(), &sealed).ok());
    EXPECT_FALSE(sealed.stopped());
    auto fence           = std::make_shared<std::promise<void>>();
    auto fence_requested = fence->get_future();
    store->pending.setDestinationFence([fence] { fence->set_value(); });
    ASSERT_EQ(fence_requested.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    EXPECT_EQ(outcome.wait_for(std::chrono::milliseconds(0)), std::future_status::timeout);
    ASSERT_TRUE(server.SealPDWriteLease(&context, &request.write_lease(), &sealed).ok());
    EXPECT_FALSE(sealed.stopped());
    auto stop = store->pending.destinationStopNotifier();
    ASSERT_TRUE(stop);
    store->pending(false, CacheStoreErrorCode::LoadRdmaWriteFailed);
    ASSERT_EQ(outcome.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    ASSERT_TRUE(outcome.get().ok());
    EXPECT_EQ(response.error_info().error_code(), ErrorCodePB::CACHE_STORE_LOAD_RDMA_WRITE_FAILED);
    EXPECT_FALSE(response.destination_writes_quiesced());
    store->pending = {};
    ASSERT_TRUE(server.SealPDWriteLease(&context, &request.write_lease(), &sealed).ok());
    EXPECT_FALSE(sealed.stopped());
    stop();
    ASSERT_TRUE(server.SealPDWriteLease(&context, &request.write_lease(), &sealed).ok());
    EXPECT_TRUE(sealed.stopped());
    EXPECT_TRUE(samePDWriteLease(request.write_lease(), sealed.lease()));
}

namespace {

class ControlledLoadWorker final: public RpcService::Service {
public:
    grpc::Status PreparePDWriteLease(grpc::ServerContext*,
                                     const PreparePDWriteLeaseRequestPB* request,
                                     PDWriteLeasePB*                     response) override {
        if (block_prepare) {
            prepare_entered.set_value();
            if (release_prepare.get_future().wait_for(std::chrono::seconds(10)) != std::future_status::ready) {
                return grpc::Status(grpc::StatusCode::DEADLINE_EXCEEDED, "test prepare timed out");
            }
        }
        if (prepare_error != grpc::StatusCode::OK)
            return grpc::Status(prepare_error, "worker declined prepare");
        *response = leases->prepare(*request);
        return grpc::Status::OK;
    }
    grpc::Status SealPDWriteLease(grpc::ServerContext*,
                                  const PDWriteLeasePB*       request,
                                  SealPDWriteLeaseResponsePB* response) override {
        bool stopped = false;
        if (!leases->seal(*request, stopped))
            return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION, "epoch");
        if (seal_calls.fetch_add(1) == 0) {
            first_seal.set_value();
            return grpc::Status(grpc::StatusCode::UNAVAILABLE, "lost first seal response");
        }
        *response->mutable_lease() = *request;
        response->set_stopped(stopped);
        return grpc::Status::OK;
    }
    grpc::Status RemoteLoad(grpc::ServerContext*,
                            const BroadcastLoadRequestPB* request,
                            BroadcastLoadResponsePB*      response) override {
        ++load_calls;
        auto writer = leases->claim(*request);
        if (!writer)
            return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION, "not prepared");
        EXPECT_TRUE(request->destination_blocks_held());
        entered.set_value();
        if (reply.wait_for(std::chrono::seconds(10)) != std::future_status::ready) {
            return grpc::Status(grpc::StatusCode::DEADLINE_EXCEEDED, "test worker timed out");
        }
        const bool acknowledged = reply.get();
        if (acknowledged) {
            writer.reset();
            *response->mutable_stopped_lease() = request->write_lease();
            response->set_destination_writes_quiesced(true);
        } else {
            // Keep the independent physical operation live after losing the RPC.
            std::lock_guard<std::mutex> lock(mu);
            physical_writer = std::move(writer);
        }
        reply_prepared.set_value();
        return acknowledged ? grpc::Status::OK : grpc::Status(grpc::StatusCode::UNAVAILABLE, "missing worker receipt");
    }
    void completePhysicalWrite() {
        std::lock_guard<std::mutex> lock(mu);
        physical_writer.reset();
    }

    grpc::StatusCode   prepare_error = grpc::StatusCode::OK;
    bool               block_prepare = false;
    std::promise<void> prepare_entered;
    std::promise<void> release_prepare;
    std::atomic<int>   load_calls{0};
    std::atomic<int>   seal_calls{0};
    std::promise<void> first_seal;
    std::promise<void> entered;
    std::promise<void> reply_prepared;
    std::promise<bool> allow_reply;
    std::future<bool>  reply{allow_reply.get_future()};

private:
    std::mutex                            mu;
    std::shared_ptr<void>                 physical_writer;
    std::shared_ptr<PDWriteLeaseRegistry> leases = std::make_shared<PDWriteLeaseRegistry>();
};

}  // namespace

enum class FanoutOutcome {
    ACKNOWLEDGED,
    LOST_RESPONSE,
    LEGACY_WORKER,
    WORKER_CANCELLED,
    CANCEL_DURING_PREPARE
};
class PdSepRdmaFanoutTest: public PdSepKVCacheReleaseTest, public ::testing::WithParamInterface<FanoutOutcome> {};

TEST_P(PdSepRdmaFanoutTest, CancelRetainsDecodeTargetUntilLastWorkerReceipt) {
    using namespace std::chrono_literals;
    prepareStreamWithConfig({1, 2, 3}, makeConfig(), 8, RoleType::DECODE);
    ASSERT_TRUE(stream_->initKVBlock().ok());
    const auto block = stream_->kvCachePtr()->blocks(0, "default").front();
    const auto pool  = cache_manager_->coordinator_manager_->groupBlockPools().front();

    EngineInitParams params;
    params.model_config_.num_layers                      = cache_manager_->cacheConfig().layer_num;
    params.pd_sep_config.load_cache_timeout_ms           = 10000;
    params.pd_sep_config.max_rpc_timeout_ms              = 10000;
    params.pd_sep_config.decode_polling_kv_cache_step_ms = 2;
    DecodeRpcServer decode;
    decode.maga_init_params_ = params;
    decode.engine_           = std::make_shared<MinimalEngine>(params, cache_manager_);
    // This rank uses TCP. The remote workers can still publish RDMA destinations.
    decode.resource_.cache_store = std::make_shared<MemoryBackedCacheStore>(false);
    ControlledLoadWorker workers[2];
    if (GetParam() == FanoutOutcome::LEGACY_WORKER)
        workers[1].prepare_error = grpc::StatusCode::UNIMPLEMENTED;
    if (GetParam() == FanoutOutcome::WORKER_CANCELLED)
        workers[1].prepare_error = grpc::StatusCode::CANCELLED;
    workers[1].block_prepare = GetParam() == FanoutOutcome::CANCEL_DURING_PREPARE;
    std::vector<std::unique_ptr<grpc::Server>> servers;
    for (auto& worker : workers) {
        grpc::ServerBuilder builder;
        int                 port = 0;
        builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
        builder.RegisterService(&worker);
        servers.push_back(builder.BuildAndStart());
        ASSERT_TRUE(servers.back());
        auto address = "127.0.0.1:" + std::to_string(port);
        decode.resource_.workers.push_back(address);
        decode.resource_.grpc_workers.push_back(address);
    }

    grpc::ServerContext          server_context;
    DecodeRpcContext             rpc_context{nullptr};
    kmonitor::MetricsReporterPtr reporter;
    CancellableDecodeLoad        context(
        rpc_context, 10000, &server_context, reporter, std::make_shared<RpcServerRuntimeMeta>());
    context.peer_addrs = {"127.0.0.1:12345:12346"};
    context.setStream(stream_);
    auto outcome = std::async(std::launch::async, [&] { return decode.loadCacheForAllRank(context); });
    if (workers[1].prepare_error != grpc::StatusCode::OK || workers[1].block_prepare) {
        if (workers[1].block_prepare) {
            EXPECT_EQ(workers[1].prepare_entered.get_future().wait_for(5s), std::future_status::ready);
            context.cancelled.store(true);
        }
        EXPECT_EQ(outcome.wait_for(5s), std::future_status::ready);
        if (workers[1].block_prepare)
            workers[1].release_prepare.set_value();
        const auto code = outcome.get().error_info.code();
        EXPECT_EQ(code, workers[1].block_prepare ? ErrorCode::CANCELLED : ErrorCode::LOAD_KV_CACHE_FAILED);
        EXPECT_EQ(workers[0].load_calls.load(), 0);
        EXPECT_EQ(workers[1].load_calls.load(), 0);
        EXPECT_EQ(pool->referencedBlocksNum(DeviceBlockRefType::PD_CONNECTOR), 0u);
        context.markRpcHandlingCompleted();
        return;
    }
    EXPECT_EQ(workers[0].entered.get_future().wait_for(5s), std::future_status::ready);
    EXPECT_EQ(workers[1].entered.get_future().wait_for(5s), std::future_status::ready);
    workers[0].allow_reply.set_value(true);
    EXPECT_EQ(workers[0].reply_prepared.get_future().wait_for(5s), std::future_status::ready);
    context.cancelled.store(true);
    EXPECT_EQ(context.cancel_observed.get_future().wait_for(5s), std::future_status::ready);

    stream_->reportError(ErrorCode::CANCELLED, "test cancelled during fanout");
    stream_->releaseResource();
    EXPECT_FALSE(pool->hasRequestRef(block));
    EXPECT_TRUE(pool->isAllocated(block));
    EXPECT_FALSE(pool->isFrozen(block));
    EXPECT_EQ(pool->referencedBlocksNum(DeviceBlockRefType::PD_CONNECTOR), 1u);
    EXPECT_EQ(outcome.wait_for(5s), std::future_status::ready);
    EXPECT_EQ(outcome.get().error_info.code(), ErrorCode::CANCELLED);
    context.markRpcHandlingCompleted();
    ASSERT_EQ(workers[1].first_seal.get_future().wait_for(5s), std::future_status::ready);

    EXPECT_FALSE(decode.waitForDrain(std::chrono::system_clock::now()));
    const auto generation   = cache_manager_->allocationGeneration();
    const bool acknowledged = GetParam() == FanoutOutcome::ACKNOWLEDGED;
    workers[1].allow_reply.set_value(acknowledged);
    EXPECT_EQ(workers[1].reply_prepared.get_future().wait_for(5s), std::future_status::ready);
    if (!acknowledged) {
        // Repeated stop queries alone cannot release a target a worker can still write.
        EXPECT_FALSE(cache_manager_->waitForAllocationChange(generation, 150));
        EXPECT_EQ(pool->referencedBlocksNum(DeviceBlockRefType::PD_CONNECTOR), 1u);
        EXPECT_FALSE(pool->isFrozen(block));
        workers[1].completePhysicalWrite();
    }
    EXPECT_TRUE(cache_manager_->waitForAllocationChange(generation, 5000));
    EXPECT_GE(workers[1].seal_calls.load(), 2);
    EXPECT_EQ(pool->referencedBlocksNum(DeviceBlockRefType::PD_CONNECTOR), 0u);
    EXPECT_FALSE(pool->isAllocated(block));
    EXPECT_TRUE(decode.waitForDrain(std::chrono::system_clock::now() + 5s));
}

INSTANTIATE_TEST_SUITE_P(WriteLeaseSettlement,
                         PdSepRdmaFanoutTest,
                         ::testing::Values(FanoutOutcome::ACKNOWLEDGED,
                                           FanoutOutcome::LOST_RESPONSE,
                                           FanoutOutcome::LEGACY_WORKER,
                                           FanoutOutcome::WORKER_CANCELLED,
                                           FanoutOutcome::CANCEL_DURING_PREPARE));

}  // namespace rtp_llm
