#include "gtest/gtest.h"
#include "rtp_llm/cpp/cache/test/TestLayoutSpec.h"
#include "gmock/gmock.h"

#define private public
#define protected public
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/cache/KVCacheTransferPlanner.h"
#include "rtp_llm/cpp/cache/connector/p2p/plan/KVCacheTransferPlanner.h"
#include "rtp_llm/cpp/cache/connector/p2p/plan/ShardLayoutFactory.h"
#include "rtp_llm/cpp/cache/connector/p2p/LayerCacheBufferUtil.h"
#include "rtp_llm/cpp/cache/connector/p2p/LayerBlockConverterImpl.h"
#include "rtp_llm/cpp/cache/KVCacheResource.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/disaggregate/cache_store/RequestBlockBufferStore.h"
#include "rtp_llm/cpp/disaggregate/cache_store/NormalCacheStore.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PSchedulerDecodeRead.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"
#include "rtp_llm/cpp/engine_base/stream/StreamCacheResource.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/config/RoleTypes.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"

#include <algorithm>
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
#include <future>
#include <set>
#include "autil/NetUtil.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/TransferBackendFactory.h"

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

// Per-request metadata for a one-context-batch, one-block cache-store write.
// Physical geometry (strides, tokens per block, opaque-store flag, tag) now lives
// in CacheConfig / LayerKVCache, so it is no longer part of the write inputs.
torch_ext::PyCacheStoreInputs makeSingleBlockWriteInputs(int64_t cache_key, int request_id_val, int tokens_per_block) {
    torch_ext::PyCacheStoreInputs inputs;
    inputs.input_lengths_host    = torch::tensor({tokens_per_block}, torch::kInt32);
    inputs.prefix_lengths_host   = torch::tensor({0}, torch::kInt32);
    inputs.publish_start_tokens  = torch::tensor({0}, torch::kInt32);
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
    inputs.publish_start_tokens  = torch::tensor({0}, torch::kInt32);
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
// PD KV cache release and reuse correctness, including grouped cache publication.
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
                               uint32_t kernel_seq_size_per_blk = kDsv4TokensPerBlock,
                               ParallelismConfig pc = ParallelismConfig{}) {
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

        KVCacheConfig     kv_config;
        kv_config.seq_size_per_block        = seq_size_per_block;
        kv_config.kernel_seq_size_per_block = kernel_seq_size_per_blk;
        auto config                         = CacheConfigCreator::createWarmupConfig(mc, pc, kv_config, 0);
        // Publish fixture capacity before consumers inspect the group layout.
        config.finalizeBlockNums(block_num, RuntimeConfig{});
        return config;
    }

    void verifyDsv4TcpRegionTransfer(uint32_t logical_tokens, uint32_t kernel_tokens);

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

TEST_F(PdSepKVCacheReleaseTest, MissingGroupIdentityRejectsAsyncLoadBeforeConnecting) {
    const auto config = makeConfig();
    P2PConnectorSchedulerConfig scheduler_config;
    scheduler_config.topology = config.topologyPtr();
    scheduler_config.parallelism_config.tp_size = 1;
    // No RPC client or worker pool: route validation must finish before either is needed.
    P2PSchedulerDecodeRead scheduler(scheduler_config, nullptr, nullptr);
    const auto plan = scheduler.planFor(1, 1, "missing-identity");
    ASSERT_TRUE(plan->ok());
    KVCacheResource resource;
    resource.cacheKeys() = {42};
    auto result = scheduler.buildDecodeRankRoutes(plan->plan, resource, {0, 1}, 1);
    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), ErrorCode::P2P_CONNECTOR_SCHEDULER_STREAM_RESOURCE_FAILED);
}

TEST_F(PdSepKVCacheReleaseTest, MalformedGroupIndexRejectsAsyncLoadBeforeConnecting) {
    const auto config = makeConfig();
    P2PConnectorSchedulerConfig scheduler_config;
    scheduler_config.topology = config.topologyPtr();
    scheduler_config.parallelism_config.tp_size = 1;
    P2PSchedulerDecodeRead scheduler(scheduler_config, nullptr, nullptr);
    const auto plan = scheduler.planFor(1, 1, "malformed-identity");
    ASSERT_TRUE(plan->ok());
    KVCacheResource resource;
    resource.initGroups(config.topologyPtr());
    resource.cacheKeys() = {42};
    auto holder = std::make_shared<BlockIds>();
    holder->assign({1});
    const auto tag = config.groupTags().front();
    const std::vector<GroupBlockIds> malformed{{{}, {holder}},
                                               {{{tag, 0}, {"extra", 0}}, {holder, holder}},
                                               {{{tag, 1}}, {holder}},
                                               {{{tag, 0}, {"extra", 2}}, {holder, holder}},
                                               {{{tag, 0}}, {nullptr}}};
    for (const auto& blocks : malformed) {
        // Fault injection: the dispatch projection must reject malformed identity before any RPC.
        const_cast<GroupBlockIds&>(resource.groupBlockIds()) = blocks;
        auto result = scheduler.buildDecodeRankRoutes(plan->plan, resource, {0, 1}, 1);
        EXPECT_FALSE(result.ok());
        EXPECT_EQ(result.status().code(), ErrorCode::P2P_CONNECTOR_SCHEDULER_STREAM_RESOURCE_FAILED);
    }
}

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
// insertIntoCache is called during releaseResource (device reuse cache)
// After releaseResource, the cache keys should be findable in the block cache
// (i.e., a subsequent allocation with the same tokens hits reuse)
// =============================================================================
TEST_F(PdSepKVCacheReleaseTest, testInsertIntoCache_CalledDuringRelease_ReuseWorks) {
    const std::vector<int> tokens = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14};
    prepareStream(tokens);
    allocateAndFinish();

    auto& resource = stream_->streamCacheResource();
    ASSERT_GT(resource.curBlocksNum(), 0);

    // Engine thread releases: should call insertIntoCache (device cache)
    stream_->releaseResource();
    EXPECT_TRUE(resource.resource_released_);

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

void PdSepKVCacheReleaseTest::verifyDsv4TcpRegionTransfer(uint32_t logical_tokens, uint32_t kernel_tokens) {
    auto config = makeDsv4Config(/*block_num=*/4, logical_tokens, kernel_tokens);
    auto manager = std::make_shared<KVCacheManager>(config, false, nullptr);
    ASSERT_TRUE(manager->init());
    transfer::TransferBackendConfig backend_config;
    backend_config.cache_store_listen_port = autil::NetUtil::randomPort();
    backend_config.messager_io_thread_count = 1;
    backend_config.messager_worker_thread_count = 1;
    auto sender = transfer::createTransferBackend(transfer::TransferBackend::kTcp, backend_config,
                                                   transfer::TransferBackendRole::kSender).sender;
    auto receiver = transfer::createTransferBackend(transfer::TransferBackend::kTcp, backend_config,
                                                     transfer::TransferBackendRole::kReceiver).receiver;
    ASSERT_NE(sender, nullptr);
    ASSERT_NE(receiver, nullptr);
    std::set<std::string> covered_groups;
    size_t region_count = 0;
    for (const auto& layer : config.topology().layers()) {
        for (const auto& tag : layer.group_tags) {
            auto source = manager->convertIndexToBuffer(layer.layer_id, tag, 1);
            auto target = manager->convertIndexToBuffer(layer.layer_id, tag, 2);
            ASSERT_FALSE(source.empty()) << tag;
            ASSERT_EQ(source.size(), target.size()) << tag;
            std::vector<std::vector<uint8_t>> expected;
            for (size_t region = 0; region < source.size(); ++region) {
                ASSERT_EQ(source[region].size_bytes, target[region].size_bytes);
                std::vector<uint8_t> bytes(source[region].size_bytes);
                for (size_t offset = 0; offset < bytes.size(); ++offset) {
                    bytes[offset] = static_cast<uint8_t>((region_count * 37 + offset) % 251);
                }
                auto host = torch::from_blob(bytes.data(), {static_cast<int64_t>(bytes.size())}, torch::kUInt8);
                auto src = torch::from_blob(source[region].addr, {static_cast<int64_t>(bytes.size())},
                    torch::TensorOptions(torch::kUInt8).device(source[region].is_cuda ? torch::kCUDA : torch::kCPU));
                auto dst = torch::from_blob(target[region].addr, {static_cast<int64_t>(bytes.size())},
                    torch::TensorOptions(torch::kUInt8).device(target[region].is_cuda ? torch::kCUDA : torch::kCPU));
                src.copy_(host);
                dst.fill_(255);
                expected.push_back(std::move(bytes));
                ++region_count;
            }
            runtimeSyncAndCheck();
            auto src_info = std::make_shared<transfer::KeyBlockInfo>();
            auto dst_info = std::make_shared<transfer::KeyBlockInfo>();
            src_info->blocks = source;
            dst_info->blocks = target;
            const auto key = "dsv4-seven-groups-" + std::to_string(layer.layer_id) + "-" + tag;
            const auto deadline = currentTimeMs() + 10000;
            transfer::RecvRequest receive;
            receive.unique_key = key;
            receive.deadline_ms = deadline;
            receive.block_info.emplace(1001, dst_info);
            auto task = receiver->recv(receive);
            ASSERT_NE(task, nullptr);
            transfer::SendRequest send;
            send.ip = "127.0.0.1";
            send.port = backend_config.cache_store_listen_port;
            send.unique_key = key;
            send.deadline_ms = deadline;
            send.timeout_ms = 10000;
            send.block_info.emplace(1001, src_info);
            auto completion = std::make_shared<std::promise<transfer::TransferErrorCode>>();
            auto done = completion->get_future();
            sender->send(send, [completion](auto code, const std::string&) { completion->set_value(code); });
            ASSERT_EQ(done.wait_for(std::chrono::seconds(15)), std::future_status::ready);
            ASSERT_EQ(done.get(), transfer::TransferErrorCode::OK) << tag;
            ASSERT_TRUE(task->success()) << task->errorMessage();
            runtimeSyncAndCheck();
            for (size_t region = 0; region < target.size(); ++region) {
                auto dst = torch::from_blob(target[region].addr, {static_cast<int64_t>(target[region].size_bytes)},
                    torch::TensorOptions(torch::kUInt8).device(target[region].is_cuda ? torch::kCUDA : torch::kCPU));
                auto actual = dst.cpu().contiguous();
                EXPECT_EQ(std::memcmp(actual.data_ptr<uint8_t>(), expected[region].data(), expected[region].size()), 0)
                    << "layer=" << layer.layer_id << " tag=" << tag << " region=" << region;
            }
            covered_groups.insert(tag);
        }
    }
    EXPECT_EQ(covered_groups.size(), kDsv4PoolNum);
    EXPECT_GT(region_count, covered_groups.size());
}

TEST_F(PdSepKVCacheReleaseTest, Dsv4RealCpPlanTransfersSuffixThroughTcpAndPreservesReusedPrefix) {
    ParallelismConfig prefill_pc;
    prefill_pc.role_type                          = RoleType::PREFILL;
    prefill_pc.tp_size                            = 2;
    prefill_pc.prefill_cp_config.method           = CPRotateMethod::ALL_GATHER;
    prefill_pc.prefill_cp_config.kv_cache_sharded = true;
    prefill_pc.prefill_cp_config.prefill_cp_size  = 2;
    auto decode_pc                                = prefill_pc;
    decode_pc.role_type                           = RoleType::DECODE;
    decode_pc.tp_size                             = 1;
    decode_pc.prefill_cp_config.method            = CPRotateMethod::PREFILL_CP;
    const auto prefill_config                     = makeDsv4Config(8, 256, 128, prefill_pc);
    const auto decode_config                      = makeDsv4Config(8, 256, 128, decode_pc);
    const auto source =
        ShardLayoutFactory::fromTopology(prefill_config.topology(), prefill_pc, RoleType::PREFILL, kDsv4TokensPerBlock);
    const auto target =
        ShardLayoutFactory::fromTopology(decode_config.topology(), decode_pc, RoleType::DECODE, kDsv4TokensPerBlock);
    const auto tags = ShardLayoutFactory::tagsOf(decode_config.topology());
    ASSERT_EQ(tags.size(), kDsv4PoolNum);
    const auto planned = KVCacheTransferPlanner::plan(source, target, tags);
    ASSERT_TRUE(planned.ok()) << planned.error.ToString();
    // Prefix filtering must happen after resolving the original request's key
    // geometry. Resolving a shortened suffix would change CP ownership and
    // compact-tail placement instead of preserving absolute key positions.
    constexpr size_t      full_key_count = 7;
    constexpr size_t      reused_keys    = 3;  // Odd prefix shifts CP ownership if incorrectly rebased.
    std::set<std::string> covered;
    std::set<int>         source_ranks;
    bool                  has_byte_slice      = false;
    bool                  has_retained_suffix = false;
    for (const auto& route : planned.plan.routes) {
        EXPECT_EQ(route.src_bytes, route.dst_bytes) << route.cache_tag;
        EXPECT_GT(route.src_bytes, 0u) << route.cache_tag;
        auto       positions = KVCacheTransferPlanner::resolveKeys(route.src_keys, full_key_count);
        const auto original  = positions;
        positions.erase(
            std::remove_if(positions.begin(), positions.end(), [](size_t pos) { return pos < reused_keys; }),
            positions.end());
        for (const auto position : positions) {
            EXPECT_GE(position, reused_keys);
            EXPECT_LT(position, full_key_count);
            EXPECT_NE(std::find(original.begin(), original.end(), position), original.end());
        }
        has_retained_suffix = has_retained_suffix || !positions.empty();
        has_byte_slice      = has_byte_slice || route.dst_slice.mode != CpBlockSliceMode::NONE;
        source_ranks.insert(route.src_rank);
        covered.insert(route.cache_tag);
    }
    EXPECT_EQ(covered.size(), kDsv4PoolNum);
    EXPECT_EQ(source_ranks, (std::set<int>{0, 1}));
    EXPECT_TRUE(has_byte_slice);
    EXPECT_TRUE(has_retained_suffix);

    std::vector<std::shared_ptr<KVCacheManager>>      source_managers;
    std::vector<std::shared_ptr<LayerBlockConverter>> source_converters;
    std::vector<KVCacheResource>                      source_resources(2);
    CacheKeysType                                     keys(full_key_count);
    std::iota(keys.begin(), keys.end(), 1001);
    BlockIndicesType physical_ids(full_key_count);
    std::iota(physical_ids.begin(), physical_ids.end(), 1);
    for (int rank = 0; rank < 2; ++rank) {
        auto manager = std::make_shared<KVCacheManager>(prefill_config, false, nullptr);
        ASSERT_TRUE(manager->init());
        source_converters.push_back(std::make_shared<LayerBlockConverterImpl>(manager->coordinator_manager_));
        source_managers.push_back(std::move(manager));
        source_resources[rank].initGroups(prefill_config.topologyPtr());
        source_resources[rank].setCacheKeys(keys);
        for (const auto& tag : tags) {
            source_resources[rank].mutableBlockIds(tag).assign(physical_ids);
        }
    }
    auto target_manager = std::make_shared<KVCacheManager>(decode_config, false, nullptr);
    ASSERT_TRUE(target_manager->init());
    auto            target_converter = std::make_shared<LayerBlockConverterImpl>(target_manager->coordinator_manager_);
    KVCacheResource target_resource;
    target_resource.initGroups(decode_config.topologyPtr());
    target_resource.setCacheKeys(keys);
    for (const auto& tag : tags) {
        target_resource.mutableBlockIds(tag).assign(physical_ids);
    }
    transfer::TransferBackendConfig backend_config;
    backend_config.cache_store_listen_port      = autil::NetUtil::randomPort();
    backend_config.messager_io_thread_count     = 1;
    backend_config.messager_worker_thread_count = 1;
    auto sender                                 = transfer::createTransferBackend(
                      transfer::TransferBackend::kTcp, backend_config, transfer::TransferBackendRole::kSender)
                      .sender;
    auto receiver = transfer::createTransferBackend(
                        transfer::TransferBackend::kTcp, backend_config, transfer::TransferBackendRole::kReceiver)
                        .receiver;
    ASSERT_NE(sender, nullptr);
    ASSERT_NE(receiver, nullptr);
    auto tensor = [](const BlockInfo& block) {
        return torch::from_blob(block.addr,
                                {static_cast<int64_t>(block.size_bytes)},
                                torch::TensorOptions(torch::kUInt8).device(block.is_cuda ? torch::kCUDA : torch::kCPU));
    };
    size_t                transferred_regions = 0;
    size_t                skipped_prefix_keys = 0;
    std::set<std::string> transferred_tags;
    for (const auto& layer : decode_config.topology().layers()) {
        for (const auto& tag : layer.group_tags) {
            std::vector<BlockInfo>            target_regions;
            std::vector<std::vector<uint8_t>> expected;
            for (const auto block_id : physical_ids) {
                for (const auto& part : target_manager->convertIndexToBuffer(layer.layer_id, tag, block_id)) {
                    tensor(part).fill_(255);
                    target_regions.push_back(part);
                    expected.emplace_back(part.size_bytes, 255);
                }
            }
            ASSERT_FALSE(target_regions.empty());
            for (const auto* route : planned.plan.forDecodeRankAndTag(0, tag)) {
                const auto all_positions = KVCacheTransferPlanner::resolveKeys(route->src_keys, full_key_count);
                auto       suffix        = all_positions;
                suffix.erase(std::remove_if(suffix.begin(), suffix.end(), [](size_t pos) { return pos < reused_keys; }),
                             suffix.end());
                if (suffix.empty()) {
                    continue;
                }
                auto src_buffers = LayerCacheBufferUtil::convertTagForRoute(source_resources.at(route->src_rank),
                                                                            prefill_config.topology(),
                                                                            tag,
                                                                            all_positions,
                                                                            source.cpRank(route->src_rank),
                                                                            source.physicalCpSize(tag),
                                                                            kDsv4TokensPerBlock);
                auto dst_buffers = LayerCacheBufferUtil::convertTagForRoute(target_resource,
                                                                            decode_config.topology(),
                                                                            tag,
                                                                            suffix,
                                                                            0,
                                                                            target.physicalCpSize(tag),
                                                                            kDsv4TokensPerBlock);
                ASSERT_TRUE(src_buffers.ok()) << src_buffers.status().ToString();
                ASSERT_TRUE(dst_buffers.ok()) << dst_buffers.status().ToString();
                auto find_layer = [&](const auto& buffers) {
                    return std::find_if(buffers.begin(), buffers.end(), [&](const auto& buffer) {
                        return buffer->getLayerId() == layer.layer_id;
                    });
                };
                auto src_layer = find_layer(src_buffers.value());
                auto dst_layer = find_layer(dst_buffers.value());
                ASSERT_NE(src_layer, src_buffers.value().end());
                ASSERT_NE(dst_layer, dst_buffers.value().end());
                if (source.effectiveMapping(tag) == CpBlockMappingMode::BLOCK_ROUND_ROBIN
                    && source.groups.at(tag).policy.group_type == CacheGroupType::FULL) {
                    for (const auto pos : all_positions) {
                        EXPECT_EQ((*src_layer)->getBlockId(keys[pos]), physical_ids[pos / 2]);
                    }
                    for (const auto pos : suffix) {
                        EXPECT_EQ((*dst_layer)->getBlockId(keys[pos]), physical_ids[pos]);
                    }
                }

                auto src_blocks =
                    LayerCacheBufferUtil::buildKeyBlockInfosSliced(source_converters.at(route->src_rank),
                                                                   *src_layer,
                                                                   route->src_partition.count,
                                                                   route->src_partition.id,
                                                                   route->src_slice,
                                                                   source.groups.at(tag).k_block_payload_bytes);
                auto dst_blocks =
                    LayerCacheBufferUtil::buildKeyBlockInfosSliced(target_converter,
                                                                   *dst_layer,
                                                                   route->dst_partition.count,
                                                                   route->dst_partition.id,
                                                                   route->dst_slice,
                                                                   target.groups.at(tag).k_block_payload_bytes);
                ASSERT_TRUE(src_blocks.ok()) << src_blocks.status().ToString();
                ASSERT_TRUE(dst_blocks.ok()) << dst_blocks.status().ToString();
                skipped_prefix_keys += src_blocks.value().size() - dst_blocks.value().size();
                transferred_tags.insert(tag);
                // Prefill publishes the full route key set. Decode receives only
                // its suffix; prefix bytes and padding must remain untouched.
                for (const auto& [key, info] : src_blocks.value()) {
                    for (size_t part_index = 0; part_index < info->blocks.size(); ++part_index) {
                        tensor(info->blocks[part_index])
                            .fill_(static_cast<int>((key * 17 + route->src_rank * 31 + part_index) % 251));
                    }
                }
                runtimeSyncAndCheck();
                for (const auto& [key, info] : dst_blocks.value()) {
                    const auto& src_parts = src_blocks.value().at(key)->blocks;
                    ASSERT_EQ(src_parts.size(), info->blocks.size());
                    for (size_t part_index = 0; part_index < info->blocks.size(); ++part_index) {
                        const auto& part = info->blocks[part_index];
                        ASSERT_EQ(part.size_bytes, src_parts[part_index].size_bytes);
                        auto       bytes   = tensor(src_parts[part_index]).cpu().contiguous();
                        bool       found   = false;
                        const auto address = reinterpret_cast<uintptr_t>(part.addr);
                        for (size_t region = 0; region < target_regions.size(); ++region) {
                            const auto base = reinterpret_cast<uintptr_t>(target_regions[region].addr);
                            if (address >= base && address - base <= expected[region].size()
                                && part.size_bytes <= expected[region].size() - (address - base)) {
                                std::memcpy(expected[region].data() + address - base,
                                            bytes.data_ptr<uint8_t>(),
                                            part.size_bytes);
                                found = true;
                                break;
                            }
                        }
                        ASSERT_TRUE(found) << tag << " key=" << key;
                        ++transferred_regions;
                    }
                }
                const auto unique_key = "dsv4-planned-" + std::to_string(layer.layer_id) + "-" + tag + "-"
                                        + std::to_string(route->route_id);
                const auto            deadline = currentTimeMs() + 10000;
                transfer::RecvRequest receive;
                receive.unique_key  = unique_key;
                receive.deadline_ms = deadline;
                receive.block_info  = dst_blocks.value();
                auto task           = receiver->recv(receive);
                ASSERT_NE(task, nullptr);
                transfer::SendRequest send;
                send.ip          = "127.0.0.1";
                send.port        = backend_config.cache_store_listen_port;
                send.unique_key  = unique_key;
                send.deadline_ms = deadline;
                send.timeout_ms  = 10000;
                send.block_info  = src_blocks.value();
                auto completion  = std::make_shared<std::promise<transfer::TransferErrorCode>>();
                auto done        = completion->get_future();
                sender->send(send, [completion](auto code, const std::string&) { completion->set_value(code); });
                ASSERT_EQ(done.wait_for(std::chrono::seconds(15)), std::future_status::ready);
                ASSERT_EQ(done.get(), transfer::TransferErrorCode::OK) << unique_key;
                ASSERT_TRUE(task->success()) << task->errorMessage();
            }
            runtimeSyncAndCheck();
            for (size_t region = 0; region < target_regions.size(); ++region) {
                auto actual = tensor(target_regions[region]).cpu().contiguous();
                EXPECT_EQ(std::memcmp(actual.data_ptr<uint8_t>(), expected[region].data(), expected[region].size()), 0)
                    << "layer=" << layer.layer_id << " tag=" << tag << " region=" << region;
            }
        }
    }
    EXPECT_GT(transferred_regions, kDsv4PoolNum);
    EXPECT_GT(skipped_prefix_keys, 0u);
    EXPECT_EQ(transferred_tags.size(), kDsv4PoolNum);
}
TEST_F(PdSepKVCacheReleaseTest, Dsv4SevenGroupsTransferEveryLayerRegionThroughTcp) {
    ASSERT_NO_FATAL_FAILURE(verifyDsv4TcpRegionTransfer(kDsv4TokensPerBlock, kDsv4TokensPerBlock));
}

TEST_F(PdSepKVCacheReleaseTest, Dsv4DecoupledKernelBlocksTransferEveryLayerRegionThroughTcp) {
    ASSERT_NO_FATAL_FAILURE(verifyDsv4TcpRegionTransfer(kDsv4TokensPerBlock, 128));
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
    for (const auto& tag : config.groupTags()) {
        ASSERT_EQ(resource.kvCache().blocksNum(0, tag), 4) << "group " << tag;
        const auto&  blocks = resource.kvCache().blocks(0, tag);
        const size_t tail   = static_cast<size_t>(config.group(tag).policy.active_tail_blocks);
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

    stream_->releaseResource();
    EXPECT_TRUE(resource.resource_released_);

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
    stream_->releaseResource();

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

}  // namespace rtp_llm
