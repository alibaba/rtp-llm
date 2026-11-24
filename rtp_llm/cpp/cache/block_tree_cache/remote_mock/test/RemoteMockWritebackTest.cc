#include "rtp_llm/cpp/cache/block_tree_cache/transfer/test/RemoteAllocationFailureTestUtils.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/test/RemoteCopyFailureTestUtils.h"
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <future>
#include <map>
#include <atomic>
#include <array>
#include <cstring>
#include <condition_variable>
#include "rtp_llm/cpp/disaggregate/cache_store/RequestBlockBufferStore.h"
#include "autil/NetUtil.h"
#include "rtp_llm/cpp/cache/block_tree_cache/remote_mock/RemoteMockAdapter.h"
#include "rtp_llm/cpp/cache/block_tree_cache/remote_mock/RemoteMockWriteback.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferRequestConverter.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/test/PerRankBlockTransferEngineTestUtils.h"
#include "rtp_llm/cpp/disaggregate/cache_store/remote_mock/RemoteMockService.h"
#include "rtp_llm/cpp/disaggregate/cache_store/TcpServer.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"

namespace rtp_llm {
namespace {
struct WriteBorrowObserver {
    std::mutex                                  mutex;
    std::condition_variable                     cv;
    std::map<std::string, size_t>               lookups;
    std::map<std::string, std::vector<uint8_t>> copied;
    RequestBlockBufferStore*                    registry{nullptr};
    std::string                                 gated_key;
    bool                                        entered{false}, opened{false};
    bool                                        hold_continuation{false}, continuation_entered{false};
    bool                                        fail_after_copy{false}, fail_after_ack{false};
    size_t                                      continuation_count{0};
    void                                        open() {
        std::lock_guard<std::mutex> lock(mutex);
        opened = true;
        cv.notify_all();
    }
};
std::mutex                         observer_mutex;
std::weak_ptr<WriteBorrowObserver> observer;
}  // namespace
extern "C" std::shared_ptr<BlockBuffer> realWriterFind(RequestBlockBufferStore*, const std::string&) asm(
    "__real__ZN7rtp_llm23RequestBlockBufferStore14findUserBufferERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE");
extern "C" std::shared_ptr<BlockBuffer> wrapWriterFind(RequestBlockBufferStore*, const std::string&) asm(
    "__wrap__ZN7rtp_llm23RequestBlockBufferStore14findUserBufferERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE");
extern "C" std::shared_ptr<BlockBuffer> wrapWriterFind(RequestBlockBufferStore* registry, const std::string& key) {
    auto                                 block = realWriterFind(registry, key);
    std::shared_ptr<WriteBorrowObserver> retained;
    {
        std::lock_guard<std::mutex> lock(observer_mutex);
        retained = observer.lock();
    }
    if (retained && block && !block->gpu_mem) {
        std::unique_lock<std::mutex> lock(retained->mutex);
        retained->registry = registry;
        if (++retained->lookups[key] == 2) {  // Actual reverse blockRead borrower, after metadata lookup.
            if (key == retained->gated_key) {
                retained->entered = true;
                retained->cv.notify_all();
                retained->cv.wait_for(lock, std::chrono::seconds(10), [&] { return retained->opened; });
            }
            const auto* bytes     = static_cast<const uint8_t*>(block->addr.get());
            retained->copied[key] = std::vector<uint8_t>(bytes, bytes + block->len);
        }
    }
    return block;
}
// Gate only the writer's third continuation: after first Host ACK, before
// the next slice. The original task pool and all transport callbacks still run.
extern "C" bool realWriterCompletion(BlockTreeTaskPool*, std::function<void()>) asm(
    "__real__ZN7rtp_llm17BlockTreeTaskPool16submitCompletionESt8functionIFvvEE");
extern "C" bool wrapWriterCompletion(BlockTreeTaskPool*, std::function<void()>) asm(
    "__wrap__ZN7rtp_llm17BlockTreeTaskPool16submitCompletionESt8functionIFvvEE");
extern "C" bool wrapWriterCompletion(BlockTreeTaskPool* pool, std::function<void()> task) {
    std::shared_ptr<WriteBorrowObserver> retained;
    {
        std::lock_guard<std::mutex> lock(observer_mutex);
        retained = observer.lock();
    }
    bool gated = false;
    if (retained && pool->thread_name_ == "RemoteWriteback") {
        std::lock_guard<std::mutex> lock(retained->mutex);
        ++retained->continuation_count;
        if (retained->fail_after_copy && retained->continuation_count == 2) {
            retained->fail_after_copy = false;
            throw std::bad_alloc();
        }
        if (retained->fail_after_ack && retained->continuation_count == 3) {
            retained->fail_after_ack = false;
            throw std::bad_alloc();
        }
        gated = retained->hold_continuation && retained->continuation_count == 3;
    }
    if (gated)
        return realWriterCompletion(pool, [retained, task = std::move(task)] {
            {
                std::unique_lock<std::mutex> lock(retained->mutex);
                retained->continuation_entered = true;
                retained->cv.notify_all();
                retained->cv.wait_for(lock, std::chrono::seconds(10), [&] { return retained->opened; });
            }
            task();
        });
    return realWriterCompletion(pool, std::move(task));
}
namespace {
using namespace block_transfer_engine_test;
class RemoteMockWritebackTest: public ::testing::Test {
protected:
    void SetUp() override {
        observation = std::make_shared<WriteBorrowObserver>();
        {
            std::lock_guard<std::mutex> lock(observer_mutex);
            observer = observation;
        }
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
        initRuntime(0);
        RemoteMockServiceOptions options;
        options.limits.total_bytes        = 1 << 20;
        options.limits.source_bytes       = 1 << 16;
        options.limits.write_slot_bytes   = 1024;
        options.limits.read_staging_bytes = options.limits.write_staging_bytes = 1 << 16;
        options.limits.max_rpc_bytes                                           = 64;
        service = std::make_unique<RemoteMockService>(options);
        ASSERT_TRUE(service->init());
        server = std::make_unique<TcpServer>();
        ASSERT_TRUE(server->init(1, 2, true, 32));
        ASSERT_TRUE(server->registerService(service.get()));
        ASSERT_TRUE(server->registerService(service->controlService()));
        const auto port = autil::NetUtil::randomPort();
        ASSERT_TRUE(server->start(port));
        config.mock_remote_cache                 = true;
        config.mock_remote_cache_hit_ratio       = 0;
        config.mock_remote_cache_address         = "127.0.0.1:" + std::to_string(port);
        config.mock_remote_cache_callback_port   = autil::NetUtil::randomPort();
        config.mock_remote_cache_source_blocks   = 2;
        config.mock_remote_cache_staging_size_mb = 1;
        config.mock_remote_cache_timeout_ms      = 3000;
        device                                   = makeTestDevicePool({{64, 16}, {64, 16}}, 4, "remote_writer_gpu");
        auto policy                              = defaultCacheGroupPolicy(CacheGroupType::FULL);
        policy.enable_prefix_reuse               = true;
        group                                    = makeTestGroupSet(0,
                                 makeTestTopology({makeTestGroupBase(policy, {0, 1}, 64, 16)}),
                                                                    {"group0"},
                                                                    {device},
                                 nullptr,
                                 nullptr,
                                 true);
        engine = std::make_shared<PerRankBlockTransferEngine>(std::vector<GroupSetPtr>{group});
        block  = poolMalloc(*device);
        device->incRef(block);  // The request retains its allocation independently of STORE.
        for (int layer = 0; layer < 2; ++layer)
            for (const auto& buffer : device->convertIndexToBuffer(layer, block))
                ASSERT_EQ(cudaMemset(buffer.addr, 0xa5, buffer.size_bytes), cudaSuccess);
        for (const auto guard : {2, 3})
            for (int layer = 0; layer < 2; ++layer)
                for (const auto& buffer : device->convertIndexToBuffer(layer, guard))
                    ASSERT_EQ(cudaMemset(buffer.addr, 0x37, buffer.size_bytes), cudaSuccess);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    }
    void attach() {
        adapter = RemoteMockAdapter::create(config, ParallelismConfig{}, {group}, engine);
        engine->setRemoteMockAdapter(adapter);
    }
    void drain() {
        std::promise<void> done;
        auto               future = done.get_future();
        adapter->drain([&] { done.set_value(); });
        ASSERT_EQ(future.wait_for(std::chrono::seconds(10)), std::future_status::ready);
    }
    void TearDown() override {
        observation->open();
        if (adapter) {
            adapter->stopAccepting();
            drain();
            adapter.reset();
        }
        engine.reset();
        device->decRef(block);
        service->shutdown();
        server.reset();
    }
    void expectBaseline() {
        auto* registry = observation->registry;
        ASSERT_NE(registry, nullptr);
        {
            std::shared_lock<std::shared_mutex> lock(registry->user_buffers_->mutex);
            EXPECT_TRUE(registry->user_buffers_->buffers.empty());
        }
        {
            std::shared_lock<std::shared_mutex> lock(registry->request_cache_map_mutex_);
            EXPECT_TRUE(registry->request_cache_map_.empty());
            EXPECT_TRUE(registry->expired_request_caches_.empty());
        }
        EXPECT_EQ(device->referencedBlocksNum(BlockTreeRefType::STORE), 0u);
        EXPECT_EQ(engine->remoteWritebackStats().retained_bytes, 0u);
    }
    std::shared_ptr<WriteBorrowObserver> observation;
    TransferTask                         task(std::chrono::milliseconds timeout = std::chrono::seconds(3)) {
        return TransferTask({TransferDescriptor::deviceToRemote(0, adapter->session(), 19, {block})}, timeout);
    }
    KVCacheConfig                               config;
    std::unique_ptr<RemoteMockService>          service;
    std::unique_ptr<TcpServer>                  server;
    DeviceBlockPoolPtr                          device;
    GroupSetPtr                                 group;
    BlockIdxType                                block;
    std::shared_ptr<PerRankBlockTransferEngine> engine;
    std::shared_ptr<RemoteMockAdapter>          adapter;
};

// Missing D2REMOTE execution would omit real reverse RPC bytes at read ratio zero.
TEST_F(RemoteMockWritebackTest, RatioZeroD2hThenReverseBlockReadAndAck) {
    attach();
    ASSERT_EQ(group->payloadBytes(), 160u);
    ASSERT_EQ(group->storageBytes(), 176u);
    auto                         context = engine->execute(task());
    std::promise<TransferResult> terminal;
    auto                         result = terminal.get_future();
    context->onDone([&](TransferResult value) { terminal.set_value(std::move(value)); });
    ASSERT_EQ(result.wait_for(std::chrono::seconds(10)), std::future_status::ready);
    EXPECT_TRUE(result.get().ok());
    drain();
    EXPECT_EQ(service->stats().write_payload_bytes, 177u);  // probe + full packed record
    EXPECT_EQ(service->stats().reverse_rpcs, 4u);           // probe + 64/64/48 slices
    EXPECT_EQ(service->stats().failed_operations, 0u);
    EXPECT_EQ(device->referencedBlocksNum(BlockTreeRefType::STORE), 0u);
    const auto stats = engine->remoteWritebackStats();
    EXPECT_EQ(stats.submitted, 1u);
    EXPECT_EQ(stats.completed, 1u);
    EXPECT_EQ(stats.failed, 0u);
    EXPECT_EQ(stats.retained_bytes, 0u);
    EXPECT_EQ(stats.acknowledged_payload_bytes, 176u);
    EXPECT_EQ(stats.peak_retained_bytes, 192u);  // Entire aligned allocation, not the 64-byte RPC slice.
    std::vector<uint8_t> received;
    {
        std::lock_guard<std::mutex> lock(observation->mutex);
        for (const auto& entry : observation->copied)
            if (entry.second.size() > 1)
                received.insert(received.end(), entry.second.begin(), entry.second.end());
    }
    std::vector<uint8_t> expected(176, 0);
    std::fill(expected.begin(), expected.begin() + 160, 0xa5);
    // Independent bitwise CRC32C of 160 bytes 0xa5 is 0xe212414d; little-endian footer.
    expected[172] = 0x4d;
    expected[173] = 0x41;
    expected[174] = 0x12;
    expected[175] = 0xe2;
    EXPECT_EQ(received, expected);
    for (const auto guard : {2, 3})
        for (int layer = 0; layer < 2; ++layer)
            for (const auto& buffer : device->convertIndexToBuffer(layer, guard)) {
                std::vector<uint8_t> bytes(buffer.size_bytes);
                ASSERT_EQ(cudaMemcpy(bytes.data(), buffer.addr, bytes.size(), cudaMemcpyDeviceToHost), cudaSuccess);
                EXPECT_EQ(bytes, std::vector<uint8_t>(bytes.size(), 0x37));
            }
    expectBaseline();
}
TEST_F(RemoteMockWritebackTest, BorrowedHostAfterTimeoutKeepsQueueBackingAndDrainUntilRealRelease) {
    config.mock_remote_cache_max_inflight = 1;
    attach();
    observation->gated_key = "rmw-" + std::to_string(adapter->session().session_id) + "-0-2-0";
    auto ensure_release    = std::shared_ptr<void>(nullptr, [observed = observation](void*) { observed->open(); });
    std::promise<ErrorInfo> terminal;
    auto                    result = terminal.get_future();
    std::atomic<int>        terminals{0};
    ASSERT_TRUE(adapter->writeback(task(std::chrono::milliseconds(200)), [&](ErrorInfo error) {
        ++terminals;
        terminal.set_value(std::move(error));
    }));
    {
        std::unique_lock<std::mutex> lock(observation->mutex);
        ASSERT_TRUE(observation->cv.wait_for(lock, std::chrono::seconds(3), [&] { return observation->entered; }));
    }
    EXPECT_EQ(result.wait_for(std::chrono::milliseconds(350)), std::future_status::timeout);
    EXPECT_EQ(engine->remoteWritebackStats().retained_bytes, 192u);
    EXPECT_EQ(device->referencedBlocksNum(BlockTreeRefType::STORE), 1u);
    std::promise<ErrorInfo> rejected;
    auto                    rejected_result = rejected.get_future();
    EXPECT_FALSE(adapter->writeback(task(), [&](ErrorInfo error) { rejected.set_value(error); }));
    ASSERT_EQ(rejected_result.wait_for(std::chrono::seconds(1)), std::future_status::ready);
    EXPECT_FALSE(rejected_result.get().ok());
    EXPECT_EQ(engine->remoteWritebackStats().submitted, 1u);
    EXPECT_EQ(engine->remoteWritebackStats().rejected, 1u);
    std::promise<void> drained;
    auto               drained_result = drained.get_future();
    adapter->drain([&] { drained.set_value(); });
    EXPECT_EQ(drained_result.wait_for(std::chrono::milliseconds(30)), std::future_status::timeout);
    ensure_release.reset();
    ASSERT_EQ(result.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    EXPECT_FALSE(result.get().ok());
    ASSERT_EQ(drained_result.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    EXPECT_EQ(terminals, 1);
    EXPECT_EQ(engine->remoteWritebackStats().failed, 1u);
    expectBaseline();
    auto again = engine->execute(task());
    again->waitDone();
    EXPECT_TRUE(again->success());
    drain();
    EXPECT_EQ(engine->remoteWritebackStats().completed, 1u);
    expectBaseline();
}
TEST_F(RemoteMockWritebackTest, RepeatedSlicedRecordsReuseOneBackingAndRemoveAllScopedEntries) {
    attach();
    for (size_t iteration = 0; iteration < 4; ++iteration) {
        auto desc    = TransferDescriptor::deviceToRemote(0, adapter->session(), iteration, {block});
        auto context = engine->execute(TransferTask({desc, desc}, std::chrono::seconds(3)));
        context->waitDone();
        EXPECT_TRUE(context->success());
        drain();
        expectBaseline();
    }
    EXPECT_EQ(engine->remoteWritebackStats().completed, 4u);
    EXPECT_EQ(engine->remoteWritebackStats().acknowledged_payload_bytes, 8u * 176);
    EXPECT_EQ(engine->remoteWritebackStats().peak_retained_bytes, 192u);
    EXPECT_EQ(service->stats().write_payload_bytes, 1u + 8 * 176);
}
TEST_F(RemoteMockWritebackTest, RemoteWriteWireValidatesNeutralSessionAndNoLocalTarget) {
    attach();
    MemoryOperationRequestPB wire;
    ASSERT_TRUE(BlockTransferRequestConverter::encodeTransfer(wire, task(), {group}));
    std::vector<TransferDescriptor> decoded;
    ASSERT_TRUE(BlockTransferRequestConverter::decodeTransfer(wire, decoded, {group}, false, nullptr, adapter.get()));
    ASSERT_EQ(decoded.size(), 1u);
    EXPECT_TRUE(decoded[0].isRemoteWrite());
    EXPECT_EQ(decoded[0].source_blocks, (std::vector<BlockIdxType>{block}));
    EXPECT_EQ(decoded[0].remote_target->record_index, 19u);
    EXPECT_TRUE(decoded[0].target_blocks.empty());
    for (int variant = 0; variant < 4; ++variant) {
        auto invalid = wire;
        if (variant == 0)
            invalid.mutable_copy_items(0)->mutable_remote_mock_target()->set_generation(0);
        if (variant == 1)
            invalid.mutable_copy_items(0)->set_mem_block(1);
        if (variant == 2)
            invalid.mutable_copy_items(0)->mutable_remote_mock_source()->set_generation(1);
        if (variant == 3)
            invalid.set_copy_direction(MemoryOperationRequestPB::D2H);
        std::vector<TransferDescriptor> out;
        EXPECT_FALSE(
            BlockTransferRequestConverter::decodeTransfer(invalid, out, {group}, false, nullptr, adapter.get()));
        EXPECT_TRUE(out.empty());
    }
}
TEST_F(RemoteMockWritebackTest, MetadataBeyondPhysicalCapacityRejectsBeforePinOrCapture) {
    attach();
    auto                            desc = TransferDescriptor::deviceToRemote(0, adapter->session(), 90, {block});
    std::vector<TransferDescriptor> excessive(device->totalBlocksNum() + 1, desc);
    ASSERT_LE(excessive.size(), 8u);
    std::promise<ErrorInfo> terminal;
    auto                    result = terminal.get_future();
    EXPECT_FALSE(adapter->writeback(TransferTask(excessive, std::chrono::seconds(3)),
                                    [&](ErrorInfo error) { terminal.set_value(error); }));
    ASSERT_EQ(result.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    EXPECT_FALSE(result.get().ok());
    drain();
    EXPECT_EQ(engine->remoteWritebackStats().submitted, 0u);
    EXPECT_EQ(engine->remoteWritebackStats().rejected, 1u);
    EXPECT_EQ(service->stats().reverse_rpcs, 1u);
    expectBaseline();
}
TEST_F(RemoteMockWritebackTest, ShutdownKeepsSourceThroughActualCudaAndRejectsNewCapture) {
    group->initialize(0, group->topologyPtr(), group->groupTags(), makeTestBackingLayout(160, false));
    engine       = std::make_shared<PerRankBlockTransferEngine>(std::vector<GroupSetPtr>{group});
    auto gate    = armRemoteCopyFailure(0);
    auto release = std::shared_ptr<void>(nullptr, [gate](void*) { retireRemoteCopyFailure(gate); });
    attach();
    auto context = engine->execute(task());
    ASSERT_TRUE(gate->waitQueued());
    adapter->stopAccepting();
    EXPECT_FALSE(context->done());
    EXPECT_EQ(device->referencedBlocksNum(BlockTreeRefType::STORE), 1u);
    std::promise<void> done;
    auto               drained = done.get_future();
    adapter->drain([&] { done.set_value(); });
    EXPECT_EQ(drained.wait_for(std::chrono::milliseconds(30)), std::future_status::timeout);
    auto rejected = engine->execute(task());
    EXPECT_TRUE(rejected->done());
    EXPECT_FALSE(rejected->success());
    gate->open();
    ASSERT_EQ(drained.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    EXPECT_TRUE(context->done());
    EXPECT_FALSE(context->success());
    EXPECT_EQ(service->stats().reverse_rpcs, 1u);
    expectBaseline();
}
TEST_F(RemoteMockWritebackTest, FullSwaCpLayerScaleGeometryOnTwoLogicalRanksOneGpu) {
    auto full_config = makeTestGroupBase(defaultCacheGroupPolicy(CacheGroupType::FULL), {0, 1}, 64, 16, 4, 2);
    auto full_spec   = full_config.group.spec->clone();
    full_spec->cache_key_token_stride = 4;
    full_config.group.spec            = full_spec;
    auto swa_policy                   = defaultCacheGroupPolicy(CacheGroupType::SWA);
    swa_policy.sliding_window_size    = 8;
    auto swa_config                   = makeTestGroupBase(swa_policy, {2}, 48, 8, 4, 2);
    auto swa_spec                     = swa_config.group.spec->clone();
    swa_spec->cache_key_token_stride  = 4;
    swa_config.group.spec             = swa_spec;
    auto                                                       topology = makeTestTopology({full_config, swa_config});
    std::array<std::vector<GroupSetPtr>, 2>                    groups;
    std::array<std::shared_ptr<PerRankBlockTransferEngine>, 2> engines;
    std::array<std::shared_ptr<RemoteMockAdapter>, 2>          adapters;
    std::array<DeviceBlockPoolPtr, 2>                          fulls, swas;
    for (int rank = 0; rank < 2; ++rank) {
        fulls[rank]   = rank == 0 ? device : makeTestDevicePool({{64, 16}, {64, 16}}, 4, "cp_peer_full");
        swas[rank]    = makeTestDevicePool({{48, 8}}, 4, "cp_swa");
        groups[rank]  = {makeTestGroupSet(0, topology, {"group0"}, {fulls[rank]}, nullptr, nullptr, true),
                         makeTestGroupSet(1, topology, {"group1"}, {swas[rank]}, nullptr, nullptr, true)};
        engines[rank] = std::make_shared<PerRankBlockTransferEngine>(groups[rank]);
        for (const auto& group : groups[rank]) {
            const auto& pool = group->devicePools()[0];
            for (int layer = 0; layer < (group->groupSetId() == 0 ? 2 : 1); ++layer) {
                for (const auto& buffer : pool->convertIndexToBuffer(layer, block))
                    ASSERT_EQ(
                        cudaMemset(buffer.addr,
                                   group->groupSetId() == 0 ? (rank == 0 ? 0xa5 : 0xb6) : (rank == 0 ? 0xc7 : 0xd8),
                                   buffer.size_bytes),
                        cudaSuccess);
                for (const auto& buffer : pool->convertIndexToBuffer(layer, 2))
                    ASSERT_EQ(cudaMemset(buffer.addr, 0x37, buffer.size_bytes), cudaSuccess);
            }
        }
    }
    ASSERT_EQ(poolMalloc(*swas[0]), block);
    swas[0]->incRef(block);
    auto owner_ref = std::shared_ptr<void>(nullptr, [pool = swas[0], target = block](void*) { pool->decRef(target); });
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::array<std::future<std::shared_ptr<RemoteMockAdapter>>, 2> boot;
    for (int rank = 0; rank < 2; ++rank) {
        ParallelismConfig parallel;
        parallel.tp_size = parallel.world_size = 2;
        parallel.tp_rank = parallel.world_rank      = rank;
        parallel.prefill_cp_config.kv_cache_sharded = true;
        boot[rank]                                  = std::async(std::launch::async, [&, rank, parallel] {
            return RemoteMockAdapter::create(config, parallel, groups[rank], engines[rank]);
        });
    }
    for (int rank = 0; rank < 2; ++rank) {
        adapters[rank] = boot[rank].get();
        engines[rank]->setRemoteMockAdapter(adapters[rank], rank == 0);
        EXPECT_EQ(adapters[rank]->layouts()[0].cp_world_size, 2u);
        EXPECT_EQ(adapters[rank]->layouts()[1].cp_rank, rank);
        EXPECT_EQ(adapters[rank]->layouts()[1].attention_window, 8u);
        auto session = adapters[rank]->session();
        auto context = engines[rank]->execute(TransferTask({TransferDescriptor::deviceToRemote(0, session, 4, {block}),
                                                            TransferDescriptor::deviceToRemote(1, session, 5, {block})},
                                                           std::chrono::seconds(3)));
        context->waitDone();
        ASSERT_TRUE(context->success());
        std::promise<void> complete;
        auto               drained = complete.get_future();
        adapters[rank]->drain([&] { complete.set_value(); });
        ASSERT_EQ(drained.wait_for(std::chrono::seconds(5)), std::future_status::ready);
        EXPECT_EQ(adapters[rank]->writebackStats().acknowledged_payload_bytes, 240u);
        EXPECT_EQ(adapters[rank]->writebackStats().peak_retained_bytes, 192u);
        EXPECT_EQ(adapters[rank]->writebackStats().retained_bytes, 0u);
        std::vector<uint8_t> received;
        const auto           prefix = "rmw-" + std::to_string(session.session_id) + "-" + std::to_string(rank) + "-";
        {
            std::lock_guard<std::mutex> lock(observation->mutex);
            for (const auto& item : observation->copied)
                if (item.first.find(prefix) == 0 && item.second.size() > 1)
                    received.insert(received.end(), item.second.begin(), item.second.end());
        }
        std::vector<uint8_t> expected(240, 0);
        std::fill(expected.begin(), expected.begin() + 160, rank == 0 ? 0xa5 : 0xb6);
        std::fill(expected.begin() + 176, expected.begin() + 232, rank == 0 ? 0xc7 : 0xd8);
        const uint32_t full_crc = rank == 0 ? 0xe212414d : 0xcc298cca;
        const uint32_t swa_crc  = rank == 0 ? 0xaa5e9ab6 : 0xfad0270e;
        std::memcpy(expected.data() + 172, &full_crc, 4);
        std::memcpy(expected.data() + 236, &swa_crc, 4);
        EXPECT_EQ(received, expected);
        for (const auto& group : groups[rank]) {
            auto pool = group->devicePools()[0];
            EXPECT_EQ(pool->referencedBlocksNum(BlockTreeRefType::STORE), 0u);
            if (rank == 1) {
                EXPECT_FALSE(pool->isAllocated(block));
            }
            for (int layer = 0; layer < (group->groupSetId() == 0 ? 2 : 1); ++layer)
                for (const auto& buffer : pool->convertIndexToBuffer(layer, 2)) {
                    std::vector<uint8_t> actual(buffer.size_bytes);
                    ASSERT_EQ(cudaMemcpy(actual.data(), buffer.addr, actual.size(), cudaMemcpyDeviceToHost),
                              cudaSuccess);
                    EXPECT_EQ(actual, std::vector<uint8_t>(actual.size(), 0x37));
                }
        }
    }
    EXPECT_EQ(service->stats().write_payload_bytes, 482u);  // 2 probes + 2 * (176 FULL + 64 SWA)
    for (auto& item : adapters)
        item->stopAccepting();
}
TEST_F(RemoteMockWritebackTest, DrainIncludesQueuedContinuationBetweenHostRpcs) {
    attach();
    observation->hold_continuation = true;
    auto release = std::shared_ptr<void>(nullptr, [observed = observation](void*) { observed->open(); });
    auto context = engine->execute(task());
    {
        std::unique_lock<std::mutex> lock(observation->mutex);
        ASSERT_TRUE(
            observation->cv.wait_for(lock, std::chrono::seconds(3), [&] { return observation->continuation_entered; }));
    }
    std::promise<void> complete;
    auto               drained = complete.get_future();
    adapter->stopAccepting();
    adapter->drain([&] { complete.set_value(); });
    EXPECT_EQ(drained.wait_for(std::chrono::milliseconds(50)), std::future_status::timeout);
    EXPECT_FALSE(context->done());
    EXPECT_EQ(engine->remoteWritebackStats().retained_bytes, 192u);
    EXPECT_EQ(device->referencedBlocksNum(BlockTreeRefType::STORE), 1u);
    release.reset();
    ASSERT_EQ(drained.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    EXPECT_TRUE(context->done());
    EXPECT_FALSE(context->success());
    EXPECT_EQ(engine->remoteWritebackStats().failed, 1u);
    expectBaseline();
}
TEST_F(RemoteMockWritebackTest, DelayedAckDeadlineAndCallbackReentryRemainOneTerminal) {
    config.mock_remote_cache_write_delay_ms = 1500;
    config.mock_remote_cache_timeout_ms     = 5000;
    attach();
    const auto         before = service->stats().write_payload_bytes;
    std::promise<void> complete;
    auto               drained = complete.get_future();
    std::atomic<int>   terminals{0};
    ASSERT_TRUE(adapter->writeback(task(std::chrono::milliseconds(500)), [&](ErrorInfo error) {
        EXPECT_FALSE(error.ok());
        ++terminals;
        adapter->stopAccepting();
        adapter->drain([&] { complete.set_value(); });
    }));
    ASSERT_EQ(drained.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    EXPECT_EQ(terminals, 1);
    EXPECT_EQ(engine->remoteWritebackStats().completed, 0u);
    EXPECT_EQ(engine->remoteWritebackStats().failed, 1u);
    EXPECT_GT(service->stats().write_payload_bytes, before);  // Received data did not imply an ACK.
    expectBaseline();
    service->shutdown();  // Cancels/drains any delayed server ACK before destruction.
    EXPECT_EQ(terminals, 1);
}
TEST_F(RemoteMockWritebackTest, HelperZeroIdInlineRejectionFailsAnAcceptedCaptureOnce) {
    config.mock_remote_cache_max_inflight = 1;
    attach();
    observation->gated_key = "rmw-" + std::to_string(adapter->session().session_id) + "-0-2-0";
    auto release           = std::shared_ptr<void>(nullptr, [observed = observation](void*) { observed->open(); });
    auto bytes             = std::make_shared<std::vector<uint8_t>>(16, 0x6b);
    RemoteMockHostTransfer occupied;
    occupied.deadline       = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    occupied.retained_bytes = bytes->size();
    occupied.slices.push_back({0, 91, 0, std::shared_ptr<void>(bytes, bytes->data()), 16});
    std::promise<ErrorInfo> helper_terminal;
    auto                    helper_result = helper_terminal.get_future();
    ASSERT_NE(adapter->transferHost(occupied, [&](ErrorInfo error) { helper_terminal.set_value(std::move(error)); }),
              0u);
    {
        std::unique_lock<std::mutex> lock(observation->mutex);
        ASSERT_TRUE(observation->cv.wait_for(lock, std::chrono::seconds(3), [&] { return observation->entered; }));
    }
    // The writer still admits/captures; its helper call now returns ID zero and
    // invokes the real rejection callback inline because the sole credit is held.
    std::promise<ErrorInfo> terminal;
    auto                    result = terminal.get_future();
    std::atomic<int>        terminals{0};
    ASSERT_TRUE(adapter->writeback(task(), [&](ErrorInfo error) {
        ++terminals;
        terminal.set_value(std::move(error));
    }));
    ASSERT_EQ(result.wait_for(std::chrono::seconds(3)), std::future_status::ready);
    EXPECT_FALSE(result.get().ok());
    EXPECT_EQ(terminals, 1);
    EXPECT_EQ(engine->remoteWritebackStats().submitted, 1u);
    EXPECT_EQ(engine->remoteWritebackStats().failed, 1u);
    EXPECT_EQ(engine->remoteWritebackStats().rejected, 0u);
    EXPECT_EQ(service->stats().reverse_rpcs, 2u);
    release.reset();
    ASSERT_EQ(helper_result.wait_for(std::chrono::seconds(3)), std::future_status::ready);
    EXPECT_TRUE(helper_result.get().ok());
    drain();
    expectBaseline();
    auto again = engine->execute(task());
    again->waitDone();
    EXPECT_TRUE(again->success());
    drain();
    EXPECT_EQ(engine->remoteWritebackStats().submitted, 2u);
    EXPECT_EQ(engine->remoteWritebackStats().completed, 1u);
    EXPECT_EQ(engine->remoteWritebackStats().failed, 1u);
    expectBaseline();
}

TEST_F(RemoteMockWritebackTest, TerminalCanSubmitNextBatchWithOneWindowWhileDrainWaitsForReturn) {
    config.mock_remote_cache_max_inflight = 1;
    attach();
    auto                    open           = std::make_shared<std::promise<void>>();
    auto                    release        = open->get_future().share();
    auto                    ensure_release = std::shared_ptr<void>(nullptr, [open](void*) { open->set_value(); });
    std::promise<bool>      next_admission;
    auto                    next_admitted = next_admission.get_future();
    std::promise<ErrorInfo> second_terminal;
    auto                    second_result = second_terminal.get_future();
    ASSERT_TRUE(adapter->writeback(task(), [&](ErrorInfo error) {
        EXPECT_TRUE(error.ok());
        next_admission.set_value(adapter->writeback(
            task(), [&](ErrorInfo next_error) { second_terminal.set_value(std::move(next_error)); }));
        release.wait_for(std::chrono::seconds(10));
    }));
    ASSERT_EQ(next_admitted.wait_for(std::chrono::seconds(3)), std::future_status::ready);
    EXPECT_TRUE(next_admitted.get());
    EXPECT_EQ(engine->remoteWritebackStats().submitted, 2u);
    EXPECT_EQ(engine->remoteWritebackStats().rejected, 0u);
    EXPECT_EQ(engine->remoteWritebackStats().retained_bytes, 192u);
    EXPECT_EQ(engine->remoteWritebackStats().peak_retained_bytes, 192u);
    std::promise<void> complete;
    auto               drained = complete.get_future();
    adapter->drain([&] { complete.set_value(); });
    EXPECT_EQ(drained.wait_for(std::chrono::milliseconds(30)), std::future_status::timeout);
    ensure_release.reset();
    ASSERT_EQ(second_result.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    EXPECT_TRUE(second_result.get().ok());
    ASSERT_EQ(drained.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    EXPECT_EQ(engine->remoteWritebackStats().completed, 2u);
    expectBaseline();
}

TEST_F(RemoteMockWritebackTest, RegistrationAllocationFailuresPrecedeD2hDispatch) {
    group  = makeTestGroupSet(0, group->topologyPtr(), {"group0"}, {device});
    engine = std::make_shared<PerRankBlockTransferEngine>(std::vector<GroupSetPtr>{group});
    attach();
    {
        // The physical owner occupies the allocation-free inline slot. The
        // dependent terminal allocates one list node before any CUDA dispatch.
        auto fault   = armRemoteCopyFailure(1);
        auto cleanup = std::shared_ptr<void>(nullptr, [fault](void*) { retireRemoteCopyFailure(fault); });
        auto context = engine->execute(task());
        ASSERT_TRUE(fault->waitInjected());
        EXPECT_EQ(fault->registrations, 1);
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        EXPECT_EQ(fault->copies, 0) << "registration must precede physical dispatch";
        if (fault->copies) {
            EXPECT_FALSE(context->done());
            EXPECT_EQ(device->referencedBlocksNum(BlockTreeRefType::STORE), 1u);
            EXPECT_GT(engine->remoteWritebackStats().retained_bytes, 0u);
            EXPECT_EQ(fault->early_frees, 0);
        }
        fault->open();
        context->waitDone();
        EXPECT_FALSE(context->success());
        drain();
        expectBaseline();
    }
    auto next = engine->execute(task());
    next->waitDone();
    EXPECT_TRUE(next->success());
    drain();
    expectBaseline();
}
TEST_F(RemoteMockWritebackTest, PartialGenericEnqueueFailureDrainsBeforeTerminalAndWindowReuse) {
    group  = makeTestGroupSet(0, group->topologyPtr(), {"group0"}, {device});
    engine = std::make_shared<PerRankBlockTransferEngine>(std::vector<GroupSetPtr>{group});
    attach();
    auto fault   = armRemoteCopyFailure(0, true);
    auto cleanup = std::shared_ptr<void>(nullptr, [fault](void*) { retireRemoteCopyFailure(fault); });
    auto context = engine->execute(task());
    ASSERT_TRUE(fault->waitInjected());
    std::promise<void> done;
    auto               drained = done.get_future();
    adapter->drain([&] { done.set_value(); });
    EXPECT_EQ(drained.wait_for(std::chrono::milliseconds(100)), std::future_status::timeout);
    EXPECT_FALSE(context->done());
    EXPECT_EQ(device->referencedBlocksNum(BlockTreeRefType::STORE), 1u);
    EXPECT_GT(engine->remoteWritebackStats().retained_bytes, 0u);
    EXPECT_EQ(fault->early_frees, 0);
    fault->open();
    ASSERT_EQ(drained.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    EXPECT_FALSE(context->success());
    cleanup.reset();
    expectBaseline();
    auto next = engine->execute(task());
    next->waitDone();
    EXPECT_TRUE(next->success());
    drain();
}

TEST_F(RemoteMockWritebackTest, SliceAllocationFailureSettlesAcceptedWindowAndAllowsProgress) {
    attach();
    armRemoteAllocationFailure(RemoteAllocationSite::SLICE);
    auto               context = engine->execute(task());
    std::promise<void> done;
    auto               drained = done.get_future();
    adapter->drain([&] { done.set_value(); });
    const bool ready = drained.wait_for(std::chrono::seconds(5)) == std::future_status::ready;
    EXPECT_TRUE(remoteAllocationFailed());
    clearRemoteAllocationFailure();
    EXPECT_TRUE(ready) << "accepted writer must settle after slice allocation failure";
    if (!ready)
        std::_Exit(1);  // Preserve RED instead of hanging in destructor of the broken implementation.
    EXPECT_FALSE(context->success());
    EXPECT_EQ(engine->remoteWritebackStats().failed, 1u);
    EXPECT_EQ(engine->remoteWritebackStats().completed, 0u);
    expectBaseline();
    auto next = engine->execute(task());
    next->waitDone();
    EXPECT_TRUE(next->success());
    drain();
    expectBaseline();
}
TEST_F(RemoteMockWritebackTest, PhysicalCompletionPostAllocationFailureSettlesAfterCuda) {
    attach();
    observation->fail_after_copy = true;
    auto               context   = engine->execute(task());
    std::promise<void> done;
    auto               drained = done.get_future();
    adapter->drain([&] { done.set_value(); });
    const bool ready = drained.wait_for(std::chrono::seconds(5)) == std::future_status::ready;
    EXPECT_FALSE(observation->fail_after_copy);
    EXPECT_TRUE(ready);
    if (!ready)
        std::_Exit(1);
    EXPECT_FALSE(context->success());
    EXPECT_EQ(engine->remoteWritebackStats().failed, 1u);
    expectBaseline();
    auto next = engine->execute(task());
    next->waitDone();
    EXPECT_TRUE(next->success());
    drain();
    expectBaseline();
}
TEST_F(RemoteMockWritebackTest, HostAckPostFailureSettlesOnceAndReturnsCredit) {
    attach();
    const auto baseline         = service->stats();
    observation->fail_after_ack = true;
    auto               context  = engine->execute(task());
    std::promise<void> done;
    auto               drained = done.get_future();
    adapter->drain([&] { done.set_value(); });
    const bool ready = drained.wait_for(std::chrono::seconds(5)) == std::future_status::ready;
    EXPECT_FALSE(observation->fail_after_ack);
    EXPECT_TRUE(ready);
    if (!ready)
        std::_Exit(1);
    EXPECT_FALSE(context->success());
    EXPECT_EQ(engine->remoteWritebackStats().failed, 1u);
    const auto after = service->stats();
    EXPECT_EQ(after.failed_operations, baseline.failed_operations);
    EXPECT_GT(after.write_ack_ns, baseline.write_ack_ns);
    const auto received = after.write_payload_bytes - baseline.write_payload_bytes;
    EXPECT_GT(received, 0u);
    EXPECT_EQ(engine->remoteWritebackStats().acknowledged_payload_bytes, received);
    expectBaseline();
    auto next = engine->execute(task());
    next->waitDone();
    EXPECT_TRUE(next->success());
    drain();
    expectBaseline();
}
}  // namespace
}  // namespace rtp_llm
