#include "rtp_llm/cpp/cache/block_tree_cache/transfer/test/RemoteCopyFailureTestUtils.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/test/RemoteAllocationFailureTestUtils.h"
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <future>
#include <limits>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <cstring>
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/AlignedHostMemory.h"
#include "autil/NetUtil.h"
#include "rtp_llm/cpp/cache/block_tree_cache/remote_mock/RemoteMockAdapter.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferRequestConverter.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/test/PerRankBlockTransferEngineTestUtils.h"
#include "rtp_llm/cpp/disaggregate/cache_store/remote_mock/RemoteMockService.h"
#include "rtp_llm/cpp/disaggregate/cache_store/TcpServer.h"
#include "rtp_llm/cpp/disaggregate/cache_store/RequestBlockBufferStore.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"

namespace rtp_llm {
namespace {
struct BorrowGate {
    std::mutex              mutex;
    std::condition_variable cv;
    bool                    entered{false}, released{false};
    size_t                  lookups{0};
    std::string             key;
    void                    open() {
        std::lock_guard<std::mutex> lock(mutex);
        released = true;
        cv.notify_all();
    }
};
std::mutex                borrow_gate_mutex;
std::weak_ptr<BorrowGate> borrow_gate;
std::atomic<bool>         fail_adapter_physical_post{false};
}  // namespace

thread_local bool fail_id_reserve = false;
extern "C" void   realReserve(std::string*,
                              size_t) asm("__real__ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7reserveEm");
extern "C" void   wrapReserve(std::string*,
                              size_t) asm("__wrap__ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7reserveEm");
extern "C" void   wrapReserve(std::string* str, size_t size) {
    if (size == sizeof("rmw---") - 1 + 3 * (std::numeric_limits<uint64_t>::digits10 + 1)
        && std::exchange(fail_id_reserve, false))
        throw std::bad_alloc();
    realReserve(str, size);
}
thread_local bool fail_hook_construction = false;
extern "C" std::shared_ptr<UserBufferRegistration>
realScopedRegistration(RequestBlockBufferStore*, const std::vector<std::shared_ptr<BlockBuffer>>&) asm(
    "__real__ZN7rtp_llm23RequestBlockBufferStore20regUserBuffersScopedERKSt6vectorISt10shared_ptrINS_11BlockBufferEESaIS4_EE");
extern "C" std::shared_ptr<UserBufferRegistration>
wrapScopedRegistration(RequestBlockBufferStore*, const std::vector<std::shared_ptr<BlockBuffer>>&) asm(
    "__wrap__ZN7rtp_llm23RequestBlockBufferStore20regUserBuffersScopedERKSt6vectorISt10shared_ptrINS_11BlockBufferEESaIS4_EE");
extern "C" std::shared_ptr<UserBufferRegistration>
wrapScopedRegistration(RequestBlockBufferStore* store, const std::vector<std::shared_ptr<BlockBuffer>>& buffers) {
    auto lease = realScopedRegistration(store, buffers);
    if (lease && std::exchange(fail_hook_construction, false))
        block_transfer_engine_test::failNextRemoteAllocationOfSize(sizeof(std::shared_ptr<void>));
    return lease;
}
thread_local bool                  fail_watch_registration = false;
thread_local std::function<void()> cancel_before_publication;
thread_local bool                  check_accepted_post = false;
extern "C" bool realWatch(RequestBlockBufferStore*, const std::string&, RequestBlockBuffer::WatchFunc&&) asm(
    "__real__ZN7rtp_llm23RequestBlockBufferStore30setRequestBlockBufferWatchFuncERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEOSt8functionIFvbRKSt6vectorISt10shared_ptrINS_11BlockBufferEESaISD_EEEE");
extern "C" bool
wrapWatch(RequestBlockBufferStore* store, const std::string& id, RequestBlockBuffer::WatchFunc&& cb) asm(
    "__wrap__ZN7rtp_llm23RequestBlockBufferStore30setRequestBlockBufferWatchFuncERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEOSt8functionIFvbRKSt6vectorISt10shared_ptrINS_11BlockBufferEESaISD_EEEE");
extern "C" bool wrapWatch(RequestBlockBufferStore* store, const std::string& id, RequestBlockBuffer::WatchFunc&& cb) {
    if (std::exchange(fail_watch_registration, false))
        throw std::bad_alloc();
    return realWatch(store, id, std::move(cb));
}
extern "C" void realAddBlocks(RequestBlockBuffer*, const std::vector<std::shared_ptr<BlockBuffer>>&) asm(
    "__real__ZN7rtp_llm18RequestBlockBuffer9addBlocksERKSt6vectorISt10shared_ptrINS_11BlockBufferEESaIS4_EE");
extern "C" void wrapAddBlocks(RequestBlockBuffer*, const std::vector<std::shared_ptr<BlockBuffer>>&) asm(
    "__wrap__ZN7rtp_llm18RequestBlockBuffer9addBlocksERKSt6vectorISt10shared_ptrINS_11BlockBufferEESaIS4_EE");
extern "C" void wrapAddBlocks(RequestBlockBuffer* request, const std::vector<std::shared_ptr<BlockBuffer>>& blocks) {
    if (cancel_before_publication) {
        auto cancel = std::move(cancel_before_publication);
        cancel();
        throw std::bad_alloc();
    }
    realAddBlocks(request, blocks);
}
// Test-only link wrapper: preserve the actual registered lookup and returned
// BlockBuffer. Gate one unique key after lookup, before blockReadImpl copies it.
extern "C" std::shared_ptr<BlockBuffer> realFindUserBuffer(RequestBlockBufferStore*, const std::string&) asm(
    "__real__ZN7rtp_llm23RequestBlockBufferStore14findUserBufferERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE");
extern "C" std::shared_ptr<BlockBuffer> wrapFindUserBuffer(RequestBlockBufferStore*, const std::string&) asm(
    "__wrap__ZN7rtp_llm23RequestBlockBufferStore14findUserBufferERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE");
extern "C" std::shared_ptr<BlockBuffer> wrapFindUserBuffer(RequestBlockBufferStore* store, const std::string& key) {
    auto                        block = realFindUserBuffer(store, key);
    std::shared_ptr<BorrowGate> gate;
    {
        std::lock_guard<std::mutex> lock(borrow_gate_mutex);
        gate = borrow_gate.lock();
    }
    if (gate && block && key == gate->key) {
        std::unique_lock<std::mutex> lock(gate->mutex);
        // The first lookup builds the outgoing transfer metadata. The second
        // is the callback server's real blockReadImpl source acquisition.
        if (++gate->lookups == 2) {
            gate->entered = true;
            gate->cv.notify_all();
            gate->cv.wait_for(lock, std::chrono::seconds(10), [&] { return gate->released; });
        }
    }
    return block;
}
// This one-shot exception represents the existing completion queue's allocation
// failure. Arm only after the real CUDA stream has entered its test gate.
extern "C" bool realAdapterCompletion(BlockTreeTaskPool*, std::function<void()>) asm(
    "__real__ZN7rtp_llm17BlockTreeTaskPool16submitCompletionESt8functionIFvvEE");
extern "C" bool wrapAdapterCompletion(BlockTreeTaskPool*, std::function<void()>) asm(
    "__wrap__ZN7rtp_llm17BlockTreeTaskPool16submitCompletionESt8functionIFvvEE");
extern "C" bool wrapAdapterCompletion(BlockTreeTaskPool* pool, std::function<void()> task) {
    if (pool->thread_name_ == "RemoteMockCrc" && fail_adapter_physical_post.exchange(false))
        throw std::bad_alloc();
    const bool accepted = realAdapterCompletion(pool, std::move(task));
    if (accepted && std::exchange(check_accepted_post, false))
        block_transfer_engine_test::failNextRemoteAllocationOfSize(sizeof("completion worker stopped before drain"));
    return accepted;
}
namespace {
using namespace block_transfer_engine_test;
class FaultInjectingService: public RemoteMockService {
public:
    using RemoteMockService::RemoteMockService;
    std::atomic<bool>       corrupt{false};
    std::atomic<uint64_t>   read_calls{0};
    std::atomic<bool>       fail_next_read_post{false};
    std::mutex              read_mutex;
    std::condition_variable read_cv;
    bool                    waitRead() {
        std::unique_lock<std::mutex> lock(read_mutex);
        return read_cv.wait_for(lock, std::chrono::seconds(3), [&] { return read_calls.load() > 0; });
    }
    void load(google::protobuf::RpcController* controller,
              const CacheLoadRequest*          request,
              CacheLoadResponse*               response,
              google::protobuf::Closure*       done) override {
        {
            std::lock_guard<std::mutex> lock(read_mutex);
            ++read_calls;
        }
        read_cv.notify_all();
        struct Completion: google::protobuf::Closure {
            CacheLoadResponse*         response;
            google::protobuf::Closure* done;
            bool                       corrupt;
            bool                       arm_post_failure;
            void                       Run() override {
                if (corrupt && response->error_code() == EC_SUCCESS && response->blocks_size() > 0
                    && !response->blocks(0).content().empty()) {
                    std::string bytes(response->blocks(0).content().data(), response->blocks(0).content().size());
                    bytes[0] ^= 1;
                    response->mutable_blocks(0)->set_content(bytes);
                }
                if (arm_post_failure)
                    fail_adapter_physical_post = true;
                done->Run();
                delete this;
            }
        };
        auto completion              = new Completion;
        completion->response         = response;
        completion->done             = done;
        completion->corrupt          = corrupt.load();
        completion->arm_post_failure = fail_next_read_post.exchange(false);
        RemoteMockService::load(controller, request, response, completion);
    }
};
class RemoteMockAdapterTest: public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
        initRuntime(0);
        RemoteMockServiceOptions options;
        options.limits.total_bytes         = 1 << 20;
        options.limits.source_bytes        = 1 << 16;
        options.limits.write_slot_bytes    = 1024;
        options.limits.read_staging_bytes  = 1 << 16;
        options.limits.write_staging_bytes = 1 << 16;
        options.limits.max_rpc_bytes       = 64;
        restartService(options);
        config.mock_remote_cache                 = true;
        config.mock_remote_cache_callback_port   = autil::NetUtil::randomPort();
        config.mock_remote_cache_source_blocks   = 2;
        config.mock_remote_cache_staging_size_mb = 1;
        config.mock_remote_cache_timeout_ms      = 3000;
        resetGeometry(64, 16);
    }
    void restartService(const RemoteMockServiceOptions& options) {
        if (service)
            service->shutdown();
        server.reset();
        service = std::make_unique<FaultInjectingService>(options);
        ASSERT_TRUE(service->init());
        server = std::make_unique<TcpServer>();
        ASSERT_TRUE(server->init(1, 2, true, 32));
        ASSERT_TRUE(server->registerService(service.get()));
        ASSERT_TRUE(server->registerService(service->controlService()));
        const auto port = autil::NetUtil::randomPort();
        ASSERT_TRUE(server->start(port));
        config.mock_remote_cache_address = "127.0.0.1:" + std::to_string(port);
    }
    void resetGeometry(size_t kv_bytes, size_t scale_bytes) {
        if (device) {
            releasePoolBlock(*device, before);
            releasePoolBlock(*device, block);
            releasePoolBlock(*device, after);
        }
        auto policy                = defaultCacheGroupPolicy(CacheGroupType::FULL);
        policy.enable_prefix_reuse = true;
        device = makeTestDevicePool({{kv_bytes, scale_bytes}, {kv_bytes, scale_bytes}}, 4, "remote_adapter_gpu");
        group  = makeTestGroupSet(0,
                                 makeTestTopology({makeTestGroupBase(policy, {0, 1}, kv_bytes, scale_bytes)}),
                                  {"group0"},
                                  {device},
                                 nullptr,
                                 nullptr,
                                 true);
        engine = std::make_shared<PerRankBlockTransferEngine>(std::vector<GroupSetPtr>{group});
        before = poolMalloc(*device);
        block  = poolMalloc(*device);
        after  = poolMalloc(*device);
        ASSERT_GT(block, 0);
        for (const auto target : {before, block, after}) {
            for (int layer = 0; layer < 2; ++layer)
                for (const auto& buffer : device->convertIndexToBuffer(layer, target))
                    ASSERT_EQ(cudaMemset(buffer.addr, target == block ? 0xa5 : 0x37, buffer.size_bytes), cudaSuccess);
        }
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    }
    void TearDown() override {
        if (adapter) {
            adapter->stopAccepting();
            std::promise<void> drained;
            auto               future = drained.get_future();
            adapter->drain([&] { drained.set_value(); });
            ASSERT_EQ(future.wait_for(std::chrono::seconds(10)), std::future_status::ready);
            adapter.reset();
        }
        engine.reset();
        releasePoolBlock(*device, before);
        if (has_request_ref)
            device->decRef(block);
        else if (!released_target)
            releasePoolBlock(*device, block);
        releasePoolBlock(*device, after);
        service->shutdown();
        server.reset();
        service.reset();
    }
    RemoteMockLoadPlan plan(std::chrono::milliseconds timeout = std::chrono::seconds(3)) {
        return {{{adapter->freezeSource(0, 19, 3), {block}}}, std::chrono::steady_clock::now() + timeout, device};
    }
    ErrorInfo run(const RemoteMockLoadPlan& plan) {
        auto done   = std::make_shared<std::promise<ErrorInfo>>();
        auto result = done->get_future();
        adapter->load(plan, [done](ErrorInfo error) { done->set_value(std::move(error)); });
        if (result.wait_for(std::chrono::seconds(10)) != std::future_status::ready)
            return ErrorInfo(ErrorCode::DEADLINE_EXCEEDED, "test load completion timed out");
        return result.get();
    }
    void expectTarget(uint8_t value) {
        for (const auto guard : {before, after}) {
            for (int layer = 0; layer < 2; ++layer) {
                for (const auto& buffer : device->convertIndexToBuffer(layer, guard)) {
                    std::vector<uint8_t> bytes(buffer.size_bytes);
                    ASSERT_EQ(cudaMemcpy(bytes.data(), buffer.addr, bytes.size(), cudaMemcpyDeviceToHost), cudaSuccess);
                    EXPECT_EQ(bytes, std::vector<uint8_t>(bytes.size(), 0x37));
                }
            }
        }
        for (int layer = 0; layer < 2; ++layer) {
            for (const auto& buffer : device->convertIndexToBuffer(layer, block)) {
                std::vector<uint8_t> bytes(buffer.size_bytes, 0xff);
                ASSERT_EQ(cudaMemcpy(bytes.data(), buffer.addr, bytes.size(), cudaMemcpyDeviceToHost), cudaSuccess);
                EXPECT_EQ(bytes, std::vector<uint8_t>(bytes.size(), value));
            }
        }
    }
    KVCacheConfig                               config;
    std::unique_ptr<FaultInjectingService>      service;
    std::unique_ptr<TcpServer>                  server;
    DeviceBlockPoolPtr                          device;
    GroupSetPtr                                 group;
    std::shared_ptr<PerRankBlockTransferEngine> engine;
    std::shared_ptr<RemoteMockAdapter>          adapter;
    BlockIdxType                                before, block, after;
    bool                                        has_request_ref{false};
    bool                                        released_target{false};
};
TEST_F(RemoteMockAdapterTest, RemoteDescriptorRoundTripAndActualCudaPins) {
    config.mock_remote_cache_read_delay_ms = 100;
    adapter                                = RemoteMockAdapter::create(config, ParallelismConfig{}, {group}, engine);
    engine->setRemoteMockAdapter(adapter);
    device->incRef(block);  // Real request ownership, distinct from the temporary LOAD pin.
    has_request_ref = true;
    auto desc       = TransferDescriptor::remoteToDevice(0, adapter->freezeSource(0, 19, 3), {block});
    EXPECT_EQ(desc.source_tier, Tier::NONE);
    EXPECT_EQ(desc.node, nullptr);
    EXPECT_TRUE(desc.source_blocks.empty());
    ASSERT_TRUE(desc.isExecutable());
    MemoryOperationRequestPB request;
    ASSERT_TRUE(
        BlockTransferRequestConverter::encodeTransfer(request, TransferTask({desc}, std::chrono::seconds(3)), {group}));
    EXPECT_EQ(request.copy_direction(), MemoryOperationRequestPB::REMOTE2D);
    std::vector<TransferDescriptor> decoded;
    ASSERT_TRUE(
        BlockTransferRequestConverter::decodeTransfer(request, decoded, {group}, false, nullptr, adapter.get()));
    ASSERT_EQ(decoded.size(), 1u);
    EXPECT_EQ(decoded[0].remote_source->admission_offset, 19u);
    EXPECT_EQ(decoded[0].remote_source->logical_index, 3u);
    auto context = engine->execute(TransferTask(decoded, std::chrono::seconds(3)));
    ASSERT_TRUE(service->waitRead());
    EXPECT_EQ(device->referencedBlocksNum(BlockTreeRefType::LOAD), 1u);
    context->waitDone();
    ASSERT_TRUE(context->success()) << context->errorInfo().ToString();
    std::promise<void> drained;
    auto               future = drained.get_future();
    adapter->drain([&] { drained.set_value(); });
    ASSERT_EQ(future.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    EXPECT_EQ(device->referencedBlocksNum(BlockTreeRefType::LOAD), 0u);
    expectTarget(0);
}

TEST_F(RemoteMockAdapterTest, WorkerFreePoolCopiesActualCudaWithoutAllocationPins) {
    adapter = RemoteMockAdapter::create(config, ParallelismConfig{}, {group}, engine);
    engine->setRemoteMockAdapter(adapter, false);
    releasePoolBlock(*device, block);
    ASSERT_FALSE(device->isAllocated(block));
    released_target        = true;
    const auto free_before = device->freeBlocksNum();
    auto       descriptor  = TransferDescriptor::remoteToDevice(0, adapter->freezeSource(0, 19, 3), {block});
    auto       context     = engine->execute(TransferTask({descriptor}, std::chrono::seconds(3)));
    context->waitDone();
    EXPECT_TRUE(context->success()) << context->errorInfo().ToString();
    EXPECT_FALSE(device->isAllocated(block));
    EXPECT_EQ(device->freeBlocksNum(), free_before);
    EXPECT_EQ(device->referencedBlocksNum(BlockTreeRefType::LOAD), 0u);
    EXPECT_EQ(adapter->stats().read_payload_bytes, group->storageBytes());
    expectTarget(0);
}

TEST_F(RemoteMockAdapterTest, RemoteCodecRejectsStalePartialAndMalformedSource) {
    adapter                       = RemoteMockAdapter::create(config, ParallelismConfig{}, {group}, engine);
    auto                     desc = TransferDescriptor::remoteToDevice(0, adapter->freezeSource(0, 19, 3), {block});
    MemoryOperationRequestPB request;
    ASSERT_TRUE(
        BlockTransferRequestConverter::encodeTransfer(request, TransferTask({desc}, std::chrono::seconds(3)), {group}));
    for (int mutation = 0; mutation < 6; ++mutation) {
        auto  malformed = request;
        auto* item      = malformed.mutable_copy_items(0);
        if (mutation == 0)
            item->mutable_remote_mock_source()->set_generation(0);
        if (mutation == 1)
            item->mutable_remote_mock_source()->set_generation(adapter->session().generation + 1);
        if (mutation == 2)
            item->set_mem_block(1);
        if (mutation == 3)
            malformed.set_copy_direction(MemoryOperationRequestPB::H2D);
        if (mutation == 4)
            item->clear_remote_mock_source();
        if (mutation == 5)
            item->mutable_remote_mock_source()->set_admission_offset(UINT64_MAX);
        std::vector<TransferDescriptor> decoded;
        EXPECT_FALSE(
            BlockTransferRequestConverter::decodeTransfer(malformed, decoded, {group}, false, nullptr, adapter.get()));
        EXPECT_TRUE(decoded.empty());
    }
    desc.source_blocks = {1};
    EXPECT_FALSE(desc.isExecutable());
    desc.source_blocks.clear();
    desc.source_tier = Tier::HOST;
    EXPECT_FALSE(desc.isExecutable());
}

TEST_F(RemoteMockAdapterTest, RemotePerRankInlineRejectionSettlesAndReleasesPinsOnce) {
    adapter = RemoteMockAdapter::create(config, ParallelismConfig{}, {group}, engine);
    engine->setRemoteMockAdapter(adapter);
    device->incRef(block);
    has_request_ref = true;
    auto desc       = TransferDescriptor::remoteToDevice(0, adapter->freezeSource(0, 19, 3), {block});
    adapter->stopAccepting();
    auto   context   = engine->execute(TransferTask({desc}, std::chrono::seconds(3)));
    size_t completed = 0;
    context->onDone([&](TransferResult result) {
        ++completed;
        EXPECT_FALSE(result.ok());
    });
    EXPECT_TRUE(context->done());
    EXPECT_EQ(completed, 1u);
    EXPECT_EQ(device->referencedBlocksNum(BlockTreeRefType::LOAD), 0u);
    EXPECT_EQ(device->refCount(block), 1u);
    expectTarget(0xa5);
}

TEST_F(RemoteMockAdapterTest, ActualTcpSlicesCrcAndCudaScatterWithoutHostCache) {
    adapter = RemoteMockAdapter::create(config, ParallelismConfig{}, {group}, engine);
    ASSERT_NE(adapter, nullptr);
    RemoteMockLoadPlan plan;
    plan.targets.push_back({adapter->freezeSource(0, 19, 3), {block}});
    plan.deadline     = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    plan.target_owner = device;
    std::promise<ErrorInfo> completed;
    auto                    future = completed.get_future();
    adapter->load(plan, [&](ErrorInfo result) { completed.set_value(std::move(result)); });
    ASSERT_EQ(future.wait_for(std::chrono::seconds(10)), std::future_status::ready);
    EXPECT_TRUE(future.get().ok());
    for (int layer = 0; layer < 2; ++layer) {
        for (const auto& buffer : device->convertIndexToBuffer(layer, block)) {
            std::vector<uint8_t> bytes(buffer.size_bytes, 0xff);
            ASSERT_EQ(cudaMemcpy(bytes.data(), buffer.addr, bytes.size(), cudaMemcpyDeviceToHost), cudaSuccess);
            EXPECT_EQ(bytes, std::vector<uint8_t>(bytes.size(), 0));
        }
    }
    EXPECT_GE(service->stats().read_payload_bytes, group->storageBytes());
    EXPECT_EQ(group->hostPool(), nullptr);
    EXPECT_GE(service->read_calls, 3u);
    expectTarget(0);
}

TEST_F(RemoteMockAdapterTest, CallbackProbeUsesExistingHostTransfer) {
    adapter = RemoteMockAdapter::create(config, ParallelismConfig{}, {group}, engine);
    EXPECT_EQ(service->stats().reverse_rpcs, 1u);
    EXPECT_EQ(service->stats().write_payload_bytes, 1u);
    EXPECT_EQ(service->stats().failed_operations, 0u);
}
TEST_F(RemoteMockAdapterTest, ReadPlanAdmissionKeepsOriginalTerminal) {
    adapter      = RemoteMockAdapter::create(config, ParallelismConfig{}, {group}, engine);
    auto request = plan();
    request.targets.resize(3, request.targets.front());
    unsigned                        terminals = 0;
    RemoteMockAdapter::DoneCallback callback  = [&](ErrorInfo e) {
        ++terminals;
        EXPECT_FALSE(e.ok());
    };
    failNextRemoteAllocationOfSize(request.targets.size() * sizeof(RemoteMockLoadTarget));
    const auto id       = adapter->load(request, std::move(callback));
    const bool injected = remoteAllocationFailed();
    failNextRemoteAllocationOfSize(0);
    EXPECT_TRUE(injected);
    EXPECT_EQ(id, 0u);
    EXPECT_EQ(terminals, 1u);
    EXPECT_EQ(adapter->stats().read_submitted, 0u);
    EXPECT_EQ(adapter->stats().read_failed, 0u);
    EXPECT_TRUE(run(plan()).ok());
}
TEST_F(RemoteMockAdapterTest, AcceptedReadPostNeedsNoDiagnosticAllocation) {
    adapter                         = RemoteMockAdapter::create(config, ParallelismConfig{}, {group}, engine);
    auto                    request = plan();
    std::promise<ErrorInfo> done;
    auto                    result = done.get_future();
    check_accepted_post            = true;
    const auto id                  = adapter->load(request, [&](ErrorInfo error) { done.set_value(std::move(error)); });
    const auto injected            = remoteAllocationFailed();
    failNextRemoteAllocationOfSize(0);
    EXPECT_FALSE(check_accepted_post);
    EXPECT_FALSE(injected);
    ASSERT_NE(id, 0u);
    ASSERT_EQ(result.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    EXPECT_TRUE(result.get().ok());
    expectTarget(0);
}
TEST_F(RemoteMockAdapterTest, HostAdmissionAndRegisteredTaskFailuresReturnOnlyOwnedSlots) {
    config.mock_remote_cache_max_inflight   = 2;
    config.mock_remote_cache_write_delay_ms = 200;
    adapter                                 = RemoteMockAdapter::create(config, ParallelismConfig{}, {group}, engine);
    ASSERT_GE(adapter->credits().write_operations, 2u);
    auto                   bytes = std::make_shared<std::string>(16, 'q');
    RemoteMockHostTransfer request;
    request.slices.push_back({0, 0, 0, std::shared_ptr<void>(bytes, bytes->data()), 16});
    request.retained_bytes = 16;
    request.deadline       = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    auto first             = std::make_shared<std::promise<ErrorInfo>>();
    auto pending           = first->get_future();
    auto id                = adapter->transferHost(request, [first](ErrorInfo e) { first->set_value(std::move(e)); });
    ASSERT_NE(id, 0u);
    auto                            rejected = std::make_shared<std::atomic<unsigned>>(0);
    RemoteMockAdapter::DoneCallback reject   = [rejected](ErrorInfo e) {
        ++*rejected;
        EXPECT_FALSE(e.ok());
    };
    fail_id_reserve  = true;
    auto rejected_id = adapter->transferHost(request, reject);
    EXPECT_FALSE(std::exchange(fail_id_reserve, false));
    ASSERT_EQ(rejected_id, 0u);
    EXPECT_EQ(rejected->load(), 1u);
    EXPECT_EQ(pending.wait_for(std::chrono::milliseconds(0)), std::future_status::timeout);
    // The rejected admission did not consume an ID or the other active Host slot.
    cancel_before_publication = [local = adapter, id] { EXPECT_TRUE(local->cancel(id + 1)); };
    EXPECT_EQ(adapter->transferHost(request, reject), 0u);
    EXPECT_FALSE(std::exchange(cancel_before_publication, {}));
    EXPECT_EQ(rejected->load(), 2u);
    // The first done hook captures one shared owner. Allocation is rejected
    // after the Host buffer registry accepted the lease, before task exposure.
    fail_hook_construction = true;
    EXPECT_EQ(adapter->transferHost(request, reject), 0u);
    EXPECT_FALSE(fail_hook_construction);
    const bool hook_failed = remoteAllocationFailed();
    failNextRemoteAllocationOfSize(0);
    EXPECT_TRUE(hook_failed);
    EXPECT_EQ(rejected->load(), 3u);
    fail_watch_registration = true;
    std::promise<ErrorInfo> watch_done;
    auto                    watching = watch_done.get_future();
    adapter->transferHost(request, [&](ErrorInfo e) { watch_done.set_value(std::move(e)); });
    EXPECT_FALSE(fail_watch_registration);
    ASSERT_EQ(watching.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    EXPECT_FALSE(watching.get().ok());
    ASSERT_EQ(pending.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    EXPECT_TRUE(pending.get().ok());
    std::promise<ErrorInfo> next;
    auto                    ready = next.get_future();
    request.deadline              = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    ASSERT_NE(adapter->transferHost(request, [&](ErrorInfo e) { next.set_value(std::move(e)); }), 0u);
    ASSERT_EQ(ready.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    EXPECT_TRUE(ready.get().ok());
    EXPECT_EQ(rejected->load(), 3u);
}
TEST_F(RemoteMockAdapterTest, FrozenSourceRejectsStaleSessionBeforeIo) {
    adapter      = RemoteMockAdapter::create(config, ParallelismConfig{}, {group}, engine);
    auto request = plan();
    ++request.targets[0].source.session.generation;
    EXPECT_FALSE(run(request).ok());
    EXPECT_EQ(service->read_calls, 0u);
    EXPECT_EQ(adapter->stats().read_submitted, 0u);
    expectTarget(0xa5);
}
TEST_F(RemoteMockAdapterTest, CrcFailureLeavesGpuTargetUnchanged) {
    adapter          = RemoteMockAdapter::create(config, ParallelismConfig{}, {group}, engine);
    service->corrupt = true;
    EXPECT_FALSE(run(plan()).ok());
    EXPECT_GE(service->read_calls, 3u);
    EXPECT_EQ(adapter->stats().read_payload_bytes, group->storageBytes());
    EXPECT_EQ(adapter->stats().read_completed, 0u);
    EXPECT_EQ(adapter->stats().read_failed, 1u);
    expectTarget(0xa5);
}
TEST_F(RemoteMockAdapterTest, TcpLoadContinuationPostFailureReturnsWindowWithoutGpuScatter) {
    config.mock_remote_cache_max_inflight = 1;
    adapter                               = RemoteMockAdapter::create(config, ParallelismConfig{}, {group}, engine);
    auto               owner              = std::make_shared<int>(17);
    std::weak_ptr<int> weak               = owner;
    auto               request            = plan();
    request.target_owner                  = std::move(owner);
    service->fail_next_read_post          = true;
    std::atomic<unsigned> terminals{0};
    auto                  result = std::make_shared<std::promise<ErrorInfo>>();
    auto                  ready  = result->get_future();
    ASSERT_NE(adapter->load(request,
                            [&, result](ErrorInfo error) {
                                EXPECT_FALSE(weak.expired());
                                ++terminals;
                                result->set_value(std::move(error));
                            }),
              0u);
    request.target_owner.reset();
    auto drain   = std::make_shared<std::promise<void>>();
    auto drained = drain->get_future();
    adapter->drain([drain] { drain->set_value(); });
    if (drained.wait_for(std::chrono::seconds(5)) != std::future_status::ready) {
        ADD_FAILURE() << "accepted TCP continuation did not retire its read window";
        std::_Exit(1);
    }
    ASSERT_EQ(ready.wait_for(std::chrono::seconds(1)), std::future_status::ready);
    EXPECT_FALSE(ready.get().ok());
    EXPECT_FALSE(fail_adapter_physical_post.load());
    EXPECT_FALSE(service->fail_next_read_post.load());
    EXPECT_GT(adapter->stats().read_payload_bytes, 0u);
    EXPECT_EQ(adapter->stats().read_failed, 1u);
    EXPECT_EQ(terminals, 1u);
    EXPECT_TRUE(weak.expired());
    expectTarget(0xa5);
    EXPECT_TRUE(run(plan()).ok());
    expectTarget(0);
}
TEST_F(RemoteMockAdapterTest, StopDrainsSocketAndKeepsTargetOwnerUntilTerminal) {
    config.mock_remote_cache_read_delay_ms = 100;
    adapter                                = RemoteMockAdapter::create(config, ParallelismConfig{}, {group}, engine);
    auto               target_owner        = std::make_shared<int>(17);
    std::weak_ptr<int> weak                = target_owner;
    auto               request             = plan();
    request.target_owner                   = target_owner;
    std::atomic<int> terminals{0};
    auto             done      = std::make_shared<std::promise<ErrorInfo>>();
    auto             result    = done->get_future();
    const auto       operation = adapter->load(request, [&, done](ErrorInfo error) {
        EXPECT_FALSE(weak.expired());
        ++terminals;
        done->set_value(std::move(error));
    });
    ASSERT_NE(operation, 0u);
    request.target_owner.reset();
    target_owner.reset();
    EXPECT_FALSE(weak.expired());
    ASSERT_TRUE(service->waitRead());
    adapter->stopAccepting();
    auto drain   = std::make_shared<std::promise<void>>();
    auto drained = drain->get_future();
    adapter->drain([drain] { drain->set_value(); });
    ASSERT_EQ(drained.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    ASSERT_EQ(result.wait_for(std::chrono::seconds(1)), std::future_status::ready);
    EXPECT_FALSE(result.get().ok());
    EXPECT_EQ(terminals, 1);
    EXPECT_TRUE(weak.expired());
    EXPECT_FALSE(adapter->cancel(operation));
    EXPECT_FALSE(run(plan()).ok());
    expectTarget(0xa5);
}
TEST_F(RemoteMockAdapterTest, ExplicitCancelSettlesOnce) {
    config.mock_remote_cache_read_delay_ms = 50;
    adapter                                = RemoteMockAdapter::create(config, ParallelismConfig{}, {group}, engine);
    auto             done                  = std::make_shared<std::promise<ErrorInfo>>();
    auto             result                = done->get_future();
    std::atomic<int> terminals{0};
    const auto       operation = adapter->load(plan(), [&, done](ErrorInfo error) {
        ++terminals;
        done->set_value(std::move(error));
    });
    ASSERT_NE(operation, 0u);
    EXPECT_TRUE(adapter->cancel(operation));
    ASSERT_EQ(result.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    EXPECT_FALSE(result.get().ok());
    EXPECT_EQ(terminals, 1);
}
TEST_F(RemoteMockAdapterTest, InvalidScalarRankPortAndCapacityFailBeforeRegistration) {
    auto invalid                         = config;
    invalid.mock_remote_cache_timeout_ms = 10;
    EXPECT_THROW(RemoteMockAdapter::create(invalid, ParallelismConfig{}, {group}, engine), std::invalid_argument);
    ParallelismConfig rank;
    rank.tp_size = rank.world_size = 2;
    rank.tp_rank = rank.world_rank          = 1;
    invalid                                 = config;
    invalid.mock_remote_cache_callback_port = 65535;
    EXPECT_THROW(RemoteMockAdapter::create(invalid, rank, {group}, engine), std::invalid_argument);
    rank.tp_size = rank.world_size = 9;
    rank.tp_rank = rank.world_rank = 0;
    EXPECT_THROW(RemoteMockAdapter::create(config, rank, {group}, engine), std::invalid_argument);
    rank            = ParallelismConfig{};
    rank.world_rank = std::numeric_limits<int64_t>::max();
    EXPECT_THROW(RemoteMockAdapter::create(config, rank, {group}, engine), std::invalid_argument);
    EXPECT_EQ(service->read_calls, 0u);
    EXPECT_EQ(service->stats().reverse_rpcs, 0u);
}
TEST_F(RemoteMockAdapterTest, ScopedHostTransferUsesActualReverseCopyAndDrains) {
    adapter = RemoteMockAdapter::create(config, ParallelismConfig{}, {group}, engine);
    for (uint64_t index = 0; index < 3; ++index) {
        auto                       bytes = std::make_shared<std::string>(16, char(0x31 + index));
        std::weak_ptr<std::string> weak  = bytes;
        RemoteMockHostTransfer     request;
        request.slices.push_back({0, index, 0, std::shared_ptr<void>(bytes, bytes->data()), 16});
        request.deadline       = std::chrono::steady_clock::now() + std::chrono::seconds(3);
        request.retained_bytes = 16;
        auto done              = std::make_shared<std::promise<ErrorInfo>>();
        auto result            = done->get_future();
        ASSERT_NE(adapter->transferHost(request, [done](ErrorInfo error) { done->set_value(std::move(error)); }), 0u);
        request.slices.clear();
        bytes.reset();
        ASSERT_EQ(result.wait_for(std::chrono::seconds(5)), std::future_status::ready);
        EXPECT_TRUE(result.get().ok());
        auto safe    = std::make_shared<std::promise<void>>();
        auto drained = safe->get_future();
        adapter->drain([safe] { safe->set_value(); });
        ASSERT_EQ(drained.wait_for(std::chrono::seconds(5)), std::future_status::ready);
        EXPECT_TRUE(weak.expired());
    }
    EXPECT_EQ(service->stats().reverse_rpcs, 4u);
    EXPECT_EQ(service->stats().write_payload_bytes, 49u);
}
TEST_F(RemoteMockAdapterTest, HostBorrowAfterOuterTimeoutRetainsAdmissionAndDrain) {
    config.mock_remote_cache_max_inflight = 1;
    adapter                               = RemoteMockAdapter::create(config, ParallelismConfig{}, {group}, engine);
    auto gate                             = std::make_shared<BorrowGate>();
    // Probe consumed sequence1. Gate the second slice so the actual blockRead
    // sources vector also retains the first slice from this same backing.
    gate->key = "rmw-" + std::to_string(adapter->session().session_id) + "-0-2-1";
    {
        std::lock_guard<std::mutex> lock(borrow_gate_mutex);
        borrow_gate = gate;
    }
    const auto                 release_gate = std::shared_ptr<void>(nullptr, [gate](void*) { gate->open(); });
    const auto                 retained     = adapter->credits().local_write_bytes - (64 << 10);
    auto                       backing      = std::make_shared<std::string>(retained, 'q');
    std::weak_ptr<std::string> weak         = backing;
    RemoteMockHostTransfer     request;
    request.slices         = {{0, 0, 0, std::shared_ptr<void>(backing, backing->data()), 16},
                              {0, 0, 16, std::shared_ptr<void>(backing, backing->data() + 16), 16}};
    request.retained_bytes = retained;
    request.deadline       = std::chrono::steady_clock::now() + std::chrono::milliseconds(200);
    auto       done        = std::make_shared<std::promise<ErrorInfo>>();
    auto       result      = done->get_future();
    auto       terminals   = std::make_shared<std::atomic<int>>(0);
    const auto operation   = adapter->transferHost(request, [done, terminals](ErrorInfo error) {
        ++*terminals;
        done->set_value(std::move(error));
    });
    ASSERT_NE(operation, 0u);
    {
        std::unique_lock<std::mutex> lock(gate->mutex);
        ASSERT_TRUE(gate->cv.wait_for(lock, std::chrono::seconds(3), [&] { return gate->entered; }));
    }
    EXPECT_EQ(service->stats().reverse_rpcs, 2u);
    request.slices.clear();
    backing.reset();
    EXPECT_FALSE(weak.expired());
    EXPECT_EQ(result.wait_for(std::chrono::milliseconds(1000)), std::future_status::timeout);
    auto safe    = std::make_shared<std::promise<void>>();
    auto drained = safe->get_future();
    adapter->drain([safe] { safe->set_value(); });
    EXPECT_EQ(drained.wait_for(std::chrono::milliseconds(100)), std::future_status::timeout);

    auto replacement = std::make_shared<std::string>(retained, 'r');
    request.slices   = {{0, 1, 0, std::shared_ptr<void>(replacement, replacement->data()), 16}};
    request.deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    auto rejected    = std::make_shared<std::promise<ErrorInfo>>();
    auto rejection   = rejected->get_future();
    EXPECT_EQ(adapter->transferHost(request, [rejected](ErrorInfo error) { rejected->set_value(std::move(error)); }),
              0u);
    ASSERT_EQ(rejection.wait_for(std::chrono::seconds(3)), std::future_status::ready);
    EXPECT_FALSE(rejection.get().ok());
    EXPECT_FALSE(weak.expired());
    EXPECT_EQ(drained.wait_for(std::chrono::milliseconds(50)), std::future_status::timeout);

    gate->open();
    ASSERT_EQ(result.wait_for(std::chrono::seconds(3)), std::future_status::ready);
    EXPECT_FALSE(result.get().ok());
    ASSERT_EQ(drained.wait_for(std::chrono::seconds(3)), std::future_status::ready);
    EXPECT_TRUE(weak.expired());
    EXPECT_EQ(*terminals, 1);
    EXPECT_FALSE(adapter->cancel(operation));

    // The same single operation/slot allowance is reusable after the last alias.
    auto retry       = std::make_shared<std::promise<ErrorInfo>>();
    auto retried     = retry->get_future();
    request.deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    ASSERT_NE(adapter->transferHost(request, [retry](ErrorInfo error) { retry->set_value(std::move(error)); }), 0u);
    ASSERT_EQ(retried.wait_for(std::chrono::seconds(3)), std::future_status::ready);
    EXPECT_TRUE(retried.get().ok());
}
TEST_F(RemoteMockAdapterTest, RemotePerRankTimeoutRetainsActualLoadPinThroughCuda) {
    // CRC-enabled groups use CrcTransferService directly and bypass plain cudaMemcpyAsync hooks.
    group   = makeTestGroupSet(0, group->topologyPtr(), {"group0"}, {device});
    engine  = std::make_shared<PerRankBlockTransferEngine>(std::vector<GroupSetPtr>{group});
    adapter = RemoteMockAdapter::create(config, ParallelismConfig{}, {group}, engine);
    engine->setRemoteMockAdapter(adapter);
    device->incRef(block);
    has_request_ref      = true;
    auto       gate      = armRemoteCopyFailure(0);
    auto       cleanup   = std::shared_ptr<void>(nullptr, [gate](void*) { retireRemoteCopyFailure(gate); });
    const auto desc    = TransferDescriptor::remoteToDevice(0, adapter->freezeSource(0, 19, 3), {block});
    auto       context = engine->execute(TransferTask({desc}, std::chrono::milliseconds(1000)));
    ASSERT_TRUE(gate->waitQueued());
    std::promise<TransferResult> terminal;
    auto                         result = terminal.get_future();
    context->onDone([&](TransferResult value) { terminal.set_value(std::move(value)); });
    EXPECT_EQ(result.wait_for(std::chrono::milliseconds(1100)), std::future_status::timeout);
    EXPECT_EQ(device->referencedBlocksNum(BlockTreeRefType::LOAD), 1u);
    std::promise<void> drained;
    auto               done = drained.get_future();
    adapter->stopAccepting();
    adapter->drain([&] { drained.set_value(); });
    EXPECT_EQ(done.wait_for(std::chrono::milliseconds(20)), std::future_status::timeout);
    gate->open();
    ASSERT_EQ(result.wait_for(std::chrono::seconds(3)), std::future_status::ready);
    EXPECT_FALSE(result.get().ok());
    ASSERT_EQ(done.wait_for(std::chrono::seconds(3)), std::future_status::ready);
    EXPECT_EQ(device->referencedBlocksNum(BlockTreeRefType::LOAD), 0u);
    EXPECT_EQ(device->refCount(block), 1u);
    expectTarget(0);
}

TEST_F(RemoteMockAdapterTest, TimeoutDuringCopyKeepsBuffers) {
    group                      = makeTestGroupSet(0, group->topologyPtr(), {"group0"}, {device});
    engine                     = std::make_shared<PerRankBlockTransferEngine>(std::vector<GroupSetPtr>{group});
    auto gate                  = armRemoteCopyFailure(0);
    auto cleanup               = std::shared_ptr<void>(nullptr, [gate](void*) { retireRemoteCopyFailure(gate); });
    adapter                    = RemoteMockAdapter::create(config, ParallelismConfig{}, {group}, engine);
    auto               owner   = std::make_shared<int>(23);
    std::weak_ptr<int> weak    = owner;
    auto               request = plan(std::chrono::milliseconds(150));
    request.target_owner       = owner;
    auto done                  = std::make_shared<std::promise<ErrorInfo>>();
    auto result                = done->get_future();
    ASSERT_NE(adapter->load(request, [done](ErrorInfo error) { done->set_value(std::move(error)); }), 0u);
    request.target_owner.reset();
    owner.reset();
    ASSERT_TRUE(gate->waitQueued());
    EXPECT_EQ(result.wait_for(std::chrono::milliseconds(200)), std::future_status::timeout);
    EXPECT_FALSE(weak.expired());
    gate->open();
    ASSERT_EQ(result.wait_for(std::chrono::seconds(3)), std::future_status::ready);
    EXPECT_FALSE(result.get().ok());
    auto safe    = std::make_shared<std::promise<void>>();
    auto drained = safe->get_future();
    adapter->drain([safe] { safe->set_value(); });
    ASSERT_EQ(drained.wait_for(std::chrono::seconds(3)), std::future_status::ready);
    EXPECT_TRUE(weak.expired());
    EXPECT_EQ(adapter->stats().read_completed, 0u);
    EXPECT_EQ(adapter->stats().read_payload_bytes, group->storageBytes());
    expectTarget(0);
}
TEST_F(RemoteMockAdapterTest, CallbackPortCollisionRollsBackSession) {
    auto invalid = config;
    invalid.mock_remote_cache_callback_port =
        std::stoul(config.mock_remote_cache_address.substr(config.mock_remote_cache_address.rfind(':') + 1));
    EXPECT_THROW(RemoteMockAdapter::create(invalid, ParallelismConfig{}, {group}, engine), std::invalid_argument);
    adapter = RemoteMockAdapter::create(config, ParallelismConfig{}, {group}, engine);
    EXPECT_GT(adapter->session().generation, 1u);
}
TEST_F(RemoteMockAdapterTest, InsufficientLocalMetadataReserveFailsAtStartup) {
    RemoteMockServiceOptions options;
    options.limits.total_bytes        = 4 << 20;
    options.limits.source_bytes       = 2 << 20;
    options.limits.write_slot_bytes   = 1024;
    options.limits.read_staging_bytes = options.limits.write_staging_bytes = 1 << 16;
    options.limits.max_rpc_bytes                                           = 64;
    restartService(options);
    resetGeometry(230 * 1024, 16);
    config.mock_remote_cache_max_inflight = 1;
    EXPECT_THROW(RemoteMockAdapter::create(config, ParallelismConfig{}, {group}, engine), std::invalid_argument);
    EXPECT_EQ(service->stats().reverse_rpcs, 0u);
}
TEST_F(RemoteMockAdapterTest, HostTransferChargesWholeRetainedAllocation) {
    adapter                      = RemoteMockAdapter::create(config, ParallelismConfig{}, {group}, engine);
    auto                   bytes = std::make_shared<std::string>(adapter->credits().local_write_bytes, 'x');
    RemoteMockHostTransfer request;
    request.slices.push_back({0, 0, 0, std::shared_ptr<void>(bytes, bytes->data()), 1});
    request.retained_bytes = bytes->size();
    request.deadline       = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    int callbacks          = 0;
    EXPECT_EQ(adapter->transferHost(request,
                                    [&](ErrorInfo error) {
                                        ++callbacks;
                                        EXPECT_FALSE(error.ok());
                                    }),
              0u);
    EXPECT_EQ(callbacks, 1);
    EXPECT_EQ(service->stats().reverse_rpcs, 1u);
}
TEST_F(RemoteMockAdapterTest, RecordLargerThanDefaultRpcCapReassemblesBeforeCuda) {
    RemoteMockServiceOptions options;
    options.limits.total_bytes        = 128 << 20;
    options.limits.source_bytes       = 64 << 20;
    options.limits.write_slot_bytes   = 1024;
    options.limits.read_staging_bytes = options.limits.write_staging_bytes = 8 << 20;
    options.limits.max_rpc_bytes                                           = 1 << 20;
    restartService(options);
    resetGeometry(8 << 20, 512 << 10);
    config.mock_remote_cache_staging_size_mb = 64;
    config.mock_remote_cache_timeout_ms      = 10000;
    adapter                                  = RemoteMockAdapter::create(config, ParallelismConfig{}, {group}, engine);
    ASSERT_GT(group->storageBytes(), 16u << 20);
    EXPECT_TRUE(run(plan(std::chrono::seconds(10))).ok());
    EXPECT_GE(service->read_calls, 18u);
    EXPECT_EQ(adapter->stats().read_payload_bytes, group->storageBytes());
    expectTarget(0);
}
TEST_F(RemoteMockAdapterTest, ReadCreditsRejectWithoutRetainingSecondTarget) {
    config.mock_remote_cache_max_inflight  = 1;
    config.mock_remote_cache_read_delay_ms = 50;
    adapter                                = RemoteMockAdapter::create(config, ParallelismConfig{}, {group}, engine);
    auto done                              = std::make_shared<std::promise<ErrorInfo>>();
    auto result                            = done->get_future();
    ASSERT_NE(adapter->load(plan(), [done](ErrorInfo error) { done->set_value(std::move(error)); }), 0u);
    ASSERT_TRUE(service->waitRead());
    int callbacks = 0;
    EXPECT_EQ(adapter->load(plan(),
                            [&](ErrorInfo error) {
                                ++callbacks;
                                EXPECT_FALSE(error.ok());
                            }),
              0u);
    EXPECT_EQ(callbacks, 1);
    ASSERT_EQ(result.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    EXPECT_TRUE(result.get().ok());
    EXPECT_EQ(adapter->stats().read_submitted, 1u);
}
TEST_F(RemoteMockAdapterTest, TerminalCallbackCanReenterCancelAndDrain) {
    config.mock_remote_cache_read_delay_ms = 25;
    adapter                                = RemoteMockAdapter::create(config, ParallelismConfig{}, {group}, engine);
    auto                  safe             = std::make_shared<std::promise<void>>();
    auto                  drained          = safe->get_future();
    std::atomic<int>      terminals{0};
    std::atomic<uint64_t> operation{0};
    operation = adapter->load(plan(), [&](ErrorInfo error) {
        EXPECT_TRUE(error.ok());
        ++terminals;
        EXPECT_TRUE(adapter->cancel(operation));
        adapter->drain([safe] { safe->set_value(); });
    });
    ASSERT_NE(operation, 0u);
    ASSERT_EQ(drained.wait_for(std::chrono::seconds(3)), std::future_status::ready);
    EXPECT_EQ(terminals, 1);
    EXPECT_FALSE(adapter->cancel(operation));
}
TEST_F(RemoteMockAdapterTest, ServiceShutdownDuringReadSettlesOnceWithoutScatter) {
    config.mock_remote_cache_read_delay_ms = 100;
    adapter                                = RemoteMockAdapter::create(config, ParallelismConfig{}, {group}, engine);
    auto             done                  = std::make_shared<std::promise<ErrorInfo>>();
    auto             result                = done->get_future();
    std::atomic<int> terminals{0};
    ASSERT_NE(adapter->load(plan(),
                            [&, done](ErrorInfo error) {
                                ++terminals;
                                done->set_value(std::move(error));
                            }),
              0u);
    ASSERT_TRUE(service->waitRead());
    service->shutdown();
    ASSERT_EQ(result.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    EXPECT_FALSE(result.get().ok());
    EXPECT_EQ(terminals, 1);
    expectTarget(0xa5);
}
TEST_F(RemoteMockAdapterTest, ConcurrentRankBootstrapSimulatedOnOneGpu) {
    ParallelismConfig rank0;
    rank0.tp_size = rank0.world_size         = 2;
    rank0.prefill_cp_config.kv_cache_sharded = true;
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
    auto topology                     = makeTestTopology({full_config, swa_config});
    auto swa_device                   = makeTestDevicePool({{48, 8}}, 4, "remote_swa_cp_gpu");
    auto swa_guard                    = poolMalloc(*swa_device);
    auto swa_block                    = poolMalloc(*swa_device);
    for (const auto& buffer : swa_device->convertIndexToBuffer(0, swa_guard))
        ASSERT_EQ(cudaMemset(buffer.addr, 0x4b, buffer.size_bytes), cudaSuccess);
    for (const auto& buffer : swa_device->convertIndexToBuffer(0, swa_block))
        ASSERT_EQ(cudaMemset(buffer.addr, 0x92, buffer.size_bytes), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    group         = makeTestGroupSet(0, topology, {"group0"}, {device}, nullptr, nullptr, true);
    auto swa      = makeTestGroupSet(1, topology, {"group1"}, {swa_device}, nullptr, nullptr, true);
    engine        = std::make_shared<PerRankBlockTransferEngine>(std::vector<GroupSetPtr>{group, swa});
    auto rank1    = rank0;
    rank1.tp_rank = rank1.world_rank = 1;
    const auto callback_base         = config.mock_remote_cache_callback_port;
    ASSERT_LT(callback_base, 65535);
    auto first =
        std::async(std::launch::async, [&] { return RemoteMockAdapter::create(config, rank0, {group, swa}, engine); });
    EXPECT_EQ(first.wait_for(std::chrono::milliseconds(25)), std::future_status::timeout);
    auto second =
        std::async(std::launch::async, [&] { return RemoteMockAdapter::create(config, rank1, {group, swa}, engine); });
    ASSERT_EQ(first.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    ASSERT_EQ(second.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    adapter   = first.get();
    auto peer = second.get();
    EXPECT_TRUE(adapter->session() == peer->session());
    EXPECT_NE(adapter->freezeSource(0, 19, 3).rank, peer->freezeSource(0, 19, 3).rank);
    EXPECT_EQ(adapter->credits().service_read_bytes + peer->credits().service_read_bytes, 1u << 16);
    EXPECT_EQ(adapter->credits().write_slots + peer->credits().write_slots, 8u);
    EXPECT_LE(adapter->credits().first_write_slot + adapter->credits().write_slots, peer->credits().first_write_slot);
    EXPECT_EQ(service->stats().reverse_rpcs, 2u);
    EXPECT_EQ(adapter->layouts()[0].tokens_per_block, 4u);
    EXPECT_EQ(adapter->layouts()[1].attention_window, 8u);
    EXPECT_EQ(adapter->layouts()[0].cp_world_size, 2u);
    EXPECT_EQ(peer->layouts()[1].cp_rank, 1u);
    EXPECT_EQ(adapter->layouts()[0].payload_bytes, 160u);
    EXPECT_EQ(adapter->layouts()[1].payload_bytes, 56u);
    EXPECT_EQ(adapter->layouts()[1].member_offsets, (std::vector<uint64_t>{0, 48}));
    auto request = plan();
    request.targets.push_back({adapter->freezeSource(1, 19, 7), {swa_block}});
    request.target_owner =
        std::make_shared<std::vector<DeviceBlockPoolPtr>>(std::vector<DeviceBlockPoolPtr>{device, swa_device});
    EXPECT_TRUE(run(request).ok());
    for (const auto target : {swa_guard, swa_block}) {
        for (const auto& buffer : swa_device->convertIndexToBuffer(0, target)) {
            std::vector<uint8_t> bytes(buffer.size_bytes);
            ASSERT_EQ(cudaMemcpy(bytes.data(), buffer.addr, bytes.size(), cudaMemcpyDeviceToHost), cudaSuccess);
            EXPECT_EQ(bytes, std::vector<uint8_t>(bytes.size(), target == swa_guard ? 0x4b : 0));
        }
    }
    expectTarget(0);
    auto done       = std::make_shared<std::promise<ErrorInfo>>();
    auto result     = done->get_future();
    request.targets = {{peer->freezeSource(1, 19, 7), {swa_block}}};
    ASSERT_NE(peer->load(request, [done](ErrorInfo error) { done->set_value(std::move(error)); }), 0u);
    ASSERT_EQ(result.wait_for(std::chrono::seconds(3)), std::future_status::ready);
    EXPECT_TRUE(result.get().ok());
    peer->stopAccepting();
    peer.reset();
    releasePoolBlock(*swa_device, swa_guard);
    releasePoolBlock(*swa_device, swa_block);
}
TEST_F(RemoteMockAdapterTest, MaximumWriteSliceIncludesAlignedBackingInCredit) {
    RemoteMockServiceOptions options;
    options.limits.total_bytes        = 32 << 20;
    options.limits.source_bytes       = 4 << 20;
    options.limits.write_slot_bytes   = 1 << 20;
    options.limits.read_staging_bytes = options.limits.write_staging_bytes = 4 << 20;
    options.limits.max_rpc_bytes                                           = 1 << 20;
    restartService(options);
    resetGeometry(128 << 10, 16);
    config.mock_remote_cache_max_inflight = 1;
    adapter                               = RemoteMockAdapter::create(config, ParallelismConfig{}, {group}, engine);
    auto bytes = std::make_shared<AlignedHostMemory>(group->storageBytes(), 16, "writer_credit_boundary");
    std::memset(bytes->data(), 0x5a, group->storageBytes());
    const auto length = adapter->credits().max_write_slice_bytes;
    ASSERT_GT(length, 64u);
    ASSERT_LT(length, group->storageBytes());
    RemoteMockHostTransfer request;
    request.deadline       = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    request.retained_bytes = group->storageBytes() + 16;
    request.slices.push_back({0, 2, 0, std::shared_ptr<void>(bytes, bytes->data()), length});
    std::promise<ErrorInfo> complete;
    auto                    result = complete.get_future();
    adapter->transferHost(request, [&](ErrorInfo error) { complete.set_value(std::move(error)); });
    ASSERT_EQ(result.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    const auto error = result.get();
    EXPECT_TRUE(error.ok()) << error.ToString();
    EXPECT_EQ(service->stats().write_payload_bytes, 1u + length);
}

TEST_F(RemoteMockAdapterTest, HostCallbackCanStartNextTransferBeforeReturningButDrainStillWaits) {
    config.mock_remote_cache_max_inflight = 1;
    adapter                               = RemoteMockAdapter::create(config, ParallelismConfig{}, {group}, engine);
    ASSERT_GT(adapter->credits().write_slots, adapter->credits().write_operations);
    auto                   bytes = std::make_shared<std::vector<uint8_t>>(16, 0x6d);
    RemoteMockHostTransfer request;
    request.deadline       = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    request.retained_bytes = bytes->size();
    request.slices.push_back({0, 19, 0, std::shared_ptr<void>(bytes, bytes->data()), 16});
    auto                    open           = std::make_shared<std::promise<void>>();
    auto                    release        = open->get_future().share();
    auto                    ensure_release = std::shared_ptr<void>(nullptr, [open](void*) { open->set_value(); });
    std::promise<uint64_t>  next_id;
    auto                    next_admitted = next_id.get_future();
    std::promise<ErrorInfo> second_terminal;
    auto                    second_result = second_terminal.get_future();
    ASSERT_NE(adapter->transferHost(request,
                                    [&](ErrorInfo error) {
                                        EXPECT_TRUE(error.ok());
                                        next_id.set_value(adapter->transferHost(request, [&](ErrorInfo next_error) {
                                            second_terminal.set_value(std::move(next_error));
                                        }));
                                        release.wait_for(std::chrono::seconds(10));
                                    }),
              0u);
    ASSERT_EQ(next_admitted.wait_for(std::chrono::seconds(3)), std::future_status::ready);
    EXPECT_NE(next_admitted.get(), 0u);
    std::promise<void> complete;
    auto               drained = complete.get_future();
    adapter->drain([&] { complete.set_value(); });
    EXPECT_EQ(drained.wait_for(std::chrono::milliseconds(30)), std::future_status::timeout);
    ensure_release.reset();
    ASSERT_EQ(second_result.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    EXPECT_TRUE(second_result.get().ok());
    ASSERT_EQ(drained.wait_for(std::chrono::seconds(5)), std::future_status::ready);
}

TEST_F(RemoteMockAdapterTest, RegistrationAllocationFailuresPrecedeH2dDispatch) {
    group   = makeTestGroupSet(0, group->topologyPtr(), {"group0"}, {device});
    engine  = std::make_shared<PerRankBlockTransferEngine>(std::vector<GroupSetPtr>{group});
    adapter = RemoteMockAdapter::create(config, ParallelismConfig{}, {group}, engine);
    engine->setRemoteMockAdapter(adapter);
    device->incRef(block);
    has_request_ref = true;
    {
        // The physical owner occupies the allocation-free inline slot. The
        // dependent terminal allocates one list node before any CUDA dispatch.
        auto fault   = armRemoteCopyFailure(1);
        auto cleanup = std::shared_ptr<void>(nullptr, [fault](void*) { retireRemoteCopyFailure(fault); });
        auto context = engine->execute(
            TransferTask({TransferDescriptor::remoteToDevice(0, adapter->freezeSource(0, 19, 3), {block})},
                         std::chrono::seconds(5)));
        ASSERT_TRUE(fault->waitInjected());
        EXPECT_EQ(fault->registrations, 1);
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        EXPECT_EQ(fault->copies, 0) << "registration must precede physical dispatch";
        if (fault->copies) {
            EXPECT_FALSE(context->done());
            EXPECT_EQ(device->referencedBlocksNum(BlockTreeRefType::LOAD), 1u);
            EXPECT_EQ(fault->early_frees, 0);
        }
        fault->open();
        context->waitDone();
        EXPECT_FALSE(context->success());
        std::promise<void> done;
        auto               drained = done.get_future();
        adapter->drain([&] { done.set_value(); });
        ASSERT_EQ(drained.wait_for(std::chrono::seconds(3)), std::future_status::ready);
        EXPECT_EQ(device->referencedBlocksNum(BlockTreeRefType::LOAD), 0u);
    }
    EXPECT_TRUE(run(plan()).ok());
    expectTarget(0);
}

class RemoteMockAdapterDeathTest: public RemoteMockAdapterTest {};
TEST_F(RemoteMockAdapterDeathTest, ThrowingDrainDoesNotEscapePostDmaFailureTerminal) {
    ::testing::FLAGS_gtest_death_test_style = "threadsafe";
    ASSERT_EXIT(
        {
            const auto exercise = [&] {
                config.mock_remote_cache_max_inflight = 1;
                group                                 = makeTestGroupSet(0, group->topologyPtr(), {"group0"}, {device});
                engine = std::make_shared<PerRankBlockTransferEngine>(std::vector<GroupSetPtr>{group});
                adapter = RemoteMockAdapter::create(config, ParallelismConfig{}, {group}, engine);
                engine->setRemoteMockAdapter(adapter);
                device->incRef(block);
                has_request_ref      = true;
                auto       gate      = armRemoteCopyFailure(0);
                auto       cleanup   = std::shared_ptr<void>(nullptr, [gate](void*) { retireRemoteCopyFailure(gate); });
                const auto make_task = [&] {
                    return TransferTask(
                        {TransferDescriptor::remoteToDevice(0, adapter->freezeSource(0, 19, 3), {block})},
                        std::chrono::seconds(5));
                };
                auto               context           = engine->execute(make_task());
                auto               callback_owner    = std::make_shared<int>(1);
                std::weak_ptr<int> callback_lifetime = callback_owner;
                context->onDone([retained = callback_owner](TransferResult) {});
                callback_owner.reset();
                {
                    std::unique_lock<std::mutex> lock(gate->mutex);
                    ASSERT_TRUE(gate->cv.wait_for(lock, std::chrono::seconds(3), [&] { return gate->queued; }));
                }
                ASSERT_EQ(device->referencedBlocksNum(BlockTreeRefType::LOAD), 1u);
                std::atomic<int>   callbacks{0};
                std::promise<void> later_callback;
                auto               later = later_callback.get_future();
                adapter->drain([&] {
                    ++callbacks;
                    throw std::runtime_error("external drain callback");
                });
                adapter->drain([&] {
                    ++callbacks;
                    later_callback.set_value();
                });
                EXPECT_EQ(later.wait_for(std::chrono::milliseconds(20)), std::future_status::timeout);
                fail_adapter_physical_post = true;
                gate->open();
                ASSERT_EQ(later.wait_for(std::chrono::seconds(3)), std::future_status::ready);
                EXPECT_FALSE(fail_adapter_physical_post.load());
                EXPECT_EQ(callbacks, 2);
                EXPECT_TRUE(context->done());
                EXPECT_FALSE(context->success());
                EXPECT_EQ(adapter->stats().read_failed, 1u);
                EXPECT_EQ(adapter->stats().read_completed, 0u);
                EXPECT_EQ(device->referencedBlocksNum(BlockTreeRefType::LOAD), 0u);
                EXPECT_EQ(device->refCount(block), 1u);
                EXPECT_TRUE(callback_lifetime.expired());
                adapter->drain([&] { ++callbacks; });
                EXPECT_EQ(callbacks, 3);
                cleanup.reset();
                auto next = engine->execute(make_task());
                next->waitDone();
                EXPECT_TRUE(next->success());
                std::promise<void> done;
                auto               drained = done.get_future();
                adapter->drain([&] { done.set_value(); });
                ASSERT_EQ(drained.wait_for(std::chrono::seconds(3)), std::future_status::ready);
                EXPECT_EQ(device->referencedBlocksNum(BlockTreeRefType::LOAD), 0u);
            };
            exercise();
            TearDown();
            std::_Exit(::testing::Test::HasFailure() ? 1 : 0);
        },
        ::testing::ExitedWithCode(0),
        "");
}

}  // namespace
}  // namespace rtp_llm
