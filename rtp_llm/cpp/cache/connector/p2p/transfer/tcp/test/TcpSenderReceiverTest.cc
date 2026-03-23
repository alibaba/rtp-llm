#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <future>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "rtp_llm/cpp/cache/BlockInfo.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/IKVCacheReceiver.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/IKVCacheSender.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/TransferErrorCode.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/Types.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/tcp/TcpKVCacheReceiver.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/tcp/TcpKVCacheSender.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/tcp/test/DeviceReserveForTest.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/core/Buffer.h"
#include "rtp_llm/cpp/devices/DeviceFactory.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "autil/NetUtil.h"

namespace rtp_llm {
namespace transfer {
namespace tcp {

// ---------------------------------------------------------------------------
// Helper: send and block until callback fires, returning the error code.
// ---------------------------------------------------------------------------

static TransferErrorCode
syncSend(TcpKVCacheSender& sender, const SendRequest& req, std::chrono::seconds timeout = std::chrono::seconds(10)) {
    TransferErrorCode  result = TransferErrorCode::UNKNOWN;
    std::promise<void> p;
    auto               future = p.get_future();

    sender.send(req, [&](TransferErrorCode code, const std::string&) {
        result = code;
        p.set_value();
    });

    future.wait_for(timeout);
    return result;
}

// ---------------------------------------------------------------------------
// Block builders
// ---------------------------------------------------------------------------

static KeyBlockInfoMap makeBlocks(int64_t cache_key, void* addr, size_t size) {
    auto kbi = std::make_shared<KeyBlockInfo>();
    kbi->blocks.push_back(BlockInfo{false, 0, 0, addr, size});
    KeyBlockInfoMap m;
    m[cache_key] = kbi;
    return m;
}

// ---------------------------------------------------------------------------
// Fixture: Sender only (no Receiver)
// ---------------------------------------------------------------------------

class TcpSenderOnlyTest: public ::testing::Test {
public:
    static void SetUpTestSuite() {
        ParallelismConfig           parallelism_config;
        ModelConfig                 model_config;
        EPLBConfig                  eplb_config;
        FMHAConfig                  fmha_config;
        DeviceResourceConfig        device_resource_config;
        MoeConfig                   moe_config;
        SpeculativeExecutionConfig  sp_config;
        MiscellaneousConfig         misc_config;
        ProfilingDebugLoggingConfig profiling_debug_logging_config;
        HWKernelConfig              hw_kernel_config;
        ConcurrencyConfig           concurrency_config;
        FfnDisAggregateConfig       ffn_disaggregate_config;
        RuntimeConfig               runtime_config;
        ModelSpecificConfig         model_specific_config;

        tcp_test_internal::apply_device_reserve_from_env(device_resource_config);
        DeviceFactory::initDevices(parallelism_config,
                                   model_config,
                                   eplb_config,
                                   fmha_config,
                                   device_resource_config,
                                   moe_config,
                                   sp_config,
                                   misc_config,
                                   profiling_debug_logging_config,
                                   hw_kernel_config,
                                   concurrency_config,
                                   ffn_disaggregate_config,
                                   runtime_config,
                                   model_specific_config,
                                   NcclCommConfig{});
        device_ = DeviceFactory::getDefaultDevice();
    }

protected:
    void SetUp() override {
        unused_port_ = autil::NetUtil::randomPort();
        sender_      = std::make_unique<TcpKVCacheSender>();
        ASSERT_TRUE(sender_->init(2));
    }

    static DeviceBase*                device_;
    std::unique_ptr<TcpKVCacheSender> sender_;
    uint32_t                          unused_port_ = 0;
};

DeviceBase* TcpSenderOnlyTest::device_ = nullptr;

// ---------------------------------------------------------------------------
// Fixture: Sender + Receiver on localhost
// ---------------------------------------------------------------------------

class TcpSenderReceiverTest: public ::testing::Test {
public:
    static void SetUpTestSuite() {
        ParallelismConfig           parallelism_config;
        ModelConfig                 model_config;
        EPLBConfig                  eplb_config;
        FMHAConfig                  fmha_config;
        DeviceResourceConfig        device_resource_config;
        MoeConfig                   moe_config;
        SpeculativeExecutionConfig  sp_config;
        MiscellaneousConfig         misc_config;
        ProfilingDebugLoggingConfig profiling_debug_logging_config;
        HWKernelConfig              hw_kernel_config;
        ConcurrencyConfig           concurrency_config;
        FfnDisAggregateConfig       ffn_disaggregate_config;
        RuntimeConfig               runtime_config;
        ModelSpecificConfig         model_specific_config;

        tcp_test_internal::apply_device_reserve_from_env(device_resource_config);
        DeviceFactory::initDevices(parallelism_config,
                                   model_config,
                                   eplb_config,
                                   fmha_config,
                                   device_resource_config,
                                   moe_config,
                                   sp_config,
                                   misc_config,
                                   profiling_debug_logging_config,
                                   hw_kernel_config,
                                   concurrency_config,
                                   ffn_disaggregate_config,
                                   runtime_config,
                                   model_specific_config,
                                   NcclCommConfig{});
        device_ = DeviceFactory::getDefaultDevice();
    }

protected:
    void SetUp() override {
        test_port_ = autil::NetUtil::randomPort();
        sender_    = std::make_unique<TcpKVCacheSender>();
        receiver_  = std::make_unique<TcpKVCacheReceiver>();
        ASSERT_TRUE(sender_->init(2));
        ASSERT_TRUE(receiver_->init(test_port_, 2, 4));
    }

    void TearDown() override {
        receiver_.reset();
        sender_.reset();
    }

    BufferPtr allocDevice(size_t size) {
        if (!device_)
            return nullptr;
        return device_->allocateBuffer({DataType::TYPE_UINT8, {size}, AllocationType::DEVICE});
    }

    bool verifyDevice(void* ptr, const std::string& expected) {
        const size_t size    = expected.size();
        auto         cpu_buf = device_->allocateBuffer({DataType::TYPE_UINT8, {size}, AllocationType::HOST});
        auto         src     = Buffer(MemoryType::MEMORY_GPU, DataType::TYPE_UINT8, {size}, ptr);
        auto         dst     = Buffer(MemoryType::MEMORY_CPU, DataType::TYPE_UINT8, {size}, cpu_buf->data());
        device_->copy({dst, src});
        device_->syncAndCheck();
        return std::memcmp(cpu_buf->data(), expected.data(), size) == 0;
    }

    static DeviceBase*                  device_;
    std::unique_ptr<TcpKVCacheSender>   sender_;
    std::unique_ptr<TcpKVCacheReceiver> receiver_;
    uint32_t                            test_port_ = 0;
};

DeviceBase* TcpSenderReceiverTest::device_ = nullptr;

// ===========================================================================
// F. Build request failures (Sender only, no network)
// ===========================================================================

// F1: block_info is empty → makeTransferRequest returns nullptr → BUILD_REQUEST_FAILED.
TEST_F(TcpSenderOnlyTest, F1_EmptyBlockInfo_ImmediateBuildFailed) {
    SendRequest req;
    req.ip          = "127.0.0.1";
    req.port        = unused_port_;
    req.unique_key  = "k_f1";
    req.block_info  = {};  // empty
    req.deadline_ms = currentTimeMs() + 5000;

    auto code = syncSend(*sender_, req);
    EXPECT_EQ(code, TransferErrorCode::BUILD_REQUEST_FAILED);
}

// ===========================================================================
// D. Connection errors
// ===========================================================================

// D1/D3: Receiver is not running; arpc returns a channel object regardless, so
//        the RPC is sent but times out → RPC_FAILED (controller->Failed()=true).
TEST_F(TcpSenderOnlyTest, D1_ConnectionFailed_ReceiverNotRunning) {
    char        dummy[64]{};
    SendRequest req;
    req.ip          = "127.0.0.1";
    req.port        = unused_port_;
    req.unique_key  = "k_d1";
    req.block_info  = makeBlocks(1, dummy, 64);
    req.deadline_ms = currentTimeMs() + 1000;  // 1s deadline to keep the test fast

    auto code = syncSend(*sender_, req, std::chrono::seconds(5));
    // anet always returns a channel object; connect failure surfaces as RPC_FAILED
    // after the controller expire time elapses.
    EXPECT_EQ(code, TransferErrorCode::RPC_FAILED);
}

// ===========================================================================
// B. Timeout (integration)
// ===========================================================================

// B1: Receiver is running but has not called recv(); the RPC enters wait_tasks_
//     and times out after the deadline.
TEST_F(TcpSenderReceiverTest, B1_SenderTimeout_ReceiverHasNoMatchingTask) {
    char        dummy[64]{};
    SendRequest req;
    req.ip          = "127.0.0.1";
    req.port        = test_port_;
    req.unique_key  = "k_b1_integ";
    req.block_info  = makeBlocks(1, dummy, 64);
    req.deadline_ms = currentTimeMs() + 500;  // short deadline

    auto code = syncSend(*sender_, req, std::chrono::seconds(5));
    EXPECT_EQ(code, TransferErrorCode::TIMEOUT);
}

// ===========================================================================
// C. Cancel (integration)
// ===========================================================================

// C1: Receiver registers a task then immediately cancels it; the RPC arrives,
//     waitCheckProc finds the already-done task and responds CANCELLED.
TEST_F(TcpSenderReceiverTest, C1_RecvCancelledBeforeRpcArrives_Integration) {
    char dummy[64]{};

    RecvRequest recv_req;
    recv_req.unique_key  = "k_c1_integ";
    recv_req.block_info  = makeBlocks(1, dummy, 64);
    recv_req.deadline_ms = currentTimeMs() + 5000;

    auto task = receiver_->recv(recv_req);
    ASSERT_NE(task, nullptr);
    task->cancel();

    SendRequest send_req;
    send_req.ip          = "127.0.0.1";
    send_req.port        = test_port_;
    send_req.unique_key  = "k_c1_integ";
    send_req.block_info  = makeBlocks(1, dummy, 64);
    send_req.deadline_ms = currentTimeMs() + 5000;

    auto code = syncSend(*sender_, send_req, std::chrono::seconds(5));
    EXPECT_EQ(code, TransferErrorCode::CANCELLED);
    EXPECT_EQ(task->errorCode(), TransferErrorCode::CANCELLED);
}

// ===========================================================================
// A. Happy Path (integration)
// ===========================================================================

// A1: Full end-to-end transfer: Receiver registers GPU buffer, Sender copies
//     data over TCP, GPU content is verified.
TEST_F(TcpSenderReceiverTest, A1_NormalTransfer_EndToEnd_Success) {
    if (!device_) {
        GTEST_SKIP() << "No GPU device";
    }

    constexpr size_t  size = 64;
    const std::string content(size, 'E');
    auto              gpu_buf = allocDevice(size);

    RecvRequest recv_req;
    recv_req.unique_key  = "k_a1_integ";
    recv_req.block_info  = makeBlocks(1, gpu_buf->data(), size);
    recv_req.deadline_ms = currentTimeMs() + 5000;

    auto task = receiver_->recv(recv_req);
    ASSERT_NE(task, nullptr);

    // Sender packs the host copy of the content and ships it.
    // We use a host buffer as the "GPU source" for CudaCopyUtil in tests;
    // the send-side copy path works with any device-visible pointer.
    auto host_buf = device_->allocateBuffer({DataType::TYPE_UINT8, {size}, AllocationType::HOST});
    std::memcpy(host_buf->data(), content.data(), size);
    // Flush so the GPU can read it.
    auto gpu_src = device_->allocateBuffer({DataType::TYPE_UINT8, {size}, AllocationType::DEVICE});
    {
        auto src = Buffer(MemoryType::MEMORY_CPU, DataType::TYPE_UINT8, {size}, host_buf->data());
        auto dst = Buffer(MemoryType::MEMORY_GPU, DataType::TYPE_UINT8, {size}, gpu_src->data());
        device_->copy({dst, src});
        device_->syncAndCheck();
    }

    SendRequest send_req;
    send_req.ip          = "127.0.0.1";
    send_req.port        = test_port_;
    send_req.unique_key  = "k_a1_integ";
    send_req.block_info  = makeBlocks(1, gpu_src->data(), size);
    send_req.deadline_ms = currentTimeMs() + 5000;

    auto code = syncSend(*sender_, send_req, std::chrono::seconds(10));
    EXPECT_EQ(code, TransferErrorCode::OK);
    EXPECT_TRUE(task->success());
    EXPECT_TRUE(verifyDevice(gpu_buf->data(), content));
}

// ===========================================================================
// G. Concurrency (integration)
// ===========================================================================

// G2: Multiple senders transfer independent keys to the same receiver;
//     each key must complete successfully with correct data.
//     Sends are issued sequentially to avoid concurrent GPU-copy races,
//     while still exercising the multi-key routing path in TcpTransferService.
TEST_F(TcpSenderReceiverTest, G2_MultipleSenders_ConcurrentKeys) {
    if (!device_) {
        GTEST_SKIP() << "No GPU device";
    }

    constexpr int    N    = 3;
    constexpr size_t size = 64;

    std::vector<BufferPtr>           src_bufs(N), dst_bufs(N);
    std::vector<IKVCacheRecvTaskPtr> tasks(N);
    std::vector<std::string>         contents(N);

    for (int i = 0; i < N; ++i) {
        contents[i] = std::string(size, static_cast<char>('0' + i));
        dst_bufs[i] = allocDevice(size);
        src_bufs[i] = allocDevice(size);

        auto host_src = device_->allocateBuffer({DataType::TYPE_UINT8, {size}, AllocationType::HOST});
        std::memcpy(host_src->data(), contents[i].data(), size);
        {
            auto src = Buffer(MemoryType::MEMORY_CPU, DataType::TYPE_UINT8, {size}, host_src->data());
            auto dst = Buffer(MemoryType::MEMORY_GPU, DataType::TYPE_UINT8, {size}, src_bufs[i]->data());
            device_->copy({dst, src});
            // Sync before host_src goes out of scope to avoid use-after-free in async copy.
            device_->syncAndCheck();
        }

        RecvRequest recv_req;
        recv_req.unique_key  = "k_g2_" + std::to_string(i);
        recv_req.block_info  = makeBlocks(1, dst_bufs[i]->data(), size);
        recv_req.deadline_ms = currentTimeMs() + 5000;
        tasks[i]             = receiver_->recv(recv_req);
        ASSERT_NE(tasks[i], nullptr);
    }

    // Issue N sends sequentially; each completes before the next starts to avoid
    // concurrent GPU-copy on shared CudaCopyUtil state.
    for (int i = 0; i < N; ++i) {
        SendRequest send_req;
        send_req.ip          = "127.0.0.1";
        send_req.port        = test_port_;
        send_req.unique_key  = "k_g2_" + std::to_string(i);
        send_req.block_info  = makeBlocks(1, src_bufs[i]->data(), size);
        send_req.deadline_ms = currentTimeMs() + 5000;

        auto code = syncSend(*sender_, send_req, std::chrono::seconds(10));
        EXPECT_EQ(code, TransferErrorCode::OK) << "key index " << i;
        EXPECT_TRUE(tasks[i]->success()) << "key index " << i;
        // batchCopyToDevice on the worker thread is non-blocking; sync all streams
        // before the D→H verification copy to avoid reading stale GPU data.
        device_->syncAndCheck();
        EXPECT_TRUE(verifyDevice(dst_bufs[i]->data(), contents[i])) << "key index " << i;
    }
}

}  // namespace tcp
}  // namespace transfer
}  // namespace rtp_llm
