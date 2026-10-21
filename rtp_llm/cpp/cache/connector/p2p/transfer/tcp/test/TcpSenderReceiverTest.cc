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
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "autil/NetUtil.h"
#include <torch/torch.h>
#include <c10/cuda/CUDAStream.h>

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
        initRuntime(/*device_id=*/0,
                    /*trace_memory=*/false,
                    /*enable_comm_overlap=*/false,
                    MlaOpsType::AUTO);
        device_initialized_ = isRuntimeInitialized();
    }

protected:
    void SetUp() override {
        unused_port_ = autil::NetUtil::randomPort();
        sender_      = std::make_unique<TcpKVCacheSender>();
        ASSERT_TRUE(sender_->init(2));
    }

    static bool                       device_initialized_;
    std::unique_ptr<TcpKVCacheSender> sender_;
    uint32_t                          unused_port_ = 0;
};

bool TcpSenderOnlyTest::device_initialized_ = false;

// ---------------------------------------------------------------------------
// Fixture: Sender + Receiver on localhost
// ---------------------------------------------------------------------------

class TcpSenderReceiverTest: public ::testing::Test {
public:
    static void SetUpTestSuite() {
        initRuntime(/*device_id=*/0,
                    /*trace_memory=*/false,
                    /*enable_comm_overlap=*/false,
                    MlaOpsType::AUTO);
        device_initialized_ = isRuntimeInitialized();
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

    torch::Tensor allocDevice(size_t size) {
        return torch::empty({static_cast<int64_t>(size)},
                            torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
    }

    bool verifyDevice(void* ptr, const std::string& expected) {
        const size_t size       = expected.size();
        auto         gpu_tensor = torch::from_blob(
            ptr, {static_cast<int64_t>(size)}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
        auto cpu_tensor = gpu_tensor.cpu();
        at::cuda::getCurrentCUDAStream().synchronize();
        return std::memcmp(cpu_tensor.data_ptr(), expected.data(), size) == 0;
    }

    static bool                         device_initialized_;
    std::unique_ptr<TcpKVCacheSender>   sender_;
    std::unique_ptr<TcpKVCacheReceiver> receiver_;
    uint32_t                            test_port_ = 0;
};

bool TcpSenderReceiverTest::device_initialized_ = false;

TEST_F(TcpSenderOnlyTest, StagedGatherFollowsSourceDeviceOnReusedThread) {
    if (torch::cuda::device_count() < 2) {
        GTEST_SKIP() << "requires two CUDA devices";
    }
    auto copied = std::async(std::launch::async, [&]() {
        for (const int device_index : {0, 1, 0}) {
            std::vector<torch::Tensor> buffers;
            auto blocks = std::make_shared<KeyBlockInfo>();
            for (int tile = 0; tile < 4; ++tile) {
                buffers.push_back(torch::full(
                    {2048},
                    10 * device_index + tile,
                    torch::TensorOptions().dtype(torch::kUInt8).device(torch::Device(torch::kCUDA, device_index))));
                blocks->blocks.push_back(BlockInfo{true, device_index, 0, buffers.back().data_ptr(), 2048});
            }
            at::cuda::getCurrentCUDAStream(device_index).synchronize();
            SendRequest request;
            request.unique_key = "staged-device-" + std::to_string(device_index);
            request.block_info[1] = blocks;
            const auto wire = sender_->makeTransferRequest(request, std::make_shared<TransferClientMetricsCollector>());
            if (!wire || wire->blocks_size() != 1 || wire->blocks(0).blocks_size() != 4) {
                return false;
            }
            for (int tile = 0; tile < 4; ++tile) {
                if (wire->blocks(0).blocks(tile).content() != std::string(2048, 10 * device_index + tile)) {
                    return false;
                }
            }
        }
        return true;
    });
    EXPECT_TRUE(copied.get());
}

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
    req.timeout_ms = 5000;

    auto code = syncSend(*sender_, req);
    EXPECT_EQ(code, TransferErrorCode::BUILD_REQUEST_FAILED);
}

// ===========================================================================
// D. Connection errors
// ===========================================================================

// D1/D3: Receiver is not running; arpc still returns a channel object.
// The queued request expires with ARPC_ERROR_TIMEOUT, which maps to TIMEOUT.
TEST_F(TcpSenderOnlyTest, D1_ConnectionFailed_ReceiverNotRunning) {
    if (!device_initialized_) {
        GTEST_SKIP() << "No GPU device";
    }
    auto        gpu_buf = torch::empty({64}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
    SendRequest req;
    req.ip          = "127.0.0.1";
    req.port        = unused_port_;
    req.unique_key  = "k_d1";
    req.block_info  = makeBlocks(1, gpu_buf.data_ptr(), 64);
    req.deadline_ms = currentTimeMs() + 1000;
    req.timeout_ms = 1000;  // 1s deadline to keep the test fast

    auto code = syncSend(*sender_, req, std::chrono::seconds(5));
    // Preserve the typed deadline failure even when connection refusal preceded it.
    EXPECT_EQ(code, TransferErrorCode::TIMEOUT);
}

// ===========================================================================
// B. Timeout (integration)
// ===========================================================================

// B1: Receiver is running but has not called recv(); the RPC enters wait_tasks_
//     and times out after the deadline.
TEST_F(TcpSenderOnlyTest, BusyStagedScratchSelectsDirectCopyWithoutWaitingForItsOwner) {
    std::unique_lock<std::mutex> owner(sender_->staged_scratch_mutex_);
    auto source = torch::ones({64}, torch::TensorOptions(torch::kUInt8).device(torch::kCUDA));
    auto target = torch::zeros({64}, torch::kUInt8);
    std::vector<CopyTask> tasks(4);
    for (size_t i = 0; i < tasks.size(); ++i) {
        tasks[i].src_ptr = source.data_ptr<uint8_t>() + 16 * i;
        tasks[i].dst_ptr = reinterpret_cast<char*>(target.data_ptr<uint8_t>()) + 16 * i;
        tasks[i].size = 16;
    }
    auto attempt = std::async(std::launch::async, [&] {
        return sender_->tryStagedGatherCopyToHost(tasks, 0);
    });
    const auto ready = attempt.wait_for(std::chrono::seconds(5));
    // Keep the owner locked through the observation; release on every path
    // before joining so a regression cannot hang test teardown.
    EXPECT_EQ(ready, std::future_status::ready);
    owner.unlock();
    EXPECT_FALSE(attempt.get());
}

TEST_F(TcpSenderOnlyTest, MixedDeviceTasksSelectDirectCopyBeforeStagedSubmission) {
    auto source = torch::ones({64}, torch::TensorOptions(torch::kUInt8).device(torch::kCUDA));
    auto target = torch::zeros({64}, torch::kUInt8);
    std::vector<CopyTask> tasks(4);
    for (size_t i = 0; i < tasks.size(); ++i) {
        tasks[i].src_ptr = source.data_ptr<uint8_t>() + 16 * i;
        tasks[i].dst_ptr = reinterpret_cast<char*>(target.data_ptr<uint8_t>()) + 16 * i;
        tasks[i].size = 16;
        tasks[i].device_index = source.get_device();
    }
    // Metadata alone must reject this batch before dereferencing on another GPU.
    tasks.back().device_index = source.get_device() + 1;
    EXPECT_FALSE(sender_->tryStagedGatherCopyToHost(tasks, source.get_device()));
    EXPECT_EQ(target.sum().item<int64_t>(), 0);
}

TEST_F(TcpSenderOnlyTest, BusyStagedScratchDirectFallbackPreservesEverySerializedByte) {
    auto source = torch::arange(64, torch::TensorOptions(torch::kUInt8).device(torch::kCUDA));
    at::cuda::getCurrentCUDAStream().synchronize();
    SendRequest request;
    request.unique_key = "busy-staged-byte-check";
    for (int i = 0; i < 4; ++i) {
        auto block = std::make_shared<KeyBlockInfo>();
        block->blocks.push_back(BlockInfo{true, 0, 0, source.data_ptr<uint8_t>() + 16 * i, 16});
        request.block_info.emplace(100 + i, std::move(block));
    }
    auto metrics = std::make_shared<TransferClientMetricsCollector>();
    std::unique_lock<std::mutex> owner(sender_->staged_scratch_mutex_);
    auto building = std::async(std::launch::async, [&] { return sender_->makeTransferRequest(request, metrics); });
    const auto ready = building.wait_for(std::chrono::seconds(5));
    EXPECT_EQ(ready, std::future_status::ready);
    owner.unlock();
    auto serialized = building.get();
    ASSERT_NE(serialized, nullptr);
    ASSERT_EQ(serialized->blocks_size(), 4);
    for (const auto& block : serialized->blocks()) {
        ASSERT_GE(block.key(), 100);
        ASSERT_LT(block.key(), 104);
        ASSERT_EQ(block.blocks_size(), 1);
        const auto& content = block.blocks(0).content();
        ASSERT_EQ(content.size(), 16u);
        for (size_t byte = 0; byte < content.size(); ++byte) {
            EXPECT_EQ(static_cast<uint8_t>(content[byte]), 16 * (block.key() - 100) + byte);
        }
    }
}

TEST_F(TcpSenderOnlyTest, ExpiredTransferDoesNotSubmitTcpRpc) {
    auto request = std::make_shared<::tcp_transfer::TcpLayerBlockTransferRequest>();
    request->set_unique_key("expired-before-submit");
    request->set_timeout_ms(60000);
    int callbacks = 0;
    sender_->loadToRemote("127.0.0.1", unused_port_, request,
        [&](TransferErrorCode code, const std::string&) {
            ++callbacks;
            EXPECT_EQ(code, TransferErrorCode::TIMEOUT);
        }, currentTimeMs() - 1);
    EXPECT_EQ(callbacks, 1);
}

TEST_F(TcpSenderReceiverTest, WireTimeoutUsesRemainingRequestBudgetInsteadOfPrefillConfiguration) {
    for (const int64_t static_timeout : {int64_t{0}, int64_t{-1}, int64_t{2147483648LL}}) {
        auto request = std::make_shared<::tcp_transfer::TcpLayerBlockTransferRequest>();
        request->set_unique_key("remaining-window-" + std::to_string(static_timeout));
        request->set_timeout_ms(static_timeout);
        auto completion = std::make_shared<std::promise<TransferErrorCode>>();
        auto done = completion->get_future();
        sender_->loadToRemote("127.0.0.1", test_port_, request,
            [completion](TransferErrorCode code, const std::string&) { completion->set_value(code); },
            currentTimeMs() + 250);
        EXPECT_GT(request->timeout_ms(), 0);
        EXPECT_LE(request->timeout_ms(), 250);
        ASSERT_EQ(done.wait_for(std::chrono::seconds(5)), std::future_status::ready);
        EXPECT_EQ(done.get(), TransferErrorCode::TIMEOUT);
    }
}

TEST_F(TcpSenderReceiverTest, B1_SenderTimeout_ReceiverHasNoMatchingTask) {
    if (!device_initialized_) {
        GTEST_SKIP() << "No GPU device";
    }
    auto        gpu_buf = allocDevice(64);
    SendRequest req;
    req.ip          = "127.0.0.1";
    req.port        = test_port_;
    req.unique_key  = "k_b1_integ";
    req.block_info  = makeBlocks(1, gpu_buf.data_ptr(), 64);
    req.deadline_ms = currentTimeMs() + 500;
    req.timeout_ms = 500;  // short deadline

    auto code = syncSend(*sender_, req, std::chrono::seconds(5));
    EXPECT_EQ(code, TransferErrorCode::TIMEOUT);
}

// ===========================================================================
// C. Cancel (integration)
// ===========================================================================

// C1: Receiver registers a task then immediately cancels it; the RPC arrives,
//     waitCheckProc finds the already-done task and responds CANCELLED.
TEST_F(TcpSenderReceiverTest, C1_RecvCancelledBeforeRpcArrives_Integration) {
    if (!device_initialized_) {
        GTEST_SKIP() << "No GPU device";
    }
    auto gpu_buf_recv = allocDevice(64);
    auto gpu_buf_send = allocDevice(64);

    RecvRequest recv_req;
    recv_req.unique_key  = "k_c1_integ";
    recv_req.block_info  = makeBlocks(1, gpu_buf_recv.data_ptr(), 64);
    recv_req.deadline_ms = currentTimeMs() + 5000;

    auto task = receiver_->recv(recv_req);
    ASSERT_NE(task, nullptr);
    task->cancel();

    SendRequest send_req;
    send_req.ip          = "127.0.0.1";
    send_req.port        = test_port_;
    send_req.unique_key  = "k_c1_integ";
    send_req.block_info  = makeBlocks(1, gpu_buf_send.data_ptr(), 64);
    send_req.deadline_ms = currentTimeMs() + 5000;
    send_req.timeout_ms = 5000;

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
    if (!device_initialized_) {
        GTEST_SKIP() << "No GPU device";
    }

    constexpr size_t  size = 64;
    const std::string content(size, 'E');
    auto              gpu_buf = allocDevice(size);

    RecvRequest recv_req;
    recv_req.unique_key  = "k_a1_integ";
    recv_req.block_info  = makeBlocks(1, gpu_buf.data_ptr(), size);
    recv_req.deadline_ms = currentTimeMs() + 5000;

    auto task = receiver_->recv(recv_req);
    ASSERT_NE(task, nullptr);

    auto gpu_src = allocDevice(size);
    {
        auto host_tensor =
            torch::from_blob(const_cast<char*>(content.data()), {static_cast<int64_t>(size)}, torch::kUInt8);
        gpu_src.copy_(host_tensor);
        at::cuda::getCurrentCUDAStream().synchronize();
    }

    SendRequest send_req;
    send_req.ip          = "127.0.0.1";
    send_req.port        = test_port_;
    send_req.unique_key  = "k_a1_integ";
    send_req.block_info  = makeBlocks(1, gpu_src.data_ptr(), size);
    send_req.deadline_ms = currentTimeMs() + 5000;
    send_req.timeout_ms = 5000;

    auto code = syncSend(*sender_, send_req, std::chrono::seconds(10));
    EXPECT_EQ(code, TransferErrorCode::OK);
    EXPECT_TRUE(task->success());
    EXPECT_TRUE(verifyDevice(gpu_buf.data_ptr(), content));
}

// ===========================================================================
// G. Concurrency (integration)
// ===========================================================================

// G2: Multiple senders transfer independent keys to the same receiver;
//     each key must complete successfully with correct data.
//     Sends are issued sequentially to avoid concurrent GPU-copy races,
//     while still exercising the multi-key routing path in TcpTransferService.
TEST_F(TcpSenderReceiverTest, G2_MultipleSenders_ConcurrentKeys) {
    if (!device_initialized_) {
        GTEST_SKIP() << "No GPU device";
    }

    constexpr int    N    = 3;
    constexpr size_t size = 64;

    std::vector<torch::Tensor>       src_bufs(N), dst_bufs(N);
    std::vector<IKVCacheRecvTaskPtr> tasks(N);
    std::vector<std::string>         contents(N);

    for (int i = 0; i < N; ++i) {
        contents[i] = std::string(size, static_cast<char>('0' + i));
        dst_bufs[i] = allocDevice(size);
        src_bufs[i] = allocDevice(size);

        {
            auto host_tensor =
                torch::from_blob(const_cast<char*>(contents[i].data()), {static_cast<int64_t>(size)}, torch::kUInt8);
            src_bufs[i].copy_(host_tensor);
            at::cuda::getCurrentCUDAStream().synchronize();
        }

        RecvRequest recv_req;
        recv_req.unique_key  = "k_g2_" + std::to_string(i);
        recv_req.block_info  = makeBlocks(1, dst_bufs[i].data_ptr(), size);
        recv_req.deadline_ms = currentTimeMs() + 5000;
        tasks[i]             = receiver_->recv(recv_req);
        ASSERT_NE(tasks[i], nullptr);
    }

    for (int i = 0; i < N; ++i) {
        SendRequest send_req;
        send_req.ip          = "127.0.0.1";
        send_req.port        = test_port_;
        send_req.unique_key  = "k_g2_" + std::to_string(i);
        send_req.block_info  = makeBlocks(1, src_bufs[i].data_ptr(), size);
        send_req.deadline_ms = currentTimeMs() + 5000;
        send_req.timeout_ms  = 5000;

        auto code = syncSend(*sender_, send_req, std::chrono::seconds(10));
        EXPECT_EQ(code, TransferErrorCode::OK) << "key index " << i;
        EXPECT_TRUE(tasks[i]->success()) << "key index " << i;
        at::cuda::getCurrentCUDAStream().synchronize();
        EXPECT_TRUE(verifyDevice(dst_bufs[i].data_ptr(), contents[i])) << "key index " << i;
    }
}

}  // namespace tcp
}  // namespace transfer
}  // namespace rtp_llm
