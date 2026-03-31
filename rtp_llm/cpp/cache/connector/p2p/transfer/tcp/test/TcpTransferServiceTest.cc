#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <future>
#include <memory>
#include <string>
#include <thread>

#include "rtp_llm/cpp/cache/BlockInfo.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/TransferErrorCode.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/TransferTask.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/Types.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/tcp/TcpTransferService.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/tcp/test/DeviceReserveForTest.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/tcp/proto/tcp_service.pb.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/core/ExecOps.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include <torch/torch.h>
#include <c10/cuda/CUDAStream.h>

namespace rtp_llm {
namespace transfer {
namespace tcp {

// ---------------------------------------------------------------------------
// Test doubles
// ---------------------------------------------------------------------------

class MockRpcController: public google::protobuf::RpcController {
public:
    void Reset() override {}
    bool Failed() const override {
        return false;
    }
    std::string ErrorText() const override {
        return "";
    }
    void StartCancel() override {}
    void SetFailed(const std::string&) override {}
    bool IsCanceled() const override {
        return false;
    }
    void NotifyOnCancel(google::protobuf::Closure*) override {}
};

/// Signals a promise when Run() is called, allowing the test thread to block
/// until the RPC closure fires.
class BlockingClosure: public google::protobuf::Closure {
public:
    void Run() override {
        p_.set_value();
    }

    bool waitFor(std::chrono::milliseconds timeout) {
        return future_.wait_for(timeout) == std::future_status::ready;
    }

private:
    std::promise<void> p_;
    std::future<void>  future_ = p_.get_future();
};

// ---------------------------------------------------------------------------
// Proto helpers
// ---------------------------------------------------------------------------

static void addRequestBlock(::tcp_transfer::TcpLayerBlockTransferRequest&        request,
                            int64_t                                              cache_key,
                            const std::vector<std::pair<uint32_t, std::string>>& sub_blocks) {
    auto* kb = request.add_blocks();
    kb->set_key(cache_key);
    for (const auto& [len, content] : sub_blocks) {
        auto* b = kb->add_blocks();
        b->set_len(len);
        b->set_content(content);
    }
}

static ::tcp_transfer::TcpLayerBlockTransferRequest makeRequest(const std::string& unique_key, int64_t deadline_ms) {
    ::tcp_transfer::TcpLayerBlockTransferRequest req;
    req.set_unique_key(unique_key);
    req.set_deadline_ms(deadline_ms);
    return req;
}

// ---------------------------------------------------------------------------
// Task / block helpers
// ---------------------------------------------------------------------------

static KeyBlockInfoMap makeBlocks(int64_t cache_key, void* addr, size_t size) {
    auto kbi = std::make_shared<KeyBlockInfo>();
    kbi->blocks.push_back(BlockInfo{false, 0, 0, addr, size});
    KeyBlockInfoMap m;
    m[cache_key] = kbi;
    return m;
}

static KeyBlockInfoMap makeTwoSubBlocks(int64_t cache_key, void* addr0, size_t sz0, void* addr1, size_t sz1) {
    auto kbi = std::make_shared<KeyBlockInfo>();
    kbi->blocks.push_back(BlockInfo{false, 0, 0, addr0, sz0});
    kbi->blocks.push_back(BlockInfo{false, 0, 0, addr1, sz1});
    KeyBlockInfoMap m;
    m[cache_key] = kbi;
    return m;
}

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

class TcpTransferServiceTest: public ::testing::Test {
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
        initExecCtx(parallelism_config,
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
        device_initialized_ = isRuntimeInitialized();
    }

protected:
    void SetUp() override {
        task_store_ = std::make_shared<TransferTaskStore>();
        service_    = std::make_shared<TcpTransferService>(task_store_);
        ASSERT_TRUE(service_->init(1000 /* 1ms */, 4));
    }

    void TearDown() override {
        service_.reset();
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
        return std::memcmp(cpu_tensor.data_ptr(), expected.data(), size) == 0;
    }

    // Issue one transfer RPC and return the closure for waiting.
    std::unique_ptr<BlockingClosure> issueTransfer(const ::tcp_transfer::TcpLayerBlockTransferRequest* req,
                                                   ::tcp_transfer::TcpLayerBlockTransferResponse*      resp) {
        auto closure = std::make_unique<BlockingClosure>();
        service_->transfer(&ctrl_, req, resp, closure.get());
        return closure;
    }

    static bool                         device_initialized_;
    MockRpcController                   ctrl_;
    std::shared_ptr<TransferTaskStore>  task_store_;
    std::shared_ptr<TcpTransferService> service_;
};

bool TcpTransferServiceTest::device_initialized_ = false;

static constexpr auto kWait = std::chrono::seconds(5);

// ===========================================================================
// A. Happy Path
// ===========================================================================

// A1: Receiver registers task first, then Sender transfer arrives.
TEST_F(TcpTransferServiceTest, A1_NormalTransfer_RecvFirst_Success) {
    if (!device_initialized_) {
        GTEST_SKIP() << "No GPU device";
    }

    constexpr size_t  size = 64;
    const std::string content(size, 'A');
    auto              gpu_buf = allocDevice(size);

    auto task = task_store_->addTask("k1", makeBlocks(1, gpu_buf.data_ptr(), size), currentTimeMs() + 5000);
    ASSERT_NE(task, nullptr);

    auto req = makeRequest("k1", currentTimeMs() + 5000);
    addRequestBlock(req, 1, {{size, content}});

    ::tcp_transfer::TcpLayerBlockTransferResponse resp;
    auto                                          closure = issueTransfer(&req, &resp);

    ASSERT_TRUE(closure->waitFor(kWait));
    EXPECT_EQ(resp.error_code(), ::tcp_transfer::TCP_TRANSFER_NONE_ERROR);
    EXPECT_TRUE(task->success());
    EXPECT_TRUE(verifyDevice(gpu_buf.data_ptr(), content));
}

// A2: Sender transfer arrives first; Receiver registers task afterwards.
TEST_F(TcpTransferServiceTest, A2_NormalTransfer_TransferFirst_RecvLater_Success) {
    if (!device_initialized_) {
        GTEST_SKIP() << "No GPU device";
    }

    constexpr size_t  size = 64;
    const std::string content(size, 'B');
    auto              gpu_buf = allocDevice(size);

    auto req = makeRequest("k2", currentTimeMs() + 5000);
    addRequestBlock(req, 1, {{size, content}});

    ::tcp_transfer::TcpLayerBlockTransferResponse resp;
    auto                                          closure = issueTransfer(&req, &resp);

    // Register task AFTER the RPC is already waiting.
    auto task = task_store_->addTask("k2", makeBlocks(1, gpu_buf.data_ptr(), size), currentTimeMs() + 5000);
    ASSERT_NE(task, nullptr);

    ASSERT_TRUE(closure->waitFor(kWait));
    EXPECT_EQ(resp.error_code(), ::tcp_transfer::TCP_TRANSFER_NONE_ERROR);
    EXPECT_TRUE(task->success());
    EXPECT_TRUE(verifyDevice(gpu_buf.data_ptr(), content));
}

// ===========================================================================
// B. Timeout
// ===========================================================================

// B1: No matching recv task; context deadline is already past → TIMEOUT.
TEST_F(TcpTransferServiceTest, B1_ContextTimeout_NoMatchingRecvTask) {
    auto                                          req = makeRequest("k_b1", currentTimeMs() - 1);  // already expired
    ::tcp_transfer::TcpLayerBlockTransferResponse resp;
    auto                                          closure = issueTransfer(&req, &resp);

    ASSERT_TRUE(closure->waitFor(kWait));
    EXPECT_EQ(resp.error_code(), ::tcp_transfer::TCP_TRANSFER_CONTEXT_TIMEOUT);
}

// B3: Task exists but its deadline is already past; notifyDone overrides any
//     lower-level error with TIMEOUT.
TEST_F(TcpTransferServiceTest, B3_ContextTimeout_OverridesErrorOnNotifyDone) {
    char dummy[64]{};

    // Task has expired deadline but done_=false, so waitCheckProc still matches it.
    auto task = task_store_->addTask("k_b3", makeBlocks(1, dummy, 64),
                                     currentTimeMs() - 1);  // expired
    ASSERT_NE(task, nullptr);

    // Request uses size=32 → mismatch prevents actual GPU copy, keeping the test GPU-free.
    auto req = makeRequest("k_b3", currentTimeMs() + 5000);
    addRequestBlock(req, 1, {{32, std::string(32, 'X')}});

    ::tcp_transfer::TcpLayerBlockTransferResponse resp;
    auto                                          closure = issueTransfer(&req, &resp);

    ASSERT_TRUE(closure->waitFor(kWait));
    // notifyDone sees isTimeout()=true → TIMEOUT overrides BUFFER_MISMATCH.
    EXPECT_EQ(resp.error_code(), ::tcp_transfer::TCP_TRANSFER_CONTEXT_TIMEOUT);
}

// ===========================================================================
// C. Cancel
// ===========================================================================

// C1: Task cancelled in PENDING state before the RPC arrives.
TEST_F(TcpTransferServiceTest, C1_RecvCancelledPending_ReturnsCancelled) {
    auto task = task_store_->addTask("k_c1", {}, currentTimeMs() + 5000);
    ASSERT_NE(task, nullptr);
    task->cancel();  // PENDING → DONE(CANCELLED)

    auto                                          req = makeRequest("k_c1", currentTimeMs() + 5000);
    ::tcp_transfer::TcpLayerBlockTransferResponse resp;
    auto                                          closure = issueTransfer(&req, &resp);

    ASSERT_TRUE(closure->waitFor(kWait));
    EXPECT_EQ(resp.error_code(), ::tcp_transfer::TCP_TRANSFER_TASK_CANCELLED);
}

// C2: Task is in TRANSFERRING state with cancel_requested_=true when
//     notifyDone is called → CANCELLED overrides BUFFER_MISMATCH.
TEST_F(TcpTransferServiceTest, C2_CancelDuringTransferring_ReturnsCancelled) {
    char dummy[64]{};

    auto task = task_store_->addTask("k_c2", makeBlocks(1, dummy, 64), currentTimeMs() + 5000);
    ASSERT_NE(task, nullptr);

    // Manually advance the task to TRANSFERRING and record cancel intent.
    task->startTransfer();
    task->cancel();  // sets cancel_requested_=true (not done yet)

    // Size mismatch forces executeCopy to fail without touching the GPU,
    // triggering notifyDone(false, BUFFER_MISMATCH) which is then overridden
    // by cancel_requested_=true → CANCELLED.
    auto req = makeRequest("k_c2", currentTimeMs() + 5000);
    addRequestBlock(req, 1, {{32, std::string(32, 'Y')}});

    ::tcp_transfer::TcpLayerBlockTransferResponse resp;
    auto                                          closure = issueTransfer(&req, &resp);

    ASSERT_TRUE(closure->waitFor(kWait));
    EXPECT_EQ(resp.error_code(), ::tcp_transfer::TCP_TRANSFER_TASK_CANCELLED);
}

// C4: Service destructor synchronously cancels all pending contexts.
TEST_F(TcpTransferServiceTest, C4_ServiceDestruction_CancelsPendingContexts) {
    // No task registered → context stays in wait_tasks_ indefinitely.
    auto                                          req = makeRequest("k_c4", currentTimeMs() + 60000);
    ::tcp_transfer::TcpLayerBlockTransferResponse resp;
    auto                                          closure = issueTransfer(&req, &resp);

    service_.reset();  // destructor fires synchronously

    // Closure must have been called by the destructor.
    ASSERT_TRUE(closure->waitFor(std::chrono::milliseconds(100)));
    EXPECT_EQ(resp.error_code(), ::tcp_transfer::TCP_TRANSFER_TASK_CANCELLED);
}

// ===========================================================================
// E. Buffer Format Errors
// ===========================================================================

// E1: Block size in request differs from what the receiver registered → BUFFER_MISMATCH.
TEST_F(TcpTransferServiceTest, E1_BufferMismatch_SizeMismatch) {
    char dummy[64]{};
    auto task = task_store_->addTask("k_e1", makeBlocks(1, dummy, 64), currentTimeMs() + 5000);
    ASSERT_NE(task, nullptr);

    auto req = makeRequest("k_e1", currentTimeMs() + 5000);
    addRequestBlock(req, 1, {{32, std::string(32, 'Z')}});  // 32 != 64

    ::tcp_transfer::TcpLayerBlockTransferResponse resp;
    auto                                          closure = issueTransfer(&req, &resp);

    ASSERT_TRUE(closure->waitFor(kWait));
    EXPECT_EQ(resp.error_code(), ::tcp_transfer::TCP_TRANSFER_BUFFER_MISMATCH);
}

// E2: cache_key present in receiver task is absent in the sender request → BUFFER_MISMATCH.
TEST_F(TcpTransferServiceTest, E2_BufferMismatch_ReceiverKeyMissingInSender) {
    char dummy[64]{};
    auto task = task_store_->addTask("k_e2", makeBlocks(1, dummy, 64), currentTimeMs() + 5000);
    ASSERT_NE(task, nullptr);

    auto req = makeRequest("k_e2", currentTimeMs() + 5000);
    addRequestBlock(req, 2, {{64, std::string(64, 'W')}});  // key=2, task expects key=1

    ::tcp_transfer::TcpLayerBlockTransferResponse resp;
    auto                                          closure = issueTransfer(&req, &resp);

    ASSERT_TRUE(closure->waitFor(kWait));
    EXPECT_EQ(resp.error_code(), ::tcp_transfer::TCP_TRANSFER_BUFFER_MISMATCH);
}

// E2b: Sender sends an extra cache_key not registered by receiver; it is
//      silently ignored and the transfer succeeds.
TEST_F(TcpTransferServiceTest, E2b_ExtraKeyInSender_IgnoredOnSuccess) {
    if (!device_initialized_) {
        GTEST_SKIP() << "No GPU device";
    }

    constexpr size_t  size = 64;
    const std::string content(size, 'V');
    auto              gpu_buf = allocDevice(size);

    auto task = task_store_->addTask("k_e2b", makeBlocks(1, gpu_buf.data_ptr(), size), currentTimeMs() + 5000);
    ASSERT_NE(task, nullptr);

    auto req = makeRequest("k_e2b", currentTimeMs() + 5000);
    addRequestBlock(req, 1, {{size, content}});                 // key=1: expected
    addRequestBlock(req, 2, {{size, std::string(size, '?')}});  // key=2: extra, ignored

    ::tcp_transfer::TcpLayerBlockTransferResponse resp;
    auto                                          closure = issueTransfer(&req, &resp);

    ASSERT_TRUE(closure->waitFor(kWait));
    EXPECT_EQ(resp.error_code(), ::tcp_transfer::TCP_TRANSFER_NONE_ERROR);
    EXPECT_TRUE(task->success());
    EXPECT_TRUE(verifyDevice(gpu_buf.data_ptr(), content));
}

// E3: Receiver task has more sub-blocks than the sender request provides → BUFFER_MISMATCH.
TEST_F(TcpTransferServiceTest, E3_BufferMismatch_SubBlockOutOfRange) {
    char dummy[128]{};

    auto task = task_store_->addTask("k_e3", makeTwoSubBlocks(1, dummy, 64, dummy + 64, 64), currentTimeMs() + 5000);
    ASSERT_NE(task, nullptr);

    // Request only provides sub_block[0]; sub_block[1] is missing.
    auto req = makeRequest("k_e3", currentTimeMs() + 5000);
    addRequestBlock(req, 1, {{64, std::string(64, 'U')}});

    ::tcp_transfer::TcpLayerBlockTransferResponse resp;
    auto                                          closure = issueTransfer(&req, &resp);

    ASSERT_TRUE(closure->waitFor(kWait));
    EXPECT_EQ(resp.error_code(), ::tcp_transfer::TCP_TRANSFER_BUFFER_MISMATCH);
}

// ===========================================================================
// G. Concurrency
// ===========================================================================

// G3: Two concurrent transfer RPCs for the same unique_key both match the
//     same task; startTransfer() has no exclusive lock, so both proceed.
//     Both closures must fire and the task must end up in a consistent state.
TEST_F(TcpTransferServiceTest, G3_ConcurrentTransfers_SameKey_BothComplete) {
    if (!device_initialized_) {
        GTEST_SKIP() << "No GPU device";
    }

    constexpr size_t  size = 64;
    const std::string content(size, 'G');
    auto              gpu_buf = allocDevice(size);

    auto task = task_store_->addTask("k_g3", makeBlocks(1, gpu_buf.data_ptr(), size), currentTimeMs() + 5000);
    ASSERT_NE(task, nullptr);

    auto req1 = makeRequest("k_g3", currentTimeMs() + 5000);
    auto req2 = makeRequest("k_g3", currentTimeMs() + 5000);
    addRequestBlock(req1, 1, {{size, content}});
    addRequestBlock(req2, 1, {{size, content}});

    ::tcp_transfer::TcpLayerBlockTransferResponse resp1, resp2;
    auto                                          closure1 = issueTransfer(&req1, &resp1);
    auto                                          closure2 = issueTransfer(&req2, &resp2);

    ASSERT_TRUE(closure1->waitFor(kWait));
    ASSERT_TRUE(closure2->waitFor(kWait));

    // Both must complete without crash; task must be in a terminal state.
    EXPECT_TRUE(task->done());
    EXPECT_TRUE(task->success());
    EXPECT_EQ(resp1.error_code(), ::tcp_transfer::TCP_TRANSFER_NONE_ERROR);
    EXPECT_EQ(resp2.error_code(), ::tcp_transfer::TCP_TRANSFER_NONE_ERROR);
}

}  // namespace tcp
}  // namespace transfer
}  // namespace rtp_llm
