#include <gtest/gtest.h>

#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/BlockInfo.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/TransferErrorCode.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/TransferTask.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/Types.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/tcp/CudaCopyUtil.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/tcp/TcpTaskContext.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/tcp/test/DeviceReserveForTest.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/tcp/proto/tcp_service.pb.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/core/Buffer.h"
#include "rtp_llm/cpp/devices/DeviceFactory.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"

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

/// Counts how many times Run() has been called — the sole observable side-effect
/// of completing (or abandoning) an RPC closure.
class MockClosure: public google::protobuf::Closure {
public:
    void Run() override {
        ++run_count_;
    }
    int run_count() const {
        return run_count_;
    }

private:
    int run_count_ = 0;
};

// ---------------------------------------------------------------------------
// Helpers for building proto objects
// ---------------------------------------------------------------------------

/// Appends one TcpCacheKeyBlockBufferInfo entry to @p request.
/// @param sub_blocks  list of (len, content) pairs, one per sub-block.
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

// ---------------------------------------------------------------------------
// Test fixture
// ---------------------------------------------------------------------------

class TcpTaskContextTest: public ::testing::Test {
public:
    /// Initialize the device once for the whole test suite.
    /// GPU tests are skipped when no device is available.
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
    // ----- Request builders -----

    /// Returns a minimal request with a single cache_key and one sub-block.
    static ::tcp_transfer::TcpLayerBlockTransferRequest makeSimpleRequest(const std::string& unique_key,
                                                                          int64_t            deadline_ms,
                                                                          int64_t            cache_key,
                                                                          uint32_t           len,
                                                                          const std::string& content) {
        ::tcp_transfer::TcpLayerBlockTransferRequest req;
        req.set_unique_key(unique_key);
        req.set_deadline_ms(deadline_ms);
        addRequestBlock(req, cache_key, {{len, content}});
        return req;
    }

    /// Returns an empty request (no blocks) with the given key and deadline.
    static ::tcp_transfer::TcpLayerBlockTransferRequest makeEmptyRequest(const std::string& unique_key,
                                                                         int64_t            deadline_ms) {
        ::tcp_transfer::TcpLayerBlockTransferRequest req;
        req.set_unique_key(unique_key);
        req.set_deadline_ms(deadline_ms);
        return req;
    }

    // ----- Task builder -----

    /// Returns a TransferTask whose sole block has addr/size as given.
    static std::shared_ptr<TransferTask>
    makeTask(int64_t cache_key, void* addr, size_t size_bytes, int64_t deadline_offset_ms = 5000) {
        auto kbi = std::make_shared<KeyBlockInfo>();
        kbi->blocks.push_back(BlockInfo{false, 0, 0, addr, size_bytes});
        KeyBlockInfoMap block_infos;
        block_infos[cache_key] = kbi;
        return std::make_shared<TransferTask>(std::move(block_infos), currentTimeMs() + deadline_offset_ms);
    }

    // ----- Context builder -----

    std::unique_ptr<TcpTaskContext> makeContext(const ::tcp_transfer::TcpLayerBlockTransferRequest* request,
                                                ::tcp_transfer::TcpLayerBlockTransferResponse*      response,
                                                MockClosure*                                        closure) {
        return std::make_unique<TcpTaskContext>(&controller_, request, response, closure, nullptr);
    }

    // ----- GPU helpers -----

    /// Allocates a device buffer of @p size bytes.
    /// Returns nullptr if device_ is null.
    BufferPtr allocDeviceBuffer(size_t size) {
        if (!device_) {
            return nullptr;
        }
        return device_->allocateBuffer({DataType::TYPE_UINT8, {size}, AllocationType::DEVICE});
    }

    /// Copies @p size bytes from @p device_ptr back to host and compares against @p expected.
    bool verifyDeviceData(void* device_ptr, const std::string& expected, size_t size) {
        auto cpu_buf = device_->allocateBuffer({DataType::TYPE_UINT8, {size}, AllocationType::HOST});
        auto src     = Buffer(MemoryType::MEMORY_GPU, DataType::TYPE_UINT8, {size}, device_ptr);
        auto dst     = Buffer(MemoryType::MEMORY_CPU, DataType::TYPE_UINT8, {size}, cpu_buf->data());
        device_->copy({dst, src});
        device_->syncAndCheck();
        return std::memcmp(cpu_buf->data(), expected.data(), size) == 0;
    }

    static DeviceBase* device_;
    MockRpcController  controller_;
};

DeviceBase* TcpTaskContextTest::device_ = nullptr;

// ===========================================================================
// Group 1: startTransfer()
// ===========================================================================

TEST_F(TcpTaskContextTest, StartTransfer_TaskNull_ReturnsFalse) {
    auto                                          req = makeEmptyRequest("k1", currentTimeMs() + 5000);
    ::tcp_transfer::TcpLayerBlockTransferResponse resp;
    MockClosure                                   closure;
    auto                                          ctx = makeContext(&req, &resp, &closure);

    EXPECT_FALSE(ctx->startTransfer());
}

TEST_F(TcpTaskContextTest, StartTransfer_TaskPending_ReturnsTrue) {
    auto                                          req = makeEmptyRequest("k1", currentTimeMs() + 5000);
    ::tcp_transfer::TcpLayerBlockTransferResponse resp;
    MockClosure                                   closure;
    auto                                          ctx = makeContext(&req, &resp, &closure);

    auto task = makeTask(1, nullptr, 0);
    ctx->setTask(task);

    EXPECT_TRUE(ctx->startTransfer());
}

TEST_F(TcpTaskContextTest, StartTransfer_TaskAlreadyCancelled_ReturnsFalse) {
    auto                                          req = makeEmptyRequest("k1", currentTimeMs() + 5000);
    ::tcp_transfer::TcpLayerBlockTransferResponse resp;
    MockClosure                                   closure;
    auto                                          ctx = makeContext(&req, &resp, &closure);

    auto task = makeTask(1, nullptr, 0);
    task->cancel();  // PENDING → DONE(CANCELLED)
    ctx->setTask(task);

    EXPECT_FALSE(ctx->startTransfer());
}

// ===========================================================================
// Group 2: executeCopy() — failure paths (no GPU required)
// ===========================================================================

TEST_F(TcpTaskContextTest, ExecuteCopy_TaskNull_ReturnsFalse) {
    auto                                          req = makeEmptyRequest("k1", currentTimeMs() + 5000);
    ::tcp_transfer::TcpLayerBlockTransferResponse resp;
    MockClosure                                   closure;
    auto                                          ctx = makeContext(&req, &resp, &closure);

    CudaCopyUtil copy_util;
    EXPECT_FALSE(ctx->executeCopy(copy_util));
}

TEST_F(TcpTaskContextTest, ExecuteCopy_CacheKeyMissingInRequest_ReturnsFalse) {
    // Request contains cache_key=99, but task expects cache_key=1.
    ::tcp_transfer::TcpLayerBlockTransferRequest req;
    req.set_unique_key("k1");
    req.set_deadline_ms(currentTimeMs() + 5000);
    addRequestBlock(req, 99, {{64, std::string(64, 'A')}});

    ::tcp_transfer::TcpLayerBlockTransferResponse resp;
    MockClosure                                   closure;
    auto                                          ctx = makeContext(&req, &resp, &closure);

    char dummy[64];
    auto task = makeTask(1, dummy, 64);
    ctx->setTask(task);

    CudaCopyUtil copy_util;
    EXPECT_FALSE(ctx->executeCopy(copy_util));
}

TEST_F(TcpTaskContextTest, ExecuteCopy_SubBlockOutOfRange_ReturnsFalse) {
    // Task has 2 sub_blocks for cache_key=1; request only has 1.
    constexpr int64_t key  = 1;
    constexpr size_t  size = 64;

    ::tcp_transfer::TcpLayerBlockTransferRequest req;
    req.set_unique_key("k1");
    req.set_deadline_ms(currentTimeMs() + 5000);
    addRequestBlock(req, key, {{size, std::string(size, 'B')}});  // only 1 sub-block

    ::tcp_transfer::TcpLayerBlockTransferResponse resp;
    MockClosure                                   closure;
    auto                                          ctx = makeContext(&req, &resp, &closure);

    auto kbi = std::make_shared<KeyBlockInfo>();
    char dummy[size];
    kbi->blocks.push_back(BlockInfo{false, 0, 0, dummy, size});  // index 0 — present in request
    kbi->blocks.push_back(BlockInfo{false, 0, 0, dummy, size});  // index 1 — NOT in request
    KeyBlockInfoMap block_infos;
    block_infos[key] = kbi;
    auto task        = std::make_shared<TransferTask>(std::move(block_infos), currentTimeMs() + 5000);
    ctx->setTask(task);

    CudaCopyUtil copy_util;
    EXPECT_FALSE(ctx->executeCopy(copy_util));
}

TEST_F(TcpTaskContextTest, ExecuteCopy_SizeMismatch_ReturnsFalse) {
    constexpr int64_t key      = 1;
    constexpr size_t  task_sz  = 64;
    constexpr size_t  proto_sz = 32;  // intentional mismatch

    char dummy[task_sz];
    auto req = makeSimpleRequest("k1", currentTimeMs() + 5000, key, proto_sz, std::string(proto_sz, 'C'));

    ::tcp_transfer::TcpLayerBlockTransferResponse resp;
    MockClosure                                   closure;
    auto                                          ctx = makeContext(&req, &resp, &closure);
    ctx->setTask(makeTask(key, dummy, task_sz));

    CudaCopyUtil copy_util;
    EXPECT_FALSE(ctx->executeCopy(copy_util));
}

TEST_F(TcpTaskContextTest, ExecuteCopy_AllBlocksEmpty_ReturnsFalse) {
    // All BlockInfo entries have addr==nullptr / size_bytes==0 → copy_tasks stays empty.
    constexpr int64_t key = 1;

    ::tcp_transfer::TcpLayerBlockTransferRequest req;
    req.set_unique_key("k1");
    req.set_deadline_ms(currentTimeMs() + 5000);
    addRequestBlock(req, key, {{64, std::string(64, 'D')}});

    ::tcp_transfer::TcpLayerBlockTransferResponse resp;
    MockClosure                                   closure;
    auto                                          ctx = makeContext(&req, &resp, &closure);
    ctx->setTask(makeTask(key, nullptr, 0));

    CudaCopyUtil copy_util;
    EXPECT_FALSE(ctx->executeCopy(copy_util));
}

// ===========================================================================
// Group 2: executeCopy() — success paths (real GPU memory)
// ===========================================================================

TEST_F(TcpTaskContextTest, ExecuteCopy_ValidData_DataCopiedToDevice) {
    if (!device_) {
        GTEST_SKIP() << "No GPU device available";
    }

    constexpr int64_t key       = 1;
    constexpr size_t  data_size = 64;
    const std::string content(data_size, 'X');

    auto gpu_buf = allocDeviceBuffer(data_size);
    auto req     = makeSimpleRequest("k1", currentTimeMs() + 5000, key, data_size, content);

    ::tcp_transfer::TcpLayerBlockTransferResponse resp;
    MockClosure                                   closure;
    auto                                          ctx = makeContext(&req, &resp, &closure);
    ctx->setTask(makeTask(key, gpu_buf->data(), data_size));

    CudaCopyUtil copy_util;
    ASSERT_TRUE(ctx->executeCopy(copy_util));
    EXPECT_TRUE(verifyDeviceData(gpu_buf->data(), content, data_size));
}

TEST_F(TcpTaskContextTest, ExecuteCopy_MixedBlocks_SkipsNullAddrBlocks) {
    if (!device_) {
        GTEST_SKIP() << "No GPU device available";
    }

    constexpr int64_t key       = 1;
    constexpr size_t  data_size = 32;
    const std::string content(data_size, 'Y');

    auto gpu_buf = allocDeviceBuffer(data_size);

    // Request: 2 sub_blocks for cache_key=1.
    // Index 0: empty (len=0) — skipped by the task-side null check.
    // Index 1: real data.
    ::tcp_transfer::TcpLayerBlockTransferRequest req;
    req.set_unique_key("k1");
    req.set_deadline_ms(currentTimeMs() + 5000);
    addRequestBlock(req, key, {{0, ""}, {data_size, content}});

    ::tcp_transfer::TcpLayerBlockTransferResponse resp;
    MockClosure                                   closure;
    auto                                          ctx = makeContext(&req, &resp, &closure);

    // Task: sub_block[0] has addr==nullptr (skipped), sub_block[1] is valid.
    auto kbi = std::make_shared<KeyBlockInfo>();
    kbi->blocks.push_back(BlockInfo{false, 0, 0, nullptr, 0});  // skipped
    kbi->blocks.push_back(BlockInfo{false, 0, 0, gpu_buf->data(), data_size});
    KeyBlockInfoMap block_infos;
    block_infos[key] = kbi;
    auto task        = std::make_shared<TransferTask>(std::move(block_infos), currentTimeMs() + 5000);
    ctx->setTask(task);

    CudaCopyUtil copy_util;
    ASSERT_TRUE(ctx->executeCopy(copy_util));
    EXPECT_TRUE(verifyDeviceData(gpu_buf->data(), content, data_size));
}

// ===========================================================================
// Group 3: run() — basic response / closure semantics (no task)
// ===========================================================================

TEST_F(TcpTaskContextTest, Run_Success_SetsResponseNoneError) {
    auto                                          req = makeEmptyRequest("k1", currentTimeMs() + 5000);
    ::tcp_transfer::TcpLayerBlockTransferResponse resp;
    MockClosure                                   closure;
    auto                                          ctx = makeContext(&req, &resp, &closure);

    ctx->run(true);

    EXPECT_EQ(resp.error_code(), ::tcp_transfer::TCP_TRANSFER_NONE_ERROR);
    EXPECT_EQ(closure.run_count(), 1);
}

TEST_F(TcpTaskContextTest, Run_Failure_Timeout_SetsProtoTimeout) {
    auto                                          req = makeEmptyRequest("k1", currentTimeMs() + 5000);
    ::tcp_transfer::TcpLayerBlockTransferResponse resp;
    MockClosure                                   closure;
    auto                                          ctx = makeContext(&req, &resp, &closure);

    ctx->run(false, TransferErrorCode::TIMEOUT);

    EXPECT_EQ(resp.error_code(), ::tcp_transfer::TCP_TRANSFER_CONTEXT_TIMEOUT);
    EXPECT_EQ(closure.run_count(), 1);
}

TEST_F(TcpTaskContextTest, Run_Failure_Cancelled_SetsProtoCancelled) {
    auto                                          req = makeEmptyRequest("k1", currentTimeMs() + 5000);
    ::tcp_transfer::TcpLayerBlockTransferResponse resp;
    MockClosure                                   closure;
    auto                                          ctx = makeContext(&req, &resp, &closure);

    ctx->run(false, TransferErrorCode::CANCELLED);

    EXPECT_EQ(resp.error_code(), ::tcp_transfer::TCP_TRANSFER_TASK_CANCELLED);
}

TEST_F(TcpTaskContextTest, Run_Failure_BufferMismatch_SetsProtoBufferMismatch) {
    auto                                          req = makeEmptyRequest("k1", currentTimeMs() + 5000);
    ::tcp_transfer::TcpLayerBlockTransferResponse resp;
    MockClosure                                   closure;
    auto                                          ctx = makeContext(&req, &resp, &closure);

    ctx->run(false, TransferErrorCode::BUFFER_MISMATCH);

    EXPECT_EQ(resp.error_code(), ::tcp_transfer::TCP_TRANSFER_BUFFER_MISMATCH);
}

TEST_F(TcpTaskContextTest, Run_Failure_Unknown_SetsProtoUnknown) {
    auto                                          req = makeEmptyRequest("k1", currentTimeMs() + 5000);
    ::tcp_transfer::TcpLayerBlockTransferResponse resp;
    MockClosure                                   closure;
    auto                                          ctx = makeContext(&req, &resp, &closure);

    ctx->run(false, TransferErrorCode::UNKNOWN);

    EXPECT_EQ(resp.error_code(), ::tcp_transfer::TCP_TRANSFER_UNKNOWN_ERROR);
}

TEST_F(TcpTaskContextTest, Run_Idempotent_SecondCallIsNoOp) {
    auto                                          req = makeEmptyRequest("k1", currentTimeMs() + 5000);
    ::tcp_transfer::TcpLayerBlockTransferResponse resp;
    MockClosure                                   closure;
    auto                                          ctx = makeContext(&req, &resp, &closure);

    ctx->run(true);
    ctx->run(false, TransferErrorCode::CANCELLED, "must be ignored");  // done_ is already nullptr

    EXPECT_EQ(closure.run_count(), 1);
    EXPECT_EQ(resp.error_code(), ::tcp_transfer::TCP_TRANSFER_NONE_ERROR);
}

// ===========================================================================
// Group 3: run() — with task (errorCode read-back semantics)
// ===========================================================================

TEST_F(TcpTaskContextTest, Run_WithTask_CancelInTransferring_OverridesSuccess) {
    auto                                          req = makeEmptyRequest("k1", currentTimeMs() + 5000);
    ::tcp_transfer::TcpLayerBlockTransferResponse resp;
    MockClosure                                   closure;
    auto                                          ctx = makeContext(&req, &resp, &closure);

    auto task = makeTask(1, nullptr, 0);
    ctx->setTask(task);
    task->startTransfer();  // PENDING → TRANSFERRING
    task->cancel();         // sets cancel_requested_=true; task is NOT done yet

    // run(true) calls notifyDone(true) internally; the cancel flag overrides the success.
    ctx->run(true);

    EXPECT_EQ(resp.error_code(), ::tcp_transfer::TCP_TRANSFER_TASK_CANCELLED);
    EXPECT_EQ(closure.run_count(), 1);
}

TEST_F(TcpTaskContextTest, Run_WithTask_ExpiredDeadline_OverridesSuccess) {
    auto                                          req = makeEmptyRequest("k1", currentTimeMs() + 5000);
    ::tcp_transfer::TcpLayerBlockTransferResponse resp;
    MockClosure                                   closure;
    auto                                          ctx = makeContext(&req, &resp, &closure);

    // Create a task whose deadline is already in the past.
    KeyBlockInfoMap block_infos;
    block_infos[1] = std::make_shared<KeyBlockInfo>();
    auto task      = std::make_shared<TransferTask>(std::move(block_infos), currentTimeMs() - 100);
    ctx->setTask(task);
    task->startTransfer();

    // run(true) calls notifyDone(true) internally; isTimeout() overrides the result.
    ctx->run(true);

    EXPECT_EQ(resp.error_code(), ::tcp_transfer::TCP_TRANSFER_CONTEXT_TIMEOUT);
}

// ===========================================================================
// Group 4: destructor semantics
// ===========================================================================

TEST_F(TcpTaskContextTest, Destructor_WithPendingDone_AutoRunsFailure) {
    auto                                          req = makeEmptyRequest("k1", currentTimeMs() + 5000);
    ::tcp_transfer::TcpLayerBlockTransferResponse resp;
    MockClosure                                   closure;
    {
        auto ctx = makeContext(&req, &resp, &closure);
        // ctx destroyed here without run() being called.
    }

    EXPECT_EQ(closure.run_count(), 1);
    EXPECT_EQ(resp.error_code(), ::tcp_transfer::TCP_TRANSFER_UNKNOWN_ERROR);
}

TEST_F(TcpTaskContextTest, Destructor_AfterRunCompleted_NoDoubleRun) {
    auto                                          req = makeEmptyRequest("k1", currentTimeMs() + 5000);
    ::tcp_transfer::TcpLayerBlockTransferResponse resp;
    MockClosure                                   closure;
    {
        auto ctx = makeContext(&req, &resp, &closure);
        ctx->run(true);  // done_ set to nullptr
    }  // destructor: done_ is already nullptr → no second Run()

    EXPECT_EQ(closure.run_count(), 1);
}

}  // namespace tcp
}  // namespace transfer
}  // namespace rtp_llm
