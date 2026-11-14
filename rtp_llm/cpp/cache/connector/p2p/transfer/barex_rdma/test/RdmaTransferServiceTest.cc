#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <future>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <torch/torch.h>
#include <cuda_runtime.h>

#include "rtp_llm/cpp/cache/BlockInfo.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/TransferErrorCode.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/TransferTask.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/Types.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/barex_rdma/RdmaInterface.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/barex_rdma/RdmaTransferService.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/barex_rdma/proto/rdma_service.pb.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/models_py/bindings/cuda/cuda_host_utils.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"

namespace rtp_llm {
namespace transfer {
namespace barex_rdma {

// ---------------------------------------------------------------------------
// Mock RPC infrastructure
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
    std::future<void> future_ = p_.get_future();
};

class CudaEventGate {
public:
    bool init() {
        if (cudaStreamCreateWithFlags(&control_stream_, cudaStreamNonBlocking) != cudaSuccess)
            return false;
        if (cudaEventCreateWithFlags(&event_, cudaEventDisableTiming) != cudaSuccess)
            return false;
        if (cudaLaunchHostFunc(control_stream_, waitCallback, this) != cudaSuccess)
            return false;
        return cudaEventRecord(event_, control_stream_) == cudaSuccess;
    }

    ~CudaEventGate() {
        release();
        if (control_stream_)
            cudaStreamSynchronize(control_stream_);
        if (event_)
            cudaEventDestroy(event_);
        if (control_stream_)
            cudaStreamDestroy(control_stream_);
    }

    cudaEvent_t event() const {
        return event_;
    }

    void release() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            released_ = true;
        }
        cv_.notify_all();
    }

private:
    static void CUDART_CB waitCallback(void* opaque) {
        auto* gate = static_cast<CudaEventGate*>(opaque);
        std::unique_lock<std::mutex> lock(gate->mutex_);
        gate->cv_.wait(lock, [gate]() { return gate->released_; });
    }

private:
    std::mutex mutex_;
    std::condition_variable cv_;
    bool released_{false};
    cudaStream_t control_stream_{nullptr};
    cudaEvent_t event_{nullptr};
};

// ---------------------------------------------------------------------------
// Mock RDMA infrastructure
// ---------------------------------------------------------------------------

class MockRdmaConnection: public IRdmaConnection {
public:
    explicit MockRdmaConnection(bool success = true): success_(success) {}

    void close() override {
        close_count_.fetch_add(1);
    }

    void read(const std::vector<std::pair<torch::Tensor, std::shared_ptr<RemoteBuffer>>>& pairs,
              RdmaReadCallback cb,
              uint64_t) override {
        if (!success_) {
            cb(TransferErrorCode::RDMA_FAILED, "mock RDMA read failed");
            return;
        }
        for (const auto& [local, remote] : pairs) {
            std::memcpy(
                local.data_ptr(), reinterpret_cast<const void*>(static_cast<uintptr_t>(remote->addr)), remote->len);
        }
        cb(TransferErrorCode::OK, "");
    }

    std::atomic<int> close_count_{0};

private:
    bool success_;
};

class MockRdmaClient: public IRdmaClient {
public:
    explicit MockRdmaClient(bool conn_ok = true, bool read_ok = true):
        conn_(std::make_shared<MockRdmaConnection>(read_ok)), conn_ok_(conn_ok) {}

    std::shared_ptr<IRdmaConnection> getConnection(const std::string&, uint32_t) override {
        if (!conn_ok_)
            return nullptr;
        return conn_;
    }

    std::shared_ptr<MockRdmaConnection> conn_;

private:
    bool conn_ok_;
};

class MockRdmaMemoryManager: public IRdmaMemoryManager {
public:
    bool regUserMr(const BlockInfo&, uint64_t) override {
        ++reg_count;
        return true;
    }

    bool deregUserMr(const BlockInfo&) override {
        ++dereg_count;
        return true;
    }

    std::shared_ptr<RemoteBuffer> findMemoryMr(const BlockInfo&) override {
        return nullptr;
    }

    std::atomic<size_t> reg_count{0};
    std::atomic<size_t> dereg_count{0};
};

/// Mock connection that never fires the read callback — simulates lost RDMA CQE.
class HangingRdmaConnection: public IRdmaConnection {
public:
    void close() override {
        close_count_.fetch_add(1);
        completeAll(false);
    }

    void read(const std::vector<std::pair<torch::Tensor, std::shared_ptr<RemoteBuffer>>>&,
              RdmaReadCallback cb,
              uint64_t) override {
        std::lock_guard<std::mutex> lock(mu_);
        held_callbacks_.push_back(std::move(cb));
    }

    size_t heldCount() {
        std::lock_guard<std::mutex> lock(mu_);
        return held_callbacks_.size();
    }

    void completeAll(bool success) {
        std::vector<RdmaReadCallback> callbacks;
        {
            std::lock_guard<std::mutex> lock(mu_);
            callbacks.swap(held_callbacks_);
        }
        for (auto& callback : callbacks) {
            if (callback) {
                callback(success ? TransferErrorCode::OK : TransferErrorCode::RDMA_FAILED,
                         success ? "" : "mock connection closed");
            }
        }
    }

    void completeOne(size_t index, TransferErrorCode ec, const std::string& message) {
        RdmaReadCallback callback;
        {
            std::lock_guard<std::mutex> lock(mu_);
            callback = std::move(held_callbacks_.at(index));
        }
        callback(ec, message);
    }

    std::atomic<int> close_count_{0};

private:
    std::mutex mu_;
    std::vector<RdmaReadCallback> held_callbacks_;
};

class HangingRdmaClient: public IRdmaClient {
public:
    HangingRdmaClient(): conn_(std::make_shared<HangingRdmaConnection>()) {}

    std::shared_ptr<IRdmaConnection> getConnection(const std::string&, uint32_t) override {
        return conn_;
    }

    std::shared_ptr<HangingRdmaConnection> conn_;
};

// ---------------------------------------------------------------------------
// Proto helpers
// ---------------------------------------------------------------------------

static ::rdma_transfer::RdmaLayerBlockTransferRequest makeRequest(const std::string& unique_key,
                                                                  int64_t deadline_ms,
                                                                  const std::string& rdma_ip = "127.0.0.1",
                                                                  uint32_t rdma_port = 19875) {
    ::rdma_transfer::RdmaLayerBlockTransferRequest req;
    req.set_unique_key(unique_key);
    req.set_timeout_ms(deadline_ms - currentTimeMs());
    req.set_server_rdma_ip(rdma_ip);
    req.set_server_rdma_port(rdma_port);
    return req;
}

static void addRdmaBlock(::rdma_transfer::RdmaLayerBlockTransferRequest& req,
                         int64_t cache_key,
                         const void* src_addr,
                         uint32_t len) {
    auto* kb = req.add_blocks();
    kb->set_key(cache_key);
    auto* b = kb->add_blocks();
    b->set_len(len);
    auto* info = b->mutable_rdma_info();
    info->set_addr(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(src_addr)));
    auto* nr = info->add_nic_rkeys();
    nr->set_nicid(0);
    nr->set_rkey(0xDEADBEEF);
}

static void addRdmaSubBlock(::rdma_transfer::RdmaCacheKeyBlockBufferInfo* kb, const void* src_addr, uint32_t len) {
    auto* b = kb->add_blocks();
    b->set_len(len);
    auto* info = b->mutable_rdma_info();
    info->set_addr(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(src_addr)));
    auto* nr = info->add_nic_rkeys();
    nr->set_nicid(0);
    nr->set_rkey(0xDEADBEEF);
}

// ---------------------------------------------------------------------------
// Block helper
// ---------------------------------------------------------------------------

static KeyBlockInfoMap makeKvBlocks(int64_t cache_key, void* dst_addr, size_t size) {
    auto kbi = std::make_shared<KeyBlockInfo>();
    kbi->cache_key = cache_key;
    kbi->blocks.push_back(BlockInfo{false, 0, 0, dst_addr, size});
    KeyBlockInfoMap result;
    result[cache_key] = kbi;
    return result;
}

static KeyBlockInfoMap makeCudaKvBlocks(int64_t cache_key, const torch::Tensor& tensor) {
    auto kbi = std::make_shared<KeyBlockInfo>();
    kbi->cache_key = cache_key;
    kbi->blocks.push_back(BlockInfo{true,
                                    static_cast<int32_t>(tensor.device().index()),
                                    static_cast<int32_t>(tensor.scalar_type()),
                                    tensor.data_ptr(),
                                    tensor.nbytes()});
    KeyBlockInfoMap result;
    result[cache_key] = kbi;
    return result;
}

static KeyBlockInfoMap
makeKvScaleBlocks(int64_t cache_key, void* kv_addr, size_t kv_size, void* scale_addr, size_t scale_size) {
    auto kbi = std::make_shared<KeyBlockInfo>();
    kbi->cache_key = cache_key;
    kbi->blocks.push_back(BlockInfo{false, 0, 0, kv_addr, kv_size});
    kbi->blocks.push_back(BlockInfo{false, 0, 0, scale_addr, scale_size});
    KeyBlockInfoMap result;
    result[cache_key] = kbi;
    return result;
}

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

class RdmaTransferServiceTest: public ::testing::Test {
public:
    static void SetUpTestSuite() {
        initRuntime(/*device_id=*/0,
                    /*trace_memory=*/false,
                    /*enable_comm_overlap=*/false,
                    MlaOpsType::AUTO);
    }

protected:
    void SetUp() override {
        task_store_ = std::make_shared<TransferTaskStore>();
        rdma_client_ = std::make_shared<MockRdmaClient>();
        service_ = std::make_shared<RdmaTransferService>(task_store_, rdma_client_);
        ASSERT_TRUE(service_->init(1000 /* 1ms */, 4));
    }

    void TearDown() override {
        service_.reset();
    }

    std::unique_ptr<BlockingClosure> issueTransfer(const ::rdma_transfer::RdmaLayerBlockTransferRequest* req,
                                                   ::rdma_transfer::RdmaLayerBlockTransferResponse* resp) {
        auto closure = std::make_unique<BlockingClosure>();
        service_->transfer(&ctrl_, req, resp, closure.get());
        return closure;
    }

    MockRpcController ctrl_;
    std::shared_ptr<TransferTaskStore> task_store_;
    std::shared_ptr<MockRdmaClient> rdma_client_;
    std::shared_ptr<RdmaTransferService> service_;
};

static constexpr auto kWait = std::chrono::seconds(5);

// ===========================================================================
// A. Happy Path
// ===========================================================================

TEST_F(RdmaTransferServiceTest, A1_NormalTransfer_RecvFirst_Success) {
    constexpr size_t size = 64;
    char src[size];
    char dst[size]{};
    std::memset(src, 'A', size);

    auto task = task_store_->addTask("k_a1", makeKvBlocks(1, dst, size), currentTimeMs() + 5000);
    ASSERT_NE(task, nullptr);

    auto req = makeRequest("k_a1", currentTimeMs() + 5000);
    addRdmaBlock(req, 1, src, size);

    ::rdma_transfer::RdmaLayerBlockTransferResponse resp;
    auto closure = issueTransfer(&req, &resp);

    ASSERT_TRUE(closure->waitFor(kWait));
    EXPECT_EQ(resp.error_code(), ::rdma_transfer::RDMA_TRANSFER_NONE_ERROR);
    EXPECT_TRUE(task->success());
    EXPECT_EQ(std::memcmp(dst, src, size), 0);
}

TEST_F(RdmaTransferServiceTest, A2_NormalTransfer_TransferFirst_RecvLater_Success) {
    constexpr size_t size = 64;
    char src[size];
    char dst[size]{};
    std::memset(src, 'B', size);

    auto req = makeRequest("k_a2", currentTimeMs() + 5000);
    addRdmaBlock(req, 1, src, size);

    ::rdma_transfer::RdmaLayerBlockTransferResponse resp;
    auto closure = issueTransfer(&req, &resp);

    auto task = task_store_->addTask("k_a2", makeKvBlocks(1, dst, size), currentTimeMs() + 5000);
    ASSERT_NE(task, nullptr);

    ASSERT_TRUE(closure->waitFor(kWait));
    EXPECT_EQ(resp.error_code(), ::rdma_transfer::RDMA_TRANSFER_NONE_ERROR);
    EXPECT_TRUE(task->success());
    EXPECT_EQ(std::memcmp(dst, src, size), 0);
}

TEST_F(RdmaTransferServiceTest, StagingPoolTimeoutDoesNotRetainReleasedSlot) {
    auto memory_manager = std::make_shared<MockRdmaMemoryManager>();
    auto pool = RdmaStagingBufferPool::create(
        memory_manager, HostStagingBlockPool::kAlignment, HostStagingBlockPool::kAlignment);
    ASSERT_NE(pool, nullptr);
    auto leases = pool->mallocBatchWithBackoff(1, std::chrono::seconds(1));
    ASSERT_TRUE(leases.has_value());
    const auto original = leases->front().blockBuffer(64).base;

    // No slot is released while waiting: expiration must not depend on free().
    EXPECT_FALSE(pool->mallocBatchWithBackoff(1, std::chrono::milliseconds(1)).has_value());
    leases.reset();
    auto reused = pool->mallocBatchWithBackoff(1, std::chrono::seconds(1));
    ASSERT_TRUE(reused.has_value());
    EXPECT_EQ(reused->front().blockBuffer(64).base, original);
    reused.reset();

    pool.reset();
    EXPECT_EQ(memory_manager->reg_count.load(), 1);
    EXPECT_EQ(memory_manager->dereg_count.load(), 1);
}

TEST_F(RdmaTransferServiceTest, A3_RdmaReadToPinnedStagingThenH2d_Success) {
    service_.reset();
    auto memory_manager = std::make_shared<MockRdmaMemoryManager>();
    auto staging_pool = RdmaStagingBufferPool::create(
        memory_manager, HostStagingBlockPool::kAlignment, HostStagingBlockPool::kAlignment);
    ASSERT_NE(staging_pool, nullptr);
    ASSERT_EQ(memory_manager->reg_count.load(), 1);

    service_ = std::make_shared<RdmaTransferService>(task_store_, rdma_client_, nullptr, staging_pool);
    ASSERT_TRUE(service_->init(1000 /* 1ms */, 4));

    constexpr size_t size = 64;
    char src[size];
    std::memset(src, 'C', size);
    auto dst =
        torch::zeros({static_cast<int64_t>(size)}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
    auto task = task_store_->addTask("k_a3", makeCudaKvBlocks(1, dst), currentTimeMs() + 5000);
    ASSERT_NE(task, nullptr);

    auto req = makeRequest("k_a3", currentTimeMs() + 5000);
    addRdmaBlock(req, 1, src, size);

    ::rdma_transfer::RdmaLayerBlockTransferResponse resp;
    auto closure = issueTransfer(&req, &resp);

    ASSERT_TRUE(closure->waitFor(kWait));
    EXPECT_EQ(resp.error_code(), ::rdma_transfer::RDMA_TRANSFER_NONE_ERROR);
    EXPECT_TRUE(task->success());
    auto dst_cpu = dst.cpu();
    EXPECT_EQ(std::memcmp(dst_cpu.data_ptr(), src, size), 0);

    service_.reset();
    staging_pool.reset();
    EXPECT_EQ(memory_manager->dereg_count.load(), 1);
}

TEST_F(RdmaTransferServiceTest, A4_TotalBytesAllowTransferToSpanDerivedStagingBlocks) {
    service_.reset();
    auto memory_manager = std::make_shared<MockRdmaMemoryManager>();
    auto staging_pool = RdmaStagingBufferPool::create(
        memory_manager, 2 * HostStagingBlockPool::kAlignment, /*block_size_bytes=*/3072);
    ASSERT_NE(staging_pool, nullptr);
    EXPECT_EQ(staging_pool->strideBytes(), HostStagingBlockPool::kAlignment);
    EXPECT_EQ(staging_pool->blockCount(), 2);
    ASSERT_EQ(memory_manager->reg_count.load(), 2);

    service_ = std::make_shared<RdmaTransferService>(task_store_, rdma_client_, nullptr, staging_pool);
    ASSERT_TRUE(service_->init(1000 /* 1ms */, 4));

    constexpr size_t size = 3072;
    char src0[size];
    char src1[size];
    std::memset(src0, 'D', size);
    std::memset(src1, 'E', size);
    auto dst0 =
        torch::zeros({static_cast<int64_t>(size)}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
    auto dst1 =
        torch::zeros({static_cast<int64_t>(size)}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));

    auto blocks = makeCudaKvBlocks(1, dst0);
    blocks.emplace(2, makeCudaKvBlocks(2, dst1).at(2));
    auto task = task_store_->addTask("k_a4", std::move(blocks), currentTimeMs() + 5000);
    ASSERT_NE(task, nullptr);

    auto req = makeRequest("k_a4", currentTimeMs() + 5000);
    addRdmaBlock(req, 1, src0, size);
    addRdmaBlock(req, 2, src1, size);

    ::rdma_transfer::RdmaLayerBlockTransferResponse resp;
    auto closure = issueTransfer(&req, &resp);

    ASSERT_TRUE(closure->waitFor(kWait));
    EXPECT_EQ(resp.error_code(), ::rdma_transfer::RDMA_TRANSFER_NONE_ERROR);
    EXPECT_TRUE(task->success());
    auto dst0_cpu = dst0.cpu();
    auto dst1_cpu = dst1.cpu();
    EXPECT_EQ(std::memcmp(dst0_cpu.data_ptr(), src0, size), 0);
    EXPECT_EQ(std::memcmp(dst1_cpu.data_ptr(), src1, size), 0);

    service_.reset();
    staging_pool.reset();
    EXPECT_EQ(memory_manager->dereg_count.load(), 2);
}

// ===========================================================================
// B. Timeout
// ===========================================================================

TEST_F(RdmaTransferServiceTest, B1_ContextTimeout_NoMatchingRecvTask) {
    auto req = makeRequest("k_b1", currentTimeMs() - 1);
    ::rdma_transfer::RdmaLayerBlockTransferResponse resp;
    auto closure = issueTransfer(&req, &resp);

    ASSERT_TRUE(closure->waitFor(kWait));
    EXPECT_EQ(resp.error_code(), ::rdma_transfer::RDMA_TRANSFER_CONTEXT_TIMEOUT);
}

TEST_F(RdmaTransferServiceTest, B3_ReadFailureIsNotReclassifiedByElapsedTaskDeadline) {
    service_.reset();
    auto failing_client = std::make_shared<MockRdmaClient>(true /*conn*/, false /*read*/);
    service_ = std::make_shared<RdmaTransferService>(task_store_, failing_client);
    ASSERT_TRUE(service_->init(1000, 4));

    char dummy[64]{};
    auto task = task_store_->addTask("k_b3", makeKvBlocks(1, dummy, 64), currentTimeMs() - 1);
    ASSERT_NE(task, nullptr);

    auto req = makeRequest("k_b3", currentTimeMs() + 5000);
    addRdmaBlock(req, 1, dummy, 64);

    ::rdma_transfer::RdmaLayerBlockTransferResponse resp;
    auto closure = issueTransfer(&req, &resp);

    ASSERT_TRUE(closure->waitFor(kWait));
    EXPECT_EQ(resp.error_code(), ::rdma_transfer::RDMA_TRANSFER_RDMA_FAILED);
    EXPECT_NE(resp.error_message().find("mock RDMA read failed"), std::string::npos);
}

TEST_F(RdmaTransferServiceTest, B4_WatchdogRepliesBeforePhysicalCallbackButTaskRemainsInflight) {
    service_.reset();
    auto hanging_client = std::make_shared<HangingRdmaClient>();
    service_ = std::make_shared<RdmaTransferService>(task_store_, hanging_client);
    ASSERT_TRUE(service_->init(1000, 4));

    char src[64]{};
    char dst[64]{};
    const int64_t deadline_ms = currentTimeMs() + 500;
    auto task = task_store_->addTask("k_b4", makeKvBlocks(1, dst, sizeof(dst)), deadline_ms);
    ASSERT_NE(task, nullptr);

    auto req = makeRequest("k_b4", deadline_ms);
    addRdmaBlock(req, 1, src, sizeof(src));

    ::rdma_transfer::RdmaLayerBlockTransferResponse resp;
    auto closure = issueTransfer(&req, &resp);

    ASSERT_TRUE(closure->waitFor(kWait));
    EXPECT_EQ(resp.error_code(), ::rdma_transfer::RDMA_TRANSFER_CONTEXT_TIMEOUT);
    EXPECT_FALSE(task->done());
    ASSERT_EQ(hanging_client->conn_->heldCount(), 1);

    hanging_client->conn_->completeAll(true);
    for (int i = 0; i < 100 && !task->done(); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    EXPECT_TRUE(task->done());
    EXPECT_EQ(task->errorCode(), TransferErrorCode::TIMEOUT);
}

TEST_F(RdmaTransferServiceTest, B5_BlockedStagingH2d_KeepsTaskInflightAfterDeadlineAndClose) {
    service_.reset();
    auto memory_manager = std::make_shared<MockRdmaMemoryManager>();
    auto staging_pool = RdmaStagingBufferPool::create(
        memory_manager, HostStagingBlockPool::kAlignment, HostStagingBlockPool::kAlignment);
    ASSERT_NE(staging_pool, nullptr);
    auto immediate_client = std::make_shared<MockRdmaClient>();
    service_ = std::make_shared<RdmaTransferService>(
        task_store_, immediate_client, nullptr, staging_pool, /*rdma_disconnect_after_deadline_ms=*/50);
    ASSERT_TRUE(service_->init(1000, 1));

    CudaEventGate gate;
    ASSERT_TRUE(gate.init());
    std::promise<cudaError_t> stream_primed;
    auto stream_primed_future = stream_primed.get_future();
    ASSERT_EQ(service_->worker_thread_pool_->pushTask([&]() {
        auto copy_stream = getNoBlockCopyStream(0).stream();
        stream_primed.set_value(cudaStreamWaitEvent(copy_stream, gate.event(), 0));
    }),
              autil::ThreadPoolBase::ERROR_NONE);
    ASSERT_EQ(stream_primed_future.wait_for(kWait), std::future_status::ready);
    ASSERT_EQ(stream_primed_future.get(), cudaSuccess);

    constexpr size_t size = 64;
    char src[size];
    std::memset(src, 'J', size);
    auto dst =
        torch::zeros({static_cast<int64_t>(size)}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
    const int64_t deadline_ms = currentTimeMs() + 100;
    auto task = task_store_->addTask("k_b5", makeCudaKvBlocks(1, dst), deadline_ms);
    ASSERT_NE(task, nullptr);

    auto req = makeRequest("k_b5", deadline_ms);
    addRdmaBlock(req, 1, src, size);
    ::rdma_transfer::RdmaLayerBlockTransferResponse resp;
    auto closure = issueTransfer(&req, &resp);

    ASSERT_TRUE(closure->waitFor(kWait));
    EXPECT_EQ(resp.error_code(), ::rdma_transfer::RDMA_TRANSFER_CONTEXT_TIMEOUT);
    task->cancel();
    EXPECT_FALSE(task->done());
    for (int i = 0; i < 1000 && immediate_client->conn_->close_count_.load() == 0; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    EXPECT_EQ(immediate_client->conn_->close_count_.load(), 1);
    EXPECT_FALSE(task->done());

    gate.release();
    for (int i = 0; i < 1000 && !task->done(); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    EXPECT_TRUE(task->done());
    EXPECT_EQ(task->errorCode(), TransferErrorCode::TIMEOUT);

    service_.reset();
    staging_pool.reset();
    EXPECT_EQ(memory_manager->dereg_count.load(), 1);
}

// ===========================================================================
// C. Cancel
// ===========================================================================

TEST_F(RdmaTransferServiceTest, C1_RecvCancelledPending_ReturnsCancelled) {
    auto task = task_store_->addTask("k_c1", {}, currentTimeMs() + 5000);
    ASSERT_NE(task, nullptr);
    task->cancel();

    auto req = makeRequest("k_c1", currentTimeMs() + 5000);
    ::rdma_transfer::RdmaLayerBlockTransferResponse resp;
    auto closure = issueTransfer(&req, &resp);

    ASSERT_TRUE(closure->waitFor(kWait));
    EXPECT_EQ(resp.error_code(), ::rdma_transfer::RDMA_TRANSFER_TASK_CANCELLED);
    EXPECT_EQ(resp.error_message(), "TransferTask cancelled");
}

TEST_F(RdmaTransferServiceTest, C2_CancelDuringTransferring_ReturnsCancelled) {
    service_.reset();
    auto failing_client = std::make_shared<MockRdmaClient>(true, false);
    service_ = std::make_shared<RdmaTransferService>(task_store_, failing_client);
    ASSERT_TRUE(service_->init(1000, 4));

    char dummy[64]{};
    auto task = task_store_->addTask("k_c2", makeKvBlocks(1, dummy, 64), currentTimeMs() + 5000);
    ASSERT_NE(task, nullptr);

    task->startTransfer();
    task->cancel();

    auto req = makeRequest("k_c2", currentTimeMs() + 5000);
    addRdmaBlock(req, 1, dummy, 64);

    ::rdma_transfer::RdmaLayerBlockTransferResponse resp;
    auto closure = issueTransfer(&req, &resp);

    ASSERT_TRUE(closure->waitFor(kWait));
    EXPECT_EQ(resp.error_code(), ::rdma_transfer::RDMA_TRANSFER_TASK_CANCELLED);
}

TEST_F(RdmaTransferServiceTest, C4_ServiceDestruction_CancelsPendingContexts) {
    auto req = makeRequest("k_c4", currentTimeMs() + 60000);
    ::rdma_transfer::RdmaLayerBlockTransferResponse resp;
    auto closure = issueTransfer(&req, &resp);

    service_.reset();

    ASSERT_TRUE(closure->waitFor(std::chrono::milliseconds(500)));
    EXPECT_EQ(resp.error_code(), ::rdma_transfer::RDMA_TRANSFER_TASK_CANCELLED);
}

// ===========================================================================
// D. RDMA-specific errors
// ===========================================================================

TEST_F(RdmaTransferServiceTest, D1_NullRdmaClient_NotInitialized) {
    service_.reset();
    service_ = std::make_shared<RdmaTransferService>(task_store_, nullptr);
    ASSERT_TRUE(service_->init(1000, 4));

    char dummy[64]{};
    auto task = task_store_->addTask("k_d1", makeKvBlocks(1, dummy, 64), currentTimeMs() + 5000);
    ASSERT_NE(task, nullptr);

    auto req = makeRequest("k_d1", currentTimeMs() + 5000);
    addRdmaBlock(req, 1, dummy, 64);

    ::rdma_transfer::RdmaLayerBlockTransferResponse resp;
    auto closure = issueTransfer(&req, &resp);

    ASSERT_TRUE(closure->waitFor(kWait));
    EXPECT_EQ(resp.error_code(), ::rdma_transfer::RDMA_TRANSFER_NOT_INITIALIZED);
}

TEST_F(RdmaTransferServiceTest, D2_GetConnectionFailed_ConnectionFailed) {
    service_.reset();
    auto no_conn_client = std::make_shared<MockRdmaClient>(false, true);
    service_ = std::make_shared<RdmaTransferService>(task_store_, no_conn_client);
    ASSERT_TRUE(service_->init(1000, 4));

    char dummy[64]{};
    auto task = task_store_->addTask("k_d2", makeKvBlocks(1, dummy, 64), currentTimeMs() + 5000);
    ASSERT_NE(task, nullptr);

    auto req = makeRequest("k_d2", currentTimeMs() + 5000);
    addRdmaBlock(req, 1, dummy, 64);

    ::rdma_transfer::RdmaLayerBlockTransferResponse resp;
    auto closure = issueTransfer(&req, &resp);

    ASSERT_TRUE(closure->waitFor(kWait));
    EXPECT_EQ(resp.error_code(), ::rdma_transfer::RDMA_TRANSFER_CONNECTION_FAILED);
}

TEST_F(RdmaTransferServiceTest, D4_RdmaReadFailed_RdmaFailed) {
    service_.reset();
    auto fail_read_client = std::make_shared<MockRdmaClient>(true, false);
    service_ = std::make_shared<RdmaTransferService>(task_store_, fail_read_client);
    ASSERT_TRUE(service_->init(1000, 4));

    char dummy[64]{};
    auto task = task_store_->addTask("k_d4", makeKvBlocks(1, dummy, 64), currentTimeMs() + 5000);
    ASSERT_NE(task, nullptr);

    auto req = makeRequest("k_d4", currentTimeMs() + 5000);
    addRdmaBlock(req, 1, dummy, 64);

    ::rdma_transfer::RdmaLayerBlockTransferResponse resp;
    auto closure = issueTransfer(&req, &resp);

    ASSERT_TRUE(closure->waitFor(kWait));
    EXPECT_EQ(resp.error_code(), ::rdma_transfer::RDMA_TRANSFER_RDMA_FAILED);
}

// ===========================================================================
// E. Buffer Format Errors
// ===========================================================================

TEST_F(RdmaTransferServiceTest, E1_BufferMismatch_CacheKeyMismatch) {
    char dummy[64]{};
    auto task = task_store_->addTask("k_e1", makeKvBlocks(1, dummy, 64), currentTimeMs() + 5000);
    ASSERT_NE(task, nullptr);

    auto req = makeRequest("k_e1", currentTimeMs() + 5000);
    addRdmaBlock(req, 2, dummy, 64);

    ::rdma_transfer::RdmaLayerBlockTransferResponse resp;
    auto closure = issueTransfer(&req, &resp);

    ASSERT_TRUE(closure->waitFor(kWait));
    EXPECT_EQ(resp.error_code(), ::rdma_transfer::RDMA_TRANSFER_BUFFER_MISMATCH);
}

TEST_F(RdmaTransferServiceTest, E2_BufferMismatch_NoRdmaInfo) {
    char dummy[64]{};
    auto task = task_store_->addTask("k_e2", makeKvBlocks(1, dummy, 64), currentTimeMs() + 5000);
    ASSERT_NE(task, nullptr);

    auto req = makeRequest("k_e2", currentTimeMs() + 5000);
    auto* kb = req.add_blocks();
    kb->set_key(1);
    auto* b = kb->add_blocks();
    b->set_len(64);

    ::rdma_transfer::RdmaLayerBlockTransferResponse resp;
    auto closure = issueTransfer(&req, &resp);

    ASSERT_TRUE(closure->waitFor(kWait));
    EXPECT_EQ(resp.error_code(), ::rdma_transfer::RDMA_TRANSFER_BUFFER_MISMATCH);
}

TEST_F(RdmaTransferServiceTest, E3_BufferMismatch_PartialExpectedKeyCoverage) {
    char src[64]{};
    char dst1[64]{};
    char dst2[64]{};
    auto expected = makeKvBlocks(1, dst1, sizeof(dst1));
    expected.emplace(2, makeKvBlocks(2, dst2, sizeof(dst2)).at(2));
    auto task = task_store_->addTask("k_e3", std::move(expected), currentTimeMs() + 5000);
    ASSERT_NE(task, nullptr);

    auto req = makeRequest("k_e3", currentTimeMs() + 5000);
    addRdmaBlock(req, 1, src, sizeof(src));

    ::rdma_transfer::RdmaLayerBlockTransferResponse resp;
    auto closure = issueTransfer(&req, &resp);

    ASSERT_TRUE(closure->waitFor(kWait));
    EXPECT_EQ(resp.error_code(), ::rdma_transfer::RDMA_TRANSFER_BUFFER_MISMATCH);
    EXPECT_EQ(task->errorCode(), TransferErrorCode::BUFFER_MISMATCH);
}

TEST_F(RdmaTransferServiceTest, E4_BufferMismatch_MissingExpectedSubBlock) {
    char src[64]{};
    char dst_kv[64]{};
    char dst_scale[16]{};
    auto task = task_store_->addTask(
        "k_e4", makeKvScaleBlocks(1, dst_kv, sizeof(dst_kv), dst_scale, sizeof(dst_scale)), currentTimeMs() + 5000);
    ASSERT_NE(task, nullptr);

    auto req = makeRequest("k_e4", currentTimeMs() + 5000);
    auto* kb = req.add_blocks();
    kb->set_key(1);
    addRdmaSubBlock(kb, src, sizeof(src));
    addRdmaSubBlock(kb, src, sizeof(src));

    ::rdma_transfer::RdmaLayerBlockTransferResponse resp;
    auto closure = issueTransfer(&req, &resp);

    ASSERT_TRUE(closure->waitFor(kWait));
    EXPECT_EQ(resp.error_code(), ::rdma_transfer::RDMA_TRANSFER_BUFFER_MISMATCH);
}

TEST_F(RdmaTransferServiceTest, E5_PrefillSubBlockSuperset_Success) {
    char expected_src[64]{};
    char extra_src[64]{};
    char dst[64]{};
    std::memset(expected_src, 'S', sizeof(expected_src));
    auto task = task_store_->addTask("k_e5", makeKvBlocks(1, dst, sizeof(dst)), currentTimeMs() + 5000);
    ASSERT_NE(task, nullptr);

    auto req = makeRequest("k_e5", currentTimeMs() + 5000);
    auto* kb = req.add_blocks();
    kb->set_key(1);
    addRdmaSubBlock(kb, expected_src, sizeof(expected_src));
    addRdmaSubBlock(kb, extra_src, sizeof(extra_src));

    ::rdma_transfer::RdmaLayerBlockTransferResponse resp;
    auto closure = issueTransfer(&req, &resp);

    ASSERT_TRUE(closure->waitFor(kWait));
    EXPECT_EQ(resp.error_code(), ::rdma_transfer::RDMA_TRANSFER_NONE_ERROR);
    EXPECT_EQ(std::memcmp(dst, expected_src, sizeof(dst)), 0);
}

TEST_F(RdmaTransferServiceTest, E6_BufferMismatch_BlockLengthMismatch) {
    char src[32]{};
    char dst[64]{};
    auto task = task_store_->addTask("k_e6", makeKvBlocks(1, dst, sizeof(dst)), currentTimeMs() + 5000);
    ASSERT_NE(task, nullptr);

    auto req = makeRequest("k_e6", currentTimeMs() + 5000);
    addRdmaBlock(req, 1, src, sizeof(src));

    ::rdma_transfer::RdmaLayerBlockTransferResponse resp;
    auto closure = issueTransfer(&req, &resp);

    ASSERT_TRUE(closure->waitFor(kWait));
    EXPECT_EQ(resp.error_code(), ::rdma_transfer::RDMA_TRANSFER_BUFFER_MISMATCH);
}

TEST_F(RdmaTransferServiceTest, E7_PrefillKeySuperset_Success) {
    char extra_src[64]{};
    char expected_src[64]{};
    char expected_dst[64]{};
    std::memset(expected_src, 'S', sizeof(expected_src));
    auto task =
        task_store_->addTask("k_e7", makeKvBlocks(2, expected_dst, sizeof(expected_dst)), currentTimeMs() + 5000);
    ASSERT_NE(task, nullptr);

    auto req = makeRequest("k_e7", currentTimeMs() + 5000);
    addRdmaBlock(req, 1, extra_src, sizeof(extra_src));
    addRdmaBlock(req, 2, expected_src, sizeof(expected_src));

    ::rdma_transfer::RdmaLayerBlockTransferResponse resp;
    auto closure = issueTransfer(&req, &resp);

    ASSERT_TRUE(closure->waitFor(kWait));
    EXPECT_EQ(resp.error_code(), ::rdma_transfer::RDMA_TRANSFER_NONE_ERROR);
    EXPECT_EQ(std::memcmp(expected_dst, expected_src, sizeof(expected_dst)), 0);
}

TEST_F(RdmaTransferServiceTest, E8_BufferMismatch_DuplicatePrefillKey) {
    char src[64]{};
    char dst[64]{};
    auto task = task_store_->addTask("k_e8", makeKvBlocks(1, dst, sizeof(dst)), currentTimeMs() + 5000);
    ASSERT_NE(task, nullptr);

    auto req = makeRequest("k_e8", currentTimeMs() + 5000);
    addRdmaBlock(req, 1, src, sizeof(src));
    addRdmaBlock(req, 1, src, sizeof(src));

    ::rdma_transfer::RdmaLayerBlockTransferResponse resp;
    auto closure = issueTransfer(&req, &resp);

    ASSERT_TRUE(closure->waitFor(kWait));
    EXPECT_EQ(resp.error_code(), ::rdma_transfer::RDMA_TRANSFER_BUFFER_MISMATCH);
}

// ===========================================================================
// H. A2 fix: deadline watchdog forces timeout when RDMA callback is lost
// ===========================================================================

TEST_F(RdmaTransferServiceTest, H1_DeadlineWatchdog_ForcesTimeout_WhenCallbackNeverFires) {
    service_.reset();
    auto hanging_client = std::make_shared<HangingRdmaClient>();
    service_ = std::make_shared<RdmaTransferService>(task_store_, hanging_client);
    ASSERT_TRUE(service_->init(1000, 4));

    constexpr size_t size = 64;
    char src[size];
    char dst[size]{};
    std::memset(src, 'H', size);

    int64_t short_deadline = currentTimeMs() + 500;
    auto task = task_store_->addTask("k_h1", makeKvBlocks(1, dst, size), short_deadline + 1000);
    ASSERT_NE(task, nullptr);

    auto req = makeRequest("k_h1", short_deadline);
    addRdmaBlock(req, 1, src, size);

    ::rdma_transfer::RdmaLayerBlockTransferResponse resp;
    auto closure = issueTransfer(&req, &resp);

    // The hanging client never fires the callback.
    // The deadline watchdog should finish the RPC at D without fabricating task completion.
    ASSERT_TRUE(closure->waitFor(std::chrono::seconds(5)));
    EXPECT_EQ(resp.error_code(), ::rdma_transfer::RDMA_TRANSFER_CONTEXT_TIMEOUT);
}

TEST_F(RdmaTransferServiceTest, H2_DeadlineWatchdog_DoesNotDoubleRun_WhenCallbackAlsoFires) {
    service_.reset();
    auto normal_client = std::make_shared<MockRdmaClient>(true, true);
    service_ = std::make_shared<RdmaTransferService>(task_store_, normal_client);
    ASSERT_TRUE(service_->init(1000, 4));

    constexpr size_t size = 64;
    char src[size];
    char dst[size]{};
    std::memset(src, 'I', size);

    // Deadline far in the future — normal callback fires first
    auto task = task_store_->addTask("k_h2", makeKvBlocks(1, dst, size), currentTimeMs() + 60000);
    ASSERT_NE(task, nullptr);

    auto req = makeRequest("k_h2", currentTimeMs() + 60000);
    addRdmaBlock(req, 1, src, size);

    ::rdma_transfer::RdmaLayerBlockTransferResponse resp;
    auto closure = issueTransfer(&req, &resp);

    ASSERT_TRUE(closure->waitFor(kWait));
    EXPECT_EQ(resp.error_code(), ::rdma_transfer::RDMA_TRANSFER_NONE_ERROR);

    // Wait a bit to let the deadline sweep run — should not crash (guardedRun prevents double-run)
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
}

TEST_F(RdmaTransferServiceTest, H3_DeadlineGraceClosesConnectionAndCompletesTask) {
    service_.reset();
    auto hanging_client = std::make_shared<HangingRdmaClient>();
    service_ = std::make_shared<RdmaTransferService>(
        task_store_, hanging_client, nullptr, nullptr, /*rdma_disconnect_after_deadline_ms=*/50);
    ASSERT_TRUE(service_->init(1000, 4));

    char src[64]{};
    char dst[64]{};
    const int64_t deadline_ms = currentTimeMs() + 50;
    auto task = task_store_->addTask("k_h3", makeKvBlocks(1, dst, sizeof(dst)), deadline_ms);
    ASSERT_NE(task, nullptr);

    auto req = makeRequest("k_h3", deadline_ms);
    addRdmaBlock(req, 1, src, sizeof(src));

    ::rdma_transfer::RdmaLayerBlockTransferResponse resp;
    auto closure = issueTransfer(&req, &resp);
    ASSERT_TRUE(closure->waitFor(std::chrono::seconds(1)));
    EXPECT_EQ(resp.error_code(), ::rdma_transfer::RDMA_TRANSFER_CONTEXT_TIMEOUT);

    for (int i = 0; i < 1000 && !task->done(); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    EXPECT_TRUE(task->done());
    EXPECT_EQ(hanging_client->conn_->close_count_.load(), 1);
}

}  // namespace barex_rdma
}  // namespace transfer
}  // namespace rtp_llm

namespace rtp_llm::transfer::barex_rdma {
TEST(RdmaReadFirstCauseTest, LaterIndexFailureSurvivesWatchdogAndCancellationUntilAllDmaCompletes) {
    auto store = std::make_shared<TransferTaskStore>();
    auto client = std::make_shared<HangingRdmaClient>();
    RdmaTransferService service(store, client);
    service.per_connection_block_count_ = 1;
    // Do not start background loops: drive completion and watchdog order explicitly.
    char source[128]{};
    char target[128]{};
    const auto deadline = currentTimeMs() + 60000;
    auto task = store->addTask("first-rdma-error", makeKvScaleBlocks(7, target, 64, target + 64, 64), deadline);
    auto request = makeRequest("first-rdma-error", deadline);
    addRdmaBlock(request, 7, source, 64);
    addRdmaSubBlock(request.mutable_blocks(0), source + 64, 64);
    ::rdma_transfer::RdmaLayerBlockTransferResponse response;
    MockRpcController controller;
    BlockingClosure closure;
    auto context = std::make_shared<RdmaTaskContext>(&controller, &request, &response, &closure, nullptr);
    context->setTask(task);
    service.transferViaRdma(context);
    ASSERT_EQ(client->conn_->heldCount(), 2u);
    client->conn_->completeOne(1, TransferErrorCode::RDMA_FAILED, "peer=10.0.0.8:9000 vendor QP error 17");
    EXPECT_FALSE(task->done());
    task->cancel();
    ASSERT_EQ(service.active_transfers_.size(), 1u);
    service.active_transfers_.front().deadline_ms = 0;
    service.rdma_disconnect_after_deadline_ms_ = std::numeric_limits<int64_t>::max();
    service.transferDeadlineCheckProc();
    ASSERT_TRUE(closure.waitFor(std::chrono::milliseconds(0)));
    EXPECT_EQ(response.error_code(), ::rdma_transfer::RDMA_TRANSFER_RDMA_FAILED);
    EXPECT_NE(response.error_message().find("batch=1/2"), std::string::npos);
    EXPECT_NE(response.error_message().find("vendor QP error 17"), std::string::npos);
    EXPECT_EQ(response.error_message().find("timeout"), std::string::npos);
    EXPECT_FALSE(task->done());
    const auto first_message = response.error_message();
    client->conn_->completeAll(false);
    EXPECT_TRUE(task->done());
    EXPECT_EQ(task->errorCode(), TransferErrorCode::RDMA_FAILED);
    EXPECT_EQ(task->errorMessage(), first_message);
    EXPECT_EQ(response.error_message(), first_message);
}
}  // namespace rtp_llm::transfer::barex_rdma
