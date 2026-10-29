#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <future>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <torch/torch.h>

#include "rtp_llm/cpp/cache/BlockInfo.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/TransferErrorCode.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/TransferTask.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/Types.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/barex_rdma/RdmaInterface.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/barex_rdma/RdmaTransferService.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/barex_rdma/proto/rdma_service.pb.h"
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

// ---------------------------------------------------------------------------
// Mock RDMA infrastructure
// ---------------------------------------------------------------------------

class MockRdmaConnection: public IRdmaConnection {
public:
    explicit MockRdmaConnection(bool success = true): success_(success) {}

    void read(const std::vector<std::pair<torch::Tensor, std::shared_ptr<RemoteBuffer>>>& pairs,
              std::function<void(bool)> cb,
              uint64_t) override {
        if (!success_) {
            cb(false);
            return;
        }
        for (const auto& [local, remote] : pairs) {
            std::memcpy(
                local.data_ptr(), reinterpret_cast<const void*>(static_cast<uintptr_t>(remote->addr)), remote->len);
        }
        cb(true);
    }

private:
    bool success_;
};

class MockRdmaClient: public IRdmaClient {
public:
    explicit MockRdmaClient(bool conn_ok = true, bool read_ok = true): conn_ok_(conn_ok), read_ok_(read_ok) {}

    std::shared_ptr<IRdmaConnection> getConnection(const std::string&, uint32_t) override {
        if (!conn_ok_)
            return nullptr;
        return std::make_shared<MockRdmaConnection>(read_ok_);
    }

private:
    bool conn_ok_;
    bool read_ok_;
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
    req.set_deadline_ms(deadline_ms);
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

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

class RdmaTransferServiceTest: public ::testing::Test {
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

TEST_F(RdmaTransferServiceTest, B3_ContextTimeout_OverridesRdmaFailed) {
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
    EXPECT_EQ(resp.error_code(), ::rdma_transfer::RDMA_TRANSFER_CONTEXT_TIMEOUT);
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

TEST_F(RdmaTransferServiceTest, D1_NullRdmaClient_RdmaFailed) {
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
    EXPECT_EQ(resp.error_code(), ::rdma_transfer::RDMA_TRANSFER_RDMA_FAILED);
}

TEST_F(RdmaTransferServiceTest, D2_GetConnectionFailed_RdmaFailed) {
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
    EXPECT_EQ(resp.error_code(), ::rdma_transfer::RDMA_TRANSFER_RDMA_FAILED);
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

// ===========================================================================
// G. Concurrency
// ===========================================================================

TEST_F(RdmaTransferServiceTest, G3_ConcurrentTransfers_SameKey_BothComplete) {
    constexpr size_t size = 64;
    char src[size];
    char dst[size]{};
    std::memset(src, 'G', size);

    auto task = task_store_->addTask("k_g3", makeKvBlocks(1, dst, size), currentTimeMs() + 5000);
    ASSERT_NE(task, nullptr);

    auto req1 = makeRequest("k_g3", currentTimeMs() + 5000);
    auto req2 = makeRequest("k_g3", currentTimeMs() + 5000);
    addRdmaBlock(req1, 1, src, size);
    addRdmaBlock(req2, 1, src, size);

    ::rdma_transfer::RdmaLayerBlockTransferResponse resp1, resp2;
    auto closure1 = issueTransfer(&req1, &resp1);
    auto closure2 = issueTransfer(&req2, &resp2);

    ASSERT_TRUE(closure1->waitFor(kWait));
    ASSERT_TRUE(closure2->waitFor(kWait));

    EXPECT_TRUE(task->done());
    EXPECT_TRUE(task->success());
    EXPECT_EQ(resp1.error_code(), ::rdma_transfer::RDMA_TRANSFER_NONE_ERROR);
    EXPECT_EQ(resp2.error_code(), ::rdma_transfer::RDMA_TRANSFER_NONE_ERROR);
}

}  // namespace barex_rdma
}  // namespace transfer
}  // namespace rtp_llm
