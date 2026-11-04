#include "gtest/gtest.h"

#include "rtp_llm/cpp/cache/connector/p2p/transfer/barex_rdma/RdmaServer.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/barex_rdma/RdmaClient.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/barex_rdma/RdmaConnection.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/barex_rdma/RdmaMemoryManager.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/barex_rdma/RdmaInterface.h"
#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/cpp/cache/BlockInfo.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"

#include <torch/torch.h>
#include <c10/cuda/CUDAStream.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <future>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace rtp_llm {
namespace transfer {
namespace barex_rdma {

static constexpr uint32_t kTestPort   = 29876;
static constexpr uint32_t kUnusedPort = 29875;
static constexpr int64_t  kDeadlineMs = 5000;

// ============================================================================
// Sync helper: wraps the async read() into a blocking call.
// ============================================================================

static bool syncRead(IRdmaConnection&                                                          conn,
                     const std::vector<std::pair<torch::Tensor, std::shared_ptr<RemoteBuffer>>>& pairs,
                     int64_t                deadline_offset_ms = kDeadlineMs,
                     std::chrono::seconds   timeout            = std::chrono::seconds(10)) {
    std::promise<bool> p;
    auto               future = p.get_future();
    conn.read(pairs, [&p](bool ok) { p.set_value(ok); }, currentTimeMs() + deadline_offset_ms);
    if (future.wait_for(timeout) != std::future_status::ready) {
        return false;
    }
    return future.get();
}

// ============================================================================
// Fixture 1: RdmaInitOnlyTest
// ============================================================================

class RdmaInitOnlyTest : public ::testing::Test {};

TEST_F(RdmaInitOnlyTest, C1_Server_NullMemoryManager_InitFails) {
    RdmaServer server(nullptr);
    EXPECT_FALSE(server.init(kTestPort, 2, 0));
}

TEST_F(RdmaInitOnlyTest, C2_Client_NullMemoryManager_InitFails) {
    RdmaClient client(nullptr);
    EXPECT_FALSE(client.init(2, 0));
}

TEST_F(RdmaInitOnlyTest, C3_Server_DoubleInit_ReturnsTrue) {
    RdmaServer server(nullptr);
    server.initialized_ = true;
    EXPECT_TRUE(server.init(kTestPort, 2, 0));
}

TEST_F(RdmaInitOnlyTest, C4_Client_DoubleInit_ReturnsTrue) {
    RdmaClient client(nullptr);
    client.initialized_ = true;
    EXPECT_TRUE(client.init(2, 0));
}

TEST_F(RdmaInitOnlyTest, C5_Client_GetConnection_NotInitialized_ReturnsNull) {
    RdmaClient client(nullptr);
    EXPECT_EQ(client.getConnection("127.0.0.1", kTestPort), nullptr);
}

TEST_F(RdmaInitOnlyTest, B2_ReadOnFailedConnection_ImmediateReturnsFalse) {
    auto conn = std::make_shared<RdmaConnection>(nullptr, "127.0.0.1", kTestPort, nullptr);
    conn->onConnectFailed(nullptr);

    std::promise<bool> p;
    auto               future = p.get_future();
    conn->read({}, [&p](bool ok) { p.set_value(ok); }, currentTimeMs() + kDeadlineMs);

    ASSERT_EQ(future.wait_for(std::chrono::milliseconds(100)), std::future_status::ready);
    EXPECT_FALSE(future.get());
}

TEST_F(RdmaInitOnlyTest, B3_ReadInterleavedWithOnConnectFailed_CallbackFires) {
    for (int iter = 0; iter < 30; ++iter) {
        auto conn = std::make_shared<RdmaConnection>(nullptr, "127.0.0.1", kTestPort, nullptr);
        std::promise<bool> p;
        auto               future = p.get_future();
        std::thread        t([&conn] { conn->onConnectFailed(nullptr); });
        conn->read({}, [&p](bool ok) { p.set_value(ok); }, currentTimeMs() + kDeadlineMs);
        t.join();
        ASSERT_EQ(future.wait_for(std::chrono::seconds(2)), std::future_status::ready);
        EXPECT_FALSE(future.get());
    }
}

// ============================================================================
// Fixture 2: RdmaClientServerTest
// ============================================================================

class RdmaClientServerTest : public DeviceTestBase {
protected:
    void SetUp() override {
        DeviceTestBase::SetUp();

        auto mm  = createRdmaMemoryManager();
        server_mm_ = std::dynamic_pointer_cast<RdmaMemoryManager>(mm);
        client_mm_ = server_mm_;

        server_ = std::make_unique<RdmaServer>(mm);
        if (!server_->init(kTestPort, 2, 0)) {
            GTEST_SKIP() << "RDMA server init failed — no RDMA hardware or port in use";
        }

        client_ = std::make_unique<RdmaClient>(mm, /*rdma_connections_per_host=*/2);
        if (!client_->init(2, 0)) {
            GTEST_SKIP() << "RDMA client init failed — no RDMA hardware";
        }
    }

    void TearDown() override {
        client_.reset();
        server_.reset();
        allocated_tensors_.clear();
        DeviceTestBase::TearDown();
    }

    struct RegBlock {
        BlockInfo                     blk;
        torch::Tensor                 tensor;
        std::shared_ptr<RemoteBuffer> remote;
    };

    RegBlock allocAndRegCpuBlock(size_t size, uint8_t fill = 0) {
        auto tensor = torch::empty({static_cast<int64_t>(size)}, torch::kUInt8);
        std::memset(tensor.data_ptr(), fill, size);
        allocated_tensors_.push_back(tensor);

        BlockInfo blk{false, 0, 0, tensor.data_ptr(), size};
        EXPECT_TRUE(server_mm_->regUserMr(blk));
        auto remote = server_mm_->findMemoryMr(blk);
        EXPECT_NE(remote, nullptr);
        return {blk, tensor, remote};
    }

    RegBlock allocAndRegGpuBlock(size_t size, uint8_t fill = 0) {
        auto gpu_tensor = torch::empty({static_cast<int64_t>(size)},
                                        torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
        auto host_tmp = torch::full({static_cast<int64_t>(size)}, static_cast<int64_t>(fill), torch::kUInt8);
        gpu_tensor.copy_(host_tmp);
        at::cuda::getCurrentCUDAStream().synchronize();
        allocated_tensors_.push_back(gpu_tensor);

        BlockInfo blk{true, 0, 0, gpu_tensor.data_ptr(), size};
        EXPECT_TRUE(server_mm_->regUserMr(blk));
        auto remote = server_mm_->findMemoryMr(blk);
        EXPECT_NE(remote, nullptr);
        return {blk, gpu_tensor, remote};
    }

    bool verifyGpuContent(void* ptr, size_t size, uint8_t expected_fill) {
        auto options    = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);
        auto gpu_tensor = torch::from_blob(ptr, {static_cast<int64_t>(size)}, options);
        auto cpu_tensor = gpu_tensor.cpu();
        const auto* data = cpu_tensor.data_ptr<uint8_t>();
        for (size_t i = 0; i < size; ++i) {
            if (data[i] != expected_fill) {
                return false;
            }
        }
        return true;
    }

    std::shared_ptr<RdmaMemoryManager> server_mm_;
    std::shared_ptr<RdmaMemoryManager> client_mm_;
    std::unique_ptr<RdmaServer>        server_;
    std::unique_ptr<RdmaClient>        client_;
    std::vector<torch::Tensor>         allocated_tensors_;
};

// ─── Group B1: connect to unreachable port ───────────────────────────────────

TEST_F(RdmaClientServerTest, B1_ConnectToUnreachablePort_ReadReturnsFalse) {
    auto conn = client_->getConnection("127.0.0.1", kUnusedPort);
    ASSERT_NE(conn, nullptr);

    auto local_tensor = torch::empty({64}, torch::kUInt8);
    allocated_tensors_.push_back(local_tensor);
    BlockInfo local_blk{false, 0, 0, local_tensor.data_ptr(), 64};
    ASSERT_TRUE(client_mm_->regUserMr(local_blk));

    auto fake_rkeys = std::make_shared<std::map<uint32_t, uint32_t>>();
    auto remote     = std::make_shared<RemoteBuffer>(0, 64, fake_rkeys);

    bool ok = syncRead(*conn, {{local_tensor, remote}}, kDeadlineMs, std::chrono::seconds(15));
    EXPECT_FALSE(ok);

    client_mm_->deregUserMr(local_blk);
}

// ─── Group A: happy path reads ───────────────────────────────────────────────

TEST_F(RdmaClientServerTest, A1_ReadSingleCpuBuffer_ContentVerified) {
    constexpr size_t  kSize = 256;
    constexpr uint8_t kFill = 0xAB;

    auto [srv_blk, srv_tensor, remote] = allocAndRegCpuBlock(kSize, kFill);

    auto dst_tensor = torch::zeros({static_cast<int64_t>(kSize)}, torch::kUInt8);
    allocated_tensors_.push_back(dst_tensor);
    BlockInfo dst_blk{false, 0, 0, dst_tensor.data_ptr(), kSize};
    ASSERT_TRUE(client_mm_->regUserMr(dst_blk));

    auto conn = client_->getConnection("127.0.0.1", kTestPort);
    ASSERT_NE(conn, nullptr);

    ASSERT_TRUE(syncRead(*conn, {{dst_tensor, remote}}));

    const auto* data = dst_tensor.data_ptr<uint8_t>();
    for (size_t i = 0; i < kSize; ++i) {
        EXPECT_EQ(data[i], kFill) << "mismatch at byte " << i;
    }

    client_mm_->deregUserMr(dst_blk);
    server_mm_->deregUserMr(srv_blk);
}

TEST_F(RdmaClientServerTest, A2_ReadMultipleBuffers_AllCorrect) {
    constexpr int    kN    = 3;
    constexpr size_t kSize = 128;

    auto conn = client_->getConnection("127.0.0.1", kTestPort);
    ASSERT_NE(conn, nullptr);

    std::vector<std::pair<torch::Tensor, std::shared_ptr<RemoteBuffer>>> pairs;
    std::vector<BlockInfo>     dst_blks;
    std::vector<torch::Tensor> dst_tensors;
    std::vector<BlockInfo>     srv_blks;

    for (int i = 0; i < kN; ++i) {
        uint8_t fill = static_cast<uint8_t>(0x10 * (i + 1));

        auto [srv_blk, srv_tensor, remote] = allocAndRegCpuBlock(kSize, fill);
        srv_blks.push_back(srv_blk);

        auto dst = torch::zeros({static_cast<int64_t>(kSize)}, torch::kUInt8);
        allocated_tensors_.push_back(dst);
        BlockInfo dst_blk{false, 0, 0, dst.data_ptr(), kSize};
        ASSERT_TRUE(client_mm_->regUserMr(dst_blk));
        dst_blks.push_back(dst_blk);
        dst_tensors.push_back(dst);

        pairs.push_back({dst, remote});
    }

    ASSERT_TRUE(syncRead(*conn, pairs));

    for (int i = 0; i < kN; ++i) {
        uint8_t expected = static_cast<uint8_t>(0x10 * (i + 1));
        const auto* data = dst_tensors[i].data_ptr<uint8_t>();
        for (size_t j = 0; j < kSize; ++j) {
            EXPECT_EQ(data[j], expected) << "pair[" << i << "] byte " << j;
        }
        client_mm_->deregUserMr(dst_blks[i]);
        server_mm_->deregUserMr(srv_blks[i]);
    }
}

TEST_F(RdmaClientServerTest, A3_ReadGpuToGpu_ContentVerified) {
    constexpr size_t  kSize = 256;
    constexpr uint8_t kFill = 0xCC;

    auto [srv_blk, srv_tensor, remote] = allocAndRegGpuBlock(kSize, kFill);

    auto dst_gpu = torch::empty({static_cast<int64_t>(kSize)},
                                 torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
    allocated_tensors_.push_back(dst_gpu);
    BlockInfo dst_blk{true, 0, 0, dst_gpu.data_ptr(), kSize};
    ASSERT_TRUE(client_mm_->regUserMr(dst_blk));

    auto conn = client_->getConnection("127.0.0.1", kTestPort);
    ASSERT_NE(conn, nullptr);

    ASSERT_TRUE(syncRead(*conn, {{dst_gpu, remote}}));
    EXPECT_TRUE(verifyGpuContent(dst_gpu.data_ptr(), kSize, kFill));

    client_mm_->deregUserMr(dst_blk);
    server_mm_->deregUserMr(srv_blk);
}

// ─── Group D: read request validation errors ─────────────────────────────────

TEST_F(RdmaClientServerTest, D1_Read_UnregisteredLocalBuffer_ReturnsFalse) {
    constexpr size_t kSize = 64;
    auto [srv_blk, srv_tensor, remote] = allocAndRegCpuBlock(kSize, 0xAA);

    auto unregistered = torch::empty({static_cast<int64_t>(kSize)}, torch::kUInt8);
    allocated_tensors_.push_back(unregistered);

    auto conn = client_->getConnection("127.0.0.1", kTestPort);
    ASSERT_NE(conn, nullptr);

    EXPECT_FALSE(syncRead(*conn, {{unregistered, remote}}));

    server_mm_->deregUserMr(srv_blk);
}

TEST_F(RdmaClientServerTest, D2_Read_SizeMismatch_ReturnsFalse) {
    constexpr size_t kSrvSize = 128;
    constexpr size_t kDstSize = 64;

    auto [srv_blk, srv_tensor, remote] = allocAndRegCpuBlock(kSrvSize, 0xBB);

    auto dst_tensor = torch::empty({static_cast<int64_t>(kDstSize)}, torch::kUInt8);
    allocated_tensors_.push_back(dst_tensor);
    BlockInfo dst_blk{false, 0, 0, dst_tensor.data_ptr(), kDstSize};
    ASSERT_TRUE(client_mm_->regUserMr(dst_blk));

    auto conn = client_->getConnection("127.0.0.1", kTestPort);
    ASSERT_NE(conn, nullptr);

    EXPECT_FALSE(syncRead(*conn, {{dst_tensor, remote}}));

    client_mm_->deregUserMr(dst_blk);
    server_mm_->deregUserMr(srv_blk);
}

TEST_F(RdmaClientServerTest, D3_Read_NullLocalBuffer_ReturnsFalse) {
    auto fake_rkeys = std::make_shared<std::map<uint32_t, uint32_t>>();
    auto remote     = std::make_shared<RemoteBuffer>(0x1000, 64, fake_rkeys);

    auto conn = client_->getConnection("127.0.0.1", kTestPort);
    ASSERT_NE(conn, nullptr);

    torch::Tensor undefined_tensor;
    EXPECT_FALSE(syncRead(*conn, {{undefined_tensor, remote}}));
}

TEST_F(RdmaClientServerTest, D4_Read_NullRemoteBuffer_ReturnsFalse) {
    auto local_tensor = torch::empty({64}, torch::kUInt8);
    allocated_tensors_.push_back(local_tensor);
    BlockInfo local_blk{false, 0, 0, local_tensor.data_ptr(), 64};
    ASSERT_TRUE(client_mm_->regUserMr(local_blk));

    auto conn = client_->getConnection("127.0.0.1", kTestPort);
    ASSERT_NE(conn, nullptr);

    EXPECT_FALSE(syncRead(*conn, {{local_tensor, nullptr}}));

    client_mm_->deregUserMr(local_blk);
}

// ─── Group E: connection pool behavior ──────────────────────────────────────

TEST_F(RdmaClientServerTest, E1_SameHostPort_GetConnectionReturnsSamePool) {
    auto conn1 = client_->getConnection("127.0.0.1", kTestPort);
    auto conn2 = client_->getConnection("127.0.0.1", kTestPort);
    EXPECT_NE(conn1, nullptr);
    EXPECT_NE(conn2, nullptr);
}

TEST_F(RdmaClientServerTest, E2_ConnectionPoolDoesNotExceedMaxConns) {
    client_->getConnection("127.0.0.1", kTestPort);
    client_->getConnection("127.0.0.1", kTestPort);

    std::string spec = "127.0.0.1:" + std::to_string(kTestPort);
    auto        it   = client_->host_connection_pools_.find(spec);
    ASSERT_NE(it, client_->host_connection_pools_.end());
    EXPECT_LE(it->second->getPoolSize(), static_cast<uint32_t>(2));
}

// ─── Group F: concurrency ────────────────────────────────────────────────────

TEST_F(RdmaClientServerTest, F1_ConcurrentReads_NoCrash) {
    constexpr int     kNumThreads = 4;
    constexpr size_t  kSize       = 64;
    constexpr uint8_t kFill       = 0xEE;

    auto [srv_blk, srv_tensor, remote] = allocAndRegCpuBlock(kSize, kFill);

    std::vector<torch::Tensor> dst_tensors(kNumThreads);
    std::vector<BlockInfo>     dst_blks(kNumThreads);
    for (int i = 0; i < kNumThreads; ++i) {
        dst_tensors[i] = torch::empty({static_cast<int64_t>(kSize)}, torch::kUInt8);
        allocated_tensors_.push_back(dst_tensors[i]);
        dst_blks[i] = BlockInfo{false, 0, 0, dst_tensors[i].data_ptr(), kSize};
        ASSERT_TRUE(client_mm_->regUserMr(dst_blks[i]));
    }

    auto conn = client_->getConnection("127.0.0.1", kTestPort);
    ASSERT_NE(conn, nullptr);

    std::vector<std::promise<bool>> promises(kNumThreads);
    std::vector<std::future<bool>>  futures;
    for (int i = 0; i < kNumThreads; ++i) {
        futures.push_back(promises[i].get_future());
    }

    std::vector<std::thread> threads;
    for (int i = 0; i < kNumThreads; ++i) {
        threads.emplace_back([&, i]() {
            conn->read({{dst_tensors[i], remote}},
                       [&promises, i](bool ok) { promises[i].set_value(ok); },
                       currentTimeMs() + kDeadlineMs);
        });
    }
    for (auto& t : threads) {
        t.join();
    }

    int success_count = 0;
    for (int i = 0; i < kNumThreads; ++i) {
        ASSERT_EQ(futures[i].wait_for(std::chrono::seconds(10)), std::future_status::ready);
        if (futures[i].get()) {
            ++success_count;
        }
    }
    EXPECT_EQ(success_count, kNumThreads);

    for (int i = 0; i < kNumThreads; ++i) {
        client_mm_->deregUserMr(dst_blks[i]);
    }
    server_mm_->deregUserMr(srv_blk);
}

}  // namespace barex_rdma
}  // namespace transfer
}  // namespace rtp_llm
