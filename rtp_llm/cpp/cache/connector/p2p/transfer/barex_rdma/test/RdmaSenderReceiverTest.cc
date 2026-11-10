#include <gtest/gtest.h>

#include <cstdlib>
#include <chrono>
#include <cstring>
#include <future>
#include <string>
#include <memory>
#include <string>
#include <vector>

#include <torch/torch.h>
#include <c10/cuda/CUDAStream.h>

#include "rtp_llm/cpp/cache/BlockInfo.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/IKVCacheReceiver.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/IKVCacheSender.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/TransferErrorCode.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/Types.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/barex_rdma/RdmaInterface.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/barex_rdma/RdmaKVCacheReceiver.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/barex_rdma/RdmaKVCacheSender.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "autil/NetUtil.h"

namespace rtp_llm {
namespace transfer {
namespace barex_rdma {

static constexpr size_t kMinMrRegBytes = 1024;

// ---------------------------------------------------------------------------
// syncSend: block until callback fires
// ---------------------------------------------------------------------------

static TransferErrorCode
syncSend(RdmaKVCacheSender& sender, const SendRequest& req, std::chrono::seconds timeout = std::chrono::seconds(10)) {
    TransferErrorCode result = TransferErrorCode::UNKNOWN;
    std::promise<void> p;
    auto future = p.get_future();

    sender.send(req, [&](TransferErrorCode code, const std::string&) {
        result = code;
        p.set_value();
    });

    future.wait_for(timeout);
    return result;
}

// ---------------------------------------------------------------------------
// Block helpers
// ---------------------------------------------------------------------------

static KeyBlockInfoMap makeBlocks(int64_t cache_key, void* addr, size_t size, bool is_cuda = false) {
    auto kbi = std::make_shared<KeyBlockInfo>();
    kbi->cache_key = cache_key;
    kbi->blocks.push_back(BlockInfo{is_cuda, 0, 0, addr, size});
    KeyBlockInfoMap m;
    m[cache_key] = kbi;
    return m;
}

static bool regAllMem(IKVCacheSender& sender, const KeyBlockInfoMap& blocks) {
    for (const auto& [_, kbi_ptr] : blocks) {
        if (!kbi_ptr) {
            continue;
        }
        for (const auto& b : kbi_ptr->blocks) {
            if (!sender.regMem(b)) {
                return false;
            }
        }
    }
    return true;
}

static bool regAllMem(IKVCacheReceiver& receiver, const KeyBlockInfoMap& blocks) {
    for (const auto& [_, kbi_ptr] : blocks) {
        if (!kbi_ptr) {
            continue;
        }
        for (const auto& b : kbi_ptr->blocks) {
            if (!receiver.regMem(b)) {
                return false;
            }
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Fixture: Sender only
// ---------------------------------------------------------------------------

class RdmaSenderOnlyTest: public ::testing::Test {
public:
    static void SetUpTestSuite() {
        initRuntime(/*device_id=*/0,
                    /*trace_memory=*/false,
                    /*enable_comm_overlap=*/false,
                    MlaOpsType::AUTO);
    }

protected:
    void SetUp() override {
        unused_port_ = autil::NetUtil::randomPort();
        mm_ = createRdmaMemoryManager();
        sender_ = std::make_unique<RdmaKVCacheSender>(mm_, nullptr);
        if (!sender_->init(2, 2, 2)) {
            GTEST_SKIP() << "RdmaKVCacheSender init failed (barex RDMA environment required)";
        }
    }

    void TearDown() override {
        sender_.reset();
        mm_.reset();
    }

    std::shared_ptr<IRdmaMemoryManager> mm_;
    std::unique_ptr<RdmaKVCacheSender> sender_;
    uint32_t unused_port_ = 0;
};

// ---------------------------------------------------------------------------
// Fixture: Sender + Receiver
// ---------------------------------------------------------------------------

class RdmaSenderReceiverTest: public ::testing::Test {
public:
    static void SetUpTestSuite() {
        initRuntime(/*device_id=*/0,
                    /*trace_memory=*/false,
                    /*enable_comm_overlap=*/false,
                    MlaOpsType::AUTO);
    }

protected:
    void SetUp() override {
        test_port_ = autil::NetUtil::randomPort();
        mm_ = createRdmaMemoryManager();
        sender_ = std::make_unique<RdmaKVCacheSender>(mm_, nullptr);
        receiver_ = std::make_unique<RdmaKVCacheReceiver>(mm_, nullptr);
        if (!sender_->init(2, 2, 2)) {
            GTEST_SKIP() << "RdmaKVCacheSender init failed (barex RDMA environment required)";
        }
        if (!receiver_->init(test_port_, 2, 4, 2, 2, 8, 0)) {
            GTEST_SKIP() << "RdmaKVCacheReceiver init failed (barex RDMA environment required)";
        }
    }

    void TearDown() override {
        receiver_.reset();
        sender_.reset();
        mm_.reset();
    }

    torch::Tensor allocDevice(size_t size) {
        return torch::empty({static_cast<int64_t>(size)},
                            torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
    }

    bool verifyDevice(void* ptr, const std::string& expected) {
        const size_t size = expected.size();
        auto options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);
        auto gpu_tensor = torch::from_blob(ptr, {static_cast<int64_t>(size)}, options);
        auto cpu_tensor = gpu_tensor.cpu();
        at::cuda::getCurrentCUDAStream().synchronize();
        return std::memcmp(cpu_tensor.data_ptr(), expected.data(), size) == 0;
    }

    std::shared_ptr<IRdmaMemoryManager> mm_;
    std::unique_ptr<RdmaKVCacheSender> sender_;
    std::unique_ptr<RdmaKVCacheReceiver> receiver_;
    uint32_t test_port_ = 0;
};

// ===========================================================================
// F. Build request failures
// ===========================================================================

TEST_F(RdmaSenderOnlyTest, F1_EmptyBlockInfo_ImmediateBuildFailed) {
    SendRequest req;
    req.ip = "127.0.0.1";
    req.port = unused_port_;
    req.unique_key = "k_f1_rdma";
    req.block_info = {};
    req.deadline_ms = currentTimeMs() + 5000;

    auto code = syncSend(*sender_, req);
    EXPECT_EQ(code, TransferErrorCode::BUILD_REQUEST_FAILED);
}

// ===========================================================================
// D. Connection errors
// ===========================================================================

TEST_F(RdmaSenderOnlyTest, D1_ConnectionFailed_ReceiverNotRunning) {
    alignas(64) char dummy[kMinMrRegBytes]{};
    auto blocks = makeBlocks(1, dummy, sizeof(dummy), false);
    ASSERT_TRUE(regAllMem(*sender_, blocks));

    SendRequest req;
    req.ip = "127.0.0.1";
    req.port = unused_port_;
    req.unique_key = "k_d1_rdma";
    req.block_info = blocks;
    req.deadline_ms = currentTimeMs() + 1000;

    auto code = syncSend(*sender_, req, std::chrono::seconds(5));
    EXPECT_TRUE(code == TransferErrorCode::RPC_FAILED || code == TransferErrorCode::CONNECTION_FAILED);
}

// ===========================================================================
// B. Timeout (integration)
// ===========================================================================

TEST_F(RdmaSenderReceiverTest, B1_SenderTimeout_ReceiverHasNoMatchingTask) {
    alignas(64) char dummy[kMinMrRegBytes]{};
    auto blocks = makeBlocks(1, dummy, sizeof(dummy), false);
    ASSERT_TRUE(regAllMem(*sender_, blocks));

    SendRequest req;
    req.ip = "127.0.0.1";
    req.port = test_port_;
    req.unique_key = "k_b1_rdma_integ";
    req.block_info = blocks;
    req.deadline_ms = currentTimeMs() + 500;

    auto code = syncSend(*sender_, req, std::chrono::seconds(5));
    EXPECT_EQ(code, TransferErrorCode::TIMEOUT);
}

// ===========================================================================
// C. Cancel (integration)
// ===========================================================================

TEST_F(RdmaSenderReceiverTest, C1_RecvCancelledBeforeRpcArrives_Integration) {
    alignas(64) char dummy[kMinMrRegBytes]{};
    auto blocks = makeBlocks(1, dummy, sizeof(dummy), false);
    ASSERT_TRUE(regAllMem(*receiver_, blocks));
    ASSERT_TRUE(regAllMem(*sender_, blocks));

    RecvRequest recv_req;
    recv_req.unique_key = "k_c1_rdma_integ";
    recv_req.block_info = blocks;
    recv_req.deadline_ms = currentTimeMs() + 5000;

    auto task = receiver_->recv(recv_req);
    ASSERT_NE(task, nullptr);
    task->cancel();

    SendRequest send_req;
    send_req.ip = "127.0.0.1";
    send_req.port = test_port_;
    send_req.unique_key = "k_c1_rdma_integ";
    send_req.block_info = blocks;
    send_req.deadline_ms = currentTimeMs() + 5000;

    auto code = syncSend(*sender_, send_req, std::chrono::seconds(5));
    EXPECT_EQ(code, TransferErrorCode::CANCELLED);
    EXPECT_EQ(task->errorCode(), TransferErrorCode::CANCELLED);
}

// ===========================================================================
// A. Happy path (integration)
// ===========================================================================

TEST_F(RdmaSenderReceiverTest, A1_NormalTransfer_EndToEnd_Success) {
    constexpr size_t size = kMinMrRegBytes;
    const std::string content(size, 'E');
    auto gpu_dst = allocDevice(size);
    auto gpu_src = allocDevice(size);

    auto dst_blocks = makeBlocks(1, gpu_dst.data_ptr(), size, true);
    auto src_blocks = makeBlocks(1, gpu_src.data_ptr(), size, true);
    ASSERT_TRUE(regAllMem(*receiver_, dst_blocks));
    ASSERT_TRUE(regAllMem(*sender_, src_blocks));

    RecvRequest recv_req;
    recv_req.unique_key = "k_a1_rdma_integ";
    recv_req.block_info = dst_blocks;
    recv_req.deadline_ms = currentTimeMs() + 5000;

    auto task = receiver_->recv(recv_req);
    ASSERT_NE(task, nullptr);

    auto host_tensor =
        torch::from_blob(const_cast<char*>(content.data()), {static_cast<int64_t>(size)}, torch::kUInt8).clone();
    gpu_src.copy_(host_tensor);
    at::cuda::getCurrentCUDAStream().synchronize();

    SendRequest send_req;
    send_req.ip = "127.0.0.1";
    send_req.port = test_port_;
    send_req.unique_key = "k_a1_rdma_integ";
    send_req.block_info = src_blocks;
    send_req.deadline_ms = currentTimeMs() + 5000;

    auto code = syncSend(*sender_, send_req, std::chrono::seconds(10));
    EXPECT_EQ(code, TransferErrorCode::OK);
    EXPECT_TRUE(task->success());
    at::cuda::getCurrentCUDAStream().synchronize();
    EXPECT_TRUE(verifyDevice(gpu_dst.data_ptr(), content));
}

// ===========================================================================
// G. Multi-key (integration)
// ===========================================================================

TEST_F(RdmaSenderReceiverTest, G2_MultipleSenders_ConcurrentKeys) {
    constexpr int N = 3;
    constexpr size_t size = kMinMrRegBytes;

    std::vector<torch::Tensor> src_tensors(N), dst_tensors(N);
    std::vector<IKVCacheRecvTaskPtr> tasks(N);
    std::vector<std::string> contents(N);

    for (int i = 0; i < N; ++i) {
        contents[i] = std::string(size, static_cast<char>('0' + i));
        dst_tensors[i] = allocDevice(size);
        src_tensors[i] = allocDevice(size);

        auto host_src =
            torch::from_blob(const_cast<char*>(contents[i].data()), {static_cast<int64_t>(size)}, torch::kUInt8)
                .clone();
        src_tensors[i].copy_(host_src);
        at::cuda::getCurrentCUDAStream().synchronize();

        auto dst_blocks = makeBlocks(1, dst_tensors[i].data_ptr(), size, true);
        auto src_blocks = makeBlocks(1, src_tensors[i].data_ptr(), size, true);
        ASSERT_TRUE(regAllMem(*receiver_, dst_blocks));
        ASSERT_TRUE(regAllMem(*sender_, src_blocks));

        RecvRequest recv_req;
        recv_req.unique_key = "k_g2_rdma_" + std::to_string(i);
        recv_req.block_info = dst_blocks;
        recv_req.deadline_ms = currentTimeMs() + 5000;
        tasks[i] = receiver_->recv(recv_req);
        ASSERT_NE(tasks[i], nullptr);
    }

    for (int i = 0; i < N; ++i) {
        auto src_blocks = makeBlocks(1, src_tensors[i].data_ptr(), size, true);

        SendRequest send_req;
        send_req.ip = "127.0.0.1";
        send_req.port = test_port_;
        send_req.unique_key = "k_g2_rdma_" + std::to_string(i);
        send_req.block_info = src_blocks;
        send_req.deadline_ms = currentTimeMs() + 5000;

        auto code = syncSend(*sender_, send_req, std::chrono::seconds(10));
        EXPECT_EQ(code, TransferErrorCode::OK) << "key index " << i;
        EXPECT_TRUE(tasks[i]->success()) << "key index " << i;
        at::cuda::getCurrentCUDAStream().synchronize();
        EXPECT_TRUE(verifyDevice(dst_tensors[i].data_ptr(), contents[i])) << "key index " << i;
    }
}

}  // namespace barex_rdma
}  // namespace transfer
}  // namespace rtp_llm
