#include "gtest/gtest.h"

#include "autil/EnvUtil.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/barex_rdma/RdmaMemoryManager.h"
#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/cpp/cache/BlockInfo.h"
#include "aios/network/accl-barex/include/accl/barex/barex_types.h"

#include <torch/torch.h>

#include <atomic>
#include <exception>
#include <thread>
#include <vector>

namespace rtp_llm {
namespace transfer {
namespace barex_rdma {

// ============================================================================
// Fixture
// ============================================================================

class RdmaMemoryManagerTest: public DeviceTestBase {
protected:
    void SetUp() override {
        DeviceTestBase::SetUp();
        manager_ = std::make_unique<RdmaMemoryManager>();
        if (!manager_->getXMempool()) {
            GTEST_SKIP() << "Barex RDMA mempool is unavailable on this test host";
        }
    }

    void TearDown() override {
        manager_.reset();
        allocated_tensors_.clear();
        DeviceTestBase::TearDown();
    }

    BlockInfo allocCpuBlock(size_t size) {
        auto tensor = torch::empty({static_cast<int64_t>(size)}, torch::kUInt8);
        allocated_tensors_.push_back(tensor);
        return BlockInfo{false, 0, 0, tensor.data_ptr(), size};
    }

    BlockInfo allocGpuBlock(size_t size) {
        auto tensor = torch::empty({static_cast<int64_t>(size)},
                                   torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
        allocated_tensors_.push_back(tensor);
        return BlockInfo{true, 0, 0, tensor.data_ptr(), size};
    }

    std::unique_ptr<RdmaMemoryManager> manager_;
    std::vector<torch::Tensor> allocated_tensors_;
};

// ============================================================================
// Group 1: regUserMr — input validation (no RDMA call required)
// ============================================================================

TEST_F(RdmaMemoryManagerTest, RegUserMr_NullAddr_ReturnsFalse) {
    BlockInfo bad{false, 0, 0, nullptr, 1024};
    EXPECT_FALSE(manager_->regUserMr(bad));
}

TEST_F(RdmaMemoryManagerTest, RegUserMr_ZeroSize_ReturnsFalse) {
    auto blk = allocCpuBlock(1024);
    blk.size_bytes = 0;
    EXPECT_FALSE(manager_->regUserMr(blk));
}

TEST_F(RdmaMemoryManagerTest, RegUserMr_AlignedSizeGtSize_ReturnsFalse) {
    EXPECT_FALSE(manager_->regUserMr(allocCpuBlock(1024), 2048));
}

TEST_F(RdmaMemoryManagerTest, RegUserMr_SizeNotDivisibleByAlignedSize_ReturnsFalse) {
    EXPECT_FALSE(manager_->regUserMr(allocCpuBlock(1024), 333));
}

// ============================================================================
// Group 2: deregUserMr — input validation
// ============================================================================

TEST_F(RdmaMemoryManagerTest, DeregUserMr_NullAddr_ReturnsFalse) {
    BlockInfo bad{false, 0, 0, nullptr, 0};
    EXPECT_FALSE(manager_->deregUserMr(bad));
}

// ============================================================================
// Group 3: findMemoryMr — input validation
// ============================================================================

TEST_F(RdmaMemoryManagerTest, FindMemoryMr_NullAddr_ReturnsNull) {
    BlockInfo bad{false, 0, 0, nullptr, 1024};
    EXPECT_EQ(manager_->findMemoryMr(bad), nullptr);
}

TEST_F(RdmaMemoryManagerTest, FindMemoryMr_ZeroSize_ReturnsNull) {
    auto blk = allocCpuBlock(1024);
    blk.size_bytes = 0;
    EXPECT_EQ(manager_->findMemoryMr(blk), nullptr);
}

TEST_F(RdmaMemoryManagerTest, FindMemoryMr_Unregistered_ReturnsNull) {
    ::accl::barex::memp_t mem;
    auto blk = allocCpuBlock(1024);
    EXPECT_EQ(manager_->findMemoryMr(blk), nullptr);
    EXPECT_FALSE(manager_->findMemoryMr(mem, blk.addr, blk.size_bytes, false));
}

// ============================================================================
// Group 4: CPU memory — full reg / find / dereg lifecycle
// ============================================================================

TEST_F(RdmaMemoryManagerTest, RegFindDereg_CpuMemory_FullLifecycle) {
    auto blk = allocCpuBlock(1024);

    ASSERT_TRUE(manager_->regUserMr(blk));

    auto remote = manager_->findMemoryMr(blk);
    ASSERT_NE(remote, nullptr);
    EXPECT_EQ(remote->addr, reinterpret_cast<int64_t>(blk.addr));
    EXPECT_EQ(remote->len, static_cast<uint32_t>(blk.size_bytes));

    ::accl::barex::memp_t mem;
    ASSERT_TRUE(manager_->findMemoryMr(mem, blk.addr, blk.size_bytes, false));
    EXPECT_EQ(mem.buf, blk.addr);
    EXPECT_EQ(mem.buf_len, blk.size_bytes);
    EXPECT_EQ(mem.d_type, ::accl::barex::CPU);

    ASSERT_TRUE(manager_->deregUserMr(blk));

    EXPECT_EQ(manager_->findMemoryMr(blk), nullptr);
    EXPECT_FALSE(manager_->findMemoryMr(mem, blk.addr, blk.size_bytes, false));
}

// ============================================================================
// Group 5: GPU memory — full reg / find / dereg lifecycle
// ============================================================================

TEST_F(RdmaMemoryManagerTest, RegFindDereg_GpuMemory_FullLifecycle) {
    auto blk = allocGpuBlock(1024);

    ASSERT_TRUE(manager_->regUserMr(blk));

    auto remote = manager_->findMemoryMr(blk);
    ASSERT_NE(remote, nullptr);
    EXPECT_EQ(remote->addr, reinterpret_cast<int64_t>(blk.addr));
    EXPECT_EQ(remote->len, static_cast<uint32_t>(blk.size_bytes));

    ::accl::barex::memp_t mem;
    ASSERT_TRUE(manager_->findMemoryMr(mem, blk.addr, blk.size_bytes, true));
    EXPECT_EQ(mem.buf, blk.addr);
    EXPECT_EQ(mem.buf_len, blk.size_bytes);
    EXPECT_EQ(mem.d_type, ::accl::barex::GPU);

    ASSERT_TRUE(manager_->deregUserMr(blk));
    EXPECT_EQ(manager_->findMemoryMr(blk), nullptr);
    EXPECT_FALSE(manager_->findMemoryMr(mem, blk.addr, blk.size_bytes, true));
}

// ============================================================================
// Group 6: aligned registration
// ============================================================================

TEST_F(RdmaMemoryManagerTest, RegUserMr_WithAlignedSize_SucceedsAndFindable) {
    auto blk = allocCpuBlock(1024);
    ASSERT_TRUE(manager_->regUserMr(blk, 512));

    auto remote = manager_->findMemoryMr(blk);
    ASSERT_NE(remote, nullptr);
    EXPECT_EQ(remote->len, static_cast<uint32_t>(blk.size_bytes));

    ASSERT_TRUE(manager_->deregUserMr(blk));
}

// ============================================================================
// Group 7: findMemoryMr — address boundary and device-type checks
// ============================================================================

TEST_F(RdmaMemoryManagerTest, FindMemoryMr_WrongGpuFlag_CpuRegistered_ReturnsNull) {
    auto blk = allocCpuBlock(1024);
    ASSERT_TRUE(manager_->regUserMr(blk));

    ::accl::barex::memp_t mem;
    EXPECT_FALSE(manager_->findMemoryMr(mem, blk.addr, blk.size_bytes, /*gpu=*/true));
    EXPECT_TRUE(manager_->findMemoryMr(mem, blk.addr, blk.size_bytes, /*gpu=*/false));

    ASSERT_TRUE(manager_->deregUserMr(blk));
}

TEST_F(RdmaMemoryManagerTest, FindMemoryMr_OneByteOverEnd_ReturnsNull) {
    auto blk = allocCpuBlock(1024);
    ASSERT_TRUE(manager_->regUserMr(blk));

    ::accl::barex::memp_t mem;
    EXPECT_FALSE(manager_->findMemoryMr(mem, blk.addr, blk.size_bytes + 1, false));

    ASSERT_TRUE(manager_->deregUserMr(blk));
}

TEST_F(RdmaMemoryManagerTest, FindMemoryMr_PartialOverlapBeyondEnd_ReturnsNull) {
    auto blk = allocCpuBlock(1024);
    ASSERT_TRUE(manager_->regUserMr(blk));

    void* partial = static_cast<uint8_t*>(blk.addr) + 512;
    ::accl::barex::memp_t mem;
    EXPECT_FALSE(manager_->findMemoryMr(mem, partial, 1024, false));

    ASSERT_TRUE(manager_->deregUserMr(blk));
}

TEST_F(RdmaMemoryManagerTest, FindMemoryMr_SubrangeWithinMR_Succeeds) {
    auto blk = allocCpuBlock(1024);
    ASSERT_TRUE(manager_->regUserMr(blk));

    void* sub_addr = static_cast<uint8_t*>(blk.addr) + 256;
    ::accl::barex::memp_t mem;
    ASSERT_TRUE(manager_->findMemoryMr(mem, sub_addr, 512, false));
    EXPECT_EQ(mem.buf, sub_addr);
    EXPECT_EQ(mem.buf_len, static_cast<uint64_t>(512));

    ASSERT_TRUE(manager_->deregUserMr(blk));
}

TEST_F(RdmaMemoryManagerTest, FindMemoryMr_AddressBeforeFirstBlock_ReturnsNull) {
    auto raw = torch::empty({8192}, torch::kUInt8);
    allocated_tensors_.push_back(raw);

    BlockInfo upper{false, 0, 0, static_cast<uint8_t*>(raw.data_ptr()) + 4096, 4096};
    ASSERT_TRUE(manager_->regUserMr(upper));

    BlockInfo lower{false, 0, 0, raw.data_ptr(), 4096};
    EXPECT_EQ(manager_->findMemoryMr(lower), nullptr);

    ASSERT_TRUE(manager_->deregUserMr(upper));
}

TEST_F(RdmaMemoryManagerTest, FindMemoryMr_AddressInGapBetweenBlocks_ReturnsNull) {
    auto raw = torch::empty({12288}, torch::kUInt8);
    allocated_tensors_.push_back(raw);

    BlockInfo blk_a{false, 0, 0, raw.data_ptr(), 4096};
    BlockInfo blk_b{false, 0, 0, static_cast<uint8_t*>(raw.data_ptr()) + 8192, 4096};
    ASSERT_TRUE(manager_->regUserMr(blk_a));
    ASSERT_TRUE(manager_->regUserMr(blk_b));

    void* gap = static_cast<uint8_t*>(raw.data_ptr()) + 4096;
    BlockInfo gap_blk{false, 0, 0, gap, 1024};
    EXPECT_EQ(manager_->findMemoryMr(gap_blk), nullptr);

    ASSERT_TRUE(manager_->deregUserMr(blk_a));
    ASSERT_TRUE(manager_->deregUserMr(blk_b));
}

TEST_F(RdmaMemoryManagerTest, FindMemoryMr_MultipleBlocks_ReturnsCorrectOne) {
    auto raw = torch::empty({8192}, torch::kUInt8);
    allocated_tensors_.push_back(raw);

    BlockInfo blk_a{false, 0, 0, raw.data_ptr(), 4096};
    BlockInfo blk_b{false, 0, 0, static_cast<uint8_t*>(raw.data_ptr()) + 4096, 4096};
    ASSERT_TRUE(manager_->regUserMr(blk_a));
    ASSERT_TRUE(manager_->regUserMr(blk_b));

    auto remote_a = manager_->findMemoryMr(blk_a);
    auto remote_b = manager_->findMemoryMr(blk_b);
    ASSERT_NE(remote_a, nullptr);
    ASSERT_NE(remote_b, nullptr);
    EXPECT_EQ(remote_a->addr, reinterpret_cast<int64_t>(blk_a.addr));
    EXPECT_EQ(remote_b->addr, reinterpret_cast<int64_t>(blk_b.addr));

    ASSERT_TRUE(manager_->deregUserMr(blk_a));
    ASSERT_TRUE(manager_->deregUserMr(blk_b));
}

// ============================================================================
// Group 8: deregUserMr — deregistration behaviors
// ============================================================================

TEST_F(RdmaMemoryManagerTest, DeregUserMr_UnregisteredAddr_ReturnsTrue_Noop) {
    auto blk = allocCpuBlock(1024);
    EXPECT_TRUE(manager_->deregUserMr(blk));
}

TEST_F(RdmaMemoryManagerTest, DeregUserMr_SecondCallSameAddr_Noop) {
    auto blk = allocCpuBlock(1024);
    ASSERT_TRUE(manager_->regUserMr(blk));
    ASSERT_TRUE(manager_->deregUserMr(blk));

    EXPECT_TRUE(manager_->deregUserMr(blk));
    EXPECT_EQ(manager_->findMemoryMr(blk), nullptr);
}

TEST_F(RdmaMemoryManagerTest, DeregUserMr_OnlyTargetBlockRemoved) {
    auto blk1 = allocCpuBlock(1024);
    auto blk2 = allocCpuBlock(1024);
    ASSERT_TRUE(manager_->regUserMr(blk1));
    ASSERT_TRUE(manager_->regUserMr(blk2));

    ASSERT_TRUE(manager_->deregUserMr(blk1));

    EXPECT_EQ(manager_->findMemoryMr(blk1), nullptr);
    EXPECT_NE(manager_->findMemoryMr(blk2), nullptr);

    ASSERT_TRUE(manager_->deregUserMr(blk2));
}

// ============================================================================
// Group 9: chunk-split registration
// ============================================================================

TEST_F(RdmaMemoryManagerTest, RegUserMr_ChunkSplit_AllChunksFindable) {
    constexpr uint64_t kChunkSize = 2048;
    constexpr uint64_t kAlignSize = 1024;
    constexpr uint64_t kTotalSize = 4 * kChunkSize;

    manager_->setMaxRegMemSize(kChunkSize);
    manager_->setMinRegMemSize(kAlignSize);

    auto blk = allocGpuBlock(kTotalSize);
    ASSERT_TRUE(manager_->regUserMr(blk, kAlignSize));

    for (uint64_t i = 0; i < kTotalSize / kAlignSize; ++i) {
        void* offset = static_cast<uint8_t*>(blk.addr) + i * kAlignSize;
        BlockInfo chunk{true, 0, 0, offset, kAlignSize};
        EXPECT_NE(manager_->findMemoryMr(chunk), nullptr) << "chunk[" << i << "] not found";
    }

    ::accl::barex::memp_t mem;
    EXPECT_FALSE(manager_->findMemoryMr(mem, static_cast<uint8_t*>(blk.addr) + kTotalSize, kAlignSize, true));

    ASSERT_TRUE(manager_->deregUserMr(blk));
}

TEST_F(RdmaMemoryManagerTest, DeregUserMr_ChunkSplitBlock_AllChunksGone) {
    constexpr uint64_t kChunkSize = 2048;
    constexpr uint64_t kAlignSize = 1024;
    constexpr uint64_t kTotalSize = 4 * kChunkSize;

    manager_->setMaxRegMemSize(kChunkSize);
    manager_->setMinRegMemSize(kAlignSize);

    auto blk = allocGpuBlock(kTotalSize);
    ASSERT_TRUE(manager_->regUserMr(blk, kAlignSize));

    ASSERT_TRUE(manager_->deregUserMr(blk));

    for (uint64_t i = 0; i < kTotalSize / kAlignSize; ++i) {
        void* offset = static_cast<uint8_t*>(blk.addr) + i * kAlignSize;
        BlockInfo chunk{true, 0, 0, offset, kAlignSize};
        EXPECT_EQ(manager_->findMemoryMr(chunk), nullptr) << "chunk[" << i << "] still findable";
    }
}

TEST_F(RdmaMemoryManagerTest, RegUserMr_LargeAligned_Succeeds) {
    constexpr uint64_t kBufLen = 14085881856UL;
    constexpr uint64_t kAlign = 294912;

    torch::Tensor raw;
    try {
        raw = torch::empty({static_cast<int64_t>(kBufLen)},
                           torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
    } catch (const std::exception& e) {
        GTEST_SKIP() << "GPU allocate failed, skipping large-aligned test: " << e.what();
    }
    if (!raw.defined()) {
        GTEST_SKIP() << "GPU OOM, skipping large-aligned test";
    }
    allocated_tensors_.push_back(raw);

    BlockInfo blk{true, 0, 0, raw.data_ptr(), kBufLen};
    bool ok = manager_->regUserMr(blk, kAlign);
    if (ok) {
        ASSERT_TRUE(manager_->deregUserMr(blk));
    }
}

// ============================================================================
// Group 10: concurrency — shared_mutex correctness
// ============================================================================

TEST_F(RdmaMemoryManagerTest, Concurrent_MultipleReaders_NoCrash) {
    auto blk = allocCpuBlock(4096);
    ASSERT_TRUE(manager_->regUserMr(blk));

    constexpr int kNumThreads = 16;
    std::atomic<int> found{0};

    std::vector<std::thread> threads;
    for (int i = 0; i < kNumThreads; ++i) {
        threads.emplace_back([&]() {
            if (manager_->findMemoryMr(blk) != nullptr) {
                found.fetch_add(1);
            }
        });
    }
    for (auto& t : threads)
        t.join();

    EXPECT_EQ(found.load(), kNumThreads);
    ASSERT_TRUE(manager_->deregUserMr(blk));
}

TEST_F(RdmaMemoryManagerTest, Concurrent_RegAndFind_DifferentKeys_NoCrash) {
    constexpr int kNumThreads = 4;
    constexpr size_t kBlockSize = 1024;

    std::vector<BlockInfo> blocks;
    for (int i = 0; i < kNumThreads; ++i) {
        blocks.push_back(allocCpuBlock(kBlockSize));
    }

    std::atomic<int> success{0};
    std::vector<std::thread> threads;
    for (int i = 0; i < kNumThreads; ++i) {
        threads.emplace_back([&, i]() {
            if (manager_->regUserMr(blocks[i]) && manager_->findMemoryMr(blocks[i]) != nullptr) {
                success.fetch_add(1);
            }
        });
    }
    for (auto& t : threads)
        t.join();

    EXPECT_EQ(success.load(), kNumThreads);
}

}  // namespace barex_rdma
}  // namespace transfer
}  // namespace rtp_llm
