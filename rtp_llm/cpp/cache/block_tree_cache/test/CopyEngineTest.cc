#include <gtest/gtest.h>

#include <cstring>
#include <unordered_map>
#include <vector>

#include <torch/torch.h>

#include "rtp_llm/cpp/cache/BlockPool.h"
#include "rtp_llm/cpp/cache/BlockPoolConfigHelper.h"
#include "rtp_llm/cpp/cache/block_tree_cache/copy_engine/CopyEngine.h"

namespace rtp_llm {
namespace {

// Helper: create a HOST BlockPool with the given block_size and usable_count.
// BlockPool reserves block 0, so we allocate usable_count+1 total blocks.
static std::shared_ptr<BlockPool> makeHostPool(size_t block_size, size_t usable_count) {
    auto cfg = BlockPoolConfigHelper::createConfig(
        /*layer_num=*/1,
        static_cast<uint32_t>(usable_count + 1),
        static_cast<uint32_t>(block_size),
        rtp_llm::TYPE_INT8);
    auto pool = std::make_shared<BlockPool>(cfg, AllocationType::HOST);
    return pool;
}

// Helper: allocate one block from pool, return NULL_BLOCK_IDX if pool is full.
static BlockIdxType poolMalloc(BlockPool& pool) {
    auto alloc = pool.malloc(1);
    return alloc.empty() ? NULL_BLOCK_IDX : alloc[0];
}

// Real CUDA device memory: allocates GPU buffers via torch and provides
// fill/readback utilities for test verification.
// Total GPU memory usage is kept minimal (< 2KB) to avoid OOM.
class CudaDeviceMemory {
public:
    void allocate(int layer_id, BlockIdxType block_idx, size_t size_bytes) {
        auto key      = makeKey(layer_id, block_idx);
        auto gpu_opts = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA, 0);
        tensors_[key] = torch::zeros({static_cast<int64_t>(size_bytes)}, gpu_opts);
    }

    // Fill GPU buffer with a repeating byte pattern.
    void fill(int layer_id, BlockIdxType block_idx, uint8_t pattern) {
        auto key = makeKey(layer_id, block_idx);
        auto it  = tensors_.find(key);
        if (it != tensors_.end()) {
            it->second.fill_(pattern);
        }
    }

    // Fill GPU buffer with sequential pattern (0, 1, 2, ..., wrapping at 255).
    void fillSequential(int layer_id, BlockIdxType block_idx) {
        auto key = makeKey(layer_id, block_idx);
        auto it  = tensors_.find(key);
        if (it != tensors_.end()) {
            int64_t              size = it->second.numel();
            std::vector<uint8_t> cpu_data(static_cast<size_t>(size));
            for (int64_t i = 0; i < size; ++i) {
                cpu_data[i] = static_cast<uint8_t>(i & 0xFF);
            }
            auto cpu_tensor = torch::from_blob(cpu_data.data(), {size}, torch::kUInt8).clone();
            it->second.copy_(cpu_tensor);
        }
    }

    // Copy GPU buffer back to CPU for verification.
    std::vector<uint8_t> readBack(int layer_id, BlockIdxType block_idx) {
        auto key = makeKey(layer_id, block_idx);
        auto it  = tensors_.find(key);
        if (it == tensors_.end())
            return {};
        auto    cpu_tensor = it->second.cpu();
        auto*   ptr        = cpu_tensor.data_ptr<uint8_t>();
        int64_t n          = cpu_tensor.numel();
        return std::vector<uint8_t>(ptr, ptr + n);
    }

    void* devicePtr(int layer_id, BlockIdxType block_idx) {
        auto key = makeKey(layer_id, block_idx);
        auto it  = tensors_.find(key);
        return it != tensors_.end() ? it->second.data_ptr() : nullptr;
    }

    size_t size(int layer_id, BlockIdxType block_idx) const {
        auto key = makeKey(layer_id, block_idx);
        auto it  = tensors_.find(key);
        return it != tensors_.end() ? static_cast<size_t>(it->second.numel()) : 0;
    }

    // Create a DeviceBufferResolver that returns real CUDA pointers.
    DeviceBufferResolver makeResolver() {
        return [this](int layer_id, BlockIdxType block_idx) -> BlockInfo {
            BlockInfo info;
            info.is_cuda      = true;
            info.device_index = 0;
            info.addr         = devicePtr(layer_id, block_idx);
            info.size_bytes   = size(layer_id, block_idx);
            return info;
        };
    }

private:
    static uint64_t makeKey(int layer_id, BlockIdxType block_idx) {
        return (static_cast<uint64_t>(layer_id) << 32) | static_cast<uint64_t>(block_idx);
    }

    std::unordered_map<uint64_t, torch::Tensor> tensors_;
};

// Fixture for CopyEngine tests with real CUDA memory.
class CopyEngineTest: public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_TRUE(torch::cuda::is_available()) << "CUDA not available, cannot run GPU tests";

        // 3 layers with different strides: 100, 200, 150 bytes
        slots_ = {
            {0, "layer_0", 100},
            {1, "layer_1", 200},
            {2, "layer_2", 150},
        };
        host_block_size_ = CopyEngine::computeHostBlockSize(slots_);  // 450

        // Create host pool (BlockPool HOST) — 10 usable blocks
        host_pool_ = makeHostPool(host_block_size_, 10);
        ASSERT_TRUE(host_pool_->init());

        copy_engine_ = std::make_shared<CopyEngine>();

        // Allocate real CUDA device memory for 3 device blocks (indices 1, 2, 3)
        device_blocks_ = {1, 2, 3};
        for (size_t i = 0; i < slots_.size(); ++i) {
            cuda_device_.allocate(slots_[i].layer_id, device_blocks_[i], slots_[i].stride_bytes);
        }
    }

    std::vector<MemoryBlockLayerTagSlot> slots_;
    size_t                               host_block_size_;
    std::shared_ptr<BlockPool>           host_pool_;
    std::shared_ptr<CopyEngine>          copy_engine_;
    std::vector<BlockIdxType>            device_blocks_;
    CudaDeviceMemory                     cuda_device_;
};

// ---- BlockPool HOST tests ----

TEST(BlockPoolHostTest, InitAndAlloc) {
    auto pool = makeHostPool(1024, 4);
    ASSERT_TRUE(pool->init());
    // totalBlocksNum returns usable block count (config_.block_num - 1)
    EXPECT_EQ(pool->totalBlocksNum(), 4u);
    EXPECT_EQ(pool->freeBlocksNum(), 4u);

    BlockIdxType b1 = poolMalloc(*pool);
    EXPECT_NE(b1, NULL_BLOCK_IDX);
    EXPECT_EQ(pool->freeBlocksNum(), 3u);

    BlockIdxType b2 = poolMalloc(*pool);
    EXPECT_NE(b2, NULL_BLOCK_IDX);
    EXPECT_NE(b1, b2);

    pool->requestFree(b1);
    // After 2 mallocs (free=2) then 1 requestFree → free=3
    EXPECT_EQ(pool->freeBlocksNum(), 3u);
}

TEST(BlockPoolHostTest, ExhaustPool) {
    auto pool = makeHostPool(256, 2);
    ASSERT_TRUE(pool->init());

    BlockIdxType b1 = poolMalloc(*pool);
    BlockIdxType b2 = poolMalloc(*pool);
    EXPECT_NE(b1, NULL_BLOCK_IDX);
    EXPECT_NE(b2, NULL_BLOCK_IDX);

    BlockIdxType b3 = poolMalloc(*pool);
    EXPECT_EQ(b3, NULL_BLOCK_IDX);  // Pool exhausted
}

TEST(BlockPoolHostTest, BlockAddr) {
    auto pool = makeHostPool(128, 3);
    ASSERT_TRUE(pool->init());

    BlockIdxType b    = poolMalloc(*pool);
    void*        addr = pool->convertIndexToAddr(0, b).kv_addr;
    EXPECT_NE(addr, nullptr);

    // Write and read back
    std::memset(addr, 0xAB, 128);
    auto* data = static_cast<uint8_t*>(addr);
    EXPECT_EQ(data[0], 0xAB);
    EXPECT_EQ(data[127], 0xAB);
}

TEST(BlockPoolHostTest, PinnedMemory) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available";
    }
    // BlockPool HOST uses pin_memory() by default (env RTP_LLM_PIN_HOST_BLOCK_POOL)
    auto pool = makeHostPool(4096, 2);
    ASSERT_TRUE(pool->init());

    BlockIdxType b    = poolMalloc(*pool);
    void*        addr = pool->convertIndexToAddr(0, b).kv_addr;
    EXPECT_NE(addr, nullptr);
}

// ---- CopyEngine DeviceToHost / HostToDevice tests (real CUDA) ----

TEST_F(CopyEngineTest, DeviceToHostPacking) {
    // Fill device blocks with distinct patterns
    cuda_device_.fill(0, 1, 0xAA);  // layer 0 → 100 bytes of 0xAA
    cuda_device_.fill(1, 2, 0xBB);  // layer 1 → 200 bytes of 0xBB
    cuda_device_.fill(2, 3, 0xCC);  // layer 2 → 150 bytes of 0xCC

    BlockIdxType host_block = poolMalloc(*host_pool_);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);

    auto resolver = cuda_device_.makeResolver();
    bool ok       = copy_engine_->deviceToHost(device_blocks_, host_block, slots_, resolver, *host_pool_);
    ASSERT_TRUE(ok);

    // Verify packed layout in host block (host memory is directly readable)
    const uint8_t* host_data = static_cast<const uint8_t*>(host_pool_->convertIndexToAddr(0, host_block).kv_addr);
    ASSERT_NE(host_data, nullptr);

    // First 100 bytes should be 0xAA (layer 0)
    for (size_t i = 0; i < 100; ++i) {
        EXPECT_EQ(host_data[i], 0xAA) << "byte " << i;
    }
    // Next 200 bytes should be 0xBB (layer 1)
    for (size_t i = 100; i < 300; ++i) {
        EXPECT_EQ(host_data[i], 0xBB) << "byte " << i;
    }
    // Last 150 bytes should be 0xCC (layer 2)
    for (size_t i = 300; i < 450; ++i) {
        EXPECT_EQ(host_data[i], 0xCC) << "byte " << i;
    }

    host_pool_->requestFree(host_block);
}

TEST_F(CopyEngineTest, HostToDeviceUnpacking) {
    // First, pack into host (D2H)
    cuda_device_.fill(0, 1, 0x11);
    cuda_device_.fill(1, 2, 0x22);
    cuda_device_.fill(2, 3, 0x33);

    BlockIdxType host_block = poolMalloc(*host_pool_);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);

    auto resolver = cuda_device_.makeResolver();
    ASSERT_TRUE(copy_engine_->deviceToHost(device_blocks_, host_block, slots_, resolver, *host_pool_));

    // Clear device buffers
    cuda_device_.fill(0, 1, 0x00);
    cuda_device_.fill(1, 2, 0x00);
    cuda_device_.fill(2, 3, 0x00);

    // Unpack host → device (H2D)
    ASSERT_TRUE(copy_engine_->hostToDevice(host_block, device_blocks_, slots_, resolver, *host_pool_));

    // Verify device buffers restored (read back from GPU)
    auto d0 = cuda_device_.readBack(0, 1);
    auto d1 = cuda_device_.readBack(1, 2);
    auto d2 = cuda_device_.readBack(2, 3);

    ASSERT_EQ(d0.size(), 100u);
    ASSERT_EQ(d1.size(), 200u);
    ASSERT_EQ(d2.size(), 150u);

    for (size_t i = 0; i < 100; ++i)
        EXPECT_EQ(d0[i], 0x11);
    for (size_t i = 0; i < 200; ++i)
        EXPECT_EQ(d1[i], 0x22);
    for (size_t i = 0; i < 150; ++i)
        EXPECT_EQ(d2[i], 0x33);

    host_pool_->requestFree(host_block);
}

TEST_F(CopyEngineTest, RoundTripSequentialData) {
    // Fill with sequential data for precise byte-level verification
    cuda_device_.fillSequential(0, 1);
    cuda_device_.fillSequential(1, 2);
    cuda_device_.fillSequential(2, 3);

    BlockIdxType host_block = poolMalloc(*host_pool_);
    auto         resolver   = cuda_device_.makeResolver();

    // D2H
    ASSERT_TRUE(copy_engine_->deviceToHost(device_blocks_, host_block, slots_, resolver, *host_pool_));

    // Clear device
    cuda_device_.fill(0, 1, 0x00);
    cuda_device_.fill(1, 2, 0x00);
    cuda_device_.fill(2, 3, 0x00);

    // H2D
    ASSERT_TRUE(copy_engine_->hostToDevice(host_block, device_blocks_, slots_, resolver, *host_pool_));

    // Verify round-trip: sequential pattern restored (read back from GPU)
    auto d0 = cuda_device_.readBack(0, 1);
    auto d1 = cuda_device_.readBack(1, 2);
    auto d2 = cuda_device_.readBack(2, 3);

    for (size_t i = 0; i < 100; ++i) {
        EXPECT_EQ(d0[i], static_cast<uint8_t>(i & 0xFF));
    }
    for (size_t i = 0; i < 200; ++i) {
        EXPECT_EQ(d1[i], static_cast<uint8_t>(i & 0xFF));
    }
    for (size_t i = 0; i < 150; ++i) {
        EXPECT_EQ(d2[i], static_cast<uint8_t>(i & 0xFF));
    }

    host_pool_->requestFree(host_block);
}

TEST_F(CopyEngineTest, NullDeviceBlockSkipped) {
    // device_blocks[1] is NULL → that slot should be zero-filled in host
    std::vector<BlockIdxType> blocks = {1, NULL_BLOCK_IDX, 3};
    cuda_device_.fill(0, 1, 0xAA);
    cuda_device_.fill(2, 3, 0xCC);

    BlockIdxType host_block = poolMalloc(*host_pool_);
    auto         resolver   = cuda_device_.makeResolver();

    ASSERT_TRUE(copy_engine_->deviceToHost(blocks, host_block, slots_, resolver, *host_pool_));

    const uint8_t* host_data = static_cast<const uint8_t*>(host_pool_->convertIndexToAddr(0, host_block).kv_addr);
    // Layer 0: 0xAA
    for (size_t i = 0; i < 100; ++i)
        EXPECT_EQ(host_data[i], 0xAA);
    // Layer 1: zero-filled (null device block)
    for (size_t i = 100; i < 300; ++i)
        EXPECT_EQ(host_data[i], 0x00);
    // Layer 2: 0xCC
    for (size_t i = 300; i < 450; ++i)
        EXPECT_EQ(host_data[i], 0xCC);

    host_pool_->requestFree(host_block);
}

TEST_F(CopyEngineTest, InvalidHostBlockFails) {
    auto resolver = cuda_device_.makeResolver();
    EXPECT_FALSE(copy_engine_->deviceToHost(device_blocks_, NULL_BLOCK_IDX, slots_, resolver, *host_pool_));
    EXPECT_FALSE(copy_engine_->deviceToHost(device_blocks_, 999, slots_, resolver, *host_pool_));
}

TEST_F(CopyEngineTest, MismatchedSlotCountFails) {
    BlockIdxType host_block = poolMalloc(*host_pool_);
    auto         resolver   = cuda_device_.makeResolver();

    std::vector<BlockIdxType> wrong_blocks = {1, 2};  // 2 blocks but 3 slots
    EXPECT_FALSE(copy_engine_->deviceToHost(wrong_blocks, host_block, slots_, resolver, *host_pool_));

    host_pool_->requestFree(host_block);
}

TEST_F(CopyEngineTest, ComputeHostBlockSize) {
    std::vector<MemoryBlockLayerTagSlot> test_slots = {
        {0, "a", 100},
        {1, "b", 200},
        {2, "c", 300},
    };
    EXPECT_EQ(CopyEngine::computeHostBlockSize(test_slots), 600u);
}

// ---- Host ↔ Disk roundtrip tests ----

class CopyEngineDiskTest: public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_TRUE(torch::cuda::is_available()) << "CUDA not available, cannot run GPU tests";

        slots_ = {
            {0, "layer_0", 128},
            {1, "layer_1", 256},
        };
        host_block_size_ = CopyEngine::computeHostBlockSize(slots_);  // 384

        host_pool_ = makeHostPool(host_block_size_, 4);
        ASSERT_TRUE(host_pool_->init());

        // Create a temp directory for disk pool
        char  tmpdir_template[] = "/tmp/copy_engine_test_XXXXXX";
        char* tmpdir            = ::mkdtemp(tmpdir_template);
        ASSERT_NE(tmpdir, nullptr) << "Failed to create temp directory";
        test_tmpdir_ = tmpdir;

        // Create a disk pool in the temp directory
        DiskBlockPoolConfig disk_config;
        disk_config.work_dir   = test_tmpdir_;
        disk_config.local_rank = 0;
        disk_config.world_rank = 0;
        // Ensure enough space: align block size up to 4096, then multiply by slot count
        size_t aligned_block_size    = ((host_block_size_ + 4095) / 4096) * 4096;
        disk_config.disk_size_bytes  = aligned_block_size * 8;  // 8 slots
        disk_config.block_size_bytes = host_block_size_;
        disk_config.buffered_io      = true;

        disk_pool_ = std::make_shared<DiskBlockPool>(std::move(disk_config));
        ASSERT_TRUE(disk_pool_->init()) << "DiskBlockPool init failed in " << test_tmpdir_;

        copy_engine_ = std::make_shared<CopyEngine>();
    }

    void TearDown() override {
        disk_pool_.reset();
        if (!test_tmpdir_.empty()) {
            std::string cmd = "rm -rf " + test_tmpdir_;
            ::system(cmd.c_str());
        }
    }

    std::string test_tmpdir_;

    std::vector<MemoryBlockLayerTagSlot> slots_;
    size_t                               host_block_size_;
    std::shared_ptr<BlockPool>           host_pool_;
    std::shared_ptr<DiskBlockPool>       disk_pool_;
    std::shared_ptr<CopyEngine>          copy_engine_;
};

TEST_F(CopyEngineDiskTest, HostToDiskRoundTrip) {
    // Fill host block with pattern
    BlockIdxType host_block = poolMalloc(*host_pool_);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);
    uint8_t* host_data = static_cast<uint8_t*>(host_pool_->convertIndexToAddr(0, host_block).kv_addr);
    for (size_t i = 0; i < host_block_size_; ++i) {
        host_data[i] = static_cast<uint8_t>(i & 0xFF);
    }

    // Write to disk
    auto disk_slot_opt = disk_pool_->malloc();
    ASSERT_TRUE(disk_slot_opt.has_value());
    int32_t disk_slot = disk_slot_opt.value();

    ASSERT_TRUE(copy_engine_->hostToDisk(host_block, disk_slot, *host_pool_, *disk_pool_));

    // Clear host block
    std::memset(host_data, 0, host_block_size_);

    // Read back from disk
    ASSERT_TRUE(copy_engine_->diskToHost(disk_slot, host_block, *host_pool_, *disk_pool_));

    // Verify data restored
    for (size_t i = 0; i < host_block_size_; ++i) {
        EXPECT_EQ(host_data[i], static_cast<uint8_t>(i & 0xFF)) << "byte " << i;
    }

    host_pool_->requestFree(host_block);
    disk_pool_->requestFree(disk_slot);
}

TEST_F(CopyEngineDiskTest, FullPipeline_D2H_H2Disk_Disk2H_H2D) {
    // Full pipeline: Device → Host → Disk → Host → Device (real CUDA)
    CudaDeviceMemory          cuda_device;
    std::vector<BlockIdxType> device_blocks = {1, 2};
    cuda_device.allocate(0, 1, 128);
    cuda_device.allocate(1, 2, 256);

    // Fill with sequential data on GPU
    cuda_device.fillSequential(0, 1);
    // Fill layer 1 with pattern: i*3 & 0xFF
    {
        std::vector<uint8_t> cpu_data(256);
        for (size_t i = 0; i < 256; ++i)
            cpu_data[i] = static_cast<uint8_t>(i * 3 & 0xFF);
        auto cpu_tensor = torch::from_blob(cpu_data.data(), {256}, torch::kUInt8).clone();
        // Get the internal tensor and copy
        auto gpu_opts   = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA, 0);
        auto gpu_tensor = torch::from_blob(cuda_device.devicePtr(1, 2), {256}, gpu_opts);
        gpu_tensor.copy_(cpu_tensor);
    }

    auto resolver = cuda_device.makeResolver();

    // Step 1: D2H
    BlockIdxType host_block_1 = poolMalloc(*host_pool_);
    ASSERT_TRUE(copy_engine_->deviceToHost(device_blocks, host_block_1, slots_, resolver, *host_pool_));

    // Step 2: H2Disk
    auto disk_slot_opt = disk_pool_->malloc();
    ASSERT_TRUE(disk_slot_opt.has_value());
    int32_t disk_slot = disk_slot_opt.value();
    ASSERT_TRUE(copy_engine_->hostToDisk(host_block_1, disk_slot, *host_pool_, *disk_pool_));
    host_pool_->requestFree(host_block_1);

    // Step 3: Disk2H
    BlockIdxType host_block_2 = poolMalloc(*host_pool_);
    ASSERT_NE(host_block_2, NULL_BLOCK_IDX);
    ASSERT_TRUE(copy_engine_->diskToHost(disk_slot, host_block_2, *host_pool_, *disk_pool_));

    // Step 4: H2D (clear device first)
    cuda_device.fill(0, 1, 0x00);
    cuda_device.fill(1, 2, 0x00);
    ASSERT_TRUE(copy_engine_->hostToDevice(host_block_2, device_blocks, slots_, resolver, *host_pool_));

    // Verify full roundtrip (read back from GPU)
    auto d0 = cuda_device.readBack(0, 1);
    auto d1 = cuda_device.readBack(1, 2);

    for (size_t i = 0; i < 128; ++i) {
        EXPECT_EQ(d0[i], static_cast<uint8_t>(i & 0xFF));
    }
    for (size_t i = 0; i < 256; ++i) {
        EXPECT_EQ(d1[i], static_cast<uint8_t>(i * 3 & 0xFF));
    }

    host_pool_->requestFree(host_block_2);
    disk_pool_->requestFree(disk_slot);
}

}  // namespace
}  // namespace rtp_llm
