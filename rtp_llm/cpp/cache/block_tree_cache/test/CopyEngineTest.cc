#include <gtest/gtest.h>

#include <cstring>
#include <unordered_map>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/copy_engine/CopyEngine.h"
#include "rtp_llm/cpp/cache/block_tree_cache/copy_engine/HostBlockPool.h"

namespace rtp_llm {
namespace {

// Mock device memory: a map of (layer_id, block_idx) → CPU buffer.
// This allows testing the CopyEngine packing logic without actual GPU memory.
class MockDeviceMemory {
public:
    // Allocate a buffer for a (layer_id, block_idx) pair.
    void allocate(int layer_id, BlockIdxType block_idx, size_t size_bytes) {
        auto key = makeKey(layer_id, block_idx);
        buffers_[key].resize(size_bytes, 0);
    }

    // Fill a buffer with a repeating pattern.
    void fill(int layer_id, BlockIdxType block_idx, uint8_t pattern) {
        auto key = makeKey(layer_id, block_idx);
        auto it  = buffers_.find(key);
        if (it != buffers_.end()) {
            std::memset(it->second.data(), pattern, it->second.size());
        }
    }

    // Fill with a sequential pattern (0, 1, 2, ...) for verification.
    void fillSequential(int layer_id, BlockIdxType block_idx) {
        auto key = makeKey(layer_id, block_idx);
        auto it  = buffers_.find(key);
        if (it != buffers_.end()) {
            for (size_t i = 0; i < it->second.size(); ++i) {
                it->second[i] = static_cast<uint8_t>(i & 0xFF);
            }
        }
    }

    // Get the raw buffer for verification.
    const uint8_t* data(int layer_id, BlockIdxType block_idx) const {
        auto key = makeKey(layer_id, block_idx);
        auto it  = buffers_.find(key);
        return it != buffers_.end() ? it->second.data() : nullptr;
    }

    uint8_t* mutableData(int layer_id, BlockIdxType block_idx) {
        auto key = makeKey(layer_id, block_idx);
        auto it  = buffers_.find(key);
        return it != buffers_.end() ? it->second.data() : nullptr;
    }

    size_t size(int layer_id, BlockIdxType block_idx) const {
        auto key = makeKey(layer_id, block_idx);
        auto it  = buffers_.find(key);
        return it != buffers_.end() ? it->second.size() : 0;
    }

    // Create a DeviceBufferResolver that points to this mock memory.
    DeviceBufferResolver makeResolver() {
        return [this](int layer_id, BlockIdxType block_idx) -> BlockInfo {
            BlockInfo info;
            info.is_cuda      = false;
            info.device_index = 0;
            info.addr         = mutableData(layer_id, block_idx);
            info.size_bytes   = size(layer_id, block_idx);
            return info;
        };
    }

private:
    static uint64_t makeKey(int layer_id, BlockIdxType block_idx) {
        return (static_cast<uint64_t>(layer_id) << 32) | static_cast<uint64_t>(block_idx);
    }

    std::unordered_map<uint64_t, std::vector<uint8_t>> buffers_;
};

// Fixture for CopyEngine tests.
class CopyEngineTest: public ::testing::Test {
protected:
    void SetUp() override {
        // 3 layers with different strides: 100, 200, 150 bytes
        slots_ = {
            {0, "layer_0", 100},
            {1, "layer_1", 200},
            {2, "layer_2", 150},
        };
        host_block_size_ = CopyEngine::computeHostBlockSize(slots_);  // 450

        // Create host pool: 450-byte blocks, 10 blocks
        host_pool_ = std::make_shared<HostBlockPool>(host_block_size_, 10);
        ASSERT_TRUE(host_pool_->init());

        // Create copy engine (no disk pool for basic tests)
        copy_engine_ = std::make_shared<CopyEngine>(host_pool_);

        // Allocate mock device memory for 3 device blocks (indices 1, 2, 3)
        device_blocks_ = {1, 2, 3};
        for (size_t i = 0; i < slots_.size(); ++i) {
            mock_device_.allocate(slots_[i].layer_id, device_blocks_[i], slots_[i].stride_bytes);
        }
    }

    std::vector<MemoryBlockLayerTagSlot> slots_;
    size_t                               host_block_size_;
    std::shared_ptr<HostBlockPool>       host_pool_;
    std::shared_ptr<CopyEngine>          copy_engine_;
    std::vector<BlockIdxType>            device_blocks_;
    MockDeviceMemory                     mock_device_;
};

// ---- HostBlockPool tests ----

TEST(HostBlockPoolTest, InitAndAlloc) {
    HostBlockPool pool(1024, 4);
    ASSERT_TRUE(pool.init());
    EXPECT_EQ(pool.totalBlocks(), 4u);
    EXPECT_EQ(pool.freeBlocks(), 4u);

    BlockIdxType b1 = pool.malloc();
    EXPECT_NE(b1, NULL_BLOCK_IDX);
    EXPECT_EQ(pool.freeBlocks(), 3u);

    BlockIdxType b2 = pool.malloc();
    EXPECT_NE(b2, NULL_BLOCK_IDX);
    EXPECT_NE(b1, b2);

    pool.free(b1);
    EXPECT_EQ(pool.freeBlocks(), 3u);
}

TEST(HostBlockPoolTest, ExhaustPool) {
    HostBlockPool pool(256, 2);
    ASSERT_TRUE(pool.init());

    BlockIdxType b1 = pool.malloc();
    BlockIdxType b2 = pool.malloc();
    EXPECT_NE(b1, NULL_BLOCK_IDX);
    EXPECT_NE(b2, NULL_BLOCK_IDX);

    BlockIdxType b3 = pool.malloc();
    EXPECT_EQ(b3, NULL_BLOCK_IDX);  // Pool exhausted
}

TEST(HostBlockPoolTest, BlockAddr) {
    HostBlockPool pool(128, 3);
    ASSERT_TRUE(pool.init());

    BlockIdxType b    = pool.malloc();
    void*        addr = pool.blockAddr(b);
    EXPECT_NE(addr, nullptr);

    // Write and read back
    std::memset(addr, 0xAB, 128);
    auto* data = static_cast<uint8_t*>(addr);
    EXPECT_EQ(data[0], 0xAB);
    EXPECT_EQ(data[127], 0xAB);
}

TEST(HostBlockPoolTest, PinnedMemory) {
    HostBlockPool pool(4096, 2, /*use_pinned_memory=*/true);
    ASSERT_TRUE(pool.init());

    BlockIdxType b    = pool.malloc();
    void*        addr = pool.blockAddr(b);
    EXPECT_NE(addr, nullptr);
    // Pinned memory should be page-aligned
    EXPECT_EQ(reinterpret_cast<uintptr_t>(addr) % 4096, 0u);
}

// ---- CopyEngine DeviceToHost / HostToDevice tests ----

TEST_F(CopyEngineTest, DeviceToHostPacking) {
    // Fill device blocks with distinct patterns
    mock_device_.fill(0, 1, 0xAA);  // layer 0 → 100 bytes of 0xAA
    mock_device_.fill(1, 2, 0xBB);  // layer 1 → 200 bytes of 0xBB
    mock_device_.fill(2, 3, 0xCC);  // layer 2 → 150 bytes of 0xCC

    BlockIdxType host_block = host_pool_->malloc();
    ASSERT_NE(host_block, NULL_BLOCK_IDX);

    auto resolver = mock_device_.makeResolver();
    bool ok       = copy_engine_->deviceToHost(device_blocks_, host_block, slots_, resolver);
    ASSERT_TRUE(ok);

    // Verify packed layout in host block
    const uint8_t* host_data = static_cast<const uint8_t*>(host_pool_->blockAddr(host_block));
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

    host_pool_->free(host_block);
}

TEST_F(CopyEngineTest, HostToDeviceUnpacking) {
    // First, pack into host
    mock_device_.fill(0, 1, 0x11);
    mock_device_.fill(1, 2, 0x22);
    mock_device_.fill(2, 3, 0x33);

    BlockIdxType host_block = host_pool_->malloc();
    ASSERT_NE(host_block, NULL_BLOCK_IDX);

    auto resolver = mock_device_.makeResolver();
    ASSERT_TRUE(copy_engine_->deviceToHost(device_blocks_, host_block, slots_, resolver));

    // Clear device buffers
    mock_device_.fill(0, 1, 0x00);
    mock_device_.fill(1, 2, 0x00);
    mock_device_.fill(2, 3, 0x00);

    // Unpack host → device
    ASSERT_TRUE(copy_engine_->hostToDevice(host_block, device_blocks_, slots_, resolver));

    // Verify device buffers restored
    const uint8_t* d0 = mock_device_.data(0, 1);
    const uint8_t* d1 = mock_device_.data(1, 2);
    const uint8_t* d2 = mock_device_.data(2, 3);

    for (size_t i = 0; i < 100; ++i)
        EXPECT_EQ(d0[i], 0x11);
    for (size_t i = 0; i < 200; ++i)
        EXPECT_EQ(d1[i], 0x22);
    for (size_t i = 0; i < 150; ++i)
        EXPECT_EQ(d2[i], 0x33);

    host_pool_->free(host_block);
}

TEST_F(CopyEngineTest, RoundTripSequentialData) {
    // Fill with sequential data for precise byte-level verification
    mock_device_.fillSequential(0, 1);
    mock_device_.fillSequential(1, 2);
    mock_device_.fillSequential(2, 3);

    BlockIdxType host_block = host_pool_->malloc();
    auto         resolver   = mock_device_.makeResolver();

    // D2H
    ASSERT_TRUE(copy_engine_->deviceToHost(device_blocks_, host_block, slots_, resolver));

    // Clear device
    mock_device_.fill(0, 1, 0x00);
    mock_device_.fill(1, 2, 0x00);
    mock_device_.fill(2, 3, 0x00);

    // H2D
    ASSERT_TRUE(copy_engine_->hostToDevice(host_block, device_blocks_, slots_, resolver));

    // Verify round-trip: sequential pattern restored
    for (size_t i = 0; i < 100; ++i) {
        EXPECT_EQ(mock_device_.data(0, 1)[i], static_cast<uint8_t>(i & 0xFF));
    }
    for (size_t i = 0; i < 200; ++i) {
        EXPECT_EQ(mock_device_.data(1, 2)[i], static_cast<uint8_t>(i & 0xFF));
    }
    for (size_t i = 0; i < 150; ++i) {
        EXPECT_EQ(mock_device_.data(2, 3)[i], static_cast<uint8_t>(i & 0xFF));
    }

    host_pool_->free(host_block);
}

TEST_F(CopyEngineTest, NullDeviceBlockSkipped) {
    // device_blocks[1] is NULL → that slot should be zero-filled
    std::vector<BlockIdxType> blocks = {1, NULL_BLOCK_IDX, 3};
    mock_device_.fill(0, 1, 0xAA);
    mock_device_.fill(2, 3, 0xCC);

    BlockIdxType host_block = host_pool_->malloc();
    auto         resolver   = mock_device_.makeResolver();

    ASSERT_TRUE(copy_engine_->deviceToHost(blocks, host_block, slots_, resolver));

    const uint8_t* host_data = static_cast<const uint8_t*>(host_pool_->blockAddr(host_block));
    // Layer 0: 0xAA
    for (size_t i = 0; i < 100; ++i)
        EXPECT_EQ(host_data[i], 0xAA);
    // Layer 1: zero-filled (null device block)
    for (size_t i = 100; i < 300; ++i)
        EXPECT_EQ(host_data[i], 0x00);
    // Layer 2: 0xCC
    for (size_t i = 300; i < 450; ++i)
        EXPECT_EQ(host_data[i], 0xCC);

    host_pool_->free(host_block);
}

TEST_F(CopyEngineTest, InvalidHostBlockFails) {
    auto resolver = mock_device_.makeResolver();
    EXPECT_FALSE(copy_engine_->deviceToHost(device_blocks_, NULL_BLOCK_IDX, slots_, resolver));
    EXPECT_FALSE(copy_engine_->deviceToHost(device_blocks_, 999, slots_, resolver));
}

TEST_F(CopyEngineTest, MismatchedSlotCountFails) {
    BlockIdxType host_block = host_pool_->malloc();
    auto         resolver   = mock_device_.makeResolver();

    std::vector<BlockIdxType> wrong_blocks = {1, 2};  // 2 blocks but 3 slots
    EXPECT_FALSE(copy_engine_->deviceToHost(wrong_blocks, host_block, slots_, resolver));

    host_pool_->free(host_block);
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
        slots_ = {
            {0, "layer_0", 128},
            {1, "layer_1", 256},
        };
        host_block_size_ = CopyEngine::computeHostBlockSize(slots_);  // 384

        host_pool_ = std::make_shared<HostBlockPool>(host_block_size_, 4);
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

        copy_engine_ = std::make_shared<CopyEngine>(host_pool_, disk_pool_);
    }

    void TearDown() override {
        // Clean up temp files
        disk_pool_.reset();
        if (!test_tmpdir_.empty()) {
            // Remove files in temp dir
            std::string cmd = "rm -rf " + test_tmpdir_;
            ::system(cmd.c_str());
        }
    }

    std::string test_tmpdir_;

    std::vector<MemoryBlockLayerTagSlot> slots_;
    size_t                               host_block_size_;
    std::shared_ptr<HostBlockPool>       host_pool_;
    std::shared_ptr<DiskBlockPool>       disk_pool_;
    std::shared_ptr<CopyEngine>          copy_engine_;
};

TEST_F(CopyEngineDiskTest, HostToDiskRoundTrip) {
    // Fill host block with pattern
    BlockIdxType host_block = host_pool_->malloc();
    ASSERT_NE(host_block, NULL_BLOCK_IDX);
    uint8_t* host_data = static_cast<uint8_t*>(host_pool_->blockAddr(host_block));
    for (size_t i = 0; i < host_block_size_; ++i) {
        host_data[i] = static_cast<uint8_t>(i & 0xFF);
    }

    // Write to disk
    auto disk_slot_opt = disk_pool_->malloc();
    ASSERT_TRUE(disk_slot_opt.has_value());
    int32_t disk_slot = disk_slot_opt.value();

    ASSERT_TRUE(copy_engine_->hostToDisk(host_block, disk_slot));

    // Clear host block
    std::memset(host_data, 0, host_block_size_);

    // Read back from disk
    ASSERT_TRUE(copy_engine_->diskToHost(disk_slot, host_block));

    // Verify data restored
    for (size_t i = 0; i < host_block_size_; ++i) {
        EXPECT_EQ(host_data[i], static_cast<uint8_t>(i & 0xFF)) << "byte " << i;
    }

    host_pool_->free(host_block);
    disk_pool_->requestFree(disk_slot);
}

TEST_F(CopyEngineDiskTest, FullPipeline_D2H_H2Disk_Disk2H_H2D) {
    // Full pipeline: Device → Host → Disk → Host → Device
    MockDeviceMemory          mock_device;
    std::vector<BlockIdxType> device_blocks = {1, 2};
    mock_device.allocate(0, 1, 128);
    mock_device.allocate(1, 2, 256);

    // Fill with sequential data
    for (size_t i = 0; i < 128; ++i)
        mock_device.mutableData(0, 1)[i] = static_cast<uint8_t>(i);
    for (size_t i = 0; i < 256; ++i)
        mock_device.mutableData(1, 2)[i] = static_cast<uint8_t>(i * 3 & 0xFF);

    auto resolver = mock_device.makeResolver();

    // Step 1: D2H
    BlockIdxType host_block_1 = host_pool_->malloc();
    ASSERT_TRUE(copy_engine_->deviceToHost(device_blocks, host_block_1, slots_, resolver));

    // Step 2: H2Disk
    auto disk_slot_opt = disk_pool_->malloc();
    ASSERT_TRUE(disk_slot_opt.has_value());
    int32_t disk_slot = disk_slot_opt.value();
    ASSERT_TRUE(copy_engine_->hostToDisk(host_block_1, disk_slot));
    host_pool_->free(host_block_1);

    // Step 3: Disk2H
    BlockIdxType host_block_2 = host_pool_->malloc();
    ASSERT_NE(host_block_2, NULL_BLOCK_IDX);
    ASSERT_TRUE(copy_engine_->diskToHost(disk_slot, host_block_2));

    // Step 4: H2D
    mock_device.fill(0, 1, 0x00);
    mock_device.fill(1, 2, 0x00);
    ASSERT_TRUE(copy_engine_->hostToDevice(host_block_2, device_blocks, slots_, resolver));

    // Verify full roundtrip
    for (size_t i = 0; i < 128; ++i) {
        EXPECT_EQ(mock_device.data(0, 1)[i], static_cast<uint8_t>(i));
    }
    for (size_t i = 0; i < 256; ++i) {
        EXPECT_EQ(mock_device.data(1, 2)[i], static_cast<uint8_t>(i * 3 & 0xFF));
    }

    host_pool_->free(host_block_2);
    disk_pool_->requestFree(disk_slot);
}

}  // namespace
}  // namespace rtp_llm
