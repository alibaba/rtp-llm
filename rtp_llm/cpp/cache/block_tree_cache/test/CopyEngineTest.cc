#include <gtest/gtest.h>

#include <algorithm>
#include <cstring>
#include <unordered_map>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <torch/torch.h>

#include "rtp_llm/cpp/cache/block_tree_cache/copy_engine/CopyEngine.h"
#include "rtp_llm/cpp/cache/block_tree_cache/host/DiskBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/host/HostBlockPool.h"

namespace rtp_llm {
namespace {

// Helper: create a v4 HostBlockPool with the given payload_bytes and usable_count.
// IBlockPool reserves block 0, so physical_block_count = usable_count + 1.
static std::shared_ptr<HostBlockPool> makeHostPool(size_t payload_bytes, size_t usable_count) {
    auto config                     = std::make_shared<HostBlockPoolConfig>();
    config->pool_type               = BlockPoolType::HOST;
    config->pool_name               = "copy_engine_host";
    config->physical_block_count    = usable_count + 1;
    config->free_block_order_policy = FreeBlockOrderPolicy::ANY_ORDER;
    config->payload_bytes           = payload_bytes;
    config->stride_bytes            = ((payload_bytes + 4095) / 4096) * 4096;
    config->enable_pinned           = true;
    config->alignment               = 4096;

    auto pool = std::make_shared<HostBlockPool>(config);
    RTP_LLM_CHECK(pool->init());
    return pool;
}

// Helper: allocate one block from pool, return NULL_BLOCK_IDX if pool is full.
static BlockIdxType poolMalloc(IBlockPool& pool) {
    auto block = pool.malloc();
    return block.has_value() ? *block : NULL_BLOCK_IDX;
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
            // Ensure fill_ (default stream) completes before subsequent copies
            // executed on the CopyEngine's dedicated no-block-copy stream.
            cudaDeviceSynchronize();
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
            cudaDeviceSynchronize();
        }
    }

    // Copy GPU buffer back to CPU for verification.
    std::vector<uint8_t> readBack(int layer_id, BlockIdxType block_idx) {
        auto key = makeKey(layer_id, block_idx);
        auto it  = tensors_.find(key);
        if (it == tensors_.end())
            return {};
        // Ensure prior GPU work (e.g. execNoBlockCopy on its own stream) is
        // visible before the D2H readback.
        cudaDeviceSynchronize();
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

        copy_engine_ = std::make_shared<CopyEngine>();

        // Allocate real CUDA device memory for 3 device blocks (indices 1, 2, 3)
        device_blocks_ = {1, 2, 3};
        for (size_t i = 0; i < slots_.size(); ++i) {
            cuda_device_.allocate(slots_[i].layer_id, device_blocks_[i], slots_[i].stride_bytes);
        }
    }

    std::vector<MemoryBlockLayerTagSlot> slots_;
    size_t                               host_block_size_;
    std::shared_ptr<HostBlockPool>      host_pool_;
    std::shared_ptr<CopyEngine>          copy_engine_;
    std::vector<BlockIdxType>            device_blocks_;
    CudaDeviceMemory                     cuda_device_;
};

// ---- BlockPool HOST tests ----

TEST(BlockPoolHostTest, InitAndAlloc) {
    auto pool = makeHostPool(1024, 4);
    // totalBlocksNum returns usable block count (config_.block_num - 1)
    EXPECT_EQ(pool->totalBlocksNum(), 4u);
    EXPECT_EQ(pool->freeBlocksNum(), 4u);

    BlockIdxType b1 = poolMalloc(*pool);
    EXPECT_NE(b1, NULL_BLOCK_IDX);
    EXPECT_EQ(pool->freeBlocksNum(), 3u);

    BlockIdxType b2 = poolMalloc(*pool);
    EXPECT_NE(b2, NULL_BLOCK_IDX);
    EXPECT_NE(b1, b2);

    pool->free(b1);
    // After 2 mallocs (free=2) then 1 free() → free=3
    EXPECT_EQ(pool->freeBlocksNum(), 3u);
}

TEST(BlockPoolHostTest, ExhaustPool) {
    auto pool = makeHostPool(256, 2);

    BlockIdxType b1 = poolMalloc(*pool);
    BlockIdxType b2 = poolMalloc(*pool);
    EXPECT_NE(b1, NULL_BLOCK_IDX);
    EXPECT_NE(b2, NULL_BLOCK_IDX);

    BlockIdxType b3 = poolMalloc(*pool);
    EXPECT_EQ(b3, NULL_BLOCK_IDX);  // Pool exhausted
}

TEST(BlockPoolHostTest, BlockAddr) {
    auto pool = makeHostPool(128, 3);

    BlockIdxType b    = poolMalloc(*pool);
    void*        addr = pool->blockBuffer(b).addr;
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

    BlockIdxType b    = poolMalloc(*pool);
    void*        addr = pool->blockBuffer(b).addr;
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
    const uint8_t* host_data = static_cast<const uint8_t*>(host_pool_->blockBuffer(host_block).addr);
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

    host_pool_->free(host_block);
}

TEST_F(CopyEngineTest, RoundTripSequentialData) {
    // Fill with sequential data for precise byte-level verification
    cuda_device_.fillSequential(0, 1);
    cuda_device_.fill(1, 2, 0x5A);
    cuda_device_.fillSequential(2, 3);

    BlockIdxType host_block = poolMalloc(*host_pool_);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);
    auto resolver = cuda_device_.makeResolver();

    ASSERT_TRUE(copy_engine_->deviceToHost(device_blocks_, host_block, slots_, resolver, *host_pool_));

    const uint8_t* host_data = static_cast<const uint8_t*>(host_pool_->blockBuffer(host_block).addr);
    ASSERT_NE(host_data, nullptr);

    for (size_t i = 0; i < slots_[0].stride_bytes; ++i) {
        EXPECT_EQ(host_data[i], static_cast<uint8_t>(i & 0xFF));
    }
    for (size_t i = 0; i < slots_[1].stride_bytes; ++i) {
        EXPECT_EQ(host_data[slots_[0].stride_bytes + i], 0x5A);
    }
    const size_t layer_2_offset = slots_[0].stride_bytes + slots_[1].stride_bytes;
    for (size_t i = 0; i < slots_[2].stride_bytes; ++i) {
        EXPECT_EQ(host_data[layer_2_offset + i], static_cast<uint8_t>(i & 0xFF));
    }

    cuda_device_.fill(0, 1, 0x00);
    cuda_device_.fill(1, 2, 0x00);
    cuda_device_.fill(2, 3, 0x00);
    ASSERT_TRUE(copy_engine_->hostToDevice(host_block, device_blocks_, slots_, resolver, *host_pool_));

    auto d0 = cuda_device_.readBack(0, 1);
    auto d1 = cuda_device_.readBack(1, 2);
    auto d2 = cuda_device_.readBack(2, 3);
    ASSERT_EQ(d0.size(), slots_[0].stride_bytes);
    ASSERT_EQ(d1.size(), slots_[1].stride_bytes);
    ASSERT_EQ(d2.size(), slots_[2].stride_bytes);

    for (size_t i = 0; i < slots_[0].stride_bytes; ++i) {
        EXPECT_EQ(d0[i], static_cast<uint8_t>(i & 0xFF));
    }
    for (auto byte : d1) {
        EXPECT_EQ(byte, 0x5A);
    }
    for (size_t i = 0; i < slots_[2].stride_bytes; ++i) {
        EXPECT_EQ(d2[i], static_cast<uint8_t>(i & 0xFF));
    }

    host_pool_->free(host_block);
}

TEST_F(CopyEngineTest, NullDeviceBlockSkipped) {
    // device_blocks[1] is NULL → that slot should be zero-filled in host
    std::vector<BlockIdxType> blocks = {1, NULL_BLOCK_IDX, 3};
    cuda_device_.fill(0, 1, 0xAA);
    cuda_device_.fill(2, 3, 0xCC);

    BlockIdxType host_block = poolMalloc(*host_pool_);
    auto         resolver   = cuda_device_.makeResolver();

    ASSERT_TRUE(copy_engine_->deviceToHost(blocks, host_block, slots_, resolver, *host_pool_));

    const uint8_t* host_data = static_cast<const uint8_t*>(host_pool_->blockBuffer(host_block).addr);
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
    auto resolver = cuda_device_.makeResolver();
    EXPECT_FALSE(copy_engine_->deviceToHost(device_blocks_, NULL_BLOCK_IDX, slots_, resolver, *host_pool_));
    EXPECT_FALSE(copy_engine_->deviceToHost(device_blocks_, 999, slots_, resolver, *host_pool_));
}

TEST_F(CopyEngineTest, DeviceToHostRejectsUnallocatedHostBlock) {
    auto resolver = cuda_device_.makeResolver();
    EXPECT_FALSE(copy_engine_->deviceToHost(device_blocks_, 1, slots_, resolver, *host_pool_));
}

TEST_F(CopyEngineTest, MismatchedSlotCountFails) {
    BlockIdxType host_block = poolMalloc(*host_pool_);
    auto         resolver   = cuda_device_.makeResolver();

    std::vector<BlockIdxType> wrong_blocks = {1, 2};  // 2 blocks but 3 slots
    EXPECT_FALSE(copy_engine_->deviceToHost(wrong_blocks, host_block, slots_, resolver, *host_pool_));

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

class CpuDeviceMemory {
public:
    void allocate(int layer_id, BlockIdxType block_idx, size_t size_bytes) {
        buffers_[makeKey(layer_id, block_idx)] = std::vector<uint8_t>(size_bytes, 0);
    }

    void fill(int layer_id, BlockIdxType block_idx, uint8_t pattern) {
        auto& buffer = buffers_[makeKey(layer_id, block_idx)];
        std::fill(buffer.begin(), buffer.end(), pattern);
    }

    std::vector<uint8_t> read(int layer_id, BlockIdxType block_idx) const {
        auto it = buffers_.find(makeKey(layer_id, block_idx));
        return it == buffers_.end() ? std::vector<uint8_t>{} : it->second;
    }

    DeviceBufferResolver makeResolver() {
        return [this](int layer_id, BlockIdxType block_idx) -> BlockInfo {
            auto it = buffers_.find(makeKey(layer_id, block_idx));
            if (it == buffers_.end()) {
                return BlockInfo{};
            }
            BlockInfo info;
            info.is_cuda    = false;
            info.addr       = it->second.data();
            info.size_bytes = it->second.size();
            return info;
        };
    }

private:
    static uint64_t makeKey(int layer_id, BlockIdxType block_idx) {
        return (static_cast<uint64_t>(layer_id) << 32) | static_cast<uint64_t>(block_idx);
    }

    std::unordered_map<uint64_t, std::vector<uint8_t>> buffers_;
};

class CopyEngineFacadeCpuTest: public ::testing::Test {
protected:
    void SetUp() override {
        slots_ = {
            {0, "layer_0", 64},
            {1, "layer_1", 32},
        };
        host_pool_ = makeHostPool(CopyEngine::computeHostBlockSize(slots_), 4);
        ASSERT_TRUE(host_pool_->init());

        device_blocks_ = {11, 12};
        cpu_device_.allocate(0, device_blocks_[0], slots_[0].stride_bytes);
        cpu_device_.allocate(1, device_blocks_[1], slots_[1].stride_bytes);

        CopyEngineTransferResources resources;
        resources.device_buffer_resolver = cpu_device_.makeResolver();
        resources.layer_slots_resolver   = [this](int) { return slots_; };
        resources.host_pool_resolver     = [this](int) { return host_pool_; };
        copy_engine_                     = std::make_shared<CopyEngine>(std::move(resources));
    }

    TransferDescriptor makeDescriptor(Tier source_tier, Tier target_tier) {
        TransferDescriptor desc;
        desc.component_group_id = 0;
        desc.source_tier        = source_tier;
        desc.target_tier        = target_tier;
        return desc;
    }

    std::vector<MemoryBlockLayerTagSlot> slots_;
    std::shared_ptr<HostBlockPool>       host_pool_;
    std::shared_ptr<CopyEngine>          copy_engine_;
    std::vector<BlockIdxType>            device_blocks_;
    CpuDeviceMemory                      cpu_device_;
};

TEST_F(CopyEngineFacadeCpuTest, SubmitDeviceToHostReturnsCompletedHandle) {
    cpu_device_.fill(0, device_blocks_[0], 0x4A);
    cpu_device_.fill(1, device_blocks_[1], 0x7B);

    BlockIdxType host_block = poolMalloc(*host_pool_);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);

    auto          desc = makeDescriptor(Tier::DEVICE, Tier::HOST);
    TransferEntry entry;
    entry.device_blocks = device_blocks_;
    entry.host_block    = host_block;
    desc.entries        = {entry};

    auto handle = copy_engine_->submit(desc);
    ASSERT_TRUE(handle.valid());
    handle.wait();
    EXPECT_TRUE(handle.done());

    bool callback_called = false;
    handle.onComplete([&](const CopyResult& result) {
        callback_called = true;
        EXPECT_TRUE(result.ok());
        EXPECT_EQ(result.completed_entries, 1u);
    });
    EXPECT_TRUE(callback_called);

    auto result = handle.result();
    EXPECT_TRUE(result.ok());
    EXPECT_EQ(result.request_id, handle.requestId());
    EXPECT_EQ(result.completed_entries, 1u);
    EXPECT_EQ(result.failed_entries, 0u);

    const uint8_t* host_data = static_cast<const uint8_t*>(host_pool_->blockBuffer(host_block).addr);
    for (size_t i = 0; i < slots_[0].stride_bytes; ++i) {
        EXPECT_EQ(host_data[i], 0x4A);
    }
    for (size_t i = slots_[0].stride_bytes; i < CopyEngine::computeHostBlockSize(slots_); ++i) {
        EXPECT_EQ(host_data[i], 0x7B);
    }

    host_pool_->free(host_block);
}

TEST_F(CopyEngineFacadeCpuTest, SubmitHostToDeviceSupportsBatchEntries) {
    BlockIdxType host_block = poolMalloc(*host_pool_);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);
    auto* host_data = static_cast<uint8_t*>(host_pool_->blockBuffer(host_block).addr);
    std::memset(host_data, 0x12, slots_[0].stride_bytes);
    std::memset(host_data + slots_[0].stride_bytes, 0x34, slots_[1].stride_bytes);

    std::vector<BlockIdxType> second_device_blocks = {21, 22};
    cpu_device_.allocate(0, second_device_blocks[0], slots_[0].stride_bytes);
    cpu_device_.allocate(1, second_device_blocks[1], slots_[1].stride_bytes);

    BlockIdxType second_host_block = poolMalloc(*host_pool_);
    ASSERT_NE(second_host_block, NULL_BLOCK_IDX);
    auto* second_host_data = static_cast<uint8_t*>(host_pool_->blockBuffer(second_host_block).addr);
    std::memset(second_host_data, 0x56, slots_[0].stride_bytes);
    std::memset(second_host_data + slots_[0].stride_bytes, 0x78, slots_[1].stride_bytes);

    auto          desc = makeDescriptor(Tier::HOST, Tier::DEVICE);
    TransferEntry first_entry;
    first_entry.host_block    = host_block;
    first_entry.device_blocks = device_blocks_;
    TransferEntry second_entry;
    second_entry.host_block    = second_host_block;
    second_entry.device_blocks = second_device_blocks;
    desc.entries              = {first_entry, second_entry};

    auto result = copy_engine_->submit(desc).result();
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result.completed_entries, 2u);

    auto d0 = cpu_device_.read(0, device_blocks_[0]);
    auto d1 = cpu_device_.read(1, device_blocks_[1]);
    ASSERT_EQ(d0.size(), slots_[0].stride_bytes);
    ASSERT_EQ(d1.size(), slots_[1].stride_bytes);
    for (auto byte : d0) {
        EXPECT_EQ(byte, 0x12);
    }
    for (auto byte : d1) {
        EXPECT_EQ(byte, 0x34);
    }

    auto d2 = cpu_device_.read(0, second_device_blocks[0]);
    auto d3 = cpu_device_.read(1, second_device_blocks[1]);
    ASSERT_EQ(d2.size(), slots_[0].stride_bytes);
    ASSERT_EQ(d3.size(), slots_[1].stride_bytes);
    for (auto byte : d2) {
        EXPECT_EQ(byte, 0x56);
    }
    for (auto byte : d3) {
        EXPECT_EQ(byte, 0x78);
    }

    host_pool_->free(host_block);
    host_pool_->free(second_host_block);
}

TEST(CopyEngineFacadeTest, SubmitInvalidDescriptorReturnsStructuredFailure) {
    CopyEngine         copy_engine;
    TransferDescriptor desc;
    desc.source_tier = Tier::DEVICE;
    desc.target_tier = Tier::HOST;

    auto result = copy_engine.submit(desc).result();
    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.status, CopyStatusCode::INVALID_DESCRIPTOR);
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

        // Create a temp directory for disk pool
        char  tmpdir_template[] = "/tmp/copy_engine_test_XXXXXX";
        char* tmpdir            = ::mkdtemp(tmpdir_template);
        ASSERT_NE(tmpdir, nullptr) << "Failed to create temp directory";
        test_tmpdir_ = tmpdir;

        // Create a disk pool in the temp directory
        auto disk_config                     = std::make_shared<DiskBlockPoolConfig>();
        disk_config->pool_type               = BlockPoolType::DISK;
        disk_config->pool_name               = "copy_engine_disk";
        disk_config->free_block_order_policy = FreeBlockOrderPolicy::ASCENDING_ORDER;
        disk_config->work_dir                = test_tmpdir_;
        disk_config->local_rank              = 0;
        disk_config->world_rank              = 0;
        // Ensure enough space: align block size up to 4096, then multiply by slot count
        size_t aligned_block_size            = ((host_block_size_ + 4095) / 4096) * 4096;
        disk_config->disk_size_bytes         = aligned_block_size * 8;  // 8 slots
        disk_config->payload_bytes           = host_block_size_;
        disk_config->stride_bytes            = aligned_block_size;
        disk_config->buffered_io             = true;

        disk_pool_ = std::make_shared<DiskBlockPool>(disk_config);
        ASSERT_TRUE(disk_pool_->init()) << "DiskBlockPool init failed in " << test_tmpdir_;

        copy_engine_ = std::make_shared<CopyEngine>();
    }

    void TearDown() override {
        disk_pool_.reset();
        if (!test_tmpdir_.empty()) {
            std::string cmd = "rm -rf " + test_tmpdir_;
            int         rc  = ::system(cmd.c_str());
            (void)rc;
        }
    }

    std::string test_tmpdir_;

    std::vector<MemoryBlockLayerTagSlot> slots_;
    size_t                               host_block_size_;
    std::shared_ptr<HostBlockPool>      host_pool_;
    std::shared_ptr<DiskBlockPool>       disk_pool_;
    std::shared_ptr<CopyEngine>          copy_engine_;
};

// TODO: Rewrite this as submit(HOST -> DISK) + submit(DISK -> HOST) facade coverage.
TEST_F(CopyEngineDiskTest, HostToDiskRoundTrip) {
    // Fill host block with pattern
    BlockIdxType host_block = poolMalloc(*host_pool_);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);
    uint8_t* host_data = static_cast<uint8_t*>(host_pool_->blockBuffer(host_block).addr);
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

    host_pool_->free(host_block);
    disk_pool_->free(disk_slot);
}

// TODO: Move this L1/L2/L3 pipeline coverage out of CopyEngine unit tests.
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
    host_pool_->free(host_block_1);

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

    host_pool_->free(host_block_2);
    disk_pool_->free(disk_slot);
}

TEST_F(CopyEngineDiskTest, HostToDiskRejectsUnallocatedDiskBlock) {
    BlockIdxType host_block = poolMalloc(*host_pool_);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);
    EXPECT_FALSE(copy_engine_->hostToDisk(host_block, 1, *host_pool_, *disk_pool_));
    host_pool_->free(host_block);
}

}  // namespace
}  // namespace rtp_llm
