#include <gtest/gtest.h>

#include <cstring>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <torch/torch.h>

#include "rtp_llm/cpp/cache/block_tree_cache/DeviceBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/FullComponentGroup.h"
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

struct DeviceLayerBufferSpec {
    size_t kv_bytes{0};
    size_t scale_bytes{0};
};

// Build a DeviceBlockPool with the given per-layer layout.
static DeviceBlockPoolPtr makeDevicePool(const std::vector<DeviceLayerBufferSpec>& specs,
                                         size_t                                    usable_count,
                                         const std::string&                        pool_name) {
    const auto physical_block_count = usable_count + 1;

    auto config                     = std::make_shared<DeviceBlockPoolConfig>();
    config->pool_type               = BlockPoolType::DEVICE;
    config->pool_name               = pool_name;
    config->physical_block_count    = physical_block_count;
    config->allocation_type         = AllocationType::DEVICE;
    config->use_cuda_malloc_backing = false;

    size_t offset = 0;
    for (const auto& spec : specs) {
        MemoryLayoutConfig layout;
        layout.layer_num                = 1;
        layout.block_num                = static_cast<uint32_t>(physical_block_count);
        layout.dtype                    = TYPE_INT8;
        layout.kv_cache_offset_bytes    = offset;
        layout.kv_block_stride_bytes    = spec.kv_bytes;
        layout.kv_block_pool_size_bytes = physical_block_count * spec.kv_bytes;
        layout.block_stride_bytes       = spec.kv_bytes + spec.scale_bytes;
        layout.total_size_bytes         = layout.kv_block_pool_size_bytes;
        offset += layout.kv_block_pool_size_bytes;

        if (spec.scale_bytes > 0) {
            layout.enable_kv_scale          = true;
            layout.kv_scale_offset_bytes    = offset;
            layout.kv_scale_stride_bytes    = spec.scale_bytes;
            layout.kv_scale_pool_size_bytes = physical_block_count * spec.scale_bytes;
            layout.total_size_bytes += layout.kv_scale_pool_size_bytes;
            offset += layout.kv_scale_pool_size_bytes;
        }

        layout.local_head_num_kv          = 1;
        layout.seq_size_per_block         = 1;
        layout.kernel_blocks_per_kv_block = 1;
        config->memory_layouts.push_back(layout);
    }
    config->total_size_bytes = offset;

    auto pool = std::make_shared<DeviceBlockPool>(config);
    RTP_LLM_CHECK(pool->init());
    return pool;
}

// DeviceBlockPool backing is always CUDA, so the byte view is unconditionally a CUDA tensor.
static torch::Tensor makePoolByteTensor(void* addr, size_t bytes) {
    return torch::from_blob(
        addr, {static_cast<int64_t>(bytes)}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
}

static void fillDeviceLayer(const DeviceBlockPoolPtr&                     pool,
                            int                                    layer_id,
                            BlockIdxType                           block,
                            const std::vector<uint8_t>&             patterns) {
    auto buffers = pool->blockBuffers(layer_id, block);
    ASSERT_EQ(buffers.size(), patterns.size());
    for (size_t i = 0; i < buffers.size(); ++i) {
        makePoolByteTensor(buffers[i].addr, buffers[i].bytes).fill_(patterns[i]);
    }
    if (pool->where() == MemoryType::MEMORY_GPU) {
        cudaDeviceSynchronize();
    }
}

static void fillDeviceLayerSequential(const DeviceBlockPoolPtr&                     pool,
                                      int                                    layer_id,
                                      BlockIdxType                           block) {
    auto buffers = pool->blockBuffers(layer_id, block);
    for (const auto& buffer : buffers) {
        std::vector<uint8_t> cpu_data(buffer.bytes);
        for (size_t i = 0; i < buffer.bytes; ++i) {
            cpu_data[i] = static_cast<uint8_t>(i & 0xFF);
        }
        auto cpu_tensor =
            torch::from_blob(cpu_data.data(), {static_cast<int64_t>(buffer.bytes)}, torch::kUInt8).clone();
        makePoolByteTensor(buffer.addr, buffer.bytes).copy_(cpu_tensor);
    }
    if (pool->where() == MemoryType::MEMORY_GPU) {
        cudaDeviceSynchronize();
    }
}

static std::vector<uint8_t> readDeviceLayer(const DeviceBlockPoolPtr& pool, int layer_id, BlockIdxType block) {
    if (pool->where() == MemoryType::MEMORY_GPU) {
        cudaDeviceSynchronize();
    }

    std::vector<uint8_t> out;
    auto                 buffers = pool->blockBuffers(layer_id, block);
    for (const auto& buffer : buffers) {
        auto  tensor = makePoolByteTensor(buffer.addr, buffer.bytes);
        auto  cpu    = tensor.cpu();
        auto* ptr    = cpu.data_ptr<uint8_t>();
        out.insert(out.end(), ptr, ptr + buffer.bytes);
    }
    return out;
}

static Component makeComponent(int component_id,
                               int component_group_id,
                               const std::vector<MemoryBlockLayerTagSlot>& slots,
                               int device_pool_index = 0) {
    Component component;
    component.component_id                 = component_id;
    component.component_group_id           = component_group_id;
    component.type                         = CacheGroupType::FULL;
    component.memory_block_layer_tag_slots = slots;
    component.device_pool_index            = device_pool_index;
    return component;
}

static ComponentGroupPtr makeGroup(int group_id,
                                   std::vector<int> component_indices,
                                   std::vector<DeviceBlockPoolPtr> device_pools,
                                   std::shared_ptr<HostBlockPool> host_pool,
                                   std::shared_ptr<DiskBlockPool> disk_pool = nullptr) {
    auto group                  = std::make_shared<FullComponentGroup>();
    group->component_group_id   = group_id;
    group->group_type           = CacheGroupType::FULL;
    group->component_indices    = std::move(component_indices);
    group->setDevicePools(std::move(device_pools));
    group->setHostPool(std::move(host_pool));
    group->setDiskPool(std::move(disk_pool));
    return group;
}

static TransferDescriptor makeDescriptor(Tier                             source_tier,
                                         Tier                             target_tier,
                                         const std::vector<BlockIdxType>& device_blocks,
                                         BlockIdxType                     host_block = NULL_BLOCK_IDX,
                                         BlockIdxType                     disk_block = NULL_BLOCK_IDX,
                                         int                              group_id   = 0) {
    if (source_tier == Tier::DEVICE && target_tier == Tier::HOST) {
        return TransferDescriptor::deviceToHost(nullptr, group_id, device_blocks, host_block);
    }
    if (source_tier == Tier::HOST && target_tier == Tier::DEVICE) {
        return TransferDescriptor::hostToDevice(nullptr, group_id, host_block, device_blocks);
    }
    if (source_tier == Tier::HOST && target_tier == Tier::DISK) {
        return TransferDescriptor::hostToDisk(nullptr, group_id, host_block, disk_block);
    }
    if (source_tier == Tier::DISK && target_tier == Tier::HOST) {
        return TransferDescriptor::diskToHost(nullptr, group_id, disk_block, host_block);
    }

    TransferDescriptor desc;
    desc.component_group_id = group_id;
    desc.source_tier        = source_tier;
    desc.target_tier        = target_tier;
    desc.device_blocks      = device_blocks;
    desc.host_block         = host_block;
    desc.disk_block         = disk_block;
    return desc;
}

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

// ---- CopyEngine submit() tests (real CUDA) ----

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

        // Create host pool — 10 usable blocks
        host_pool_ = makeHostPool(host_block_size_, 10);

        device_pool_ = makeDevicePool({{100, 0}, {200, 0}, {150, 0}}, 10, "copy_engine_device");
        device_block_ = poolMalloc(*device_pool_);
        ASSERT_NE(device_block_, NULL_BLOCK_IDX);
        device_blocks_ = {device_block_};

        component_      = makeComponent(0, 0, slots_);
        component_group_ = makeGroup(0, {0}, {device_pool_}, host_pool_);
        copy_engine_ =
            std::make_shared<CopyEngine>(std::vector<ComponentGroupPtr>{component_group_},
                                         std::vector<Component>{component_});
    }

    std::vector<MemoryBlockLayerTagSlot> slots_;
    size_t                               host_block_size_;
    std::shared_ptr<HostBlockPool>      host_pool_;
    std::shared_ptr<CopyEngine>          copy_engine_;
    DeviceBlockPoolPtr                  device_pool_;
    BlockIdxType                         device_block_;
    std::vector<BlockIdxType>            device_blocks_;
    Component                            component_;
    ComponentGroupPtr                    component_group_;
};

TEST_F(CopyEngineTest, SubmitDeviceToHostPacking) {
    fillDeviceLayer(device_pool_, 0, device_block_, {0xAA});
    fillDeviceLayer(device_pool_, 1, device_block_, {0xBB});
    fillDeviceLayer(device_pool_, 2, device_block_, {0xCC});

    BlockIdxType host_block = poolMalloc(*host_pool_);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);

    auto desc  = makeDescriptor(Tier::DEVICE, Tier::HOST, device_blocks_, host_block);
    auto result = copy_engine_->submit(desc).result();
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result.completed_entries, 1u);

    const uint8_t* host_data = static_cast<const uint8_t*>(host_pool_->blockBuffer(host_block).addr);
    for (size_t i = 0; i < 100; ++i)
        EXPECT_EQ(host_data[i], 0xAA) << "byte " << i;
    for (size_t i = 100; i < 300; ++i)
        EXPECT_EQ(host_data[i], 0xBB) << "byte " << i;
    for (size_t i = 300; i < 450; ++i)
        EXPECT_EQ(host_data[i], 0xCC) << "byte " << i;

    host_pool_->free(host_block);
}

TEST_F(CopyEngineTest, SubmitHostToDeviceUnpacking) {
    fillDeviceLayer(device_pool_, 0, device_block_, {0x11});
    fillDeviceLayer(device_pool_, 1, device_block_, {0x22});
    fillDeviceLayer(device_pool_, 2, device_block_, {0x33});

    BlockIdxType host_block = poolMalloc(*host_pool_);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);

    // Pack D2H first
    auto d2h_desc  = makeDescriptor(Tier::DEVICE, Tier::HOST, device_blocks_, host_block);
    ASSERT_TRUE(copy_engine_->submit(d2h_desc).result().ok());

    // Clear device buffers
    fillDeviceLayer(device_pool_, 0, device_block_, {0x00});
    fillDeviceLayer(device_pool_, 1, device_block_, {0x00});
    fillDeviceLayer(device_pool_, 2, device_block_, {0x00});

    // Unpack H2D
    auto h2d_desc  = makeDescriptor(Tier::HOST, Tier::DEVICE, device_blocks_, host_block);
    ASSERT_TRUE(copy_engine_->submit(h2d_desc).result().ok());

    auto d0 = readDeviceLayer(device_pool_, 0, device_block_);
    auto d1 = readDeviceLayer(device_pool_, 1, device_block_);
    auto d2 = readDeviceLayer(device_pool_, 2, device_block_);
    ASSERT_EQ(d0.size(), 100u);
    ASSERT_EQ(d1.size(), 200u);
    ASSERT_EQ(d2.size(), 150u);
    for (size_t i = 0; i < 100; ++i) EXPECT_EQ(d0[i], 0x11);
    for (size_t i = 0; i < 200; ++i) EXPECT_EQ(d1[i], 0x22);
    for (size_t i = 0; i < 150; ++i) EXPECT_EQ(d2[i], 0x33);

    host_pool_->free(host_block);
}

TEST_F(CopyEngineTest, SubmitRoundTripSequentialData) {
    fillDeviceLayerSequential(device_pool_, 0, device_block_);
    fillDeviceLayer(device_pool_, 1, device_block_, {0x5A});
    fillDeviceLayerSequential(device_pool_, 2, device_block_);

    BlockIdxType host_block = poolMalloc(*host_pool_);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);

    auto d2h_desc = makeDescriptor(Tier::DEVICE, Tier::HOST, device_blocks_, host_block);
    ASSERT_TRUE(copy_engine_->submit(d2h_desc).result().ok());

    const uint8_t* host_data = static_cast<const uint8_t*>(host_pool_->blockBuffer(host_block).addr);
    for (size_t i = 0; i < slots_[0].stride_bytes; ++i)
        EXPECT_EQ(host_data[i], static_cast<uint8_t>(i & 0xFF));
    for (size_t i = 0; i < slots_[1].stride_bytes; ++i)
        EXPECT_EQ(host_data[slots_[0].stride_bytes + i], 0x5A);
    const size_t off2 = slots_[0].stride_bytes + slots_[1].stride_bytes;
    for (size_t i = 0; i < slots_[2].stride_bytes; ++i)
        EXPECT_EQ(host_data[off2 + i], static_cast<uint8_t>(i & 0xFF));

    fillDeviceLayer(device_pool_, 0, device_block_, {0x00});
    fillDeviceLayer(device_pool_, 1, device_block_, {0x00});
    fillDeviceLayer(device_pool_, 2, device_block_, {0x00});
    auto h2d_desc = makeDescriptor(Tier::HOST, Tier::DEVICE, device_blocks_, host_block);
    ASSERT_TRUE(copy_engine_->submit(h2d_desc).result().ok());

    auto d0 = readDeviceLayer(device_pool_, 0, device_block_);
    auto d1 = readDeviceLayer(device_pool_, 1, device_block_);
    auto d2 = readDeviceLayer(device_pool_, 2, device_block_);
    for (size_t i = 0; i < slots_[0].stride_bytes; ++i)
        EXPECT_EQ(d0[i], static_cast<uint8_t>(i & 0xFF));
    for (auto byte : d1) EXPECT_EQ(byte, 0x5A);
    for (size_t i = 0; i < slots_[2].stride_bytes; ++i)
        EXPECT_EQ(d2[i], static_cast<uint8_t>(i & 0xFF));

    host_pool_->free(host_block);
}

TEST_F(CopyEngineTest, SubmitNullDeviceComponentZeroFillsAllSlots) {
    std::vector<BlockIdxType> blocks = {NULL_BLOCK_IDX};

    BlockIdxType host_block = poolMalloc(*host_pool_);
    auto desc  = makeDescriptor(Tier::DEVICE, Tier::HOST, blocks, host_block);
    ASSERT_TRUE(copy_engine_->submit(desc).result().ok());

    const uint8_t* host_data = static_cast<const uint8_t*>(host_pool_->blockBuffer(host_block).addr);
    for (size_t i = 0; i < host_block_size_; ++i) EXPECT_EQ(host_data[i], 0x00);

    host_pool_->free(host_block);
}

TEST_F(CopyEngineTest, SubmitInvalidHostBlockReturnsStructuredFailure) {
    auto desc  = makeDescriptor(Tier::DEVICE, Tier::HOST, device_blocks_, NULL_BLOCK_IDX);
    auto result = copy_engine_->submit(desc).result();
    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.status, CopyStatus::INVALID_ARGS);
}

TEST_F(CopyEngineTest, SubmitRejectsUnallocatedHostBlock) {
    auto desc  = makeDescriptor(Tier::DEVICE, Tier::HOST, device_blocks_, 1);
    auto result = copy_engine_->submit(desc).result();
    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.status, CopyStatus::INVALID_ARGS);
}

TEST_F(CopyEngineTest, SubmitMismatchedSlotCountFails) {
    BlockIdxType host_block = poolMalloc(*host_pool_);
    std::vector<BlockIdxType> wrong_blocks = {device_block_, device_block_};
    auto desc  = makeDescriptor(Tier::DEVICE, Tier::HOST, wrong_blocks, host_block);
    auto result = copy_engine_->submit(desc).result();
    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.status, CopyStatus::INVALID_ARGS);
    host_pool_->free(host_block);
}

TEST_F(CopyEngineTest, SubmitInvalidDescriptorReturnsStructuredFailure) {
    TransferDescriptor desc;
    desc.source_tier = Tier::DEVICE;
    desc.target_tier = Tier::HOST;
    // No component_group_id.
    auto result = copy_engine_->submit(desc).result();
    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.status, CopyStatus::INVALID_ARGS);
}

TEST_F(CopyEngineTest, SubmitMissingDevicePoolFails) {
    BlockIdxType host_block = poolMalloc(*host_pool_);
    auto desc  = makeDescriptor(Tier::DEVICE, Tier::HOST, device_blocks_, host_block);
    auto group = makeGroup(0, {0}, {}, host_pool_);
    auto copy_engine = std::make_shared<CopyEngine>(std::vector<ComponentGroupPtr>{group},
                                                    std::vector<Component>{component_});
    auto result = copy_engine->submit(desc).result();
    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.status, CopyStatus::INVALID_ARGS);
    host_pool_->free(host_block);
}

TEST_F(CopyEngineTest, SubmitDeviceTransferWithEmptyComponentsFails) {
    BlockIdxType host_block = poolMalloc(*host_pool_);
    auto desc  = makeDescriptor(Tier::DEVICE, Tier::HOST, device_blocks_, host_block);
    auto group = makeGroup(0, {}, {device_pool_}, host_pool_);
    auto copy_engine = std::make_shared<CopyEngine>(std::vector<ComponentGroupPtr>{group}, std::vector<Component>{});
    auto result = copy_engine->submit(desc).result();
    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.status, CopyStatus::INVALID_ARGS);
    host_pool_->free(host_block);
}

TEST_F(CopyEngineTest, SubmitReturnsCompletedHandle) {
    fillDeviceLayer(device_pool_, 0, device_block_, {0xAA});
    fillDeviceLayer(device_pool_, 1, device_block_, {0xBB});
    fillDeviceLayer(device_pool_, 2, device_block_, {0xCC});

    BlockIdxType host_block = poolMalloc(*host_pool_);
    auto desc = makeDescriptor(Tier::DEVICE, Tier::HOST, device_blocks_, host_block);

    auto handle = copy_engine_->submit(desc);
    ASSERT_TRUE(handle.valid());
    EXPECT_TRUE(handle.done());
    handle.wait();

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

TEST_F(CopyEngineTest, SubmitHostToDeviceIndependentDescriptors) {
    BlockIdxType second_device_block = poolMalloc(*device_pool_);
    ASSERT_NE(second_device_block, NULL_BLOCK_IDX);
    std::vector<BlockIdxType> second_device_blocks = {second_device_block};

    BlockIdxType host_block_1 = poolMalloc(*host_pool_);
    auto* host_data_1 = static_cast<uint8_t*>(host_pool_->blockBuffer(host_block_1).addr);
    std::memset(host_data_1, 0x12, slots_[0].stride_bytes);
    std::memset(host_data_1 + slots_[0].stride_bytes, 0x34, slots_[1].stride_bytes);
    std::memset(host_data_1 + slots_[0].stride_bytes + slots_[1].stride_bytes, 0x9A, slots_[2].stride_bytes);

    BlockIdxType host_block_2 = poolMalloc(*host_pool_);
    auto* host_data_2 = static_cast<uint8_t*>(host_pool_->blockBuffer(host_block_2).addr);
    std::memset(host_data_2, 0x56, slots_[0].stride_bytes);
    std::memset(host_data_2 + slots_[0].stride_bytes, 0x78, slots_[1].stride_bytes);
    std::memset(host_data_2 + slots_[0].stride_bytes + slots_[1].stride_bytes, 0xBC, slots_[2].stride_bytes);

    auto result_1 =
        copy_engine_->submit(makeDescriptor(Tier::HOST, Tier::DEVICE, device_blocks_, host_block_1)).result();
    ASSERT_TRUE(result_1.ok());
    EXPECT_EQ(result_1.completed_entries, 1u);

    auto result_2 =
        copy_engine_->submit(makeDescriptor(Tier::HOST, Tier::DEVICE, second_device_blocks, host_block_2)).result();
    ASSERT_TRUE(result_2.ok());
    EXPECT_EQ(result_2.completed_entries, 1u);

    for (auto byte : readDeviceLayer(device_pool_, 0, device_block_)) EXPECT_EQ(byte, 0x12);
    for (auto byte : readDeviceLayer(device_pool_, 1, device_block_)) EXPECT_EQ(byte, 0x34);
    for (auto byte : readDeviceLayer(device_pool_, 2, device_block_)) EXPECT_EQ(byte, 0x9A);
    for (auto byte : readDeviceLayer(device_pool_, 0, second_device_block)) EXPECT_EQ(byte, 0x56);
    for (auto byte : readDeviceLayer(device_pool_, 1, second_device_block)) EXPECT_EQ(byte, 0x78);
    for (auto byte : readDeviceLayer(device_pool_, 2, second_device_block)) EXPECT_EQ(byte, 0xBC);

    host_pool_->free(host_block_1);
    host_pool_->free(host_block_2);
    device_pool_->free(second_device_block);
}

TEST_F(CopyEngineTest, SubmitUsesComponentOrdinalForMultipleDevicePools) {
    std::vector<MemoryBlockLayerTagSlot> comp0_slots = {{0, "comp0", 64}};
    std::vector<MemoryBlockLayerTagSlot> comp1_slots = {{0, "comp1", 96}};
    auto host_pool = makeHostPool(160, 2);
    auto pool0     = makeDevicePool({{64, 0}}, 4, "copy_engine_multi_pool_0");
    auto pool1     = makeDevicePool({{96, 0}}, 4, "copy_engine_multi_pool_1");
    auto block0    = poolMalloc(*pool0);
    auto block1    = poolMalloc(*pool1);
    ASSERT_NE(block0, NULL_BLOCK_IDX);
    ASSERT_NE(block1, NULL_BLOCK_IDX);

    fillDeviceLayer(pool0, 0, block0, {0xA1});
    fillDeviceLayer(pool1, 0, block1, {0xB2});

    std::vector<Component> components = {
        makeComponent(0, 0, comp0_slots, 0),
        makeComponent(1, 0, comp1_slots, 1),
    };
    auto group = makeGroup(0, {0, 1}, {pool0, pool1}, host_pool);
    auto copy_engine =
        std::make_shared<CopyEngine>(std::vector<ComponentGroupPtr>{group}, components);

    auto host_block = poolMalloc(*host_pool);
    auto desc       = makeDescriptor(Tier::DEVICE, Tier::HOST, {block0, block1}, host_block);
    auto result     = copy_engine->submit(desc).result();
    ASSERT_TRUE(result.ok());

    const auto* host_data = static_cast<const uint8_t*>(host_pool->blockBuffer(host_block).addr);
    for (size_t i = 0; i < 64; ++i) EXPECT_EQ(host_data[i], 0xA1);
    for (size_t i = 64; i < 160; ++i) EXPECT_EQ(host_data[i], 0xB2);

    host_pool->free(host_block);
    pool0->free(block0);
    pool1->free(block1);
}

TEST_F(CopyEngineTest, SubmitCopiesAllBuffersReturnedByDeviceBlockPool) {
    std::vector<MemoryBlockLayerTagSlot> slots = {{0, "fp8_kv", 80}};
    auto host_pool  = makeHostPool(CopyEngine::computeHostBlockSize(slots), 2);
    auto scale_pool = makeDevicePool({{64, 16}}, 4, "copy_engine_scale_device");
    auto block      = poolMalloc(*scale_pool);
    ASSERT_NE(block, NULL_BLOCK_IDX);

    fillDeviceLayer(scale_pool, 0, block, {0x4A, 0x7B});

    auto component   = makeComponent(0, 0, slots);
    auto group       = makeGroup(0, {0}, {scale_pool}, host_pool);
    auto copy_engine = std::make_shared<CopyEngine>(std::vector<ComponentGroupPtr>{group},
                                                    std::vector<Component>{component});
    auto host_block  = poolMalloc(*host_pool);
    auto d2h_desc    = makeDescriptor(Tier::DEVICE, Tier::HOST, {block}, host_block);
    auto d2h_result  = copy_engine->submit(d2h_desc).result();
    ASSERT_TRUE(d2h_result.ok());

    const auto* host_data = static_cast<const uint8_t*>(host_pool->blockBuffer(host_block).addr);
    for (size_t i = 0; i < 64; ++i) EXPECT_EQ(host_data[i], 0x4A);
    for (size_t i = 64; i < 80; ++i) EXPECT_EQ(host_data[i], 0x7B);

    auto* mutable_host_data = static_cast<uint8_t*>(host_pool->blockBuffer(host_block).addr);
    std::memset(mutable_host_data, 0x11, 64);
    std::memset(mutable_host_data + 64, 0x22, 16);

    fillDeviceLayer(scale_pool, 0, block, {0x00, 0x00});
    auto h2d_desc   = makeDescriptor(Tier::HOST, Tier::DEVICE, {block}, host_block);
    auto h2d_result = copy_engine->submit(h2d_desc).result();
    ASSERT_TRUE(h2d_result.ok());

    auto device_bytes = readDeviceLayer(scale_pool, 0, block);
    ASSERT_EQ(device_bytes.size(), 80u);
    for (size_t i = 0; i < 64; ++i) EXPECT_EQ(device_bytes[i], 0x11);
    for (size_t i = 64; i < 80; ++i) EXPECT_EQ(device_bytes[i], 0x22);

    host_pool->free(host_block);
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

        device_pool_  = makeDevicePool({{128, 0}, {256, 0}}, 4, "copy_engine_disk_device");
        device_block_ = poolMalloc(*device_pool_);
        ASSERT_NE(device_block_, NULL_BLOCK_IDX);

        component_       = makeComponent(0, 0, slots_);
        component_group_ = makeGroup(0, {0}, {device_pool_}, host_pool_, disk_pool_);
        copy_engine_ =
            std::make_shared<CopyEngine>(std::vector<ComponentGroupPtr>{component_group_},
                                         std::vector<Component>{component_});
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
    DeviceBlockPoolPtr                  device_pool_;
    BlockIdxType                         device_block_;
    Component                            component_;
    ComponentGroupPtr                    component_group_;
};

TEST_F(CopyEngineDiskTest, SubmitHostToDiskRoundTrip) {
    BlockIdxType host_block = poolMalloc(*host_pool_);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);
    uint8_t* host_data = static_cast<uint8_t*>(host_pool_->blockBuffer(host_block).addr);
    for (size_t i = 0; i < host_block_size_; ++i)
        host_data[i] = static_cast<uint8_t>(i & 0xFF);

    auto disk_slot_opt = disk_pool_->malloc();
    ASSERT_TRUE(disk_slot_opt.has_value());
    int32_t disk_slot = disk_slot_opt.value();

    // H2Disk
    auto h2d_desc = makeDescriptor(Tier::HOST, Tier::DISK, {}, host_block, disk_slot);
    ASSERT_TRUE(copy_engine_->submit(h2d_desc).result().ok());

    // Clear host
    std::memset(host_data, 0, host_block_size_);

    // Disk2H
    auto d2h_desc = makeDescriptor(Tier::DISK, Tier::HOST, {}, host_block, disk_slot);
    ASSERT_TRUE(copy_engine_->submit(d2h_desc).result().ok());

    for (size_t i = 0; i < host_block_size_; ++i)
        EXPECT_EQ(host_data[i], static_cast<uint8_t>(i & 0xFF)) << "byte " << i;

    host_pool_->free(host_block);
    disk_pool_->free(disk_slot);
}

TEST_F(CopyEngineDiskTest, SubmitHostToDiskDoesNotRequireComponents) {
    auto group = makeGroup(0, {0}, {}, host_pool_, disk_pool_);
    auto copy_engine = std::make_shared<CopyEngine>(std::vector<ComponentGroupPtr>{group}, std::vector<Component>{});

    BlockIdxType host_block = poolMalloc(*host_pool_);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);
    auto* host_data = static_cast<uint8_t*>(host_pool_->blockBuffer(host_block).addr);
    std::memset(host_data, 0x5C, host_block_size_);

    auto disk_slot_opt = disk_pool_->malloc();
    ASSERT_TRUE(disk_slot_opt.has_value());
    auto disk_slot = disk_slot_opt.value();

    auto desc = makeDescriptor(Tier::HOST, Tier::DISK, {}, host_block, disk_slot);
    auto result = copy_engine->submit(desc).result();
    EXPECT_TRUE(result.ok());

    host_pool_->free(host_block);
    disk_pool_->free(disk_slot);
}

// TODO: Move this L1/L2/L3 pipeline coverage out of CopyEngine unit tests.
TEST_F(CopyEngineDiskTest, FullPipeline_D2H_H2Disk_Disk2H_H2D) {
    std::vector<BlockIdxType> device_blocks = {device_block_};

    fillDeviceLayerSequential(device_pool_, 0, device_block_);
    {
        std::vector<uint8_t> cpu_data(256);
        for (size_t i = 0; i < 256; ++i)
            cpu_data[i] = static_cast<uint8_t>(i * 3 & 0xFF);
        auto cpu_tensor = torch::from_blob(cpu_data.data(), {256}, torch::kUInt8).clone();
        auto buffers    = device_pool_->blockBuffers(1, device_block_);
        ASSERT_EQ(buffers.size(), 1u);
        makePoolByteTensor(buffers[0].addr, buffers[0].bytes).copy_(cpu_tensor);
        cudaDeviceSynchronize();
    }

    // Step 1: D2H
    BlockIdxType host_block_1 = poolMalloc(*host_pool_);
    auto d2h_desc = makeDescriptor(Tier::DEVICE, Tier::HOST, device_blocks, host_block_1);
    ASSERT_TRUE(copy_engine_->submit(d2h_desc).result().ok());

    // Step 2: H2Disk
    auto disk_slot_opt = disk_pool_->malloc();
    ASSERT_TRUE(disk_slot_opt.has_value());
    int32_t disk_slot = disk_slot_opt.value();
    auto h2disk_desc = makeDescriptor(Tier::HOST, Tier::DISK, {}, host_block_1, disk_slot);
    ASSERT_TRUE(copy_engine_->submit(h2disk_desc).result().ok());
    host_pool_->free(host_block_1);

    // Step 3: Disk2H
    BlockIdxType host_block_2 = poolMalloc(*host_pool_);
    ASSERT_NE(host_block_2, NULL_BLOCK_IDX);
    auto disk2h_desc = makeDescriptor(Tier::DISK, Tier::HOST, {}, host_block_2, disk_slot);
    ASSERT_TRUE(copy_engine_->submit(disk2h_desc).result().ok());

    // Step 4: H2D
    fillDeviceLayer(device_pool_, 0, device_block_, {0x00});
    fillDeviceLayer(device_pool_, 1, device_block_, {0x00});
    auto h2d_desc = makeDescriptor(Tier::HOST, Tier::DEVICE, device_blocks, host_block_2);
    ASSERT_TRUE(copy_engine_->submit(h2d_desc).result().ok());

    auto d0 = readDeviceLayer(device_pool_, 0, device_block_);
    auto d1 = readDeviceLayer(device_pool_, 1, device_block_);
    for (size_t i = 0; i < 128; ++i)
        EXPECT_EQ(d0[i], static_cast<uint8_t>(i & 0xFF));
    for (size_t i = 0; i < 256; ++i)
        EXPECT_EQ(d1[i], static_cast<uint8_t>(i * 3 & 0xFF));

    host_pool_->free(host_block_2);
    disk_pool_->free(disk_slot);
}

TEST_F(CopyEngineDiskTest, SubmitHostToDiskRejectsUnallocatedDiskBlock) {
    BlockIdxType host_block = poolMalloc(*host_pool_);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);
    auto desc = makeDescriptor(Tier::HOST, Tier::DISK, {}, host_block, 1);
    auto result = copy_engine_->submit(desc).result();
    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.status, CopyStatus::INVALID_ARGS);
    host_pool_->free(host_block);
}

}  // namespace
}  // namespace rtp_llm
