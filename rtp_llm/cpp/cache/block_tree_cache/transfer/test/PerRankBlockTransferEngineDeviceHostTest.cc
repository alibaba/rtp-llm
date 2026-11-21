#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <torch/torch.h>

#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/DeviceBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/FullGroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/PerRankBlockTransferEngine.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/test/PerRankBlockTransferEngineTestUtils.h"
#include "rtp_llm/models_py/bindings/NoBlockCopy.h"
#include "rtp_llm/cpp/utils/AsyncProfilingScope.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferStageTimer.h"

namespace rtp_llm {
class TransferStageTimerTestPeer {
public:
    static void setAllocationProbe(void (*probe)(const char*)) {
        TransferStageTimer::allocation_probe_ = probe;
    }
};
namespace {

using block_transfer_engine_test::TempDirGuard;
using block_transfer_engine_test::DirectAlignmentDiskBlockIO;
using block_transfer_engine_test::TestGroupConfig;
using block_transfer_engine_test::expectStatus;
using block_transfer_engine_test::makeDescriptor;
using block_transfer_engine_test::makeDiskPool;
using block_transfer_engine_test::makeHostPool;
using block_transfer_engine_test::makeTestGroupBase;
using block_transfer_engine_test::makeTestGroupSet;
using block_transfer_engine_test::makeTestTopology;
using block_transfer_engine_test::makeTransferTask;
using block_transfer_engine_test::poolMalloc;
using block_transfer_engine_test::releasePoolBlock;
using block_transfer_engine_test::executeSucceeded;

struct DeviceLayerBufferSpec {
    size_t kv_bytes{0};
    size_t scale_bytes{0};
};

static DeviceBlockPoolPtr
makeDevicePool(const std::vector<DeviceLayerBufferSpec>& specs, size_t usable_count, const std::string& pool_name) {
    const auto physical_block_count = usable_count + 1;

    auto config                       = std::make_shared<DeviceBlockPoolConfig>();
    config->pool_type                 = BlockPoolType::DEVICE;
    config->pool_name                 = pool_name;
    config->physical_block_count      = physical_block_count;
    config->use_device_malloc_backing = true;

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
    // Finish layout initialization before handing the backing to a worker's copy stream.
    RTP_LLM_CHECK(cudaDeviceSynchronize() == cudaSuccess);
    return pool;
}

// DeviceBlockPool backing is always CUDA, so the byte view is unconditionally a CUDA tensor.
static torch::Tensor makePoolByteTensor(void* addr, size_t bytes) {
    return torch::from_blob(
        addr, {static_cast<int64_t>(bytes)}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
}

static void fillDeviceLayer(const DeviceBlockPoolPtr&   pool,
                            int                         layer_id,
                            BlockIdxType                block,
                            const std::vector<uint8_t>& patterns) {
    auto buffers = pool->convertIndexToBuffer(layer_id, block);
    ASSERT_EQ(buffers.size(), patterns.size());
    for (size_t i = 0; i < buffers.size(); ++i) {
        makePoolByteTensor(buffers[i].addr, buffers[i].size_bytes).fill_(patterns[i]);
    }
    if (pool->where() == MemoryType::MEMORY_GPU) {
        cudaDeviceSynchronize();
    }
}

static void fillDeviceLayerSequential(const DeviceBlockPoolPtr& pool, int layer_id, BlockIdxType block) {
    auto buffers = pool->convertIndexToBuffer(layer_id, block);
    for (const auto& buffer : buffers) {
        std::vector<uint8_t> cpu_data(buffer.size_bytes);
        for (size_t i = 0; i < buffer.size_bytes; ++i) {
            cpu_data[i] = static_cast<uint8_t>(i & 0xFF);
        }
        auto cpu_tensor =
            torch::from_blob(cpu_data.data(), {static_cast<int64_t>(buffer.size_bytes)}, torch::kUInt8).clone();
        makePoolByteTensor(buffer.addr, buffer.size_bytes).copy_(cpu_tensor);
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
    auto                 buffers = pool->convertIndexToBuffer(layer_id, block);
    for (const auto& buffer : buffers) {
        auto  tensor = makePoolByteTensor(buffer.addr, buffer.size_bytes);
        auto  cpu    = tensor.cpu();
        auto* ptr    = cpu.data_ptr<uint8_t>();
        out.insert(out.end(), ptr, ptr + buffer.size_bytes);
    }
    return out;
}

static TestGroupConfig makeGroupBase(std::vector<int> layer_ids, size_t kv_bytes, size_t scale_bytes = 0) {
    auto policy                = defaultCacheGroupPolicy(CacheGroupType::FULL);
    policy.enable_prefix_reuse = true;
    return makeTestGroupBase(std::move(policy), std::move(layer_ids), kv_bytes, scale_bytes);
}

static GroupSetPtr makeDeviceHostGroup(const std::vector<DeviceBlockPoolPtr>& device_pools,
                                       const std::shared_ptr<HostBlockPool>&  host_pool,
                                       std::vector<TestGroupConfig>           groups,
                                       const std::shared_ptr<DiskBlockPool>&  disk_pool = nullptr) {
    auto topology = makeTestTopology(std::move(groups));
    return makeTestGroupSet(0, topology, topology->groupTags(), device_pools, host_pool, disk_pool, true);
}

static std::shared_ptr<PerRankBlockTransferEngine> makeEngine(std::vector<GroupSetPtr> groups) {
    return std::make_shared<PerRankBlockTransferEngine>(std::move(groups), true, 128, 8, 4, 10000, nullptr, 0, 4);
}

class PerRankBlockTransferEngineTest: public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_TRUE(torch::cuda::is_available()) << "CUDA not available, cannot run GPU tests";

        layer_bytes_     = {100, 100, 100};
        host_block_size_ = 300;

        // Create host pool — 10 usable blocks
        host_pool_ = makeHostPool(host_block_size_, 10, true);

        device_pool_  = makeDevicePool({{100, 0}, {100, 0}, {100, 0}}, 10, "per_rank_transfer_engine_device");
        device_block_ = poolMalloc(*device_pool_);
        ASSERT_NE(device_block_, NULL_BLOCK_IDX);
        device_blocks_ = {device_block_};

        group_set_ = makeDeviceHostGroup({device_pool_}, host_pool_, {makeGroupBase({0, 1, 2}, 100)});
        ASSERT_EQ(group_set_->payloadBytes(), host_block_size_);
        per_rank_transfer_engine_ = makeEngine({group_set_});
    }

    std::vector<size_t>                         layer_bytes_;
    size_t                                      host_block_size_;
    std::shared_ptr<HostBlockPool>              host_pool_;
    std::shared_ptr<PerRankBlockTransferEngine> per_rank_transfer_engine_;
    DeviceBlockPoolPtr                          device_pool_;
    BlockIdxType                                device_block_;
    std::vector<BlockIdxType>                   device_blocks_;
    GroupSetPtr                                 group_set_;
};

TEST_F(PerRankBlockTransferEngineTest, SubmitDeviceHostRoundTripPreservesLayout) {
    fillDeviceLayerSequential(device_pool_, 0, device_block_);
    fillDeviceLayer(device_pool_, 1, device_block_, {0x5A});
    fillDeviceLayerSequential(device_pool_, 2, device_block_);

    BlockIdxType host_block = poolMalloc(*host_pool_);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);

    auto d2h_desc = makeDescriptor(Tier::DEVICE, Tier::HOST, device_blocks_, host_block);
    ASSERT_TRUE(executeSucceeded(per_rank_transfer_engine_, d2h_desc));

    const uint8_t* host_data = static_cast<const uint8_t*>(host_pool_->blockBuffer(host_block).addr);
    for (size_t i = 0; i < layer_bytes_[0]; ++i)
        EXPECT_EQ(host_data[i], static_cast<uint8_t>(i & 0xFF));
    for (size_t i = 0; i < layer_bytes_[1]; ++i)
        EXPECT_EQ(host_data[layer_bytes_[0] + i], 0x5A);
    const size_t off2 = layer_bytes_[0] + layer_bytes_[1];
    for (size_t i = 0; i < layer_bytes_[2]; ++i)
        EXPECT_EQ(host_data[off2 + i], static_cast<uint8_t>(i & 0xFF));

    fillDeviceLayer(device_pool_, 0, device_block_, {0x00});
    fillDeviceLayer(device_pool_, 1, device_block_, {0x00});
    fillDeviceLayer(device_pool_, 2, device_block_, {0x00});
    auto h2d_desc = makeDescriptor(Tier::HOST, Tier::DEVICE, device_blocks_, host_block);
    ASSERT_TRUE(executeSucceeded(per_rank_transfer_engine_, h2d_desc));

    auto d0 = readDeviceLayer(device_pool_, 0, device_block_);
    auto d1 = readDeviceLayer(device_pool_, 1, device_block_);
    auto d2 = readDeviceLayer(device_pool_, 2, device_block_);
    for (size_t i = 0; i < layer_bytes_[0]; ++i)
        EXPECT_EQ(d0[i], static_cast<uint8_t>(i & 0xFF));
    for (size_t i = 0; i < layer_bytes_[1]; ++i)
        EXPECT_EQ(d1[i], 0x5A);
    for (size_t i = 0; i < layer_bytes_[2]; ++i)
        EXPECT_EQ(d2[i], static_cast<uint8_t>(i & 0xFF));

    releasePoolBlock(*host_pool_, host_block);
}

TEST_F(PerRankBlockTransferEngineTest, SubmitHostToDeviceIndependentDescriptors) {
    BlockIdxType second_device_block = poolMalloc(*device_pool_);
    ASSERT_NE(second_device_block, NULL_BLOCK_IDX);
    std::vector<BlockIdxType> second_device_blocks = {second_device_block};

    BlockIdxType host_block_1 = poolMalloc(*host_pool_);
    auto*        host_data_1  = static_cast<uint8_t*>(host_pool_->blockBuffer(host_block_1).addr);
    std::memset(host_data_1, 0x12, layer_bytes_[0]);
    std::memset(host_data_1 + layer_bytes_[0], 0x34, layer_bytes_[1]);
    std::memset(host_data_1 + layer_bytes_[0] + layer_bytes_[1], 0x9A, layer_bytes_[2]);

    BlockIdxType host_block_2 = poolMalloc(*host_pool_);
    auto*        host_data_2  = static_cast<uint8_t*>(host_pool_->blockBuffer(host_block_2).addr);
    std::memset(host_data_2, 0x56, layer_bytes_[0]);
    std::memset(host_data_2 + layer_bytes_[0], 0x78, layer_bytes_[1]);
    std::memset(host_data_2 + layer_bytes_[0] + layer_bytes_[1], 0xBC, layer_bytes_[2]);

    block_transfer_engine_test::encodeTestRecord(host_data_1, host_block_size_);
    block_transfer_engine_test::encodeTestRecord(host_data_2, host_block_size_);
    auto context = per_rank_transfer_engine_->execute(
        makeTransferTask({makeDescriptor(Tier::HOST, Tier::DEVICE, device_blocks_, host_block_1),
                          makeDescriptor(Tier::HOST, Tier::DEVICE, second_device_blocks, host_block_2)}));
    context->waitDone();
    ASSERT_TRUE(context->success());

    const auto first_layer0  = readDeviceLayer(device_pool_, 0, device_block_);
    const auto first_layer1  = readDeviceLayer(device_pool_, 1, device_block_);
    const auto first_layer2  = readDeviceLayer(device_pool_, 2, device_block_);
    const auto second_layer0 = readDeviceLayer(device_pool_, 0, second_device_block);
    const auto second_layer1 = readDeviceLayer(device_pool_, 1, second_device_block);
    const auto second_layer2 = readDeviceLayer(device_pool_, 2, second_device_block);
    for (size_t i = 0; i < 100; ++i) {
        EXPECT_EQ(first_layer0[i], 0x12);
        EXPECT_EQ(first_layer1[i], 0x34);
        EXPECT_EQ(first_layer2[i], 0x9A);
        EXPECT_EQ(second_layer0[i], 0x56);
        EXPECT_EQ(second_layer1[i], 0x78);
        EXPECT_EQ(second_layer2[i], 0xBC);
    }

    releasePoolBlock(*host_pool_, host_block_1);
    releasePoolBlock(*host_pool_, host_block_2);
    device_pool_->incRef(second_device_block);
    device_pool_->decRef(second_device_block);
}

TEST_F(PerRankBlockTransferEngineTest, BatchAllowsSharedReadEndpoint) {
    fillDeviceLayer(device_pool_, 0, device_block_, {0x4A});
    const BlockIdxType first_host_block  = poolMalloc(*host_pool_);
    const BlockIdxType second_host_block = poolMalloc(*host_pool_);
    ASSERT_NE(first_host_block, NULL_BLOCK_IDX);
    ASSERT_NE(second_host_block, NULL_BLOCK_IDX);

    auto context = per_rank_transfer_engine_->execute(
        makeTransferTask({makeDescriptor(Tier::DEVICE, Tier::HOST, device_blocks_, first_host_block),
                          makeDescriptor(Tier::DEVICE, Tier::HOST, device_blocks_, second_host_block)}));
    context->waitDone();
    EXPECT_TRUE(context->success());

    releasePoolBlock(*host_pool_, first_host_block);
    releasePoolBlock(*host_pool_, second_host_block);
}

class PerRankBlockTransferEngineMultiMemberTest: public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_TRUE(torch::cuda::is_available()) << "CUDA not available, cannot run GPU tests";
        host_pool_ = makeHostPool(240, 4, true);
        disk_pool_ = makeDiskPool(240, 4, temp_dir_.path, nullptr, "multi_member", true, true);
        pools_     = {
            makeDevicePool({{64, 16}}, 4, "per_rank_transfer_engine_multi_member_0"),
            makeDevicePool({{64, 16}}, 4, "per_rank_transfer_engine_multi_member_1"),
            makeDevicePool({{64, 16}}, 4, "per_rank_transfer_engine_multi_member_2"),
        };
        for (const auto& pool : pools_) {
            blocks_.push_back(poolMalloc(*pool));
            ASSERT_NE(blocks_.back(), NULL_BLOCK_IDX);
        }

        GroupSetPtr group =
            makeDeviceHostGroup(pools_,
                                host_pool_,
                                {makeGroupBase({0}, 64, 16), makeGroupBase({0}, 64, 16), makeGroupBase({0}, 64, 16)},
                                disk_pool_);
        engine_ = makeEngine({group});

        host_block_ = poolMalloc(*host_pool_);
        disk_block_ = poolMalloc(*disk_pool_);
        ASSERT_NE(host_block_, NULL_BLOCK_IDX);
        ASSERT_NE(disk_block_, NULL_BLOCK_IDX);
    }

    TempDirGuard                                temp_dir_{"per_rank_transfer_engine_multi_member"};
    std::shared_ptr<HostBlockPool>              host_pool_;
    DiskBlockPoolPtr                            disk_pool_;
    std::vector<DeviceBlockPoolPtr>             pools_;
    std::vector<BlockIdxType>                   blocks_;
    std::shared_ptr<PerRankBlockTransferEngine> engine_;
    BlockIdxType                                host_block_{NULL_BLOCK_IDX};
    BlockIdxType                                disk_block_{NULL_BLOCK_IDX};
};

TEST_F(PerRankBlockTransferEngineMultiMemberTest, CompleteMultiMemberThreeTierRoundTripPreservesOffsets) {
    fillDeviceLayer(pools_[0], 0, blocks_[0], {0xA1, 0xA2});
    fillDeviceLayer(pools_[1], 0, blocks_[1], {0xB1, 0xB2});
    fillDeviceLayer(pools_[2], 0, blocks_[2], {0xC1, 0xC2});
    auto* host_data = static_cast<uint8_t*>(host_pool_->blockBuffer(host_block_).addr);
    std::memset(host_data, 0xFF, 240);

    const auto expect_host_payload = [host_data]() {
        for (size_t i = 0; i < 64; ++i)
            EXPECT_EQ(host_data[i], 0xA1);
        for (size_t i = 64; i < 80; ++i)
            EXPECT_EQ(host_data[i], 0xA2);
        for (size_t i = 80; i < 144; ++i)
            EXPECT_EQ(host_data[i], 0xB1);
        for (size_t i = 144; i < 160; ++i)
            EXPECT_EQ(host_data[i], 0xB2);
        for (size_t i = 160; i < 224; ++i)
            EXPECT_EQ(host_data[i], 0xC1);
        for (size_t i = 224; i < 240; ++i)
            EXPECT_EQ(host_data[i], 0xC2);
    };

    expectStatus(engine_,
                 makeDescriptor(Tier::DEVICE, Tier::HOST, {blocks_[0], blocks_[1], blocks_[2]}, host_block_),
                 TransferStatus::OK);
    expect_host_payload();

    expectStatus(engine_, makeDescriptor(Tier::HOST, Tier::DISK, {}, host_block_, disk_block_), TransferStatus::OK);
    std::memset(host_data, 0, 240);
    ASSERT_EQ(disk_pool_->read(disk_block_, host_data, disk_pool_->strideBytes()), BlockIOStatus::OK);
    expect_host_payload();

    fillDeviceLayer(pools_[0], 0, blocks_[0], {0x00, 0x00});
    fillDeviceLayer(pools_[1], 0, blocks_[1], {0x00, 0x00});
    fillDeviceLayer(pools_[2], 0, blocks_[2], {0x00, 0x00});
    expectStatus(
        engine_,
        makeDescriptor(Tier::DISK, Tier::DEVICE, {blocks_[0], blocks_[1], blocks_[2]}, NULL_BLOCK_IDX, disk_block_),
        TransferStatus::OK);

    const auto comp0 = readDeviceLayer(pools_[0], 0, blocks_[0]);
    const auto comp1 = readDeviceLayer(pools_[1], 0, blocks_[1]);
    const auto comp2 = readDeviceLayer(pools_[2], 0, blocks_[2]);
    for (size_t i = 0; i < 64; ++i) {
        EXPECT_EQ(comp0[i], 0xA1);
        EXPECT_EQ(comp1[i], 0xB1);
        EXPECT_EQ(comp2[i], 0xC1);
    }
    for (size_t i = 64; i < 80; ++i) {
        EXPECT_EQ(comp0[i], 0xA2);
        EXPECT_EQ(comp1[i], 0xB2);
        EXPECT_EQ(comp2[i], 0xC2);
    }
}

TEST(PerRankBlockTransferEngineIntegrationTest, DeviceHostDiskDeviceRoundTrip) {
    ASSERT_TRUE(torch::cuda::is_available()) << "CUDA not available, cannot run GPU tests";
    TempDirGuard     temp_dir("per_rank_transfer_engine_three_tier");
    constexpr size_t payload_bytes = 80;
    auto             host_pool     = makeHostPool(payload_bytes, 2, true);
    auto             disk_pool     = makeDiskPool(payload_bytes, 2, temp_dir.path, nullptr, "three_tier", true, true);
    auto             device_pool   = makeDevicePool({{64, 16}}, 2, "per_rank_transfer_engine_three_tier_device");
    auto             device_block  = poolMalloc(*device_pool);
    auto             host_block    = poolMalloc(*host_pool);
    auto             disk_block    = poolMalloc(*disk_pool);
    ASSERT_NE(device_block, NULL_BLOCK_IDX);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);
    ASSERT_NE(disk_block, NULL_BLOCK_IDX);

    auto group  = makeDeviceHostGroup({device_pool}, host_pool, {makeGroupBase({0}, 64, 16)}, disk_pool);
    auto engine = makeEngine({group});
    fillDeviceLayer(device_pool, 0, device_block, {0x6A, 0xD3});
    const auto expected  = readDeviceLayer(device_pool, 0, device_block);
    auto       profiling = AsyncProfileSession::start();

    expectStatus(engine,
                 makeDescriptor(Tier::DEVICE, Tier::HOST, {device_block}, host_block, NULL_BLOCK_IDX, 0),
                 TransferStatus::OK);
    const auto* host_data = static_cast<const uint8_t*>(host_pool->blockBuffer(host_block).addr);
    EXPECT_TRUE(std::equal(expected.begin(), expected.end(), host_data));

    expectStatus(engine, makeDescriptor(Tier::HOST, Tier::DISK, {}, host_block, disk_block, 0), TransferStatus::OK);
    std::memset(host_pool->blockBuffer(host_block).addr, 0, payload_bytes);
    fillDeviceLayer(device_pool, 0, device_block, {0x00, 0x00});

    ASSERT_EQ(disk_pool->read(disk_block, host_pool->blockBuffer(host_block).addr, disk_pool->strideBytes()),
              BlockIOStatus::OK);
    EXPECT_TRUE(std::equal(expected.begin(), expected.end(), host_data));
    expectStatus(engine,
                 makeDescriptor(Tier::DISK, Tier::DEVICE, {device_block}, NULL_BLOCK_IDX, disk_block, 0),
                 TransferStatus::OK);
    EXPECT_EQ(readDeviceLayer(device_pool, 0, device_block), expected);
    const auto events = profiling->finish();
    for (const auto* path :
         {"device_to_host.d2h", "host_to_disk.disk_write", "disk_to_device.disk_read", "disk_to_device.h2d"}) {
        EXPECT_EQ(std::count_if(events.begin(),
                                events.end(),
                                [path](const auto& event) {
                                    return event.name.find(std::string("cache.transfer.") + path + "(") == 0;
                                }),
                  1)
            << path;
    }
}

TEST(PerRankBlockTransferEngineIntegrationTest, HostDiskDirectRoundTripThroughDevice) {
    ASSERT_TRUE(torch::cuda::is_available()) << "CUDA not available, cannot run GPU tests";
    TempDirGuard     temp_dir("per_rank_direct_round_trip");
    constexpr size_t payload_bytes = 80;
    auto             owned_io      = std::make_unique<DirectAlignmentDiskBlockIO>();
    auto*            direct_io     = owned_io.get();
    auto             disk_pool =
        makeDiskPool(payload_bytes, 2, temp_dir.path, std::move(owned_io), "device_disk_direct", false, true);
    auto host_pool    = makeHostPool(payload_bytes, 2, true);
    auto device_pool  = makeDevicePool({{64, 16}}, 2, "per_rank_direct_round_trip_device");
    auto host_block   = poolMalloc(*host_pool);
    auto device_block = poolMalloc(*device_pool);
    auto disk_block   = poolMalloc(*disk_pool);
    ASSERT_NE(device_block, NULL_BLOCK_IDX);
    ASSERT_NE(disk_block, NULL_BLOCK_IDX);

    auto group  = makeDeviceHostGroup({device_pool}, host_pool, {makeGroupBase({0}, 64, 16)}, disk_pool);
    auto engine = makeEngine({group});

    fillDeviceLayer(device_pool, 0, device_block, {0x6A, 0xD3});
    const auto expected  = readDeviceLayer(device_pool, 0, device_block);
    auto       profiling = AsyncProfileSession::start();

    expectStatus(engine,
                 makeDescriptor(Tier::DEVICE, Tier::HOST, {device_block}, host_block, NULL_BLOCK_IDX, 0),
                 TransferStatus::OK);
    expectStatus(engine, makeDescriptor(Tier::HOST, Tier::DISK, {}, host_block, disk_block, 0), TransferStatus::OK);
    EXPECT_FALSE(direct_io->bufferedIo());
    EXPECT_EQ(direct_io->lastWriteBytes(), disk_pool->strideBytes());
    fillDeviceLayer(device_pool, 0, device_block, {0x00, 0x00});
    expectStatus(engine,
                 makeDescriptor(Tier::DISK, Tier::DEVICE, {device_block}, NULL_BLOCK_IDX, disk_block, 0),
                 TransferStatus::OK);
    EXPECT_EQ(readDeviceLayer(device_pool, 0, device_block), expected);
    EXPECT_EQ(direct_io->lastReadBytes(), disk_pool->strideBytes());
    const auto events = profiling->finish();
    EXPECT_FALSE(events.empty());
}

TEST(TransferStageTimerTest, DisabledObserverDoesNotAttemptSharedAllocation) {
    EXPECT_FALSE(AsyncProfileSession::enabled());
    TransferStageTimerTestPeer::setAllocationProbe(
        [](const char*) { ADD_FAILURE() << "disabled observer attempted shared allocation"; });
    auto timer = TransferStageTimer::create(nullptr, "probe", 1, 0, 1);
    EXPECT_FALSE(timer);
    timer.finish(true);
    timer.finish(false);
    TransferStageTimerTestPeer::setAllocationProbe(nullptr);
}

class StagedMemoryCopyTest: public ::testing::TestWithParam<bool> {
protected:
    void SetUp() override {
        ASSERT_TRUE(torch::cuda::is_available()) << "CUDA not available, cannot run GPU tests";
    }

    void TearDown() override {
        releaseStagedMemoryCopyScratch(scratch_);
    }

    void expectRoundTrip(bool segmented, size_t tile_count, bool reuse_scratch) {
        constexpr size_t            allocation_bytes = 24576;
        const std::array<size_t, 3> tile_bytes{31, 257, 4097};
        const std::array<size_t, 3> device_offsets{5, 4103, 16387};
        const std::array<size_t, 3> host_offsets{16448, 64, 8256};
        auto                        host =
            torch::empty({allocation_bytes},
                         torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU).pinned_memory(GetParam()));
        auto device =
            torch::empty({allocation_bytes}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
        ASSERT_EQ(host.is_pinned(), GetParam());
        auto*                host_data   = host.data_ptr<uint8_t>();
        auto*                device_data = device.data_ptr<uint8_t>();
        std::vector<uint8_t> expected_device(allocation_bytes, 0xA5);
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy(device_data, expected_device.data(), allocation_bytes, cudaMemcpyHostToDevice));
        ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
        std::fill_n(host_data, allocation_bytes, 0xCD);
        std::vector<uint8_t> expected_host(allocation_bytes, 0xE3);

        StagedMemoryCopyParams params;
        params.host_base = host_data + 64;
        ASSERT_EQ(cudaSuccess, cudaGetDevice(&params.device_index));
        for (size_t tile = 0; tile < tile_count; ++tile) {
            const size_t staging_offset = segmented ? (params.host_bytes + 15) / 16 * 16 : params.host_bytes;
            const size_t host_offset    = segmented ? host_offsets[tile] : 64 + staging_offset;
            params.tiles.push_back({device_data + device_offsets[tile], staging_offset, tile_bytes[tile]});
            if (segmented) {
                params.host_segments.push_back({host_data + host_offset, staging_offset, tile_bytes[tile]});
            }
            for (size_t byte = 0; byte < tile_bytes[tile]; ++byte) {
                const uint8_t value                          = (tile * 61 + byte * 17 + tile_count) % 251;
                host_data[host_offset + byte]                = value;
                expected_device[device_offsets[tile] + byte] = value;
                expected_host[host_offset + byte]            = value;
            }
            params.host_bytes = staging_offset + tile_bytes[tile];
        }

        params.direction = StagedMemoryCopyDirection::H2D;
        ASSERT_TRUE(execStagedMemoryCopy(params, reuse_scratch ? &scratch_ : nullptr));
        std::vector<uint8_t> actual_device(allocation_bytes);
        ASSERT_EQ(cudaSuccess, cudaMemcpy(actual_device.data(), device_data, allocation_bytes, cudaMemcpyDeviceToHost));
        EXPECT_EQ(actual_device, expected_device);

        std::fill_n(host_data, allocation_bytes, 0xE3);
        params.direction = StagedMemoryCopyDirection::D2H;
        ASSERT_TRUE(execStagedMemoryCopy(params, reuse_scratch ? &scratch_ : nullptr));
        EXPECT_EQ(std::vector<uint8_t>(host_data, host_data + allocation_bytes), expected_host);
    }

    StagedMemoryCopyScratch scratch_;
};

TEST_P(StagedMemoryCopyTest, DisjointHostSegmentsPreserveGuardsAcrossScratchGrowthAndReuse) {
    for (size_t tile_count : {1u, 3u, 2u}) {
        SCOPED_TRACE(tile_count);
        ASSERT_NO_FATAL_FAILURE(expectRoundTrip(true, tile_count, true));
    }
}

TEST_P(StagedMemoryCopyTest, ContiguousHostBaseWorksWithoutReusableScratch) {
    ASSERT_NO_FATAL_FAILURE(expectRoundTrip(false, 3, false));
}

INSTANTIATE_TEST_SUITE_P(HostMemory, StagedMemoryCopyTest, ::testing::Bool());

}  // namespace
}  // namespace rtp_llm
