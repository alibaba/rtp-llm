#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "kmonitor/client/MetricsReporter.h"
#include "kmonitor/client/core/MetricsData.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCacheMetricsReporter.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/FullGroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/PerRankBlockTransferEngine.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/HostDiskTransferExecutor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/test/PerRankBlockTransferEngineTestUtils.h"

namespace rtp_llm {
namespace {

TEST(PerRankBlockTransferEngineConfigTest, UsesFourSharedWorkersByDefault) {
    PerRankBlockTransferEngine engine(std::vector<GroupSetPtr>{});
    EXPECT_EQ(engine.transferWorkerCount(), 4u);
}

TEST(PerRankBlockTransferEngineConfigTest, PreservesExplicitWorkerOverride) {
    PerRankBlockTransferEngine engine({}, false, {}, 4, 64, 7);
    EXPECT_EQ(engine.transferWorkerCount(), 7u);
}

using block_transfer_engine_test::TempDirGuard;
using block_transfer_engine_test::DirectAlignmentDiskBlockIO;
using block_transfer_engine_test::StatusDiskBlockIO;
using block_transfer_engine_test::expectStatus;
using block_transfer_engine_test::makeDescriptor;
using block_transfer_engine_test::makeDiskPool;
using block_transfer_engine_test::makeHostPool;
using block_transfer_engine_test::makeTestDevicePool;
using block_transfer_engine_test::makeTestGroupBase;
using block_transfer_engine_test::makeTestGroupSet;
using block_transfer_engine_test::makeTestTopology;
using block_transfer_engine_test::makeTransferTask;
using block_transfer_engine_test::poolMalloc;
using block_transfer_engine_test::releasePoolBlock;
using block_transfer_engine_test::executeSucceeded;

GroupSetPtr makeHostDiskGroup(size_t                                  group_set_id,
                              std::shared_ptr<HostBlockPool>          host_pool,
                              std::shared_ptr<BlockTreeDiskBlockPool> disk_pool,
                              size_t                                  payload_bytes) {
    auto policy                = defaultCacheGroupPolicy(CacheGroupType::FULL);
    policy.enable_prefix_reuse = true;
    auto topology              = makeTestTopology({makeTestGroupBase(policy, {0}, payload_bytes)});
    auto device_pool = makeTestDevicePool({{payload_bytes, 0}}, 2, "host_disk_group_" + std::to_string(group_set_id));
    auto group       = makeTestGroupSet(
        group_set_id, std::move(topology), {0}, {std::move(device_pool)}, std::move(host_pool), std::move(disk_pool));
    return group;
}

std::shared_ptr<PerRankBlockTransferEngine> makeEngine(std::vector<GroupSetPtr> groups) {
    return std::make_shared<PerRankBlockTransferEngine>(std::move(groups));
}

class RecordingBatchDiskBlockIO: public StatusDiskBlockIO {
public:
    RecordingBatchDiskBlockIO(): StatusDiskBlockIO(DiskBlockIOStatus::OK) {}

    DiskBlockIOStatus write(const std::vector<DiskWrite>& writes) override {
        batch_sizes.push_back(writes.size());
        return StatusDiskBlockIO::write(writes);
    }

    std::vector<size_t> batch_sizes;
};

class BlockingBatchDiskBlockIO: public StatusDiskBlockIO {
public:
    enum class Operation {
        READ,
        WRITE
    };

    explicit BlockingBatchDiskBlockIO(Operation operation):
        StatusDiskBlockIO(DiskBlockIOStatus::OK), operation_(operation) {}

    DiskBlockIOStatus read(const std::vector<DiskRead>& reads) override {
        block(Operation::READ);
        return StatusDiskBlockIO::read(reads);
    }
    DiskBlockIOStatus write(const std::vector<DiskWrite>& writes) override {
        block(Operation::WRITE);
        return StatusDiskBlockIO::write(writes);
    }
    bool waitForBlockedCalls(size_t count, std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        return cv_.wait_for(lock, timeout, [&] { return blocked_calls_ >= count; });
    }
    void release() {
        std::lock_guard<std::mutex> lock(mutex_);
        released_ = true;
        cv_.notify_all();
    }

private:
    void block(Operation operation) {
        if (operation != operation_) {
            return;
        }
        std::unique_lock<std::mutex> lock(mutex_);
        ++blocked_calls_;
        cv_.notify_all();
        cv_.wait(lock, [&] { return released_; });
    }

    Operation               operation_;
    std::mutex              mutex_;
    std::condition_variable cv_;
    size_t                  blocked_calls_{0};
    bool                    released_{false};
};

class PerRankBlockTransferEngineHostDiskTest: public ::testing::Test {
protected:
    void SetUp() override {
        host_block_size_ = 384;

        host_pool_ = makeHostPool(host_block_size_, 4);
        disk_pool_ = makeDiskPool(host_block_size_, 7, temp_dir_.path);

        group_set_ = makeHostDiskGroup(0, host_pool_, disk_pool_, host_block_size_);
        ASSERT_EQ(group_set_->payloadBytes(), host_block_size_);
        per_rank_transfer_engine_ = makeEngine({group_set_});
    }

    TempDirGuard                                temp_dir_{"block_transfer_engine_test"};
    size_t                                      host_block_size_;
    std::shared_ptr<HostBlockPool>              host_pool_;
    std::shared_ptr<BlockTreeDiskBlockPool>     disk_pool_;
    std::shared_ptr<PerRankBlockTransferEngine> per_rank_transfer_engine_;
    GroupSetPtr                                 group_set_;
};

TEST_F(PerRankBlockTransferEngineHostDiskTest, DisabledDiskCacheSkipsDeviceDiskExecutor) {
    auto engine = std::make_shared<PerRankBlockTransferEngine>(std::vector<GroupSetPtr>{group_set_}, false);

    EXPECT_EQ(engine->device_disk_executor_, nullptr);
}

TEST_F(PerRankBlockTransferEngineHostDiskTest, SubmitHostToDiskRoundTrip) {
    BlockIdxType host_block = poolMalloc(*host_pool_);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);
    uint8_t* host_data = static_cast<uint8_t*>(host_pool_->blockBuffer(host_block).addr);
    for (size_t i = 0; i < host_block_size_; ++i)
        host_data[i] = static_cast<uint8_t>(i & 0xFF);

    auto disk_block_opt = disk_pool_->malloc();
    ASSERT_TRUE(disk_block_opt.has_value());
    int32_t disk_block = disk_block_opt.value();

    auto host_to_disk = makeDescriptor(Tier::HOST, Tier::DISK, {}, host_block, disk_block);
    ASSERT_TRUE(executeSucceeded(per_rank_transfer_engine_, host_to_disk));

    std::memset(host_data, 0, host_block_size_);

    auto disk_to_host = makeDescriptor(Tier::DISK, Tier::HOST, {}, host_block, disk_block);
    ASSERT_TRUE(executeSucceeded(per_rank_transfer_engine_, disk_to_host));

    for (size_t i = 0; i < host_block_size_; ++i)
        EXPECT_EQ(host_data[i], static_cast<uint8_t>(i & 0xFF)) << "byte " << i;

    releasePoolBlock(*host_pool_, host_block);
    releasePoolBlock(*disk_pool_, disk_block);
}

TEST_F(PerRankBlockTransferEngineHostDiskTest, ReportsSuccessfulQueueWaitThroughMetricsReporter) {
    auto metrics_reporter       = std::make_shared<kmonitor::MetricsReporter>("", "", kmonitor::MetricsTags{});
    auto cache_metrics_reporter = std::make_shared<BlockTreeCacheMetricsReporter>(metrics_reporter);
    auto engine                 = std::make_shared<PerRankBlockTransferEngine>(std::vector<GroupSetPtr>{group_set_},
                                                               false,
                                                               DeviceHostCopyOptions{},
                                                               4,
                                                               8,
                                                               4,
                                                               10000,
                                                               cache_metrics_reporter);

    const BlockIdxType host_block = poolMalloc(*host_pool_);
    const BlockIdxType disk_block = poolMalloc(*disk_pool_);
    const auto         descriptor = makeDescriptor(Tier::HOST, Tier::DISK, {}, host_block, disk_block);
    ASSERT_TRUE(executeSucceeded(engine, descriptor));

    auto* transfer_metrics = metrics_reporter->getMetricsGroup<RtpLLMCacheTransferMetrics>();
    ASSERT_NE(transfer_metrics, nullptr);
    kmonitor::MetricsTags tags("pool_type", "transfer");
    tags.AddTag("source_tier", tierName(Tier::HOST));
    tags.AddTag("target_tier", tierName(Tier::DISK));
    auto* waiting_tasks = transfer_metrics->task_queue_waiting_tasks_metric;
    ASSERT_NE(waiting_tasks, nullptr);
    EXPECT_EQ(waiting_tasks->metric_data_->Size(), 1u);
    auto* metric = waiting_tasks->DeclareMetric(&tags);
    ASSERT_NE(metric, nullptr);
    kmonitor::MetricsRecord record(nullptr, nullptr, 0);
    metric->Snapshot(&record, 1000);
    EXPECT_TRUE(waiting_tasks->UndeclareMetric(metric));
    ASSERT_EQ(record.Values().size(), 1u);
    EXPECT_DOUBLE_EQ(std::stod(record.Values().front()->Value()), 1);
    auto* latency = transfer_metrics->transfer_task_queue_wait_latency_us_metric;
    ASSERT_NE(latency, nullptr);
    EXPECT_EQ(latency->metric_data_->Size(), 1u);

    releasePoolBlock(*host_pool_, host_block);
    releasePoolBlock(*disk_pool_, disk_block);
}

TEST_F(PerRankBlockTransferEngineHostDiskTest, HostDiskExecutorReportsTaskPoolSubmissionRejection) {
    BlockTreeTaskPool task_pool(1, 8, "HostDiskExecutorTest");
    ASSERT_TRUE(task_pool.start());
    task_pool.stopAdmission();
    HostDiskTransferExecutor executor(task_pool, 16);

    const BlockIdxType host_block = poolMalloc(*host_pool_);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);
    const BlockIdxType disk_block = poolMalloc(*disk_pool_);
    ASSERT_NE(disk_block, NULL_BLOCK_IDX);
    const HostBlockBuffer host_buffer = host_pool_->blockBuffer(host_block);
    const HostBufferView  host{host_buffer.addr, host_buffer.payload_bytes, host_buffer.stride_bytes};
    const auto            descriptor = makeDescriptor(Tier::HOST, Tier::DISK, {}, host_block, disk_block);

    auto context = executor.execute(makeTransferTask({descriptor}), {host}, {group_set_.get()});
    context->waitDone();

    EXPECT_FALSE(context->success());
    EXPECT_EQ(context->errorInfo().code(), ErrorCode::EXECUTION_EXCEPTION);
    EXPECT_NE(context->errorInfo().ToString().find("RESOURCE_EXHAUSTED"), std::string::npos);
    releasePoolBlock(*host_pool_, host_block);
    releasePoolBlock(*disk_pool_, disk_block);
}

TEST_F(PerRankBlockTransferEngineHostDiskTest, MaxBatchSizeSplitsOneLogicalBatch) {
    auto  owned_io  = std::make_unique<RecordingBatchDiskBlockIO>();
    auto* io        = owned_io.get();
    auto  host_pool = makeHostPool(host_block_size_, 8);
    auto  disk_pool = makeDiskPool(host_block_size_, 8, temp_dir_.path, std::move(owned_io), "split_batch");
    auto  group     = makeHostDiskGroup(0, host_pool, disk_pool, host_block_size_);
    auto  engine    = std::make_shared<PerRankBlockTransferEngine>(
        std::vector<GroupSetPtr>{group}, false, DeviceHostCopyOptions{}, 4, 2, 4);
    std::vector<TransferDescriptor> descriptors;
    for (size_t index = 0; index < 5; ++index) {
        descriptors.push_back(
            makeDescriptor(Tier::HOST, Tier::DISK, {}, poolMalloc(*host_pool), poolMalloc(*disk_pool)));
    }

    auto context = engine->execute(makeTransferTask(descriptors));
    context->waitDone();

    ASSERT_TRUE(context->success());
    EXPECT_EQ(io->batch_sizes, (std::vector<size_t>{2, 2, 1}));
}

TEST_F(PerRankBlockTransferEngineHostDiskTest, DefaultBatchSizeSplitsSeventeenDescriptorsIntoEightEightOne) {
    auto  owned_io  = std::make_unique<RecordingBatchDiskBlockIO>();
    auto* io        = owned_io.get();
    auto  host_pool = makeHostPool(host_block_size_, 17);
    auto  disk_pool = makeDiskPool(host_block_size_, 17, temp_dir_.path, std::move(owned_io), "default_batch");
    auto  group     = makeHostDiskGroup(0, host_pool, disk_pool, host_block_size_);
    auto  engine    = makeEngine({group});
    std::vector<TransferDescriptor> descriptors;
    for (size_t index = 0; index < 17; ++index) {
        descriptors.push_back(
            makeDescriptor(Tier::HOST, Tier::DISK, {}, poolMalloc(*host_pool), poolMalloc(*disk_pool)));
    }

    auto context = engine->execute(makeTransferTask(descriptors));
    context->waitDone();

    ASSERT_TRUE(context->success());
    EXPECT_EQ(io->batch_sizes, (std::vector<size_t>{8, 8, 1}));
}

TEST_F(PerRankBlockTransferEngineHostDiskTest, SameDirectionHostToDiskTasksMayUseSharedWorkers) {
    auto       owned_io  = std::make_unique<BlockingBatchDiskBlockIO>(BlockingBatchDiskBlockIO::Operation::WRITE);
    auto*      io        = owned_io.get();
    auto       host_pool = makeHostPool(host_block_size_, 3);
    auto       disk_pool = makeDiskPool(host_block_size_, 2, temp_dir_.path, std::move(owned_io), "serialized_write");
    auto       group     = makeHostDiskGroup(0, host_pool, disk_pool, host_block_size_);
    auto       engine    = makeEngine({group});
    const auto first     = makeDescriptor(Tier::HOST, Tier::DISK, {}, poolMalloc(*host_pool), poolMalloc(*disk_pool));
    const auto second    = makeDescriptor(Tier::HOST, Tier::DISK, {}, poolMalloc(*host_pool), poolMalloc(*disk_pool));

    auto first_context = engine->execute(makeTransferTask({first}));
    ASSERT_TRUE(io->waitForBlockedCalls(1, std::chrono::seconds(5)));
    auto       second_context                = engine->execute(makeTransferTask({second}));
    const bool second_started_before_release = io->waitForBlockedCalls(2, std::chrono::milliseconds(200));

    io->release();
    first_context->waitDone();
    second_context->waitDone();
    EXPECT_TRUE(second_started_before_release);
    EXPECT_TRUE(first_context->success());
    EXPECT_TRUE(second_context->success());
}

TEST_F(PerRankBlockTransferEngineHostDiskTest, SameDirectionDiskToHostTasksMayUseSharedWorkers) {
    auto       owned_io   = std::make_unique<BlockingBatchDiskBlockIO>(BlockingBatchDiskBlockIO::Operation::READ);
    auto*      io         = owned_io.get();
    auto       host_pool  = makeHostPool(host_block_size_, 3);
    auto       disk_pool  = makeDiskPool(host_block_size_, 2, temp_dir_.path, std::move(owned_io), "shared_read");
    auto       group      = makeHostDiskGroup(0, host_pool, disk_pool, host_block_size_);
    auto       engine     = makeEngine({group});
    const auto disk_block = poolMalloc(*disk_pool);
    const auto first      = makeDescriptor(Tier::DISK, Tier::HOST, {}, poolMalloc(*host_pool), disk_block);
    const auto second     = makeDescriptor(Tier::DISK, Tier::HOST, {}, poolMalloc(*host_pool), disk_block);

    auto first_context = engine->execute(makeTransferTask({first}));
    ASSERT_TRUE(io->waitForBlockedCalls(1, std::chrono::seconds(5)));
    auto       second_context                = engine->execute(makeTransferTask({second}));
    const bool second_started_before_release = io->waitForBlockedCalls(2, std::chrono::milliseconds(200));

    io->release();
    first_context->waitDone();
    second_context->waitDone();
    EXPECT_TRUE(second_started_before_release);
    EXPECT_TRUE(first_context->success());
    EXPECT_TRUE(second_context->success());
}

TEST_F(PerRankBlockTransferEngineHostDiskTest, HostDiskDirectIoWritesAlignedStrideAndZeroPads) {
    auto  owned_io  = std::make_unique<DirectAlignmentDiskBlockIO>();
    auto* direct_io = owned_io.get();
    auto  direct_disk =
        makeDiskPool(host_block_size_, 4, temp_dir_.path, std::move(owned_io), "host_disk_direct", false);
    auto                     group = makeHostDiskGroup(0, host_pool_, direct_disk, host_block_size_);
    BlockTreeTaskPool        task_pool(1, 8, "HostDiskExecutorDirectIoTest");
    HostDiskTransferExecutor executor(task_pool, 16);
    const size_t             stride = direct_disk->strideBytes();
    ASSERT_GT(stride, host_block_size_);
    EXPECT_FALSE(direct_io->bufferedIo());

    const BlockIdxType host_block = poolMalloc(*host_pool_);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);
    uint8_t* host_data = static_cast<uint8_t*>(host_pool_->blockBuffer(host_block).addr);
    // Seed non-zero padding to verify that the executor clears it.
    for (size_t i = 0; i < stride; ++i) {
        host_data[i] = static_cast<uint8_t>((i * 7 + 1) & 0xFF);
    }

    const auto disk_block = poolMalloc(*direct_disk);
    ASSERT_NE(disk_block, NULL_BLOCK_IDX);
    ASSERT_EQ(executor.executeBatch({HostBufferView{host_data, host_block_size_, stride}},
                                    {TransferDescriptor::deviceToDisk(0, {0}, disk_block)},
                                    {group.get()}),
              TransferStatus::OK);
    EXPECT_EQ(direct_io->lastWriteBytes(), stride);

    const BlockIdxType dst_block = poolMalloc(*host_pool_);
    ASSERT_NE(dst_block, NULL_BLOCK_IDX);
    uint8_t* dst_data = static_cast<uint8_t*>(host_pool_->blockBuffer(dst_block).addr);
    std::memset(dst_data, 0xAB, stride);
    ASSERT_EQ(executor.executeBatch({HostBufferView{dst_data, host_block_size_, stride}},
                                    {TransferDescriptor::diskToDevice(0, disk_block, {0})},
                                    {group.get()}),
              TransferStatus::OK);
    EXPECT_EQ(direct_io->lastReadBytes(), stride);

    for (size_t i = 0; i < host_block_size_; ++i) {
        EXPECT_EQ(dst_data[i], static_cast<uint8_t>((i * 7 + 1) & 0xFF)) << "payload byte " << i;
    }
    for (size_t i = host_block_size_; i < stride; ++i) {
        EXPECT_EQ(dst_data[i], 0) << "padding byte " << i;
    }

    releasePoolBlock(*host_pool_, host_block);
    releasePoolBlock(*host_pool_, dst_block);
    releasePoolBlock(*direct_disk, disk_block);
}

TEST_F(PerRankBlockTransferEngineHostDiskTest, SubmitHostToDiskAcceptsValidUnallocatedDiskBlock) {
    const BlockIdxType host_block = poolMalloc(*host_pool_);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);

    constexpr BlockIdxType unallocated_disk_block = 1;
    expectStatus(per_rank_transfer_engine_,
                 makeDescriptor(Tier::HOST, Tier::DISK, {}, host_block, unallocated_disk_block),
                 TransferStatus::OK);

    releasePoolBlock(*host_pool_, host_block);
}

TEST_F(PerRankBlockTransferEngineHostDiskTest, SubmitHostToDiskRejectsOutOfRangeDiskBlock) {
    const BlockIdxType host_block = poolMalloc(*host_pool_);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);

    const BlockIdxType out_of_range = static_cast<BlockIdxType>(disk_pool_->totalBlocksNum() + 1);
    expectStatus(per_rank_transfer_engine_,
                 makeDescriptor(Tier::HOST, Tier::DISK, {}, host_block, out_of_range),
                 TransferStatus::INVALID_ARGS);

    releasePoolBlock(*host_pool_, host_block);
}

TEST_F(PerRankBlockTransferEngineHostDiskTest, HostDiskStatusMapping) {
    const std::array<std::pair<BlockIOStatus, TransferStatus>, 6> mappings = {
        std::pair{BlockIOStatus::OK, TransferStatus::OK},
        std::pair{BlockIOStatus::INVALID_BLOCK, TransferStatus::INVALID_ARGS},
        std::pair{BlockIOStatus::INVALID_SIZE, TransferStatus::INVALID_ARGS},
        std::pair{BlockIOStatus::ALIGNMENT_ERROR, TransferStatus::INVALID_ARGS},
        std::pair{BlockIOStatus::IO_ERROR, TransferStatus::DISK_IO_ERROR},
        std::pair{BlockIOStatus::PARTIAL_FAILURE, TransferStatus::DISK_IO_ERROR},
    };
    for (const auto& [input, expected] : mappings) {
        EXPECT_EQ(HostDiskTransferExecutor::blockIOStatusToTransferStatus(input), expected);
    }

    const std::array<std::pair<DiskBlockIOStatus, TransferStatus>, 5> io_mappings = {
        std::pair{DiskBlockIOStatus::OK, TransferStatus::OK},
        std::pair{DiskBlockIOStatus::INVALID_SIZE, TransferStatus::INVALID_ARGS},
        std::pair{DiskBlockIOStatus::ALIGNMENT_ERROR, TransferStatus::INVALID_ARGS},
        std::pair{DiskBlockIOStatus::IO_ERROR, TransferStatus::DISK_IO_ERROR},
        std::pair{DiskBlockIOStatus::PARTIAL_FAILURE, TransferStatus::DISK_IO_ERROR},
    };
    int pool_suffix = 0;
    for (const auto& [io_status, expected] : io_mappings) {
        SCOPED_TRACE(::testing::Message() << "io_mapping=" << pool_suffix);
        auto disk_pool  = makeDiskPool(host_block_size_,
                                      2,
                                      temp_dir_.path,
                                      std::make_unique<StatusDiskBlockIO>(io_status),
                                      "per_rank_transfer_engine_status_" + std::to_string(pool_suffix++));
        auto host_block = poolMalloc(*host_pool_);
        auto disk_block = poolMalloc(*disk_pool);
        auto group      = makeHostDiskGroup(0, host_pool_, disk_pool, host_block_size_);
        auto engine     = makeEngine({group});
        expectStatus(engine, makeDescriptor(Tier::HOST, Tier::DISK, {}, host_block, disk_block), expected);
        expectStatus(engine, makeDescriptor(Tier::DISK, Tier::HOST, {}, host_block, disk_block), expected);
        releasePoolBlock(*host_pool_, host_block);
    }
}

TEST(GroupSetPayloadTest, PayloadBytesUsesLogicalStridesAcrossLayers) {
    auto policy                = defaultCacheGroupPolicy(CacheGroupType::FULL);
    policy.enable_prefix_reuse = true;
    auto topology              = makeTestTopology({makeTestGroupBase(policy, {0, 1, 2}, 160, 40)});
    auto pool                  = makeTestDevicePool({{200, 40}, {220, 40}, {240, 40}}, 2, "group_set_payload_test");
    auto group                 = makeTestGroupSet(0, std::move(topology), {0}, {std::move(pool)});
    EXPECT_EQ(group->payloadBytes(), 600u);
}

}  // namespace
}  // namespace rtp_llm
