#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <stdexcept>

#include <cuda_runtime.h>
#include <torch/torch.h>

#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/DeviceBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/legacy/DeviceHostTransferExecutor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/test/PerRankBlockTransferEngineTestUtils.h"
#include "rtp_llm/cpp/config/StaticConfig.h"

namespace rtp_llm {
namespace {
using namespace block_transfer_engine_test;

class ScriptedCopyStrategy final: public DeviceHostCopyStrategy {
public:
    explicit ScriptedCopyStrategy(StrategyResult result): result_(result) {}
    StrategyResult tryExecute(const DeviceHostCopyPlan&,
                              const DeviceHostCopyOptions&,
                              const DeviceHostCopyExecutionContext&) override {
        return result_;
    }
private:
    StrategyResult result_;
};

struct CopyProofInjection {
    const char* phase;
    bool throw_after_submit{false};
    bool fail_generic{false};
    size_t drains{0};
    size_t generic_submits{0};
};

thread_local CopyProofInjection* proof_injection = nullptr;

int injectCopyProofFailure(const char* phase, int actual) {
    if (std::strstr(phase, "drain") || std::strstr(phase, "completion"))
        ++proof_injection->drains;
    const bool generic = std::strcmp(phase, "generic_submit") == 0;
    if (generic)
        ++proof_injection->generic_submits;
    if (actual == cudaSuccess
        && (std::strcmp(phase, proof_injection->phase) == 0 || (generic && proof_injection->fail_generic))) {
        if (proof_injection->throw_after_submit)
            throw std::runtime_error("injected copy failure after real submission");
        return cudaErrorInvalidValue;
    }
    return actual;
}

class ScopedCopyProofInjection {
public:
    explicit ScopedCopyProofInjection(CopyProofInjection& injection):
        previous_(proof_injection), previous_probe_(no_block_copy_detail::cuda_result_for_test) {
        proof_injection = &injection;
        no_block_copy_detail::cuda_result_for_test = injectCopyProofFailure;
    }
    ~ScopedCopyProofInjection() {
        no_block_copy_detail::cuda_result_for_test = previous_probe_;
        proof_injection = previous_;
    }
private:
    CopyProofInjection* previous_;
    int (*previous_probe_)(const char*, int);
};

class DeviceHostBufferSafetyTest: public ::testing::Test {
protected:
    void SetUp() override {
        previous_core_dump_ = StaticConfig::user_ft_core_dump_on_exception;
        StaticConfig::user_ft_core_dump_on_exception = false;
        ASSERT_TRUE(torch::cuda::is_available());
        device_ = makeTestDevicePool({{64, 0}}, 2, "buffer_safety_device");
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        host_ = makeHostPool(64, 2);
        setGroup(device_);
    }
    void TearDown() override {
        StaticConfig::user_ft_core_dump_on_exception = previous_core_dump_;
    }
    void setGroup(const DeviceBlockPoolPtr& pool) {
        auto group = makeTestGroupBase(defaultCacheGroupPolicy(CacheGroupType::FULL), {0}, 64);
        const auto topology = makeTestTopology({group});
        group_ = makeTestGroupSet(0, topology, topology->groupTags(), {pool}, host_);
    }
    TransferResult execute(DeviceHostTransferExecutor& executor) {
        return executor.executeBatch({{bytes_.data(), bytes_.size(), bytes_.size()}},
                                     {TransferDescriptor::hostToDevice(0, 1, {1})},
                                     {group_.get()});
    }
    BlockTreeTaskPool pool_{1, 8, "BufferSafetyTest"};
    DeviceBlockPoolPtr device_;
    std::shared_ptr<HostBlockPool> host_;
    GroupSetPtr group_;
    std::array<uint8_t, 64> bytes_{};
    bool previous_core_dump_{false};
};

TEST_F(DeviceHostBufferSafetyTest, InvalidBatchDoesNotStartCopy) {
    DeviceHostTransferExecutor executor(pool_, 8);
    auto result = executor.executeBatch({}, {}, {});
    EXPECT_FALSE(result.ok());
    EXPECT_TRUE(result.buffer_safe);
    result = executor.executeBatch({{nullptr, 64, 64}},
                                   {TransferDescriptor::hostToDevice(0, 1, {1})}, {group_.get()});
    EXPECT_FALSE(result.ok());
    EXPECT_TRUE(result.buffer_safe);
}

TEST_F(DeviceHostBufferSafetyTest, HostBackedDeviceCopyIsSafeAfterMemcpy) {
    auto config = std::make_shared<DeviceBlockPoolConfig>(device_->config());
    config->use_device_malloc_backing = false;
    config->use_pinned_cpu_backing = true;
    auto pinned = std::make_shared<DeviceBlockPool>(config);
    ASSERT_TRUE(pinned->init());
    setGroup(pinned);
    DeviceHostTransferExecutor executor(pool_, 8);
    bytes_.fill(0x5A);
    auto result = execute(executor);
    ASSERT_TRUE(result.ok());
    EXPECT_TRUE(result.buffer_safe);
    const auto buffers = pinned->convertIndexToBuffer(0, 1);
    ASSERT_EQ(buffers.size(), 1u);
    EXPECT_EQ(std::memcmp(buffers.front().addr, bytes_.data(), bytes_.size()), 0);
}

TEST_F(DeviceHostBufferSafetyTest, NormalGpuCopyPublishesProofAfterDataArrives) {
    DeviceHostCopyOptions options;
    options.cuda_batch_copy_enabled = false;
    options.staged_sm_copy_enabled = false;
    DeviceHostTransferExecutor executor(pool_, 8, options);
    bytes_.fill(0xA5);
    const auto result = execute(executor);
    ASSERT_TRUE(result.ok());
    EXPECT_TRUE(result.buffer_safe);
    std::array<uint8_t, 64> copied{};
    const auto buffers = device_->convertIndexToBuffer(0, 1);
    ASSERT_EQ(buffers.size(), 1u);
    ASSERT_EQ(cudaMemcpy(copied.data(), buffers.front().addr, copied.size(), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(copied, bytes_);
}

TEST_F(DeviceHostBufferSafetyTest, GenericFailurePublishesProofAfterSuccessfulDrain) {
    DeviceHostCopyOptions options;
    options.cuda_batch_copy_enabled = false;
    options.staged_sm_copy_enabled = false;
    for (const bool throw_after_submit : {false, true}) {
        DeviceHostTransferExecutor executor(pool_, 8, options);
        CopyProofInjection injection{"generic_submit", throw_after_submit};
        ScopedCopyProofInjection scope(injection);
        const auto result = execute(executor);
        EXPECT_FALSE(result.ok());
        EXPECT_EQ(result.copy_error, CacheCopyError::COPY_FAILED);
        EXPECT_TRUE(result.buffer_safe);
        EXPECT_EQ(injection.generic_submits, 1u);
        EXPECT_EQ(injection.drains, 1u);
    }
}

TEST_F(DeviceHostBufferSafetyTest, DrainedStagedFailureKeepsProofThroughGenericFallback) {
    DeviceHostCopyOptions options;
    options.cuda_batch_copy_enabled = false;
    options.staged_sm_min_tile_count = 1;
    options.staged_sm_min_bytes = 1;
    bytes_.fill(0xA7);
    for (const char* phase : {"staged_metadata_submit", "staged_payload_submit"}) {
        for (const bool fail_generic : {false, true}) {
            SCOPED_TRACE(::testing::Message() << "phase=" << phase << " fail_generic=" << fail_generic);
            DeviceHostTransferExecutor executor(pool_, 8, options);
            CopyProofInjection injection{phase, false, fail_generic};
            ScopedCopyProofInjection scope(injection);
            const auto result = execute(executor);
            EXPECT_EQ(result.ok(), !fail_generic);
            EXPECT_EQ(result.copy_error, fail_generic ? CacheCopyError::COPY_FAILED : CacheCopyError::NONE);
            EXPECT_TRUE(result.buffer_safe);
            EXPECT_EQ(injection.generic_submits, 1u);
            EXPECT_EQ(injection.drains, fail_generic ? 2u : 1u);
            std::array<uint8_t, 64> copied{};
            const auto buffers = device_->convertIndexToBuffer(0, 1);
            ASSERT_EQ(cudaMemcpy(copied.data(), buffers.front().addr, copied.size(), cudaMemcpyDeviceToHost), cudaSuccess);
            EXPECT_EQ(copied, bytes_);
        }
    }
}

TEST_F(DeviceHostBufferSafetyTest, SuccessfulFallbackCannotEraseUnprovenAttempt) {
    for (const bool first_safe : {false, true}) {
        DeviceHostTransferExecutor executor(pool_, 8);
        executor.strategies_.clear();
        executor.strategies_.push_back(
            std::make_unique<ScriptedCopyStrategy>(StrategyResult::notApplicable(first_safe)));
        executor.strategies_.push_back(std::make_unique<ScriptedCopyStrategy>(StrategyResult::done(true)));
        const auto result = execute(executor);
        EXPECT_TRUE(result.ok());
        EXPECT_EQ(result.buffer_safe, first_safe);
    }
}

TEST_F(DeviceHostBufferSafetyTest, SafeFailureStillFailsBusinessOperation) {
    for (const bool failed_safe : {false, true}) {
        DeviceHostTransferExecutor executor(pool_, 8);
        executor.strategies_.clear();
        executor.strategies_.push_back(std::make_unique<ScriptedCopyStrategy>(
            StrategyResult::failed(TransferStatus::DEVICE_IO_ERROR, failed_safe)));
        const auto result = execute(executor);
        EXPECT_FALSE(result.ok());
        EXPECT_EQ(result.copy_error, CacheCopyError::COPY_FAILED);
        EXPECT_EQ(result.buffer_safe, failed_safe);
    }
}

TEST_F(DeviceHostBufferSafetyTest, NormalBatchAndStagedCompletionCarryProof) {
    auto host = torch::full({64}, 0x3C, torch::TensorOptions().dtype(torch::kUInt8).pinned_memory(true));
    auto device = torch::zeros({64}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    DeviceHostCopyPlan plan{false, 0, {host.data_ptr(), 64, 64},
                           {{host.data_ptr(), device.data_ptr(), 64, device.get_device(), 0, 0}}};
    const DeviceHostCopyExecutionContext context{acquireDeviceHostCopyStreams(device.get_device()),
                                                 DeviceHostCopyDirection::H2D};
    CudaBatchDeviceHostCopyStrategy batch;
    auto result = batch.tryExecute(plan, {}, context);
    ASSERT_NE(result.status, StrategyStatus::FAILED);
    EXPECT_TRUE(result.buffer_safe);  // Unsupported is a proven pre-submission skip.
    if (result.status == StrategyStatus::DONE) {
        EXPECT_TRUE(device.cpu().equal(host));
    }
    StagedSmDeviceHostCopyStrategy staged;
    DeviceHostCopyOptions options;
    options.staged_sm_min_tile_count = 1;
    options.staged_sm_min_bytes = 1;
    result = staged.tryExecute(plan, options, context);
    ASSERT_EQ(result.status, StrategyStatus::DONE);
    EXPECT_TRUE(result.buffer_safe);
    EXPECT_TRUE(device.cpu().equal(host));
}

}  // namespace
}  // namespace rtp_llm
