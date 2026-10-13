#include <memory>

#include "gtest/gtest.h"
#include "torch/all.h"
#include "rtp_llm/cpp/engine_base/schedulers/BatchDecodeScheduler.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/testing/TestBase.h"

namespace rtp_llm {

class BatchDecodeSchedulerTest: public DeviceTestBase {
protected:
    void SetUp() override {
        DeviceTestBase::SetUp();
        cache_manager_ = std::make_shared<KVCacheManager>(makeMhaCacheConfig(1, 16, 1, 4, 8, DataType::TYPE_FP16));
        ASSERT_TRUE(cache_manager_->init());
        model_config_.max_seq_len                                                       = 8192;
        runtime_config_.batch_decode_scheduler_config.batch_decode_scheduler_batch_size = 8;
        scheduler_ = std::make_unique<BatchDecodeScheduler>(runtime_config_, cache_manager_, nullptr);
        scheduler_->setForcePoll(true);
    }

    void TearDown() override {
        if (scheduler_) {
            EXPECT_TRUE(scheduler_->stop().ok());
            scheduler_.reset();
        }
        cache_manager_.reset();
        DeviceTestBase::TearDown();
    }

    GenerateStreamPtr makeStream() {
        auto input             = std::make_shared<GenerateInput>();
        input->input_ids       = torch::tensor({1}, torch::kInt32);
        input->generate_config = std::make_shared<GenerateConfig>();
        ResourceContext resources;
        resources.cache_manager = cache_manager_;
        return std::make_shared<NormalGenerateStream>(input, model_config_, runtime_config_, resources, nullptr);
    }

    void expectTerminalCleanup(const GenerateStreamPtr& stream) {
        EXPECT_EQ(stream->getStatus(), StreamState::FINISHED);
        EXPECT_TRUE(stream->streamCacheResource().isResourceReleased());
    }

    ModelConfig                           model_config_;
    RuntimeConfig                         runtime_config_;
    std::shared_ptr<KVCacheManager>       cache_manager_;
    std::unique_ptr<BatchDecodeScheduler> scheduler_;
};

TEST_F(BatchDecodeSchedulerTest, CancelledPartialBatchDoesNotBlockDrain) {
    auto stream = makeStream();
    ASSERT_TRUE(scheduler_->enqueue(stream).ok());
    stream->reportError(ErrorCode::CANCELLED, "request cancelled while waiting for batch");
    ASSERT_TRUE(scheduler_->schedule().ok());
    expectTerminalCleanup(stream);
    EXPECT_EQ(scheduler_->onflightStreams(), 0);
    EXPECT_TRUE(scheduler_->drainManager().waitDrained(0));
}

TEST_F(BatchDecodeSchedulerTest, ErrorPartialBatchDoesNotBlockDrain) {
    auto stream = makeStream();
    ASSERT_TRUE(scheduler_->enqueue(stream).ok());
    stream->reportError(ErrorCode::UNKNOWN_ERROR, "failed before full batch");
    ASSERT_TRUE(scheduler_->schedule().ok());
    expectTerminalCleanup(stream);
    EXPECT_EQ(scheduler_->onflightStreams(), 0);
    EXPECT_TRUE(scheduler_->drainManager().waitDrained(0));
}

TEST_F(BatchDecodeSchedulerTest, HealthyPartialBatchStillWaitsForBatch) {
    auto stream = makeStream();
    ASSERT_TRUE(scheduler_->enqueue(stream).ok());
    ASSERT_TRUE(scheduler_->schedule().ok());
    EXPECT_EQ(stream->getStatus(), StreamState::WAITING);
    EXPECT_FALSE(stream->streamCacheResource().isResourceReleased());
    EXPECT_EQ(scheduler_->onflightStreams(), 1);
    EXPECT_FALSE(scheduler_->drainManager().waitDrained(0));
}

TEST_F(BatchDecodeSchedulerTest, AlreadyFinishedWaiterCleanupIsIdempotent) {
    auto stream = makeStream();
    ASSERT_TRUE(scheduler_->enqueue(stream).ok());
    stream->reportError(ErrorCode::CANCELLED, "finished by another cleanup owner");
    ASSERT_EQ(stream->moveToNext(), StreamState::FINISHED);
    expectTerminalCleanup(stream);
    const auto free_before = cache_manager_->freeBlocksNum();
    ASSERT_TRUE(scheduler_->schedule().ok());
    expectTerminalCleanup(stream);
    EXPECT_EQ(cache_manager_->freeBlocksNum(), free_before);
    EXPECT_EQ(scheduler_->onflightStreams(), 0);
    ASSERT_TRUE(scheduler_->schedule().ok());
    EXPECT_EQ(cache_manager_->freeBlocksNum(), free_before);
    EXPECT_TRUE(scheduler_->drainManager().waitDrained(0));
}

TEST_F(BatchDecodeSchedulerTest, FullQueueCancellationCleansResourcesAndPreservesHealthyWaiter) {
    auto healthy = makeStream();
    ASSERT_TRUE(scheduler_->enqueue(healthy).ok());
    std::vector<GenerateStreamPtr> cancelled;
    for (int i = 0; i < 7; ++i) {
        auto stream = makeStream();
        ASSERT_TRUE(scheduler_->enqueue(stream).ok());
        stream->reportError(ErrorCode::CANCELLED, "cancelled full queue member");
        cancelled.push_back(std::move(stream));
    }
    ASSERT_TRUE(scheduler_->schedule().ok());
    for (const auto& stream : cancelled) {
        expectTerminalCleanup(stream);
    }
    EXPECT_EQ(healthy->getStatus(), StreamState::WAITING);
    EXPECT_EQ(scheduler_->onflightStreams(), 1);
}

}  // namespace rtp_llm
