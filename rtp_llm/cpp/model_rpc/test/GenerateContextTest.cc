#include <gtest/gtest.h>

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <unistd.h>

#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/model_rpc/GenerateContext.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/testing/TestBase.h"

namespace {

// stopStream waits for the scheduler with usleep(). Run one scheduler tick at
// that exact boundary, after its cancellation decision, without racing threads
// or depending on a sleep duration. Other tests/threads retain libc behaviour.
thread_local std::function<void()> context_wait_scheduler_tick;

class ScopedContextSchedulerTick {
public:
    explicit ScopedContextSchedulerTick(std::function<void()> tick) {
        EXPECT_FALSE(context_wait_scheduler_tick);
        context_wait_scheduler_tick = std::move(tick);
    }

    ~ScopedContextSchedulerTick() {
        context_wait_scheduler_tick = nullptr;
    }
};

}  // namespace

extern "C" int __real_usleep(useconds_t usec);

extern "C" int __wrap_usleep(useconds_t usec) {
    if (context_wait_scheduler_tick) {
        auto tick                   = std::move(context_wait_scheduler_tick);
        context_wait_scheduler_tick = nullptr;
        tick();
        return 0;
    }
    return __real_usleep(usec);
}

namespace rtp_llm {
namespace {

class GenerateContextCachePublicationTest: public DeviceTestBase, public ::testing::WithParamInterface<bool> {
protected:
    void SetUp() override {
        DeviceTestBase::SetUp();
        auto config = test::makeSimpleMhaCacheConfig(
            /*layer_num=*/1, /*block_num=*/16, /*tokens_per_block=*/2, DataType::TYPE_FP16);
        cache_manager_ = std::make_shared<KVCacheManager>(config);
        ASSERT_TRUE(cache_manager_->init());
    }

    std::shared_ptr<NormalGenerateStream> makeStream(int64_t request_id) {
        auto input                             = std::make_shared<GenerateInput>();
        input->request_id                      = request_id;
        input->begin_time_us                   = currentTimeUs();
        input->input_ids                       = torch::tensor({1, 2, 3, 4, 5}, torch::kInt32);
        input->generate_config                 = std::make_shared<GenerateConfig>();
        input->generate_config->max_new_tokens = 1;

        ModelConfig model_config;
        model_config.max_seq_len = 16;
        model_config.vocab_size  = 32;
        ResourceContext resources;
        resources.cache_manager       = cache_manager_;
        resources.reuse_cache         = true;
        resources.enable_device_cache = true;
        auto stream = std::make_shared<NormalGenerateStream>(input, model_config, RuntimeConfig{}, resources, nullptr);
        stream->setNeedReleaseResource(true);
        return stream;
    }

    void start(const std::shared_ptr<NormalGenerateStream>& stream) {
        ASSERT_TRUE(stream->initKVBlock().ok());
        ASSERT_GT(stream->curBlocksNum(), 0);
        stream->generate_status_->status.store(StreamState::RUNNING);
    }

    void finishOutput(const std::shared_ptr<NormalGenerateStream>& stream) {
        const auto token = torch::tensor({{6}}, torch::kInt32);
        stream->update({token,
                        1,
                        torch::Tensor(),
                        torch::Tensor(),
                        torch::Tensor(),
                        torch::Tensor(),
                        torch::Tensor(),
                        torch::Tensor(),
                        torch::Tensor(),
                        torch::Tensor()});

        const auto final_output = stream->nextOutput(1);
        ASSERT_TRUE(final_output.ok());
        ASSERT_EQ(final_output.value().generate_outputs.size(), 1);
        ASSERT_TRUE(final_output.value().generate_outputs[0].finished);
        const auto drained = stream->nextOutput(1);
        ASSERT_FALSE(drained.ok());
        ASSERT_EQ(drained.status().code(), ErrorCode::FINISHED);
        ASSERT_EQ(stream->getStatus(), StreamState::RUNNING);
        ASSERT_TRUE(stream->statusInfo().ok());
        ASSERT_FALSE(stream->stream_cache_resource_->isResourceReleased());
    }

    void cleanupContext(const std::shared_ptr<NormalGenerateStream>& stream) {
        kmonitor::MetricsReporterPtr reporter;
        auto                         meta = std::make_shared<RpcServerRuntimeMeta>();
        auto context = std::make_unique<GenerateContext>(stream->streamId(), 0, nullptr, reporter, meta);
        context->setStream(stream);
        ScopedContextSchedulerTick scheduler_tick([stream] {
            const auto state = stream->moveToNext();
            EXPECT_EQ(state, StreamState::FINISHED);
            if (state != StreamState::FINISHED) {
                // Preserve the failure above, but unblock stopStream if a
                // regression forgot to cancel an active request.
                stream->reportError(ErrorCode::EXECUTION_EXCEPTION, "test scheduler cleanup");
                stream->moveToNext();
            }
        });
        if (GetParam()) {
            context.reset();
        } else {
            context->stopStream();
            EXPECT_EQ(context->getStream(), nullptr);
            context.reset();
        }
        // A cleanup implementation may leave completed work to the scheduler
        // without waiting. WAITING cancellation also needs a scheduler tick.
        if (stream->getStatus() != StreamState::FINISHED) {
            EXPECT_EQ(stream->moveToNext(), StreamState::FINISHED);
        }
        EXPECT_TRUE(stream->stream_cache_resource_->isResourceReleased());
    }

    void expectPrefixReuse(int expected_tokens) {
        auto probe = makeStream(42);
        ASSERT_TRUE(probe->initKVBlock().ok());
        EXPECT_EQ(probe->reuseLength(), expected_tokens);
        // The probe did not execute, so it must not publish additional cache.
        probe->reportError(ErrorCode::CANCELLED, "test probe cleanup");
        EXPECT_EQ(probe->moveToNext(), StreamState::FINISHED);
    }

    std::shared_ptr<KVCacheManager> cache_manager_;
};

TEST_P(GenerateContextCachePublicationTest, PendingNormalCompletionPublishesPrefix) {
    auto stream = makeStream(41);
    ASSERT_NO_FATAL_FAILURE(start(stream));
    ASSERT_NO_FATAL_FAILURE(finishOutput(stream));

    cleanupContext(stream);

    EXPECT_TRUE(stream->statusInfo().ok());
    EXPECT_FALSE(stream->hasEvent(StreamEvents::Error));
    expectPrefixReuse(4);
}

TEST_P(GenerateContextCachePublicationTest, CommittedNormalCompletionKeepsPublishedPrefix) {
    auto stream = makeStream(41);
    ASSERT_NO_FATAL_FAILURE(start(stream));
    ASSERT_NO_FATAL_FAILURE(finishOutput(stream));
    ASSERT_EQ(stream->moveToNext(), StreamState::FINISHED);

    cleanupContext(stream);

    EXPECT_TRUE(stream->statusInfo().ok());
    expectPrefixReuse(4);
}

TEST_P(GenerateContextCachePublicationTest, ActiveStreamStillCancelsWithoutPublishing) {
    auto stream = makeStream(41);
    ASSERT_NO_FATAL_FAILURE(start(stream));

    cleanupContext(stream);

    EXPECT_EQ(stream->statusInfo().code(), ErrorCode::CANCELLED);
    expectPrefixReuse(0);
}

TEST_P(GenerateContextCachePublicationTest, WaitingStreamStillCancelsWithoutPublishing) {
    auto stream = makeStream(41);
    ASSERT_EQ(stream->getStatus(), StreamState::WAITING);

    cleanupContext(stream);

    EXPECT_EQ(stream->statusInfo().code(), ErrorCode::CANCELLED);
    expectPrefixReuse(0);
}

TEST_P(GenerateContextCachePublicationTest, ExistingErrorIsPreservedWithoutPublishing) {
    auto stream = makeStream(41);
    ASSERT_NO_FATAL_FAILURE(start(stream));
    ASSERT_NO_FATAL_FAILURE(finishOutput(stream));
    // A real execution failure after GenerateDone must still win; only normal
    // context cleanup should avoid introducing a new cancellation error.
    stream->reportError(ErrorCode::EXECUTION_EXCEPTION, "original execution failure");

    cleanupContext(stream);

    EXPECT_EQ(stream->statusInfo().code(), ErrorCode::EXECUTION_EXCEPTION);
    EXPECT_NE(stream->statusInfo().ToString().find("original execution failure"), std::string::npos);
    expectPrefixReuse(0);
}

TEST_P(GenerateContextCachePublicationTest, ExplicitCancellationAfterCompletionStillPreventsPublication) {
    auto stream = makeStream(41);
    ASSERT_NO_FATAL_FAILURE(start(stream));
    ASSERT_NO_FATAL_FAILURE(finishOutput(stream));
    stream->reportError(ErrorCode::CANCELLED, "forced cancellation");

    cleanupContext(stream);

    EXPECT_EQ(stream->statusInfo().code(), ErrorCode::CANCELLED);
    EXPECT_NE(stream->statusInfo().ToString().find("forced cancellation"), std::string::npos);
    expectPrefixReuse(0);
}

TEST_P(GenerateContextCachePublicationTest, CancellationBeforeCompletionIsPreserved) {
    auto stream = makeStream(41);
    ASSERT_NO_FATAL_FAILURE(start(stream));
    stream->reportError(ErrorCode::CANCELLED, "cancelled before completion");
    stream->reportEvent(StreamEvents::GenerateDone);

    cleanupContext(stream);

    EXPECT_EQ(stream->statusInfo().code(), ErrorCode::CANCELLED);
    EXPECT_NE(stream->statusInfo().ToString().find("cancelled before completion"), std::string::npos);
    expectPrefixReuse(0);
}

INSTANTIATE_TEST_SUITE_P(CleanupEntry,
                         GenerateContextCachePublicationTest,
                         ::testing::Values(false, true),
                         [](const ::testing::TestParamInfo<bool>& info) {
                             return info.param ? "Destructor" : "StopStream";
                         });

}  // namespace
}  // namespace rtp_llm
