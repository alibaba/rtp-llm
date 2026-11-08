#include <chrono>
#include <memory>
#include <thread>

#include "gtest/gtest.h"
#include "autil/TimeUtility.h"
#include "torch/all.h"

#define protected public
#include "rtp_llm/cpp/engine_base/schedulers/FIFOScheduler.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/config/ConfigModules.h"

namespace rtp_llm {

namespace {

// The targeted validation lane runs driver-less (CUDA stub); KVCacheManager::init()
// would allocate GPU memory. The coalesce tests therefore drive the hold mechanics
// directly: streams are injected into the scheduler queues and
// maybeCoalesceContextRound() is invoked with the scheduler lock held, exactly as
// schedule() does. The cache manager is left uninitialized on purpose.
struct CoalesceFixture {
    std::shared_ptr<KVCacheManager> cache_manager = std::make_shared<KVCacheManager>(CacheConfig{});
    ResourceContext                 resource_context;
    ModelConfig                     model_config;
    RuntimeConfig                   runtime_config;
    PDSepConfig                     pd_sep_config;
    ParallelismConfig               parallelism_config;
    ModelSpecificConfig             model_specific_config;
    std::unique_ptr<FIFOScheduler>  scheduler;

    CoalesceFixture(int64_t window_ms, int64_t target_streams) {
        resource_context.cache_manager                         = cache_manager;
        model_config.max_seq_len                                = 8192;
        runtime_config.max_generate_batch_size                  = 16;
        runtime_config.fifo_scheduler_config.max_batch_tokens_size = 8192;
        runtime_config.fifo_scheduler_config.context_coalesce_window_ms       = window_ms;
        runtime_config.fifo_scheduler_config.context_coalesce_target_streams = target_streams;
        scheduler = std::make_unique<FIFOScheduler>(
            runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);
    }

    std::shared_ptr<GenerateStream> makeStream(const std::vector<int>& ids) {
        auto query                                   = std::make_shared<GenerateInput>();
        query->input_ids                             = torch::tensor(ids, torch::kInt32);
        auto generate_config                         = std::make_shared<GenerateConfig>();
        generate_config->max_new_tokens               = 1;
        generate_config->num_return_sequences         = 1;
        query->generate_config                       = generate_config;
        return std::make_shared<NormalGenerateStream>(
            query, model_config, runtime_config, resource_context, nullptr);
    }

    // Simulates an enqueue arriving while the round is held: the scheduler lock is
    // taken, the stream joins this round's new_streams_ (the join-at-boundary case
    // the production code reaches via admitWaitingStreams), the trigger is set and
    // the condition variable notified.
    void lateArrivalJoinsRound(std::shared_ptr<GenerateStream> stream, int delay_ms) {
        std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
        std::unique_lock<std::mutex> lock(scheduler->lock_);
        scheduler->new_streams_.push_back(stream);
        scheduler->schedule_trigger_ = true;
        scheduler->cond_.notify_all();
    }

    long holdMs() {
        std::unique_lock<std::mutex> lock(scheduler->lock_);
        scheduler->schedule_trigger_ = false;
        const auto t0 = std::chrono::steady_clock::now();
        scheduler->maybeCoalesceContextRound(lock);
        return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
    }
};

}  // namespace

TEST(FIFOSchedulerCoalesceTest, HoldsSoloRoundForConcurrentArrival) {
    CoalesceFixture fixture(/*window_ms=*/500, /*target_streams=*/2);
    auto            stream1 = fixture.makeStream({1, 2});
    fixture.scheduler->new_streams_.push_back(stream1);

    auto        stream2 = fixture.makeStream({3, 4});
    std::thread late_arrival(
        [&fixture, stream2]() { fixture.lateArrivalJoinsRound(stream2, /*delay_ms=*/60); });

    const long hold_ms = fixture.holdMs();
    late_arrival.join();

    EXPECT_EQ(fixture.scheduler->new_streams_.size(), 2u)
        << "the held round must carry the concurrent arrival";
    EXPECT_GE(hold_ms, 50) << "the solo round must have been held for the arrival";
    EXPECT_LT(hold_ms, 1000);
}

TEST(FIFOSchedulerCoalesceTest, WindowExpiresSoloRound) {
    CoalesceFixture fixture(/*window_ms=*/200, /*target_streams=*/2);
    auto            stream1 = fixture.makeStream({1, 2});
    fixture.scheduler->new_streams_.push_back(stream1);

    const long hold_ms = fixture.holdMs();

    EXPECT_EQ(fixture.scheduler->new_streams_.size(), 1u)
        << "no arrival: the window expires and the solo round dispatches";
    EXPECT_GE(hold_ms, 150) << "the round must have been held for (roughly) the full window";
    EXPECT_LT(hold_ms, 2000);
}

TEST(FIFOSchedulerCoalesceTest, DisabledReturnsImmediately) {
    CoalesceFixture fixture(/*window_ms=*/0, /*target_streams=*/2);
    auto            stream1 = fixture.makeStream({1, 2});
    fixture.scheduler->new_streams_.push_back(stream1);

    const long hold_ms = fixture.holdMs();

    EXPECT_EQ(fixture.scheduler->new_streams_.size(), 1u);
    EXPECT_LT(hold_ms, 100) << "window=0 must keep the legacy immediate dispatch";
}

TEST(FIFOSchedulerCoalesceTest, SkipsHoldWhenTargetAlreadyQueued) {
    CoalesceFixture fixture(/*window_ms=*/500, /*target_streams=*/2);
    auto            stream1 = fixture.makeStream({1, 2});
    auto            stream2 = fixture.makeStream({3, 4});
    fixture.scheduler->new_streams_.push_back(stream1);
    fixture.scheduler->new_streams_.push_back(stream2);

    const long hold_ms = fixture.holdMs();

    EXPECT_EQ(fixture.scheduler->new_streams_.size(), 2u)
        << "a round already at target is dispatched without any hold";
    EXPECT_LT(hold_ms, 100) << "no hold is needed when the round already reaches the target";
}

TEST(FIFOSchedulerCoalesceTest, NonContextStreamBypassesHold) {
    CoalesceFixture fixture(/*window_ms=*/500, /*target_streams=*/2);
    auto            stream1 = fixture.makeStream({1, 2});
    // A non-context (decode-side) stream must never be held by the coalescer.
    stream1->setIsContextStream(false);
    ASSERT_FALSE(stream1->isContextStream());
    fixture.scheduler->new_streams_.push_back(stream1);

    const long hold_ms = fixture.holdMs();

    EXPECT_LT(hold_ms, 100) << "non-context streams must never be held";
    EXPECT_EQ(fixture.scheduler->new_streams_.size(), 1u);
}

}  // namespace rtp_llm
