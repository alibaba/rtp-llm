#include <chrono>
#include <functional>
#include <future>
#include <memory>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>

#include "torch/all.h"
#include "gmock/gmock-actions.h"
#include "gmock/gmock-function-mocker.h"
#include "gtest/gtest.h"
#include "autil/TimeUtility.h"
#include "kmonitor/client/common/MinMaxCalculator.h"
#include "kmonitor/client/core/MetricsCollector.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"

#define private public
#define protected public
#include "rtp_llm/cpp/engine_base/schedulers/PPScheduler.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/engine_base/stream/StreamCacheResource.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/cache/AsyncContext.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/block_tree_cache/load/LoadAsyncContext.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/cache/test/mock/MockCoordinatorCacheManager.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/models_py/bindings/core/Types.h"

using namespace std;

namespace rtp_llm {

/** Observe the real idle wait predicate so stop is requested after schedule reaches its blocking path. */
class IdleWaitPPScheduler: public PPScheduler {
public:
    using PPScheduler::PPScheduler;

    std::promise<void> idle_wait;

private:
    bool waitPredicate() override {
        const bool ready = PPScheduler::waitPredicate();
        if (!ready && !observed_idle_) {
            observed_idle_ = true;
            idle_wait.set_value();
        }
        return ready;
    }

    bool observed_idle_ = false;
};

std::shared_ptr<LoadAsyncContext> makeControlledAllocatorContext() {
    auto coordinator = std::make_shared<LoadContextCoordinator>(
        [](const std::shared_ptr<LoadAsyncContext>&) { return true; }, [](LoadAsyncContext&) {});
    std::vector<TransferDescriptor> descriptors{TransferDescriptor{nullptr,
                                                                   /*group_set_id=*/0,
                                                                   /*path_index=*/0,
                                                                   Tier::HOST,
                                                                   Tier::DEVICE,
                                                                   BlockIndicesType{1}}};
    auto context = coordinator->create(std::move(descriptors), {false}, /*matched_blocks=*/1);
    if (!coordinator->registerContext(context)) {
        throw std::runtime_error("failed to register controlled allocator context");
    }
    return context;
}

class PPSchedulerTest: public DeviceTestBase {
protected:
    using ContextSelector = std::function<std::shared_ptr<AsyncContext>(const MallocInfo&)>;

    PPSchedulerTest(): perf_scope("PERF_TEST", "0") {}

    void SetUp() override {
        DeviceTestBase::SetUp();
        cache_config_ = test::makeSimpleMhaCacheConfig(
            /*layer_num=*/1, /*block_num=*/21, /*tokens_per_block=*/2, rtp_llm::DataType::TYPE_INT8);
        cache_manager_ = std::make_shared<KVCacheManager>(cache_config_);
        ASSERT_TRUE(cache_manager_->init());
    }

    void TearDown() override {
        if (real_allocator_) {
            cache_manager_->coordinator_manager_ = real_allocator_;
        }
        DeviceTestBase::TearDown();
    }

    template<typename Scheduler = PPScheduler>
    std::shared_ptr<Scheduler> createScheduler(size_t max_generate_batch_size = 100,
                                                 RoleType role = RoleType::PDFUSION,
                                                 SpeculativeType type = SP_TYPE_NONE,
                                                 bool cp_enabled = false) {
        ModelConfig model_config;
        model_config.max_seq_len = 8192;
        model_config.vocab_size = 128;
        model_config.special_tokens.eos_token_id = -1;
        RuntimeConfig runtime_config;
        runtime_config.max_generate_batch_size = max_generate_batch_size;
        runtime_config.fifo_scheduler_config.max_batch_tokens_size = 8192;
        PDSepConfig pd_sep_config;
        pd_sep_config.role_type = role;
        ParallelismConfig parallelism_config;
        parallelism_config.pp_size = 2;
        if (cp_enabled) {
            parallelism_config.tp_size = 2;
            parallelism_config.prefill_cp_config.method = CPRotateMethod::ALL_GATHER;
        }
        SpeculativeExecutionConfig sp_config;
        sp_config.type = type;
        sp_config.gen_num_per_cycle = 3;
        return std::make_shared<Scheduler>(runtime_config, model_config, pd_sep_config,
                                             parallelism_config, ModelSpecificConfig{}, sp_config, cache_manager_);
    }

    GenerateStreamPtr createStream(const std::vector<int>& input_tokens        = {1, 2, 3},
                                   bool                    reuse_cache         = false,
                                   bool                    enable_memory_cache = false,
                                   int                     max_new_tokens      = 1,
                                   const std::vector<int>& variable_num_beams  = {},
                                   RoleType                role_type           = RoleType::PDFUSION) {
        ResourceContext resource_context;
        resource_context.cache_manager       = cache_manager_;
        resource_context.reuse_cache         = reuse_cache;
        resource_context.enable_memory_cache = enable_memory_cache;
        resource_context.role_type           = role_type;

        ModelConfig model_config;
        model_config.max_seq_len = 8192;
        model_config.vocab_size = 128;
        model_config.special_tokens.eos_token_id = -1;
        RuntimeConfig runtime_config;

        auto query                           = std::make_shared<GenerateInput>();
        auto generate_config                 = std::make_shared<GenerateConfig>();
        query->request_id                    = next_request_id_++;
        generate_config->reuse_cache         = reuse_cache;
        generate_config->enable_memory_cache = enable_memory_cache;
        generate_config->max_new_tokens      = max_new_tokens;
        generate_config->variable_num_beams  = variable_num_beams;
        query->input_ids                     = torch::tensor(input_tokens, torch::kInt32);
        query->generate_config               = generate_config;
        auto stream = std::make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
        return stream;
    }

    void installReadinessAllocator(ContextSelector selector) {
        real_allocator_ = cache_manager_->coordinator_manager_;
        mock_allocator_ = std::make_shared<testing::NiceMock<MockCoordinatorCacheManager>>(cache_manager_->config_);
        initial_malloc_calls_ = 0;

        ON_CALL(*mock_allocator_, initMallocForCommonLen(testing::_))
            .WillByDefault(testing::Invoke([this, selector](const MallocInfo& info) {
                ++initial_malloc_calls_;
                auto result = real_allocator_->initMallocForCommonLen(info);
                if (result.success) {
                    result.async_context = selector(info);
                }
                return result;
            }));
        ON_CALL(*mock_allocator_, incrMalloc(testing::_)).WillByDefault(testing::Invoke([this](const MallocInfo& info) {
            return real_allocator_->incrMalloc(info);
        }));
        ON_CALL(*mock_allocator_, free(testing::_)).WillByDefault(testing::Invoke([this](const FreeInfo& info) {
            real_allocator_->free(info);
        }));
        ON_CALL(*mock_allocator_, insertIntoCache(testing::_, testing::_))
            .WillByDefault(testing::Invoke([this](const InsertInfo& info, size_t& resident_prefix_length) {
                real_allocator_->insertIntoCache(info, resident_prefix_length);
            }));
        ON_CALL(*mock_allocator_, singleBatchNeedBlocks(testing::_, testing::_, testing::_))
            .WillByDefault(
                testing::Invoke([this](const BatchKVCacheResourcePtr& resource, int seq_len, int reserve_step) {
                    return real_allocator_->singleBatchNeedBlocks(resource, seq_len, reserve_step);
                }));
        ON_CALL(*mock_allocator_, seqSizePerBlock()).WillByDefault(testing::Invoke([this]() {
            return real_allocator_->seqSizePerBlock();
        }));
        ON_CALL(*mock_allocator_, maxAvailableTokensNum()).WillByDefault(testing::Invoke([this]() {
            return real_allocator_->maxAvailableTokensNum();
        }));

        cache_manager_->coordinator_manager_ = mock_allocator_;
    }

protected:
    autil::EnvGuard                                                 perf_scope;
    CacheConfig                                                     cache_config_;
    std::shared_ptr<KVCacheManager>                                 cache_manager_;
    int64_t                                                         next_request_id_{1};
    CoordinatorCacheManagerPtr                                      real_allocator_;
    std::shared_ptr<testing::NiceMock<MockCoordinatorCacheManager>> mock_allocator_;
    size_t                                                          initial_malloc_calls_{0};
};

TEST_F(PPSchedulerTest, StopWakesSingleDpScheduleWithNoOutstandingRequests) {
    auto scheduler = createScheduler<IdleWaitPPScheduler>();
    ASSERT_FALSE(scheduler->need_fill_fake_stream_);
    ASSERT_TRUE(scheduler->empty());
    auto idle = scheduler->idle_wait.get_future();
    auto scheduled = std::async(std::launch::async, [scheduler] { return scheduler->schedule(); });
    /** Keep assertions nonfatal until the waiting schedule has been released. */
    EXPECT_EQ(idle.wait_for(std::chrono::seconds(10)), std::future_status::ready);
    EXPECT_EQ(scheduled.wait_for(std::chrono::milliseconds(0)), std::future_status::timeout);
    EXPECT_TRUE(scheduler->stop().ok());
    const auto stopped = scheduled.wait_for(std::chrono::seconds(10));
    EXPECT_EQ(stopped, std::future_status::ready);
    if (stopped != std::future_status::ready) {
        /** Release the future even when the regression is a missing stop notification or stop predicate. */
        {
            std::lock_guard<std::mutex> lock(scheduler->lock_);
            scheduler->schedule_trigger_ = true;
        }
        scheduler->cond_.notify_all();
    }
    const auto output = scheduled.get();
    ASSERT_TRUE(output.ok()) << output.status().ToString();
    EXPECT_TRUE(output->streams.empty());
    EXPECT_TRUE(output->finished_request_ids.empty());
}

TEST_F(PPSchedulerTest, AdmissionMetricsResetWhenReadyDecodeSkipsAdmission) {
    auto monitor = std::make_shared<kmonitor::KMonitor>("pp_scheduler_metrics_test");
    monitor->SetServiceName("");
    /** Exercise ready-decode scheduling on PDFUSION; PREFILL rejects local multi-token requests. */
    auto scheduler = createScheduler(1, RoleType::PDFUSION);
    scheduler->metrics_reporter_ =
        std::make_shared<kmonitor::MetricsReporter>(monitor, "", kmonitor::MetricsTags("dp_rank", "1"));
    int64_t snapshot_time_ms = 0;
    auto    expect_metrics   = [&](int admitted, int tokens, int running, int waiting, bool has_wait_age) {
        kmonitor::MetricsCollector snapshot;
        monitor->GetMetrics(&snapshot, {kmonitor::NORMAL}, snapshot_time_ms += 10000);
        std::map<std::string, std::string> values;
        for (const auto* record : snapshot.GetRecords().getRecords()) {
            EXPECT_EQ(record->Tags()->FindTag("dp_rank"), "1");
            for (const auto* value : record->Values()) {
                EXPECT_TRUE(values.emplace(value->Name(), value->Value()).second);
            }
        }
        const std::map<std::string, int> expected{{"rtp_llm_scheduler_admitted_context_batch_size", admitted},
                                                  {"rtp_llm_scheduler_admitted_context_token_size", tokens},
                                                  {"rtp_llm_running_stream_size", running},
                                                  {"rtp_llm_wait_stream_size", waiting}};
        for (const auto& [name, value] : expected) {
            kmonitor::MinMaxCalculator gauge;
            gauge.Add(value);
            EXPECT_EQ(values.at(name), gauge.ToString()) << name;
        }
        kmonitor::MinMaxCalculator zero;
        zero.Add(0);
        EXPECT_EQ(values.at("rtp_llm_scheduler_waiting_oldest_age_us") != zero.ToString(), has_wait_age);
    };

    auto first = createStream({1, 2, 3}, false, false, 4);
    first->recordSchedulerEnqueueTime(autil::TimeUtility::currentTimeInMicroSeconds() - 10000);
    ASSERT_TRUE(scheduler->enqueue(first).ok());
    ASSERT_EQ(scheduler->schedule()->streams.size(), 1u);
    expect_metrics(1, 3, 1, 0, true);

    auto second = createStream({5, 6}, false, false, 4);
    second->recordSchedulerEnqueueTime(autil::TimeUtility::currentTimeInMicroSeconds() - 10000);
    ASSERT_TRUE(scheduler->enqueue(second).ok());
    first->update({torch::tensor({{4}}, torch::kInt32), 1}, false);
    first->clearPPInflight();
    const auto decode = scheduler->schedule();
    ASSERT_TRUE(decode.ok());
    ASSERT_EQ(decode->streams.size(), 1u);
    EXPECT_EQ(decode->streams.front(), first);
    EXPECT_EQ(second->getStatus(), StreamState::WAITING);
    expect_metrics(0, 0, 1, 1, false);

    /** While A is in flight, B may enter a separate batch. A stays in the running count. */
    const auto prefill = scheduler->schedule();
    ASSERT_TRUE(prefill.ok());
    ASSERT_EQ(prefill->streams.size(), 1u);
    EXPECT_EQ(prefill->streams.front(), second);
    EXPECT_TRUE(first->isPPInflight());
    EXPECT_TRUE(second->isPPInflight());
    expect_metrics(1, 2, 2, 0, true);

    const auto empty = scheduler->schedule();
    ASSERT_TRUE(empty.ok());
    EXPECT_TRUE(empty->streams.empty());
    expect_metrics(0, 0, 2, 0, false);
}

TEST_F(PPSchedulerTest, SingleDpInflightKeepsProducingEmptySchedulesUntilResultCommit) {
    auto scheduler = createScheduler();
    ASSERT_FALSE(scheduler->need_fill_fake_stream_);
    auto stream = createStream({1, 2, 3}, false, false, 4);
    ASSERT_TRUE(scheduler->enqueue(stream).ok());
    auto first = scheduler->schedule();
    ASSERT_TRUE(first.ok());
    ASSERT_EQ(first->streams.size(), 1u);
    ASSERT_TRUE(stream->isPPInflight());
    /** With no queued requests, inflight work alone must keep the scheduler awake without fake streams. */
    for (int round = 0; round < 3; ++round) {
        SCOPED_TRACE(round);
        ASSERT_FALSE(scheduler->empty());
        ASSERT_TRUE(scheduler->waitPredicate());
        auto pending = scheduler->schedule();
        ASSERT_TRUE(pending.ok());
        EXPECT_TRUE(pending->streams.empty());
        EXPECT_TRUE(pending->finished_request_ids.empty());
        EXPECT_TRUE(stream->isPPInflight());
    }
    stream->update({torch::tensor({{4}}, torch::kInt32), 1}, false);
    stream->clearPPInflight();
    auto next = scheduler->schedule();
    ASSERT_TRUE(next.ok());
    ASSERT_EQ(next->streams.size(), 1u);
    EXPECT_EQ(next->streams.front(), stream);
    EXPECT_TRUE(stream->isPPInflight());
}

TEST_F(PPSchedulerTest, CancelledInflightKeepsKvUntilResponseAndRetiresOnce) {
    auto scheduler = createScheduler();
    auto stream = createStream({1, 2, 3});
    ASSERT_TRUE(scheduler->enqueue(stream).ok());
    ASSERT_TRUE(scheduler->schedule().ok());
    const auto blocks = stream->curBlocksNum();
    ASSERT_GT(blocks, 0u);
    stream->reportError(ErrorCode::CANCELLED, "cancelled while PP owns KV");
    auto pending = scheduler->schedule();
    ASSERT_TRUE(pending.ok());
    EXPECT_TRUE(pending->streams.empty());
    EXPECT_TRUE(pending->finished_request_ids.empty());
    EXPECT_EQ(stream->curBlocksNum(), blocks);
    EXPECT_FALSE(stream->streamCacheResource().isResourceReleased());
    stream->update({torch::tensor({{4}}, torch::kInt32), 1}, false);
    stream->clearPPInflight();
    auto retired = scheduler->schedule();
    ASSERT_TRUE(retired.ok());
    EXPECT_TRUE(retired->streams.empty());
    EXPECT_EQ(retired->finished_request_ids, (std::vector<int64_t>{stream->streamId()}));
    EXPECT_TRUE(stream->streamCacheResource().isResourceReleased());
    EXPECT_EQ(stream->seqLength(), 3);
    /** Only after retirement is there no outstanding work. The next real request must not repeat the cleanup ID. */
    EXPECT_TRUE(scheduler->empty());
    EXPECT_FALSE(scheduler->waitPredicate());
    auto next_stream = createStream({4, 5, 6});
    ASSERT_TRUE(scheduler->enqueue(next_stream).ok());
    auto next = scheduler->schedule();
    ASSERT_TRUE(next.ok());
    ASSERT_EQ(next->streams.size(), 1u);
    EXPECT_EQ(next->streams.front(), next_stream);
    EXPECT_TRUE(next->finished_request_ids.empty());
}

TEST_F(PPSchedulerTest, LoadedRequestsReenterAdmissionAndRespectCurrentBatchLimit) {
    auto scheduler = createScheduler(1);
    auto first = createStream({1, 2, 3}, true);
    auto second = createStream({4, 5, 6}, true);
    auto first_context = makeControlledAllocatorContext();
    auto second_context = makeControlledAllocatorContext();
    ASSERT_TRUE(scheduler->enqueue(first).ok());
    ASSERT_TRUE(scheduler->enqueue(second).ok());
    installReadinessAllocator([&](const MallocInfo& info) {
        return info.request_id == first->streamId() ? first_context : second_context;
    });
    auto loading = scheduler->schedule();
    ASSERT_TRUE(loading.ok());
    EXPECT_TRUE(loading->streams.empty());
    ASSERT_EQ(first->getStatus(), StreamState::LOADING_CACHE);
    ASSERT_EQ(second->getStatus(), StreamState::LOADING_CACHE);
    ASSERT_TRUE(first_context->completeTransfers(1, true));
    ASSERT_TRUE(second_context->completeTransfers(1, true));
    auto ready = scheduler->schedule();
    ASSERT_TRUE(ready.ok());
    ASSERT_EQ(ready->streams.size(), 1u);
    EXPECT_EQ(ready->streams.front(), first);
    EXPECT_EQ(second->getStatus(), StreamState::WAITING);
    EXPECT_FALSE(second->hasEvent(StreamEvents::CanRun));
    EXPECT_EQ(initial_malloc_calls_, 2u);
    auto next = scheduler->schedule();
    ASSERT_TRUE(next.ok());
    ASSERT_EQ(next->streams.size(), 1u);
    EXPECT_EQ(next->streams.front(), second);
    EXPECT_TRUE(first->isPPInflight());
    EXPECT_TRUE(second->isPPInflight());
    EXPECT_EQ(initial_malloc_calls_, 2u);
}

TEST_F(PPSchedulerTest, PartitionsExecutionModesWithoutDrainingInflightBatches) {
    for (const auto role : {RoleType::PDFUSION, RoleType::PREFILL}) {
        for (const bool first_prefill_only : {false, true}) {
            SCOPED_TRACE(testing::Message() << "role=" << static_cast<int>(role)
                                           << ", first_prefill_only=" << first_prefill_only);
            auto scheduler = createScheduler(4, role);
            const auto make_stream = [&](bool prefill_only) {
                auto stream = createStream({1, 2, 3}, false, false, prefill_only ? 0 : 2, {}, role);
                stream->generateConfig()->pd_separation = role == RoleType::PREFILL && !prefill_only;
                return stream;
            };
            auto first      = make_stream(first_prefill_only);
            auto other      = make_stream(!first_prefill_only);
            auto first_peer = make_stream(first_prefill_only);
            auto other_peer = make_stream(!first_prefill_only);
            for (const auto& stream : {first, other, first_peer, other_peer}) {
                ASSERT_TRUE(stream->isContextStream());
                ASSERT_TRUE(scheduler->enqueue(stream).ok());
            }

            auto first_batch = scheduler->schedule();
            ASSERT_TRUE(first_batch.ok());
            EXPECT_EQ(first_batch->streams, (std::list<GenerateStreamPtr>{first, first_peer}));
            for (const auto& stream : {other, other_peer}) {
                EXPECT_EQ(stream->getStatus(), StreamState::WAITING);
                EXPECT_FALSE(stream->hasEvent(StreamEvents::CanRun));
                EXPECT_EQ(stream->curBlocksNum(), 0u);
            }

            /** All four requests are context streams; the second mode can enter a separate PP batch immediately. */
            auto other_batch = scheduler->schedule();
            ASSERT_TRUE(other_batch.ok());
            EXPECT_EQ(other_batch->streams, (std::list<GenerateStreamPtr>{other, other_peer}));
            for (const auto& stream : {first, other, first_peer, other_peer}) {
                EXPECT_TRUE(stream->isPPInflight());
                EXPECT_FALSE(stream->hasError());
            }
            EXPECT_EQ(scheduler->waitingStreamsSize(), 0);
        }
    }
}

TEST_F(PPSchedulerTest, AsyncLoadSelectsModeAndCompletedLoadRequiresReadmission) {
    auto scheduler    = createScheduler(2, RoleType::PREFILL);
    auto generation   = createStream({1, 2, 3}, true, false, 1, {}, RoleType::PREFILL);
    auto prefill_only = createStream({4, 5, 6}, false, false, 0, {}, RoleType::PREFILL);
    auto context      = makeControlledAllocatorContext();
    installReadinessAllocator([&](const MallocInfo& info) -> std::shared_ptr<AsyncContext> {
        return info.request_id == generation->streamId() ? context : nullptr;
    });
    ASSERT_TRUE(scheduler->enqueue(generation).ok());
    ASSERT_TRUE(scheduler->enqueue(prefill_only).ok());

    auto loading = scheduler->schedule();
    ASSERT_TRUE(loading.ok());
    EXPECT_TRUE(loading->streams.empty());
    ASSERT_EQ(generation->getStatus(), StreamState::LOADING_CACHE);
    ASSERT_GT(generation->curBlocksNum(), 0u);
    EXPECT_EQ(prefill_only->getStatus(), StreamState::WAITING);
    EXPECT_FALSE(prefill_only->hasEvent(StreamEvents::CanRun));
    EXPECT_EQ(prefill_only->curBlocksNum(), 0u);
    EXPECT_EQ(initial_malloc_calls_, 1u);

    /** The completed load joins the queue behind the opposite mode and must not reuse its old CanRun admission. */
    ASSERT_TRUE(context->completeTransfers(1, true));
    auto prefill = scheduler->schedule();
    ASSERT_TRUE(prefill.ok());
    EXPECT_EQ(prefill->streams, (std::list<GenerateStreamPtr>{prefill_only}));
    EXPECT_EQ(generation->getStatus(), StreamState::WAITING);
    EXPECT_FALSE(generation->hasEvent(StreamEvents::CanRun));
    EXPECT_EQ(initial_malloc_calls_, 2u);

    auto loaded = scheduler->schedule();
    ASSERT_TRUE(loaded.ok());
    EXPECT_EQ(loaded->streams, (std::list<GenerateStreamPtr>{generation}));
    EXPECT_TRUE(prefill_only->isPPInflight());
    EXPECT_EQ(generation->getStatus(), StreamState::RUNNING);
    EXPECT_FALSE(generation->hasError());
    EXPECT_EQ(initial_malloc_calls_, 2u);
}

TEST_F(PPSchedulerTest, RetryableKvShortageDoesNotPinModeOrStarveLoadedRequest) {
    auto scheduler    = createScheduler(2, RoleType::PREFILL);
    auto generation   = createStream(std::vector<int>(30, 1), true, false, 1, {}, RoleType::PREFILL);
    auto prefill_only = createStream(std::vector<int>(12, 2), false, false, 0, {}, RoleType::PREFILL);
    auto context      = makeControlledAllocatorContext();
    installReadinessAllocator([&](const MallocInfo& info) -> std::shared_ptr<AsyncContext> {
        return info.request_id == generation->streamId() ? context : nullptr;
    });
    ASSERT_TRUE(scheduler->enqueue(generation).ok());
    auto loading = scheduler->schedule();
    ASSERT_TRUE(loading.ok());
    EXPECT_TRUE(loading->streams.empty());
    ASSERT_EQ(generation->getStatus(), StreamState::LOADING_CACHE);
    ASSERT_EQ(generation->curBlocksNum(), 15u);
    ASSERT_EQ(real_allocator_->freeBlocksNum(), 5u);

    /** The waiter needs six blocks, so its retryable allocation failure must leave this round's mode unset. */
    ASSERT_TRUE(scheduler->enqueue(prefill_only).ok());
    auto blocked = scheduler->schedule();
    ASSERT_TRUE(blocked.ok());
    EXPECT_TRUE(blocked->streams.empty());
    ASSERT_EQ(prefill_only->getStatus(), StreamState::WAITING);
    ASSERT_FALSE(prefill_only->hasError());
    ASSERT_TRUE(prefill_only->hasEvent(StreamEvents::CanRun));
    ASSERT_FALSE(prefill_only->hasEvent(StreamEvents::LoadInitiated));
    ASSERT_EQ(prefill_only->curBlocksNum(), 0u);

    ASSERT_TRUE(context->completeTransfers(1, true));
    auto ready = scheduler->schedule();
    ASSERT_TRUE(ready.ok());
    EXPECT_EQ(ready->streams, (std::list<GenerateStreamPtr>{generation}));
    EXPECT_EQ(prefill_only->getStatus(), StreamState::WAITING);
    EXPECT_FALSE(prefill_only->hasError());
    EXPECT_EQ(prefill_only->curBlocksNum(), 0u);
    EXPECT_EQ(initial_malloc_calls_, 3u);

    generation->reportEvent(StreamEvents::GenerateDone);
    generation->clearPPInflight();
    auto recovered = scheduler->schedule();
    ASSERT_TRUE(recovered.ok());
    EXPECT_EQ(recovered->streams, (std::list<GenerateStreamPtr>{prefill_only}));
    EXPECT_EQ(recovered->finished_request_ids, (std::vector<int64_t>{generation->streamId()}));
    EXPECT_EQ(prefill_only->getStatus(), StreamState::RUNNING);
    EXPECT_EQ(prefill_only->curBlocksNum(), 6u);
    EXPECT_EQ(initial_malloc_calls_, 4u);
}

TEST_F(PPSchedulerTest, OppositeModeTimeoutsAndErrorsRetireOnceWithoutAdmission) {
    auto scheduler  = createScheduler();
    auto generation = createStream({1, 2, 3});
    auto expired    = createStream({4, 5, 6}, false, false, 0);
    auto cancelled  = createStream({7, 8, 9}, false, false, 0);
    for (const auto& stream : {generation, expired, cancelled}) {
        ASSERT_TRUE(scheduler->enqueue(stream).ok());
    }
    expired->generateConfig()->timeout_ms = 1;
    expired->begin_time_us_               = autil::TimeUtility::currentTimeInMicroSeconds() - 1000000;
    cancelled->reportError(ErrorCode::CANCELLED, "cancelled before admission");

    auto output = scheduler->schedule();
    ASSERT_TRUE(output.ok());
    EXPECT_EQ(output->streams, (std::list<GenerateStreamPtr>{generation}));
    EXPECT_EQ(output->finished_request_ids, (std::vector<int64_t>{expired->streamId(), cancelled->streamId()}));
    EXPECT_EQ(expired->statusInfo().code(), ErrorCode::GENERATE_TIMEOUT);
    EXPECT_EQ(cancelled->statusInfo().code(), ErrorCode::CANCELLED);
    for (const auto& stream : {expired, cancelled}) {
        EXPECT_EQ(stream->getStatus(), StreamState::FINISHED);
        EXPECT_FALSE(stream->hasEvent(StreamEvents::CanRun));
        EXPECT_EQ(stream->curBlocksNum(), 0u);
        EXPECT_TRUE(stream->streamCacheResource().isResourceReleased());
    }
    EXPECT_EQ(scheduler->waitingStreamsSize(), 0);

    auto next = scheduler->schedule();
    ASSERT_TRUE(next.ok());
    EXPECT_TRUE(next->streams.empty());
    EXPECT_TRUE(next->finished_request_ids.empty());
}

TEST_F(PPSchedulerTest, PrefillRejectsLocalMultiTokenBeforeKvAllocation) {
    installReadinessAllocator([](const MallocInfo&) { return std::shared_ptr<AsyncContext>{}; });
    for (const auto type : {SP_TYPE_NONE, SP_TYPE_MTP, SP_TYPE_EAGLE, SP_TYPE_DSPARK}) {
        for (const bool cp_enabled : {false, true}) {
            SCOPED_TRACE(testing::Message() << "speculative_type=" << static_cast<int>(type)
                                           << ", cp_enabled=" << cp_enabled);
            auto scheduler = createScheduler(1, RoleType::PREFILL, type, cp_enabled);
            auto stream = createStream({1, 2, 3}, false, false, 2, {}, RoleType::PREFILL);
            stream->generateConfig()->can_use_pd_separation = true;
            stream->generateConfig()->pd_separation = false;
            ASSERT_TRUE(scheduler->enqueue(stream).ok());

            auto output = scheduler->schedule();
            ASSERT_TRUE(output.ok());
            EXPECT_TRUE(output->streams.empty());
            EXPECT_EQ(output->finished_request_ids, (std::vector<int64_t>{stream->streamId()}));
            EXPECT_EQ(stream->getStatus(), StreamState::FINISHED);
            EXPECT_EQ(stream->statusInfo().code(), ErrorCode::INVALID_PARAMS);
            EXPECT_NE(stream->statusInfo().ToString().find("not using PD separation"), std::string::npos);
            EXPECT_FALSE(stream->hasEvent(StreamEvents::CanRun));
            EXPECT_EQ(initial_malloc_calls_, 0u);
            EXPECT_EQ(stream->curBlocksNum(), 0u);
            EXPECT_TRUE(stream->streamCacheResource().isResourceReleased());
            EXPECT_TRUE(scheduler->empty());
        }
    }
}

TEST_F(PPSchedulerTest, AdmitsPdSingleTokenAndNonPrefillRequests) {
    struct RequestCase {
        RoleType role;
        int      max_new_tokens;
        bool     pd_separation;
    };
    const RequestCase cases[] = {
        {RoleType::PREFILL, 2, true},
        {RoleType::PREFILL, 1, false},
        {RoleType::PDFUSION, 2, false},
        {RoleType::DECODE, 2, false},
    };
    for (const auto& request : cases) {
        SCOPED_TRACE(testing::Message() << "role=" << static_cast<int>(request.role)
                                       << ", max_new_tokens=" << request.max_new_tokens
                                       << ", pd_separation=" << request.pd_separation);
        auto scheduler = createScheduler(1, request.role);
        auto stream = createStream({1, 2, 3}, false, false, request.max_new_tokens, {}, request.role);
        stream->generateConfig()->pd_separation = request.pd_separation;
        if (request.role == RoleType::DECODE) {
            ASSERT_TRUE(stream->initKVBlock().ok());
            stream->reportEvent(StreamEvents::LoadInitiated);
        }
        ASSERT_TRUE(scheduler->enqueue(stream).ok());

        auto output = scheduler->schedule();
        ASSERT_TRUE(output.ok());
        ASSERT_EQ(output->streams.size(), 1u);
        EXPECT_EQ(output->streams.front(), stream);
        EXPECT_TRUE(output->finished_request_ids.empty());
        EXPECT_FALSE(stream->hasError());
        EXPECT_EQ(stream->getStatus(), StreamState::RUNNING);
    }
}

TEST_F(PPSchedulerTest, PrefillRejectsInvalidGroupMembersWhenBatchIsFull) {
    auto scheduler = createScheduler(1, RoleType::PREFILL);
    auto valid = createStream({1, 2, 3}, false, false, 1, {}, RoleType::PREFILL);
    auto invalid = createStream({4, 5, 6}, false, false, 2, {}, RoleType::PREFILL);
    auto cancelled = createStream({7, 8, 9}, false, false, 2, {}, RoleType::PREFILL);
    cancelled->reportError(ErrorCode::CANCELLED, "cancelled before admission");
    const auto enqueued = scheduler->enqueueGroup({valid, invalid, cancelled});
    ASSERT_EQ(enqueued.first, (std::vector<bool>{true, true, true}));

    auto output = scheduler->schedule();
    ASSERT_TRUE(output.ok());
    ASSERT_EQ(output->streams.size(), 1u);
    EXPECT_EQ(output->streams.front(), valid);
    EXPECT_FALSE(valid->hasError());
    EXPECT_TRUE(valid->isPPInflight());
    EXPECT_EQ(output->finished_request_ids, (std::vector<int64_t>{invalid->streamId(), cancelled->streamId()}));
    EXPECT_EQ(invalid->statusInfo().code(), ErrorCode::INVALID_PARAMS);
    EXPECT_EQ(cancelled->statusInfo().code(), ErrorCode::CANCELLED);
    EXPECT_EQ(cancelled->statusInfo().ToString(), "cancelled before admission");
    for (const auto& rejected : {invalid, cancelled}) {
        EXPECT_EQ(rejected->getStatus(), StreamState::FINISHED);
        EXPECT_FALSE(rejected->hasEvent(StreamEvents::CanRun));
        EXPECT_EQ(rejected->curBlocksNum(), 0u);
        EXPECT_TRUE(rejected->streamCacheResource().isResourceReleased());
    }
    EXPECT_EQ(scheduler->waitingStreamsSize(), 0);

    /** Keep the valid stream in flight so another schedule can confirm rejected IDs are retired only once. */
    auto next = scheduler->schedule();
    ASSERT_TRUE(next.ok());
    EXPECT_TRUE(next->streams.empty());
    EXPECT_TRUE(next->finished_request_ids.empty());
}

TEST_F(PPSchedulerTest, PdDecodeInitializesCandidatesOnceAndKeepsTailProposal) {
    for (const auto type : {SP_TYPE_MTP, SP_TYPE_DSPARK}) {
        auto scheduler = createScheduler(1, RoleType::DECODE, type);
        auto stream = createStream({1, 2, 3}, false, false, 8, {}, RoleType::DECODE);
        ASSERT_TRUE(stream->initKVBlock().ok());
        stream->update({torch::tensor({{4}}, torch::kInt32), 1});
        if (type == SP_TYPE_MTP) {
            stream->setProposeToken({4, 5});
            /** RPC already installed the anchor and draft slots before scheduler admission. */
            auto buffer = std::make_shared<SpeculativeExecutorStreamOutput>();
            buffer->propose_step = 3;
            buffer->tokens = torch::zeros({1, 2}, torch::TensorOptions().dtype(torch::kInt32).pinned_memory(true));
            buffer->tokens.copy_(torch::tensor({{4, 5}}, torch::kInt32));
            stream->setSPOutputBuffer(buffer);
        } else {
            /** DSpARK P publishes features only; D has no proposal before scheduler admission. */
            EXPECT_TRUE(stream->getProposeToken().empty());
            ASSERT_EQ(stream->getSPOutputBuffer(), nullptr);
        }
        stream->reportEvent(StreamEvents::LoadInitiated);
        ASSERT_TRUE(scheduler->enqueue(stream).ok());
        auto first = scheduler->schedule();
        ASSERT_TRUE(first.ok());
        ASSERT_EQ(first->streams.size(), 1u);
        EXPECT_EQ(stream->getProposeToken(),
                  (type == SP_TYPE_DSPARK ? std::vector<int>{4, 0, 0, 0} : std::vector<int>{4, 5, 0, 0}));
        ASSERT_NE(stream->getSPOutputBuffer(), nullptr);
        EXPECT_EQ(stream->getSPOutputBuffer()->propose_step, 3);
        EXPECT_FALSE(stream->getSPOutputBuffer()->all_probs.defined());
        const auto& tokens = stream->getSPOutputBuffer()->tokens;
        ASSERT_TRUE(tokens.defined());
        EXPECT_EQ(tokens.sizes().vec(), (std::vector<int64_t>{1, 4}));
        EXPECT_TRUE(torch::equal(tokens, torch::tensor(stream->getProposeToken(), torch::kInt32).reshape({1, 4})));
        stream->specUpdate({torch::tensor({{5}}, torch::kInt32), 1,
                            torch::tensor({6, 7, 8}, torch::kInt32), torch::Tensor(), torch::Tensor()}, false);
        stream->clearPPInflight();
        auto next = scheduler->schedule();
        ASSERT_TRUE(next.ok());
        ASSERT_EQ(next->streams.size(), 1u);
        EXPECT_EQ(stream->getProposeToken(), (std::vector<int>{5, 6, 7, 8}));
    }
}

}  // namespace rtp_llm
