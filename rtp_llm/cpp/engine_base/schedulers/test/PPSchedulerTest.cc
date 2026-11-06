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
    std::shared_ptr<Scheduler> createScheduler(size_t          max_generate_batch_size = 100,
                                               RoleType        role                    = RoleType::PDFUSION,
                                               SpeculativeType type                    = SP_TYPE_NONE,
                                               bool            cp_enabled              = false) {
        ModelConfig model_config;
        model_config.max_seq_len                 = model_max_seq_len_;
        model_config.vocab_size                  = 128;
        model_config.special_tokens.eos_token_id = -1;
        RuntimeConfig runtime_config;
        runtime_config.max_generate_batch_size                     = max_generate_batch_size;
        runtime_config.fifo_scheduler_config.max_batch_tokens_size = model_max_seq_len_;
        PDSepConfig pd_sep_config;
        pd_sep_config.role_type = role;
        ParallelismConfig parallelism_config;
        parallelism_config.pp_size = 2;
        if (cp_enabled) {
            parallelism_config.tp_size                  = 2;
            parallelism_config.prefill_cp_config.method = CPRotateMethod::ALL_GATHER;
        }
        SpeculativeExecutionConfig sp_config;
        sp_config.type              = type;
        sp_config.gen_num_per_cycle = 3;
        return std::make_shared<Scheduler>(runtime_config,
                                           model_config,
                                           pd_sep_config,
                                           parallelism_config,
                                           ModelSpecificConfig{},
                                           sp_config,
                                           cache_manager_);
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
        model_config.max_seq_len                 = model_max_seq_len_;
        model_config.vocab_size                  = 128;
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
        auto stream =
            std::make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
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
    int                                                             model_max_seq_len_{8192};
    int64_t                                                         next_request_id_{1};
    CoordinatorCacheManagerPtr                                      real_allocator_;
    std::shared_ptr<testing::NiceMock<MockCoordinatorCacheManager>> mock_allocator_;
    size_t                                                          initial_malloc_calls_{0};
};

TEST_F(PPSchedulerTest, StopWakesSingleDpScheduleWithNoOutstandingRequests) {
    auto scheduler = createScheduler<IdleWaitPPScheduler>();
    ASSERT_FALSE(scheduler->need_fill_fake_stream_);
    ASSERT_TRUE(scheduler->empty());
    auto idle      = scheduler->idle_wait.get_future();
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

TEST_F(PPSchedulerTest, CapacityBlockedWaitingStreamBacksOffBetweenPolls) {
    auto scheduler = createScheduler(0);
    auto stream    = createStream();
    ASSERT_TRUE(scheduler->enqueue(stream).ok());
    EXPECT_TRUE(scheduler->waitPredicate());

    auto first = scheduler->schedule();
    ASSERT_TRUE(first.ok());
    EXPECT_TRUE(first->streams.empty());
    EXPECT_EQ(stream->getStatus(), StreamState::WAITING);
    EXPECT_FALSE(scheduler->waitPredicate());

    const auto start   = std::chrono::steady_clock::now();
    auto       second  = scheduler->schedule();
    const auto elapsed = std::chrono::steady_clock::now() - start;
    ASSERT_TRUE(second.ok());
    EXPECT_TRUE(second->streams.empty());
    EXPECT_EQ(stream->getStatus(), StreamState::WAITING);
    EXPECT_GE(elapsed, std::chrono::milliseconds(50));
}

TEST_F(PPSchedulerTest, AdmissionMetricsResetWhenReadyDecodeSkipsAdmission) {
    auto monitor = std::make_shared<kmonitor::KMonitor>("pp_scheduler_metrics_test");
    monitor->SetServiceName("");
    /** Exercise ready-decode scheduling on PDFUSION; PREFILL rejects local multi-token requests. */
    auto scheduler = createScheduler(2, RoleType::PDFUSION);
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
    expect_metrics(0, 0, 1, 1, true);

    /** The second concurrency slot allows B to enter a separate batch while A remains in flight. */
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

TEST_F(PPSchedulerTest, DecodeTopUpCountsReadyAndInflightRequestsOnce) {
    auto scheduler = createScheduler(3, RoleType::DECODE);
    std::vector<GenerateStreamPtr> streams;
    for (int i = 0; i < 4; ++i) {
        auto stream = createStream({1, 2, 3}, false, false, 8, {}, RoleType::DECODE);
        ASSERT_TRUE(stream->initKVBlock().ok());
        stream->update({torch::tensor({{4}}, torch::kInt32), 1});
        stream->reportEvent(StreamEvents::LoadInitiated);
        ASSERT_FALSE(stream->isContextStream());
        streams.push_back(stream);
    }
    ASSERT_TRUE(scheduler->enqueue(streams[0]).ok());
    ASSERT_TRUE(scheduler->enqueue(streams[1]).ok());
    auto initial = scheduler->schedule();
    ASSERT_TRUE(initial.ok());
    ASSERT_EQ(initial->streams, (std::list<GenerateStreamPtr>{streams[0], streams[1]}));

    ASSERT_TRUE(scheduler->enqueue(streams[2]).ok());
    ASSERT_TRUE(scheduler->enqueue(streams[3]).ok());
    streams[0]->clearPPInflight();
    /** One ready request and one inflight request leave exactly one slot for a preallocated waiter. */
    auto topped_up = scheduler->schedule();
    ASSERT_TRUE(topped_up.ok());
    EXPECT_EQ(topped_up->streams, (std::list<GenerateStreamPtr>{streams[0], streams[2]}));
    EXPECT_TRUE(streams[1]->isPPInflight());
    EXPECT_EQ(scheduler->runningStreamsSize(), 3);
    EXPECT_EQ(streams[3]->getStatus(), StreamState::WAITING);
    EXPECT_FALSE(streams[3]->hasEvent(StreamEvents::CanRun));
    EXPECT_GT(streams[3]->curBlocksNum(), 0u);

    streams[1]->reportEvent(StreamEvents::GenerateDone);
    streams[1]->clearPPInflight();
    auto admitted = scheduler->schedule();
    ASSERT_TRUE(admitted.ok());
    EXPECT_EQ(admitted->streams, (std::list<GenerateStreamPtr>{streams[3]}));
    EXPECT_EQ(admitted->finished_request_ids, (std::vector<int64_t>{streams[1]->streamId()}));
    EXPECT_EQ(scheduler->runningStreamsSize(), 3);
}

TEST_F(PPSchedulerTest, CancelledInflightKeepsConcurrencyAndKvUntilResultCommit) {
    auto scheduler = createScheduler(1);
    auto stream    = createStream({1, 2, 3});
    ASSERT_TRUE(scheduler->enqueue(stream).ok());
    ASSERT_TRUE(scheduler->schedule().ok());
    ASSERT_TRUE(stream->isPPInflight());
    const auto blocks = stream->curBlocksNum();
    ASSERT_GT(blocks, 0u);
    /** Inflight work alone keeps single-DP scheduling awake while waiting for its result. */
    EXPECT_FALSE(scheduler->need_fill_fake_stream_);
    EXPECT_FALSE(scheduler->empty());
    ASSERT_TRUE(scheduler->waitPredicate());

    auto next_stream = createStream({4, 5, 6});
    ASSERT_TRUE(scheduler->enqueue(next_stream).ok());
    stream->reportError(ErrorCode::CANCELLED, "cancelled while PP owns KV");
    auto pending = scheduler->schedule();
    ASSERT_TRUE(pending.ok());
    EXPECT_TRUE(pending->streams.empty());
    EXPECT_TRUE(pending->finished_request_ids.empty());
    EXPECT_EQ(stream->curBlocksNum(), blocks);
    EXPECT_FALSE(stream->streamCacheResource().isResourceReleased());
    EXPECT_EQ(next_stream->getStatus(), StreamState::WAITING);
    EXPECT_FALSE(next_stream->hasEvent(StreamEvents::CanRun));
    EXPECT_EQ(next_stream->curBlocksNum(), 0u);
    stream->update({torch::tensor({{4}}, torch::kInt32), 1}, false);
    stream->clearPPInflight();
    auto retired = scheduler->schedule();
    ASSERT_TRUE(retired.ok());
    EXPECT_EQ(retired->streams, (std::list<GenerateStreamPtr>{next_stream}));
    EXPECT_EQ(retired->finished_request_ids, (std::vector<int64_t>{stream->streamId()}));
    EXPECT_TRUE(stream->streamCacheResource().isResourceReleased());
    EXPECT_EQ(stream->seqLength(), 3);
    EXPECT_EQ(scheduler->runningStreamsSize(), 1);
    auto next = scheduler->schedule();
    ASSERT_TRUE(next.ok());
    EXPECT_TRUE(next->streams.empty());
    EXPECT_TRUE(next->finished_request_ids.empty());

    next_stream->reportEvent(StreamEvents::GenerateDone);
    next_stream->clearPPInflight();
    auto finished = scheduler->schedule();
    ASSERT_TRUE(finished.ok());
    EXPECT_TRUE(finished->streams.empty());
    EXPECT_EQ(finished->finished_request_ids, (std::vector<int64_t>{next_stream->streamId()}));
    EXPECT_TRUE(scheduler->empty());
    EXPECT_FALSE(scheduler->waitPredicate());
}

TEST_F(PPSchedulerTest, AsyncLoadsRespectAdmissionCapacityAndReenterAfterInflightRetires) {
    auto scheduler      = createScheduler(1);
    auto first          = createStream({1, 2, 3}, true);
    auto second         = createStream({4, 5, 6}, true);
    auto first_context  = makeControlledAllocatorContext();
    auto second_context = makeControlledAllocatorContext();
    ASSERT_TRUE(scheduler->enqueue(first).ok());
    ASSERT_TRUE(scheduler->enqueue(second).ok());
    installReadinessAllocator(
        [&](const MallocInfo& info) { return info.request_id == first->streamId() ? first_context : second_context; });
    auto loading = scheduler->schedule();
    ASSERT_TRUE(loading.ok());
    EXPECT_TRUE(loading->streams.empty());
    ASSERT_EQ(first->getStatus(), StreamState::LOADING_CACHE);
    ASSERT_EQ(second->getStatus(), StreamState::WAITING);
    EXPECT_FALSE(second->hasEvent(StreamEvents::CanRun));
    EXPECT_EQ(second->curBlocksNum(), 0u);
    EXPECT_EQ(initial_malloc_calls_, 1u);
    EXPECT_FALSE(scheduler->waitPredicate());

    /** Earlier loads retain KV ownership but do not reserve a later round's concurrency admission. */
    auto next_loading = scheduler->schedule();
    ASSERT_TRUE(next_loading.ok());
    EXPECT_TRUE(next_loading->streams.empty());
    ASSERT_EQ(first->getStatus(), StreamState::LOADING_CACHE);
    ASSERT_EQ(second->getStatus(), StreamState::LOADING_CACHE);
    EXPECT_EQ(initial_malloc_calls_, 2u);
    EXPECT_FALSE(scheduler->waitPredicate());
    ASSERT_TRUE(first_context->completeTransfers(1, true));
    ASSERT_TRUE(second_context->completeTransfers(1, true));
    auto ready = scheduler->schedule();
    ASSERT_TRUE(ready.ok());
    ASSERT_EQ(ready->streams.size(), 1u);
    EXPECT_EQ(ready->streams.front(), first);
    EXPECT_EQ(second->getStatus(), StreamState::WAITING);
    EXPECT_FALSE(second->hasEvent(StreamEvents::CanRun));
    EXPECT_EQ(initial_malloc_calls_, 2u);
    auto blocked = scheduler->schedule();
    ASSERT_TRUE(blocked.ok());
    EXPECT_TRUE(blocked->streams.empty());
    EXPECT_EQ(second->getStatus(), StreamState::WAITING);
    EXPECT_FALSE(second->hasEvent(StreamEvents::CanRun));
    EXPECT_TRUE(first->isPPInflight());
    EXPECT_FALSE(second->isPPInflight());
    EXPECT_EQ(initial_malloc_calls_, 2u);

    first->reportEvent(StreamEvents::GenerateDone);
    first->clearPPInflight();
    auto next = scheduler->schedule();
    ASSERT_TRUE(next.ok());
    EXPECT_EQ(next->streams, (std::list<GenerateStreamPtr>{second}));
    EXPECT_EQ(next->finished_request_ids, (std::vector<int64_t>{first->streamId()}));
    EXPECT_TRUE(second->isPPInflight());
    EXPECT_EQ(scheduler->runningStreamsSize(), 1);
    EXPECT_EQ(initial_malloc_calls_, 2u);
}

TEST_F(PPSchedulerTest, KvLimitCountsExistingAndNewOwnersOnce) {
    auto scheduler = createScheduler(4);
    scheduler->max_inited_kv_cache_streams_ = 3;
    auto existing = createStream({1, 2, 3});
    auto first    = createStream({4, 5, 6});
    auto second   = createStream({7, 8, 9});
    auto waiting  = createStream({10, 11, 12});
    ASSERT_TRUE(existing->initKVBlock().ok());
    ASSERT_GT(existing->curBlocksNum(), 0u);
    existing->reportEvent(StreamEvents::LoadInitiated);
    for (const auto& stream : {existing, first, second, waiting}) {
        ASSERT_TRUE(scheduler->enqueue(stream).ok());
    }

    /** Readmission of a KV owner leaves two KV slots for fresh allocations. */
    auto admitted = scheduler->schedule();
    ASSERT_TRUE(admitted.ok());
    EXPECT_EQ(admitted->streams, (std::list<GenerateStreamPtr>{existing, first, second}));
    EXPECT_EQ(scheduler->runningStreamsSize(), 3);
    EXPECT_GT(first->curBlocksNum(), 0u);
    EXPECT_GT(second->curBlocksNum(), 0u);
    EXPECT_EQ(waiting->getStatus(), StreamState::WAITING);
    EXPECT_FALSE(waiting->hasEvent(StreamEvents::CanRun));
    EXPECT_EQ(waiting->curBlocksNum(), 0u);
}

TEST_F(PPSchedulerTest, AsyncLoadsRespectCurrentBatchLimitsAndExecutionMode) {
    struct BudgetCase {
        const char* name;
        size_t      first_tokens;
        size_t      second_tokens;
        size_t      token_limit;
        bool        cp_enabled;
        bool        second_fits;
        bool        second_prefill_only;
    };
    const BudgetCase cases[] = {
        {"cp_single_prefill", 3, 3, 8192, true, false, false},
        {"token_sum", 3, 3, 6, false, false, false},
        {"padded_tokens", 4, 2, 7, false, false, false},
        {"oversized_first_prefill", 8, 3, 5, false, false, false},
        {"remaining_budget", 3, 3, 7, false, true, false},
        {"execution_mode", 3, 3, 8192, false, false, true},
    };
    int64_t                          loading_request_id = 0;
    std::shared_ptr<LoadAsyncContext> loading_context;
    installReadinessAllocator([&](const MallocInfo& info) -> std::shared_ptr<AsyncContext> {
        return info.request_id == loading_request_id ? loading_context : nullptr;
    });

    for (const auto& budget : cases) {
        SCOPED_TRACE(budget.name);
        auto scheduler = createScheduler(2, RoleType::PREFILL, SP_TYPE_NONE, budget.cp_enabled);
        scheduler->max_batch_tokens_size_ = budget.token_limit;
        auto first  = createStream(std::vector<int>(budget.first_tokens, 1), true, false, 1, {}, RoleType::PREFILL);
        auto second = createStream(std::vector<int>(budget.second_tokens, 2),
                                   false,
                                   false,
                                   budget.second_prefill_only ? 0 : 1,
                                   {},
                                   RoleType::PREFILL);
        loading_request_id = first->streamId();
        loading_context    = makeControlledAllocatorContext();
        const auto malloc_calls_before = initial_malloc_calls_;
        ASSERT_TRUE(scheduler->enqueue(first).ok());
        ASSERT_TRUE(scheduler->enqueue(second).ok());

        auto loading = scheduler->schedule();
        ASSERT_TRUE(loading.ok());
        ASSERT_EQ(first->getStatus(), StreamState::LOADING_CACHE);
        ASSERT_GT(first->curBlocksNum(), 0u);
        ASSERT_TRUE(loading_context->completeTransfers(1, true));
        if (budget.second_fits) {
            EXPECT_EQ(loading->streams, (std::list<GenerateStreamPtr>{second}));
            EXPECT_EQ(second->getStatus(), StreamState::RUNNING);
            EXPECT_EQ(initial_malloc_calls_, malloc_calls_before + 2);
            continue;
        }
        EXPECT_TRUE(loading->streams.empty());
        EXPECT_EQ(second->getStatus(), StreamState::WAITING);
        EXPECT_FALSE(second->hasEvent(StreamEvents::CanRun));
        EXPECT_EQ(second->curBlocksNum(), 0u);
        EXPECT_EQ(initial_malloc_calls_, malloc_calls_before + 1);

        /** A completed load must pass the batch limits and execution-mode check again. */
        auto ready = scheduler->schedule();
        ASSERT_TRUE(ready.ok());
        EXPECT_EQ(ready->streams, (std::list<GenerateStreamPtr>{second}));
        EXPECT_EQ(first->getStatus(), StreamState::WAITING);
        EXPECT_FALSE(first->hasEvent(StreamEvents::CanRun));

        /** Inflight batches occupy concurrency slots without constraining the next batch's mode or token budget. */
        auto next = scheduler->schedule();
        ASSERT_TRUE(next.ok());
        EXPECT_EQ(next->streams, (std::list<GenerateStreamPtr>{first}));
        EXPECT_TRUE(first->isPPInflight());
        EXPECT_TRUE(second->isPPInflight());
        EXPECT_EQ(scheduler->runningStreamsSize(), 2);
        EXPECT_EQ(initial_malloc_calls_, malloc_calls_before + 2);
    }
}

TEST_F(PPSchedulerTest, FullLengthPrefillOnlyRespectsTokenBudget) {
    model_max_seq_len_ = 8;
    for (const auto role : {RoleType::PDFUSION, RoleType::PREFILL}) {
        for (const size_t budget : {0, 7, 8, 9}) {
            SCOPED_TRACE(testing::Message() << "role=" << static_cast<int>(role) << ", budget=" << budget);
            auto scheduler = createScheduler(2, role);
            scheduler->max_batch_tokens_size_ = budget;
            auto stream = createStream(std::vector<int>(8, 1), false, false, 0, {}, role);
            ASSERT_EQ(stream->inputLength(), stream->maxSeqLen());
            ASSERT_TRUE(scheduler->enqueue(stream).ok());

            const auto result = scheduler->schedule();
            ASSERT_TRUE(result.ok());
            const bool fits = budget >= 8;
            ASSERT_EQ(result->streams.size(), fits ? 1u : 0u);
            EXPECT_EQ(scheduler->waitingStreamsSize(), fits ? 0 : 1);
            EXPECT_EQ(stream->getStatus(), fits ? StreamState::RUNNING : StreamState::WAITING);
            EXPECT_EQ(stream->hasEvent(StreamEvents::CanRun), fits);
            EXPECT_FALSE(stream->hasError());
            if (fits) {
                EXPECT_EQ(result->streams.front(), stream);
                /** Another in-flight PP batch consumes concurrency, not this round's token budget. */
                auto next = createStream(std::vector<int>(8, 2), false, false, 0, {}, role);
                ASSERT_TRUE(scheduler->enqueue(next).ok());
                const auto next_batch = scheduler->schedule();
                ASSERT_TRUE(next_batch.ok());
                EXPECT_EQ(next_batch->streams, (std::list<GenerateStreamPtr>{next}));
                EXPECT_TRUE(stream->isPPInflight());
            }
        }
    }
}

TEST_F(PPSchedulerTest, PrefillTokenAndPaddingLimitsPreserveGenerationReserve) {
    model_max_seq_len_ = 8;
    for (const int max_new_tokens : {0, 1}) {
        for (const int second_length : {2, 4}) {
            for (const size_t budget : {7, 8, 9}) {
                SCOPED_TRACE(testing::Message() << "max_new_tokens=" << max_new_tokens
                                               << ", second_length=" << second_length << ", budget=" << budget);
                auto scheduler = createScheduler(2);
                scheduler->max_batch_tokens_size_ = budget;
                auto first  = createStream(std::vector<int>(4, 1), false, false, max_new_tokens);
                auto second = createStream(std::vector<int>(second_length, 2), false, false, max_new_tokens);
                ASSERT_TRUE(scheduler->enqueue(first).ok());
                ASSERT_TRUE(scheduler->enqueue(second).ok());

                const auto result = scheduler->schedule();
                ASSERT_TRUE(result.ok());
                /** Both shapes need an eight-token rectangle; generation must also leave one token free. */
                const bool both_fit = budget >= (max_new_tokens == 0 ? 8u : 9u);
                EXPECT_EQ(result->streams, both_fit ? (std::list<GenerateStreamPtr>{first, second}) :
                                                    (std::list<GenerateStreamPtr>{first}));
                EXPECT_EQ(second->getStatus(), both_fit ? StreamState::RUNNING : StreamState::WAITING);
                EXPECT_FALSE(first->hasError());
                EXPECT_FALSE(second->hasError());
            }
        }
    }
}

TEST_F(PPSchedulerTest, PartitionsExecutionModesWithoutDrainingInflightBatches) {
    for (const auto role : {RoleType::PDFUSION, RoleType::PREFILL}) {
        for (const bool first_prefill_only : {false, true}) {
            SCOPED_TRACE(testing::Message()
                         << "role=" << static_cast<int>(role) << ", first_prefill_only=" << first_prefill_only);
            auto       scheduler   = createScheduler(4, role);
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

TEST_F(PPSchedulerTest, RetryableKvShortageDoesNotPinModeOrStarveLoadedRequest) {
    auto scheduler    = createScheduler(1, RoleType::PREFILL);
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

    /** A failed six-block allocation must leave the sole admission slot and execution mode available. */
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

TEST_F(PPSchedulerTest, WaitingCleanupRunsWhenAdmissionIsDeferred) {
    for (const auto role : {RoleType::PDFUSION, RoleType::PREFILL, RoleType::DECODE}) {
        SCOPED_TRACE(static_cast<int>(role));
        /** P/PDFUSION defer admission for ready decode; D defers it when concurrency is full. */
        auto scheduler = createScheduler(role == RoleType::DECODE ? 1 : 2, role);
        auto ready     = createStream({1, 2, 3}, false, false, 8, {}, role);
        ready->generateConfig()->pd_separation = role == RoleType::PREFILL;
        if (role == RoleType::DECODE) {
            ASSERT_TRUE(ready->initKVBlock().ok());
            ready->update({torch::tensor({{4}}, torch::kInt32), 1});
            ready->reportEvent(StreamEvents::LoadInitiated);
        }
        ASSERT_TRUE(scheduler->enqueue(ready).ok());
        auto first = scheduler->schedule();
        ASSERT_TRUE(first.ok());
        ASSERT_EQ(first->streams, (std::list<GenerateStreamPtr>{ready}));
        /** Keep the PREFILL stream local to exercise a ready fallback decode batch. */
        StreamUpdateInfo update{torch::tensor({{5}}, torch::kInt32), 1};
        update.update_remote_generate = false;
        ready->update(update, false);
        ready->clearPPInflight();

        const int error_max_tokens = role == RoleType::DECODE ? 8 : 0;
        auto expired   = createStream({5, 6, 7}, false, false, error_max_tokens, {}, role);
        auto cancelled = createStream({8, 9, 10}, false, false, error_max_tokens, {}, role);
        auto waiting   = createStream({11, 12, 13}, false, false, role == RoleType::DECODE ? 8 : 1, {}, role);
        /** Cleanup must release existing KV even for requests of the opposite execution mode. */
        for (const auto& stream : {expired, cancelled, waiting}) {
            ASSERT_TRUE(stream->initKVBlock().ok());
            ASSERT_GT(stream->curBlocksNum(), 0u);
            if (role == RoleType::DECODE) {
                stream->update({torch::tensor({{4}}, torch::kInt32), 1});
            }
            stream->reportEvent(StreamEvents::LoadInitiated);
            ASSERT_TRUE(scheduler->enqueue(stream).ok());
        }
        waiting->reportEvent(StreamEvents::CanRun);
        expired->generateConfig()->timeout_ms = 1;
        expired->begin_time_us_               = autil::TimeUtility::currentTimeInMicroSeconds() - 1000000;
        cancelled->reportError(ErrorCode::CANCELLED, "cancelled while waiting behind ready decode");
        const auto waiting_blocks = waiting->curBlocksNum();

        auto output = scheduler->schedule();
        ASSERT_TRUE(output.ok());
        EXPECT_EQ(output->streams, (std::list<GenerateStreamPtr>{ready}));
        EXPECT_EQ(output->finished_request_ids, (std::vector<int64_t>{expired->streamId(), cancelled->streamId()}));
        EXPECT_EQ(expired->statusInfo().code(), ErrorCode::GENERATE_TIMEOUT);
        EXPECT_EQ(cancelled->statusInfo().code(), ErrorCode::CANCELLED);
        for (const auto& stream : {expired, cancelled}) {
            EXPECT_EQ(stream->getStatus(), StreamState::FINISHED);
            EXPECT_FALSE(stream->hasEvent(StreamEvents::CanRun));
            EXPECT_EQ(stream->curBlocksNum(), 0u);
            EXPECT_TRUE(stream->streamCacheResource().isResourceReleased());
        }
        EXPECT_EQ(waiting->getStatus(), StreamState::WAITING);
        EXPECT_EQ(waiting->curBlocksNum(), waiting_blocks);
        EXPECT_FALSE(waiting->isPPInflight());
        EXPECT_EQ(scheduler->waitingStreamsSize(), 1);

        ready->reportEvent(StreamEvents::GenerateDone);
        ready->clearPPInflight();
        auto admitted = scheduler->schedule();
        ASSERT_TRUE(admitted.ok());
        EXPECT_EQ(admitted->streams, (std::list<GenerateStreamPtr>{waiting}));
        EXPECT_EQ(admitted->finished_request_ids, (std::vector<int64_t>{ready->streamId()}));
    }
}

TEST_F(PPSchedulerTest, RejectsUnsupportedRequestsBeforeKvAdmissionEvenAtCapacity) {
    struct RequestCase {
        RoleType role;
        size_t   capacity;
        bool     pd_separation;
    };
    const RequestCase cases[] = {
        {RoleType::PREFILL, 1, false},
        {RoleType::PREFILL, 0, false},
        {RoleType::DECODE, 1, false},
        {RoleType::DECODE, 0, true},
    };
    installReadinessAllocator([](const MallocInfo&) { return std::shared_ptr<AsyncContext>{}; });
    for (const auto& request : cases) {
        SCOPED_TRACE(testing::Message() << "role=" << static_cast<int>(request.role)
                                       << ", capacity=" << request.capacity
                                       << ", pd_separation=" << request.pd_separation);
        auto scheduler = createScheduler(request.capacity, request.role);
        auto stream    = createStream({1, 2, 3}, false, false, 2, {}, request.role);
        stream->generateConfig()->can_use_pd_separation = true;
        stream->generateConfig()->pd_separation         = request.pd_separation;
        ASSERT_TRUE(scheduler->enqueue(stream).ok());

        auto output = scheduler->schedule();
        ASSERT_TRUE(output.ok());
        EXPECT_TRUE(output->streams.empty());
        EXPECT_EQ(output->finished_request_ids, (std::vector<int64_t>{stream->streamId()}));
        EXPECT_EQ(stream->statusInfo().code(), ErrorCode::INVALID_PARAMS);
        EXPECT_FALSE(stream->hasEvent(StreamEvents::CanRun));
        EXPECT_EQ(initial_malloc_calls_, 0u);
        EXPECT_EQ(stream->curBlocksNum(), 0u);
        EXPECT_TRUE(stream->streamCacheResource().isResourceReleased());
        EXPECT_TRUE(scheduler->empty());
    }
}

TEST_F(PPSchedulerTest, DecodeRejectsContextStreams) {
    for (const bool fake : {false, true}) {
        for (const bool perf : {false, true}) {
            SCOPED_TRACE(testing::Message() << "fake=" << fake << ", perf=" << perf);
            auto scheduler = createScheduler(1, RoleType::DECODE, SP_TYPE_MTP);
            auto stream    = createStream({1, 2, 3}, false, false, 2, {}, RoleType::DECODE);
            stream->setIsFakeStream(fake);
            stream->setPerfTest(perf);
            ASSERT_TRUE(stream->initKVBlock().ok());
            stream->reportEvent(StreamEvents::LoadInitiated);
            ASSERT_TRUE(stream->isContextStream());
            ASSERT_TRUE(scheduler->enqueue(stream).ok());

            auto output = scheduler->schedule();
            ASSERT_TRUE(output.ok());
            EXPECT_TRUE(output->streams.empty());
            EXPECT_EQ(output->finished_request_ids, (std::vector<int64_t>{stream->streamId()}));
            EXPECT_EQ(stream->statusInfo().code(), ErrorCode::INVALID_PARAMS);
            EXPECT_EQ(stream->getStatus(), StreamState::FINISHED);
            EXPECT_FALSE(stream->isPPInflight());
            EXPECT_EQ(stream->curBlocksNum(), 0u);
            EXPECT_TRUE(scheduler->empty());
        }
    }
}

TEST_F(PPSchedulerTest, DecodeContinuesReadyStreamsWhenContextIsRejected) {
    auto scheduler = createScheduler(2, RoleType::DECODE, SP_TYPE_MTP);
    auto ready     = createStream({1, 2, 3}, false, false, 8, {}, RoleType::DECODE);
    ASSERT_TRUE(ready->initKVBlock().ok());
    ready->update({torch::tensor({{4}}, torch::kInt32), 1});
    ready->reportEvent(StreamEvents::LoadInitiated);
    ASSERT_FALSE(ready->isContextStream());
    ASSERT_TRUE(scheduler->enqueue(ready).ok());
    auto first = scheduler->schedule();
    ASSERT_TRUE(first.ok());
    ASSERT_EQ(first->streams, (std::list<GenerateStreamPtr>{ready}));
    ready->clearPPInflight();

    auto context = createStream({5, 6, 7}, false, false, 8, {}, RoleType::DECODE);
    context->setPerfTest(true);
    ASSERT_TRUE(context->initKVBlock().ok());
    context->reportEvent(StreamEvents::LoadInitiated);
    ASSERT_TRUE(context->isContextStream());
    ASSERT_TRUE(scheduler->enqueue(context).ok());

    auto output = scheduler->schedule();
    ASSERT_TRUE(output.ok());
    EXPECT_EQ(output->streams, (std::list<GenerateStreamPtr>{ready}));
    EXPECT_EQ(output->finished_request_ids, (std::vector<int64_t>{context->streamId()}));
    EXPECT_EQ(context->statusInfo().code(), ErrorCode::INVALID_PARAMS);
    EXPECT_EQ(context->getStatus(), StreamState::FINISHED);
    EXPECT_FALSE(context->isPPInflight());
    EXPECT_EQ(context->curBlocksNum(), 0u);
    EXPECT_FALSE(ready->hasError());
    EXPECT_EQ(ready->getStatus(), StreamState::RUNNING);
    EXPECT_TRUE(ready->isPPInflight());
    EXPECT_EQ(scheduler->waitingStreamsSize(), 0u);
}

TEST_F(PPSchedulerTest, PdDecodeInitializesCandidatesOnceAndKeepsTailProposal) {
    for (const auto type : {SP_TYPE_MTP, SP_TYPE_EAGLE, SP_TYPE_DSPARK}) {
        auto scheduler = createScheduler(1, RoleType::DECODE, type);
        auto stream    = createStream({1, 2, 3}, false, false, 8, {}, RoleType::DECODE);
        ASSERT_TRUE(stream->initKVBlock().ok());
        stream->update({torch::tensor({{4}}, torch::kInt32), 1});
        if (type != SP_TYPE_DSPARK) {
            stream->setProposeToken({4, 5});
            /** RPC already installed the anchor and draft slots before scheduler admission. */
            auto buffer          = std::make_shared<SpeculativeExecutorStreamOutput>();
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
        stream->specUpdate({torch::tensor({{5}}, torch::kInt32),
                            1,
                            torch::tensor({6, 7, 8}, torch::kInt32),
                            torch::Tensor(),
                            torch::Tensor()},
                           false);
        stream->clearPPInflight();
        auto next = scheduler->schedule();
        ASSERT_TRUE(next.ok());
        ASSERT_EQ(next->streams.size(), 1u);
        EXPECT_EQ(stream->getProposeToken(), (std::vector<int>{5, 6, 7, 8}));
    }
}

}  // namespace rtp_llm
