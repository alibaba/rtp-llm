#include <chrono>
#include <functional>
#include <future>
#include <memory>
#include <thread>
#include <unordered_map>

#include "torch/all.h"
#include "gmock/gmock-actions.h"
#include "gmock/gmock-function-mocker.h"
#include "gtest/gtest.h"

#define private public
#define protected public
#include "rtp_llm/cpp/engine_base/schedulers/FIFOScheduler.h"
#include "rtp_llm/cpp/engine_base/schedulers/PDFusionRatioScheduler.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/engine_base/stream/StreamCacheResource.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/cache/AsyncContext.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/block_tree_cache/load/LoadAsyncContext.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/cache/test/mock/MockKVCacheAllocator.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/models_py/bindings/core/Types.h"

using namespace std;

namespace rtp_llm {

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
    EXPECT_TRUE(coordinator->registerContext(context));
    return context;
}

class FIFOSchedulerAsyncCacheTest: public DeviceTestBase {
protected:
    using ContextSelector = std::function<std::shared_ptr<AsyncContext>(const MallocInfo&)>;

    FIFOSchedulerAsyncCacheTest(): perf_scope("PERF_TEST", "1") {}

    void SetUp() override {
        DeviceTestBase::SetUp();
        cache_config_ = test::makeSimpleMhaCacheConfig(
            /*layer_num=*/1, /*block_num=*/21, /*tokens_per_block=*/2, rtp_llm::DataType::TYPE_INT8);
        cache_manager_ = std::make_shared<KVCacheManager>(cache_config_);
        ASSERT_TRUE(cache_manager_->init());
    }

    void TearDown() override {
        if (real_allocator_) {
            cache_manager_->allocator_ = real_allocator_;
        }
        DeviceTestBase::TearDown();
    }

    std::shared_ptr<FIFOScheduler> createScheduler(size_t max_generate_batch_size     = 100,
                                                   size_t max_inited_kv_cache_streams = 0) {
        ModelConfig model_config;
        model_config.max_seq_len = 8192;
        RuntimeConfig runtime_config;
        runtime_config.max_generate_batch_size                           = max_generate_batch_size;
        runtime_config.fifo_scheduler_config.max_batch_tokens_size       = 8192;
        runtime_config.fifo_scheduler_config.max_inited_kv_cache_streams = max_inited_kv_cache_streams;
        PDSepConfig         pd_sep_config;
        ParallelismConfig   parallelism_config;
        ModelSpecificConfig model_specific_config;
        return std::make_shared<FIFOScheduler>(
            runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager_);
    }

    std::shared_ptr<PDFusionRatioScheduler> createPDFusionRatioScheduler(size_t max_generate_batch_size = 100) {
        ModelConfig model_config;
        model_config.max_seq_len = 8192;
        RuntimeConfig runtime_config;
        runtime_config.max_generate_batch_size                     = max_generate_batch_size;
        runtime_config.fifo_scheduler_config.max_batch_tokens_size = 8192;
        PDSepConfig pd_sep_config;
        pd_sep_config.role_type = RoleType::PDFUSION;
        ParallelismConfig   parallelism_config;
        ModelSpecificConfig model_specific_config;
        return std::make_shared<PDFusionRatioScheduler>(
            runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager_);
    }

    GenerateStreamPtr createStream(const std::vector<int>& input_tokens       = {1, 2, 3},
                                   bool                    reuse_cache        = false,
                                   bool                    enable_host_cache  = false,
                                   int                     max_new_tokens     = 1,
                                   const std::vector<int>& variable_num_beams = {},
                                   RoleType                role_type          = RoleType::PDFUSION) {
        ResourceContext resource_context;
        resource_context.cache_manager     = cache_manager_;
        resource_context.reuse_cache       = reuse_cache;
        resource_context.enable_host_cache = enable_host_cache;
        resource_context.role_type         = role_type;

        ModelConfig model_config;
        model_config.max_seq_len = 8192;
        RuntimeConfig runtime_config;

        auto query                          = std::make_shared<GenerateInput>();
        auto generate_config                = std::make_shared<GenerateConfig>();
        query->request_id                   = next_request_id_++;
        generate_config->reuse_cache        = reuse_cache;
        generate_config->enable_host_cache  = enable_host_cache;
        generate_config->max_new_tokens     = max_new_tokens;
        generate_config->variable_num_beams = variable_num_beams;
        query->input_ids                    = torch::tensor(input_tokens, torch::kInt32);
        query->generate_config              = generate_config;
        return std::make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    }

    void installReadinessAllocator(ContextSelector selector) {
        real_allocator_       = cache_manager_->allocator_;
        mock_allocator_       = std::make_shared<testing::NiceMock<MockKVCacheAllocator>>(cache_manager_->config_);
        initial_malloc_calls_ = 0;
        free_calls_           = 0;
        insert_calls_         = 0;

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
            ++free_calls_;
            real_allocator_->free(info);
        }));
        ON_CALL(*mock_allocator_, insertIntoCache(testing::_))
            .WillByDefault(testing::Invoke([this](const InsertInfo& info) {
                ++insert_calls_;
                real_allocator_->insertIntoCache(info);
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

        cache_manager_->allocator_ = mock_allocator_;
    }

    void installRetryableAllocator() {
        real_allocator_       = cache_manager_->allocator_;
        mock_allocator_       = std::make_shared<testing::NiceMock<MockKVCacheAllocator>>(cache_manager_->config_);
        initial_malloc_calls_ = 0;

        ON_CALL(*mock_allocator_, totalBlocksNum()).WillByDefault(testing::Return(64));
        ON_CALL(*mock_allocator_, getNeedBlocks(testing::_)).WillByDefault(testing::Return(1));
        ON_CALL(*mock_allocator_, initMallocForCommonLen(testing::_))
            .WillByDefault(testing::Invoke([this](const MallocInfo&) {
                ++initial_malloc_calls_;
                return MallocResult{false, 0, 0, MallocStatus::RETRYABLE_RESOURCE_EXHAUSTED};
            }));

        cache_manager_->allocator_ = mock_allocator_;
    }

protected:
    autil::EnvGuard                                          perf_scope;
    CacheConfig                                              cache_config_;
    std::shared_ptr<KVCacheManager>                          cache_manager_;
    int64_t                                                  next_request_id_{1};
    KVCacheAllocatorPtr                                      real_allocator_;
    std::shared_ptr<testing::NiceMock<MockKVCacheAllocator>> mock_allocator_;
    size_t                                                   initial_malloc_calls_{0};
    size_t                                                   free_calls_{0};
    size_t                                                   insert_calls_{0};
};

TEST_F(FIFOSchedulerAsyncCacheTest, testScheduleNew_NoReuseCache_DirectlyRunning) {
    auto scheduler = createScheduler();
    auto stream    = createStream({1, 2, 3}, /*reuse_cache=*/false);

    ASSERT_TRUE(scheduler->enqueue(stream).ok());
    auto result = scheduler->schedule();

    ASSERT_TRUE(result.ok());
    ASSERT_EQ(result.value().size(), 1);
    EXPECT_EQ(result.value().front(), stream);
    EXPECT_EQ(scheduler->loading_cache_streams_.size(), 0);
    EXPECT_EQ(scheduler->waitingStreamsSize(), 0);
    EXPECT_EQ(scheduler->runningStreamsSize(), 1);
}

TEST_F(FIFOSchedulerAsyncCacheTest, testScheduleNew_WithAllocatorReadiness_EntersLoadingCache) {
    auto scheduler = createScheduler();
    auto stream    = createStream({1, 2, 3}, /*reuse_cache=*/true);
    auto context   = makeControlledAllocatorContext();
    ASSERT_TRUE(scheduler->enqueue(stream).ok());
    installReadinessAllocator([context](const MallocInfo&) { return context; });

    auto result = scheduler->schedule();

    ASSERT_TRUE(result.ok());
    EXPECT_TRUE(result.value().empty());
    EXPECT_EQ(stream->getStatus(), StreamState::LOADING_CACHE);
    EXPECT_EQ(stream->streamCacheResource().allocator_load_context_, context);
    EXPECT_EQ(scheduler->loading_cache_streams_.size(), 1);
    EXPECT_EQ(scheduler->waitingStreamsSize(), 0);
    EXPECT_EQ(scheduler->runningStreamsSize(), 0);
}

TEST_F(FIFOSchedulerAsyncCacheTest, testEvaluateLoadingCache_AllocatorSuccess_MovesToRunning) {
    auto scheduler = createScheduler();
    auto stream    = createStream({1, 2, 3}, /*reuse_cache=*/true);
    auto context   = makeControlledAllocatorContext();
    ASSERT_TRUE(scheduler->enqueue(stream).ok());
    installReadinessAllocator([context](const MallocInfo&) { return context; });

    auto first = scheduler->schedule();
    ASSERT_TRUE(first.ok());
    ASSERT_EQ(stream->getStatus(), StreamState::LOADING_CACHE);
    EXPECT_TRUE(context->completeOne(true));

    auto second = scheduler->schedule();
    ASSERT_TRUE(second.ok());
    ASSERT_EQ(second.value().size(), 1);
    EXPECT_EQ(second.value().front(), stream);
    EXPECT_EQ(stream->getStatus(), StreamState::RUNNING);
    EXPECT_EQ(stream->streamCacheResource().allocator_load_context_, nullptr);
    EXPECT_EQ(scheduler->loading_cache_streams_.size(), 0);
    EXPECT_EQ(scheduler->waitingStreamsSize(), 0);
    EXPECT_EQ(scheduler->runningStreamsSize(), 1);
    EXPECT_FALSE(stream->hasError());
}

TEST_F(FIFOSchedulerAsyncCacheTest, testEvaluateLoadingCache_AllocatorFailure_Evicted) {
    auto scheduler = createScheduler();
    auto stream    = createStream({1, 2, 3}, /*reuse_cache=*/true);
    auto context   = makeControlledAllocatorContext();
    ASSERT_TRUE(scheduler->enqueue(stream).ok());
    installReadinessAllocator([context](const MallocInfo&) { return context; });

    auto first = scheduler->schedule();
    ASSERT_TRUE(first.ok());
    ASSERT_GT(stream->curBlocksNum(), 0);
    EXPECT_TRUE(context->completeOne(false));

    auto second = scheduler->schedule();
    ASSERT_TRUE(second.ok());
    EXPECT_TRUE(second.value().empty());
    EXPECT_TRUE(stream->isFinished());
    EXPECT_TRUE(stream->hasError());
    EXPECT_TRUE(stream->streamCacheResource().isResourceReleased());
    EXPECT_EQ(stream->curBlocksNum(), 0);
    EXPECT_EQ(stream->streamCacheResource().allocator_load_context_, nullptr);
    EXPECT_EQ(scheduler->loading_cache_streams_.size(), 0);
    EXPECT_EQ(scheduler->waitingStreamsSize(), 0);
    EXPECT_EQ(scheduler->runningStreamsSize(), 0);
    EXPECT_EQ(insert_calls_, 0);
    EXPECT_EQ(free_calls_, 1);
}

TEST_F(FIFOSchedulerAsyncCacheTest, testEvaluateLoadingCache_PrefillAllocatorFailure_MovesToRunning) {
    auto scheduler = createScheduler();
    auto stream    = createStream({1, 2, 3},
                               /*reuse_cache=*/true,
                               /*enable_host_cache=*/false,
                               /*max_new_tokens=*/1,
                               /*variable_num_beams=*/{},
                               RoleType::PREFILL);
    auto context   = makeControlledAllocatorContext();
    ASSERT_TRUE(scheduler->enqueue(stream).ok());
    installReadinessAllocator([context](const MallocInfo&) { return context; });

    auto first = scheduler->schedule();
    ASSERT_TRUE(first.ok());
    ASSERT_EQ(stream->getStatus(), StreamState::LOADING_CACHE);
    EXPECT_TRUE(context->completeOne(false));

    auto second = scheduler->schedule();
    ASSERT_TRUE(second.ok());
    ASSERT_EQ(second.value().size(), 1);
    EXPECT_EQ(second.value().front(), stream);
    EXPECT_EQ(stream->getStatus(), StreamState::RUNNING);
    EXPECT_FALSE(stream->hasError());
    EXPECT_FALSE(stream->streamCacheResource().isResourceReleased());
    EXPECT_GT(stream->curBlocksNum(), 0);
    EXPECT_EQ(stream->streamCacheResource().allocator_load_context_, nullptr);
    EXPECT_EQ(scheduler->loading_cache_streams_.size(), 0);
    EXPECT_EQ(scheduler->waitingStreamsSize(), 0);
    EXPECT_EQ(scheduler->runningStreamsSize(), 1);
    EXPECT_EQ(free_calls_, 0);
}

TEST_F(FIFOSchedulerAsyncCacheTest, testPendingAllocatorLoads_RespectAdmissionBatchLimit) {
    auto scheduler = createScheduler(/*max_generate_batch_size=*/2);
    auto stream1   = createStream({1}, /*reuse_cache=*/true);
    auto stream2   = createStream({2}, /*reuse_cache=*/true);
    auto stream3   = createStream({3}, /*reuse_cache=*/true);
    auto context1  = makeControlledAllocatorContext();
    auto context2  = makeControlledAllocatorContext();
    auto context3  = makeControlledAllocatorContext();
    ASSERT_TRUE(scheduler->enqueue(stream1).ok());
    ASSERT_TRUE(scheduler->enqueue(stream2).ok());
    ASSERT_TRUE(scheduler->enqueue(stream3).ok());
    const std::unordered_map<int64_t, std::shared_ptr<AsyncContext>> contexts{
        {stream1->streamId(), context1}, {stream2->streamId(), context2}, {stream3->streamId(), context3}};
    installReadinessAllocator([contexts](const MallocInfo& info) { return contexts.at(info.request_id); });

    auto result = scheduler->schedule();

    ASSERT_TRUE(result.ok());
    EXPECT_TRUE(result.value().empty());
    EXPECT_EQ(scheduler->loading_cache_streams_.size(), 2);
    EXPECT_EQ(scheduler->waitingStreamsSize(), 1);
    EXPECT_EQ(scheduler->runningStreamsSize(), 0);
    EXPECT_EQ(stream1->getStatus(), StreamState::LOADING_CACHE);
    EXPECT_EQ(stream2->getStatus(), StreamState::LOADING_CACHE);
    EXPECT_EQ(stream3->getStatus(), StreamState::WAITING);
    EXPECT_EQ(initial_malloc_calls_, 2);
}

TEST_F(FIFOSchedulerAsyncCacheTest, testScheduleNew_ReturningFromLoadingCache_SkipsDuplicateInitialReadiness) {
    auto scheduler = createScheduler();
    auto stream    = createStream({1, 2, 3}, /*reuse_cache=*/true);
    auto context   = makeControlledAllocatorContext();
    ASSERT_TRUE(scheduler->enqueue(stream).ok());
    installReadinessAllocator([context](const MallocInfo&) { return context; });

    auto first = scheduler->schedule();
    ASSERT_TRUE(first.ok());
    ASSERT_EQ(stream->getStatus(), StreamState::LOADING_CACHE);
    EXPECT_EQ(initial_malloc_calls_, 1);
    EXPECT_TRUE(context->completeOne(true));

    auto second = scheduler->schedule();
    ASSERT_TRUE(second.ok());
    ASSERT_EQ(second.value().size(), 1);
    EXPECT_EQ(second.value().front(), stream);
    EXPECT_EQ(stream->getStatus(), StreamState::RUNNING);
    EXPECT_EQ(initial_malloc_calls_, 1);
}

TEST_F(FIFOSchedulerAsyncCacheTest, testLoadingCacheStreams_IncludedInEmptyAndOnflight) {
    auto scheduler = createScheduler();
    auto stream    = createStream({1, 2, 3}, /*reuse_cache=*/true);
    auto context   = makeControlledAllocatorContext();
    ASSERT_TRUE(scheduler->enqueue(stream).ok());
    installReadinessAllocator([context](const MallocInfo&) { return context; });

    auto result = scheduler->schedule();

    ASSERT_TRUE(result.ok());
    ASSERT_EQ(scheduler->loading_cache_streams_.size(), 1);
    EXPECT_FALSE(scheduler->empty());
    EXPECT_EQ(scheduler->onflightStreams(), 1);
    EXPECT_EQ(scheduler->waitingStreamsSize(), 0);
    EXPECT_EQ(scheduler->runningStreamsSize(), 0);
    EXPECT_EQ(stream->streamCacheResource().allocator_load_context_, context);
    EXPECT_FALSE(context->done());
}

TEST_F(FIFOSchedulerAsyncCacheTest, testWaitPredicate_DoesNotSpinForLoadingCacheStreams) {
    auto scheduler = createScheduler();
    auto stream    = createStream({1, 2, 3}, /*reuse_cache=*/true);
    auto context   = makeControlledAllocatorContext();
    ASSERT_TRUE(scheduler->enqueue(stream).ok());
    installReadinessAllocator([context](const MallocInfo&) { return context; });

    auto first = scheduler->schedule();
    ASSERT_TRUE(first.ok());
    ASSERT_EQ(stream->getStatus(), StreamState::LOADING_CACHE);

    // Consume the admission-triggered round while the allocator context remains pending.
    auto second = scheduler->schedule();
    ASSERT_TRUE(second.ok());
    ASSERT_EQ(stream->getStatus(), StreamState::LOADING_CACHE);
    EXPECT_FALSE(scheduler->waitPredicate());
}

TEST_F(FIFOSchedulerAsyncCacheTest, testRetryableKVShortageDoesNotSpinAndIsPolled) {
    auto scheduler = createScheduler();
    auto stream    = createStream({1, 2, 3},
                               /*reuse_cache=*/true,
                               /*enable_host_cache=*/false,
                               /*max_new_tokens=*/1,
                               /*variable_num_beams=*/{},
                               RoleType::PREFILL);
    ASSERT_TRUE(scheduler->enqueue(stream).ok());
    installRetryableAllocator();

    auto first = scheduler->schedule();
    ASSERT_TRUE(first.ok());
    EXPECT_TRUE(first.value().empty());
    EXPECT_EQ(stream->getStatus(), StreamState::WAITING);
    EXPECT_EQ(initial_malloc_calls_, 1u);
    EXPECT_FALSE(scheduler->waitPredicate());

    std::promise<absl::StatusOr<std::list<GenerateStreamPtr>>> promise;
    auto                                                       future = promise.get_future();
    std::thread schedule_thread([&] { promise.set_value(scheduler->schedule()); });

    EXPECT_EQ(future.wait_for(std::chrono::milliseconds(2)), std::future_status::timeout);
    const auto wait_status = future.wait_for(std::chrono::milliseconds(250));
    if (wait_status != std::future_status::ready) {
        EXPECT_TRUE(scheduler->stop().ok());
    }
    schedule_thread.join();

    ASSERT_EQ(wait_status, std::future_status::ready);
    auto second = future.get();
    ASSERT_TRUE(second.ok());
    EXPECT_TRUE(second.value().empty());
    EXPECT_EQ(stream->getStatus(), StreamState::WAITING);
    EXPECT_EQ(initial_malloc_calls_, 2u);
}

TEST_F(FIFOSchedulerAsyncCacheTest, testEvictDoneStreams_HandlesExternalError) {
    auto scheduler = createScheduler();
    auto stream    = createStream({1, 2, 3});

    ASSERT_TRUE(scheduler->enqueue(stream).ok());
    auto first = scheduler->schedule();
    ASSERT_TRUE(first.ok());
    ASSERT_EQ(first.value().size(), 1);

    stream->reportError(ErrorCode::CANCELLED, "cancelled by RPC");
    auto second = scheduler->schedule();

    ASSERT_TRUE(second.ok());
    EXPECT_TRUE(second.value().empty());
    EXPECT_TRUE(stream->isFinished());
    EXPECT_EQ(scheduler->runningStreamsSize(), 0);
}

TEST_F(FIFOSchedulerAsyncCacheTest, testMixedAllocatorReadinessAndDirectStreams) {
    auto scheduler      = createScheduler();
    auto loading_stream = createStream({1, 2}, /*reuse_cache=*/true);
    auto direct_stream  = createStream({3, 4}, /*reuse_cache=*/false);
    auto context        = makeControlledAllocatorContext();
    ASSERT_TRUE(scheduler->enqueue(loading_stream).ok());
    ASSERT_TRUE(scheduler->enqueue(direct_stream).ok());
    installReadinessAllocator([context, loading_stream](const MallocInfo& info) -> std::shared_ptr<AsyncContext> {
        return info.request_id == loading_stream->streamId() ? context : nullptr;
    });

    auto result = scheduler->schedule();

    ASSERT_TRUE(result.ok());
    ASSERT_EQ(result.value().size(), 1);
    EXPECT_EQ(result.value().front(), direct_stream);
    EXPECT_EQ(loading_stream->getStatus(), StreamState::LOADING_CACHE);
    EXPECT_EQ(direct_stream->getStatus(), StreamState::RUNNING);
    EXPECT_EQ(scheduler->loading_cache_streams_.size(), 1);
    EXPECT_EQ(scheduler->waitingStreamsSize(), 0);
    EXPECT_EQ(scheduler->runningStreamsSize(), 1);
    EXPECT_EQ(scheduler->onflightStreams(), 2);
}

TEST_F(FIFOSchedulerAsyncCacheTest, testEvaluateLoadingCache_AllocatorPending_StaysInQueue) {
    auto scheduler = createScheduler();
    auto stream    = createStream({1, 2, 3}, /*reuse_cache=*/true);
    auto context   = makeControlledAllocatorContext();
    ASSERT_TRUE(scheduler->enqueue(stream).ok());
    installReadinessAllocator([context](const MallocInfo&) { return context; });

    auto first = scheduler->schedule();
    ASSERT_TRUE(first.ok());
    ASSERT_EQ(stream->getStatus(), StreamState::LOADING_CACHE);
    auto second = scheduler->schedule();

    ASSERT_TRUE(second.ok());
    EXPECT_TRUE(second.value().empty());
    EXPECT_EQ(stream->getStatus(), StreamState::LOADING_CACHE);
    EXPECT_EQ(stream->streamCacheResource().allocator_load_context_, context);
    EXPECT_EQ(scheduler->loading_cache_streams_.size(), 1);
    EXPECT_EQ(scheduler->waitingStreamsSize(), 0);
    EXPECT_EQ(scheduler->runningStreamsSize(), 0);
    EXPECT_FALSE(stream->hasError());
    EXPECT_EQ(free_calls_, 0);
}

TEST_F(FIFOSchedulerAsyncCacheTest, testScheduleOrdering_LoadDoneRejoinsWaitingTail) {
    auto scheduler        = createScheduler(/*max_generate_batch_size=*/1);
    auto completed_stream = createStream({1, 2}, /*reuse_cache=*/true);
    auto older_waiter     = createStream({3, 4}, /*reuse_cache=*/false);
    auto context          = makeControlledAllocatorContext();
    ASSERT_TRUE(scheduler->enqueue(completed_stream).ok());
    ASSERT_TRUE(scheduler->enqueue(older_waiter).ok());
    installReadinessAllocator([context, completed_stream](const MallocInfo& info) {
        return info.request_id == completed_stream->streamId() ? context : nullptr;
    });

    auto first = scheduler->schedule();
    ASSERT_TRUE(first.ok());
    ASSERT_EQ(completed_stream->getStatus(), StreamState::LOADING_CACHE);
    EXPECT_TRUE(context->completeOne(true));

    auto second = scheduler->schedule();
    ASSERT_TRUE(second.ok());
    ASSERT_EQ(second.value().size(), 1);
    EXPECT_EQ(second.value().front(), older_waiter);
    EXPECT_EQ(older_waiter->getStatus(), StreamState::RUNNING);
    EXPECT_EQ(completed_stream->getStatus(), StreamState::WAITING);
    EXPECT_EQ(completed_stream->streamCacheResource().allocator_load_context_, nullptr);
    ASSERT_EQ(scheduler->waiting_streams_.size(), 1);
    EXPECT_EQ(scheduler->waiting_streams_.front(), completed_stream);
    EXPECT_EQ(scheduler->loading_cache_streams_.size(), 0);
    EXPECT_EQ(scheduler->runningStreamsSize(), 1);
}

TEST_F(FIFOSchedulerAsyncCacheTest, testLoadingGroupDoesNotBlockOrdinaryWaitingStreams) {
    auto scheduler      = createScheduler();
    auto direct_stream  = createStream({1, 2}, /*reuse_cache=*/false);
    auto loading_stream = createStream({3, 4}, /*reuse_cache=*/true);
    auto waiting_stream = createStream({5, 6}, /*reuse_cache=*/false);
    auto context        = makeControlledAllocatorContext();

    auto [enqueue_successes, streams] = scheduler->enqueueGroup({direct_stream, loading_stream});
    ASSERT_EQ(enqueue_successes, std::vector<bool>({true, true}));
    ASSERT_EQ(streams.size(), 2);
    installReadinessAllocator([context, loading_stream](const MallocInfo& info) {
        return info.request_id == loading_stream->streamId() ? context : nullptr;
    });

    auto group_result = scheduler->schedule();
    ASSERT_TRUE(group_result.ok());
    EXPECT_TRUE(group_result.value().empty());
    EXPECT_EQ(direct_stream->getStatus(), StreamState::RUNNING);
    EXPECT_EQ(loading_stream->getStatus(), StreamState::LOADING_CACHE);
    ASSERT_EQ(scheduler->loading_cache_group_queue_.size(), 1);
    EXPECT_EQ(scheduler->loading_cache_group_queue_.front().size(), 2);

    ASSERT_TRUE(scheduler->enqueue(waiting_stream).ok());
    auto waiting_result = scheduler->schedule();
    ASSERT_TRUE(waiting_result.ok());
    ASSERT_EQ(waiting_result.value().size(), 1);
    EXPECT_EQ(waiting_stream->getStatus(), StreamState::RUNNING);
    ASSERT_EQ(scheduler->loading_cache_group_queue_.size(), 1);
    EXPECT_EQ(scheduler->waitingStreamsSize(), 0);
}

TEST_F(FIFOSchedulerAsyncCacheTest, testOrdinaryCacheLoadDoesNotStarveWaitingGroup) {
    auto scheduler      = createScheduler();
    auto loading_stream = createStream({1, 2}, /*reuse_cache=*/true);
    auto context        = makeControlledAllocatorContext();

    ASSERT_TRUE(scheduler->enqueue(loading_stream).ok());
    installReadinessAllocator([context, loading_stream](const MallocInfo& info) {
        return info.request_id == loading_stream->streamId() ? context : nullptr;
    });
    ASSERT_TRUE(scheduler->schedule().ok());
    ASSERT_EQ(scheduler->loading_cache_streams_.size(), 1);

    auto group_stream_1 = createStream({3, 4}, /*reuse_cache=*/false);
    auto group_stream_2 = createStream({5, 6}, /*reuse_cache=*/false);
    ASSERT_EQ(scheduler->enqueueGroup({group_stream_1, group_stream_2}).first, std::vector<bool>({true, true}));

    ASSERT_TRUE(context->completeOne(true));
    auto result = scheduler->schedule();
    ASSERT_TRUE(result.ok());
    ASSERT_EQ(result.value().size(), 1);
    EXPECT_EQ(result.value().front(), loading_stream);
    EXPECT_EQ(scheduler->waiting_group_queue_.size(), 1);
    EXPECT_TRUE(scheduler->loading_cache_group_queue_.empty());
    EXPECT_EQ(group_stream_1->getStatus(), StreamState::WAITING);
    EXPECT_EQ(group_stream_2->getStatus(), StreamState::WAITING);
    EXPECT_EQ(loading_stream->getStatus(), StreamState::RUNNING);

    loading_stream->reportEvent(StreamEvents::GenerateDone);
    auto group_result = scheduler->schedule();
    ASSERT_TRUE(group_result.ok());
    ASSERT_EQ(group_result.value().size(), 2);
    EXPECT_EQ(group_stream_1->getStatus(), StreamState::RUNNING);
    EXPECT_EQ(group_stream_2->getStatus(), StreamState::RUNNING);
}

TEST_F(FIFOSchedulerAsyncCacheTest, testPreparedGroupFinishesLoadingInOneRound) {
    auto scheduler      = createScheduler();
    auto direct_stream  = createStream({1, 2}, /*reuse_cache=*/false);
    auto loading_stream = createStream({3, 4}, /*reuse_cache=*/true);
    auto waiting_stream = createStream({5, 6}, /*reuse_cache=*/false);
    auto context        = makeControlledAllocatorContext();
    ASSERT_TRUE(context->completeOne(true));

    ASSERT_EQ(scheduler->enqueueGroup({direct_stream, loading_stream}).first, std::vector<bool>({true, true}));
    installReadinessAllocator([context, loading_stream](const MallocInfo& info) {
        return info.request_id == loading_stream->streamId() ? context : nullptr;
    });
    auto first_result = scheduler->schedule();
    ASSERT_TRUE(first_result.ok());
    EXPECT_TRUE(first_result.value().empty());
    EXPECT_EQ(scheduler->loading_cache_group_queue_.size(), 1);

    ASSERT_TRUE(scheduler->enqueue(waiting_stream).ok());
    auto second_result = scheduler->schedule();
    ASSERT_TRUE(second_result.ok());
    EXPECT_EQ(second_result.value().size(), 2);
    EXPECT_EQ(direct_stream->getStatus(), StreamState::RUNNING);
    EXPECT_EQ(loading_stream->getStatus(), StreamState::RUNNING);
    EXPECT_EQ(waiting_stream->getStatus(), StreamState::WAITING);
    EXPECT_TRUE(scheduler->loading_cache_group_queue_.empty());
    EXPECT_EQ(scheduler->waitingStreamsSize(), 1);

    direct_stream->reportEvent(StreamEvents::GenerateDone);
    loading_stream->reportEvent(StreamEvents::GenerateDone);
    auto third_result = scheduler->schedule();
    ASSERT_TRUE(third_result.ok());
    ASSERT_EQ(third_result.value().size(), 1);
    EXPECT_EQ(waiting_stream->getStatus(), StreamState::RUNNING);
    EXPECT_EQ(scheduler->waitingStreamsSize(), 0);
}

TEST_F(FIFOSchedulerAsyncCacheTest, testReadyLoadingGroupDrainsNormalLaneBeforeDispatch) {
    auto scheduler      = createScheduler();
    auto direct_stream  = createStream({1, 2}, /*reuse_cache=*/false);
    auto loading_stream = createStream({3, 4}, /*reuse_cache=*/true);
    auto normal_stream  = createStream({5, 6}, /*reuse_cache=*/false);
    auto normal_tail    = createStream({7, 8}, /*reuse_cache=*/false);
    auto context        = makeControlledAllocatorContext();

    ASSERT_EQ(scheduler->enqueueGroup({direct_stream, loading_stream}).first, std::vector<bool>({true, true}));
    installReadinessAllocator([context, loading_stream](const MallocInfo& info) {
        return info.request_id == loading_stream->streamId() ? context : nullptr;
    });
    ASSERT_TRUE(scheduler->schedule().ok());
    ASSERT_EQ(scheduler->loading_cache_group_queue_.size(), 1);

    ASSERT_TRUE(scheduler->enqueue(normal_stream).ok());
    auto normal_result = scheduler->schedule();
    ASSERT_TRUE(normal_result.ok());
    ASSERT_EQ(normal_result.value().size(), 1);
    EXPECT_EQ(normal_stream->getStatus(), StreamState::RUNNING);

    ASSERT_TRUE(context->completeOne(true));
    ASSERT_TRUE(scheduler->enqueue(normal_tail).ok());
    auto drain_result = scheduler->schedule();
    ASSERT_TRUE(drain_result.ok());
    ASSERT_EQ(drain_result.value().size(), 1);
    EXPECT_EQ(normal_tail->getStatus(), StreamState::WAITING);
    ASSERT_EQ(scheduler->loading_cache_group_queue_.size(), 1);

    normal_stream->reportEvent(StreamEvents::GenerateDone);
    auto group_result = scheduler->schedule();
    ASSERT_TRUE(group_result.ok());
    ASSERT_EQ(group_result.value().size(), 2);
    EXPECT_EQ(direct_stream->getStatus(), StreamState::RUNNING);
    EXPECT_EQ(loading_stream->getStatus(), StreamState::RUNNING);
    EXPECT_EQ(normal_tail->getStatus(), StreamState::WAITING);
}

TEST_F(FIFOSchedulerAsyncCacheTest, testOrdinaryCacheCompletionDoesNotMixIntoRunningGroup) {
    auto scheduler        = createScheduler();
    auto ordinary_loading = createStream({1, 2}, /*reuse_cache=*/true);
    auto context          = makeControlledAllocatorContext();

    ASSERT_TRUE(scheduler->enqueue(ordinary_loading).ok());
    installReadinessAllocator([context, ordinary_loading](const MallocInfo& info) {
        return info.request_id == ordinary_loading->streamId() ? context : nullptr;
    });
    ASSERT_TRUE(scheduler->schedule().ok());
    ASSERT_EQ(ordinary_loading->getStatus(), StreamState::LOADING_CACHE);

    auto group_stream_1 = createStream({3, 4}, /*reuse_cache=*/false);
    auto group_stream_2 = createStream({5, 6}, /*reuse_cache=*/false);
    ASSERT_EQ(scheduler->enqueueGroup({group_stream_1, group_stream_2}).first, std::vector<bool>({true, true}));
    auto group_result = scheduler->schedule();
    ASSERT_TRUE(group_result.ok());
    ASSERT_EQ(group_result.value().size(), 2);

    ASSERT_TRUE(context->completeOne(true));
    auto isolated_result = scheduler->schedule();
    ASSERT_TRUE(isolated_result.ok());
    ASSERT_EQ(isolated_result.value().size(), 2);
    EXPECT_EQ(ordinary_loading->getStatus(), StreamState::LOADING_CACHE);
    EXPECT_EQ(scheduler->loading_cache_streams_.size(), 1);

    group_stream_1->reportEvent(StreamEvents::GenerateDone);
    group_stream_2->reportEvent(StreamEvents::GenerateDone);
    auto ordinary_result = scheduler->schedule();
    ASSERT_TRUE(ordinary_result.ok());
    ASSERT_EQ(ordinary_result.value().size(), 1);
    EXPECT_EQ(ordinary_loading->getStatus(), StreamState::RUNNING);
}

TEST_F(FIFOSchedulerAsyncCacheTest, testOrdinaryLoadingReleasesInitedLimitBeforeWaitingGroup) {
    auto scheduler        = createScheduler(/*max_generate_batch_size=*/100, /*max_inited_kv_cache_streams=*/1);
    auto ordinary_loading = createStream({1, 2}, /*reuse_cache=*/true);
    auto context          = makeControlledAllocatorContext();

    ASSERT_TRUE(scheduler->enqueue(ordinary_loading).ok());
    installReadinessAllocator([context, ordinary_loading](const MallocInfo& info) {
        return info.request_id == ordinary_loading->streamId() ? context : nullptr;
    });
    ASSERT_TRUE(scheduler->schedule().ok());
    ASSERT_EQ(ordinary_loading->getStatus(), StreamState::LOADING_CACHE);
    ASSERT_EQ(scheduler->countInitedKVCacheStreams(), 1);

    auto group_stream = createStream({3, 4}, /*reuse_cache=*/false);
    ASSERT_EQ(scheduler->enqueueGroup({group_stream}).first, std::vector<bool>({true}));

    ASSERT_TRUE(context->completeOne(true));
    auto ordinary_result = scheduler->schedule();
    ASSERT_TRUE(ordinary_result.ok());
    ASSERT_EQ(ordinary_result.value().size(), 1);
    EXPECT_EQ(ordinary_loading->getStatus(), StreamState::RUNNING);
    EXPECT_EQ(group_stream->getStatus(), StreamState::WAITING);

    ordinary_loading->reportEvent(StreamEvents::GenerateDone);
    auto group_result = scheduler->schedule();
    ASSERT_TRUE(group_result.ok());
    ASSERT_EQ(group_result.value().size(), 1);
    EXPECT_EQ(group_stream->getStatus(), StreamState::RUNNING);
}

TEST_F(FIFOSchedulerAsyncCacheTest, testGroupedSurvivorContinuesLoadingAfterPeerTimeout) {
    auto scheduler                              = createScheduler();
    auto direct_stream                          = createStream({1, 2}, /*reuse_cache=*/false);
    auto loading_stream                         = createStream({3, 4}, /*reuse_cache=*/true);
    auto context                                = makeControlledAllocatorContext();
    direct_stream->generateConfig()->timeout_ms = 1;
    direct_stream->resetBeginTime(autil::TimeUtility::currentTimeInMicroSeconds());

    auto [enqueue_successes, streams] = scheduler->enqueueGroup({direct_stream, loading_stream});
    ASSERT_EQ(enqueue_successes, std::vector<bool>({true, true}));
    ASSERT_EQ(streams.size(), 2);
    installReadinessAllocator([context, loading_stream](const MallocInfo& info) {
        return info.request_id == loading_stream->streamId() ? context : nullptr;
    });

    auto first_result = scheduler->schedule();
    ASSERT_TRUE(first_result.ok());
    EXPECT_TRUE(first_result.value().empty());
    ASSERT_EQ(direct_stream->getStatus(), StreamState::RUNNING);
    ASSERT_EQ(loading_stream->getStatus(), StreamState::LOADING_CACHE);

    std::this_thread::sleep_for(std::chrono::milliseconds(3));
    auto second_result = scheduler->schedule();
    ASSERT_TRUE(second_result.ok());
    EXPECT_TRUE(second_result.value().empty());
    ASSERT_EQ(direct_stream->getStatus(), StreamState::FINISHED);
    ASSERT_EQ(direct_stream->statusInfo().code(), ErrorCode::GENERATE_TIMEOUT);
    ASSERT_EQ(loading_stream->getStatus(), StreamState::LOADING_CACHE);
    ASSERT_EQ(scheduler->loading_cache_group_queue_.size(), 1);
    ASSERT_EQ(scheduler->loading_cache_group_queue_.front().size(), 1);
    ASSERT_TRUE(scheduler->loading_cache_streams_.empty());
    ASSERT_EQ(scheduler->waitingStreamsSize(), 0);

    ASSERT_TRUE(context->completeOne(true));
    auto final_result = scheduler->schedule();
    ASSERT_TRUE(final_result.ok());
    ASSERT_EQ(final_result.value().size(), 1);
    EXPECT_EQ(loading_stream->getStatus(), StreamState::RUNNING);
    EXPECT_TRUE(scheduler->loading_cache_group_queue_.empty());
}

TEST_F(FIFOSchedulerAsyncCacheTest, testLoadingGroupCompletionIsPolledWithoutExternalWakeup) {
    auto scheduler = createScheduler();
    auto stream    = createStream({1, 2}, /*reuse_cache=*/true);
    auto context   = makeControlledAllocatorContext();

    ASSERT_EQ(scheduler->enqueueGroup({stream}).first, std::vector<bool>({true}));
    installReadinessAllocator([context, stream](const MallocInfo& info) {
        return info.request_id == stream->streamId() ? context : nullptr;
    });
    auto first = scheduler->schedule();
    ASSERT_TRUE(first.ok());
    EXPECT_TRUE(first.value().empty());
    ASSERT_EQ(scheduler->loading_cache_group_queue_.size(), 1);
    EXPECT_TRUE(scheduler->loading_cache_streams_.empty());
    EXPECT_EQ(scheduler->runningStreamsSize(), 0);
    EXPECT_EQ(scheduler->waitingStreamsSize(), 0);
    EXPECT_FALSE(scheduler->waitPredicate());

    ASSERT_TRUE(context->completeOne(true));
    std::promise<absl::StatusOr<std::list<GenerateStreamPtr>>> promise;
    auto                                                       future = promise.get_future();
    std::thread schedule_thread([&] { promise.set_value(scheduler->schedule()); });

    const auto wait_status = future.wait_for(std::chrono::milliseconds(250));
    if (wait_status != std::future_status::ready) {
        EXPECT_TRUE(scheduler->stop().ok());
    }
    schedule_thread.join();
    ASSERT_EQ(wait_status, std::future_status::ready);
    auto result = future.get();
    ASSERT_TRUE(result.ok());
    ASSERT_EQ(result.value().size(), 1);
    EXPECT_EQ(result.value().front(), stream);
    EXPECT_EQ(stream->getStatus(), StreamState::RUNNING);
}

TEST_F(FIFOSchedulerAsyncCacheTest, testPDFusionAllocatorLoadingLifecyclePromotesToDecode) {
    auto scheduler = createPDFusionRatioScheduler();
    auto stream    = createStream({1, 2, 3}, /*reuse_cache=*/true);
    auto context   = makeControlledAllocatorContext();
    ASSERT_TRUE(scheduler->enqueue(stream).ok());
    installReadinessAllocator([context](const MallocInfo&) { return context; });

    auto loading = scheduler->schedule();
    ASSERT_TRUE(loading.ok());
    ASSERT_TRUE(loading.value().empty());
    ASSERT_EQ(stream->getStatus(), StreamState::LOADING_CACHE);
    ASSERT_EQ(stream->streamCacheResource().allocator_load_context_, context);
    ASSERT_EQ(scheduler->loading_cache_streams_.size(), 1);
    ASSERT_EQ(scheduler->waitingStreamsSize(), 0);
    ASSERT_EQ(scheduler->pendingDecodeStreamsSize(), 0);
    ASSERT_EQ(initial_malloc_calls_, 1);

    EXPECT_TRUE(context->completeOne(true));
    auto prefill = scheduler->schedule();
    ASSERT_TRUE(prefill.ok());
    ASSERT_EQ(prefill.value().size(), 1);
    ASSERT_EQ(prefill.value().front(), stream);
    ASSERT_EQ(stream->getStatus(), StreamState::RUNNING);
    ASSERT_EQ(stream->streamCacheResource().allocator_load_context_, nullptr);
    ASSERT_EQ(scheduler->loading_cache_streams_.size(), 0);
    ASSERT_EQ(scheduler->waitingStreamsSize(), 0);
    ASSERT_EQ(scheduler->runningStreamsSize(), 0);
    ASSERT_EQ(scheduler->pendingDecodeStreamsSize(), 1);
    ASSERT_EQ(initial_malloc_calls_, 1);

    stream->setSeqLength(stream->seqLength() + 1);
    auto decode = scheduler->schedule();
    ASSERT_TRUE(decode.ok());
    ASSERT_EQ(decode.value().size(), 1);
    ASSERT_EQ(decode.value().front(), stream);
    ASSERT_EQ(scheduler->runningStreamsSize(), 1);
    ASSERT_EQ(scheduler->pendingDecodeStreamsSize(), 0);
    ASSERT_EQ(initial_malloc_calls_, 1);
}

TEST_F(FIFOSchedulerAsyncCacheTest, testPDFusionPendingAllocatorLoadsCountTowardInflightLimit) {
    auto scheduler = createPDFusionRatioScheduler(/*max_generate_batch_size=*/2);
    auto stream1   = createStream({1}, /*reuse_cache=*/true);
    auto stream2   = createStream({2}, /*reuse_cache=*/true);
    auto stream3   = createStream({3}, /*reuse_cache=*/true);
    auto context1  = makeControlledAllocatorContext();
    auto context2  = makeControlledAllocatorContext();
    auto context3  = makeControlledAllocatorContext();
    ASSERT_TRUE(scheduler->enqueue(stream1).ok());
    ASSERT_TRUE(scheduler->enqueue(stream2).ok());
    ASSERT_TRUE(scheduler->enqueue(stream3).ok());
    const std::unordered_map<int64_t, std::shared_ptr<AsyncContext>> contexts{
        {stream1->streamId(), context1}, {stream2->streamId(), context2}, {stream3->streamId(), context3}};
    installReadinessAllocator([contexts](const MallocInfo& info) { return contexts.at(info.request_id); });

    auto first = scheduler->schedule();
    ASSERT_TRUE(first.ok());
    ASSERT_TRUE(first.value().empty());
    ASSERT_EQ(scheduler->loading_cache_streams_.size(), 2);
    ASSERT_EQ(scheduler->waitingStreamsSize(), 1);
    ASSERT_EQ(scheduler->pendingDecodeStreamsSize(), 0);
    ASSERT_EQ(initial_malloc_calls_, 2);
    EXPECT_EQ(stream1->streamCacheResource().allocator_load_context_, context1);
    EXPECT_EQ(stream2->streamCacheResource().allocator_load_context_, context2);
    EXPECT_EQ(stream3->streamCacheResource().allocator_load_context_, nullptr);

    auto second = scheduler->schedule();
    ASSERT_TRUE(second.ok());
    ASSERT_TRUE(second.value().empty());
    ASSERT_EQ(scheduler->loading_cache_streams_.size(), 2);
    ASSERT_EQ(scheduler->waitingStreamsSize(), 1);
    ASSERT_EQ(scheduler->pendingDecodeStreamsSize(), 0);
    ASSERT_EQ(initial_malloc_calls_, 2);
}

}  // namespace rtp_llm
