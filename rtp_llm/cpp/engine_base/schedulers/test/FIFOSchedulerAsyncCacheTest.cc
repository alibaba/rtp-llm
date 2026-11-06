#include <chrono>
#include <functional>
#include <future>
#include <memory>
#include <stdexcept>
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
#include "rtp_llm/cpp/cache/test/mock/MockCoordinatorCacheManager.h"
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
    if (!coordinator->registerContext(context)) {
        throw std::runtime_error("failed to register controlled allocator context");
    }
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
            cache_manager_->coordinator_manager_ = real_allocator_;
        }
        DeviceTestBase::TearDown();
    }

    std::shared_ptr<FIFOScheduler> createScheduler(size_t   max_generate_batch_size     = 100,
                                                   size_t   max_inited_kv_cache_streams = 0,
                                                   RoleType role                        = RoleType::PDFUSION,
                                                   int      prefill_chunk_size          = 0) {
        ModelConfig model_config;
        model_config.max_seq_len = 8192;
        RuntimeConfig runtime_config;
        runtime_config.max_generate_batch_size                           = max_generate_batch_size;
        runtime_config.fifo_scheduler_config.max_batch_tokens_size       = 8192;
        runtime_config.fifo_scheduler_config.max_inited_kv_cache_streams = max_inited_kv_cache_streams;
        runtime_config.fifo_scheduler_config.prefill_chunk_size          = prefill_chunk_size;
        PDSepConfig pd_sep_config;
        pd_sep_config.role_type = role;
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

    GenerateStreamPtr createStream(const std::vector<int>&                     input_tokens        = {1, 2, 3},
                                   bool                                        reuse_cache         = false,
                                   bool                                        enable_memory_cache = false,
                                   int                                         max_new_tokens      = 1,
                                   const std::vector<int>&                     variable_num_beams  = {},
                                   RoleType                                    role_type           = RoleType::PDFUSION,
                                   const std::function<void(GenerateConfig&)>& configure           = {}) {
        ResourceContext resource_context;
        resource_context.cache_manager       = cache_manager_;
        resource_context.reuse_cache         = reuse_cache;
        resource_context.enable_memory_cache = enable_memory_cache;
        resource_context.role_type           = role_type;

        ModelConfig model_config;
        model_config.max_seq_len = 8192;
        RuntimeConfig runtime_config;

        auto query                           = std::make_shared<GenerateInput>();
        auto generate_config                 = std::make_shared<GenerateConfig>();
        query->request_id                    = next_request_id_++;
        generate_config->reuse_cache         = reuse_cache;
        generate_config->enable_memory_cache = enable_memory_cache;
        generate_config->max_new_tokens      = max_new_tokens;
        generate_config->variable_num_beams  = variable_num_beams;
        if (configure) {
            configure(*generate_config);
        }
        query->input_ids       = torch::tensor(input_tokens, torch::kInt32);
        query->generate_config = generate_config;
        return std::make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    }

    GenerateStreamPtr createPrefillStream(const std::vector<int>& input_tokens   = {1, 2, 3},
                                          bool                    reuse_cache    = false,
                                          int                     max_new_tokens = 1) {
        return createStream(input_tokens, reuse_cache, false, max_new_tokens, {}, RoleType::PREFILL);
    }

    void installReadinessAllocator(ContextSelector selector) {
        real_allocator_ = cache_manager_->coordinator_manager_;
        mock_allocator_ = std::make_shared<testing::NiceMock<MockCoordinatorCacheManager>>(cache_manager_->config_);
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
        ON_CALL(*mock_allocator_, insertIntoCache(testing::_, testing::_))
            .WillByDefault(testing::Invoke([this](const InsertInfo& info, size_t& resident_prefix_length) {
                ++insert_calls_;
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
        ON_CALL(*mock_allocator_, availableBlocksNum()).WillByDefault(testing::Invoke([this]() {
            return real_allocator_->availableBlocksNum();
        }));

        cache_manager_->coordinator_manager_ = mock_allocator_;
    }

    void installRetryableAllocator() {
        real_allocator_ = cache_manager_->coordinator_manager_;
        mock_allocator_ = std::make_shared<testing::NiceMock<MockCoordinatorCacheManager>>(cache_manager_->config_);
        initial_malloc_calls_ = 0;

        ON_CALL(*mock_allocator_, maxAvailableTokensNum()).WillByDefault(testing::Invoke([this] {
            return real_allocator_->maxAvailableTokensNum();
        }));
        ON_CALL(*mock_allocator_, availableBlocksNum()).WillByDefault(testing::Invoke([this] {
            return real_allocator_->availableBlocksNum();
        }));
        ON_CALL(*mock_allocator_, seqSizePerBlock()).WillByDefault(testing::Invoke([this] {
            return real_allocator_->seqSizePerBlock();
        }));
        ON_CALL(*mock_allocator_, totalBlocksNum()).WillByDefault(testing::Return(64));
        ON_CALL(*mock_allocator_, getNeedBlocks(testing::_)).WillByDefault(testing::Return(1));
        ON_CALL(*mock_allocator_, initMallocForCommonLen(testing::_))
            .WillByDefault(testing::Invoke([this](const MallocInfo&) {
                ++initial_malloc_calls_;
                return MallocResult{false, 0, 0, MallocStatus::RETRYABLE_RESOURCE_EXHAUSTED};
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
    std::atomic<size_t>                                             initial_malloc_calls_{0};
    size_t                                                          free_calls_{0};
    size_t                                                          insert_calls_{0};
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
    EXPECT_TRUE(context->completeTransfers(1, true));

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
    EXPECT_TRUE(context->completeTransfers(1, false));

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
                               /*enable_memory_cache=*/false,
                               /*max_new_tokens=*/1,
                               /*variable_num_beams=*/{},
                               RoleType::PREFILL);
    auto context   = makeControlledAllocatorContext();
    ASSERT_TRUE(scheduler->enqueue(stream).ok());
    installReadinessAllocator([context](const MallocInfo&) { return context; });

    auto first = scheduler->schedule();
    ASSERT_TRUE(first.ok());
    ASSERT_EQ(stream->getStatus(), StreamState::LOADING_CACHE);
    EXPECT_TRUE(context->completeTransfers(1, false));

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
    EXPECT_TRUE(context->completeTransfers(1, true));

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
                               /*enable_memory_cache=*/false,
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
    EXPECT_TRUE(context->completeTransfers(1, true));

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

    ASSERT_TRUE(context->completeTransfers(1, true));
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
    ASSERT_TRUE(context->completeTransfers(1, true));

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

    ASSERT_TRUE(context->completeTransfers(1, true));
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

    ASSERT_TRUE(context->completeTransfers(1, true));
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

    ASSERT_TRUE(context->completeTransfers(1, true));
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

    ASSERT_TRUE(context->completeTransfers(1, true));
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

    ASSERT_TRUE(context->completeTransfers(1, true));
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

    EXPECT_TRUE(context->completeTransfers(1, true));
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

namespace {
bool waitForCache(const std::function<bool()>& predicate) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (std::chrono::steady_clock::now() < deadline) {
        if (predicate()) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return predicate();
}
}  // namespace

TEST_F(FIFOSchedulerAsyncCacheTest, AsyncPrepareRejectsPDFusion) {
    autil::EnvGuard flag("RTP_LLM_ASYNC_PREPARE_CACHE", "1");
    EXPECT_THROW(createScheduler(2, 0, RoleType::PDFUSION), std::exception);
}

TEST_F(FIFOSchedulerAsyncCacheTest, AsyncPrepareDisabledSupportsLocalDecode) {
    autil::EnvGuard flag("RTP_LLM_ASYNC_PREPARE_CACHE", "0");
    auto            scheduler = createScheduler(2, 0, RoleType::PREFILL);
    auto            stream    = createPrefillStream({1, 2, 3}, false, 8);
    ASSERT_TRUE(scheduler->enqueue(stream).ok());
    EXPECT_FALSE(scheduler->async_cache_prepare_enabled_);
    EXPECT_FALSE(stream->hasError());
    EXPECT_EQ(stream->curBlocksNum(), 0);
    auto batch = scheduler->schedule();
    ASSERT_TRUE(batch.ok());
    ASSERT_EQ(batch->size(), 1);
    EXPECT_EQ(batch->front(), stream);
    const auto prefill_blocks = stream->curBlocksNum();
    stream->setIsContextStream(false);
    stream->complete_token_ids_->setSeqLength(6);
    ASSERT_TRUE(scheduler->schedule().ok());
    EXPECT_FALSE(stream->hasError());
    EXPECT_GT(stream->curBlocksNum(), prefill_blocks);
}

TEST_F(FIFOSchedulerAsyncCacheTest, AsyncPrepareAcceptsPDSeparatedGeneration) {
    autil::EnvGuard flag("RTP_LLM_ASYNC_PREPARE_CACHE", "1");
    auto            scheduler               = createScheduler(2, 0, RoleType::PREFILL);
    auto            stream                  = createPrefillStream({1, 2, 3}, false, 8);
    stream->generateConfig()->pd_separation = true;
    ASSERT_TRUE(scheduler->enqueue(stream).ok());
    ASSERT_TRUE(waitForCache([&] { return stream->hasEvent(StreamEvents::CachePrepared); }));
    EXPECT_FALSE(stream->hasError());
    EXPECT_GT(stream->curBlocksNum(), 0);
    auto batch = scheduler->schedule();
    ASSERT_TRUE(batch.ok());
    ASSERT_EQ(batch->size(), 1);
    EXPECT_EQ(batch->front(), stream);
}

TEST_F(FIFOSchedulerAsyncCacheTest, AsyncPrepareReadyTailUnblocksMemoryStarvedHead) {
    autil::EnvGuard flag("RTP_LLM_ASYNC_PREPARE_CACHE", "1");
    auto            head = createPrefillStream(std::vector<int>(16, 1));
    auto            tail = createPrefillStream(std::vector<int>(30, 2));
    // Reproduce the state after a head load needs to retry allocation while a
    // later, already-prepared request holds the blocks required for that retry.
    ASSERT_EQ(tail->prepareCache(), GenerateStream::CachePrepareResult::DONE);
    ASSERT_FALSE(tail->hasError());
    auto scheduler = createScheduler(1, 0, RoleType::PREFILL);
    ASSERT_TRUE(scheduler->enqueue(head).ok());
    ASSERT_TRUE(scheduler->enqueue(tail).ok());
    ASSERT_TRUE(waitForCache([&] {
        std::lock_guard<std::mutex> stream_lock(*head->mutex_);
        return head->streamCacheResource().mallocFailedTimes() > 0;
    }));

    auto batch = scheduler->schedule();
    ASSERT_TRUE(batch.ok());
    ASSERT_EQ(batch->size(), 1);
    EXPECT_EQ(batch->front(), tail);
    EXPECT_EQ(head->getStatus(), StreamState::WAITING);

    tail->reportEvent(StreamEvents::GenerateDone);
    ASSERT_TRUE(scheduler->schedule().ok());
    ASSERT_TRUE(waitForCache([&] { return head->hasEvent(StreamEvents::CachePrepared); }));
    auto resumed = scheduler->schedule();
    ASSERT_TRUE(resumed.ok());
    ASSERT_EQ(resumed->size(), 1);
    EXPECT_EQ(resumed->front(), head);
    EXPECT_FALSE(head->hasError());
}

TEST_F(FIFOSchedulerAsyncCacheTest, AsyncPrepareCompletesInflightTailAfterHeadAllocationShortage) {
    autil::EnvGuard flag("RTP_LLM_ASYNC_PREPARE_CACHE", "1");
    auto            head    = createPrefillStream(std::vector<int>(16, 1));
    auto            tail    = createPrefillStream(std::vector<int>(30, 2), true);
    auto            context = makeControlledAllocatorContext();
    installReadinessAllocator(
        [&](const MallocInfo& info) { return info.request_id == tail->streamId() ? context : nullptr; });
    ASSERT_EQ(tail->prepareCache(), GenerateStream::CachePrepareResult::WAIT);
    auto scheduler = createScheduler(1, 0, RoleType::PREFILL);
    ASSERT_TRUE(scheduler->enqueue(head).ok());
    ASSERT_TRUE(scheduler->enqueue(tail).ok());
    ASSERT_TRUE(waitForCache([&] {
        std::lock_guard<std::mutex> stream_lock(*head->mutex_);
        return head->streamCacheResource().mallocFailedTimes() > 0;
    }));
    const auto available = cache_manager_->availableBlocksNum();
    ASSERT_TRUE(context->completeTransfers(1, true));
    ASSERT_TRUE(waitForCache([&] { return tail->hasEvent(StreamEvents::CachePrepared); }));
    EXPECT_EQ(cache_manager_->availableBlocksNum(), available);
    auto batch = scheduler->schedule();
    ASSERT_TRUE(batch.ok());
    ASSERT_EQ(batch->size(), 1);
    EXPECT_EQ(batch->front(), tail);
    tail->reportEvent(StreamEvents::GenerateDone);
    ASSERT_TRUE(scheduler->schedule().ok());
    ASSERT_TRUE(waitForCache([&] { return head->hasEvent(StreamEvents::CachePrepared); }));
    batch = scheduler->schedule();
    ASSERT_TRUE(batch.ok());
    ASSERT_EQ(batch->size(), 1);
    EXPECT_EQ(batch->front(), head);
}

TEST_F(FIFOSchedulerAsyncCacheTest, AsyncPrepareShortageDoesNotBlockSmallerRequestsOrGroups) {
    autil::EnvGuard flag("RTP_LLM_ASYNC_PREPARE_CACHE", "1");
    auto            held = createPrefillStream(std::vector<int>(30, 9));
    ASSERT_TRUE(held->initKVBlock().ok());
    auto head      = createPrefillStream(std::vector<int>(16, 1));
    auto tail      = createPrefillStream({2});
    auto grouped   = createPrefillStream({3});
    auto scheduler = createScheduler(2, 0, RoleType::PREFILL);
    ASSERT_TRUE(scheduler->enqueue(head).ok());
    ASSERT_TRUE(scheduler->enqueue(tail).ok());
    ASSERT_EQ(scheduler->enqueueGroup({grouped}).first, std::vector<bool>({true}));
    ASSERT_TRUE(waitForCache([&] { return tail->hasEvent(StreamEvents::CachePrepared); }));
    ASSERT_TRUE(waitForCache([&] { return grouped->hasEvent(StreamEvents::CachePrepared); }));
    EXPECT_FALSE(head->hasEvent(StreamEvents::CachePrepared));
    auto batch = scheduler->schedule();
    ASSERT_TRUE(batch.ok());
    ASSERT_EQ(batch->size(), 1);
    EXPECT_EQ(batch->front(), tail);
    tail->reportEvent(StreamEvents::GenerateDone);
    batch = scheduler->schedule();
    ASSERT_TRUE(batch.ok());
    ASSERT_EQ(batch->size(), 1);
    EXPECT_EQ(batch->front(), grouped);
    grouped->reportEvent(StreamEvents::GenerateDone);
    ASSERT_TRUE(scheduler->schedule().ok());
    held->releaseResource();
    ASSERT_TRUE(waitForCache([&] { return head->hasEvent(StreamEvents::CachePrepared); }));
}

TEST_F(FIFOSchedulerAsyncCacheTest, AsyncPrepareShortageDoesNotBlockLaterReadyGroup) {
    autil::EnvGuard flag("RTP_LLM_ASYNC_PREPARE_CACHE", "1");
    auto            held = createPrefillStream(std::vector<int>(30, 9));
    ASSERT_TRUE(held->initKVBlock().ok());
    auto head      = createPrefillStream(std::vector<int>(16, 1));
    auto tail      = createPrefillStream({2});
    auto scheduler = createScheduler(2, 0, RoleType::PREFILL);
    ASSERT_EQ(scheduler->enqueueGroup({head}).first, std::vector<bool>({true}));
    ASSERT_EQ(scheduler->enqueueGroup({tail}).first, std::vector<bool>({true}));
    ASSERT_TRUE(waitForCache([&] { return tail->hasEvent(StreamEvents::CachePrepared); }));
    auto batch = scheduler->schedule();
    ASSERT_TRUE(batch.ok());
    ASSERT_EQ(batch->size(), 1);
    EXPECT_EQ(batch->front(), tail);
    EXPECT_FALSE(head->hasEvent(StreamEvents::CachePrepared));
    tail->reportEvent(StreamEvents::GenerateDone);
    ASSERT_TRUE(scheduler->schedule().ok());
    held->releaseResource();
    ASSERT_TRUE(waitForCache([&] { return head->hasEvent(StreamEvents::CachePrepared); }));
    batch = scheduler->schedule();
    ASSERT_TRUE(batch.ok());
    ASSERT_EQ(batch->size(), 1);
    EXPECT_EQ(batch->front(), head);
}

TEST_F(FIFOSchedulerAsyncCacheTest, AsyncPrepareEnforcesPurePDContractForLocalFallbackConfigurations) {
    const std::vector<std::function<void(GenerateConfig&)>> configurations = {
        [](GenerateConfig& config) { config.num_beams = 2; },
        [](GenerateConfig& config) { config.variable_num_beams = {1, 2}; },
        [](GenerateConfig& config) { config.num_return_sequences = 2; },
        [](GenerateConfig& config) { config.can_use_pd_separation = false; },
    };
    for (const char* enabled : {"0", "1"}) {
        autil::EnvGuard flag("RTP_LLM_ASYNC_PREPARE_CACHE", enabled);
        for (size_t i = 0; i < configurations.size(); ++i) {
            SCOPED_TRACE(enabled);
            SCOPED_TRACE(i);
            auto stream    = createStream({1, 2}, false, false, 8, {}, RoleType::PREFILL, configurations[i]);
            auto scheduler = createScheduler(2, 0, RoleType::PREFILL);
            ASSERT_TRUE(scheduler->enqueue(stream).ok());
            if (enabled[0] == '1') {
                ASSERT_TRUE(waitForCache([&] { return stream->hasEvent(StreamEvents::CachePrepared); }));
                EXPECT_EQ(stream->statusInfo().code(), ErrorCode::INVALID_PARAMS);
                EXPECT_NE(stream->statusInfo().ToString().find("pure PD"), std::string::npos);
                EXPECT_EQ(stream->curBlocksNum(), 0);
                EXPECT_FALSE(stream->hasEvent(StreamEvents::LoadInitiated));
                auto batch = scheduler->schedule();
                ASSERT_TRUE(batch.ok());
                EXPECT_TRUE(batch->empty());
                EXPECT_EQ(stream->getStatus(), StreamState::FINISHED);
            } else {
                auto batch = scheduler->schedule();
                ASSERT_TRUE(batch.ok());
                ASSERT_EQ(batch->size(), 1);
                EXPECT_EQ(batch->front(), stream);
                EXPECT_FALSE(stream->hasError()) << stream->statusInfo().ToString();
            }
            ASSERT_TRUE(scheduler->stop().ok());
        }
    }
}

TEST_F(FIFOSchedulerAsyncCacheTest, AsyncPrepareRejectsLocalDecodeWithoutDisruptingInflightLoads) {
    autil::EnvGuard flag("RTP_LLM_ASYNC_PREPARE_CACHE", "1");
    auto            context = makeControlledAllocatorContext();
    auto            pending = createPrefillStream(std::vector<int>(30, 2), true);
    installReadinessAllocator(
        [&](const MallocInfo& info) { return info.request_id == pending->streamId() ? context : nullptr; });
    auto scheduler = createScheduler(1, 1, RoleType::PREFILL);
    ASSERT_TRUE(scheduler->enqueue(pending).ok());
    ASSERT_TRUE(waitForCache([&] { return pending->hasEvent(StreamEvents::LoadInitiated); }));
    auto local = createPrefillStream({1, 2, 3}, false, 8);
    auto fresh = createPrefillStream({4, 5});
    ASSERT_TRUE(scheduler->enqueue(local).ok());
    ASSERT_TRUE(scheduler->enqueue(fresh).ok());
    ASSERT_TRUE(waitForCache([&] { return local->hasEvent(StreamEvents::CachePrepared); }));
    EXPECT_EQ(local->statusInfo().code(), ErrorCode::INVALID_PARAMS);
    EXPECT_EQ(local->curBlocksNum(), 0);
    EXPECT_FALSE(local->hasEvent(StreamEvents::LoadInitiated));
    auto batch = scheduler->schedule();
    ASSERT_TRUE(batch.ok());
    EXPECT_TRUE(batch->empty());
    EXPECT_EQ(local->getStatus(), StreamState::FINISHED);
    EXPECT_FALSE(pending->hasError());
    EXPECT_EQ(fresh->curBlocksNum(), 0);
    ASSERT_TRUE(context->completeTransfers(1, true));
    ASSERT_TRUE(waitForCache([&] { return pending->hasEvent(StreamEvents::CachePrepared); }));
    batch = scheduler->schedule();
    ASSERT_TRUE(batch.ok());
    ASSERT_EQ(batch->size(), 1);
    EXPECT_EQ(batch->front(), pending);
    pending->reportEvent(StreamEvents::GenerateDone);
    ASSERT_TRUE(scheduler->schedule().ok());
    ASSERT_TRUE(waitForCache([&] { return fresh->hasEvent(StreamEvents::CachePrepared); }));
    batch = scheduler->schedule();
    ASSERT_TRUE(batch.ok());
    ASSERT_EQ(batch->size(), 1);
    EXPECT_EQ(batch->front(), fresh);
    EXPECT_FALSE(fresh->hasError());
    std::lock_guard<std::mutex> lock(scheduler->lock_);
    EXPECT_TRUE(scheduler->async_cache_prepare_enabled_);
}

TEST_F(FIFOSchedulerAsyncCacheTest, AsyncPrepareRejectsLocalGroupMemberAndAdmitsPDMember) {
    autil::EnvGuard flag("RTP_LLM_ASYNC_PREPARE_CACHE", "1");
    auto            scheduler           = createScheduler(2, 0, RoleType::PREFILL);
    auto            local               = createPrefillStream({1, 2}, false, 8);
    auto            pd                  = createPrefillStream({2, 3}, false, 8);
    pd->generateConfig()->pd_separation = true;
    ASSERT_EQ(scheduler->enqueueGroup({local, pd}).first, std::vector<bool>({true, true}));
    ASSERT_TRUE(waitForCache([&] { return local->hasEvent(StreamEvents::CachePrepared); }));
    ASSERT_TRUE(waitForCache([&] { return pd->hasEvent(StreamEvents::CachePrepared); }));
    EXPECT_EQ(local->statusInfo().code(), ErrorCode::INVALID_PARAMS);
    EXPECT_EQ(local->curBlocksNum(), 0);
    EXPECT_FALSE(local->hasEvent(StreamEvents::LoadInitiated));
    auto batch = scheduler->schedule();
    ASSERT_TRUE(batch.ok());
    ASSERT_EQ(batch->size(), 1);
    EXPECT_EQ(batch->front(), pd);
    EXPECT_FALSE(pd->hasError());
    EXPECT_EQ(local->getStatus(), StreamState::FINISHED);
}

TEST_F(FIFOSchedulerAsyncCacheTest, AsyncPrepareStartsBeforeAdmissionAndPreservesWaitLatency) {
    autil::EnvGuard flag("RTP_LLM_ASYNC_PREPARE_CACHE", "1");
    auto            context = makeControlledAllocatorContext();
    installReadinessAllocator([context](const MallocInfo&) { return context; });
    auto scheduler = createScheduler(1, 0, RoleType::PREFILL);
    auto stream    = createPrefillStream({1, 2, 3}, true);
    stream->resetBeginTime(autil::TimeUtility::currentTimeInMicroSeconds() - 1000000);
    ASSERT_TRUE(scheduler->enqueue(stream).ok());
    ASSERT_TRUE(waitForCache([&] { return stream->hasEvent(StreamEvents::LoadInitiated); }));
    EXPECT_GT(stream->curBlocksNum(), 0);
    EXPECT_EQ(stream->getStatus(), StreamState::WAITING);
    EXPECT_EQ(stream->getTimeInfo().wait_time_us, 0);
    ASSERT_TRUE(context->completeTransfers(1, true));
    ASSERT_TRUE(waitForCache([&] { return stream->hasEvent(StreamEvents::CachePrepared); }));
    auto batch = scheduler->schedule();
    ASSERT_TRUE(batch.ok());
    ASSERT_EQ(batch->size(), 1);
    EXPECT_GE(stream->getTimeInfo().wait_time_us, 1000000);
    EXPECT_EQ(initial_malloc_calls_.load(), 1);
}

TEST_F(FIFOSchedulerAsyncCacheTest, AsyncPrepareAdmitsReadyTailBeforePendingHead) {
    autil::EnvGuard flag("RTP_LLM_ASYNC_PREPARE_CACHE", "1");
    auto            first_context  = makeControlledAllocatorContext();
    auto            second_context = makeControlledAllocatorContext();
    auto            first          = createPrefillStream({1, 2}, true);
    auto            second         = createPrefillStream({3, 4}, true);
    installReadinessAllocator(
        [&](const MallocInfo& info) { return info.request_id == first->streamId() ? first_context : second_context; });
    auto scheduler = createScheduler(2, 0, RoleType::PREFILL);
    ASSERT_TRUE(scheduler->enqueue(first).ok());
    ASSERT_TRUE(scheduler->enqueue(second).ok());
    ASSERT_TRUE(waitForCache([&] { return second->hasEvent(StreamEvents::LoadInitiated); }));
    ASSERT_TRUE(second_context->completeTransfers(1, true));
    ASSERT_TRUE(waitForCache([&] { return second->hasEvent(StreamEvents::CachePrepared); }));
    auto ready = scheduler->schedule();
    ASSERT_TRUE(ready.ok());
    ASSERT_EQ(ready->size(), 1);
    EXPECT_EQ(ready->front(), second);
    EXPECT_EQ(first->getStatus(), StreamState::WAITING);
    ASSERT_TRUE(first_context->completeTransfers(1, true));
    ASSERT_TRUE(waitForCache([&] { return first->hasEvent(StreamEvents::CachePrepared); }));
    second->reportEvent(StreamEvents::GenerateDone);
    auto batch = scheduler->schedule();
    ASSERT_TRUE(batch.ok());
    ASSERT_EQ(batch->size(), 1);
    EXPECT_EQ(batch->front(), first);
}

TEST_F(FIFOSchedulerAsyncCacheTest, AsyncPrepareGroupWaitsForAllLoads) {
    autil::EnvGuard flag("RTP_LLM_ASYNC_PREPARE_CACHE", "1");
    auto            context = makeControlledAllocatorContext();
    auto            first   = createPrefillStream({1, 2}, true);
    auto            second  = createPrefillStream({3, 4});
    installReadinessAllocator(
        [context, first](const MallocInfo& info) { return info.request_id == first->streamId() ? context : nullptr; });
    auto scheduler = createScheduler(2, 0, RoleType::PREFILL);
    ASSERT_EQ(scheduler->enqueueGroup({first, second}).first, std::vector<bool>({true, true}));
    ASSERT_TRUE(waitForCache([&] { return second->hasEvent(StreamEvents::CachePrepared); }));
    auto pending = scheduler->schedule();
    ASSERT_TRUE(pending.ok());
    EXPECT_TRUE(pending->empty());
    ASSERT_TRUE(context->completeTransfers(1, true));
    ASSERT_TRUE(waitForCache([&] { return first->hasEvent(StreamEvents::CachePrepared); }));
    ASSERT_FALSE(first->hasError()) << first->statusInfo().ToString();
    ASSERT_FALSE(second->hasError()) << second->statusInfo().ToString();
    auto batch = scheduler->schedule();
    ASSERT_TRUE(batch.ok());
    EXPECT_EQ(batch->size(), 2);
}

TEST_F(FIFOSchedulerAsyncCacheTest, AsyncPrepareGroupRecoversFromAllocationShortage) {
    autil::EnvGuard flag("RTP_LLM_ASYNC_PREPARE_CACHE", "1");
    auto            scheduler = createScheduler(2, 0, RoleType::PREFILL);
    auto            first     = createPrefillStream(std::vector<int>(16, 1));
    auto            blocked   = createPrefillStream(std::vector<int>(30, 2));
    ASSERT_EQ(scheduler->enqueueGroup({first, blocked}).first, std::vector<bool>({true, true}));
    ASSERT_TRUE(waitForCache([&] {
        std::lock_guard<std::mutex> stream_lock(*blocked->mutex_);
        return blocked->streamCacheResource().mallocFailedTimes() > 0;
    }));
    auto prefix = scheduler->schedule();
    ASSERT_TRUE(prefix.ok());
    ASSERT_EQ(prefix->size(), 1);
    EXPECT_EQ(prefix->front(), first);
    first->reportEvent(StreamEvents::GenerateDone);
    ASSERT_TRUE(scheduler->schedule().ok());
    ASSERT_TRUE(waitForCache([&] { return blocked->hasEvent(StreamEvents::CachePrepared); }));
    auto residual = scheduler->schedule();
    ASSERT_TRUE(residual.ok());
    ASSERT_EQ(residual->size(), 1);
    EXPECT_EQ(residual->front(), blocked);
}

TEST_F(FIFOSchedulerAsyncCacheTest, AsyncPrepareHonorsInitializedStreamQuota) {
    autil::EnvGuard flag("RTP_LLM_ASYNC_PREPARE_CACHE", "1");
    auto            scheduler = createScheduler(2, 1, RoleType::PREFILL);
    auto            first     = createPrefillStream({1, 2});
    auto            second    = createPrefillStream({3, 4});
    ASSERT_TRUE(scheduler->enqueue(first).ok());
    ASSERT_TRUE(scheduler->enqueue(second).ok());
    ASSERT_TRUE(waitForCache([&] { return first->hasEvent(StreamEvents::CachePrepared); }));
    std::this_thread::sleep_for(std::chrono::milliseconds(150));
    EXPECT_EQ(second->curBlocksNum(), 0);
    auto batch = scheduler->schedule();
    ASSERT_TRUE(batch.ok());
    ASSERT_EQ(batch->size(), 1);
    first->reportEvent(StreamEvents::GenerateDone);
    ASSERT_TRUE(scheduler->schedule().ok());
    ASSERT_TRUE(waitForCache([&] { return second->hasEvent(StreamEvents::CachePrepared); }));
}

TEST_F(FIFOSchedulerAsyncCacheTest, AsyncPrepareDoesNotSpinOnRetryableShortage) {
    autil::EnvGuard flag("RTP_LLM_ASYNC_PREPARE_CACHE", "1");
    installRetryableAllocator();
    auto scheduler = createScheduler(2, 0, RoleType::PREFILL);
    auto first     = createPrefillStream({1, 2});
    auto trailing  = createPrefillStream({3, 4});
    ASSERT_TRUE(scheduler->enqueue(first).ok());
    ASSERT_TRUE(scheduler->enqueue(trailing).ok());
    ASSERT_TRUE(waitForCache([&] { return initial_malloc_calls_.load() >= 2; }));
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
    EXPECT_GE(initial_malloc_calls_.load(), 2);
    EXPECT_LE(initial_malloc_calls_.load(), 8);
    EXPECT_EQ(trailing->curBlocksNum(), 0);
    first->reportError(ErrorCode::CANCELLED, "cancel blocked request");
    ASSERT_TRUE(waitForCache([&] { return first->hasEvent(StreamEvents::CachePrepared); }));
    ASSERT_TRUE(scheduler->schedule().ok());
    EXPECT_EQ(first->getStatus(), StreamState::FINISHED);
}

TEST_F(FIFOSchedulerAsyncCacheTest, AsyncPrepareRetriesCapacityReleasedDuringFailedAllocation) {
    autil::EnvGuard flag("RTP_LLM_ASYNC_PREPARE_CACHE", "1");
    auto            held = createPrefillStream(std::vector<int>(10, 9));
    ASSERT_TRUE(held->initKVBlock().ok());
    installReadinessAllocator([](const MallocInfo&) { return nullptr; });
    ON_CALL(*mock_allocator_, initMallocForCommonLen(testing::_))
        .WillByDefault(testing::Invoke([&](const MallocInfo& info) {
            if (++initial_malloc_calls_ == 1) {
                held->releaseResource();
                return MallocResult{false, 0, 0, MallocStatus::RETRYABLE_RESOURCE_EXHAUSTED};
            }
            return real_allocator_->initMallocForCommonLen(info);
        }));
    auto scheduler = createScheduler(1, 0, RoleType::PREFILL);
    auto stream    = createPrefillStream({1, 2});
    ASSERT_TRUE(scheduler->enqueue(stream).ok());
    ASSERT_TRUE(waitForCache([&] { return stream->hasEvent(StreamEvents::CachePrepared); }));
    EXPECT_EQ(initial_malloc_calls_.load(), 2);
    auto batch = scheduler->schedule();
    ASSERT_TRUE(batch.ok());
    ASSERT_EQ(batch->size(), 1);
    EXPECT_EQ(batch->front(), stream);
}

TEST_F(FIFOSchedulerAsyncCacheTest, AsyncPreparePreservesGlobalChunkBudgetAndInitialReuse) {
    autil::EnvGuard flag("RTP_LLM_ASYNC_PREPARE_CACHE", "1");
    installReadinessAllocator([](const MallocInfo&) { return nullptr; });
    auto scheduler = createScheduler(2, 0, RoleType::PREFILL, 4);
    auto stream    = createPrefillStream({1, 2, 3, 4, 5, 6});
    stream->setChunkSize(4);
    ASSERT_TRUE(scheduler->enqueue(stream).ok());
    ASSERT_TRUE(waitForCache([&] { return stream->hasEvent(StreamEvents::CachePrepared); }));
    auto first = scheduler->schedule();
    ASSERT_TRUE(first.ok());
    ASSERT_EQ(first->size(), 1);
    EXPECT_EQ(first->front(), stream);
    EXPECT_EQ(stream->contextLength(), 4);
    EXPECT_TRUE(stream->isMiddleChunk());
    stream->advanceChunk();
    auto second = scheduler->schedule();
    ASSERT_TRUE(second.ok());
    ASSERT_EQ(second->size(), 1);
    EXPECT_EQ(second->front(), stream);
    EXPECT_EQ(stream->contextLength(), 2);
    EXPECT_FALSE(stream->isMiddleChunk());
    EXPECT_EQ(stream->initialReuseLength(), 0);
    EXPECT_EQ(stream->synchronizedPrefixLength(), 4);
    EXPECT_EQ(initial_malloc_calls_.load(), 1);
}

TEST_F(FIFOSchedulerAsyncCacheTest, AsyncPrepareTimesOutWhileCapacityIsUnchanged) {
    autil::EnvGuard flag("RTP_LLM_ASYNC_PREPARE_CACHE", "1");
    installRetryableAllocator();
    auto scheduler                       = createScheduler(1, 0, RoleType::PREFILL);
    auto stream                          = createPrefillStream({1, 2});
    stream->generateConfig()->timeout_ms = 100;
    stream->resetBeginTime(autil::TimeUtility::currentTimeInMicroSeconds());
    ASSERT_TRUE(scheduler->enqueue(stream).ok());
    ASSERT_TRUE(waitForCache([&] { return initial_malloc_calls_.load() >= 1; }));
    ASSERT_TRUE(waitForCache([&] { return stream->hasEvent(StreamEvents::CachePrepared); }));
    EXPECT_TRUE(stream->hasError());
    EXPECT_LE(initial_malloc_calls_.load(), 2);
    ASSERT_TRUE(scheduler->schedule().ok());
    EXPECT_EQ(stream->getStatus(), StreamState::FINISHED);
    EXPECT_TRUE(stream->streamCacheResource().isResourceReleased());
}

TEST_F(FIFOSchedulerAsyncCacheTest, AsyncPrepareTimesOutUnpreparedTailBehindBlockedHead) {
    autil::EnvGuard flag("RTP_LLM_ASYNC_PREPARE_CACHE", "1");
    installRetryableAllocator();
    auto scheduler                     = createScheduler(2, 0, RoleType::PREFILL);
    auto head                          = createPrefillStream({1, 2});
    auto tail                          = createPrefillStream({3, 4});
    tail->generateConfig()->timeout_ms = 100;
    ASSERT_TRUE(scheduler->enqueue(head).ok());
    ASSERT_TRUE(scheduler->enqueue(tail).ok());
    ASSERT_TRUE(waitForCache([&] { return initial_malloc_calls_.load() >= 2; }));
    tail->resetBeginTime(autil::TimeUtility::currentTimeInMicroSeconds() - 1000000);

    auto batch = scheduler->schedule();
    ASSERT_TRUE(batch.ok());
    EXPECT_TRUE(batch->empty());
    EXPECT_EQ(tail->getStatus(), StreamState::FINISHED);
    EXPECT_EQ(tail->statusInfo().code(), ErrorCode::GENERATE_TIMEOUT);
    EXPECT_TRUE(tail->streamCacheResource().isResourceReleased());
    EXPECT_EQ(head->getStatus(), StreamState::WAITING);
    EXPECT_EQ(scheduler->waitingStreamsSize(), 1);
}

TEST_F(FIFOSchedulerAsyncCacheTest, AsyncPrepareStopJoinsWorkerAndReleasesPendingLoads) {
    autil::EnvGuard flag("RTP_LLM_ASYNC_PREPARE_CACHE", "1");
    auto            context = makeControlledAllocatorContext();
    installReadinessAllocator([context](const MallocInfo&) { return context; });
    auto scheduler = createScheduler(2, 0, RoleType::PREFILL);
    auto stream    = createPrefillStream({1, 2}, true);
    ASSERT_EQ(scheduler->enqueueGroup({stream}).first, std::vector<bool>({true}));
    ASSERT_TRUE(waitForCache([&] { return stream->hasEvent(StreamEvents::LoadInitiated); }));
    ASSERT_TRUE(scheduler->stop().ok());
    EXPECT_FALSE(scheduler->cache_prepare_thread_.joinable());
    EXPECT_TRUE(scheduler->empty());
    EXPECT_TRUE(stream->hasError());
    EXPECT_EQ(stream->statusInfo().code(), ErrorCode::CANCELLED);
    EXPECT_TRUE(stream->streamCacheResource().isResourceReleased());
    ASSERT_TRUE(scheduler->stop().ok());
}

TEST_F(FIFOSchedulerAsyncCacheTest, AsyncPrepareIsDisabledForDecodeAndByDefault) {
    {
        autil::EnvGuard flag("RTP_LLM_ASYNC_PREPARE_CACHE", "0");
        auto            scheduler = createScheduler(2);
        EXPECT_FALSE(scheduler->async_cache_prepare_enabled_);
        ASSERT_TRUE(scheduler->enqueue(createStream()).ok());
        ASSERT_EQ(scheduler->schedule()->size(), 1);
    }
    autil::EnvGuard flag("RTP_LLM_ASYNC_PREPARE_CACHE", "1");
    auto            scheduler = createScheduler(2, 0, RoleType::DECODE);
    EXPECT_FALSE(scheduler->async_cache_prepare_enabled_);
}

TEST_F(FIFOSchedulerAsyncCacheTest, AsyncPrepareCleansCanceledTailBehindPendingHead) {
    autil::EnvGuard flag("RTP_LLM_ASYNC_PREPARE_CACHE", "1");
    auto            context  = makeControlledAllocatorContext();
    auto            first    = createPrefillStream({1, 2, 3}, true);
    auto            canceled = createPrefillStream({4, 5, 6});
    installReadinessAllocator(
        [context, first](const MallocInfo& info) { return info.request_id == first->streamId() ? context : nullptr; });
    auto scheduler = createScheduler(2, 0, RoleType::PREFILL);
    ASSERT_TRUE(scheduler->enqueue(first).ok());
    ASSERT_TRUE(scheduler->enqueue(canceled).ok());
    ASSERT_TRUE(waitForCache([&] { return first->hasEvent(StreamEvents::LoadInitiated); }));
    canceled->reportError(ErrorCode::CANCELLED, "cancel while waiting for head");
    auto batch = scheduler->schedule();
    ASSERT_TRUE(batch.ok());
    EXPECT_TRUE(batch->empty());
    EXPECT_EQ(first->getStatus(), StreamState::WAITING);
    EXPECT_EQ(canceled->getStatus(), StreamState::FINISHED);
    EXPECT_TRUE(canceled->streamCacheResource().isResourceReleased());
    EXPECT_EQ(scheduler->waitingStreamsSize(), 1);
    ASSERT_TRUE(context->completeTransfers(1, true));
    ASSERT_TRUE(waitForCache([&] { return first->hasEvent(StreamEvents::CachePrepared); }));
    batch = scheduler->schedule();
    ASSERT_TRUE(batch.ok());
    ASSERT_EQ(batch->size(), 1);
    EXPECT_EQ(batch->front(), first);
}

}  // namespace rtp_llm
