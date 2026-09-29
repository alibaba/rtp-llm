#include <functional>
#include <memory>
#include <stdexcept>

#include "torch/all.h"
#include "gmock/gmock-actions.h"
#include "gmock/gmock-function-mocker.h"
#include "gtest/gtest.h"

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

    std::shared_ptr<PPScheduler> createScheduler(size_t max_generate_batch_size = 100,
                                                 RoleType role = RoleType::PDFUSION,
                                                 SpeculativeType type = SP_TYPE_NONE) {
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
        SpeculativeExecutionConfig sp_config;
        sp_config.type = type;
        sp_config.gen_num_per_cycle = 3;
        return std::make_shared<PPScheduler>(runtime_config, model_config, pd_sep_config,
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
        stream->setPipelineParallel(true);
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

TEST_F(PPSchedulerTest, InflightIsSkippedUntilResultCommit) {
    auto scheduler = createScheduler();
    auto stream = createStream({1, 2, 3}, false, false, 4);
    ASSERT_TRUE(scheduler->enqueue(stream).ok());
    auto first = scheduler->schedule();
    ASSERT_TRUE(first.ok());
    ASSERT_EQ(first->streams.size(), 1u);
    ASSERT_TRUE(stream->isPPInflight());
    auto pending = scheduler->schedule();
    ASSERT_TRUE(pending.ok());
    EXPECT_TRUE(pending->streams.empty());
    EXPECT_TRUE(pending->finished_request_ids.empty());
    stream->updateFromPP({torch::tensor({{4}}, torch::kInt32), 1});
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
    stream->updateFromPP({torch::tensor({{4}}, torch::kInt32), 1});
    stream->clearPPInflight();
    auto retired = scheduler->schedule();
    ASSERT_TRUE(retired.ok());
    EXPECT_TRUE(retired->streams.empty());
    EXPECT_EQ(retired->finished_request_ids, (std::vector<int64_t>{stream->streamId()}));
    EXPECT_TRUE(stream->streamCacheResource().isResourceReleased());
    EXPECT_EQ(stream->seqLength(), 3);
    /** Allow an idle round without enqueueing another request; retirement must not repeat. */
    scheduler->schedule_trigger_ = true;
    auto idle = scheduler->schedule();
    ASSERT_TRUE(idle.ok());
    EXPECT_TRUE(idle->streams.empty());
    EXPECT_TRUE(idle->finished_request_ids.empty());
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
