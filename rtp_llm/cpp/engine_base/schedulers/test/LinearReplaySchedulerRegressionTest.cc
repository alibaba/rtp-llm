#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "torch/all.h"

#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/engine_base/schedulers/BatchDecodeScheduler.h"
#include "rtp_llm/cpp/engine_base/schedulers/FIFOScheduler.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/testing/TestBase.h"

namespace rtp_llm {

class LinearReplaySchedulerRegressionTest: public DeviceTestBase {
protected:
    void SetUp() override {
        DeviceTestBase::SetUp();
        model_.num_layers                                    = 4;
        model_.max_seq_len                                   = 256;
        model_.vocab_size                                    = 1024;
        model_.special_tokens.eos_token_id                   = 999;
        model_.attn_config.tokens_per_block                  = 4;
        auto& linear                                         = model_.linear_attention_config;
        linear.linear_num_key_heads                          = 1;
        linear.linear_num_value_heads                        = 1;
        linear.linear_key_head_dim                           = 1;
        linear.linear_value_head_dim                         = 1;
        linear.linear_conv_kernel_dim                        = 2;
        linear.conv_state_dtype                              = DataType::TYPE_FP16;
        runtime_.enable_linear_replay                        = true;
        runtime_.max_generate_batch_size                     = 1;
        runtime_.fifo_scheduler_config.max_batch_tokens_size = 128;
        speculative_.type                                    = SP_TYPE_MTP;
        speculative_.gen_num_per_cycle                       = 3;
        parallelism_.role_type                               = RoleType::DECODE;
    }

    CacheConfig replayConfig() const {
        auto config = test::makeSimpleHybridMhaCacheConfig(4, 64, 4, DataType::TYPE_FP16, 2);
        CacheConfigCreator::configureLinearReplay(config, model_, parallelism_, runtime_, speculative_, true);
        return config;
    }

    ResourceContext resources(const std::shared_ptr<KVCacheManager>& manager, RoleType role) const {
        ResourceContext context;
        context.cache_manager = manager;
        context.role_type     = role;
        context.reuse_cache   = false;
        return context;
    }

    GenerateStreamPtr stream(int length, const ResourceContext& context) const {
        auto input                             = std::make_shared<GenerateInput>();
        input->input_ids                       = torch::ones({length}, torch::kInt32);
        input->generate_config                 = std::make_shared<GenerateConfig>();
        input->generate_config->max_new_tokens = 32;
        return std::make_shared<NormalGenerateStream>(input, model_, runtime_, context, nullptr);
    }

    GenerateStreamPtr pdDecodeStream(const ResourceContext& context) const {
        auto result = stream(6, context);
        // The Decode RPC initializes prompt KV before enqueueing, then includes
        // the Prefill-produced first token when switching to Decode execution.
        result->reportEvent(StreamEvents::CanRun);
        EXPECT_EQ(result->moveToNext(), StreamState::WAITING);
        EXPECT_TRUE(result->hasEvent(StreamEvents::LoadInitiated));
        result->setSeqLength(7);
        result->setIsContextStream(false);
        return result;
    }

    void checkFixedDecodeBootstrap(int prompt_length) {
        runtime_.use_batch_decode_scheduler                                      = true;
        runtime_.batch_decode_scheduler_config.batch_decode_scheduler_batch_size = 1;
        auto config                                                              = replayConfig();
        auto manager = std::make_shared<KVCacheManager>(config);
        ASSERT_TRUE(manager->init());
        BatchDecodeScheduler scheduler(runtime_, manager, nullptr);
        auto                 candidate = stream(prompt_length, resources(manager, RoleType::PDFUSION));
        ASSERT_TRUE(scheduler.enqueue(candidate).ok());
        auto scheduled = scheduler.schedule();
        ASSERT_TRUE(scheduled.ok()) << scheduled.status();
        ASSERT_EQ(scheduled->streams.size(), 1);
        EXPECT_EQ(scheduled->streams.front(), candidate);
        ASSERT_FALSE(candidate->hasError()) << candidate->stopReason();
        EXPECT_EQ(candidate->getStatus(), StreamState::RUNNING);
        EXPECT_EQ(candidate->seqLength(), prompt_length);
        EXPECT_EQ(candidate->getCompleteTokenIds()->completeTokenIdsVec(0), std::vector<int>(prompt_length, 1));
        EXPECT_TRUE(candidate->prepareLinearReplayRound().ok());
    }

    void checkRetiringSlotBackpressure(bool pending_bookkeeping, bool cancelled) {
        auto config = replayConfig();
        ASSERT_EQ(config.linear_replay.slotCount(), 1);
        auto manager = std::make_shared<KVCacheManager>(config);
        ASSERT_TRUE(manager->init());
        const auto  context = resources(manager, RoleType::DECODE);
        PDSepConfig pd;
        pd.role_type = RoleType::DECODE;
        FIFOScheduler scheduler(runtime_, model_, pd, parallelism_, manager);
        auto          old = pdDecodeStream(context);
        ASSERT_TRUE(scheduler.enqueue(old).ok());
        auto initial = scheduler.schedule();
        ASSERT_TRUE(initial.ok()) << initial.status();
        ASSERT_EQ(initial->streams.size(), 1);
        ASSERT_EQ(initial->streams.front(), old);
        ASSERT_FALSE(old->hasError()) << old->stopReason();

        std::shared_ptr<void> bookkeeping;
        if (pending_bookkeeping) {
            old->incPendingAsyncBookkeeping();
            bookkeeping =
                std::shared_ptr<void>(nullptr, [old](void*) { old->decPendingAsyncBookkeepingAndMaybeRelease(); });
        }
        auto round = old->prepareLinearReplayRound();
        ASSERT_TRUE(round.ok()) << round.status();
        auto waiting = pdDecodeStream(context);
        ASSERT_TRUE(scheduler.enqueue(waiting).ok());
        if (cancelled) {
            old->reportError(ErrorCode::CANCELLED, "cancel while replay round owns resources");
        } else {
            old->reportEvent(StreamEvents::GenerateDone);
        }

        auto blocked = scheduler.schedule();
        ASSERT_TRUE(blocked.ok()) << blocked.status();
        EXPECT_EQ(old->getStatus(), StreamState::FINISHED);
        ASSERT_FALSE(waiting->hasError()) << waiting->stopReason();
        EXPECT_TRUE(blocked->streams.empty());
        EXPECT_EQ(waiting->getStatus(), StreamState::WAITING);
        EXPECT_EQ(scheduler.waitingStreamsSize(), 1);

        // Finishing host bookkeeping alone cannot recycle a log slot still
        // retained by the dispatched round. Neither ownership may be bypassed.
        bookkeeping.reset();
        auto still_blocked = scheduler.schedule();
        ASSERT_TRUE(still_blocked.ok()) << still_blocked.status();
        ASSERT_FALSE(waiting->hasError()) << waiting->stopReason();
        EXPECT_TRUE(still_blocked->streams.empty());
        round = absl::FailedPreconditionError("round completed");

        auto resumed = scheduler.schedule();
        ASSERT_TRUE(resumed.ok()) << resumed.status();
        ASSERT_EQ(resumed->streams.size(), 1);
        EXPECT_EQ(resumed->streams.front(), waiting);
        EXPECT_FALSE(waiting->hasError()) << waiting->stopReason();
        EXPECT_EQ(waiting->getStatus(), StreamState::RUNNING);
    }

    void checkUnusedReplayReservationDoesNotBlockPreparedStream(bool grouped) {
        // A two-member explicit group must fit the actual scheduler capacity.
        // One completed round still owns one of the two replay slots.
        runtime_.max_generate_batch_size = 2;
        parallelism_.role_type           = RoleType::PDFUSION;
        auto config                      = replayConfig();
        auto groups                      = config.topology().groups();
        for (auto& group : groups) {
            if (group.policy.group_type == CacheGroupType::FULL) {
                group.block_num = 5;  // Four usable blocks after the reserved zero block.
            }
        }
        config.setTopology(std::move(groups), config.topology().layers());
        ASSERT_EQ(config.linear_replay.slotCount(), 2);
        KVCacheConfig options;
        options.reserve_block_ratio                          = 0;
        options.block_tree_device_evict_low_watermark_ratio  = 0;
        options.block_tree_device_evict_high_watermark_ratio = 0;
        PDSepConfig pd;
        pd.role_type = RoleType::PDFUSION;
        auto manager =
            std::make_shared<KVCacheManager>(config, false, nullptr, options, parallelism_, runtime_, speculative_, pd);
        ASSERT_TRUE(manager->init());
        DeviceBlockPoolPtr full_pool;
        for (const auto& group_set : manager->blockTreeCache()->groupSets()) {
            if (group_set->groupType() == CacheGroupType::FULL) {
                ASSERT_EQ(group_set->devicePools().size(), 1);
                full_pool = group_set->devicePools().front();
            }
        }
        ASSERT_NE(full_pool, nullptr);
        ASSERT_EQ(full_pool->totalBlocksNum(), 4);
        const auto    context = resources(manager, RoleType::PDFUSION);
        FIFOScheduler scheduler(runtime_, model_, pd, parallelism_, manager);
        auto          old = stream(2, context);
        ASSERT_TRUE(scheduler.enqueue(old).ok());
        auto initial = scheduler.schedule();
        ASSERT_TRUE(initial.ok()) << initial.status();
        ASSERT_EQ(initial->streams.size(), 1);
        old->setSeqLength(3);
        old->setIsContextStream(false);
        ASSERT_EQ(old->moveToNext(), StreamState::RUNNING);
        auto round = old->prepareLinearReplayRound();
        ASSERT_TRUE(round.ok()) << round.status();
        old->reportEvent(StreamEvents::GenerateDone);
        auto retired = scheduler.schedule();
        ASSERT_TRUE(retired.ok()) << retired.status();
        ASSERT_TRUE(retired->streams.empty());
        ASSERT_EQ(old->getStatus(), StreamState::FINISHED);
        ASSERT_EQ(full_pool->freeBlocksNum(), 4);

        auto waiting  = stream(8, context);
        auto prepared = stream(9, context);
        // Exercise the real pre-enqueue KV initialization entry used by the
        // Decode RPC, with a consistent PDFUSION resource/scheduler role.
        // Nine tokens hold three FULL blocks, but do not reserve a replay slot
        // while the stream is still in its context phase.
        ASSERT_TRUE(prepared->streamCacheResource().initKVBlock().ok());
        ASSERT_EQ(full_pool->freeBlocksNum(), 1);
        ASSERT_EQ(prepared->streamCacheResource().linearReplayLease(), nullptr);
        // Eight tokens need only two of the four usable FULL blocks. The
        // shortage is temporary, and failed allocation must roll back its KV.
        const auto allocation = waiting->streamCacheResource().initKVBlock();
        ASSERT_TRUE(absl::IsUnavailable(allocation)) << allocation;
        ASSERT_EQ(waiting->curBlocksNum(), 0);
        ASSERT_EQ(full_pool->freeBlocksNum(), 1);
        if (grouped) {
            const auto enqueued = scheduler.enqueueGroup({waiting, prepared});
            ASSERT_EQ(enqueued.first.size(), 2);
            ASSERT_TRUE(enqueued.first[0]);
            ASSERT_TRUE(enqueued.first[1]);
        } else {
            ASSERT_TRUE(scheduler.enqueue(waiting).ok());
            ASSERT_TRUE(scheduler.enqueue(prepared).ok());
        }
        auto scheduled = scheduler.schedule();
        ASSERT_TRUE(scheduled.ok()) << scheduled.status();
        EXPECT_EQ(scheduled->streams.size(), 1);
        if (!scheduled->streams.empty()) {
            EXPECT_EQ(scheduled->streams.front(), prepared);
        }
        EXPECT_EQ(waiting->getStatus(), StreamState::WAITING);
        EXPECT_FALSE(waiting->hasError()) << waiting->stopReason();
        EXPECT_EQ(waiting->curBlocksNum(), 0);
        EXPECT_FALSE(waiting->hasEvent(StreamEvents::LoadInitiated));
        EXPECT_EQ(waiting->streamCacheResource().linearReplayLease(), nullptr);
        EXPECT_EQ(prepared->getStatus(), StreamState::RUNNING);
        EXPECT_FALSE(prepared->hasError()) << prepared->stopReason();
        EXPECT_NE(prepared->streamCacheResource().linearReplayLease(), nullptr);
        EXPECT_EQ(full_pool->freeBlocksNum(), 1);
        EXPECT_EQ(scheduler.waitingStreamsSize(), 1);
    }

    ModelConfig                model_;
    RuntimeConfig              runtime_;
    ParallelismConfig          parallelism_;
    SpeculativeExecutionConfig speculative_;
};

TEST_F(LinearReplaySchedulerRegressionTest, CancelledPendingRoundDoesNotFailQueuedDecode) {
    checkRetiringSlotBackpressure(/*pending_bookkeeping=*/true, /*cancelled=*/true);
}

TEST_F(LinearReplaySchedulerRegressionTest, FinishedRoundOwnershipDoesNotFailQueuedDecode) {
    checkRetiringSlotBackpressure(/*pending_bookkeeping=*/false, /*cancelled=*/false);
}

TEST_F(LinearReplaySchedulerRegressionTest, RetryableKvShortageDoesNotPinUnusedReplayReservation) {
    checkUnusedReplayReservationDoesNotBlockPreparedStream(/*grouped=*/false);
}

TEST_F(LinearReplaySchedulerRegressionTest, RetryableKvShortageDoesNotPinUnusedGroupReplayReservation) {
    checkUnusedReplayReservationDoesNotBlockPreparedStream(/*grouped=*/true);
}

TEST_F(LinearReplaySchedulerRegressionTest, TerminalStreamCannotReacquireReplayReservation) {
    auto manager = std::make_shared<KVCacheManager>(replayConfig());
    ASSERT_TRUE(manager->init());
    auto cancelled = stream(6, resources(manager, RoleType::PDFUSION));
    cancelled->reportError(ErrorCode::CANCELLED, "cancel before replay reservation");
    ASSERT_EQ(cancelled->moveToNext(), StreamState::FINISHED);
    ASSERT_TRUE(cancelled->streamCacheResource().isResourceReleased());
    // Cancellation can finish between the scheduler's precheck and reserve.
    EXPECT_FALSE(cancelled->streamCacheResource().reserveLinearReplaySlot());
    EXPECT_EQ(cancelled->streamCacheResource().linearReplayLease(), nullptr);
    EXPECT_NE(manager->acquireLinearReplaySlot(), nullptr);
}

TEST_F(LinearReplaySchedulerRegressionTest, FixedDecodeBootstrapKeepsBoundaryCanonicalState) {
    checkFixedDecodeBootstrap(/*prompt_length=*/9);
}

TEST_F(LinearReplaySchedulerRegressionTest, FixedDecodeBootstrapSupportsNonBoundaryPrompt) {
    checkFixedDecodeBootstrap(/*prompt_length=*/10);
}

TEST_F(LinearReplaySchedulerRegressionTest, PipelineExecutorKeepsLegacyLinearCacheContract) {
    ASSERT_TRUE(replayConfig().linear_replay.enabled());
    parallelism_.pp_size = 2;
    // PPExecutor has no LinearReplayRound producer, so publishing a log pool
    // there would require metadata its existing attention path never receives.
    auto config = replayConfig();
    EXPECT_FALSE(config.linear_replay.enabled());
    EXPECT_TRUE(config.isLegacyLinearGroup(static_cast<int>(config.topology().groupIdForTag("linear"))));
}

TEST_F(LinearReplaySchedulerRegressionTest, FixedBatchCapacityIncludesItsConfiguredBatchSize) {
    runtime_.use_batch_decode_scheduler                                      = true;
    runtime_.batch_decode_scheduler_config.batch_decode_scheduler_batch_size = 2;
    auto config                                                              = replayConfig();
    EXPECT_GE(config.linear_replay.slotCount(), 2);
    auto manager = std::make_shared<KVCacheManager>(config);
    ASSERT_TRUE(manager->init());
    BatchDecodeScheduler scheduler(runtime_, manager, nullptr);
    const auto           context = resources(manager, RoleType::PDFUSION);
    auto                 first   = stream(6, context);
    auto                 second  = stream(6, context);
    ASSERT_TRUE(scheduler.enqueue(first).ok());
    ASSERT_TRUE(scheduler.enqueue(second).ok());
    auto scheduled = scheduler.schedule();
    ASSERT_TRUE(scheduled.ok()) << scheduled.status();
    ASSERT_EQ(scheduled->streams.size(), 2);
    for (const auto& candidate : scheduled->streams) {
        EXPECT_FALSE(candidate->hasError()) << candidate->stopReason();
        EXPECT_EQ(candidate->getStatus(), StreamState::RUNNING);
        EXPECT_TRUE(candidate->prepareLinearReplayRound().ok());
    }
}

TEST_F(LinearReplaySchedulerRegressionTest, PdfusionAdmissionReservesReplayBeforePrefill) {
    for (bool grouped : {false, true}) {
        SCOPED_TRACE(grouped);
        auto config  = replayConfig();
        auto manager = std::make_shared<KVCacheManager>(config);
        ASSERT_TRUE(manager->init());
        const auto  context = resources(manager, RoleType::PDFUSION);
        PDSepConfig pd;
        pd.role_type = RoleType::PDFUSION;
        FIFOScheduler scheduler(runtime_, model_, pd, parallelism_, manager);
        auto          old = stream(6, context);
        ASSERT_TRUE(scheduler.enqueue(old).ok());
        auto initial = scheduler.schedule();
        ASSERT_TRUE(initial.ok()) << initial.status();
        ASSERT_EQ(initial->streams.size(), 1);
        ASSERT_TRUE(old->isContextStream());
        ASSERT_NE(old->streamCacheResource().linearReplayLease(), nullptr);

        // Prefill has committed its first output; the first Decode round now
        // retains the lease even after the scheduler releases the old stream.
        old->setSeqLength(7);
        old->setIsContextStream(false);
        ASSERT_EQ(old->moveToNext(), StreamState::RUNNING);
        auto round = old->prepareLinearReplayRound();
        ASSERT_TRUE(round.ok()) << round.status();
        old->reportEvent(StreamEvents::GenerateDone);
        auto waiting = stream(6, context);
        if (grouped) {
            const auto enqueued = scheduler.enqueueGroup({waiting});
            ASSERT_EQ(enqueued.first.size(), 1);
            ASSERT_TRUE(enqueued.first.front());
        } else {
            ASSERT_TRUE(scheduler.enqueue(waiting).ok());
        }
        auto blocked = scheduler.schedule();
        ASSERT_TRUE(blocked.ok()) << blocked.status();
        EXPECT_TRUE(blocked->streams.empty());
        EXPECT_EQ(old->getStatus(), StreamState::FINISHED);
        EXPECT_EQ(waiting->getStatus(), StreamState::WAITING);
        EXPECT_FALSE(waiting->hasError()) << waiting->stopReason();
        EXPECT_EQ(waiting->curBlocksNum(), 0);
        EXPECT_EQ(scheduler.waitingStreamsSize(), 1);

        round        = absl::FailedPreconditionError("round completed");
        auto resumed = scheduler.schedule();
        ASSERT_TRUE(resumed.ok()) << resumed.status();
        ASSERT_EQ(resumed->streams.size(), 1);
        EXPECT_EQ(resumed->streams.front(), waiting);
        EXPECT_EQ(waiting->getStatus(), StreamState::RUNNING);
        EXPECT_TRUE(waiting->isContextStream());
        EXPECT_FALSE(waiting->hasError()) << waiting->stopReason();
        EXPECT_NE(waiting->streamCacheResource().linearReplayLease(), nullptr);
    }
}

TEST_F(LinearReplaySchedulerRegressionTest, FixedBatchWaitsForEveryReplayReservation) {
    runtime_.use_batch_decode_scheduler                                      = true;
    runtime_.batch_decode_scheduler_config.batch_decode_scheduler_batch_size = 2;
    auto config                                                              = replayConfig();
    auto manager                                                             = std::make_shared<KVCacheManager>(config);
    ASSERT_TRUE(manager->init());
    const auto           context = resources(manager, RoleType::PDFUSION);
    BatchDecodeScheduler scheduler(runtime_, manager, nullptr);
    auto                 first  = stream(6, context);
    auto                 second = stream(6, context);
    ASSERT_TRUE(scheduler.enqueue(first).ok());
    ASSERT_TRUE(scheduler.enqueue(second).ok());
    auto initial = scheduler.schedule();
    ASSERT_TRUE(initial.ok()) << initial.status();
    ASSERT_EQ(initial->streams.size(), 2);
    auto round = first->prepareLinearReplayRound();
    ASSERT_TRUE(round.ok()) << round.status();
    first->reportEvent(StreamEvents::GenerateDone);
    second->reportEvent(StreamEvents::GenerateDone);
    auto next_first  = stream(6, context);
    auto next_second = stream(6, context);
    ASSERT_TRUE(scheduler.enqueue(next_first).ok());
    ASSERT_TRUE(scheduler.enqueue(next_second).ok());

    // Only one slot is free. Reserving it must not dispatch half a fixed batch
    // or turn the other member's temporary slot shortage into a malloc error.
    auto blocked = scheduler.schedule();
    ASSERT_TRUE(blocked.ok()) << blocked.status();
    EXPECT_TRUE(blocked->streams.empty());
    for (const auto& waiting : {next_first, next_second}) {
        EXPECT_EQ(waiting->getStatus(), StreamState::WAITING);
        EXPECT_FALSE(waiting->hasError()) << waiting->stopReason();
        EXPECT_EQ(waiting->curBlocksNum(), 0);
    }
    round        = absl::FailedPreconditionError("round completed");
    auto resumed = scheduler.schedule();
    ASSERT_TRUE(resumed.ok()) << resumed.status();
    ASSERT_EQ(resumed->streams.size(), 2);
    for (const auto& candidate : resumed->streams) {
        EXPECT_EQ(candidate->getStatus(), StreamState::RUNNING);
        EXPECT_FALSE(candidate->hasError()) << candidate->stopReason();
        EXPECT_TRUE(candidate->prepareLinearReplayRound().ok());
    }
}

TEST_F(LinearReplaySchedulerRegressionTest, FixedDecodeBootstrapSupportsSingleTokenPrompt) {
    checkFixedDecodeBootstrap(/*prompt_length=*/1);
}

TEST_F(LinearReplaySchedulerRegressionTest, FixedDecodeKeepsLegacyAndFullCacheBootstrap) {
    runtime_.use_batch_decode_scheduler                                      = true;
    runtime_.batch_decode_scheduler_config.batch_decode_scheduler_batch_size = 1;
    for (bool full_only : {false, true}) {
        SCOPED_TRACE(full_only);
        runtime_.enable_linear_replay = full_only;
        auto config = full_only ? test::makeSimpleMhaCacheConfig(4, 64, 4, DataType::TYPE_FP16) : replayConfig();
        CacheConfigCreator::configureLinearReplay(config, model_, parallelism_, runtime_, speculative_, true);
        ASSERT_FALSE(config.linear_replay.enabled());
        auto manager = std::make_shared<KVCacheManager>(config);
        ASSERT_TRUE(manager->init());
        BatchDecodeScheduler scheduler(runtime_, manager, nullptr);
        auto                 candidate = stream(9, resources(manager, RoleType::PDFUSION));
        ASSERT_TRUE(scheduler.enqueue(candidate).ok());
        auto scheduled = scheduler.schedule();
        ASSERT_TRUE(scheduled.ok()) << scheduled.status();
        ASSERT_EQ(scheduled->streams.size(), 1);
        EXPECT_FALSE(candidate->hasError()) << candidate->stopReason();
        EXPECT_EQ(candidate->getStatus(), StreamState::RUNNING);
        EXPECT_EQ(candidate->seqLength(), 9);
        EXPECT_EQ(candidate->streamCacheResource().linearReplayLease(), nullptr);
    }
}

TEST_F(LinearReplaySchedulerRegressionTest, ReplayTailPrivatizationCanReclaimEvictableCache) {
    for (const bool spare : {false, true}) {
        SCOPED_TRACE(spare);
        auto config          = test::makeSimpleLinearCacheConfig(1, spare ? 3 : 2, 4, DataType::TYPE_FP16, 1, 1);
        config.linear_replay = LinearReplayPlan::create({0}, 1, 4, {1, 1}, {1, 1}, false, DataType::TYPE_FP16, 1);
        KVCacheConfig options;
        options.block_tree_device_evict_low_watermark_ratio  = 0;
        options.block_tree_device_evict_high_watermark_ratio = 0;
        auto manager = std::make_shared<KVCacheManager>(config, false, nullptr, options);
        ASSERT_TRUE(manager->init());
        const auto cache = manager->blockTreeCache();
        ASSERT_NE(cache, nullptr);
        ASSERT_EQ(cache->groupSets().size(), 1);
        const auto pool   = cache->groupSets().front()->devicePools().front();
        auto       blocks = pool->malloc(spare ? 2 : 1);
        ASSERT_TRUE(blocks.has_value());
        pool->incRef(*blocks);

        // One immutable Prefill checkpoint is shared with the active request.
        // A second, unrelated cached checkpoint is the reclaimable spare block.
        std::vector<std::vector<GroupSetResource>> cached(1, std::vector<GroupSetResource>(1));
        cached[0][0].device_blocks = {blocks->at(0)};
        cache->insert({100}, cached, Tier::DEVICE);
        if (spare) {
            cached[0][0].device_blocks = {blocks->at(1)};
            cache->insert({200}, cached, Tier::DEVICE);
            pool->decRef(blocks->at(1));
        }
        ASSERT_EQ(pool->freeBlocksNum(), 0);

        auto resource = std::make_shared<BatchKVCacheResource>();
        resource->resetBatchSize(1);
        resource->initGroups(config.topologyPtr());
        resource->setBatchBlocks(0, "linear", {blocks->at(0)});
        const auto release_request =
            std::shared_ptr<void>(nullptr, [pool, resource](void*) { pool->decRef(resource->blocks(0, "linear")); });
        const bool privatized = manager->makeLinearReplayTailsPrivate(resource, 0);
        EXPECT_TRUE(privatized);
        if (privatized) {
            EXPECT_TRUE(pool->isExclusiveRequestBlock(resource->blocks(0, "linear").front()));
        } else {
            // Demonstrate that the failed COW was not a true capacity shortage.
            cache->evictForGroup("linear", 1);
            EXPECT_TRUE(spare ? pool->freeBlocksNum() > 0 : pool->isExclusiveRequestBlock(blocks->at(0)));
        }
    }
}

}  // namespace rtp_llm
