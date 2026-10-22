
#include "gtest/gtest.h"
#include <thread>

#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/engine_base/stream/StreamGroups.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/models/logits_processor/ThinkModeLogitsProcessor.h"

using namespace std;

namespace rtp_llm {

TEST(GenerateConfigCaptureTest, JsonRoundTripPreservesIndependentCaptureIntent) {
    GenerateConfig config;
    EXPECT_FALSE(config.capture_hidden_states);
    autil::legacy::FromJsonString(config, "{\"max_new_tokens\":4,\"capture_hidden_states\":true,\"reuse_cache\":true}");
    EXPECT_NO_THROW(config.validatePrefillOnly());
    EXPECT_TRUE(config.capture_hidden_states);
    EXPECT_TRUE(config.reuse_cache);
    GenerateConfig copy;
    autil::legacy::FromJsonString(copy, autil::legacy::ToJsonString(config));
    EXPECT_TRUE(copy.capture_hidden_states);
    EXPECT_TRUE(copy.reuse_cache);
    EXPECT_EQ(copy.max_new_tokens, 4);
    EXPECT_NE(copy.debugString().find("capture_hidden_states: 1"), std::string::npos);
}

class GenerateStreamBuilder {
public:
    GenerateStreamBuilder() {
        model_config_.max_seq_len = 2048;
        model_config_.vocab_size  = 128;
    }

    CacheConfig init_config() {
        return test::makeSimpleMhaCacheConfig(
            /*layer_num=*/3, /*block_num=*/9, /*tokens_per_block=*/2, rtp_llm::DataType::TYPE_INT8);
    }

    GenerateStreamPtr createContextStream(std::vector<int>                input_ids,
                                          std::shared_ptr<GenerateConfig> generate_config = nullptr) {
        std::shared_ptr<GenerateInput> generate_input(new GenerateInput());
        ResourceContext                resource_context;
        if (generate_config == nullptr) {
            generate_config.reset(new GenerateConfig());
        }
        generate_input->generate_config = generate_config;
        generate_input->input_ids =
            torch::tensor(std::vector<int32_t>(input_ids.begin(), input_ids.end()), torch::kInt32);
        return std::make_shared<NormalGenerateStream>(
            generate_input, model_config_, runtime_config_, resource_context, nullptr);
    };

    GenerateStreamPtr createComplexContextStream(std::vector<int> input_ids, int num_return_sequences = 2) {
        autil::EnvGuard perf_scope("PERF_TEST", "1");

        auto cache_config  = init_config();
        auto cache_manager = std::make_shared<KVCacheManager>(cache_config);
        cache_manager->init();
        ResourceContext resource_context;
        resource_context.cache_manager = cache_manager;
        resource_context.reuse_cache   = true;

        std::shared_ptr<GenerateInput>  generate_input(new GenerateInput());
        std::shared_ptr<GenerateConfig> generate_config(new GenerateConfig());
        generate_config->num_return_sequences = num_return_sequences;
        generate_input->input_ids =
            torch::tensor(std::vector<int32_t>(input_ids.begin(), input_ids.end()), torch::kInt32);
        generate_input->generate_config = generate_config;
        ModelConfig   model_config;
        RuntimeConfig runtime_config;
        model_config.max_seq_len = 2048;
        auto stream              = std::make_shared<NormalGenerateStream>(
            generate_input, model_config, runtime_config, resource_context, nullptr);

        return stream;
    }

    GenerateStreamPtr createDecoderStream(std::vector<int> input_ids, std::vector<int> new_token_ids) {
        std::shared_ptr<GenerateInput>  generate_input(new GenerateInput());
        std::shared_ptr<GenerateConfig> generate_config(new GenerateConfig());
        ResourceContext                 resource_context;
        generate_input->generate_config = generate_config;
        generate_input->input_ids =
            torch::tensor(std::vector<int32_t>(input_ids.begin(), input_ids.end()), torch::kInt32);
        auto stream_ptr = std::make_shared<NormalGenerateStream>(
            generate_input, model_config_, runtime_config_, resource_context, nullptr);
        stream_ptr->setIsContextStream(false);
        auto complete_ids = stream_ptr->completeTokenIds();
        std::memcpy(complete_ids.data_ptr<int32_t>() + stream_ptr->seqLength(),
                    new_token_ids.data(),
                    new_token_ids.size() * sizeof(int));
        stream_ptr->setSeqLength(stream_ptr->seqLength() + new_token_ids.size());
        return stream_ptr;
    };

private:
    ModelConfig   model_config_;
    RuntimeConfig runtime_config_;
};

class GenerateStreamTest: public DeviceTestBase {
protected:
};

static GenerateStreamPtr makeChunkStream(int      input_length,
                                         RoleType role                  = RoleType::PREFILL,
                                         int      max_new_tokens        = 8,
                                         bool     capture_hidden_states = false,
                                         bool     reuse_cache           = false,
                                         int      chunk_size            = 16,
                                         bool     global_reuse          = true) {
    auto input                                    = std::make_shared<GenerateInput>();
    input->input_ids                              = torch::arange(input_length, torch::kInt32);
    input->generate_config                        = std::make_shared<GenerateConfig>();
    input->generate_config->max_new_tokens        = max_new_tokens;
    input->generate_config->capture_hidden_states = capture_hidden_states;
    input->generate_config->reuse_cache           = reuse_cache;
    input->generate_config->ignore_eos            = true;
    ModelConfig model;
    model.max_seq_len = 128;
    model.vocab_size  = 128;
    RuntimeConfig runtime;
    runtime.fifo_scheduler_config.prefill_chunk_size    = chunk_size;
    runtime.fifo_scheduler_config.max_batch_tokens_size = 8;
    ResourceContext resource;
    resource.role_type   = role;
    resource.reuse_cache = global_reuse;
    return std::make_shared<NormalGenerateStream>(input, model, runtime, resource, nullptr);
}

TEST_F(GenerateStreamTest, ChunkPrefillWindowDoesNotPublishIntermediateTokens) {
    for (bool speculative : {false, true}) {
        SCOPED_TRACE(speculative);
        auto stream = makeChunkStream(18);
        ASSERT_EQ(stream->maxChunkSize(), 8);
        EXPECT_EQ(stream->contextLength(), 18);
        stream->setReuseLength(8);
        stream->setInitialReuseLength(8);
        stream->generate_status_->status = StreamState::RUNNING;
        EXPECT_EQ(stream->currentExecuteTokens(), (std::vector<int>{8, 9, 10, 11, 12, 13, 14, 15}));
        EXPECT_EQ(stream->nextChunkBoundaryToken(), 16);
        EXPECT_EQ(stream->cacheStorePublishStartToken(), 0);
        auto token = torch::tensor({{101}}, torch::kInt32);
        if (speculative) {
            stream->specUpdate({token, 1, 42, torch::Tensor(), torch::Tensor()});
        } else {
            stream->update({token, 1});
        }
        EXPECT_TRUE(stream->isContextStream());
        EXPECT_EQ(stream->seqLength(), 18);
        EXPECT_EQ(stream->reuseLength(), 16);
        EXPECT_EQ(stream->initialReuseLength(), 8);
        EXPECT_EQ(stream->cacheStorePublishStartToken(), 16);
        EXPECT_EQ(stream->currentExecuteTokens(), (std::vector<int>{16, 17}));
        ASSERT_TRUE(stream->isLastChunk());
        stream->update({token, 1});
        EXPECT_FALSE(stream->isContextStream());
        EXPECT_EQ(stream->seqLength(), 19);
    }
}

TEST_F(GenerateStreamTest, ChunkPrefillActivationAndCancellation) {
    for (auto role : {RoleType::PREFILL, RoleType::PDFUSION, RoleType::DECODE}) {
        for (int max_new_tokens : {0, 8}) {
            auto stream = makeChunkStream(18, role, max_new_tokens);
            const bool rejected = role != RoleType::DECODE && max_new_tokens == 0;
            EXPECT_EQ(stream->hasError(), rejected);
            EXPECT_EQ(stream->chunkedPrefillEnabled(), role != RoleType::DECODE && !rejected);
        }
    }
    auto stream                      = makeChunkStream(18);
    stream->generate_status_->status = StreamState::RUNNING;
    ASSERT_TRUE(stream->isMiddleChunk());
    stream->reportError(ErrorCode::CANCELLED, "cancel during chunked prefill");
    stream->update({torch::tensor({{101}}, torch::kInt32), 1});
    EXPECT_EQ(stream->reuseLength(), 0);
    EXPECT_EQ(stream->seqLength(), 18);
    EXPECT_TRUE(stream->hasError());
}

TEST_F(GenerateStreamTest, FullContextCaptureRejectsChunkModeBeforeReservation) {
    for (auto role : {RoleType::PREFILL, RoleType::PDFUSION, RoleType::DECODE}) {
        for (int max_new_tokens : {0, 8}) {
            for (bool capture : {false, true}) {
                for (bool reuse : {false, true}) {
                    for (bool global_reuse : {false, true}) {
                        for (int chunk_size : {0, 16}) {
                            SCOPED_TRACE(::testing::Message()
                                         << "role=" << static_cast<int>(role) << " max_new_tokens=" << max_new_tokens
                                         << " capture=" << capture << " reuse=" << reuse
                                         << " global_reuse=" << global_reuse << " chunk_size=" << chunk_size);
                            auto stream =
                                makeChunkStream(18, role, max_new_tokens, capture, reuse, chunk_size, global_reuse);
                            const bool effective_reuse = global_reuse && reuse && max_new_tokens != 0;
                            const bool full_context =
                                role != RoleType::DECODE && (max_new_tokens == 0 || (capture && !effective_reuse));
                            EXPECT_EQ(stream->streamCacheResource().reuseCache(), effective_reuse);
                            ASSERT_EQ(stream->requiresFullContextForHiddenStateCapture(), full_context);
                            const bool rejected = chunk_size > 0 && full_context;
                            ASSERT_EQ(stream->hasError(), rejected);
                            EXPECT_EQ(stream->chunkedPrefillEnabled(),
                                      chunk_size > 0 && role != RoleType::DECODE && !rejected);
                            EXPECT_EQ(stream->isHiddenStateCaptureCandidate(), full_context && !rejected);
                            EXPECT_EQ(stream->generateConfig()->reuse_cache, reuse);
                            if (rejected) {
                                EXPECT_EQ(stream->statusInfo().code(), ErrorCode::INVALID_PARAMS);
                                EXPECT_EQ(stream->stopReason(),
                                          "chunked prefill incompatible: full-context hidden-state capture");
                                EXPECT_FALSE(stream->hasHiddenStateCaptureReservation());
                                EXPECT_EQ(stream->contextLength(), 18);
                                EXPECT_EQ(stream->moveToNext(), StreamState::FINISHED);
                            } else if (full_context) {
                                EXPECT_EQ(stream->contextLength(), stream->inputLength());
                                stream->setReuseLength(1);
                                EXPECT_FALSE(stream->isHiddenStateCaptureCandidate());
                                stream->setReuseLength(0);
                                EXPECT_TRUE(stream->isHiddenStateCaptureCandidate());
                                stream->setSeqLength(stream->inputLength() + 1);
                                EXPECT_FALSE(stream->isHiddenStateCaptureCandidate());
                            }
                        }
                    }
                }
            }
        }
    }
}

TEST_F(GenerateStreamTest, UnexecutedCaptureReservationIsCancelledOnDestruction) {
    auto cancelled = std::make_shared<std::atomic<int>>(0);
    {
        auto stream = GenerateStreamBuilder().createContextStream({1, 2});
        stream->setHiddenStateCaptureReservation(
            std::make_shared<HiddenStateCaptureReservation>([cancelled] { ++*cancelled; }));
        EXPECT_TRUE(stream->hasHiddenStateCaptureReservation());
    }
    EXPECT_EQ(cancelled->load(), 1);
}

TEST_F(GenerateStreamTest, CaptureCancellationIsIdempotentAcrossConcurrentCallers) {
    auto cancelled = std::make_shared<std::atomic<int>>(0);
    auto stream    = GenerateStreamBuilder().createContextStream({1, 2});
    stream->setHiddenStateCaptureReservation(
        std::make_shared<HiddenStateCaptureReservation>([cancelled] { ++*cancelled; }));
    std::thread first([&] { stream->cancelHiddenStateCapture(); });
    std::thread second([&] { stream->cancelHiddenStateCapture(); });
    first.join();
    second.join();
    stream->releaseResource();
    EXPECT_FALSE(stream->hasHiddenStateCaptureReservation());
    EXPECT_EQ(cancelled->load(), 1);
}

TEST_F(GenerateStreamTest, CaptureCancellationFailureDoesNotBecomeAServingError) {
    auto stream = GenerateStreamBuilder().createContextStream({1, 2});
    stream->setHiddenStateCaptureReservation(
        std::make_shared<HiddenStateCaptureReservation>([] { throw std::runtime_error("capture owner unavailable"); }));
    EXPECT_NO_THROW(stream->cancelHiddenStateCapture());
    EXPECT_FALSE(stream->hasHiddenStateCaptureReservation());
    EXPECT_FALSE(stream->hasError());
}

TEST_F(GenerateStreamTest, CaptureReservationIsSharedWithoutWrappingItsCallback) {
    auto cancelled   = std::make_shared<int>(0);
    auto reservation = std::make_shared<HiddenStateCaptureReservation>([cancelled] { ++*cancelled; });
    auto stream      = GenerateStreamBuilder().createContextStream({1, 2});
    stream->setHiddenStateCaptureReservation(reservation);
    EXPECT_EQ(reservation.use_count(), 2);
    EXPECT_TRUE(stream->hasHiddenStateCaptureReservation());
    reservation->cancel();
    EXPECT_FALSE(stream->hasHiddenStateCaptureReservation());
    stream->cancelHiddenStateCapture();
    stream.reset();
    EXPECT_EQ(*cancelled, 1);
}

TEST_F(GenerateStreamTest, ReplacingCaptureReservationRollsBackOnlyThePreviousToken) {
    auto old_cancelled = std::make_shared<int>(0);
    auto new_cancelled = std::make_shared<int>(0);
    auto stream        = GenerateStreamBuilder().createContextStream({1, 2});
    stream->setHiddenStateCaptureReservation(
        std::make_shared<HiddenStateCaptureReservation>([old_cancelled] { ++*old_cancelled; }));
    stream->setHiddenStateCaptureReservation(
        std::make_shared<HiddenStateCaptureReservation>([new_cancelled] { ++*new_cancelled; }));
    EXPECT_EQ(*old_cancelled, 1);
    EXPECT_EQ(*new_cancelled, 0);
    EXPECT_TRUE(stream->hasHiddenStateCaptureReservation());
    stream->setHiddenStateCaptureReservation(nullptr);
    EXPECT_EQ(*new_cancelled, 1);
    EXPECT_FALSE(stream->hasHiddenStateCaptureReservation());
}

TEST_F(GenerateStreamTest, DraftCopyDoesNotOwnTargetCaptureReservation) {
    auto target_cancelled = std::make_shared<int>(0);
    auto draft_cancelled  = std::make_shared<int>(0);
    auto target           = GenerateStreamBuilder().createContextStream({1, 2});
    auto draft            = GenerateStreamBuilder().createContextStream({1, 2});
    target->setHiddenStateCaptureReservation(
        std::make_shared<HiddenStateCaptureReservation>([target_cancelled] { ++*target_cancelled; }));
    draft->setHiddenStateCaptureReservation(
        std::make_shared<HiddenStateCaptureReservation>([draft_cancelled] { ++*draft_cancelled; }));
    draft->CopyOnWrite(*target, false, true);
    EXPECT_EQ(*draft_cancelled, 1);
    EXPECT_FALSE(draft->hasHiddenStateCaptureReservation());
    draft.reset();
    EXPECT_TRUE(target->hasHiddenStateCaptureReservation());
    EXPECT_EQ(*target_cancelled, 0);
    target->cancelHiddenStateCapture();
    EXPECT_EQ(*target_cancelled, 1);
}

TEST(HiddenStateCaptureReservationTest, CancellationAndDestructionIsolateAllCallbackExceptions) {
    for (bool explicit_cancel : {false, true}) {
        for (bool standard_exception : {false, true}) {
            SCOPED_TRACE(explicit_cancel);
            SCOPED_TRACE(standard_exception);
            int calls = 0;
            {
                HiddenStateCaptureReservation reservation([&] {
                    ++calls;
                    if (standard_exception) {
                        throw std::runtime_error("capture cancellation failed");
                    }
                    throw 42;
                });
                EXPECT_TRUE(reservation.active());
                if (explicit_cancel) {
                    EXPECT_NO_THROW(reservation.cancel());
                    EXPECT_FALSE(reservation.active());
                    EXPECT_NO_THROW(reservation.cancel());
                }
            }
            EXPECT_EQ(calls, 1);
        }
    }
}

TEST_F(GenerateStreamTest, testAsyncReleaseHandoffAfterWorkerDrains) {
    auto stream = GenerateStreamBuilder().createContextStream({1, 2});
    stream->incPendingAsyncBookkeeping();
    ASSERT_TRUE(stream->hasPendingAsyncBookkeeping());
    // The last worker can drain after the old caller's pending-count check.
    stream->decPendingAsyncBookkeepingAndMaybeRelease();
    EXPECT_FALSE(stream->markDeferredRelease());
    EXPECT_FALSE(stream->isDeferredReleasePending());
}

TEST_F(GenerateStreamTest, testAsyncReleaseWaitsForLastWorker) {
    auto stream = GenerateStreamBuilder().createComplexContextStream({1, 2});
    ASSERT_TRUE(stream->initKVBlock().ok());
    ASSERT_GT(stream->stream_cache_resource_->curBlocksNum(), 0);
    stream->incPendingAsyncBookkeeping();
    stream->incPendingAsyncBookkeeping();
    {
        std::lock_guard<std::mutex> lock(*stream->mutex_);
        stream->generate_status_->releaseResource();
    }
    EXPECT_FALSE(stream->stream_cache_resource_->isResourceReleased());
    stream->decPendingAsyncBookkeepingAndMaybeRelease();
    EXPECT_FALSE(stream->stream_cache_resource_->isResourceReleased());
    stream->decPendingAsyncBookkeepingAndMaybeRelease();
    EXPECT_TRUE(stream->stream_cache_resource_->isResourceReleased());
    EXPECT_FALSE(stream->isDeferredReleasePending());
}

TEST_F(GenerateStreamTest, testAsyncReleaseConcurrentHandoff) {
    auto stream = GenerateStreamBuilder().createComplexContextStream({1, 2});
    for (int trial = 0; trial < 2000; ++trial) {
        stream->stream_cache_resource_->init(2);
        ASSERT_TRUE(stream->initKVBlock().ok());
        ASSERT_GT(stream->stream_cache_resource_->curBlocksNum(), 0);
        stream->incPendingAsyncBookkeeping();
        std::atomic<bool> start{false};
        std::thread       worker([&] {
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            stream->decPendingAsyncBookkeepingAndMaybeRelease();
        });
        start.store(true, std::memory_order_release);
        {
            std::lock_guard<std::mutex> lock(*stream->mutex_);
            stream->generate_status_->releaseResource();
        }
        worker.join();
        ASSERT_TRUE(stream->stream_cache_resource_->isResourceReleased()) << "trial=" << trial;
        ASSERT_EQ(stream->stream_cache_resource_->curBlocksNum(), 0) << "trial=" << trial;
        ASSERT_FALSE(stream->hasPendingAsyncBookkeeping());
        ASSERT_FALSE(stream->isDeferredReleasePending());
    }
}

TEST_F(GenerateStreamTest, testMtpIgnoresUpdateAfterFinalOutputBeforeStateTransition) {
    for (const std::string reason : {"eos", "stop", "length"}) {
        SCOPED_TRACE(reason);
        auto config = std::make_shared<GenerateConfig>();
        config->is_streaming = false;
        config->ignore_eos = reason != "eos";
        config->max_new_tokens = reason == "length" ? 1 : 100;
        if (reason == "stop") {
            config->stop_words_list = {{7}};
        }
        auto stream = GenerateStreamBuilder().createContextStream({1, 2}, config);
        stream->special_tokens_.eos_token_id = reason == "eos" ? 7 : 9;
        auto sp_output = std::make_shared<SpeculativeExecutorStreamOutput>();
        sp_output->tokens = torch::zeros({2}, torch::kInt32);
        stream->setSPOutputBuffer(sp_output);
        stream->update({torch::tensor({{7}}, torch::kInt32), 1});
        ASSERT_TRUE(stream->hasEvent(StreamEvents::GenerateDone));
        ASSERT_TRUE(stream->isSubGenerateDoneWithoutLock(0));
        ASSERT_FALSE(stream->isFinished());
        auto final_output = stream->nextOutput();
        ASSERT_TRUE(final_output.ok());
        ASSERT_TRUE(final_output.value().generate_outputs[0].finished);
        ASSERT_FALSE(stream->hasOutput());
        const auto seq_len = stream->seqLength();
        const auto tokens = stream->completeTokenIds().clone();
        stream->specUpdate({torch::tensor({{8}}, torch::kInt32),
                            1,
                            -1,
                            torch::Tensor(),
                            torch::Tensor(),
                            torch::Tensor(),
                            torch::Tensor(),
                            torch::Tensor(),
                            torch::Tensor(),
                            torch::Tensor()});
        EXPECT_EQ(stream->seqLength(), seq_len);
        EXPECT_TRUE(torch::equal(stream->completeTokenIds(), tokens));
        EXPECT_FALSE(stream->hasOutput());
        EXPECT_FALSE(stream->hasError());
    }
}

TEST_F(GenerateStreamTest, testMtpForwardGroupsDeferHostTimelineEligibility) {
    auto stream = GenerateStreamBuilder().createDecoderStream({1, 2}, {3});
    stream->generateConfig()->profile_step = 2;
    stream->setGenTimeline(true);
    GenerateStream::MtpAsyncDeviceState state;
    state.next_real_seq_len = 7;
    stream->setMtpAsyncDeviceState(std::move(state));

    // The worker may still be updating the host length during forward gather.
    // Profiling eligibility is only read when rebuilding after bookkeeping.
    StreamGroups forward_groups({stream}, /*mtp_forward_only=*/true);
    EXPECT_EQ(forward_groups.maxSeqLen(), 7);
    EXPECT_FALSE(forward_groups.genTimeline());
    StreamGroups ready_groups({stream});
    EXPECT_EQ(ready_groups.maxSeqLen(), 3);
    EXPECT_TRUE(ready_groups.genTimeline());
    stream->setSeqLength(4);
    StreamGroups after_profile_window({stream});
    EXPECT_FALSE(after_profile_window.genTimeline());
}

TEST_F(GenerateStreamTest, testMtpForwardClaimRejectsUnavailableStreams) {
    for (const std::string reason : {"context", "empty", "multiple", "finished", "error", "done", "deferred", "released"}) {
        SCOPED_TRACE(reason);
        auto stream = GenerateStreamBuilder().createComplexContextStream({1, 2}, reason == "multiple" ? 2 : 1);
        if (reason != "empty") {
            ASSERT_TRUE(stream->initKVBlock().ok());
        }
        stream->setIsContextStream(reason == "context");
        if (reason == "finished") {
            stream->generate_status_->status.store(StreamState::FINISHED);
        } else if (reason == "error") {
            stream->reportError(ErrorCode::CANCELLED, "cancelled before forward");
        } else if (reason == "done") {
            stream->reportEvent(StreamEvents::GenerateDone);
        } else if (reason == "deferred") {
            stream->incPendingAsyncBookkeeping();
            ASSERT_TRUE(stream->markDeferredRelease());
        } else if (reason == "released") {
            stream->releaseResource();
        }
        EXPECT_FALSE(stream->tryAcquireMtpForwardClaim());
        if (reason == "deferred") {
            stream->decPendingAsyncBookkeepingAndMaybeRelease();
        }
        EXPECT_FALSE(stream->hasPendingAsyncBookkeeping());
    }
}

TEST_F(GenerateStreamTest, testMtpForwardClaimOutlivesPreviousBookkeeping) {
    auto stream = GenerateStreamBuilder().createComplexContextStream({1, 2}, 1);
    ASSERT_TRUE(stream->initKVBlock().ok());
    stream->setIsContextStream(false);
    stream->incPendingAsyncBookkeeping();
    ASSERT_TRUE(stream->tryAcquireMtpForwardClaim());
    {
        std::lock_guard<std::mutex> lock(*stream->mutex_);
        stream->generate_status_->releaseResource();
    }
    // Previous CPU work finishes while the next forward still owns the KV.
    stream->decPendingAsyncBookkeepingAndMaybeRelease();
    EXPECT_TRUE(stream->hasPendingAsyncBookkeeping());
    EXPECT_FALSE(stream->stream_cache_resource_->isResourceReleased());
    EXPECT_FALSE(stream->tryAcquireMtpForwardClaim());
    // The executor drops this claim only after the forward completion event.
    stream->decPendingAsyncBookkeepingAndMaybeRelease();
    EXPECT_TRUE(stream->stream_cache_resource_->isResourceReleased());
    EXPECT_FALSE(stream->hasPendingAsyncBookkeeping());
    EXPECT_FALSE(stream->isDeferredReleasePending());
}

TEST_F(GenerateStreamTest, testMtpForwardClaimConcurrentRelease) {
    auto stream = GenerateStreamBuilder().createComplexContextStream({1, 2}, 1);
    stream->setIsContextStream(false);
    for (int trial = 0; trial < 1000; ++trial) {
        stream->stream_cache_resource_->init(1);
        ASSERT_TRUE(stream->initKVBlock().ok());
        std::atomic<bool> start{false};
        std::thread releaser([&] {
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            stream->releaseResource();
        });
        start.store(true, std::memory_order_release);
        if (stream->tryAcquireMtpForwardClaim()) {
            {
                std::lock_guard<std::mutex> lock(*stream->mutex_);
                EXPECT_FALSE(stream->stream_cache_resource_->isResourceReleased()) << "trial=" << trial;
            }
            stream->decPendingAsyncBookkeepingAndMaybeRelease();
        }
        releaser.join();
        ASSERT_TRUE(stream->stream_cache_resource_->isResourceReleased()) << "trial=" << trial;
        ASSERT_FALSE(stream->hasPendingAsyncBookkeeping());
    }
}

TEST_F(GenerateStreamTest, testMtpMinNewTokensIgnoresEarlyStopWithinAcceptedBatch) {
    for (bool use_eos : {false, true}) {
        for (bool stop_at_minimum : {false, true}) {
            auto config            = std::make_shared<GenerateConfig>();
            config->min_new_tokens = 8;
            config->max_new_tokens = 8;
            config->aux_info       = true;
            config->is_streaming   = false;
            config->ignore_eos     = !use_eos;
            if (!use_eos) {
                config->stop_words_list = {{7}};
            }
            auto stream                          = GenerateStreamBuilder().createContextStream({1, 2}, config);
            stream->special_tokens_.eos_token_id = 7;
            stream->update({torch::tensor({{3, 4, 5, 6}}, torch::kInt32), 4});
            ASSERT_FALSE(stream->hasOutput());
            // The second MTP batch crosses min_new_tokens; its first stop is
            // at output five and must not truncate the final output to five.
            stream->update({torch::tensor({{7, 4, 5, stop_at_minimum ? 7 : 6}}, torch::kInt32), 4});
            auto output = stream->nextOutput();
            ASSERT_TRUE(output.ok());
            ASSERT_EQ(output.value().generate_outputs.size(), 1u);
            const auto& result = output.value().generate_outputs[0];
            EXPECT_TRUE(result.finished);
            EXPECT_EQ(result.output_ids.numel(), 8);
            EXPECT_EQ(result.aux_info.output_len, 8);
        }
    }
}

TEST_F(GenerateStreamTest, testConstruct) {
    auto builder = GenerateStreamBuilder();
    auto stream1 = builder.createContextStream({{1, 2, 3, 4, 5}, {}});
    auto stream2 = builder.createDecoderStream({1, 2, 3, 4, 5}, {1, 2, 3});
}

TEST_F(GenerateStreamTest, testAuxInfoKeepsIndependentReturnSequenceLengths) {
    auto config                  = std::make_shared<GenerateConfig>();
    config->aux_info             = true;
    config->ignore_eos           = true;
    config->is_streaming         = false;
    config->num_return_sequences = 2;
    config->stop_words_list      = {{7}};
    auto stream                  = GenerateStreamBuilder().createContextStream({1, 2}, config);

    // The first sequence finishes one step before the second. The sampler
    // continues to append padding/EOS tokens to finished rows, but their
    // request-level output length must remain frozen at the finish point.
    stream->update({torch::tensor({{7}, {8}}, torch::kInt32), 1});
    ASSERT_FALSE(stream->hasOutput());
    stream->update({torch::tensor({{0}, {7}}, torch::kInt32), 1});

    auto output = stream->nextOutput();
    ASSERT_TRUE(output.ok());
    ASSERT_EQ(output.value().generate_outputs.size(), 2);
    const auto& first  = output.value().generate_outputs[0];
    const auto& second = output.value().generate_outputs[1];
    EXPECT_TRUE(first.finished);
    EXPECT_TRUE(second.finished);
    EXPECT_EQ(first.output_ids.numel(), 2);
    EXPECT_EQ(second.output_ids.numel(), 2);
    EXPECT_EQ(first.aux_info.output_len, 1);
    EXPECT_EQ(second.aux_info.output_len, 2);
    EXPECT_EQ(first.aux_info.output_len + second.aux_info.output_len, 3);
}

TEST_F(GenerateStreamTest, testSlowNonStreamingMetricConsumerDoesNotFillBusinessOutputQueue) {
    constexpr int kOutputLength       = 1100;
    auto          config              = std::make_shared<GenerateConfig>();
    config->aux_info                  = true;
    config->ignore_eos                = true;
    config->is_streaming              = false;
    config->frontend_metric_streaming = true;
    config->max_new_tokens            = kOutputLength;
    auto stream                       = GenerateStreamBuilder().createContextStream({1}, config);

    // Do not consume while generating more frames than the business queue's
    // capacity. Metric-only cumulative snapshots must coalesce instead of
    // terminating the otherwise valid non-streaming request.
    for (int i = 0; i < kOutputLength; ++i) {
        stream->addFrontendGenerateExecuteMetrics(/*execute_time_us=*/10, /*generate_token_num=*/1);
        stream->update({torch::tensor({{2}}, torch::kInt32), 1});
    }

    ASSERT_FALSE(stream->hasError());
    ASSERT_TRUE(stream->hasEvent(StreamEvents::GenerateDone));

    auto metric_output = stream->nextOutput();
    ASSERT_TRUE(metric_output.ok());
    EXPECT_TRUE(metric_output.value().frontend_metric_only);
    ASSERT_TRUE(metric_output.value().frontend_generate_token_num.has_value());
    EXPECT_EQ(metric_output.value().frontend_generate_token_num.value(), kOutputLength - 1);

    auto final_output = stream->nextOutput();
    ASSERT_TRUE(final_output.ok());
    EXPECT_FALSE(final_output.value().frontend_metric_only);
    ASSERT_EQ(final_output.value().generate_outputs.size(), 1);
    EXPECT_EQ(final_output.value().generate_outputs[0].output_ids.numel(), kOutputLength);
    EXPECT_EQ(final_output.value().generate_outputs[0].aux_info.output_len, kOutputLength);
    ASSERT_TRUE(final_output.value().frontend_generate_token_num.has_value());
    EXPECT_EQ(final_output.value().frontend_generate_token_num.value(), kOutputLength);
    ASSERT_TRUE(final_output.value().frontend_generate_execute_time_us.has_value());
    EXPECT_EQ(final_output.value().frontend_generate_execute_time_us.value(), kOutputLength * 10);
    EXPECT_FALSE(stream->hasError());
}

TEST_F(GenerateStreamTest, testCancelledNonStreamingRequestDrainsLatestMetricBeforeError) {
    constexpr int kExecutedSteps      = 5;
    auto          config              = std::make_shared<GenerateConfig>();
    config->aux_info                  = true;
    config->ignore_eos                = true;
    config->is_streaming              = false;
    config->frontend_metric_streaming = true;
    config->max_new_tokens            = 100;
    auto stream                       = GenerateStreamBuilder().createContextStream({1}, config);

    for (int i = 0; i < kExecutedSteps; ++i) {
        stream->addFrontendGenerateExecuteMetrics(/*execute_time_us=*/10, /*generate_token_num=*/1);
        stream->update({torch::tensor({{2}}, torch::kInt32), 1});
    }
    stream->reportError(ErrorCode::CANCELLED, "cancelled by test");

    // Mirror LocalRpcServer's real polling condition. It must consume the
    // latest cumulative snapshot and then iterate once more to expose the
    // terminal error instead of returning grpc::Status::OK.
    int       metric_output_count = 0;
    ErrorCode terminal_error      = ErrorCode::NONE_ERROR;
    while (stream->isActive() || stream->hasOutput()) {
        auto output = stream->nextOutput();
        if (!output.ok()) {
            terminal_error = output.status().code();
            break;
        }
        ++metric_output_count;
        EXPECT_TRUE(output.value().frontend_metric_only);
        ASSERT_TRUE(output.value().frontend_generate_token_num.has_value());
        EXPECT_EQ(output.value().frontend_generate_token_num.value(), kExecutedSteps);
        ASSERT_TRUE(output.value().frontend_generate_execute_time_us.has_value());
        EXPECT_EQ(output.value().frontend_generate_execute_time_us.value(), kExecutedSteps * 10);
    }

    EXPECT_EQ(metric_output_count, 1);
    EXPECT_EQ(terminal_error, ErrorCode::CANCELLED);
    EXPECT_FALSE(stream->hasOutput());
}

TEST_F(GenerateStreamTest, testLogprobsHistoryUsesBoundedGeometricGrowth) {
    auto generate_config             = std::make_shared<GenerateConfig>();
    generate_config->return_logprobs = true;
    generate_config->top_logprobs    = 20;
    generate_config->max_new_tokens  = 2000;
    auto stream                      = GenerateStreamBuilder().createContextStream({1}, generate_config);

    EXPECT_FALSE(stream->getTokenLogProbs().defined());
    EXPECT_FALSE(stream->getTopLogprobTokenIds().defined());
    EXPECT_FALSE(stream->getTopLogProbs().defined());

    auto selected_first = torch::arange(64, torch::kFloat32).reshape({1, 64});
    auto top_ids_first  = torch::arange(64 * 20, torch::kInt32).reshape({1, 64, 20});
    auto top_first      = torch::arange(64 * 20, torch::kFloat32).reshape({1, 64, 20});
    stream->setLogProbs(selected_first, top_ids_first, top_first, torch::Tensor(), stream->inputLength());

    ASSERT_EQ(stream->getTokenLogProbs().sizes(), (torch::IntArrayRef{1, 64}));
    ASSERT_EQ(stream->getTopLogprobTokenIds().sizes(), (torch::IntArrayRef{1, 64, 20}));
    ASSERT_EQ(stream->getTopLogProbs().sizes(), (torch::IntArrayRef{1, 64, 20}));
    EXPECT_LT(stream->getTokenLogProbs().size(1), generate_config->max_new_tokens);

    auto selected_last = torch::tensor({64.0f}, torch::kFloat32).reshape({1, 1});
    auto top_ids_last  = torch::arange(64 * 20, 65 * 20, torch::kInt32).reshape({1, 1, 20});
    auto top_last      = torch::arange(64 * 20, 65 * 20, torch::kFloat32).reshape({1, 1, 20});
    stream->setLogProbs(selected_last, top_ids_last, top_last, torch::Tensor(), stream->inputLength() + 64);

    ASSERT_EQ(stream->getTokenLogProbs().size(1), 128);
    EXPECT_TRUE(
        torch::equal(stream->getTokenLogProbs().narrow(1, 0, 65), torch::arange(65, torch::kFloat32).reshape({1, 65})));
    EXPECT_TRUE(torch::equal(stream->getTopLogprobTokenIds().narrow(1, 0, 65),
                             torch::arange(65 * 20, torch::kInt32).reshape({1, 65, 20})));
    EXPECT_TRUE(torch::equal(stream->getTopLogProbs().narrow(1, 0, 65),
                             torch::arange(65 * 20, torch::kFloat32).reshape({1, 65, 20})));
}

TEST_F(GenerateStreamTest, testGenerateStreamReuseCacheMethod) {
    auto builder = GenerateStreamBuilder();
    auto stream  = builder.createContextStream({1, 2, 3, 4, 5, 6});

    // default true
    ASSERT_TRUE(stream->reuseCache());

    // flip to false and verify
    stream->generate_input_->generate_config->reuse_cache = false;
    ASSERT_FALSE(stream->reuseCache());

    // flip back to true and verify
    stream->generate_input_->generate_config->reuse_cache = true;
    ASSERT_TRUE(stream->reuseCache());
}

TEST_F(GenerateStreamTest, testMaxTokenNumExcludesThinkingTokens) {
    autil::EnvGuard guard("RTP_LLM_MAX_TOKENS_EXCLUDE_THINKING", "true");
    auto            builder       = GenerateStreamBuilder();
    auto            config        = std::make_shared<GenerateConfig>();
    config->max_new_tokens        = 1;
    config->in_think_mode         = true;
    config->max_thinking_tokens   = 5;
    config->begin_think_token_ids = {7};
    config->end_think_token_ids   = {8, 9};
    auto stream                   = builder.createContextStream({1, 2}, config);

    ASSERT_EQ(stream->maxTokenNum(), 8);

    auto processors = stream->getAllLogitsProcessorPtr();
    ASSERT_FALSE(processors.empty());
    auto think_processor = std::dynamic_pointer_cast<ThinkModeLogitsProcessor>(processors[0]);
    ASSERT_NE(think_processor, nullptr);

    think_processor->updateStatus(torch::tensor({{8, 9}}, torch::kInt32), 2);
    ASSERT_EQ(think_processor->finishedThinkOutputLen(), 2);
    ASSERT_EQ(stream->maxTokenNum(), 5);
}

TEST_F(GenerateStreamTest, testNegativeMaxThinkingTokensKeepsInclusiveMaxNewTokensBudget) {
    autil::EnvGuard guard("RTP_LLM_MAX_TOKENS_EXCLUDE_THINKING", "true");
    auto            builder       = GenerateStreamBuilder();
    auto            config        = std::make_shared<GenerateConfig>();
    config->max_new_tokens        = 4;
    config->in_think_mode         = true;
    config->max_thinking_tokens   = -1;
    config->begin_think_token_ids = {7};
    config->end_think_token_ids   = {8, 9};
    auto stream                   = builder.createContextStream({1, 2}, config);

    ASSERT_EQ(config->max_thinking_tokens, -1);
    ASSERT_EQ(stream->maxTokenNum(), 6);
}

TEST_F(GenerateStreamTest, testThinkEndTrimsMtpBatchToOneContentToken) {
    autil::EnvGuard guard("RTP_LLM_MAX_TOKENS_EXCLUDE_THINKING", "true");
    auto            builder       = GenerateStreamBuilder();
    auto            config        = std::make_shared<GenerateConfig>();
    config->max_new_tokens        = 1;
    config->in_think_mode         = true;
    config->max_thinking_tokens   = 100;
    config->begin_think_token_ids = {7};
    config->end_think_token_ids   = {8, 9};
    auto stream                   = builder.createContextStream({1, 2}, config);

    // Simulate one speculative accept batch crossing the think boundary:
    // one think token, a two-token </think>, and two content tokens.
    auto new_tokens = torch::tensor({{10, 8, 9, 11, 12}}, torch::kInt32);
    stream->update({new_tokens, 5});

    ASSERT_FALSE(stream->hasError());
    ASSERT_TRUE(stream->hasEvent(StreamEvents::GenerateDone));
    ASSERT_EQ(stream->completeTokenIdsVec(), std::vector<int>({1, 2, 10, 8, 9, 11}));

    auto processors = stream->getAllLogitsProcessorPtr();
    ASSERT_FALSE(processors.empty());
    auto think_processor = std::dynamic_pointer_cast<ThinkModeLogitsProcessor>(processors[0]);
    ASSERT_NE(think_processor, nullptr);
    ASSERT_EQ(think_processor->finishedThinkOutputLen(), 3);
    ASSERT_EQ(think_processor->acceptedTokenLen(), 4);

    // The dynamic limit is input + actual think output + exactly one content token.
    ASSERT_EQ(stream->seqLength(), stream->maxTokenNum());
    ASSERT_EQ(stream->outputTokenLen() - think_processor->finishedThinkOutputLen(), 1);

    auto output = stream->nextOutput();
    ASSERT_TRUE(output.ok());
    ASSERT_EQ(output.value().generate_outputs.size(), 1);
    ASSERT_TRUE(output.value().generate_outputs[0].finished);
    ASSERT_EQ(output.value().generate_outputs[0].output_ids.numel(), 4);
    ASSERT_EQ(output.value().generate_outputs[0].output_ids.data_ptr<int32_t>()[3], 11);
}

TEST_F(GenerateStreamTest, testThinkEndScanSkippedWhenMaxNewTokensCoversMtpBatch) {
    autil::EnvGuard guard("RTP_LLM_MAX_TOKENS_EXCLUDE_THINKING", "true");
    auto            builder       = GenerateStreamBuilder();
    auto            config        = std::make_shared<GenerateConfig>();
    config->max_new_tokens        = 5;
    config->in_think_mode         = true;
    config->max_thinking_tokens   = 100;
    config->begin_think_token_ids = {7};
    config->end_think_token_ids   = {8, 9};
    auto stream                   = builder.createContextStream({1, 2}, config);

    auto new_tokens = torch::tensor({{10, 8, 9, 11, 12}}, torch::kInt32);
    stream->update({new_tokens, 5});

    ASSERT_FALSE(stream->hasError());
    ASSERT_FALSE(stream->hasEvent(StreamEvents::GenerateDone));
    ASSERT_EQ(stream->completeTokenIdsVec(), std::vector<int>({1, 2, 10, 8, 9, 11, 12}));
}

TEST_F(GenerateStreamTest, testThinkingLogprobsContentBoundaryUsesFirstCloseToken) {
    auto config                 = std::make_shared<GenerateConfig>();
    config->return_logprobs     = true;
    config->top_logprobs        = 1;
    config->max_new_tokens      = 10;
    config->in_think_mode       = true;
    config->end_think_token_ids = {8, 9};
    config->ignore_eos          = true;
    auto stream                 = GenerateStreamBuilder().createContextStream({1}, config);

    EXPECT_FALSE(stream->hasLogprobsContentStarted());
    EXPECT_FALSE(stream->getTokenLogProbs().defined());
    EXPECT_EQ(stream->logprobsContentOffset(torch::tensor({{10, 11}}, torch::kInt32), 2), 2);
    // The first close token remains reasoning. The delimiter tail and answer
    // that follow it are already content-logprob positions.
    EXPECT_EQ(stream->logprobsContentOffset(torch::tensor({{10, 8, 9, 11}}, torch::kInt32), 4), 2);

    stream->update({torch::tensor({{10}}, torch::kInt32),
                    1,
                    torch::Tensor(),
                    torch::Tensor(),
                    torch::Tensor(),
                    torch::Tensor(),
                    torch::Tensor(),
                    torch::Tensor(),
                    torch::Tensor(),
                    torch::Tensor(),
                    true,
                    false,
                    torch::Tensor(),
                    torch::Tensor(),
                    torch::Tensor(),
                    -1,
                    1});
    EXPECT_FALSE(stream->hasLogprobsContentStarted());

    stream->update({torch::tensor({{8}}, torch::kInt32),
                    1,
                    torch::Tensor(),
                    torch::Tensor(),
                    torch::Tensor(),
                    torch::Tensor(),
                    torch::Tensor(),
                    torch::Tensor(),
                    torch::Tensor(),
                    torch::Tensor(),
                    true,
                    false,
                    torch::Tensor(),
                    torch::Tensor(),
                    torch::Tensor(),
                    -1,
                    1});
    EXPECT_TRUE(stream->hasLogprobsContentStarted());
    EXPECT_EQ(stream->logprobsContentOffset(torch::tensor({{9}}, torch::kInt32), 1), 0);
}

TEST_F(GenerateStreamTest, testThinkingLogprobsPhaseKeepsCloseTokenAfterIgnoredStop) {
    auto config                 = std::make_shared<GenerateConfig>();
    config->return_logprobs     = true;
    config->top_logprobs        = 1;
    config->max_new_tokens      = 10;
    config->in_think_mode       = true;
    config->end_think_token_ids = {8, 9};
    config->stop_words_list     = {{7}};
    auto stream                 = GenerateStreamBuilder().createContextStream({1}, config);

    stream->update({torch::tensor({{7, 8}}, torch::kInt32),
                    2,
                    torch::Tensor(),
                    torch::Tensor(),
                    torch::Tensor(),
                    torch::Tensor(),
                    torch::Tensor(),
                    torch::Tensor(),
                    torch::Tensor(),
                    torch::Tensor(),
                    true,
                    false,
                    torch::Tensor(),
                    torch::Tensor(),
                    torch::Tensor(),
                    -1,
                    2});

    EXPECT_EQ(stream->completeTokenIdsVec(), (std::vector<int>{1, 7, 8}));
    EXPECT_TRUE(stream->hasLogprobsContentStarted());
}

// clearMtpAsyncDeviceState rejects stale epochs. A worker that
// captured epoch N must not clear state that step N+1 already published
// under epoch N+1.
TEST_F(GenerateStreamTest, testMtpAsyncDeviceStateStaleEpochReject) {
    auto builder = GenerateStreamBuilder();
    auto stream  = builder.createContextStream({1, 2, 3, 4, 5, 6});

    // Start: epoch counter is 0, state is default-constructed.
    ASSERT_EQ(stream->getMtpAsyncDeviceState().epoch, 0u);
    ASSERT_FALSE(stream->getMtpAsyncDeviceState().accept_len_gpu.defined());

    // Step 1: publish state, capture epoch_1.
    GenerateStream::MtpAsyncDeviceState s1;
    s1.accept_len_gpu      = torch::ones({1}, torch::kInt32);
    const uint64_t epoch_1 = stream->setMtpAsyncDeviceState(std::move(s1));
    ASSERT_EQ(epoch_1, 1u);
    ASSERT_TRUE(stream->getMtpAsyncDeviceState().accept_len_gpu.defined());

    // Step 2: another publish before the worker for epoch_1 ran. Counter
    // bumps; old epoch should now be stale.
    GenerateStream::MtpAsyncDeviceState s2;
    s2.accept_len_gpu      = torch::ones({1}, torch::kInt32) * 2;
    const uint64_t epoch_2 = stream->setMtpAsyncDeviceState(std::move(s2));
    ASSERT_EQ(epoch_2, 2u);
    ASSERT_NE(epoch_1, epoch_2);

    // Stale worker for epoch_1 attempts to clear: must be rejected, state
    // for epoch_2 must remain intact.
    ASSERT_FALSE(stream->clearMtpAsyncDeviceState(epoch_1));
    ASSERT_TRUE(stream->getMtpAsyncDeviceState().accept_len_gpu.defined());
    ASSERT_EQ(stream->getMtpAsyncDeviceState().epoch, epoch_2);

    // Worker for epoch_2 clears successfully.
    ASSERT_TRUE(stream->clearMtpAsyncDeviceState(epoch_2));
    ASSERT_FALSE(stream->getMtpAsyncDeviceState().accept_len_gpu.defined());
    ASSERT_EQ(stream->getMtpAsyncDeviceState().epoch, 0u);

    // Repeated stale clear after the live state is gone is also a no-op
    // (epoch 0 != epoch_2 since state was reset to default).
    ASSERT_FALSE(stream->clearMtpAsyncDeviceState(epoch_2));
}

TEST_F(GenerateStreamTest, testMtpAsyncDeviceStateTracksRealAndUpperBoundSeqLen) {
    auto builder = GenerateStreamBuilder();
    auto stream  = builder.createContextStream({1, 2, 3, 4, 5, 6});

    GenerateStream::MtpAsyncDeviceState state;
    state.last_real_seq_len = stream->seqLength();
    state.next_real_seq_len = state.last_real_seq_len + 2;
    stream->setMtpAsyncDeviceState(std::move(state));

    ASSERT_EQ(stream->getMtpAsyncDeviceState().last_real_seq_len, stream->seqLength());
    ASSERT_EQ(stream->getMtpAsyncDeviceState().next_real_seq_len, stream->seqLength() + 2);
}

// setSpecDecodeDeviceState / clearSpecDecodeDeviceState
// continue to work as wrappers around the new struct API.
TEST_F(GenerateStreamTest, testMtpAsyncDeviceStateBackCompatWrappers) {
    auto builder = GenerateStreamBuilder();
    auto stream  = builder.createContextStream({1, 2, 3, 4, 5, 6});

    auto accept_len     = torch::ones({1}, torch::kInt32);
    auto accept_tokens  = torch::ones({1, 2}, torch::kInt32);
    auto next_seq_len   = torch::ones({1}, torch::kInt32) * 7;
    auto propose_tokens = torch::ones({1, 4}, torch::kInt32);

    stream->setSpecDecodeDeviceState(accept_len, accept_tokens, next_seq_len, propose_tokens);
    ASSERT_TRUE(stream->getAcceptLenGpu().defined());
    ASSERT_TRUE(stream->getAcceptTokensGpu().defined());
    ASSERT_TRUE(stream->getNextSeqLenGpu().defined());
    ASSERT_TRUE(stream->getProposeTokensGpu().defined());

    stream->clearSpecDecodeDeviceState();
    ASSERT_FALSE(stream->getAcceptLenGpu().defined());
    ASSERT_FALSE(stream->getAcceptTokensGpu().defined());
    ASSERT_FALSE(stream->getNextSeqLenGpu().defined());
    ASSERT_FALSE(stream->getProposeTokensGpu().defined());
}


TEST(CompleteTokenIdsStopWordsTest, OnlyContentStopsThinkingRequests) {
    auto input = std::make_shared<GenerateInput>();
    input->generate_config = std::make_shared<GenerateConfig>();
    input->input_ids = torch::tensor({8, 9, 7}, torch::kInt32);
    CompleteTokenIds ids(1, 1, 64, 1);
    ids.init(input);
    const auto append = [&](std::vector<int> tokens) {
        const int old_len = ids.seqLength();
        std::copy(tokens.begin(), tokens.end(), ids.data(0) + old_len);
        ids.setSeqLength(old_len + tokens.size());
    };
    append({7, 8});
    // The prompt's closing marker and an incomplete generated marker do not count.
    EXPECT_FALSE(ids.matchStopWordsList(0, {7}, 3, 3, true, {8, 9}));
    append({9, 6, 7, 5});
    EXPECT_FALSE(ids.matchStopWordsList(0, {8, 9}, 3, 3, true, {8, 9}));
    EXPECT_FALSE(ids.matchStopWordsList(0, {9, 6}, 3, 3, true, {8, 9}));
    EXPECT_TRUE(ids.matchStopWordsList(0, {7}, 3, 3, true, {8, 9}));
    EXPECT_EQ(ids.completeTokenIdsVec(0), (std::vector<int>{8, 9, 7, 7, 8, 9, 6, 7}));
}

TEST(CompleteTokenIdsStopWordsTest, MultiTokenStopAcrossUpdatesAndMissingMarker) {
    auto input = std::make_shared<GenerateInput>();
    input->generate_config = std::make_shared<GenerateConfig>();
    input->input_ids = torch::tensor({1}, torch::kInt32);
    CompleteTokenIds ids(1, 1, 64, 1);
    ids.init(input);
    std::vector<int> tokens{7, 6, 8, 9, 7};
    std::copy(tokens.begin(), tokens.end(), ids.data(0) + 1);
    ids.setSeqLength(6);
    EXPECT_FALSE(ids.matchStopWordsList(0, {7, 6}, 1, 1, true, {8, 9}));
    ids.data(0)[6] = 6;
    ids.setSeqLength(7);
    EXPECT_FALSE(ids.matchStopWordsList(0, {7, 6}, 1, 1, true, {}));
    EXPECT_FALSE(ids.matchStopWordsList(0, {7, 6}, 8, 1, true, {8, 9}));
    EXPECT_TRUE(ids.matchStopWordsList(0, {7, 6}, 7, 1, true, {8, 9}));
    EXPECT_EQ(ids.seqLength(), 7);
}

TEST_F(GenerateStreamTest, StopWordsIgnoreThinkAndHonorMinTokens) {
    auto config = std::make_shared<GenerateConfig>();
    config->in_think_mode = true;
    config->end_think_token_ids = {8, 9};
    config->stop_words_list = {{7}};
    config->min_new_tokens = 4;
    auto stream = GenerateStreamBuilder().createContextStream({1}, config);
    stream->update({torch::tensor({{7, 8, 9}}, torch::kInt32), 3});
    EXPECT_FALSE(stream->hasEvent(StreamEvents::GenerateDone));
    stream->update({torch::tensor({{7}}, torch::kInt32), 1});
    EXPECT_TRUE(stream->hasEvent(StreamEvents::GenerateDone));
    EXPECT_TRUE(stream->isSubGenerateDoneWithoutLock(0));
    EXPECT_EQ(stream->completeTokenIdsVec(), (std::vector<int>{1, 7, 8, 9, 7}));
}

TEST_F(GenerateStreamTest, prefillOnlyValidationPreservesCacheSetting) {
    GenerateConfig config;
    config.max_new_tokens = 0;
    config.reuse_cache    = true;

    EXPECT_NO_THROW(config.validatePrefillOnly());
    EXPECT_TRUE(config.reuse_cache);
}

TEST_F(GenerateStreamTest, prefillOnlyValidationRejectsUnsupportedConfigWithoutMutation) {
    struct UnsupportedConfig {
        const char* name;
        void (*set)(GenerateConfig&);
    };
    constexpr UnsupportedConfig fields[] = {
        {"min_new_tokens", +[](GenerateConfig& config) { config.min_new_tokens = 1; }},
        {"num_beams", +[](GenerateConfig& config) { config.num_beams = 2; }},
        {"variable_num_beams", +[](GenerateConfig& config) { config.variable_num_beams = {1}; }},
        {"num_return_sequences", +[](GenerateConfig& config) { config.num_return_sequences = 2; }},
        {"return_logits", +[](GenerateConfig& config) { config.return_logits = true; }},
        {"calculate_loss", +[](GenerateConfig& config) { config.calculate_loss = 1; }},
        {"return_softmax_probs", +[](GenerateConfig& config) { config.return_softmax_probs = true; }},
        {"return_all_probs", +[](GenerateConfig& config) { config.return_all_probs = true; }},
        {"return_cum_log_probs", +[](GenerateConfig& config) { config.return_cum_log_probs = true; }},
        {"return_logprobs", +[](GenerateConfig& config) { config.return_logprobs = true; }},
        {"return_hidden_states", +[](GenerateConfig& config) { config.return_hidden_states = true; }},
        {"return_all_hidden_states", +[](GenerateConfig& config) { config.return_all_hidden_states = true; }},
    };
    for (const auto& field : fields) {
        SCOPED_TRACE(field.name);
        GenerateConfig config;
        config.max_new_tokens = 0;
        config.reuse_cache    = true;
        field.set(config);

        try {
            config.validatePrefillOnly();
            FAIL() << field.name << " should be rejected";
        } catch (const std::invalid_argument& error) {
            EXPECT_NE(std::string(error.what()).find(field.name), std::string::npos);
        }
        EXPECT_TRUE(config.reuse_cache);
    }
}

TEST_F(GenerateStreamTest, prefillOnlyRejectsNegativeTokenBudget) {
    GenerateConfig config;
    config.max_new_tokens = -1;
    EXPECT_THROW(config.validatePrefillOnly(), std::invalid_argument);
    EXPECT_EQ(config.max_new_tokens, -1);
    config.max_new_tokens = 1;
    EXPECT_NO_THROW(config.validatePrefillOnly());
}

}  // namespace rtp_llm
