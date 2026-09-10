
#include "gtest/gtest.h"

#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/cpp/config/ConfigModules.h"

using namespace std;

namespace rtp_llm {

class GenerateStreamBuilder {
public:
    GenerateStreamBuilder() {
        model_config_.max_seq_len = 2048;
    }

    CacheConfig init_config() {
        return test::makeSimpleMhaCacheConfig(
            /*layer_num=*/3, /*block_num=*/9, /*tokens_per_block=*/2, rtp_llm::DataType::TYPE_INT8);
    }

    GenerateStreamPtr createContextStream(std::vector<int> input_ids) {
        std::shared_ptr<GenerateInput>  generate_input(new GenerateInput());
        std::shared_ptr<GenerateConfig> generate_config(new GenerateConfig());
        ResourceContext                 resource_context;
        generate_input->generate_config = generate_config;
        generate_input->input_ids =
            torch::tensor(std::vector<int32_t>(input_ids.begin(), input_ids.end()), torch::kInt32);
        return std::make_shared<NormalGenerateStream>(
            generate_input, model_config_, runtime_config_, resource_context, nullptr);
    };

    GenerateStreamPtr createComplexContextStream(std::vector<int> input_ids) {
        autil::EnvGuard perf_scope("PERF_TEST", "1");

        auto cache_config  = init_config();
        auto cache_manager = std::make_shared<KVCacheManager>(cache_config);
        cache_manager->init();
        ResourceContext resource_context;
        resource_context.cache_manager = cache_manager;
        resource_context.reuse_cache   = true;

        std::shared_ptr<GenerateInput>  generate_input(new GenerateInput());
        std::shared_ptr<GenerateConfig> generate_config(new GenerateConfig());
        generate_config->num_return_sequences = 2;
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

TEST_F(GenerateStreamTest, testConstruct) {
    auto builder = GenerateStreamBuilder();
    auto stream1 = builder.createContextStream({{1, 2, 3, 4, 5}, {}});
    auto stream2 = builder.createDecoderStream({1, 2, 3, 4, 5}, {1, 2, 3});
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

TEST_F(GenerateStreamTest, testInitialReuseLengthMustBeLessThanSeqLength) {
    auto builder = GenerateStreamBuilder();
    auto stream  = builder.createContextStream({1, 2, 3, 4, 5, 6});

    stream->setInitialReuseLength(stream->seqLength() - 1);
    ASSERT_EQ(stream->initialReuseLength(), stream->seqLength() - 1);

    EXPECT_THROW(stream->setInitialReuseLength(stream->seqLength()), RTPException);
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

TEST_F(GenerateStreamTest, testLinearReplayWindowSurvivesMtpStateClearAndRejectsStalePublish) {
    auto stream                                          = GenerateStreamBuilder().createContextStream({1, 2, 3, 4});
    auto first                                           = std::make_shared<GenerateStream::LinearReplayWindow>();
    first->verify_epoch                                  = 3;
    first->lease                                         = std::make_shared<LinearReplayLease>();
    first->lease->slot_id                                = 1;
    first->lease->generation                             = 2;
    stream->stream_cache_resource_->linear_replay_lease_ = first->lease;
    stream->linear_replay_epoch_counter_                 = first->verify_epoch;
    first->accept_len_gpu                                = torch::tensor({3}, torch::kInt32);
    std::weak_ptr<const GenerateStream::LinearReplayWindow> first_weak = first;
    stream->publishLinearReplayWindow(std::move(first));

    GenerateStream::MtpAsyncDeviceState state;
    state.linear_replay_epoch = 3;
    const auto mtp_epoch      = stream->setMtpAsyncDeviceState(std::move(state));
    ASSERT_TRUE(stream->clearMtpAsyncDeviceState(mtp_epoch));
    stream->clearSpecDecodeDeviceState();
    ASSERT_FALSE(first_weak.expired());

    auto next                            = std::make_shared<GenerateStream::LinearReplayWindow>();
    next->verify_epoch                   = 4;
    next->lease                          = stream->stream_cache_resource_->linear_replay_lease_;
    stream->linear_replay_epoch_counter_ = next->verify_epoch;
    std::weak_ptr<const GenerateStream::LinearReplayWindow> next_weak = next;
    stream->publishLinearReplayWindow(std::move(next));
    ASSERT_TRUE(first_weak.expired());

    auto stale          = std::make_shared<GenerateStream::LinearReplayWindow>();
    stale->verify_epoch = 3;
    stale->lease        = stream->stream_cache_resource_->linear_replay_lease_;
    std::weak_ptr<const GenerateStream::LinearReplayWindow> stale_weak = stale;
    stream->publishLinearReplayWindow(std::move(stale));
    ASSERT_TRUE(stale_weak.expired());
    ASSERT_FALSE(next_weak.expired());

    stream->clearLinearReplayWindow();
    ASSERT_TRUE(next_weak.expired());

    auto old_lease_window                                = std::make_shared<GenerateStream::LinearReplayWindow>();
    old_lease_window->verify_epoch                       = 5;
    old_lease_window->lease                              = stream->stream_cache_resource_->linear_replay_lease_;
    stream->stream_cache_resource_->linear_replay_lease_ = std::make_shared<LinearReplayLease>();
    stream->linear_replay_epoch_counter_                 = 5;
    std::weak_ptr<const GenerateStream::LinearReplayWindow> old_lease_weak = old_lease_window;
    stream->publishLinearReplayWindow(std::move(old_lease_window));
    ASSERT_TRUE(old_lease_weak.expired());
}

TEST_F(GenerateStreamTest, testLinearReplayPublicationIgnoresFinishedOrReleasedStream) {
    for (bool resources_released : {false, true}) {
        auto stream       = GenerateStreamBuilder().createContextStream({1, 2, 3, 4});
        auto lease        = std::make_shared<LinearReplayLease>();
        lease->slot_id    = 1;
        lease->generation = 2;
        stream->stream_cache_resource_->linear_replay_lease_ = lease;
        stream->linear_replay_epoch_counter_                 = 3;
        stream->generate_status_->status = resources_released ? StreamState::RUNNING : StreamState::FINISHED;
        stream->stream_cache_resource_->resource_released_ = resources_released;

        auto window          = std::make_shared<GenerateStream::LinearReplayWindow>();
        window->verify_epoch = 3;
        window->lease        = std::move(lease);
        std::weak_ptr<const GenerateStream::LinearReplayWindow> weak_window = window;
        EXPECT_NO_THROW(stream->publishLinearReplayWindow(std::move(window)));
        EXPECT_TRUE(weak_window.expired());
    }
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

}  // namespace rtp_llm
