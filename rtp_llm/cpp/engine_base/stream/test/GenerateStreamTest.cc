
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

    GenerateStreamPtr createMockStream(std::vector<int> input_ids, int64_t /*batch_padding*/) {
        std::shared_ptr<GenerateInput>  generate_input(new GenerateInput());
        std::shared_ptr<GenerateConfig> generate_config(new GenerateConfig());
        ResourceContext                 resource_context;
        generate_input->generate_config = generate_config;
        generate_input->input_ids =
            torch::tensor(std::vector<int32_t>(input_ids.begin(), input_ids.end()), torch::kInt32);
        return std::make_shared<NormalGenerateStream>(
            generate_input, model_config_, runtime_config_, resource_context, nullptr);
    }

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

TEST_F(GenerateStreamTest, P2PRequestDeadlineDoesNotRestartWithStreamBeginTime) {
    auto builder = GenerateStreamBuilder();
    auto stream  = builder.createMockStream({1, 2, 3}, 1001);
    const int64_t deadline_ms = autil::TimeUtility::currentTimeInMicroSeconds() / 1000 + 1000;
    stream->generate_input_->request_deadline_ms = deadline_ms;
    stream->resetBeginTime((deadline_ms + 5000) * 1000);
    EXPECT_EQ(stream->deadlineMs(), deadline_ms);
}

TEST_F(GenerateStreamTest, PrefillFallbackVariableBeamOutputUsesCompletedStep) {
    autil::EnvGuard perf_scope("PERF_TEST", "0");
    // Cover shrinking, expanding and reordered fixed-width beams, after prefill and decode.
    for (const auto& widths : std::vector<std::vector<int>>{{4, 2, 4}, {2, 4, 2}, {2, 2, 2}}) {
        for (const int max_new_tokens : {1, 2}) {
            SCOPED_TRACE(::testing::Message()
                         << "first_width=" << widths.front() << ", max_new_tokens=" << max_new_tokens);
            auto input                   = std::make_shared<GenerateInput>();
            input->input_ids             = torch::tensor({1, 2}, torch::kInt32);
            input->generate_config       = std::make_shared<GenerateConfig>();
            auto& config                 = *input->generate_config;
            config.variable_num_beams    = widths;
            config.max_new_tokens        = max_new_tokens;
            config.can_use_pd_separation = false;
            config.is_streaming          = true;
            config.ignore_eos            = true;
            config.reuse_cache           = false;
            config.return_cum_log_probs  = true;
            config.return_logits        = true;
            config.return_hidden_states = true;

            ModelConfig model_config;
            model_config.max_seq_len                  = 32;
            model_config.vocab_size                   = 128;
            model_config.attn_config.tokens_per_block = 2;
            ResourceContext resource_context;
            resource_context.role_type       = RoleType::PREFILL;
            resource_context.decode_entrance = true;
            resource_context.reuse_cache     = false;
            auto stream =
                std::make_shared<NormalGenerateStream>(input, model_config, RuntimeConfig{}, resource_context, nullptr);
            // Exercise sampled-token updates and output queuing without a model or KV allocation.
            stream->generate_status_->status = StreamState::RUNNING;
            ASSERT_FALSE(stream->queryPdSep());
            ASSERT_FALSE(stream->isStreaming());

            torch::Tensor        sampled_tokens;
            torch::Tensor        logits;
            torch::Tensor        hidden_states;
            std::vector<int32_t> source_rows;
            for (int step = 0; step < max_new_tokens; ++step) {
                const int beam_count = widths[step];
                sampled_tokens       = torch::empty({beam_count, input->inputLength() + step + 1}, torch::kInt32);
                for (int beam = 0; beam < beam_count; ++beam) {
                    auto* row = sampled_tokens.data_ptr<int32_t>() + beam * sampled_tokens.size(1);
                    row[0]    = 1;
                    row[1]    = 2;
                    for (int token = 0; token <= step; ++token) {
                        row[input->inputLength() + token] = 10 * (beam + 1) + token;
                    }
                }
                torch::Tensor src_batch_indices;
                if (step + 1 == max_new_tokens) {
                    const int input_rows = step == 0 ? 1 : widths[step - 1];
                    logits = torch::arange(input_rows * model_config.vocab_size, torch::kFloat32)
                                 .reshape({input_rows, model_config.vocab_size});
                    hidden_states = torch::arange(input_rows * 3, torch::kFloat32).reshape({input_rows, 3}) + 1000;
                    source_rows.resize(beam_count);
                    for (int beam = 0; beam < beam_count; ++beam) {
                        source_rows[beam] = input_rows - 1 - beam % input_rows;
                    }
                    // The final update skips KV remapping, so this remains a stream-only test.
                    src_batch_indices = torch::tensor(source_rows, torch::kInt32);
                }
                StreamUpdateInfo update_info{.new_tokens        = sampled_tokens,
                                             .num_new_tokens    = 1,
                                             .hidden_states     = hidden_states,
                                             .logits            = logits,
                                             .cum_log_probs     = torch::arange(beam_count, torch::kFloat32),
                                             .src_batch_indices = src_batch_indices};
                stream->update(update_info);
                ASSERT_FALSE(stream->hasError());
                if (step + 1 < max_new_tokens) {
                    ASSERT_FALSE(stream->hasOutput());
                }
            }

            ASSERT_TRUE(stream->hasEvent(StreamEvents::GenerateDone));
            ASSERT_TRUE(stream->hasOutput());
            auto result = stream->nextOutput();
            ASSERT_TRUE(result.ok());
            const auto& outputs = result.value().generate_outputs;
            ASSERT_EQ(outputs.size(), static_cast<size_t>(widths[max_new_tokens - 1]));
            for (size_t beam = 0; beam < outputs.size(); ++beam) {
                const auto& output = outputs[beam];
                EXPECT_TRUE(output.finished);
                EXPECT_FALSE(output.aux_info.pd_sep);
                EXPECT_EQ(output.aux_info.output_len, max_new_tokens);
                EXPECT_TRUE(
                    torch::equal(output.output_ids,
                                 sampled_tokens.narrow(0, beam, 1).narrow(1, input->inputLength(), max_new_tokens)));
                ASSERT_TRUE(output.aux_info.cum_log_probs.has_value());
                EXPECT_FLOAT_EQ(output.aux_info.cum_log_probs->item<float>(), static_cast<float>(beam));
                ASSERT_TRUE(output.logits.has_value());
                EXPECT_TRUE(torch::equal(*output.logits, logits.narrow(0, source_rows[beam], 1)));
                ASSERT_TRUE(output.hidden_states.has_value());
                EXPECT_TRUE(torch::equal(*output.hidden_states, hidden_states.narrow(0, source_rows[beam], 1)));
            }
            EXPECT_FALSE(stream->hasOutput());
        }
    }
}

}  // namespace rtp_llm
