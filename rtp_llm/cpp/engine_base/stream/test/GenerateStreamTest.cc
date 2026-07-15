
#include "gtest/gtest.h"

#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include <functional>
#include <numeric>

using namespace std;

namespace rtp_llm {

using ChunkedInputConfigurer = std::function<void(GenerateInput&, GenerateConfig&)>;

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
        return createContextStreamImpl(input_ids, model_config_, runtime_config_, ResourceContext{}, {});
    };

    GenerateStreamPtr createChunkWindowStream(std::vector<int> input_ids) {
        ResourceContext resource_context;
        resource_context.cache_manager = std::make_shared<KVCacheManager>(init_config());
        ModelConfig model_config        = model_config_;
        model_config.vocab_size         = 2048;
        return createContextStreamImpl(input_ids, model_config, runtime_config_, resource_context, {});
    }

    GenerateStreamPtr createChunkedContextStream(std::vector<int>             input_ids,
                                                 RoleType                     role_type = RoleType::PREFILL,
                                                 int64_t                      chunk_size = 8,
                                                 const ChunkedInputConfigurer& configure  = {}) {
        RuntimeConfig runtime_config = runtime_config_;
        runtime_config.fifo_scheduler_config.prefill_chunk_size = chunk_size;

        ResourceContext resource_context;
        resource_context.role_type = role_type;
        return createContextStreamImpl(input_ids, model_config_, runtime_config, resource_context, configure);
    }

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
    GenerateStreamPtr createContextStreamImpl(std::vector<int>                 input_ids,
                                              const ModelConfig&               model_config,
                                              const RuntimeConfig&             runtime_config,
                                              const ResourceContext&           resource_context,
                                              const ChunkedInputConfigurer&    configure) {
        auto generate_input             = std::make_shared<GenerateInput>();
        generate_input->generate_config = std::make_shared<GenerateConfig>();
        generate_input->input_ids =
            torch::tensor(std::vector<int32_t>(input_ids.begin(), input_ids.end()), torch::kInt32);
        if (configure) {
            configure(*generate_input, *generate_input->generate_config);
        }
        return std::make_shared<NormalGenerateStream>(
            generate_input, model_config, runtime_config, resource_context, nullptr);
    }

    ModelConfig   model_config_;
    RuntimeConfig runtime_config_;
};

class GenerateStreamTest: public DeviceTestBase {
protected:
    // useChunkWindow() gates on getStatus() == RUNNING. Unit tests skip the real
    // state machine (WAITING → LOADING_CACHE → RUNNING) and force the status directly.
    static void markRunning(const GenerateStreamPtr& s) {
        s->generate_status_->status = StreamState::RUNNING;
    }

    static std::vector<int> makeInputIds(int count) {
        std::vector<int> input_ids(count);
        std::iota(input_ids.begin(), input_ids.end(), 1);
        return input_ids;
    }

    static void expectChunkWindow(const GenerateStreamPtr& stream, int prefix, int length, bool is_last) {
        EXPECT_EQ(stream->prefixLength(), prefix);
        EXPECT_EQ(stream->currentChunkLen(), length);
        EXPECT_EQ(stream->contextLength(), length);
        EXPECT_EQ(stream->isLastChunk(), is_last);
        EXPECT_EQ(stream->isMiddleChunk(), !is_last);
    }

    static StreamUpdateInfo singleTokenUpdate(int token_id) {
        return {torch::tensor(std::vector<int32_t>{token_id}, torch::kInt32).reshape({1, 1}),
                1,
                torch::Tensor(),
                torch::Tensor(),
                torch::Tensor(),
                torch::Tensor(),
                torch::Tensor(),
                torch::Tensor(),
                torch::Tensor(),
                torch::Tensor()};
    }
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

TEST_F(GenerateStreamTest, testChunkedPrefillWindowAndUpdate) {
    auto builder = GenerateStreamBuilder();
    auto stream  = builder.createChunkWindowStream(makeInputIds(10));
    ASSERT_TRUE(stream->isContextStream());

    ASSERT_FALSE(stream->useChunkWindow());
    ASSERT_EQ(stream->contextLength(), 10);
    ASSERT_EQ(stream->prefixLength(), 0);

    stream->setChunkSize(/*chunk_size=*/8);
    ASSERT_FALSE(stream->useChunkWindow());
    ASSERT_EQ(stream->contextLength(), 10);

    markRunning(stream);
    expectChunkWindow(stream, /*prefix=*/0, /*length=*/8, /*is_last=*/false);

    const int  original_seq_len = stream->seqLength();
    const auto update_info      = singleTokenUpdate(101);
    stream->update(update_info);
    ASSERT_TRUE(stream->isContextStream());
    ASSERT_EQ(stream->seqLength(), original_seq_len);
    expectChunkWindow(stream, /*prefix=*/8, /*length=*/2, /*is_last=*/true);

    stream->update(update_info);
    ASSERT_FALSE(stream->isContextStream());
    ASSERT_EQ(stream->seqLength(), original_seq_len + 1);
}

TEST_F(GenerateStreamTest, testChunkedPrefillReuseStartAndInitialReuseFrozen) {
    auto builder = GenerateStreamBuilder();
    auto stream  = builder.createChunkWindowStream(makeInputIds(18));
    stream->setReuseLength(8);
    stream->setInitialReuseLength(8);
    stream->setChunkSize(/*chunk_size=*/8);
    markRunning(stream);

    expectChunkWindow(stream, /*prefix=*/8, /*length=*/8, /*is_last=*/false);
    stream->advanceChunk();
    expectChunkWindow(stream, /*prefix=*/16, /*length=*/2, /*is_last=*/true);
    ASSERT_EQ(stream->initialReuseLength(), 8);
}

TEST_F(GenerateStreamTest, testChunkedPrefillSpecUpdateMiddleChunkDiscards) {
    auto builder = GenerateStreamBuilder();
    auto stream  = builder.createChunkWindowStream(makeInputIds(10));
    stream->setChunkSize(/*chunk_size=*/8);
    markRunning(stream);
    const int original_seq_len = stream->seqLength();

    auto new_tokens = torch::tensor(std::vector<int32_t>{101}, torch::kInt32).reshape({1, 1});
    stream->specUpdate({new_tokens, 1, /*draft_token=*/42, torch::Tensor(), torch::Tensor()});
    ASSERT_TRUE(stream->isContextStream());
    ASSERT_FALSE(stream->hasError());
    ASSERT_EQ(stream->seqLength(), original_seq_len);
    ASSERT_EQ(stream->prefixLength(), 8);
    ASSERT_TRUE(stream->isLastChunk());
}

TEST_F(GenerateStreamTest, testChunkedPrefillActivationGates) {
    GenerateStreamBuilder builder;
    const auto            input_ids = makeInputIds(9);

    struct RoleCase {
        const char* name;
        RoleType    role;
        bool        enabled;
    };
    const RoleCase role_cases[] = {{"prefill", RoleType::PREFILL, true},
                                   {"pdfusion", RoleType::PDFUSION, true},
                                   {"decode", RoleType::DECODE, false}};
    for (const auto& test_case : role_cases) {
        SCOPED_TRACE(test_case.name);
        auto stream = builder.createChunkedContextStream(input_ids, test_case.role);
        EXPECT_EQ(stream->chunkedPrefillEnabled(), test_case.enabled);
        EXPECT_FALSE(stream->hasError());
    }

    struct RejectionCase {
        const char*            name;
        ChunkedInputConfigurer configure;
    };
    const std::vector<RejectionCase> rejection_cases{
        {"force_batch", [](GenerateInput&, GenerateConfig& config) { config.force_batch = true; }},
        {"num_beams", [](GenerateInput&, GenerateConfig& config) { config.num_beams = 2; }},
        {"return_logits", [](GenerateInput&, GenerateConfig& config) { config.return_logits = true; }},
        {"calculate_loss", [](GenerateInput&, GenerateConfig& config) { config.calculate_loss = 1; }},
        {"return_hidden_states", [](GenerateInput&, GenerateConfig& config) { config.return_hidden_states = true; }},
        {"return_all_hidden_states",
         [](GenerateInput&, GenerateConfig& config) { config.return_all_hidden_states = true; }},
        {"return_all_probs",
         [](GenerateInput&, GenerateConfig& config) { config.return_all_probs = ReturnAllProbsMode::DEFAULT; }},
        {"multimodal",
         [](GenerateInput& input, GenerateConfig&) {
             input.multimodal_features = std::vector<torch::Tensor>{torch::zeros({1, 1}, torch::kFloat32)};
         }},
    };
    for (const auto& test_case : rejection_cases) {
        SCOPED_TRACE(test_case.name);
        auto stream = builder.createChunkedContextStream(
            input_ids, RoleType::PREFILL, /*chunk_size=*/8, test_case.configure);
        EXPECT_FALSE(stream->chunkedPrefillEnabled());
        EXPECT_EQ(stream->stopReason(), "chunked prefill incompatible: " + std::string(test_case.name));
    }

    auto non_chunked_force_batch_stream = builder.createChunkedContextStream(
        input_ids,
        RoleType::PREFILL,
        /*chunk_size=*/0,
        [](GenerateInput&, GenerateConfig& config) { config.force_batch = true; });
    ASSERT_FALSE(non_chunked_force_batch_stream->chunkedPrefillEnabled());
    ASSERT_FALSE(non_chunked_force_batch_stream->hasError());
}

}  // namespace rtp_llm
