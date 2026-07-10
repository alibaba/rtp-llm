
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
        model_config_.vocab_size  = 2048;
    }

    CacheConfig init_config() {
        return test::makeSimpleMhaCacheConfig(
            /*layer_num=*/3, /*block_num=*/9, /*tokens_per_block=*/8, rtp_llm::DataType::TYPE_INT8);
    }

    GenerateStreamPtr createContextStream(std::vector<int> input_ids) {
        std::shared_ptr<GenerateConfig> generate_config(new GenerateConfig());
        ResourceContext                 resource_context;
        return createContextStreamWithConfig(input_ids, runtime_config_, resource_context, generate_config, false);
    };

    GenerateStreamPtr createChunkedContextStream(std::vector<int> input_ids,
                                                 RoleType         role_type        = RoleType::PREFILL,
                                                 int              num_beams        = 1,
                                                 bool             return_logits    = false,
                                                 int              calculate_loss   = 0,
                                                 ReturnAllProbsMode return_all_probs = ReturnAllProbsMode::NONE,
                                                 bool             has_multimodal   = false,
                                                 bool             return_hidden_states = false,
                                                 bool             return_all_hidden_states = false) {
        RuntimeConfig runtime_config = runtime_config_;
        runtime_config.fifo_scheduler_config.prefill_chunk_size = 8;

        ResourceContext resource_context;
        resource_context.role_type = role_type;

        std::shared_ptr<GenerateConfig> generate_config(new GenerateConfig());
        generate_config->num_beams        = num_beams;
        generate_config->return_logits    = return_logits;
        generate_config->calculate_loss   = calculate_loss;
        generate_config->return_all_probs = return_all_probs;
        generate_config->return_hidden_states     = return_hidden_states;
        generate_config->return_all_hidden_states = return_all_hidden_states;

        return createContextStreamWithConfig(input_ids, runtime_config, resource_context, generate_config, has_multimodal);
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
    GenerateStreamPtr createContextStreamWithConfig(std::vector<int>                input_ids,
                                                    const RuntimeConfig&            runtime_config,
                                                    const ResourceContext&          resource_context,
                                                    std::shared_ptr<GenerateConfig> generate_config,
                                                    bool                            has_multimodal) {
        std::shared_ptr<GenerateInput> generate_input(new GenerateInput());
        generate_input->generate_config = generate_config;
        generate_input->input_ids =
            torch::tensor(std::vector<int32_t>(input_ids.begin(), input_ids.end()), torch::kInt32);
        if (has_multimodal) {
            generate_input->multimodal_features =
                std::vector<torch::Tensor>{torch::zeros({1, 1}, torch::kFloat32)};
        }
        ResourceContext stream_resource_context = resource_context;
        if (!stream_resource_context.cache_manager) {
            stream_resource_context.cache_manager = std::make_shared<KVCacheManager>(init_config());
        }
        return std::make_shared<NormalGenerateStream>(
            generate_input, model_config_, runtime_config, stream_resource_context, nullptr);
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
};

TEST_F(GenerateStreamTest, testConstruct) {
    auto builder = GenerateStreamBuilder();
    auto stream1 = builder.createContextStream({1, 2, 3, 4, 5});
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

// chunked prefill: 窗口函数 + chunk 推进的核心数学（不依赖 server / PD / 真 role）
TEST_F(GenerateStreamTest, testChunkedPrefillWindow) {
    auto builder = GenerateStreamBuilder();
    auto stream  = builder.createContextStream(
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18});  // seqLength = 18
    ASSERT_TRUE(stream->isContextStream());

    // 未启用 chunked -> 整段语义
    ASSERT_FALSE(stream->useChunkWindow());
    ASSERT_EQ(stream->contextLength(), 18);
    ASSERT_EQ(stream->prefixLength(), 0);

    // 启用但仍在 WAITING（模拟准入期）-> 回退整段（useChunkWindow gated on RUNNING）
    stream->setChunkSize(/*chunk_size=*/8);
    ASSERT_FALSE(stream->useChunkWindow());
    ASSERT_EQ(stream->contextLength(), 18);

    // 转 RUNNING 后才切到 chunk 窗口
    markRunning(stream);
    ASSERT_TRUE(stream->useChunkWindow());

    // 第 0 块 [0,8)
    ASSERT_EQ(stream->prefixLength(), 0);
    ASSERT_EQ(stream->currentChunkLen(), 8);
    ASSERT_EQ(stream->contextLength(), 8);
    ASSERT_FALSE(stream->isLastChunk());
    ASSERT_TRUE(stream->isChunkedMiddleChunk());
    stream->advanceChunk();

    // 第 1 块 [8,16)
    ASSERT_EQ(stream->prefixLength(), 8);
    ASSERT_EQ(stream->currentChunkLen(), 8);
    ASSERT_EQ(stream->contextLength(), 8);
    ASSERT_FALSE(stream->isLastChunk());
    ASSERT_TRUE(stream->isChunkedMiddleChunk());
    stream->advanceChunk();

    // 第 2 块（末块）[16,18) 余数 2
    ASSERT_EQ(stream->prefixLength(), 16);
    ASSERT_EQ(stream->currentChunkLen(), 2);
    ASSERT_EQ(stream->contextLength(), 2);
    ASSERT_TRUE(stream->isLastChunk());
    ASSERT_FALSE(stream->isChunkedMiddleChunk());  // 末块不是中间块
}

// 短 prompt（< chunk_size）：单块即末块，不是中间块
TEST_F(GenerateStreamTest, testChunkedPrefillShortPromptSingleChunk) {
    auto builder = GenerateStreamBuilder();
    auto stream  = builder.createContextStream({1, 2, 3});  // seqLength = 3
    stream->setChunkSize(/*chunk_size=*/8);
    markRunning(stream);
    ASSERT_EQ(stream->currentChunkLen(), 3);  // min(8, 3-0)
    ASSERT_EQ(stream->contextLength(), 3);
    ASSERT_TRUE(stream->isLastChunk());
    ASSERT_FALSE(stream->isChunkedMiddleChunk());
}


// initial_reuse_length_ / initialReuseLength() must stay frozen after chunk advance
// so cache-hit-rate metric is not diluted by chunked forward progress.
TEST_F(GenerateStreamTest, testChunkedPrefillInitialReuseFrozenAcrossChunks) {
    auto builder = GenerateStreamBuilder();
    auto stream  = builder.createContextStream(
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18});
    stream->setReuseLength(8);
    stream->setInitialReuseLength(8);
    stream->setChunkSize(/*chunk_size=*/8);
    markRunning(stream);

    ASSERT_EQ(stream->initialReuseLength(), 8);
    stream->advanceChunk();  // reuse_length_ 8 -> 16 (last-chunk start)
    ASSERT_EQ(stream->prefixLength(), 16);
    ASSERT_EQ(stream->initialReuseLength(), 8);  // frozen; hit-rate metric uses this
}

TEST_F(GenerateStreamTest, testChunkedPrefillReuseStart) {
    auto builder = GenerateStreamBuilder();
    auto stream  = builder.createContextStream(
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18});
    stream->setReuseLength(8);
    stream->setInitialReuseLength(8);  // freeze cache-hit prefix used by hit-rate metrics
    stream->setChunkSize(/*chunk_size=*/8);
    markRunning(stream);

    ASSERT_EQ(stream->prefixLength(), 8);
    ASSERT_EQ(stream->currentChunkLen(), 8);
    ASSERT_FALSE(stream->isLastChunk());
    stream->advanceChunk();

    ASSERT_EQ(stream->prefixLength(), 16);
    ASSERT_EQ(stream->currentChunkLen(), 2);
    ASSERT_TRUE(stream->isLastChunk());
}

TEST_F(GenerateStreamTest, testChunkedPrefillUpdateMiddleAndLastChunk) {
    auto builder = GenerateStreamBuilder();
    auto stream  = builder.createContextStream(
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18});
    stream->setChunkSize(/*chunk_size=*/8);
    markRunning(stream);
    const int original_seq_len = stream->seqLength();

    auto new_tokens = torch::tensor(std::vector<int32_t>{101}, torch::kInt32).reshape({1, 1});
    stream->update({new_tokens,
                    1,
                    torch::Tensor(),
                    torch::Tensor(),
                    torch::Tensor(),
                    torch::Tensor(),
                    torch::Tensor(),
                    torch::Tensor(),
                    torch::Tensor(),
                    torch::Tensor()});
    ASSERT_TRUE(stream->isContextStream());
    ASSERT_EQ(stream->seqLength(), original_seq_len);
    ASSERT_EQ(stream->prefixLength(), 8);

    stream->update({new_tokens,
                    1,
                    torch::Tensor(),
                    torch::Tensor(),
                    torch::Tensor(),
                    torch::Tensor(),
                    torch::Tensor(),
                    torch::Tensor(),
                    torch::Tensor(),
                    torch::Tensor()});
    ASSERT_TRUE(stream->isContextStream());
    ASSERT_EQ(stream->seqLength(), original_seq_len);
    ASSERT_EQ(stream->prefixLength(), 16);
    ASSERT_TRUE(stream->isLastChunk());

    stream->update({new_tokens,
                    1,
                    torch::Tensor(),
                    torch::Tensor(),
                    torch::Tensor(),
                    torch::Tensor(),
                    torch::Tensor(),
                    torch::Tensor(),
                    torch::Tensor(),
                    torch::Tensor()});
    ASSERT_FALSE(stream->isContextStream());
    ASSERT_EQ(stream->seqLength(), original_seq_len + 1);
}

// Middle chunk specUpdate is compute-then-discard: no decode flip, no token append.
// Last-chunk specUpdate uses the existing spec path and is covered by MTP tests.
TEST_F(GenerateStreamTest, testChunkedPrefillSpecUpdateMiddleChunkDiscards) {
    auto builder = GenerateStreamBuilder();
    auto stream  = builder.createContextStream(
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18});
    stream->setChunkSize(/*chunk_size=*/8);
    markRunning(stream);
    const int original_seq_len = stream->seqLength();

    auto new_tokens = torch::tensor(std::vector<int32_t>{101}, torch::kInt32).reshape({1, 1});

    // Middle chunk 1: specUpdate should early-return and advance the chunk window.
    stream->specUpdate({new_tokens, 1, /*draft_token=*/42, torch::Tensor(), torch::Tensor()});
    ASSERT_TRUE(stream->isContextStream());
    ASSERT_FALSE(stream->hasError());
    ASSERT_EQ(stream->seqLength(), original_seq_len);
    ASSERT_EQ(stream->prefixLength(), 8);

    // Middle chunk 2: same.
    stream->specUpdate({new_tokens, 1, /*draft_token=*/43, torch::Tensor(), torch::Tensor()});
    ASSERT_TRUE(stream->isContextStream());
    ASSERT_FALSE(stream->hasError());
    ASSERT_EQ(stream->prefixLength(), 16);
    ASSERT_TRUE(stream->isLastChunk());
}

TEST_F(GenerateStreamTest, testChunkedPrefillActivationGates) {
    auto                   builder = GenerateStreamBuilder();
    const std::vector<int> input_ids{1, 2, 3, 4, 5, 6, 7, 8, 9};

    // PREFILL / PDFUSION roles + no request-level exclusions -> chunked enabled, no error.
    for (auto role : {RoleType::PREFILL, RoleType::PDFUSION}) {
        auto s = builder.createChunkedContextStream(input_ids, role);
        ASSERT_TRUE(s->chunkedPrefillEnabled());
        ASSERT_FALSE(s->hasError());
    }
    // Non-{PREFILL, PDFUSION} roles: chunked silently disabled (role is a node property).
    {
        auto s = builder.createChunkedContextStream(input_ids, RoleType::DECODE);
        ASSERT_FALSE(s->chunkedPrefillEnabled());
        ASSERT_FALSE(s->hasError());
    }
    // Request-level exclusions: chunked disabled AND the stream is rejected with an error.
    auto expect_rejected = [&](GenerateStreamPtr s) {
        ASSERT_FALSE(s->chunkedPrefillEnabled());
        ASSERT_TRUE(s->hasError());
    };
    expect_rejected(builder.createChunkedContextStream(input_ids, RoleType::PREFILL, /*num_beams=*/2));
    expect_rejected(builder.createChunkedContextStream(input_ids,
                                                      RoleType::PREFILL,
                                                      /*num_beams=*/1,
                                                      /*return_logits=*/true));
    expect_rejected(builder.createChunkedContextStream(input_ids,
                                                      RoleType::PREFILL,
                                                      /*num_beams=*/1,
                                                      /*return_logits=*/false,
                                                      /*calculate_loss=*/1));
    expect_rejected(builder.createChunkedContextStream(input_ids,
                                                      RoleType::PREFILL,
                                                      /*num_beams=*/1,
                                                      /*return_logits=*/false,
                                                      /*calculate_loss=*/0,
                                                      ReturnAllProbsMode::DEFAULT));
    expect_rejected(builder.createChunkedContextStream(input_ids,
                                                      RoleType::PREFILL,
                                                      /*num_beams=*/1,
                                                      /*return_logits=*/false,
                                                      /*calculate_loss=*/0,
                                                      ReturnAllProbsMode::NONE,
                                                      /*has_multimodal=*/true));
    expect_rejected(builder.createChunkedContextStream(input_ids,
                                                      RoleType::PREFILL,
                                                      /*num_beams=*/1,
                                                      /*return_logits=*/false,
                                                      /*calculate_loss=*/0,
                                                      ReturnAllProbsMode::NONE,
                                                      /*has_multimodal=*/false,
                                                      /*return_hidden_states=*/true));
    expect_rejected(builder.createChunkedContextStream(input_ids,
                                                      RoleType::PREFILL,
                                                      /*num_beams=*/1,
                                                      /*return_logits=*/false,
                                                      /*calculate_loss=*/0,
                                                      ReturnAllProbsMode::NONE,
                                                      /*has_multimodal=*/false,
                                                      /*return_hidden_states=*/false,
                                                      /*return_all_hidden_states=*/true));
}

TEST_F(GenerateStreamTest, testChunkedPrefillActivationNotDisabledByPerfTestEnv) {
    autil::EnvGuard perf_scope("PERF_TEST", "1");
    auto            builder = GenerateStreamBuilder();
    auto            stream  = builder.createChunkedContextStream({1, 2, 3, 4, 5, 6, 7, 8, 9}, RoleType::PDFUSION);

    ASSERT_TRUE(stream->isPerfTest());
    ASSERT_TRUE(stream->chunkedPrefillEnabled());
    ASSERT_FALSE(stream->hasError());
}

TEST_F(GenerateStreamTest, testChunkedPrefillInvariantRejectsUnalignedReuse) {
    auto builder = GenerateStreamBuilder();
    auto stream  = builder.createContextStream({1, 2, 3, 4, 5, 6, 7, 8, 9});
    stream->setReuseLength(1);
    stream->setChunkSize(/*chunk_size=*/8);
    markRunning(stream);
    EXPECT_FALSE(stream->checkChunkWindowInvariant());
    EXPECT_FALSE(stream->hasError());
    stream->advanceChunk();
    EXPECT_TRUE(stream->hasError());
}

}  // namespace rtp_llm
