#include <cstdlib>
#include "torch/all.h"
#include <ATen/cuda/CUDAContext.h>
#include "rtp_llm/models_py/bindings/cuda/kernels/greedy_rejection.h"
#include "gtest/gtest.h"
#define private public
#include "rtp_llm/cpp/normal_engine/speculative/MtpBatchStreamProcessor.h"
#undef private
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/testing/TestBase.h"

namespace rtp_llm {
class MtpCompactSamplerHistoryTest: public DeviceTestBase {};

void checkWorkspace(int proposals, bool history_penalty = false) {
    const char* flag = std::getenv("RTP_LLM_MTP_COMPACT_SAMPLER_HISTORY");
    const bool compact = (flag == nullptr || std::string(flag) == "1") && !history_penalty;
    ModelConfig model_config;
    model_config.max_seq_len = 83968;
    model_config.vocab_size = 4;
    model_config.num_layers = 1;
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size = 2;
    SpeculativeExecutionConfig sp_config;
    sp_config.gen_num_per_cycle = proposals;
    PDSepConfig pd_config;
    ProfilingDebugLoggingConfig logging_config;
    CacheConfig cache_config;
    cache_config.group_types = {CacheGroupType::FULL};
    ResourceContext resources;
    std::list<GenerateStreamPtr> streams;
    for (int i = 0; i < 2; ++i) {
        auto input = std::make_shared<GenerateInput>();
        input->input_ids = torch::ones({81920}, torch::kInt32);
        input->generate_config = std::make_shared<GenerateConfig>();
        input->generate_config->top_k = 1;
        input->generate_config->top_p = 1.0f;
        input->generate_config->temperature = 1.0f;
        input->generate_config->repetition_penalty = history_penalty ? 1.1f : 1.0f;
        input->generate_config->do_sample = false;
        input->generate_config->ignore_eos = true;
        auto stream = std::make_shared<NormalGenerateStream>(input, model_config, runtime_config, resources, nullptr);
        BatchKVCacheResource cache;
        cache.resetBatchSize(1);
        cache.initGroups(1, 1, {0});
        cache.setBatchBlocks(0, 0, {i + 1});
        stream->setKVCache(cache);
        stream->setReturnAllProbs(ReturnAllProbsMode::DEFAULT);
        stream->generate_status_->status = StreamState::RUNNING;
        stream->setNeedReleaseResource(false);
        stream->setScoreLen(proposals + 1);
        streams.push_back(stream);
    }
    StreamGroups groups(streams);
    MtpBatchStreamProcessor processor(model_config, pd_config, logging_config, cache_config, sp_config, false);
    GptModelInputs inputs;
    GptModelOutputs outputs;
    outputs.logits = torch::zeros({2 * (proposals + 1), 4}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    auto result = processor.gatherSpecSamplerInput(groups, inputs, outputs);
    ASSERT_TRUE(result.ok());
    const auto& sampler = result.value();
    EXPECT_EQ(sampler.token_ids.is_cuda(), compact);
    EXPECT_EQ(sampler.token_ids.scalar_type(), torch::kInt32);
    EXPECT_EQ(sampler.token_ids.size(0), 2 * (proposals + 1));
    EXPECT_EQ(sampler.token_ids.size(1), compact ? 1 : sampler.step + 1);
    EXPECT_GE(sampler.step, 81920);
}
TEST_F(MtpCompactSamplerHistoryTest, LongContextMtp3) {
    checkWorkspace(3);
}

TEST_F(MtpCompactSamplerHistoryTest, LongContextMtp4) {
    checkWorkspace(4);
}

TEST_F(MtpCompactSamplerHistoryTest, HistoryPenaltyRequiresFullWorkspace) {
    checkWorkspace(4, true);
}
TEST_F(MtpCompactSamplerHistoryTest, GreedyAcceptanceMatchesEveryRejectionPosition) {
    for (int proposals : {0, 1, 3, 4, 5, 31}) {
        for (int stride : {1, 3}) {
            const int batch = proposals + 1;
            auto draft = torch::empty({batch, proposals}, torch::kInt32);
            auto target = torch::full({batch * (proposals + 1), stride}, -1, torch::kInt32);
            auto expected = torch::zeros({batch, proposals + 1}, torch::kInt32);
            auto expected_lengths = torch::empty({batch}, torch::kInt32);
            for (int row = 0; row < batch; ++row) {
                expected_lengths.data_ptr<int32_t>()[row] = row + 1;
                for (int col = 0; col <= proposals; ++col) {
                    const int token = 100 + row * 32 + col;
                    target.data_ptr<int32_t>()[(row * (proposals + 1) + col) * stride + stride - 1] = token;
                    if (col < proposals) {
                        draft.data_ptr<int32_t>()[row * proposals + col] = token + (col == row ? 1 : 0);
                    }
                    if (col <= row) {
                        expected.data_ptr<int32_t>()[row * (proposals + 1) + col] = token;
                    }
                }
            }
            auto draft_d = draft.to(torch::kCUDA);
            auto target_d = target.to(torch::kCUDA);
            auto output = torch::empty_like(expected, expected.options().device(torch::kCUDA));
            auto accepted = torch::empty_like(expected_lengths, expected_lengths.options().device(torch::kCUDA));
            auto stream = at::cuda::getCurrentCUDAStream();
            ASSERT_EQ(rtpLlmGreedyRejection(draft_d.data_ptr<int32_t>(), target_d.data_ptr<int32_t>(),
                                           output.data_ptr<int32_t>(), accepted.data_ptr<int32_t>(),
                                           batch, proposals, stride, stream), cudaSuccess);
            ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
            EXPECT_TRUE(torch::equal(output.cpu(), expected)) << "proposals=" << proposals << " stride=" << stride;
            EXPECT_TRUE(torch::equal(accepted.cpu(), expected_lengths));
        }
    }
}
}  // namespace rtp_llm
