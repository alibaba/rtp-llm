
#include "gtest/gtest.h"

#include "rtp_llm/cpp/devices/testing/TestBase.h"
#include "rtp_llm/cpp/models/logits_processor/ThinkModeLogitsProcessor.h"
#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorStates.h"
#include "rtp_llm/cpp/core/BufferHelper.h"

using namespace std;

namespace rtp_llm {

class SamplerDataBuilder {
public:
    SamplerDataBuilder(): device_(rtp_llm::DeviceFactory::getDefaultDevice()) {};

    struct Config {
        size_t            batch_size;
        size_t            vocab_size;
        size_t            max_length;
        rtp_llm::DataType logits_type = rtp_llm::DataType::TYPE_FP32;
    };

    BaseLogitsProcessorPtr generateLogitsProcessor(bool             think_mode,
                                                   std::vector<int> max_thinking_tokens,
                                                   std::vector<int> end_think_token_ids,
                                                   std::vector<int> think_status) {
        std::vector<StreamThinkInfo> think_infos;

        size_t batch_size = max_thinking_tokens.size();
        for (size_t i = 0; i < batch_size; i++) {
            auto think_info = StreamThinkInfo(think_mode,
                                              max_thinking_tokens[i],
                                              end_think_token_ids,
                                              0,
                                              0,
                                              0,
                                              std::make_shared<StringContainDFA<size_t, int>>(end_think_token_ids));
            think_info.dfa_ptr->forceSetStatus(think_status[i]);
            think_infos.push_back(think_info);
        }

        BaseLogitsProcessorPtr processor_ptr = std::make_shared<ThinkModeLogitsProcessor>(device_, think_infos);
        return processor_ptr;
    }

    SamplerInputs allocate(Config config, std::vector<BaseLogitsProcessorPtr> processors, std::vector<size_t> nums) {
        SamplerInputs sampler_inputs;

        sampler_inputs.step                = config.max_length;
        sampler_inputs.batch_size          = config.batch_size;
        sampler_inputs.batch_size_out      = config.batch_size;
        sampler_inputs.vocab_size          = config.vocab_size;
        LogitsProcessorStatesPtr state_ptr = std::make_shared<LogitsProcessorStates>();
        for (size_t i = 0, idx = 0; i < processors.size(); i++) {
            state_ptr->insert(processors[i], idx, idx + nums[i]);
            idx += nums[i];
        }
        sampler_inputs.logits_processor_states_ptr = state_ptr;
        sampler_inputs.logits                      = device_->allocateBuffer(
            {config.logits_type, {config.batch_size, config.vocab_size}, rtp_llm::AllocationType::HOST}, {});
        sampler_inputs.sequence_lengths = device_->allocateBuffer(
            {rtp_llm::DataType::TYPE_INT32, {config.batch_size}, rtp_llm::AllocationType::HOST}, {});
        sampler_inputs.input_lengths = device_->allocateBuffer(
            {rtp_llm::DataType::TYPE_INT32, {config.batch_size}, rtp_llm::AllocationType::HOST}, {});
        sampler_inputs.num_beams_in = device_->allocateBuffer(
            {rtp_llm::DataType::TYPE_UINT64, {config.batch_size}, rtp_llm::AllocationType::HOST}, {});
        sampler_inputs.num_beams_out = device_->allocateBuffer(
            {rtp_llm::DataType::TYPE_UINT64, {config.batch_size}, rtp_llm::AllocationType::HOST}, {});
        sampler_inputs.top_k = device_->allocateBuffer(
            {rtp_llm::DataType::TYPE_UINT32, {config.batch_size}, rtp_llm::AllocationType::HOST}, {});
        sampler_inputs.top_p = device_->allocateBuffer(
            {rtp_llm::DataType::TYPE_FP32, {config.batch_size}, rtp_llm::AllocationType::HOST}, {});
        sampler_inputs.temperature = device_->allocateBuffer(
            {rtp_llm::DataType::TYPE_FP32, {config.batch_size}, rtp_llm::AllocationType::HOST}, {});
        sampler_inputs.repetition_penalty = device_->allocateBuffer(
            {rtp_llm::DataType::TYPE_FP32, {config.batch_size}, rtp_llm::AllocationType::HOST}, {});
        sampler_inputs.cum_log_probs = device_->allocateBuffer(
            {rtp_llm::DataType::TYPE_FP32, {config.batch_size}, rtp_llm::AllocationType::HOST}, {});
        sampler_inputs.token_ids = device_->allocateBuffer({rtp_llm::DataType::TYPE_INT32,
                                                            {config.batch_size, sampler_inputs.step + 1},
                                                            rtp_llm::AllocationType::HOST},
                                                           {});
        device_->bufMemset(*sampler_inputs.logits, 0);
        device_->bufMemset(*sampler_inputs.token_ids, 0);
        return sampler_inputs;
    };

    void setSequenceLengths(SamplerInputs& sampler_inputs, std::vector<int>& sequence_lengths) {
        RTP_LLM_CHECK(sequence_lengths.size() == sampler_inputs.batch_size);
        sampler_inputs.sequence_lengths = rtp_llm::vector2Buffer(sequence_lengths);
    };

    void setTokenIds(SamplerInputs& sampler_inputs, std::vector<std::vector<int>>& token_ids) {
        RTP_LLM_CHECK(token_ids.size() == sampler_inputs.batch_size);
        RTP_LLM_CHECK(token_ids[0].size() == sampler_inputs.step + 1);
        for (auto i = 0; i < sampler_inputs.batch_size; i++) {
            auto tensor = Buffer2torchTensor(*sampler_inputs.token_ids->index(i), false);
            for (auto j = 0; j < sampler_inputs.step + 1; j++) {
                tensor[j] = token_ids[i][j];
            }
        }
    }

    rtp_llm::DeviceBase* device_;
};

class SamplerTest: public DeviceTestBase {
protected:
    void SetUp() override {
        DeviceTestBase::SetUp();
    }

    void TearDown() override {
        DeviceTestBase::TearDown();
    }

    rtp_llm::BufferPtr randint(int start, int end, std::vector<int64_t> shape, bool is_host) {
        auto tensor  = torch::randint(start, end, shape, at::TensorOptions().dtype(at::ScalarType::Int));
        auto alloc_t = is_host ? AllocationType::HOST : AllocationType::DEVICE;
        return tensorToBuffer(tensor, alloc_t);
    }

    rtp_llm::BufferPtr rand(std::vector<int64_t> shape, bool is_host) {
        auto tensor  = torch::rand(torch::IntArrayRef(shape));
        auto alloc_t = is_host ? AllocationType::HOST : AllocationType::DEVICE;
        return tensorToBuffer(tensor, alloc_t);
    }
};

#define EXPECT_SIMILAR(vec1, vec2, eps)                                                                                \
    do {                                                                                                               \
        bool similar = true;                                                                                           \
        if (vec1.size() != vec2.size()) {                                                                              \
            similar = false;                                                                                           \
        } else {                                                                                                       \
            for (size_t i = 0; i < vec1.size(); ++i) {                                                                 \
                if (std::fabs(vec1[i] - vec2[i]) >= eps) {                                                             \
                    similar = false;                                                                                   \
                    break;                                                                                             \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
        EXPECT_TRUE(similar) << "Vectors are not similar";                                                             \
    } while (0)

TEST_F(SamplerTest, testMemFill) {
    SamplerDataBuilder builder;

    std::vector<int>       end_think_token_ids = {101, 102};
    std::vector<int>       max_thinking_tokens = {3, 4, 5, 4};
    std::vector<int>       think_status        = {0, 0, 0, 0};
    BaseLogitsProcessorPtr processor =
        builder.generateLogitsProcessor(true, max_thinking_tokens, end_think_token_ids, think_status);

    SamplerInputs    sampler_inputs   = builder.allocate({4, 1024, 1024}, {processor}, {(size_t)4});
    std::vector<int> sequence_lengths = {1, 2, 3, 4};
    builder.setSequenceLengths(sampler_inputs, sequence_lengths);
    EXPECT_EQ(buffer2vector<int>(*sampler_inputs.sequence_lengths), std::vector<int>({1, 2, 3, 4}));

    torch::Tensor tensor2 = torch::tensor({{2, 2, 2, 2, 2}, {2, 2, 2, 2, 2}, {2, 2, 2, 2, 2}, {2, 2, 2, 2, 2}},
                                          torch::dtype(torch::kDouble));
    auto          logits2 = torchTensor2Buffer(tensor2);
    processor->memFill(logits2->index(0), 5, 0);
    processor->memFill(logits2->index(1), 5, 1);
    processor->memFill(logits2->index(2), 5, 2);
    processor->memFill(logits2->index(3), 5, 3);

    float neg_inf = -std::numeric_limits<float>::max();

    EXPECT_SIMILAR(
        buffer2vector<double>(*logits2->index(0)), std::vector<double>({1, neg_inf, neg_inf, neg_inf, neg_inf}), 1e-6);
    EXPECT_SIMILAR(
        buffer2vector<double>(*logits2->index(1)), std::vector<double>({neg_inf, 1, neg_inf, neg_inf, neg_inf}), 1e-6);
    EXPECT_SIMILAR(
        buffer2vector<double>(*logits2->index(2)), std::vector<double>({neg_inf, neg_inf, 1, neg_inf, neg_inf}), 1e-6);
    EXPECT_SIMILAR(
        buffer2vector<double>(*logits2->index(3)), std::vector<double>({neg_inf, neg_inf, neg_inf, 1, neg_inf}), 1e-6);
}

template<typename Dtype>
void setBuffer(rtp_llm::BufferPtr buf, std::vector<std::vector<Dtype>> content) {
    RTP_LLM_CHECK(buf->shape().size() == 2);
    RTP_LLM_CHECK(buf->shape()[0] == content.size());
    RTP_LLM_CHECK(buf->shape()[1] == content[0].size());
    for (auto i = 0; i < buf->shape()[0]; i++) {
        auto tensor = Buffer2torchTensor(*buf->index(i), false);
        for (auto j = 0; j < buf->shape()[1]; j++) {
            tensor[j] = content[i][j];
        }
    }
}

TEST_F(SamplerTest, testUpdateStatus) {
    {
        SamplerDataBuilder     builder;
        size_t                 batch_size          = 4;
        std::vector<int>       end_think_token_ids = {5};
        std::vector<int>       max_thinking_tokens = {3, 3, 3, 3};
        std::vector<int>       think_status        = {0, 0, 0, 0};
        BaseLogitsProcessorPtr processor =
            builder.generateLogitsProcessor(true, max_thinking_tokens, end_think_token_ids, think_status);

        rtp_llm::BufferPtr new_token = device_->allocateBuffer(
            {rtp_llm::DataType::TYPE_INT32, {batch_size, 1}, rtp_llm::AllocationType::HOST}, {});
        std::vector<std::vector<int>> new_token_ids = {{0}, {1}, {5}, {9}};
        setBuffer(new_token, new_token_ids);

        processor->updateStatus(new_token, 1);

        auto                proc        = std::dynamic_pointer_cast<ThinkModeLogitsProcessor>(processor);
        std::vector<size_t> status_list = proc->thinkEndTokensStatus();
        EXPECT_EQ(0, status_list[0]);
        EXPECT_EQ(0, status_list[1]);
        EXPECT_EQ(1, status_list[2]);
        EXPECT_EQ(0, status_list[3]);
    }

    {
        SamplerDataBuilder     builder;
        size_t                 batch_size          = 4;
        std::vector<int>       end_think_token_ids = {5, 5};
        std::vector<int>       max_thinking_tokens = {3, 3, 3, 3};
        std::vector<int>       think_status        = {0, 0, 1, 1};
        BaseLogitsProcessorPtr processor =
            builder.generateLogitsProcessor(true, max_thinking_tokens, end_think_token_ids, think_status);

        rtp_llm::BufferPtr new_token = device_->allocateBuffer(
            {rtp_llm::DataType::TYPE_INT32, {batch_size, 1}, rtp_llm::AllocationType::HOST}, {});
        std::vector<std::vector<int>> new_token_ids = {{0}, {1}, {5}, {9}};
        setBuffer(new_token, new_token_ids);

        processor->updateStatus(new_token, 1);

        auto                proc        = std::dynamic_pointer_cast<ThinkModeLogitsProcessor>(processor);
        std::vector<size_t> status_list = proc->thinkEndTokensStatus();
        EXPECT_EQ(0, status_list[0]);
        EXPECT_EQ(0, status_list[1]);
        EXPECT_EQ(2, status_list[2]);
        EXPECT_EQ(0, status_list[3]);
    }

    {
        SamplerDataBuilder     builder;
        size_t                 batch_size          = 4;
        std::vector<int>       end_think_token_ids = {5, 6};
        std::vector<int>       max_thinking_tokens = {3, 3, 3, 3};
        std::vector<int>       think_status        = {0, 0, 1, 1};
        BaseLogitsProcessorPtr processor =
            builder.generateLogitsProcessor(true, max_thinking_tokens, end_think_token_ids, think_status);

        rtp_llm::BufferPtr new_token = device_->allocateBuffer(
            {rtp_llm::DataType::TYPE_INT32, {batch_size, 1}, rtp_llm::AllocationType::HOST}, {});
        std::vector<std::vector<int>> new_token_ids = {{5}, {6}, {5}, {6}};
        setBuffer(new_token, new_token_ids);

        processor->updateStatus(new_token, 1);

        auto                proc        = std::dynamic_pointer_cast<ThinkModeLogitsProcessor>(processor);
        std::vector<size_t> status_list = proc->thinkEndTokensStatus();
        EXPECT_EQ(1, status_list[0]);
        EXPECT_EQ(0, status_list[1]);
        EXPECT_EQ(1, status_list[2]);
        EXPECT_EQ(2, status_list[3]);
    }
}

std::vector<float> tensorToVector(const at::Tensor& tensor, size_t size) {
    std::vector<float> vec(size, 0);
    for (size_t i = 0; i < tensor.size(0); ++i) {
        vec[i] = tensor[i].item<float>();
    }
    return vec;
}

TEST_F(SamplerTest, testSetVocabMask) {
    {
        SamplerDataBuilder     builder;
        size_t                 batch_size          = 4;
        size_t                 vocab_size          = 10;
        size_t                 max_length          = 10;
        std::vector<int>       end_think_token_ids = {5, 6};
        std::vector<int>       max_thinking_tokens = {3, 3, 3, 3};
        std::vector<int>       think_status        = {0, 0, 1, 1};
        BaseLogitsProcessorPtr processor =
            builder.generateLogitsProcessor(true, max_thinking_tokens, end_think_token_ids, think_status);

        SamplerInputs sampler_inputs =
            builder.allocate({batch_size, vocab_size, max_length}, {processor}, {batch_size});
        std::vector<int> sequence_lengths = {1, 2, 3, 4};
        builder.setSequenceLengths(sampler_inputs, sequence_lengths);
        EXPECT_EQ(buffer2vector<int>(*sampler_inputs.sequence_lengths), std::vector<int>({1, 2, 3, 4}));

        auto think_processor = std::dynamic_pointer_cast<ThinkModeLogitsProcessor>(processor);

        for (size_t i = 0; i < batch_size; i++) {
            think_processor->setVocabMask(think_processor->think_infos_[i].dfa_ptr,
                                          sampler_inputs.logits->index(i),
                                          1,
                                          end_think_token_ids,
                                          vocab_size,
                                          i % 2 == 0 ? true : false);
        }

        float neg_inf = -std::numeric_limits<float>::max();

        std::vector<float> expect_vec_0 = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        std::vector<float> expect_vec_1 = {
            neg_inf, neg_inf, neg_inf, neg_inf, neg_inf, 1, neg_inf, neg_inf, neg_inf, neg_inf};
        std::vector<float> expect_vec_2 = {
            neg_inf, neg_inf, neg_inf, neg_inf, neg_inf, neg_inf, 1, neg_inf, neg_inf, neg_inf};
        EXPECT_SIMILAR(
            expect_vec_1, tensorToVector(Buffer2torchTensor(*sampler_inputs.logits->index(0), false), 10), 1e-6);
        EXPECT_SIMILAR(
            expect_vec_0, tensorToVector(Buffer2torchTensor(*sampler_inputs.logits->index(1), false), 10), 1e-6);
        EXPECT_SIMILAR(
            expect_vec_2, tensorToVector(Buffer2torchTensor(*sampler_inputs.logits->index(2), false), 10), 1e-6);
        EXPECT_SIMILAR(
            expect_vec_0, tensorToVector(Buffer2torchTensor(*sampler_inputs.logits->index(3), false), 10), 1e-6);
    }
}

#undef EXPECT_SIMILAR

}  // namespace rtp_llm
