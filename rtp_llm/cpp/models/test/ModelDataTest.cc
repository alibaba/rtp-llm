
#include "gtest/gtest.h"

#include "rtp_llm/cpp/devices/testing/TestBase.h"
#include "rtp_llm/cpp/models/GptModel.h"
#include "rtp_llm/cpp/models/Sampler.h"
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

    SamplerInputs allocate(Config config) {
        SamplerInputs sampler_inputs;
        sampler_inputs.step           = config.max_length;
        sampler_inputs.batch_size     = config.batch_size;
        sampler_inputs.batch_size_out = config.batch_size;
        sampler_inputs.logits         = device_->allocateBuffer(
            {config.logits_type, {config.batch_size, config.vocab_size}, rtp_llm::AllocationType::DEVICE}, {});
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
        return sampler_inputs;
    };

    void setSequenceLengths(SamplerInputs& sampler_inputs, std::vector<int>& sequence_lengths) {
        RTP_LLM_CHECK(sequence_lengths.size() == sampler_inputs.batch_size);
        sampler_inputs.sequence_lengths = rtp_llm::vector2Buffer(sequence_lengths);
    };

    rtp_llm::DeviceBase* device_;
};

class ModelDataTest: public DeviceTestBase {
protected:
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

TEST_F(ModelDataTest, testConstruct) {
    SamplerDataBuilder builder;
    SamplerInputs      sampler_inputs   = builder.allocate({4, 1024, 1024});
    std::vector<int>   sequence_lengths = {1, 2, 3, 4};
    builder.setSequenceLengths(sampler_inputs, sequence_lengths);
    EXPECT_EQ(buffer2vector<int>(*sampler_inputs.sequence_lengths), std::vector<int>({1, 2, 3, 4}));
}

}  // namespace rtp_llm
