#include "maga_transformer/cpp/speculative_engine/SpeculativeStreamOutput.h"
#include "src/fastertransformer/devices/testing/TestBase.h"
#include "maga_transformer/cpp/utils/Logger.h"
#include <memory>
#define private public
#include "maga_transformer/cpp/speculative_engine/speculative_sampler/RejectionSampler.h"


using namespace std;
using namespace rtp_llm;
using namespace fastertransformer;
// TODO: make this test device-independent
class SpeculativeSamplerTest : public DeviceTestBase {
public:
};

TEST_F(SpeculativeSamplerTest, top1Sample) {
    std::unique_ptr<RejectionSampler> sampler = std::make_unique<RejectionSampler>(device_);
    size_t propose_step = 5;
    SpeculativeExecutorStreamOutputPtr propose_stream_output = std::make_shared<SpeculativeExecutorStreamOutput>();
    SpeculativeExecutorStreamOutputPtr score_stream_output = std::make_shared<SpeculativeExecutorStreamOutput>();
    propose_stream_output->tokens = createBuffer<int32_t>({5}, {1, 2, 3, 4, 5}, ft::AllocationType::HOST);
    score_stream_output->tokens = createBuffer<int32_t>({5}, {1, 2, 5, 3, 4}, ft::AllocationType::HOST);

    size_t accepted_len = sampler->top1Sample(propose_step, propose_stream_output, score_stream_output);
    ASSERT_EQ(accepted_len, 3);
}


TEST_F(SpeculativeSamplerTest, stochasticSample) {
        std::unique_ptr<RejectionSampler> sampler = std::make_unique<RejectionSampler>(device_);
    size_t propose_step = 5;
    SpeculativeExecutorStreamOutputPtr propose_stream_output = std::make_shared<SpeculativeExecutorStreamOutput>();
    SpeculativeExecutorStreamOutputPtr score_stream_output = std::make_shared<SpeculativeExecutorStreamOutput>();
    propose_stream_output->tokens = createBuffer<int32_t>({5}, {1, 2, 3, 4, 5}, ft::AllocationType::HOST);
    score_stream_output->tokens = createBuffer<int32_t>({5}, {1, 2, 5, 3, 4}, ft::AllocationType::HOST);
    propose_stream_output->all_probs = createBuffer<float>({5, 6}, {
        0.1, 0.1, 0.2, 0.1, 0.3, 0.2,
         0, 0, 1, 0, 0, 0,  
        0.1, 0.1, 0.2, 0.3, 0.1, 0.2,
        0.1, 0.1, 0.3, 0.0, 0.5, 0.0,
        0.1, 0.1, 0.2, 0.3, 0.1, 0.2,
    }, ft::AllocationType::HOST);
    score_stream_output->all_probs = createBuffer<float>({5, 6}, {
        0.1, 0.2, 0.1, 0.1, 0.3, 0.2,
        0, 0, 1, 0, 0, 0, 
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.1, 0.1, 0.3, 0.1, 0.2, 0.2,
        0.1, 0.1, 0.3, 0.0, 0.5, 0.0,
    }, ft::AllocationType::HOST);
    size_t accepted_len = sampler->top1Sample(propose_step, propose_stream_output, score_stream_output);
    ASSERT_EQ(accepted_len, 3);
}

