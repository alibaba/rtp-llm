
#include "gtest/gtest.h"

#include "maga_transformer/cpp/stream/GenerateStream.h"
#include "maga_transformer/cpp/normal_engine/NormalGenerateStream.h"
#include "src/fastertransformer/devices/testing/TestBase.h"


using namespace std;

namespace rtp_llm {

class GenerateStreamBuilder {
public:

    GenerateStreamBuilder(ft::GptInitParameter params) :
        params_(params), device_(ft::DeviceFactory::getDefaultDevice()) {
        params_.max_seq_len_ = 2048;
    }

    GenerateStreamPtr createContextStream(std::vector<int> input_ids) {
        std::shared_ptr<GenerateInput>  generate_input(new GenerateInput());
        std::shared_ptr<GenerateConfig> generate_config(new GenerateConfig());
        ResourceContext resource_context;
        generate_input->generate_config = generate_config;
        generate_input->input_ids = ft::vector2Buffer(input_ids);
        return std::make_shared<NormalGenerateStream>(generate_input, params_, resource_context, nullptr);
    };

    GenerateStreamPtr createDecoderStream(std::vector<int> input_ids, std::vector<int> new_token_ids) {
        std::shared_ptr<GenerateInput>  generate_input(new GenerateInput());
        std::shared_ptr<GenerateConfig> generate_config(new GenerateConfig());
        ResourceContext resource_context;
        generate_input->generate_config = generate_config;
        generate_input->input_ids = ft::vector2Buffer(input_ids);
        auto stream_ptr = std::make_shared<NormalGenerateStream>(generate_input, params_, resource_context, nullptr);
        stream_ptr->setIsContextStream(false);
        auto new_tokens_ptr = ft::vector2Buffer(new_token_ids);
        device_->copy({*(stream_ptr->completeTokenIds()->index(0)->slice(stream_ptr->seqLength(), new_token_ids.size())),
                       *new_tokens_ptr});
        stream_ptr->setSeqLength(stream_ptr->seqLength() + new_token_ids.size());
        return stream_ptr;

    };

private:
    ft::GptInitParameter params_;
    ft::DeviceBase* device_;

};

class GenerateStreamTest: public DeviceTestBase {
protected:


};

TEST_F(GenerateStreamTest, testConstruct) {
    ft::GptInitParameter params;
    auto builder = GenerateStreamBuilder(params);
    auto stream1 = builder.createContextStream({{1, 2, 3, 4, 5}, {}});
    auto stream2 = builder.createDecoderStream({1, 2, 3, 4, 5}, {1, 2, 3});
}

}  // namespace rtp_llm
