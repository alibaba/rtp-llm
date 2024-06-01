
#include "gtest/gtest.h"

#define private public
#include "maga_transformer/cpp/dataclass/GenerateStream.h"
#include "src/fastertransformer/devices/testing/TestBase.h"

#include <chrono>
#include <memory>
#include <thread>

using namespace std;

namespace rtp_llm {

class GenerateStreamTest: public DeviceTestBase {
protected:

};

TEST_F(GenerateStreamTest, testConstruct) {
    std::shared_ptr<GenerateInput>  generate_input(new GenerateInput());
    std::shared_ptr<GenerateConfig> generate_config(new GenerateConfig());
    auto                            vec   = vector<int>{1, 2, 3, 4, 5, 6};
    std::vector<size_t>             shape = {6};
    generate_input->input_ids = std::make_unique<ft::Buffer>(ft::MEMORY_CPU, ft::TYPE_INT32, shape, (void*)(vec.data()));
    generate_input->generate_config = generate_config;
    ResourceContext resource_context;
    ft::GptInitParameter params;
    params.max_seq_len_ = 2048;

    GenerateStream stream(generate_input, params, resource_context, nullptr);
}

}  // namespace rtp_llm
