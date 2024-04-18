#include "c10/util/intrusive_ptr.h"
#include "src/fastertransformer/core/Types.h"
#include "torch/all.h"

#include "maga_transformer/cpp/engines/NormalEngine.h"
#include "maga_transformer/cpp/schedulers/FIFOScheduler.h"
#include "maga_transformer/cpp/engines/test/MockEngine.h"
#include "src/fastertransformer/core/Buffer.h"
#include "src/fastertransformer/devices/testing/TestBase.h"
#include "src/fastertransformer/models/W.h"
#include "gmock/gmock-actions.h"
#include "gmock/gmock-function-mocker.h"
#include "gtest/gtest.h"
#include <memory>

using namespace std;
namespace W  = ft::W;
namespace ft = fastertransformer;
namespace rtp_llm {

class NormalEngineTest: public DeviceTestBase {
public:

};

TEST_F(NormalEngineTest, testSimple) {
    auto engine = createMockEngine(device_);

    std::shared_ptr<GenerateInput> query   = make_shared<GenerateInput>();
    query->input_ids                       = createBuffer<int32_t>({7}, {1, 2, 3, 4, 5, 6, 7}, AllocationType::HOST);
    query->generate_config                 = make_shared<GenerateConfig>();
    query->generate_config->max_new_tokens = 3;
    shared_ptr<GenerateStream> stream      = make_shared<GenerateStream>(query);

    ASSERT_TRUE(engine->enqueue(stream).ok());
    auto output1 = stream->nextOutput();
    ASSERT_TRUE(output1.ok());
    ASSERT_EQ(output1.value().aux_info.output_len, 1);

    auto output2 = stream->nextOutput();
    ASSERT_TRUE(output2.ok());
    ASSERT_EQ(output2.value().aux_info.output_len, 2);

    auto output3 = stream->nextOutput();
    ASSERT_TRUE(output3.ok());
    ASSERT_EQ(output3.value().aux_info.output_len, 3);

    ASSERT_TRUE(stream->finished());
    auto output4 = stream->nextOutput();
    ASSERT_TRUE(!output4.ok());
}

}  // namespace rtp_llm
