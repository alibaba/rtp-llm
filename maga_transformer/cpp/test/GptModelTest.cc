#include "src/fastertransformer/devices/testing/TestBase.h"
#include "src/fastertransformer/devices/cuda_impl/CudaDevice.h"
#include "maga_transformer/cpp/models/GptModel.h"
#include "maga_transformer/cpp/test/ModelTestUtil.h"

using namespace std;
using namespace rtp_llm;

class GptModelTest: public DeviceTestBase<DeviceType::Cuda> {
public:
    void SetUp() override {
        DeviceTestBase<DeviceType::Cuda>::SetUp();
    }
    void TearDown() override {
        DeviceTestBase<DeviceType::Cuda>::TearDown();
    }
};

TEST_F(GptModelTest, testSimple) {
    const auto path = test_data_path_ + "../../test/model_test/fake_test/testdata/qwen_0.5b";
    auto weights = loadWeightsFromDir(path);
    assert(weights->lm_head->kernel);
    assert(weights->embedding);
    assert(weights->layers.size() == 24);

    GptModelDescription description;
    GptModel model({device_, *weights, description});

    const auto combo_tokens = createBuffer<int32_t>({3}, {13048, 11, 220});
    const auto input_lenghts = createBuffer<int32_t>({1}, {3});
    const auto sequence_lenghts = createBuffer<int32_t>({}, {});

    // TODO: fill these blokcs when BlockManager is done.
    const auto kv_cache_blocks = createBuffer<int64_t>({1, 1}, {0});

    GptModelInputs inputs = {
        *combo_tokens, *input_lenghts, *sequence_lenghts,
        nullopt, nullopt, *kv_cache_blocks
    };

    try {
        auto outputs = model.forward(inputs);
    } catch (const OpException& e) {
        cout << e.what() << endl;
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
