#include "rtp_llm/cpp/devices/testing/TestBase.h"
#include "rtp_llm/cpp/devices/cpu_impl/CpuDevice.h"
#include <torch/torch.h>

using namespace std;
using namespace rtp_llm;

class CpuEmbeddingTest: public DeviceTestBase {
public:
    void SetUp() override {
        DeviceTestBase::SetUp();
        rtol_ = 1e-2;
        atol_ = 1e-2;
    }

protected:
    void testembeddinglookup(DataType data_type) {
        const auto torch_dtype = dataTypeToTorchType(data_type);
        const auto vocab_size  = 102400;
        const auto hidden_size = 1024;

        auto ids_tensor = torch::tensor({100, 20000, 2010, 1024}, torch::kInt32);
        auto ids        = tensorToBuffer(ids_tensor);

        auto table_tensor    = torch::rand({vocab_size, hidden_size}).to(torch_dtype);
        auto table           = tensorToBuffer(table_tensor);
        auto output          = device_->embeddingLookup({*ids, *table});
        auto expected_values = table_tensor.index_select(0, ids_tensor);

        assertTensorClose(expected_values, bufferToTensor(*(output)));
    }
};

TEST_F(CpuEmbeddingTest, testEmbeddingLookup) {
    testembeddinglookup(DataType::TYPE_BF16);
    testembeddinglookup(DataType::TYPE_FP16);
}
