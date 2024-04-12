#include "src/fastertransformer/devices/cuda_impl/tests/CudaTestUtils.h"
#include "src/fastertransformer/devices/cuda_impl/CudaDevice.h"

#include <torch/torch.h>

using namespace std;
using namespace fastertransformer;

class EmbeddingLookupTest: public DeviceTestBase {
};

TEST_F(EmbeddingLookupTest, testEmbeddingLookup) {
    const auto vocab_size = 102400;
    const auto hidden_size = 1024;
    const auto seq_len = 4;

    auto ids_vec = vector<int32_t>{100, 20000, 2010, 1024};
    auto ids = createBuffer<int32_t>({seq_len}, ids_vec);
    auto table_tensor = torch::rand(
        {vocab_size, hidden_size}, torch::Device(torch::kCPU)
    ).to(torch::kHalf);
    auto table = createDeviceBuffer<half>(table_tensor);
    auto output = device_->embeddingLookup({*ids, *table});
    auto output_tensor = bufferToTensor(*output);
    std::cout << "output_tensor: " << output_tensor << std::endl;

    auto ids_tensor = bufferToTensor(*ids);
    std::cout << "ids: " << ids_tensor << std::endl;
    auto expected_values = table_tensor.index_select(0, ids_tensor);
    std::cout << "expected: " << expected_values << output_tensor << std::endl;

    ASSERT_TRUE(torch::allclose(expected_values, output_tensor, 1e-03, 1e-03));

    sync_check_cuda_error();
}
