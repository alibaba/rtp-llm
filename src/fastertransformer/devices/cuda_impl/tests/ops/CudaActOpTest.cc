#include "src/fastertransformer/devices/cuda_impl/tests/CudaTestUtils.h"
#include "src/fastertransformer/devices/cuda_impl/CudaDevice.h"

using namespace std;
using namespace fastertransformer;

class CudaActOpTest: public CudaDeviceTestBase {
public:
    void BasicActTest(ActivationType atype, size_t m, size_t n);
    void GateActTest(ActivationType atype, size_t m, size_t n);

};

void CudaActOpTest::BasicActTest(ActivationType atype, size_t m, size_t n) {
    auto input_host = torch::rand({(int)m, (int)n}, torch::Device(torch::kCPU)).to(torch::kFloat);
    auto input_device = CreateDeviceBuffer<half>(input_host);

    ActivationParams params {atype, *input_device};
    device_->activation(params);

    torch::Tensor output_host;

    if (atype == ActivationType::Silu) {
        output_host = torch::silu(input_host);
    } else if (atype == ActivationType::Gelu) {
        output_host = torch::gelu(input_host);
    }
    auto output_device = bufferToTensor(*input_device).to(output_host.dtype());

    ASSERT_TRUE(torch::allclose(output_host, output_device, rtol_, atol_));
}


void CudaActOpTest::GateActTest(ActivationType atype, size_t m, size_t n) {
    auto input_host = torch::rand({(int)m, (int)n}, torch::Device(torch::kCPU)).to(torch::kFloat);
    auto gate_host = torch::rand({(int)m, (int)n}, torch::Device(torch::kCPU)).to(torch::kFloat);
    auto gate_bias_host = torch::zeros({(int)m}, torch::Device(torch::kCPU)).to(torch::kFloat);

    auto input_device = CreateDeviceBuffer<half>(input_host);
    auto gate_device = CreateDeviceBuffer<half>(gate_host);
    auto gate_bias_device = CreateDeviceBuffer<half>(gate_bias_host);

    ActivationParams params {atype, *input_device, std::nullopt, *gate_device, * gate_bias_device};

    device_->activation(params);

    torch::Tensor output_host;

    if (atype == ActivationType::Silu) {
        output_host = torch::silu(input_host);
    } else if (atype == ActivationType::Gelu) {
        output_host = torch::gelu(input_host);
    }
    output_host = output_host * gate_host;

    auto output_device = bufferToTensor(*input_device).to(output_host.dtype());

    ASSERT_TRUE(torch::allclose(output_host, output_device, rtol_, atol_));
}


TEST_F(CudaActOpTest, testSiluOp) {
    BasicActTest(ActivationType::Silu, 100, 100);
    BasicActTest(ActivationType::Silu, 1024, 1024);
    BasicActTest(ActivationType::Silu, 1024, 4096);
    GateActTest(ActivationType::Silu, 100, 100);
    GateActTest(ActivationType::Silu, 1024, 1024);
    GateActTest(ActivationType::Silu, 1024, 4096);
}


TEST_F(CudaActOpTest, testGeluOp) {
    BasicActTest(ActivationType::Gelu, 100, 100);
    BasicActTest(ActivationType::Gelu, 1024, 1024);
    GateActTest(ActivationType::Gelu, 100, 100);
    GateActTest(ActivationType::Gelu, 1024, 1024);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
