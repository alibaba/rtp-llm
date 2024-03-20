#include "src/fastertransformer/devices/cuda_impl/tests/CudaTestUtils.h"
#include "src/fastertransformer/devices/cuda_impl/CudaDevice.h"

#include <torch/torch.h>

using namespace std;
using namespace fastertransformer;

class CudaFFnOpTest: public CudaDeviceTestBase {
public:

    double rtol_;
    double atol_;

    void SetUp() override {
        CudaDeviceTestBase::SetUp();
        rtol_ = 1e-03;
        atol_ = 1e-03;
    }
    void TearDown() override {
        CudaDeviceTestBase::TearDown();
    }
};

struct MLPImpl : torch::nn::Module {
    MLPImpl(int hidden_size, int intermediate_size) :
        gate_proj(torch::nn::LinearOptions(hidden_size, intermediate_size)
            .bias(false)),
        up_proj(torch::nn::LinearOptions(hidden_size, intermediate_size)
            .bias(false)),
        down_proj(torch::nn::LinearOptions(intermediate_size, hidden_size)
            .bias(false)) {
        // register_module() is needed if we want to use the parameters() method later on
        register_module("gate_proj", gate_proj);
        register_module("up_proj", up_proj);
        register_module("down_proj", down_proj);
   }

    torch::Tensor forward(torch::Tensor x) {
        return down_proj(torch::silu(gate_proj(x)) * up_proj(x));
    }

    torch::nn::Linear gate_proj, up_proj, down_proj;
};
TORCH_MODULE(MLP);


TEST_F(CudaFFnOpTest, testFFNOp) {
    size_t token_num = 100;
    size_t hidden_size = 1024;
    size_t inter_size = 2048;

    MLP mlp(hidden_size, inter_size);
    mlp.ptr()->to(torch::Device(torch::kCPU));
    auto state_dict = mlp.ptr()->named_parameters();
    torch::NoGradGuard no_grad;

    auto input_host = 0.001 * torch::rand({(int)token_num, (int)hidden_size}, torch::Device(torch::kCPU)).to(torch::kFloat);
    // gate
    auto gate_proj_host = 0.001 * torch::rand({(int)hidden_size, (int)inter_size}, torch::Device(torch::kCPU)).to(torch::kFloat);
    state_dict["gate_proj.weight"].set_data(gate_proj_host.t());
    // up
    auto up_proj_host = 0.001 * torch::rand({(int)hidden_size, (int)inter_size}, torch::Device(torch::kCPU)).to(torch::kFloat);
    state_dict["up_proj.weight"].set_data(up_proj_host.t());
    // down
    auto down_proj_host = 0.001 * torch::rand({(int)inter_size, (int)hidden_size}, torch::Device(torch::kCPU)).to(torch::kFloat);
    state_dict["down_proj.weight"].set_data(down_proj_host.t());

    auto output_host = torch::zeros({(int)token_num, (int)hidden_size}, torch::Device(torch::kCPU)).to(torch::kFloat);

    auto input_device = CreateDeviceBuffer<half>(input_host);
    auto gate_proj_device = CreateDeviceBuffer<half>(gate_proj_host);
    auto up_proj_device = CreateDeviceBuffer<half>(up_proj_host);
    auto down_proj_device = CreateDeviceBuffer<half>(down_proj_host);
    auto output_device = CreateDeviceBuffer<half>(output_host);

    auto input      = CreateTensor(*input_device);
    auto gate_proj  = CreateTensor(*gate_proj_device);
    auto up_proj    = CreateTensor(*up_proj_device);
    auto down_proj  = CreateTensor(*down_proj_device);

    ASSERT_TRUE(torch::allclose(input, input_host, rtol_, atol_));
    ASSERT_TRUE(torch::allclose(gate_proj, gate_proj_host, rtol_, atol_));
    ASSERT_TRUE(torch::allclose(up_proj, up_proj_host, rtol_, atol_));
    ASSERT_TRUE(torch::allclose(down_proj, down_proj_host, rtol_, atol_));

    ActivationType atype = ActivationType::Silu;

    // FfnLayerParams params(*input_device,
    //                       *gate_proj_device,
    //                       *up_proj_device,
    //                       *down_proj_device,
    //                       *output_device,
    //                       atype);
    // device_->ffnLayer(params);

    // auto result     = CreateTensor(*output_device);

    // auto result_host = mlp->forward(input_host);

    // std::cout << result << std::endl;
    // std::cout << result_host << std::endl;
    // ASSERT_TRUE(torch::allclose(result_host, result, rtol_, atol_));

}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
