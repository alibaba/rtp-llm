#include "src/fastertransformer/devices/cuda_impl/tests/CudaTestUtils.h"
#include "src/fastertransformer/devices/cuda_impl/CudaDevice.h"

#include <torch/torch.h>

using namespace std;
using namespace fastertransformer;

class CudaFFnOpTest: public CudaDeviceTestBase {
public:

    void FFNOpTest(size_t token_num,
                   size_t hidden_size,
                   size_t inter_size,
                   ActivationType act_t);
    
    void FFNLoraOpTest(size_t token_num,
                       size_t hidden_size,
                       size_t inter_size,
                       ActivationType act_t);
    
    void MoeOpTest(size_t token_num,
                   size_t hidden_size,
                   size_t inter_size,
                   ActivationType act_t);
    
    void MoeLoraOpTest(size_t token_num,
                       size_t hidden_size,
                       size_t inter_size,
                       ActivationType act_t);
};

torch::Tensor GeGluNoneApproximate(const torch::Tensor& tensor) {
    return torch::gelu(tensor, "tanh");
}

torch::Tensor gelu(const torch::Tensor& tensor) {
    return torch::gelu(tensor);
}


std::unordered_map<ActivationType, 
                   std::function<torch::Tensor(const torch::Tensor&)>> ACT2FUN {
    {ActivationType::Geglu , gelu},
    {ActivationType::GeGluNoneApproximate , GeGluNoneApproximate},
    {ActivationType::Swiglu , torch::silu},
    {ActivationType::Silu, torch::silu}
};

struct MLPImpl : torch::nn::Module {
    MLPImpl(int hidden_size, int intermediate_size, ActivationType act_t) :
        gate_proj(torch::nn::LinearOptions(hidden_size, intermediate_size)
            .bias(false)),
        up_proj(torch::nn::LinearOptions(hidden_size, intermediate_size)
            .bias(false)),
        down_proj(torch::nn::LinearOptions(intermediate_size, hidden_size)
            .bias(false)),
        act_t(act_t) {
        // register_module() is needed if we want to use the parameters() method later on
        register_module("gate_proj", gate_proj);
        register_module("up_proj", up_proj);
        register_module("down_proj", down_proj);
   }

    torch::Tensor forward(torch::Tensor x) {
        if (isGatedActivation(act_t)) {
            return down_proj(ACT2FUN[act_t](gate_proj(x)) * up_proj(x));
        }
        return down_proj(ACT2FUN[act_t](up_proj(x)));
    }

    torch::Tensor forward_test(torch::Tensor x) {
        return ACT2FUN[act_t](gate_proj(x));
    }

    torch::Tensor forward_up(torch::Tensor x) {
        return up_proj(x);
    }

    torch::nn::Linear gate_proj, up_proj, down_proj;
    ActivationType act_t;
};
TORCH_MODULE(MLP);

void CudaFFnOpTest::FFNOpTest(size_t token_num,
                              size_t hidden_size,
                              size_t inter_size,
                              ActivationType act_t) {

    MLP mlp(hidden_size, inter_size, act_t);
    mlp.ptr()->to(torch::Device(torch::kCPU));
    auto state_dict = mlp.ptr()->named_parameters();
    torch::NoGradGuard no_grad;

    auto input_host = 0.01 * torch::rand(
        {(int)token_num, (int)hidden_size}, torch::Device(torch::kCPU)).to(torch::kFloat);
    // gate
    auto gate_proj_host = 0.01 * torch::rand(
        {(int)hidden_size, (int)inter_size}, torch::Device(torch::kCPU)).to(torch::kFloat);
    state_dict["gate_proj.weight"].set_data(gate_proj_host.t());
    // up
    auto up_proj_host = 0.01 * torch::rand(
        {(int)hidden_size, (int)inter_size}, torch::Device(torch::kCPU)).to(torch::kFloat);
    state_dict["up_proj.weight"].set_data(up_proj_host.t());
    // down
    auto down_proj_host = 0.01 * torch::rand(
        {(int)inter_size, (int)hidden_size}, torch::Device(torch::kCPU)).to(torch::kFloat);
    state_dict["down_proj.weight"].set_data(down_proj_host.t());

    auto input_device = CreateDeviceBuffer<half>(input_host);
    auto gate_proj_device = CreateDeviceBuffer<half>(gate_proj_host);
    auto up_proj_device = CreateDeviceBuffer<half>(up_proj_host);
    auto down_proj_device = CreateDeviceBuffer<half>(down_proj_host);

    auto input      = bufferToTensor(*input_device);
    auto gate_proj  = bufferToTensor(*gate_proj_device);
    auto up_proj    = bufferToTensor(*up_proj_device);
    auto down_proj  = bufferToTensor(*down_proj_device);

    assertTensorClose(input, input_host);
    assertTensorClose(gate_proj, gate_proj_host);
    assertTensorClose(up_proj, up_proj_host);
    assertTensorClose(down_proj, down_proj_host);


    FfnLayerWeights weights (
        std::make_unique<const DenseWeights>(
            DenseWeights(up_proj_device)),
        std::make_unique<const DenseWeights>(
            DenseWeights(gate_proj_device)),
        std::make_unique<const DenseWeights>(
            DenseWeights(down_proj_device))
    );

    FfnLayerParams params(*input_device,
                          weights,
                          act_t);
    
    auto output_device  = device_->ffnLayer(params);
    auto result         = bufferToTensor(*(output_device.hidden_states));
    auto result_host    = mlp->forward(input_host).to(result.dtype());;
    
    ASSERT_TRUE(torch::allclose(result_host, result, rtol_, atol_));

}

TEST_F(CudaFFnOpTest, FfnGatedActivationOp) {
    FFNOpTest(4, 2048, 128, ActivationType::Swiglu);
    FFNOpTest(4, 2048, 4096, ActivationType::Swiglu);
    FFNOpTest(128, 2048, 128, ActivationType::Swiglu);
    FFNOpTest(1000, 2048, 128, ActivationType::Swiglu);
    FFNOpTest(1, 2, 4096, ActivationType::Swiglu);
    FFNOpTest(1000, 2048, 128, ActivationType::Swiglu);
}

TEST_F(CudaFFnOpTest, FfnNoGatedActivationOp) {
    FFNOpTest(4, 2048, 128, ActivationType::Silu);
    FFNOpTest(4, 2048, 4096, ActivationType::Silu);
    FFNOpTest(128, 2048, 128, ActivationType::Silu);
    FFNOpTest(1000, 2048, 128, ActivationType::Silu);
    FFNOpTest(1, 2, 4096, ActivationType::Silu);
    FFNOpTest(1000, 2048, 128, ActivationType::Silu);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
