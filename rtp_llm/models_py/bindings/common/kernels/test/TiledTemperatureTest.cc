#include <gtest/gtest.h>
#include <torch/torch.h>
#include <ATen/cuda/CUDAContext.h>
#include <limits>
#include "rtp_llm/models_py/bindings/common/kernels/sampling_penalty_kernels.h"
#include "rtp_llm/models_py/bindings/cuda/cuda_host_utils.h"

using namespace rtp_llm;

TEST(TiledTemperatureTest, testTiledTemperatureMatchesNativeBitsAndUnalignedViews) {
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto stream  = at::cuda::getCurrentCUDAStream().stream();
    torch::manual_seed(20261020);
    for (int batch : {1, 3, 4, 17, 32}) {
        for (int offset : {0, 1}) {
            auto storage = torch::empty({batch * 248320 + offset}, options);
            auto actual  = storage.slice(0, offset).reshape({batch, 248320});
            for (float temp : {0.0f, 0.01f, 0.7f, 1.0f, 2.0f, 1.e20f}) {
                actual.normal_();
                actual.slice(1, 0, 1).fill_(std::numeric_limits<float>::denorm_min());
                actual.slice(1, 1, 2).fill_(-std::numeric_limits<float>::denorm_min());
                actual.slice(1, 2, 3).fill_(std::numeric_limits<float>::infinity());
                actual.slice(1, 3, 4).fill_(-std::numeric_limits<float>::infinity());
                actual.slice(1, 4, 5).fill_(std::numeric_limits<float>::quiet_NaN());
                actual.slice(1, 5, 6).fill_(-0.0f);
                auto expected     = actual.clone();
                auto temperatures = torch::full({batch}, temp, options);
                if (batch > 1)
                    temperatures.slice(0, 1, 2).fill_(0.31f);
                invokeBatchApplyTemperaturePenalty<float>(
                    expected.data_ptr<float>(), nullptr, temperatures.data_ptr<float>(), batch, 248320, 248320, stream);
                invokeTiledTemperaturePenalty(
                    actual.data_ptr<float>(), temperatures.data_ptr<float>(), batch, 248320, stream);
                check_cuda_error();
                ASSERT_TRUE(torch::equal(actual.view(torch::kInt32), expected.view(torch::kInt32)));
            }
        }
    }
}
