#include <gtest/gtest.h>
#include "src/fastertransformer/devices/utils/DebugUtils.h"
#include "trt_plugins/mixtureOfExperts/mixtureOfExpertsPlugin.h"
#include "src/fastertransformer/core/torch_utils/BufferTorchUtils.h"
#include "src/fastertransformer/devices/testing/TestBase.h"
#include "maga_transformer/cpp/utils/SignalUtils.h"
#include <nvtx3/nvToolsExt.h>
#include <cuda_fp8.h>

using namespace fastertransformer;
using namespace rtp_llm;

class CudaQuantizeTest : public DeviceTestBase {
public:
    void RunCudaQuantizeFp8() {
        int m = 15, n = 256;
        auto input = torch::randn({m, n}).to(torch::kBFloat16).to(torch::kCUDA);
        auto output = device_->quantize({*torchTensor2Buffer(input),
                                         DataType::TYPE_FP8_E4M3,
                                         1,
                                         QScheme::Qfp8PerTokenBlock});

        auto kernel = Buffer2torchTensor(reinterpret_cast<const QBuffer&>(*output).kernel(), false);
        auto scales = Buffer2torchTensor(reinterpret_cast<const QBuffer&>(*output).scales(), false);
        auto m_pad = (m + 127) / 128 * 128;
        
        auto mul_res = kernel.to(torch::kFloat32) * scales.repeat({1, 128}).reshape({m_pad, 128, n / 128}).permute({0, 2, 1}).reshape({m_pad, n});
        auto out = mul_res.slice(0, 0, m);
        auto sum = torch::sum(out * out + input * input).to(torch::kFloat32);
        EXPECT_NEAR(1, (2 * torch::sum(input * out) / sum).item<double>(), 0.001);
        printf("%lf\n", (2 * torch::sum(input * out) / sum).item<double>());
        printBufferData(*torchTensor2Buffer(input), "input");
        printBufferData(*torchTensor2Buffer(out), "out");
    }
};

TEST_F(CudaQuantizeTest, Test1) {
    RunCudaQuantizeFp8();
}
