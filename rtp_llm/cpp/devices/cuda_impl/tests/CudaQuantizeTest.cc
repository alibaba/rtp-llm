#include <gtest/gtest.h>
#include "rtp_llm/cpp/devices/utils/DebugUtils.h"
#include "trt_plugins/mixtureOfExperts/mixtureOfExpertsPlugin.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/devices/testing/TestBase.h"
#include <cuda_fp8.h>

using namespace rtp_llm;

class CudaQuantizeTest: public DeviceTestBase {
public:
    void RunCudaQuantizeFp8() {
        int  m = 15, n = 256;
        auto input  = torch::randn({m, n}).to(torch::kBFloat16).to(torch::kCUDA);
        auto output = device_->quantize(
            {*torchTensor2Buffer(input), DataType::TYPE_FP8_E4M3, 1, QScheme::Qfp8PerTokenBlock, 128});
        printBufferData(*output, "quant_out");

        auto kernel = Buffer2torchTensor(reinterpret_cast<const QBuffer&>(*output).kernel(), false);
        auto scales = Buffer2torchTensor(reinterpret_cast<const QBuffer&>(*output).scales(), false);
        auto m_pad  = (m + 127) / 128 * 128;

        auto mul_res = kernel.to(torch::kFloat32)
                       * scales.transpose(0, 1)
                             .repeat({1, 128})
                             .reshape({m_pad, 128, n / 128})
                             .permute({0, 2, 1})
                             .reshape({m_pad, n});
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
