#include <gtest/gtest.h>
#include "src/fastertransformer/devices/utils/DebugUtils.h"
#include "trt_plugins/mixtureOfExperts/mixtureOfExpertsPlugin.h"
#include "src/fastertransformer/core/torch_utils/BufferTorchUtils.h"
#include "src/fastertransformer/devices/testing/TestBase.h"
#include "maga_transformer/cpp/utils/SignalUtils.h"
#include <nvtx3/nvToolsExt.h>
#include <cuda_fp8.h>
#include "src/fastertransformer/deep_gemm/include/fp8_gemm.cuh"

using namespace fastertransformer;
using namespace rtp_llm;
using namespace deep_gemm;

class DeepGemmTest : public DeviceTestBase {
public:
    void RunDeepGeemTest() {
        using gemm_runner = deep_gemm::Gemm<128, 256, 64, 32, 128, 1, 8, 1, deep_gemm::GemmType::Normal>;
        auto input =  torch::randint(-100, 100, {(int)16, (int)256}, torch::device(torch::kCUDA)).to(torch::kFloat8_e4m3fn);
        auto input_scale = torch::rand({16,2}, torch::device(torch::kCUDA)).to(torch::kFloat32);
        auto weight = torch::randn({(int)256, (int)128}, torch::device(torch::kCUDA)).to(torch::kFloat8_e4m3fn);
        auto weight_scale = torch::randn({(int)(256 / 128), (int)(128 / 128)}, torch::device(torch::kCUDA)).to(torch::kFloat32);
        auto input_scale_t = input_scale.repeat({1, 128}).reshape({1, 16, 128, (int)(256 / 128)}).permute({1, 0, 3, 2}).reshape({16, 256});
        auto weight_scale_t = weight_scale.repeat({128, 128}).reshape({(int)128, (int)(256 / 128), 128, (int)(128 / 128)}).permute({1, 0, 3, 2}).reshape({(int)(256), (int)(128)});
        auto ref_output = torch::matmul(input.to(torch::kFloat32) * input_scale_t, weight.to(torch::kFloat32) * weight_scale_t).to(torch::kBFloat16);

        printBufferData(*fastertransformer::torchTensor2Buffer(ref_output), "ref", device_, true);

        auto input_scale_col_major = torch::transpose(torch::empty({2,16}).to(torch::kFloat32).to(torch::kCUDA), 0, 1);
        input_scale_col_major.index_put_({torch::indexing::Slice(), torch::indexing::Slice()}, input_scale);

        auto output = device_->allocateBuffer({DataType::TYPE_BF16, {16, 128}, AllocationType::DEVICE});
        auto tma_a_desc = gemm_runner::make_2d_tma_a_desc<__nv_fp8_e4m3>((__nv_fp8_e4m3*)input.data_ptr(), (uint32_t)16);
        auto tma_b_desc = gemm_runner::make_2d_tma_b_desc<__nv_fp8_e4m3>((__nv_fp8_e4m3*)weight.transpose(0, 1).contiguous().data_ptr());
        auto tma_scales_a_desc = gemm_runner::make_2d_tma_scales_a_desc<float>(input_scale_col_major.data<float>(), (uint32_t)16);
        auto tma_d_desc = gemm_runner::make_2d_tma_d_desc<__nv_bfloat16>(output->data<__nv_bfloat16>(), (uint32_t)16);

        gemm_runner::run(output->data<__nv_bfloat16>(), (float*)weight_scale.transpose(0, 1).contiguous().data_ptr(), nullptr, (uint32_t)16, 
                          tma_a_desc, tma_b_desc, tma_scales_a_desc, tma_d_desc, 0, 72, (uint32_t)104800);

        auto gemm_output = torch::from_blob(output->data(), {(int64_t)16, (int64_t)128}, torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA));
        printBufferData(*output, "output", device_, true);
        auto res = torch::all(torch::isclose(ref_output, gemm_output)).to(torch::kCPU);        
        printf("%d\n", *res.data_ptr<bool>());
    }
};

TEST_F(DeepGemmTest, Test1) {
    RunDeepGeemTest();
}
