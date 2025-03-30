#include <gtest/gtest.h>
#include "src/fastertransformer/devices/utils/DebugUtils.h"
#include "trt_plugins/mixtureOfExperts/mixtureOfExpertsPlugin.h"
#include "src/fastertransformer/core/torch_utils/BufferTorchUtils.h"
#include "src/fastertransformer/devices/testing/TestBase.h"
#include "maga_transformer/cpp/utils/SignalUtils.h"
#include <nvtx3/nvToolsExt.h>
#include <cuda_fp8.h>
#include "src/fastertransformer/deep_gemm/DeepGemmPlugin.h"

using namespace fastertransformer;
using namespace rtp_llm;

class CudaMlaBmmTest : public DeviceTestBase {
public:
    torch::Tensor QInputBatchMatmulWrapper(Buffer& q, Buffer& w, int nope_dim, int t) {
        // from [batch_size, head_num, nope_head_dim + rope_head_dim] to [batch_size, head_num, nope_head_dim] with stride
        auto q_nope_t = Buffer2torchTensorWithStride(
            q,
            {(int64_t)q.shape()[0], (int64_t)q.shape()[1], (int64_t)nope_dim});

        if (t == 1){
            size_t m = q.shape()[0];
            size_t m_pad = (m + 127) / 128 * 128;
            auto bmm_indices_host = device_->allocateBuffer({DataType::TYPE_INT32, {m_pad * q.shape()[1], 1}, AllocationType::HOST});
            int* index = bmm_indices_host->data<int>();
            for (int i = 0;i < m_pad * q.shape()[1]; ++i) {
                index[i] = i / q.shape()[1];
            }
            auto bmm_indices = device_->clone({*bmm_indices_host, AllocationType::DEVICE});

            auto q_nope_t_pad = torch::zeros({(int64_t)m_pad, (int64_t)q.shape()[1], (int64_t)nope_dim}, torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA));
            q_nope_t_pad.index_put_({torch::indexing::Slice(0, m)}, q_nope_t);
            q_nope_t_pad = q_nope_t_pad.transpose(0, 1).reshape({(int64_t)(q.shape()[1] * m_pad), (int64_t)nope_dim}).contiguous();
            auto q_nope_t_pad_quant = device_->quantize({*torchTensor2Buffer(q_nope_t_pad),
                                                        DataType::TYPE_FP8_E4M3,
                                                        1,
                                                        QScheme::Qfp8PerTokenBlock});        

            auto output_tensor = torch::zeros({(int64_t)q.shape()[1] * (int64_t)m_pad, (int64_t)w.shape()[1]}, torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA));
            // num_groups = head_num
            DeepGemmPlugin::groupedGemmFp8Contiguous(*q_nope_t_pad_quant, w, *torchTensor2Buffer(output_tensor), *bmm_indices, 0);

            return output_tensor.reshape({(int64_t)q.shape()[1], (int64_t)m_pad, (int64_t)w.shape()[1]}).transpose(0, 1).index({torch::indexing::Slice(0, m)}).contiguous();
        } else {
            // shape: [head_num, nope_head_dim, kv_lora_rank]
            auto w_kc_t = Buffer2torchTensor(w, false);
            auto q_nope_out = torch::bmm(q_nope_t.transpose(0, 1), w_kc_t.transpose(1, 2));
            return q_nope_out.transpose(0, 1).contiguous();
        }
    }

    void RunQInputBatchMatmulWrapper() {
        int m = 16, head_num = 128, nope_dim = 128, rope_dim = 64, kv_lora_rank = 512;
        auto input = torch::randn({m, head_num, nope_dim + rope_dim}, torch::device(torch::kCUDA)).to(torch::kBFloat16);
        auto weight_scale = torch::randn({head_num, kv_lora_rank / 128, nope_dim / 128}, torch::device(torch::kCUDA)).to(torch::kFloat32);
        auto weight_kernel = torch::randn({head_num, kv_lora_rank, nope_dim}, torch::device(torch::kCUDA)).to(torch::kFloat8_e4m3fn);
        auto weight_scale_t = weight_scale.repeat({1, 128, 128}).reshape({head_num, 128, kv_lora_rank / 128, 128, nope_dim / 128}).permute({0, 2, 1, 4, 3}).reshape({head_num, kv_lora_rank, nope_dim});
        auto weight_unquant = (weight_scale_t * weight_kernel.to(torch::kFloat32)).to(torch::kBFloat16);

        auto ref_output = torch::bmm(input.transpose(0, 1).index({torch::indexing::Slice(), torch::indexing::Slice(),torch::indexing::Slice(0, nope_dim)}),
                                     weight_unquant.transpose(1,2)).transpose(0,1).contiguous(); 
        printBufferData(*torchTensor2Buffer(ref_output), "ref");
        auto weight_quant = QBuffer(torchTensor2Buffer(weight_kernel), torchTensor2Buffer(weight_scale), BufferPtr(new Buffer(torchTensor2Buffer(weight_kernel)->where(), DataType::TYPE_INVALID, {0}, nullptr)));
        auto gemm_output = QInputBatchMatmulWrapper(*torchTensor2Buffer(input), weight_quant, nope_dim, 1);
        printBufferData(*torchTensor2Buffer(gemm_output), "out");
        auto sum = torch::sum(ref_output * ref_output + (gemm_output * gemm_output)).to(torch::kFloat32);
        EXPECT_NEAR(1, (2 * torch::sum(ref_output * gemm_output) / sum).item<double>(), 0.001);
        std::cout << (2 * torch::sum(ref_output * gemm_output) / sum).item<double>() << std::endl;
    }
    // void RunDeepGeemPluginGroupedContiguousTest() {
    //     int m = 128, n = 32, k = 7168, num_groups = 4;

    //     auto input =  torch::ones({(int)num_groups, (int)m, (int)k}, torch::device(torch::kCUDA)).to(torch::kFloat8_e4m3fn);
    //     auto input_scale = torch::randn({num_groups, m, int(k / 128)}, torch::device(torch::kCUDA)).to(torch::kFloat32);
    //     auto weight = torch::randn({num_groups, n, k}, torch::device(torch::kCUDA)).to(torch::kFloat8_e4m3fn);
    //     auto weight_scale = torch::randn({num_groups, int(n / 128), int(k / 128)}, torch::device(torch::kCUDA)).to(torch::kFloat32);
    //     auto input_scale_t = input_scale.repeat({1, 1, 128}).reshape({num_groups, m, 128, (int)(k / 128)}).permute({0, 1, 3, 2}).reshape({num_groups, m, k});
    //     auto weight_scale_t = weight_scale.repeat({1, 128, 128}).reshape({num_groups, (int)128, (int)(n / 128), 128, (int)(k / 128)}).permute({0, 2, 1, 4, 3}).reshape({(int)num_groups, (int)(n), (int)(k)});
    //     auto weight_scale_tt = (weight.to(torch::kFloat32) * weight_scale_t).index({torch::indexing::Slice(0,1), torch::indexing::Slice(), torch::indexing::Slice()}).transpose(1,2).reshape({k, n});
    //     for (auto& x: weight_scale_tt.sizes()) std::cout << x << " ";
    //     std::cout << std::endl;
    //     auto ref_output = torch::matmul(input.to(torch::kFloat32) * input_scale_t, weight_scale_tt).reshape({num_groups * m, n});
    //     // auto ref_output = torch::einsum("gmk,gnk->gmn", {input.to(torch::kFloat32) * input_scale_t, (weight.to(torch::kFloat32) * weight_scale_t)}).reshape({num_groups * m, n});
    //     printBufferData(*fastertransformer::torchTensor2Buffer(ref_output), "ref", device_, true);

    //     BufferPtr input_k, input_s;
    //     input_k = torchTensor2Buffer(input.reshape({num_groups * m, k}));
    //     input_s = torchTensor2Buffer(input_scale.reshape({num_groups * m, k / 128}));
    //     BufferPtr lhs = std::make_shared<QBuffer>(std::move(input_k), std::move(input_s), std::move(BufferPtr(new Buffer(input_k->where(), DataType::TYPE_INVALID, {0}, nullptr))));

    //     BufferPtr weight_k, weight_s;
    //     weight_k = torchTensor2Buffer(weight);
    //     weight_s = torchTensor2Buffer(weight_scale);

    //     //auto m_indices = torch::arange(0, num_groups, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    //     //m_indices = m_indices.unsqueeze(-1).expand({num_groups, m}).contiguous().reshape({num_groups * m});
    //     auto m_indices = torch::zeros({num_groups * m}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));

    //     BufferPtr rhs = std::make_shared<QBuffer>(std::move(weight_k), std::move(weight_s), std::move(BufferPtr(new Buffer(weight_k->where(), DataType::TYPE_INVALID, {0}, nullptr))));
    //     BufferPtr output = device_->allocateBuffer({DataType::TYPE_BF16, {(unsigned long)m * num_groups, (unsigned long)n}, AllocationType::DEVICE});

    //     DeepGemmPlugin::groupedGemmFp8Contiguous(*lhs, *rhs, *output, *(torchTensor2Buffer(m_indices)), 0);
    //     auto gemm_output = torch::from_blob(output->data(), {(int64_t)m * num_groups, (int64_t)n}, torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA));
    //     auto sum = torch::sum(ref_output * ref_output + (gemm_output * gemm_output)).to(torch::kFloat32);
    //     printBufferData(*output, "output", device_, true);
    //     std::cout << 2 * torch::sum(ref_output * gemm_output) / sum << std::endl;
    //     EXPECT_NEAR(1, (2 * torch::sum(ref_output * gemm_output) / sum).item<double>(), 0.001);
    // }
};

TEST_F(CudaMlaBmmTest, Test1) {
    RunQInputBatchMatmulWrapper();
}
