#include <gtest/gtest.h>
#include "rtp_llm/cpp/devices/utils/DebugUtils.h"
#include "trt_plugins/mixtureOfExperts/mixtureOfExpertsPlugin.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/devices/testing/TestBase.h"

using namespace std;

using namespace rtp_llm;
using namespace tensorrt_llm::kernels;
using namespace tensorrt_llm::plugins;

class MoeTest: public DeviceTestBase {
public:
    void RunMoeTest(uint64_t    token_num,
                    uint64_t    hidden_dim,
                    int         moe_inter_size,
                    int         num_expert,
                    uint64_t    ep_size,
                    uint64_t    top_k,
                    uint64_t    bench_round = 0,
                    const char* range_msg   = nullptr) {
        std::string range_msg_str;
        if (range_msg == nullptr) {
            range_msg_str = "moe-bs" + std::to_string(token_num) + "-dim" + std::to_string(hidden_dim) + "-indim"
                            + std::to_string(moe_inter_size) + "-exp_num" + std::to_string(num_expert) + "-ep"
                            + std::to_string(ep_size) + "-top" + std::to_string(top_k);
            range_msg = range_msg_str.c_str();
        }

        std::cout << range_msg_str << std::endl;

        MixtureOfExpertsPlugin moe_plugin;

        moe_plugin.init(num_expert,
                        top_k,
                        /*normalize_expert_scale=*/0,
                        hidden_dim,
                        moe_inter_size,
                        rtp_llm::ActivationType::Swiglu,
                        nvinfer1::DataType::kHALF,
                        nvinfer1::DataType::kHALF,
                        /*has_zeros=*/false,
                        /*group_size=*/0,
                        tensorrt_llm::kernels::MOEExpertScaleNormalizationMode::RENORMALIZE,
                        /*ep_size=*/ep_size,
                        /*ep_rank=*/0);

        const auto ws_size    = moe_plugin.getWorkspaceSize(token_num);
        const auto worksapce  = device_->allocateBuffer({DataType::TYPE_BYTES, {ws_size}});
        const auto fc2_result = device_->allocateBuffer({DataType::TYPE_FP16, {token_num, top_k, hidden_dim}});
        const auto expert_scales =
            device_->allocateBuffer({DataType::TYPE_FP32, {rtp_llm::pad_to_multiple_of_16(token_num * top_k)}});

        const auto expanded_source_row_to_dest =
            device_->allocateBuffer({DataType::TYPE_INT32, {rtp_llm::pad_to_multiple_of_16(token_num * top_k)}});
        const auto expert_for_source_row =
            device_->allocateBuffer({DataType::TYPE_INT32, {rtp_llm::pad_to_multiple_of_16(token_num * top_k)}});
        auto output = device_->allocateBuffer({DataType::TYPE_FP16, {token_num, hidden_dim}});

        auto input = torch::randn({(int)token_num, (int)hidden_dim}, torch::device(torch::kCUDA)).to(torch::kFloat16);
        auto gate  = torch::randn({(int)token_num, (int)num_expert}, torch::device(torch::kCUDA)).to(torch::kFloat32);
        auto fc1_weight = torch::randn({(int)(num_expert / ep_size), (int)2 * moe_inter_size, (int)hidden_dim},
                                       torch::Device(torch::kCUDA))
                              .to(torch::kFloat16);
        auto fc2_weight = torch::randn({(int)(num_expert / ep_size), (int)hidden_dim, (int)moe_inter_size},
                                       torch::Device(torch::kCUDA))
                              .to(torch::kFloat16);

        const uint64_t round = bench_round + 1;
        for (int i = 0; i < round; ++i) {
            if (i != 0)
                nvtxRangePushA(range_msg);

            moe_plugin.enqueue(input.data_ptr(),
                               (float*)(gate.data_ptr()),
                               (float*)(gate.data_ptr()),
                               fc1_weight.data_ptr(),
                               nullptr,
                               nullptr,
                               nullptr,
                               fc2_weight.data_ptr(),
                               nullptr,
                               nullptr,
                               nullptr,
                               token_num,
                               worksapce->data<char>(),
                               // output
                               output->data(),
                               fc2_result->data(),
                               nullptr,  // finished
                               expert_scales->data(),
                               expanded_source_row_to_dest->data<int>(),
                               expert_for_source_row->data<int>(),
                               0);

            if (i == 0) {
                printBufferData(*rtp_llm::torchTensor2Buffer(input), "input");
                printBufferData(*rtp_llm::torchTensor2Buffer(gate), "gate");
                printBufferData(*expert_for_source_row, "expert_for_source_row");
                printBufferData(*output, "output");
            }

            if (i != 0)
                nvtxRangePop();
        }
    }
};

TEST_F(MoeTest, Test1) {
    installSighandler();

    // for (auto ep: {1, 32, 64}) {
    //     for (auto bs: {1, 32, 256, 512, 1024, 2048, 4096}) {
    //         RunMoeTest(bs, 7168, 2048, 256, ep, 8, 1000);
    //     }
    // }

    for (auto bs : {1, 32, 64, 128, 256}) {
        RunMoeTest(bs, 7168, 2048, 8, 1, 1, 1000);
    }
}
