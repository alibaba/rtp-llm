#include "rtp_llm/cpp/core/ExecOps.h"
#include "rtp_llm/cpp/cuda/ops/tests/CudaTestUtils.h"
#include "rtp_llm/cpp/utils/DebugUtils.h"
#include "rtp_llm/cpp/kernels/banRepeatNgram.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>  // For half
#include <cuda_bf16.h>  // For __nv_bfloat16

using namespace std;
using namespace rtp_llm;

class CudaMaskLogitsOpTest: public DeviceTestBase {
public:
protected:
};

#ifndef CUDART_INF_FP16
#define CUDART_INF_FP16 __ushort_as_half((unsigned short)0x7C00U)
#endif

#ifndef CUDART_INF_BF16
#define CUDART_INF_BF16 __ushort_as_bfloat16((unsigned short)0x7F80U)
#endif

TEST_F(CudaMaskLogitsOpTest, testMaskLogits) {
    {

        size_t batch_size = 4;
        auto   logits     = torch::tensor(
                          {
                              0.f,    0.f,    0.f,      0.1f, 0.2f, 0.3f,   0.f, 0.f,     0.f,   0.01f,
                              0.987f, 0.887f, 0.99999f, 0.1f, 0.2f, 0.3f,   0.f, 0.f,     0.99f, 0.989f,
                              0.221f, 0.f,    0.f,      0.1f, 0.2f, 0.321f, 0.f, 0.4432f, 0.44f, 0.01f,
                              0.221f, 0.f,    0.f,      0.1f, 0.2f, 0.321f, 0.f, 0.4432f, 0.44f, 0.01f,
                          },
                          torch::kFloat32)
                          .reshape({(int64_t)batch_size, 10})
                          .to(torch::kCUDA);

        std::vector<uint8_t> mask_data = {
            0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        };
        auto vocab_mask =
            torch::from_blob(mask_data.data(), {(int64_t)batch_size, 10}, torch::kUInt8).clone().to(torch::kCUDA);

        printBufferData(logits, "MYDEBUG_BEFORE_MASK_LOGITS");
        runtimeMaskLogits(logits, vocab_mask);
        printBufferData(logits, "MYDEBUG_AFTER_MASK_LOGITS");

        auto logits_cpu = logits.cpu().contiguous();
        auto logits_ptr = logits_cpu.data_ptr<float>();

        std::vector<float> expect_vec = {
            0.f,       -INFINITY, 0.f,      -INFINITY, 0.2f,      0.3f,   0.f, 0.f,     0.f,   0.01f,
            -INFINITY, 0.887f,    0.99999f, 0.1f,      0.2f,      0.3f,   0.f, 0.f,     0.99f, 0.989f,
            0.221f,    0.f,       0.f,      0.1f,      0.2f,      0.321f, 0.f, 0.4432f, 0.44f, -INFINITY,
            0.221f,    0.f,       0.f,      0.1f,      -INFINITY, 0.321f, 0.f, 0.4432f, 0.44f, 0.01f,
        };

        ASSERT_EQ((size_t)logits_cpu.numel(), expect_vec.size()) << "Vectors x and y are of unequal length";
        for (size_t i = 0; i < expect_vec.size(); i++) {
            ASSERT_TRUE(logits_ptr[i] == expect_vec[i]);
        }
    }
    {
        size_t batch_size = 2;
        auto logits = torch::tensor({0.f, 0.f, 0.f, 0.1f, 0.2f, 0.987f, 0.887f, 0.99999f, 0.1f, 0.2f}, torch::kFloat32)
                          .reshape({(int64_t)batch_size, 5})
                          .to(torch::kHalf)
                          .to(torch::kCUDA);

        std::vector<uint8_t> mask_data2 = {0, 1, 0, 1, 0, 1, 0, 0, 0, 0};
        auto                 vocab_mask =
            torch::from_blob(mask_data2.data(), {(int64_t)batch_size, 5}, torch::kUInt8).clone().to(torch::kCUDA);

        printBufferData(logits, "MYDEBUG_BEFORE_MASK_LOGITS");
        runtimeMaskLogits(logits, vocab_mask);
        printBufferData(logits, "MYDEBUG_AFTER_MASK_LOGITS");

        // Compare in float space
        auto logits_cpu = logits.cpu().to(torch::kFloat32).contiguous();
        auto logits_ptr = logits_cpu.data_ptr<float>();

        // Expected: masked positions become -inf, others unchanged (in fp16 precision)
        // Position [0][1], [0][3], [1][0] should be -inf
        ASSERT_TRUE(std::isinf(logits_ptr[1]) && logits_ptr[1] < 0);  // [0][1]
        ASSERT_TRUE(std::isinf(logits_ptr[3]) && logits_ptr[3] < 0);  // [0][3]
        ASSERT_TRUE(std::isinf(logits_ptr[5]) && logits_ptr[5] < 0);  // [1][0]
        // Non-masked positions should be finite
        ASSERT_FALSE(std::isinf(logits_ptr[0]));  // [0][0]
        ASSERT_FALSE(std::isinf(logits_ptr[2]));  // [0][2]
        ASSERT_FALSE(std::isinf(logits_ptr[4]));  // [0][4]
        ASSERT_FALSE(std::isinf(logits_ptr[6]));  // [1][1]
    }
    {
        size_t batch_size = 2;
        auto logits = torch::tensor({0.f, 0.f, 0.f, 0.1f, 0.2f, 0.987f, 0.887f, 0.99999f, 0.1f, 0.2f}, torch::kFloat32)
                          .reshape({(int64_t)batch_size, 5})
                          .to(torch::kBFloat16)
                          .to(torch::kCUDA);

        std::vector<uint8_t> mask_data3 = {0, 1, 0, 1, 0, 1, 0, 0, 0, 0};
        auto                 vocab_mask =
            torch::from_blob(mask_data3.data(), {(int64_t)batch_size, 5}, torch::kUInt8).clone().to(torch::kCUDA);

        printBufferData(logits, "MYDEBUG_BEFORE_MASK_LOGITS");
        runtimeMaskLogits(logits, vocab_mask);
        printBufferData(logits, "MYDEBUG_AFTER_MASK_LOGITS");

        // Compare in float space
        auto logits_cpu = logits.cpu().to(torch::kFloat32).contiguous();
        auto logits_ptr = logits_cpu.data_ptr<float>();

        // Expected: masked positions become -inf, others unchanged
        ASSERT_TRUE(std::isinf(logits_ptr[1]) && logits_ptr[1] < 0);  // [0][1]
        ASSERT_TRUE(std::isinf(logits_ptr[3]) && logits_ptr[3] < 0);  // [0][3]
        ASSERT_TRUE(std::isinf(logits_ptr[5]) && logits_ptr[5] < 0);  // [1][0]
        ASSERT_FALSE(std::isinf(logits_ptr[0]));                      // [0][0]
        ASSERT_FALSE(std::isinf(logits_ptr[2]));                      // [0][2]
        ASSERT_FALSE(std::isinf(logits_ptr[4]));                      // [0][4]
        ASSERT_FALSE(std::isinf(logits_ptr[6]));                      // [1][1]
    }
}
