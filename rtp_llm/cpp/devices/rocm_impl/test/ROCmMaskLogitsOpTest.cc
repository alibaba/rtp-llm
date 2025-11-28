#include "rtp_llm/cpp/devices/rocm_impl/RocmTestUtils.h"
#include "rtp_llm/cpp/devices/rocm_impl/ROCmDevice.h"
#include "rtp_llm/cpp/devices/utils/DebugUtils.h"
#include "rtp_llm/cpp/kernels/banRepeatNgram.h"
#include "rtp_llm/cpp/core/BufferHelper.h"

using namespace std;
using namespace rtp_llm;

class ROCmMaskLogitsOpTest: public DeviceTestBase {
public:
protected:
};

#ifndef CUDART_INF_FP16
#define CUDART_INF_FP16 __ushort_as_half((unsigned short)0x7C00U)
#endif

#ifndef CUDART_INF_BF16
#define CUDART_INF_BF16 __ushort_as_bfloat16((unsigned short)0x7F80U)
#endif

TEST_F(ROCmMaskLogitsOpTest, testMaskLogits) {
    {
        device_ = new ROCmDevice(DeviceInitParams());
        device_->init();

        size_t    batch_size = 4;
        BufferPtr logits     = createBuffer<float>({batch_size, 10},
                                                   {
                                                   0,     0,     0,       0.1, 0.2, 0.3,   0, 0,      0,    0.01,
                                                   0.987, 0.887, 0.99999, 0.1, 0.2, 0.3,   0, 0,      0.99, 0.989,
                                                   0.221, 0,     0,       0.1, 0.2, 0.321, 0, 0.4432, 0.44, 0.01,
                                                   0.221, 0,     0,       0.1, 0.2, 0.321, 0, 0.4432, 0.44, 0.01,
                                               },
                                               rtp_llm::AllocationType::DEVICE);
        BufferPtr vocab_mask = createBuffer<uint8_t>({batch_size, 10},
                                                     {
                                                         0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                         0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                                                     },
                                                     rtp_llm::AllocationType::DEVICE);

        printBufferData(*logits, "MYDEBUG_BEFORE_MASK_LOGITS");
        device_->maskLogits(*logits, *vocab_mask);
        printBufferData(*logits, "MYDEBUG_AFTER_MASK_LOGITS");

        auto logits_hosts = getBufferValues<float>(*logits);

        std::vector<float> expect_vec = {
            0,         -INFINITY, 0,       -INFINITY, 0.2,       0.3,   0, 0,      0,    0.01,
            -INFINITY, 0.887,     0.99999, 0.1,       0.2,       0.3,   0, 0,      0.99, 0.989,
            0.221,     0,         0,       0.1,       0.2,       0.321, 0, 0.4432, 0.44, -INFINITY,
            0.221,     0,         0,       0.1,       -INFINITY, 0.321, 0, 0.4432, 0.44, 0.01,
        };

        ASSERT_EQ(logits_hosts.size(), expect_vec.size()) << "Vectors x and y are of unequal length";
        for (size_t i = 0; i < expect_vec.size(); i++) {
            ASSERT_TRUE(logits_hosts[i] == expect_vec[i]);
        }
    }
    {
        device_ = new ROCmDevice(DeviceInitParams());
        device_->init();

        size_t    batch_size = 2;
        BufferPtr logits     = createBuffer<half>({batch_size, 5},
                                                  {
                                                  __float2half(0),
                                                  __float2half(0),
                                                  __float2half(0),
                                                  __float2half(0.1),
                                                  __float2half(0.2),
                                                  __float2half(0.987),
                                                  __float2half(0.887),
                                                  __float2half(0.99999),
                                                  __float2half(0.1),
                                                  __float2half(0.2),
                                              },
                                              rtp_llm::AllocationType::DEVICE);
        BufferPtr vocab_mask = createBuffer<uint8_t>({batch_size, 5},
                                                     {
                                                         0,
                                                         1,
                                                         0,
                                                         1,
                                                         0,
                                                         1,
                                                         0,
                                                         0,
                                                         0,
                                                         0,
                                                     },
                                                     rtp_llm::AllocationType::DEVICE);

        printBufferData(*logits, "MYDEBUG_BEFORE_MASK_LOGITS");
        device_->maskLogits(*logits, *vocab_mask);
        printBufferData(*logits, "MYDEBUG_AFTER_MASK_LOGITS");

        auto logits_hosts = getBufferValues<half>(*logits);

        std::vector<half> expect_vec = {
            __float2half(0),
            -CUDART_INF_FP16,
            __float2half(0),
            -CUDART_INF_FP16,
            __float2half(0.2),
            -CUDART_INF_FP16,
            __float2half(0.887),
            __float2half(0.99999),
            __float2half(0.1),
            __float2half(0.2),
        };

        ASSERT_EQ(logits_hosts.size(), expect_vec.size()) << "Vectors x and y are of unequal length";
        for (size_t i = 0; i < expect_vec.size(); i++) {
            ASSERT_TRUE(logits_hosts[i] == expect_vec[i]);
        }
    }
    {
        device_ = new ROCmDevice(DeviceInitParams());
        device_->init();

        size_t    batch_size = 2;
        BufferPtr logits     = createBuffer<__nv_bfloat16>({batch_size, 5},
                                                           {
                                                           __float2bfloat16(0),
                                                           __float2bfloat16(0),
                                                           __float2bfloat16(0),
                                                           __float2bfloat16(0.1),
                                                           __float2bfloat16(0.2),
                                                           __float2bfloat16(0.987),
                                                           __float2bfloat16(0.887),
                                                           __float2bfloat16(0.99999),
                                                           __float2bfloat16(0.1),
                                                           __float2bfloat16(0.2),
                                                       },
                                                       rtp_llm::AllocationType::DEVICE);
        BufferPtr vocab_mask = createBuffer<uint8_t>({batch_size, 5},
                                                     {
                                                         0,
                                                         1,
                                                         0,
                                                         1,
                                                         0,
                                                         1,
                                                         0,
                                                         0,
                                                         0,
                                                         0,
                                                     },
                                                     rtp_llm::AllocationType::DEVICE);

        printBufferData(*logits, "MYDEBUG_BEFORE_MASK_LOGITS");
        device_->maskLogits(*logits, *vocab_mask);
        printBufferData(*logits, "MYDEBUG_AFTER_MASK_LOGITS");

        auto logits_hosts = getBufferValues<__nv_bfloat16>(*logits);

        std::vector<__nv_bfloat16> expect_vec = {
            __float2bfloat16(0),
            -CUDART_INF_BF16,
            __float2bfloat16(0),
            -CUDART_INF_BF16,
            __float2bfloat16(0.2),
            -CUDART_INF_BF16,
            __float2bfloat16(0.887),
            __float2bfloat16(0.99999),
            __float2bfloat16(0.1),
            __float2bfloat16(0.2),
        };

        ASSERT_EQ(logits_hosts.size(), expect_vec.size()) << "Vectors x and y are of unequal length";
        for (size_t i = 0; i < expect_vec.size(); i++) {
            ASSERT_TRUE(logits_hosts[i] == expect_vec[i]);
        }
    }
}
