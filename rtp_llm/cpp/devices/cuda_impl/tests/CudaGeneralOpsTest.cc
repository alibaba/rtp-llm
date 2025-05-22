#include "rtp_llm/cpp/devices/cuda_impl/tests/CudaTestUtils.h"
#include "rtp_llm/cpp/devices/base_tests/GeneralOpsTest.hpp"

RTP_LLM_RUN_DEVICE_TEST(GeneralOpsTest, testCopyWithSlicing);
RTP_LLM_RUN_DEVICE_TEST(GeneralOpsTest, testTranspose);
RTP_LLM_RUN_DEVICE_TEST(GeneralOpsTest, testConvert);
RTP_LLM_RUN_DEVICE_TEST(GeneralOpsTest, testQBufferCopy);
RTP_LLM_RUN_DEVICE_TEST(GeneralOpsTest, testSelect1d);
RTP_LLM_RUN_DEVICE_TEST(GeneralOpsTest, testSelect);
RTP_LLM_RUN_DEVICE_TEST(GeneralOpsTest, testConcat);
RTP_LLM_RUN_DEVICE_TEST(GeneralOpsTest, testSplit);
RTP_LLM_RUN_DEVICE_TEST(GeneralOpsTest, testEmbeddingLookup);
RTP_LLM_RUN_DEVICE_TEST(GeneralOpsTest, testMultiply);
RTP_LLM_RUN_DEVICE_TEST(GeneralOpsTest, testLoss);
