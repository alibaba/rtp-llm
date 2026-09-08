#include "rtp_llm/cpp/testing/RejectionSamplingOpTest.hpp"

class CudaRejectionSamplingOpTest: public RejectionSamplingOpTest {};

TEST_F(CudaRejectionSamplingOpTest, referenceCases) {
    runReferenceCases();
}

TEST_F(CudaRejectionSamplingOpTest, zeroAndOneSpeculativeTokenCases) {
    runZeroAndOneSpeculativeTokenCases();
}

TEST_F(CudaRejectionSamplingOpTest, rejectsInvalidTensorMetadata) {
    runRejectsInvalidTensorMetadata();
}
