#include "rtp_llm/cpp/testing/RejectionSamplingOpTest.hpp"

class RocmRejectionSamplingOpTest: public RejectionSamplingOpTest {};

TEST_F(RocmRejectionSamplingOpTest, referenceCases) {
    runReferenceCases();
}

TEST_F(RocmRejectionSamplingOpTest, stochasticSemanticsCases) {
    runStochasticSemanticsCases();
}

TEST_F(RocmRejectionSamplingOpTest, pointMassDraftCases) {
    runPointMassDraftCases();
}

TEST_F(RocmRejectionSamplingOpTest, zeroAndOneSpeculativeTokenCases) {
    runZeroAndOneSpeculativeTokenCases();
}

TEST_F(RocmRejectionSamplingOpTest, rejectsInvalidTensorMetadata) {
    runRejectsInvalidTensorMetadata();
}
