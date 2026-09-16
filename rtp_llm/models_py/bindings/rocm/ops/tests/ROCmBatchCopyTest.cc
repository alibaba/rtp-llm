#include "rtp_llm/cpp/testing/BatchCopyTest.hpp"
#include <gtest/gtest.h>

class ROCmBatchCopyTest: public BatchCopyTest {};

TEST_F(ROCmBatchCopyTest, D2DSingleCopy) {
    testD2DSingleCopy();
}
TEST_F(ROCmBatchCopyTest, D2DMultipleCopies) {
    testD2DMultipleCopies();
}
TEST_F(ROCmBatchCopyTest, D2DUniformSizes) {
    testD2DUniformSizes();
}
TEST_F(ROCmBatchCopyTest, D2DVariableSizes) {
    testD2DVariableSizes();
}
TEST_F(ROCmBatchCopyTest, D2DLargeBatch) {
    testD2DLargeBatch();
}
TEST_F(ROCmBatchCopyTest, D2DWithH2DAndD2H) {
    testD2DWithH2DAndD2H();
}
