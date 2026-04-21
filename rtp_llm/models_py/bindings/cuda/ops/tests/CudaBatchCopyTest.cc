#include "rtp_llm/cpp/testing/BatchCopyTest.hpp"

class CudaBatchCopyTest: public BatchCopyTest {};

TEST_F(CudaBatchCopyTest, D2DSingleCopy) {
    testD2DSingleCopy();
}
TEST_F(CudaBatchCopyTest, D2DMultipleCopies) {
    testD2DMultipleCopies();
}
TEST_F(CudaBatchCopyTest, D2DUniformSizes) {
    testD2DUniformSizes();
}
TEST_F(CudaBatchCopyTest, D2DVariableSizes) {
    testD2DVariableSizes();
}
TEST_F(CudaBatchCopyTest, D2DLargeBatch) {
    testD2DLargeBatch();
}
TEST_F(CudaBatchCopyTest, D2DWithH2DAndD2H) {
    testD2DWithH2DAndD2H();
}
