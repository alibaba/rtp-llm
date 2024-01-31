#include "tests/unittests/gtest_utils.h"

using namespace fastertransformer;

class CudaOpsTest: public FtTestBase {
public:
    void SetUp() override
    {
        FtTestBase::SetUp();
    }
    void TearDown() override
    {
        FtTestBase::TearDown();
    }

};

TEST_F(CudaOpsTest, testBasic) {
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
