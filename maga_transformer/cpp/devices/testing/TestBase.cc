#include "maga_transformer/cpp/devices/testing/TestBase.h"

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}