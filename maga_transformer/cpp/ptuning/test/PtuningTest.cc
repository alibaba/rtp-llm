
#include "gtest/gtest.h"

#define private public
#include "maga_transformer/cpp/ptuning/Ptuning.h"
#include "maga_transformer/cpp/ptuning/PtuningConstructor.h"
#include "maga_transformer/cpp/utils/TimeUtility.h"
#include "src/fastertransformer/devices/testing/TestBase.h"
#include "src/fastertransformer/core/Tensor.h"

#include <memory>
#include <thread>
#include <chrono>

using namespace std;

namespace rtp_llm {

class PtuningTest : public DeviceTestBase {
protected:

protected:
};

TEST_F(PtuningTest, testConstruct) {
}

}
