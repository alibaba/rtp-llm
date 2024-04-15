
#include "gtest/gtest.h"

#define private public
#include "maga_transformer/cpp/ptuning/Ptuning.h"
#include "maga_transformer/cpp/ptuning/PtuningConstructor.h"
#include "maga_transformer/cpp/utils/TimeUtility.h"
#include "maga_transformer/cpp/test/MockEngine.h"
#include "src/fastertransformer/devices/testing/TestBase.h"
#include "src/fastertransformer/core/Tensor.h"

#include <memory>
#include <thread>
#include <chrono>

using namespace std;

namespace rtp_llm {

class PtuningConstructorTest : public DeviceTestBase {
};

TEST_F(PtuningConstructorTest, testConstruct) {
    PtuningConstructor constructor;
    GptInitParameter params;
    // params.multi_task_prompt_tokens;
    NormalEngine* engine = createMockEngine(device_);
    //  std::map<int, PrefixParams> construct(const GptInitParameter& params, NormalEngine* engine) {
    // constructor.
}

}
