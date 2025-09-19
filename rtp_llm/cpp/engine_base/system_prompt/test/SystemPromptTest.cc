
#include "gtest/gtest.h"

#define private public
#include "rtp_llm/cpp/engine_base/system_prompt/SystemPrompt.h"
#include "rtp_llm/cpp/engine_base/system_prompt/SystemPromptConstructor.h"
#include "rtp_llm/cpp/devices/testing/TestBase.h"

#include <memory>
#include <thread>
#include <chrono>

using namespace std;

namespace rtp_llm {

class SystemPromptTest: public DeviceTestBase {
protected:
protected:
};

TEST_F(SystemPromptTest, testConstruct) {}

}  // namespace rtp_llm
