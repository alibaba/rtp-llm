#include <gtest/gtest.h>
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include "autil/Log.h"

// Same pattern as rtp_llm/cpp/api_server/test/TestMain.cc: the cases need a live Python
// interpreter, so the gtest binary is exposed as a pybind module and driven from a py_test.
const std::string UNITTEST_DEFAULT_LOG_CONF = R"conf(
alog.rootLogger=INFO, unittestAppender
alog.max_msg_len=4096
alog.appender.unittestAppender=ConsoleAppender
alog.appender.unittestAppender.flush=true
alog.appender.unittestAppender.layout=PatternLayout
alog.appender.unittestAppender.layout.LogPattern=[%%d] [%%l] [%%t,%%F -- %%f():%%n] [%%m]
alog.logger.arpc=WARN
)conf";

namespace py = pybind11;
namespace rtp_llm {

// The module name must match the name of the generated shared library.
PYBIND11_MODULE(config_unittest_lib, m) {
    m.def(
        "RunCppUnittest",
        []() {
            AUTIL_LOG_CONFIG_FROM_STRING(UNITTEST_DEFAULT_LOG_CONF.c_str());
            ::testing::InitGoogleTest();
            return RUN_ALL_TESTS();
        },
        "run all cpp unittest case");
}

}  // namespace rtp_llm
