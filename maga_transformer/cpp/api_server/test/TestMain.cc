#include <gtest/gtest.h>
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include "autil/Log.h"

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

// 模块名与生成的动态库的名字必须一致
PYBIND11_MODULE(api_server_unittest_lib, m) {
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
