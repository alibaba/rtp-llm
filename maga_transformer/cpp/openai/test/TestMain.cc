#include <gtest/gtest.h>
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
namespace rtp_llm {

// 模块名与生成的动态库的名字必须一致
PYBIND11_MODULE(openai_unittest_lib, m) {
    m.def(
        "RunCppUnittest",
        []() {
            ::testing::InitGoogleTest();
            return RUN_ALL_TESTS();
        },
        "run all cpp unittest case");
}

}  // namespace rtp_llm
