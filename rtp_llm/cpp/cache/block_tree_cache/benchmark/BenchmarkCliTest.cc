#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/BenchmarkCli.h"

#include <cstdlib>
#include <vector>
#include <gtest/gtest.h>

namespace rtp_llm::benchmark {
namespace {
struct Result {
    int         code;
    bool        ran;
    std::string output;
    std::string error;
};
Result invoke(std::vector<std::string> args) {
    args.insert(args.begin(), "benchmark");
    std::vector<char*> argv;
    for (auto& arg : args)
        argv.push_back(arg.data());
    argv.push_back(nullptr);
    bool ran = false;
    testing::internal::CaptureStdout();
    testing::internal::CaptureStderr();
    int        code   = runBenchmarkCli(static_cast<int>(args.size()), argv.data(), [&](const auto&) {
        ran = true;
        return true;
    });
    const auto output = testing::internal::GetCapturedStdout();
    const auto error  = testing::internal::GetCapturedStderr();
    return {code, ran, output, error};
}

TEST(BenchmarkCliTest, DispatchesHelpWithoutRunningGpuCode) {
    const std::vector<std::pair<std::vector<std::string>, std::string>> cases = {
        {{"--help"}, "<tree|transfer>"},
        {{"tree", "--model-profile=/does/not/exist", "--help"}, "--task-pool-size"},
        {{"transfer", "-h"}, "--transfer-concurrency"},
    };
    for (const auto& [args, expected] : cases) {
        const auto result = invoke(args);
        EXPECT_EQ(result.code, 0);
        EXPECT_FALSE(result.ran);
        EXPECT_TRUE(result.error.empty());
        EXPECT_NE(result.output.find(expected), std::string::npos);
    }
}

TEST(BenchmarkCliTest, ReportsErrorsBeforeRunningGpuCode) {
    for (const auto& args : std::vector<std::vector<std::string>>{{"tree", "--unknown=1"},
                                                                  {"tree", "--seed=abc"},
                                                                  {"tree", "--seed"},
                                                                  {"tree", "--task-pool-size=0"},
                                                                  {"transfer", "--transfer-concurrency=0"},
                                                                  {"tree", "--model-profile=/does/not/exist"}}) {
        SCOPED_TRACE(testing::PrintToString(args));
        const auto result = invoke(args);
        EXPECT_EQ(result.code, 1);
        EXPECT_FALSE(result.ran);
        EXPECT_NE(result.error.find("Benchmark failed:"), std::string::npos);
    }
}

TEST(BenchmarkCliTest, DispatchesMixedCommonAndTreeOptions) {
    const char* runfiles  = std::getenv("TEST_SRCDIR");
    const char* workspace = std::getenv("TEST_WORKSPACE");
    ASSERT_NE(runfiles, nullptr);
    ASSERT_NE(workspace, nullptr);
    const std::string path =
        std::string(runfiles) + "/" + workspace
        + "/rtp_llm/cpp/cache/block_tree_cache/benchmark/profiles/deepseek_v4_pro_fp8_tp1_cp1.json";
    const auto result = invoke({"tree", "--task-pool-size=8", "--model-profile", path, "--seed=7"});
    EXPECT_EQ(result.code, 0);
    EXPECT_TRUE(result.ran);
    EXPECT_TRUE(result.error.empty());
}
TEST(BenchmarkCliTest, RejectsInsufficientQueueForOwnersBeforeLoadingProfile) {
    const auto result = invoke({"transfer", "--transfer-worker-count=3", "--transfer-queue-max-size=2"});
    EXPECT_EQ(result.code, 1);
    EXPECT_FALSE(result.ran);
    EXPECT_NE(result.error.find("Q must be >= worker count"), std::string::npos);
}

TEST(BenchmarkCliTest, WorkLoopDefaultsAndLegacyOnlyArgumentsAreExplicit) {
    TransferOptions defaults;
    EXPECT_TRUE(defaults.btc_transfer_engine_async);
    EXPECT_TRUE(defaults.enable_crc);
    EXPECT_EQ(defaults.btc_crc_slot_count, 8u);
    for (const auto& argument : {"--copy-strategy=batch",
                                 "--transfer-worker-count=0",
                                 "--btc-transfer-engine-async=invalid",
                                 "--btc-crc-slot-count=0",
                                 "--enable-crc=false"}) {
        const auto result = invoke({"transfer", argument});
        EXPECT_EQ(result.code, 1);
        EXPECT_FALSE(result.ran);
    }
    const char* runfiles  = std::getenv("TEST_SRCDIR");
    const char* workspace = std::getenv("TEST_WORKSPACE");
    ASSERT_NE(runfiles, nullptr);
    ASSERT_NE(workspace, nullptr);
    const std::string profile =
        std::string(runfiles) + "/" + workspace
        + "/rtp_llm/cpp/cache/block_tree_cache/benchmark/profiles/deepseek_v4_pro_fp8_tp1_cp1.json";
    for (const auto& mode : {"true", "false"}) {
        std::vector<std::string> args{"transfer",
                                      "--model-profile=" + profile,
                                      std::string("--btc-transfer-engine-async=") + mode,
                                      "--btc-crc-slot-count=3",
                                      "--transfer-worker-count=2"};
        if (std::string(mode) == "false") {
            args.push_back("--copy-strategy=batch");
            args.push_back("--enable-crc=false");
        }
        const auto result = invoke(args);
        EXPECT_EQ(result.code, 0) << result.error;
        EXPECT_TRUE(result.ran);
    }
}

}  // namespace
}  // namespace rtp_llm::benchmark
