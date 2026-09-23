#include "rtp_llm/cpp/normal_engine/test/MockEngine.h"
#include "rtp_llm/cpp/observability/ExecutionRecorder.h"
#include "alog/Configurator.h"
#include <filesystem>
#include <algorithm>
#include <fstream>
#include <iterator>
#include <set>
#include <sstream>
#include <unistd.h>

namespace rtp_llm {
class RecordingEngineTest: public DeviceTestBase {
public:
    void SetUp() override {}  // Initialize CUDA only in the fresh child process.
    void runLifecycleAndSnapshot();
};

void RecordingEngineTest::runLifecycleAndSnapshot() {
    DeviceTestBase::SetUp();
    char directory[] = "/tmp/rtp-recording-engine-XXXXXX";
    ASSERT_NE(mkdtemp(directory), nullptr);
    setenv("RTP_LLM_RECORD_DIR", directory, 1);
    setenv("RTP_LLM_RECORD_SESSION", "engine-test", 1);
    auto       engine = createMockEngine(CustomConfig{});
    const auto batch_config =
        std::string("alog.max_msg_len=0\nalog.logger.batch_schedule=INFO, batchScheduleAppender\n")
        + "inherit.batch_schedule=false\nalog.appender.batchScheduleAppender=FileAppender\n"
        + "alog.appender.batchScheduleAppender.fileName=" + directory + "/batch_schedule.log\n"
        + "alog.appender.batchScheduleAppender.layout=PatternLayout\n"
        + "alog.appender.batchScheduleAppender.layout.LogPattern=%%m\n"
        + "alog.appender.batchScheduleAppender.async_flush=true\n";
    alog::Configurator::configureLoggerFromString(batch_config.c_str());
    auto input                             = std::make_shared<GenerateInput>();
    input->input_ids                       = torch::tensor({1, 2, 3}, torch::kInt32);
    input->generate_config                 = std::make_shared<GenerateConfig>();
    input->generate_config->max_new_tokens = 4;
    input->generate_config->is_streaming   = true;
    auto stream                            = engine->enqueue(input);
    ASSERT_TRUE(stream->nextOutput().ok());
    ASSERT_TRUE(stream->nextOutput().ok());
    ASSERT_TRUE(stream->nextOutput().ok());
    ASSERT_TRUE(stream->nextOutput().ok());
    ASSERT_FALSE(stream->nextOutput().ok());
    ASSERT_TRUE(engine->stop().ok());
    ExecutionRecorder::instance().close();
    std::ifstream file(std::string(directory) + "/batch_schedule.log");
    ASSERT_TRUE(file.good());
    const std::string batches((std::istreambuf_iterator<char>(file)), {});
    // Exercise the production snapshot path: nanosecond-sized IDs must remain
    // distinct decimal strings after a JSON round trip, even above 2^53.
    std::istringstream    lines(batches);
    std::set<std::string> execution_ids;
    for (std::string line; std::getline(lines, line);) {
        if (line.empty())
            continue;
        const auto record = autil::legacy::AnyCast<ExecutionRecorder::Json>(autil::legacy::json::ParseJson(line));
        const auto id     = autil::legacy::AnyCast<std::string>(record.at("exec_id"));
        EXPECT_GT(std::stoll(id), 9007199254740991LL);
        EXPECT_EQ(std::to_string(std::stoll(id)), id);
        EXPECT_TRUE(execution_ids.insert(id).second);
        EXPECT_EQ(autil::legacy::AnyCast<std::string>(
                      autil::legacy::json::ParseJson(ExecutionRecorder::toJson(record.at("exec_id")))),
                  id);
    }
    EXPECT_FALSE(execution_ids.empty());
    // Current optional fields are omitted for a single real sequence.
    for (const auto* field : {"sequence_id", "is_fake"}) {
        EXPECT_EQ(batches.find(std::string("\"") + field + "\":"), std::string::npos) << field << batches;
    }
    EXPECT_NE(batches.find("\"schema_version\":1"), std::string::npos) << batches;
    EXPECT_NE(batches.find("\"owner_id\":"), std::string::npos) << batches;
    EXPECT_NE(batches.find("\"exec_id\":"), std::string::npos) << batches;
    EXPECT_NE(batches.find("\"request_id\":"), std::string::npos) << batches;
    EXPECT_NE(batches.find("\"input_len\":3"), std::string::npos) << batches;
    EXPECT_NE(batches.find("\"reuse_len\":0"), std::string::npos) << batches;
    EXPECT_NE(batches.find("\"q_len\":3"), std::string::npos) << batches;
    EXPECT_NE(batches.find("\"kv_len\":0"), std::string::npos) << batches;
    EXPECT_NE(batches.find("\"ttft_us\":"), std::string::npos) << batches;
    EXPECT_EQ(batches.find("\"ttft_us\":null"), std::string::npos) << batches;
    const auto* decode_env    = std::getenv("RTP_LLM_RECORD_DECODE");
    const bool  record_decode = decode_env && std::string(decode_env) == "1";
    if (record_decode) {
        EXPECT_NE(batches.find("\"phase\":\"decode\""), std::string::npos) << batches;
        EXPECT_NE(batches.find("\"q_len\":1"), std::string::npos) << batches;
    } else {
        EXPECT_EQ(batches.find("\"phase\":\"decode\""), std::string::npos) << batches;
        EXPECT_EQ(batches.find("\"q_len\":1"), std::string::npos) << batches;
    }
    unsetenv("RTP_LLM_RECORD_DIR");
    unsetenv("RTP_LLM_RECORD_SESSION");
}

TEST_F(RecordingEngineTest, LifecycleAndSnapshot) {
    // Re-exec rather than fork with an initialized CUDA runtime. Each repeat
    // gets fresh recorder and alog state; failures in the child reach the parent.
    ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
    ASSERT_EXIT(
        {
            runLifecycleAndSnapshot();
            _exit(::testing::Test::HasFailure() ? 1 : 0);
        },
        ::testing::ExitedWithCode(0),
        "");
}

}  // namespace rtp_llm
