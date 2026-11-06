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
    void runEmptyAlignmentRoundNotRecorded();
    void runFakeOnlyBatchNotRecorded();
};

void RecordingEngineTest::runLifecycleAndSnapshot() {
    DeviceTestBase::SetUp();
    char directory[] = "/tmp/rtp-recording-engine-XXXXXX";
    ASSERT_NE(mkdtemp(directory), nullptr);
    setenv("LOG_PATH", directory, 1);
    CustomConfig config;
    config.enable_batch_record = true;
    auto       engine          = createMockEngine(config);
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
    // record_decode defaults to false: decode batches are never recorded.
    EXPECT_EQ(batches.find("\"phase\":\"decode\""), std::string::npos) << batches;
    EXPECT_EQ(batches.find("\"q_len\":1"), std::string::npos) << batches;
    // Config-driven enablement publishes the manifest under LOG_PATH.
    bool manifest_found = false;
    for (const auto& entry : std::filesystem::directory_iterator(std::filesystem::path(directory) / "batch_record")) {
        if (entry.path().filename().string().rfind("owner-", 0) == 0
            && std::filesystem::exists(entry.path() / "manifest.json"))
            manifest_found = true;
    }
    EXPECT_TRUE(manifest_found);
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

// FFN-disaggregate deployments push empty alignment rounds through
// NormalExecutor::process() (skip_run stays false so the ranks stay in
// lockstep). Those rounds carry no request slots; recording them would
// flood the batch log with requests:[] lines and drain the recorder's
// total-bytes budget until it fails permanently.
void RecordingEngineTest::runEmptyAlignmentRoundNotRecorded() {
    DeviceTestBase::SetUp();
    char directory[] = "/tmp/rtp-recording-empty-XXXXXX";
    ASSERT_NE(mkdtemp(directory), nullptr);
    setenv("LOG_PATH", directory, 1);
    CustomConfig config;
    config.enable_batch_record  = true;
    config.enable_ffn_disaggregate = true;
    auto       engine              = createMockEngine(config);
    const auto batch_config =
        std::string("alog.max_msg_len=0\nalog.logger.batch_schedule=INFO, batchScheduleAppender\n")
        + "inherit.batch_schedule=false\nalog.appender.batchScheduleAppender=FileAppender\n"
        + "alog.appender.batchScheduleAppender.fileName=" + directory + "/batch_schedule.log\n"
        + "alog.appender.batchScheduleAppender.layout=PatternLayout\n"
        + "alog.appender.batchScheduleAppender.layout.LogPattern=%%m\n"
        + "alog.appender.batchScheduleAppender.async_flush=true\n";
    alog::Configurator::configureLoggerFromString(batch_config.c_str());
    // Death tests only surface the tail of child stderr; persist a step trace and
    // the recorded log content as undeclared outputs so remote runs stay diagnosable.
    const std::string trace_path =
        std::getenv("TEST_UNDECLARED_OUTPUTS_DIR") ?
            std::string(std::getenv("TEST_UNDECLARED_OUTPUTS_DIR")) + "/empty_align_trace.log" :
            std::string(directory) + "/empty_align_trace.log";
    auto trace = std::ofstream(trace_path, std::ios::app);
    // Mirror what NormalEngine::step does for an alignment round: the executor
    // receives an empty stream list and must run process() to completion.
    const std::list<GenerateStreamPtr> empty;
    for (int i = 0; i < 3; ++i) {
        const auto status = engine->executor_->process(empty, 0);
        trace << "process[" << i << "] ok=" << status.ok() << " status=" << status.ToString() << "\n";
        ASSERT_TRUE(status.ok());
    }
    const auto stop_status = engine->stop();
    trace << "stop ok=" << stop_status.ok() << "\n";
    ASSERT_TRUE(stop_status.ok());
    ExecutionRecorder::instance().close();
    std::ifstream file(std::string(directory) + "/batch_schedule.log");
    trace << "log file good=" << file.good() << "\n";
    ASSERT_TRUE(file.good());
    const std::string batches((std::istreambuf_iterator<char>(file)), {});
    trace << "---- batch_schedule.log ----\n" << batches << "---- end ----\n";
    EXPECT_EQ(batches.find("\"requests\""), std::string::npos) << batches;
}

TEST_F(RecordingEngineTest, EmptyAlignmentRoundIsNotRecorded) {
    // Re-exec for the same reasons as LifecycleAndSnapshot.
    ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
    ASSERT_EXIT(
        {
            runEmptyAlignmentRoundNotRecorded();
            _exit(::testing::Test::HasFailure() ? 1 : 0);
        },
        ::testing::ExitedWithCode(0),
        "");
}

// dp>1 deployments fill idle steps with fake streams (mayAddFakeStream ->
// createMinFakeStream). A fake-only batch carries no request slots; recording
// it would flood the batch log with is_fake rows and drain the recorder's
// total-bytes budget until it fails permanently.
void RecordingEngineTest::runFakeOnlyBatchNotRecorded() {
    DeviceTestBase::SetUp();
    char directory[] = "/tmp/rtp-recording-fake-XXXXXX";
    ASSERT_NE(mkdtemp(directory), nullptr);
    setenv("LOG_PATH", directory, 1);
    CustomConfig config;
    config.enable_batch_record = true;
    // PREFILL keeps the fake filler stream a context stream; PDFUSION/DECODE
    // roles would turn it into a decode stream, which the decode gate already
    // excludes by default, masking the bug under test.
    config.role_type  = RoleType::PREFILL;
    auto       engine = createMockEngine(config);
    const auto batch_config =
        std::string("alog.max_msg_len=0\nalog.logger.batch_schedule=INFO, batchScheduleAppender\n")
        + "inherit.batch_schedule=false\nalog.appender.batchScheduleAppender=FileAppender\n"
        + "alog.appender.batchScheduleAppender.fileName=" + directory + "/batch_schedule.log\n"
        + "alog.appender.batchScheduleAppender.layout=PatternLayout\n"
        + "alog.appender.batchScheduleAppender.layout.LogPattern=%%m\n"
        + "alog.appender.batchScheduleAppender.async_flush=true\n";
    alog::Configurator::configureLoggerFromString(batch_config.c_str());
    // Death tests only surface the tail of child stderr; persist a step trace and
    // the recorded log content as undeclared outputs so remote runs stay diagnosable.
    const std::string trace_path =
        std::getenv("TEST_UNDECLARED_OUTPUTS_DIR") ?
            std::string(std::getenv("TEST_UNDECLARED_OUTPUTS_DIR")) + "/fake_only_trace.log" :
            std::string(directory) + "/fake_only_trace.log";
    auto trace = std::ofstream(trace_path, std::ios::app);
    // Mirror the production idle-step shape: the scheduler hands the executor a
    // batch that contains only a fake filler stream.
    for (int i = 0; i < 3; ++i) {
        const std::list<GenerateStreamPtr> streams{engine->createMinFakeStream(1)};
        const auto                         status = engine->executor_->process(streams, 0);
        trace << "process[" << i << "] ok=" << status.ok() << " status=" << status.ToString() << "\n";
        ASSERT_TRUE(status.ok());
    }
    const auto stop_status = engine->stop();
    trace << "stop ok=" << stop_status.ok() << "\n";
    ASSERT_TRUE(stop_status.ok());
    ExecutionRecorder::instance().close();
    std::ifstream file(std::string(directory) + "/batch_schedule.log");
    trace << "log file good=" << file.good() << "\n";
    ASSERT_TRUE(file.good());
    const std::string batches((std::istreambuf_iterator<char>(file)), {});
    trace << "---- batch_schedule.log ----\n" << batches << "---- end ----\n";
    EXPECT_EQ(batches.find("\"requests\""), std::string::npos) << batches;
}

TEST_F(RecordingEngineTest, FakeOnlyBatchIsNotRecorded) {
    // Re-exec for the same reasons as LifecycleAndSnapshot.
    ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
    ASSERT_EXIT(
        {
            runFakeOnlyBatchNotRecorded();
            _exit(::testing::Test::HasFailure() ? 1 : 0);
        },
        ::testing::ExitedWithCode(0),
        "");
}

}  // namespace rtp_llm
