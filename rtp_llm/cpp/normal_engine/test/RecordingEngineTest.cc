#include "rtp_llm/cpp/normal_engine/test/MockEngine.h"
#include "rtp_llm/cpp/normal_engine/speculative/MtpExecutor.h"
#include "rtp_llm/cpp/observability/ExecutionRecorder.h"
#include "alog/Configurator.h"
#include <filesystem>
#include <algorithm>
#include <cstring>
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
    void runDecodeSampleRateThinsDecodeRecords();
    void runRecordingWithoutCacheManager(bool speculative);
};

void RecordingEngineTest::runRecordingWithoutCacheManager(bool speculative) {
    DeviceTestBase::SetUp();
    char directory[] = "/tmp/rtp-recording-no-cache-XXXXXX";
    ASSERT_NE(mkdtemp(directory), nullptr);
    setenv("LOG_PATH", directory, 1);
    EngineInitParams params;
    params.model_id                                = 0;
    params.model_config_.max_seq_len               = 8;
    params.model_config_.vocab_size                = 10;
    params.model_config_.num_layers                = 1;
    params.model_config_.attn_config.head_num      = 2;
    params.model_config_.attn_config.kv_head_num   = 2;
    params.model_config_.attn_config.size_per_head = 64;
    params.model_config_.hidden_size               = 128;
    params.py_model                                = py::none();
    params.py_sp_model                             = py::none();
    ASSERT_TRUE(params.profiling_debug_logging_config.enable_batch_record);
    std::unique_ptr<NormalExecutor> normal_executor;
    std::unique_ptr<MtpExecutor>    mtp_executor;
    if (speculative) {
        auto draft_params = std::make_unique<std::vector<std::unique_ptr<EngineInitParams>>>();
        draft_params->push_back(std::make_unique<EngineInitParams>(params));
        auto propose_params = std::make_unique<ProposeModelEngineInitParams>(SP_TYPE_MTP, 2, std::move(draft_params));
        mtp_executor        = std::make_unique<MtpExecutor>(params, propose_params, nullptr);
    } else {
        normal_executor = std::make_unique<NormalExecutor>(params, nullptr);
    }
    auto& recorder = ExecutionRecorder::instance();
    ASSERT_TRUE(recorder.enabled());
    recorder.close();
    std::ifstream file(std::filesystem::path(directory) / "batch_record" / ("owner-" + recorder.owner())
                       / "manifest.json");
    ASSERT_TRUE(file.good());
    const std::string manifest((std::istreambuf_iterator<char>(file)), {});
    const auto        scope = speculative ? "mtp_executor" : "normal_executor";
    EXPECT_NE(manifest.find(std::string("\"scope\":\"") + scope + "\""), std::string::npos) << manifest;
    EXPECT_NE(manifest.find("\"vocab_size\":10"), std::string::npos) << manifest;
    for (const auto* field : {"cache_block_tokens", "cache_groups", "cache_config"}) {
        EXPECT_EQ(manifest.find(std::string("\"") + field + "\":"), std::string::npos) << manifest;
    }
}

TEST_F(RecordingEngineTest, RecordingWithoutCacheManager) {
    ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
    ASSERT_EXIT(
        {
            runRecordingWithoutCacheManager(false);
            _exit(::testing::Test::HasFailure() ? 1 : 0);
        },
        ::testing::ExitedWithCode(0),
        "");
}

TEST_F(RecordingEngineTest, SpeculativeRecordingWithoutCacheManager) {
    ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
    ASSERT_EXIT(
        {
            runRecordingWithoutCacheManager(true);
            _exit(::testing::Test::HasFailure() ? 1 : 0);
        },
        ::testing::ExitedWithCode(0),
        "");
}

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
    // Every submitted record carries the executor step's own wall duration.
    {
        const auto key_pos = batches.find("\"step_duration_us\":");
        ASSERT_NE(key_pos, std::string::npos) << batches;
        ASSERT_GT(std::stoull(batches.substr(key_pos + strlen("\"step_duration_us\":"), 12)), 0u) << batches;
    }
    // The mock config explicitly uses rate 1.0 for full decode coverage:
    // a 4-token streaming request records its prefill plus every decode step.
    EXPECT_NE(batches.find("\"phase\":\"decode\""), std::string::npos) << batches;
    EXPECT_NE(batches.find("\"q_len\":1"), std::string::npos) << batches;
    // Config-driven enablement publishes the manifest under LOG_PATH.
    bool manifest_found = false;
    for (const auto& entry : std::filesystem::directory_iterator(std::filesystem::path(directory) / "batch_record")) {
        if (entry.path().filename().string().rfind("owner-", 0) == 0
            && std::filesystem::exists(entry.path() / "manifest.json")) {
            manifest_found = true;
            std::ifstream     manifest_file(entry.path() / "manifest.json");
            const std::string manifest((std::istreambuf_iterator<char>(manifest_file)), {});
            for (const auto* field : {"cache_block_tokens", "cache_groups", "cache_config"}) {
                EXPECT_NE(manifest.find(std::string("\"") + field + "\":"), std::string::npos) << manifest;
            }
        }
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
    config.enable_batch_record     = true;
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

// record_decode_sample_rate deterministically thins decode-step recording:
// the credit accumulator records exactly floor(steps * rate) decode steps
// with even spacing (rate 0.5 -> every second step), while prefill batches
// stay fully recorded regardless of the rate.
void RecordingEngineTest::runDecodeSampleRateThinsDecodeRecords() {
    DeviceTestBase::SetUp();
    char directory[] = "/tmp/rtp-recording-sample-XXXXXX";
    ASSERT_NE(mkdtemp(directory), nullptr);
    setenv("LOG_PATH", directory, 1);
    CustomConfig config;
    config.enable_batch_record       = true;
    config.record_decode_sample_rate = 0.5;
    auto       engine                = createMockEngine(config);
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
    input->generate_config->max_new_tokens = 9;
    input->generate_config->is_streaming   = true;
    auto stream                            = engine->enqueue(input);
    for (int i = 0; i < 9; ++i) {
        ASSERT_TRUE(stream->nextOutput().ok());
    }
    ASSERT_FALSE(stream->nextOutput().ok());
    ASSERT_TRUE(engine->stop().ok());
    ExecutionRecorder::instance().close();
    std::ifstream file(std::string(directory) + "/batch_schedule.log");
    ASSERT_TRUE(file.good());
    const std::string  batches((std::istreambuf_iterator<char>(file)), {});
    size_t             prefill_rows = 0, decode_rows = 0;
    std::istringstream lines(batches);
    for (std::string line; std::getline(lines, line);) {
        if (line.empty())
            continue;
        const auto record = autil::legacy::AnyCast<ExecutionRecorder::Json>(autil::legacy::json::ParseJson(line));
        for (const auto& row : autil::legacy::AnyCast<ExecutionRecorder::JsonArray>(record.at("requests"))) {
            const auto parsed = autil::legacy::AnyCast<ExecutionRecorder::Json>(row);
            const auto phase  = autil::legacy::AnyCast<std::string>(parsed.at("phase"));
            if (phase == "decode")
                ++decode_rows;
            else
                ++prefill_rows;
        }
    }
    // One 3-token prefill step is always recorded. The remaining 8 decode
    // steps halve to exactly floor(8 * 0.5) = 4 recorded decode steps.
    EXPECT_EQ(prefill_rows, 1u) << batches;
    EXPECT_EQ(decode_rows, 4u) << batches;
}

TEST_F(RecordingEngineTest, DecodeSampleRateThinsDecodeRecords) {
    // Re-exec for the same reasons as LifecycleAndSnapshot.
    ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
    ASSERT_EXIT(
        {
            runDecodeSampleRateThinsDecodeRecords();
            _exit(::testing::Test::HasFailure() ? 1 : 0);
        },
        ::testing::ExitedWithCode(0),
        "");
}

}  // namespace rtp_llm
