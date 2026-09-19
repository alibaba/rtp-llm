#include "rtp_llm/cpp/normal_engine/test/MockEngine.h"
#include "rtp_llm/cpp/observability/ExecutionRecorder.h"
#include "alog/Configurator.h"
#include <filesystem>
#include <algorithm>
#include <fstream>
#include <iterator>
#include <unistd.h>

namespace rtp_llm {
class RecordingEngineTest: public DeviceTestBase {};

TEST_F(RecordingEngineTest, LifecycleAndSnapshot) {
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
    std::string batches;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(directory)) {
        const auto name = entry.path().filename().string();
        EXPECT_NE(name, "request_events.jsonl");
        EXPECT_NE(name, "executions.jsonl");
        EXPECT_NE(name, "requests.jsonl");
        if (name != "batch_schedule.log")
            continue;
        std::ifstream file(entry.path());
        std::string   text((std::istreambuf_iterator<char>(file)), {});
        batches += text;
    }
    for (const auto* field : {"enqueue_time_unix_ns", "first_scheduled_time_unix_ns", "finish_time_unix_ns",
                              "scheduler_step_id", "session_id", "replica_id", "dp_rank", "world_rank",
                              "prompt_tokens", "prefix_cache_hit_tokens",
                              "execution_id", "owner_instance_id", "q_tokens", "kv_tokens_before",
                              "model_role", "length_source", "batch_slot", "first_token_produced", "sequence_id", "is_fake"}) {
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

}  // namespace rtp_llm
