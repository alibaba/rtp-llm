#include "rtp_llm/cpp/observability/ExecutionRecorder.h"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <future>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <vector>
#include "alog/Logger.h"
#include "alog/Appender.h"
#include "alog/Configurator.h"
#include "rtp_llm/cpp/utils/Logger.h"

static void configureBatch(const std::string& path, const std::string& level = "INFO") {
    const auto config = "alog.max_msg_len=64\nalog.logger.batch_schedule=" + level
                        + ", batchScheduleAppender\n"
                          "inherit.batch_schedule=false\nalog.appender.batchScheduleAppender=FileAppender\n"
                          "alog.appender.batchScheduleAppender.fileName="
                        + path
                        + "\n"
                          "alog.appender.batchScheduleAppender.layout=PatternLayout\n"
                          "alog.appender.batchScheduleAppender.layout.LogPattern=%%m\n"
                          "alog.appender.batchScheduleAppender.async_flush=true\n"
                          "alog.appender.batchScheduleAppender.flush_interval=20\n"
                          "alog.appender.batchScheduleAppender.flush_threshold=64\n"
                          "alog.appender.batchScheduleAppender.max_file_size=0\n";
    alog::Configurator::configureLoggerFromString(config.c_str());
}

// CPU-only test; link against alog (no CUDA dependency).
static std::string readFile(const std::string& path) {
    std::ifstream input(path);
    assert(input.good());
    std::ostringstream contents;
    contents << input.rdbuf();
    return contents.str();
}

int main(int argc, char** argv) {
    const char* directory = argc == 2 ? argv[1] : std::getenv("TEST_TMPDIR");
    assert(directory);
    const auto root_path = std::string(directory) + "/root.log";
    alog::Logger::getRootLogger()->setAppender(alog::FileAppender::getAppender(root_path.c_str()));
    RTP_LLM_BATCH_SCHEDULE_LOG_INFO("{\"unconfigured\":true}");  // Must not inherit root.
    using namespace rtp_llm;
    const std::string special = "中文\n\r\t\"\\" + std::string(1, '\0');
    const auto        encoded = ExecutionRecorder::quote(special);
    assert(encoded.find('\n') == std::string::npos);
    assert(autil::legacy::AnyCast<std::string>(autil::legacy::json::ParseJson(encoded)) == special);
    const auto structured =
        ExecutionRecorder::toJson(ExecutionRecorder::Json{{"null", ExecutionRecorder::JsonValue()},
                                                          {"flag", true},
                                                          {"text", special},
                                                          {"id", int64_t(9007199254740993LL)},
                                                          {"items", ExecutionRecorder::JsonArray{int32_t(3)}}});
    assert(structured.find("9007199254740993") != std::string::npos);
    assert(structured.find("\"null\":null") != std::string::npos);
    assert(structured.find("\"flag\":true") != std::string::npos);
    assert(structured.find("\"items\":[3]") != std::string::npos);
    {
        const auto path = std::string(directory) + "/delayed-batch.log";
        configureBatch(path);
        ExecutionRecorder recorder(std::string(directory) + "/delayed", "ttft", 0, 0);
        auto              make_batch = [&](int64_t execution) {
            auto record            = recorder.identity();
            record["exec_id"] = execution;
            return std::make_shared<RecordedBatch>(
                recorder,
                std::move(record),
                std::vector<ExecutionRecorder::Json>{{{"request_id", std::string("r1")}, {"q_len", 4}}});
        };
        // No output update (e.g. a non-final prefill chunk): unknown TTFT.
        make_batch(1).reset();
        auto               final_chunk = make_batch(2);
        std::promise<void> release;
        auto               gate = release.get_future().share();
        std::thread        worker([batch = std::move(final_chunk), gate]() mutable {
            gate.wait();
            batch->timings[0] = {1250, true};
            batch.reset();  // Dispatch ownership ends here, not on the caller.
        });
        // Drain earlier recorder work while dispatch is blocked. The pending
        // batch must not have been submitted just because forward finished.
        std::promise<void> drained;
        assert(recorder.submit("barrier.jsonl", [&] {
            drained.set_value();
            return "{}";
        }));
        drained.get_future().wait();
        getBatchScheduleLogger()->flush();
        assert(readFile(path).find("\"exec_id\":2") == std::string::npos);
        release.set_value();
        worker.join();
        {
            auto decode        = make_batch(3);
            decode->timings[0] = {1250, false};
        }
        try {
            auto failed = make_batch(4);
            throw std::runtime_error("forward failed before dispatch");
        } catch (const std::runtime_error&) {}
        recorder.close();
        std::istringstream input(readFile(path));
        std::string        line;
        int                count = 0;
        while (std::getline(input, line)) {
            const auto record   = autil::legacy::AnyCast<ExecutionRecorder::Json>(autil::legacy::json::ParseJson(line));
            const auto requests = autil::legacy::AnyCast<ExecutionRecorder::JsonArray>(record.at("requests"));
            const auto row      = autil::legacy::AnyCast<ExecutionRecorder::Json>(requests.at(0));
            ++count;
            assert(line.find("\"q_len\":4") != std::string::npos);
            assert(row.count("first_token_produced") == 0);
            if (count == 1 || count == 4)
                assert(row.count("ttft_us") == 0);
            else
                assert(line.find("\"ttft_us\":1250") != std::string::npos);
        }
        assert(count == 4);
    }
    {
        ExecutionRecorder recorder(std::string(directory) + "/normal", "test", 2, 1, 1024, 1024);
        assert(recorder.enabled());
        recorder.setMetadata({{"model_config", special},
                              {"record_decode", false},
                              {"model_fingerprint", ExecutionRecorder::JsonValue()}});
        RecordedRequest request(recorder);
        assert(!request.scheduleTiming().enqueue_time_unix_ns);
        request.scheduled();  // Scheduling before enqueue must not invent a timestamp.
        assert(!request.scheduleTiming().first_scheduled_time_unix_ns);
        request.enqueue();
        const auto enqueued = request.scheduleTiming();
        request.enqueue();
        std::vector<std::thread> threads;
        for (int i = 0; i < 16; ++i)
            threads.emplace_back([&] { request.scheduled(); });
        for (auto& thread : threads)
            thread.join();
        const auto scheduled = request.scheduleTiming();
        assert(scheduled.enqueue_time_unix_ns == enqueued.enqueue_time_unix_ns);
        assert(scheduled.first_scheduled_time_unix_ns.has_value());
        request.scheduled();
        assert(request.scheduleTiming().first_scheduled_time_unix_ns == scheduled.first_scheduled_time_unix_ns);
        RecordedRequest other(recorder);
        other.enqueue();
        assert(other.id() != request.id());
        assert(!other.scheduleTiming().first_scheduled_time_unix_ns);
        recorder.close();
        assert(recorder.dropped() == 0);
        const auto owner_dir = std::string(directory) + "/normal/owner-" + recorder.owner();
        for (const auto& entry : std::filesystem::directory_iterator(owner_dir))
            assert(entry.path().filename() == "manifest.json");
        const auto manifest = readFile(owner_dir + "/manifest.json");
        const auto parsed   = autil::legacy::AnyCast<ExecutionRecorder::Json>(autil::legacy::json::ParseJson(manifest));
        const auto metadata = autil::legacy::AnyCast<ExecutionRecorder::Json>(parsed.at("engine"));
        assert(autil::legacy::AnyCast<std::string>(metadata.at("model_config")) == special);
        assert(metadata.at("model_fingerprint").IsEmpty());
        assert(autil::legacy::AnyCast<std::string>(parsed.at("owner_id")) == recorder.owner());
        assert(parsed.count("owner_instance_id") == 0);
        assert(manifest.find("\"submitted_to_alog\":0") != std::string::npos);
        assert(manifest.find("\"written\":null") != std::string::npos);
        assert(manifest.find("\"complete\":false") != std::string::npos);
        assert(manifest.find("\"closed\":true") != std::string::npos);
    }
    {
        ExecutionRecorder  recorder(std::string(directory) + "/overflow", "test", 0, 0, 1);
        std::promise<void> entered, release;
        auto               gate = release.get_future().share();
        assert(recorder.submit("events.jsonl", [&] {
            entered.set_value();
            gate.wait();
            return "{}";
        }));
        entered.get_future().wait();
        assert(recorder.submit("events.jsonl", "{}"));
        assert(!recorder.submit("events.jsonl", "{}"));
        release.set_value();
        recorder.close();
        assert(recorder.dropped() == 1);
    }
    {
        ExecutionRecorder disabled("", "test", 0, 0);
        assert(!disabled.enabled());
        assert(!disabled.submit("events.jsonl", "{}"));
    }
    {
        const auto batch_path = std::string(directory) + "/configured-batches.jsonl";
        configureBatch(batch_path);
        ExecutionRecorder recorder(std::string(directory) + "/single_file", "test", 0, 0);
        for (int i = 0; i < 128; ++i)
            assert(recorder.submit("batches.jsonl", "{\"exec_id\":" + std::to_string(i + 1) + "}"));
        recorder.close();
        const auto    owner_dir = std::string(directory) + "/single_file/owner-" + recorder.owner();
        std::ifstream input(batch_path);
        std::string   line;
        for (int i = 0; i < 128; ++i) {
            assert(std::getline(input, line));
            assert(line == "{\"exec_id\":" + std::to_string(i + 1) + "}");
        }
        assert(!std::getline(input, line));
        assert(!std::filesystem::exists(owner_dir + "/batches.jsonl"));
        assert(!std::filesystem::exists(batch_path + ".1"));
        assert(readFile(owner_dir + "/manifest.json").find("\"max_file_bytes\":0") != std::string::npos);
    }
    {
        const auto path = std::string(directory) + "/configured-live.jsonl";
        configureBatch(path);
        ExecutionRecorder recorder(std::string(directory) + "/live", "test", 0, 0);
        assert(recorder.submit("batches.jsonl", "{\"live\":true}"));
        bool visible = false;
        for (int i = 0; i < 200; ++i) {
            if (std::filesystem::exists(path) && readFile(path) == "{\"live\":true}\n") {
                visible = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        assert(visible);  // Periodic alog flush works before recorder.close().
        recorder.close();
    }
    {
        // Exercise messages larger than even the production alog.max_msg_len.
        const auto path = std::string(directory) + "/configured-large.jsonl";
        configureBatch(path);
        ExecutionRecorder recorder(std::string(directory) + "/large", "test", 0, 0);
        const auto        line = "{\"payload\":\"" + std::string(3 * 1024 * 1024, 'x') + "\"}";
        assert(recorder.submit("batches.jsonl", line));
        recorder.close();
        assert(readFile(path) == line + "\n");
    }
    {
        const auto path = std::string(directory) + "/configured-level.jsonl";
        configureBatch(path, "ERROR");
        bool evaluated = false;
        RTP_LLM_BATCH_SCHEDULE_LOG_INFO(([&] {
            evaluated = true;
            return "{\"filtered\":true}";
        })());
        assert(!evaluated);
        ExecutionRecorder recorder(std::string(directory) + "/filtered", "test", 0, 0);
        assert(recorder.submit("batches.jsonl", "{\"filtered\":true}"));
        recorder.close();
        assert(readFile(path).empty());
        RTP_LLM_BATCH_SCHEDULE_LOG(alog::LOG_LEVEL_ERROR, "{\"percent\":\"100%\"}");
        rtp_llm::getBatchScheduleLogger()->flush();
        assert(readFile(path) == "{\"percent\":\"100%\"}\n");
    }
    {
        ExecutionRecorder recorder(std::string(directory) + "/producer_error", "test", 0, 0);
        assert(recorder.submit("events.jsonl", []() -> std::string { throw std::runtime_error("snapshot failed"); }));
        recorder.close();
        const auto manifest =
            readFile(std::string(directory) + "/producer_error/owner-" + recorder.owner() + "/manifest.json");
        assert(manifest.find("\"errors\":1") != std::string::npos);
    }
    {
        ExecutionRecorder limited(std::string(directory) + "/budget", "test", 0, 0, 8, 100, 4);
        assert(limited.submit("events.jsonl", "12345"));
        limited.close();
        assert(limited.dropped() == 1);
    }
    alog::Logger::getRootLogger()->flush();
    assert(readFile(root_path).empty());  // Recorder JSON never inherits the engine/root appender.
    {
        // Exercise the shipped config: all batch records append to the fixed file.
        const auto cwd     = std::filesystem::current_path();
        const auto config  = cwd / "rtp_llm/config/alog.conf";
        const auto workdir = std::filesystem::absolute(std::string(directory) + "/real_config");
        std::filesystem::create_directories(workdir);
        std::filesystem::current_path(workdir);
        alog::Configurator::configureLogger(config.c_str());
        RTP_LLM_BATCH_SCHEDULE_LOG_INFO("{\"actual_config\":true}");
        RTP_LLM_BATCH_SCHEDULE_LOG_INFO("{\"second_batch\":true}");
        rtp_llm::getBatchScheduleLogger()->flush();
        assert(readFile("logs/batch_schedule.log") == "{\"actual_config\":true}\n{\"second_batch\":true}\n");
        for (const auto& entry : std::filesystem::directory_iterator("logs")) {
            const auto name = entry.path().filename().string();
            if (name.find("batches.") == 0)
                assert(name == "batch_schedule.log");
        }
        assert(readFile("logs/engine.log").empty());
        std::filesystem::current_path(cwd);
    }
}
