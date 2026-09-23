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
#include <sys/wait.h>
#include <unistd.h>
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

static void testMultiprocessConfig(const std::string& directory) {
    const auto config  = std::filesystem::current_path() / "rtp_llm/config/alog.conf";
    const auto workdir = std::filesystem::absolute(directory + "/multiprocess_config");
    std::filesystem::create_directories(workdir);
    int start[2];
    assert(pipe(start) == 0);
    std::vector<pid_t> children;
    // Fork before initializing alog or starting any of its background threads.
    for (int owner = 0; owner < 4; ++owner) {
        const auto pid = fork();
        assert(pid >= 0);
        if (pid == 0) {
            close(start[1]);
            std::filesystem::current_path(workdir);
            alog::Configurator::configureLogger(config.c_str());
            char token;
            assert(read(start[0], &token, 1) == 1);
            close(start[0]);
            const std::string payload(64 * 1024, 'a' + owner);
            for (int sequence = 0; sequence < 300; ++sequence) {
                const auto record = "{\"pid\":" + std::to_string(getpid()) + ",\"sequence\":" + std::to_string(sequence)
                                    + ",\"payload\":\"" + payload + "\"}";
                RTP_LLM_BATCH_SCHEDULE_LOG_INFO(record);
            }
            rtp_llm::Logger::getBatchScheduleLogger()->flush();
            _exit(0);
        }
        children.push_back(pid);
    }
    close(start[0]);
    assert(write(start[1], "1234", 4) == 4);
    close(start[1]);
    for (const auto pid : children) {
        int status;
        assert(waitpid(pid, &status, 0) == pid);
        assert(WIFEXITED(status) && WEXITSTATUS(status) == 0);
    }
    for (size_t owner = 0; owner < children.size(); ++owner) {
        const auto    pid = children[owner];
        std::ifstream input(workdir / "logs" / ("batch_schedule." + std::to_string(pid) + ".log"));
        assert(input.good());
        const std::string payload(64 * 1024, 'a' + owner);
        std::string       line;
        for (int sequence = 0; sequence < 300; ++sequence) {
            assert(std::getline(input, line));
            autil::legacy::json::ParseJson(line);
            const auto expected = "{\"pid\":" + std::to_string(pid) + ",\"sequence\":" + std::to_string(sequence)
                                  + ",\"payload\":\"" + payload + "\"}";
            assert(line == expected);
        }
        assert(!std::getline(input, line));
    }
    assert(!std::filesystem::exists(workdir / "logs/batch_schedule.log"));
}

int main(int argc, char** argv) {
    const char* directory = argc == 2 ? argv[1] : std::getenv("TEST_TMPDIR");
    assert(directory);
    testMultiprocessConfig(directory);
    const auto root_path = std::string(directory) + "/root.log";
    alog::Logger::getRootLogger()->setAppender(alog::FileAppender::getAppender(root_path.c_str()));
    RTP_LLM_BATCH_SCHEDULE_LOG_INFO("{\"unconfigured\":true}");  // Must not inherit root.
    using namespace rtp_llm;
    {
        const auto path = std::string(directory) + "/initialization.log";
        configureBatch(path);
        ExecutionRecorder recorder(std::string(directory) + "/initialization", "test");
        assert(recorder.needsConfiguration());
        assert(!recorder.enabled());
        bool produced = false;
        assert(!recorder.submitBatch([&] {
            produced = true;
            return "{}";
        }));
        assert(!produced);
        assert(!std::filesystem::exists(std::string(directory) + "/initialization/owner-" + recorder.owner()));
        assert(recorder.configure(3, 2, {{"model", std::string("target")}}));
        assert(recorder.enabled());
        assert(!recorder.needsConfiguration());
        const auto manifest_path =
            std::string(directory) + "/initialization/owner-" + recorder.owner() + "/manifest.json";
        const auto initial = readFile(manifest_path);
        assert(initial.find("\"world_rank\":3") != std::string::npos);
        assert(initial.find("\"dp_rank\":2") != std::string::npos);
        assert(initial.find("\"model\":\"target\"") != std::string::npos);
        assert(initial.find("\"generated\":0") != std::string::npos);
        assert(!recorder.configure(9, 9, {{"model", std::string("replacement")}}));
        assert(readFile(manifest_path) == initial);
        assert(recorder.submitBatch("{}"));
        assert(!recorder.configure(9, 9, {}));
        recorder.close();
        assert(!recorder.enabled());
        assert(!recorder.configure(3, 2, {}));
        assert(!recorder.submitBatch("{}"));
        assert(readFile(path) == "{}\n");
        const auto final_manifest = readFile(manifest_path);
        assert(final_manifest.find("\"world_rank\":3") != std::string::npos);
        assert(final_manifest.find("\"model\":\"target\"") != std::string::npos);
        assert(final_manifest.find("\"generated\":1") != std::string::npos);
        assert(final_manifest.find("\"closed\":true") != std::string::npos);
    }
    {
        // Competing initializers must publish exactly one coherent configuration.
        ExecutionRecorder        recorder(std::string(directory) + "/concurrent-init", "test");
        std::atomic<int>         configured{0};
        std::vector<std::thread> threads;
        for (int i = 0; i < 8; ++i) {
            threads.emplace_back([&, i] {
                if (recorder.configure(i, i, {{"initializer", i}}))
                    ++configured;
            });
        }
        for (auto& thread : threads)
            thread.join();
        assert(configured == 1);
        assert(recorder.enabled());
        recorder.close();
        const auto manifest = autil::legacy::AnyCast<ExecutionRecorder::Json>(autil::legacy::json::ParseJson(
            readFile(std::string(directory) + "/concurrent-init/owner-" + recorder.owner() + "/manifest.json")));
        const auto metadata = autil::legacy::AnyCast<ExecutionRecorder::Json>(manifest.at("engine"));
        assert(ExecutionRecorder::toJson(manifest.at("world_rank"))
               == ExecutionRecorder::toJson(metadata.at("initializer")));
        assert(ExecutionRecorder::toJson(manifest.at("dp_rank"))
               == ExecutionRecorder::toJson(metadata.at("initializer")));
    }
    {
        ExecutionRecorder closed(std::string(directory) + "/closed-before-init", "test");
        closed.close();
        assert(!closed.needsConfiguration());
        assert(!closed.configure(0, 0, {}));
        assert(!closed.submitBatch("{}"));

        const auto blocker = std::string(directory) + "/init-blocker";
        std::ofstream(blocker).put('x');
        ExecutionRecorder failed(blocker, "test");
        assert(!failed.configure(0, 0, {}));
        assert(!failed.enabled());
        assert(!failed.needsConfiguration());
        assert(!failed.configure(0, 0, {}));
        assert(!failed.submitBatch("{}"));
    }
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
        ExecutionRecorder recorder(std::string(directory) + "/delayed", "ttft");
        assert(recorder.configure(0, 0, {}));
        auto make_batch = [&](int64_t execution) {
            auto record       = recorder.identity();
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
            batch->timings[0] = {1250};
            batch.reset();  // Dispatch ownership ends here, not on the caller.
        });
        // Drain earlier recorder work while dispatch is blocked. The pending
        // batch must not have been submitted just because forward finished.
        std::promise<void> drained;
        assert(recorder.submitBatch([&] {
            drained.set_value();
            return "{\"barrier\":true}";
        }));
        drained.get_future().wait();
        Logger::getBatchScheduleLogger()->flush();
        assert(readFile(path).find("\"exec_id\":2") == std::string::npos);
        release.set_value();
        worker.join();
        {
            auto decode        = make_batch(3);
            decode->timings[0] = {1250};
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
            const auto record = autil::legacy::AnyCast<ExecutionRecorder::Json>(autil::legacy::json::ParseJson(line));
            if (record.count("barrier"))
                continue;
            const auto requests = autil::legacy::AnyCast<ExecutionRecorder::JsonArray>(record.at("requests"));
            const auto row      = autil::legacy::AnyCast<ExecutionRecorder::Json>(requests.at(0));
            ++count;
            assert(line.find("\"q_len\":4") != std::string::npos);
            if (count == 1 || count == 4)
                assert(row.count("ttft_us") == 0);
            else
                assert(line.find("\"ttft_us\":1250") != std::string::npos);
        }
        assert(count == 4);
    }
    {
        ExecutionRecorder recorder(std::string(directory) + "/normal", "test", 1024);
        assert(!recorder.enabled());
        assert(recorder.configure(2,
                                  1,
                                  {{"model_config", special},
                                   {"record_decode", false},
                                   {"model_fingerprint", ExecutionRecorder::JsonValue()}}));
        assert(recorder.enabled());
        RecordedRequest request(recorder);
        RecordedRequest other(recorder);
        assert(other.id() != request.id());
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
        assert(manifest.find("\"submitted_to_alog\":0") != std::string::npos);
        assert(manifest.find("\"written\":null") != std::string::npos);
        assert(manifest.find("\"complete\":false") != std::string::npos);
        assert(manifest.find("\"closed\":true") != std::string::npos);
    }
    {
        const auto path = std::string(directory) + "/overflow.log";
        configureBatch(path);
        ExecutionRecorder recorder(std::string(directory) + "/overflow", "test", 1);
        assert(recorder.configure(0, 0, {}));
        std::promise<void> entered, release;
        auto               gate = release.get_future().share();
        assert(recorder.submitBatch([&] {
            entered.set_value();
            gate.wait();
            return "{}";
        }));
        entered.get_future().wait();
        assert(recorder.submitBatch("{}"));
        assert(!recorder.submitBatch("{}"));
        release.set_value();
        recorder.close();
        assert(recorder.dropped() == 1);
        assert(readFile(path) == "{}\n{}\n");
    }
    {
        ExecutionRecorder disabled("", "test");
        assert(!disabled.enabled());
        assert(!disabled.needsConfiguration());
        assert(!disabled.configure(0, 0, {}));
        assert(!disabled.submitBatch("{}"));
    }
    {
        const auto batch_path = std::string(directory) + "/configured-batches.jsonl";
        configureBatch(batch_path);
        ExecutionRecorder recorder(std::string(directory) + "/single_file", "test");
        assert(recorder.configure(0, 0, {}));
        for (int i = 0; i < 128; ++i)
            assert(recorder.submitBatch("{\"exec_id\":" + std::to_string(i + 1) + "}"));
        recorder.close();
        const auto    owner_dir = std::string(directory) + "/single_file/owner-" + recorder.owner();
        std::ifstream input(batch_path);
        std::string   line;
        for (int i = 0; i < 128; ++i) {
            assert(std::getline(input, line));
            assert(line == "{\"exec_id\":" + std::to_string(i + 1) + "}");
        }
        assert(!std::getline(input, line));
        assert(!std::filesystem::exists(batch_path + ".1"));
        for (const auto& entry : std::filesystem::directory_iterator(owner_dir))
            assert(entry.path().filename() == "manifest.json");
    }
    {
        const auto path = std::string(directory) + "/configured-live.jsonl";
        configureBatch(path);
        ExecutionRecorder recorder(std::string(directory) + "/live", "test");
        assert(recorder.configure(0, 0, {}));
        assert(recorder.submitBatch("{\"live\":true}"));
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
        ExecutionRecorder recorder(std::string(directory) + "/large", "test");
        assert(recorder.configure(0, 0, {}));
        const auto line = "{\"payload\":\"" + std::string(3 * 1024 * 1024, 'x') + "\"}";
        assert(recorder.submitBatch(line));
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
        // A filtered record exceeds the entire budget but must not consume it
        // or even invoke the JSON producer. The next accepted record fits exactly.
        ExecutionRecorder recorder(std::string(directory) + "/filtered", "test", 8, 3);
        assert(recorder.configure(0, 0, {}));
        bool serialized = false;
        assert(recorder.submitBatch([&] {
            serialized = true;
            return "{\"filtered\":true}";
        }));
        assert(recorder.submitBatch("{}"));
        // Wait for both filtered tasks without using a separate file sink.
        for (int i = 0; i < 200 && recorder.dropped() < 2; ++i)
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        assert(recorder.dropped() == 2);
        assert(recorder.enabled());  // Filtering must not exhaust the budget.
        Logger::getBatchScheduleLogger()->flush();
        assert(readFile(path).empty());
        Logger::getBatchScheduleLogger()->setLevel(alog::LOG_LEVEL_INFO);
        assert(recorder.submitBatch("{}"));  // Still has the full three-byte budget.
        recorder.close();
        assert(!serialized);
        assert(readFile(path) == "{}\n");
        const auto manifest =
            readFile(std::string(directory) + "/filtered/owner-" + recorder.owner() + "/manifest.json");
        assert(manifest.find("\"generated\":3") != std::string::npos);
        assert(manifest.find("\"submitted_to_alog\":1") != std::string::npos);
        assert(manifest.find("\"bytes_submitted\":3") != std::string::npos);
        assert(manifest.find("\"dropped\":2") != std::string::npos);
        assert(manifest.find("\"errors\":0") != std::string::npos);
        RTP_LLM_BATCH_SCHEDULE_LOG(alog::LOG_LEVEL_ERROR, "{\"percent\":\"100%\"}");
        rtp_llm::Logger::getBatchScheduleLogger()->flush();
        assert(readFile(path) == "{}\n{\"percent\":\"100%\"}\n");
    }
    {
        const auto path = std::string(directory) + "/producer-error.log";
        configureBatch(path);
        ExecutionRecorder recorder(std::string(directory) + "/producer_error", "test");
        assert(recorder.configure(0, 0, {}));
        assert(recorder.submitBatch([]() -> std::string { throw std::runtime_error("snapshot failed"); }));
        recorder.close();
        const auto manifest =
            readFile(std::string(directory) + "/producer_error/owner-" + recorder.owner() + "/manifest.json");
        assert(manifest.find("\"errors\":1") != std::string::npos);
        assert(manifest.find("\"submitted_to_alog\":0") != std::string::npos);
        assert(readFile(path).empty());
    }
    {
        const auto path = std::string(directory) + "/oversized.log";
        configureBatch(path);
        ExecutionRecorder limited(std::string(directory) + "/budget", "test", 8, 4);
        assert(limited.configure(0, 0, {}));
        assert(limited.submitBatch("12345"));
        limited.close();
        assert(limited.dropped() == 1);
        assert(readFile(path).empty());
    }
    {
        const auto path = std::string(directory) + "/batch-budget.jsonl";
        configureBatch(path);
        ExecutionRecorder recorder(std::string(directory) + "/batch-budget", "test", 8, 3);
        assert(recorder.configure(0, 0, {}));
        assert(recorder.submitBatch("{}"));
        assert(recorder.submitBatch("{}"));
        recorder.close();
        assert(readFile(path) == "{}\n");
        const auto manifest =
            readFile(std::string(directory) + "/batch-budget/owner-" + recorder.owner() + "/manifest.json");
        assert(manifest.find("\"submitted_to_alog\":1") != std::string::npos);
        assert(manifest.find("\"bytes_submitted\":3") != std::string::npos);
        assert(manifest.find("\"dropped\":1") != std::string::npos);
        assert(manifest.find("\"errors\":1") != std::string::npos);
    }
    alog::Logger::getRootLogger()->flush();
    assert(readFile(root_path).empty());  // Recorder JSON never inherits the engine/root appender.
    {
        // Exercise the shipped config: batch records append to this process's file.
        const auto cwd     = std::filesystem::current_path();
        const auto config  = cwd / "rtp_llm/config/alog.conf";
        const auto workdir = std::filesystem::absolute(std::string(directory) + "/real_config");
        std::filesystem::create_directories(workdir);
        std::filesystem::current_path(workdir);
        alog::Configurator::configureLogger(config.c_str());
        RTP_LLM_BATCH_SCHEDULE_LOG_INFO("{\"actual_config\":true}");
        RTP_LLM_BATCH_SCHEDULE_LOG_INFO("{\"second_batch\":true}");
        rtp_llm::Logger::getBatchScheduleLogger()->flush();
        assert(readFile("logs/batch_schedule." + std::to_string(getpid()) + ".log")
               == "{\"actual_config\":true}\n{\"second_batch\":true}\n");
        assert(readFile("logs/engine.log").empty());
        std::filesystem::current_path(cwd);
    }
    {
        const auto path   = std::string(directory) + "/snapshot-warnings.log";
        const auto config = "alog.max_msg_len=0\nalog.logger.engine=WARN, snapshotWarnings\n"
                            "inherit.engine=false\nalog.logger.console=WARN, snapshotWarnings\n"
                            "inherit.console=false\nalog.appender.snapshotWarnings=FileAppender\n"
                            "alog.appender.snapshotWarnings.fileName="
                            + path
                            + "\n"
                              "alog.appender.snapshotWarnings.layout=PatternLayout\n"
                              "alog.appender.snapshotWarnings.layout.LogPattern=%%m\n";
        alog::Configurator::configureLoggerFromString(config.c_str());
        ExecutionRecorder recorder(std::string(directory) + "/snapshot-errors", "test");
        assert(recorder.configure(0, 0, {}));
        for (int i = 0; i < 2; ++i) {
            RecordedBatch batch(recorder, {}, {});
            batch.markTimingError();
            batch.markTimingError();  // Multiple bad slots in a batch count once.
        }
        std::vector<std::thread> threads;
        for (int i = 0; i < 16; ++i)
            threads.emplace_back([&] { recorder.reportSnapshotError("invalid lengths"); });
        for (auto& thread : threads)
            thread.join();
        recorder.close();
        alog::Logger::getLogger("engine")->flush();
        alog::Logger::getLogger("console")->flush();
        const auto warnings = readFile(path);
        const auto first    = warnings.find("batch recording snapshot failed:");
        assert(first != std::string::npos);
        assert(warnings.find("batch recording snapshot failed:", first + 1) == std::string::npos);
        const auto manifest =
            readFile(std::string(directory) + "/snapshot-errors/owner-" + recorder.owner() + "/manifest.json");
        assert(manifest.find("\"errors\":18") != std::string::npos);
    }
}
