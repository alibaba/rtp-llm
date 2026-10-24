#include "rtp_llm/cpp/utils/ExecutionStageRecorder.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <unistd.h>

using rtp_llm::ExecutionStageRecorder;

namespace {
void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void testDisabled() {
    ExecutionStageRecorder recorder({});
    require(recorder.begin("copy", "") == 0, "disabled recorder created an event");
    require(recorder.snapshot().history.empty(), "disabled recorder retained history");
}

void testBoundedHistoryRetainsBlockedOperation() {
    ExecutionStageRecorder::Config config;
    config.enabled          = true;
    config.capacity         = 4;
    config.stall_timeout_ms = 60000;
    ExecutionStageRecorder recorder(config);
    auto                   blocked = recorder.begin("blocked", "device=0 stream=0", 123);
    for (int i = 0; i < 20; ++i) {
        recorder.end(recorder.begin("finished", "", 456, blocked));
    }
    auto data = recorder.snapshot();
    require(data.history.size() == 4, "history exceeded capacity");
    require(data.active.size() == 1 && data.active[0].id == blocked, "lost blocked operation when history wrapped");
    require(data.active[0].request_id == 123, "lost request identity");
    require(data.history.back().parent_id == blocked, "lost parent stage");
    recorder.end(blocked, "exception");
    data = recorder.snapshot();
    require(data.active.empty(), "completed operation is still active");
    require(data.history.back().phase == "exception", "lost exception outcome");
}

void testActiveOverflowIsBoundedAndReported() {
    ExecutionStageRecorder::Config config;
    config.enabled          = true;
    config.capacity         = 2;
    config.stall_timeout_ms = 60000;
    ExecutionStageRecorder recorder(config);
    auto                   first  = recorder.begin("oldest", "");
    auto                   second = recorder.begin("second", "");
    recorder.end(recorder.begin("overflow", ""));
    auto data = recorder.snapshot();
    require(data.active.size() == 2 && data.dropped_active == 1, "active overflow was not bounded or counted");
    require(data.active.front().id == first, "active overflow evicted oldest blocked operation");
    recorder.end(first);
    recorder.end(second);
}

void testConcurrentWriters() {
    ExecutionStageRecorder::Config config;
    config.enabled          = true;
    config.capacity         = 64;
    config.stall_timeout_ms = 60000;
    ExecutionStageRecorder   recorder(config);
    std::vector<std::thread> threads;
    for (int i = 0; i < 8; ++i) {
        threads.emplace_back([&] {
            for (int j = 0; j < 1000; ++j) {
                auto id = recorder.begin("concurrent", "");
                recorder.snapshot();
                recorder.end(id);
            }
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }
    auto data = recorder.snapshot();
    require(data.active.empty(), "concurrent completion leaked active operations");
    require(data.history.size() == config.capacity, "concurrent history is not bounded");
    require(data.dropped_active == 0, "unexpected active operation loss");
}

void testMonitorDumpsWithoutWorkerProgress(const std::string& prefix) {
    ExecutionStageRecorder::Config config;
    config.enabled          = true;
    config.stall_timeout_ms = 20;
    config.file_prefix      = prefix;
    const auto file         = prefix + "." + std::to_string(getpid()) + ".jsonl";
    {
        ExecutionStageRecorder recorder(config);
        recorder.setRank(3);
        auto        id       = recorder.begin("blocked\"copy", "line\nslash\\tab\t", 789);
        const auto  deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
        std::string output;
        while (std::chrono::steady_clock::now() < deadline) {
            std::ifstream in(file);
            output.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
            if (output.find("stage_stalled") != std::string::npos) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        require(output.find("stage_stalled") != std::string::npos, "monitor failed to dump blocked worker");
        require(output.find("\"rank\":3") != std::string::npos, "dump lost rank");
        require(output.find("blocked\\\"copy") != std::string::npos, "dump did not escape JSON string");
        recorder.end(id);
        recorder.dump("manual");
    }
    // Keep this fixture for external JSON parsing when requested by the caller.
    if (!std::getenv("RTP_STAGE_TEST_KEEP_OUTPUT")) {
        std::remove(file.c_str());
    }
}
}  // namespace

int main() {
    try {
        testDisabled();
        testBoundedHistoryRetainsBlockedOperation();
        testActiveOverflowIsBoundedAndReported();
        testConcurrentWriters();
        const char* temp = std::getenv("TEST_TMPDIR");
        testMonitorDumpsWithoutWorkerProgress(std::string(temp ? temp : "/tmp") + "/execution_stage_recorder_test");
        std::puts("ExecutionStageRecorder: 5 tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
