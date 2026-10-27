#include "rtp_llm/cpp/utils/ExecutionStageRecorder.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <dirent.h>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <sys/syscall.h>
#include <unistd.h>

using rtp_llm::ExecutionStageRecorder;
using rtp_llm::ExecutionStageScope;

namespace {
void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

size_t threadCount() {
    auto* directory = ::opendir("/proc/self/task");
    require(directory != nullptr, "cannot inspect Linux thread inventory");
    size_t count = 0;
    while (const auto* entry = ::readdir(directory)) {
        count += entry->d_name[0] != '.';
    }
    ::closedir(directory);
    return count;
}

void testDisabled(const std::string& prefix) {
    ExecutionStageRecorder::Config config;
    config.stall_timeout_ms = 1;
    config.file_prefix      = prefix;
    const auto file         = prefix + "." + std::to_string(getpid()) + ".jsonl";
    require(::access(file.c_str(), F_OK) != 0, "disabled fixture path already exists");
    const auto threads_before = threadCount();
    {
        ExecutionStageRecorder recorder(config);
        require(threadCount() == threads_before, "disabled recorder started a monitor thread");
        require(recorder.begin("copy", "", 3608368576110075969LL) == 0, "disabled recorder created an event");
        ExecutionStageScope scope(recorder, "disabled_scope");
        scope.annotate("unused");
        scope.finish("completed");
        require(scope.id() == 0 && scope.parentId() == 0, "disabled scope created an identity");
        recorder.setRank(3);
        recorder.dump("disabled_manual_dump");
        const auto data = recorder.snapshot();
        require(data.history.empty() && data.active.empty() && data.dropped_active == 0,
                "disabled recorder retained events or changed overflow counters");
        require(threadCount() == threads_before, "disabled operations started a monitor thread");
        require(::access(file.c_str(), F_OK) != 0, "disabled recorder wrote a trace file");
    }
    require(threadCount() == threads_before, "disabled recorder leaked a thread");
    require(::access(file.c_str(), F_OK) != 0, "disabled destruction wrote a trace file");
}

void testScopeEnrichmentAndNesting() {
    ExecutionStageRecorder::Config config;
    config.enabled          = true;
    config.stall_timeout_ms = 60000;
    ExecutionStageRecorder recorder(config);
    uint64_t               parent_id;
    {
        ExecutionStageScope parent(recorder, "prepare", {}, 3608368576110075969LL);
        parent_id        = parent.id();
        const auto start = recorder.snapshot().active.front().monotonic_us;
        const auto identity = recorder.snapshot().active.front();
        require(identity.thread_id == ::syscall(SYS_gettid), "event lost its actual Linux thread identity");
        require(identity.thread_start_ticks > 0, "event lost its Linux task start identity");
        parent.annotate("stream=unobserved device=0");
        auto data = recorder.snapshot();
        require(data.active.front().monotonic_us == start, "annotation changed operation start time");
        require(data.active.front().details == "stream=unobserved device=0", "active scope lost annotation");
        {
            ExecutionStageScope child(recorder, "launch");
            require(child.parentId() == parent_id, "scope did not expose its resolved parent identity");
            data = recorder.snapshot();
            require(data.active.size() == 2 && data.active.back().parent_id == parent_id,
                    "nested host scopes lost parent identity");
            require(data.active.back().request_id == 3608368576110075969LL,
                    "nested host scope did not inherit its 64-bit request identity");
            child.finish("host_launch_returned");
            child.finish();
        }
        data = recorder.snapshot();
        require(data.active.size() == 1, "explicit scope completion leaked an active operation");
        require(data.history.back().phase == "host_launch_returned", "scope overwrote explicit phase");
    }
    const auto data = recorder.snapshot();
    require(data.active.empty(), "scope destructor did not end operation");
    require(data.history.back().id == parent_id && data.history.back().phase == "returned",
            "scope destructor did not record host return");
    require(data.history.back().request_id == 3608368576110075969LL, "scope lost 64-bit request identity");
}

void testRequestIdentityInheritanceAndIsolation() {
    ExecutionStageRecorder::Config config;
    config.enabled          = true;
    config.stall_timeout_ms = 60000;
    ExecutionStageRecorder recorder(config);
    ExecutionStageRecorder independent(config);
    ExecutionStageScope    parent(recorder, "request", {}, 3608368576110075969LL);
    {
        ExecutionStageScope override(recorder, "explicit_request", {}, 3608856155824222323LL);
        ExecutionStageScope child(recorder, "copy");
        const auto          active = recorder.snapshot().active;
        require(active.back().request_id == 3608856155824222323LL,
                "child did not inherit its nearest explicit request identity");
        require(active.back().parent_id == override.id(), "child skipped its immediate parent");
        ExecutionStageScope isolated(independent, "other_recorder");
        const auto          other = independent.snapshot().active;
        require(other.size() == 1 && other.front().parent_id == 0 && other.front().request_id == 0,
                "request identity leaked between recorder instances");
    }
    const auto active = recorder.snapshot().active;
    require(active.size() == 1 && active.front().request_id == 3608368576110075969LL,
            "ending an overridden child changed the parent's request identity");
}

void testScopeExceptionAndOutOfOrderExit() {
    ExecutionStageRecorder::Config config;
    config.enabled          = true;
    config.stall_timeout_ms = 60000;
    ExecutionStageRecorder recorder(config);
    try {
        ExecutionStageScope scope(recorder, "throwing");
        throw std::runtime_error("expected");
    } catch (const std::runtime_error&) {}
    require(recorder.snapshot().history.back().phase == "exception", "scope lost exception outcome");
    {
        ExecutionStageScope parent(recorder, "parent");
        ExecutionStageScope child(recorder, "child");
        parent.finish();
        child.finish();
        ExecutionStageScope next(recorder, "next");
        require(recorder.snapshot().active.back().parent_id == 0, "TLS restored an ended parent");
    }
}

void testScopeCrossThreadCorrelation() {
    ExecutionStageRecorder::Config config;
    config.enabled          = true;
    config.stall_timeout_ms = 60000;
    ExecutionStageRecorder recorder(config);
    ExecutionStageScope    producer(recorder, "producer", {}, 3608856155824222323LL);
    std::thread            worker([&] {
        ExecutionStageScope isolated(recorder, "isolated");
        require(recorder.snapshot().active.back().parent_id == 0, "implicit parent crossed worker threads");
        ExecutionStageScope consumer(recorder, "consumer", "sequence=42", 0, producer.id());
        const auto          active = recorder.snapshot().active;
        require(active.back().parent_id == producer.id(), "explicit cross-thread parent was lost");
        require(active.back().request_id == 3608856155824222323LL,
                "explicit cross-thread parent did not carry its request identity");
        require(active.back().thread_id != active.front().thread_id, "worker thread identity was lost");
        require(active.back().thread_id == ::syscall(SYS_gettid) && active.back().thread_start_ticks > 0,
                "worker event lost its native thread identity or start ticks");
        consumer.finish("completed");  // Only an already-observed completion may use this phase.
    });
    worker.join();
    require(recorder.snapshot().active.size() == 1, "worker scopes leaked active operations");
}

void testScopeEndedOnAnotherThread() {
    ExecutionStageRecorder::Config config;
    config.enabled          = true;
    config.stall_timeout_ms = 60000;
    ExecutionStageRecorder recorder(config);
    ExecutionStageScope    handed_off(recorder, "handed_off");
    std::thread            worker([&] { handed_off.finish(); });
    worker.join();
    ExecutionStageScope next(recorder, "next");
    require(recorder.snapshot().active.back().parent_id == 0, "cross-thread exit left a stale TLS parent");
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
    for (int i = 0; i < 3; ++i) {
        recorder.end(recorder.begin("overflow", ""));
    }
    auto data = recorder.snapshot();
    require(data.active.size() == 2 && data.dropped_active == 3, "active overflow was not bounded or counted");
    require(data.active.front().id == first && data.active.back().id == second,
            "active overflow evicted an older blocked operation");
    recorder.end(first);
    const auto replacement = recorder.begin("replacement", "");
    data = recorder.snapshot();
    require(data.active.size() == 2 && data.active.front().id == second && data.active.back().id == replacement,
            "active slot was not reusable after an observed completion");
    require(data.dropped_active == 3, "successful retention reset or incremented the overflow counter");
    recorder.end(second);
    recorder.end(replacement);
    require(recorder.snapshot().active.empty(), "overflow recovery leaked active operations");
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
        const char* temp = std::getenv("TEST_TMPDIR");
        const auto prefix = std::string(temp ? temp : "/tmp") + "/execution_stage_recorder_test_"
                            + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
        testDisabled(prefix + "_disabled");
        testScopeEnrichmentAndNesting();
        testRequestIdentityInheritanceAndIsolation();
        testScopeExceptionAndOutOfOrderExit();
        testScopeCrossThreadCorrelation();
        testScopeEndedOnAnotherThread();
        testBoundedHistoryRetainsBlockedOperation();
        testActiveOverflowIsBoundedAndReported();
        testConcurrentWriters();
        testMonitorDumpsWithoutWorkerProgress(prefix);
        std::puts("ExecutionStageRecorder: 10 tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
