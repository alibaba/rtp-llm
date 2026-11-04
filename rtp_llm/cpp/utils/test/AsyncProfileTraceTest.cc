#include "rtp_llm/cpp/utils/AsyncProfileTrace.h"
#include "autil/legacy/json.h"
#include <gtest/gtest.h>
#include <fstream>
#include <iterator>
#include <set>
#include <map>
#include <sys/stat.h>
#include <thread>

namespace rtp_llm {
namespace {
using autil::legacy::AnyCast;
using autil::legacy::json::JsonArray;
using autil::legacy::json::JsonMap;
using autil::legacy::json::ParseJson;
using autil::legacy::json::ToString;

JsonArray traceEvents(const std::string& text) {
    return AnyCast<JsonArray>(AnyCast<JsonMap>(ParseJson(text)).at("traceEvents"));
}

AsyncProfileEvent sample(AsyncProfileKind kind = AsyncProfileKind::Execution, uint64_t id = 1) {
    AsyncProfileEvent e{"cache.transfer.host_to_device.h2d(request_id=3621580797405763233)", 1000010, 1000030, {}};
    e.record.kind        = kind;
    e.record.stage       = "h2d";
    e.record.path        = "host_to_device";
    e.record.request_id  = 3621580797405763233;
    e.record.transfer_id = 57003;
    e.record.start_tid   = 123;
    e.record.scope       = "worker_batch";
    e.record.operation   = "load";
    e.pid                = 99;
    e.end_tid            = 123;
    e.session_id         = 7;
    e.span_id            = id;
    return e;
}

AsyncProfileEvent backgroundSample(AsyncProfileKind kind, uint64_t id = 1) {
    auto e              = sample(kind, id);
    e.record.request_id = -1;
    return e;
}

TEST(AsyncProfileTraceTest, NativeEventsMetadataPrecisionAndFlowsAreBytePreserved) {
    const std::string native        = R"({"schemaVersion":1,"traceEvents":[
{"ph":"X","cat":"cpu_op","name":"engine.normal.schedule","pid":99,"tid":42,"ts":1.234,"dur":2.001},
{"ph":"X","cat":"kernel","name":"kernel\"[x]","pid":0,"tid":7,"ts":3.333,"dur":4.444},
{"ph":"X","cat":"gpu_memcpy","name":"HtoD","ts":5.001,"dur":1},
{"ph":"X","cat":"cuda_runtime","name":"cudaMemcpyAsync","ts":2,"dur":1},
{"ph":"s","cat":"ac2g","id":9007199254740993,"pid":99,"tid":42,"ts":2},
{"ph":"f","cat":"ac2g","id":9007199254740993,"pid":0,"tid":7,"ts":3},
{"ph":"M","name":"thread_name","pid":99,"tid":42,"args":{"name":"engine"}}
],"baseTimeNanoseconds":1000000000,"traceName":"x","other":{"traceEvents":["escaped\\\"[]"]}})";
    auto              worker        = sample();
    worker.record.start_thread_name = "btc_load";
    const auto merged               = mergeAsyncProfileTrace(native, {worker});
    const auto original_events      = traceEvents(native);
    const auto events               = traceEvents(merged);
    ASSERT_EQ(events.size(), original_events.size() + 2);
    for (size_t i = 0; i < original_events.size(); ++i)
        EXPECT_EQ(ToString(events[i], true), ToString(original_events[i], true));
    // Only insertion is allowed; native text on both sides remains identical.
    auto close = native.find("],\"baseTimeNanoseconds\"");
    ASSERT_NE(close, std::string::npos);
    EXPECT_EQ(merged.substr(0, close), native.substr(0, close));
    EXPECT_EQ(merged.substr(close + merged.size() - native.size()), native.substr(close));
    auto e = AnyCast<JsonMap>(events.back());
    EXPECT_EQ(AnyCast<std::string>(e.at("ph")), "X");
    EXPECT_EQ(ToString(e.at("pid"), true), "99");
    EXPECT_EQ(ToString(e.at("tid"), true), "123");
    EXPECT_DOUBLE_EQ(std::stod(ToString(e.at("ts"), true)), 10);
    EXPECT_DOUBLE_EQ(std::stod(ToString(e.at("dur"), true)), 20);
    auto args = AnyCast<JsonMap>(e.at("args"));
    EXPECT_EQ(AnyCast<std::string>(args.at("request_id")), "3621580797405763233");
    EXPECT_EQ(AnyCast<std::string>(args.at("kind")), "execution");
}

TEST(AsyncProfileTraceTest, LogicalRangesShareTrackScopeAndRetainKindsAndUniqueSpanIds) {
    auto       total   = backgroundSample(AsyncProfileKind::Lifecycle, 1);
    auto       wait    = backgroundSample(AsyncProfileKind::Wait, 2);
    auto       derived = backgroundSample(AsyncProfileKind::Derived, 3);
    const auto events  = traceEvents(
        mergeAsyncProfileTrace("{\"traceEvents\":[],\"baseTimeNanoseconds\":1000000500}", {total, wait, derived}));
    ASSERT_EQ(events.size(), 6u);
    std::set<std::string> ids, scopes, kinds;
    for (size_t i = 0; i < events.size(); i += 2) {
        auto begin = AnyCast<JsonMap>(events[i]);
        auto end   = AnyCast<JsonMap>(events[i + 1]);
        EXPECT_EQ(AnyCast<std::string>(begin.at("ph")), "b");
        EXPECT_EQ(AnyCast<std::string>(end.at("ph")), "e");
        EXPECT_EQ(ToString(begin.at("tid"), true), "0");
        EXPECT_EQ(ToString(begin.at("id2"), true), ToString(end.at("id2"), true));
        EXPECT_DOUBLE_EQ(std::stod(ToString(begin.at("ts"), true)), 9.5);
        ids.insert(ToString(begin.at("id2"), true));
        scopes.insert(AnyCast<std::string>(begin.at("scope")));
        kinds.insert(AnyCast<std::string>(AnyCast<JsonMap>(begin.at("args")).at("kind")));
        EXPECT_EQ(AnyCast<std::string>(begin.at("name")), "KV background");
    }
    EXPECT_EQ(ids.size(), 3u);
    EXPECT_EQ(scopes, (std::set<std::string>{"cache.transfer.logical"}));
    EXPECT_EQ(kinds, (std::set<std::string>{"lifecycle", "wait", "derived"}));
}

TEST(AsyncProfileTraceTest, CrossThreadAndUnknownStartsCannotAppearAsThreadExecution) {
    auto crossed              = sample();
    crossed.end_tid           = 456;
    auto unknown              = sample(AsyncProfileKind::Execution, 2);
    unknown.record.start_tid  = 0;
    crossed.record.request_id = unknown.record.request_id = -1;
    auto events = traceEvents(mergeAsyncProfileTrace("{\"traceEvents\":[]}", {crossed, unknown}));
    ASSERT_EQ(events.size(), 4u);
    for (const auto& item : events) {
        auto e = AnyCast<JsonMap>(item);
        EXPECT_NE(AnyCast<std::string>(e.at("ph")), "X");
        EXPECT_EQ(AnyCast<std::string>(AnyCast<JsonMap>(e.at("args")).at("kind")), "lifecycle");
    }
}

TEST(AsyncProfileTraceTest, RequestLogicalRangesAreRemovedAndPhysicalRequestIdsRemain) {
    auto io                         = sample();
    auto zero                       = sample(AsyncProfileKind::Execution, 2);
    zero.record.request_id          = 0;
    auto total                      = sample(AsyncProfileKind::Lifecycle, 3);
    auto wait                       = sample(AsyncProfileKind::Wait, 4);
    auto request                    = total;
    request.record.path             = "request";
    request.record.stage            = "request_lifetime";
    auto unknown_thread             = io;
    unknown_thread.record.start_tid = 0;
    const auto events =
        traceEvents(mergeAsyncProfileTrace(R"({"traceEvents":[]})", {io, zero, total, wait, request, unknown_thread}));
    ASSERT_EQ(events.size(), 2u);
    EXPECT_EQ(AnyCast<std::string>(AnyCast<JsonMap>(events[0]).at("ph")), "X");
    const auto first  = AnyCast<JsonMap>(AnyCast<JsonMap>(events[0]).at("args"));
    const auto second = AnyCast<JsonMap>(AnyCast<JsonMap>(events[1]).at("args"));
    EXPECT_EQ(AnyCast<std::string>(first.at("request_id")), "3621580797405763233");
    EXPECT_EQ(AnyCast<std::string>(second.at("request_id")), "0");
    EXPECT_EQ(AnyCast<std::string>(first.at("kind")), "execution");
    EXPECT_TRUE(
        asyncProfileWorkflowTrace(R"({"traceEvents":[]})", {io, zero, total, wait, request, unknown_thread}).empty());
}

TEST(AsyncProfileTraceTest, WorkerNamesAreDeduplicatedByProcessAndThreadWithoutOverridingNativeNames) {
    const std::string native   = R"({"traceEvents":[
{"ph":"M","name":"thread_name","pid":"99","tid":"123","args":{"name":"native worker \"[x]"}},
{"ph":"M","name":"thread_name","pid":99,"tid":42,"args":{"name":"normal_engine"}},
{"ph":"M","name":"thread_name","pid":99,"tid":456,"args":{"name":""}},
{"ph":"X","name":"thread_name","pid":99,"tid":789,"args":{"name":"not metadata"},"ts":1,"dur":1}
]})";
    auto              existing = sample(), missing = sample(AsyncProfileKind::Execution, 2),
                      repeated        = sample(AsyncProfileKind::Execution, 3);
    existing.record.start_thread_name = "do not replace native worker";
    missing.record.start_tid = missing.end_tid = 456;
    missing.record.start_thread_name           = "btc_load";
    repeated                                   = missing;
    repeated.span_id                           = 3;
    auto other_pid                             = existing;
    other_pid.pid                              = 100;
    other_pid.span_id                          = 4;
    auto non_metadata                          = sample(AsyncProfileKind::Execution, 5);
    non_metadata.record.start_tid = non_metadata.end_tid = 789;
    non_metadata.record.start_thread_name                = "btc_task";
    const auto events =
        traceEvents(mergeAsyncProfileTrace(native, {existing, missing, repeated, other_pid, non_metadata}));
    ASSERT_EQ(events.size(), 4u + 3u + 5u);
    std::set<std::string> added;
    for (size_t i = 4; i < 7; ++i) {
        const auto event = AnyCast<JsonMap>(events[i]);
        EXPECT_EQ(AnyCast<std::string>(event.at("ph")), "M");
        EXPECT_EQ(AnyCast<std::string>(event.at("name")), "thread_name");
        added.insert(ToString(event.at("pid"), true) + ":" + ToString(event.at("tid"), true));
    }
    EXPECT_EQ(added, (std::set<std::string>{"99:456", "99:789", "100:123"}));
    const auto original = traceEvents(native);
    for (size_t i = 0; i < original.size(); ++i)
        EXPECT_EQ(ToString(events[i], true), ToString(original[i], true));
}

TEST(AsyncProfileTraceTest, LogicalOrUnknownExecutionCannotNameARealThread) {
    auto total = backgroundSample(AsyncProfileKind::Lifecycle, 1), wait = backgroundSample(AsyncProfileKind::Wait, 2);
    auto crossed = sample(AsyncProfileKind::Execution, 3), unknown = sample(AsyncProfileKind::Execution, 4);
    for (auto* event : {&total, &wait, &crossed, &unknown}) {
        event->record.request_id        = -1;
        event->record.start_thread_name = "must not become a thread track";
    }
    crossed.end_tid          = 456;
    unknown.record.start_tid = 0;
    const auto events = traceEvents(mergeAsyncProfileTrace("{\"traceEvents\":[]}", {total, wait, crossed, unknown}));
    ASSERT_EQ(events.size(), 8u);
    for (const auto& item : events) {
        const auto event = AnyCast<JsonMap>(item);
        EXPECT_NE(AnyCast<std::string>(event.at("ph")), "M");
        EXPECT_EQ(ToString(event.at("tid"), true), "0");
    }
}

TEST(AsyncProfileTraceTest, ThreadNamesSurviveWorkerExitAndJsonEscaping) {
    std::vector<AsyncProfileEvent> captured;
    auto                           session = AsyncProfileSession::start();
    std::thread                    worker([] {
#if defined(__linux__)
        ASSERT_EQ(pthread_setname_np(pthread_self(), "btc_\"load"), 0);
#elif defined(__APPLE__)
        ASSERT_EQ(pthread_setname_np("btc_\"load"), 0);
#endif
        AsyncProfileSpan span("disk_read", 42, 0, 1);
        span.finish();
    });
    worker.join();
    captured = session->finish();
    ASSERT_EQ(captured.size(), 1u);
    session.reset();
    const auto events = traceEvents(mergeAsyncProfileTrace("{\"traceEvents\":[]}", captured));
    ASSERT_EQ(events.size(), 2u);
    const auto metadata = AnyCast<JsonMap>(events.front());
    EXPECT_EQ(AnyCast<std::string>(AnyCast<JsonMap>(metadata.at("args")).at("name")), "btc_\"load");
    EXPECT_EQ(ToString(metadata.at("tid"), true), std::to_string(captured[0].record.start_tid));
}

TEST(AsyncProfileTraceTest, CompactLogicalLabelsRetainKindsGranularityAndOriginalMetadata) {
    auto total = backgroundSample(AsyncProfileKind::Lifecycle, 1), wait = backgroundSample(AsyncProfileKind::Wait, 2);
    total.record.path = wait.record.path = "mixed_to_device";
    total.record.stage                   = "completion_total";
    wait.record.stage                    = "completion_queue_wait";
    auto task                            = backgroundSample(AsyncProfileKind::Lifecycle, 3),
         batch                           = backgroundSample(AsyncProfileKind::Lifecycle, 4);
    task.record.stage                    = "worker_task_total";
    batch.record.stage                   = "worker_batch_total";
    const auto events = traceEvents(mergeAsyncProfileTrace("{\"traceEvents\":[]}", {total, wait, task, batch}));
    ASSERT_EQ(events.size(), 8u);
    const std::vector<std::string> labels(4, "KV background");
    for (size_t i = 0; i < labels.size(); ++i) {
        const auto begin = AnyCast<JsonMap>(events[2 * i]), end = AnyCast<JsonMap>(events[2 * i + 1]);
        EXPECT_EQ(AnyCast<std::string>(begin.at("name")), labels[i]);
        EXPECT_EQ(AnyCast<std::string>(end.at("name")), labels[i]);
        EXPECT_EQ(ToString(begin.at("id2"), true), ToString(end.at("id2"), true));
        EXPECT_EQ(ToString(begin.at("args"), true), ToString(end.at("args"), true));
    }
    const auto args = AnyCast<JsonMap>(AnyCast<JsonMap>(events[0]).at("args"));
    EXPECT_EQ(AnyCast<std::string>(args.at("path")), "mixed_to_device");
    EXPECT_EQ(AnyCast<std::string>(args.at("stage")), "completion_total");
    EXPECT_EQ(AnyCast<std::string>(args.at("original_name")), total.name);
}

TEST(AsyncProfileTraceTest, SnapshotOwnsMetadataAfterSessionAndSourceNamesAreDestroyed) {
    std::vector<AsyncProfileEvent> events;
    {
        std::string stage = "callback", path = "host_to_device", scope = "business", operation = "load";
        auto        session = AsyncProfileSession::start();
        auto        span    = AsyncProfileSpan::fromSteady(stage.c_str(),
                                                           42,
                                                           0,
                                                           8,
                                                           AsyncProfileSession::steadyNowUs(),
                                                           -1,
                                                           77,
                                                           path.c_str(),
                                                           scope.c_str(),
                                                           operation.c_str(),
                                                           AsyncProfileKind::Execution);
        span.finish();
        events = session->finish();
        stage.assign(1000, 's');
        path.assign(1000, 'p');
        scope.assign(1000, 'c');
        operation.assign(1000, 'o');
    }
    ASSERT_EQ(events.size(), 1u);
    // Exercise snapshot moves as well as session/source destruction (including SSO).
    auto       moved = std::move(events);
    const auto saved = traceEvents(mergeAsyncProfileTrace("{\"traceEvents\":[]}", moved));
    ASSERT_EQ(saved.size(), 2u);
    const auto event = AnyCast<JsonMap>(saved.back());
    EXPECT_EQ(AnyCast<std::string>(event.at("name")), moved.front().name);
    const auto args = AnyCast<JsonMap>(event.at("args"));
    EXPECT_EQ(AnyCast<std::string>(args.at("stage")), "callback");
    EXPECT_EQ(AnyCast<std::string>(args.at("scope")), "business");
    EXPECT_EQ(AnyCast<std::string>(args.at("operation")), "load");
}

TEST(AsyncProfileTraceTest, OverlappingBatchesZeroDurationAndReusedSessionSlotsPairIndependently) {
    AsyncProfileSession         first(100, 3);
    AsyncProfileSession::Record r;
    r.kind        = AsyncProfileKind::Wait;
    r.stage       = "queue_wait";
    r.scope       = "worker_batch";
    r.transfer_id = 77;
    r.request_id  = -1;
    auto add      = [&](AsyncProfileSession& session, size_t batch, int64_t begin, int64_t end) {
        r.begin     = batch;
        r.end       = batch + 8;
        r.start_us  = begin;
        auto* entry = session.add(r);
        EXPECT_NE(entry, nullptr);
        if (entry)
            entry->end_us.store(end, std::memory_order_release);
    };
    add(first, 0, 110, 150);
    add(first, 8, 120, 160);
    add(first, 16, 130, 130);
    auto                events = first.finish(200);
    AsyncProfileSession second(100, 1);
    add(second, 0, 110, 150);
    auto next = second.finish(200);
    ASSERT_EQ(events.size(), 3u);
    ASSERT_EQ(next.size(), 1u);
    EXPECT_EQ(events.front().span_id, next.front().span_id);
    EXPECT_NE(events.front().session_id, next.front().session_id);
    events.push_back(std::move(next.front()));
    const auto saved = traceEvents(mergeAsyncProfileTrace("{\"traceEvents\":[]}", events));
    ASSERT_EQ(saved.size(), 8u);
    std::set<std::string> ids;
    for (size_t i = 0; i < events.size(); ++i) {
        const auto begin = AnyCast<JsonMap>(saved[2 * i]);
        const auto end   = AnyCast<JsonMap>(saved[2 * i + 1]);
        EXPECT_EQ(AnyCast<std::string>(begin.at("ph")), "b");
        EXPECT_EQ(AnyCast<std::string>(end.at("ph")), "e");
        EXPECT_EQ(ToString(begin.at("id2"), true), ToString(end.at("id2"), true));
        EXPECT_EQ(ToString(begin.at("scope"), true), ToString(end.at("scope"), true));
        EXPECT_EQ(ToString(begin.at("name"), true), ToString(end.at("name"), true));
        EXPECT_DOUBLE_EQ(std::stod(ToString(begin.at("ts"), true)), events[i].start_us);
        EXPECT_DOUBLE_EQ(std::stod(ToString(end.at("ts"), true)), events[i].end_us);
        EXPECT_EQ(ToString(AnyCast<JsonMap>(begin.at("args")).at("batch_begin"), true),
                  std::to_string(events[i].record.begin));
        ids.insert(ToString(begin.at("id2"), true));
    }
    EXPECT_EQ(ids.size(), 4u);
}

TEST(AsyncProfileTraceTest, SerializationStreamsPropagateBufferFailures) {
    class RejectingBuffer: public std::streambuf {
        std::streamsize xsputn(const char*, std::streamsize) override {
            return 0;
        }
        int_type overflow(int_type) override {
            return traits_type::eof();
        }
    } buffer;
    auto output = detail::asyncProfileOutputStream();
    static_cast<std::ostream&>(output).rdbuf(&buffer);
    EXPECT_THROW(output << "incomplete JSON", std::ios_base::failure);
    EXPECT_TRUE(output.bad());
}

TEST(AsyncProfileTraceTest, ClippingFlagsAndEscapedNamesRoundTrip) {
    auto e          = sample();
    e.name          = "quote\" backslash\\ newline\n tab\t";
    e.start_clipped = true;
    e.end_clipped   = true;
    e.end_tid       = 0;
    auto events     = traceEvents(mergeAsyncProfileTrace("{\"traceEvents\": [ ]}", {e}));
    ASSERT_EQ(events.size(), 1u);
    auto event = AnyCast<JsonMap>(events.front());
    EXPECT_EQ(AnyCast<std::string>(event.at("name")), e.name);
    auto args = AnyCast<JsonMap>(event.at("args"));
    EXPECT_TRUE(AnyCast<bool>(args.at("start_clipped")));
    EXPECT_TRUE(AnyCast<bool>(args.at("end_clipped")));
    EXPECT_TRUE(AnyCast<bool>(args.at("incomplete")));
}

TEST(AsyncProfileTraceTest, EmptyCustomCaptureIsAnExactNoOp) {
    const std::string native = "{\"traceEvents\":[{\"id\":9007199254740993,\"ts\":1.234567}]}";
    EXPECT_EQ(mergeAsyncProfileTrace(native, {}), native);
    EXPECT_NO_THROW(appendAsyncProfileTrace("/does/not/exist", {}));
}

TEST(AsyncProfileTraceTest, SuccessfulAppendPreservesNativeFilePermissions) {
    char path[] = "/tmp/async-profile-mode-test-XXXXXX";
    auto fd     = mkstemp(path);
    ASSERT_GE(fd, 0);
    close(fd);
    ASSERT_EQ(chmod(path, 0640), 0);
    const std::string native = "{\"traceEvents\":[{\"name\":\"native\",\"ph\":\"X\",\"ts\":1,\"dur\":1}]}";
    {
        std::ofstream out(path);
        out << native;
    }
    ASSERT_NO_THROW(appendAsyncProfileTrace(path, {sample()}));
    struct stat saved_stat;
    ASSERT_EQ(stat(path, &saved_stat), 0);
    EXPECT_EQ(saved_stat.st_mode & 0777, 0640u);
    std::ifstream     in(path);
    const std::string saved{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
    EXPECT_EQ(traceEvents(saved).size(), 2u);
    EXPECT_EQ(saved.substr(0, native.find(']')), native.substr(0, native.find(']')));
    std::remove(path);
}

TEST(AsyncProfileTraceTest, StartClippingAloneIsMarkedIncomplete) {
    auto e            = sample();
    e.start_clipped   = true;
    const auto events = traceEvents(mergeAsyncProfileTrace("{\"traceEvents\":[]}", {e}));
    const auto args   = AnyCast<JsonMap>(AnyCast<JsonMap>(events.front()).at("args"));
    EXPECT_TRUE(AnyCast<bool>(args.at("incomplete")));
    EXPECT_FALSE(AnyCast<bool>(args.at("end_clipped")));
}

TEST(AsyncProfileTraceTest, UnsupportedExportRetainsOriginalFile) {
    char path[] = "/tmp/async-profile-export-test-XXXXXX";
    auto fd     = mkstemp(path);
    ASSERT_GE(fd, 0);
    close(fd);
    const std::string native = "{\"traceEvents\":[],\"baseTimeNanoseconds\":\"unknown\"}";
    {
        std::ofstream out(path);
        out << native;
    }
    EXPECT_THROW(appendAsyncProfileTrace(path, {sample()}), std::exception);
    std::ifstream     in(path);
    const std::string saved{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
    EXPECT_EQ(saved, native);
    std::remove(path);
    EXPECT_THROW(mergeAsyncProfileTrace("{}", {sample()}), std::exception);
    EXPECT_THROW(mergeAsyncProfileTrace("{\"traceEvents\":[", {sample()}), std::exception);
    EXPECT_THROW(mergeAsyncProfileTrace("{\"traceEvents\":[],\"baseTimeNanoseconds\":9999999999999999999}", {sample()}),
                 std::exception);
}

}  // namespace

TEST(AsyncProfileTraceTest, WorkflowResolutionUsesExplicitLinksAndProcessNamespace) {
    auto stage                     = sample();
    auto unrelated                 = stage;
    unrelated.pid                  = stage.pid + 1;
    auto link                      = stage;
    link.record.stage              = "task_link";
    link.record.workflow_id        = 17;
    link.record.parent_transfer_id = 15;
    const auto resolved            = detail::resolveAsyncProfileLinks({stage, unrelated, link});
    ASSERT_EQ(resolved.size(), 3u);
    EXPECT_EQ(resolved[0].record.workflow_id, 17u);
    EXPECT_EQ(resolved[0].record.parent_transfer_id, 15u);
    EXPECT_EQ(resolved[1].record.workflow_id, unrelated.record.transfer_id);
    auto json = traceEvents(mergeAsyncProfileTrace(R"({"traceEvents":[]})", resolved));
    auto args = AnyCast<JsonMap>(AnyCast<JsonMap>(json[0]).at("args"));
    EXPECT_EQ(AnyCast<std::string>(args.at("workflow_id")), "17");
    EXPECT_EQ(AnyCast<std::string>(args.at("parent_transfer_id")), "15");
}

TEST(AsyncProfileTraceTest, BundleContainsUnchangedNativeGpuFlowsAndPhysicalWorkerEvents) {
    const std::string native =
        R"({"baseTimeNanoseconds":1000000000,"traceEvents":[{"ph":"X","name":"kernel","cat":"kernel","pid":3,"tid":7,"ts":1.125,"dur":2.875},{"ph":"s","name":"flow","cat":"ac2g","pid":3,"tid":7,"ts":1.125,"id":9007199254740993}]})";
    auto e                     = sample();
    e.record.start_thread_name = "disk_worker";
    auto wait                  = backgroundSample(AsyncProfileKind::Wait, 2);
    wait.record.stage          = "queue_wait";
    wait.start_us              = 1000001;
    wait.end_us                = 1000009;
    const std::string path     = ::testing::TempDir() + "request_bundle_" + std::to_string(getpid()) + ".zip";
    writeAsyncProfileBundle(path, native, {e, wait});
    std::ifstream     in(path, std::ios::binary);
    const std::string archive{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
    EXPECT_EQ(archive.substr(0, 4), std::string("PK\x03\x04", 4));
    EXPECT_NE(archive.find("timeline.json"), std::string::npos);
    EXPECT_NE(archive.find("requests.pftrace"), std::string::npos);
    EXPECT_NE(archive.find(mergeAsyncProfileTrace(native, {e})), std::string::npos);
    EXPECT_NE(archive.find("KV background"), std::string::npos);
    EXPECT_EQ(archive.find("KV requests"), std::string::npos);
    EXPECT_EQ(archive.find("Request " + std::to_string(e.record.request_id)), std::string::npos);
    EXPECT_NE(archive.find("queue_wait [wait]"), std::string::npos);
    std::remove(path.c_str());
}

TEST(AsyncProfileTraceTest, WorkflowProtoRetainsLabelsClippingAndBackgroundSeparation) {
    auto a               = backgroundSample(AsyncProfileKind::Lifecycle);
    a.record.stage       = "completion_total";
    a.record.workflow_id = 10;
    auto b               = a;
    b.span_id            = 2;
    b.start_us += 10;
    b.end_us += 10;
    b.record.kind                 = AsyncProfileKind::Wait;
    b.record.stage                = "queue_wait";
    auto background               = a;
    background.record.request_id  = -1;
    background.record.operation   = "evict";
    background.record.workflow_id = 11;
    background.span_id            = 3;
    background.end_clipped        = true;
    const auto proto              = asyncProfileWorkflowTrace(R"({"traceEvents":[]})", {a, b, background});
    EXPECT_NE(proto.find("KV background"), std::string::npos);
    EXPECT_EQ(proto.find("evict / workflow 11"), std::string::npos);
    EXPECT_NE(proto.find("completion_total [lifetime]"), std::string::npos);
    EXPECT_NE(proto.find("queue_wait [wait]"), std::string::npos);
    EXPECT_EQ(proto.find("overlap 1"), std::string::npos);
    EXPECT_NE(proto.find("end_clipped"), std::string::npos);
    EXPECT_THROW(asyncProfileWorkflowTrace(R"({"baseTimeNanoseconds":9000000000,"traceEvents":[]})", {a}),
                 std::runtime_error);
}

namespace {
struct WireField {
    unsigned         tag;
    uint64_t         value;
    std::string_view data;
};
std::vector<WireField> decodeWire(std::string_view input) {
    size_t pos  = 0;
    auto   read = [&]() {
        uint64_t value = 0;
        for (unsigned shift = 0; shift < 64 && pos < input.size(); shift += 7) {
            const auto byte = static_cast<uint8_t>(input[pos++]);
            value |= uint64_t(byte & 127) << shift;
            if (!(byte & 128))
                return value;
        }
        throw std::runtime_error("malformed test protobuf");
    };
    std::vector<WireField> fields;
    while (pos < input.size()) {
        auto      key = read();
        WireField field{static_cast<unsigned>(key >> 3), 0, {}};
        switch (key & 7) {
            case 0:
                field.value = read();
                break;
            case 1:
                if (pos + 8 > input.size())
                    throw std::runtime_error("truncated fixed64");
                pos += 8;
                break;
            case 2: {
                const auto size = read();
                if (size > input.size() - pos)
                    throw std::runtime_error("truncated message");
                field.data = input.substr(pos, size);
                pos += size;
                break;
            }
            default:
                throw std::runtime_error("unsupported test wire type");
        }
        fields.push_back(field);
    }
    return fields;
}
}  // namespace

TEST(AsyncProfileTraceTest, PackedWorkflowKeepsExactPairsAndSharesSequentialKinds) {
    auto total               = backgroundSample(AsyncProfileKind::Lifecycle, 1);
    total.start_us           = 1000010;
    total.end_us             = 1000030;
    total.record.workflow_id = 9;
    total.record.stage       = "total";
    auto crossing            = total;
    crossing.span_id         = 2;
    crossing.start_us        = 1000020;
    crossing.end_us          = 1000040;
    crossing.record.kind     = AsyncProfileKind::Wait;
    crossing.record.stage    = "crossing_wait";
    auto next                = total;
    next.span_id             = 3;
    next.start_us            = 1000040;
    next.end_us              = 1000050;
    next.record.kind         = AsyncProfileKind::Execution;
    next.record.stage        = "callback";
    const auto proto =
        asyncProfileWorkflowTrace(R"({"baseTimeNanoseconds":1000000000,"traceEvents":[]})", {total, crossing, next});
    std::map<uint64_t, std::vector<std::pair<uint64_t, std::string>>> stacks;
    std::map<std::string, std::pair<uint64_t, uint64_t>>              intervals;
    std::map<std::string, uint64_t>                                   tracks;
    for (const auto& packet : decodeWire(proto)) {
        ASSERT_EQ(packet.tag, 1u);
        uint64_t timestamp = 0;
        for (const auto& field : decodeWire(packet.data)) {
            if (field.tag == 8)
                timestamp = field.value;
            if (field.tag == 60) {
                for (const auto& descriptor : decodeWire(field.data)) {
                    EXPECT_NE(descriptor.tag, 3u);  // no ProcessDescriptor
                    EXPECT_NE(descriptor.tag, 4u);  // no ThreadDescriptor
                    EXPECT_NE(descriptor.tag, 5u);  // no child tracks
                    if (descriptor.tag == 2) {
                        EXPECT_EQ(descriptor.data, "KV background");
                    }
                    if (descriptor.tag == 15) {
                        EXPECT_EQ(descriptor.value, 1u);
                    }  // merge sibling lanes
                }
            }
            if (field.tag != 11)
                continue;
            uint64_t    track = 0, type = 0;
            std::string name;
            for (const auto& event : decodeWire(field.data)) {
                if (event.tag == 11)
                    track = event.value;
                if (event.tag == 9)
                    type = event.value;
                if (event.tag == 23)
                    name = std::string(event.data);
            }
            if (type == 1) {
                stacks[track].push_back({timestamp, name});
                tracks[name] = track;
            } else if (type == 2) {
                ASSERT_FALSE(stacks[track].empty());
                const auto start = stacks[track].back();
                stacks[track].pop_back();
                EXPECT_GE(timestamp, start.first);
                intervals[start.second] = {start.first, timestamp};
            }
        }
    }
    for (const auto& stack : stacks)
        EXPECT_TRUE(stack.second.empty());
    ASSERT_EQ(intervals.size(), 3u);
    EXPECT_EQ(intervals["total [lifetime] / host_to_device"], std::make_pair(uint64_t(10000), uint64_t(30000)));
    EXPECT_EQ(intervals["crossing_wait [wait] / host_to_device"], std::make_pair(uint64_t(20000), uint64_t(40000)));
    EXPECT_EQ(intervals["callback [execution] / host_to_device"], std::make_pair(uint64_t(40000), uint64_t(50000)));
    EXPECT_EQ(tracks["total [lifetime] / host_to_device"], tracks["callback [execution] / host_to_device"]);
    EXPECT_NE(tracks["total [lifetime] / host_to_device"], tracks["crossing_wait [wait] / host_to_device"]);
}

TEST(AsyncProfileTraceTest, ManyRequestsAndStagesDoNotProduceLogicalTracks) {
    std::vector<AsyncProfileEvent> events;
    for (uint64_t i = 0; i < 200; ++i) {
        auto e               = sample(i % 2 ? AsyncProfileKind::Wait : AsyncProfileKind::Lifecycle, i + 1);
        e.record.request_id  = i;
        e.record.workflow_id = i + 1000;
        e.record.stage       = i % 2 ? "queue_wait" : "completion_total";
        e.start_us += i * 30;
        e.end_us += i * 30;
        events.push_back(e);
    }
    const std::string native = R"({"traceEvents":[]})";
    EXPECT_TRUE(asyncProfileWorkflowTrace(native, events).empty());
    EXPECT_EQ(mergeAsyncProfileTrace(native, events), native);
}

TEST(AsyncProfileTraceTest, RequestLifetimesAreHiddenIncludingUnknownRequestIds) {
    auto first                = sample(AsyncProfileKind::Lifecycle, 1);
    first.record.path         = "request";
    first.record.stage        = "request_lifetime";
    first.start_us            = 1000010;
    first.end_us              = 1000100;
    auto unknown              = first;
    unknown.record.request_id = -1;
    unknown.span_id           = 2;
    const std::string native  = R"({"traceEvents":[]})";
    EXPECT_TRUE(asyncProfileWorkflowTrace(native, {first, unknown}).empty());
    EXPECT_EQ(mergeAsyncProfileTrace(native, {first, unknown}), native);
}

TEST(AsyncProfileTraceTest, RequestOnlyBundleOmitsEmptyLogicalMember) {
    const std::string native =
        R"({"traceEvents":[{"ph":"X","cat":"kernel","name":"gpu","pid":0,"tid":7,"ts":1.125,"dur":2.875}]})";
    const auto io   = sample();
    const auto path = ::testing::TempDir() + "physical_only_" + std::to_string(getpid()) + ".zip";
    writeAsyncProfileBundle(path, native, {io});
    std::ifstream     in(path, std::ios::binary);
    const std::string archive{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
    EXPECT_NE(archive.find("timeline.json"), std::string::npos);
    EXPECT_EQ(archive.find("requests.pftrace"), std::string::npos);
    EXPECT_EQ(archive.find("KV requests"), std::string::npos);
    EXPECT_NE(archive.find(mergeAsyncProfileTrace(native, {io})), std::string::npos);
    ASSERT_GE(archive.size(), 22u);
    EXPECT_EQ(archive.substr(archive.size() - 14, 4), std::string("\1\0\1\0", 4));  // one ZIP member
    std::remove(path.c_str());
}

}  // namespace rtp_llm
