#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#include <thread>

#include <fstream>
#include <iterator>
#include <cstdlib>

#include "autil/legacy/json.h"
#include "c10/util/Exception.h"
#include "gtest/gtest.h"
#include "kineto/GenericTraceActivity.h"
#include "kineto/IActivityProfiler.h"
#include "kineto/libkineto.h"
#include "kineto/output_base.h"
#include "rtp_llm/cpp/engine_base/TorchProfiler.h"

#include "rtp_llm/cpp/utils/ProfilingScope.h"

namespace rtp_llm {

struct TorchProfileTestPeer {
    static void useCpuActivities(TorchProfile& profile) {
        profile.activities_ = {tpi::ActivityType::CPU};
    }
    static std::string fileName(const TorchProfile& profile) {
        return profile.output_dir_ + "/" + profile.prefix_ + std::to_string(profile.count_.load()) + ".json";
    }
};

namespace {

namespace tap = torch::autograd::profiler;
namespace tpi = torch::profiler::impl;

class DuplicateFlowSession: public libkineto::IActivityProfilerSession {
public:
    DuplicateFlowSession(): span_(0, 0, "duplicate_flow_span") {
        for (int64_t id = 1; id <= 2; ++id) {
            libkineto::GenericTraceActivity activity(
                span_, libkineto::ActivityType::CUDA_RUNTIME, "duplicate_flow_launch_" + std::to_string(id));
            activity.startTime = 100 + id;
            activity.endTime   = 200 + id;
            // Match CuptiActivity exactly: flowId() returns correlationId().
            // The two distinct API activities carry the same bad backend ID.
            activity.id         = 42;
            activity.flow.id    = 42;
            activity.flow.type  = libkineto::kLinkAsyncCpuGpu;
            activity.flow.start = true;
            activities_.push_back(std::move(activity));
        }
    }

    void start() override {
        status_ = libkineto::TraceStatus::RECORDING;
    }

    void stop() override {
        status_ = libkineto::TraceStatus::PROCESSING;
    }

    std::vector<std::string> errors() override {
        return {};
    }

    void processTrace(libkineto::ActivityLogger& logger) override {
        for (const auto& activity : activities_) {
            logger.handleGenericActivity(activity);
        }
    }

    std::unique_ptr<libkineto::DeviceInfo> getDeviceInfo() override {
        return nullptr;
    }

    std::vector<libkineto::ResourceInfo> getResourceInfos() override {
        return {};
    }

    std::unique_ptr<libkineto::CpuTraceBuffer> getTraceBuffer() override {
        return nullptr;
    }

private:
    libkineto::TraceSpan                         span_;
    std::vector<libkineto::GenericTraceActivity> activities_;
};

class DuplicateFlowProfiler: public libkineto::IActivityProfiler {
public:
    const std::string& name() const override {
        return name_;
    }

    const std::set<libkineto::ActivityType>& availableActivities() const override {
        return activities_;
    }

    std::unique_ptr<libkineto::IActivityProfilerSession> configure(const std::set<libkineto::ActivityType>&,
                                                                   const libkineto::Config&) override {
        return std::make_unique<DuplicateFlowSession>();
    }

    std::unique_ptr<libkineto::IActivityProfilerSession>
    configure(int64_t,
              int64_t,
              const std::set<libkineto::ActivityType>& activity_types,
              const libkineto::Config&                 config) override {
        return configure(activity_types, config);
    }

private:
    std::string                       name_{"duplicate_flow_profiler"};
    std::set<libkineto::ActivityType> activities_{libkineto::ActivityType::CUDA_RUNTIME};
};

void startCpuProfiler() {
    tpi::ProfilerConfig         config(tpi::ProfilerState::KINETO, /*report_input_shapes=*/false);
    std::set<tpi::ActivityType> activities{tpi::ActivityType::CPU};
    tap::prepareProfiler(config, activities);
    tap::enableProfiler(config, activities);
}

std::unique_ptr<tap::ProfilerResult> throwProfilerError() {
    throw std::runtime_error("synthetic profiler finalization failure");
}

std::unique_ptr<tap::ProfilerResult> throwUnknownProfilerError() {
    throw 1;
}

std::unique_ptr<tap::ProfilerResult> returnProfilerResult() {
    return std::make_unique<tap::ProfilerResult>();
}

TEST(TorchProfilerTest, DisableResultIsReturned) {
    EXPECT_NE(detail::tryDisableProfiler(&returnProfilerResult), nullptr);
}

TEST(TorchProfilerTest, DisableFailureIsNonFatal) {
    EXPECT_EQ(detail::tryDisableProfiler(&throwProfilerError), nullptr);
}

TEST(TorchProfilerTest, UnknownDisableFailureIsNonFatal) {
    EXPECT_EQ(detail::tryDisableProfiler(&throwUnknownProfilerError), nullptr);
}

TEST(TorchProfilerTest, DrainCompletesBeforeProfilerIsDisabled) {
    static bool drained;
    drained      = false;
    auto disable = +[]() {
        EXPECT_TRUE(drained);
        return returnProfilerResult();
    };
    auto result = detail::tryDisableProfiler(disable, [] { drained = true; });
    EXPECT_TRUE(drained);
    EXPECT_NE(result, nullptr);
}

TEST(TorchProfilerTest, DrainFailureStillDisablesOnceAndDropsIncompleteTrace) {
    static int disable_calls;
    disable_calls = 0;
    auto disable  = +[]() {
        ++disable_calls;
        return returnProfilerResult();
    };
    auto result = detail::tryDisableProfiler(disable, [] { throw std::runtime_error("synthetic device failure"); });
    EXPECT_EQ(disable_calls, 1);
    EXPECT_EQ(result, nullptr);
}

TEST(TorchProfilerTest, UnknownDrainFailureStillDisablesOnce) {
    static int disable_calls;
    disable_calls = 0;
    auto disable  = +[]() {
        ++disable_calls;
        return returnProfilerResult();
    };
    EXPECT_EQ(detail::tryDisableProfiler(disable, [] { throw 1; }), nullptr);
    EXPECT_EQ(disable_calls, 1);
}

TEST(TorchProfilerTest, CpuStopRecordsTeardownAndIsIdempotent) {
    TorchProfile profile("teardown_test_");
    TorchProfileTestPeer::useCpuActivities(profile);
    profile.start();
    auto [result, name] = profile.stopAndCollect();
    ASSERT_NE(result, nullptr);
    EXPECT_FALSE(name.empty());
    size_t teardown_events = 0;
    for (const auto& event : result->events()) {
        teardown_events += event.name() == "profiler.teardown(device_drain)";
    }
    EXPECT_EQ(teardown_events, 1);
    auto [second_result, second_name] = profile.stopAndCollect();
    EXPECT_EQ(second_result, nullptr);
    EXPECT_TRUE(second_name.empty());
}

TEST(TorchProfilerTest, AsyncTransferEventsStayOutsideKinetoAndLateCompletionIsSafe) {
    TorchProfile profile("async_transfer_test_");
    TorchProfileTestPeer::useCpuActivities(profile);
    profile.start();
    std::thread worker([] {
        AsyncProfileSpan span("disk_read", 42, 0, 8);
        span.finish();
    });
    worker.join();
    AsyncProfileSpan pending("staging_wait", 43, 0, 8);
    auto [result, name] = profile.stopAndCollect();
    ASSERT_NE(result, nullptr);
    size_t complete = 0, incomplete = 0;
    for (const auto& event : result->async_events) {
        complete += event.name.find("cache.transfer.disk_to_device.disk_read(request_id=42") == 0;
        incomplete += event.name.find("staging_wait(request_id=43") != std::string::npos
                      && event.name.find("incomplete=1") != std::string::npos;
    }
    for (const auto& event : result->events())
        EXPECT_NE(event.name().find("cache.transfer."), 0u);
    EXPECT_EQ(complete, 1u);
    EXPECT_EQ(incomplete, 1u);
    EXPECT_NO_THROW(pending.finish());
}

using autil::legacy::AnyCast;
using autil::legacy::json::JsonMap;
using autil::legacy::json::JsonArray;
using autil::legacy::json::ParseJson;
using autil::legacy::json::ToString;

std::string temporaryTrace() {
    char path[] = "/tmp/rtp-torch-profiler-test-XXXXXX";
    int  fd     = mkstemp(path);
    if (fd < 0)
        throw std::runtime_error("cannot create test trace");
    close(fd);
    return path;
}

JsonMap readTraceObject(const std::string& path) {
    std::ifstream     in(path);
    const std::string text{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
    if (const auto* outputs = std::getenv("TEST_UNDECLARED_OUTPUTS_DIR")) {
        std::ofstream evidence(std::string(outputs) + "/" + path.substr(path.find_last_of('/') + 1) + ".json");
        evidence << text;
    }
    return AnyCast<JsonMap>(ParseJson(text));
}

JsonArray readTrace(const std::string& path) {
    return AnyCast<JsonArray>(readTraceObject(path).at("traceEvents"));
}

TEST(TorchProfilerTest, ExportPlacesWorkersAndInlineScopesOnNativeThreadIds) {
    TorchProfile profile("thread_export_");
    TorchProfileTestPeer::useCpuActivities(profile);
    profile.start();
    { RTP_LLM_PROFILE_SCOPE("original_timeline_marker"); }
    AsyncProfileSpan inline_span("prepare", 42, 0, 8);
    inline_span.finish();
    int64_t     worker_tid = 0;
    std::thread worker([&] {
#if defined(__linux__)
        EXPECT_EQ(pthread_setname_np(pthread_self(), "btc_export"), 0);
#elif defined(__APPLE__)
        EXPECT_EQ(pthread_setname_np("btc_export"), 0);
#endif
        worker_tid = asyncProfileThreadId();
        AsyncProfileSpan span("disk_read", 42, 0, 8);
        span.finish();
    });
    worker.join();
    auto [result, unused] = profile.stopAndCollect();
    ASSERT_NE(result, nullptr);
    const auto path = temporaryTrace();
    result->save(path);
    const auto    root       = readTraceObject(path);
    const auto    events     = AnyCast<JsonArray>(root.at("traceEvents"));
    const auto    base_field = root.find("baseTimeNanoseconds");
    const int64_t base_ns    = base_field == root.end() ? 0 : std::stoll(ToString(base_field->second, true));
    std::remove(path.c_str());
    size_t native = 0, custom = 0, worker_names = 0;
    for (const auto& item : events) {
        auto e    = AnyCast<JsonMap>(item);
        auto name = AnyCast<std::string>(e.at("name"));
        if (name == "thread_name" && ToString(e.at("tid"), true) == std::to_string(worker_tid)) {
            ++worker_names;
            EXPECT_EQ(AnyCast<std::string>(AnyCast<JsonMap>(e.at("args")).at("name")), "btc_export");
        }
        if (name == "original_timeline_marker") {
            ++native;
            EXPECT_EQ(AnyCast<std::string>(e.at("cat")), "cpu_op");
            EXPECT_EQ(std::stoll(ToString(e.at("tid"), true)), asyncProfileThreadId());
        }
        if (name.find("cache.transfer.") != 0)
            continue;
        ++custom;
        EXPECT_EQ(AnyCast<std::string>(e.at("ph")), "X");
        EXPECT_EQ(AnyCast<std::string>(e.at("cat")), "cache.transfer.execution");
        EXPECT_EQ(std::stoll(ToString(e.at("pid"), true)), getpid());
        EXPECT_EQ(std::stoll(ToString(e.at("tid"), true)),
                  name.find(".prepare(") != std::string::npos ? asyncProfileThreadId() : worker_tid);
        size_t matched = 0;
        for (const auto& recorded : result->async_events) {
            if (recorded.name == name) {
                ++matched;
                EXPECT_DOUBLE_EQ(std::stod(ToString(e.at("ts"), true)),
                                 static_cast<double>(recorded.start_us * 1000 - base_ns) / 1000);
                EXPECT_DOUBLE_EQ(std::stod(ToString(e.at("dur"), true)), recorded.end_us - recorded.start_us);
            }
        }
        EXPECT_EQ(matched, 1u);
    }
    EXPECT_EQ(native, 1u);
    EXPECT_EQ(custom, 2u);
    EXPECT_EQ(worker_names, 1u);
}

TEST(TorchProfilerTest, BackgroundExportOmitsRequestViewButKeepsRequestExecution) {
    const auto path = temporaryTrace();
    {
        ProfilerSaveWorker saver;
        TorchProfile       profile("async_export_");
        TorchProfileTestPeer::useCpuActivities(profile);
        profile.start();
        { RTP_LLM_PROFILE_SCOPE("original_background_marker"); }
        AsyncProfileSpan total("total", -1, 0, 1, 0, -1, 77, "host_to_device", AsyncProfileKind::Lifecycle);
        AsyncProfileSpan wait("queue_wait", -1, 0, 1, 0, -1, 77, "host_to_device", AsyncProfileKind::Wait);
        AsyncProfileSpan request_total(
            "request_total", 42, 0, 1, 0, -1, 88, "host_to_device", AsyncProfileKind::Lifecycle);
        std::thread worker([&] {
            AsyncProfileSpan io("h2d", 42, 0, 1);
            io.finish();
            request_total.finish();
            wait.finish();
            total.finish();
        });
        worker.join();
        auto [result, unused] = profile.stopAndCollect();
        ASSERT_NE(result, nullptr);
        saver.enqueue(std::move(result), path);
    }  // Joins the saver before inspecting its output.
    auto events = readTrace(path);
    std::remove(path.c_str());
    size_t native = 0, begin = 0, end = 0, request_execution = 0;
    for (const auto& item : events) {
        auto e    = AnyCast<JsonMap>(item);
        auto name = AnyCast<std::string>(e.at("name"));
        native += name == "original_background_marker";
        EXPECT_NE(name, "KV requests");
        if (name.find("cache.transfer.") == 0 && AnyCast<std::string>(e.at("ph")) == "X") {
            const auto args = AnyCast<JsonMap>(e.at("args"));
            EXPECT_EQ(AnyCast<std::string>(args.at("request_id")), "42");
            ++request_execution;
        }
        if (name.find("KV ") != 0)
            continue;
        auto ph = AnyCast<std::string>(e.at("ph"));
        begin += ph == "b";
        end += ph == "e";
        EXPECT_NE(ph, "X");
        EXPECT_EQ(ToString(e.at("tid"), true), "0");
    }
    EXPECT_EQ(native, 1u);
    EXPECT_EQ(request_execution, 1u);
    EXPECT_EQ(begin, 2u);
    EXPECT_EQ(end, 2u);
}

TEST(TorchProfilerTest, SynchronousStopExportsNativeOnlyCapture) {
    const auto prefix = temporaryTrace();
    std::remove(prefix.c_str());
    TorchProfile profile(prefix.substr(5), "/tmp");
    TorchProfileTestPeer::useCpuActivities(profile);
    profile.start();
    { RTP_LLM_PROFILE_SCOPE("original_sync_marker"); }
    const auto path = TorchProfileTestPeer::fileName(profile);
    profile.stop();
    auto events = readTrace(path);
    std::remove(path.c_str());
    size_t marker = 0, custom = 0;
    for (const auto& item : events) {
        const auto name = AnyCast<std::string>(AnyCast<JsonMap>(item).at("name"));
        marker += name == "original_sync_marker";
        custom += name.find("cache.transfer.") == 0;
    }
    EXPECT_EQ(marker, 1u);
    EXPECT_EQ(custom, 0u);
}

TEST(TorchProfilerTest, SynchronousStopIncludesCustomExecution) {
    const auto prefix = temporaryTrace();
    std::remove(prefix.c_str());
    TorchProfile profile(prefix.substr(5), "/tmp");
    TorchProfileTestPeer::useCpuActivities(profile);
    profile.start();
    AsyncProfileSpan span("prepare", 42, 0, 1);
    span.finish();
    const auto path = TorchProfileTestPeer::fileName(profile);
    profile.stop();
    auto events = readTrace(path);
    std::remove(path.c_str());
    size_t custom = 0;
    for (const auto& item : events) {
        auto e = AnyCast<JsonMap>(item);
        if (AnyCast<std::string>(e.at("name")).find("cache.transfer.") == 0) {
            ++custom;
            EXPECT_EQ(AnyCast<std::string>(e.at("ph")), "X");
            EXPECT_EQ(std::stoll(ToString(e.at("tid"), true)), asyncProfileThreadId());
        }
    }
    EXPECT_EQ(custom, 1u);
}

TEST(TorchProfilerTest, FailedCustomExportStillSavesOriginalTimeline) {
    TorchProfile profile("failed_custom_export_");
    TorchProfileTestPeer::useCpuActivities(profile);
    profile.start();
    { RTP_LLM_PROFILE_SCOPE("retained_native_marker"); }
    AsyncProfileSpan span("prepare", 42, 0, 1);
    span.finish();
    auto [result, unused] = profile.stopAndCollect();
    ASSERT_NE(result, nullptr);
    ASSERT_EQ(result->async_events.size(), 1u);
    result->async_events[0].start_us = -1;
    const auto path                  = temporaryTrace();
    EXPECT_NO_THROW(result->save(path));
    auto events = readTrace(path);
    std::remove(path.c_str());
    size_t native = 0;
    for (const auto& item : events) {
        auto name = AnyCast<std::string>(AnyCast<JsonMap>(item).at("name"));
        native += name == "retained_native_marker";
        EXPECT_NE(name.find("cache.transfer."), 0u);
    }
    EXPECT_EQ(native, 1u);
}

TEST(TorchProfilerTest, DestructorDoesNotThrowWhenTraceCannotBeSaved) {
    EXPECT_NO_THROW({
        // /dev/null is a file, so no trace file can be created beneath it.
        TorchProfile profile("teardown_test_", "/dev/null/not_a_directory");
        TorchProfileTestPeer::useCpuActivities(profile);
        profile.start();
    });
}

TEST(TorchProfilerTest, RealDuplicateFlowDoesNotEscapeWrapper) {
    libkineto::api().registerProfilerFactory([]() { return std::make_unique<DuplicateFlowProfiler>(); });

    startCpuProfiler();
    try {
        // Newer PyTorch versions handle this malformed backend trace and
        // return a result. The incident version takes the catch branch below.
        EXPECT_NE(tap::disableProfiler(), nullptr);
    } catch (const c10::Error& e) {
        EXPECT_NE(std::string(e.what()).find("collection.cpp"), std::string::npos);
    }

    startCpuProfiler();
    EXPECT_NO_THROW((void)detail::tryDisableProfiler(&tap::disableProfiler));
}

}  // namespace
}  // namespace rtp_llm
