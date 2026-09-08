#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "c10/util/Exception.h"
#include "gtest/gtest.h"
#include "kineto/GenericTraceActivity.h"
#include "kineto/IActivityProfiler.h"
#include "kineto/libkineto.h"
#include "kineto/output_base.h"
#include "rtp_llm/cpp/engine_base/TorchProfiler.h"

namespace rtp_llm {
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
