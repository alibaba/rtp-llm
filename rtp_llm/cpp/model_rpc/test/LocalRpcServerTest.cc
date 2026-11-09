#include <gtest/gtest.h>

#include "rtp_llm/cpp/model_rpc/LocalRpcServer.h"

namespace rtp_llm {
namespace {

class WorkerStatusTestServer: public LocalRpcServer {
public:
    WorkerStatusTestServer(RoleType role, int concurrency_limit) {
        maga_init_params_.pd_sep_config.role_type              = role;
        maga_init_params_.concurrency_config.concurrency_limit = concurrency_limit;
    }

    // Requests can be in KV preparation or waiting for remote decode without
    // a local runtime task. No engine or GPU is needed for these status tests.
    EngineScheduleInfo getEngineScheduleInfo(int64_t) override {
        return EngineScheduleInfo{};
    }

    std::unique_ptr<AtomicGuard> startRequest() {
        return std::make_unique<AtomicGuard>(onflight_requests_);
    }
};

class LocalRpcServerWorkerStatusTest: public ::testing::TestWithParam<RoleType> {};

TEST_P(LocalRpcServerWorkerStatusTest, TracksRequestsOutsideTheLocalTaskList) {
    WorkerStatusTestServer server(GetParam(), 2);
    auto                  status = server.getWorkerStatusInfo(-1);
    EXPECT_EQ(status.role, GetParam());
    EXPECT_EQ(status.available_concurrency, 2);

    auto first = server.startRequest();
    status     = server.getWorkerStatusInfo(-1);
    EXPECT_TRUE(status.engine_schedule_info.running_task_info_list.empty());
    EXPECT_EQ(status.available_concurrency, 1);

    auto second = server.startRequest();
    EXPECT_EQ(server.getWorkerStatusInfo(-1).available_concurrency, 0);

    // Direct RPCs may exceed the configured limit. Never report a negative
    // value or wrap around the unsigned in-flight counter.
    auto third = server.startRequest();
    EXPECT_EQ(server.getWorkerStatusInfo(-1).available_concurrency, 0);
    second.reset();
    EXPECT_EQ(server.getWorkerStatusInfo(-1).available_concurrency, 0);
    first.reset();
    EXPECT_EQ(server.getWorkerStatusInfo(-1).available_concurrency, 1);
    third.reset();
    EXPECT_EQ(server.getWorkerStatusInfo(-1).available_concurrency, 2);
}

TEST_P(LocalRpcServerWorkerStatusTest, NonPositiveLimitReportsNoAvailableConcurrency) {
    for (const int limit : {0, -1}) {
        WorkerStatusTestServer server(GetParam(), limit);
        EXPECT_EQ(server.getWorkerStatusInfo(-1).available_concurrency, 0);
        auto request = server.startRequest();
        EXPECT_EQ(server.getWorkerStatusInfo(-1).available_concurrency, 0);
        request.reset();
        EXPECT_EQ(server.getWorkerStatusInfo(-1).available_concurrency, 0);
    }
}

INSTANTIATE_TEST_SUITE_P(Roles,
                        LocalRpcServerWorkerStatusTest,
                        ::testing::Values(RoleType::PREFILL, RoleType::DECODE, RoleType::PDFUSION));

}  // namespace
}  // namespace rtp_llm
