#include <gtest/gtest.h>

#include <list>
#include <memory>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/model_rpc/LocalRpcServer.h"

namespace rtp_llm {
namespace {

class WorkerStatusTestScheduler final: public SchedulerBase {
public:
    absl::Status enqueue(const GenerateStreamPtr&) override {
        return absl::OkStatus();
    }

    std::pair<std::vector<bool>, std::vector<GenerateStreamPtr>>
    enqueueGroup(const std::vector<GenerateStreamPtr>&) override {
        return {{}, {}};
    }

    absl::StatusOr<std::list<GenerateStreamPtr>> schedule() override {
        return std::list<GenerateStreamPtr>();
    }

    absl::Status stop() override {
        return absl::OkStatus();
    }

    bool empty() override {
        return true;
    }

    int64_t lastScheduleTime() override {
        return 0;
    }

    int64_t onflightStreams() override {
        return 0;
    }
};

class WorkerStatusTestEngine final: public EngineBase {
public:
    WorkerStatusTestEngine(): EngineBase(EngineInitParams()) {
        scheduler_ = std::make_unique<WorkerStatusTestScheduler>();
    }

    std::shared_ptr<GenerateStream> enqueue(const std::shared_ptr<GenerateInput>&) override {
        return nullptr;
    }

    void enqueue(std::shared_ptr<GenerateStream>&) override {}

    absl::Status stop() override {
        return absl::OkStatus();
    }

    absl::StatusOr<GenerateStreamPtr> preRun(const std::shared_ptr<GenerateInput>&, preRunMode) override {
        return absl::UnimplementedError("unused in test");
    }

    KVCacheInfo getCacheStatusInfo(int64_t, bool) override {
        return KVCacheInfo();
    }
};

class WorkerStatusTestServer final: public LocalRpcServer {
public:
    explicit WorkerStatusTestServer(EngineScheduleInfo schedule_info): schedule_info_(std::move(schedule_info)) {
        engine_ = std::make_shared<WorkerStatusTestEngine>();
    }

    EngineScheduleInfo getEngineScheduleInfo(int64_t) override {
        return schedule_info_;
    }

private:
    EngineScheduleInfo schedule_info_;
};

TEST(LocalRpcServerWorkerStatusTest, SerializesIntegerRequestIdsAsStrings) {
    EngineScheduleInfo schedule_info;
    schedule_info.running_task_info_list.push_back(EngineScheduleInfo::TaskInfo{/*request_id=*/1'234'567'890'123,
                                                                                /*prefix_length=*/64,
                                                                                /*input_length=*/128,
                                                                                /*waiting_time_ms=*/3});
    schedule_info.finished_task_info_list.push_back(EngineScheduleInfo::TaskInfo{/*request_id=*/0,
                                                                                 /*prefix_length=*/32,
                                                                                 /*input_length=*/96,
                                                                                 /*waiting_time_ms=*/5});

    WorkerStatusTestServer server(std::move(schedule_info));
    grpc::ServerContext    context;
    StatusVersionPB        request;
    WorkerStatusPB         response;

    ASSERT_TRUE(server.GetWorkerStatus(&context, &request, &response).ok());
    ASSERT_EQ(response.running_task_info_size(), 1);
    ASSERT_EQ(response.finished_task_list_size(), 1);
    EXPECT_EQ(response.running_task_info(0).request_id(), "1234567890123");
    EXPECT_EQ(response.finished_task_list(0).request_id(), "0");

    std::string serialized;
    ASSERT_TRUE(response.SerializeToString(&serialized));
    WorkerStatusPB parsed;
    ASSERT_TRUE(parsed.ParseFromString(serialized));
    ASSERT_EQ(parsed.running_task_info_size(), 1);
    ASSERT_EQ(parsed.finished_task_list_size(), 1);
    EXPECT_EQ(parsed.running_task_info(0).request_id(), "1234567890123");
    EXPECT_EQ(parsed.finished_task_list(0).request_id(), "0");
}

}  // namespace
}  // namespace rtp_llm
