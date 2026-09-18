#include <chrono>
#include <string>
#include <utility>

#include <gtest/gtest.h>
#include "google/protobuf/stubs/common.h"
#include "rtp_llm/cpp/model_rpc/LocalRpcServer.h"
#include "rtp_llm/cpp/model_rpc/RpcErrorMessage.h"

namespace rtp_llm {
namespace {

// No model/weights. Only EngineBase's normal runtime initialization is needed;
// execution is protected by the standard GPU test lock.
class StatusTestEngine: public EngineBase {
public:
    StatusTestEngine(): EngineBase(EngineInitParams{}) {}
    GenerateStreamPtr enqueue(const std::shared_ptr<GenerateInput>&) override {
        return nullptr;
    }
    void         enqueue(GenerateStreamPtr&) override {}
    absl::Status stop() override {
        return absl::OkStatus();
    }
    absl::StatusOr<GenerateStreamPtr> preRun(const std::shared_ptr<GenerateInput>&, preRunMode) override {
        return absl::UnimplementedError("unused");
    }
    KVCacheInfo getCacheStatusInfo(int64_t, bool) override {
        return KVCacheInfo{};
    }
};

class StatusTestServer: public LocalRpcServer {
public:
    StatusTestServer() {
        engine_ = std::make_shared<StatusTestEngine>();
    }
    EngineScheduleInfo getEngineScheduleInfo(int64_t version) override {
        auto result = history;
        if (version >= result.latest_finished_version) {
            result.finished_task_info_list.clear();
        }
        return result;
    }
    EngineScheduleInfo history;
};

std::string binarySleepDetails() {
    ErrorDetailsPB details;
    details.set_error_code(8600);
    details.set_error_message("engine unavailable: DRAINING (sleep_epoch=1)");
    details.set_error_code_str("ENGINE_UNAVAILABLE");
    details.set_state("DRAINING");
    details.set_sleep_epoch(1);
    return details.SerializeAsString();
}

// This is the captured legacy shape, deliberately retained as a regression
// fixture even after the producer is fixed: existing history can contain it.
std::string poisonedHistoryMessage() {
    return "failed to load kv cache in rank: grpc_details=" + binarySleepDetails();
}

void expectValidMessage(const std::string& message) {
    EXPECT_TRUE(google::protobuf::internal::IsStructurallyValidUTF8(message));
}

class ErrorTransportService: public RpcService::Service {
public:
    ErrorTransportService(std::string message, ErrorCode code): message_(std::move(message)), code_(code) {}

    grpc::Status GetWorkerStatus(grpc::ServerContext*, const StatusVersionPB*, WorkerStatusPB*) override {
        // Exercise the production serializer, not a test copy of its policy.
        return serializer_.serializeErrorMsg("grpc-wire-regression", ErrorInfo(code_, message_));
    }

private:
    const std::string message_;
    const ErrorCode   code_;
    LocalRpcServer    serializer_;
};

void expectTransportedError(const std::string& input,
                            const std::string& expected,
                            ErrorCode          code = ErrorCode::ENGINE_UNAVAILABLE) {
    ErrorTransportService service(input, code);
    grpc::ServerBuilder   builder;
    int                   port = 0;
    builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(&service);
    auto server = builder.BuildAndStart();
    ASSERT_NE(server, nullptr);

    grpc::ChannelArguments args;
    // Do not mask oversized trailers by raising the production default.
    args.SetInt(GRPC_ARG_MAX_METADATA_SIZE, 8 * 1024);
    auto channel =
        grpc::CreateCustomChannel("127.0.0.1:" + std::to_string(port), grpc::InsecureChannelCredentials(), args);
    auto                stub = RpcService::NewStub(channel);
    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(10));
    StatusVersionPB request;
    WorkerStatusPB  response;
    const auto      status = stub->GetWorkerStatus(&context, request, &response);
    server->Shutdown();
    server->Wait();

    // An oversized metadata block becomes RESOURCE_EXHAUSTED and loses the
    // original application code. Parse the trailers actually received on wire.
    ASSERT_EQ(status.error_code(), transErrorCodeToGrpc(code)) << status.error_message();
    EXPECT_EQ(status.error_message(), expected);
    expectValidMessage(status.error_message());
    ErrorDetailsPB parsed;
    ASSERT_TRUE(parsed.ParseFromString(status.error_details()));
    EXPECT_EQ(parsed.error_code(), static_cast<int64_t>(code));
    EXPECT_EQ(parsed.error_message(), expected);
    expectValidMessage(parsed.error_message());
}

TEST(RpcErrorMessageRegressionTest, GrpcChannelCarriesSanitizedBinaryDetailsAndOriginalCode) {
    const auto input = poisonedHistoryMessage();
    ASSERT_FALSE(google::protobuf::internal::IsStructurallyValidUTF8(input));
    const auto expected = safeRpcErrorMessage(input);
    ASSERT_NE(expected.find("invalid UTF-8; hex="), std::string::npos);
    expectTransportedError(input, expected);
}

TEST(RpcErrorMessageRegressionTest, GrpcChannelPreservesShortUnicodeWithoutTruncation) {
    const std::string input = u8"正常错误信息：资源不足，稍后重试 🌏";
    expectTransportedError(input, input);
}

TEST(RpcErrorMessageRegressionTest, LegacyBinaryErrorIsSanitizedByRealGrpcSerializer) {
    LocalRpcServer server;
    const auto     poisoned = poisonedHistoryMessage();
    ASSERT_FALSE(google::protobuf::internal::IsStructurallyValidUTF8(poisoned));
    const auto status =
        server.serializeErrorMsg("captured-request", ErrorInfo(ErrorCode::ENGINE_UNAVAILABLE, poisoned));
    EXPECT_EQ(status.error_code(), grpc::StatusCode::UNAVAILABLE);
    expectValidMessage(status.error_message());
    ErrorDetailsPB parsed;
    ASSERT_TRUE(parsed.ParseFromString(status.error_details()));
    EXPECT_EQ(parsed.error_code(), 8600);
    expectValidMessage(parsed.error_message());
}

TEST(RpcErrorMessageRegressionTest, RealGrpcSerializerPreservesNormalUnicodeAndCode) {
    LocalRpcServer    server;
    const std::string text = u8"正常错误信息：资源不足，稍后重试 🌏";
    const auto     status = server.serializeErrorMsg("unicode-request", ErrorInfo(ErrorCode::ENGINE_UNAVAILABLE, text));
    ErrorDetailsPB parsed;
    ASSERT_TRUE(parsed.ParseFromString(status.error_details()));
    EXPECT_EQ(parsed.error_code(), 8600);
    EXPECT_EQ(parsed.error_message(), text);
    EXPECT_EQ(status.error_message(), text);
}

TEST(RpcErrorMessageRegressionTest, RealWorkerStatusKeepsThousandTasksWithFivePoisonedRecords) {
    StatusTestServer server;
    server.history.latest_finished_version = 1001;
    for (int64_t i = 1; i <= 1000; ++i) {
        EngineScheduleInfo::TaskInfo task{};
        task.request_id  = i;
        task.end_time_ms = i * 10;
        if (i <= 5) {
            task.error_code    = 8600;
            task.error_message = poisonedHistoryMessage();
        }
        server.history.finished_task_info_list.push_back(task);
    }
    const auto          original = server.history.finished_task_info_list.front().error_message;
    grpc::ServerContext context;
    StatusVersionPB     request;
    request.set_latest_finished_version(-1);
    WorkerStatusPB response;
    ASSERT_TRUE(server.GetWorkerStatus(&context, &request, &response).ok());
    ASSERT_EQ(response.finished_task_list_size(), 1000);
    for (int i = 0; i < 1000; ++i) {
        const auto& task = response.finished_task_list(i);
        EXPECT_EQ(task.request_id(), i + 1);
        EXPECT_EQ(task.end_time_ms(), (i + 1) * 10);
        if (i < 5) {
            EXPECT_EQ(task.error_info().error_code(), 8600);
            expectValidMessage(task.error_info().error_message());
        }
    }
    WorkerStatusPB parsed;
    ASSERT_TRUE(parsed.ParseFromString(response.SerializeAsString()));
    EXPECT_EQ(parsed.finished_task_list_size(), 1000);
    EXPECT_EQ(parsed.latest_finished_version(), 1001);
    // Output-only defense must not lose/mutate the internal completion record.
    EXPECT_EQ(server.history.finished_task_info_list.front().error_message, original);
    request.set_latest_finished_version(response.latest_finished_version());
    WorkerStatusPB delta;
    ASSERT_TRUE(server.GetWorkerStatus(&context, &request, &delta).ok());
    EXPECT_EQ(delta.finished_task_list_size(), 0);
    EXPECT_TRUE(parsed.ParseFromString(delta.SerializeAsString()));
}

TEST(RpcErrorMessageRegressionTest, RealWorkerStatusSanitizesRunningTaskErrorsToo) {
    StatusTestServer             server;
    EngineScheduleInfo::TaskInfo task{};
    task.request_id    = 42;
    task.error_code    = 8600;
    task.error_message = poisonedHistoryMessage();
    server.history.running_task_info_list.push_back(task);
    grpc::ServerContext context;
    StatusVersionPB     request;
    request.set_latest_finished_version(-1);
    WorkerStatusPB response;
    ASSERT_TRUE(server.GetWorkerStatus(&context, &request, &response).ok());
    ASSERT_EQ(response.running_task_info_size(), 1);
    EXPECT_EQ(response.running_task_info(0).request_id(), 42);
    EXPECT_EQ(response.running_task_info(0).error_info().error_code(), 8600);
    expectValidMessage(response.running_task_info(0).error_info().error_message());
    WorkerStatusPB parsed;
    EXPECT_TRUE(parsed.ParseFromString(response.SerializeAsString()));
}

}  // namespace
}  // namespace rtp_llm
