#include <chrono>
#include <future>
#include <memory>
#include <string>

#include <gtest/gtest.h>

#include "rtp_llm/cpp/model_rpc/GenerateContext.h"
// Build the cache-store registry without transport threads, including on HIP
// toolchains that do not honor the test target's -fno-access-control option.
#define private public
#undef private
#include "rtp_llm/cpp/testing/TestLogCapture.h"

namespace rtp_llm::test {
namespace {

class LifecycleTestStream: public GenerateStream {
public:
    explicit LifecycleTestStream(int64_t request_id):
        GenerateStream(makeInput(request_id), makeModelConfig(), RuntimeConfig{}, ResourceContext{}, nullptr) {}

    ErrorResult<GenerateOutputs> nextOutput(int64_t /*wait_timeout_ms*/ = 0) override {
        return ErrorResult<GenerateOutputs>(GenerateOutputs{});
    }

    void updateOutput(const StreamUpdateInfo&) override {}

    void setState(StreamState state) {
        std::lock_guard<std::mutex> lock(*mutex_);
        generate_status_->status.store(state);
        consumer_cv_->notify_all();
    }

private:
    static std::shared_ptr<GenerateInput> makeInput(int64_t request_id) {
        auto input             = std::make_shared<GenerateInput>();
        input->request_id      = request_id;
        input->begin_time_us   = currentTimeUs();
        input->generate_config = std::make_shared<GenerateConfig>();
        input->input_ids       = torch::tensor({1, 2, 3}, torch::kInt32);
        return input;
    }

    static ModelConfig makeModelConfig() {
        ModelConfig config;
        config.max_seq_len = 16;
        return config;
    }
};

std::unique_ptr<GenerateContext> makeContext(int64_t request_id, const std::shared_ptr<RpcServerRuntimeMeta>& meta) {
    static kmonitor::MetricsReporterPtr metrics_reporter;
    return std::make_unique<GenerateContext>(request_id, 0, nullptr, metrics_reporter, meta);
}

class CancellationProbeContext: public GenerateContext {
public:
    using GenerateContext::GenerateContext;
    using GenerateContext::cancelStreamOnTeardown;

    bool isRequestCancelled() const override {
        ++cancellation_queries;
        return false;
    }

    mutable int cancellation_queries = 0;
};

TEST(GenerateContextLifecycleTest, TeardownWithoutActiveStreamDoesNotQueryRpcCancellation) {
    kmonitor::MetricsReporterPtr reporter;
    auto                         meta = std::make_shared<RpcServerRuntimeMeta>();
    CancellationProbeContext     context(1006, 0, nullptr, reporter, meta);
    context.markRpcHandlingCompleted();

    context.cancelStreamOnTeardown();
    EXPECT_EQ(context.cancellation_queries, 0);

    auto stream = std::make_shared<LifecycleTestStream>(1006);
    stream->setState(StreamState::FINISHED);
    context.setStream(stream);
    context.cancelStreamOnTeardown();
    EXPECT_EQ(context.cancellation_queries, 0);

    stream->setState(StreamState::RUNNING);
    stream->reportError(ErrorCode::EXECUTION_EXCEPTION, "already failed");
    context.cancelStreamOnTeardown();
    EXPECT_EQ(context.cancellation_queries, 0);
    EXPECT_EQ(stream->statusInfo().code(), ErrorCode::EXECUTION_EXCEPTION);
}

bool destroyWithoutSchedulerProgress(std::unique_ptr<GenerateContext>            context,
                                     const std::shared_ptr<LifecycleTestStream>& stream) {
    auto destroyed   = std::async(std::launch::async, [context = std::move(context)]() mutable { context.reset(); });
    const auto ready = destroyed.wait_for(std::chrono::milliseconds(200)) == std::future_status::ready;
    if (!ready) {
        // Keep a regression from hanging the test process indefinitely.
        stream->setState(StreamState::FINISHED);
    }
    destroyed.get();
    return ready;
}

TEST(GenerateContextLifecycleTest, CompletedSuccessDoesNotCancelOrWaitForRunningStream) {
    auto meta    = std::make_shared<RpcServerRuntimeMeta>();
    auto stream  = std::make_shared<LifecycleTestStream>(1001);
    auto context = makeContext(1001, meta);
    stream->setState(StreamState::RUNNING);
    context->setStream(stream);
    context->markRpcHandlingCompleted();

    ASSERT_TRUE(destroyWithoutSchedulerProgress(std::move(context), stream));
    EXPECT_EQ(stream->getStatus(), StreamState::RUNNING);
    EXPECT_FALSE(stream->hasError());

    const auto runtime_info = meta->getEngineScheduleInfo(/*latest_finished_version=*/-1);
    EXPECT_TRUE(runtime_info.running_task_info_list.empty());
    ASSERT_EQ(runtime_info.finished_task_info_list.size(), 1);
    EXPECT_EQ(runtime_info.finished_task_info_list[0].request_id, 1001);
}

TEST(GenerateContextLifecycleTest, CompletedFailureCancelsOnceWithoutWaiting) {
    auto meta    = std::make_shared<RpcServerRuntimeMeta>();
    auto stream  = std::make_shared<LifecycleTestStream>(1002);
    auto context = makeContext(1002, meta);
    stream->setState(StreamState::RUNNING);
    context->setStream(stream);
    context->error_info   = ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "writer failed");
    context->error_status = grpc::Status(grpc::StatusCode::INTERNAL, "writer failed");
    context->markRpcHandlingCompleted();

    ASSERT_TRUE(destroyWithoutSchedulerProgress(std::move(context), stream));
    ASSERT_TRUE(stream->hasError());
    EXPECT_EQ(stream->statusInfo().code(), ErrorCode::EXECUTION_EXCEPTION);

    const auto runtime_info = meta->getEngineScheduleInfo(/*latest_finished_version=*/-1);
    ASSERT_EQ(runtime_info.finished_task_info_list.size(), 1);
    EXPECT_EQ(runtime_info.finished_task_info_list[0].error_code, static_cast<int64_t>(ErrorCode::EXECUTION_EXCEPTION));
}

TEST(GenerateContextLifecycleTest, GrpcFailureWithoutStreamErrorCancelsWithoutWaiting) {
    auto meta    = std::make_shared<RpcServerRuntimeMeta>();
    auto stream  = std::make_shared<LifecycleTestStream>(1005);
    auto context = makeContext(1005, meta);
    stream->setState(StreamState::RUNNING);
    context->setStream(stream);
    context->error_status = grpc::Status(grpc::StatusCode::INTERNAL, "writer failed");
    context->markRpcHandlingCompleted();

    ASSERT_TRUE(destroyWithoutSchedulerProgress(std::move(context), stream));
    ASSERT_TRUE(stream->hasError());
    EXPECT_EQ(stream->statusInfo().code(), ErrorCode::CANCELLED);
}

TEST(GenerateContextLifecycleTest, UnexpectedDestructionDiagnosesAndCancelsWithoutWaiting) {
    TestLogCapture capture("generate_context_unexpected_exit");
    auto           meta    = std::make_shared<RpcServerRuntimeMeta>();
    auto           stream  = std::make_shared<LifecycleTestStream>(1003);
    auto           context = makeContext(1003, meta);
    stream->setState(StreamState::RUNNING);
    context->setStream(stream);

    ASSERT_TRUE(destroyWithoutSchedulerProgress(std::move(context), stream));
    ASSERT_TRUE(stream->hasError());
    EXPECT_EQ(stream->statusInfo().code(), ErrorCode::CANCELLED);

    const auto logs       = capture.content();
    const auto diagnostic = std::string("GenerateContext destroyed before RPC handling completed");
    const auto first      = logs.find(diagnostic);
    ASSERT_NE(first, std::string::npos);
    EXPECT_EQ(logs.find(diagnostic, first + diagnostic.size()), std::string::npos);
}

TEST(GenerateContextLifecycleTest, RetryReplacementStopsOnlyTheAbandonedAttempt) {
    auto meta       = std::make_shared<RpcServerRuntimeMeta>();
    auto old_stream = std::make_shared<LifecycleTestStream>(1004);
    auto new_stream = std::make_shared<LifecycleTestStream>(1004);
    auto context    = makeContext(1004, meta);
    old_stream->setState(StreamState::WAITING);
    new_stream->setState(StreamState::RUNNING);

    context->setStream(old_stream);
    context->setStream(new_stream);
    ASSERT_TRUE(old_stream->hasError());
    EXPECT_EQ(old_stream->statusInfo().code(), ErrorCode::CANCELLED);

    context->markRpcHandlingCompleted();
    ASSERT_TRUE(destroyWithoutSchedulerProgress(std::move(context), new_stream));
    EXPECT_FALSE(new_stream->hasError());
}

}  // namespace
}  // namespace rtp_llm::test
