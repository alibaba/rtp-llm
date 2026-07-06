#include <gtest/gtest.h>

#include <set>
#include <vector>

#include "rtp_llm/cpp/model_rpc/PrefillBatchRpcServer.h"

namespace rtp_llm {
namespace {

class TestPrefillBatchRpcServer: public PrefillBatchRpcServer {
public:
    grpc::Status EnqueueGroup(grpc::ServerContext*,
                              const EnqueueGroupRequestPB* request,
                              EnqueueBatchResponsePB*      response) override {
        ++enqueue_group_calls;
        captured_group_request = *request;
        response->set_batch_id(request->batch_id());
        const int result_count = request->requests_size() - (omit_last_result ? 1 : 0);
        for (int i = 0; i < result_count; ++i) {
            const auto& group_input = request->requests(i);
            if (group_input.has_input()) {
                response->add_successes()->set_request_id(group_input.input().request_id());
            } else {
                auto* error = response->add_errors();
                error->set_request_id(0);
                error->mutable_error_info()->set_error_code(grpc::StatusCode::INVALID_ARGUMENT);
                error->mutable_error_info()->set_error_message("missing input");
            }
        }
        return grpc::Status::OK;
    }

    void setParallelism(int64_t dp_size, int64_t dp_rank) {
        maga_init_params_.parallelism_config.dp_size = dp_size;
        maga_init_params_.parallelism_config.dp_rank = dp_rank;
    }

    int                   enqueue_group_calls = 0;
    bool                  omit_last_result    = false;
    EnqueueGroupRequestPB captured_group_request;
};

EnqueueBatchExternalInputPB* addInput(EnqueueBatchDpSlotPB* slot, int64_t request_id) {
    auto* external_input = slot->add_requests();
    external_input->mutable_input()->set_request_id(request_id);
    return external_input;
}

std::set<int64_t> successIds(const EnqueueBatchResponsePB& response) {
    std::set<int64_t> ids;
    for (const auto& success : response.successes()) {
        ids.insert(success.request_id());
    }
    return ids;
}

TEST(PrefillBatchRpcServerTest, FlattensLocalSlotsAndPreservesMissingInputResult) {
    TestPrefillBatchRpcServer server;
    server.setParallelism(/*dp_size=*/1, /*dp_rank=*/0);

    EnqueueBatchRequestPB request;
    request.set_batch_id(101);
    auto* first_slot = request.add_dp_slots();
    first_slot->set_dp_rank(0);
    addInput(first_slot, 11);
    first_slot->add_requests();
    auto* second_slot = request.add_dp_slots();
    second_slot->set_dp_rank(0);
    addInput(second_slot, 12);

    EnqueueBatchResponsePB response;
    ASSERT_TRUE(server.EnqueueBatch(nullptr, &request, &response).ok());

    EXPECT_EQ(server.enqueue_group_calls, 1);
    EXPECT_EQ(server.captured_group_request.batch_id(), 101);
    EXPECT_EQ(server.captured_group_request.dp_rank(), 0);
    ASSERT_EQ(server.captured_group_request.requests_size(), 3);
    EXPECT_EQ(server.captured_group_request.requests(0).input().request_id(), 11);
    EXPECT_FALSE(server.captured_group_request.requests(1).has_input());
    EXPECT_EQ(server.captured_group_request.requests(2).input().request_id(), 12);
    EXPECT_EQ(response.batch_id(), 101);
    EXPECT_EQ(successIds(response), (std::set<int64_t>{11, 12}));
    ASSERT_EQ(response.errors_size(), 1);
    EXPECT_EQ(response.errors(0).request_id(), 0);
    EXPECT_EQ(response.errors(0).error_info().error_code(), grpc::StatusCode::INVALID_ARGUMENT);
    EXPECT_EQ(response.successes_size() + response.errors_size(), 3);
}

TEST(PrefillBatchRpcServerTest, RejectsInvalidRankWithoutBlockingLocalRequests) {
    TestPrefillBatchRpcServer server;
    server.setParallelism(/*dp_size=*/1, /*dp_rank=*/0);

    EnqueueBatchRequestPB request;
    request.set_batch_id(102);
    auto* local_slot = request.add_dp_slots();
    local_slot->set_dp_rank(0);
    addInput(local_slot, 21);
    auto* invalid_slot = request.add_dp_slots();
    invalid_slot->set_dp_rank(1);
    addInput(invalid_slot, 22);

    EnqueueBatchResponsePB response;
    ASSERT_TRUE(server.EnqueueBatch(nullptr, &request, &response).ok());

    EXPECT_EQ(server.enqueue_group_calls, 1);
    ASSERT_EQ(server.captured_group_request.requests_size(), 1);
    EXPECT_EQ(server.captured_group_request.requests(0).input().request_id(), 21);
    EXPECT_EQ(successIds(response), (std::set<int64_t>{21}));
    ASSERT_EQ(response.errors_size(), 1);
    EXPECT_EQ(response.errors(0).request_id(), 22);
    EXPECT_EQ(response.errors(0).error_info().error_code(), grpc::StatusCode::INVALID_ARGUMENT);
    EXPECT_EQ(response.successes_size() + response.errors_size(), 2);
}

TEST(PrefillBatchRpcServerTest, RejectsWholeBatchWhenRequestIdIsDuplicatedAcrossSlots) {
    TestPrefillBatchRpcServer server;
    server.setParallelism(/*dp_size=*/1, /*dp_rank=*/0);

    EnqueueBatchRequestPB request;
    request.set_batch_id(103);
    auto* local_slot = request.add_dp_slots();
    local_slot->set_dp_rank(0);
    addInput(local_slot, 31);
    local_slot->add_requests();
    auto* invalid_slot = request.add_dp_slots();
    invalid_slot->set_dp_rank(9);
    addInput(invalid_slot, 31);

    EnqueueBatchResponsePB response;
    ASSERT_TRUE(server.EnqueueBatch(nullptr, &request, &response).ok());

    EXPECT_EQ(server.enqueue_group_calls, 0);
    EXPECT_EQ(response.successes_size(), 0);
    ASSERT_EQ(response.errors_size(), 3);
    EXPECT_EQ(response.errors(0).request_id(), 31);
    EXPECT_EQ(response.errors(0).error_info().error_code(), grpc::StatusCode::ALREADY_EXISTS);
    EXPECT_EQ(response.errors(1).request_id(), 0);
    EXPECT_EQ(response.errors(1).error_info().error_code(), grpc::StatusCode::INVALID_ARGUMENT);
    EXPECT_EQ(response.errors(2).request_id(), 31);
    EXPECT_EQ(response.errors(2).error_info().error_code(), grpc::StatusCode::ALREADY_EXISTS);
}

TEST(PrefillBatchRpcServerTest, FailsFastWhenMultiDpIsConfigured) {
    TestPrefillBatchRpcServer server;
    server.setParallelism(/*dp_size=*/2, /*dp_rank=*/0);
    EnqueueBatchRequestPB  request;
    EnqueueBatchResponsePB response;

    EXPECT_ANY_THROW(server.EnqueueBatch(nullptr, &request, &response));
    EXPECT_EQ(server.enqueue_group_calls, 0);
}

TEST(PrefillBatchRpcServerTest, FailsFastWhenEnqueueGroupOmitsAResult) {
    TestPrefillBatchRpcServer server;
    server.setParallelism(/*dp_size=*/1, /*dp_rank=*/0);
    server.omit_last_result = true;
    EnqueueBatchRequestPB request;
    auto*                 slot = request.add_dp_slots();
    slot->set_dp_rank(0);
    addInput(slot, 51);
    addInput(slot, 52);
    EnqueueBatchResponsePB response;

    EXPECT_ANY_THROW(server.EnqueueBatch(nullptr, &request, &response));
    EXPECT_EQ(server.enqueue_group_calls, 1);
    EXPECT_EQ(response.successes_size() + response.errors_size(), 1);
}

}  // namespace
}  // namespace rtp_llm
