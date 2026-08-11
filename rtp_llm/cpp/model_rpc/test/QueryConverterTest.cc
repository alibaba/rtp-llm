#include "rtp_llm/cpp/testing/TestBase.h"
#include <limits>
#include <memory>
#include <optional>

#include "grpcpp/impl/codegen/time.h"
#define private public
#include "rtp_llm/cpp/model_rpc/DecodeRpcServer.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"
#include "rtp_llm/cpp/model_rpc/LocalRpcServer.h"
#include "rtp_llm/cpp/model_rpc/PrefillRpcServer.h"
#include "rtp_llm/cpp/model_rpc/QueryConverter.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.grpc.pb.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"

using namespace std;
namespace rtp_llm {

class QueryConverterTest: public DeviceTestBase {};

class TestableLocalRpcServer: public LocalRpcServer {
public:
    using LocalRpcServer::prepareInput;
};

TEST_F(QueryConverterTest, testTransInput) {
    GenerateInputPB input;
    input.add_token_ids(0);
    input.add_token_ids(1);

    auto generate_config_pb = input.mutable_generate_config();
    generate_config_pb->set_min_new_tokens(4);
    generate_config_pb->set_max_new_tokens(5);
    generate_config_pb->set_num_beams(1);
    generate_config_pb->set_num_return_sequences(1);
    generate_config_pb->set_top_k(6);
    generate_config_pb->set_top_p(0.6);
    generate_config_pb->set_temperature(0.1);
    generate_config_pb->set_repetition_penalty(0.2);
    generate_config_pb->mutable_top_p_decay()->set_value(0.7);
    generate_config_pb->mutable_top_p_min()->set_value(0.3);
    generate_config_pb->mutable_top_p_reset_ids()->set_value(7);
    generate_config_pb->mutable_task_id()->set_value("8");
    generate_config_pb->set_calculate_loss(1);
    generate_config_pb->set_return_hidden_states(true);
    generate_config_pb->set_unique_key("reuse-session-a");
    for (int i = 0; i < 2; ++i) {
        auto* stop_words = generate_config_pb->mutable_stop_words_list()->add_rows();
        for (int j = 0; j < 3; ++j) {
            stop_words->add_values(i * 3 + j);
        }
    }
    auto  generate_input = QueryConverter::transQuery(&input);
    auto& input_ids      = generate_input->input_ids;
    ASSERT_EQ(input_ids.numel(), 2);
    ASSERT_EQ(input_ids.data_ptr<int32_t>()[0], 0);
    auto generate_config = generate_input->generate_config;
    ASSERT_EQ(generate_config->min_new_tokens, 4);
    ASSERT_EQ(generate_config->max_new_tokens, 5);
    ASSERT_EQ(generate_config->num_beams, 1);
    ASSERT_EQ(generate_config->num_return_sequences, 1);
    ASSERT_EQ(generate_config->top_k, 6);
    ASSERT_FLOAT_EQ(generate_config->top_p, 0.6);
    ASSERT_FLOAT_EQ(generate_config->temperature, 0.1);
    ASSERT_FLOAT_EQ(generate_config->repetition_penalty, 0.2);
    ASSERT_FLOAT_EQ(generate_config->top_p_decay.value(), 0.7);
    ASSERT_FLOAT_EQ(generate_config->top_p_min.value(), 0.3);
    ASSERT_EQ(generate_config->top_p_reset_ids.value(), 7);
    ASSERT_EQ(generate_config->task_id.value(), "8");
    ASSERT_EQ(generate_config->calculate_loss, 1);
    ASSERT_TRUE(generate_config->return_hidden_states);
    ASSERT_FALSE(generate_config->return_logits);
    ASSERT_EQ(generate_config->unique_key, "reuse-session-a");
    ASSERT_EQ(generate_config->stop_words_list.size(), 2);
    vector<int> stop_words_1{0, 1, 2};
    vector<int> stop_words_2{3, 4, 5};
    ASSERT_EQ(generate_config->stop_words_list[0], stop_words_1);
    ASSERT_EQ(generate_config->stop_words_list[1], stop_words_2);
}

TEST_F(QueryConverterTest, TransQueryPreservesLegacyRelativeTimeout) {
    GenerateInputPB input;
    input.add_token_ids(1);
    input.mutable_generate_config()->set_timeout_ms(500);
    input.set_start_time(1);

    const auto before = currentTimeUs();
    const auto query  = QueryConverter::transQuery(&input);
    const auto after  = currentTimeUs();

    EXPECT_GE(query->begin_time_us, before);
    EXPECT_LE(query->begin_time_us, after);
    EXPECT_EQ(query->generate_config->timeout_ms, 500);
}

TEST_F(QueryConverterTest, TransQueryAnchorsTimeoutToAbsoluteDeadline) {
    GenerateInputPB input;
    input.add_token_ids(1);
    input.mutable_generate_config()->set_timeout_ms(500);
    const auto now_us      = currentTimeUs();
    const auto deadline_ms = now_us / 1000 + 300;
    input.set_request_deadline_unix_ms(deadline_ms);

    const auto query = QueryConverter::transQuery(&input);

    EXPECT_EQ(query->generate_config->timeout_ms, 500);
    EXPECT_NEAR(query->begin_time_us, deadline_ms * 1000 - 500 * 1000, 1000);
}

TEST_F(QueryConverterTest, ExpiredDeadlineIsRejectedBeforeEngineAdmission) {
    GenerateInputPB input;
    input.add_token_ids(1);
    input.mutable_generate_config()->set_timeout_ms(500);
    input.set_request_deadline_unix_ms(currentTimeUs() / 1000 - 1);
    std::shared_ptr<GenerateInput> query;
    TestableLocalRpcServer         server;

    const auto error = server.prepareInput(input, query);

    EXPECT_EQ(error.code(), ErrorCode::GENERATE_TIMEOUT);
    EXPECT_EQ(query, nullptr);
}

TEST_F(QueryConverterTest, PrefillToDecodeHandoffPreservesRequestDeadline) {
    GenerateInputPB input;
    input.set_request_id(17);
    input.set_request_deadline_unix_ms(21'000);
    input.mutable_generate_config()->set_timeout_ms(500);

    GenerateRequestPB handoff;
    handoff.set_stage(RemoteStage::ALLOCATE);
    handoff.mutable_input()->CopyFrom(input);

    EXPECT_EQ(handoff.input().request_id(), 17);
    EXPECT_EQ(handoff.input().request_deadline_unix_ms(), 21'000);
    EXPECT_EQ(handoff.input().generate_config().timeout_ms(), 500);
}

TEST_F(QueryConverterTest, ExpiredPrefillRequestDoesNotReachEngineOrRouting) {
    grpc::ServerContext server_context;
    grpc::Timepoint2Timespec(std::chrono::system_clock::time_point(std::chrono::microseconds(10'000'000)),
                             &server_context.deadline_);
    GenerateInputPB            request;
    request.set_request_id(17);
    request.mutable_generate_config()->set_timeout_ms(500);
    RPCContext                   rpc_context{&request, nullptr};
    RemoteServerResource         resource;
    kmonitor::MetricsReporterPtr metrics;
    PrefillGenerateContext       context(&resource, rpc_context, 500, &server_context, metrics, nullptr);
    PrefillRpcServer             server;

    server.getRpcConnection(context);

    EXPECT_EQ(context.error_info.code(), ErrorCode::GENERATE_TIMEOUT);
    EXPECT_EQ(context.error_status.error_code(), grpc::StatusCode::DEADLINE_EXCEEDED);
    EXPECT_EQ(context.generate_input, nullptr);
}

TEST_F(QueryConverterTest, ExpiredDecodeRequestDoesNotReachEngineOrCacheAllocation) {
    grpc::ServerContext server_context;
    grpc::Timepoint2Timespec(std::chrono::system_clock::time_point(std::chrono::microseconds(10'000'000)),
                             &server_context.deadline_);
    DecodeRpcContext             rpc_context{nullptr};
    kmonitor::MetricsReporterPtr metrics;
    DecodeGenerateContext        context(rpc_context, 500, &server_context, metrics, nullptr);
    DecodeRpcServer              server;

    server.allocateResource(context);

    EXPECT_EQ(context.error_info.code(), ErrorCode::GENERATE_TIMEOUT);
    EXPECT_EQ(context.error_status.error_code(), grpc::StatusCode::DEADLINE_EXCEEDED);
    EXPECT_EQ(context.getStream(), nullptr);
}

TEST_F(QueryConverterTest, testTransOutput) {
    constexpr int64_t long_duration_us = static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 12345;
    auto output_token_ids = torch::empty({1, 3}, torch::kInt32);
    auto data             = output_token_ids.data_ptr<int>();
    for (int i = 0; i < 3; ++i) {
        data[i] = i;
    }
    GenerateOutputs outputs;
    GenerateOutput  res;
    res.output_ids            = output_token_ids;
    res.finished              = true;
    res.aux_info.cost_time_us             = long_duration_us;
    res.aux_info.first_token_cost_time_us = long_duration_us - 1;
    res.aux_info.wait_time_us             = long_duration_us - 2;
    res.aux_info.iter_count   = 9;
    res.aux_info.input_len    = 8;
    res.aux_info.output_len   = 7;
    auto hidden_states_tensor = torch::empty({3, 2}, torch::kFloat32);
    auto hidden_states_data   = hidden_states_tensor.data_ptr<float>();
    for (int i = 0; i < 6; ++i) {
        hidden_states_data[i] = i;
    }
    res.hidden_states.emplace(hidden_states_tensor);
    outputs.generate_outputs.push_back(res);

    GenerateOutputsPB outputs_pb;
    QueryConverter::transResponse(&outputs_pb, &outputs, true, "", 10000);

    auto& output_pb   = outputs_pb.flatten_output();
    auto  aux_info_pb = output_pb.aux_info(0);
    EXPECT_EQ(aux_info_pb.cost_time_us(), long_duration_us);
    EXPECT_EQ(aux_info_pb.first_token_cost_time_us(), long_duration_us - 1);
    EXPECT_EQ(aux_info_pb.wait_time_us(), long_duration_us - 2);
    EXPECT_EQ(aux_info_pb.iter_count(), 9);
    EXPECT_EQ(aux_info_pb.input_len(), 8);
    EXPECT_EQ(aux_info_pb.output_len(), 7);
    auto output_ids_pb = output_pb.output_ids();
    ASSERT_EQ(output_ids_pb.data_type(), TensorPB_DataType::TensorPB_DataType_INT32);
    ASSERT_EQ(output_ids_pb.shape_size(), 3);
    ASSERT_EQ(output_ids_pb.shape(0), 1);
    ASSERT_EQ(output_ids_pb.shape(1), 1);
    ASSERT_EQ(output_ids_pb.shape(2), 3);
    auto            output_ids_string = output_ids_pb.int32_data();
    vector<int32_t> output_ids_vector;
    output_ids_vector.resize(output_ids_string.size() / sizeof(int32_t));
    std::memcpy(output_ids_vector.data(), output_ids_string.data(), output_ids_string.size());
    for (int i = 0; i < 3; ++i) {
        ASSERT_EQ(output_ids_vector[i], i);
    }
    ASSERT_TRUE(output_pb.has_hidden_states());
    auto hidden_states_pb = output_pb.hidden_states();
    ASSERT_EQ(hidden_states_pb.data_type(), TensorPB_DataType::TensorPB_DataType_FP32);
    ASSERT_EQ(hidden_states_pb.shape_size(), 3);
    ASSERT_EQ(hidden_states_pb.shape(0), 1);
    ASSERT_EQ(hidden_states_pb.shape(1), 3);
    ASSERT_EQ(hidden_states_pb.shape(2), 2);
    auto          hidden_states_string = hidden_states_pb.fp32_data();
    vector<float> hidden_states_vector;
    hidden_states_vector.resize(hidden_states_string.size() / sizeof(float));
    std::memcpy(hidden_states_vector.data(), hidden_states_string.data(), hidden_states_string.size());
    for (int i = 0; i < 6; ++i) {
        ASSERT_FLOAT_EQ(hidden_states_vector[i], i);
    }
}

TEST_F(QueryConverterTest, TransTensorPB_FP32) {

    torch::Tensor tensor = torch::rand({2, 3}, torch::kFloat32);
    TensorPB      tensor_pb;
    QueryConverter::transTensorPB(&tensor_pb, tensor);
    EXPECT_EQ(tensor_pb.data_type(), TensorPB::FP32);
    ASSERT_EQ(tensor_pb.shape_size(), 2);
    EXPECT_EQ(tensor_pb.shape(0), 2);
    EXPECT_EQ(tensor_pb.shape(1), 3);

    // 验证数据一致性
    const std::string& proto_data        = tensor_pb.fp32_data();
    const float*       proto_ptr         = reinterpret_cast<const float*>(proto_data.data());
    torch::Tensor      contiguous_tensor = tensor.contiguous();
    const float*       tensor_ptr        = contiguous_tensor.data_ptr<float>();

    ASSERT_EQ(proto_data.size(), contiguous_tensor.numel() * sizeof(float));
    for (int i = 0; i < contiguous_tensor.numel(); ++i) {
        EXPECT_FLOAT_EQ(proto_ptr[i], tensor_ptr[i]);
    }
}

TEST_F(QueryConverterTest, TransTensorPB_BF16) {
    torch::Tensor tensor = torch::rand({3}, torch::kBFloat16);
    TensorPB      tensor_pb;
    QueryConverter::transTensorPB(&tensor_pb, tensor);

    EXPECT_EQ(tensor_pb.data_type(), TensorPB::BF16);

    const std::string& proto_data    = tensor_pb.bf16_data();
    size_t             expected_size = tensor.numel() * sizeof(c10::BFloat16);
    ASSERT_EQ(proto_data.size(), expected_size);

    const char* tensor_data = static_cast<const char*>(tensor.contiguous().data_ptr());
    EXPECT_EQ(std::memcmp(proto_data.data(), tensor_data, expected_size), 0);
}

TEST_F(QueryConverterTest, TransTensorPB_ScalarShape) {
    torch::Tensor tensor = torch::tensor(42, torch::kInt32);
    TensorPB      tensor_pb;
    QueryConverter::transTensorPB(&tensor_pb, tensor);
    EXPECT_EQ(tensor_pb.shape_size(), 0);
}

TEST_F(QueryConverterTest, TransTensorPB_NonContiguous) {
    torch::Tensor tensor = torch::rand({3, 4}, torch::kFloat32).transpose(0, 1);
    TensorPB      tensor_pb;
    QueryConverter::transTensorPB(&tensor_pb, tensor);

    torch::Tensor      contiguous_tensor = tensor.contiguous();
    const std::string& proto_data        = tensor_pb.fp32_data();
    const float*       proto_ptr         = reinterpret_cast<const float*>(proto_data.data());
    const float*       tensor_ptr        = contiguous_tensor.data_ptr<float>();

    for (int i = 0; i < contiguous_tensor.numel(); ++i) {
        EXPECT_FLOAT_EQ(proto_ptr[i], tensor_ptr[i]);
    }
}

TEST_F(QueryConverterTest, TransTensorPB_UnsupportedType) {
    torch::Tensor tensor = torch::ones({1}, torch::kInt64);
    TensorPB      tensor_pb;
    tensor_pb.add_shape(7);
    tensor_pb.set_fp32_data(std::string(sizeof(float), '\0'));

    EXPECT_THROW(QueryConverter::transTensorPB(&tensor_pb, tensor), std::runtime_error);
    ASSERT_EQ(tensor_pb.shape_size(), 1);
    EXPECT_EQ(tensor_pb.shape(0), 7);
    EXPECT_EQ(tensor_pb.fp32_data().size(), sizeof(float));
}

TEST_F(QueryConverterTest, TransTensorRejectsOversizedPayload) {
    TensorPB tensor_pb;
    tensor_pb.set_data_type(TensorPB::FP32);
    tensor_pb.add_shape(1);
    tensor_pb.set_fp32_data(std::string(2 * sizeof(float), '\0'));

    EXPECT_THROW(QueryConverter::transTensor(tensor_pb), std::invalid_argument);
}

TEST_F(QueryConverterTest, TransTensorRejectsTruncatedAndUnexpectedPayloads) {
    TensorPB tensor_pb;
    tensor_pb.set_data_type(TensorPB::FP32);
    tensor_pb.add_shape(2);
    tensor_pb.set_fp32_data(std::string(sizeof(float), '\0'));
    EXPECT_THROW(QueryConverter::transTensor(tensor_pb), std::invalid_argument);

    tensor_pb.set_fp32_data(std::string(2 * sizeof(float), '\0'));
    tensor_pb.set_int32_data(std::string(2 * sizeof(int32_t), '\0'));
    EXPECT_THROW(QueryConverter::transTensor(tensor_pb), std::invalid_argument);
}

TEST_F(QueryConverterTest, TransTensorRejectsInvalidShape) {
    TensorPB tensor_pb;
    tensor_pb.set_data_type(TensorPB::FP16);
    tensor_pb.add_shape(-1);
    EXPECT_THROW(QueryConverter::transTensor(tensor_pb), std::invalid_argument);

    tensor_pb.clear_shape();
    tensor_pb.add_shape(std::numeric_limits<int64_t>::max());
    tensor_pb.add_shape(2);
    EXPECT_THROW(QueryConverter::transTensor(tensor_pb), std::invalid_argument);
}

TEST_F(QueryConverterTest, TransTensorAcceptsScalarAndZeroSizedShape) {
    TensorPB scalar_pb;
    scalar_pb.set_data_type(TensorPB::INT32);
    const int32_t value = 42;
    scalar_pb.set_int32_data(&value, sizeof(value));

    const auto scalar = QueryConverter::transTensor(scalar_pb);
    EXPECT_EQ(scalar.dim(), 0);
    EXPECT_EQ(scalar.item<int32_t>(), value);

    TensorPB empty_pb;
    empty_pb.set_data_type(TensorPB::BF16);
    empty_pb.add_shape(2);
    empty_pb.add_shape(0);
    empty_pb.add_shape(3);

    const auto empty = QueryConverter::transTensor(empty_pb);
    EXPECT_EQ(empty.sizes(), torch::IntArrayRef({2, 0, 3}));
    EXPECT_EQ(empty.numel(), 0);
}

TEST_F(QueryConverterTest, TransTensorPBClearsReusedMessage) {
    TensorPB tensor_pb;
    tensor_pb.add_shape(9);
    tensor_pb.set_fp32_data(std::string(sizeof(float), '\0'));

    const auto tensor = torch::tensor({7, 8}, torch::kInt32);
    QueryConverter::transTensorPB(&tensor_pb, tensor);

    ASSERT_EQ(tensor_pb.shape_size(), 1);
    EXPECT_EQ(tensor_pb.shape(0), 2);
    EXPECT_TRUE(tensor_pb.fp32_data().empty());
    EXPECT_EQ(tensor_pb.int32_data().size(), 2 * sizeof(int32_t));
    EXPECT_TRUE(torch::equal(QueryConverter::transTensor(tensor_pb), tensor));
}

TEST_F(QueryConverterTest, TransTensorPBRoundTripsZeroSizedTensor) {
    const auto tensor = torch::empty({2, 0, 3}, torch::kBFloat16);
    TensorPB  tensor_pb;

    QueryConverter::transTensorPB(&tensor_pb, tensor);

    ASSERT_EQ(tensor_pb.shape_size(), 3);
    EXPECT_EQ(tensor_pb.shape(0), 2);
    EXPECT_EQ(tensor_pb.shape(1), 0);
    EXPECT_EQ(tensor_pb.shape(2), 3);
    EXPECT_TRUE(tensor_pb.bf16_data().empty());
    const auto restored = QueryConverter::transTensor(tensor_pb);
    EXPECT_EQ(restored.sizes(), tensor.sizes());
    EXPECT_EQ(restored.scalar_type(), tensor.scalar_type());
}

TEST_F(QueryConverterTest, TransTensorPBFailureLeavesReusedMessageUnchanged) {
    TensorPB tensor_pb;
    tensor_pb.set_data_type(TensorPB::INT32);
    tensor_pb.add_shape(1);
    const int32_t value = 17;
    tensor_pb.set_int32_data(&value, sizeof(value));
    const auto original = tensor_pb.SerializeAsString();

    const auto meta_tensor =
        torch::empty({2}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kMeta));
    EXPECT_THROW(QueryConverter::transTensorPB(&tensor_pb, meta_tensor), std::exception);
    EXPECT_EQ(tensor_pb.SerializeAsString(), original);
}

}  // namespace rtp_llm
