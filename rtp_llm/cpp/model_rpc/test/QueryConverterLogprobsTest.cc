#include <cstring>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"
#include "rtp_llm/cpp/model_rpc/QueryConverter.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"

namespace rtp_llm {

TEST(QueryConverterLogprobsTest, TranslatesGenerateConfig) {
    GenerateInputPB input;
    input.add_token_ids(1);
    auto* config = input.mutable_generate_config();
    config->set_return_logprobs(true);
    config->set_top_logprobs(2);

    auto generate_input = QueryConverter::transQuery(&input);

    ASSERT_NE(generate_input, nullptr);
    ASSERT_NE(generate_input->generate_config, nullptr);
    EXPECT_TRUE(generate_input->generate_config->return_logprobs);
    EXPECT_EQ(generate_input->generate_config->top_logprobs, 2);
}

TEST(QueryConverterLogprobsTest, TranslatesCompactOutputTensors) {
    GenerateOutput output;
    output.output_ids      = torch::tensor({{1, 2, 3}}, torch::kInt32);
    output.finished        = true;
    output.logprobs_offset = 0;
    output.logprobs_count  = 3;
    output.token_logprobs.emplace(torch::tensor({-0.1f, -0.2f, -0.3f}, torch::kFloat32));
    output.top_logprob_token_ids.emplace(torch::tensor({{10, 11}, {12, 13}, {14, 15}}, torch::kInt32));
    output.top_logprobs.emplace(torch::tensor({{-1.0f, -2.0f}, {-1.1f, -2.1f}, {-1.2f, -2.2f}}, torch::kFloat32));

    GenerateOutputs outputs;
    outputs.request_id = 1;
    outputs.generate_outputs.push_back(std::move(output));

    GenerateOutputsPB outputs_pb;
    QueryConverter::transResponse(&outputs_pb, &outputs, false, "", 0);

    const auto& flatten_output = outputs_pb.flatten_output();

    ASSERT_EQ(flatten_output.logprobs_offsets_size(), 1);
    ASSERT_EQ(flatten_output.logprobs_counts_size(), 1);
    EXPECT_EQ(flatten_output.logprobs_offsets(0), 0);
    EXPECT_EQ(flatten_output.logprobs_counts(0), 3);

    ASSERT_TRUE(flatten_output.has_token_logprobs());
    const auto& token_logprobs = flatten_output.token_logprobs();
    EXPECT_EQ(token_logprobs.data_type(), TensorPB::FP32);
    ASSERT_EQ(token_logprobs.shape_size(), 2);
    EXPECT_EQ(token_logprobs.shape(0), 1);
    EXPECT_EQ(token_logprobs.shape(1), 3);
    std::vector<float> token_logprob_values(3);
    std::memcpy(token_logprob_values.data(), token_logprobs.fp32_data().data(), token_logprobs.fp32_data().size());
    EXPECT_FLOAT_EQ(token_logprob_values[0], -0.1f);
    EXPECT_FLOAT_EQ(token_logprob_values[2], -0.3f);

    ASSERT_TRUE(flatten_output.has_top_logprob_token_ids());
    const auto& top_token_ids = flatten_output.top_logprob_token_ids();
    EXPECT_EQ(top_token_ids.data_type(), TensorPB::INT32);
    ASSERT_EQ(top_token_ids.shape_size(), 3);
    EXPECT_EQ(top_token_ids.shape(0), 1);
    EXPECT_EQ(top_token_ids.shape(1), 3);
    EXPECT_EQ(top_token_ids.shape(2), 2);
    std::vector<int32_t> top_token_id_values(6);
    std::memcpy(top_token_id_values.data(), top_token_ids.int32_data().data(), top_token_ids.int32_data().size());
    EXPECT_EQ(top_token_id_values[0], 10);
    EXPECT_EQ(top_token_id_values[5], 15);

    ASSERT_TRUE(flatten_output.has_top_logprobs());
    const auto& top_logprobs = flatten_output.top_logprobs();
    EXPECT_EQ(top_logprobs.data_type(), TensorPB::FP32);
    ASSERT_EQ(top_logprobs.shape_size(), 3);
    EXPECT_EQ(top_logprobs.shape(0), 1);
    EXPECT_EQ(top_logprobs.shape(1), 3);
    EXPECT_EQ(top_logprobs.shape(2), 2);
    std::vector<float> top_logprob_values(6);
    std::memcpy(top_logprob_values.data(), top_logprobs.fp32_data().data(), top_logprobs.fp32_data().size());
    EXPECT_FLOAT_EQ(top_logprob_values[0], -1.0f);
    EXPECT_FLOAT_EQ(top_logprob_values[5], -2.2f);
}

TEST(QueryConverterLogprobsTest, OmitsLogprobSubmessagesWhenDisabled) {
    GenerateOutput output;
    output.output_ids = torch::tensor({{1, 2}}, torch::kInt32);
    output.finished   = true;

    GenerateOutputs outputs;
    outputs.request_id = 2;
    outputs.generate_outputs.push_back(std::move(output));

    GenerateOutputsPB outputs_pb;
    QueryConverter::transResponse(&outputs_pb, &outputs, false, "", 0);

    ASSERT_TRUE(outputs_pb.has_flatten_output());
    const auto& flatten_output = outputs_pb.flatten_output();
    EXPECT_FALSE(flatten_output.has_token_logprobs());
    EXPECT_FALSE(flatten_output.has_top_logprob_token_ids());
    EXPECT_FALSE(flatten_output.has_top_logprobs());
    EXPECT_EQ(flatten_output.logprobs_offsets_size(), 0);
    EXPECT_EQ(flatten_output.logprobs_counts_size(), 0);
}

TEST(QueryConverterLogprobsTest, PreservesZeroTopKLogprobSubmessages) {
    GenerateInputPB input;
    input.add_token_ids(1);
    auto* config = input.mutable_generate_config();
    config->set_return_logprobs(true);
    config->set_top_logprobs(0);

    auto generate_input = QueryConverter::transQuery(&input);
    ASSERT_NE(generate_input, nullptr);
    ASSERT_NE(generate_input->generate_config, nullptr);
    EXPECT_TRUE(generate_input->generate_config->return_logprobs);
    EXPECT_EQ(generate_input->generate_config->top_logprobs, 0);

    GenerateOutput output;
    output.output_ids      = torch::tensor({{1, 2}}, torch::kInt32);
    output.finished        = true;
    output.logprobs_offset = 0;
    output.logprobs_count  = 2;
    output.token_logprobs.emplace(torch::tensor({-0.1f, -0.2f}, torch::kFloat32));
    output.top_logprob_token_ids.emplace(torch::empty({2, 0}, torch::kInt32));
    output.top_logprobs.emplace(torch::empty({2, 0}, torch::kFloat32));

    GenerateOutputs outputs;
    outputs.request_id = 3;
    outputs.generate_outputs.push_back(std::move(output));

    GenerateOutputsPB outputs_pb;
    QueryConverter::transResponse(&outputs_pb, &outputs, false, "", 0);

    const auto& flatten_output = outputs_pb.flatten_output();
    ASSERT_TRUE(flatten_output.has_token_logprobs());
    EXPECT_EQ(flatten_output.token_logprobs().data_type(), TensorPB::FP32);
    ASSERT_EQ(flatten_output.token_logprobs().shape_size(), 2);
    EXPECT_EQ(flatten_output.token_logprobs().shape(0), 1);
    EXPECT_EQ(flatten_output.token_logprobs().shape(1), 2);

    ASSERT_TRUE(flatten_output.has_top_logprob_token_ids());
    EXPECT_EQ(flatten_output.top_logprob_token_ids().data_type(), TensorPB::INT32);
    ASSERT_EQ(flatten_output.top_logprob_token_ids().shape_size(), 3);
    EXPECT_EQ(flatten_output.top_logprob_token_ids().shape(0), 1);
    EXPECT_EQ(flatten_output.top_logprob_token_ids().shape(1), 2);
    EXPECT_EQ(flatten_output.top_logprob_token_ids().shape(2), 0);
    EXPECT_TRUE(flatten_output.top_logprob_token_ids().int32_data().empty());

    ASSERT_TRUE(flatten_output.has_top_logprobs());
    EXPECT_EQ(flatten_output.top_logprobs().data_type(), TensorPB::FP32);
    ASSERT_EQ(flatten_output.top_logprobs().shape_size(), 3);
    EXPECT_EQ(flatten_output.top_logprobs().shape(0), 1);
    EXPECT_EQ(flatten_output.top_logprobs().shape(1), 2);
    EXPECT_EQ(flatten_output.top_logprobs().shape(2), 0);
    EXPECT_TRUE(flatten_output.top_logprobs().fp32_data().empty());
}

TEST(QueryConverterLogprobsTest, PadsVariableCompactRowsAndSerializesPlacement) {
    GenerateOutput boundary_output;
    boundary_output.output_ids            = torch::tensor({{10, 11, 12, 13}}, torch::kInt32);
    boundary_output.finished              = false;
    boundary_output.logprobs_offset       = 2;
    boundary_output.logprobs_count        = 2;
    boundary_output.token_logprobs        = torch::tensor({-0.1f, -0.2f}, torch::kFloat32);
    boundary_output.top_logprob_token_ids = torch::tensor({{12}, {13}}, torch::kInt32);
    boundary_output.top_logprobs          = torch::tensor({{-0.1f}, {-0.2f}}, torch::kFloat32);

    GenerateOutput content_output;
    content_output.output_ids            = torch::tensor({{20, 21, 22}}, torch::kInt32);
    content_output.finished              = false;
    content_output.logprobs_offset       = 0;
    content_output.logprobs_count        = 3;
    content_output.token_logprobs        = torch::tensor({-0.3f, -0.4f, -0.5f}, torch::kFloat32);
    content_output.top_logprob_token_ids = torch::tensor({{20}, {21}, {22}}, torch::kInt32);
    content_output.top_logprobs          = torch::tensor({{-0.3f}, {-0.4f}, {-0.5f}}, torch::kFloat32);

    GenerateOutputs outputs;
    outputs.request_id = 4;
    outputs.generate_outputs.push_back(std::move(boundary_output));
    outputs.generate_outputs.push_back(std::move(content_output));

    GenerateOutputsPB outputs_pb;
    QueryConverter::transResponse(&outputs_pb, &outputs, false, "", 0);

    const auto& flatten_output = outputs_pb.flatten_output();
    ASSERT_EQ(flatten_output.logprobs_offsets_size(), 2);
    ASSERT_EQ(flatten_output.logprobs_counts_size(), 2);
    EXPECT_EQ(flatten_output.logprobs_offsets(0), 2);
    EXPECT_EQ(flatten_output.logprobs_counts(0), 2);
    EXPECT_EQ(flatten_output.logprobs_offsets(1), 0);
    EXPECT_EQ(flatten_output.logprobs_counts(1), 3);

    ASSERT_EQ(flatten_output.token_logprobs().shape_size(), 2);
    EXPECT_EQ(flatten_output.token_logprobs().shape(0), 2);
    EXPECT_EQ(flatten_output.token_logprobs().shape(1), 3);
    std::vector<float> values(6);
    std::memcpy(values.data(),
                flatten_output.token_logprobs().fp32_data().data(),
                flatten_output.token_logprobs().fp32_data().size());
    EXPECT_FLOAT_EQ(values[0], -0.1f);
    EXPECT_FLOAT_EQ(values[1], -0.2f);
    EXPECT_FLOAT_EQ(values[2], 0.0f);
    EXPECT_FLOAT_EQ(values[5], -0.5f);
}

TEST(QueryConverterLogprobsTest, SuppliesZeroWidthRowsForThinkingOnlyOutputInMixedBatch) {
    GenerateOutput thinking_output;
    thinking_output.output_ids      = torch::tensor({{10, 11}}, torch::kInt32);
    thinking_output.finished        = false;
    thinking_output.logprobs_offset = 2;
    thinking_output.logprobs_count  = 0;

    GenerateOutput content_output;
    content_output.output_ids            = torch::tensor({{20}}, torch::kInt32);
    content_output.finished              = false;
    content_output.logprobs_offset       = 0;
    content_output.logprobs_count        = 1;
    content_output.token_logprobs        = torch::tensor({-0.3f}, torch::kFloat32);
    content_output.top_logprob_token_ids = torch::empty({1, 0}, torch::kInt32);
    content_output.top_logprobs          = torch::empty({1, 0}, torch::kFloat32);

    GenerateOutputs outputs;
    outputs.request_id = 5;
    outputs.generate_outputs.push_back(std::move(thinking_output));
    outputs.generate_outputs.push_back(std::move(content_output));

    GenerateOutputsPB outputs_pb;
    QueryConverter::transResponse(&outputs_pb, &outputs, false, "", 0);

    const auto& flatten_output = outputs_pb.flatten_output();
    ASSERT_EQ(flatten_output.logprobs_offsets_size(), 2);
    EXPECT_EQ(flatten_output.logprobs_offsets(0), 2);
    EXPECT_EQ(flatten_output.logprobs_counts(0), 0);
    ASSERT_EQ(flatten_output.token_logprobs().shape_size(), 2);
    EXPECT_EQ(flatten_output.token_logprobs().shape(0), 2);
    EXPECT_EQ(flatten_output.token_logprobs().shape(1), 1);
}

}  // namespace rtp_llm
