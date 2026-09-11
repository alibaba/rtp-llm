#include "rtp_llm/cpp/devices/testing/TestBase.h"
#include <memory>
#include <optional>

#define private public
#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"
#include "rtp_llm/cpp/model_rpc/LocalRpcServer.h"
#include "rtp_llm/cpp/model_rpc/QueryConverter.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.grpc.pb.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"
#include "rtp_llm/cpp/devices/DeviceFactory.h"

using namespace std;
namespace rtp_llm {

class QueryConverterTest: public DeviceTestBase {};

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
    for (int i = 0; i < 2; ++i) {
        auto* stop_words = generate_config_pb->mutable_stop_words_list()->add_rows();
        for (int j = 0; j < 3; ++j) {
            stop_words->add_values(i * 3 + j);
        }
    }
    auto generate_input = QueryConverter::transQuery(&input);
    auto input_ids      = generate_input->input_ids.get();
    ASSERT_EQ(input_ids->size(), 2);
    ASSERT_EQ(*(int*)(input_ids->data()), 0);
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
    ASSERT_EQ(generate_config->stop_words_list.size(), 2);
    vector<int> stop_words_1{0, 1, 2};
    vector<int> stop_words_2{3, 4, 5};
    ASSERT_EQ(generate_config->stop_words_list[0], stop_words_1);
    ASSERT_EQ(generate_config->stop_words_list[1], stop_words_2);
}

TEST_F(QueryConverterTest, testTransOutput) {
    auto device = rtp_llm::DeviceFactory::getDefaultDevice();
    auto output_token_ids =
        device->allocateBuffer({rtp_llm::DataType::TYPE_INT32, {1, 3}, rtp_llm::AllocationType::HOST}, {});
    auto data = (int*)output_token_ids->data();
    for (int i = 0; i < 3; ++i) {
        data[i] = i;
    }
    GenerateOutputs outputs;
    GenerateOutput  res;
    res.output_ids            = std::move(output_token_ids);
    res.finished              = true;
    res.aux_info.cost_time_us = 1000;
    res.aux_info.iter_count   = 9;
    res.aux_info.input_len    = 8;
    res.aux_info.output_len   = 7;
    auto hidden_states =
        device->allocateBuffer({rtp_llm::DataType::TYPE_FP32, {3, 2}, rtp_llm::AllocationType::HOST}, {});
    auto hidden_states_data = (float*)hidden_states->data();
    for (int i = 0; i < 6; ++i) {
        hidden_states_data[i] = i;
    }
    res.hidden_states.emplace(std::move(hidden_states));
    outputs.generate_outputs.push_back(res);

    GenerateOutputsPB outputs_pb;
    QueryConverter::transResponse(&outputs_pb, &outputs, "");

    auto& output_pb   = outputs_pb.generate_outputs(0);
    auto  aux_info_pb = output_pb.aux_info();
    EXPECT_EQ(aux_info_pb.cost_time_us(), 1000);
    EXPECT_EQ(aux_info_pb.iter_count(), 9);
    EXPECT_EQ(aux_info_pb.input_len(), 8);
    EXPECT_EQ(aux_info_pb.output_len(), 7);
    auto output_ids_pb = output_pb.output_ids();
    ASSERT_EQ(output_ids_pb.data_type(), TensorPB_DataType::TensorPB_DataType_INT32);
    ASSERT_EQ(output_ids_pb.shape_size(), 2);
    ASSERT_EQ(output_ids_pb.shape(0), 1);
    ASSERT_EQ(output_ids_pb.shape(1), 3);
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
    ASSERT_EQ(hidden_states_pb.shape_size(), 2);
    ASSERT_EQ(hidden_states_pb.shape(0), 3);
    ASSERT_EQ(hidden_states_pb.shape(1), 2);
    auto          hidden_states_string = hidden_states_pb.fp32_data();
    vector<float> hidden_states_vector;
    hidden_states_vector.resize(hidden_states_string.size() / sizeof(float));
    std::memcpy(hidden_states_vector.data(), hidden_states_string.data(), hidden_states_string.size());
    for (int i = 0; i < 6; ++i) {
        ASSERT_FLOAT_EQ(hidden_states_vector[i], i);
    }
}

TEST_F(QueryConverterTest, testBatchedOutputNegotiationAndLegacyFallback) {
    GenerateInputPB input;
    input.mutable_generate_config();
    auto config = QueryConverter::transQuery(&input)->generate_config;
    EXPECT_FALSE(config->accept_batched_output);
    EXPECT_TRUE(config->aux_info);
    input.mutable_generate_config()->set_accept_batched_output(true);
    input.mutable_generate_config()->mutable_aux_info()->set_value(false);
    config = QueryConverter::transQuery(&input)->generate_config;
    EXPECT_TRUE(config->accept_batched_output);
    EXPECT_FALSE(config->aux_info);

    GenerateOutputs source;
    source.request_id = 42;
    for (int i = 0; i < 3; ++i) {
        GenerateOutput row;
        row.finished               = i == 2;
        row.output_ids             = createBuffer<int32_t>({1, 2}, {17, 20 + i}, AllocationType::HOST);
        row.aux_info.output_len    = 2;
        row.aux_info.cost_time_us  = 12345;
        row.aux_info.cum_log_probs = createBuffer<float>({1}, {-float(i)}, AllocationType::HOST);
        row.aux_info.softmax_probs = createBuffer<float>({2}, {0.5f, 0.25f}, AllocationType::HOST);
        row.logits                 = createBuffer<float>({1, 2}, {float(i), 1.0f}, AllocationType::HOST);
        source.generate_outputs.push_back(row);
    }
    GenerateOutputsPB legacy, batched;
    QueryConverter::transResponse(&legacy, &source, "metadata");
    QueryConverter::transResponse(&batched, &source, "metadata", true, false);
    EXPECT_FALSE(legacy.has_batched_output());
    ASSERT_EQ(batched.generate_outputs_size(), 3);
    ASSERT_TRUE(batched.batched_output().has_output_ids());
    EXPECT_EQ(batched.batched_output().output_ids().shape(0), 3);
    EXPECT_EQ(batched.batched_output().output_ids().shape(1), 1);
    EXPECT_EQ(batched.batched_output().output_ids().shape(2), 2);
    std::string tokens, scores, probabilities, logits;
    for (int i = 0; i < 3; ++i) {
        tokens += legacy.generate_outputs(i).output_ids().int32_data();
        scores += legacy.generate_outputs(i).aux_info().cum_log_probs().fp32_data();
        probabilities += legacy.generate_outputs(i).aux_info().softmax_probs().fp32_data();
        logits += legacy.generate_outputs(i).logits().fp32_data();
        EXPECT_FALSE(batched.generate_outputs(i).has_output_ids());
        EXPECT_FALSE(batched.generate_outputs(i).aux_info().has_cum_log_probs());
        EXPECT_EQ(batched.generate_outputs(i).aux_info().output_len(), 2);
        EXPECT_EQ(batched.generate_outputs(i).aux_info().cost_time_us(), 0);
        EXPECT_EQ(batched.generate_outputs(i).finished(), legacy.generate_outputs(i).finished());
    }
    EXPECT_EQ(batched.batched_output().output_ids().int32_data(), tokens);
    EXPECT_EQ(batched.batched_output().cum_log_probs().fp32_data(), scores);
    EXPECT_EQ(batched.batched_output().softmax_probs().fp32_data(), probabilities);
    EXPECT_EQ(batched.batched_output().logits().fp32_data(), logits);

    // Variable SID lengths / partially present optional tensors must remain exact.
    source.generate_outputs[1].output_ids = createBuffer<int32_t>({1, 1}, {31}, AllocationType::HOST);
    source.generate_outputs[2].logits.reset();
    GenerateOutputsPB ragged;
    QueryConverter::transResponse(&ragged, &source, "metadata", true);
    EXPECT_FALSE(ragged.batched_output().has_output_ids());
    EXPECT_FALSE(ragged.batched_output().has_logits());
    EXPECT_EQ(ragged.generate_outputs(1).output_ids().shape(1), 1);
    EXPECT_TRUE(ragged.batched_output().has_cum_log_probs());
}

}  // namespace rtp_llm
