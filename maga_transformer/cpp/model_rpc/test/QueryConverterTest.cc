#include "src/fastertransformer/devices/testing/TestBase.h"
#include <memory>
#include <optional>

#define private public
#include "maga_transformer/cpp/dataclass/Query.h"
#include "maga_transformer/cpp/model_rpc/ModelRpcServer.h"
#include "maga_transformer/cpp/model_rpc/QueryConverter.h"
#include "maga_transformer/cpp/proto/model_rpc_service.grpc.pb.h"
#include "maga_transformer/cpp/proto/model_rpc_service.pb.h"
#include "src/fastertransformer/devices/DeviceFactory.h"

using namespace std;
namespace rtp_llm {

class QueryConverterTest: public DeviceTestBase {
};

TEST_F(QueryConverterTest, testTransInput) {
    ResourceContext resource_context;
    GenerateInputPB input;
    input.add_token_ids(0);
    input.add_token_ids(1);
    input.set_prefix_length(3);
    auto generate_config_pb = input.mutable_generate_config();
    generate_config_pb->set_min_new_tokens(4);
    generate_config_pb->set_max_new_tokens(5);
    generate_config_pb->set_num_beams(1);
    generate_config_pb->set_num_return_sequences(1);
    generate_config_pb->mutable_top_k()->set_value(6);
    generate_config_pb->mutable_top_p()->set_value(0.6);
    generate_config_pb->set_calculate_loss(1);
    generate_config_pb->set_return_hidden_states(true);
    auto query     = QueryConverter::transQuery(resource_context, &input);
    auto input_ids = query->generateInput()->input_ids.get();
    EXPECT_EQ(input_ids->size(), 2);
    ASSERT_EQ(*(int*)(input_ids->data()), 0);
    ASSERT_EQ(query->generateInput()->prefix_length, 3);
    auto generate_config = query->generateInput()->generate_config;
    ASSERT_EQ(generate_config->min_new_tokens, 4);
    ASSERT_EQ(generate_config->max_new_tokens, 5);
    ASSERT_EQ(generate_config->num_beams, 1);
    ASSERT_EQ(generate_config->num_return_sequences, 1);
    ASSERT_EQ(generate_config->top_k.value(), 6);
    ASSERT_FLOAT_EQ(generate_config->top_p.value(), 0.6);
    ASSERT_EQ(generate_config->calculate_loss, 1);
    ASSERT_TRUE(generate_config->return_hidden_states);
    ASSERT_FALSE(generate_config->return_logits);
}

TEST_F(QueryConverterTest, testTransOutput) {
    auto device           = ft::DeviceFactory::getDevice(ft::DeviceType::Cuda);
    auto output_token_ids = device->allocateBuffer({ft::DataType::TYPE_INT32, {1, 3}, ft::AllocationType::HOST}, {});
    auto data             = (int*)output_token_ids->data();
    for (int i = 0; i < 3; ++i) {
        data[i] = i;
    }
    GenerateOutputs outputs;
    GenerateOutput res;
    res.output_ids            = std::move(output_token_ids);
    res.finished              = true;
    res.aux_info.cost_time_us = 1000;
    res.aux_info.iter_count   = 9;
    res.aux_info.input_len    = 8;
    res.aux_info.output_len   = 7;
    auto hidden_states        = device->allocateBuffer({ft::DataType::TYPE_FP32, {3, 2}, ft::AllocationType::HOST}, {});
    auto hidden_states_data   = (float*)hidden_states->data();
    for (int i = 0; i < 6; ++i) {
        hidden_states_data[i] = i;
    }
    res.hidden_states.emplace(std::move(hidden_states));
    outputs.generate_outputs.push_back(res);

    GenerateOutputsPB outputs_pb;
    QueryConverter::transResponse(&outputs_pb, &outputs);
    auto& output_pb = outputs_pb.generate_outputs(0);
    auto aux_info_pb = output_pb.aux_info();
    EXPECT_EQ(aux_info_pb.cost_time_us(), 1000);
    EXPECT_EQ(aux_info_pb.iter_count(), 9);
    EXPECT_EQ(aux_info_pb.input_len(), 8);
    EXPECT_EQ(aux_info_pb.output_len(), 7);
    auto output_ids_pb = output_pb.output_ids();
    ASSERT_EQ(output_ids_pb.data_type(), TensorPB_DataType::TensorPB_DataType_INT32);
    ASSERT_EQ(output_ids_pb.shape_size(), 2);
    ASSERT_EQ(output_ids_pb.shape(0), 1);
    ASSERT_EQ(output_ids_pb.shape(1), 3);
    auto output_ids_string = output_ids_pb.int32_data();
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
    auto hidden_states_string = hidden_states_pb.fp32_data();
    vector<float> hidden_states_vector;
    hidden_states_vector.resize(hidden_states_string.size() / sizeof(float));
    std::memcpy(hidden_states_vector.data(), hidden_states_string.data(), hidden_states_string.size());
    for (int i = 0; i < 6; ++i) {
        ASSERT_FLOAT_EQ(hidden_states_vector[i], i);
    }
}

}  // namespace rtp_llm
