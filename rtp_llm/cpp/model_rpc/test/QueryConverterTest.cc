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
    QueryConverter::transResponse(&outputs_pb, &outputs, true, "", 10000);

    auto& output_pb   = outputs_pb.flatten_output();
    auto  aux_info_pb = output_pb.aux_info(0);
    EXPECT_EQ(aux_info_pb.cost_time_us(), 1000);
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
    EXPECT_THROW(QueryConverter::transTensorPB(&tensor_pb, tensor), std::runtime_error);
}

TEST_F(QueryConverterTest, testTransInputWithInputEmbeddings_FP32) {
    GenerateInputPB input;
    input.add_token_ids(0);
    input.add_token_ids(1);

    // 创建 input_embeddings
    auto* input_embeddings_pb = input.mutable_input_embeddings();

    // 添加第一个 embedding (FP32, shape [2, 3])
    auto* embedding1_pb = input_embeddings_pb->add_embeddings();
    embedding1_pb->set_data_type(TensorPB::FP32);
    embedding1_pb->add_shape(2);
    embedding1_pb->add_shape(3);
    std::vector<float> embedding1_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    embedding1_pb->set_fp32_data(reinterpret_cast<const char*>(embedding1_data.data()),
                                 embedding1_data.size() * sizeof(float));

    // 添加第二个 embedding (FP32, shape [1, 4])
    auto* embedding2_pb = input_embeddings_pb->add_embeddings();
    embedding2_pb->set_data_type(TensorPB::FP32);
    embedding2_pb->add_shape(1);
    embedding2_pb->add_shape(4);
    std::vector<float> embedding2_data = {7.0f, 8.0f, 9.0f, 10.0f};
    embedding2_pb->set_fp32_data(reinterpret_cast<const char*>(embedding2_data.data()),
                                 embedding2_data.size() * sizeof(float));

    // 添加 embedding_locs
    input_embeddings_pb->add_embedding_locs(5);
    input_embeddings_pb->add_embedding_locs(10);

    auto generate_input = QueryConverter::transQuery(&input);

    // 验证 input_embeddings 转换
    ASSERT_TRUE(generate_input->input_embeddings.has_value());
    ASSERT_TRUE(generate_input->input_embeddings_locs.has_value());

    const auto& embeddings = generate_input->input_embeddings.value();
    const auto& locs       = generate_input->input_embeddings_locs.value();

    ASSERT_EQ(embeddings.size(), 2);
    ASSERT_EQ(locs.size(), 2);

    // 验证第一个 embedding
    ASSERT_EQ(embeddings[0].dtype(), torch::kFloat32);
    ASSERT_EQ(embeddings[0].dim(), 2);
    ASSERT_EQ(embeddings[0].size(0), 2);
    ASSERT_EQ(embeddings[0].size(1), 3);
    auto embedding1_ptr = embeddings[0].data_ptr<float>();
    for (int i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(embedding1_ptr[i], embedding1_data[i]);
    }

    // 验证第二个 embedding
    ASSERT_EQ(embeddings[1].dtype(), torch::kFloat32);
    ASSERT_EQ(embeddings[1].dim(), 2);
    ASSERT_EQ(embeddings[1].size(0), 1);
    ASSERT_EQ(embeddings[1].size(1), 4);
    auto embedding2_ptr = embeddings[1].data_ptr<float>();
    for (int i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(embedding2_ptr[i], embedding2_data[i]);
    }

    // 验证 embedding_locs
    EXPECT_EQ(locs[0], 5);
    EXPECT_EQ(locs[1], 10);
}

TEST_F(QueryConverterTest, testTransInputWithInputEmbeddings_FP16) {
    GenerateInputPB input;
    input.add_token_ids(0);

    // 创建 input_embeddings (FP16)
    auto* input_embeddings_pb = input.mutable_input_embeddings();
    auto* embedding_pb        = input_embeddings_pb->add_embeddings();
    embedding_pb->set_data_type(TensorPB::FP16);
    embedding_pb->add_shape(2);
    embedding_pb->add_shape(2);

    std::vector<c10::Half> embedding_data = {1.0f, 2.0f, 3.0f, 4.0f};
    embedding_pb->set_fp16_data(reinterpret_cast<const char*>(embedding_data.data()),
                                embedding_data.size() * sizeof(c10::Half));

    input_embeddings_pb->add_embedding_locs(3);

    auto generate_input = QueryConverter::transQuery(&input);

    ASSERT_TRUE(generate_input->input_embeddings.has_value());
    const auto& embeddings = generate_input->input_embeddings.value();
    const auto& locs       = generate_input->input_embeddings_locs.value();

    ASSERT_EQ(embeddings.size(), 1);
    ASSERT_EQ(embeddings[0].dtype(), torch::kFloat16);
    ASSERT_EQ(embeddings[0].dim(), 2);
    EXPECT_EQ(locs[0], 3);
}

TEST_F(QueryConverterTest, testTransInputWithInputEmbeddings_BF16) {
    GenerateInputPB input;
    input.add_token_ids(0);

    // 创建 input_embeddings (BF16)
    auto* input_embeddings_pb = input.mutable_input_embeddings();
    auto* embedding_pb        = input_embeddings_pb->add_embeddings();
    embedding_pb->set_data_type(TensorPB::BF16);
    embedding_pb->add_shape(1);
    embedding_pb->add_shape(3);

    std::vector<c10::BFloat16> embedding_data = {1.0f, 2.0f, 3.0f};
    embedding_pb->set_bf16_data(reinterpret_cast<const char*>(embedding_data.data()),
                                embedding_data.size() * sizeof(c10::BFloat16));

    input_embeddings_pb->add_embedding_locs(2);

    auto generate_input = QueryConverter::transQuery(&input);

    ASSERT_TRUE(generate_input->input_embeddings.has_value());
    const auto& embeddings = generate_input->input_embeddings.value();

    ASSERT_EQ(embeddings.size(), 1);
    ASSERT_EQ(embeddings[0].dtype(), torch::kBFloat16);
    ASSERT_EQ(embeddings[0].dim(), 2);
}

TEST_F(QueryConverterTest, testTransInputWithoutInputEmbeddings) {
    GenerateInputPB input;
    input.add_token_ids(0);
    input.add_token_ids(1);

    auto generate_input = QueryConverter::transQuery(&input);

    // 验证没有 input_embeddings 时，字段为空
    ASSERT_FALSE(generate_input->input_embeddings.has_value());
    ASSERT_FALSE(generate_input->input_embeddings_locs.has_value());
}

}  // namespace rtp_llm
