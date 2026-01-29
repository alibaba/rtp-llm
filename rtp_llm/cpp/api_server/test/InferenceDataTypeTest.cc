
#include <gtest/gtest.h>

#include "rtp_llm/cpp/api_server/InferenceDataType.h"

namespace rtp_llm {

class InferenceDataTypeTest: public ::testing::Test {};

TEST(InferenceDataTypeTest, RawRequest) {
    std::string jsonStr = R"({"source": "alibaba", "private_request": true})";
    RawRequest  req1;
    FromJsonString(req1, jsonStr);
    ASSERT_EQ(req1.source, "alibaba");
    ASSERT_EQ(req1.private_request, true);

    jsonStr = R"({"source": "alibaba"})";
    RawRequest req2;
    FromJsonString(req2, jsonStr);
    ASSERT_EQ(req2.source, "alibaba");
    ASSERT_EQ(req2.private_request, std::nullopt);

    jsonStr = R"({"private_request": true})";
    RawRequest req3;
    FromJsonString(req3, jsonStr);
    ASSERT_EQ(req3.source, "unknown");
    ASSERT_EQ(req3.private_request, true);

    jsonStr = R"({"prompt_batch": ["prompt1", "prompt2", "prompt3"]})";
    RawRequest req5;
    FromJsonString(req5, jsonStr);
    ASSERT_EQ(req5.prompt_batch.has_value(), true);
    ASSERT_EQ(req5.prompt_batch.value().size(), 3);
    ASSERT_EQ(req5.prompt_batch.value()[0], "prompt1");
    ASSERT_EQ(req5.prompt_batch.value()[1], "prompt2");
    ASSERT_EQ(req5.prompt_batch.value()[2], "prompt3");
}

TEST(InferenceDataTypeTest, RawRequest_GenerateConfig_TopK) {
    std::string jsonStr = R"({"generate_config": {"top_k": 1}})";
    RawRequest  req;
    FromJsonString(req, jsonStr);
    ASSERT_EQ(req.generate_config.has_value(), true);
    ASSERT_EQ(req.generate_config.value().top_k, 1);
}

TEST(InferenceDataTypeTest, RawRequest_GenerateConfig_HiddenStates_False) {
    std::string jsonStr = R"({"generate_config": {"top_k": 1}})";
    RawRequest  req;
    FromJsonString(req, jsonStr);
    ASSERT_EQ(req.generate_config.has_value(), true);
    ASSERT_EQ(req.generate_config.value().top_k, 1);
    ASSERT_EQ(req.generate_config.value().return_hidden_states, false);
}

TEST(InferenceDataTypeTest, RawRequest_GenerateConfig_HiddenStates_True) {
    std::string jsonStr = R"({"generate_config": {"top_k": 1, "return_hidden_states": true}})";
    RawRequest  req;
    FromJsonString(req, jsonStr);
    ASSERT_EQ(req.generate_config.has_value(), true);
    ASSERT_EQ(req.generate_config.value().top_k, 1);
    ASSERT_EQ(req.generate_config.value().return_hidden_states, true);
}

TEST(InferenceDataTypeTest, RawRequest_Images_Vector) {
    std::string jsonStr = R"({"images": ["prompt1", "prompt2", "prompt3"]})";
    RawRequest  req;
    FromJsonString(req, jsonStr);
    ASSERT_EQ(req.images.has_value(), true);
    ASSERT_EQ(req.images_batch.has_value(), false);

    const auto& images_vector = req.images.value();
    ASSERT_EQ(images_vector.size(), 3);
    ASSERT_EQ(images_vector[0], "prompt1");
    ASSERT_EQ(images_vector[1], "prompt2");
    ASSERT_EQ(images_vector[2], "prompt3");
}

TEST(InferenceDataTypeTest, RawRequest_Images_VectorVector) {
    std::string jsonStr = R"({"images": [["prompt1"],
                                        ["prompt1", "prompt2"],
                                        ["prompt1", "prompt2", "prompt3"]]})";
    RawRequest  req;
    FromJsonString(req, jsonStr);
    ASSERT_EQ(req.images.has_value(), false);
    ASSERT_EQ(req.images_batch.has_value(), true);

    const auto& images_vector_vector = req.images_batch.value();
    ASSERT_EQ(images_vector_vector.size(), 3);
    ASSERT_EQ(images_vector_vector[0].size(), 1);
    ASSERT_EQ(images_vector_vector[1].size(), 2);
    ASSERT_EQ(images_vector_vector[2].size(), 3);
}

TEST(InferenceDataTypeTest, RawRequest_Text) {
    std::string jsonStr = R"({"text": "alibaba"})";
    RawRequest  req;
    FromJsonString(req, jsonStr);
    ASSERT_EQ(req.prompt, "alibaba");
}

TEST(InferenceDataTypeTest, RawRequest_Prompt) {
    std::string jsonStr = R"({"prompt": "alibaba"})";
    RawRequest  req;
    FromJsonString(req, jsonStr);
    ASSERT_EQ(req.prompt, "alibaba");
}

TEST(InferenceDataTypeTest, AuxInfoAdapter) {
    AuxInfo aux_info;
    aux_info.cost_time_us = 1000;
    AuxInfoAdapter aux_info_adapter(aux_info);
    std::string    jsonStr = ToJsonString(aux_info_adapter, true);
    ASSERT_TRUE(jsonStr.find(R"("cost_time":1)") != std::string::npos);
    ASSERT_TRUE(jsonStr.find(R"("iter_count":0)") != std::string::npos);
    ASSERT_TRUE(jsonStr.find(R"("input_len":0)") != std::string::npos);
    ASSERT_TRUE(jsonStr.find(R"("prefix_len":0)") != std::string::npos);
    ASSERT_TRUE(jsonStr.find(R"("output_len":0)") != std::string::npos);
    ASSERT_TRUE(jsonStr.find(R"("pd_sep":false)") != std::string::npos);
    ASSERT_TRUE(jsonStr.find(R"("step_output_len":0)") != std::string::npos);
    ASSERT_TRUE(jsonStr.find(R"("beam_responses":[])") != std::string::npos);
    ASSERT_TRUE(jsonStr.find(R"("cum_log_probs":)") == std::string::npos);
}

TEST(InferenceDataTypeTest, MultiSeqsResponse) {
    MultiSeqsResponse res;
    res.response.push_back("fake response 1");
    res.response.push_back("fake response 2");
    res.finished = true;
    res.aux_info.push_back(AuxInfoAdapter());

    std::string jsonStr = ToJsonString(res, true);
    ASSERT_TRUE(jsonStr.find(R"("finished":true)") != std::string::npos);
    ASSERT_TRUE(jsonStr.find(R"("fake response 1")") != std::string::npos);
    ASSERT_TRUE(jsonStr.find(R"("fake response 2")") != std::string::npos);
    ASSERT_TRUE(jsonStr.find(R"("aux_info":[)") != std::string::npos);
}

TEST(InferenceDataTypeTest, MultiSeqsResponse_OptionalHasValue) {
    MultiSeqsResponse res;
    res.response.push_back("fake response");
    res.finished = true;
    res.aux_info.push_back(AuxInfoAdapter());

    std::vector<float> vec = {1.111f, 2.222f, 3.333f};
    res.logits.emplace();
    res.logits.value().push_back(vec);

    std::string jsonStr = ToJsonString(res, true);
    ASSERT_TRUE(jsonStr.find(R"("finished":true)") != std::string::npos);
    ASSERT_TRUE(jsonStr.find(R"("response":"fake response")") != std::string::npos);
    ASSERT_TRUE(jsonStr.find(R"("aux_info":[)") != std::string::npos);
    ASSERT_TRUE(jsonStr.find(R"("logits":[[1.111,2.222,3.333]])") != std::string::npos);
}

TEST(InferenceDataTypeTest, MultiSeqsResponse_OptionalNull) {
    MultiSeqsResponse res;
    res.response.push_back("fake response");
    res.finished = true;
    res.aux_info.push_back(AuxInfoAdapter());

    std::string jsonStr = ToJsonString(res, true);
    ASSERT_TRUE(jsonStr.find(R"("finished":true)") != std::string::npos);
    ASSERT_TRUE(jsonStr.find(R"("response":"fake response")") != std::string::npos);
    ASSERT_TRUE(jsonStr.find(R"("aux_info":[)") != std::string::npos);
    ASSERT_TRUE(jsonStr.find(R"("logits":[)") == std::string::npos);
}

TEST(InferenceDataTypeTest, BatchResponse) {
    std::vector<MultiSeqsResponse> batch;
    batch.push_back(MultiSeqsResponse());
    BatchResponse res(batch);
    std::string   jsonStr = ToJsonString(res, true);
    ASSERT_TRUE(jsonStr.find(R"("response_batch":[)") != std::string::npos);
    ASSERT_TRUE(jsonStr.find(R"("finished":)") != std::string::npos);
    ASSERT_TRUE(jsonStr.find(R"("response":[)") != std::string::npos);
    ASSERT_TRUE(jsonStr.find(R"("aux_info":[)") != std::string::npos);
}

}  // namespace rtp_llm
