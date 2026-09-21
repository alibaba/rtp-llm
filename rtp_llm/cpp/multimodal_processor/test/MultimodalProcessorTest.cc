#include <memory>
#include "gtest/gtest.h"
#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/cpp/multimodal_processor/test/FakeMultimodalProcessor.h"
#include "rtp_llm/cpp/multimodal_processor/RemoteMultimodalProcessor.h"

using namespace std;

namespace rtp_llm {

class MultimodalProcessorTest: public DeviceTestBase {};

namespace {
class PreparedV41Processor: public FakeMultimodalProcessor {
public:
    PreparedV41Processor(): FakeMultimodalProcessor(py::none(), {}, false, 32) {}
    int calls = 0;

protected:
    ErrorResult<MultimodalOutput>
    V41MultimodalEmbedding(const V41RequestInputs& inputs, const std::string&, grpc::ClientContext*) override {
        ++calls;
        MultimodalOutput output;
        for (const auto& image : inputs.images) {
            output.mm_features.push_back(torch::ones({image.types.numel(), 4}));
        }
        return output;
    }
};

std::shared_ptr<GenerateInput> preparedV41Input() {
    auto input           = std::make_shared<GenerateInput>();
    input->input_ids     = torch::zeros({11}, torch::kInt32);
    auto prepared_ptr    = std::make_shared<V41RequestInputs>();
    input->v41_inputs    = prepared_ptr;
    auto& prepared       = *prepared_ptr;
    prepared.token_types = torch::tensor({-1, 0, 1, 2, 3, -1, 0, 1, 2, 3, -1}, torch::kInt32);
    prepared.image_mask  = prepared.token_types.ne(-1);
    for (int32_t start : {1, 6}) {
        V41ImageInput image;
        image.start = start;
        image.types = torch::tensor({0, 1, 2, 3}, torch::kInt32);
        prepared.images.push_back(std::move(image));
    }
    return input;
}
}  // namespace

TEST_F(MultimodalProcessorTest, V41PreparedMetadataPreservesImagesAndTextOnlyRequests) {
    PreparedV41Processor processor;
    auto                 input = preparedV41Input();
    ASSERT_TRUE(processor.updateMultimodalFeatures(input).ok());
    EXPECT_EQ(processor.calls, 1);
    EXPECT_TRUE(torch::equal(input->mm_locs.value(), torch::tensor({1, 6}, torch::kInt32)));
    EXPECT_TRUE(
        torch::equal(input->text_tokens_mask.value(), input->v41_inputs->image_mask.logical_not().to(torch::kInt32)));
    EXPECT_EQ(input->multimodal_features->size(), 2);
    input     = preparedV41Input();
    auto text = std::make_shared<V41RequestInputs>(*input->v41_inputs);
    text->images.clear();
    text->token_types.fill_(-1);
    text->image_mask.fill_(false);
    input->v41_inputs = text;
    ASSERT_TRUE(processor.updateMultimodalFeatures(input).ok());
    EXPECT_EQ(processor.calls, 1);
    EXPECT_TRUE(input->multimodal_features->empty());
}

TEST_F(MultimodalProcessorTest, V41MalformedMetadataNeverReachesEmbedding) {
    for (const std::string invalid : {"token_shape",
                                      "token_dtype",
                                      "mask_length",
                                      "mask_dtype",
                                      "mask_value",
                                      "invalid_kind",
                                      "negative_start",
                                      "overlap",
                                      "past_end",
                                      "empty_types",
                                      "types_dtype",
                                      "types_mismatch",
                                      "missing_image",
                                      "uncovered_gap"}) {
        SCOPED_TRACE(invalid);
        PreparedV41Processor processor;
        auto                 input        = preparedV41Input();
        auto                 prepared_ptr = std::make_shared<V41RequestInputs>(*input->v41_inputs);
        input->v41_inputs                 = prepared_ptr;
        auto& prepared                    = *prepared_ptr;
        if (invalid == "token_shape")
            prepared.token_types = prepared.token_types.unsqueeze(0);
        if (invalid == "token_dtype")
            prepared.token_types = prepared.token_types.to(torch::kFloat32);
        if (invalid == "mask_length")
            prepared.image_mask = prepared.image_mask.slice(0, 0, 10);
        if (invalid == "mask_dtype")
            prepared.image_mask = prepared.image_mask.to(torch::kInt32);
        if (invalid == "mask_value")
            prepared.image_mask.index_put_({1}, false);
        if (invalid == "invalid_kind")
            prepared.token_types.index_put_({1}, 4);
        if (invalid == "negative_start")
            prepared.images[0].start = -1;
        if (invalid == "overlap")
            prepared.images[1].start = 3;
        if (invalid == "past_end")
            prepared.images[1].start = 10;
        if (invalid == "empty_types")
            prepared.images[1].types = torch::empty({0}, torch::kInt32);
        if (invalid == "types_dtype")
            prepared.images[1].types = prepared.images[1].types.to(torch::kFloat32);
        if (invalid == "types_mismatch")
            prepared.images[1].types.index_put_({1}, 2);
        if (invalid == "missing_image")
            prepared.images.pop_back();
        if (invalid == "uncovered_gap") {
            prepared.token_types.index_put_({5}, 1);
            prepared.image_mask.index_put_({5}, true);
        }
        EXPECT_FALSE(processor.updateMultimodalFeatures(input).ok());
        EXPECT_EQ(processor.calls, 0);
        EXPECT_FALSE(input->multimodal_features.has_value());
    }
}

TEST_F(MultimodalProcessorTest, StrictRemoteRejectsInlineAndReleasesMixedOutputs) {
    class FakeService: public MultimodalRpcService::Service {
    public:
        MultimodalOutputsPB      outputs;
        std::vector<std::string> released;
        grpc::Status             RemoteMultimodalEmbedding(grpc::ServerContext*,
                                                           const MultimodalInputsPB*,
                                                           MultimodalOutputsPB* response) override {
            response->CopyFrom(outputs);
            return grpc::Status::OK;
        }
        grpc::Status ReleaseEmbedding(grpc::ServerContext*, const ReleaseEmbeddingPB* request, EmptyPB*) override {
            released.assign(request->handle().begin(), request->handle().end());
            return grpc::Status::OK;
        }
    } service;
    QueryConverter::transTensorPB(service.outputs.add_multimodal_outputs()->mutable_multimodal_embedding(),
                                  torch::ones({2, 4}));
    grpc::ServerBuilder builder;
    int                 port = 0;
    builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(&service);
    auto server = builder.BuildAndStart();
    ASSERT_NE(server, nullptr);
    RemoteMultimodalProcessor processor(py::none(), MMModelConfig{}, 32);
    auto                      call = [&](const std::vector<MultimodalInput>& inputs) {
        grpc::ClientContext context;
        context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(5));
        return processor.MultimodalEmbedding(inputs, "127.0.0.1:" + std::to_string(port), &context);
    };
    // Simulate an auto client without a usable provider; inline features remain supported.
    processor.vit_config_.mm_transport_mode = "auto";
    EXPECT_TRUE(call({MultimodalInput("a")}).ok());
    processor.vit_config_.mm_transport_mode = "rdma";
    EXPECT_FALSE(call({MultimodalInput("a")}).ok());
    EXPECT_TRUE(service.released.empty());
    service.outputs.add_multimodal_outputs()->mutable_output_rdma()->set_handle("unread-slot");
    EXPECT_FALSE(call({MultimodalInput("a"), MultimodalInput("b")}).ok());
    EXPECT_EQ(service.released, std::vector<std::string>({"unread-slot"}));
    server->Shutdown();
    server->Wait();
}

TEST_F(MultimodalProcessorTest, SameUrlDifferentImageFeaturesHaveDifferentCacheTokens) {
    auto processor = FakeMultimodalProcessor::createFakeMultimodalProcessor({{1}}, false, 32);
    auto expand    = [&processor]() {
        auto input               = std::make_shared<GenerateInput>();
        input->input_ids         = torch::tensor({0, 1, 2}, torch::kInt32);
        input->multimodal_inputs = std::vector<MultimodalInput>{MultimodalInput("8")};
        auto status              = processor.updateMultimodalFeatures(input);
        EXPECT_TRUE(status.ok());
        return input->input_ids;
    };
    auto first = expand();
    EXPECT_TRUE(torch::equal(first, expand()));
    processor.feature_value = 1.0f;
    auto second             = expand();
    EXPECT_FALSE(torch::equal(first, second));
    EXPECT_EQ(first[0].item<int32_t>(), second[0].item<int32_t>());
    EXPECT_EQ(first[-1].item<int32_t>(), second[-1].item<int32_t>());
}

TEST_F(MultimodalProcessorTest, RequestDeadlineDoesNotExtendParentDeadline) {
    auto       processor = FakeMultimodalProcessor::createFakeMultimodalProcessor({{1}}, false, 32);
    const auto begin     = std::chrono::system_clock::now();
    for (bool shorter_parent : {false, true}) {
        auto input                         = std::make_shared<GenerateInput>();
        input->input_ids                   = torch::tensor({0, 1, 2}, torch::kInt32);
        input->multimodal_inputs           = std::vector<MultimodalInput>{MultimodalInput("3")};
        input->generate_config             = std::make_shared<GenerateConfig>();
        input->generate_config->timeout_ms = 100;
        input->begin_time_us = std::chrono::duration_cast<std::chrono::microseconds>(begin.time_since_epoch()).count();
        grpc::ClientContext context;
        auto request_deadline = std::chrono::system_clock::time_point(std::chrono::microseconds(input->begin_time_us))
                                + std::chrono::milliseconds(100);
        auto parent_deadline = request_deadline + std::chrono::milliseconds(shorter_parent ? -50 : 50);
        context.set_deadline(parent_deadline);
        ASSERT_TRUE(processor.updateMultimodalFeatures(input, &context).ok());
        EXPECT_EQ(context.deadline(), std::min(parent_deadline, request_deadline));
    }
}

TEST_F(MultimodalProcessorTest, testSimple) {
    FakeMultimodalProcessor        processor = FakeMultimodalProcessor::createFakeMultimodalProcessor({{1}}, false, 10);
    std::shared_ptr<GenerateInput> input     = std::make_shared<GenerateInput>();
    input->input_ids                         = torch::tensor({0, 1, 2, 3}, torch::kInt32);
    auto mm_inputs                           = std::vector<MultimodalInput>();
    mm_inputs.emplace_back("3");
    input->multimodal_inputs = mm_inputs;
    auto res                 = processor.updateMultimodalFeatures(input);
    EXPECT_EQ(res.ok(), true);
    EXPECT_EQ(processor.last_mm_padding_size, std::vector<int32_t>({0}));

    auto input_ids = input->input_ids.data_ptr<int32_t>();
    EXPECT_EQ(input->input_ids.numel(), 6);
    EXPECT_EQ(input_ids[0], 0);
    EXPECT_EQ(input_ids[4], 2);
    EXPECT_EQ(input_ids[5], 3);

    EXPECT_TRUE(input->text_tokens_mask);
    auto text_tokens_mask = input->text_tokens_mask.value().data_ptr<int32_t>();
    EXPECT_EQ(input->text_tokens_mask.value().numel(), 6);
    EXPECT_EQ(text_tokens_mask[0], 1);
    EXPECT_EQ(text_tokens_mask[1], 0);
    EXPECT_EQ(text_tokens_mask[2], 0);
    EXPECT_EQ(text_tokens_mask[3], 0);
    EXPECT_EQ(text_tokens_mask[4], 1);
    EXPECT_EQ(text_tokens_mask[5], 1);

    EXPECT_TRUE(input->mm_locs);
    auto locs = input->mm_locs.value().data_ptr<int32_t>();
    EXPECT_EQ(input->mm_locs.value().numel(), 1);
    EXPECT_EQ(locs[0], 1);

    EXPECT_TRUE(input->multimodal_features);
    EXPECT_EQ(input->multimodal_features.value().size(), 1);
}

TEST_F(MultimodalProcessorTest, testMultiInput) {
    FakeMultimodalProcessor processor =
        FakeMultimodalProcessor::createFakeMultimodalProcessor({{1}, {2, 3}}, false, 10);
    std::shared_ptr<GenerateInput> input = std::make_shared<GenerateInput>();
    input->input_ids                     = torch::tensor({0, 1, 2, 3}, torch::kInt32);
    auto mm_inputs                       = std::vector<MultimodalInput>();
    mm_inputs.emplace_back("3");
    mm_inputs.emplace_back("2");
    input->multimodal_inputs = mm_inputs;
    auto res                 = processor.updateMultimodalFeatures(input);
    EXPECT_EQ(res.ok(), true);
    EXPECT_EQ(processor.last_mm_padding_size, std::vector<int32_t>({0, 0}));

    EXPECT_EQ(input->input_ids.numel(), 8);

    EXPECT_TRUE(input->text_tokens_mask);
    auto text_tokens_mask = input->text_tokens_mask.value().data_ptr<int32_t>();
    EXPECT_EQ(input->text_tokens_mask.value().numel(), 8);
    EXPECT_EQ(text_tokens_mask[0], 1);
    EXPECT_EQ(text_tokens_mask[4], 1);
    EXPECT_EQ(text_tokens_mask[7], 1);

    EXPECT_TRUE(input->mm_locs);
    auto locs = input->mm_locs.value().data_ptr<int32_t>();
    EXPECT_EQ(input->mm_locs.value().numel(), 2);
    EXPECT_EQ(locs[0], 1);
    EXPECT_EQ(locs[1], 5);

    EXPECT_TRUE(input->multimodal_features);
    EXPECT_EQ(input->multimodal_features.value().size(), 2);
}

TEST_F(MultimodalProcessorTest, testMMPaddingUsesConfiguredAlignmentAndInterImageText) {
    for (int padding_size : {0, 2, 4}) {
        FakeMultimodalProcessor processor =
            FakeMultimodalProcessor::createFakeMultimodalProcessor({{1}}, false, 32, padding_size);
        auto input       = std::make_shared<GenerateInput>();
        input->input_ids = torch::tensor({9, 1, 7, 7, 1, 8}, torch::kInt32);
        auto mm_inputs   = std::vector<MultimodalInput>();
        mm_inputs.emplace_back("8");
        mm_inputs.emplace_back("6");
        input->multimodal_inputs = mm_inputs;

        auto res = processor.updateMultimodalFeatures(input);

        ASSERT_TRUE(res.ok()) << res.ToString();
        const std::vector<int32_t> expected =
            padding_size == 4 ? std::vector<int32_t>{2, 0} : std::vector<int32_t>{0, 0};
        EXPECT_EQ(processor.last_mm_padding_size, expected);
        EXPECT_EQ(input->input_ids.numel(), 18);
    }
}

TEST_F(MultimodalProcessorTest, testWrongMMTag) {
    FakeMultimodalProcessor processor = FakeMultimodalProcessor::createFakeMultimodalProcessor({{2, 3, 4}}, false, 10);
    std::shared_ptr<GenerateInput> input = std::make_shared<GenerateInput>();
    input->input_ids                     = torch::tensor({0, 1, 2, 3, 4}, torch::kInt32);
    auto mm_inputs                       = std::vector<MultimodalInput>();
    mm_inputs.emplace_back("2");
    input->multimodal_inputs = mm_inputs;
    auto res                 = processor.updateMultimodalFeatures(input);
    EXPECT_EQ(res.ok(), false);
    EXPECT_EQ(res.ToString(), "more than 2 sep tokens or no sep tokens for multimodal model is not supported");
    EXPECT_EQ(res.code(), ErrorCode::MM_WRONG_FORMAT_ERROR);

    processor.sep_token_ids_ = {{3, 5}};
    res                      = processor.updateMultimodalFeatures(input);
    EXPECT_EQ(res.ok(), false);
    EXPECT_EQ(res.ToString(), "unclosed multimodal tag pairs");
    EXPECT_EQ(res.code(), ErrorCode::MM_WRONG_FORMAT_ERROR);
}

TEST_F(MultimodalProcessorTest, testTooLongInput) {
    FakeMultimodalProcessor processor    = FakeMultimodalProcessor::createFakeMultimodalProcessor({{1, 2}}, false, 10);
    std::shared_ptr<GenerateInput> input = std::make_shared<GenerateInput>();
    input->input_ids                     = torch::tensor({0, 1, 2, 3}, torch::kInt32);
    auto mm_inputs                       = std::vector<MultimodalInput>();
    mm_inputs.emplace_back("10");
    input->multimodal_inputs = mm_inputs;
    auto res                 = processor.updateMultimodalFeatures(input);
    EXPECT_EQ(res.ok(), false);
    EXPECT_EQ(res.ToString(), "input after multimodal process is 14 > max_seq_len(10)");
    EXPECT_EQ(res.code(), ErrorCode::MM_LONG_PROMPT_ERROR);
}

TEST_F(MultimodalProcessorTest, testGetMMFeatures) {
    FakeMultimodalProcessor processor    = FakeMultimodalProcessor::createFakeMultimodalProcessor({{1, 2}}, false, 10);
    std::shared_ptr<GenerateInput> input = std::make_shared<GenerateInput>();
    input->input_ids                     = torch::tensor({0, 1, 2, 3}, torch::kInt32);
    auto mm_inputs                       = std::vector<MultimodalInput>();
    mm_inputs.emplace_back("2");
    input->multimodal_inputs = mm_inputs;
    auto res                 = processor.getMultimodalFeatures(input->input_ids, mm_inputs).value();
    EXPECT_EQ(res.features.size(), 1);
    EXPECT_EQ(res.text_tokens_mask.numel(), 6);
    EXPECT_EQ(res.locs.numel(), 1);
    EXPECT_EQ(res.expanded_ids.numel(), 6);
}

}  // namespace rtp_llm
