#include <pybind11/embed.h>
#include "gtest/gtest.h"
#include "rtp_llm/cpp/multimodal_processor/RemoteMultimodalProcessor.h"
#include "rtp_llm/cpp/multimodal_processor/test/FakeMultimodalProcessor.h"

namespace rtp_llm {

class V41RemoteMultimodalProcessorTest: public ::testing::Test {
protected:
    class Service: public MultimodalRpcService::Service {
    public:
        MultimodalInputsPB  request;
        MultimodalOutputsPB outputs;
        grpc::Status        status;
        grpc::Status        RemoteMultimodalEmbedding(grpc::ServerContext*,
                                                      const MultimodalInputsPB* input,
                                                      MultimodalOutputsPB*      output) override {
            request.CopyFrom(*input);
            output->CopyFrom(outputs);
            return status;
        }
    } service;
    inline static std::unique_ptr<py::scoped_interpreter> interpreter;
    std::unique_ptr<grpc::Server>                         server;
    int                                                   port = 0;

    void SetUp() override {
        if (!Py_IsInitialized()) {
            interpreter = std::make_unique<py::scoped_interpreter>();
        }
        grpc::ServerBuilder builder;
        builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
        builder.RegisterService(&service);
        server = builder.BuildAndStart();
        ASSERT_NE(server, nullptr);
    }

    void TearDown() override {
        server->Shutdown();
        server->Wait();
    }

    std::shared_ptr<GenerateInput> input() {
        auto input                         = std::make_shared<GenerateInput>();
        input->input_ids                   = torch::arange(11, torch::kInt32);
        input->generate_config             = std::make_shared<GenerateConfig>();
        input->generate_config->timeout_ms = 5000;
        input->generate_config->role_addrs.emplace_back(RoleType::VIT, "127.0.0.1", 0, port);
        auto prepared         = std::make_shared<V41RequestInputs>();
        prepared->token_types = torch::tensor({-1, 0, 1, 2, 3, -1, 0, 1, 2, 3, -1}, torch::kInt32);
        prepared->image_mask  = prepared->token_types.ne(-1);
        for (int start : {1, 6}) {
            V41ImageInput image;
            image.start              = start;
            image.n_vit_h            = 3;
            image.n_vit_w            = 3;
            image.patches            = torch::full({9, 3, 14, 14}, start, torch::kBFloat16);
            image.types              = torch::tensor({0, 1, 2, 3}, torch::kInt32);
            image.content_sha256     = std::string(64, 'a' + start);
            image.processor_identity = "processor-v41";
            prepared->images.push_back(image);
            QueryConverter::transTensorPB(service.outputs.add_multimodal_outputs()->mutable_multimodal_embedding(),
                                          torch::full({4, 8}, start, torch::kBFloat16));
        }
        input->v41_inputs = prepared;
        return input;
    }
};

TEST_F(V41RemoteMultimodalProcessorTest, RoundTripPreservesTypedImagesAndCanonicalTokens) {
    RemoteMultimodalProcessor processor(py::none(), MMModelConfig{}, 32);
    auto                      request   = input();
    const auto                token_ids = request->input_ids.clone();
    ASSERT_TRUE(processor.updateMultimodalFeatures(request).ok());
    ASSERT_TRUE(service.request.has_v41_inputs());
    EXPECT_EQ(service.request.multimodal_inputs_size(), 0);
    EXPECT_FALSE(service.request.metadata_only());
    EXPECT_FALSE(service.request.support_rdma());
    const auto& typed = service.request.v41_inputs();
    EXPECT_EQ(typed.schema_version(), 1);
    EXPECT_EQ(typed.token_types_size(), 11);
    EXPECT_EQ(typed.image_mask_size(), 11);
    ASSERT_EQ(typed.images_size(), 2);
    ASSERT_EQ(request->multimodal_features->size(), 2);
    for (int index = 0; index < 2; ++index) {
        const auto& source = request->v41_inputs->images[index];
        const auto& image  = typed.images(index);
        EXPECT_EQ(image.start(), source.start);
        EXPECT_EQ(image.n_vit_h(), 3);
        EXPECT_EQ(image.n_vit_w(), 3);
        EXPECT_EQ(image.content_sha256(), source.content_sha256);
        EXPECT_EQ(image.processor_identity(), source.processor_identity);
        EXPECT_TRUE(torch::equal(QueryConverter::transTensor(image.patches()), source.patches));
        EXPECT_TRUE(
            torch::equal(request->multimodal_features->at(index), torch::full({4, 8}, source.start, torch::kBFloat16)));
    }
    EXPECT_TRUE(torch::equal(request->input_ids, token_ids));
    EXPECT_TRUE(torch::equal(*request->mm_locs, torch::tensor({1, 6}, torch::kInt32)));
    EXPECT_TRUE(
        torch::equal(*request->text_tokens_mask, request->v41_inputs->image_mask.logical_not().to(torch::kInt32)));
}

TEST_F(V41RemoteMultimodalProcessorTest, MissingFeaturesAndBadSpansFail) {
    RemoteMultimodalProcessor processor(py::none(), MMModelConfig{}, 32);
    auto                      request = input();
    service.outputs.mutable_multimodal_outputs()->RemoveLast();
    EXPECT_FALSE(processor.updateMultimodalFeatures(request).ok());
    QueryConverter::transTensorPB(service.outputs.add_multimodal_outputs()->mutable_multimodal_embedding(),
                                  torch::ones({3, 8}, torch::kBFloat16));
    EXPECT_FALSE(processor.updateMultimodalFeatures(request).ok());
}

TEST_F(V41RemoteMultimodalProcessorTest, RpcFailuresPropagate) {
    RemoteMultimodalProcessor processor(py::none(), MMModelConfig{}, 32);
    auto                      request = input();
    for (auto code : {grpc::StatusCode::CANCELLED, grpc::StatusCode::DEADLINE_EXCEEDED}) {
        service.status = grpc::Status(code, "interrupted");
        EXPECT_EQ(processor.updateMultimodalFeatures(request).code(),
                  code == grpc::StatusCode::CANCELLED ? ErrorCode::CANCELLED : ErrorCode::GENERATE_TIMEOUT);
    }
}

TEST_F(V41RemoteMultimodalProcessorTest, ParentDeadlineIsPreservedAndCancellationPropagates) {
    RemoteMultimodalProcessor processor(py::none(), MMModelConfig{}, 32);
    auto                      request = input();
    grpc::ClientContext       expired;
    const auto                deadline = std::chrono::system_clock::now() - std::chrono::milliseconds(1);
    expired.set_deadline(deadline);
    EXPECT_EQ(processor.updateMultimodalFeatures(request, &expired).code(), ErrorCode::GENERATE_TIMEOUT);
    EXPECT_EQ(expired.deadline(), deadline);
    grpc::ClientContext cancelled;
    cancelled.TryCancel();
    EXPECT_EQ(processor.updateMultimodalFeatures(request, &cancelled).code(), ErrorCode::CANCELLED);
}

TEST_F(V41RemoteMultimodalProcessorTest, LocalPreparedImagesStillUseLocalEngine) {
    auto     request = input();
    py::dict scope;
    py::exec("import torch\n"
             "from types import SimpleNamespace\n"
             "class Engine:\n"
             "    def submit_v41(self, images):\n"
             "        return SimpleNamespace(embeddings=[torch.full((len(x['types']), 8), x['start'], "
             "dtype=torch.bfloat16) for x in images])\n"
             "engine = Engine()\n",
             scope);
    FakeMultimodalProcessor processor(scope["engine"], {}, false, 32);
    ASSERT_TRUE(processor.updateMultimodalFeatures(request).ok());
    ASSERT_EQ(request->multimodal_features->size(), 2);
    EXPECT_FALSE(service.request.has_v41_inputs());
    EXPECT_TRUE(torch::equal(request->multimodal_features->at(1), torch::full({4, 8}, 6, torch::kBFloat16)));
}

}  // namespace rtp_llm
