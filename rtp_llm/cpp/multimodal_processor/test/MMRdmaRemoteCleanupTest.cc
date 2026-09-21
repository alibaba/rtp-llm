#include <mutex>
#include <stdexcept>

#include <gtest/gtest.h>
#include <pybind11/embed.h>

#include "rtp_llm/cpp/multimodal_processor/RemoteMultimodalProcessor.h"

namespace rtp_llm {

class MMRdmaRemoteCleanupTest: public ::testing::Test {
protected:
    class Service: public MultimodalRpcService::Service {
    public:
        grpc::Status RemoteMultimodalEmbedding(grpc::ServerContext*,
                                               const MultimodalInputsPB*,
                                               MultimodalOutputsPB* output) override {
            auto* desc = output->add_multimodal_outputs()->mutable_output_rdma();
            desc->set_addr(4096);
            desc->set_handle("late-v41-image");
            desc->set_rdma_ip("127.0.0.1");
            desc->set_rdma_port(1234);
            desc->set_nbytes(64);
            desc->add_nic_rkeys()->set_rkey(1);
            auto* tensor = desc->add_tensors();
            tensor->set_role(MMRdmaTensorPB::EMBEDDING);
            tensor->set_data_type(TensorPB::BF16);
            tensor->add_shape(4);
            tensor->add_shape(8);
            tensor->set_nbytes(64);
            return grpc::Status::OK;
        }

        grpc::Status ReleaseEmbedding(grpc::ServerContext*, const ReleaseEmbeddingPB* request, EmptyPB*) override {
            std::lock_guard<std::mutex> lock(mutex);
            handles.insert(handles.end(), request->handle().begin(), request->handle().end());
            return grpc::Status::OK;
        }

        std::vector<std::string> releasedHandles() {
            std::lock_guard<std::mutex> lock(mutex);
            return handles;
        }

    private:
        std::mutex mutex;
        std::vector<std::string> handles;
    } service;

    class Transport: public MMRdmaTransport {
    public:
        bool exportEmbedding(const std::vector<torch::Tensor>&,
                             const std::vector<MMRdmaTensorPB::Role>&,
                             MMRdmaDescPB*) override {
            return false;
        }
        void releaseEmbedding(const std::vector<std::string>&) override {}
        MMRdmaReadStatus readEmbedding(const MMRdmaDescPB&,
                                       std::vector<torch::Tensor>* out,
                                       int64_t,
                                       std::function<void()> release_after_completion,
                                       bool* release_deferred) override {
            *release_deferred = false;
            if (throw_before_submission) {
                throw std::bad_alloc();
            }
            if (status == MMRdmaReadStatus::SUCCESS) {
                out->push_back(torch::ones({4, 8}, torch::kBFloat16));
            }
            if (status == MMRdmaReadStatus::IN_FLIGHT || throw_after_submission) {
                deferred_release = std::move(release_after_completion);
                *release_deferred = true;
                if (complete_before_return) {
                    deferred_release();
                    deferred_release = {};
                }
            }
            if (throw_after_submission) {
                throw std::runtime_error("submission outcome unknown");
            }
            return status;
        }

        MMRdmaReadStatus status{MMRdmaReadStatus::IN_FLIGHT};
        bool throw_before_submission{false};
        bool throw_after_submission{false};
        bool complete_before_return{false};
        std::function<void()> deferred_release;
    };

    inline static std::unique_ptr<py::scoped_interpreter> interpreter;
    std::unique_ptr<grpc::Server> server;
    int port{0};

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
        auto input = std::make_shared<GenerateInput>();
        input->input_ids = torch::arange(6, torch::kInt32);
        input->generate_config = std::make_shared<GenerateConfig>();
        input->generate_config->timeout_ms = 5000;
        input->generate_config->role_addrs.emplace_back(RoleType::VIT, "127.0.0.1", 0, port);
        auto prepared = std::make_shared<V41RequestInputs>();
        prepared->token_types = torch::tensor({-1, 0, 1, 2, 3, -1}, torch::kInt32);
        prepared->image_mask = prepared->token_types.ne(-1);
        V41ImageInput image;
        image.start = 1;
        image.n_vit_h = 3;
        image.n_vit_w = 3;
        image.types = torch::tensor({0, 1, 2, 3}, torch::kInt32);
        image.patches = torch::ones({9, 3, 14, 14}, torch::kBFloat16);
        prepared->images.push_back(std::move(image));
        input->v41_inputs = std::move(prepared);
        return input;
    }
};

TEST_F(MMRdmaRemoteCleanupTest, LateCompletionOwnsHandleAndStubBeyondProcessorLifetime) {
    auto transport = std::make_shared<Transport>();
    {
        RemoteMultimodalProcessor processor(py::none(), MMModelConfig{}, 32);
        processor.rdma_transport_ = transport;
        auto request = input();
        EXPECT_FALSE(processor.updateMultimodalFeatures(request).ok());
        EXPECT_TRUE(service.releasedHandles().empty());
    }
    ASSERT_TRUE(transport->deferred_release);
    transport->deferred_release();
    EXPECT_EQ(service.releasedHandles(), std::vector<std::string>{"late-v41-image"});
}

TEST_F(MMRdmaRemoteCleanupTest, SubmissionExceptionDefersRemoteReleaseUntilConfirmedCompletion) {
    auto transport = std::make_shared<Transport>();
    transport->throw_after_submission = true;
    {
        RemoteMultimodalProcessor processor(py::none(), MMModelConfig{}, 32);
        processor.rdma_transport_ = transport;
        auto request = input();
        EXPECT_FALSE(processor.updateMultimodalFeatures(request).ok());
        EXPECT_TRUE(service.releasedHandles().empty());
    }
    ASSERT_TRUE(transport->deferred_release);
    transport->deferred_release();
    EXPECT_EQ(service.releasedHandles(), std::vector<std::string>{"late-v41-image"});
}

TEST_F(MMRdmaRemoteCleanupTest, AllocationFailureBeforeSubmissionStillReleasesRemoteHandle) {
    auto transport = std::make_shared<Transport>();
    transport->throw_before_submission = true;
    RemoteMultimodalProcessor processor(py::none(), MMModelConfig{}, 32);
    processor.rdma_transport_ = transport;
    auto request = input();
    EXPECT_FALSE(processor.updateMultimodalFeatures(request).ok());
    EXPECT_FALSE(transport->deferred_release);
    EXPECT_EQ(service.releasedHandles(), std::vector<std::string>{"late-v41-image"});
}

TEST_F(MMRdmaRemoteCleanupTest, CompletionDuringTimeoutHandoffReleasesOnlyOnce) {
    auto transport = std::make_shared<Transport>();
    transport->complete_before_return = true;
    RemoteMultimodalProcessor processor(py::none(), MMModelConfig{}, 32);
    processor.rdma_transport_ = transport;
    auto request = input();
    EXPECT_FALSE(processor.updateMultimodalFeatures(request).ok());
    EXPECT_EQ(service.releasedHandles(), std::vector<std::string>{"late-v41-image"});
}

TEST_F(MMRdmaRemoteCleanupTest, CompletedReadKeepsSynchronousReleaseOwnership) {
    auto transport = std::make_shared<Transport>();
    transport->status = MMRdmaReadStatus::SUCCESS;
    RemoteMultimodalProcessor processor(py::none(), MMModelConfig{}, 32);
    processor.rdma_transport_ = transport;
    auto request = input();
    EXPECT_TRUE(processor.updateMultimodalFeatures(request).ok());
    EXPECT_FALSE(transport->deferred_release);
    EXPECT_EQ(service.releasedHandles(), std::vector<std::string>{"late-v41-image"});
}

}  // namespace rtp_llm
