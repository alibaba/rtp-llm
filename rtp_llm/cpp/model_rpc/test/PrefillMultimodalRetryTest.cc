#include <gtest/gtest.h>

#include "rtp_llm/cpp/model_rpc/PrefillRpcServer.h"
#include "rtp_llm/cpp/model_rpc/QueryConverter.h"
#include "rtp_llm/cpp/multimodal_processor/MultimodalProcessor.h"

namespace rtp_llm {
namespace {

class VideoFramesProcessor: public MultimodalProcessor {
public:
    explicit VideoFramesProcessor(int frames):
        MultimodalProcessor(py::none(), MMModelConfig{true, {{10, 11}, {12, 13}}, false}, 4096), frames_(frames) {}

private:
    ErrorResult<MultimodalOutput>
    MultimodalEmbedding(const std::vector<MultimodalInput>, std::string, int64_t, grpc::ServerContext*) override {
        MultimodalOutput output;
        output.mm_extra_input.emplace();
        for (int i = 0; i < frames_; ++i) {
            output.mm_features.push_back(torch::zeros({1, 1}));
            output.mm_extra_input->push_back(torch::tensor({-53530053, i == 0 ? 1 : 0, 1, 1, 10, 11}, torch::kInt32));
        }
        return output;
    }

    int frames_;
};

TEST(PrefillMultimodalRetryTest, VideoAllocationRetryPreservesOriginalPrompt) {
    // Thirteen/sixteen generated frame pairs plus the enclosing video pair
    // reproduce the observed tags=14/17 on a second expansion of one video.
    for (int frames : {13, 16}) {
        PrefillRpcServer server;
        server.mm_processor_              = std::make_shared<VideoFramesProcessor>(frames);
        auto                         meta = std::make_shared<RpcServerRuntimeMeta>();
        kmonitor::MetricsReporterPtr reporter;
        GenerateInputPB              original;
        original.set_request_id(5353);
        for (int id : {0, 12, 14, 13, 99}) {
            original.add_token_ids(id);
        }
        original.add_multimodal_inputs()->set_multimodal_url("fake://video");
        original.set_group_size(2);
        original.mutable_group_id()->set_value(7);
        original.mutable_generate_config()->mutable_group_timeout()->set_value(100);
        original.mutable_generate_config()->set_max_new_tokens(16);
        const auto             original_wire = original.SerializeAsString();
        RPCContext             rpc{&original, nullptr};
        PrefillGenerateContext context(&server.resource(), rpc, 0, nullptr, reporter, meta);

        std::string first_allocation;
        for (int attempt = 0; attempt < 2; ++attempt) {
            context.reset();
            context.generate_input = QueryConverter::transQuery(&original);
            server.multimodalProcess(context);
            ASSERT_TRUE(context.ok()) << context.error_info.ToString();
            ASSERT_EQ(context.generate_input->multimodal_inputs->size(), frames);
            EXPECT_EQ(original.SerializeAsString(), original_wire);

            auto allocation = server.makeRemoteAllocateInput(context);
            ASSERT_EQ(allocation->token_ids_size(), 4 + frames * 3);
            const auto* ids = context.generate_input->input_ids.data_ptr<int32_t>();
            for (int i = 0; i < allocation->token_ids_size(); ++i) {
                EXPECT_EQ(allocation->token_ids(i), ids[i]);
            }
            EXPECT_EQ(allocation->multimodal_inputs_size(), 1);
            EXPECT_EQ(allocation->group_size(), 0);
            EXPECT_FALSE(allocation->has_group_id());
            EXPECT_FALSE(allocation->generate_config().has_group_timeout());
            EXPECT_EQ(allocation->generate_config().max_new_tokens(), 16);
            if (attempt == 0) {
                // Reusing the forwarded packet for another preprocessing pass
                // reproduces both live errors. It must never replace original.
                auto expanded_again = QueryConverter::transQuery(allocation.get());
                auto mismatch       = server.getMultimodalProcessor()->updateMultimodalFeatures(expanded_again);
                EXPECT_EQ(mismatch.code(), ErrorCode::MM_WRONG_FORMAT_ERROR);
                EXPECT_EQ(mismatch.ToString(),
                          "interleaved multimodal media groups, tags and inputs must match: groups=1, tags="
                              + std::to_string(frames + 1) + ", inputs=1");
                first_allocation     = allocation->SerializeAsString();
                context.error_status = grpc::Status(grpc::StatusCode::UNAVAILABLE, "injected allocation failure");
            } else {
                EXPECT_EQ(allocation->SerializeAsString(), first_allocation);
            }
        }
    }
}

}  // namespace
}  // namespace rtp_llm
