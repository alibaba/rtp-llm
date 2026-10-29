#include <gtest/gtest.h>

#include <limits>

#include <c10/cuda/CUDAGuard.h>
#include <torch/torch.h>

#include "rtp_llm/cpp/config/RdmaConfig.h"
#include "rtp_llm/cpp/multimodal_processor/transport/rdma/MMRdmaReader.h"
#include "rtp_llm/cpp/rdma_transport/RdmaTransport.h"

namespace rtp_llm::rdma_transport {
namespace {

class ReleasingControlClient: public MMControlClient {
public:
    explicit ReleasingControlClient(std::shared_ptr<RdmaExport> exporter): exporter_(std::move(exporter)) {}

    ErrorResult<MultimodalOutputPB>
    request(const std::string&, MultimodalInputsPB&, DeadlineBudget&) override {
        return ErrorInfo(ErrorCode::UNKNOWN_ERROR, "request is not used by the RDMA smoke test");
    }

    void release(const std::string&, const std::vector<std::string>& handles, DeadlineBudget&) override {
        releaseAsync("", std::vector<std::string>(handles));
    }

    void releaseAsync(std::string, std::vector<std::string> handles) override {
        released.push_back(handles);
        exporter_->release(handles);
    }

    std::vector<std::vector<std::string>> released;

private:
    std::shared_ptr<RdmaExport> exporter_;
};

TEST(BarexRdmaTransportSmokeTest, ExportReadReleaseAcrossGpus) {
    ASSERT_TRUE(torch::cuda::is_available());
    ASSERT_GE(torch::cuda::device_count(), 2);
    ASSERT_TRUE(hasRdmaImplementation());

    RdmaConfig config;
    config.connect_timeout_ms = 3000;
    config.read_timeout_ms    = 10000;
    config.qp_count           = 2;
    config.max_slot_bytes     = 4 * 1024 * 1024;

    std::shared_ptr<RdmaExport> exporter;
    RdmaDescriptor             descriptor;
    torch::Tensor              expected_float;
    torch::Tensor              expected_int;
    RdmaDescriptor             multimodal_descriptor;
    torch::Tensor              expected_embedding;
    torch::Tensor              expected_position_ids;
    std::vector<torch::Tensor> expected_extra_inputs;
    {
        at::cuda::CUDAGuard export_guard(0);
        exporter = createRdmaExport(config, 0);
        ASSERT_NE(exporter, nullptr);

        auto float_tensor = torch::arange(
                                0, 32 * 64, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, 0))
                                .reshape({32, 64});
        auto int_tensor = torch::arange(
                              0, 128, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, 0))
                              .reshape({16, 8});
        expected_float = float_tensor.cpu();
        expected_int   = int_tensor.cpu();
        descriptor     = exporter->create({float_tensor, int_tensor});

        auto embedding = torch::arange(
                             0, 5 * 4, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, 0))
                             .reshape({5, 4});
        auto position_ids = torch::arange(
                                0, 5 * 3, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, 0))
                                .reshape({5, 3});
        std::vector<torch::Tensor> extra_inputs{
            torch::arange(0, 4, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, 0)),
            torch::arange(0, 6, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, 0)),
        };
        expected_embedding     = embedding.cpu();
        expected_position_ids  = position_ids.cpu();
        expected_extra_inputs  = {extra_inputs[0].cpu(), extra_inputs[1].cpu()};
        multimodal_descriptor  = exporter->create({embedding, position_ids, extra_inputs[0], extra_inputs[1]});
    }

    ASSERT_FALSE(descriptor.lease_id.empty());
    ASSERT_FALSE(descriptor.host.empty());
    ASSERT_NE(descriptor.port, 0u);
    ASSERT_NE(descriptor.remote_addr, 0u);
    ASSERT_EQ(descriptor.tensors.size(), 2u);

    std::shared_ptr<RdmaRead> reader;
    {
        at::cuda::CUDAGuard read_guard(1);
        reader = createRdmaRead(config, 1);
        ASSERT_NE(reader, nullptr);

        auto duplicate = reader->read({descriptor, descriptor}, config.read_timeout_ms);
        EXPECT_FALSE(duplicate.status.ok());
        EXPECT_EQ(duplicate.status.code(), ErrorCode::INVALID_PARAMS);

        auto overflowing = descriptor;
        overflowing.remote_addr =
            std::numeric_limits<uint64_t>::max() - overflowing.payload_bytes + 1;
        auto overflow_result = reader->read({overflowing}, config.read_timeout_ms);
        EXPECT_FALSE(overflow_result.status.ok());
        EXPECT_EQ(overflow_result.status.code(), ErrorCode::INVALID_PARAMS);

        auto overlapping = descriptor;
        ASSERT_GE(overlapping.tensors.size(), 2u);
        overlapping.tensors[1].offset = overlapping.tensors[0].offset;
        auto overlap_result = reader->read({overlapping}, config.read_timeout_ms);
        EXPECT_FALSE(overlap_result.status.ok());
        EXPECT_EQ(overlap_result.status.code(), ErrorCode::INVALID_PARAMS);

        auto result = reader->read({descriptor}, config.read_timeout_ms);
        ASSERT_TRUE(result.status.ok()) << result.status.ToString();
        ASSERT_EQ(result.tensors.size(), 2u);
        EXPECT_TRUE(result.tensors[0].is_cuda());
        EXPECT_TRUE(result.tensors[1].is_cuda());
        EXPECT_EQ(result.tensors[0].device().index(), 1);
        EXPECT_EQ(result.tensors[1].device().index(), 1);
        EXPECT_EQ(result.tensors[0].scalar_type(), torch::kFloat32);
        EXPECT_EQ(result.tensors[1].scalar_type(), torch::kInt32);
        EXPECT_TRUE(torch::equal(result.tensors[0].cpu(), expected_float));
        EXPECT_TRUE(torch::equal(result.tensors[1].cpu(), expected_int));

        MultimodalOutputPB receipt;
        auto*              slot = receipt.add_output_rdma_slots();
        toProto(multimodal_descriptor, slot->mutable_rdma_descriptor());
        slot->add_roles(MMRdmaSlotPB::EMBEDDING);
        slot->add_roles(MMRdmaSlotPB::POS_ID);
        slot->add_roles(MMRdmaSlotPB::EXTRA_INPUT);
        slot->add_roles(MMRdmaSlotPB::EXTRA_INPUT);
        receipt.add_split_size(2);
        receipt.add_split_size(3);

        MMRdmaReader           production_reader(reader, config);
        ReleasingControlClient control(exporter);
        DeadlineBudget         budget(config.read_timeout_ms);
        const std::string      endpoint = "barex-smoke";
        DeliveryContext        context{endpoint, budget, control};
        auto                   consume_result = production_reader.consume(receipt, context);

        ASSERT_TRUE(consume_result.succeeded());
        auto& output = consume_result.output();
        ASSERT_EQ(output.mm_features.size(), 2u);
        EXPECT_TRUE(torch::equal(torch::cat(output.mm_features, 0).cpu(), expected_embedding));
        ASSERT_TRUE(output.mm_position_ids.has_value());
        EXPECT_TRUE(torch::equal(torch::cat(*output.mm_position_ids, 0), expected_position_ids));
        ASSERT_TRUE(output.mm_extra_input.has_value());
        ASSERT_EQ(output.mm_extra_input->size(), 2u);
        EXPECT_TRUE(torch::equal((*output.mm_extra_input)[0].cpu(), expected_extra_inputs[0]));
        EXPECT_TRUE(torch::equal((*output.mm_extra_input)[1].cpu(), expected_extra_inputs[1]));
        ASSERT_EQ(control.released.size(), 1u);
        EXPECT_EQ(control.released[0], std::vector<std::string>({multimodal_descriptor.lease_id}));

        reader.reset();
    }

    {
        at::cuda::CUDAGuard export_guard(0);
        exporter->release({descriptor.lease_id});
        exporter->release({descriptor.lease_id});
        exporter.reset();
    }
}

}  // namespace
}  // namespace rtp_llm::rdma_transport
