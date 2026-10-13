#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "rtp_llm/cpp/multimodal_processor/transport/rdma/MMRdmaReader.h"
#include "rtp_llm/cpp/multimodal_processor/transport/rdma/MMRdmaExporter.h"
#include "rtp_llm/cpp/rdma_transport/RdmaTransport.h"

namespace rtp_llm {
namespace {

uint64_t alignUp(uint64_t value) {
    return (value + rdma_transport::kRdmaSlotAlign - 1) / rdma_transport::kRdmaSlotAlign
           * rdma_transport::kRdmaSlotAlign;
}

uint64_t tensorBytes(const torch::Tensor& tensor) {
    return static_cast<uint64_t>(tensor.numel()) * tensor.element_size();
}

rdma_transport::TensorDataType dataTypeOf(const torch::Tensor& tensor) {
    switch (tensor.scalar_type()) {
        case torch::kFloat32:
            return rdma_transport::TensorDataType::FLOAT32;
        case torch::kInt32:
            return rdma_transport::TensorDataType::INT32;
        case torch::kHalf:
            return rdma_transport::TensorDataType::FLOAT16;
        case torch::kBFloat16:
            return rdma_transport::TensorDataType::BFLOAT16;
        default:
            return rdma_transport::TensorDataType::FLOAT32;
    }
}

torch::ScalarType torchTypeOf(rdma_transport::TensorDataType dtype) {
    switch (dtype) {
        case rdma_transport::TensorDataType::INT32:
            return torch::kInt32;
        case rdma_transport::TensorDataType::FLOAT16:
            return torch::kHalf;
        case rdma_transport::TensorDataType::BFLOAT16:
            return torch::kBFloat16;
        case rdma_transport::TensorDataType::FLOAT32:
        default:
            return torch::kFloat32;
    }
}

class RecordingExport: public rdma_transport::RdmaExport {
public:
    struct Slot {
        std::string   lease_id;
        torch::Tensor backing;
    };

    size_t                   fail_at = std::numeric_limits<size_t>::max();
    std::vector<Slot>        slots;
    std::vector<std::string> released;

    rdma_transport::RdmaDescriptor create(const std::vector<torch::Tensor>& tensors) override {
        rdma_transport::RdmaDescriptor descriptor;
        if (slots.size() == fail_at) {
            return descriptor;
        }
        uint64_t offset = 0;
        for (const auto& tensor : tensors) {
            rdma_transport::TensorMeta meta;
            meta.shape.assign(tensor.sizes().begin(), tensor.sizes().end());
            meta.dtype  = dataTypeOf(tensor);
            meta.offset = offset;
            meta.nbytes = tensorBytes(tensor);
            descriptor.tensors.push_back(std::move(meta));
            offset += alignUp(tensorBytes(tensor));
        }
        Slot slot;
        slot.lease_id = "lease-" + std::to_string(slots.size());
        slot.backing  = torch::zeros({static_cast<int64_t>(std::max<uint64_t>(offset, 1))}, torch::kUInt8);
        auto* base = slot.backing.data_ptr<uint8_t>();
        for (size_t i = 0; i < tensors.size(); ++i) {
            const auto contiguous = tensors[i].contiguous();
            std::memcpy(base + descriptor.tensors[i].offset, contiguous.data_ptr(), tensorBytes(tensors[i]));
        }
        descriptor.host          = "127.0.0.1";
        descriptor.port          = 1;
        descriptor.remote_addr   = reinterpret_cast<uint64_t>(base);
        descriptor.payload_bytes = offset;
        descriptor.lease_id      = slot.lease_id;
        slots.push_back(std::move(slot));
        return descriptor;
    }

    void release(const std::vector<std::string>& lease_ids) override {
        released.insert(released.end(), lease_ids.begin(), lease_ids.end());
    }

    bool read(const ::RdmaDescriptorPB& descriptor, std::vector<torch::Tensor>* tensors) const {
        const auto it = std::find_if(slots.begin(), slots.end(), [&](const Slot& slot) {
            return slot.lease_id == descriptor.lease_id();
        });
        if (it == slots.end()) {
            return false;
        }
        const auto* base = it->backing.data_ptr<uint8_t>();
        for (const auto& meta : descriptor.tensors()) {
            rdma_transport::TensorDataType dtype;
            switch (meta.data_type()) {
                case ::RDMA_TENSOR_INT32:
                    dtype = rdma_transport::TensorDataType::INT32;
                    break;
                case ::RDMA_TENSOR_FLOAT16:
                    dtype = rdma_transport::TensorDataType::FLOAT16;
                    break;
                case ::RDMA_TENSOR_BFLOAT16:
                    dtype = rdma_transport::TensorDataType::BFLOAT16;
                    break;
                default:
                    dtype = rdma_transport::TensorDataType::FLOAT32;
                    break;
            }
            std::vector<int64_t> shape(meta.shape().begin(), meta.shape().end());
            auto view = torch::from_blob(const_cast<uint8_t*>(base) + meta.offset(), shape,
                                         torch::TensorOptions().dtype(torchTypeOf(dtype)));
            if (tensorBytes(view) != meta.nbytes() || meta.offset() + meta.nbytes() > descriptor.payload_bytes()) {
                return false;
            }
            tensors->push_back(view.clone());
        }
        return true;
    }
};

torch::Tensor rowMarkedTensor(int64_t rows, int64_t cols) {
    return torch::arange(0, rows * cols, torch::kFloat32).reshape({rows, cols});
}

class MMRdmaTransportTest: public ::testing::Test {
protected:
    void SetUp() override {
        exporter = std::make_shared<RecordingExport>();
    }

    MMRdmaExporter makeAdapter(int64_t max_slot_bytes) {
        return MMRdmaExporter(exporter, max_slot_bytes);
    }

    std::shared_ptr<RecordingExport> exporter;
};

TEST_F(MMRdmaTransportTest, packsRolesIntoOneSlotInOrder) {
    auto adapter = makeAdapter(1 << 20);
    std::vector<MMRdmaSlotPB> slots;
    const auto embedding = rowMarkedTensor(4, 8);
    const auto pos_id = torch::zeros({4, 3}, torch::kInt32);
    std::vector<torch::Tensor> extras{torch::zeros({5}, torch::kInt32), torch::zeros({6}, torch::kInt32)};

    ASSERT_TRUE(adapter.exportSlots(embedding, pos_id, extras, &slots));
    ASSERT_EQ(slots.size(), 1u);
    const std::vector<MMRdmaSlotPB::Role> expected{
        MMRdmaSlotPB::EMBEDDING, MMRdmaSlotPB::POS_ID, MMRdmaSlotPB::EXTRA_INPUT, MMRdmaSlotPB::EXTRA_INPUT};
    std::vector<MMRdmaSlotPB::Role> actual;
    for (int i = 0; i < slots[0].roles_size(); ++i) {
        actual.push_back(slots[0].roles(i));
    }
    EXPECT_EQ(actual, expected);
    EXPECT_EQ(slots[0].rdma_descriptor().tensors_size(), 4);
}

TEST_F(MMRdmaTransportTest, rowSplitsAndRoundTripsInOrder) {
    auto adapter = makeAdapter(300);
    std::vector<MMRdmaSlotPB> slots;
    const auto embedding = rowMarkedTensor(6, 25);

    ASSERT_TRUE(adapter.exportSlots(embedding, std::nullopt, {}, &slots));
    ASSERT_EQ(slots.size(), 3u);
    std::vector<torch::Tensor> chunks;
    for (const auto& slot : slots) {
        ASSERT_EQ(slot.rdma_descriptor().tensors_size(), 1);
        EXPECT_EQ(slot.rdma_descriptor().tensors(0).shape(0), 2);
        EXPECT_LE(slot.rdma_descriptor().payload_bytes(), 300u);
        ASSERT_TRUE(exporter->read(slot.rdma_descriptor(), &chunks));
    }
    EXPECT_TRUE(torch::equal(torch::cat(chunks, 0), embedding));
}

TEST_F(MMRdmaTransportTest, rejectsEmbeddingRowLargerThanSlot) {
    auto adapter = makeAdapter(256);
    std::vector<MMRdmaSlotPB> slots;
    EXPECT_FALSE(adapter.exportSlots(rowMarkedTensor(2, 128), std::nullopt, {}, &slots));
    EXPECT_TRUE(slots.empty());
}

TEST_F(MMRdmaTransportTest, rollsBackEarlierLeasesOnCreateFailure) {
    exporter->fail_at = 2;
    auto adapter = makeAdapter(512);
    std::vector<MMRdmaSlotPB> slots;
    EXPECT_FALSE(adapter.exportSlots(rowMarkedTensor(8, 64), std::nullopt, {}, &slots));
    EXPECT_TRUE(slots.empty());
    EXPECT_EQ(exporter->released, (std::vector<std::string>{"lease-0", "lease-1"}));
}

TEST_F(MMRdmaTransportTest, reassemblesChunkedMultimodalOutput) {
    auto adapter = makeAdapter(512);
    std::vector<MMRdmaSlotPB> slots;
    const auto embedding = rowMarkedTensor(7, 64);
    const auto pos_id = torch::zeros({7, 3}, torch::kInt32);
    ASSERT_TRUE(adapter.exportSlots(embedding, pos_id, {}, &slots));

    std::vector<torch::Tensor> tensors;
    std::vector<MMRdmaSlotPB::Role> roles;
    for (const auto& slot : slots) {
        ASSERT_TRUE(exporter->read(slot.rdma_descriptor(), &tensors));
        for (int i = 0; i < slot.roles_size(); ++i) {
            roles.push_back(slot.roles(i));
        }
    }
    MultimodalOutputPB output_pb;
    output_pb.add_split_size(3);
    output_pb.add_split_size(4);
    MultimodalOutput output;
    ASSERT_TRUE(assembleMMRdmaOutput(tensors, roles, &output_pb, &output));
    EXPECT_TRUE(torch::equal(torch::cat(output.mm_features, 0), embedding));
    ASSERT_TRUE(output.mm_position_ids.has_value());
}

TEST_F(MMRdmaTransportTest, rejectsUnsetRoleAndInconsistentSplit) {
    const auto embedding = rowMarkedTensor(4, 4);
    MultimodalOutput output;
    MultimodalOutputPB output_pb;
    output_pb.add_split_size(3);
    EXPECT_FALSE(assembleMMRdmaOutput(
        {embedding}, {MMRdmaSlotPB::ROLE_UNSPECIFIED}, &output_pb, &output));
    EXPECT_FALSE(assembleMMRdmaOutput(
        {embedding}, {MMRdmaSlotPB::EMBEDDING}, &output_pb, &output));
}

}  // namespace
}  // namespace rtp_llm
