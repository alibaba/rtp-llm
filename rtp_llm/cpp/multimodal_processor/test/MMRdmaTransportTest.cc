#include <cstring>
#include <limits>
#include <memory>
#include <vector>

#include "gtest/gtest.h"

#include "rtp_llm/cpp/multimodal_processor/transport/rdma/MMRdmaEncoderOp.h"
#include "rtp_llm/cpp/multimodal_processor/transport/rdma/MMRdmaOutputAssembler.h"
#include "rtp_llm/cpp/multimodal_processor/transport/rdma/MMRdmaTransport.h"

// Covers the slot planning (encoder side) and the manifest reassembly (LLM side) of the ViT
// output RDMA path against a fake transport, so neither needs RDMA hardware. The package's
// test copts disable access control, which is how the tests reach exportSlots() /
// assembleRdmaOutput() without widening the production API.
namespace rtp_llm {

namespace {

uint64_t alignUp(uint64_t x, uint64_t a) {
    return (x + a - 1) / a * a;
}

uint64_t tensorBytes(const torch::Tensor& t) {
    return static_cast<uint64_t>(t.numel()) * t.element_size();
}

// Mirrors the dtype mapping the real transport records in the manifest.
TensorPB::DataType toManifestDataType(const torch::Tensor& t) {
    switch (t.scalar_type()) {
        case torch::kFloat32:
            return TensorPB::FP32;
        case torch::kInt32:
            return TensorPB::INT32;
        case torch::kHalf:
            return TensorPB::FP16;
        case torch::kBFloat16:
            return TensorPB::BF16;
        default:
            return TensorPB::FP32;
    }
}

torch::ScalarType fromManifestDataType(TensorPB::DataType dtype) {
    switch (dtype) {
        case TensorPB::INT32:
            return torch::kInt32;
        case TensorPB::FP16:
            return torch::kHalf;
        case TensorPB::BF16:
            return torch::kBFloat16;
        case TensorPB::FP32:
        default:
            return torch::kFloat32;
    }
}

}  // namespace

// Records every exported slot and released handle. Unlike a clone-back stub, exportEmbedding
// actually copies each tensor's bytes into a contiguous backing buffer at its manifest offset,
// and readEmbedding reconstructs the tensors by slicing that buffer back out per offset/shape/
// dtype -- so the round-trip exercises the offset packing rather than the fake echoing its input.
class RecordingTransport: public MMRdmaTransport {
public:
    struct Slot {
        std::vector<torch::Tensor>        tensors;   // originals, for plan assertions
        std::vector<MMRdmaTensorPB::Role> roles;
        std::string                       handle;
        torch::Tensor                     backing;   // packed bytes, one contiguous uint8 buffer
    };

    // Index of the export call that must fail; defaults to "never".
    size_t                   fail_export_at = std::numeric_limits<size_t>::max();
    std::vector<Slot>        exported;
    std::vector<std::string> released;

    bool exportEmbedding(const std::vector<torch::Tensor>&        tensors,
                         const std::vector<MMRdmaTensorPB::Role>& roles,
                         MMRdmaDescPB*                            desc) override {
        if (exported.size() == fail_export_at) {
            return false;
        }
        Slot                  slot{tensors, roles, "handle-" + std::to_string(exported.size()), {}};
        uint64_t              offset = 0;
        std::vector<uint64_t> offsets;
        for (size_t i = 0; i < tensors.size(); ++i) {
            const uint64_t nbytes = tensorBytes(tensors[i]);
            auto*          entry  = desc->add_tensors();
            entry->set_role(roles[i]);
            for (auto dim : tensors[i].sizes()) {
                entry->add_shape(dim);
            }
            entry->set_data_type(toManifestDataType(tensors[i]));
            entry->set_offset(offset);
            entry->set_nbytes(nbytes);
            offsets.push_back(offset);
            offset += alignUp(nbytes, kMMRdmaSlotAlign);
        }
        // Pack the bytes into one buffer at exactly the advertised offsets, mirroring what the
        // real MR packer does, so a wrong offset would corrupt the round-trip below.
        slot.backing = torch::zeros({static_cast<int64_t>(offset == 0 ? 1 : offset)}, torch::kUInt8);
        auto* base   = slot.backing.data_ptr<uint8_t>();
        for (size_t i = 0; i < tensors.size(); ++i) {
            const auto contiguous = tensors[i].contiguous();
            std::memcpy(base + offsets[i], contiguous.data_ptr(), tensorBytes(tensors[i]));
        }
        desc->set_handle(slot.handle);
        desc->set_nbytes(offset);
        exported.push_back(std::move(slot));
        return true;
    }

    void releaseEmbedding(const std::vector<std::string>& handles) override {
        released.insert(released.end(), handles.begin(), handles.end());
    }

    bool readEmbedding(const MMRdmaDescPB& desc, std::vector<torch::Tensor>* out, int64_t = 0) override {
        for (const auto& slot : exported) {
            if (slot.handle != desc.handle()) {
                continue;
            }
            const auto* base = slot.backing.data_ptr<uint8_t>();
            for (int i = 0; i < desc.tensors_size(); ++i) {
                const auto& entry = desc.tensors(i);
                // Interpret the bytes using only what the manifest declares, so a manifest that
                // lies about shape or dtype cannot be rescued by the original tensor's metadata.
                const auto         dtype = fromManifestDataType(entry.data_type());
                std::vector<int64_t> shape(entry.shape().begin(), entry.shape().end());
                const auto options  = torch::TensorOptions().dtype(dtype);
                int64_t    numel    = 1;
                for (auto dim : shape) {
                    numel *= dim;
                }
                const uint64_t implied = static_cast<uint64_t>(numel) * torch::elementSize(dtype);
                if (implied != entry.nbytes() || entry.offset() + entry.nbytes() > desc.nbytes()) {
                    return false;  // self-inconsistent manifest: the real reader degrades too
                }
                auto flat = torch::from_blob(const_cast<uint8_t*>(base) + entry.offset(), {numel}, options).clone();
                out->push_back(flat.reshape(shape));
            }
            return true;
        }
        return false;
    }
};

namespace {

std::shared_ptr<RecordingTransport> g_transport;

std::shared_ptr<MMRdmaTransport> createRecordingTransport(const MMRdmaConfig&, MMRdmaRole) {
    return g_transport;
}

class ScopedTransportCreator {
public:
    explicit ScopedTransportCreator(MMRdmaTransportCreator creator):
        previous_(registerMMRdmaTransportCreator(creator)) {}
    ~ScopedTransportCreator() {
        registerMMRdmaTransportCreator(previous_);
    }

private:
    MMRdmaTransportCreator previous_;
};

}  // namespace

class MMRdmaTransportTest: public ::testing::Test {
protected:
    void SetUp() override {
        g_transport = std::make_shared<RecordingTransport>();
        previous_creator_ = registerMMRdmaTransportCreator(&createRecordingTransport);
    }

    void TearDown() override {
        registerMMRdmaTransportCreator(previous_creator_);
        g_transport.reset();
    }

    MMRdmaTransportCreator previous_creator_ = nullptr;

    static MMRdmaConfig configWithMaxSlot(int64_t max_slot_bytes) {
        MMRdmaConfig config;
        config.max_slot_bytes = max_slot_bytes;
        return config;
    }

    // [rows, cols] float32 tensor whose first element of row i is i * cols, so a test can tell
    // which rows ended up in which chunk.
    static torch::Tensor rowMarkedTensor(int64_t rows, int64_t cols) {
        return torch::arange(0, rows * cols, torch::kFloat32).reshape({rows, cols});
    }

    static std::vector<MMRdmaTensorPB::Role> rolesOf(const RecordingTransport::Slot& slot) {
        return slot.roles;
    }
};

TEST(MMRdmaTransportFactoryTest, returnsNullWithoutImplementation) {
    ScopedTransportCreator creator(nullptr);
    MMRdmaConfig config;
    EXPECT_EQ(createMMRdmaTransport(config, MMRdmaRole::LLM_CLIENT), nullptr);
}

TEST(MMRdmaTransportFactoryTest, usesRegisteredImplementation) {
    ScopedTransportCreator creator(&createRecordingTransport);
    g_transport = std::make_shared<RecordingTransport>();
    MMRdmaConfig config;

    EXPECT_EQ(createMMRdmaTransport(config, MMRdmaRole::LLM_CLIENT), g_transport);
    g_transport.reset();
}

TEST_F(MMRdmaTransportTest, packsEveryRoleIntoOneSlotInOrder) {
    MMRdmaEncoderOp           op(configWithMaxSlot(1 << 20));
    std::vector<MMRdmaDescPB> descs;
    const auto                embedding = rowMarkedTensor(4, 8);
    const auto                pos_id    = torch::zeros({4, 3}, torch::kInt32);
    std::vector<torch::Tensor> extras{torch::zeros({5}, torch::kInt32), torch::zeros({6}, torch::kInt32)};

    ASSERT_TRUE(op.exportSlots(embedding, pos_id, extras, &descs));
    ASSERT_EQ(descs.size(), 1u);
    EXPECT_EQ(g_transport->exported.size(), 1u);
    const std::vector<MMRdmaTensorPB::Role> expected{MMRdmaTensorPB::EMBEDDING,
                                                     MMRdmaTensorPB::POS_ID,
                                                     MMRdmaTensorPB::EXTRA_INPUT,
                                                     MMRdmaTensorPB::EXTRA_INPUT};
    EXPECT_EQ(rolesOf(g_transport->exported[0]), expected);
    EXPECT_EQ(descs[0].tensors_size(), 4);
    // NOTE: offset alignment and in-slot bounds are the *fake's* packing contract, not something
    // exportSlots decides -- the production packer lives in the internal transport. What the
    // encoder actually determines is the grouping and role order asserted above. These are kept
    // as a self-check that the fake stays a faithful stand-in for the manifest reader below.
    for (const auto& tensor : descs[0].tensors()) {
        EXPECT_EQ(tensor.offset() % kMMRdmaSlotAlign, 0u);
        EXPECT_LE(tensor.offset() + tensor.nbytes(), descs[0].nbytes());
    }
    EXPECT_TRUE(g_transport->released.empty());
}

TEST_F(MMRdmaTransportTest, rowSplitsEmbeddingAcrossSlotsPreservingOrder) {
    // 64 float32 columns => 256 B per row, so a 512 B slot holds exactly 2 rows.
    MMRdmaEncoderOp           op(configWithMaxSlot(512));
    std::vector<MMRdmaDescPB> descs;
    const auto                embedding = rowMarkedTensor(8, 64);

    ASSERT_TRUE(op.exportSlots(embedding, std::nullopt, {}, &descs));
    ASSERT_EQ(descs.size(), 4u);
    for (size_t slot = 0; slot < descs.size(); ++slot) {
        ASSERT_EQ(descs[slot].tensors_size(), 1);
        EXPECT_EQ(descs[slot].tensors(0).role(), MMRdmaTensorPB::EMBEDDING);
        EXPECT_EQ(descs[slot].tensors(0).shape(0), 2);
        EXPECT_LE(descs[slot].nbytes(), 512u);
        // Chunk k must start at row 2k, i.e. carry value 2k * 64.
        const auto& chunk = g_transport->exported[slot].tensors[0];
        EXPECT_FLOAT_EQ(chunk.flatten()[0].item<float>(), static_cast<float>(slot * 2 * 64));
    }
}

TEST_F(MMRdmaTransportTest, roundsRowSplitBudgetDownToTheSlotAlignment) {
    // 25 float32 columns => 100 B per row, which is not a multiple of the 256 B slot alignment.
    // A 300 B cap floors to 256 B, so a chunk holds 2 rows (200 B -> 256 B once padded). Budgeting
    // against the raw cap instead yields 3-row chunks whose padded occupancy is 512 B, i.e. slots
    // larger than the configured maximum.
    MMRdmaEncoderOp           op(configWithMaxSlot(300));
    std::vector<MMRdmaDescPB> descs;
    const auto                embedding = rowMarkedTensor(6, 25);

    ASSERT_TRUE(op.exportSlots(embedding, std::nullopt, {}, &descs));
    ASSERT_EQ(descs.size(), 3u);
    std::vector<torch::Tensor> read_tensors;
    for (const auto& desc : descs) {
        ASSERT_EQ(desc.tensors_size(), 1);
        EXPECT_EQ(desc.tensors(0).shape(0), 2);
        EXPECT_LE(desc.nbytes(), 300u);
        std::vector<torch::Tensor> chunk;
        ASSERT_TRUE(g_transport->readEmbedding(desc, &chunk));
        read_tensors.insert(read_tensors.end(), chunk.begin(), chunk.end());
    }
    EXPECT_TRUE(torch::equal(torch::cat(read_tensors, 0), embedding));
}

TEST_F(MMRdmaTransportTest, keepsSingleSlotWhenFootprintExactlyFits) {
    MMRdmaEncoderOp           op(configWithMaxSlot(1024));
    std::vector<MMRdmaDescPB> descs;
    const auto                embedding = rowMarkedTensor(4, 64);  // 1024 B, already 256 B aligned

    ASSERT_TRUE(op.exportSlots(embedding, std::nullopt, {}, &descs));
    ASSERT_EQ(descs.size(), 1u);
    EXPECT_EQ(descs[0].tensors(0).shape(0), 4);
}

TEST_F(MMRdmaTransportTest, treatsZeroMaxSlotAsUnlimited) {
    MMRdmaEncoderOp           op(configWithMaxSlot(0));
    std::vector<MMRdmaDescPB> descs;

    ASSERT_TRUE(op.exportSlots(rowMarkedTensor(1024, 64), std::nullopt, {}, &descs));
    EXPECT_EQ(descs.size(), 1u);
}

TEST_F(MMRdmaTransportTest, rejectsEmbeddingRowLargerThanOneSlot) {
    MMRdmaEncoderOp           op(configWithMaxSlot(256));
    std::vector<MMRdmaDescPB> descs;

    // 128 float32 columns => 512 B per row, which no 256 B slot can ever hold.
    EXPECT_FALSE(op.exportSlots(rowMarkedTensor(2, 128), std::nullopt, {}, &descs));
    EXPECT_TRUE(descs.empty());
    EXPECT_TRUE(g_transport->exported.empty());
}

TEST_F(MMRdmaTransportTest, rejectsPosIdLargerThanOneSlot) {
    MMRdmaEncoderOp           op(configWithMaxSlot(512));
    std::vector<MMRdmaDescPB> descs;
    const auto                pos_id = torch::zeros({4, 64}, torch::kInt32);  // 1024 B

    EXPECT_FALSE(op.exportSlots(rowMarkedTensor(1, 4), pos_id, {}, &descs));
    EXPECT_TRUE(descs.empty());
    // Nothing is exported before the whole plan is known, so there is nothing to roll back.
    EXPECT_TRUE(g_transport->exported.empty());
    EXPECT_TRUE(g_transport->released.empty());
}

TEST_F(MMRdmaTransportTest, rejectsExtraInputLargerThanOneSlot) {
    MMRdmaEncoderOp           op(configWithMaxSlot(512));
    std::vector<MMRdmaDescPB> descs;
    std::vector<torch::Tensor> extras{torch::zeros({256}, torch::kInt32)};

    EXPECT_FALSE(op.exportSlots(rowMarkedTensor(1, 4), std::nullopt, extras, &descs));
    EXPECT_TRUE(descs.empty());
    EXPECT_TRUE(g_transport->exported.empty());
}

TEST_F(MMRdmaTransportTest, releasesAlreadyExportedSlotsWhenOneSlotFails) {
    g_transport->fail_export_at = 2;
    MMRdmaEncoderOp           op(configWithMaxSlot(512));
    std::vector<MMRdmaDescPB> descs;

    EXPECT_FALSE(op.exportSlots(rowMarkedTensor(8, 64), std::nullopt, {}, &descs));
    EXPECT_TRUE(descs.empty());
    const std::vector<std::string> expected{"handle-0", "handle-1"};
    EXPECT_EQ(g_transport->released, expected);
}

TEST_F(MMRdmaTransportTest, reassemblesChunkedEmbeddingAndSplitsPerImage) {
    const auto                 chunk0 = rowMarkedTensor(2, 4);
    const auto                 chunk1 = rowMarkedTensor(2, 4) + 100.0f;
    std::vector<torch::Tensor>  tensors{chunk0,
                                       chunk1,
                                       torch::zeros({4, 3}, torch::kInt32),
                                       torch::zeros({5}, torch::kInt32),
                                       torch::zeros({6}, torch::kInt32)};
    std::vector<MMRdmaTensorPB::Role> roles{MMRdmaTensorPB::EMBEDDING,
                                            MMRdmaTensorPB::EMBEDDING,
                                            MMRdmaTensorPB::POS_ID,
                                            MMRdmaTensorPB::EXTRA_INPUT,
                                            MMRdmaTensorPB::EXTRA_INPUT};
    MultimodalOutputPB output_pb;
    output_pb.add_split_size(1);
    output_pb.add_split_size(3);

    MultimodalOutput output;
    ASSERT_TRUE(assembleMMRdmaOutput(tensors, roles, &output_pb, &output));
    ASSERT_EQ(output.mm_features.size(), 2u);
    EXPECT_EQ(output.mm_features[0].size(0), 1);
    EXPECT_EQ(output.mm_features[1].size(0), 3);
    // The chunks must be concatenated in order, so the second image starts inside chunk0.
    EXPECT_FLOAT_EQ(output.mm_features[0].flatten()[0].item<float>(), 0.0f);
    EXPECT_FLOAT_EQ(output.mm_features[1].flatten()[0].item<float>(), 4.0f);
    ASSERT_TRUE(output.mm_position_ids.has_value());
    EXPECT_EQ(output.mm_position_ids.value().size(), 2u);
    ASSERT_TRUE(output.mm_extra_input.has_value());
    EXPECT_EQ(output.mm_extra_input.value().size(), 2u);
}

TEST_F(MMRdmaTransportTest, rejectsInconsistentManifestInsteadOfAsserting) {
    const auto       embedding = rowMarkedTensor(4, 4);
    MultimodalOutput output;

    {  // split_size is required
        std::vector<torch::Tensor> tensors{embedding};
        std::vector<MMRdmaTensorPB::Role> roles{MMRdmaTensorPB::EMBEDDING};
        MultimodalOutputPB output_pb;
        output.mm_features = {torch::ones({1})};
        EXPECT_FALSE(assembleMMRdmaOutput(tensors, roles, &output_pb, &output));
        ASSERT_EQ(output.mm_features.size(), 1u);
    }
    {  // position ids must have the same row count as embedding
        std::vector<torch::Tensor> tensors{embedding, torch::zeros({3, 2}, torch::kInt32)};
        std::vector<MMRdmaTensorPB::Role> roles{MMRdmaTensorPB::EMBEDDING, MMRdmaTensorPB::POS_ID};
        MultimodalOutputPB output_pb;
        output_pb.add_split_size(4);
        EXPECT_FALSE(assembleMMRdmaOutput(tensors, roles, &output_pb, &output));
    }
    {  // unknown future roles are rejected without partially replacing output
        std::vector<torch::Tensor> tensors{embedding, torch::zeros({1})};
        std::vector<MMRdmaTensorPB::Role> roles{MMRdmaTensorPB::EMBEDDING,
                                                static_cast<MMRdmaTensorPB::Role>(99)};
        MultimodalOutputPB output_pb;
        output_pb.add_split_size(4);
        output.mm_features = {torch::ones({1})};
        EXPECT_FALSE(assembleMMRdmaOutput(tensors, roles, &output_pb, &output));
        ASSERT_EQ(output.mm_features.size(), 1u);
        EXPECT_EQ(output.mm_features[0].item<float>(), 1.0f);
    }

    {  // split_size sum does not match the embedding rows
        std::vector<torch::Tensor>        tensors{embedding};
        std::vector<MMRdmaTensorPB::Role> roles{MMRdmaTensorPB::EMBEDDING};
        MultimodalOutputPB                output_pb;
        output_pb.add_split_size(2);
        output_pb.add_split_size(3);
        EXPECT_FALSE(assembleMMRdmaOutput(tensors, roles, &output_pb, &output));
    }
    {  // no embedding at all
        std::vector<torch::Tensor>        tensors{torch::zeros({4, 3}, torch::kInt32)};
        std::vector<MMRdmaTensorPB::Role> roles{MMRdmaTensorPB::POS_ID};
        MultimodalOutputPB                output_pb;
        output_pb.add_split_size(4);
        EXPECT_FALSE(assembleMMRdmaOutput(tensors, roles, &output_pb, &output));
    }
    {  // two position id tensors
        std::vector<torch::Tensor>        tensors{embedding,
                                           torch::zeros({4, 3}, torch::kInt32),
                                           torch::zeros({4, 3}, torch::kInt32)};
        std::vector<MMRdmaTensorPB::Role> roles{MMRdmaTensorPB::EMBEDDING,
                                                MMRdmaTensorPB::POS_ID,
                                                MMRdmaTensorPB::POS_ID};
        MultimodalOutputPB                output_pb;
        output_pb.add_split_size(4);
        EXPECT_FALSE(assembleMMRdmaOutput(tensors, roles, &output_pb, &output));
    }
    {  // manifest and payload disagree on how many tensors there are
        std::vector<torch::Tensor>        tensors{embedding};
        std::vector<MMRdmaTensorPB::Role> roles{MMRdmaTensorPB::EMBEDDING, MMRdmaTensorPB::POS_ID};
        MultimodalOutputPB                output_pb;
        output_pb.add_split_size(4);
        EXPECT_FALSE(assembleMMRdmaOutput(tensors, roles, &output_pb, &output));
    }
    {  // one extra_input per image is required
        std::vector<torch::Tensor>        tensors{embedding,
                                           torch::zeros({5}, torch::kInt32),
                                           torch::zeros({5}, torch::kInt32),
                                           torch::zeros({5}, torch::kInt32)};
        std::vector<MMRdmaTensorPB::Role> roles{MMRdmaTensorPB::EMBEDDING,
                                                MMRdmaTensorPB::EXTRA_INPUT,
                                                MMRdmaTensorPB::EXTRA_INPUT,
                                                MMRdmaTensorPB::EXTRA_INPUT};
        MultimodalOutputPB                output_pb;
        output_pb.add_split_size(2);
        output_pb.add_split_size(2);
        EXPECT_FALSE(assembleMMRdmaOutput(tensors, roles, &output_pb, &output));
    }
}

TEST_F(MMRdmaTransportTest, roundTripsChunkedOutputThroughTheManifest) {
    // Export chunked, then replay what the LLM does minus gRPC: read every slot in descriptor
    // order, collect the manifest roles, reassemble.
    MMRdmaEncoderOp           op(configWithMaxSlot(512));
    std::vector<MMRdmaDescPB> descs;
    const auto                embedding = rowMarkedTensor(8, 64);
    const auto                pos_id    = torch::zeros({8, 3}, torch::kInt32);

    ASSERT_TRUE(op.exportSlots(embedding, pos_id, {}, &descs));
    ASSERT_GT(descs.size(), 1u);

    std::vector<torch::Tensor>        read_tensors;
    std::vector<MMRdmaTensorPB::Role> roles;
    for (const auto& desc : descs) {
        std::vector<torch::Tensor> chunk;
        ASSERT_TRUE(g_transport->readEmbedding(desc, &chunk));
        for (int i = 0; i < desc.tensors_size(); ++i) {
            roles.push_back(desc.tensors(i).role());
        }
        read_tensors.insert(read_tensors.end(), chunk.begin(), chunk.end());
    }

    MultimodalOutputPB output_pb;
    output_pb.add_split_size(3);
    output_pb.add_split_size(5);

    MultimodalOutput output;
    ASSERT_TRUE(assembleMMRdmaOutput(read_tensors, roles, &output_pb, &output));
    ASSERT_EQ(output.mm_features.size(), 2u);
    EXPECT_EQ(output.mm_features[0].size(0), 3);
    EXPECT_EQ(output.mm_features[1].size(0), 5);
    EXPECT_TRUE(torch::equal(torch::cat({output.mm_features[0], output.mm_features[1]}, 0), embedding));
    ASSERT_TRUE(output.mm_position_ids.has_value());
    EXPECT_EQ(output.mm_position_ids.value().size(), 2u);
}

TEST_F(MMRdmaTransportTest, rowSplitLeavesUnevenTailChunkAndRoundTripsExactly) {
    // 64 f32 cols => 256 B/row; a 512 B slot holds 2 rows, so 7 rows split 2,2,2,1.
    MMRdmaEncoderOp           op(configWithMaxSlot(512));
    std::vector<MMRdmaDescPB> descs;
    const auto                embedding = rowMarkedTensor(7, 64);

    ASSERT_TRUE(op.exportSlots(embedding, std::nullopt, {}, &descs));
    ASSERT_EQ(descs.size(), 4u);
    const std::vector<int64_t> expected_rows{2, 2, 2, 1};
    std::vector<torch::Tensor> read_tensors;
    for (size_t slot = 0; slot < descs.size(); ++slot) {
        ASSERT_EQ(descs[slot].tensors_size(), 1);
        EXPECT_EQ(descs[slot].tensors(0).shape(0), expected_rows[slot]);
        std::vector<torch::Tensor> chunk;
        ASSERT_TRUE(g_transport->readEmbedding(descs[slot], &chunk));
        read_tensors.insert(read_tensors.end(), chunk.begin(), chunk.end());
    }
    // Reconstructing straight from the packed buffers must reproduce the original tensor.
    EXPECT_TRUE(torch::equal(torch::cat(read_tensors, 0), embedding));
}

TEST_F(MMRdmaTransportTest, rejectsZeroDimEmbeddingThatCannotBeRowSplit) {
    // A 0-dim tensor has no row axis; forced past a single slot it hits the rows<=0 guard and
    // must fall back to inline bytes rather than being packed as a bogus one-row embedding.
    MMRdmaEncoderOp           op(configWithMaxSlot(128));  // < one 0-dim footprint (256 B aligned)
    std::vector<MMRdmaDescPB> descs;

    EXPECT_FALSE(op.exportSlots(torch::tensor(1.0f), std::nullopt, {}, &descs));
    EXPECT_TRUE(descs.empty());
    EXPECT_TRUE(g_transport->exported.empty());
}

TEST_F(MMRdmaTransportTest, rejectsManifestWithUnsetRole) {
    // proto3 default: an encoder that forgot to set role yields ROLE_UNSPECIFIED, which the
    // assembler must treat as a protocol error rather than a stray EMBEDDING chunk.
    std::vector<torch::Tensor>        tensors{rowMarkedTensor(4, 4)};
    std::vector<MMRdmaTensorPB::Role> roles{MMRdmaTensorPB::ROLE_UNSPECIFIED};
    MultimodalOutputPB                output_pb;
    output_pb.add_split_size(4);

    MultimodalOutput output;
    output.mm_features = {torch::ones({1})};
    EXPECT_FALSE(assembleMMRdmaOutput(tensors, roles, &output_pb, &output));
    ASSERT_EQ(output.mm_features.size(), 1u);
    EXPECT_EQ(output.mm_features[0].item<float>(), 1.0f);
}

TEST_F(MMRdmaTransportTest, rejectsManifestWhoseShapeOrDtypeContradictsNbytes) {
    MMRdmaEncoderOp           op(configWithMaxSlot(1 << 20));
    std::vector<MMRdmaDescPB> descs;

    ASSERT_TRUE(op.exportSlots(rowMarkedTensor(4, 8), std::nullopt, {}, &descs));
    ASSERT_EQ(descs.size(), 1u);

    {  // shape no longer implies the declared nbytes
        auto corrupted = descs[0];
        corrupted.mutable_tensors(0)->set_shape(1, 9);
        std::vector<torch::Tensor> out;
        EXPECT_FALSE(g_transport->readEmbedding(corrupted, &out));
        EXPECT_TRUE(out.empty());
    }
    {  // dtype width no longer implies the declared nbytes
        // BF16 rather than INT32: int32 is also 4 bytes wide and would not contradict nbytes.
        auto corrupted = descs[0];
        corrupted.mutable_tensors(0)->set_data_type(TensorPB::BF16);
        std::vector<torch::Tensor> out;
        EXPECT_FALSE(g_transport->readEmbedding(corrupted, &out));
        EXPECT_TRUE(out.empty());
    }
    // The untouched descriptor still round-trips, so the rejections above are the mutation's doing.
    std::vector<torch::Tensor> ok;
    EXPECT_TRUE(g_transport->readEmbedding(descs[0], &ok));
    ASSERT_EQ(ok.size(), 1u);
    EXPECT_TRUE(torch::equal(ok[0], rowMarkedTensor(4, 8)));
}

TEST_F(MMRdmaTransportTest, malformedManifestThatMakesTorchThrowReturnsFalseWithoutMutating) {
    MultimodalOutput output;
    output.mm_features = {torch::ones({1})};  // sentinel: must survive a rejected assemble

    {  // two EMBEDDING chunks with different column counts => torch::cat throws
        std::vector<torch::Tensor>        tensors{rowMarkedTensor(2, 4), rowMarkedTensor(2, 5)};
        std::vector<MMRdmaTensorPB::Role> roles{MMRdmaTensorPB::EMBEDDING, MMRdmaTensorPB::EMBEDDING};
        MultimodalOutputPB                output_pb;
        output_pb.add_split_size(4);
        EXPECT_FALSE(assembleMMRdmaOutput(tensors, roles, &output_pb, &output));
        ASSERT_EQ(output.mm_features.size(), 1u);
        EXPECT_EQ(output.mm_features[0].item<float>(), 1.0f);
    }
    {  // split sizes sum to the row count but contain a negative entry => torch::split throws
        std::vector<torch::Tensor>        tensors{rowMarkedTensor(4, 4)};
        std::vector<MMRdmaTensorPB::Role> roles{MMRdmaTensorPB::EMBEDDING};
        MultimodalOutputPB                output_pb;
        output_pb.add_split_size(5);
        output_pb.add_split_size(-1);
        EXPECT_FALSE(assembleMMRdmaOutput(tensors, roles, &output_pb, &output));
        ASSERT_EQ(output.mm_features.size(), 1u);
        EXPECT_EQ(output.mm_features[0].item<float>(), 1.0f);
    }
}

}  // namespace rtp_llm
