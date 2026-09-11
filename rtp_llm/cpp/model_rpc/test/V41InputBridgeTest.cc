#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/cpp/cache/KVCacheHashUtil.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"
#include "rtp_llm/cpp/model_rpc/QueryConverter.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"

namespace rtp_llm {
namespace {

GenerateInputPB imageRequest() {
    GenerateInputPB request;
    request.mutable_generate_config()->set_max_new_tokens(256);
    auto* prepared = request.mutable_v41_inputs();
    prepared->set_schema_version(1);
    for (int32_t token : {7, 8, 9, 129264, 129264, 129264, 129264, 10, 11, 12, 13}) {
        request.add_token_ids(token);
    }
    for (int32_t kind : {-1, -1, -1, 0, 1, 2, 3, -1, -1, -1, -1}) {
        prepared->add_token_types(kind);
        prepared->add_image_mask(kind != -1);
    }
    auto* image = prepared->add_images();
    image->set_start(3);
    image->set_n_vit_h(1);
    image->set_n_vit_w(1);
    image->set_content_sha256(std::string(64, 'a'));
    image->set_processor_identity(std::string(64, 'b'));
    for (int32_t kind : {0, 1, 2, 3}) {
        image->add_types(kind);
    }
    QueryConverter::transTensorPB(image->mutable_patches(), torch::ones({1, 3, 14, 14}, torch::kBFloat16));
    return request;
}

struct Rows {
    torch::Tensor types;
    torch::Tensor valid;
    torch::Tensor history;
    torch::Tensor history_valid;
};

Rows readRows(CompleteTokenIds& tokens, int begin, int count, int batch = 0) {
    Rows rows{torch::empty({count}, torch::kInt32),
              torch::empty({count}, torch::kBool),
              torch::empty({count, 3}, torch::kInt32),
              torch::empty({count, 3}, torch::kBool)};
    tokens.writeV41Rows(batch,
                        begin,
                        count,
                        rows.types.data_ptr<int32_t>(),
                        rows.valid.data_ptr<bool>(),
                        rows.history.data_ptr<int32_t>(),
                        rows.history_valid.data_ptr<bool>());
    return rows;
}

class V41InputBridgeTest: public DeviceTestBase {};

TEST_F(V41InputBridgeTest, RpcOwnsExactCanonicalPayloadAfterProtoLifetime) {
    auto wire  = imageRequest();
    auto input = QueryConverter::transQuery(&wire);
    wire.Clear();
    ASSERT_TRUE(input->v41_inputs);
    EXPECT_NO_THROW(input->v41_inputs->validate(input->input_ids));
    ASSERT_EQ(input->v41_inputs->images.size(), 1);
    EXPECT_EQ(input->v41_inputs->images[0].content_sha256, std::string(64, 'a'));
    EXPECT_EQ(input->v41_inputs->images[0].processor_identity, std::string(64, 'b'));
    EXPECT_TRUE(torch::equal(input->v41_inputs->images[0].patches, torch::ones({1, 3, 14, 14}, torch::kBFloat16)));
    EXPECT_TRUE(torch::equal(input->input_ids.slice(0, 3, 7), torch::full({4}, 129264, torch::kInt32)));
    EXPECT_EQ(input->v41_inputs->image_mask.sum().item<int64_t>(), 4);
    EXPECT_TRUE(input->multimodal_inputs->empty());
}

TEST_F(V41InputBridgeTest, RpcRejectsMalformedImageMetadataAndPayload) {
    const auto valid   = imageRequest();
    auto       request = valid;
    request.mutable_v41_inputs()->set_schema_version(2);
    EXPECT_THROW(QueryConverter::transQuery(&request), std::runtime_error);
    request = valid;
    request.mutable_v41_inputs()->set_image_mask(3, false);
    EXPECT_THROW(QueryConverter::transQuery(&request), std::runtime_error);
    request = valid;
    request.set_token_ids(4, 200000);
    EXPECT_THROW(QueryConverter::transQuery(&request), std::runtime_error);
    request = valid;
    request.mutable_v41_inputs()->mutable_images(0)->set_types(2, 1);
    EXPECT_THROW(QueryConverter::transQuery(&request), std::runtime_error);
    request = valid;
    request.mutable_v41_inputs()->mutable_images(0)->set_content_sha256("missing");
    EXPECT_THROW(QueryConverter::transQuery(&request), std::runtime_error);
    request = valid;
    request.mutable_v41_inputs()->mutable_images(0)->mutable_patches()->mutable_bf16_data()->pop_back();
    EXPECT_THROW(QueryConverter::transQuery(&request), std::runtime_error);
    request = valid;
    request.add_multimodal_inputs()->set_multimodal_url("https://unused.invalid/image");
    EXPECT_THROW(QueryConverter::transQuery(&request), std::runtime_error);
    request = valid;
    request.mutable_generate_config()->set_max_new_tokens(1048576);
    EXPECT_THROW(QueryConverter::transQuery(&request), std::runtime_error);
}

TEST_F(V41InputBridgeTest, CanonicalHistorySurvivesChunksImagesAndRequestCopies) {
    auto             wire  = imageRequest();
    auto             input = QueryConverter::transQuery(&wire);
    CompleteTokenIds tokens(1, 1, 64, 4);
    tokens.init(input);
    const auto full = readRows(tokens, 0, 11);
    const auto tail = readRows(tokens, 7, 4);
    EXPECT_TRUE(torch::equal(tail.history, full.history.slice(0, 7, 11)));
    EXPECT_TRUE(torch::equal(tail.history_valid, full.history_valid.slice(0, 7, 11)));
    EXPECT_TRUE(torch::equal(tail.history_valid.sum(1), torch::tensor({0, 1, 2, 3}, torch::kInt64)));
    EXPECT_EQ(full.history_valid[0].sum().item<int64_t>(), 0);
    EXPECT_TRUE(torch::equal(full.history[3], torch::tensor({7, 8, 9}, torch::kInt32)));
    EXPECT_EQ(full.valid.sum().item<int64_t>(), 11);

    CompleteTokenIds candidate(tokens, false);
    int              error = -1;
    ASSERT_TRUE(candidate.update(
        torch::tensor({31, 32}, torch::kInt32).reshape({1, 2}), 0, 2, 11, 64, 129280, false, 1, error));
    EXPECT_EQ(tokens.seqLength(), 11);
    EXPECT_TRUE(torch::equal(readRows(tokens, 7, 4).history, tail.history));
    candidate.setSeqLength(12);
    ASSERT_TRUE(
        candidate.update(torch::tensor({41}, torch::kInt32).reshape({1, 1}), 0, 1, 11, 64, 129280, false, 1, error));
    EXPECT_TRUE(torch::equal(readRows(candidate, 12, 1).history[0], torch::tensor({12, 13, 31}, torch::kInt32)));
    EXPECT_THROW(readRows(candidate, 12, 2), std::runtime_error);
    EXPECT_THROW(readRows(tokens, 0, 1, -1), std::runtime_error);
    CompleteTokenIds shifted(tokens, true, 1);
    EXPECT_THROW(readRows(shifted, 0, 1), std::runtime_error);
}

TEST_F(V41InputBridgeTest, CacheImageIdentityIsSeparateFromCanonicalModelTokens) {
    auto first_wire  = imageRequest();
    auto second_wire = first_wire;
    second_wire.mutable_v41_inputs()->mutable_images(0)->set_content_sha256(std::string(64, 'c'));
    CompleteTokenIds first(1, 1, 64, 4), second(1, 1, 64, 4);
    first.init(QueryConverter::transQuery(&first_wire));
    second.init(QueryConverter::transQuery(&second_wire));
    EXPECT_TRUE(torch::equal(first.completeTokenIds(), second.completeTokenIds()));
    EXPECT_TRUE(first.imageCacheIdentity(0, 3).empty());
    EXPECT_TRUE(first.imageCacheIdentity(7, 4).empty());
    EXPECT_NE(first.imageCacheIdentity(0, 4), second.imageCacheIdentity(0, 4));
    EXPECT_NE(first.imageCacheIdentity(4, 4), second.imageCacheIdentity(4, 4));
    second_wire = first_wire;
    second_wire.mutable_v41_inputs()->mutable_images(0)->set_processor_identity(std::string(64, 'd'));
    second.init(QueryConverter::transQuery(&second_wire));
    EXPECT_NE(first.imageCacheIdentity(3, 4), second.imageCacheIdentity(3, 4));
}

TEST_F(V41InputBridgeTest, WholeImageChunkValidationAndTextRequestAreExplicit) {
    auto        wire     = imageRequest();
    auto        input    = QueryConverter::transQuery(&wire);
    const auto& prepared = *input->v41_inputs;
    EXPECT_NO_THROW(prepared.validateChunk(0, 3));
    EXPECT_NO_THROW(prepared.validateChunk(3, 7));
    EXPECT_NO_THROW(prepared.validateChunk(7, 11));
    EXPECT_THROW(prepared.validateChunk(0, 4), std::runtime_error);
    EXPECT_THROW(prepared.validateChunk(4, 11), std::runtime_error);

    GenerateInputPB text;
    text.mutable_v41_inputs()->set_schema_version(1);
    for (int token : {1, 2, 3}) {
        text.add_token_ids(token);
        text.mutable_v41_inputs()->add_token_types(-1);
        text.mutable_v41_inputs()->add_image_mask(false);
    }
    auto text_input = QueryConverter::transQuery(&text);
    EXPECT_TRUE(text_input->v41_inputs->images.empty());
    CompleteTokenIds tokens(1, 1, 64, 4);
    tokens.init(input);
    tokens.init(text_input);
    EXPECT_TRUE(tokens.imageCacheIdentity(0, 3).empty());
    EXPECT_EQ(readRows(tokens, 0, 3).types.max().item<int32_t>(), -1);
    text.set_token_ids(1, 129264);
    EXPECT_THROW(QueryConverter::transQuery(&text), std::runtime_error);
}

TEST_F(V41InputBridgeTest, FullAndIncrementalCacheKeysAgreeAcrossImagePartialBlock) {
    auto wire   = imageRequest();
    auto tokens = std::make_shared<CompleteTokenIds>(1, 1, 64, 4);
    tokens->init(QueryConverter::transQuery(&wire));
    auto incremental = std::make_shared<BatchKVCacheResource>();
    incremental->resetBatchSize(1);
    tokens->setSeqLength(5);
    initCacheKeys(incremental, tokens, 4);
    tokens->setSeqLength(8);
    updateCacheKeys(incremental, tokens, 4);
    auto complete = std::make_shared<BatchKVCacheResource>();
    complete->resetBatchSize(1);
    initCacheKeys(complete, tokens, 4);
    ASSERT_EQ(complete->cacheKeys().size(), 2);
    EXPECT_EQ(incremental->cacheKeys(), complete->cacheKeys());

    wire.mutable_v41_inputs()->mutable_images(0)->set_content_sha256(std::string(64, 'c'));
    auto different_tokens = std::make_shared<CompleteTokenIds>(1, 1, 64, 4);
    different_tokens->init(QueryConverter::transQuery(&wire));
    different_tokens->setSeqLength(8);
    auto different = std::make_shared<BatchKVCacheResource>();
    different->resetBatchSize(1);
    initCacheKeys(different, different_tokens, 4);
    EXPECT_TRUE(torch::equal(tokens->completeTokenIds(), different_tokens->completeTokenIds()));
    EXPECT_NE(complete->cacheKeys()[0], different->cacheKeys()[0]);
    EXPECT_NE(complete->cacheKeys()[1], different->cacheKeys()[1]);
}

}  // namespace
}  // namespace rtp_llm
