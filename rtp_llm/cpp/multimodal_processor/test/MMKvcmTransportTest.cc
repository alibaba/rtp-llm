#include <algorithm>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "rtp_llm/cpp/multimodal_processor/transport/kvcm/MMKvcmReader.h"

namespace rtp_llm {
namespace {

class FakeKvcmClient final: public MMKvcmClient {
public:
    std::string save(const std::string&, const std::vector<MMKvcmBuffer>&) override {
        return {};
    }

    std::string load(const std::string&, const std::vector<MMKvcmBuffer>& objects, int64_t timeout_ms) override {
        ++load_calls;
        last_load_timeout_ms = timeout_ms;
        if (throw_standard_load) {
            throw std::runtime_error("injected object-client exception");
        }
        if (throw_unknown_load) {
            throw 7;
        }
        loaded_keys.clear();
        loaded_sizes.clear();
        for (const auto& object : objects) {
            loaded_keys.push_back(object.key);
            loaded_sizes.push_back(object.nbytes);
            if (load_error.empty()) {
                std::memset(object.data, 0, static_cast<size_t>(object.nbytes));
            }
        }
        return load_error;
    }

    std::string remove(const std::string&, const std::vector<std::string>&) override {
        return {};
    }

    size_t                   load_calls           = 0;
    int64_t                  last_load_timeout_ms = 0;
    std::string              load_error;
    bool                     throw_standard_load = false;
    bool                     throw_unknown_load  = false;
    std::vector<std::string> loaded_keys;
    std::vector<uint64_t>    loaded_sizes;
};

class FakeControl final: public MMControlClient {
public:
    ErrorResult<MultimodalOutputPB> request(const std::string&, MultimodalInputsPB&, DeadlineBudget&) override {
        return ErrorInfo(ErrorCode::UNKNOWN_ERROR, "unused");
    }

    void release(const std::string&, const std::vector<std::string>& handles, DeadlineBudget&) override {
        if (throw_synchronous_release) {
            throw std::runtime_error("injected synchronous release failure");
        }
        if (throw_unknown_synchronous_release) {
            throw 7;
        }
        synchronous_releases.push_back(handles);
    }

    void releaseAsync(std::string, std::vector<std::string> handles) override {
        if (throw_asynchronous_release) {
            throw std::runtime_error("injected asynchronous release failure");
        }
        if (throw_unknown_asynchronous_release) {
            throw 7;
        }
        asynchronous_releases.push_back(std::move(handles));
    }

    bool                                  throw_synchronous_release          = false;
    bool                                  throw_asynchronous_release         = false;
    bool                                  throw_unknown_synchronous_release  = false;
    bool                                  throw_unknown_asynchronous_release = false;
    std::vector<std::vector<std::string>> synchronous_releases;
    std::vector<std::vector<std::string>> asynchronous_releases;
};

void addObject(MultimodalOutputPB*  receipt,
               const std::string&   key,
               MMRdmaSlotPB::Role   role,
               uint32_t             logical_index,
               std::vector<int64_t> shape,
               TensorDataTypePB     dtype,
               uint64_t             nbytes) {
    auto* object = receipt->add_output_kvcm_objects();
    object->set_key(key);
    object->set_value_size(nbytes);
    object->set_role(role);
    object->set_logical_index(logical_index);
    object->mutable_tensor()->set_data_type(dtype);
    object->mutable_tensor()->set_nbytes(nbytes);
    for (const auto dimension : shape) {
        object->mutable_tensor()->add_shape(dimension);
    }
}

MultimodalOutputPB validReceipt() {
    MultimodalOutputPB receipt;
    receipt.add_split_size(2);
    receipt.add_split_size(3);
    addObject(&receipt, "embedding-0", MMRdmaSlotPB::EMBEDDING, 0, {2, 4}, RDMA_TENSOR_FLOAT32, 32);
    addObject(&receipt, "embedding-1", MMRdmaSlotPB::EMBEDDING, 0, {3, 4}, RDMA_TENSOR_FLOAT32, 48);
    addObject(&receipt, "position", MMRdmaSlotPB::POS_ID, 0, {5}, RDMA_TENSOR_INT32, 20);
    addObject(&receipt, "extra-0", MMRdmaSlotPB::EXTRA_INPUT, 0, {3}, RDMA_TENSOR_FLOAT16, 6);
    addObject(&receipt, "extra-1", MMRdmaSlotPB::EXTRA_INPUT, 1, {2}, RDMA_TENSOR_INT32, 8);
    return receipt;
}

MMKvcmConfig validConfig() {
    MMKvcmConfig config;
    config.addresses              = {"127.0.0.1:1234"};
    config.instance_id            = "model";
    config.instance_group         = "epd-emb";
    config.transfer_client_config = R"({"type":"local"})";
    return config;
}

struct Harness {
    Harness() {
        client                   = std::make_shared<FakeKvcmClient>();
        config.max_object_bytes  = 64;
        config.max_receipt_bytes = 256;
        reader                   = std::make_unique<MMKvcmReader>(client, config, -1);
    }

    ConsumeResult consume(const MultimodalOutputPB& receipt) {
        return reader->consume(receipt, context);
    }

    std::shared_ptr<FakeKvcmClient> client;
    MMKvcmConfig                    config;
    std::unique_ptr<MMKvcmReader>   reader;
    FakeControl                     control;
    std::string                     endpoint = "vit:1234";
    DeadlineBudget                  budget{10000};
    DeliveryContext                 context{endpoint, budget, control};
};

using ReceiptMutation = std::function<void(MultimodalOutputPB*)>;

void expectRejectedBeforeProviderIo(const std::string& name, const ReceiptMutation& mutate) {
    SCOPED_TRACE(name);
    Harness h;
    auto    receipt = validReceipt();
    mutate(&receipt);

    const auto result = h.consume(receipt);

    EXPECT_FALSE(result.succeeded());
    EXPECT_EQ(h.client->load_calls, 0u);
    EXPECT_TRUE(h.control.asynchronous_releases.empty());
}

}  // namespace

TEST(MMKvcmTransportTest, advertisesOnlyWithAClient) {
    MultimodalInputsPB request;
    MMKvcmReader       disabled(nullptr);
    EXPECT_FALSE(disabled.advertise("vit", request));
    EXPECT_FALSE(request.support_kvcm());

    auto         client = std::make_shared<FakeKvcmClient>();
    MMKvcmReader enabled(client);
    EXPECT_TRUE(enabled.advertise("vit", request));
    EXPECT_TRUE(request.support_kvcm());
}

TEST(MMKvcmTransportTest, matchesOnlyNonEmptyKvcmReceipts) {
    MMKvcmReader       reader(nullptr);
    MultimodalOutputPB receipt;
    EXPECT_FALSE(reader.matches(receipt));

    receipt.add_output_kvcm_objects();
    EXPECT_TRUE(reader.matches(receipt));
}

TEST(MMKvcmTransportTest, loadsVariableSizeObjectsAndReassemblesLogicalValues) {
    Harness h;
    auto    receipt = validReceipt();

    auto result = h.consume(receipt);

    ASSERT_TRUE(result.succeeded()) << result.error().ToString();
    EXPECT_EQ(h.client->load_calls, 1u);
    EXPECT_GT(h.client->last_load_timeout_ms, 0);
    EXPECT_LE(h.client->last_load_timeout_ms, 10000);
    EXPECT_EQ(h.client->loaded_keys,
              (std::vector<std::string>{"embedding-0", "embedding-1", "position", "extra-0", "extra-1"}));
    EXPECT_EQ(h.client->loaded_sizes, (std::vector<uint64_t>{32, 48, 20, 6, 8}));
    ASSERT_EQ(result.output().mm_features.size(), 2u);
    EXPECT_EQ(result.output().mm_features[0].size(0), 2);
    EXPECT_EQ(result.output().mm_features[0].size(1), 4);
    EXPECT_EQ(result.output().mm_features[1].size(0), 3);
    EXPECT_EQ(result.output().mm_features[1].size(1), 4);
    ASSERT_TRUE(result.output().mm_position_ids.has_value());
    EXPECT_EQ(result.output().mm_position_ids->size(), 2u);
    ASSERT_TRUE(result.output().mm_extra_input.has_value());
    EXPECT_EQ(result.output().mm_extra_input->size(), 2u);
    EXPECT_TRUE(h.control.synchronous_releases.empty());
    EXPECT_EQ(
        h.control.asynchronous_releases,
        (std::vector<std::vector<std::string>>{{"embedding-0", "embedding-1", "position", "extra-0", "extra-1"}}));
}

TEST(MMKvcmTransportTest, loadsEmbeddingOnlyReceiptAtConfiguredByteBoundaries) {
    Harness h;
    h.config.max_object_bytes  = 48;
    h.config.max_receipt_bytes = 80;
    h.reader                   = std::make_unique<MMKvcmReader>(h.client, h.config, -1);
    MultimodalOutputPB receipt;
    receipt.add_split_size(2);
    receipt.add_split_size(3);
    addObject(&receipt, "embedding-0", MMRdmaSlotPB::EMBEDDING, 0, {2, 4}, RDMA_TENSOR_FLOAT32, 32);
    addObject(&receipt, "embedding-1", MMRdmaSlotPB::EMBEDDING, 0, {3, 4}, RDMA_TENSOR_FLOAT32, 48);

    auto result = h.consume(receipt);

    ASSERT_TRUE(result.succeeded()) << result.error().ToString();
    ASSERT_EQ(result.output().mm_features.size(), 2u);
    EXPECT_EQ(result.output().mm_features[0].sizes(), (torch::IntArrayRef{2, 4}));
    EXPECT_EQ(result.output().mm_features[1].sizes(), (torch::IntArrayRef{3, 4}));
    EXPECT_FALSE(result.output().mm_position_ids.has_value());
    EXPECT_FALSE(result.output().mm_extra_input.has_value());
    EXPECT_EQ(h.control.asynchronous_releases, (std::vector<std::vector<std::string>>{{"embedding-0", "embedding-1"}}));
}

TEST(MMKvcmTransportTest, reassemblesChunkedPositionAndExtraValues) {
    Harness            h;
    MultimodalOutputPB receipt;
    receipt.add_split_size(2);
    receipt.add_split_size(3);
    addObject(&receipt, "embedding", MMRdmaSlotPB::EMBEDDING, 0, {5, 2}, RDMA_TENSOR_BFLOAT16, 20);
    addObject(&receipt, "position-0", MMRdmaSlotPB::POS_ID, 0, {2}, RDMA_TENSOR_INT32, 8);
    addObject(&receipt, "position-1", MMRdmaSlotPB::POS_ID, 0, {3}, RDMA_TENSOR_INT32, 12);
    addObject(&receipt, "extra-0-0", MMRdmaSlotPB::EXTRA_INPUT, 0, {1}, RDMA_TENSOR_FLOAT16, 2);
    addObject(&receipt, "extra-0-1", MMRdmaSlotPB::EXTRA_INPUT, 0, {2}, RDMA_TENSOR_FLOAT16, 4);
    addObject(&receipt, "extra-1", MMRdmaSlotPB::EXTRA_INPUT, 1, {4}, RDMA_TENSOR_FLOAT32, 16);

    auto result = h.consume(receipt);

    ASSERT_TRUE(result.succeeded()) << result.error().ToString();
    ASSERT_TRUE(result.output().mm_position_ids.has_value());
    ASSERT_EQ(result.output().mm_position_ids->size(), 2u);
    EXPECT_EQ(result.output().mm_position_ids->at(0).numel(), 2);
    EXPECT_EQ(result.output().mm_position_ids->at(1).numel(), 3);
    ASSERT_TRUE(result.output().mm_extra_input.has_value());
    ASSERT_EQ(result.output().mm_extra_input->size(), 2u);
    EXPECT_EQ(result.output().mm_extra_input->at(0).numel(), 3);
    EXPECT_EQ(result.output().mm_extra_input->at(1).numel(), 4);
}

TEST(MMKvcmTransportTest, rejectsReceiptAfterDeadlineWithoutProviderIo) {
    Harness h;
    h.budget = DeadlineBudget(0);

    auto result = h.consume(validReceipt());

    EXPECT_FALSE(result.succeeded());
    EXPECT_EQ(h.client->load_calls, 0u);
    EXPECT_TRUE(h.control.synchronous_releases.empty());
    ASSERT_EQ(h.control.asynchronous_releases.size(), 1u);
    EXPECT_EQ(h.control.asynchronous_releases[0].size(), 5u);
}

TEST(MMKvcmTransportTest, rejectsReceiptWhenClientIsMissingAndReleasesSynchronously) {
    FakeControl     control;
    std::string     endpoint = "vit:1234";
    DeadlineBudget  budget{10000};
    DeliveryContext context{endpoint, budget, control};
    const auto      config = validConfig();
    MMKvcmReader    reader(nullptr, config, -1);
    const auto      result = reader.consume(validReceipt(), context);

    EXPECT_FALSE(result.succeeded());
    ASSERT_EQ(control.synchronous_releases.size(), 1u);
    EXPECT_EQ(control.synchronous_releases[0].size(), 5u);
    EXPECT_TRUE(control.asynchronous_releases.empty());
}

TEST(MMKvcmTransportTest, asynchronousReleaseExceptionFallsBackWithoutFailingLoadedOutput) {
    Harness h;
    h.control.throw_asynchronous_release = true;

    auto result = h.consume(validReceipt());

    ASSERT_TRUE(result.succeeded()) << result.error().ToString();
    EXPECT_TRUE(h.control.asynchronous_releases.empty());
    ASSERT_EQ(h.control.synchronous_releases.size(), 1u);
    EXPECT_EQ(h.control.synchronous_releases[0].size(), 5u);
}

TEST(MMKvcmTransportTest, unknownAsynchronousReleaseExceptionAlsoFallsBack) {
    Harness h;
    h.control.throw_unknown_asynchronous_release = true;

    auto result = h.consume(validReceipt());

    ASSERT_TRUE(result.succeeded()) << result.error().ToString();
    EXPECT_TRUE(h.control.asynchronous_releases.empty());
    ASSERT_EQ(h.control.synchronous_releases.size(), 1u);
    EXPECT_EQ(h.control.synchronous_releases[0].size(), 5u);
}

TEST(MMKvcmTransportTest, synchronousReleaseExceptionDoesNotEscapeLeaseDestructor) {
    Harness h;
    h.control.throw_synchronous_release = true;
    auto receipt                        = validReceipt();
    receipt.mutable_output_kvcm_objects(0)->set_value_size(31);

    auto result = h.consume(receipt);

    EXPECT_FALSE(result.succeeded());
    EXPECT_EQ(h.client->load_calls, 0u);
}

TEST(MMKvcmTransportTest, synchronousReleaseExceptionDoesNotEscapeDiscard) {
    Harness h;
    h.control.throw_synchronous_release = true;

    EXPECT_NO_THROW(h.reader->discard(validReceipt(), h.context));
    EXPECT_TRUE(h.control.synchronous_releases.empty());
}

TEST(MMKvcmTransportTest, unknownReleaseExceptionsDoNotEscapeCleanup) {
    Harness h;
    h.control.throw_unknown_synchronous_release  = true;
    h.control.throw_unknown_asynchronous_release = true;

    EXPECT_NO_THROW(h.reader->discard(validReceipt(), h.context));
    EXPECT_NO_THROW({
        auto result = h.consume(validReceipt());
        EXPECT_TRUE(result.succeeded()) << result.error().ToString();
    });
    EXPECT_TRUE(h.control.synchronous_releases.empty());
    EXPECT_TRUE(h.control.asynchronous_releases.empty());
}

TEST(MMKvcmTransportTest, rejectsSizeMismatchBeforeProviderIoAndReleasesKeys) {
    Harness h;
    auto    receipt = validReceipt();
    receipt.mutable_output_kvcm_objects(1)->set_value_size(47);

    auto result = h.consume(receipt);

    EXPECT_FALSE(result.succeeded());
    EXPECT_EQ(h.client->load_calls, 0u);
    EXPECT_EQ(h.control.synchronous_releases.size(), 1u);
    EXPECT_TRUE(h.control.asynchronous_releases.empty());
}

TEST(MMKvcmTransportTest, rejectsDuplicateKeysBeforeProviderIoAndDeduplicatesRelease) {
    Harness h;
    auto    receipt = validReceipt();
    receipt.mutable_output_kvcm_objects(1)->set_key("embedding-0");

    auto result = h.consume(receipt);

    EXPECT_FALSE(result.succeeded());
    EXPECT_EQ(h.client->load_calls, 0u);
    ASSERT_EQ(h.control.synchronous_releases.size(), 1u);
    EXPECT_EQ(h.control.synchronous_releases[0].size(), 4u);
}

TEST(MMKvcmTransportTest, rejectsOversizedKeyBeforeProviderIo) {
    Harness h;
    auto    receipt = validReceipt();
    receipt.mutable_output_kvcm_objects(0)->set_key(std::string(kMMKvcmMaxKeyBytes + 1, 'k'));

    auto result = h.consume(receipt);

    EXPECT_FALSE(result.succeeded());
    EXPECT_EQ(h.client->load_calls, 0u);
    ASSERT_EQ(h.control.synchronous_releases.size(), 1u);
    EXPECT_EQ(h.control.synchronous_releases[0].size(), 4u);
    EXPECT_EQ(std::find(h.control.synchronous_releases[0].begin(),
                        h.control.synchronous_releases[0].end(),
                        std::string(kMMKvcmMaxKeyBytes + 1, 'k')),
              h.control.synchronous_releases[0].end());
}

TEST(MMKvcmTransportTest, rejectsOutOfRangeExtraIndexBeforeProviderIo) {
    Harness h;
    auto    receipt = validReceipt();
    receipt.mutable_output_kvcm_objects(4)->set_logical_index(std::numeric_limits<uint32_t>::max());

    auto result = h.consume(receipt);

    EXPECT_FALSE(result.succeeded());
    EXPECT_EQ(h.client->load_calls, 0u);
    EXPECT_EQ(h.control.synchronous_releases.size(), 1u);
    EXPECT_TRUE(h.control.asynchronous_releases.empty());
}

TEST(MMKvcmTransportTest, rejectsNonFlatExtraInputBeforeProviderIo) {
    Harness h;
    auto    receipt = validReceipt();
    auto*   tensor  = receipt.mutable_output_kvcm_objects(3)->mutable_tensor();
    tensor->clear_shape();
    tensor->add_shape(1);
    tensor->add_shape(3);

    auto result = h.consume(receipt);

    EXPECT_FALSE(result.succeeded());
    EXPECT_EQ(h.client->load_calls, 0u);
    EXPECT_EQ(h.control.synchronous_releases.size(), 1u);
    EXPECT_TRUE(h.control.asynchronous_releases.empty());
}

TEST(MMKvcmTransportTest, rejectsMalformedManifestVariantsBeforeProviderIo) {
    const std::vector<std::pair<std::string, ReceiptMutation>> cases = {
        {"empty object list", [](MultimodalOutputPB* receipt) { receipt->clear_output_kvcm_objects(); }},
        {"mixed RDMA data plane", [](MultimodalOutputPB* receipt) { receipt->add_output_rdma_slots(); }},
        {"mixed inline embedding", [](MultimodalOutputPB* receipt) { receipt->mutable_multimodal_embedding(); }},
        {"mixed inline position", [](MultimodalOutputPB* receipt) { receipt->mutable_multimodal_pos_id(); }},
        {"mixed inline extra", [](MultimodalOutputPB* receipt) { receipt->add_multimodal_extra_input(); }},
        {"empty split list", [](MultimodalOutputPB* receipt) { receipt->clear_split_size(); }},
        {"zero split", [](MultimodalOutputPB* receipt) { receipt->set_split_size(0, 0); }},
        {"negative split", [](MultimodalOutputPB* receipt) { receipt->set_split_size(0, -1); }},
        {"empty key", [](MultimodalOutputPB* receipt) { receipt->mutable_output_kvcm_objects(0)->clear_key(); }},
        {"unspecified role",
         [](MultimodalOutputPB* receipt) {
             receipt->mutable_output_kvcm_objects(0)->set_role(MMRdmaSlotPB::ROLE_UNSPECIFIED);
         }},
        {"descending role order",
         [](MultimodalOutputPB* receipt) { receipt->mutable_output_kvcm_objects()->SwapElements(0, 2); }},
        {"embedding logical index",
         [](MultimodalOutputPB* receipt) { receipt->mutable_output_kvcm_objects(0)->set_logical_index(1); }},
        {"position logical index",
         [](MultimodalOutputPB* receipt) { receipt->mutable_output_kvcm_objects(2)->set_logical_index(1); }},
        {"descending extra indices",
         [](MultimodalOutputPB* receipt) {
             receipt->mutable_output_kvcm_objects(3)->set_logical_index(1);
             receipt->mutable_output_kvcm_objects(4)->set_logical_index(0);
         }},
        {"nonzero tensor offset",
         [](MultimodalOutputPB* receipt) { receipt->mutable_output_kvcm_objects(0)->mutable_tensor()->set_offset(1); }},
        {"empty tensor shape",
         [](MultimodalOutputPB* receipt) { receipt->mutable_output_kvcm_objects(0)->mutable_tensor()->clear_shape(); }},
        {"too many tensor dimensions",
         [](MultimodalOutputPB* receipt) {
             auto* tensor = receipt->mutable_output_kvcm_objects(0)->mutable_tensor();
             tensor->clear_shape();
             for (size_t i = 0; i <= kMMKvcmMaxTensorDimensions; ++i) {
                 tensor->add_shape(1);
             }
         }},
        {"unsupported tensor dtype",
         [](MultimodalOutputPB* receipt) {
             receipt->mutable_output_kvcm_objects(0)->mutable_tensor()->set_data_type(
                 static_cast<TensorDataTypePB>(99));
         }},
        {"zero tensor dimension",
         [](MultimodalOutputPB* receipt) {
             receipt->mutable_output_kvcm_objects(0)->mutable_tensor()->set_shape(0, 0);
         }},
        {"negative tensor dimension",
         [](MultimodalOutputPB* receipt) {
             receipt->mutable_output_kvcm_objects(0)->mutable_tensor()->set_shape(0, -1);
         }},
        {"tensor shape overflow",
         [](MultimodalOutputPB* receipt) {
             auto* tensor = receipt->mutable_output_kvcm_objects(0)->mutable_tensor();
             tensor->clear_shape();
             tensor->add_shape(std::numeric_limits<int64_t>::max());
             tensor->add_shape(2);
         }},
        {"tensor metadata byte mismatch",
         [](MultimodalOutputPB* receipt) {
             receipt->mutable_output_kvcm_objects(0)->mutable_tensor()->set_nbytes(31);
         }},
        {"chunk rank mismatch",
         [](MultimodalOutputPB* receipt) {
             auto* object = receipt->mutable_output_kvcm_objects(1);
             object->mutable_tensor()->add_shape(1);
         }},
        {"chunk dtype mismatch",
         [](MultimodalOutputPB* receipt) {
             auto* object = receipt->mutable_output_kvcm_objects(1);
             object->set_value_size(24);
             object->mutable_tensor()->set_nbytes(24);
             object->mutable_tensor()->set_data_type(RDMA_TENSOR_FLOAT16);
         }},
        {"chunk trailing shape mismatch",
         [](MultimodalOutputPB* receipt) {
             auto* object = receipt->mutable_output_kvcm_objects(1);
             object->set_value_size(24);
             object->mutable_tensor()->set_nbytes(24);
             object->mutable_tensor()->set_shape(1, 2);
         }},
        {"embedding row mismatch", [](MultimodalOutputPB* receipt) { receipt->set_split_size(1, 4); }},
        {"position row mismatch",
         [](MultimodalOutputPB* receipt) {
             auto* object = receipt->mutable_output_kvcm_objects(2);
             object->set_value_size(16);
             object->mutable_tensor()->set_nbytes(16);
             object->mutable_tensor()->set_shape(0, 4);
         }},
        {"extra logical index gap",
         [](MultimodalOutputPB* receipt) { receipt->mutable_output_kvcm_objects()->DeleteSubrange(3, 1); }},
        {"missing final extra logical value",
         [](MultimodalOutputPB* receipt) { receipt->mutable_output_kvcm_objects()->RemoveLast(); }},
    };

    for (const auto& [name, mutate] : cases) {
        expectRejectedBeforeProviderIo(name, mutate);
    }
}

TEST(MMKvcmTransportTest, rejectsConfiguredObjectAndReceiptByteLimitViolationsBeforeProviderIo) {
    {
        Harness h;
        h.config.max_object_bytes = 0;
        h.reader                  = std::make_unique<MMKvcmReader>(h.client, h.config, -1);
        const auto result         = h.consume(validReceipt());
        EXPECT_FALSE(result.succeeded());
        EXPECT_EQ(h.client->load_calls, 0u);
    }
    {
        Harness h;
        h.config.max_receipt_bytes = h.config.max_object_bytes - 1;
        h.reader                   = std::make_unique<MMKvcmReader>(h.client, h.config, -1);
        const auto result          = h.consume(validReceipt());
        EXPECT_FALSE(result.succeeded());
        EXPECT_EQ(h.client->load_calls, 0u);
    }
    {
        Harness h;
        h.config.max_object_bytes = 47;
        h.reader                  = std::make_unique<MMKvcmReader>(h.client, h.config, -1);
        const auto result         = h.consume(validReceipt());
        EXPECT_FALSE(result.succeeded());
        EXPECT_EQ(h.client->load_calls, 0u);
    }
    {
        Harness h;
        h.config.max_receipt_bytes = 113;
        h.reader                   = std::make_unique<MMKvcmReader>(h.client, h.config, -1);
        const auto result          = h.consume(validReceipt());
        EXPECT_FALSE(result.succeeded());
        EXPECT_EQ(h.client->load_calls, 0u);
    }
}

TEST(MMKvcmTransportTest, rejectsUnboundedLogicalValueCountBeforeProviderIo) {
    Harness h;
    auto    receipt = validReceipt();
    receipt.clear_split_size();
    for (size_t i = 0; i < 16385; ++i) {
        receipt.add_split_size(1);
    }

    auto result = h.consume(receipt);

    EXPECT_FALSE(result.succeeded());
    EXPECT_EQ(h.client->load_calls, 0u);
    EXPECT_EQ(h.control.synchronous_releases.size(), 1u);
    EXPECT_TRUE(h.control.asynchronous_releases.empty());
}

TEST(MMKvcmTransportTest, rejectsUnboundedObjectCountWithBoundedReleaseSnapshot) {
    Harness            h;
    MultimodalOutputPB receipt;
    receipt.add_split_size(1);
    for (size_t i = 0; i <= kMMKvcmMaxObjectsPerReceipt; ++i) {
        receipt.add_output_kvcm_objects()->set_key("oversized-" + std::to_string(i));
    }

    auto result = h.consume(receipt);

    EXPECT_FALSE(result.succeeded());
    EXPECT_EQ(h.client->load_calls, 0u);
    ASSERT_EQ(h.control.synchronous_releases.size(), 1u);
    EXPECT_EQ(h.control.synchronous_releases[0].size(), kMMKvcmMaxObjectsPerReceipt);
    EXPECT_TRUE(h.control.asynchronous_releases.empty());
}

TEST(MMKvcmTransportTest, acceptsObjectCountAtAsyncReleaseCapacity) {
    Harness h;
    h.config.max_object_bytes  = 4;
    h.config.max_receipt_bytes = 4 * kMMKvcmMaxObjectsPerReceipt;
    h.reader                   = std::make_unique<MMKvcmReader>(h.client, h.config, -1);
    MultimodalOutputPB receipt;
    receipt.add_split_size(static_cast<int32_t>(kMMKvcmMaxObjectsPerReceipt));
    for (size_t i = 0; i < kMMKvcmMaxObjectsPerReceipt; ++i) {
        addObject(
            &receipt, "boundary-" + std::to_string(i), MMRdmaSlotPB::EMBEDDING, 0, {1, 1}, RDMA_TENSOR_FLOAT32, 4);
    }

    auto result = h.consume(receipt);

    ASSERT_TRUE(result.succeeded()) << result.error().ToString();
    // The reader issues one logical load; the production KVCM client performs
    // the 64-object service batching beneath this interface.
    EXPECT_EQ(h.client->load_calls, 1u);
    ASSERT_EQ(result.output().mm_features.size(), 1u);
    EXPECT_EQ(result.output().mm_features[0].size(0), static_cast<int64_t>(kMMKvcmMaxObjectsPerReceipt));
    ASSERT_EQ(h.control.asynchronous_releases.size(), 1u);
    EXPECT_EQ(h.control.asynchronous_releases[0].size(), kMMKvcmMaxObjectsPerReceipt);
    EXPECT_TRUE(h.control.synchronous_releases.empty());
}

TEST(MMKvcmTransportTest, providerFailureDoesNotMaterializeAndReleasesSynchronously) {
    Harness h;
    h.client->load_error = "injected load failure";

    auto result = h.consume(validReceipt());

    EXPECT_FALSE(result.succeeded());
    EXPECT_EQ(h.client->load_calls, 1u);
    EXPECT_EQ(h.control.synchronous_releases.size(), 1u);
    EXPECT_TRUE(h.control.asynchronous_releases.empty());
}

TEST(MMKvcmTransportTest, providerExceptionsBecomeFailuresAndReleaseSynchronously) {
    for (const bool unknown_exception : {false, true}) {
        SCOPED_TRACE(unknown_exception ? "unknown exception" : "standard exception");
        Harness h;
        h.client->throw_standard_load = !unknown_exception;
        h.client->throw_unknown_load  = unknown_exception;

        auto result = h.consume(validReceipt());

        EXPECT_FALSE(result.succeeded());
        EXPECT_EQ(h.client->load_calls, 1u);
        EXPECT_EQ(h.control.synchronous_releases.size(), 1u);
        EXPECT_TRUE(h.control.asynchronous_releases.empty());
    }
}

TEST(MMKvcmTransportTest, directAssemblyRejectsInvalidArgumentsAndIncompatibleChunks) {
    const auto                 receipt = validReceipt();
    std::vector<torch::Tensor> tensors = {
        torch::zeros({2, 4}),
        torch::zeros({3, 4}),
        torch::zeros({5}, torch::TensorOptions().dtype(torch::kInt32)),
        torch::zeros({3}, torch::TensorOptions().dtype(torch::kFloat16)),
        torch::zeros({2}, torch::TensorOptions().dtype(torch::kInt32)),
    };
    MultimodalOutput output;
    EXPECT_FALSE(assembleMMKvcmOutput(tensors, receipt, nullptr));

    tensors.pop_back();
    EXPECT_FALSE(assembleMMKvcmOutput(tensors, receipt, &output));

    tensors.push_back(torch::zeros({2}, torch::TensorOptions().dtype(torch::kInt32)));
    tensors[1] = torch::zeros({3, 5});
    EXPECT_FALSE(assembleMMKvcmOutput(tensors, receipt, &output));
}

TEST(MMKvcmTransportTest, directAssemblyRejectsInvalidSplitAndRoleMetadata) {
    std::vector<torch::Tensor> tensors = {
        torch::zeros({2, 4}),
        torch::zeros({3, 4}),
        torch::zeros({5}, torch::TensorOptions().dtype(torch::kInt32)),
        torch::zeros({3}, torch::TensorOptions().dtype(torch::kFloat16)),
        torch::zeros({2}, torch::TensorOptions().dtype(torch::kInt32)),
    };
    for (const auto& mutate : std::vector<ReceiptMutation>{
             [](MultimodalOutputPB* receipt) { receipt->clear_split_size(); },
             [](MultimodalOutputPB* receipt) { receipt->set_split_size(0, 0); },
             [](MultimodalOutputPB* receipt) { receipt->set_split_size(1, 4); },
             [](MultimodalOutputPB* receipt) {
                 receipt->mutable_output_kvcm_objects(0)->set_role(MMRdmaSlotPB::ROLE_UNSPECIFIED);
             },
             [](MultimodalOutputPB* receipt) { receipt->mutable_output_kvcm_objects(4)->set_logical_index(2); },
         }) {
        auto receipt = validReceipt();
        mutate(&receipt);
        MultimodalOutput output;
        EXPECT_FALSE(assembleMMKvcmOutput(tensors, receipt, &output));
    }
}

TEST(MMKvcmTransportTest, validatesServiceLimitsBeforeClientCreation) {
    auto config = validConfig();
    EXPECT_TRUE(validateMMKvcmConfig(config).empty());

    const std::vector<std::pair<std::string, std::function<void(MMKvcmConfig*)>>> cases = {
        {"no addresses", [](MMKvcmConfig* value) { value->addresses.clear(); }},
        {"too many addresses", [](MMKvcmConfig* value) { value->addresses.assign(65, "address"); }},
        {"empty address", [](MMKvcmConfig* value) { value->addresses = {""}; }},
        {"oversized address", [](MMKvcmConfig* value) { value->addresses = {std::string(1025, 'a')}; }},
        {"duplicate addresses", [](MMKvcmConfig* value) { value->addresses = {"a", "a"}; }},
        {"empty instance id", [](MMKvcmConfig* value) { value->instance_id.clear(); }},
        {"oversized instance id",
         [](MMKvcmConfig* value) { value->instance_id = std::string(kMMKvcmMaxInstanceIdBytes + 1, 'i'); }},
        {"empty instance group", [](MMKvcmConfig* value) { value->instance_group.clear(); }},
        {"oversized instance group",
         [](MMKvcmConfig* value) { value->instance_group = std::string(kMMKvcmMaxInstanceGroupBytes + 1, 'g'); }},
        {"oversized user data",
         [](MMKvcmConfig* value) { value->user_data = std::string(kMMKvcmMaxUserDataBytes + 1, 'u'); }},
        {"empty transfer config", [](MMKvcmConfig* value) { value->transfer_client_config.clear(); }},
        {"zero call timeout", [](MMKvcmConfig* value) { value->call_timeout_ms = 0; }},
        {"excessive call timeout", [](MMKvcmConfig* value) { value->call_timeout_ms = 600001; }},
        {"zero write timeout", [](MMKvcmConfig* value) { value->write_timeout_seconds = 0; }},
        {"excessive write timeout",
         [](MMKvcmConfig* value) { value->write_timeout_seconds = kMMKvcmMaxWriteTimeoutSecs + 1; }},
        {"zero GC timeout", [](MMKvcmConfig* value) { value->object_gc_timeout_ms = 0; }},
        {"zero max object", [](MMKvcmConfig* value) { value->max_object_bytes = 0; }},
        {"excessive max object",
         [](MMKvcmConfig* value) { value->max_object_bytes = static_cast<int64_t>(kMMKvcmMaxObjectBytes + 1); }},
        {"zero max receipt", [](MMKvcmConfig* value) { value->max_receipt_bytes = 0; }},
        {"receipt smaller than object",
         [](MMKvcmConfig* value) { value->max_receipt_bytes = value->max_object_bytes - 1; }},
    };
    for (const auto& [name, mutate] : cases) {
        SCOPED_TRACE(name);
        config = validConfig();
        mutate(&config);
        EXPECT_FALSE(validateMMKvcmConfig(config).empty());
    }
}

TEST(MMKvcmTransportTest, validatesCompleteClientObjectBatchBeforeProviderIo) {
    char                      storage = 0;
    std::vector<MMKvcmBuffer> boundary_objects;
    boundary_objects.reserve(kMMKvcmMaxObjectsPerReceipt);
    for (size_t i = 0; i < kMMKvcmMaxObjectsPerReceipt; ++i) {
        boundary_objects.push_back({"object-" + std::to_string(i), &storage, 1, false});
    }
    boundary_objects.front().key = std::string(kMMKvcmMaxKeyBytes, 'k');
    EXPECT_TRUE(validateMMKvcmObjects(boundary_objects, 1, kMMKvcmMaxObjectsPerReceipt).empty());

    const std::vector<std::pair<std::string, std::function<void(std::vector<MMKvcmBuffer>*)>>> cases = {
        {"empty batch", [](std::vector<MMKvcmBuffer>* objects) { objects->clear(); }},
        {"too many objects",
         [&storage](std::vector<MMKvcmBuffer>* objects) { objects->push_back({"one-too-many", &storage, 1, false}); }},
        {"empty key", [](std::vector<MMKvcmBuffer>* objects) { objects->front().key.clear(); }},
        {"oversized key",
         [](std::vector<MMKvcmBuffer>* objects) { objects->front().key = std::string(kMMKvcmMaxKeyBytes + 1, 'k'); }},
        {"duplicate key", [](std::vector<MMKvcmBuffer>* objects) { objects->back().key = objects->front().key; }},
        {"null address", [](std::vector<MMKvcmBuffer>* objects) { objects->front().data = nullptr; }},
        {"zero size", [](std::vector<MMKvcmBuffer>* objects) { objects->front().nbytes = 0; }},
        {"object over configured limit", [](std::vector<MMKvcmBuffer>* objects) { objects->front().nbytes = 2; }},
    };
    for (const auto& [name, mutate] : cases) {
        SCOPED_TRACE(name);
        auto objects = boundary_objects;
        mutate(&objects);
        EXPECT_FALSE(validateMMKvcmObjects(objects, 1, kMMKvcmMaxObjectsPerReceipt).empty());
    }

    const std::vector<MMKvcmBuffer> two_objects = {
        {"first", &storage, 2, false},
        {"second", &storage, 3, false},
    };
    EXPECT_TRUE(validateMMKvcmObjects(two_objects, 3, 5).empty());
    EXPECT_FALSE(validateMMKvcmObjects(two_objects, 3, 4).empty());
    EXPECT_FALSE(validateMMKvcmObjects(two_objects, 0, 5).empty());
    EXPECT_FALSE(validateMMKvcmObjects(two_objects, 3, 2).empty());
}

TEST(MMKvcmTransportTest, batchesByBothItemCountAndTotalBytes) {
    std::vector<MMKvcmBuffer> objects(65);
    for (auto& object : objects) {
        object.nbytes = 1;
    }
    EXPECT_EQ(nextMMKvcmBatchEnd(objects, 0), kMMKvcmMaxBatchItems);
    EXPECT_EQ(nextMMKvcmBatchEnd(objects, kMMKvcmMaxBatchItems), objects.size());

    objects.resize(5);
    for (auto& object : objects) {
        object.nbytes = kMMKvcmMaxObjectBytes;
    }
    EXPECT_EQ(nextMMKvcmBatchEnd(objects, 0), 4u);
    EXPECT_EQ(nextMMKvcmBatchEnd(objects, 4), 5u);

    objects.assign(4, MMKvcmBuffer{});
    for (auto& object : objects) {
        object.nbytes = kMMKvcmMaxObjectBytes;
    }
    EXPECT_EQ(nextMMKvcmBatchEnd(objects, 0), objects.size());
    EXPECT_EQ(nextMMKvcmBatchEnd(objects, objects.size()), objects.size());
}

}  // namespace rtp_llm
