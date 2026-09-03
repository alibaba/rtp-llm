#include <chrono>
#include <condition_variable>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "gtest/gtest.h"

#include "rtp_llm/cpp/model_rpc/TensorPbConvert.h"
#include "rtp_llm/cpp/multimodal_processor/transport/MMRemoteOutputTransport.h"
#include "rtp_llm/cpp/multimodal_processor/transport/grpc/MMGrpcTransport.h"
#include "rtp_llm/cpp/multimodal_processor/transport/rdma/MMRdmaReader.h"

// Orchestration of the LLM-side output transport: which reader gets a receipt and what happens
// when the encoder answers with a data plane that does not match the configured mode. The data
// plane itself (slot planning, manifest reassembly) is covered by MMRdmaTransportTest.cc; here
// both the control plane and the RDMA transport are fakes, so nothing needs hardware or a server.
namespace rtp_llm {

namespace {

std::string uniqueEndpoint(const char* name) {
    return std::string("127.0.0.1:") + name;
}

torch::Tensor rows(int64_t n, int64_t cols = 4) {
    return torch::arange(0, n * cols, torch::kFloat32).reshape({n, cols});
}

// A receipt with one descriptor per handle, each declaring a single EMBEDDING chunk of `chunk_rows`
// rows. split_size describes the un-chunked per-image row counts, as the encoder sends it.
MultimodalOutputPB rdmaReceipt(const std::vector<std::string>& handles,
                               int64_t                         chunk_rows,
                               const std::vector<int64_t>&     split_size) {
    MultimodalOutputPB receipt;
    for (const auto& handle : handles) {
        auto* slot = receipt.add_output_rdma_slots();
        auto* desc = slot->mutable_rdma_descriptor();
        desc->set_lease_id(handle);
        desc->set_host("127.0.0.1");
        desc->set_port(1);
        desc->set_remote_addr(0x1000);
        auto* nic_key = desc->add_nic_keys();
        nic_key->set_nic_id(0);
        nic_key->set_rkey(1);
        auto* tensor = desc->add_tensors();
        tensor->add_shape(chunk_rows);
        tensor->add_shape(4);
        tensor->set_data_type(::RDMA_TENSOR_FLOAT32);
        tensor->set_nbytes(chunk_rows * 4 * 4);
        desc->set_payload_bytes(tensor->nbytes());
        slot->add_roles(MMRdmaSlotPB::EMBEDDING);
    }
    for (auto size : split_size) {
        receipt.add_split_size(size);
    }
    return receipt;
}

MultimodalOutputPB inlineReceipt(int64_t total_rows) {
    MultimodalOutputPB receipt;
    receipt.add_split_size(total_rows);
    return receipt;
}

// Returns a canned receipt per round and records what was advertised each time, so a test can
// assert both the number of ViT round trips and the capabilities each one carried.
class FakeControlClient: public MMControlClient {
public:
    std::vector<MultimodalOutputPB>       responses;
    std::vector<bool>                     advertised_rdma;
    std::vector<std::vector<std::string>> released;
    std::vector<std::string>*             log      = nullptr;
    size_t                                requests = 0;
    size_t                                failure_round = std::numeric_limits<size_t>::max();
    ErrorInfo                             failure = ErrorInfo::OkStatus();

    ErrorResult<MultimodalOutputPB>
    request(const std::string&, MultimodalInputsPB& request_pb, DeadlineBudget&) override {
        advertised_rdma.push_back(request_pb.support_rdma());
        if (log) {
            log->push_back("request");
        }
        const size_t round = requests++;
        if (round == failure_round) {
            return failure;
        }
        if (round >= responses.size()) {
            // Surfaces "the transport asked for more ViT forwards than the test allowed" as a
            // failure instead of undefined behaviour.
            return ErrorInfo(ErrorCode::UNKNOWN_ERROR, "unexpected extra vit round trip");
        }
        // ErrorResult only takes T by rvalue, and a vector element is an lvalue.
        MultimodalOutputPB receipt = responses[round];
        return receipt;
    }

    void release(const std::string&, const std::vector<std::string>& handles, DeadlineBudget&) override {
        released.push_back(handles);
        if (log) {
            log->push_back("release");
        }
    }

    void releaseAsync(std::string, std::vector<std::string> handles) override {
        released.push_back(std::move(handles));
        if (log) {
            log->push_back("release");
        }
    }
};

// Only the LLM half matters here; the encoder half is never called.
class FakeRdmaTransport: public rdma_transport::RdmaRead {
public:
    bool                      read_ok    = true;
    std::vector<std::string>* log        = nullptr;
    size_t                    reads      = 0;
    bool                      block_read = false;

    bool waitUntilReadEntered(std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(block_mutex_);
        return block_cv_.wait_for(lock, timeout, [this] { return read_entered_; });
    }

    void unblockRead() {
        {
            std::lock_guard<std::mutex> lock(block_mutex_);
            read_unblocked_ = true;
        }
        block_cv_.notify_all();
    }

    rdma_transport::RdmaReadResult
    read(const std::vector<rdma_transport::RdmaDescriptor>& descriptors, int64_t) override {
        ++reads;
        if (block_read) {
            std::unique_lock<std::mutex> lock(block_mutex_);
            read_entered_ = true;
            block_cv_.notify_all();
            block_cv_.wait(lock, [this] { return read_unblocked_; });
        }
        rdma_transport::RdmaReadResult result;
        if (log) {
            log->push_back("read");
        }
        if (!read_ok) {
            result.status = ErrorInfo(ErrorCode::REMOTE_ALLOCATE_RESOURCE_READ_FAILED, "fake read failed");
            return result;
        }
        for (const auto& desc : descriptors) {
            for (const auto& tensor : desc.tensors) {
                result.tensors.push_back(rows(tensor.shape.at(0)));
            }
        }
        result.status = ErrorInfo::OkStatus();
        return result;
    }

private:
    std::mutex              block_mutex_;
    std::condition_variable block_cv_;
    bool                    read_entered_   = false;
    bool                    read_unblocked_ = false;
};

// Stands in for InlineReceiptReader so the orchestration tests can observe *whether* the terminal
// was reached without depending on the decode. The real InlineReceiptReader is covered separately
// below.
class FakeTerminalReader: public MMTerminalReceiptReader {
public:
    size_t                    consumed = 0;
    std::vector<std::string>* log      = nullptr;

    const char* name() const override {
        return "inline";
    }

    ErrorResult<MultimodalOutput> consumeTerminal(const MultimodalOutputPB& receipt, DeliveryContext&) override {
        ++consumed;
        if (log) {
            log->push_back("terminal");
        }
        MultimodalOutput output;
        int64_t          total_rows = 0;
        for (auto size : receipt.split_size()) {
            total_rows += size;
        }
        output.mm_features = {rows(total_rows > 0 ? total_rows : 1)};
        return output;
    }
};

struct Harness {
    FakeControlClient*                       control    = nullptr;
    FakeRdmaTransport*                       transport  = nullptr;
    FakeTerminalReader*                      terminal   = nullptr;
    std::unique_ptr<MMRemoteOutputTransport> under_test;
    std::vector<std::string>                 log;

    // `with_transport == false` models a build that links no RDMA implementation: the reader is
    // still registered (only it can recognise a descriptor receipt) but advertises nothing, which
    // is exactly what createMMRemoteOutputTransport() sets up.
    explicit Harness(bool with_transport = true, std::optional<RdmaConfig> validated_config = std::nullopt) {
        auto control_up = std::make_unique<FakeControlClient>();
        control         = control_up.get();
        control->log    = &log;
        auto terminal_up = std::make_unique<FakeTerminalReader>();
        terminal         = terminal_up.get();
        terminal->log    = &log;

        std::shared_ptr<FakeRdmaTransport> transport_sp;
        if (with_transport) {
            transport_sp   = std::make_shared<FakeRdmaTransport>();
            transport      = transport_sp.get();
            transport->log = &log;
        }
        std::vector<std::unique_ptr<MMReceiptReader>> readers;
        if (validated_config.has_value()) {
            readers.push_back(std::make_unique<MMRdmaReader>(transport_sp, *validated_config));
        } else {
            readers.push_back(std::make_unique<MMRdmaReader>(transport_sp));
        }
        under_test = std::make_unique<MMRemoteOutputTransport>(
            std::move(readers), std::move(terminal_up), std::move(control_up));
    }

    ErrorResult<MultimodalOutput> fetch(const std::string& endpoint) {
        MultimodalInputsPB request_pb;
        request_pb.add_multimodal_inputs();
        return under_test->fetch(endpoint, request_pb);
    }
};

}  // namespace

TEST(MMRemoteOutputTransportTest, mixedExplicitAndDefaultTimeoutKeepsDefaultBudget) {
    MultimodalInputsPB request;
    request.add_multimodal_inputs()->mutable_mm_preprocess_config()->set_mm_timeout_ms(2000);
    request.add_multimodal_inputs()->mutable_mm_preprocess_config()->set_mm_timeout_ms(-1);

    EXPECT_EQ(resolveRpcTimeoutMs(request), kDefaultVitRpcTimeoutMs);
}

TEST(MMRemoteOutputTransportTest, rdmaReceiptIsReadAndSlotsAreReleasedOnce) {
    Harness h;
    h.control->responses = {rdmaReceipt({"h0", "h1"}, /*chunk_rows=*/2, /*split_size=*/{2, 2})};

    auto result = h.fetch(uniqueEndpoint("rdma-hit"));

    ASSERT_TRUE(result.ok());
    EXPECT_EQ(h.transport->reads, 1u);
    // One release RPC carrying every handle, not one per slot.
    ASSERT_EQ(h.control->released.size(), 1u);
    EXPECT_EQ(h.control->released[0], std::vector<std::string>({"h0", "h1"}));
    EXPECT_EQ(result.value().mm_features.size(), 2u);
    EXPECT_TRUE(h.control->advertised_rdma[0]);
}

TEST(MMRemoteOutputTransportTest, invalidDescriptorsAreRejectedBeforeProviderRead) {
    using MutateReceipt = std::function<void(MultimodalOutputPB*)>;
    const std::vector<std::pair<std::string, MutateReceipt>> cases{
        {"zero address", [](auto* receipt) { receipt->mutable_output_rdma_slots(0)
                                                  ->mutable_rdma_descriptor()
                                                  ->set_remote_addr(0); }},
        {"tensor outside payload", [](auto* receipt) { receipt->mutable_output_rdma_slots(0)
                                                             ->mutable_rdma_descriptor()
                                                             ->mutable_tensors(0)
                                                             ->set_offset(256); }},
        {"overlapping tensors", [](auto* receipt) {
             auto* slot   = receipt->mutable_output_rdma_slots(0);
             auto* tensor = slot->mutable_rdma_descriptor()->add_tensors();
             tensor->add_shape(1);
             tensor->add_shape(4);
             tensor->set_data_type(::RDMA_TENSOR_FLOAT32);
             tensor->set_offset(0);
             tensor->set_nbytes(16);
             slot->add_roles(MMRdmaSlotPB::EMBEDDING);
         }},
        {"oversized payload", [](auto* receipt) { receipt->mutable_output_rdma_slots(0)
                                                        ->mutable_rdma_descriptor()
                                                        ->set_payload_bytes(65); }},
        {"duplicate NIC key", [](auto* receipt) {
             auto* key = receipt->mutable_output_rdma_slots(0)->mutable_rdma_descriptor()->add_nic_keys();
             key->set_nic_id(0);
             key->set_rkey(2);
         }},
        {"invalid dtype", [](auto* receipt) { receipt->mutable_output_rdma_slots(0)
                                                    ->mutable_rdma_descriptor()
                                                    ->mutable_tensors(0)
                                                    ->set_data_type(static_cast<::TensorDataTypePB>(99)); }},
    };

    RdmaConfig config;
    config.max_slot_bytes    = 64;
    config.max_receipt_bytes = 128;
    for (const auto& [name, mutate] : cases) {
        SCOPED_TRACE(name);
        Harness h(/*with_transport=*/true, config);
        auto    invalid = rdmaReceipt({"lease"}, /*chunk_rows=*/1, /*split_size=*/{1});
        mutate(&invalid);
        h.control->responses = {std::move(invalid)};

        auto result = h.fetch(uniqueEndpoint("invalid-descriptor"));

        ASSERT_FALSE(result.ok());
        EXPECT_EQ(h.control->requests, 1u);
        EXPECT_EQ(h.transport->reads, 0u);
        EXPECT_EQ(h.terminal->consumed, 0u);
        ASSERT_EQ(h.control->released.size(), 1u);
        EXPECT_EQ(h.control->released[0], std::vector<std::string>({"lease"}));
    }
}

TEST(MMRemoteOutputTransportTest, readerLockWaitHonorsDeadlineAndReleasesLeaseAsync) {
    auto transport       = std::make_shared<FakeRdmaTransport>();
    transport->block_read = true;
    MMRdmaReader reader(transport);
    FakeControlClient control;
    const std::string endpoint = uniqueEndpoint("reader-lock-deadline");
    auto first_receipt  = rdmaReceipt({"first"}, /*chunk_rows=*/1, /*split_size=*/{1});
    auto second_receipt = rdmaReceipt({"second"}, /*chunk_rows=*/1, /*split_size=*/{1});
    bool first_succeeded = false;

    std::thread first([&] {
        DeadlineBudget  budget(1000);
        DeliveryContext context{endpoint, budget, control};
        first_succeeded = reader.consume(first_receipt, context).succeeded();
    });
    const bool read_entered = transport->waitUntilReadEntered(std::chrono::milliseconds(500));
    if (!read_entered) {
        transport->unblockRead();
        first.join();
    }
    ASSERT_TRUE(read_entered);

    DeadlineBudget  short_budget(20);
    DeliveryContext short_context{endpoint, short_budget, control};
    const auto      begin = std::chrono::steady_clock::now();
    auto            second_result = reader.consume(second_receipt, short_context);
    const auto elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - begin);
    const auto reads_before_unblock    = transport->reads;
    const auto released_before_unblock = control.released;

    transport->unblockRead();
    first.join();

    EXPECT_FALSE(second_result.succeeded());
    EXPECT_EQ(second_result.error().code(), ErrorCode::MM_PROCESS_ERROR);
    EXPECT_LT(elapsed.count(), 500);
    EXPECT_EQ(reads_before_unblock, 1u);
    ASSERT_EQ(released_before_unblock.size(), 1u);
    EXPECT_EQ(released_before_unblock[0], std::vector<std::string>({"second"}));
    EXPECT_TRUE(first_succeeded);
}

TEST(MMRemoteOutputTransportTest, readFailureReturnsErrorWithoutAnotherVitRequest) {
    Harness h;
    h.control->responses = {rdmaReceipt({"h0"}, 2, {2})};
    h.transport->read_ok = false;

    auto result = h.fetch(uniqueEndpoint("read-error"));

    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), ErrorCode::MM_PROCESS_ERROR);
    EXPECT_EQ(h.control->requests, 1u);
    ASSERT_EQ(h.control->advertised_rdma.size(), 1u);
    EXPECT_TRUE(h.control->advertised_rdma[0]);
    EXPECT_EQ(h.terminal->consumed, 0u);
    ASSERT_EQ(h.control->released.size(), 1u);
    EXPECT_EQ(h.control->released[0], std::vector<std::string>({"h0"}));
    const std::vector<std::string> expected{"request", "read", "release"};
    EXPECT_EQ(h.log, expected);
}

TEST(MMRemoteOutputTransportTest, inlineReceiptIsRejectedWhenRdmaWasAdvertised) {
    Harness h;
    h.control->responses = {inlineReceipt(2)};

    auto result = h.fetch(uniqueEndpoint("inline-in-rdma-mode"));

    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), ErrorCode::MM_PROCESS_ERROR);
    EXPECT_EQ(h.control->requests, 1u);
    EXPECT_EQ(h.terminal->consumed, 0u);
}

TEST(MMRemoteOutputTransportTest, receiptForANeverAdvertisedPlaneIsAProtocolErrorNotADegrade) {
    // No transport linked, so nothing was advertised -- yet the encoder answered with descriptors
    // (version skew, or an encoder bug). Degrading here would be silent corruption: an RDMA
    // receipt has EMPTY inline tensor fields, so the terminal would decode it into a valid-looking
    // but wrong MultimodalOutput. It must fail loudly, and must not spend another ViT forward.
    Harness h(/*with_transport=*/false);
    h.control->responses = {rdmaReceipt({"h0"}, 2, {2})};

    auto result = h.fetch(uniqueEndpoint("never-advertised"));

    ASSERT_FALSE(result.ok());
    EXPECT_EQ(h.control->requests, 1u);
    EXPECT_EQ(h.terminal->consumed, 0u);
    // The slots are real even though the receipt is illegal, so they go back over the control
    // plane instead of waiting out the encoder's 60s GC.
    ASSERT_EQ(h.control->released.size(), 1u);
    EXPECT_EQ(h.control->released[0], std::vector<std::string>({"h0"}));
}

// The real terminal, not the fake: a malformed inline response must come back as an error rather
// than tripping an RTP_LLM_CHECK, which aborts the process under FT_CORE_DUMP_ON_EXCEPTION. This is
// the same reasoning the RDMA manifest path already had; the inline path used to lack it.
TEST(MMRemoteOutputTransportTest, realInlineTerminalDecodesAndRejectsInconsistentResponses) {
    auto              terminal = createGrpcInlineReceiptReader();
    DeadlineBudget    budget(kDefaultVitRpcTimeoutMs);
    FakeControlClient control;
    const std::string endpoint = uniqueEndpoint("inline-real");
    DeliveryContext   context{endpoint, budget, control};

    {  // split_size agrees with the embedding: decoded and split per image
        MultimodalOutputPB receipt;
        receipt.add_split_size(2);
        receipt.add_split_size(3);
        TensorPbConvert::torchToPb(receipt.mutable_multimodal_embedding(), rows(5));

        auto result = terminal->consumeTerminal(receipt, context);
        ASSERT_TRUE(result.ok());
        ASSERT_EQ(result.value().mm_features.size(), 2u);
        EXPECT_EQ(result.value().mm_features[0].size(0), 2);
        EXPECT_EQ(result.value().mm_features[1].size(0), 3);
    }
    {  // split_size sums to 4 but only 5 rows arrived -> error, not a CHECK abort
        MultimodalOutputPB receipt;
        receipt.add_split_size(4);
        TensorPbConvert::torchToPb(receipt.mutable_multimodal_embedding(), rows(5));

        EXPECT_FALSE(terminal->consumeTerminal(receipt, context).ok());
    }
    {  // an RDMA receipt reaching the terminal has EMPTY inline fields: must not decode to garbage
        EXPECT_FALSE(terminal->consumeTerminal(rdmaReceipt({"h0"}, 2, {2}), context).ok());
    }
}

}  // namespace rtp_llm
