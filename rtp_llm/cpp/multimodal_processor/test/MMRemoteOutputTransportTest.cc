#include <memory>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "rtp_llm/cpp/model_rpc/TensorPbConvert.h"
#include "rtp_llm/cpp/multimodal_processor/transport/MMRemoteOutputTransport.h"
#include "rtp_llm/cpp/multimodal_processor/transport/grpc/MMGrpcTransport.h"
#include "rtp_llm/cpp/multimodal_processor/transport/rdma/MMRdmaReceiptReader.h"

// Orchestration of the LLM-side output transport: which reader gets a receipt, what a degrade
// costs, and what happens when the encoder answers with a data plane nobody asked for. The data
// plane itself (slot planning, manifest reassembly) is covered by MMRdmaTransportTest.cc; here
// both the control plane and the RDMA transport are fakes, so nothing needs hardware or a server.
namespace rtp_llm {

namespace {

// Each test uses its own endpoint string: RdmaCircuitBreaker keeps its table in process-wide
// statics (deliberately -- see its comment), so sharing a name would leak failures between cases.
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
        auto* desc = receipt.add_output_rdma_slots();
        desc->set_handle(handle);
        auto* tensor = desc->add_tensors();
        tensor->set_role(MMRdmaTensorPB::EMBEDDING);
        tensor->add_shape(chunk_rows);
        tensor->add_shape(4);
        tensor->set_nbytes(chunk_rows * 4 * 4);
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

    ErrorResult<MultimodalOutputPB>
    request(const std::string&, MultimodalInputsPB& request_pb, DeadlineBudget&) override {
        advertised_rdma.push_back(request_pb.support_rdma());
        if (log) {
            log->push_back("request");
        }
        const size_t round = requests++;
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
};

// Only the LLM half matters here; the encoder half is never called.
class FakeRdmaTransport: public MMRdmaTransport {
public:
    bool                      read_ok = true;
    std::vector<std::string>* log     = nullptr;
    size_t                    reads   = 0;

    bool exportEmbedding(const std::vector<torch::Tensor>&,
                         const std::vector<MMRdmaTensorPB::Role>&,
                         MMRdmaDescPB*) override {
        return false;
    }

    void releaseEmbedding(const std::vector<std::string>&) override {}

    bool readEmbedding(const MMRdmaDescPB& desc, std::vector<torch::Tensor>* out, int64_t) override {
        ++reads;
        if (log) {
            log->push_back("read");
        }
        if (!read_ok) {
            return false;
        }
        for (int i = 0; i < desc.tensors_size(); ++i) {
            out->push_back(rows(desc.tensors(i).shape(0)));
        }
        return true;
    }
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
    explicit Harness(bool with_transport = true) {
        auto metrics    = std::make_shared<const MMTransportMetrics>(nullptr);
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
        readers.push_back(std::make_unique<RdmaReceiptReader>(transport_sp, metrics));
        under_test = std::make_unique<MMRemoteOutputTransport>(
            std::move(readers), std::move(terminal_up), std::move(control_up), metrics);
    }

    ErrorResult<MultimodalOutput> fetch(const std::string& endpoint) {
        MultimodalInputsPB request_pb;
        request_pb.add_multimodal_inputs();
        return under_test->fetch(endpoint, request_pb);
    }
};

}  // namespace

TEST(MMRemoteOutputTransportTest, inlineReceiptGoesToTheTerminalAndNeverTouchesRdma) {
    Harness h;
    h.control->responses = {inlineReceipt(2)};

    auto result = h.fetch(uniqueEndpoint("inline"));

    ASSERT_TRUE(result.ok());
    EXPECT_EQ(h.control->requests, 1u);
    EXPECT_EQ(h.transport->reads, 0u);
    EXPECT_EQ(h.terminal->consumed, 1u);
    EXPECT_TRUE(h.control->released.empty());
}

TEST(MMRemoteOutputTransportTest, rdmaReceiptIsReadAndSlotsAreReleasedOnce) {
    Harness h;
    h.control->responses = {rdmaReceipt({"h0", "h1"}, /*chunk_rows=*/2, /*split_size=*/{2, 2})};

    auto result = h.fetch(uniqueEndpoint("rdma-hit"));

    ASSERT_TRUE(result.ok());
    EXPECT_EQ(h.transport->reads, 2u);
    // One release RPC carrying every handle, not one per slot.
    ASSERT_EQ(h.control->released.size(), 1u);
    EXPECT_EQ(h.control->released[0], std::vector<std::string>({"h0", "h1"}));
    EXPECT_EQ(result.value().mm_features.size(), 2u);
    EXPECT_TRUE(h.control->advertised_rdma[0]);
}

TEST(MMRemoteOutputTransportTest, readFailureDegradesInExactlyOneExtraRoundWithCapabilityWithdrawn) {
    Harness h;
    h.control->responses  = {rdmaReceipt({"h0"}, 2, {2}), inlineReceipt(2)};
    h.transport->read_ok = false;

    auto result = h.fetch(uniqueEndpoint("degrade"));

    ASSERT_TRUE(result.ok());
    // Two ViT forwards, never three: the degrade is a bounded retry, not a candidate loop.
    EXPECT_EQ(h.control->requests, 2u);
    ASSERT_EQ(h.control->advertised_rdma.size(), 2u);
    EXPECT_TRUE(h.control->advertised_rdma[0]);
    EXPECT_FALSE(h.control->advertised_rdma[1]);  // withdrawn before the retry
}

TEST(MMRemoteOutputTransportTest, slotsAreReleasedAfterReadingEvenWhenTheReadFails) {
    Harness h;
    h.control->responses  = {rdmaReceipt({"h0"}, 2, {2}), inlineReceipt(2)};
    h.transport->read_ok = false;

    ASSERT_TRUE(h.fetch(uniqueEndpoint("release-on-failure")).ok());

    ASSERT_EQ(h.control->released.size(), 1u);
    EXPECT_EQ(h.control->released[0], std::vector<std::string>({"h0"}));
    // Ordering is the point, not just the fact: telling the encoder to reclaim the slot before
    // the READ finished would let it reuse memory still being read.
    const std::vector<std::string> expected{"request", "read", "release", "request", "terminal"};
    EXPECT_EQ(h.log, expected);
}

TEST(MMRemoteOutputTransportTest, inconsistentManifestDegradesAndStillReleases) {
    Harness h;
    // split_size sums to 3 but the single chunk carries 2 rows -> assemble rejects it.
    h.control->responses = {rdmaReceipt({"h0"}, /*chunk_rows=*/2, /*split_size=*/{3}), inlineReceipt(3)};

    auto result = h.fetch(uniqueEndpoint("manifest"));

    ASSERT_TRUE(result.ok());
    EXPECT_EQ(h.control->requests, 2u);
    ASSERT_EQ(h.control->released.size(), 1u);
    EXPECT_EQ(h.log, std::vector<std::string>({"request", "read", "release", "request", "terminal"}));
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

TEST(MMRemoteOutputTransportTest, descriptorInTheDegradedRoundIsAProtocolError) {
    Harness h;
    // The encoder answers the capability-free retry with descriptors anyway.
    h.control->responses  = {rdmaReceipt({"h0"}, 2, {2}), rdmaReceipt({"h1"}, 2, {2})};
    h.transport->read_ok = false;

    auto result = h.fetch(uniqueEndpoint("degraded-descriptor"));

    ASSERT_FALSE(result.ok());
    EXPECT_EQ(h.control->requests, 2u);
    // Both rounds allocated slots on the encoder: the first via the read attempt's lease, the
    // second via discard of the illegal fallback receipt. Neither may be left to the GC.
    ASSERT_EQ(h.control->released.size(), 2u);
    EXPECT_EQ(h.control->released[0], std::vector<std::string>({"h0"}));
    EXPECT_EQ(h.control->released[1], std::vector<std::string>({"h1"}));
}

TEST(MMRemoteOutputTransportTest, controlPlaneErrorIsReturnedWithoutADegrade) {
    Harness h;
    h.control->responses = {};  // first request already fails

    auto result = h.fetch(uniqueEndpoint("rpc-error"));

    ASSERT_FALSE(result.ok());
    EXPECT_EQ(h.control->requests, 1u);
    EXPECT_EQ(h.transport->reads, 0u);
}

TEST(MMRemoteOutputTransportTest, circuitBreakerStopsAdvertisingAfterRepeatedFailures) {
    const auto endpoint = uniqueEndpoint("circuit");
    // A fresh Harness per attempt mirrors production (one fetch per request) while the breaker's
    // process-wide table accumulates across them.
    for (int attempt = 0; attempt < RdmaCircuitBreaker::kFailuresToOpen; ++attempt) {
        Harness h;
        h.control->responses  = {rdmaReceipt({"h0"}, 2, {2}), inlineReceipt(2)};
        h.transport->read_ok = false;
        ASSERT_TRUE(h.fetch(endpoint).ok());
        EXPECT_TRUE(h.control->advertised_rdma[0]);
    }

    Harness after;
    after.control->responses = {inlineReceipt(2)};
    ASSERT_TRUE(after.fetch(endpoint).ok());

    // Circuit open: one round trip, nothing advertised, no wasted RDMA attempt.
    EXPECT_EQ(after.control->requests, 1u);
    ASSERT_EQ(after.control->advertised_rdma.size(), 1u);
    EXPECT_FALSE(after.control->advertised_rdma[0]);
    EXPECT_EQ(after.transport->reads, 0u);
}

TEST(MMRemoteOutputTransportTest, budgetIsDerivedFromTheRequestTimeout) {
    MultimodalInputsPB request_pb;
    EXPECT_EQ(resolveRpcTimeoutMs(request_pb), kDefaultVitRpcTimeoutMs);

    request_pb.add_multimodal_inputs()->mutable_mm_preprocess_config()->set_mm_timeout_ms(1200);
    request_pb.add_multimodal_inputs()->mutable_mm_preprocess_config()->set_mm_timeout_ms(3400);
    // The longest wins: one slow image must not shorten the whole request's budget.
    EXPECT_EQ(resolveRpcTimeoutMs(request_pb), 3400);
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

TEST(MMRemoteOutputTransportTest, exhaustedBudgetIsAnErrorNotASilentDegrade) {
    DeadlineBudget spent(0);
    EXPECT_TRUE(spent.exhausted());
    EXPECT_LE(spent.remainingMs(), 0);

    DeadlineBudget fresh(kDefaultVitRpcTimeoutMs);
    EXPECT_FALSE(fresh.exhausted());
    EXPECT_GT(fresh.remainingMs(), 0);
}

}  // namespace rtp_llm
