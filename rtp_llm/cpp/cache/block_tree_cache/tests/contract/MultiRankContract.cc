#include "rtp_llm/cpp/cache/block_tree_cache/transfer/MultiRankBlockTransferEngine.h"

#include <stdexcept>

using namespace rtp_llm;

namespace {
class Delivered final: public RankTransferBroadcast {
public:
    explicit Delivered(std::vector<RankTransferReply> replies): replies_(std::move(replies)) {}
    void waitDone() override {}
    void onDone(DoneCallback callback) override {
        callback();
    }
    bool done() const override {
        return true;
    }
    std::string failureSummary() const override {
        return {};
    }
    std::vector<RankTransferReply> replies() const override {
        return replies_;
    }

private:
    std::vector<RankTransferReply> replies_;
};

RankTransferReply reply(size_t rank, std::optional<TransferResult> result) {
    TransferRankSnapshot snapshot;
    snapshot.rank = rank;
    return {std::move(snapshot), std::move(result)};
}

TransferResult collect(std::vector<RankTransferReply> replies) {
    RankTransferTransport transport;
    transport.prepare = [delivered = std::make_shared<Delivered>(std::move(replies))](const TransferTask&) {
        return RankTransferRequest{{{3, "A"}, {7, "B"}}, [delivered](std::chrono::milliseconds) { return delivered; }};
    };
    MultiRankBlockTransferEngine engine({}, std::move(transport));
    auto task = TransferTask({TransferDescriptor::hostToDevice(0, 5, {2}), TransferDescriptor::hostToDevice(0, 6, {3})},
                             std::chrono::seconds(1));
    return engine.execute(std::move(task))->result();
}

void expectUnknown(std::vector<RankTransferReply> replies) {
    const auto result = collect(std::move(replies));
    if (result.status.code() != ErrorCode::TRANSFER_COMPLETION_UNKNOWN || result.buffer_safe)
        throw std::runtime_error("an incomplete set of rank results allowed publication or reuse");
}
}  // namespace

int main() {
    bool safe = false;
    TransferTask expired({TransferDescriptor::hostToDevice(0, 1, {2})}, std::chrono::milliseconds(0));
    expired.setCompletionProof(std::make_shared<TransferBufferSafety>([&] { safe = true; }));
    MultiRankBlockTransferEngine no_transport({}, RankTransferTransport{});
    auto refused = no_transport.execute(std::move(expired))->result();
    if (refused.status.code() != ErrorCode::DEADLINE_EXCEEDED || !safe || !refused.buffer_safe)
        throw std::runtime_error("an expired transfer must finish safely before touching the transport");
    TransferResult copied;
    copied.buffer_safe = true;
    auto success       = collect({reply(7, copied), reply(3, copied)});
    if (!success.ok() || !success.buffer_safe || success.block_errors.size() != 2
        || success.execution_diagnostics->ranks.size() != 2)
        throw std::runtime_error("complete, out-of-order replies from nonconsecutive ranks must settle");
    auto failed         = TransferResult::executionException(true, "failed copy");
    failed.block_errors = {CacheCopyError::NONE, CacheCopyError::COPY_FAILED};
    auto partial        = collect({reply(3, failed), reply(7, copied)});
    if (partial.ok() || !partial.buffer_safe || partial.block_errors != failed.block_errors)
        throw std::runtime_error("the coordinator must preserve per-descriptor errors and safe completion");
    expectUnknown({reply(3, copied), reply(3, copied)});
    expectUnknown({reply(3, copied), reply(9, copied)});
    expectUnknown({reply(3, failed), reply(7, std::nullopt)});
    failed.block_errors.resize(1);
    expectUnknown({reply(3, copied), reply(7, failed)});
}
