#include <cassert>
#include <utility>

#include "../SdkWriteSession.h"

namespace {
namespace sdk = kv_cache_manager;
using rtp_llm::kvcm::SdkWriteSession;
using Proof = rtp_llm::kvcm::WorkerPayloadProof::Result;

void compact_allocations() {
    SdkWriteSession prefix(sdk::WriteLocation{"id", sdk::BlockMaskOffset{2}, {{{"g", "u"}}}});
    assert(prefix.allocatedKeys(3) == std::vector<size_t>{2});
    SdkWriteSession sparse(sdk::WriteLocation{
        "id", sdk::BlockMaskVector{true, false, true, false}, {{{"g", "u"}}, {{"g", "v"}}}});
    assert((sparse.allocatedKeys(4) == std::vector<size_t>{1, 3}));
    SdkWriteSession empty(sdk::WriteLocation{"id", sdk::BlockMaskOffset{2}, {}});
    assert(empty.allocatedKeys(2).empty());
    bool malformed = false;
    try {
        (void)empty.allocatedKeys(3);
    } catch (const std::invalid_argument&) {
        malformed = true;
    }
    assert(malformed);
}

void one_terminal_owner() {
    SdkWriteSession original(sdk::WriteLocation{"lease", sdk::BlockMaskOffset{0}, {{{"g", "u"}}}});
    SdkWriteSession active(std::move(original));
    int attempts = 0;
    auto finish = [&](const std::string& id, bool success) {
        assert(id == "lease" && success);
        ++attempts;
    };
    auto confirm = [&] {
        assert(attempts == 1);
        return true;
    };
    assert(!original.finish(Proof::SUCCESS, [] {}, finish, confirm));
    assert(!active.finish(Proof::UNKNOWN, [] {}, finish, confirm));
    active.abort(Proof::UNKNOWN, false, [](const std::string&, bool) { assert(false); });
    assert(active.finish(Proof::SUCCESS, [] {}, finish, confirm));
    assert(!active.finish(Proof::SUCCESS, [] {}, finish, confirm));
    assert(attempts == 1);
}

void partial_or_unknown_cleanup() {
    SdkWriteSession safe(sdk::WriteLocation{"safe", sdk::BlockMaskOffset{0}, {{{"g", "u"}}}});
    bool retired = false;
    assert(!safe.finish(Proof::BUFFER_SAFE_FAILURE,
                       [] {},
                       [&](const std::string& id, bool success) {
                           assert(id == "safe" && !success);
                           retired = true;
                       },
                       [] { assert(false); return false; }));
    assert(retired);
    safe.abort(Proof::BUFFER_SAFE_FAILURE, false, [](const std::string&, bool) { assert(false); });

    SdkWriteSession unknown(sdk::WriteLocation{"unknown", sdk::BlockMaskOffset{0}, {{{"g", "u"}}}});
    unknown.abort(Proof::UNKNOWN, true, [&](const std::string& id, bool uncertain) {
        assert(id == "unknown" && uncertain);
    });
    assert(!unknown.finish(Proof::SUCCESS, [] {}, [](const std::string&, bool) { assert(false); }, [] { return true; }));
}
}  // namespace

int main() {
    compact_allocations();
    one_terminal_owner();
    partial_or_unknown_cleanup();
}
