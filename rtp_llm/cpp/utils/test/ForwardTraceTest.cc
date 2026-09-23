#include "rtp_llm/cpp/utils/ForwardTrace.h"
#include <cassert>
#include <iostream>

// This test intentionally needs no GPU: it checks ownership, parent identity,
// failure visibility, and the disabled fast path. CUDA ordering is a separate
// integration gate; a CPU test cannot prove the absence of GPU synchronization.
int main() {
    using namespace rtp_llm;
    std::cerr << "starting ForwardTrace test" << std::endl;
    {
        ForwardTraceScope off;
        assert(!off);
    }
    std::cerr << "disabled path passed" << std::endl;
    ForwardTraceSession session(0, 4);
    setActiveForwardTrace(&session);
    std::cerr << "session ready" << std::endl;
    auto lengths = torch::tensor({8192, 128}, torch::kInt32);
    {
        ForwardTraceScope parent;
        assert(parent && parent.record().id == 1);
        parent.snapshot("q", lengths);
        lengths.fill_(7);
        assert(parent.record().arrays.at("q").host == std::vector<int64_t>({8192, 128}));
        {
            ForwardTraceScope child;
            assert(child.record().parent_id == parent.record().id);
            child.snapshot("bad_type", torch::ones({2}, torch::kInt64));
            assert(!child.record().arrays.at("bad_type").valid);
        }
        assert(session.parent_id == parent.record().id);
        auto id = recordForwardTraceChunk(2, {5, 1}, {128, 64}, {32640, 0}, 3, 256);
        assert(id == 3);
        assert(session.records[2]->parent_id == 1);
        finishForwardTraceChunk(id);
        assert(session.records[2]->strings.at("status") == "ok");
    }
    assert(session.parent_id == 0);
    assert(session.records[0]->strings.at("status") == "ok");
    try {
        ForwardTraceScope failed;
        throw std::runtime_error("test");
    } catch (const std::runtime_error&) {}
    assert(session.records[3]->strings.at("status") == "error");
    {
        ForwardTraceScope overflow;
        assert(!overflow);
        assert(session.dropped == 1);
    }
    setActiveForwardTrace(nullptr);
    session.seal();
    session.materialize();
    assert(session.records[0]->arrays.at("q").host[0] == 8192);
    assert(session.append() == nullptr);
    ForwardTraceRecord shape;
    shape.integers["logical_sequences"] = 2;
    shape.arrays["input_lengths"].host = {8192, 128};
    shape.arrays["sequence_lengths"].host = {};
    shape.arrays["prefix_lengths"].host = {0, 32640};
    std::vector<int64_t> q, prefix;
    assert(normalizeForwardTraceLengths(shape, q, prefix));
    assert(q == std::vector<int64_t>({8192, 128}));
    assert(q[1] + prefix[1] == 32768);
    // A mixed batch keeps decode rows first, followed by context rows.
    shape.arrays["input_lengths"].host = {4096, 128};
    shape.arrays["sequence_lengths"].host = {8191};
    shape.arrays["prefix_lengths"].host = {32640};
    assert(normalizeForwardTraceLengths(shape, q, prefix));
    assert(q == std::vector<int64_t>({1, 128}));
    assert(prefix == std::vector<int64_t>({8191, 32640}));
    // Decode's original prompt lengths must NOT become Q lengths.
    shape.arrays["sequence_lengths"].host = {32767, 8191};
    shape.arrays["prefix_lengths"].host = {};
    assert(normalizeForwardTraceLengths(shape, q, prefix));
    assert(q == std::vector<int64_t>({1, 1}));
    assert(q[0] + prefix[0] == 32768);
    // MTP physical dummy rows are excluded using the logical count.
    shape.arrays["input_lengths"].host = {3, 3, 3, 3};
    shape.arrays["sequence_lengths"].host = {};
    shape.arrays["prefix_lengths"].host = {32765, 8192, 0, 0};
    assert(normalizeForwardTraceLengths(shape, q, prefix));
    assert(q.size() == 2 && q[0] + prefix[0] == 32768);
    shape.integers["logical_sequences"] = 0;
    assert(normalizeForwardTraceLengths(shape, q, prefix) && q.empty());
    shape.integers.erase("logical_sequences");
    assert(!normalizeForwardTraceLengths(shape, q, prefix));
    shape.strings["kind"] = "chunk";
    shape.arrays["q_lens"].host = {128};
    shape.arrays["prefix_lens"].host = {32640};
    assert(normalizeForwardTraceLengths(shape, q, prefix) && q[0] + prefix[0] == 32768);
    shape.arrays["q_lens"].valid = false;
    assert(!normalizeForwardTraceLengths(shape, q, prefix));
    std::cout << "ForwardTrace CPU ownership, nesting, overflow, exception and length-schema tests passed\n";
}
