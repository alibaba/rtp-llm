#include "rtp_llm/cpp/utils/cuda_call_trace/RecordQueue.h"

#include <atomic>
#include <cstdlib>
#include <thread>
#include <vector>

using rtp_llm::cuda_trace::detail::RecordQueue;

namespace {
void check(bool condition) {
    if (!condition) {
        std::abort();
    }
}
struct Record {
    uint64_t producer;
    uint64_t sequence;
    uint64_t check;
};
}  // namespace

int main() {
    RecordQueue<uint64_t, 4> bounded;
    for (uint64_t i = 0; i != 4; ++i) {
        check(bounded.push(i));
    }
    check(!bounded.push(4));
    uint64_t value = 0;
    for (uint64_t i = 0; i != 4; ++i) {
        check(bounded.pop(value) && value == i);
    }
    check(!bounded.pop(value));
    for (uint64_t i = 0; i != 10000; ++i) {
        check(bounded.push(i));
        check(bounded.pop(value) && value == i);
    }

    // Deliberately allow producer drops: every published record must be whole,
    // unique, in per-producer order, and accounted for exactly once.
    RecordQueue<Record, 256> queue;
    constexpr uint64_t       producers  = 8;
    constexpr uint64_t       iterations = 100000;
    std::atomic<uint64_t>    done{0}, accepted{0};
    std::vector<std::thread> workers;
    for (uint64_t p = 0; p != producers; ++p) {
        workers.emplace_back([&, p] {
            for (uint64_t n = 1; n <= iterations; ++n) {
                if (queue.push(Record{p, n, (p << 32) ^ n})) {
                    accepted.fetch_add(1, std::memory_order_relaxed);
                }
            }
            done.fetch_add(1, std::memory_order_release);
        });
    }
    uint64_t last[producers] = {};
    uint64_t received        = 0;
    auto     consume         = [&](const Record& r) {
        check(r.producer < producers && r.sequence > last[r.producer]);
        check(r.check == ((r.producer << 32) ^ r.sequence));
        last[r.producer] = r.sequence;
        ++received;
    };
    Record record{};
    while (done.load(std::memory_order_acquire) != producers) {
        if (queue.pop(record)) {
            consume(record);
        } else {
            std::this_thread::yield();
        }
    }
    for (auto& worker : workers) {
        worker.join();
    }
    while (queue.pop(record)) {
        consume(record);
    }
    check(received == accepted.load());
    return 0;
}
