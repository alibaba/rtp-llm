#include "rtp_llm/cpp/utils/cuda_call_trace/ActiveRecords.h"
#include "rtp_llm/cpp/utils/cuda_call_trace/RecordQueue.h"

#include <atomic>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <thread>
#include <vector>

using namespace rtp_llm::cuda_trace::detail;
void check(bool condition) {
    if (!condition) {
        std::abort();
    }
}
struct Entry {
    uint64_t id;
    uint64_t words[64];
};
Entry entry(uint64_t id) {
    Entry value{};
    value.id = id;
    for (auto& word : value.words)
        word = id;
    return value;
}
void valid(const Entry& value) {
    for (auto word : value.words)
        check(word == value.id);
}
int main() {
    ActiveRecords<Entry, 4> live;
    RecordQueue<Entry, 4>   history;
    for (uint64_t i = 1; i <= 4; ++i)
        check(history.push(entry(i)));
    const auto old = live.begin(entry(5), 5);
    check(old && live.contains(old));
    check(!history.push(entry(5)));  // Lost ENTER and EXIT must not own live state.
    check(live.close(old));
    check(!live.contains(old));
    check(!history.push(entry(5)));
    unsigned count = 0;
    live.snapshot([&](const Entry&, uint64_t) { ++count; });
    check(count == 0);
    const auto next = live.begin(entry(9), 9);
    check(next && live.contains(next) && !live.contains(old));
    check(!live.close(old));  // A stale generation cannot close the reused slot.
    check(live.close(next));

    uint64_t full[4]{};
    for (uint64_t i = 0; i < 4; ++i) {
        full[i] = live.begin(entry(20 + i), 20 + i);
        check(full[i]);
    }
    check(live.begin(entry(24), 24) == 0);
    for (auto handle : full)
        check(live.close(handle));
    check(!live.contains(ActiveRecords<Entry, 4>::untracked(24)));

    // Exercise read-pin/EXIT/reuse races with private-copy consistency checks.
    // Run this same CPU-only test under ThreadSanitizer when available.
    ActiveRecords<Entry, 64> concurrent;
    std::atomic<bool>        done{false};
    std::atomic<uint64_t>    ids{1}, seen{0};
    std::thread              reader([&] {
        while (!done.load(std::memory_order_acquire)) {
            concurrent.snapshot([&](const Entry& value, uint64_t handle) {
                valid(value);
                check((ActiveRecords<Entry, 64>::id(handle) == value.id));
                ++seen;
            });
        }
    });
    std::vector<std::thread> writers;
    for (unsigned t = 0; t != 8; ++t) {
        writers.emplace_back([&] {
            for (unsigned i = 0; i != 100000; ++i) {
                const auto id     = ids.fetch_add(1, std::memory_order_relaxed);
                const auto handle = concurrent.begin(entry(id), id);
                if (handle)
                    check(concurrent.close(handle));
            }
        });
    }
    for (auto& writer : writers)
        writer.join();
    done.store(true, std::memory_order_release);
    reader.join();
    count = 0;
    concurrent.snapshot([&](const Entry& value, uint64_t) {
        valid(value);
        ++count;
    });
    check(count == 0);
    check(seen.load() > 0);
    std::puts("PASS active ledger survives lost history, exhaustion, concurrent close and reuse");
}
