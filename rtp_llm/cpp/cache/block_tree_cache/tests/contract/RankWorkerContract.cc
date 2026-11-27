#include "block_tree_cache_worker.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <mutex>
#include <thread>
#include <vector>

namespace {

void check(bool result, const char* expression, int line) {
    if (!result) {
        std::fprintf(stderr, "RankWorkerContract:%d: %s\n", line, expression);
        std::abort();
    }
}

#define CHECK(expression) check((expression), #expression, __LINE__)

struct Batch {
    size_t                first;
    size_t                count;
    btc_rank_copy_done_fn done;
    void*                 context;
};

struct CopyFixture {
    std::vector<Batch> take(size_t count) {
        std::unique_lock lock(mutex);
        // A synchronous owner could not dispatch all four batches while
        // completion is withheld. Fail without hanging the test process.
        CHECK(ready.wait_for(lock, std::chrono::seconds(10), [&] { return batches.size() == count; }));
        std::vector<Batch> result;
        result.swap(batches);
        return result;
    }

    std::array<int, 100> device{};
    std::array<int, 100> host{};
    std::vector<Batch>   batches;
    std::mutex           mutex;
    std::condition_variable ready;
    std::atomic<size_t>  released{0};
    std::thread::id      caller{std::this_thread::get_id()};
};

void delayedSubmit(void* opaque, size_t first, size_t count, btc_rank_copy_done_fn done, void* context) {
    auto& fixture = *static_cast<CopyFixture*>(opaque);
    CHECK(std::this_thread::get_id() != fixture.caller);
    {
        std::lock_guard lock(fixture.mutex);
        fixture.batches.push_back({first, count, done, context});
    }
    fixture.ready.notify_one();
}

void inlineFailure(void*, size_t, size_t, btc_rank_copy_done_fn done, void* context) {
    done(context, false);
}

void released(void* opaque) {
    auto& fixture = *static_cast<CopyFixture*>(opaque);
    ++fixture.released;
}

}  // namespace

int main() {
    const btc_rank_worker_config config{BTC_RANK_WORKER_ABI_VERSION, 0, 2, 100, 100};
    btc_rank_worker* worker = nullptr;
    CHECK(btc_rank_worker_open(&config, &worker) == BTC_OK);

    CopyFixture fixture;
    std::vector<btc_rank_page> pages;
    for (int i = 0; i < 75; ++i) {
        fixture.device[i + 1] = i * 3 + 11;
        pages.push_back({i < 70 ? 0u : 1u, i + 1, i});
    }
    const auto send = [&](uint64_t id, btc_rank_submit_fn submit = delayedSubmit) {
        return btc_rank_worker_submit(worker, id, 0, pages.data(), pages.size(), &fixture, submit, released);
    };

    CHECK(send(1) == BTC_OK);
    CHECK(send(1) == BTC_INVALID);
    CHECK(fixture.released == 1);
    auto batches = fixture.take(4);
    std::sort(batches.begin(), batches.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    CHECK((std::array<size_t, 4>{batches[0].count, batches[1].count, batches[2].count, batches[3].count}
            == std::array<size_t, 4>{32, 32, 6, 5}));
    auto complete = [&](const Batch& batch) {
        for (size_t index = batch.first; index < batch.first + batch.count; ++index) {
            const auto& row = pages[index];
            fixture.host[row.host_slot] = fixture.device[row.device_slot];
        }
        batch.done(batch.context, true);
    };
    btc_rank_completion result{};
    for (size_t batch = batches.size(); batch > 1; --batch) {
        complete(batches[batch - 1]);
        CHECK(btc_rank_worker_poll(worker, 1, &result) == BTC_PENDING);
        CHECK(fixture.released == 1);
    }
    complete(batches.front());
    CHECK(btc_rank_worker_wait(worker, 1, &result) == BTC_OK);
    CHECK(result.result == BTC_SUCCESS && result.safe);
    CHECK(fixture.released == 2);
    CHECK(btc_rank_worker_poll(worker, 1, &result) == BTC_INVALID);
    for (const auto& row : pages)
        CHECK(fixture.host[row.host_slot] == fixture.device[row.device_slot]);

    pages.resize(1);
    CHECK(send(2, inlineFailure) == BTC_OK);
    CHECK(btc_rank_worker_wait(worker, 2, &result) == BTC_OK);
    CHECK(result.result == BTC_FAILED && result.safe);
    CHECK(fixture.released == 3);
    const btc_rank_page invalid{0, 0, 0};
    CHECK(btc_rank_worker_submit(worker, 3, 1, &invalid, 1, &fixture, delayedSubmit, released) == BTC_INVALID);
    CHECK(fixture.released == 4);

    CHECK(send(4) == BTC_OK);
    const auto live = fixture.take(1).front();
    btc_rank_worker_stats stats{};
    CHECK(btc_rank_worker_get_stats(worker, &stats) == BTC_OK);
    CHECK(stats.submitted_jobs == 3 && stats.executed_batches == 6 && stats.failed_batches == 1);
    auto closing = std::async(std::launch::async, [worker] { return btc_rank_worker_close(worker); });
    CHECK(closing.wait_for(std::chrono::milliseconds(25)) == std::future_status::timeout);
    CHECK(fixture.released == 4);
    complete(live);
    CHECK(closing.get() == BTC_OK);
    CHECK(fixture.released == 5);
}
