#include <mutex>
#include <memory>
#include <functional>
#include <condition_variable>
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeTaskPool.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <new>
#include <stdexcept>
#include <thread>
#include <vector>

#ifdef NDEBUG
#error "CompletionQueueHostTest needs active checks"
#endif

namespace {
std::atomic<size_t> checks{0};
std::atomic<int> fail_after{-1};
}
void* operator new(std::size_t size) {
    if (fail_after.load() >= 0 && fail_after.fetch_sub(1) == 0) throw std::bad_alloc();
    if (void* result = std::malloc(size ? size : 1)) return result;
    throw std::bad_alloc();
}
void* operator new[](std::size_t size) { return ::operator new(size); }
void operator delete(void* value) noexcept { std::free(value); }
void operator delete[](void* value) noexcept { std::free(value); }
void operator delete(void* value, std::size_t) noexcept { std::free(value); }
void operator delete[](void* value, std::size_t) noexcept { std::free(value); }

namespace {
#define CHECK(expression) do { ++checks; if (!(expression)) { std::fprintf(stderr, "%d: %s\n", __LINE__, #expression); std::abort(); } } while (false)
using rtp_llm::BlockTreeTaskPool;
using rtp_llm::BlockTreeTaskClass;
struct Gate {
    std::mutex mutex;
    std::condition_variable cv;
    bool entered{false}, released{false};
    void run() {
        std::unique_lock<std::mutex> lock(mutex);
        entered = true; cv.notify_all();
        CHECK(cv.wait_for(lock, std::chrono::seconds(8), [this] { return released; }));
    }
    void wait() {
        std::unique_lock<std::mutex> lock(mutex);
        CHECK(cv.wait_for(lock, std::chrono::seconds(8), [this] { return entered; }));
    }
    void release() { std::lock_guard<std::mutex> lock(mutex); released = true; cv.notify_all(); }
};

void completionDoesNotConsumeNormalCapacity() {
    auto pool = std::make_shared<BlockTreeTaskPool>(1, 1, "completion_test");
    Gate gate;
    std::atomic<size_t> calls{0};
    CHECK(pool->start());
    CHECK(pool->submit(BlockTreeTaskClass::BACKGROUND, [&] { gate.run(); }));
    gate.wait();
    CHECK(pool->submit(BlockTreeTaskClass::BACKGROUND, [] {}));
    CHECK(!pool->submit(BlockTreeTaskClass::LOAD, [] {}));
    // A queue invariant, not evidence of the SDK submission boundary.
    for (size_t i = 0; i < 256; ++i) CHECK(pool->submitCompletion([&] { ++calls; }));
    pool->stopAdmission();
    CHECK(!pool->submit(BlockTreeTaskClass::LOAD, [] {}));
    CHECK(pool->submitCompletion([&] { ++calls; }));
    CHECK(pool->queueSizes().completion == 257);
    gate.release(); pool->waitForIdle(); pool->shutdown();
    CHECK(calls == 257);
    CHECK(!pool->submitCompletion([] {}));
}

void completionRunsOnWorkerBeforeNormalWorkAndReleasesOwner() {
    auto pool = std::make_shared<BlockTreeTaskPool>(1, 4, "completion_test");
    Gate gate;
    std::vector<int> order;
    auto owner = std::make_shared<int>(7);
    std::weak_ptr<int> weak = owner;
    const auto caller = std::this_thread::get_id();
    std::thread::id worker;
    CHECK(pool->start());
    CHECK(pool->submit(BlockTreeTaskClass::BACKGROUND, [&] { gate.run(); }));
    gate.wait();
    CHECK(pool->submit(BlockTreeTaskClass::LOAD, [&] { order.push_back(1); }));
    CHECK(pool->submitCompletion([&, retained = owner] {
        CHECK(*retained == 7); worker = std::this_thread::get_id(); order.push_back(0);
    }));
    owner.reset(); CHECK(!weak.expired());
    gate.release(); pool->waitForIdle(); pool->shutdown();
    CHECK(order == (std::vector<int>{0, 1}));
    CHECK(worker != caller && weak.expired());
}

void anotherWorkerCanSettleWhileOneCompletionBlocks() {
    auto pool = std::make_shared<BlockTreeTaskPool>(2, 1, "completion_test");
    Gate gate;
    std::promise<void> settled;
    auto result = settled.get_future();
    CHECK(pool->start());
    CHECK(pool->submitCompletion([&] { gate.run(); })); gate.wait();
    CHECK(pool->submitCompletion([&] { settled.set_value(); }));
    const auto state = result.wait_for(std::chrono::seconds(5));
    gate.release(); pool->waitForIdle(); pool->shutdown();
    CHECK(state == std::future_status::ready);
}

void acceptedBusinessPostsAfterAdmissionStops() {
    auto pool = std::make_shared<BlockTreeTaskPool>(1, 4, "completion_test");
    Gate gate;
    std::promise<bool> settled;
    auto result = settled.get_future();
    CHECK(pool->start());
    CHECK(pool->submit(BlockTreeTaskClass::BACKGROUND, [&] { gate.run(); })); gate.wait();
    CHECK(pool->submit(BlockTreeTaskClass::LOAD, [&] {
        if (!pool->submitCompletion([&] { settled.set_value(true); })) settled.set_value(false);
    }));
    pool->stopAdmission(); gate.release();
    CHECK(result.wait_for(std::chrono::seconds(5)) == std::future_status::ready);
    CHECK(result.get()); pool->waitForIdle(); pool->shutdown();
}

void allocationFailurePreservesAcceptedQueueAndPendingCount() {
    auto pool = std::make_shared<BlockTreeTaskPool>(1, 1, "completion_test");
    Gate gate;
    std::atomic<size_t> calls{0};
    CHECK(pool->start());
    CHECK(pool->submitCompletion([&] { gate.run(); })); gate.wait();
    size_t accepted = 0;
    bool failed = false;
    fail_after = 0;
    try {
        for (; accepted < 1024; ++accepted) CHECK(pool->submitCompletion([&] { ++calls; }));
    } catch (const std::bad_alloc&) { failed = true; }
    fail_after = -1;
    CHECK(failed && pool->queueSizes().completion == accepted);
    gate.release(); pool->waitForIdle(); pool->shutdown();
    CHECK(calls == accepted);
}

void throwingCompletionReleasesItsOwnerAndKeepsWorkerAlive() {
    auto pool = std::make_shared<BlockTreeTaskPool>(1, 1, "completion_test");
    auto owner = std::make_shared<int>(7);
    std::weak_ptr<int> weak = owner;
    std::atomic<size_t> calls{0};
    CHECK(pool->start());
    CHECK(pool->submitCompletion([held = owner] { CHECK(*held == 7); throw std::runtime_error("business"); }));
    owner.reset();
    CHECK(pool->submitCompletion([&] { ++calls; }));
    pool->waitForIdle(); pool->shutdown();
    CHECK(weak.expired() && calls == 1);
}
}  // namespace

int main() {
    completionDoesNotConsumeNormalCapacity();
    completionRunsOnWorkerBeforeNormalWorkAndReleasesOwner();
    anotherWorkerCanSettleWhileOneCompletionBlocks();
    acceptedBusinessPostsAfterAdmissionStops();
    allocationFailurePreservesAcceptedQueueAndPendingCount();
    throwingCompletionReleasesItsOwnerAndKeepsWorkerAlive();
    std::printf("CompletionQueueHostTest: 6 original-pool scenarios, %zu checks PASS\n", checks.load());
}
