#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeTaskPool.h"
#include <thread>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <new>
#include <stdexcept>
#include <vector>

#ifdef NDEBUG
#error "ReservedCompletionLaneHostTest needs active checks"
#endif

namespace {
std::atomic<size_t> allocations{0}, checks{0};
std::atomic<int> fail_after{-1};
}
void* operator new(std::size_t size) {
    ++allocations;
    const int next = fail_after.load();
    if (next >= 0 && fail_after.fetch_sub(1) == 0) throw std::bad_alloc();
    if (void* result = std::malloc(size ? size : 1)) return result;
    throw std::bad_alloc();
}
void* operator new[](std::size_t size) { return ::operator new(size); }
void operator delete(void* value) noexcept { std::free(value); }
void operator delete[](void* value) noexcept { std::free(value); }
void operator delete(void* value, std::size_t) noexcept { std::free(value); }
void operator delete[](void* value, std::size_t) noexcept { std::free(value); }

namespace {
// Test-only spelling for the original pool's completion reservation API.
struct Lane: rtp_llm::BlockTreeTaskPool {
    using Ticket = Completion;
    Lane(size_t capacity, size_t workers): BlockTreeTaskPool(workers, 0, "completion_test", capacity) {}
    static std::shared_ptr<Lane> create(size_t capacity, size_t workers = 1) {
        if (!capacity || !workers) throw std::invalid_argument("invalid test completion configuration");
        return std::make_shared<Lane>(capacity, workers);
    }
    auto reserve(std::function<void()> task, bool load) {
        return reserveCompletion(std::move(task), load ? rtp_llm::BlockTreeTaskClass::LOAD : rtp_llm::BlockTreeTaskClass::BACKGROUND);
    }
    auto snapshot() const { return completionSnapshot(); }
    void stopAdmission() { stopCompletionAdmission(); }
    void waitForIdle() { waitForCompletionIdle(); }
};
#define CHECK(expression) do { ++checks; if (!(expression)) { std::fprintf(stderr, "%d: %s\n", __LINE__, #expression); std::abort(); } } while (false)
struct Gate {
    std::mutex mutex; std::condition_variable cv;
    bool entered{false}, released{false};
    void run() {
        std::unique_lock<std::mutex> lock(mutex); entered = true; cv.notify_all();
        CHECK(cv.wait_for(lock, std::chrono::seconds(8), [this] { return released; }));
    }
    void wait() {
        std::unique_lock<std::mutex> lock(mutex);
        CHECK(cv.wait_for(lock, std::chrono::seconds(8), [this] { return entered; }));
    }
    void release() { std::lock_guard<std::mutex> lock(mutex); released = true; cv.notify_all(); }
};

void preallocationAndAbandonNeverRunBusiness() {
    bool invalid = false;
    try { (void)Lane::create(0); } catch (const std::invalid_argument&) { invalid = true; }
    CHECK(invalid);
    invalid = false;
    try { (void)Lane::create(1, 0); } catch (const std::invalid_argument&) { invalid = true; }
    CHECK(invalid);
    auto lane = Lane::create(1); CHECK(lane->start()); CHECK(!lane->start());
    std::atomic<int> calls{0};
    auto ticket = lane->reserve([&] { ++calls; }, false); CHECK(ticket);
    CHECK(lane->snapshot().outstanding == 1 && calls == 0);
    CHECK(!lane->reserve([&] { ++calls; }, true));
    CHECK(lane->snapshot().rejected == 1);
    ticket.reset(); lane->waitForIdle(); CHECK(calls == 0);
    CHECK(lane->snapshot().outstanding == 0);
    lane->shutdown(); CHECK(!lane->reserve([] {}, false));
}

void postAllocatesNothingAndNeverRunsInline() {
    auto lane = Lane::create(2); CHECK(lane->start()); Gate gate;
    auto blocker = lane->reserve([&] { gate.run(); }, false); CHECK(blocker->post()); gate.wait();
    const auto notifier = std::this_thread::get_id(); std::thread::id business;
    std::atomic<int> calls{0};
    auto ticket = lane->reserve([&] { business = std::this_thread::get_id(); ++calls; }, true);
    const auto before = allocations.load();
    CHECK(ticket->post()); CHECK(!ticket->post());
    CHECK(allocations.load() == before); CHECK(calls == 0);
    CHECK(lane->snapshot().queued_load == 1);
    gate.release(); lane->waitForIdle(); CHECK(calls == 1 && business != notifier);
    CHECK(lane->snapshot().outstanding == 0);
    lane->shutdown();
}

void backgroundSaturationLeavesLoadAdmissionAndAcceptedPostsIntact() {
    for (const size_t capacity : {size_t{2}, size_t{8}}) {
        auto lane = Lane::create(capacity); CHECK(lane->start());
        const size_t writes = capacity / 2;
        std::atomic<size_t> completed{0};
        std::vector<std::shared_ptr<Lane::Ticket>> tickets;
        for (size_t i = 0; i < writes; ++i) {
            tickets.push_back(lane->reserve([&] { ++completed; }, false)); CHECK(tickets.back());
        }
        CHECK(!lane->reserve([] {}, false)); // Outstanding writes have not even posted their results yet.
        for (size_t i = writes; i < capacity; ++i) {
            tickets.push_back(lane->reserve([&] { ++completed; }, true)); CHECK(tickets.back());
        }
        CHECK(!lane->reserve([] {}, true)); CHECK(lane->snapshot().outstanding == capacity);
        lane->stopAdmission();
        for (const auto& ticket : tickets) CHECK(ticket->post());
        lane->waitForIdle(); CHECK(completed == capacity); CHECK(lane->snapshot().outstanding == 0);
        lane->shutdown();
    }
}

void stopAdmissionStillHonorsAnAcceptedReservation() {
    auto lane = Lane::create(1); CHECK(lane->start()); std::atomic<int> calls{0};
    auto ticket = lane->reserve([&] { ++calls; }, false);
    lane->stopAdmission(); CHECK(!lane->reserve([] {}, true));
    CHECK(ticket->post()); lane->waitForIdle(); CHECK(calls == 1);
    lane->shutdown();
}

void shutdownWaitsForPreReservedButNotYetPostedResult() {
    auto lane = Lane::create(1); CHECK(lane->start()); std::atomic<int> calls{0};
    auto ticket = lane->reserve([&] { ++calls; }, false);
    Gate begun; std::atomic<bool> stopped{false};
    std::thread stopper([&] { begun.release(); lane->shutdown(); stopped = true; });
    { std::unique_lock<std::mutex> lock(begun.mutex);
      CHECK(begun.cv.wait_for(lock, std::chrono::seconds(8), [&] { return begun.released; })); }
    CHECK(!stopped); CHECK(ticket->post()); stopper.join();
    CHECK(stopped && calls == 1 && lane->snapshot().outstanding == 0);
}

void loadPriorityHasFiniteBurstForBackgroundBusiness() {
    auto lane = Lane::create(7); CHECK(lane->start()); Gate gate;
    auto blocker = lane->reserve([&] { gate.run(); }, false); CHECK(blocker->post()); gate.wait();
    std::vector<int> order; order.reserve(6);
    std::vector<std::shared_ptr<Lane::Ticket>> tickets; tickets.reserve(6);
    tickets.push_back(lane->reserve([&] { order.push_back(99); }, false)); CHECK(tickets.back()->post());
    for (int i = 0; i < 5; ++i) {
        tickets.push_back(lane->reserve([&, i] { order.push_back(i); }, true)); CHECK(tickets.back()->post());
    }
    gate.release(); lane->waitForIdle();
    CHECK(order == (std::vector<int>{0, 1, 2, 3, 99, 4})); lane->shutdown();
}

void parallelCompletionDoesNotSerializeLoadBehindOneBlockedWrite() {
    auto lane = Lane::create(2, 2); CHECK(lane->start()); Gate write;
    auto blocked = lane->reserve([&] { write.run(); }, false); CHECK(blocked->post()); write.wait();
    std::promise<void> finished;
    auto done = finished.get_future();
    auto load = lane->reserve([&] { finished.set_value(); }, true); CHECK(load->post());
    const auto status = done.wait_for(std::chrono::seconds(5));
    write.release(); lane->waitForIdle(); lane->shutdown();
    CHECK(status == std::future_status::ready);
}

void concurrentNotificationClaimsExactlyOnce() {
    auto lane = Lane::create(1); CHECK(lane->start()); std::atomic<int> calls{0}, accepted{0};
    auto ticket = lane->reserve([&] { ++calls; }, true);
    std::vector<std::thread> threads;
    for (int i = 0; i < 16; ++i) threads.emplace_back([&] { if (ticket->post()) ++accepted; });
    for (auto& thread : threads) thread.join();
    lane->waitForIdle(); CHECK(calls == 1 && accepted == 1); lane->shutdown();
}

void allocationFailureCannotDebitAnExistingReservation() {
    auto lane = Lane::create(3); CHECK(lane->start());
    auto existing = lane->reserve([] {}, false);
    // The merged callback record has one allocation, instead of a node,
    // ticket and ticket control block. Fail that allocation before admission.
    fail_after = 0; bool rejected = false;
    try { (void)lane->reserve([] {}, true); } catch (const std::bad_alloc&) { rejected = true; }
    fail_after = -1;
    CHECK(rejected); CHECK(lane->snapshot().outstanding == 1);
    auto next = lane->reserve([] {}, true); CHECK(next);
    CHECK(lane->snapshot().outstanding == 2);
    next.reset(); CHECK(lane->snapshot().outstanding == 1);
    existing.reset(); lane->waitForIdle(); CHECK(lane->snapshot().outstanding == 0); lane->shutdown();
}

void queuedBusinessMayReserveAfterNormalAdmissionStops() {
    auto pool = std::make_shared<rtp_llm::BlockTreeTaskPool>(1, 4, "queued_reserve", 2);
    CHECK(pool->start()); Gate gate;
    CHECK(pool->submit(rtp_llm::BlockTreeTaskClass::BACKGROUND, [&] { gate.run(); }));
    gate.wait();
    std::promise<bool> settled; auto result = settled.get_future();
    CHECK(pool->submit(rtp_llm::BlockTreeTaskClass::LOAD, [&] {
        auto ticket = pool->reserveCompletion([&] { settled.set_value(true); }, rtp_llm::BlockTreeTaskClass::LOAD);
        if (!ticket || !ticket->post()) settled.set_value(false);
    }));
    pool->stopAdmission(); // The queued operation is already accepted.
    CHECK(!pool->submit(rtp_llm::BlockTreeTaskClass::LOAD, [] {}));
    gate.release();
    CHECK(result.wait_for(std::chrono::seconds(5)) == std::future_status::ready);
    CHECK(result.get()); pool->shutdown();
    CHECK(pool->completionSnapshot().outstanding == 0);
}

void thrownBusinessAndLateTicketDestructionKeepWorkerAndOwnersSafe() {
    auto lane = Lane::create(2); CHECK(lane->start());
    auto real_owner = std::make_shared<int>(7); const std::weak_ptr<int> weak = real_owner;
    auto first = lane->reserve([owner = real_owner] { if (*owner == 7) throw std::runtime_error("business"); }, false);
    real_owner.reset(); CHECK(!weak.expired()); CHECK(first->post());
    lane->waitForIdle(); CHECK(weak.expired()); CHECK(lane->snapshot().outstanding == 0);
    std::atomic<int> calls{0}; auto next = lane->reserve([&] { ++calls; }, true); CHECK(next->post());
    lane->waitForIdle(); CHECK(calls == 1); lane->shutdown();
    lane.reset(); first.reset(); next.reset(); // Tickets may outlive a stopped executor/lane.
}
} // namespace

int main() {
    preallocationAndAbandonNeverRunBusiness(); postAllocatesNothingAndNeverRunsInline();
    backgroundSaturationLeavesLoadAdmissionAndAcceptedPostsIntact();
    stopAdmissionStillHonorsAnAcceptedReservation(); shutdownWaitsForPreReservedButNotYetPostedResult();
    loadPriorityHasFiniteBurstForBackgroundBusiness(); parallelCompletionDoesNotSerializeLoadBehindOneBlockedWrite();
    concurrentNotificationClaimsExactlyOnce();
    allocationFailureCannotDebitAnExistingReservation(); queuedBusinessMayReserveAfterNormalAdmissionStops();
    thrownBusinessAndLateTicketDestructionKeepWorkerAndOwnersSafe();
    std::printf("ReservedCompletionLaneHostTest: 11 original-pool scenarios, %zu checks PASS\n", checks.load());
}
