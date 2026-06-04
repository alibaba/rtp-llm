#include <atomic>
#include <chrono>
#include <thread>
#include <vector>
#include <gtest/gtest.h>

#include "rtp_llm/cpp/model_rpc/RequestSession.h"
#include "autil/TimeUtility.h"

namespace rtp_llm::test {

// ========================== BoundedRelay Tests ==========================

TEST(BoundedRelayTest, PushAndDrain) {
    BoundedRelay relay(10);
    EXPECT_TRUE(relay.empty());
    EXPECT_EQ(relay.size(), 0);

    GenerateOutputsPB output;
    output.set_request_id(42);
    EXPECT_TRUE(relay.push(output));
    EXPECT_FALSE(relay.empty());
    EXPECT_EQ(relay.size(), 1);

    std::vector<GenerateOutputsPB> drained;
    EXPECT_EQ(relay.drainTo(&drained), 1);
    EXPECT_EQ(drained.size(), 1);
    EXPECT_EQ(drained[0].request_id(), 42);
    EXPECT_TRUE(relay.empty());
}

TEST(BoundedRelayTest, TryPop) {
    BoundedRelay relay(10);

    GenerateOutputsPB out;
    EXPECT_FALSE(relay.tryPop(&out));

    GenerateOutputsPB input;
    input.set_request_id(1);
    relay.push(input);

    EXPECT_TRUE(relay.tryPop(&out));
    EXPECT_EQ(out.request_id(), 1);
    EXPECT_FALSE(relay.tryPop(&out));
}

TEST(BoundedRelayTest, ClosedRelayRejectsPush) {
    BoundedRelay relay(10);
    relay.close();
    EXPECT_TRUE(relay.isClosed());

    GenerateOutputsPB output;
    EXPECT_FALSE(relay.push(output));
}

TEST(BoundedRelayTest, CloseUnblocksWait) {
    BoundedRelay relay(10);

    std::atomic<bool> waited{false};
    std::thread waiter([&] {
        relay.waitForData(std::chrono::milliseconds(5000));
        waited.store(true);
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_FALSE(waited.load());

    relay.close();
    waiter.join();
    EXPECT_TRUE(waited.load());
}

TEST(BoundedRelayTest, CapBlocksAndUnblocks) {
    BoundedRelay relay(2);

    GenerateOutputsPB o1, o2;
    EXPECT_TRUE(relay.push(o1));
    EXPECT_TRUE(relay.push(o2));

    std::atomic<bool> push_done{false};
    std::thread pusher([&] {
        GenerateOutputsPB o3;
        relay.push(o3);
        push_done.store(true);
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_FALSE(push_done.load());

    GenerateOutputsPB popped;
    relay.tryPop(&popped);

    pusher.join();
    EXPECT_TRUE(push_done.load());
    EXPECT_EQ(relay.size(), 2);
}

TEST(BoundedRelayTest, CapBlockUnblockedByClose) {
    BoundedRelay relay(1);

    GenerateOutputsPB o1;
    relay.push(o1);

    std::atomic<bool> push_returned{false};
    bool              push_result = true;
    std::thread pusher([&] {
        GenerateOutputsPB o2;
        push_result = relay.push(o2);
        push_returned.store(true);
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_FALSE(push_returned.load());

    relay.close();
    pusher.join();
    EXPECT_TRUE(push_returned.load());
    EXPECT_FALSE(push_result);
}

TEST(BoundedRelayTest, WaitForDataReturnsTrueOnData) {
    BoundedRelay relay(10);

    std::thread producer([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        GenerateOutputsPB output;
        relay.push(output);
    });

    bool got_data = relay.waitForData(std::chrono::milliseconds(5000));
    EXPECT_TRUE(got_data);
    producer.join();
}

TEST(BoundedRelayTest, WaitForDataTimesOut) {
    BoundedRelay relay(10);
    bool got_data = relay.waitForData(std::chrono::milliseconds(10));
    EXPECT_FALSE(got_data);
}

TEST(BoundedRelayTest, DrainMultiple) {
    BoundedRelay relay(100);
    for (int i = 0; i < 5; i++) {
        GenerateOutputsPB output;
        output.set_request_id(i);
        relay.push(output);
    }
    EXPECT_EQ(relay.size(), 5);

    std::vector<GenerateOutputsPB> drained;
    EXPECT_EQ(relay.drainTo(&drained), 5);
    EXPECT_TRUE(relay.empty());
    for (int i = 0; i < 5; i++) {
        EXPECT_EQ(drained[i].request_id(), i);
    }
}

// ========================== RequestSession Tests ==========================

TEST(RequestSessionTest, InitialState) {
    int64_t now = autil::TimeUtility::currentTimeInMicroSeconds();
    auto session = std::make_shared<RequestSession>(100, 1, now);

    EXPECT_EQ(session->requestId(), 100);
    EXPECT_EQ(session->batchId(), 1);
    EXPECT_EQ(session->admittedAtUs(), now);
    EXPECT_EQ(session->state(), SessionState::ADMITTED);
    EXPECT_FALSE(session->isTerminal());
    EXPECT_EQ(session->finishedAtUs(), 0);
}

TEST(RequestSessionTest, LeaseAcquireRelease) {
    auto session = std::make_shared<RequestSession>(1, 1, 0);

    EXPECT_FALSE(session->hasConsumer());
    EXPECT_TRUE(session->acquireLease());
    EXPECT_TRUE(session->hasConsumer());
    EXPECT_FALSE(session->acquireLease());

    session->releaseLease();
    EXPECT_FALSE(session->hasConsumer());
    EXPECT_TRUE(session->acquireLease());
}

TEST(RequestSessionTest, CancelWithoutStream) {
    auto session = std::make_shared<RequestSession>(1, 1, 0);
    session->cancel(CancelReason::EXPLICIT_CANCEL);

    EXPECT_EQ(session->state(), SessionState::CANCELLED);
    EXPECT_TRUE(session->isTerminal());
    EXPECT_GT(session->finishedAtUs(), 0);
    EXPECT_EQ(session->cancelReason(), CancelReason::EXPLICIT_CANCEL);
    EXPECT_TRUE(session->getRelay().isClosed());
}

TEST(RequestSessionTest, CancelIdempotent) {
    auto session = std::make_shared<RequestSession>(1, 1, 0);
    session->cancel(CancelReason::EXPLICIT_CANCEL);
    auto finished_at = session->finishedAtUs();

    session->cancel(CancelReason::ATTACH_DEADLINE);
    EXPECT_EQ(session->state(), SessionState::CANCELLED);
    EXPECT_EQ(session->finishedAtUs(), finished_at);
    EXPECT_EQ(session->cancelReason(), CancelReason::EXPLICIT_CANCEL);
}

TEST(RequestSessionTest, MarkFinished) {
    auto session = std::make_shared<RequestSession>(1, 1, 0);
    session->markFinished();

    EXPECT_EQ(session->state(), SessionState::FINISHED);
    EXPECT_TRUE(session->isTerminal());
    EXPECT_GT(session->finishedAtUs(), 0);
    EXPECT_TRUE(session->getRelay().isClosed());
}

TEST(RequestSessionTest, MarkError) {
    auto session = std::make_shared<RequestSession>(1, 1, 0);
    session->markError("test error");

    EXPECT_EQ(session->state(), SessionState::ERROR);
    EXPECT_TRUE(session->isTerminal());
    EXPECT_TRUE(session->getRelay().isClosed());
}

TEST(RequestSessionTest, TerminalStateNotOverwritten) {
    auto session = std::make_shared<RequestSession>(1, 1, 0);
    session->markFinished();

    session->cancel(CancelReason::EXPLICIT_CANCEL);
    EXPECT_EQ(session->state(), SessionState::FINISHED);

    session->markError("error");
    EXPECT_EQ(session->state(), SessionState::FINISHED);
}

TEST(RequestSessionTest, DeriveStateWithoutStream) {
    auto session = std::make_shared<RequestSession>(1, 1, 0);
    EXPECT_EQ(session->deriveState(), SessionState::ADMITTED);
}

TEST(RequestSessionTest, DeriveStateAfterMarkFinished) {
    auto session = std::make_shared<RequestSession>(1, 1, 0);
    session->markFinished();
    EXPECT_EQ(session->deriveState(), SessionState::FINISHED);
}

TEST(RequestSessionTest, DeriveStateAfterMarkError) {
    auto session = std::make_shared<RequestSession>(1, 1, 0);
    session->markError("engine failure");
    EXPECT_EQ(session->deriveState(), SessionState::ERROR);
}

TEST(RequestSessionTest, DeriveStateAfterCancel) {
    auto session = std::make_shared<RequestSession>(1, 1, 0);
    session->cancel(CancelReason::EXPLICIT_CANCEL);
    EXPECT_EQ(session->deriveState(), SessionState::CANCELLED);
}

TEST(RequestSessionTest, ConcurrentDeriveStateAndCancel) {
    for (int round = 0; round < 100; round++) {
        auto session = std::make_shared<RequestSession>(1, 1, 0);
        std::atomic<SessionState> derived_state{SessionState::ADMITTED};

        std::thread canceller([&] {
            session->cancel(CancelReason::SLO_DEADLINE);
        });
        std::thread deriver([&] {
            derived_state.store(session->deriveState());
        });

        canceller.join();
        deriver.join();

        auto final_state = session->state();
        EXPECT_EQ(final_state, SessionState::CANCELLED);
        auto derived = derived_state.load();
        EXPECT_TRUE(derived == SessionState::ADMITTED || derived == SessionState::CANCELLED)
            << "round " << round << ": unexpected state " << static_cast<int>(derived);
    }
}

// ========================== SessionManager Tests ==========================

TEST(SessionManagerTest, RegisterAndLookup) {
    SessionManager mgr;
    auto session = std::make_shared<RequestSession>(100, 1, 0);

    EXPECT_TRUE(mgr.registerSession(100, session));
    EXPECT_EQ(mgr.size(), 1);

    auto [result, found] = mgr.lookup(100);
    EXPECT_EQ(result, LookupResult::RUNNING);
    EXPECT_EQ(found.get(), session.get());
}

TEST(SessionManagerTest, RegisterDuplicate) {
    SessionManager mgr;
    auto s1 = std::make_shared<RequestSession>(100, 1, 0);
    auto s2 = std::make_shared<RequestSession>(100, 2, 0);

    EXPECT_TRUE(mgr.registerSession(100, s1));
    EXPECT_FALSE(mgr.registerSession(100, s2));
    EXPECT_EQ(mgr.size(), 1);
}

TEST(SessionManagerTest, LookupNotFound) {
    SessionManager mgr;
    auto [result, session] = mgr.lookup(999);
    EXPECT_EQ(result, LookupResult::NOT_FOUND);
    EXPECT_EQ(session, nullptr);
}

TEST(SessionManagerTest, LookupFinishedInTTL) {
    SessionManager mgr;
    auto session = std::make_shared<RequestSession>(100, 1, 0);
    mgr.registerSession(100, session);

    session->markFinished();

    auto [result, found] = mgr.lookup(100);
    EXPECT_EQ(result, LookupResult::FINISHED_IN_TTL);
    EXPECT_EQ(found.get(), session.get());
}

TEST(SessionManagerTest, CancelSession) {
    SessionManager mgr;
    auto session = std::make_shared<RequestSession>(100, 1, 0);
    mgr.registerSession(100, session);

    EXPECT_TRUE(mgr.cancelSession(100, CancelReason::SLO_DEADLINE));
    EXPECT_EQ(session->state(), SessionState::CANCELLED);

    EXPECT_FALSE(mgr.cancelSession(999, CancelReason::EXPLICIT_CANCEL));
}

TEST(SessionManagerTest, RemoveSession) {
    SessionManager mgr;
    auto session = std::make_shared<RequestSession>(100, 1, 0);
    mgr.registerSession(100, session);
    EXPECT_EQ(mgr.size(), 1);

    mgr.removeSession(100);
    EXPECT_EQ(mgr.size(), 0);

    auto [result, found] = mgr.lookup(100);
    EXPECT_EQ(result, LookupResult::NOT_FOUND);
}

TEST(SessionManagerTest, GcMovesToTombstone) {
    SessionManager mgr(/*terminal_ttl_us=*/1000);  // 1ms TTL
    auto session = std::make_shared<RequestSession>(100, 1, 0);
    mgr.registerSession(100, session);
    session->markFinished();

    std::this_thread::sleep_for(std::chrono::milliseconds(5));

    auto swept = mgr.gcOnce();
    EXPECT_GE(swept, 1);
    EXPECT_EQ(mgr.size(), 0);

    auto [result, found] = mgr.lookup(100);
    EXPECT_EQ(result, LookupResult::GONE);
}

TEST(SessionManagerTest, GcRemovesTombstone) {
    SessionManager mgr(/*terminal_ttl_us=*/500, /*attach_deadline_us=*/1000000, /*tombstone_ttl_us=*/1000);
    auto session = std::make_shared<RequestSession>(100, 1, 0);
    mgr.registerSession(100, session);
    session->markFinished();

    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    mgr.gcOnce();

    auto [result1, _] = mgr.lookup(100);
    EXPECT_EQ(result1, LookupResult::GONE);

    std::this_thread::sleep_for(std::chrono::milliseconds(3));
    mgr.gcOnce();

    auto [result2, __] = mgr.lookup(100);
    EXPECT_EQ(result2, LookupResult::NOT_FOUND);
}

TEST(SessionManagerTest, GcDoesNotRemoveActive) {
    SessionManager mgr(/*terminal_ttl_us=*/1000);
    auto session = std::make_shared<RequestSession>(100, 1, 0);
    mgr.registerSession(100, session);

    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    auto swept = mgr.gcOnce();
    EXPECT_EQ(swept, 0);
    EXPECT_EQ(mgr.size(), 1);
}

TEST(SessionManagerTest, CancelAll) {
    SessionManager mgr;
    auto s1 = std::make_shared<RequestSession>(1, 1, 0);
    auto s2 = std::make_shared<RequestSession>(2, 1, 0);
    mgr.registerSession(1, s1);
    mgr.registerSession(2, s2);

    mgr.cancelAll();
    EXPECT_EQ(s1->state(), SessionState::CANCELLED);
    EXPECT_EQ(s2->state(), SessionState::CANCELLED);
}

TEST(SessionManagerTest, ReapAttachDeadline) {
    SessionManager mgr(/*terminal_ttl_us=*/1000000, /*attach_deadline_us=*/1000);  // 1ms deadline
    int64_t now = autil::TimeUtility::currentTimeInMicroSeconds();
    auto session = std::make_shared<RequestSession>(100, 1, now);
    mgr.registerSession(100, session);

    EXPECT_EQ(mgr.reapAttachDeadline(), 0);

    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    EXPECT_EQ(mgr.reapAttachDeadline(), 1);
    EXPECT_EQ(session->state(), SessionState::CANCELLED);
    EXPECT_EQ(session->cancelReason(), CancelReason::ATTACH_DEADLINE);
}

TEST(SessionManagerTest, ReapSkipsAttachedSession) {
    SessionManager mgr(/*terminal_ttl_us=*/1000000, /*attach_deadline_us=*/1000);
    int64_t now = autil::TimeUtility::currentTimeInMicroSeconds();
    auto session = std::make_shared<RequestSession>(100, 1, now);
    mgr.registerSession(100, session);
    session->acquireLease();

    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    EXPECT_EQ(mgr.reapAttachDeadline(), 0);
    EXPECT_EQ(session->state(), SessionState::ADMITTED);
}

TEST(SessionManagerTest, StartStopGc) {
    SessionManager mgr(/*terminal_ttl_us=*/1000);
    mgr.startGc();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    mgr.stopGc();
}

// ========================== Lookup ALREADY_ATTACHED ==========================

TEST(SessionManagerTest, LookupAlreadyAttached) {
    SessionManager mgr;
    auto session = std::make_shared<RequestSession>(100, 1, 0);
    mgr.registerSession(100, session);
    session->acquireLease();

    auto [result, found] = mgr.lookup(100);
    EXPECT_EQ(result, LookupResult::ALREADY_ATTACHED);
    EXPECT_EQ(found.get(), session.get());

    session->releaseLease();
    auto [result2, found2] = mgr.lookup(100);
    EXPECT_EQ(result2, LookupResult::RUNNING);
}

// ========================== Lookup GONE ==========================

TEST(SessionManagerTest, LookupGoneAfterGc) {
    SessionManager mgr(/*terminal_ttl_us=*/500, /*attach_deadline_us=*/1000000, /*tombstone_ttl_us=*/500000);
    auto session = std::make_shared<RequestSession>(100, 1, 0);
    mgr.registerSession(100, session);
    session->markFinished();

    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    mgr.gcOnce();

    auto [result, found] = mgr.lookup(100);
    EXPECT_EQ(result, LookupResult::GONE);
    EXPECT_EQ(found, nullptr);
}

// ========================== Cancel then Lookup ==========================

TEST(SessionManagerTest, CancelThenLookupReturnsFinishedInTTL) {
    SessionManager mgr;
    auto session = std::make_shared<RequestSession>(100, 1, 0);
    mgr.registerSession(100, session);

    mgr.cancelSession(100, CancelReason::EXPLICIT_CANCEL);

    auto [result, found] = mgr.lookup(100);
    EXPECT_EQ(result, LookupResult::FINISHED_IN_TTL);
    EXPECT_EQ(found->state(), SessionState::CANCELLED);
}

// ========================== BoundedRelay: cap=0 rejects push ==========================

TEST(BoundedRelayTest, ZeroCapRejectsPush) {
    BoundedRelay relay(0);
    GenerateOutputsPB output;

    std::atomic<bool> push_returned{false};
    bool push_result = true;
    std::thread pusher([&] {
        push_result = relay.push(output);
        push_returned.store(true);
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_FALSE(push_returned.load());

    relay.close();
    pusher.join();
    EXPECT_TRUE(push_returned.load());
    EXPECT_FALSE(push_result);
}

// ========================== RequestSession: markError state ==========================

TEST(RequestSessionTest, MarkErrorThenCancelNoOverwrite) {
    auto session = std::make_shared<RequestSession>(1, 1, 0);
    session->markError("engine failure");
    EXPECT_EQ(session->state(), SessionState::ERROR);

    session->cancel(CancelReason::EXPLICIT_CANCEL);
    EXPECT_EQ(session->state(), SessionState::ERROR);
}

// ========================== RequestSession: cancel closes relay ==========================

TEST(RequestSessionTest, CancelClosesRelay) {
    auto session = std::make_shared<RequestSession>(1, 1, 0, /*is_pd=*/true);
    EXPECT_FALSE(session->getRelay().isClosed());

    session->cancel(CancelReason::SLO_DEADLINE);
    EXPECT_TRUE(session->getRelay().isClosed());
    EXPECT_EQ(session->cancelReason(), CancelReason::SLO_DEADLINE);
}

// ========================== RequestSession: PD relay cap ==========================

TEST(RequestSessionTest, PdSessionHasRelayCap) {
    auto pd_session = std::make_shared<RequestSession>(1, 1, 0, /*is_pd=*/true);
    auto local_session = std::make_shared<RequestSession>(2, 1, 0, /*is_pd=*/false);

    EXPECT_TRUE(pd_session->isPd());
    EXPECT_FALSE(local_session->isPd());

    GenerateOutputsPB output;
    EXPECT_TRUE(pd_session->getRelay().push(output));
    EXPECT_EQ(pd_session->getRelay().size(), 1);
}

// ========================== Concurrent lease: two threads race ==========================

TEST(RequestSessionTest, ConcurrentLeaseRace) {
    auto session = std::make_shared<RequestSession>(1, 1, 0);
    std::atomic<int> winners{0};

    auto try_lease = [&] {
        if (session->acquireLease()) {
            winners.fetch_add(1);
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            session->releaseLease();
        }
    };

    std::thread t1(try_lease);
    std::thread t2(try_lease);
    t1.join();
    t2.join();

    EXPECT_GE(winners.load(), 1);
    EXPECT_LE(winners.load(), 2);
}

// ========================== Relay: push unblocked by close during backpressure ==========================

TEST(BoundedRelayTest, PushUnblockedByCloseUnderBackpressure) {
    BoundedRelay relay(1);
    GenerateOutputsPB fill;
    relay.push(fill);

    std::atomic<bool> push2_done{false};
    bool push2_result = true;
    std::thread pusher([&] {
        GenerateOutputsPB o;
        push2_result = relay.push(o);
        push2_done.store(true);
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_FALSE(push2_done.load());

    relay.close();
    pusher.join();
    EXPECT_TRUE(push2_done.load());
    EXPECT_FALSE(push2_result);
}

// ========================== RelayWriter ==========================

TEST(RelayWriterTest, WritePushesToRelay) {
    BoundedRelay relay(10);
    RelayWriter writer(&relay);

    GenerateOutputsPB output;
    output.set_request_id(42);
    EXPECT_TRUE(writer.Write(output, grpc::WriteOptions()));
    EXPECT_EQ(relay.size(), 1);

    GenerateOutputsPB popped;
    EXPECT_TRUE(relay.tryPop(&popped));
    EXPECT_EQ(popped.request_id(), 42);
}

TEST(RelayWriterTest, WriteReturnsFalseWhenClosed) {
    BoundedRelay relay(10);
    RelayWriter writer(&relay);

    relay.close();

    GenerateOutputsPB output;
    EXPECT_FALSE(writer.Write(output, grpc::WriteOptions()));
}

// ========================== Reaper skips terminal sessions ==========================

TEST(SessionManagerTest, ReapSkipsTerminalSession) {
    SessionManager mgr(/*terminal_ttl_us=*/1000000, /*attach_deadline_us=*/1000);
    int64_t now = autil::TimeUtility::currentTimeInMicroSeconds();
    auto session = std::make_shared<RequestSession>(100, 1, now);
    mgr.registerSession(100, session);
    session->markFinished();

    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    EXPECT_EQ(mgr.reapAttachDeadline(), 0);
}

// ========================== Tombstone independent TTL ==========================

TEST(SessionManagerTest, TombstoneIndependentTtl) {
    SessionManager mgr(/*terminal_ttl_us=*/500, /*attach_deadline_us=*/1000000, /*tombstone_ttl_us=*/2000);
    auto session = std::make_shared<RequestSession>(100, 1, 0);
    mgr.registerSession(100, session);
    session->markFinished();

    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    mgr.gcOnce();
    auto [r1, _1] = mgr.lookup(100);
    EXPECT_EQ(r1, LookupResult::GONE);

    mgr.gcOnce();
    auto [r2, _2] = mgr.lookup(100);
    EXPECT_EQ(r2, LookupResult::GONE);

    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    mgr.gcOnce();
    auto [r3, _3] = mgr.lookup(100);
    EXPECT_EQ(r3, LookupResult::NOT_FOUND);
}

// ========================== drainTo after close still returns residual data ==========================

TEST(BoundedRelayTest, DrainAfterCloseReturnsResidual) {
    BoundedRelay relay(10);
    GenerateOutputsPB o1, o2;
    o1.set_request_id(1);
    o2.set_request_id(2);
    relay.push(o1);
    relay.push(o2);

    relay.close();
    EXPECT_TRUE(relay.isClosed());

    std::vector<GenerateOutputsPB> drained;
    EXPECT_EQ(relay.drainTo(&drained), 2);
    EXPECT_EQ(drained[0].request_id(), 1);
    EXPECT_EQ(drained[1].request_id(), 2);
    EXPECT_TRUE(relay.empty());
}

// ========================== Concurrent cancel + drain ==========================

TEST(RequestSessionTest, ConcurrentCancelAndDrain) {
    auto session = std::make_shared<RequestSession>(1, 1, 0, /*is_pd=*/true);
    auto& relay = session->getRelay();

    for (int i = 0; i < 100; i++) {
        GenerateOutputsPB output;
        output.set_request_id(i);
        relay.push(output);
    }

    std::vector<GenerateOutputsPB> drained;
    std::atomic<bool> cancel_done{false};

    std::thread canceller([&] {
        session->cancel(CancelReason::EXPLICIT_CANCEL);
        cancel_done.store(true);
    });

    std::thread drainer([&] {
        while (!cancel_done.load() || !relay.empty()) {
            std::vector<GenerateOutputsPB> batch;
            relay.drainTo(&batch);
            for (auto& item : batch) {
                drained.push_back(std::move(item));
            }
            if (batch.empty() && !cancel_done.load()) {
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
        }
        std::vector<GenerateOutputsPB> tail;
        relay.drainTo(&tail);
        for (auto& item : tail) {
            drained.push_back(std::move(item));
        }
    });

    canceller.join();
    drainer.join();

    EXPECT_TRUE(session->isTerminal());
    EXPECT_TRUE(relay.isClosed());
    EXPECT_EQ(drained.size(), 100);
}

}  // namespace rtp_llm::test
