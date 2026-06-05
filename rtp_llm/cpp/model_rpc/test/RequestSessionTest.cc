#include <atomic>
#include <chrono>
#include <thread>
#include <vector>
#include <gtest/gtest.h>

#include "rtp_llm/cpp/model_rpc/RequestSession.h"
#include "autil/TimeUtility.h"

namespace rtp_llm::test {

static int64_t nowUs() { return autil::TimeUtility::currentTimeInMicroSeconds(); }

static SessionCreateOptions makeOpts(int64_t request_id,
                                     int64_t batch_id = 1,
                                     int64_t payload_hash = 42,
                                     size_t max_buffer_bytes = 0) {
    SessionCreateOptions opts;
    opts.request_id = request_id;
    opts.batch_id = batch_id;
    opts.payload_hash = payload_hash;
    opts.create_time_us = nowUs();
    opts.max_buffer_bytes = max_buffer_bytes;
    return opts;
}

static GenerateOutputsPB makeOutput(int64_t id) {
    GenerateOutputsPB out;
    out.set_request_id(id);
    return out;
}

// ========================== RequestOutputBuffer ==========================

TEST(RequestOutputBufferTest, PushAndPopLive) {
    RequestOutputBuffer buf;
    EXPECT_EQ(buf.push(makeOutput(1)), PushResult::OK);
    EXPECT_EQ(buf.push(makeOutput(2)), PushResult::OK);
    EXPECT_EQ(buf.push(makeOutput(3)), PushResult::OK);

    auto pop = buf.popLive(0, 0);
    ASSERT_EQ(pop.status, PopResult::OUTPUT);
    ASSERT_EQ(pop.outputs.size(), 3);
    EXPECT_EQ(pop.outputs[0].request_id(), 1);
    EXPECT_EQ(pop.outputs[1].request_id(), 2);
    EXPECT_EQ(pop.outputs[2].request_id(), 3);
}

TEST(RequestOutputBufferTest, TerminalSnapshotPreservesConsumedPrefix) {
    RequestOutputBuffer buf;
    buf.push(makeOutput(1));
    buf.push(makeOutput(2));

    auto pop = buf.popLive(0, 0);
    ASSERT_EQ(pop.outputs.size(), 2);

    buf.push(makeOutput(3));

    TerminalInfo ti;
    ti.reason = TerminalReason::FINISHED;
    auto snap = buf.freeze(100, 1, ti);

    ASSERT_NE(snap, nullptr);
    ASSERT_EQ(snap->outputs.size(), 3);
    EXPECT_EQ(snap->outputs[0].request_id(), 1);
    EXPECT_EQ(snap->outputs[1].request_id(), 2);
    EXPECT_EQ(snap->outputs[2].request_id(), 3);
}

TEST(RequestOutputBufferTest, SnapshotRereadConsistent) {
    RequestOutputBuffer buf;
    buf.push(makeOutput(1));
    buf.push(makeOutput(2));

    TerminalInfo ti;
    ti.reason = TerminalReason::FINISHED;
    buf.freeze(100, 1, ti);

    auto snap1 = buf.snapshot();
    auto snap2 = buf.snapshot();
    EXPECT_EQ(snap1.get(), snap2.get());
    ASSERT_EQ(snap1->outputs.size(), 2);
}

TEST(RequestOutputBufferTest, CloseWakesBlockedPopLive) {
    RequestOutputBuffer buf;
    std::atomic<bool> popped{false};

    std::thread waiter([&] {
        auto pop = buf.popLive(0, 5000);
        EXPECT_EQ(pop.status, PopResult::CLOSED);
        popped.store(true);
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_FALSE(popped.load());

    buf.close();
    waiter.join();
    EXPECT_TRUE(popped.load());
}

TEST(RequestOutputBufferTest, PopLiveTimeout) {
    RequestOutputBuffer buf;
    auto pop = buf.popLive(0, 10);
    EXPECT_EQ(pop.status, PopResult::WAIT_TIMEOUT);
    EXPECT_TRUE(pop.outputs.empty());
}

TEST(RequestOutputBufferTest, ByteBudget) {
    RequestOutputBuffer buf(10);
    auto big_output = makeOutput(1);
    big_output.set_aux_info("padding_that_makes_it_large_enough_to_exceed_budget");

    auto result = buf.push(std::move(big_output));
    EXPECT_EQ(result, PushResult::BUDGET_EXCEEDED);
}

TEST(RequestOutputBufferTest, PushAfterCloseReturnslosed) {
    RequestOutputBuffer buf;
    buf.close();
    EXPECT_EQ(buf.push(makeOutput(1)), PushResult::CLOSED);
}

TEST(RequestOutputBufferTest, DropSnapshotFreesMemory) {
    RequestOutputBuffer buf;
    buf.push(makeOutput(1));
    buf.push(makeOutput(2));
    TerminalInfo ti;
    buf.freeze(1, 1, ti);
    EXPECT_GT(buf.bytes(), 0);

    buf.dropSnapshot();
    EXPECT_EQ(buf.bytes(), 0);
    EXPECT_EQ(buf.snapshot(), nullptr);
}

TEST(RequestOutputBufferTest, PendingLiveCount) {
    RequestOutputBuffer buf;
    buf.push(makeOutput(1));
    buf.push(makeOutput(2));
    buf.push(makeOutput(3));
    EXPECT_EQ(buf.pendingLiveCount(), 3);

    buf.popLive(0, 0);
    EXPECT_EQ(buf.pendingLiveCount(), 0);
}

// ========================== RequestSession ==========================

TEST(RequestSessionTest, InitialState) {
    auto opts = makeOpts(100);
    auto session = std::make_shared<RequestSession>(opts, 1);

    EXPECT_EQ(session->requestId(), 100);
    EXPECT_EQ(session->sessionEpoch(), 1);
    EXPECT_EQ(session->batchId(), 1);
    EXPECT_EQ(session->payloadHash(), 42);
    EXPECT_FALSE(session->isTerminal());
    EXPECT_FALSE(session->hasConsumer());
}

TEST(RequestSessionTest, SamePayload) {
    auto opts = makeOpts(100, 1, 42);
    auto session = std::make_shared<RequestSession>(opts, 1);
    EXPECT_TRUE(session->samePayload(42));
    EXPECT_FALSE(session->samePayload(99));
}

TEST(RequestSessionTest, FinalizeTerminalCAS) {
    auto opts = makeOpts(100);
    auto session = std::make_shared<RequestSession>(opts, 1);

    auto now = nowUs();
    EXPECT_TRUE(session->finalizeTerminal(TerminalReason::FINISHED, grpc::Status::OK, now));
    EXPECT_TRUE(session->isTerminal());

    EXPECT_FALSE(session->finalizeTerminal(TerminalReason::ERROR,
        grpc::Status(grpc::StatusCode::INTERNAL, "late"), now));

    auto ti = session->terminalInfo();
    EXPECT_EQ(ti.reason, TerminalReason::FINISHED);
    EXPECT_TRUE(ti.status.ok());
}

TEST(RequestSessionTest, CancelCallsFinalizeTerminal) {
    auto opts = makeOpts(100);
    auto session = std::make_shared<RequestSession>(opts, 1);
    auto now = nowUs();

    EXPECT_TRUE(session->cancel(CancelReason::EXPLICIT_CANCEL, now));
    EXPECT_TRUE(session->isTerminal());

    auto ti = session->terminalInfo();
    EXPECT_EQ(ti.reason, TerminalReason::CANCELLED);
}

TEST(RequestSessionTest, CancelIdempotent) {
    auto opts = makeOpts(100);
    auto session = std::make_shared<RequestSession>(opts, 1);
    auto now = nowUs();

    EXPECT_TRUE(session->cancel(CancelReason::EXPLICIT_CANCEL, now));
    EXPECT_FALSE(session->cancel(CancelReason::ATTACH_DEADLINE, now));

    auto ti = session->terminalInfo();
    EXPECT_EQ(ti.reason, TerminalReason::CANCELLED);
}

TEST(RequestSessionTest, FinishThenCancelNoOverwrite) {
    auto opts = makeOpts(100);
    auto session = std::make_shared<RequestSession>(opts, 1);
    auto now = nowUs();

    session->finalizeTerminal(TerminalReason::FINISHED, grpc::Status::OK, now);
    EXPECT_FALSE(session->cancel(CancelReason::EXPLICIT_CANCEL, now));

    auto ti = session->terminalInfo();
    EXPECT_EQ(ti.reason, TerminalReason::FINISHED);
}

TEST(RequestSessionTest, ConcurrentTerminalCAS) {
    for (int round = 0; round < 100; round++) {
        auto opts = makeOpts(round);
        auto session = std::make_shared<RequestSession>(opts, 1);
        std::atomic<int> winners{0};

        auto try_finalize = [&](TerminalReason r) {
            if (session->finalizeTerminal(r, grpc::Status::OK, nowUs())) {
                winners.fetch_add(1);
            }
        };

        std::thread t1([&] { try_finalize(TerminalReason::FINISHED); });
        std::thread t2([&] { try_finalize(TerminalReason::CANCELLED); });
        std::thread t3([&] { try_finalize(TerminalReason::ERROR); });
        t1.join(); t2.join(); t3.join();

        EXPECT_EQ(winners.load(), 1) << "round " << round;
        EXPECT_TRUE(session->isTerminal());
    }
}

TEST(RequestSessionTest, LeaseIdPreventsStaleRelease) {
    auto opts = makeOpts(100);
    auto session = std::make_shared<RequestSession>(opts, 1);

    auto lease_a = session->acquireLiveLease();
    EXPECT_EQ(lease_a.state, AttachState::LIVE);
    auto id_a = lease_a.lease_id;

    session->releaseLiveLease(id_a);
    EXPECT_FALSE(session->hasConsumer());

    auto lease_b = session->acquireLiveLease();
    EXPECT_EQ(lease_b.state, AttachState::LIVE);

    session->releaseLiveLease(id_a);
    EXPECT_TRUE(session->hasConsumer());

    session->releaseLiveLease(lease_b.lease_id);
    EXPECT_FALSE(session->hasConsumer());
}

TEST(RequestSessionTest, LeaseExclusive) {
    auto opts = makeOpts(100);
    auto session = std::make_shared<RequestSession>(opts, 1);

    auto lease1 = session->acquireLiveLease();
    EXPECT_EQ(lease1.state, AttachState::LIVE);

    auto lease2 = session->acquireLiveLease();
    EXPECT_EQ(lease2.state, AttachState::ALREADY_ATTACHED);
}

TEST(RequestSessionTest, ConcurrentLeaseRace) {
    auto opts = makeOpts(100);
    auto session = std::make_shared<RequestSession>(opts, 1);
    std::atomic<int> winners{0};

    auto try_lease = [&] {
        auto result = session->acquireLiveLease();
        if (result.state == AttachState::LIVE) {
            winners.fetch_add(1);
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            session->releaseLiveLease(result.lease_id);
        }
    };

    std::thread t1(try_lease);
    std::thread t2(try_lease);
    t1.join();
    t2.join();

    EXPECT_GE(winners.load(), 1);
    EXPECT_LE(winners.load(), 2);
}

TEST(RequestSessionTest, PushOutputAfterTerminal) {
    auto opts = makeOpts(100);
    auto session = std::make_shared<RequestSession>(opts, 1);
    session->finalizeTerminal(TerminalReason::FINISHED, grpc::Status::OK, nowUs());

    EXPECT_EQ(session->pushOutput(makeOutput(1)), PushResult::CLOSED);
}

TEST(RequestSessionTest, BindStreamAfterTerminal) {
    auto opts = makeOpts(100);
    auto session = std::make_shared<RequestSession>(opts, 1);
    session->finalizeTerminal(TerminalReason::FINISHED, grpc::Status::OK, nowUs());

    EXPECT_FALSE(session->bindStream(nullptr));
}

TEST(RequestSessionTest, BuildLookupLive) {
    auto opts = makeOpts(100);
    auto session = std::make_shared<RequestSession>(opts, 1);
    auto lr = session->buildLookup(nowUs());
    EXPECT_EQ(lr.state, AttachState::LIVE);
}

TEST(RequestSessionTest, BuildLookupFinishedInTTL) {
    auto opts = makeOpts(100);
    opts.payload_ttl_us = 10LL * 60 * 1000 * 1000;
    auto session = std::make_shared<RequestSession>(opts, 1);
    session->pushOutput(makeOutput(1));
    session->finalizeTerminal(TerminalReason::FINISHED, grpc::Status::OK, nowUs());

    auto lr = session->buildLookup(nowUs());
    EXPECT_EQ(lr.state, AttachState::FINISHED_IN_TTL);
    ASSERT_NE(lr.snapshot, nullptr);
    EXPECT_EQ(lr.snapshot->outputs.size(), 1);
}

TEST(RequestSessionTest, PayloadExpired) {
    auto opts = makeOpts(100);
    opts.payload_ttl_us = 1000;
    auto session = std::make_shared<RequestSession>(opts, 1);
    auto now = nowUs();
    session->finalizeTerminal(TerminalReason::FINISHED, grpc::Status::OK, now);

    EXPECT_FALSE(session->payloadExpired(now));
    EXPECT_TRUE(session->payloadExpired(now + 2000));
}

TEST(RequestSessionTest, TerminalClosesBuffer) {
    auto opts = makeOpts(100);
    auto session = std::make_shared<RequestSession>(opts, 1);
    session->pushOutput(makeOutput(1));
    session->finalizeTerminal(TerminalReason::FINISHED, grpc::Status::OK, nowUs());

    EXPECT_EQ(session->pushOutput(makeOutput(2)), PushResult::CLOSED);
}

TEST(RequestSessionTest, SnapshotAfterTerminal) {
    auto opts = makeOpts(100);
    auto session = std::make_shared<RequestSession>(opts, 1);
    session->pushOutput(makeOutput(10));
    session->pushOutput(makeOutput(20));
    session->finalizeTerminal(TerminalReason::FINISHED, grpc::Status::OK, nowUs());

    auto snap = session->snapshot();
    ASSERT_NE(snap, nullptr);
    ASSERT_EQ(snap->outputs.size(), 2);
    EXPECT_EQ(snap->outputs[0].request_id(), 10);
    EXPECT_EQ(snap->outputs[1].request_id(), 20);
    EXPECT_EQ(snap->request_id, 100);
    EXPECT_EQ(snap->session_epoch, 1);
    EXPECT_EQ(snap->terminal.reason, TerminalReason::FINISHED);
}

// ========================== SessionManager ==========================

TEST(SessionManagerTest, CreateAndLookup) {
    SessionManager mgr;
    auto opts = makeOpts(100);
    auto cr = mgr.create(opts);

    EXPECT_EQ(cr.status, BatchEnqueueStatus::ADMITTED);
    ASSERT_NE(cr.session, nullptr);
    EXPECT_GT(cr.session_epoch, 0);
    EXPECT_EQ(mgr.size(), 1);

    auto lr = mgr.lookup(100, cr.session_epoch, nowUs());
    EXPECT_EQ(lr.state, AttachState::LIVE);
    EXPECT_EQ(lr.session.get(), cr.session.get());
}

TEST(SessionManagerTest, CreateIdempotentSamePayload) {
    SessionManager mgr;
    auto opts = makeOpts(100, 1, 42);
    auto cr1 = mgr.create(opts);
    EXPECT_EQ(cr1.status, BatchEnqueueStatus::ADMITTED);

    auto cr2 = mgr.create(opts);
    EXPECT_EQ(cr2.status, BatchEnqueueStatus::ALREADY_ADMITTED);
    EXPECT_EQ(cr2.session_epoch, cr1.session_epoch);
    EXPECT_EQ(cr2.session.get(), cr1.session.get());
    EXPECT_EQ(mgr.size(), 1);
}

TEST(SessionManagerTest, CreateConflictPayload) {
    SessionManager mgr;
    auto opts1 = makeOpts(100, 1, 42);
    mgr.create(opts1);

    auto opts2 = makeOpts(100, 1, 99);
    auto cr = mgr.create(opts2);
    EXPECT_EQ(cr.status, BatchEnqueueStatus::CONFLICT_PAYLOAD);
    EXPECT_EQ(cr.session, nullptr);
    EXPECT_EQ(mgr.size(), 1);
}

TEST(SessionManagerTest, LookupNotFound) {
    SessionManager mgr;
    auto lr = mgr.lookup(999, 0, nowUs());
    EXPECT_EQ(lr.state, AttachState::NOT_FOUND);
    EXPECT_EQ(lr.session, nullptr);
}

TEST(SessionManagerTest, LookupEpochMismatch) {
    SessionManager mgr;
    auto opts = makeOpts(100);
    auto cr = mgr.create(opts);

    auto lr = mgr.lookup(100, cr.session_epoch + 999, nowUs());
    EXPECT_EQ(lr.state, AttachState::EPOCH_MISMATCH);
}

TEST(SessionManagerTest, LookupEpochZeroWildcard) {
    SessionManager mgr;
    auto opts = makeOpts(100);
    mgr.create(opts);

    auto lr = mgr.lookup(100, 0, nowUs());
    EXPECT_EQ(lr.state, AttachState::LIVE);
}

TEST(SessionManagerTest, LookupFinishedInTTL) {
    SessionManager mgr;
    auto opts = makeOpts(100);
    auto cr = mgr.create(opts);
    cr.session->finalizeTerminal(TerminalReason::FINISHED, grpc::Status::OK, nowUs());

    auto lr = mgr.lookup(100, 0, nowUs());
    EXPECT_EQ(lr.state, AttachState::FINISHED_IN_TTL);
    ASSERT_NE(lr.snapshot, nullptr);
}

TEST(SessionManagerTest, CancelSession) {
    SessionManager mgr;
    auto opts = makeOpts(100);
    auto cr = mgr.create(opts);
    auto now = nowUs();

    EXPECT_TRUE(mgr.cancelSession(100, cr.session_epoch, CancelReason::EXPLICIT_CANCEL, now));
    EXPECT_TRUE(cr.session->isTerminal());
    EXPECT_EQ(cr.session->terminalInfo().reason, TerminalReason::CANCELLED);
}

TEST(SessionManagerTest, CancelSessionEpochMismatch) {
    SessionManager mgr;
    auto opts = makeOpts(100);
    auto cr = mgr.create(opts);

    EXPECT_FALSE(mgr.cancelSession(100, cr.session_epoch + 1, CancelReason::EXPLICIT_CANCEL, nowUs()));
    EXPECT_FALSE(cr.session->isTerminal());
}

TEST(SessionManagerTest, CancelSessionNotFound) {
    SessionManager mgr;
    EXPECT_FALSE(mgr.cancelSession(999, 0, CancelReason::EXPLICIT_CANCEL, nowUs()));
}

TEST(SessionManagerTest, GcMovesToTombstone) {
    SessionManager mgr(/*payload_ttl=*/1000, /*attach_deadline=*/1000000, /*tombstone_ttl=*/1000000);
    auto opts = makeOpts(100);
    opts.payload_ttl_us = 1000;
    auto cr = mgr.create(opts);
    cr.session->finalizeTerminal(TerminalReason::FINISHED, grpc::Status::OK, nowUs());

    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    auto swept = mgr.gcOnce();
    EXPECT_GE(swept, 1);
    EXPECT_EQ(mgr.size(), 0);
    EXPECT_EQ(mgr.tombstoneCount(), 1);

    auto lr = mgr.lookup(100, 0, nowUs());
    EXPECT_EQ(lr.state, AttachState::GONE);
}

TEST(SessionManagerTest, GcRemovesTombstone) {
    SessionManager mgr;
    auto opts = makeOpts(100);
    opts.payload_ttl_us = 500;
    opts.tombstone_ttl_us = 1000;
    auto cr = mgr.create(opts);
    cr.session->finalizeTerminal(TerminalReason::FINISHED, grpc::Status::OK, nowUs());

    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    mgr.gcOnce();
    EXPECT_EQ(mgr.tombstoneCount(), 1);
    auto lr1 = mgr.lookup(100, 0, nowUs());
    EXPECT_EQ(lr1.state, AttachState::GONE);

    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    mgr.gcOnce();
    EXPECT_EQ(mgr.tombstoneCount(), 0);
    auto lr2 = mgr.lookup(100, 0, nowUs());
    EXPECT_EQ(lr2.state, AttachState::NOT_FOUND);
}

TEST(SessionManagerTest, GcDoesNotRemoveActive) {
    SessionManager mgr;
    auto opts = makeOpts(100);
    mgr.create(opts);

    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    EXPECT_EQ(mgr.gcOnce(), 0);
    EXPECT_EQ(mgr.size(), 1);
}

TEST(SessionManagerTest, ReapAttachDeadline) {
    SessionManager mgr;
    auto opts = makeOpts(100);
    opts.attach_deadline_us = 1000;
    auto cr = mgr.create(opts);

    auto now = nowUs();
    EXPECT_EQ(mgr.reapTimeouts(now), 0);

    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    now = nowUs();
    EXPECT_EQ(mgr.reapTimeouts(now), 1);
    EXPECT_TRUE(cr.session->isTerminal());
    EXPECT_EQ(cr.session->terminalInfo().reason, TerminalReason::TIMEOUT);
}

TEST(SessionManagerTest, ReapSkipsAttachedSession) {
    SessionManager mgr;
    auto opts = makeOpts(100);
    opts.attach_deadline_us = 1000;
    auto cr = mgr.create(opts);
    cr.session->acquireLiveLease();

    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    EXPECT_EQ(mgr.reapTimeouts(nowUs()), 0);
    EXPECT_FALSE(cr.session->isTerminal());
}

TEST(SessionManagerTest, ReapSkipsTerminalSession) {
    SessionManager mgr;
    auto opts = makeOpts(100);
    opts.attach_deadline_us = 1000;
    auto cr = mgr.create(opts);
    cr.session->finalizeTerminal(TerminalReason::FINISHED, grpc::Status::OK, nowUs());

    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    EXPECT_EQ(mgr.reapTimeouts(nowUs()), 0);
}

TEST(SessionManagerTest, Shutdown) {
    SessionManager mgr;
    auto cr1 = mgr.create(makeOpts(1));
    auto cr2 = mgr.create(makeOpts(2, 1, 99));

    mgr.shutdown(nowUs());
    EXPECT_TRUE(cr1.session->isTerminal());
    EXPECT_TRUE(cr2.session->isTerminal());
}

TEST(SessionManagerTest, StartStopGc) {
    SessionManager mgr;
    mgr.startGc();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    mgr.stopGc();
}

TEST(SessionManagerTest, MonotonicEpoch) {
    SessionManager mgr;
    auto cr1 = mgr.create(makeOpts(1, 1, 10));
    auto cr2 = mgr.create(makeOpts(2, 1, 20));
    auto cr3 = mgr.create(makeOpts(3, 1, 30));

    EXPECT_LT(cr1.session_epoch, cr2.session_epoch);
    EXPECT_LT(cr2.session_epoch, cr3.session_epoch);
}

TEST(SessionManagerTest, TombstoneEpochMismatch) {
    SessionManager mgr;
    auto opts = makeOpts(100);
    opts.payload_ttl_us = 500;
    opts.tombstone_ttl_us = 10000000;
    auto cr = mgr.create(opts);
    int64_t epoch = cr.session_epoch;
    cr.session->finalizeTerminal(TerminalReason::FINISHED, grpc::Status::OK, nowUs());

    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    mgr.gcOnce();

    auto lr_match = mgr.lookup(100, epoch, nowUs());
    EXPECT_EQ(lr_match.state, AttachState::GONE);

    auto lr_mismatch = mgr.lookup(100, epoch + 1, nowUs());
    EXPECT_EQ(lr_mismatch.state, AttachState::EPOCH_MISMATCH);
}

// ========================== SessionWriter ==========================

TEST(SessionWriterTest, WritePushesToSession) {
    auto opts = makeOpts(100);
    auto session = std::make_shared<RequestSession>(opts, 1);
    SessionWriter writer(session);

    auto output = makeOutput(42);
    EXPECT_TRUE(writer.Write(output, grpc::WriteOptions()));

    auto lease = session->acquireLiveLease();
    auto pop = session->popLive(lease.lease_id, 0);
    ASSERT_EQ(pop.status, PopResult::OUTPUT);
    ASSERT_EQ(pop.outputs.size(), 1);
    EXPECT_EQ(pop.outputs[0].request_id(), 42);
}

TEST(SessionWriterTest, WriteReturnsFalseAfterTerminal) {
    auto opts = makeOpts(100);
    auto session = std::make_shared<RequestSession>(opts, 1);
    SessionWriter writer(session);

    session->finalizeTerminal(TerminalReason::FINISHED, grpc::Status::OK, nowUs());

    auto output = makeOutput(1);
    EXPECT_FALSE(writer.Write(output, grpc::WriteOptions()));
}

// ========================== Concurrent cancel + pop ==========================

TEST(RequestSessionTest, ConcurrentCancelAndPop) {
    auto opts = makeOpts(100);
    auto session = std::make_shared<RequestSession>(opts, 1);

    for (int i = 0; i < 50; i++) {
        session->pushOutput(makeOutput(i));
    }

    auto lease = session->acquireLiveLease();
    std::vector<GenerateOutputsPB> collected;
    std::atomic<bool> cancel_done{false};

    std::thread canceller([&] {
        session->cancel(CancelReason::EXPLICIT_CANCEL, nowUs());
        cancel_done.store(true);
    });

    std::thread reader([&] {
        while (true) {
            auto pop = session->popLive(lease.lease_id, 10);
            if (pop.status == PopResult::OUTPUT) {
                for (auto& o : pop.outputs) {
                    collected.push_back(std::move(o));
                }
            } else {
                break;
            }
        }
    });

    canceller.join();
    reader.join();

    EXPECT_TRUE(session->isTerminal());
    EXPECT_EQ(collected.size(), 50);
}

// ========================== Full lifecycle ==========================

TEST(RequestSessionTest, FullLifecycle) {
    SessionManager mgr;
    auto opts = makeOpts(100);
    opts.payload_ttl_us = 2000;
    opts.tombstone_ttl_us = 3000;
    auto cr = mgr.create(opts);
    ASSERT_EQ(cr.status, BatchEnqueueStatus::ADMITTED);
    auto session = cr.session;

    session->pushOutput(makeOutput(1));
    session->pushOutput(makeOutput(2));

    auto lr_live = mgr.lookup(100, cr.session_epoch, nowUs());
    EXPECT_EQ(lr_live.state, AttachState::LIVE);

    auto lease = session->acquireLiveLease();
    auto pop = session->popLive(lease.lease_id, 0);
    EXPECT_EQ(pop.outputs.size(), 2);

    session->pushOutput(makeOutput(3));
    session->finalizeTerminal(TerminalReason::FINISHED, grpc::Status::OK, nowUs());

    auto lr_ttl = mgr.lookup(100, cr.session_epoch, nowUs());
    EXPECT_EQ(lr_ttl.state, AttachState::FINISHED_IN_TTL);
    ASSERT_NE(lr_ttl.snapshot, nullptr);
    EXPECT_EQ(lr_ttl.snapshot->outputs.size(), 3);

    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    mgr.gcOnce();
    auto lr_gone = mgr.lookup(100, cr.session_epoch, nowUs());
    EXPECT_EQ(lr_gone.state, AttachState::GONE);

    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    mgr.gcOnce();
    auto lr_nf = mgr.lookup(100, cr.session_epoch, nowUs());
    EXPECT_EQ(lr_nf.state, AttachState::NOT_FOUND);
}

}  // namespace rtp_llm::test
