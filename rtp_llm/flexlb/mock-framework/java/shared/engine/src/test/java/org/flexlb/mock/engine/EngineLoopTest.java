package org.flexlb.mock.engine;

import static org.junit.jupiter.api.Assertions.*;
import java.util.*;
import java.util.concurrent.*;
import org.junit.jupiter.api.Test;
import org.flexlb.mock.api.model.ComputeSpec;
import org.flexlb.mock.api.model.Types.Role;
import org.flexlb.mock.compute.*;

class EngineLoopTest {
    private static Scheduler.Work work(String id) {
        var item = new ForwardBatch.Request(id, 4, 0, 8, 0, 4, 0, List.of());
        return new Scheduler.Work(new Object(), new ForwardBatch(id, List.of(new ForwardBatch.Partition(0, List.of(item)))));
    }

    /** Two policies with different queue order exercise the same execution loop. */
    private static class Policy implements Scheduler {
        final Deque<Work> queue = new ArrayDeque<>();
        final BlockingQueue<String> finished = new LinkedBlockingQueue<>();
        final List<Work> failures = new ArrayList<>();
        final List<Runnable> notices = new ArrayList<>();
        final boolean newestFirst;
        final BlockingQueue<Work> scheduled = new LinkedBlockingQueue<>();
        Work cancelled;
        Policy(boolean newestFirst) { this.newestFirst = newestFirst; }
        public Work schedule() { var work = newestFirst ? queue.pollLast() : queue.pollFirst(); if (work != null) scheduled.add(work); return work; }
        public void computeCompleted(Work work, SimulatedCompute.Outcome result) { finished.add(work.batch().id()); }
        public void failed(Work work, RuntimeException error) { failures.add(work); }
        public boolean cancelled(Work work) { return cancelled == work; }
        public void maintain() {}
        public long nextDeadline() { return Long.MAX_VALUE; }
        public Status status(long latest) { return new Status(List.of(), List.of(), latest, 0, 0); }
        public void transferCompleted(Request request, TransferResult result) { throw new UnsupportedOperationException(); }
        public List<Runnable> takeNotifications() { var result = List.copyOf(notices); notices.clear(); return result; }
        public void close() { queue.clear(); }
        public List<Request> requests() { return List.of(); }
        public QueueSnapshot queues() { return new QueueSnapshot(queue.size(), 0, 0, 0); }
        public String scrapeMetrics() { return ""; }
        public Request input(Input input, Listener listener) { throw new UnsupportedOperationException(); }
        public void control(Request request, Control event) { throw new UnsupportedOperationException(); }
        public void finish(Request request, RequestFailure failure) { throw new UnsupportedOperationException(); }
        public RequestState.Use beginTransfer(Request request) { throw new UnsupportedOperationException(); }
    }
    private EngineLoop engine(Object lock, Policy policy, String expression) {
        return new EngineLoop(lock, policy, new org.flexlb.mock.cache.KvCache(1, 0, org.flexlb.mock.cache.KvCache.EvictionPolicy.LRU), new ComputeModel(new ComputeSpec(new ComputeSpec.Forward(expression, null), null, null)), Role.prefill, "test");
    }

    @Test void idleLoopsCannotExhaustVirtualThreadCarriers() throws Exception {
        var loops = new ArrayList<EngineLoop>();
        var policies = new ArrayList<Policy>();
        try {
            for (int i = 0; i < 300; i++) {
                var policy = new Policy(false); policies.add(policy);
                policy.queue.add(work("progress"));
                var loop = engine(policy, policy, "1"); loops.add(loop); loop.wake();
            }
            for (var policy : policies) assertEquals("progress", policy.finished.poll(5, TimeUnit.SECONDS));
        } finally { loops.forEach(EngineLoop::close); }
    }

    @Test void replacingQueuePolicyDoesNotChangeComputeOrLoop() throws Exception {
        for (boolean newest : List.of(false, true)) {
            var policy = new Policy(newest);
            policy.queue.add(work("first")); policy.queue.add(work("second"));
            try (var loop = engine(policy, policy, "1")) {
                loop.wake(); loop.wake();
                assertEquals(newest ? "second" : "first", policy.finished.poll(2, TimeUnit.SECONDS));
                assertEquals(newest ? "first" : "second", policy.finished.poll(2, TimeUnit.SECONDS));
                synchronized (policy) { assertTrue(policy.failures.isEmpty()); }
            }
        }
    }

    @Test void schedulerCanScheduleAnotherRoundAfterResultCommit() throws Exception {
        var policy = new Policy(false) {
            @Override public void computeCompleted(Work work, SimulatedCompute.Outcome result) {
                if (work.batch().id().equals("first")) queue.add(work("next"));
                super.computeCompleted(work, result);
            }
        };
        policy.queue.add(work("first"));
        try (var loop = engine(policy, policy, "1")) {
            loop.wake();
            assertEquals("first", policy.finished.poll(2, TimeUnit.SECONDS));
            assertEquals("next", policy.finished.poll(2, TimeUnit.SECONDS));
        }
    }

    @Test void cancelledExecutionCallbackCannotCompleteItsReplacement() throws Exception {
        var policy = new Policy(false);
        var first = work("same"); var second = work("same");
        policy.queue.add(first); policy.queue.add(second);
        try (var loop = engine(policy, policy, "100")) {
            loop.wake();
            assertSame(first, policy.scheduled.poll(2, TimeUnit.SECONDS));
            synchronized (policy) { policy.cancelled = first; loop.wake(); }
            assertSame(second, policy.scheduled.poll(2, TimeUnit.SECONDS));
            assertEquals("same", policy.finished.poll(2, TimeUnit.SECONDS));
            assertNull(policy.finished.poll(150, TimeUnit.MILLISECONDS));
            synchronized (policy) { assertEquals(List.of(first), policy.failures); }
        }
    }

    @Test void planningFailuresReleaseEveryWorkAndDoNotStrandQueue() throws Exception {
        var policy = new Policy(false);
        policy.queue.add(work("a")); policy.queue.add(work("b"));
        try (var loop = engine(policy, policy, "-1")) {
            loop.wake();
            long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(2);
            while (true) {
                synchronized (policy) { if (policy.failures.size() == 2) break; }
                assertTrue(System.nanoTime() < deadline); Thread.sleep(1);
            }
            synchronized (policy) { assertTrue(policy.queue.isEmpty()); }
        }
    }

    @Test void callbacksRunOutsideDomainLockAndCannotBlockCompute() throws Exception {
        var policy = new Policy(false);
        var entered = new CompletableFuture<Boolean>(); var release = new CountDownLatch(1);
        policy.notices.add(() -> {
            entered.complete(Thread.holdsLock(policy));
            try { assertTrue(release.await(3, TimeUnit.SECONDS)); }
            catch (InterruptedException error) { Thread.currentThread().interrupt(); throw new RuntimeException(error); }
        });
        policy.queue.add(work("forward"));
        try (var loop = engine(policy, policy, "1")) {
            loop.wake();
            assertFalse(entered.get(2, TimeUnit.SECONDS));
            assertEquals("forward", policy.finished.poll(2, TimeUnit.SECONDS));
        } finally { release.countDown(); }
    }
    @Test void idleLoopSleepsAndOnlyWakesForTheNearestDeadline() throws Exception {
        var calls = new java.util.concurrent.atomic.AtomicInteger();
        var expired = new CompletableFuture<Void>();
        var policy = new Policy(false) {
            long deadline = Long.MAX_VALUE;
            boolean armed;
            @Override public long nextDeadline() { return deadline; }
            @Override public void maintain() {
                calls.incrementAndGet();
                if (!armed) { armed = true; deadline = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(100); }
                else if (System.nanoTime() >= deadline) { deadline = Long.MAX_VALUE; expired.complete(null); }
            }
        };
        try (var loop = engine(policy, policy, "1")) {
            Thread.sleep(100);
            assertEquals(0, calls.get(), "Idle Engines must not poll");
            loop.wake();
            expired.get(2, TimeUnit.SECONDS);
            int committed = calls.get();
            assertEquals(2, committed);
            Thread.sleep(160);
            assertEquals(committed, calls.get(), "No repeated deadline wake after expiration");
        }
    }

    @Test void externalCapacityReleaseWakesAnOtherwiseIdleEngine() throws Exception {
        try (var cache = new org.flexlb.mock.cache.KvCache(1, 0, org.flexlb.mock.cache.KvCache.EvictionPolicy.LRU)) {
            var owner = cache.acquire(1, List.of()).lease();
            var tried = new CountDownLatch(1);
            var attempts = new java.util.concurrent.atomic.AtomicInteger();
            var policy = new Policy(false) {
                org.flexlb.mock.cache.KvCache.Lease lease;
                @Override public Work schedule() {
                    if (lease != null) return null;
                    attempts.incrementAndGet();
                    var admission = cache.acquire(1, List.of());
                    if (admission.lease() == null) { tried.countDown(); return null; }
                    lease = admission.lease();
                    return work("capacity");
                }
                @Override public void computeCompleted(Work work, SimulatedCompute.Outcome result) {
                    lease.close(); super.computeCompleted(work, result);
                }
            };
            try (var loop = new EngineLoop(policy, policy, cache, new ComputeModel(new ComputeSpec(new ComputeSpec.Forward("1", null), null, null)), Role.prefill, "capacity")) {
                loop.wake();
                assertTrue(tried.await(2, TimeUnit.SECONDS));
                Thread.sleep(100);
                assertEquals(1, attempts.get(), "Waiting for capacity must not poll");
                owner.close();
                assertEquals("capacity", policy.finished.poll(2, TimeUnit.SECONDS));
            } finally { owner.close(); }
        }
    }

}
