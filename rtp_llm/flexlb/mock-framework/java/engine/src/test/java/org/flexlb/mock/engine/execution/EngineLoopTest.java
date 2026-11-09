package org.flexlb.mock.engine.execution;

import java.util.*;
import java.util.concurrent.*;
import org.flexlb.mock.engine.forward.ComputeSpec;
import org.flexlb.mock.protocol.Role;
import org.flexlb.mock.engine.forward.*;
import org.flexlb.mock.engine.scheduler.EngineScheduler.QueueSnapshot;
import org.flexlb.mock.engine.scheduler.EngineScheduler.RequestFailure;
import org.flexlb.mock.engine.scheduler.RequestState;
import org.flexlb.mock.engine.scheduler.EngineScheduler;
import org.flexlb.mock.engine.scheduler.TaskHistory;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class EngineLoopTest {
    private static EngineScheduler.Work work(String id) {
        var item = new ForwardBatch.Request(id, 4, 0, 8, 0, 4, 0, List.of());
        return new EngineScheduler.Work(new Object(), new ForwardBatch(id, List.of(item)));
    }

    /** Two policies with different queue order exercise the same execution loop. */
    private static class Policy implements EngineScheduler {
        final Deque<Work> queue = new ArrayDeque<>();
        final BlockingQueue<String> finished = new LinkedBlockingQueue<>();
        final List<Work> failures = new ArrayList<>();
        final List<Runnable> notices = new ArrayList<>();
        final boolean newestFirst;
        final BlockingQueue<Work> scheduled = new LinkedBlockingQueue<>();
        Work cancelled;
        Policy(boolean newestFirst) { this.newestFirst = newestFirst; }
        public Work schedule() { var work = newestFirst ? queue.pollLast() : queue.pollFirst(); if (work != null) scheduled.add(work); return work; }
        public void processOutputs(Work work, SimulatedCompute.Outcome result) { finished.add(work.batch().id()); }
        public void failForward(Work work, RuntimeException error) { failures.add(work); }
        public boolean isForwardCancelled(Work work) { return cancelled == work; }
        public void prepareKvAndCheckTimeouts() {}
        public long nextDeadline() { return Long.MAX_VALUE; }
        public List<Task> tasks() { return List.of(); }
        public void transferCompleted(Request request, TransferResult result) { throw new UnsupportedOperationException(); }
        public List<Runnable> takeNotifications() { var result = List.copyOf(notices); notices.clear(); return result; }
        public void close() { queue.clear(); }
        public List<Request> requests() { return List.of(); }
        public QueueSnapshot queues() { return new QueueSnapshot(queue.size(), 0, 0, 0); }
        public Request enqueue(Input input, Listener listener) { throw new UnsupportedOperationException(); }
        public List<InputResult> enqueue(List<InputItem> batch) { throw new UnsupportedOperationException(); }
        public void control(Request request, Control event) { throw new UnsupportedOperationException(); }
        public void abort(Request request, RequestFailure failure) { throw new UnsupportedOperationException(); }
        public RequestState.Use beginTransfer(Request request) { throw new UnsupportedOperationException(); }
    }
    private EngineLoop engine(Object lock, Policy policy, String expression) {
        return new EngineLoop(lock, (domain, cache, finished) -> policy, new TaskHistory(), new org.flexlb.mock.engine.cache.KvCacheManager(1, 1, 0), new ComputeModel(new ComputeSpec(new ComputeSpec.Forward(expression, null), null, null)), Role.prefill, "test");
    }

    @Test void healthChecksExposeLoopFailures() throws Exception {
        var fail = new java.util.concurrent.atomic.AtomicBoolean();
        var policy = new Policy(false) {
            @Override public Work schedule() {
                if (fail.get()) throw new IllegalStateException("broken scheduling loop");
                return super.schedule();
            }
        };
        try (var loop = engine(policy, policy, "1")) {
            loop.checkHealthy(); loop.checkHealthy();
            assertThrows(NullPointerException.class, () -> loop.abortRequest(null, null));
            fail.set(true); loop.wakeUp();
            long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(2);
            while (true) {
                try { loop.checkHealthy(); }
                catch (IllegalStateException error) { assertTrue(error.getMessage().contains("broken scheduling loop")); break; }
                assertTrue(System.nanoTime() < deadline);
                Thread.sleep(5);
            }
        }
    }

    @Test void idleLoopsCannotExhaustVirtualThreadCarriers() throws Exception {
        var loops = new ArrayList<EngineLoop>();
        var policies = new ArrayList<Policy>();
        try {
            for (int i = 0; i < 300; i++) {
                var policy = new Policy(false); policies.add(policy);
                policy.queue.add(work("progress"));
                var loop = engine(policy, policy, "1"); loops.add(loop); loop.wakeUp();
            }
            for (var policy : policies) assertEquals("progress", policy.finished.poll(5, TimeUnit.SECONDS));
        } finally { loops.forEach(EngineLoop::close); }
    }

    @Test void replacingQueuePolicyDoesNotChangeComputeOrLoop() throws Exception {
        for (boolean newest : List.of(false, true)) {
            var policy = new Policy(newest);
            policy.queue.add(work("first")); policy.queue.add(work("second"));
            try (var loop = engine(policy, policy, "1")) {
                loop.wakeUp(); loop.wakeUp();
                assertEquals(newest ? "second" : "first", policy.finished.poll(2, TimeUnit.SECONDS));
                assertEquals(newest ? "first" : "second", policy.finished.poll(2, TimeUnit.SECONDS));
                synchronized (policy) { assertTrue(policy.failures.isEmpty()); }
            }
        }
    }

    @Test void schedulerCanScheduleAnotherRoundAfterResultCommit() throws Exception {
        var policy = new Policy(false) {
            @Override public void processOutputs(Work work, SimulatedCompute.Outcome result) {
                if (work.batch().id().equals("first")) queue.add(work("next"));
                super.processOutputs(work, result);
            }
        };
        policy.queue.add(work("first"));
        try (var loop = engine(policy, policy, "1")) {
            loop.wakeUp();
            assertEquals("first", policy.finished.poll(2, TimeUnit.SECONDS));
            assertEquals("next", policy.finished.poll(2, TimeUnit.SECONDS));
        }
    }

    @Test void cancelledExecutionCallbackCannotCompleteItsReplacement() throws Exception {
        var policy = new Policy(false);
        var first = work("same"); var second = work("same");
        policy.queue.add(first); policy.queue.add(second);
        try (var loop = engine(policy, policy, "100")) {
            loop.wakeUp();
            assertSame(first, policy.scheduled.poll(2, TimeUnit.SECONDS));
            synchronized (policy) { policy.cancelled = first; loop.wakeUp(); }
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
            loop.wakeUp();
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
            loop.wakeUp();
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
            @Override public void prepareKvAndCheckTimeouts() {
                calls.incrementAndGet();
                if (!armed) { armed = true; deadline = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(100); }
                else if (System.nanoTime() >= deadline) { deadline = Long.MAX_VALUE; expired.complete(null); }
            }
        };
        try (var loop = engine(policy, policy, "1")) {
            Thread.sleep(100);
            assertEquals(0, calls.get(), "Idle Engines must not poll");
            loop.wakeUp();
            expired.get(2, TimeUnit.SECONDS);
            int committed = calls.get();
            assertEquals(2, committed);
            Thread.sleep(160);
            assertEquals(committed, calls.get(), "No repeated deadline wake after expiration");
        }
    }

    @Test void externalCapacityReleaseWakesAnOtherwiseIdleEngine() throws Exception {
        try (var cache = new org.flexlb.mock.engine.cache.KvCacheManager(1, 1, 0)) {
            var owner = cache.acquire(1, List.of()).lease();
            var tried = new CountDownLatch(1);
            var attempts = new java.util.concurrent.atomic.AtomicInteger();
            var policy = new Policy(false) {
                org.flexlb.mock.engine.cache.KvCacheManager.Lease lease;
                @Override public Work schedule() {
                    if (lease != null) return null;
                    attempts.incrementAndGet();
                    var admission = cache.acquire(1, List.of());
                    if (admission.lease() == null) { tried.countDown(); return null; }
                    lease = admission.lease();
                    return work("capacity");
                }
                @Override public void processOutputs(Work work, SimulatedCompute.Outcome result) {
                    lease.close(); super.processOutputs(work, result);
                }
            };
            try (var loop = new EngineLoop(policy, (domain, storage, finished) -> policy, new TaskHistory(), cache, new ComputeModel(new ComputeSpec(new ComputeSpec.Forward("1", null), null, null)), Role.prefill, "capacity")) {
                loop.wakeUp();
                assertTrue(tried.await(2, TimeUnit.SECONDS));
                Thread.sleep(100);
                assertEquals(1, attempts.get(), "Waiting for capacity must not poll");
                owner.close();
                assertEquals("capacity", policy.finished.poll(2, TimeUnit.SECONDS));
            } finally { owner.close(); }
        }
    }

}
