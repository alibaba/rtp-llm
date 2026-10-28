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
        Work cancelled;
        Policy(boolean newestFirst) { this.newestFirst = newestFirst; }
        public Work schedule() { return newestFirst ? queue.pollLast() : queue.pollFirst(); }
        public void completed(Work work, SimulatedCompute.Outcome result) { finished.add(work.batch().id()); }
        public void failed(Work work, RuntimeException error) { failures.add(work); }
        public boolean cancelled(Work work) { return cancelled == work; }
        public void maintain() {}
        public List<Runnable> takeNotifications() { var result = List.copyOf(notices); notices.clear(); return result; }
        public void close() { queue.clear(); }
        public List<Request> requests() { return List.of(); }
        public QueueSnapshot queues() { return new QueueSnapshot(queue.size(), 0, 0, 0); }
        public String scrapeMetrics() { return ""; }
        public Request submit(Input input, Listener listener) { throw new UnsupportedOperationException(); }
        public void dependencyCompleted(Request request, Dependency dependency) { throw new UnsupportedOperationException(); }
        public void finish(Request request, RequestFailure failure) { throw new UnsupportedOperationException(); }
        public RequestState.Use readReady(Request request) { throw new UnsupportedOperationException(); }
    }
    private EngineLoop engine(Object lock, Policy policy, String expression) {
        return new EngineLoop(lock, policy, new ComputeModel(new ComputeSpec(expression, null)), Role.prefill, "test");
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
            @Override public void completed(Work work, SimulatedCompute.Outcome result) {
                if (work.batch().id().equals("first")) queue.add(work("next"));
                super.completed(work, result);
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
            synchronized (policy) {
                loop.wake(); policy.cancelled = first; loop.wake();
                assertEquals(List.of(first), policy.failures);
                loop.wake();
            }
            assertEquals("same", policy.finished.poll(2, TimeUnit.SECONDS));
            assertNull(policy.finished.poll(150, TimeUnit.MILLISECONDS));
            synchronized (policy) { assertEquals(List.of(first), policy.failures); }
        }
    }

    @Test void planningFailuresReleaseEveryWorkAndDoNotStrandQueue() {
        var policy = new Policy(false);
        policy.queue.add(work("a")); policy.queue.add(work("b"));
        try (var loop = engine(policy, policy, "-1")) {
            loop.wake();
            synchronized (policy) { assertEquals(2, policy.failures.size()); assertTrue(policy.queue.isEmpty()); }
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
}
