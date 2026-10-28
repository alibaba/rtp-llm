package org.flexlb.mock.engine;

import static org.junit.jupiter.api.Assertions.*;
import java.util.*;
import java.util.concurrent.*;
import org.junit.jupiter.api.Test;
import org.flexlb.mock.api.model.ComputeSpec;
import org.flexlb.mock.api.model.Types.Role;
import org.flexlb.mock.compute.*;

class EngineLoopTest {
    static ForwardBatch batch(String id) {
        return new ForwardBatch(id, List.of(new ForwardBatch.Partition(0, List.of(
            new ForwardBatch.Request(id, 4, 0, 8, 0, 4, 0, List.of())))));
    }
    static class Domain implements Scheduler<Object> {
        final Queue<Work<Object>> waiting = new ArrayDeque<>();
        final BlockingQueue<SimulatedCompute.Outcome> results = new LinkedBlockingQueue<>();
        final List<Object> owners = new ArrayList<>();
        final List<RuntimeException> failures = new ArrayList<>();
        public Work<Object> schedule() { return waiting.poll(); }
        public void completed(Object owner, SimulatedCompute.Outcome result) { owners.add(owner); results.add(result); }
        public void failed(Object owner, RuntimeException error) { failures.add(error); }
    }
    static EngineLoop<Object> loop(Object lock, Domain domain, String expression) {
        return new EngineLoop<>(lock, domain, new ComputeModel(new ComputeSpec(expression, null)), Role.prefill, "p");
    }

    @Test void queuePolicyAndRequeueRemainOwnedByScheduler() throws Exception {
        Object lock = new Object(); Object owner = new Object();
        var domain = new Domain() {
            @Override public void completed(Object current, SimulatedCompute.Outcome outcome) {
                if (owners.isEmpty()) waiting.add(new Work<>(current, batch("next")));
                super.completed(current, outcome);
            }
        };
        domain.waiting.add(new Scheduler.Work<>(owner, batch("first")));
        try (var engine = loop(lock, domain, "1")) {
            engine.wake(); engine.wake();
            assertEquals(SimulatedCompute.State.COMPLETED, domain.results.poll(2, TimeUnit.SECONDS).state());
            assertEquals(SimulatedCompute.State.COMPLETED, domain.results.poll(2, TimeUnit.SECONDS).state());
            synchronized (lock) { assertEquals(List.of(owner, owner), domain.owners); assertTrue(domain.failures.isEmpty()); }
        }
    }

    @Test void cancellationUsesObjectIdentityAndDoesNotCancelNextOwnerWithSameId() throws Exception {
        Object lock = new Object(); Object first = new String("same"), second = new String("same");
        var domain = new Domain();
        domain.waiting.add(new Scheduler.Work<>(first, batch("same")));
        domain.waiting.add(new Scheduler.Work<>(second, batch("same")));
        try (var engine = loop(lock, domain, "3600000")) {
            engine.wake(); engine.cancel(first);
            assertEquals(SimulatedCompute.State.CANCELLED, domain.results.poll(2, TimeUnit.SECONDS).state());
            synchronized (lock) { engine.cancel(first); assertEquals(1, domain.owners.size()); engine.cancel(second); }
            assertEquals(SimulatedCompute.State.CANCELLED, domain.results.poll(2, TimeUnit.SECONDS).state());
            synchronized (lock) { assertSame(second, domain.owners.get(1)); }
        }
    }

    @Test void modelFailureReleasesThroughOwnerAndDoesNotStrandQueuedWork() {
        var domain = new Domain();
        domain.waiting.add(new Scheduler.Work<>(new Object(), batch("a")));
        domain.waiting.add(new Scheduler.Work<>(new Object(), batch("b")));
        try (var engine = loop(domain, domain, "-1")) {
            engine.wake();
            assertEquals(2, domain.failures.size()); assertTrue(domain.waiting.isEmpty());
            assertTrue(domain.results.isEmpty());
        }
    }

    @Test void oneCancelledMemberDoesNotCancelItsBatchOrReleaseResourcesEarly() throws Exception {
        Object lock = new Object(); var ended = new ArrayList<RequestFailure>();
        int[] closed = new int[2], published = new int[2];
        var members = List.of(new RequestState(Long.MAX_VALUE, ended::add), new RequestState(Long.MAX_VALUE, ended::add));
        var uses = new ArrayList<RequestState.Use>();
        for (int i = 0; i < members.size(); i++) {
            final int index = i;
            members.get(i).attach(() -> closed[index]++, () -> published[index]++);
        }
        var domain = new Domain() {
            @Override public Work<Object> schedule() {
                var work = super.schedule();
                if (work != null) members.forEach(member -> uses.add(member.forward()));
                return work;
            }
            @Override public void completed(Object owner, SimulatedCompute.Outcome outcome) {
                for (int i = 0; i < members.size(); i++) {
                    var member = members.get(i);
                    if (!member.terminating()) { member.commit(1, outcome.elapsedNanos()); member.finish(null); }
                    uses.get(i).close();
                }
                super.completed(owner, outcome);
            }
        };
        var items = List.of(batch("a").dps().getFirst().requests().getFirst(), batch("b").dps().getFirst().requests().getFirst());
        domain.waiting.add(new Scheduler.Work<>(new Object(), new ForwardBatch("shared", List.of(new ForwardBatch.Partition(0, items)))));
        try (var engine = loop(lock, domain, "10")) {
            synchronized (lock) {
                engine.wake(); members.getFirst().finish(new RequestFailure(RequestFailure.Reason.CANCELLED, "cancel one"));
                assertArrayEquals(new int[]{0, 0}, closed); assertTrue(ended.isEmpty());
            }
            assertEquals(SimulatedCompute.State.COMPLETED, domain.results.poll(2, TimeUnit.SECONDS).state());
            synchronized (lock) {
                assertArrayEquals(new int[]{1, 1}, closed); assertArrayEquals(new int[]{0, 1}, published);
                assertEquals(2, ended.size()); assertNull(ended.getLast()); assertTrue(domain.failures.isEmpty());
            }
        }
    }

    @Test void actualForwardIncludesWaitingToReenterEngineDomain() throws Exception {
        var domain = new Domain(); Object lock = new Object();
        domain.waiting.add(new Scheduler.Work<>(new Object(), batch("a")));
        try (var engine = loop(lock, domain, "1")) {
            synchronized (lock) { engine.wake(); Thread.sleep(40); }
            var result = domain.results.poll(2, TimeUnit.SECONDS);
            assertNotNull(result); assertTrue(result.elapsedNanos() >= 40_000_000);
            assertEquals(1_000_000, result.sample().planned().toNanos());
        }
    }
}
