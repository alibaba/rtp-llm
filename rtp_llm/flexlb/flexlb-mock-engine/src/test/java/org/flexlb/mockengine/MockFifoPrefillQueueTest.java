package org.flexlb.mockengine;

import org.junit.jupiter.api.Test;
import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

/** Behavioral vectors from FIFOSchedulerTest group/normal-lane regressions. */
class MockFifoPrefillQueueTest {
    private final MockFifoPrefillQueue<Integer> queue = new MockFifoPrefillQueue<>();

    private List<Integer> poll(int cap, Set<Integer> blocked, Set<Integer> cancelled) {
        queue.executionFinished();
        return queue.poll(new MockFifoPrefillQueue.Driver<>() {
            public boolean hasError(Integer id) { return cancelled.contains(id); }
            public MockFifoPrefillQueue.State advance(Integer id, MockFifoPrefillQueue.State state) {
                return cancelled.contains(id) ? MockFifoPrefillQueue.State.FINISHED : state;
            }
            public MockFifoPrefillQueue.Round<Integer> round(boolean group, List<Integer> ready) {
                return new MockFifoPrefillQueue.Round<>() {
                    int admitted = ready.size();
                    public MockFifoPrefillQueue.Admission<Integer> admit(Integer id) {
                        if (blocked.contains(id) || admitted >= cap) return null;
                        admitted++;
                        return new MockFifoPrefillQueue.Admission<>(id, MockFifoPrefillQueue.State.RUNNING);
                    }
                };
            }
        });
    }

    @Test void normalAndExplicitGroupsOwnSeparateAlternatingBoundaries() {
        queue.enqueue(List.of(10, 11), true, 4, 0);
        queue.enqueue(List.of(20, 21), true, 4, 0);
        queue.enqueue(List.of(1, 2), false, 4, 0);
        assertEquals(List.of(1), poll(1, Set.of(), Set.of()));
        assertEquals(List.of(10), poll(1, Set.of(), Set.of()));
        assertEquals(List.of(2), poll(4, Set.of(), Set.of()));
        assertEquals(List.of(11), poll(4, Set.of(), Set.of()), "residual group keeps its head position");
        assertEquals(List.of(20, 21), poll(4, Set.of(), Set.of()), "groups never merge");
        assertEquals(0, queue.size());
    }

    @Test void oversizedGroupsBecomeOrdinaryStreamsAtTheArrivalTail() {
        queue.enqueue(List.of(1), false, 2, 0);
        queue.enqueue(List.of(2, 3, 4), true, 2, 0);
        queue.enqueue(List.of(5), false, 2, 0);
        assertEquals(0, queue.groupCount());
        assertEquals(List.of(1, 2, 3, 4, 5), poll(8, Set.of(), Set.of()));
        queue.enqueue(List.of(6, 7), true, 4, 1);
        assertEquals(0, queue.groupCount(), "initialized-stream cap also causes group fallback");
    }

    @Test void capacityBlockedCandidatesDoNotPinTheRoundOrBypassTheGroupHead() {
        queue.enqueue(List.of(10, 11), true, 4, 0);
        queue.enqueue(List.of(20), true, 4, 0);
        assertEquals(List.of(11), poll(4, Set.of(10), Set.of()));
        queue.enqueue(List.of(1), false, 4, 0);
        assertEquals(List.of(1), poll(4, Set.of(10), Set.of()));
        assertTrue(poll(4, Set.of(10), Set.of()).isEmpty(), "later group cannot pass blocked head");
        assertEquals(List.of(10), poll(4, Set.of(), Set.of()));
        assertEquals(List.of(20), poll(4, Set.of(), Set.of()));
    }

    @Test void cancellationBehindSaturatedBudgetStillLeavesTheQueue() {
        queue.enqueue(List.of(1, 2, 3), false, 4, 0);
        assertEquals(List.of(1), poll(1, Set.of(), Set.of(2, 3)));
        assertEquals(0, queue.size());
    }
    private static final class LoadingDriver implements MockFifoPrefillQueue.Driver<Integer> {
        final java.util.Set<Integer> loading = new java.util.HashSet<>();
        final java.util.Set<Integer> completed = new java.util.HashSet<>();
        final java.util.Set<Integer> cancelled = new java.util.HashSet<>();
        final java.util.Set<Integer> finalized = new java.util.HashSet<>();
        final java.util.Set<Integer> initialized = new java.util.HashSet<>();
        public boolean hasError(Integer id) { return cancelled.contains(id); }
        public MockFifoPrefillQueue.State advance(Integer id, MockFifoPrefillQueue.State state) {
            if (cancelled.contains(id)) { finalized.add(id); return MockFifoPrefillQueue.State.FINISHED; }
            if (state == MockFifoPrefillQueue.State.LOADING_CACHE && completed.contains(id)) return MockFifoPrefillQueue.State.WAITING;
            if (state == MockFifoPrefillQueue.State.WAITING && initialized.contains(id)) return MockFifoPrefillQueue.State.RUNNING;
            return state;
        }
        public MockFifoPrefillQueue.Round<Integer> round(boolean group, List<Integer> ready) {
            return id -> {
                boolean fresh = initialized.add(id);
                return new MockFifoPrefillQueue.Admission<>(id, fresh && loading.contains(id)
                        ? MockFifoPrefillQueue.State.LOADING_CACHE : MockFifoPrefillQueue.State.RUNNING);
            };
        }
    }

    @Test void ordinaryCacheCompletionReturnsToWaitingTailAndCannotJoinGroup() {
        var driver = new LoadingDriver(); driver.loading.add(1);
        queue.enqueue(List.of(1), false, 4, 0);
        assertTrue(queue.poll(driver).isEmpty());
        assertEquals(1, queue.loadingSize()); assertEquals(0, queue.size());
        queue.enqueue(List.of(2), false, 4, 0);
        driver.completed.add(1);
        assertEquals(List.of(2, 1), queue.poll(driver), "normal completion re-enters stable admission at the tail");
        queue.executionFinished();
        queue.enqueue(List.of(10), true, 4, 0);
        assertEquals(List.of(10), queue.poll(driver));
    }

    @Test void loadingGroupOwnsAnIsolatedBoundaryOnlyWhenAllMembersAreReady() {
        var driver = new LoadingDriver(); driver.loading.add(10);
        queue.enqueue(List.of(10, 11), true, 4, 0);
        assertTrue(queue.poll(driver).isEmpty());
        assertEquals(2, queue.loadingSize(), "ready member remains owned by loading group");
        queue.enqueue(List.of(1), false, 4, 0);
        assertEquals(List.of(1), queue.poll(driver), "pending I/O does not pin execution");
        driver.completed.add(10);
        assertTrue(queue.poll(driver).isEmpty(), "ready group cannot mix into active normal execution");
        queue.executionFinished();
        assertEquals(List.of(10, 11), queue.poll(driver));
        assertEquals(0, queue.loadingSize());
    }

    @Test void cancelledLaterGroupWaitsForItsHeadToBeVisited() {
        var driver = new LoadingDriver(); driver.loading.add(10);
        queue.enqueue(List.of(10), true, 4, 0);
        queue.enqueue(List.of(20), true, 4, 0);
        driver.cancelled.add(20);
        assertTrue(queue.poll(driver).isEmpty());
        assertFalse(driver.finalized.contains(20), "cancel intent alone does not publish FINISHED");
        driver.cancelled.add(10);
        queue.poll(driver);
        assertTrue(driver.finalized.containsAll(List.of(10, 20)));
        assertEquals(0, queue.loadingSize()); assertEquals(0, queue.size());
    }

}
