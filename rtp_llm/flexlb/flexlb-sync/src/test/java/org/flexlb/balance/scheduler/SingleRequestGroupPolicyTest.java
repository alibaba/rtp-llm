package org.flexlb.balance.scheduler;

import org.flexlb.balance.scheduler.ScheduledRequest;
import org.flexlb.dao.master.WorkerStatus;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.flexlb.balance.scheduler.GroupPolicyTestSupport.NOW_MS;
import static org.flexlb.balance.scheduler.GroupPolicyTestSupport.capacity;
import static org.flexlb.balance.scheduler.GroupPolicyTestSupport.single;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Final one-request grouping and exact resource revalidation contract. */
class SingleRequestGroupPolicyTest {

    private final SingleRequestGroupPolicy policy =
            new SingleRequestGroupPolicy();

    @Test
    void emptyQueueHasNoAction() {
        GroupPolicyTestSupport.Fixture fixture = single(false);

        BatcherCycleResult result = policy.processQueue(fixture.context());

        assertSame(BatcherCycleResult.NO_ACTION, result);
        assertTrue(fixture.delivery().attempts().isEmpty());
    }

    @Test
    void priorityQueueAdmitsExactlyOneStrictHeadPerPass() {
        GroupPolicyTestSupport.Fixture fixture = single(true);
        ScheduledRequest low = fixture.add(
                1L, 10, 10L, NOW_MS, Long.MAX_VALUE);
        fixture.add(2L, 100, 10L, NOW_MS + 1L, Long.MAX_VALUE);

        BatcherCycleResult first = policy.processQueue(fixture.context());
        assertSame(BatcherCycleResult.Status.ADMITTED, first.status());
        BatcherCycleResult second = policy.processQueue(fixture.context());
        assertSame(BatcherCycleResult.Status.ADMITTED, second.status());

        assertEquals(List.of(2L), ids(first.items()));
        assertEquals("single_request", first.metadata().decisionReason());
        assertEquals(1, first.metadata().remainingQueueDepth());
        assertEquals(List.of(1L), ids(second.items()));
        assertEquals("single_request", second.metadata().decisionReason());
        assertEquals(0, second.metadata().remainingQueueDepth());
        assertEquals(List.of(List.of(2L), List.of(1L)),
                fixture.delivery().committedGroups());
        assertTrue(fixture.activeItems().isEmpty());
        assertSame(low, second.items().getFirst());
    }

    @Test
    void computeCapacityDropDoesNotBlockStandaloneRequest() {
        GroupPolicyTestSupport.Fixture fixture = single(false);
        WorkerStatus.EngineObservation initiallyFits =
                capacity(200L, 0L, 0L, 0L);
        WorkerStatus.EngineObservation strictEqualityRejects =
                capacity(100L, 0L, 0L, 0L);
        fixture.statusSequence(initiallyFits, strictEqualityRejects);
        ScheduledRequest head = fixture.add(
                1L, 50, 100L, NOW_MS, Long.MAX_VALUE);

        BatcherCycleResult admitted = policy.processQueue(fixture.context());
        assertSame(BatcherCycleResult.Status.ADMITTED, admitted.status());

        assertEquals(List.of(head), admitted.items());
        assertTrue(fixture.lifecycle().offerFailureItems().isEmpty());
        assertTrue(fixture.activeItems().isEmpty());
        assertEquals(2, fixture.statusReads());
    }

    @Test
    void kvDropAtFinalReadReturnsExactEventDrivenWait() {
        GroupPolicyTestSupport.Fixture fixture = single(false);
        fixture.statusSequence(
                capacity(1_000L, 0L, 1_000L, 200L),
                capacity(1_000L, 0L, 1_000L, 50L));
        ScheduledRequest head = fixture.add(
                1L, 50, 100L, NOW_MS, NOW_MS + 10_000L);

        BatcherCycleResult waiting = policy.processQueue(fixture.context());
        assertSame(BatcherCycleResult.Status.AWAITING_SCHEDULING_CHANGE, waiting.status());

        assertSame(head, waiting.head());
        assertEquals(1L, waiting.queueVersion());
        assertEquals(0L, waiting.schedulingInputVersion());
        assertEquals(NOW_MS + 10_000L, waiting.wakeAtMs());
        assertSame(BatcherCycleResult.SchedulingWaitReason.PREFILL_KV_CAPACITY,
                waiting.reason());
        assertEquals(2, fixture.statusReads());
        assertEquals(List.of(head), fixture.activeItems());
        assertTrue(fixture.delivery().attempts().isEmpty());
    }

    @Test
    void initiallyInsufficientKvWaitsWithoutPredictionOrDelivery() {
        GroupPolicyTestSupport.Fixture fixture = single(false);
        fixture.status(capacity(1_000L, 0L, 1_000L, 99L));
        fixture.bumpSchedulingInputVersion();
        fixture.bumpSchedulingInputVersion();
        ScheduledRequest head = fixture.add(
                1L, 50, 100L, NOW_MS - 1_000L,
                NOW_MS + 10_000L);

        BatcherCycleResult waiting = policy.processQueue(fixture.context());
        assertSame(BatcherCycleResult.Status.AWAITING_SCHEDULING_CHANGE, waiting.status());

        assertSame(head, waiting.head());
        assertEquals(1L, waiting.queueVersion());
        assertEquals(2L, waiting.schedulingInputVersion());
        assertEquals(NOW_MS + 10_000L, waiting.wakeAtMs());
        assertSame(BatcherCycleResult.SchedulingWaitReason.PREFILL_KV_CAPACITY,
                waiting.reason());
        assertEquals(1, fixture.statusReads());
        assertTrue(fixture.delivery().projectedGroups().isEmpty());
        assertTrue(fixture.delivery().attempts().isEmpty());
        assertEquals(List.of(head), fixture.activeItems());
    }

    @Test
    void deliveryCapacityBlockKeepsExactRequestActive() {
        GroupPolicyTestSupport.Fixture fixture = single(false);
        fixture.delivery().block();
        ScheduledRequest head = fixture.add(
                1L, 50, 100L, NOW_MS, Long.MAX_VALUE);

        BatcherCycleResult blocked = policy.processQueue(fixture.context());
        assertSame(BatcherCycleResult.Status.CAPACITY_BLOCKED, blocked.status());

        assertSame(head, blocked.item());
        assertEquals(List.of(List.of(1L)), fixture.delivery().attempts());
        assertEquals(List.of(head), fixture.activeItems());
        assertTrue(fixture.delivery().committedGroups().isEmpty());
    }

    @Test
    void longStandaloneRequestIgnoresBatchTokenCapacity() {
        GroupPolicyTestSupport.Fixture fixture = single(false);
        fixture.status(capacity(409_600L, 1_048_576L, 0L, 0L));
        ScheduledRequest head = fixture.add(
                1L, 50, 910_537L, NOW_MS, Long.MAX_VALUE);

        BatcherCycleResult admitted = policy.processQueue(fixture.context());
        assertSame(BatcherCycleResult.Status.ADMITTED, admitted.status());

        assertEquals(List.of(head), admitted.items());
        assertTrue(fixture.lifecycle().offerFailureItems().isEmpty());
        assertTrue(fixture.activeItems().isEmpty());
    }

    @Test
    void expiredHeadIsTerminalizedBeforeCapacityRead() {
        GroupPolicyTestSupport.Fixture fixture = single(false);
        ScheduledRequest expired = fixture.add(
                1L, 50, 100L, NOW_MS - 10L, NOW_MS);

        BatcherCycleResult result = policy.processQueue(fixture.context());

        assertSame(BatcherCycleResult.QUEUE_CHANGED, result);
        assertEquals(List.of(expired), fixture.lifecycle().expired());
        assertEquals(0, fixture.statusReads());
        assertTrue(fixture.delivery().attempts().isEmpty());
        assertTrue(fixture.activeItems().isEmpty());
    }

    private static List<Long> ids(List<? extends ScheduledRequest> items) {
        return items.stream().map(ScheduledRequest::requestId).toList();
    }
}
