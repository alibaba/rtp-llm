package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.strategy.CacheTokenMatch;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.RoutingConfig.QueuedRequestPolicy;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Set;
import java.util.function.Predicate;

import static org.flexlb.config.RoutingConfig.QueuedRequestPolicy.DISABLED;
import static org.flexlb.config.RoutingConfig.QueuedRequestPolicy.HIGHEST_HIT_RATE_FIRST;
import static org.flexlb.config.RoutingConfig.QueuedRequestPolicy.LOWEST_HIT_RATE_FIRST;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

class QueuedMigrationCandidatesTest {
    @Test
    void choosesByLatestContiguousPrefixRatioAndBreaksTiesTowardTheDispatchTail() {
        var cache = new CacheStatus();
        cache.setBlockSize(4L);
        cache.setCachedKeys(Set.of(1L, 2L, 4L));
        var firstHalf = queued(10, 8, 0, List.of(1L, 3L, 4L));
        var empty = queued(15, 0, 0, List.of());
        var shorterThanOneBlock = queued(17, 2, 2, List.of(1L));
        var cold = queued(20, 16, 16, List.of(3L, 1L));
        var secondHalf = queued(30, 16, 0, List.of(1L, 2L, 3L, 4L));
        var placementHalf = queued(31, 16, 8, null);
        var hot = queued(40, 12, 0, List.of(1L, 2L, 4L));
        var source = source(cache, List.of(secondHalf, cold, placementHalf, firstHalf, hot, shorterThanOneBlock, empty));

        assertEquals(hot, QueuedMigrationCandidates.candidate(source, HIGHEST_HIT_RATE_FIRST, true, item -> true, new QueuedMigrationCandidates.Cursor()));
        assertEquals(empty, QueuedMigrationCandidates.candidate(source, LOWEST_HIT_RATE_FIRST, false, item -> true, new QueuedMigrationCandidates.Cursor()));
        var tied = Set.of(firstHalf, secondHalf, placementHalf);
        assertEquals(firstHalf, QueuedMigrationCandidates.candidate(source, HIGHEST_HIT_RATE_FIRST, true, tied::contains, new QueuedMigrationCandidates.Cursor()));
        assertEquals(firstHalf, QueuedMigrationCandidates.candidate(source, LOWEST_HIT_RATE_FIRST, false, tied::contains, new QueuedMigrationCandidates.Cursor()));
    }

    @Test
    void costOrderedColdPolicyStopsAtTheFirstEligibleTailRequestAcrossPriorities() {
        var highPriorityCold = queued(1, 8, 0, List.of());
        var lowPriorityWarm = queued(2, 8, 6, List.of());
        var ineligibleTail = queued(3, 8, 0, List.of());
        when(highPriorityCold.priority()).thenReturn(90);
        when(lowPriorityWarm.priority()).thenReturn(30);
        when(ineligibleTail.priority()).thenReturn(30);
        var workerOrder = PrefillCostOrder.workerOrder(ScheduledRequest::priority, ScheduledRequest::seqLen,
                ScheduledRequest::hitCache, Comparator.comparingLong(ScheduledRequest::enqueueSeq));
        var dispatch = List.of(lowPriorityWarm, ineligibleTail, highPriorityCold).stream().sorted(workerOrder).toList();
        var source = source(null, dispatch);
        var visited = new ArrayList<ScheduledRequest>();
        assertEquals(lowPriorityWarm, QueuedMigrationCandidates.candidate(source, LOWEST_HIT_RATE_FIRST, true,
                item -> {
                    visited.add(item);
                    return item != ineligibleTail;
                }, new QueuedMigrationCandidates.Cursor()));
        assertEquals(List.of(ineligibleTail, lowPriorityWarm), visited);
        verify(source, never()).getStatus();
        assertEquals(highPriorityCold, QueuedMigrationCandidates.candidate(source, LOWEST_HIT_RATE_FIRST, false,
                item -> item != ineligibleTail, new QueuedMigrationCandidates.Cursor()));
    }

    @Test
    void oversizedPrefixRetainsTheSameLastBlockRecomputationAsRouting() {
        long block = Long.MAX_VALUE / 2 + 1;
        assertEquals(new CacheTokenMatch(Long.MAX_VALUE - block, Long.MAX_VALUE),
                CacheTokenMatch.fromBlocks(Long.MAX_VALUE, block, 2));
        assertEquals(new CacheTokenMatch(0L, 2L), CacheTokenMatch.fromBlocks(2, 4, 1));
    }

    @Test
    void twoRejectedTailsYieldToAThirdRequestOnTheThirdInspectionDespiteNewArrivals() {
        var third = queued(1, 8, 0, List.of());
        var second = queued(2, 8, 0, List.of());
        var tail = queued(3, 8, 0, List.of());
        var arrival = queued(4, 8, 0, List.of());
        var urgent = queued(5, 8, 0, List.of());
        var source = source(null, List.of(third, second, tail));
        var cursor = new QueuedMigrationCandidates.Cursor();
        var visited = new ArrayList<ScheduledRequest>();
        Predicate<ScheduledRequest> eligible = visited::add;
        assertEquals(tail, QueuedMigrationCandidates.candidate(source, LOWEST_HIT_RATE_FIRST, true, eligible, cursor));
        // The first two targets cannot accept their candidate within budget. Neither rejection resets the cursor.
        when(source.captureQueueSnapshot()).thenReturn(
                new WorkerBatcher.QueueSnapshot("source", 1, List.of(urgent, third, second, tail, arrival)));
        assertEquals(second, QueuedMigrationCandidates.candidate(source, LOWEST_HIT_RATE_FIRST, true, eligible, cursor));
        assertEquals(third, QueuedMigrationCandidates.candidate(source, LOWEST_HIT_RATE_FIRST, true, eligible, cursor));
        assertEquals(List.of(tail, second, third), visited);
        cursor.reset();
        assertEquals(arrival, QueuedMigrationCandidates.candidate(source, LOWEST_HIT_RATE_FIRST, true, eligible, cursor));
        verify(source, never()).getStatus();
    }

    @ParameterizedTest
    @EnumSource(value = QueuedRequestPolicy.class, names = {"HIGHEST_HIT_RATE_FIRST", "LOWEST_HIT_RATE_FIRST"})
    void hitRateCursorVisitsEachRankThenWrapsWithTiesInReverseDispatchOrder(QueuedRequestPolicy policy) {
        var hot = queued(1, 8, 8, List.of());
        var cold = queued(2, 8, 0, List.of());
        var nearHead = queued(3, 8, 4, List.of());
        var nearTail = queued(4, 4, 2, List.of());
        var source = source(null, List.of(hot, nearHead, cold, nearTail));
        var cursor = new QueuedMigrationCandidates.Cursor();
        var expected = policy == HIGHEST_HIT_RATE_FIRST ? List.of(hot, nearTail, nearHead, cold, hot)
                : List.of(cold, nearTail, nearHead, hot, cold);
        for (var request : expected) {
            assertEquals(request, QueuedMigrationCandidates.candidate(source, policy, false, item -> true, cursor));
        }
    }

    @Test
    void missingCacheReportFallsBackToExactPlacementHitRatios() {
        var belowHalf = queued(1, Long.MAX_VALUE, Long.MAX_VALUE / 2, List.of());
        var half = queued(2, 2, 1, List.of());
        var source = source(null, List.of(belowHalf, half));
        assertEquals(half, QueuedMigrationCandidates.candidate(source, HIGHEST_HIT_RATE_FIRST, false, item -> true, new QueuedMigrationCandidates.Cursor()));
        assertEquals(belowHalf, QueuedMigrationCandidates.candidate(source, LOWEST_HIT_RATE_FIRST, false, item -> true, new QueuedMigrationCandidates.Cursor()));
        assertNull(QueuedMigrationCandidates.candidate(source, HIGHEST_HIT_RATE_FIRST, false, item -> false, new QueuedMigrationCandidates.Cursor()));
    }

    @Test
    void disabledSkipsTheWorkerQueueAndCacheReport() {
        var source = mock(PrefillEndpoint.class);
        assertNull(QueuedMigrationCandidates.candidate(source, DISABLED, true, item -> true, new QueuedMigrationCandidates.Cursor()));
        verifyNoInteractions(source);
    }

    private PrefillEndpoint source(CacheStatus cache, List<ScheduledRequest> items) {
        var source = mock(PrefillEndpoint.class);
        var status = mock(WorkerStatus.class);
        when(source.getStatus()).thenReturn(status);
        when(status.getCacheStatus()).thenReturn(cache);
        when(source.captureQueueSnapshot()).thenReturn(new WorkerBatcher.QueueSnapshot("source", 0, items));
        return source;
    }

    private ScheduledRequest queued(long sequence, long input, long placementHits, List<Long> keys) {
        var request = new Request();
        request.setBlockCacheKeys(keys);
        var context = new BalanceContext(new FlexlbConfig());
        context.setRequest(request);
        var item = mock(ScheduledRequest.class);
        when(item.enqueueSeq()).thenReturn(sequence);
        when(item.seqLen()).thenReturn(input);
        when(item.hitCache()).thenReturn(placementHits);
        when(item.ctx()).thenReturn(context);
        return item;
    }
}
