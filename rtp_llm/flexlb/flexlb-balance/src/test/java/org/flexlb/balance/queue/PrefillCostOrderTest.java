package org.flexlb.balance.queue;

import org.flexlb.balance.request.Request;
import org.flexlb.balance.schedule.RequestContext;
import org.flexlb.balance.schedule.SchedulingMetadata;
import org.flexlb.balance.schedule.SchedulingTestConfig;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.List;
import java.util.OptionalLong;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Checks priority composition, snapshot selection and stale asynchronous estimates.
 */
class PrefillCostOrderTest {
    @Test
    void priorityAndCostAreIndependentAndUnknownCostsUseStableOrder() {
        for (boolean priority : new boolean[] {false, true}) {
            var queue = new OrderedRequestQueue(priority, true);
            var high = entry(90);
            var cheap = entry(10);
            queue.add(high);
            queue.add(cheap);
            assertEquals(List.of(high), queue.scanForPlanningCandidates(1, 10, r -> true));
            queue.retry(high);
            complete(queue, high, 90, cheap, 10);
            queue.beginScan();
            assertEquals(
                    List.of(priority ? high : cheap),
                    queue.scanForPlanningCandidates(1, 10, r -> true));
        }
    }

    @Test
    void lateEstimateCannotChangeRestoredMembershipAndHugeProductsRemainExact() {
        var queue = new OrderedRequestQueue(false, true);
        var first = entry(50, Long.MAX_VALUE);
        var second = entry(50, Long.MAX_VALUE);
        queue.add(first);
        var old = queue.estimates(1).getFirst();
        queue.remove(first);
        queue.restore(first);
        queue.add(second);
        old.complete(OptionalLong.of(0));
        complete(queue, first, Long.MAX_VALUE - 1, second, 1);
        assertEquals(List.of(second), queue.scanForPlanningCandidates(1, 10, r -> true));
        assertEquals(2, queue.drain().size());
        assertTrue(queue.estimates(10).isEmpty());
    }

    @ParameterizedTest
    @ValueSource(ints = {1, 4})
    void takesTopRequestsFromTheWholeSnapshot(int limit) {
        var queue = new OrderedRequestQueue(false, true);
        var expensive = entry(50);
        var cheapest = entry(50);
        var ready = List.of(expensive, entry(50), entry(50), entry(50), cheapest);
        var cancelled = entry(50);
        queue.add(cancelled);
        ready.forEach(queue::add);
        long compute = 50;
        for (var ticket : queue.estimates(6)) {
            if (ticket.request() != cancelled) {
                ticket.complete(OptionalLong.of(compute));
                compute -= 10;
            }
        }
        var ranked = ready.reversed();
        assertEquals(
                ranked.subList(0, limit),
                queue.scanForPlanningCandidates(limit, 1, request -> request != cancelled));
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void eachCallIncludesTheLatestBestArrival(boolean priority) {
        var queue = new OrderedRequestQueue(priority, true);
        var original = entry(50);
        queue.add(original);
        queue.estimates(1).getFirst().complete(OptionalLong.of(50));
        assertEquals(List.of(original), queue.scanForPlanningCandidates(1, 1, r -> true));

        for (int remaining = 10; remaining > 0; remaining--) {
            var arrival = entry(90);
            queue.add(arrival);
            queue.estimates(1).getFirst().complete(OptionalLong.of(remaining));
            assertEquals(List.of(arrival), queue.scanForPlanningCandidates(1, 1, r -> true));
        }
    }

    private static void complete(
            OrderedRequestQueue queue,
            RequestContext first,
            long a,
            RequestContext second,
            long b) {
        for (var ticket : queue.estimates(10))
            ticket.complete(OptionalLong.of(ticket.request() == first ? a : b));
    }

    private static RequestContext entry(int priority) {
        return entry(priority, 100);
    }

    private static RequestContext entry(int priority, long inputTokens) {
        var context = new RequestContext(SchedulingTestConfig.newConfig());
        context.setRequest(Request.builder().seqLen(inputTokens).build());
        context.setSchedulingMetadata(SchedulingMetadata.of(priority, null, Long.MAX_VALUE, 50));
        return context;
    }
}
