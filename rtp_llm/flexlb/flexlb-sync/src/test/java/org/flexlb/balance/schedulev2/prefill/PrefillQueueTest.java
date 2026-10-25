package org.flexlb.balance.schedulev2.prefill;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.estimate.PrefillTtftEstimate;
import org.flexlb.balance.estimate.WorkView;
import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.scheduler.SchedulingTestConfig;
import org.flexlb.balance.schedulev2.Request;
import org.flexlb.balance.schedulev2.V2TestSupport;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.enums.TaskPhase;
import org.junit.jupiter.api.Test;

import java.util.Comparator;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class PrefillQueueTest {

    private final FlexlbConfig config = SchedulingTestConfig.newConfig();
    private final PrefillEndpoint endpoint = V2TestSupport.prefillEndpoint("p1:8000");
    private final PrefillQueue queue = new PrefillQueue(endpoint, false);

    @Test
    void inflightLimitCountsDispatchedAndUnknownEngineRequests() {
        queue.addDispatched(new DispatchedRequest(request("1"), 10L, 5L));
        V2TestSupport.report(endpoint, 1L, 1_000L, 1_000L, Map.of("other", TaskPhase.RUNNING));
        queue.updateFromEngine();

        assertEquals(1L, queue.unknownRequestCount());
        assertTrue(queue.canDispatch(10L, 3));
        assertFalse(queue.canDispatch(10L, 2));
    }

    @Test
    void kvCheckSubtractsOnlyRequestsTheEngineHasNotReported() {
        queue.addDispatched(new DispatchedRequest(request("1"), 60L, 5L));
        V2TestSupport.report(endpoint, 1L, 100L, 1_000L, Map.of());
        queue.updateFromEngine();

        assertTrue(queue.canDispatch(40L, 10));
        assertFalse(queue.canDispatch(41L, 10));

        V2TestSupport.report(endpoint, 2L, 100L, 1_000L, Map.of("1", TaskPhase.PENDING));
        queue.updateFromEngine();

        assertTrue(queue.canDispatch(100L, 10));
    }

    @Test
    void emptyMachineAcceptsRequestLargerThanAvailableKv() {
        V2TestSupport.report(endpoint, 1L, 10L, 1_000L, Map.of());
        queue.updateFromEngine();

        assertTrue(queue.canDispatch(500L, 10));
    }

    @Test
    void recordIsRemovedAfterEngineReportedItAndThenDropsIt() {
        Request reported = request("1");
        Request notReported = request("2");
        queue.addDispatched(new DispatchedRequest(reported, 10L, 5L));
        queue.addDispatched(new DispatchedRequest(notReported, 10L, 5L));
        V2TestSupport.report(endpoint, 1L, 1_000L, 1_000L, Map.of("1", TaskPhase.RUNNING));
        queue.updateFromEngine();

        V2TestSupport.report(endpoint, 2L, 1_000L, 1_000L, Map.of());
        assertTrue(queue.updateFromEngine());

        assertEquals(1, queue.dispatchedCount());
        assertFalse(queue.removeDispatched(reported));
        assertTrue(queue.removeDispatched(notReported));
    }

    @Test
    void sameStatusVersionIsHandledOnce() {
        V2TestSupport.report(endpoint, 1L, 1_000L, 1_000L, Map.of());

        assertTrue(queue.updateFromEngine());
        assertFalse(queue.updateFromEngine());
    }

    @Test
    void loadViewHasWaitingItemsDispatchedWorkAndUnknownCount() {
        Request waiting = request("w");
        V2TestSupport.choose(waiting, endpoint, 4L, 20L);
        queue.addWaiting(waiting);
        Request dispatched = request("d");
        queue.addDispatched(new DispatchedRequest(dispatched, 10L, 30L));
        V2TestSupport.report(endpoint, 1L, 1_000L, 1_000L, Map.of("d", TaskPhase.RUNNING, "x", TaskPhase.PENDING));
        queue.updateFromEngine();
        GroupPlanner.Constraints constraints = new GroupPlanner.Constraints(1, Long.MAX_VALUE, Long.MAX_VALUE, 0L, 0L);
        Comparator<GroupPlanner.Item> order = Comparator.comparingLong(GroupPlanner.Item::enqueueSeq);

        PrefillTtftEstimate.LoadView view = queue.loadView(1_000L, order, constraints);

        assertEquals(List.of("w"), view.queue().activeItems().stream().map(GroupPlanner.Item::requestId).toList());
        assertEquals(4L, view.queue().activeItems().getFirst().hitCache());
        assertEquals(List.of(new WorkView.RequestWork("d", WorkView.Phase.ENGINE_RUNNING, 30L)),
                view.work().requests());
        assertEquals(1L, view.work().unknownRequestCount());

        queue.removeWaiting(waiting);
        assertNotEquals(view.ownershipVersion(), queue.loadView(1_000L, order, constraints).ownershipVersion());
    }

    private Request request(String requestId) {
        return V2TestSupport.request(config, requestId, 50, 16L);
    }
}
