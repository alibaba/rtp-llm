package org.flexlb.balance.endpoint;

import org.flexlb.balance.scheduler.AbstractRequestScheduler;
import org.flexlb.balance.scheduler.ContextResult;
import org.flexlb.balance.scheduler.PlacementAvailability;
import org.flexlb.balance.scheduler.RequestContext;
import org.flexlb.balance.scheduler.RequestData;
import org.flexlb.balance.scheduler.RouteExecution;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.Test;

import static org.flexlb.balance.endpoint.EndpointTestSupport.runWithStateLock;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class EncoderEndpointTest {
    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(booleans = {false, true})
    void eitherClosedAdmissionGateRejectsCommitButAllowsRelease(boolean retired) {
        var endpoint = endpoint();
        var prepared = route("prepared", 20);
        try {
            assertTrue(endpoint.commit(prepared, 10));
            if (retired) endpoint.beginRetirement();
            else endpoint.getStatus().markRemovedFromDiscovery();
            assertEquals(retired, endpoint.isRetired());
            assertEquals(retired, endpoint.getStatus().isAvailableForScheduling());
            assertFalse(endpoint.acceptsNewRequests());
            assertFalse(endpoint.commit(route("late", 20), 10));
            endpoint.release(prepared);
            assertEquals(0L, endpoint.requestCount());
        } finally {
            endpoint.close();
            endpoint.awaitRetirement(2000L);
        }
    }

    private final AbstractRequestScheduler lifecycle = mock(AbstractRequestScheduler.class);
    private final PlacementAvailability availability = mock(PlacementAvailability.class);

    @Test
    void scoringCountsEngineAndPendingTokensWithoutHoldingTheLockDuringEngineTraversal() {
        var status = spy(endpoint().getStatus());
        var endpoint = new EncoderEndpoint(status, availability);
        var engine = mock(WorkerStatus.EngineObservation.class);
        var task = mock(WorkerStatus.TaskObservation.class);
        when(status.committedEngineObservation()).thenReturn(engine);
        when(engine.runningQueryLen()).thenReturn(1L);
        when(engine.runningTaskList()).thenReturn(java.util.Map.of("running", task));
        doAnswer(
                        call -> {
                            assertFalse(endpoint.stateLock().isHeldByCurrentThread());
                            return 100L;
                        })
                .when(task)
                .inputLength();
        var pending = route("pending", 20);
        var request = spy(pending.ctx().data().getRequest());
        when(pending.ctx().data().getRequest()).thenReturn(request);
        doAnswer(
                        call -> {
                            assertFalse(endpoint.stateLock().isHeldByCurrentThread());
                            return call.callRealMethod();
                        })
                .when(request)
                .getSeqLen();
        try {
            assertTrue(endpoint.commit(route("running", 10), 4));
            assertTrue(endpoint.commit(pending, 4));
            assertEquals(120L, endpoint.uncachedTokens());
            endpoint.release(pending);
            assertEquals(100L, endpoint.uncachedTokens());
        } finally {
            endpoint.close();
            endpoint.awaitRetirement(2000L);
        }
    }

    @Test
    void retirementCleansEarlierAdmissionAndRejectsLaterAdmission() throws Exception {
        var endpoint = endpoint();
        var route = route("request", 10);
        doAnswer(
                        call -> {
                            assertFalse(endpoint.stateLock().isHeldByCurrentThread());
                            assertFalse(endpoint.getStatus().lock.isHeldByCurrentThread());
                            ContextResult.EndpointRetirement evidence = call.getArgument(1);
                            org.junit.jupiter.api.Assertions.assertSame(
                                    endpoint, evidence.endpoint());
                            return null;
                        })
                .when(lifecycle)
                .onEndpointRetired(any(), any());
        runWithStateLock(
                endpoint.stateLock(),
                () -> {
                    assertTrue(endpoint.commit(route, 2));
                    endpoint.close();
                    assertFalse(endpoint.commit(route("late", 10), 2));
                });
        endpoint.awaitRetirement(2000L);
        assertEquals(0, endpoint.requestCount());
        verify(lifecycle).onEndpointRetired(eq(route.ctx()), any());
    }

    @Test
    void closingSeveralPendingRequestsNotifiesCapacityOnce() {
        var endpoint = endpoint();
        {
            var selectedEndpoint = endpoint;

            assertTrue(endpoint.commit(route("request-1", 10), 2));
            assertTrue(endpoint.commit(route("request-2", 20), 2));
            assertFalse(endpoint.commit(route("request-1", 10), 2));
        }
        assertEquals(2, endpoint.requestCount());
        endpoint.close();
        endpoint.awaitRetirement(2000L);
        assertEquals(0, endpoint.requestCount());
        verify(availability).changed(endpoint);
        verify(lifecycle, times(2)).onEndpointRetired(any(), any());
    }

    @Test
    void releaseRequiresTheExactRouteAndCannotRemoveAReplacement() {
        var endpoint = endpoint();
        var original = route("same-id", 10);
        var replacement = route("same-id", 20);
        try {
            var selectedEndpoint = endpoint;

            assertTrue(endpoint.commit(original, 1));
            endpoint.release(original);
            assertTrue(endpoint.commit(replacement, 1));
            endpoint.release(original);
            assertEquals(1, endpoint.requestCount());
            assertEquals(20, endpoint.uncachedTokens());
            endpoint.release(replacement);
        } finally {
            endpoint.close();
            endpoint.awaitRetirement(2000L);
        }
    }

    private RouteExecution route(String id, long tokens) {
        var route = mock(RouteExecution.class);
        var context = mock(RequestContext.class);
        var data = mock(RequestData.class);
        when(route.requestId()).thenReturn(String.valueOf(id));
        when(route.ctx()).thenReturn(context);
        when(context.data()).thenReturn(data);
        when(data.getRequestScheduler()).thenReturn(lifecycle);
        when(data.getRequest())
                .thenReturn(Request.builder().requestId(String.valueOf(id)).seqLen(tokens).build());
        return route;
    }

    private EncoderEndpoint endpoint() {
        var status =
                WorkerStatus.createDiscovered(
                        RoleType.ENCODER, "default", "127.0.0.1", 8001, 8002, "");
        status.lock.lock();
        try {
            status.recordSuccessfulPoll(true);
        } finally {
            status.lock.unlock();
        }
        return new EncoderEndpoint(status, availability);
    }
}
