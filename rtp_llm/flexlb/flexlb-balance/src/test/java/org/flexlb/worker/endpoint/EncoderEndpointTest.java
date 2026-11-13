package org.flexlb.worker.endpoint;

import org.flexlb.balance.schedule.AbstractRequestScheduler;
import org.flexlb.balance.schedule.RequestContext;
import org.flexlb.balance.schedule.RequestResult;
import org.flexlb.config.service.RoleType;
import org.flexlb.worker.status.WorkerStatus;
import org.junit.jupiter.api.Test;

import static org.flexlb.worker.endpoint.EndpointTestSupport.runWithStateLock;
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

/**
 * Checks Encoder admission and load accounting across release and cleanup, including stale route identities.
 */
class EncoderEndpointTest {
    private final AbstractRequestScheduler lifecycle = mock(AbstractRequestScheduler.class);
    private final CapacityNotifier availability = mock(CapacityNotifier.class);

    @Test
    void registrationChargesCapacityBeforeRouteConstructionAndReleasesExactRecord() {
        var endpoint = endpoint();
        var first = endpoint.tryRegisterSelection("first", 1, 17L);
        assertEquals(WorkerEndpoint.RegistrationStatus.REGISTERED, first.status());
        assertEquals(1L, endpoint.selectionSnapshot().requestCount());
        assertEquals(17L, endpoint.selectionSnapshot().uncachedTokens());
        assertEquals(
                WorkerEndpoint.RegistrationStatus.CAPACITY_FULL,
                endpoint.tryRegisterSelection("second", 1, 20L).status());
        assertTrue(endpoint.releaseRequest(first.value()));
        var replacement = endpoint.tryRegisterSelection("first", 1, 20L);
        assertFalse(endpoint.releaseRequest(first.value()));
        assertEquals(20L, endpoint.selectionSnapshot().uncachedTokens());
        assertTrue(endpoint.releaseRequest(replacement.value()));
    }

    @Test
    void selectionSnapshotIncludesUnreportedRequestsAndTheirWorkTogether() {
        var endpoint = endpoint();
        assertTrue(EndpointTestSupport.commitEncoder(endpoint, route("snapshot", 17L), 2, 17L));
        var snapshot = endpoint.selectionSnapshot();
        assertEquals(1L, snapshot.requestCount());
        assertEquals(17L, snapshot.uncachedTokens());
        assertEquals(endpoint.ipPort(), snapshot.endpoint().ipPort());
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(booleans = {false, true})
    void orphanSweepCannotRemoveAReplacementOrAFreshWorkerObservation(boolean refreshStatus) {
        var endpoint = endpoint();
        var original = route("orphan", 10L);
        assertTrue(EndpointTestSupport.commitEncoder(endpoint, original, 1, 10L));
        assertEquals(
                0,
                endpoint.evictExpiredRequests(
                        -1L,
                        id -> {
                            assertFalse(endpoint.stateLock().isHeldByCurrentThread());
                            if (refreshStatus) {
                                var task = new org.flexlb.worker.status.TaskInfo();
                                task.setRequestId(id);
                                task.setPhase(org.flexlb.worker.status.TaskPhase.RUNNING);
                                var response = new org.flexlb.worker.status.WorkerStatusResponse();
                                response.setRunningTaskInfo(java.util.Map.of(id, task));
                                EndpointTestSupport.applyStatus(endpoint, response).run();
                            } else {
                                endpoint.releaseRequest(original);
                                assertTrue(
                                        EndpointTestSupport.commitEncoder(
                                                endpoint, route(id, 20L), 1, 20L));
                            }
                            return false;
                        }));
        assertEquals(1, endpoint.evictExpiredRequests(-1L, ignored -> false));
        assertEquals(0, endpoint.evictExpiredRequests(-1L, ignored -> false));
    }

    @Test
    void endpointsWithoutRequestRecordsNeedNoExpirationWork() {
        WorkerEndpoint endpoint =
                new WorkerEndpoint(
                        WorkerStatus.createDiscovered(
                                RoleType.VIT, "g", "127.0.0.1", 8001, 8002, ""));
        assertEquals(
                0,
                endpoint.evictExpiredRequests(
                        -1L,
                        id -> {
                            throw new AssertionError("VIT has no local request records");
                        }));
    }

    @Test
    void activeAndFinishedEventsKeepTheirOrderAndPublishOutsideLocks() {
        var endpoint = endpoint();
        var route = route("request", 20);
        assertTrue(EndpointTestSupport.commitEncoder(endpoint, route, 10, 20L));
        var task = new org.flexlb.worker.status.TaskInfo();
        task.setRequestId("request");
        task.setPhase(org.flexlb.worker.status.TaskPhase.RUNNING);
        var response = new org.flexlb.worker.status.WorkerStatusResponse();
        response.setRunningTaskInfo(java.util.Map.of("request", task));
        response.setFinishedTaskInfo(java.util.Map.of("request", task));
        var events = new java.util.ArrayList<org.flexlb.worker.endpoint.RequestEventType>();
        var context = route.ctx();
        doAnswer(
                        call -> {
                            assertFalse(endpoint.stateLock().isHeldByCurrentThread());
                            assertFalse(endpoint.getStatus().lock.isHeldByCurrentThread());
                            org.flexlb.worker.endpoint.WorkerRequestEvent event =
                                    call.getArgument(1);
                            events.add(event.type());
                            return null;
                        })
                .when(lifecycle)
                .onWorkerRequestEvent(eq(context), any());

        var publish = EndpointTestSupport.applyStatus(endpoint, response);
        assertTrue(events.isEmpty());
        publish.run();
        assertEquals(
                java.util.List.of(
                        org.flexlb.worker.endpoint.RequestEventType.ACTIVE,
                        org.flexlb.worker.endpoint.RequestEventType.COMPLETED),
                events);
        assertEquals(0L, endpoint.selectionSnapshot().requestCount());
        EndpointTestSupport.applyStatus(endpoint, response).run();
        assertEquals(2, events.size());
    }

    @Test
    void snapshotPublicationFailureStopsFurtherAdmission() {
        var status = spy(endpoint().getStatus());
        var endpoint = new EncoderEndpoint(status, availability);
        org.mockito.Mockito.doThrow(new IllegalStateException("publication failed"))
                .when(status)
                .publishReducedStatus(any());
        org.junit.jupiter.api.Assertions.assertThrows(
                IllegalStateException.class,
                () ->
                        EndpointTestSupport.applyStatus(
                                endpoint, new org.flexlb.worker.status.WorkerStatusResponse()));
        assertTrue(endpoint.hasStoppedAcceptingRequests());
        assertFalse(EndpointTestSupport.commitEncoder(endpoint, route("late", 20), 10, 20L));
        assertFalse(endpoint.stateLock().isHeldByCurrentThread());
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(booleans = {false, true})
    void eitherClosedAdmissionGateRejectsCommitButAllowsRelease(boolean retired) {
        var endpoint = endpoint();
        var prepared = route("prepared", 20);
        try {
            assertTrue(EndpointTestSupport.commitEncoder(endpoint, prepared, 10, 20L));
            if (retired) endpoint.stopAcceptingRequests();
            else endpoint.getStatus().markRemovedFromDiscovery();
            assertEquals(retired, endpoint.hasStoppedAcceptingRequests());
            assertFalse(endpoint.getStatus().isAvailableForScheduling());
            assertFalse(endpoint.acceptsNewRequests());
            assertFalse(EndpointTestSupport.commitEncoder(endpoint, route("late", 20), 10, 20L));
            endpoint.releaseRequest(prepared);
            assertEquals(0L, endpoint.selectionSnapshot().requestCount());
        } finally {
            org.flexlb.worker.endpoint.EndpointTestSupport.closeAndWait(endpoint);
            EndpointTestSupport.awaitClosed(endpoint, 2000L);
        }
    }

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
        var request = spy(pending.ctx().getRequest());
        when(pending.ctx().getRequest()).thenReturn(request);
        doAnswer(
                        call -> {
                            assertFalse(endpoint.stateLock().isHeldByCurrentThread());
                            return call.callRealMethod();
                        })
                .when(request)
                .getSeqLen();
        try {
            assertTrue(EndpointTestSupport.commitEncoder(endpoint, route("running", 10), 4, 10L));
            assertTrue(EndpointTestSupport.commitEncoder(endpoint, pending, 4, 20L));
            assertEquals(120L, endpoint.selectionSnapshot().uncachedTokens());
            endpoint.releaseRequest(pending);
            assertEquals(100L, endpoint.selectionSnapshot().uncachedTokens());
        } finally {
            org.flexlb.worker.endpoint.EndpointTestSupport.closeAndWait(endpoint);
            EndpointTestSupport.awaitClosed(endpoint, 2000L);
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
                            org.flexlb.worker.endpoint.WorkerRequestEvent evidence =
                                    call.getArgument(1);
                            org.junit.jupiter.api.Assertions.assertSame(
                                    endpoint, evidence.endpoint());
                            return null;
                        })
                .when(lifecycle)
                .onWorkerRequestEvent(any(), any());
        runWithStateLock(
                endpoint.stateLock(),
                () -> {
                    assertTrue(EndpointTestSupport.commitEncoder(endpoint, route, 2, 10L));
                    org.flexlb.worker.endpoint.EndpointTestSupport.closeAndWait(endpoint);
                    assertFalse(
                            EndpointTestSupport.commitEncoder(endpoint, route("late", 10), 2, 10L));
                });
        EndpointTestSupport.awaitClosed(endpoint, 2000L);
        assertEquals(0, endpoint.selectionSnapshot().requestCount());
        verify(lifecycle).onWorkerRequestEvent(eq(route.ctx()), any());
    }

    @Test
    void closingSeveralPendingRequestsNotifiesCapacityOnce() {
        var endpoint = endpoint();
        {
            var selectedEndpoint = endpoint;

            assertTrue(EndpointTestSupport.commitEncoder(endpoint, route("request-1", 10), 2, 10L));
            assertTrue(EndpointTestSupport.commitEncoder(endpoint, route("request-2", 20), 2, 20L));
            assertFalse(
                    EndpointTestSupport.commitEncoder(endpoint, route("request-1", 10), 2, 10L));
        }
        assertEquals(2, endpoint.selectionSnapshot().requestCount());
        org.flexlb.worker.endpoint.EndpointTestSupport.closeAndWait(endpoint);
        EndpointTestSupport.awaitClosed(endpoint, 2000L);
        assertEquals(0, endpoint.selectionSnapshot().requestCount());
        verify(availability).notifyChanged(endpoint);
        verify(lifecycle, times(2)).onWorkerRequestEvent(any(), any());
    }

    @Test
    void releaseRequiresTheExactRouteAndCannotRemoveAReplacement() {
        var endpoint = endpoint();
        var original = route("same-id", 10);
        var replacement = route("same-id", 20);
        try {
            var selectedEndpoint = endpoint;

            assertTrue(EndpointTestSupport.commitEncoder(endpoint, original, 1, 10L));
            endpoint.releaseRequest(original);
            assertTrue(EndpointTestSupport.commitEncoder(endpoint, replacement, 1, 20L));
            endpoint.releaseRequest(original);
            verify(availability, times(1)).notifyChanged(endpoint);
            assertEquals(1, endpoint.selectionSnapshot().requestCount());
            assertEquals(20, endpoint.selectionSnapshot().uncachedTokens());
            endpoint.releaseRequest(replacement);
            endpoint.releaseRequest(replacement);
            verify(availability, times(2)).notifyChanged(endpoint);
        } finally {
            org.flexlb.worker.endpoint.EndpointTestSupport.closeAndWait(endpoint);
            EndpointTestSupport.awaitClosed(endpoint, 2000L);
        }
    }

    private RequestResult route(String id, long tokens) {
        var route = mock(RequestResult.class);
        var context = mock(RequestContext.class);

        when(route.requestId()).thenReturn(String.valueOf(id));
        when(route.ctx()).thenReturn(context);

        when(context.getRequestScheduler()).thenReturn(lifecycle);
        when(context.getRequest())
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
