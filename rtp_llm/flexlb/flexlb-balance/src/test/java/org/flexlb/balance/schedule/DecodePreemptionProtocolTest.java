package org.flexlb.balance.schedule;

import org.flexlb.balance.metrics.RequestSchedulerReporter;
import org.flexlb.balance.strategy.RoutingResult;
import org.flexlb.config.scheduler.PreemptionConfig;
import org.flexlb.config.service.RoleType;
import org.flexlb.worker.endpoint.DecodeEndpoint;
import org.flexlb.worker.endpoint.DecodeEndpoint.RequestRegistration;
import org.flexlb.worker.endpoint.DecodeResources.AdmissionCapacity;
import org.flexlb.worker.endpoint.DecodeResources.Capacity;
import org.flexlb.worker.endpoint.DecodeResources.DecodeRequestView;
import org.flexlb.worker.endpoint.DecodeResources.ResourceSnapshot;
import org.flexlb.worker.endpoint.Request;
import org.flexlb.worker.status.DecodeTaskPhase;
import org.flexlb.worker.status.WorkerStatus;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.RETURNS_DEEP_STUBS;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoMoreInteractions;
import static org.mockito.Mockito.when;

/**
 * A selected victim uses ordinary cancellation; the incoming request returns to the normal capacity queue.
 * Closing the incoming request cannot revoke a cancellation already handed to the victim's scheduler.
 */
class DecodePreemptionProtocolTest {
    @Test
    void cancellationReturnsCapacityWaitWithoutReservingForIncoming() {
        var f = new Fixture();
        var result = f.preemption.tryPreempt(f.incoming, f.endpoint).join();
        assertEquals(RoutingResult.Status.BLOCKED, result.status());
        verify(f.scheduler).cancel(f.victim, CancelReason.PRIORITY_PREEMPTED);
        verify(f.endpoint, never()).releaseRequest(any());
        verify(f.endpoint, never())
                .tryRegisterSelection(any(), anyString(), anyLong(), anyLong(), anyInt(), any());
    }

    @Test
    void incomingCancellationDoesNotCancelOrReleaseVictimsCleanup() {
        var f = new Fixture();
        var result = f.preemption.tryPreempt(f.incoming, f.endpoint);
        f.incoming.getFuture().cancel(false);
        assertEquals(RoutingResult.Status.BLOCKED, result.join().status());
        verify(f.scheduler, times(1)).cancel(f.victim, CancelReason.PRIORITY_PREEMPTED);
        verify(f.endpoint, never()).releaseRequest(any());
        f.preemption.close();
        verifyNoMoreInteractions(f.scheduler);
    }

    @Test
    void expiredIncomingDoesNotStartCancellation() {
        var f = new Fixture();
        when(f.incoming.getRequestExpiresAtMs()).thenReturn(1L);
        assertNull(f.preemption.tryPreempt(f.incoming, f.endpoint));
        verify(f.scheduler, never()).cancel(any(RequestContext.class), any());
    }

    @Test
    void replacedReservationCannotBeCancelledUsingOldSnapshot() {
        var f = new Fixture();
        when(f.route.decodeRegistration()).thenReturn(new RequestRegistration(1L, "a"));
        assertEquals(
                RoutingResult.Status.BLOCKED,
                f.preemption.tryPreempt(f.incoming, f.endpoint).join().status());
        verify(f.scheduler, never()).cancel(any(RequestContext.class), any());
    }

    @Test
    void replacedEndpointCannotBeCancelledUsingOldSnapshot() {
        var f = new Fixture();
        when(f.route.decodeEndpoint()).thenReturn(mock(DecodeEndpoint.class));
        f.preemption.tryPreempt(f.incoming, f.endpoint).join();
        verify(f.scheduler, never()).cancel(any(RequestContext.class), any());
    }

    @Test
    void stoppedPreemptionDoesNotStartMoreCancellations() {
        var f = new Fixture();
        f.preemption.stopAdmission();
        assertNull(f.preemption.tryPreempt(f.incoming, f.endpoint));
        assertTrue(f.preemption.awaitIdle(System.nanoTime()));
        verify(f.scheduler, never()).cancel(any(RequestContext.class), any());
    }

    @Test
    void policyCannotSubstituteARequestOutsideTheCapturedSnapshot() {
        var f = new Fixture();
        var forged =
                new DecodeRequestView(
                        "other",
                        10,
                        100,
                        100,
                        DecodeTaskPhase.RUNNING,
                        new RequestRegistration(1L, "other"));
        f.proposal = List.of(forged);
        assertThrows(
                IllegalArgumentException.class,
                () -> f.preemption.tryPreempt(f.incoming, f.endpoint));
        verify(f.scheduler, never()).cancel(any(RequestContext.class), any());
    }

    /**
     * Supplies one engine request, an independent waiter and a custom policy through the public preemption entry.
     */
    static final class Fixture {
        final RequestRepository repository = mock(RequestRepository.class);
        final AbstractRequestScheduler scheduler = mock(AbstractRequestScheduler.class);
        final AbstractRequestScheduler incomingScheduler = mock(AbstractRequestScheduler.class);
        final DecodeEndpoint endpoint = mock(DecodeEndpoint.class);
        final RequestContext incoming = mock(RequestContext.class);
        final RequestContext victim = mock(RequestContext.class, RETURNS_DEEP_STUBS);
        final RequestResult route = mock(RequestResult.class);
        final RequestRegistration reservation = new RequestRegistration(1L, "a");
        final DefaultDecodePreemption preemption;
        List<DecodeRequestView> proposal;

        Fixture() {
            var status =
                    WorkerStatus.createDiscovered(
                            RoleType.DECODE, null, "127.0.0.1", 8000, 8001, null);
            when(endpoint.getStatus()).thenReturn(status);
            when(endpoint.ipPort()).thenReturn("127.0.0.1:8000");
            when(endpoint.acceptsNewRequests()).thenReturn(true);
            var selected =
                    new DecodeRequestView("a", 10, 100, 100, DecodeTaskPhase.RUNNING, reservation);
            proposal = List.of(selected);
            var routing = mock(DecodeEndpoint.SelectionSnapshot.class);
            when(routing.address()).thenReturn("127.0.0.1:8000");
            var snapshot = new ResourceSnapshot(routing, Map.of("a", selected));
            when(endpoint.resourceSnapshot()).thenReturn(snapshot);
            when(repository.findActive("a")).thenReturn(victim);
            when(victim.snapshot().route()).thenReturn(route);
            when(victim.getRequestScheduler()).thenReturn(scheduler);
            when(route.decodeEndpoint()).thenReturn(endpoint);
            when(route.decodeRegistration()).thenReturn(reservation);
            when(incoming.getRequest())
                    .thenReturn(Request.builder().requestId("b").seqLen(100L).build());
            when(incoming.getRequestId()).thenReturn("b");
            when(incoming.getPriority()).thenReturn(90);
            when(incoming.getSchedulingMetadata())
                    .thenReturn(SchedulingMetadata.explicit(90, Long.MAX_VALUE));
            when(incoming.getRequestExpiresAtMs()).thenReturn(Long.MAX_VALUE);
            when(incoming.getFuture()).thenReturn(new CompletableFuture<>());
            when(incoming.getRequestScheduler()).thenReturn(incomingScheduler);
            when(incomingScheduler.isAdmissionOpen(incoming)).thenReturn(true);
            var config = SchedulingTestConfig.newConfig();
            SchedulingTestConfig.usePriorityQueue(config);
            config.priorityOrdering().setPreemption(new PreemptionConfig());
            DecodePreemptionPolicy policy =
                    (request, priority, capacity, observed, settings, failures) ->
                            new DecodePreemptionPolicy.DecodePreemptionDecision(
                                    new Capacity(1, 0, 0),
                                    new DecodePreemptionPolicy.DecodePreemptionPlan(
                                            endpoint.ipPort(),
                                            proposal,
                                            "decode_slot_full",
                                            100,
                                            1));
            preemption =
                    new DefaultDecodePreemption(
                            repository,
                            mock(RequestSchedulerReporter.class),
                            SchedulingTestConfig.configService(config),
                            new AdmissionCapacity(1, 100),
                            policy);
        }
    }
}
