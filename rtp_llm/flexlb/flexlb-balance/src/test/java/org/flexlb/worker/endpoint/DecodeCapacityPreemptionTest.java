package org.flexlb.worker.endpoint;

import org.flexlb.balance.schedule.AbstractRequestScheduler;
import org.flexlb.balance.schedule.CancelReason;
import org.flexlb.balance.schedule.DecodePreemptionPolicy.DecodePreemptionPlan;
import org.flexlb.balance.schedule.DefaultDecodePreemption;
import org.flexlb.balance.schedule.PriorityDecodePreemptionPolicy;
import org.flexlb.balance.schedule.RequestContext;
import org.flexlb.balance.schedule.RequestRepository;
import org.flexlb.balance.schedule.RequestResult;
import org.flexlb.balance.schedule.SchedulingMetadata;
import org.flexlb.balance.schedule.SchedulingTestConfig;
import org.flexlb.config.scheduler.PreemptionConfig;
import org.flexlb.config.scheduler.PreemptionStage;
import org.flexlb.config.service.RoleType;
import org.flexlb.worker.status.WorkerStatusResponse;

import java.util.EnumSet;
import java.util.HashMap;
import java.util.List;
import java.util.concurrent.CompletableFuture;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.RETURNS_DEEP_STUBS;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Exercises Decode admission limits, worker calibration and exact resource release.
 * A competing admission must not make the same waiter cancel another request after its first preemption.
 */
class DecodeCapacityPreemptionTest {
    @org.junit.jupiter.api.Test
    void releaseNotifiesOnceOutsideLockAfterCapacityIsAvailable() {
        var notifier = new CapacityNotifier();
        var endpoint =
                new DecodeEndpoint(
                        EndpointTestSupport.workerStatus(RoleType.DECODE, "10.0.0.1", 8080, 8081),
                        mock(RequestRepository.class),
                        notifier);
        updateCapacity(endpoint, 1000);
        var capacity = new DecodeResources.AdmissionCapacity(1, 100);
        var first =
                endpoint.tryRegisterSelection(
                                endpoint.selectionSnapshot(), "first", 100, 100, 10, capacity)
                        .value();
        assertNotNull(first);
        var notifications = new java.util.concurrent.atomic.AtomicInteger();
        var underLock = new java.util.concurrent.atomic.AtomicBoolean();
        var admitted =
                new java.util.concurrent.atomic.AtomicReference<
                        WorkerEndpoint.RegistrationStatus>();
        CapacityNotifier.Listener listener =
                key -> {
                    notifications.incrementAndGet();
                    underLock.set(endpoint.stateLock().isHeldByCurrentThread());
                    admitted.set(
                            endpoint.tryRegisterSelection(
                                            endpoint.selectionSnapshot(),
                                            "competitor",
                                            100,
                                            100,
                                            20,
                                            capacity)
                                    .status());
                };
        notifier.addListener(listener);
        try {
            assertTrue(endpoint.releaseRequest(first));
            assertFalse(endpoint.releaseRequest(first));
            assertEquals(1, notifications.get());
            assertFalse(underLock.get());
            assertEquals(WorkerEndpoint.RegistrationStatus.REGISTERED, admitted.get());
            assertEquals(1, endpoint.selectionSnapshot().usage().occupiedRequests());
        } finally {
            notifier.removeListener(listener);
            EndpointTestSupport.closeAndWait(endpoint);
        }
    }

    @org.junit.jupiter.api.Test
    void incomingMustNotCancelAnotherRequestAfterItsFreedSlotWasTaken() {
        var endpoint = endpoint(1000L);
        var capacity = new DecodeResources.AdmissionCapacity(1, 100);
        var repository = mock(RequestRepository.class);
        var victimScheduler = mock(AbstractRequestScheduler.class);
        var incomingScheduler = mock(AbstractRequestScheduler.class);
        var incoming = mock(RequestContext.class);
        when(incoming.getRequest())
                .thenReturn(Request.builder().requestId("incoming").seqLen(100L).build());
        when(incoming.getRequestId()).thenReturn("incoming");
        when(incoming.getPriority()).thenReturn(90);
        when(incoming.getSchedulingMetadata())
                .thenReturn(SchedulingMetadata.explicit(90, Long.MAX_VALUE));
        when(incoming.getRequestExpiresAtMs()).thenReturn(Long.MAX_VALUE);
        when(incoming.getFuture()).thenReturn(new CompletableFuture<>());
        when(incoming.getRequestScheduler()).thenReturn(incomingScheduler);
        when(incomingScheduler.isAdmissionOpen(incoming)).thenReturn(true);
        var config = SchedulingTestConfig.newConfig();
        SchedulingTestConfig.usePriorityQueue(config);
        var settings = new PreemptionConfig();
        settings.setAllowedPreemptionStages(EnumSet.of(PreemptionStage.DECODE_ENGINE_OWNED));
        config.priorityOrdering().setPreemption(settings);
        var preemption =
                new DefaultDecodePreemption(
                        repository,
                        mock(org.flexlb.balance.metrics.RequestSchedulerReporter.class),
                        SchedulingTestConfig.configService(config),
                        capacity,
                        new PriorityDecodePreemptionPolicy());
        try {
            var first =
                    endpoint.tryRegisterSelection(
                                    endpoint.selectionSnapshot(), "first", 100, 100, 10, capacity)
                            .value();
            assertNotNull(first);
            reportRunning(endpoint, "first", 900);
            var firstContext = bindRequest(repository, victimScheduler, endpoint, first);
            assertEquals(
                    org.flexlb.balance.strategy.RoutingResult.Status.BLOCKED,
                    preemption.tryPreempt(incoming, endpoint).join().status());
            verify(victimScheduler).cancel(firstContext, CancelReason.PRIORITY_PREEMPTED);
            assertEquals(
                    WorkerEndpoint.RegistrationStatus.CAPACITY_FULL,
                    endpoint.tryRegisterSelection(
                                    endpoint.selectionSnapshot(),
                                    "incoming",
                                    100,
                                    100,
                                    90,
                                    capacity)
                            .status());

            // Confirmed cleanup frees the slot; a competing request reaches admission before the
            // waiter.
            assertTrue(endpoint.releaseRequest(first));
            var competitor =
                    endpoint.tryRegisterSelection(
                                    endpoint.selectionSnapshot(),
                                    "competitor",
                                    100,
                                    100,
                                    20,
                                    capacity)
                            .value();
            assertNotNull(competitor);
            reportRunning(endpoint, "competitor", 900);
            var competitorContext = bindRequest(repository, victimScheduler, endpoint, competitor);
            assertEquals(
                    WorkerEndpoint.RegistrationStatus.CAPACITY_FULL,
                    endpoint.tryRegisterSelection(
                                    endpoint.selectionSnapshot(),
                                    "incoming",
                                    100,
                                    100,
                                    90,
                                    capacity)
                            .status());

            preemption.tryPreempt(incoming, endpoint).join();
            verify(victimScheduler, never())
                    .cancel(competitorContext, CancelReason.PRIORITY_PREEMPTED);
        } finally {
            preemption.close();
            EndpointTestSupport.closeAndWait(endpoint);
        }
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.CsvSource({
        "1000, 1, 100, 100, 100, 100, 100, false",
        "1000, 2, 100, 100, 100, 100, 100, true",
        "1000, 0, 100, 100, 100, 100, 100, true",
        "300, 0, 100, 100, 100, 200, 200, true",
        "300, 0, 100, 100, 100, 201, 201, false",
        "1000, 0, 80, 100, 400, 100, 400, true",
        "1000, 0, 80, 100, 400, 100, 401, false"
    })
    void admissionChecksRequestHardKvAndExpectedKvBoundaries(
            long available,
            long requestLimit,
            long kvPercent,
            long firstHard,
            long firstExpected,
            long nextHard,
            long nextExpected,
            boolean admitted) {
        var endpoint = endpoint(available);
        var capacity = new DecodeResources.AdmissionCapacity(requestLimit, kvPercent);
        try {
            var first =
                    endpoint.tryRegisterSelection(
                                    endpoint.selectionSnapshot(),
                                    "first",
                                    firstHard,
                                    firstExpected,
                                    10,
                                    capacity)
                            .value();
            assertNotNull(first);
            var before = endpoint.selectionSnapshot().usage();
            var result =
                    endpoint.tryRegisterSelection(
                            endpoint.selectionSnapshot(),
                            "next",
                            nextHard,
                            nextExpected,
                            20,
                            capacity);
            assertEquals(
                    admitted
                            ? WorkerEndpoint.RegistrationStatus.REGISTERED
                            : WorkerEndpoint.RegistrationStatus.CAPACITY_FULL,
                    result.status());
            if (!admitted)
                assertEquals(
                        before,
                        endpoint.selectionSnapshot().usage(),
                        "rejection must not consume capacity");
            if (admitted) assertTrue(endpoint.releaseRequest(result.value()));
            assertTrue(endpoint.releaseRequest(first));
            assertEquals(0, endpoint.selectionSnapshot().usage().occupiedRequests());
            assertEquals(0L, endpoint.selectionSnapshot().usage().hardReservedKvTokens());
        } finally {
            EndpointTestSupport.closeAndWait(endpoint);
        }
    }

    @org.junit.jupiter.api.Test
    void duplicateAndOldIdentityReleaseCannotFreeReplacementCapacity() {
        var endpoint = endpoint(1000);
        var capacity = new DecodeResources.AdmissionCapacity(1, 100);
        try {
            var first =
                    endpoint.tryRegisterSelection(
                                    endpoint.selectionSnapshot(), "same", 100, 200, 10, capacity)
                            .value();
            assertNotNull(first);
            assertFalse(
                    endpoint.releaseRequest(
                            new DecodeEndpoint.RequestRegistration(
                                    endpoint.getStatus().getGenerationId() + 1, "same")));
            assertTrue(endpoint.releaseRequest(first));
            assertFalse(endpoint.releaseRequest(first));
            var replacement =
                    endpoint.tryRegisterSelection(
                                    endpoint.selectionSnapshot(), "same", 200, 300, 10, capacity)
                            .value();
            assertNotNull(replacement);
            var before = endpoint.selectionSnapshot();
            assertFalse(endpoint.releaseRequest(first));
            assertEquals(before, endpoint.selectionSnapshot());
            assertEquals(1, before.usage().occupiedRequests());
            assertEquals(200L, before.usage().hardReservedKvTokens());
        } finally {
            EndpointTestSupport.closeAndWait(endpoint);
        }
    }

    @org.junit.jupiter.api.Test
    void workerReportsDetermineConfirmedRequestSlots() {
        var endpoint = endpoint(1000);
        var capacity = new DecodeResources.AdmissionCapacity(2, 100);
        try {
            var local =
                    endpoint.tryRegisterSelection(
                                    endpoint.selectionSnapshot(), "local", 100, 200, 10, capacity)
                            .value();
            assertNotNull(local);
            reportRunning(endpoint, "local", 900);
            assertEquals(1, endpoint.selectionSnapshot().usage().occupiedRequests());
            assertEquals(0L, endpoint.selectionSnapshot().usage().hardReservedKvTokens());
            assertTrue(endpoint.releaseRequest(local));
            assertEquals(1, endpoint.selectionSnapshot().usage().occupiedRequests());
            reportRunning(endpoint, "local", 900);
            assertEquals(
                    1,
                    endpoint.selectionSnapshot().usage().occupiedRequests(),
                    "local cleanup does not change reported capacity");
            updateCapacity(endpoint, 1000);
            reportRunning(endpoint, "unknown", 900);
            assertEquals(
                    1,
                    endpoint.selectionSnapshot().usage().occupiedRequests(),
                    "worker requests without local registrations still consume slots");
        } finally {
            EndpointTestSupport.closeAndWait(endpoint);
        }
    }

    @org.junit.jupiter.api.Test
    void cancellationPlanDoesNotReserveCapacityForIncomingRequest() {
        var endpoint = endpoint(500L);
        var policy = new DecodeResources.AdmissionCapacity(1L, 100L);
        var a =
                endpoint.tryRegisterSelection(
                                endpoint.selectionSnapshot(), "1", 100, 100, 30, policy)
                        .value();
        assertNotNull(a);
        var task = new org.flexlb.worker.status.TaskInfo();
        task.setRequestId("1");
        task.setPhase(org.flexlb.worker.status.TaskPhase.RUNNING);
        task.setInputLength(100L);
        var status = new WorkerStatusResponse();
        status.setTotalKvCacheTokens(1000L);
        status.setAvailableKvCacheTokens(500L);
        status.setRunningTaskInfo(java.util.Map.of("1", task));
        EndpointTestSupport.applyStatus(endpoint, status).run();
        var proposal = plan(endpoint, 100, 100, policy, PreemptionStage.DECODE_ENGINE_OWNED);
        assertEquals(
                List.of("1"),
                proposal.requestsToPreempt().stream().map(v -> v.requestId()).toList());
        assertEquals(
                WorkerEndpoint.RegistrationStatus.CAPACITY_FULL,
                endpoint.tryRegisterSelection(
                                endpoint.selectionSnapshot(), "b", 100, 100, 70, policy)
                        .status());
        assertTrue(endpoint.releaseRequest(a));
        assertEquals(
                WorkerEndpoint.RegistrationStatus.CAPACITY_FULL,
                endpoint.tryRegisterSelection(
                                endpoint.selectionSnapshot(), "c", 100, 100, 60, policy)
                        .status());
        updateCapacity(endpoint, 1000L);
        // The released slot belongs to ordinary admission, regardless of which request requested
        // cancellation.
        assertEquals(
                WorkerEndpoint.RegistrationStatus.REGISTERED,
                endpoint.tryRegisterSelection(
                                endpoint.selectionSnapshot(), "c", 100, 100, 60, policy)
                        .status());
        assertEquals(
                WorkerEndpoint.RegistrationStatus.CAPACITY_FULL,
                endpoint.tryRegisterSelection(
                                endpoint.selectionSnapshot(), "b", 100, 100, 70, policy)
                        .status());
    }

    private static RequestContext bindRequest(
            RequestRepository repository,
            AbstractRequestScheduler scheduler,
            DecodeEndpoint endpoint,
            DecodeEndpoint.RequestRegistration reservation) {
        var context = mock(RequestContext.class, RETURNS_DEEP_STUBS);
        var route = mock(RequestResult.class);
        when(repository.findActive(reservation.requestId())).thenReturn(context);
        when(context.getRequestScheduler()).thenReturn(scheduler);
        when(context.snapshot().route()).thenReturn(route);
        when(route.decodeEndpoint()).thenReturn(endpoint);
        when(route.decodeRegistration()).thenReturn(reservation);
        return context;
    }

    private static void reportRunning(DecodeEndpoint endpoint, String id, long available) {
        var task = new org.flexlb.worker.status.TaskInfo();
        task.setRequestId(id);
        task.setPhase(org.flexlb.worker.status.TaskPhase.RUNNING);
        task.setInputLength(100L);
        var status = new WorkerStatusResponse();
        status.setTotalKvCacheTokens(1000L);
        status.setAvailableKvCacheTokens(available);
        status.setRunningTaskInfo(java.util.Map.of(id, task));
        EndpointTestSupport.applyStatus(endpoint, status).run();
    }

    private static DecodePreemptionPlan plan(
            DecodeEndpoint endpoint,
            long hardKvTokens,
            long expectedKvTokens,
            DecodeResources.AdmissionCapacity policy,
            PreemptionStage stage) {
        PreemptionConfig preemption = new PreemptionConfig();
        preemption.setAllowedPreemptionStages(EnumSet.of(stage));
        var failures = new HashMap<String, String>();
        var request =
                org.flexlb.worker.endpoint.Request.builder()
                        .requestId("99")
                        .seqLen(hardKvTokens)
                        .maxNewTokens((int) (expectedKvTokens - hardKvTokens))
                        .build();
        var snapshot = endpoint.resourceSnapshot();
        DecodePreemptionPlan proposal =
                SchedulingTestConfig.planDecode(request, 70, policy, snapshot, preemption, failures)
                        .proposal();
        assertNotNull(proposal, failures.toString());
        return proposal;
    }

    private static DecodeEndpoint endpoint(long availableKv) {
        DecodeEndpoint endpoint =
                EndpointTestSupport.decode(
                        EndpointTestSupport.workerStatus(RoleType.DECODE, "10.0.0.1", 8080, 8081),
                        org.flexlb.balance.schedule.SchedulerTestSupport.repository(
                                EndpointTestSupport.noopEventSink()));
        updateCapacity(endpoint, availableKv);
        return endpoint;
    }

    private static void updateCapacity(DecodeEndpoint endpoint, long availableKv) {
        WorkerStatusResponse status = new WorkerStatusResponse();
        status.setTotalKvCacheTokens(1000L);
        status.setAvailableKvCacheTokens(availableKv);
        EndpointTestSupport.applyStatus(endpoint, status).run();
    }
}
