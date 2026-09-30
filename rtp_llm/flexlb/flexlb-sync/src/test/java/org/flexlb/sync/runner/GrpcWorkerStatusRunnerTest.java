package org.flexlb.sync.runner;

import com.google.protobuf.UnknownFieldSet;
import org.flexlb.balance.delivery.DeliveryStrategy;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.scheduler.EndpointEventProjector;
import org.flexlb.balance.scheduler.PlacementAvailability;
import org.flexlb.balance.scheduler.ScheduledRequest;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.ConfigService;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.service.grpc.EngineGrpcService;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.status.WorkerDirectory;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.mockito.ArgumentCaptor;

import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.clearInvocations;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

class GrpcWorkerStatusRunnerTest {

    @ParameterizedTest
    @CsvSource({"DECODE", "PREFILL", "PDFUSION"})
    void closedAdmissionPreservesGenerationAndInflightUntilWake(RoleType role)
            throws Exception {
        ConfigService config = mock(ConfigService.class);
        when(config.loadBalanceConfig()).thenReturn(
                org.flexlb.balance.scheduler.SchedulingTestConfig.newConfig());
        PlacementAvailability availability = mock(PlacementAvailability.class);
        EndpointRegistry registry = new EndpointRegistry(config,
                mock(EndpointEventProjector.class), mock(BatchSchedulerReporter.class),
                mock(DeliveryStrategy.class), availability);
        WorkerStatus status = RunnerTestSupport.discovered(
                role, null, "127.0.0.1", 8080, 8081, "test-site");
        WorkerDirectory directory = directory(registry, status);
        WorkerEndpoint endpoint = RunnerTestSupport.publishEndpoint(
                registry, role, status.getIpPort(), status);
        DecodeEndpoint.ReservationHandle decodeReservation = null;
        CacheAwareService cache = mock(CacheAwareService.class);
        try {
            poll(directory, status, role, EngineRpcService.WorkerStatusPB.newBuilder()
                    .setRole(role.getCode()).setAlive(true).setStatusVersion(2L).build(), cache);
            try (var pin = registry.capture(role, status.getIpPort())) {
                assertNotNull(pin, "legacy responses without admission_closed remain routable");
                if (endpoint instanceof DecodeEndpoint decode) {
                    decodeReservation = decode.reserveUnqueued(pin, 123L, 0L, 0L, 0);
                } else if (endpoint instanceof PrefillEndpoint prefill) {
                    ScheduledRequest item = mock(ScheduledRequest.class);
                    when(item.requestId()).thenReturn(123L);
                    try (var reservation = prefill.reserveUnqueuedRoute(pin, item, 1L).reservation();
                         var commit = prefill.tryBeginRouteCommitAdmission();
                         var handoff = commit.commit(List.of(item), List.of(reservation))) {
                        assertNotNull(handoff);
                    }
                }
            }
            // DRAINING, SLEEPING and WAKING all close admission without changing generation.
            for (long version = 3L; version <= 5L; version++) {
                poll(directory, status, role, lifecycleResponse(role, version, true, true), cache);
                assertSame(status, directory.statusSnapshot(role).get(status.getIpPort()));
                assertSame(endpoint, registry.get(role, status.getIpPort()));
                assertTrue(status.isActiveGeneration());
                try (var newPlacement = registry.capture(role, status.getIpPort())) {
                    assertNull(newPlacement, "closed admission must reject new placement");
                }
                if (role == RoleType.DECODE) {
                    assertTrue(registry.decodeRoutingSnapshot().isEmpty());
                    DecodeEndpoint decode = (DecodeEndpoint) endpoint;
                    assertTrue(decode.isAcceptedByEngine(decodeReservation));
                    assertEquals(1, decode.resourceSnapshot().runningCount());
                    assertFalse(decode.isRetired());
                } else {
                    assertTrue(registry.prefillRoutingSnapshot(role).isEmpty());
                    assertEquals(1, ((PrefillEndpoint) endpoint).getIndividuallyTrackedRequestCount());
                }
                try (var continuation = endpoint.tryPinGeneration()) {
                    assertNotNull(continuation, "already-owned work may finish its handoff");
                }
                verifyNoInteractions(cache);
            }
            clearInvocations(availability);
            poll(directory, status, role, lifecycleResponse(role, 6L, true, false), cache);
            verify(availability).capacityChanged(role, null, status.getIpPort());
            assertSame(endpoint, registry.get(role, status.getIpPort()));
            try (var pin = registry.capture(role, status.getIpPort())) {
                assertNotNull(pin, "wake reopens placement on the same generation");
            }
            if (role == RoleType.DECODE) {
                assertEquals(1, registry.decodeRoutingSnapshot().size());
            } else {
                assertEquals(1, registry.prefillRoutingSnapshot(role).size());
            }
            // A real death must still retire the generation, even if admission was closed.
            poll(directory, status, role, lifecycleResponse(role, 7L, false, true), cache);
            assertNull(registry.get(role, status.getIpPort()));
            assertFalse(status.isActiveGeneration());
        } finally {
            endpoint.close();
            endpoint.awaitRetirement();
        }
    }

    @Test
    void decodeDispatchAckAfterAdmissionClosesRetainsOwnership() throws Exception {
        ConfigService config = mock(ConfigService.class);
        when(config.loadBalanceConfig()).thenReturn(
                org.flexlb.balance.scheduler.SchedulingTestConfig.newConfig());
        EndpointRegistry registry = RunnerTestSupport.endpointRegistry(config);
        WorkerStatus status = RunnerTestSupport.discovered(
                RoleType.DECODE, null, "127.0.0.2", 8080, 8081, "test-site");
        WorkerDirectory directory = directory(registry, status);
        DecodeEndpoint endpoint = (DecodeEndpoint) RunnerTestSupport.publishEndpoint(
                registry, RoleType.DECODE, status.getIpPort(), status);
        try {
            DecodeEndpoint.ReservationHandle reservation;
            try (var pin = registry.capture(RoleType.DECODE, status.getIpPort())) {
                reservation = endpoint.reserve(pin, 123L, 0L, 0L, 0);
            }
            var acquired = endpoint.acquireDispatchPermit(
                    reservation, new DecodeEndpoint.AdmissionCapacity(0L, 100L));
            assertNotNull(acquired.permit());
            poll(directory, status, RoleType.DECODE,
                    lifecycleResponse(RoleType.DECODE, 2L, true, true).toBuilder()
                            .clearRunningTaskInfo().build(), mock(CacheAwareService.class));
            assertNull(registry.capture(RoleType.DECODE, status.getIpPort()));
            assertEquals(DecodeEndpoint.EngineDispatchPermitTransferStatus.TRANSFERRED,
                    endpoint.dispatch(acquired.permit(), DecodeEndpoint.DispatchOutcome.ENGINE_OWNED));
            assertNotNull(endpoint.reservationHandle(123L));
            assertFalse(endpoint.isRetired());
        } finally {
            endpoint.close();
            endpoint.awaitRetirement();
        }
    }

    private static EngineRpcService.WorkerStatusPB lifecycleResponse(
            RoleType role, long version, boolean alive, boolean admissionClosed) throws Exception {
        // Construct the wire field directly so this regression runs against the old schema too.
        var wire = EngineRpcService.WorkerStatusPB.newBuilder()
                .setRole(role.getCode())
                .setStatusVersion(version)
                .setAlive(alive)
                .addRunningTaskInfo(EngineRpcService.TaskInfoPB.newBuilder()
                        .setRequestId(123L)
                        .setPhase(EngineRpcService.TaskPhase.TASK_PHASE_RUNNING))
                .setUnknownFields(UnknownFieldSet.newBuilder().addField(23,
                        UnknownFieldSet.Field.newBuilder().addVarint(admissionClosed ? 1L : 0L).build()).build())
                .build().toByteArray();
        return EngineRpcService.WorkerStatusPB.parseFrom(wire);
    }

    private static void poll(WorkerDirectory directory, WorkerStatus status, RoleType role,
                             EngineRpcService.WorkerStatusPB response, CacheAwareService cache) {
        EngineGrpcService grpc = mock(EngineGrpcService.class);
        when(grpc.getWorkerStatusAsync(anyString(), anyInt(), anyLong(), anyLong(), any()))
                .thenReturn(CompletableFuture.completedFuture(response));
        var lease = status.tryBeginStatusPoll();
        assertNotNull(lease, "non-serving workers continue status polling");
        new GrpcWorkerStatusRunner(
                "test-model", status.getIpPort(), "test-site", role, null,
                status, lease, directory, mock(EngineHealthReporter.class),
                grpc, 5_000L, cache, Runnable::run).run();
    }

    @ParameterizedTest
    @CsvSource({"DECODE,false", "PREFILL,true", "PDFUSION,true"})
    void repeatedRpcFailuresRetireWorkerAndOnlyClearDetailedCacheIndexes(
            RoleType role, boolean needsCacheKeys) {
        ConfigService config = mock(ConfigService.class);
        when(config.loadBalanceConfig()).thenReturn(
                org.flexlb.balance.scheduler.SchedulingTestConfig.newConfig());
        EndpointRegistry registry = RunnerTestSupport.endpointRegistry(config);
        WorkerStatus status = RunnerTestSupport.discovered(
                role, null, "127.0.0.1", 8080, 8081, "test-site");
        WorkerDirectory directory = directory(registry, status);
        WorkerEndpoint endpoint = RunnerTestSupport.publishEndpoint(
                registry, role, status.getIpPort(), status);
        CacheAwareService cache = mock(CacheAwareService.class);
        EngineGrpcService grpc = mock(EngineGrpcService.class);
        when(grpc.getWorkerStatusAsync(
                anyString(), anyInt(), anyLong(), anyLong(), any()))
                .thenReturn(CompletableFuture.failedFuture(
                        io.grpc.Status.UNAVAILABLE.asRuntimeException()));

        for (int failure = 1; failure <= 3; failure++) {
            WorkerStatus.PollLease lease = status.tryBeginStatusPoll();
            assertNotNull(lease);
            new GrpcWorkerStatusRunner(
                    "test-model", status.getIpPort(), "test-site", role, null,
                    status, lease, directory, mock(EngineHealthReporter.class),
                    grpc, 5_000L, cache, Runnable::run).run();
            if (failure < 3) {
                assertSame(status, directory.statusSnapshot(role).get(status.getIpPort()));
                assertSame(endpoint, registry.get(role, status.getIpPort()));
                verifyNoInteractions(cache);
            }
        }

        assertNull(registry.get(role, status.getIpPort()));
        assertTrue(directory.statusSnapshot(role).isEmpty());
        assertNull(status.tryBeginStatusPoll(), "retired generation cannot poll again");
        if (needsCacheKeys) {
            verify(cache).removeEngineBlockCache(status.getIpPort());
        } else {
            verifyNoInteractions(cache);
        }
    }

    @Test
    void newGenerationProjectionRunsOutsideWorkerStatusLock() {
        String ipPort = "127.0.0.1:8080";
        WorkerStatus status = RunnerTestSupport.discovered(
                RoleType.DECODE, null, "127.0.0.1",
                8080, 8081, "test-site");
        WorkerStatus.PollLease pollLease = status.tryBeginStatusPoll();
        assertNotNull(pollLease);

        AtomicBoolean publicationHeldLock = new AtomicBoolean();
        AtomicBoolean projectionReleasedLock = new AtomicBoolean();
        WorkerEndpoint endpoint = mock(WorkerEndpoint.class);
        EndpointRegistry registry = mock(EndpointRegistry.class);
        WorkerDirectory directory = directory(registry, status);
        when(registry.publishPreparedEndpoint(
                org.mockito.Mockito.eq(ipPort),
                org.mockito.Mockito.eq(status),
                any(WorkerStatus.PreparedStatus.class)))
                .thenAnswer(invocation -> {
                    publicationHeldLock.set(
                            status.lock.isHeldByCurrentThread());
                    status.publishPreparedStatus(invocation.getArgument(2));
                    return new EndpointRegistry.EndpointPublication(
                            endpoint,
                            () -> projectionReleasedLock.set(
                                    !status.lock.isHeldByCurrentThread()));
                });

        EngineRpcService.WorkerStatusPB response =
                EngineRpcService.WorkerStatusPB.newBuilder()
                        .setRole(RoleType.DECODE.getCode())
                        .setRoleType(
                                EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE)
                        .setStatusVersion(1L)
                        .setAlive(true)
                        .build();
        EngineGrpcService grpc = mock(EngineGrpcService.class);
        when(grpc.getWorkerStatusAsync(
                anyString(), anyInt(), anyLong(), anyLong(), any()))
                .thenReturn(CompletableFuture.completedFuture(response));

        new GrpcWorkerStatusRunner(
                "test-model", ipPort, "test-site", RoleType.DECODE, null,
                status, pollLease, directory,
                mock(EngineHealthReporter.class), grpc, 5_000L,
                mock(CacheAwareService.class), Runnable::run)
                .run();

        assertTrue(publicationHeldLock.get(),
                "publication must commit under the generation lock");
        assertTrue(projectionReleasedLock.get(),
                "endpoint facts must project after releasing that lock");
    }

    @Test
    void sameVersionResponseProjectsExactEndpointActivity() {
        String ipPort = "127.0.0.1:8080";
        WorkerStatus status = RunnerTestSupport.alive(
                RoleType.DECODE, null, "127.0.0.1",
                8080, 8081, "test-site");
        WorkerStatus.PollLease pollLease = status.tryBeginStatusPoll();
        assertNotNull(pollLease);

        WorkerEndpoint endpoint = mock(WorkerEndpoint.class);
        EndpointRegistry registry = mock(EndpointRegistry.class);
        WorkerDirectory directory = directory(registry, status);
        when(registry.get(RoleType.DECODE, ipPort, status))
                .thenReturn(endpoint);
        java.util.concurrent.atomic.AtomicBoolean projected =
                new java.util.concurrent.atomic.AtomicBoolean();
        Runnable activity = () -> projected.set(true);
        when(endpoint.observeStatusHeartbeat(any(), any()))
                .thenReturn(activity);

        EngineRpcService.TaskInfoPB task = EngineRpcService.TaskInfoPB.newBuilder()
                .setRequestId(123L)
                .setPhase(EngineRpcService.TaskPhase.TASK_PHASE_RUNNING)
                .build();
        EngineRpcService.WorkerStatusPB response =
                EngineRpcService.WorkerStatusPB.newBuilder()
                        .setRole(RoleType.DECODE.getCode())
                        .setRoleType(EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE)
                        .setStatusVersion(
                                status.appliedStatusCursor().statusVersion())
                        .setAlive(true)
                        .addRunningTaskInfo(task)
                        .build();
        EngineGrpcService grpc = mock(EngineGrpcService.class);
        when(grpc.getWorkerStatusAsync(
                anyString(), anyInt(), anyLong(), anyLong(), any()))
                .thenReturn(CompletableFuture.completedFuture(response));
        new GrpcWorkerStatusRunner(
                "test-model", ipPort, "test-site", RoleType.DECODE, null,
                status, pollLease, directory,
                mock(EngineHealthReporter.class), grpc, 5_000L,
                mock(CacheAwareService.class), Runnable::run)
                .run();

        ArgumentCaptor<WorkerStatus.StatusObservation> observation =
                ArgumentCaptor.forClass(WorkerStatus.StatusObservation.class);
        verify(endpoint).observeStatusHeartbeat(
                org.mockito.Mockito.eq(status), observation.capture());
        assertTrue(observation.getValue().runningTasks().values().stream()
                .anyMatch(active -> active.requestId() == 123L));
        assertTrue(projected.get());
        WorkerStatus.PollLease nextPoll = status.tryBeginStatusPoll();
        assertNotNull(nextPoll, "the asynchronous owner must close the exact poll lease");
        nextPoll.close();
    }

    private static WorkerDirectory directory(
            EndpointRegistry registry, WorkerStatus status) {
        WorkerDirectory directory = new WorkerDirectory(registry);
        directory.currentOrDiscover(
                status.getRole(), status.getIpPort(), () -> status);
        return directory;
    }
}
