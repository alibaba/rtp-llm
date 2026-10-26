package org.flexlb.balance.endpoint;

import org.flexlb.balance.estimate.NonBatchTtftEstimator;
import org.flexlb.balance.scheduler.EndpointEventHandler;
import org.flexlb.balance.scheduler.PlacementAvailability;
import org.flexlb.balance.scheduler.QueueCapacity;
import org.flexlb.balance.schedulev2.prefill.PrefillQueueRegistry;
import org.flexlb.config.ConfigService;
import org.flexlb.config.DispatcherConfig;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import org.mockito.Mockito;

import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class EndpointRegistryRoleTest {

    private EndpointRegistry registry;

    @BeforeEach
    void setUp() {
        ConfigService configService = Mockito.mock(ConfigService.class);
        Mockito.when(configService.loadBalanceConfig()).thenReturn(org.flexlb.balance.scheduler.SchedulingTestConfig.newConfig());
        EndpointTestSupport.TestRequestRuntime requestRuntime =
                EndpointTestSupport.requestRuntime();
        registry = new EndpointRegistry(
                configService,
                requestRuntime.events(),
                Mockito.mock(BatchSchedulerReporter.class),
                EndpointTestSupport.routeStrategy(requestRuntime),
                new NonBatchTtftEstimator(),
                new PlacementAvailability(),
                new QueueCapacity(), new PrefillQueueRegistry());
    }

    @AfterEach
    void tearDown() {
        registry.close();
    }

    @ParameterizedTest
    @EnumSource(DispatcherConfig.Type.class)
    void endpointOwnsInitializedQueueBeforeLoopStarts(DispatcherConfig.Type dispatcherType) {
        ConfigService service = Mockito.mock(ConfigService.class);
        var config = org.flexlb.balance.scheduler.SchedulingTestConfig.newConfig();
        config.getDispatcher().setType(dispatcherType);
        Mockito.when(service.loadBalanceConfig()).thenReturn(config);
        var queues = Mockito.spy(new PrefillQueueRegistry());
        var runtime = EndpointTestSupport.requestRuntime();
        var availability = Mockito.mock(PlacementAvailability.class);
        var events = new EndpointEventHandler(runtime.requests());
        var localRegistry = new EndpointRegistry(service, events, Mockito.mock(BatchSchedulerReporter.class),
                EndpointTestSupport.routeStrategy(runtime), new NonBatchTtftEstimator(), availability,
                new QueueCapacity(), queues);
        String address = "127.0.0.1:8001";
        Mockito.doAnswer(call -> {
            PrefillEndpoint endpoint = call.getArgument(0);
            assertSame(endpoint, localRegistry.get(RoleType.PREFILL, address));
            assertNotNull(endpoint.prefillQueue());
            assertSame(endpoint.prefillQueue(), queues.find(endpoint));
            assertEquals(1L, endpoint.prefillQueue().inflightCount());
            call.callRealMethod();
            return null;
        }).when(queues).start(Mockito.any(PrefillEndpoint.class));
        Mockito.doAnswer(call -> {
            var endpoint = (PrefillEndpoint) localRegistry.get(RoleType.PREFILL, address);
            assertNotNull(queues.find(endpoint));
            return null;
        }).when(availability).topologyChanged(Mockito.eq(RoleType.PREFILL), Mockito.any(), Mockito.eq(address));
        try {
            TaskInfo task = new TaskInfo();
            task.setRequestId("1");
            task.setPhase(org.flexlb.enums.TaskPhase.RUNNING);
            WorkerStatusResponse firstStatus = new WorkerStatusResponse();
            firstStatus.setRunningTaskInfo(Map.of("1", task));
            var endpoint = (PrefillEndpoint) EndpointTestSupport.publishEndpoint(localRegistry,
                    RoleType.PREFILL, address, status(RoleType.PREFILL, 8001), firstStatus);
            assertEquals(0, queues.find(endpoint).waitingCount());
            Mockito.verify(queues).start(endpoint);

            var originalQueue = endpoint.prefillQueue();
            WorkerStatusResponse nextStatus = new WorkerStatusResponse();
            nextStatus.setRunningTaskInfo(Map.of());
            EndpointTestSupport.applyStatus(endpoint, nextStatus).run();
            assertSame(originalQueue, endpoint.prefillQueue());
            assertEquals(1L, originalQueue.inflightCount());
            originalQueue.updateFromEngine();
            assertEquals(0L, originalQueue.inflightCount());

            endpoint.close();
            assertNull(queues.find(endpoint));
        } finally {
            localRegistry.close();
            queues.close();
        }
    }

    @Test
    void should_register_and_resolve_supported_roles_independently() {
        WorkerEndpoint prefill = EndpointTestSupport.publishEndpoint(registry,
                RoleType.PREFILL, "127.0.0.1:8001", status(RoleType.PREFILL, 8001));
        WorkerEndpoint decode = EndpointTestSupport.publishEndpoint(registry,
                RoleType.DECODE, "127.0.0.1:8002", status(RoleType.DECODE, 8002));
        WorkerEndpoint pdFusion = EndpointTestSupport.publishEndpoint(registry,
                RoleType.PDFUSION, "127.0.0.1:8003", status(RoleType.PDFUSION, 8003));
        WorkerEndpoint vit = EndpointTestSupport.publishEndpoint(registry,
                RoleType.VIT, "127.0.0.1:8004", status(RoleType.VIT, 8004));

        assertInstanceOf(PrefillEndpoint.class, prefill);
        assertInstanceOf(DecodeEndpoint.class, decode);
        assertInstanceOf(PrefillEndpoint.class, pdFusion);
        assertEquals(WorkerEndpoint.class, vit.getClass());
        assertSame(prefill, registry.get(RoleType.PREFILL, "127.0.0.1:8001"));
        assertSame(decode, registry.get(RoleType.DECODE, "127.0.0.1:8002"));
        assertSame(pdFusion, registry.get(RoleType.PDFUSION, "127.0.0.1:8003"));
        assertSame(vit, registry.get(RoleType.VIT, "127.0.0.1:8004"));
        EndpointRegistry.PrefillRoutingEntry pdFusionRoute =
                registry.prefillRoutingSnapshot(RoleType.PDFUSION).getFirst();
        assertEquals("127.0.0.1:8003", pdFusionRoute.address());
        assertSame(pdFusion, pdFusionRoute.endpoint());

        for (RoleType roleType : new RoleType[]{
                RoleType.PREFILL, RoleType.DECODE, RoleType.PDFUSION, RoleType.VIT}) {
            assertEquals(1, registry.getEndpointCount(roleType));
        }
    }

    @Test
    void should_not_mix_roles_when_ip_port_is_shared() {
        String ipPort = "127.0.0.1:8080";
        WorkerEndpoint prefill = EndpointTestSupport.publishEndpoint(registry,
                RoleType.PREFILL, ipPort, status(RoleType.PREFILL, 8080));
        WorkerEndpoint decode = EndpointTestSupport.publishEndpoint(registry,
                RoleType.DECODE, ipPort, status(RoleType.DECODE, 8080));

        assertNotSame(prefill, decode);
        assertSame(prefill, registry.get(RoleType.PREFILL, ipPort));
        assertSame(decode, registry.get(RoleType.DECODE, ipPort));
    }

    @Test
    void should_register_multi_dp_prefill_roles() {
        for (RoleType role : List.of(RoleType.PREFILL, RoleType.PDFUSION)) {
            int port = role == RoleType.PREFILL ? 8090 : 8092;
            String address = "127.0.0.1:" + port;
            WorkerStatusResponse response = new WorkerStatusResponse();
            response.setDpSize(2);
            response.setDpRank(1);

            WorkerEndpoint endpoint = EndpointTestSupport.publishEndpoint(
                    registry, role, address, status(role, port), response);

            assertInstanceOf(PrefillEndpoint.class, endpoint);
            assertSame(endpoint, registry.get(role, address));
            assertEquals(1L, endpoint.getStatus().getDpRank());
        }
    }

    @Test
    void decode_routing_view_refreshes_discovery_topology_without_status_commit() {
        String address = "decode-route-alias";
        WorkerStatus status = EndpointTestSupport.workerStatus(
                RoleType.DECODE,
                "group-a",
                "127.0.0.1",
                8010,
                8011,
                "site-a");
        EndpointTestSupport.publishEndpoint(
                registry, RoleType.DECODE, address, status);

        DecodeEndpoint.DecodeRoutingView before =
                registry.decodeRoutingSnapshot().getFirst();
        assertEquals(address, before.address());
        assertEquals("group-a", before.topology().group());
        assertEquals("site-a", before.topology().site());

        status.lock.lock();
        try {
            status.updateDiscoveryLabels("site-b", "group-b");
        } finally {
            status.lock.unlock();
        }

        DecodeEndpoint.DecodeRoutingView after =
                registry.decodeRoutingSnapshot().getFirst();
        assertNotSame(before, after,
                "topology is an independent routing-cache generation");
        assertSame(before.workerStatus(), after.workerStatus());
        assertEquals(before.admissionVersion(), after.admissionVersion());
        assertEquals(address, after.address());
        assertEquals("group-b", after.topology().group());
        assertEquals("site-b", after.topology().site());
    }

    @Test
    void decode_routing_view_captures_only_its_exact_same_address_generation() {
        String address = "decode-replacement-alias";
        WorkerStatus oldStatus = status(RoleType.DECODE, 8020);
        EndpointTestSupport.publishEndpoint(
                registry, RoleType.DECODE, address, oldStatus);
        DecodeEndpoint.DecodeRoutingView oldView =
                registry.decodeRoutingSnapshot().getFirst();

        retire(RoleType.DECODE, address, oldStatus);
        WorkerStatus replacementStatus = status(RoleType.DECODE, 8020);
        DecodeEndpoint replacement = (DecodeEndpoint)
                EndpointTestSupport.publishEndpoint(
                        registry,
                        RoleType.DECODE,
                        address,
                        replacementStatus);

        assertNull(registry.holdDecodeEndpoint(oldView), "an old view must not hold a same-address replacement");
        DecodeEndpoint.DecodeRoutingView replacementView =
                registry.decodeRoutingSnapshot().getFirst();
        assertEquals(address, replacementView.address());
        assertTrue(oldView.generationId() != replacementView.generationId());
        try (WorkerEndpoint.EndpointHold hold = registry.holdDecodeEndpoint(replacementView)) {
            assertNotNull(hold);
            assertSame(replacement, hold.endpoint());
        }
    }

    @Test
    void prefill_routing_directory_tracks_exact_advisory_generations() {
        String firstAddress = "127.0.0.1:8001";
        String secondAddress = "127.0.0.1:8002";
        String thirdAddress = "127.0.0.1:8003";
        WorkerStatus firstStatus = status(RoleType.PREFILL, 8001);
        WorkerStatus secondStatus = status(RoleType.PREFILL, 8002);
        WorkerEndpoint first = EndpointTestSupport.publishEndpoint(registry,
                RoleType.PREFILL, firstAddress, firstStatus);
        EndpointTestSupport.publishEndpoint(registry,
                RoleType.PREFILL, secondAddress, secondStatus);
        EndpointTestSupport.publishEndpoint(registry,
                RoleType.PREFILL,
                thirdAddress,
                status(RoleType.PREFILL, 8003));

        List<EndpointRegistry.PrefillRoutingEntry> originalDirectory =
                registry.prefillRoutingSnapshot(RoleType.PREFILL);
        assertEquals(
                List.of(firstAddress, secondAddress, thirdAddress),
                originalDirectory.stream()
                        .map(EndpointRegistry.PrefillRoutingEntry::address)
                        .toList());
        assertSame(first, originalDirectory.getFirst().endpoint());
        assertThrows(UnsupportedOperationException.class,
                () -> originalDirectory.add(
                        new EndpointRegistry.PrefillRoutingEntry(
                                "127.0.0.1:8004",
                                (PrefillEndpoint) first)));

        retire(RoleType.PREFILL, firstAddress, firstStatus);
        WorkerEndpoint replacement = EndpointTestSupport.publishEndpoint(registry,
                RoleType.PREFILL,
                firstAddress,
                status(RoleType.PREFILL, 8001));
        assertNotSame(first, replacement);
        List<EndpointRegistry.PrefillRoutingEntry> replacementDirectory =
                registry.prefillRoutingSnapshot(RoleType.PREFILL);
        assertEquals(
                originalDirectory.stream()
                        .map(EndpointRegistry.PrefillRoutingEntry::address)
                        .collect(java.util.stream.Collectors.toSet()),
                replacementDirectory.stream()
                        .map(EndpointRegistry.PrefillRoutingEntry::address)
                        .collect(java.util.stream.Collectors.toSet()),
                "same-address replacement must retain the address membership");
        assertSame(first, originalDirectory.getFirst().endpoint(),
                "an in-flight traversal keeps its advisory old generation");
        assertSame(replacement, replacementDirectory.stream()
                .filter(entry -> entry.address().equals(firstAddress))
                .findFirst()
                .orElseThrow()
                .endpoint());
        WorkerEndpoint.EndpointHold replacementHold =
                registry.hold(RoleType.PREFILL, originalDirectory.getFirst().address());
        assertNotNull(replacementHold);
        try (replacementHold) {
            assertSame(replacement, replacementHold.endpoint());
            assertNotSame(originalDirectory.getFirst().endpoint(), replacementHold.endpoint(),
                    "winner identity validation must reject the stale route");
        }

        EndpointRegistry.DetachedGeneration detached =
                detachUnderGenerationLock(
                        RoleType.PREFILL, secondAddress, secondStatus);
        assertNotNull(detached);
        detached.retireAndAwait();
        assertEquals(Set.of(firstAddress, thirdAddress),
                registry.prefillRoutingSnapshot(RoleType.PREFILL).stream()
                        .map(EndpointRegistry.PrefillRoutingEntry::address)
                        .collect(java.util.stream.Collectors.toSet()));

        registry.close();
        assertTrue(registry.prefillRoutingSnapshot(RoleType.PREFILL).isEmpty());
    }

    @Test
    void simple_endpoint_should_report_status_task_count_as_load() {
        WorkerStatus status = status(RoleType.VIT, 8080);
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setAlive(true);
        response.setRunningTaskInfo(
                Map.of("1", new TaskInfo(), "2", new TaskInfo()));
        WorkerEndpoint endpoint = EndpointTestSupport.publishEndpoint(
                registry, RoleType.VIT, "127.0.0.1:8080", status, response);

        assertEquals(2L, endpoint.getLoadMetric().orElseThrow());
    }

    @Test
    void should_not_remove_new_generation_with_expired_status() {
        String ipPort = "127.0.0.1:8080";
        WorkerStatus expired = status(RoleType.VIT, 8080);
        WorkerEndpoint oldEndpoint = EndpointTestSupport.publishEndpoint(registry, RoleType.VIT, ipPort, expired);

        retire(RoleType.VIT, ipPort, expired);
        WorkerStatus replacement = status(RoleType.VIT, 8080);
        WorkerEndpoint newEndpoint = EndpointTestSupport.publishEndpoint(registry, RoleType.VIT, ipPort, replacement);

        assertNotSame(oldEndpoint, newEndpoint);
        assertNull(oldEndpoint.tryHold());
        assertSame(newEndpoint, registry.get(RoleType.VIT, ipPort));

        EndpointRegistry.DetachedGeneration detached =
                detachUnderGenerationLock(
                        RoleType.VIT, ipPort, replacement);
        assertNotNull(detached);
        detached.retireAndAwait();
        assertNull(registry.get(RoleType.VIT, ipPort));
    }

    @Test
    void sameAddressReplacementCannotBindOrReleaseOldDecodeReservation() {
        String ipPort = "127.0.0.1:8080";
        DecodeEndpoint oldEndpoint = (DecodeEndpoint)
                EndpointTestSupport.publishEndpoint(registry,
                        RoleType.DECODE, ipPort,
                        status(RoleType.DECODE, 8080));
        DecodeEndpoint.ReservationHandle oldReservation;
        try (WorkerEndpoint.EndpointHold hold = oldEndpoint.tryHold()) {
            assertTrue(hold != null);
            oldReservation = oldEndpoint.reserve(hold, "41", 100L, 110L, 50);
        }

        retire(RoleType.DECODE, ipPort, oldEndpoint.getStatus());
        DecodeEndpoint replacement = (DecodeEndpoint)
                EndpointTestSupport.publishEndpoint(registry,
                        RoleType.DECODE, ipPort,
                        status(RoleType.DECODE, 8080));

        assertNotSame(oldEndpoint, replacement);
        assertTrue(oldEndpoint.isRetired());
        assertNull(oldEndpoint.reservationHandle(oldReservation.requestId()),
                "close must retire A's queued ownership before B is routable");
        assertNull(replacement.reservationHandle(oldReservation.requestId()));
        replacement.release(oldReservation, DecodeEndpoint.ReleaseReason.LOCAL_ROLLBACK);
        assertTrue(replacement.resourceSnapshot().reserved().isEmpty(),
                "A's exact generation handle must never mutate same-address B");
    }

    private static WorkerStatus status(RoleType roleType, int port) {
        return EndpointTestSupport.workerStatus(
                roleType, "127.0.0.1", port, port + 1);
    }

    private EndpointRegistry.DetachedGeneration detachUnderGenerationLock(
            RoleType role,
            String address,
            WorkerStatus expectedStatus) {
        expectedStatus.lock.lock();
        try {
            return registry.detachAndBeginRetirement(
                    role, address, expectedStatus);
        } finally {
            expectedStatus.lock.unlock();
        }
    }

    private void retire(
            RoleType role,
            String address,
            WorkerStatus status) {
        EndpointRegistry.DetachedGeneration detached =
                detachUnderGenerationLock(role, address, status);
        assertNotNull(detached);
        detached.retireAndAwait();
    }
}
