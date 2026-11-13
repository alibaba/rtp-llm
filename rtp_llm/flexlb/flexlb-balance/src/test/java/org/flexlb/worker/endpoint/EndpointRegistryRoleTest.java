package org.flexlb.worker.endpoint;

import org.flexlb.config.service.RoleType;
import org.flexlb.worker.status.TaskInfo;
import org.flexlb.worker.status.WorkerStatus;
import org.flexlb.worker.status.WorkerStatusResponse;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.flexlb.worker.endpoint.EndpointTestSupport.decodeRoutingSnapshot;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;

/**
 * Checks role-specific discovery and replacement while keeping routing views and resource cleanup tied to the exact worker generation.
 */
class EndpointRegistryRoleTest {
    private EndpointRegistry registry;
    private CapacityNotifier availability;

    @org.junit.jupiter.api.BeforeEach
    void setUp() {
        var config = mock(org.flexlb.config.ConfigService.class);
        Mockito.when(config.loadBalanceConfig()).thenReturn(new org.flexlb.config.FlexlbConfig());
        availability = mock(CapacityNotifier.class);
        registry =
                new EndpointRegistry(
                        config,
                        new org.flexlb.balance.schedule.RequestRepository(),
                        mock(org.flexlb.balance.metrics.DeliveryMetricsReporter.class),
                        availability,
                        org.mockito.Mockito.mock(org.flexlb.cache.match.CacheAwareService.class));
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.EnumSource(
            value = RoleType.class,
            names = {"PREFILL", "PDFUSION", "DECODE", "ENCODER", "VIT"})
    void endpointQueryFiltersGroupsAndRetainsAnImmutableDirectory(RoleType role) {
        var first = WorkerStatus.createDiscovered(role, "a", "127.0.0.1", 8400, 8401, null);
        var second = WorkerStatus.createDiscovered(role, "b", "127.0.0.1", 8500, 8501, null);
        var a =
                EndpointTestSupport.publishEndpoint(
                        registry, role, first.getLogicalIpPort(), first);
        var b =
                EndpointTestSupport.publishEndpoint(
                        registry, role, second.getLogicalIpPort(), second);
        assertEquals(List.of(a, b), registry.endpoints(role, null));
        assertEquals(List.of(a), registry.endpoints(role, "a"));
        assertEquals(List.of(b), registry.endpoints(role, "b"));
        assertTrue(registry.endpoints(role, "missing").isEmpty());
        assertThrows(
                UnsupportedOperationException.class, () -> registry.endpoints(role, "a").clear());
        assertSame(a, registry.getEndpoint(role, first.getLogicalIpPort()));
    }

    @Test
    void vitSelectionSkipsRetiringEndpointsBeforeTheirRegistryEntriesAreRemoved() {
        var worker = status(RoleType.VIT, 8001);
        var address = worker.getLogicalIpPort();
        var endpoint = EndpointTestSupport.publishEndpoint(registry, RoleType.VIT, address, worker);
        assertEquals(
                List.of(address),
                registry.endpoints(RoleType.VIT, null).stream()
                        .map(org.flexlb.worker.endpoint.WorkerEndpoint::ipPort)
                        .toList());
        endpoint.stopAcceptingRequests();
        assertSame(endpoint, registry.getEndpoint(RoleType.VIT, address));
        assertEquals(List.of(endpoint), registry.endpoints(RoleType.VIT, null));
        assertFalse(registry.getEndpoint(RoleType.VIT, address).acceptsNewRequests());
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.EnumSource(
            value = RoleType.class,
            names = {"PREFILL", "PDFUSION", "DECODE"})
    void routingDirectoriesStayOrderedAcrossInsertionRetirementAndReplacement(RoleType role) {
        var statuses = new java.util.HashMap<Integer, WorkerStatus>();
        for (int port : new int[] {8003, 8001, 8002}) {
            var worker = status(role, port);
            statuses.put(port, worker);
            EndpointTestSupport.publishEndpoint(registry, role, worker.getLogicalIpPort(), worker);
        }
        java.util.function.Supplier<List<String>> addresses =
                () ->
                        role == RoleType.DECODE
                                ? decodeRoutingSnapshot(registry, null).stream()
                                        .map(DecodeEndpoint.SelectionSnapshot::address)
                                        .toList()
                                : registry
                                        .<org.flexlb.worker.endpoint.PrefillEndpoint>endpoints(
                                                role, null)
                                        .stream()
                                        .map(PrefillEndpoint::ipPort)
                                        .toList();
        var original = addresses.get();
        assertEquals(List.of("127.0.0.1:8001@0", "127.0.0.1:8002@0", "127.0.0.1:8003@0"), original);
        closeWorker(role, "127.0.0.1:8002@0", statuses.get(8002));
        assertEquals(List.of("127.0.0.1:8001@0", "127.0.0.1:8003@0"), addresses.get());
        var replacement = status(role, 8002);
        EndpointTestSupport.publishEndpoint(
                registry, role, replacement.getLogicalIpPort(), replacement);
        assertEquals(original, addresses.get());
        assertEquals(3, original.size(), "an earlier directory remains an immutable snapshot");
    }

    @AfterEach
    void tearDown() {
        registry.close();
    }

    @Test
    void orphanSweepAlsoCleansEncoderRegistrationsAndHonorsRetentionAndTtl() {
        var endpoint =
                (EncoderEndpoint)
                        EndpointTestSupport.publishEndpoint(
                                registry,
                                RoleType.ENCODER,
                                "127.0.0.1:8001@0",
                                status(RoleType.ENCODER, 8001));
        var route = mock(org.flexlb.balance.schedule.RequestResult.class);
        var context = mock(org.flexlb.balance.schedule.RequestContext.class);
        Mockito.when(route.requestId()).thenReturn("encoder-orphan");
        Mockito.when(route.ctx()).thenReturn(context);
        Mockito.when(context.getRequestScheduler())
                .thenReturn(mock(org.flexlb.balance.schedule.AbstractRequestScheduler.class));
        assertTrue(EndpointTestSupport.commitEncoder(endpoint, route, 1, 20L));

        registry.evictExpiredOrphans(Long.MAX_VALUE, ignored -> false);
        assertEquals(1, endpoint.selectionSnapshot().requestCount());
        registry.evictExpiredOrphans(-1L, id -> id.equals("encoder-orphan"));
        assertEquals(1, endpoint.selectionSnapshot().requestCount());
        registry.evictExpiredOrphans(-1L, ignored -> false);
        assertEquals(0, endpoint.selectionSnapshot().requestCount());
        assertEquals(0, endpoint.selectionSnapshot().uncachedTokens());
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.EnumSource(
            value = RoleType.class,
            names = {"PREFILL", "PDFUSION"})
    void orphanSweepUsesSchedulerRetentionForEveryPrefillRole(RoleType role) {
        PrefillEndpoint endpoint =
                (PrefillEndpoint)
                        EndpointTestSupport.publishEndpoint(
                                registry, role, "127.0.0.1:8001@0", status(role, 8001));
        EndpointTestSupport.commitUnqueued(endpoint, 91L, 20L);
        registry.evictExpiredOrphans(0, id -> id.equals("91"));
        assertEquals(1, endpoint.stats().locallyOwnedRequests());
        registry.evictExpiredOrphans(0, ignored -> false);
        assertEquals(0, endpoint.stats().locallyOwnedRequests());
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.EnumSource(
            value = RoleType.class,
            names = {"PREFILL", "PDFUSION", "DECODE", "VIT"})
    void notificationFailureWithdrawsPublishedGenerationAndClosesItsResources(RoleType role) {
        WorkerStatus status = status(role, 8091);
        String address = status.getLogicalIpPort();
        IllegalStateException failure =
                new IllegalStateException("publication notification failed");
        java.util.concurrent.atomic.AtomicReference<WorkerEndpoint> published =
                new java.util.concurrent.atomic.AtomicReference<>();
        Mockito.doAnswer(
                        invocation -> {
                            if (published.get() != null) return null;
                            WorkerEndpoint candidate = registry.getEndpoint(role, address);
                            assertNotNull(candidate, "notification must occur after the map write");
                            assertEquals(1L, status.appliedStatusCursor().statusVersion());
                            published.set(candidate);
                            throw failure;
                        })
                .when(availability)
                .notifyChanged(Mockito.argThat(endpoint -> endpoint.ipPort().equals(address)));

        assertSame(
                failure,
                assertThrows(
                        IllegalStateException.class,
                        () ->
                                EndpointTestSupport.publishEndpoint(
                                        registry, role, address, status)));

        assertNull(registry.getEndpoint(role, address));
        assertNull(registry.getEndpoint(role, address));
        assertTrue(
                registry.endpoints(role, null).stream()
                        .map(org.flexlb.worker.endpoint.WorkerEndpoint::ipPort)
                        .toList()
                        .isEmpty());
        assertNotNull(published.get());
        org.flexlb.worker.endpoint.EndpointTestSupport.awaitClosed(published.get());
        assertTrue(published.get().hasStoppedAcceptingRequests());
        registry.close();
    }

    @Test
    void should_register_and_resolve_supported_roles_independently() {
        WorkerEndpoint prefill =
                EndpointTestSupport.publishEndpoint(
                        registry,
                        RoleType.PREFILL,
                        "127.0.0.1:8001@0",
                        status(RoleType.PREFILL, 8001));
        WorkerEndpoint decode =
                EndpointTestSupport.publishEndpoint(
                        registry,
                        RoleType.DECODE,
                        "127.0.0.1:8002@0",
                        status(RoleType.DECODE, 8002));
        WorkerEndpoint pdFusion =
                EndpointTestSupport.publishEndpoint(
                        registry,
                        RoleType.PDFUSION,
                        "127.0.0.1:8003@0",
                        status(RoleType.PDFUSION, 8003));
        WorkerEndpoint vit =
                EndpointTestSupport.publishEndpoint(
                        registry, RoleType.VIT, "127.0.0.1:8004@0", status(RoleType.VIT, 8004));

        assertInstanceOf(PrefillEndpoint.class, prefill);
        assertInstanceOf(DecodeEndpoint.class, decode);
        assertInstanceOf(PrefillEndpoint.class, pdFusion);
        assertEquals(WorkerEndpoint.class, vit.getClass());
        assertSame(prefill, registry.getEndpoint(RoleType.PREFILL, "127.0.0.1:8001@0"));
        assertSame(decode, registry.getEndpoint(RoleType.DECODE, "127.0.0.1:8002@0"));
        assertSame(pdFusion, registry.getEndpoint(RoleType.PDFUSION, "127.0.0.1:8003@0"));
        assertSame(vit, registry.getEndpoint(RoleType.VIT, "127.0.0.1:8004@0"));
        PrefillEndpoint pdFusionRoute =
                registry.<org.flexlb.worker.endpoint.PrefillEndpoint>endpoints(
                                RoleType.PDFUSION, null)
                        .getFirst();
        assertEquals("127.0.0.1:8003@0", pdFusionRoute.ipPort());
        assertSame(pdFusion, pdFusionRoute);

        for (RoleType roleType :
                new RoleType[] {
                    RoleType.PREFILL, RoleType.DECODE, RoleType.PDFUSION, RoleType.VIT
                }) {
            assertEquals(1, registry.counts(roleType).published());
        }
    }

    @Test
    void should_not_mix_roles_when_ip_port_is_shared() {
        String ipPort = "127.0.0.1:8080@0";
        WorkerEndpoint prefill =
                EndpointTestSupport.publishEndpoint(
                        registry, RoleType.PREFILL, ipPort, status(RoleType.PREFILL, 8080));
        WorkerEndpoint decode =
                EndpointTestSupport.publishEndpoint(
                        registry, RoleType.DECODE, ipPort, status(RoleType.DECODE, 8080));

        assertNotSame(prefill, decode);
        assertSame(prefill, registry.getEndpoint(RoleType.PREFILL, ipPort));
        assertSame(decode, registry.getEndpoint(RoleType.DECODE, ipPort));
    }

    @Test
    void decode_routing_view_refreshes_discovery_topology_without_status_commit() {
        String address = "127.0.0.1:8010@0";
        WorkerStatus status =
                EndpointTestSupport.workerStatus(
                        RoleType.DECODE, "group-a", "127.0.0.1", 8010, 8011, "site-a");
        EndpointTestSupport.publishEndpoint(registry, RoleType.DECODE, address, status);

        DecodeEndpoint.SelectionSnapshot before = decodeRoutingSnapshot(registry, null).getFirst();
        assertEquals(address, before.address());
        assertEquals("group-a", before.topology().group());
        assertEquals("site-a", before.topology().site());

        status.lock.lock();
        try {
            status.updateDiscoveryLabels("site-b", "group-b");
        } finally {
            status.lock.unlock();
        }

        DecodeEndpoint.SelectionSnapshot after = decodeRoutingSnapshot(registry, null).getFirst();
        assertNotSame(before, after, "topology is an independent routing-cache generation");
        assertSame(before.workerStatus(), after.workerStatus());
        assertEquals(before.admissionVersion(), after.admissionVersion());
        assertEquals(address, after.address());
        assertEquals("group-b", after.topology().group());
        assertEquals("site-b", after.topology().site());
    }

    @Test
    void decode_routing_view_captures_only_its_exact_same_address_generation() {
        String address = "127.0.0.1:8020@0";
        WorkerStatus oldStatus = status(RoleType.DECODE, 8020);
        EndpointTestSupport.publishEndpoint(registry, RoleType.DECODE, address, oldStatus);
        DecodeEndpoint.SelectionSnapshot oldView = decodeRoutingSnapshot(registry, null).getFirst();

        closeWorker(RoleType.DECODE, address, oldStatus);
        WorkerStatus replacementStatus = status(RoleType.DECODE, 8020);
        DecodeEndpoint replacement =
                (DecodeEndpoint)
                        EndpointTestSupport.publishEndpoint(
                                registry, RoleType.DECODE, address, replacementStatus);

        org.junit.jupiter.api.Assertions.assertNotEquals(
                oldView.generationId(),
                registry.getEndpoint(RoleType.DECODE, oldView.address())
                        .getStatus()
                        .getGenerationId());
        DecodeEndpoint.SelectionSnapshot replacementView =
                decodeRoutingSnapshot(registry, null).getFirst();
        assertEquals(address, replacementView.address());
        assertTrue(oldView.generationId() != replacementView.generationId());
        {
            WorkerEndpoint selectedEndpoint =
                    registry.getEndpoint(
                            org.flexlb.config.service.RoleType.DECODE, replacementView.address());

            assertNotNull(selectedEndpoint);
            assertSame(replacement, selectedEndpoint);
        }
    }

    @Test
    void prefill_routing_directory_tracks_exact_advisory_generations() {
        String firstAddress = "127.0.0.1:8001@0";
        String secondAddress = "127.0.0.1:8002@0";
        String thirdAddress = "127.0.0.1:8003@0";
        WorkerStatus firstStatus = status(RoleType.PREFILL, 8001);
        WorkerStatus secondStatus = status(RoleType.PREFILL, 8002);
        WorkerEndpoint first =
                EndpointTestSupport.publishEndpoint(
                        registry, RoleType.PREFILL, firstAddress, firstStatus);
        EndpointTestSupport.publishEndpoint(
                registry, RoleType.PREFILL, secondAddress, secondStatus);
        EndpointTestSupport.publishEndpoint(
                registry, RoleType.PREFILL, thirdAddress, status(RoleType.PREFILL, 8003));

        List<PrefillEndpoint> originalDirectory =
                registry.<org.flexlb.worker.endpoint.PrefillEndpoint>endpoints(
                        RoleType.PREFILL, null);
        assertEquals(
                List.of(firstAddress, secondAddress, thirdAddress),
                originalDirectory.stream().map(PrefillEndpoint::ipPort).toList());
        assertSame(first, originalDirectory.getFirst());
        assertThrows(
                UnsupportedOperationException.class,
                () -> originalDirectory.add((PrefillEndpoint) first));

        closeWorker(RoleType.PREFILL, firstAddress, firstStatus);
        WorkerEndpoint replacement =
                EndpointTestSupport.publishEndpoint(
                        registry, RoleType.PREFILL, firstAddress, status(RoleType.PREFILL, 8001));
        assertNotSame(first, replacement);
        List<PrefillEndpoint> replacementDirectory =
                registry.<org.flexlb.worker.endpoint.PrefillEndpoint>endpoints(
                        RoleType.PREFILL, null);
        assertEquals(
                List.of(firstAddress, secondAddress, thirdAddress),
                replacementDirectory.stream().map(PrefillEndpoint::ipPort).toList(),
                "the replacement generation retains address ordering");
        assertSame(
                first,
                originalDirectory.getFirst(),
                "an in-flight traversal keeps its advisory old generation");
        assertSame(
                replacement,
                replacementDirectory.stream()
                        .filter(entry -> entry.ipPort().equals(firstAddress))
                        .findFirst()
                        .orElseThrow());
        WorkerEndpoint replacementWorker =
                registry.getEndpoint(RoleType.PREFILL, originalDirectory.getFirst().ipPort());
        assertNotNull(replacementWorker);
        {
            assertSame(replacement, replacementWorker);
            assertNotSame(
                    originalDirectory.getFirst(),
                    replacementWorker,
                    "winner identity validation must reject the stale route");
        }

        java.util.concurrent.CompletionStage<Void> detached =
                detachUnderGenerationLock(RoleType.PREFILL, secondAddress, secondStatus);
        assertNotNull(detached);
        EndpointTestSupport.awaitWorkerCleanup(detached);
        assertEquals(
                Set.of(firstAddress, thirdAddress),
                registry
                        .<org.flexlb.worker.endpoint.PrefillEndpoint>endpoints(
                                RoleType.PREFILL, null)
                        .stream()
                        .map(PrefillEndpoint::ipPort)
                        .collect(java.util.stream.Collectors.toSet()));

        registry.close();
        assertTrue(
                registry.<org.flexlb.worker.endpoint.PrefillEndpoint>endpoints(
                                RoleType.PREFILL, null)
                        .isEmpty());
    }

    @Test
    void simple_endpoint_should_report_status_task_count_as_load() {
        WorkerStatus status = status(RoleType.VIT, 8080);
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setAlive(true);
        response.setRunningTaskInfo(Map.of("1", new TaskInfo(), "2", new TaskInfo()));
        WorkerEndpoint endpoint =
                EndpointTestSupport.publishEndpoint(
                        registry, RoleType.VIT, "127.0.0.1:8080@0", status, response);

        assertEquals(2L, endpoint.tryGetLoadMetric().orElseThrow());
    }

    @Test
    void should_not_remove_new_generation_with_expired_status() {
        String ipPort = "127.0.0.1:8080@0";
        WorkerStatus expired = status(RoleType.VIT, 8080);
        WorkerEndpoint oldEndpoint =
                EndpointTestSupport.publishEndpoint(registry, RoleType.VIT, ipPort, expired);

        closeWorker(RoleType.VIT, ipPort, expired);
        WorkerStatus replacement = status(RoleType.VIT, 8080);
        WorkerEndpoint newEndpoint =
                EndpointTestSupport.publishEndpoint(registry, RoleType.VIT, ipPort, replacement);

        assertNotSame(oldEndpoint, newEndpoint);
        assertTrue(oldEndpoint.hasStoppedAcceptingRequests());
        assertSame(newEndpoint, registry.getEndpoint(RoleType.VIT, ipPort));

        java.util.concurrent.CompletionStage<Void> detached =
                detachUnderGenerationLock(RoleType.VIT, ipPort, replacement);
        assertNotNull(detached);
        EndpointTestSupport.awaitWorkerCleanup(detached);
        assertNull(registry.getEndpoint(RoleType.VIT, ipPort));
    }

    @Test
    void sameAddressReplacementCannotBindOrReleaseOldDecodeReservation() {
        String ipPort = "127.0.0.1:8080@0";
        DecodeEndpoint oldEndpoint =
                (DecodeEndpoint)
                        EndpointTestSupport.publishEndpoint(
                                registry, RoleType.DECODE, ipPort, status(RoleType.DECODE, 8080));
        DecodeEndpoint.RequestRegistration oldReservation;
        {
            WorkerEndpoint selectedEndpoint = oldEndpoint;

            assertTrue(selectedEndpoint != null);
            oldReservation =
                    org.flexlb.balance.schedule.SelectionTestSupport.registerQueuedDecode(
                                    oldEndpoint, String.valueOf(41L), 100L, 110L, 50)
                            .value();
        }

        closeWorker(RoleType.DECODE, ipPort, oldEndpoint.getStatus());
        DecodeEndpoint replacement =
                (DecodeEndpoint)
                        EndpointTestSupport.publishEndpoint(
                                registry, RoleType.DECODE, ipPort, status(RoleType.DECODE, 8080));

        assertNotSame(oldEndpoint, replacement);
        assertTrue(oldEndpoint.hasStoppedAcceptingRequests());
        assertNull(
                EndpointTestSupport.decodeRegistration(oldEndpoint, oldReservation.requestId()),
                "close must closeWorker A's queued ownership before B is routable");
        assertNull(EndpointTestSupport.decodeRegistration(replacement, oldReservation.requestId()));
        replacement.releaseRequest(oldReservation);
        assertTrue(
                replacement.resourceSnapshot().requests().values().stream()
                                .filter(
                                        snapshotRequest ->
                                                !snapshotRequest.phase().isEngineConfirmed())
                                .count()
                        == 0,
                "A's exact generation handle must never mutate same-address B");
    }

    @Test
    void unpublishedRetirementOwnsFinalizationAndCannotRemoveItsReplacement() throws Exception {
        WorkerStatus status = status(RoleType.PREFILL, 8300);
        String address = status.getLogicalIpPort();
        registry.currentOrDiscover(RoleType.PREFILL, address, () -> status);
        var cache = mock(org.flexlb.cache.match.CacheAwareService.class);
        org.springframework.test.util.ReflectionTestUtils.setField(
                registry, "cacheAwareService", cache);
        var failed = new java.util.concurrent.CountDownLatch(1);
        var releaseRetry = new java.util.concurrent.CountDownLatch(1);
        org.mockito.Mockito.doAnswer(
                        call -> {
                            failed.countDown();
                            throw new IllegalStateException("cache unavailable");
                        })
                .doAnswer(
                        call -> {
                            assertTrue(
                                    releaseRetry.await(5, java.util.concurrent.TimeUnit.SECONDS));
                            return null;
                        })
                .when(cache)
                .removeEngineBlockCache(address);
        var cleanup = registry.closeWorker(RoleType.PREFILL, address, status);
        assertTrue(failed.await(2, java.util.concurrent.TimeUnit.SECONDS));
        assertSame(
                status,
                registry.statusSnapshot(RoleType.PREFILL).get(address),
                "failed cache cleanup must keep the old generation fenced");
        releaseRetry.countDown();
        EndpointTestSupport.awaitWorkerCleanup(cleanup);
        long deadline = System.nanoTime() + java.util.concurrent.TimeUnit.SECONDS.toNanos(3L);
        while (registry.statusSnapshot(RoleType.PREFILL).containsKey(address)
                && System.nanoTime() < deadline) {
            Thread.sleep(5L);
        }
        assertFalse(registry.statusSnapshot(RoleType.PREFILL).containsKey(address));

        WorkerStatus replacement = status(RoleType.PREFILL, 8300);
        registry.currentOrDiscover(RoleType.PREFILL, address, () -> replacement);
        EndpointTestSupport.awaitWorkerCleanup(cleanup);
        assertSame(replacement, registry.statusSnapshot(RoleType.PREFILL).get(address));
        org.mockito.Mockito.verify(cache, org.mockito.Mockito.times(2))
                .removeEngineBlockCache(address);
        registry.close(); // An unpublished cleanup must not leave a detached-endpoint barrier.
    }

    private static WorkerStatus status(RoleType roleType, int port) {
        return EndpointTestSupport.workerStatus(roleType, "127.0.0.1", port, port + 1);
    }

    private void closeWorker(RoleType role, String address, WorkerStatus status) {
        java.util.concurrent.CompletionStage<Void> detached =
                detachUnderGenerationLock(role, address, status);
        assertNotNull(detached);
        EndpointTestSupport.awaitWorkerCleanup(detached);
    }

    private java.util.concurrent.CompletionStage<Void> detachUnderGenerationLock(
            RoleType role, String address, WorkerStatus expectedStatus) {
        expectedStatus.lock.lock();
        try {
            return registry.closeWorker(role, address, expectedStatus);
        } finally {
            expectedStatus.lock.unlock();
        }
    }
}
