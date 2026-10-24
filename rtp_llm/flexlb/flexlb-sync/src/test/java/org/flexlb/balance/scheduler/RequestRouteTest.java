package org.flexlb.balance.scheduler;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.dao.loadbalance.DebugInfo;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

class RequestRouteTest {
    @Test
    void successReturnsTheSameScopeAndFreezesFactsWithoutAnotherOwner() {
        var context = context();
        var worker = prefill();
        var pin = worker.tryPinGeneration();
        var metadata = metadata(worker);
        metadata.setDpRank(7L);
        var debug = new DebugInfo();
        debug.setHitCacheLen(17L);
        metadata.setDebugInfo(debug);
        try {
            var result = SchedulingTestConfig.prepareRoute(context, route -> {
                route.add(pin, metadata, 10L, 5L);
                return PlacementResult.success(route);
            });
            try (var route = result.value()) {
                metadata.setGroup("changed");
                metadata.setDpRank(99L);
                debug.setHitCacheLen(99L);
                route.prefill().setGroup("export changed");
                route.successResponse(true).getServerStatus().getFirst().getDebugInfo().setHitCacheLen(1L);
                assertEquals("original", route.prefill().getGroup());
                assertEquals(17L, route.hitCache());
                assertEquals(7L, route.prefillDpRank());
                assertEquals(10L, route.prefillWorkMs());
                assertSame(pin, route.prefillPin());
                assertSame(worker, route.prefillEp());
                assertTrue(pin.isOpen());
            }
            assertFalse(pin.isOpen());
            assertEquals("original", result.value().prefill().getGroup());
            result.value().close();
        } finally { worker.close(); worker.awaitRetirement(); }
    }

    @ParameterizedTest
    @ValueSource(strings = {"failed", "request", "address", "work", "version", "closed", "role"})
    void validationFailureClosesTheIncomingPin(String invalid) {
        var worker = prefill();
        var pin = worker.tryPinGeneration();
        var metadata = metadata(worker);
        if (invalid.equals("failed")) { metadata.setSuccess(false); }
        if (invalid.equals("request")) { metadata.setRequestId(99L); }
        if (invalid.equals("address")) { metadata.setHttpPort(9999); }
        if (invalid.equals("closed")) { pin.close(); }
        if (invalid.equals("role")) { metadata.setRole(RoleType.DECODE); }
        try (var route = new RequestRoute(context())) {
            assertThrows(RuntimeException.class, () -> route.add(pin, metadata,
                    invalid.equals("work") ? -1L : 10L, invalid.equals("version") ? -1L : 0L));
            assertFalse(pin.isOpen());
        } finally { worker.close(); worker.awaitRetirement(); }
    }

    @Test
    void duplicateRoleClosesBothPinsThroughTheSingleScope() {
        var worker = prefill();
        var first = worker.tryPinGeneration();
        var duplicate = worker.tryPinGeneration();
        try {
            assertThrows(IllegalStateException.class, () -> SchedulingTestConfig.prepareRoute(context(), route -> {
                route.add(first, metadata(worker), 1L, 0L);
                route.add(duplicate, metadata(worker), 1L, 0L);
                return PlacementResult.success(route);
            }));
            assertFalse(first.isOpen());
            assertFalse(duplicate.isOpen());
        } finally { worker.close(); worker.awaitRetirement(); }
    }

    @Test
    void blockedResultReleasesPreviouslySelectedPins() {
        var worker = prefill();
        var pin = worker.tryPinGeneration();
        try {
            var result = SchedulingTestConfig.prepareRoute(context(), route -> {
                route.add(pin, metadata(worker), 1L, 0L);
                return PlacementResult.blocked(new PlacementKey(RoleType.DECODE, null, null));
            });
            assertEquals(PlacementResult.Status.BLOCKED, result.status());
            assertFalse(pin.isOpen());
        } finally { worker.close(); worker.awaitRetirement(); }
    }

    @Test
    void closingAnEmptyConstructionPreventsReopeningAndConsumesIncomingPin() {
        var worker = prefill();
        var pin = worker.tryPinGeneration();
        try (var route = new RequestRoute(context())) {
            route.close();
            assertThrows(IllegalStateException.class, () -> route.add(pin, metadata(worker), 1L, 0L));
            assertFalse(pin.isOpen());
            assertThrows(IllegalStateException.class, route::prepare);
        } finally { worker.close(); worker.awaitRetirement(); }
    }

    @Test
    void closedConstructionCannotPublishOrAcquireCapacity() {
        var worker = prefill();
        var context = context();
        try (var route = new RequestRoute(context)) {
            route.add(worker.tryPinGeneration(), metadata(worker), 1L, 0L);
            assertThrows(IllegalStateException.class, () -> RequestRoute.create(context, route, null));
            route.close();
            assertThrows(IllegalStateException.class, route::prepare);
            assertThrows(IllegalStateException.class, () -> RequestRoute.create(context, route, null));
        } finally { worker.close(); worker.awaitRetirement(); }
    }

    @Test
    void failedAddCannotBeRecoveredIntoACompletedRoute() {
        var worker = prefill();
        var invalid = metadata(worker);
        invalid.setSuccess(false);
        try (var route = new RequestRoute(context())) {
            assertThrows(IllegalArgumentException.class,
                    () -> route.add(worker.tryPinGeneration(), invalid, 1L, 0L));
            var nextPin = worker.tryPinGeneration();
            assertThrows(IllegalStateException.class, () -> route.add(nextPin, metadata(worker), 1L, 0L));
            assertFalse(nextPin.isOpen());
            assertThrows(IllegalStateException.class, route::prepare);
        } finally { worker.close(); worker.awaitRetirement(); }
    }

    @Test
    void prepareRejectsAnExternallyClosedPin() {
        var worker = prefill();
        var pin = worker.tryPinGeneration();
        try (var route = new RequestRoute(context())) {
            route.add(pin, metadata(worker), 1L, 0L);
            pin.close();
            assertThrows(IllegalStateException.class, route::prepare);
            assertThrows(IllegalStateException.class, route::prepare);
        } finally { worker.close(); worker.awaitRetirement(); }
    }

    @ParameterizedTest
    @ValueSource(strings = {"PDV", "PVD", "DPV", "DVP", "VPD", "VDP"})
    void preparationPreservesSelectionOrderAndOnlyReleasesVit(String order) {
        var prefill = prefill();
        var decode = org.flexlb.balance.endpoint.EndpointTestSupport.decode(
                WorkerStatus.createDiscovered(RoleType.DECODE, "original", "decode", 8080, 8090, null),
                new RequestRepository());
        var vit = new WorkerEndpoint(WorkerStatus.createDiscovered(RoleType.VIT, "original", "vit", 8080, 8090, null));
        var pins = new java.util.ArrayList<WorkerEndpoint.GenerationPin>();
        var roles = new java.util.ArrayList<RoleType>();
        try (var route = new RequestRoute(context())) {
            for (char role : order.toCharArray()) {
                WorkerEndpoint worker = switch (role) { case 'P' -> prefill; case 'D' -> decode; default -> vit; };
                var pin = worker.tryPinGeneration();
                pins.add(pin);
                roles.add(worker.getStatus().getRole());
                route.add(pin, metadata(worker), role == 'P' ? 1L : -1L, 0L);
            }
            assertSame(route, route.prepare());
            assertEquals(roles, route.successResponse(true).getServerStatus().stream().map(ServerStatus::getRole).toList());
            for (var pin : pins) { assertEquals(pin.endpoint() != vit, pin.isOpen()); }
            assertThrows(IllegalStateException.class, route::prepare);
            route.close();
            assertTrue(pins.stream().noneMatch(WorkerEndpoint.GenerationPin::isOpen));
            assertEquals(roles, route.successResponse(true).getServerStatus().stream().map(ServerStatus::getRole).toList());
        } finally {
            prefill.close(); decode.close(); vit.close();
            prefill.awaitRetirement(); decode.awaitRetirement(); vit.awaitRetirement();
        }
    }

    @Test
    void cleanupFailureKeepsTheOriginalValidationFailure() {
        var worker = prefill();
        var pin = spy(worker.tryPinGeneration());
        var cleanup = new IllegalStateException("pin cleanup failed");
        doAnswer(call -> { call.callRealMethod(); throw cleanup; }).when(pin).close();
        var metadata = metadata(worker);
        metadata.setSuccess(false);
        try (var route = new RequestRoute(context())) {
            var failure = assertThrows(IllegalArgumentException.class, () -> route.add(pin, metadata, 1L, 0L));
            assertEquals(List.of(cleanup), List.of(failure.getSuppressed()));
            assertFalse(pin.isOpen());
        } finally { worker.close(); worker.awaitRetirement(); }
    }

    @Test
    void vitCleanupFailurePreventsPublishingAndReleasesStatefulPins() {
        var worker = prefill();
        var vit = new WorkerEndpoint(WorkerStatus.createDiscovered(RoleType.VIT, "original", "vit", 8080, 8090, null));
        var prefillPin = worker.tryPinGeneration();
        var vitPin = spy(vit.tryPinGeneration());
        var cleanup = new IllegalStateException("VIT release failed");
        doAnswer(call -> { call.callRealMethod(); throw cleanup; }).when(vitPin).close();
        try {
            var failure = assertThrows(IllegalStateException.class, () -> SchedulingTestConfig.prepareRoute(context(), route -> {
                route.add(prefillPin, metadata(worker), 1L, 0L);
                route.add(vitPin, metadata(vit), -1L, 0L);
                return PlacementResult.success(route);
            }));
            assertSame(cleanup, failure);
            assertFalse(prefillPin.isOpen());
            assertFalse(vitPin.isOpen());
        } finally { worker.close(); vit.close(); worker.awaitRetirement(); vit.awaitRetirement(); }
    }

    @Test
    @Timeout(5)
    void concurrentCloseOfSelectedAndCapacityRoutesDrainsGenerationOnce() throws Exception {
        var worker = prefill();
        var pin = worker.tryPinGeneration();
        var context = context();
        var route = SchedulingTestConfig.prepareRoute(context, scope -> {
            scope.add(pin, metadata(worker), 1L, 0L);
            return PlacementResult.success(scope);
        }).value();
        var acquired = RequestRoute.create(context, route, null);
        var barrier = new CyclicBarrier(2);
        try {
            worker.close();
            var first = CompletableFuture.runAsync(() -> closeAfterBarrier(route, barrier));
            var second = CompletableFuture.runAsync(() -> closeAfterBarrier(acquired, barrier));
            CompletableFuture.allOf(first, second).get(3, TimeUnit.SECONDS);
            worker.awaitRetirement();
            assertFalse(pin.isOpen());
        } finally { route.close(); acquired.close(); worker.close(); worker.awaitRetirement(); }
    }

    private static void closeAfterBarrier(RequestRoute route, CyclicBarrier barrier) {
        try { barrier.await(2, TimeUnit.SECONDS); route.close(); }
        catch (Exception failure) { throw new AssertionError(failure); }
    }

    private static RequestContext context() {
        return RequestProtocolTestSupport.context(SchedulingTestConfig.newConfig(), 42L);
    }

    private static PrefillEndpoint prefill() {
        var status = WorkerStatus.createDiscovered(RoleType.PREFILL, "original", "127.0.0.1", 8080, 8090, null);
        return org.flexlb.balance.endpoint.EndpointTestSupport.realPrefill(status, SchedulingTestConfig.newConfig(),
                mock(DeliveryDispatcher.class), new RequestRepository(), mock(DeliveryMetricsReporter.class));
    }

    private static ServerStatus metadata(WorkerEndpoint worker) {
        var result = ServerStatus.forWorker(worker.getStatus().getRole(), 42L, worker.getStatus().topologySnapshot(),
                worker.getStatus().committedEngineObservation());
        return result;
    }
}
