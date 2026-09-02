package org.flexlb.balance.endpoint;

import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class EndpointRegistryRoleTest {

    private EndpointRegistry registry;

    @BeforeEach
    void setUp() {
        ConfigService configService = Mockito.mock(ConfigService.class);
        Mockito.when(configService.loadBalanceConfig()).thenReturn(new FlexlbConfig());
        EndpointTestSupport.TestRequestRuntime requestRuntime =
                EndpointTestSupport.requestRuntime();
        registry = new EndpointRegistry(
                configService,
                requestRuntime,
                Mockito.mock(BatchSchedulerReporter.class),
                EndpointTestSupport.routeStrategy(requestRuntime));
    }

    @AfterEach
    void tearDown() {
        registry.close();
    }

    @Test
    void should_register_and_resolve_supported_roles_independently() {
        WorkerEndpoint prefill = registry.registerPreinitializedEndpoint(
                RoleType.PREFILL, "127.0.0.1:8001", status(RoleType.PREFILL, 8001));
        WorkerEndpoint decode = registry.registerPreinitializedEndpoint(
                RoleType.DECODE, "127.0.0.1:8002", status(RoleType.DECODE, 8002));
        WorkerEndpoint pdFusion = registry.registerPreinitializedEndpoint(
                RoleType.PDFUSION, "127.0.0.1:8003", status(RoleType.PDFUSION, 8003));
        WorkerEndpoint vit = registry.registerPreinitializedEndpoint(
                RoleType.VIT, "127.0.0.1:8004", status(RoleType.VIT, 8004));

        assertInstanceOf(PrefillEndpoint.class, prefill);
        assertInstanceOf(DecodeEndpoint.class, decode);
        assertInstanceOf(PrefillEndpoint.class, pdFusion);
        assertInstanceOf(SimpleWorkerEndpoint.class, vit);
        assertSame(prefill, registry.get(RoleType.PREFILL, "127.0.0.1:8001"));
        assertSame(decode, registry.get(RoleType.DECODE, "127.0.0.1:8002"));
        assertSame(pdFusion, registry.get(RoleType.PDFUSION, "127.0.0.1:8003"));
        assertSame(vit, registry.get(RoleType.VIT, "127.0.0.1:8004"));

        for (RoleType roleType : new RoleType[]{
                RoleType.PREFILL, RoleType.DECODE, RoleType.PDFUSION, RoleType.VIT}) {
            assertEquals(1, registry.getEndpointCount(roleType));
        }
    }

    @Test
    void should_not_mix_roles_when_ip_port_is_shared() {
        String ipPort = "127.0.0.1:8080";
        WorkerEndpoint prefill = registry.registerPreinitializedEndpoint(
                RoleType.PREFILL, ipPort, status(RoleType.PREFILL, 8080));
        WorkerEndpoint decode = registry.registerPreinitializedEndpoint(
                RoleType.DECODE, ipPort, status(RoleType.DECODE, 8080));

        assertNotSame(prefill, decode);
        assertSame(prefill, registry.get(RoleType.PREFILL, ipPort));
        assertSame(decode, registry.get(RoleType.DECODE, ipPort));
    }

    @Test
    void prefill_address_directory_tracks_membership_without_owning_generation() {
        String firstAddress = "127.0.0.1:8001";
        String secondAddress = "127.0.0.1:8002";
        String thirdAddress = "127.0.0.1:8003";
        WorkerStatus firstStatus = status(RoleType.PREFILL, 8001);
        WorkerStatus secondStatus = status(RoleType.PREFILL, 8002);
        WorkerEndpoint first = registry.registerPreinitializedEndpoint(
                RoleType.PREFILL, firstAddress, firstStatus);
        registry.registerPreinitializedEndpoint(
                RoleType.PREFILL, secondAddress, secondStatus);
        registry.registerPreinitializedEndpoint(
                RoleType.PREFILL,
                thirdAddress,
                status(RoleType.PREFILL, 8003));

        List<String> originalDirectory =
                registry.endpointAddressSnapshot(RoleType.PREFILL);
        assertEquals(
                List.of(firstAddress, secondAddress, thirdAddress),
                originalDirectory);
        assertThrows(UnsupportedOperationException.class,
                () -> originalDirectory.add("127.0.0.1:8004"));

        WorkerEndpoint replacement = registry.registerPreinitializedEndpoint(
                RoleType.PREFILL,
                firstAddress,
                status(RoleType.PREFILL, 8001));
        assertNotSame(first, replacement);
        assertSame(originalDirectory,
                registry.endpointAddressSnapshot(RoleType.PREFILL),
                "same-address replacement must retain the address directory");
        WorkerEndpoint.GenerationPin replacementPin =
                registry.capture(RoleType.PREFILL, firstAddress);
        assertNotNull(replacementPin);
        try (replacementPin) {
            assertSame(replacement, replacementPin.endpoint());
        }

        EndpointRegistry.DetachedGeneration detached =
                detachUnderGenerationLock(
                        RoleType.PREFILL, secondAddress, secondStatus);
        assertNotNull(detached);
        detached.retireAndAwait();
        assertEquals(List.of(firstAddress, thirdAddress),
                registry.endpointAddressSnapshot(RoleType.PREFILL));

        registry.close();
        assertTrue(registry.endpointAddressSnapshot(RoleType.PREFILL).isEmpty());
    }

    @Test
    void simple_endpoint_should_report_status_task_count_as_load() {
        WorkerStatus status = status(RoleType.VIT, 8080);
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setAlive(true);
        response.setRunningTaskInfo(
                Map.of("1", new TaskInfo(), "2", new TaskInfo()));
        EndpointTestSupport.publishStatus(status, response);
        SimpleWorkerEndpoint endpoint = (SimpleWorkerEndpoint) registry.registerPreinitializedEndpoint(
                RoleType.VIT, "127.0.0.1:8080", status);

        assertEquals(2L, endpoint.getLoadMetric().orElseThrow());
    }

    @Test
    void should_not_remove_new_generation_with_expired_status() {
        String ipPort = "127.0.0.1:8080";
        WorkerStatus expired = status(RoleType.VIT, 8080);
        WorkerEndpoint oldEndpoint = registry.registerPreinitializedEndpoint(RoleType.VIT, ipPort, expired);

        WorkerStatus replacement = status(RoleType.VIT, 8080);
        WorkerEndpoint newEndpoint = registry.registerPreinitializedEndpoint(RoleType.VIT, ipPort, replacement);

        assertNotSame(oldEndpoint, newEndpoint);
        assertNull(detachUnderGenerationLock(
                RoleType.VIT, ipPort, expired));
        assertTrue(
                expired.isActiveGeneration(),
                "an exact-generation detach miss must not mutate lifecycle state");
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
                registry.registerPreinitializedEndpoint(
                        RoleType.DECODE, ipPort,
                        status(RoleType.DECODE, 8080));
        DecodeEndpoint.ReservationHandle oldReservation;
        try (WorkerEndpoint.GenerationPin pin =
                     oldEndpoint.tryPinGeneration()) {
            assertTrue(pin != null);
            oldReservation = oldEndpoint.reserveQueuedPinned(
                    pin, 41L, 100L, 110L, 50);
        }

        DecodeEndpoint replacement = (DecodeEndpoint)
                registry.registerPreinitializedEndpoint(
                        RoleType.DECODE, ipPort,
                        status(RoleType.DECODE, 8080));

        assertNotSame(oldEndpoint, replacement);
        assertTrue(oldEndpoint.isRetired());
        assertNull(oldEndpoint.reservationHandle(oldReservation.requestId()),
                "close must retire A's queued ownership before B is routable");
        assertNull(replacement.reservationHandle(oldReservation.requestId()));
        replacement.rollbackExact(oldReservation);
        assertTrue(replacement.layeredAdmissionView().reserved().isEmpty(),
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
}
