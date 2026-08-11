package org.flexlb.sync.schedule;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerLifecycleState;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ExpirationCleanerTest {

    private EndpointRegistry registry;

    @BeforeEach
    void setUp() {
        ConfigService configService = Mockito.mock(ConfigService.class);
        Mockito.when(configService.loadBalanceConfig()).thenReturn(new FlexlbConfig());
        registry = new EndpointRegistry(
                configService, () -> null, Mockito.mock(BatchSchedulerReporter.class));
    }

    @AfterEach
    void tearDown() {
        registry.close();
    }

    @Test
    void should_remove_expired_status_and_endpoint_for_all_supported_roles() {
        ExpirationCleaner cleaner = new ExpirationCleaner(registry, 1_000L);
        for (RoleType role : new RoleType[]{
                RoleType.PREFILL, RoleType.DECODE, RoleType.PDFUSION, RoleType.VIT}) {
            int port = 8000 + role.ordinal();
            String ipPort = "127.0.0.1:" + port;
            WorkerStatus status = status(role, port);
            status.getStatusLastUpdateTime().set(System.nanoTime() / 1000 - 2_000L);
            Map<String, WorkerStatus> statusMap = new ConcurrentHashMap<>();
            statusMap.put(ipPort, status);
            registry.ensureEndpoint(role, ipPort, status);

            cleaner.doClean(statusMap, role);

            assertTrue(statusMap.isEmpty());
            assertFalse(status.isAlive());
            assertNull(registry.get(role, ipPort));
        }
    }

    @Test
    void should_keep_fresh_status_and_endpoint() {
        ExpirationCleaner cleaner = new ExpirationCleaner(registry, 1_000_000L);
        String ipPort = "127.0.0.1:8080";
        WorkerStatus status = status(RoleType.VIT, 8080);
        status.getStatusLastUpdateTime().set(System.nanoTime() / 1000);
        Map<String, WorkerStatus> statusMap = new ConcurrentHashMap<>();
        statusMap.put(ipPort, status);
        registry.ensureEndpoint(RoleType.VIT, ipPort, status);

        cleaner.doClean(statusMap, RoleType.VIT);

        assertSame(status, statusMap.get(ipPort));
        assertTrue(status.isAlive());
        assertSame(status, registry.get(RoleType.VIT, ipPort).getStatus());
    }

    @Test
    void should_keep_fresh_probing_generation_without_status_heartbeat() {
        ExpirationCleaner cleaner = new ExpirationCleaner(registry, 1_000_000L);
        String ipPort = "127.0.0.1:8080";
        WorkerStatus probing = probingStatus(RoleType.VIT, 8080);
        probing.recordDiscoverySeen(System.nanoTime() / 1000);
        Map<String, WorkerStatus> statusMap = new ConcurrentHashMap<>();
        statusMap.put(ipPort, probing);

        cleaner.doClean(statusMap, RoleType.VIT);

        assertSame(probing, statusMap.get(ipPort));
        assertEquals(-1L, probing.getStatusLastUpdateTime().get());
        assertEquals(WorkerLifecycleState.PROBING, probing.getLifecycleState());
        assertNull(registry.get(RoleType.VIT, ipPort));
    }

    @Test
    void should_expire_probing_generation_only_when_discovery_is_stale() {
        ExpirationCleaner cleaner = new ExpirationCleaner(registry, 1_000L);
        String ipPort = "127.0.0.1:8080";
        WorkerStatus probing = probingStatus(RoleType.VIT, 8080);
        probing.getDiscoveryLastSeenTime().set(System.nanoTime() / 1000 - 2_000L);
        Map<String, WorkerStatus> statusMap = new ConcurrentHashMap<>();
        statusMap.put(ipPort, probing);

        cleaner.doClean(statusMap, RoleType.VIT);

        assertTrue(statusMap.isEmpty());
        assertEquals(WorkerLifecycleState.CLOSED, probing.getLifecycleState());
        assertNull(registry.get(RoleType.VIT, ipPort));
    }

    @Test
    void should_recheck_ready_heartbeat_after_waiting_for_lifecycle_lock()
            throws InterruptedException {
        ExpirationCleaner cleaner = new ExpirationCleaner(registry, 1_000L);
        String ipPort = "127.0.0.1:8080";
        WorkerStatus status = status(RoleType.VIT, 8080);
        status.getStatusLastUpdateTime().set(System.nanoTime() / 1000 - 2_000L);
        Map<String, WorkerStatus> statusMap = new ConcurrentHashMap<>();
        statusMap.put(ipPort, status);
        registry.ensureEndpoint(RoleType.VIT, ipPort, status);

        Thread cleanerThread = new Thread(
                () -> cleaner.doClean(statusMap, RoleType.VIT), "expiration-cleaner-test");
        status.lock.lock();
        try {
            cleanerThread.start();
            awaitQueuedOnLifecycleLock(status, cleanerThread);
            status.recordStatusSuccess();
        } finally {
            status.lock.unlock();
        }
        cleanerThread.join(TimeUnit.SECONDS.toMillis(2));

        assertFalse(cleanerThread.isAlive());
        assertSame(status, statusMap.get(ipPort));
        assertEquals(WorkerLifecycleState.READY, status.getLifecycleState());
        assertSame(status, registry.get(RoleType.VIT, ipPort).getStatus());
    }

    @Test
    void should_recheck_probing_discovery_after_waiting_for_lifecycle_lock()
            throws InterruptedException {
        ExpirationCleaner cleaner = new ExpirationCleaner(registry, 1_000L);
        String ipPort = "127.0.0.1:8080";
        WorkerStatus probing = probingStatus(RoleType.VIT, 8080);
        probing.getDiscoveryLastSeenTime().set(System.nanoTime() / 1000 - 2_000L);
        Map<String, WorkerStatus> statusMap = new ConcurrentHashMap<>();
        statusMap.put(ipPort, probing);

        Thread cleanerThread = new Thread(
                () -> cleaner.doClean(statusMap, RoleType.VIT), "probing-cleaner-test");
        probing.lock.lock();
        try {
            cleanerThread.start();
            awaitQueuedOnLifecycleLock(probing, cleanerThread);
            assertTrue(probing.recordDiscoverySeen(System.nanoTime() / 1000));
        } finally {
            probing.lock.unlock();
        }
        cleanerThread.join(TimeUnit.SECONDS.toMillis(2));

        assertFalse(cleanerThread.isAlive());
        assertSame(probing, statusMap.get(ipPort));
        assertEquals(WorkerLifecycleState.PROBING, probing.getLifecycleState());
    }

    private static void awaitQueuedOnLifecycleLock(WorkerStatus status, Thread contender) {
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(2);
        while (!status.lock.hasQueuedThread(contender) && System.nanoTime() < deadline) {
            Thread.onSpinWait();
        }
        assertTrue(status.lock.hasQueuedThread(contender),
                "cleaner did not reach the lifecycle retirement check");
    }

    private static WorkerStatus status(RoleType role, int port) {
        WorkerStatus status = probingStatus(role, port);
        status.setAlive(true);
        return status;
    }

    private static WorkerStatus probingStatus(RoleType role, int port) {
        WorkerStatus status = new WorkerStatus();
        status.setRole(role);
        status.setIp("127.0.0.1");
        status.setPort(port);
        status.setGrpcPort(port + 1);
        return status;
    }
}
