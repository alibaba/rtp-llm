package org.flexlb.sync.schedule;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.sync.status.WorkerGenerationFence;
import org.flexlb.sync.status.WorkerGenerationManager;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.never;

class ExpirationCleanerTest {

    private EndpointRegistry registry;
    private CacheAwareService cacheAwareService;
    private WorkerGenerationManager generationManager;

    @BeforeEach
    void setUp() {
        ConfigService configService = Mockito.mock(ConfigService.class);
        Mockito.when(configService.loadBalanceConfig()).thenReturn(new FlexlbConfig());
        registry = new EndpointRegistry(
                configService, () -> null, Mockito.mock(BatchSchedulerReporter.class));
        cacheAwareService = Mockito.mock(CacheAwareService.class);
        generationManager = new WorkerGenerationManager(
                registry, cacheAwareService, new WorkerGenerationFence());
    }

    @AfterEach
    void tearDown() {
        registry.close();
    }

    @Test
    void should_remove_expired_status_and_endpoint_for_all_supported_roles() {
        ExpirationCleaner cleaner = new ExpirationCleaner(generationManager, 1_000L);
        for (RoleType role : new RoleType[]{
                RoleType.PREFILL, RoleType.DECODE, RoleType.PDFUSION, RoleType.VIT}) {
            int port = 8000 + role.ordinal();
            String ipPort = "127.0.0.1:" + port;
            WorkerStatus status = status(role, port);
            status.getStatusLastUpdateTime().set(System.nanoTime() / 1000 - 2_000L);
            ConcurrentMap<String, WorkerStatus> statusMap = new ConcurrentHashMap<>();
            statusMap.put(ipPort, status);
            registry.ensureEndpoint(role, ipPort, status);

            cleaner.doClean(statusMap, role);

            assertTrue(statusMap.isEmpty());
            assertFalse(status.isAlive());
            assertNull(registry.get(role, ipPort));
            if (role == RoleType.PREFILL || role == RoleType.PDFUSION) {
                verify(cacheAwareService).clearEngineCache(ipPort);
            } else {
                verify(cacheAwareService, never()).clearEngineCache(ipPort);
            }
        }
    }

    @Test
    void should_keep_fresh_status_and_endpoint() {
        ExpirationCleaner cleaner = new ExpirationCleaner(
                generationManager, 1_000_000L);
        String ipPort = "127.0.0.1:8080";
        WorkerStatus status = status(RoleType.VIT, 8080);
        status.getStatusLastUpdateTime().set(System.nanoTime() / 1000);
        ConcurrentMap<String, WorkerStatus> statusMap = new ConcurrentHashMap<>();
        statusMap.put(ipPort, status);
        registry.ensureEndpoint(RoleType.VIT, ipPort, status);

        cleaner.doClean(statusMap, RoleType.VIT);

        assertSame(status, statusMap.get(ipPort));
        assertTrue(status.isAlive());
        assertSame(status, registry.get(RoleType.VIT, ipPort).getStatus());
    }

    private static WorkerStatus status(RoleType role, int port) {
        WorkerStatus status = new WorkerStatus();
        status.setRole(role);
        status.setIp("127.0.0.1");
        status.setPort(port);
        status.setGrpcPort(port + 1);
        status.setAlive(true);
        return status;
    }
}
