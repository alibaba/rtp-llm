package org.flexlb.sync.schedule;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ExpirationCleanerTest {

    private EndpointRegistry registry;

    @BeforeEach
    void setUp() {
        ConfigService configService = Mockito.mock(ConfigService.class);
        Mockito.when(configService.loadBalanceConfig()).thenReturn(new FlexlbConfig());
        registry = new EndpointRegistry(
                configService, null, Mockito.mock(BatchSchedulerReporter.class));
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
