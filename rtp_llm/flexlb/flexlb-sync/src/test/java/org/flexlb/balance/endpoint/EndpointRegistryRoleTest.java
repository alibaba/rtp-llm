package org.flexlb.balance.endpoint;

import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

class EndpointRegistryRoleTest {

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
    void should_register_and_resolve_supported_roles_independently() {
        WorkerEndpoint prefill = registry.ensureEndpoint(
                RoleType.PREFILL, "127.0.0.1:8001", status(RoleType.PREFILL, 8001));
        WorkerEndpoint decode = registry.ensureEndpoint(
                RoleType.DECODE, "127.0.0.1:8002", status(RoleType.DECODE, 8002));
        WorkerEndpoint pdFusion = registry.ensureEndpoint(
                RoleType.PDFUSION, "127.0.0.1:8003", status(RoleType.PDFUSION, 8003));
        WorkerEndpoint vit = registry.ensureEndpoint(
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
        WorkerEndpoint prefill = registry.ensureEndpoint(
                RoleType.PREFILL, ipPort, status(RoleType.PREFILL, 8080));
        WorkerEndpoint decode = registry.ensureEndpoint(
                RoleType.DECODE, ipPort, status(RoleType.DECODE, 8080));

        assertNotSame(prefill, decode);
        assertSame(prefill, registry.get(RoleType.PREFILL, ipPort));
        assertSame(decode, registry.get(RoleType.DECODE, ipPort));
    }

    @Test
    void simple_endpoint_should_report_status_task_count_as_load() {
        WorkerStatus status = status(RoleType.VIT, 8080);
        status.setRunningTaskList(Map.of("1", new TaskInfo(), "2", new TaskInfo()));
        SimpleWorkerEndpoint endpoint = (SimpleWorkerEndpoint) registry.ensureEndpoint(
                RoleType.VIT, "127.0.0.1:8080", status);

        assertEquals(2, endpoint.getLoadMetric());
    }

    @Test
    void should_not_remove_new_generation_with_expired_status() {
        String ipPort = "127.0.0.1:8080";
        WorkerStatus expired = status(RoleType.VIT, 8080);
        WorkerEndpoint oldEndpoint = registry.ensureEndpoint(RoleType.VIT, ipPort, expired);

        WorkerStatus replacement = status(RoleType.VIT, 8080);
        WorkerEndpoint newEndpoint = registry.ensureEndpoint(RoleType.VIT, ipPort, replacement);

        assertNotSame(oldEndpoint, newEndpoint);
        assertFalse(registry.remove(RoleType.VIT, ipPort, expired));
        assertFalse(expired.isAlive());
        assertSame(newEndpoint, registry.get(RoleType.VIT, ipPort));

        assertTrue(registry.remove(RoleType.VIT, ipPort, replacement));
        assertNull(registry.get(RoleType.VIT, ipPort));
    }

    private static WorkerStatus status(RoleType roleType, int port) {
        WorkerStatus status = new WorkerStatus();
        status.setRole(roleType);
        status.setIp("127.0.0.1");
        status.setPort(port);
        status.setGrpcPort(port + 1);
        status.setAlive(true);
        return status;
    }
}
