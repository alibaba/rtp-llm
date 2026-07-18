package org.flexlb.sync.status;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

class EngineWorkerStatusTest {

    private EngineWorkerStatus engineWorkerStatus;
    private EndpointRegistry registry;
    private ConfigService configService;
    private WorkerStatus workerStatus1;
    private WorkerStatus workerStatus2;

    @BeforeEach
    void setUp() {
        for (RoleType roleType : RoleType.values()) {
            EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getRoleStatusMap(roleType).clear();
        }
        configService = Mockito.mock(ConfigService.class);
        Mockito.when(configService.loadBalanceConfig()).thenReturn(new FlexlbConfig());
        registry = new EndpointRegistry(configService, () -> null,
                Mockito.mock(BatchSchedulerReporter.class));
        engineWorkerStatus = new EngineWorkerStatus(new ModelMetaConfig(), registry);
        workerStatus1 = new WorkerStatus();
        workerStatus1.setGroup("group1");
        workerStatus2 = new WorkerStatus();
        workerStatus2.setGroup("group2");
    }

    @Test
    void should_create_engine_worker_status_with_config_when_constructing_with_model_meta_config() {
        // Given
        ModelMetaConfig config = new ModelMetaConfig();

        // When
        EngineWorkerStatus status = new EngineWorkerStatus(config, registry);

        // Then
        assertNotNull(status);
        assertSame(config, status.getModelMetaConfig());
    }

    @Test
    void should_return_filtered_worker_status_when_selecting_model_worker_status_with_group_filter() {
        // Given
        String ipPort1 = "127.0.0.1:8080";
        String ipPort2 = "127.0.0.1:8081";

        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getDecodeStatusMap().clear();
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getDecodeStatusMap().put(ipPort1, workerStatus1); // group1
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getDecodeStatusMap().put(ipPort2, workerStatus2); // group2

        // Register corresponding DecodeEndpoints
        workerStatus1.setIp("127.0.0.1");
        workerStatus1.setPort(8080);
        workerStatus1.setGrpcPort(9090);
        workerStatus2.setIp("127.0.0.1");
        workerStatus2.setPort(8081);
        workerStatus2.setGrpcPort(9091);
        registry.ensureDecodeEndpoint(ipPort1, workerStatus1);
        registry.ensureDecodeEndpoint(ipPort2, workerStatus2);

        // When
        var result = engineWorkerStatus.selectModelWorkerStatus(RoleType.DECODE, "group1");

        // Then
        assertNotNull(result);
        assertEquals(1, result.size());
        assertTrue(result.containsKey(ipPort1));
        assertFalse(result.containsKey(ipPort2));

        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getDecodeStatusMap().clear();
    }

    @Test
    void should_return_all_worker_status_when_selecting_model_worker_status_without_group_filter() {
        // Given
        String ipPort1 = "127.0.0.1:8080";
        String ipPort2 = "127.0.0.1:8081";

        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap().clear();
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap().put(ipPort1, workerStatus1);
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap().put(ipPort2, workerStatus2);

        // Register corresponding PrefillEndpoints
        workerStatus1.setIp("127.0.0.1");
        workerStatus1.setPort(8080);
        workerStatus1.setGrpcPort(9090);
        workerStatus2.setIp("127.0.0.1");
        workerStatus2.setPort(8081);
        workerStatus2.setGrpcPort(9091);
        registry.ensurePrefillEndpoint(ipPort1, workerStatus1);
        registry.ensurePrefillEndpoint(ipPort2, workerStatus2);

        // When
        var result = engineWorkerStatus.selectModelWorkerStatus(RoleType.PREFILL, null);

        // Then
        assertNotNull(result);
        assertEquals(2, result.size());
        assertTrue(result.containsKey(ipPort1));
        assertTrue(result.containsKey(ipPort2));

        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap().clear();
    }

    @Test
    void should_return_empty_map_when_selecting_model_worker_status_with_null_status() {
        // Given - clear all status maps
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPdFusionStatusMap().clear();

        // When
        var result = engineWorkerStatus.selectModelWorkerStatus(RoleType.PDFUSION, null);

        // Then
        assertNotNull(result);
        assertTrue(result.isEmpty());
    }

    @Test
    void should_return_empty_map_after_group_filtering_when_no_matching_group_exists() {
        // Given
        String ipPort = "127.0.0.1:8080";

        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getVitStatusMap().clear();
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getVitStatusMap().put(ipPort, workerStatus1); // group1

        // When
        var result = engineWorkerStatus.selectModelWorkerStatus(RoleType.VIT, "nonExistentGroup");

        // Then
        assertNotNull(result);
        assertTrue(result.isEmpty());

        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getVitStatusMap().clear();
    }

    @Test
    void should_exclude_null_group_worker_when_group_specified() {
        // Given - worker1 has group "groupA", worker2 has no group (null)
        String ipPort1 = "127.0.0.1:8080";
        String ipPort2 = "127.0.0.1:8081";

        WorkerStatus ws1 = new WorkerStatus();
        ws1.setGroup("groupA");
        ws1.setIp("127.0.0.1");
        ws1.setPort(8080);
        ws1.setGrpcPort(9090);

        WorkerStatus ws2 = new WorkerStatus();
        // ws2 group left as null (not set)
        ws2.setIp("127.0.0.1");
        ws2.setPort(8081);
        ws2.setGrpcPort(9091);

        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getDecodeStatusMap().clear();
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getDecodeStatusMap().put(ipPort1, ws1);
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getDecodeStatusMap().put(ipPort2, ws2);

        registry.ensureDecodeEndpoint(ipPort1, ws1);
        registry.ensureDecodeEndpoint(ipPort2, ws2);

        // When - select with group "groupA"
        var result = engineWorkerStatus.selectModelWorkerStatus(RoleType.DECODE, "groupA");

        // Then - only worker1 should be returned, worker2 (null group) should be excluded
        assertNotNull(result);
        assertEquals(1, result.size());
        assertTrue(result.containsKey(ipPort1));
        assertFalse(result.containsKey(ipPort2));

        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getDecodeStatusMap().clear();
    }

    @Test
    void should_include_all_workers_when_group_is_null() {
        // Given - worker1 has group "groupA", worker2 has no group (null)
        String ipPort1 = "127.0.0.1:8080";
        String ipPort2 = "127.0.0.1:8081";

        WorkerStatus ws1 = new WorkerStatus();
        ws1.setGroup("groupA");
        ws1.setIp("127.0.0.1");
        ws1.setPort(8080);
        ws1.setGrpcPort(9090);

        WorkerStatus ws2 = new WorkerStatus();
        // ws2 group left as null (not set)
        ws2.setIp("127.0.0.1");
        ws2.setPort(8081);
        ws2.setGrpcPort(9091);

        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getDecodeStatusMap().clear();
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getDecodeStatusMap().put(ipPort1, ws1);
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getDecodeStatusMap().put(ipPort2, ws2);

        registry.ensureDecodeEndpoint(ipPort1, ws1);
        registry.ensureDecodeEndpoint(ipPort2, ws2);

        // When - select with group null (no group constraint)
        var result = engineWorkerStatus.selectModelWorkerStatus(RoleType.DECODE, null);

        // Then - both workers should be returned
        assertNotNull(result);
        assertEquals(2, result.size());
        assertTrue(result.containsKey(ipPort1));
        assertTrue(result.containsKey(ipPort2));

        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getDecodeStatusMap().clear();
    }

    @Test
    void should_visit_registered_endpoints_without_materializing_result() {
        String matching = "127.0.0.1:8080";
        String filtered = "127.0.0.1:8081";
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getDecodeStatusMap().clear();
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getDecodeStatusMap().put(matching, workerStatus1);
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getDecodeStatusMap().put(filtered, workerStatus2);
        workerStatus1.setIp("127.0.0.1");
        workerStatus1.setPort(8080);
        workerStatus2.setIp("127.0.0.1");
        workerStatus2.setPort(8081);
        registry.ensureDecodeEndpoint(matching, workerStatus1);
        registry.ensureDecodeEndpoint(filtered, workerStatus2);
        AtomicInteger visited = new AtomicInteger();

        int count = engineWorkerStatus.forEachModelWorkerEndpoint(
                RoleType.DECODE, "group1", (ipPort, endpoint) -> {
                    assertEquals(matching, ipPort);
                    visited.incrementAndGet();
                });

        assertEquals(1, count);
        assertEquals(1, visited.get());
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getDecodeStatusMap().remove(filtered);
        assertEquals(2, engineWorkerStatus.getModelWorkerCapacity(RoleType.DECODE));
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getDecodeStatusMap().clear();
    }

    @Test
    void should_resolve_pd_fusion_and_vit_from_role_specific_registries() {
        assertRoleEndpoint(RoleType.PDFUSION, "127.0.0.1:8101");
        assertRoleEndpoint(RoleType.VIT, "127.0.0.1:8102");
    }

    private void assertRoleEndpoint(RoleType roleType, String ipPort) {
        WorkerStatus status = new WorkerStatus();
        status.setRole(roleType);
        status.setIp("127.0.0.1");
        status.setPort(Integer.parseInt(ipPort.substring(ipPort.lastIndexOf(':') + 1)));
        status.setAlive(true);
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getRoleStatusMap(roleType).put(ipPort, status);
        var registered = registry.ensureEndpoint(roleType, ipPort, status);

        var selected = engineWorkerStatus.selectModelWorkerStatus(roleType, null);

        assertEquals(1, selected.size());
        assertSame(registered, selected.get(ipPort));
    }
}
