package org.flexlb.sync.status;

import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

class EngineWorkerStatusTest {

    private EngineWorkerStatus engineWorkerStatus;
    private WorkerStatus workerStatus1;
    private WorkerStatus workerStatus2;

    @BeforeEach
    void setUp() {
        engineWorkerStatus = new EngineWorkerStatus(new ModelMetaConfig());
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
        EngineWorkerStatus status = new EngineWorkerStatus(config);

        // Then
        assertNotNull(status);
        assertSame(config, status.getModelMetaConfig());
    }

    @Test
    void should_return_filtered_worker_status_when_selecting_model_worker_status_with_group_filter() {
        // Given
        String modelName = "testModel";
        String ipPort1 = "127.0.0.1:8080";
        String ipPort2 = "127.0.0.1:8081";

        ModelWorkerStatus modelStatus = new ModelWorkerStatus();
        modelStatus.getDecodeStatusMap().put(ipPort1, workerStatus1); // group1
        modelStatus.getDecodeStatusMap().put(ipPort2, workerStatus2); // group2
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS_MAP.put(modelName, modelStatus);

        // When
        var result = engineWorkerStatus.selectModelWorkerStatus(modelName, RoleType.DECODE, "group1");

        // Then
        assertNotNull(result);
        assertEquals(1, result.size());
        assertTrue(result.containsKey(ipPort1));
        assertFalse(result.containsKey(ipPort2));
    }

    @Test
    void should_return_all_worker_status_when_selecting_model_worker_status_without_group_filter() {
        // Given
        String modelName = "testModel";
        String ipPort1 = "127.0.0.1:8080";
        String ipPort2 = "127.0.0.1:8081";

        ModelWorkerStatus modelStatus = new ModelWorkerStatus();
        modelStatus.getPrefillStatusMap().put(ipPort1, workerStatus1);
        modelStatus.getPrefillStatusMap().put(ipPort2, workerStatus2);
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS_MAP.put(modelName, modelStatus);

        // When
        var result = engineWorkerStatus.selectModelWorkerStatus(modelName, RoleType.PREFILL, null);

        // Then
        assertNotNull(result);
        assertEquals(2, result.size());
        assertTrue(result.containsKey(ipPort1));
        assertTrue(result.containsKey(ipPort2));
    }

    @Test
    void should_return_empty_map_when_selecting_model_worker_status_with_nonexistent_model() {
        // Given - no model in the map

        // When
        var result = engineWorkerStatus.selectModelWorkerStatus("nonExistentModel", RoleType.PDFUSION, null);

        // Then
        assertNotNull(result);
        assertTrue(result.isEmpty());
    }

    @Test
    void should_return_empty_map_after_group_filtering_when_no_matching_group_exists() {
        // Given
        String modelName = "testModel";
        String ipPort = "127.0.0.1:8080";

        ModelWorkerStatus modelStatus = new ModelWorkerStatus();
        modelStatus.getVitStatusMap().put(ipPort, workerStatus1); // group1
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS_MAP.put(modelName, modelStatus);

        // When
        var result = engineWorkerStatus.selectModelWorkerStatus(modelName, RoleType.VIT, "nonExistentGroup");

        // Then
        assertNotNull(result);
        assertTrue(result.isEmpty());
    }
}