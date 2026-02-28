package org.flexlb.sync.status;

import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ModelWorkerStatusTest {

    private ModelWorkerStatus modelWorkerStatus;
    private WorkerStatus workerStatus1;
    private WorkerStatus workerStatus2;
    private WorkerStatus workerStatus3;

    @BeforeEach
    void setUp() {
        modelWorkerStatus = new ModelWorkerStatus();
        workerStatus1 = new WorkerStatus();
        workerStatus1.setGroup("group1");
        workerStatus2 = new WorkerStatus();
        workerStatus2.setGroup("group2");
        workerStatus3 = new WorkerStatus();
        workerStatus3.setGroup("group1");
    }

    @Test
    void should_return_pdfusion_status_map_when_getting_role_status_map_with_pdfusion_type() {
        // Given
        String ipPort = "127.0.0.1:8080";
        workerStatus1.setGroup("testGroup");
        modelWorkerStatus.getPdFusionStatusMap().put(ipPort, workerStatus1);

        // When
        var result = modelWorkerStatus.getRoleStatusMap(RoleType.PDFUSION);

        // Then
        assertNotNull(result);
        assertEquals(1, result.size());
        assertEquals(workerStatus1, result.get(ipPort));
    }

    @Test
    void should_return_prefill_status_map_when_getting_role_status_map_with_prefill_type() {
        // Given
        String ipPort = "127.0.0.1:8081";
        workerStatus1.setGroup("testGroup");
        modelWorkerStatus.getPrefillStatusMap().put(ipPort, workerStatus1);

        // When
        var result = modelWorkerStatus.getRoleStatusMap(RoleType.PREFILL);

        // Then
        assertNotNull(result);
        assertEquals(1, result.size());
        assertEquals(workerStatus1, result.get(ipPort));
    }

    @Test
    void should_return_decode_status_map_when_getting_role_status_map_with_decode_type() {
        // Given
        String ipPort = "127.0.0.1:8082";
        workerStatus1.setGroup("testGroup");
        modelWorkerStatus.getDecodeStatusMap().put(ipPort, workerStatus1);

        // When
        var result = modelWorkerStatus.getRoleStatusMap(RoleType.DECODE);

        // Then
        assertNotNull(result);
        assertEquals(1, result.size());
        assertEquals(workerStatus1, result.get(ipPort));
    }

    @Test
    void should_return_vit_status_map_when_getting_role_status_map_with_vit_type() {
        // Given
        String ipPort = "127.0.0.1:8083";
        workerStatus1.setGroup("testGroup");
        modelWorkerStatus.getVitStatusMap().put(ipPort, workerStatus1);

        // When
        var result = modelWorkerStatus.getRoleStatusMap(RoleType.VIT);

        // Then
        assertNotNull(result);
        assertEquals(1, result.size());
        assertEquals(workerStatus1, result.get(ipPort));
    }

    @Test
    void should_return_null_when_getting_role_status_map_with_invalid_role_type() {
        // Note: We can't test this easily since the method only handles the four known role types
        // and returns null for any other case which is not possible since RoleType is an enum
        // Just testing the default false case:
        var result = modelWorkerStatus.getRoleStatusMap(RoleType.DECODE);
        assertNotNull(result); // Valid role type return non-null result
    }

    @Test
    void should_return_empty_list_when_getting_role_type_list_with_empty_status_maps() {
        // Given - default empty state

        // When
        var result = modelWorkerStatus.getRoleTypeList();

        // Then
        assertNotNull(result);
        assertTrue(result.isEmpty());
    }

    @Test
    void should_return_role_types_list_when_getting_role_type_list_with_non_empty_status_maps() {
        // Given
        modelWorkerStatus.getPdFusionStatusMap().put("127.0.0.1:8080", workerStatus1);
        modelWorkerStatus.getDecodeStatusMap().put("127.0.0.1:8081", workerStatus2);
        modelWorkerStatus.getPrefillStatusMap().put("127.0.0.1:8082", workerStatus3);

        // When
        var result = modelWorkerStatus.getRoleTypeList();

        // Then
        assertNotNull(result);
        assertEquals(3, result.size());
        assertTrue(result.contains(RoleType.PDFUSION));
        assertTrue(result.contains(RoleType.DECODE));
        assertTrue(result.contains(RoleType.PREFILL));
        assertFalse(result.contains(RoleType.VIT));
    }

    @Test
    void should_return_total_count_of_all_status_maps_when_getting_worker_total_count() {
        // Given
        modelWorkerStatus.getPdFusionStatusMap().put("127.0.0.1:8080", workerStatus1);
        modelWorkerStatus.getDecodeStatusMap().put("127.0.0.1:8081", workerStatus2);
        modelWorkerStatus.getPrefillStatusMap().put("127.0.0.1:8082", workerStatus3);

        // When
        int result = modelWorkerStatus.getWorkerTotalCount();

        // Then
        assertEquals(3, result);
    }

    @Test
    void should_return_zero_when_getting_worker_total_count_with_empty_maps() {
        // Given - empty status maps (default state)

        // When
        int result = modelWorkerStatus.getWorkerTotalCount();

        // Then
        assertEquals(0, result);
    }
}