package org.flexlb.sync.status;

import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ModelWorkerStatusTest {

    private ModelWorkerStatus modelWorkerStatus;

    @BeforeEach
    void setUp() {
        modelWorkerStatus = new ModelWorkerStatus();
    }

    @Test
    void should_return_pdfusion_status_map_for_pdfusion_role() {
        assertRoleMap(
                RoleType.PDFUSION,
                modelWorkerStatus.getPdFusionStatusMap(),
                "127.0.0.1:8080");
    }

    @Test
    void should_return_prefill_status_map_for_prefill_role() {
        assertRoleMap(
                RoleType.PREFILL,
                modelWorkerStatus.getPrefillStatusMap(),
                "127.0.0.1:8081");
    }

    @Test
    void should_return_decode_status_map_for_decode_role() {
        assertRoleMap(
                RoleType.DECODE,
                modelWorkerStatus.getDecodeStatusMap(),
                "127.0.0.1:8082");
    }

    @Test
    void should_return_vit_status_map_for_vit_role() {
        assertRoleMap(
                RoleType.VIT,
                modelWorkerStatus.getVitStatusMap(),
                "127.0.0.1:8083");
    }

    @Test
    void should_return_frontend_status_map_for_frontend_role() {
        assertRoleMap(
                RoleType.FRONTEND,
                modelWorkerStatus.getFrontendStatusMap(),
                "127.0.0.1:8084");
    }

    @Test
    void null_role_returns_immutable_empty_projection() {
        Map<String, WorkerStatus> result =
                modelWorkerStatus.getRoleStatusMap(null);

        assertTrue(result.isEmpty());
        assertThrows(UnsupportedOperationException.class,
                () -> result.put("127.0.0.1:8080",
                        status(RoleType.VIT, "group", 8080)));
    }

    @Test
    void role_maps_are_independent_exact_owners() {
        WorkerStatus prefill = status(
                RoleType.PREFILL, "group1", 8080);
        WorkerStatus decode = status(
                RoleType.DECODE, "group2", 8081);
        modelWorkerStatus.getPrefillStatusMap()
                .put(prefill.getIpPort(), prefill);
        modelWorkerStatus.getDecodeStatusMap()
                .put(decode.getIpPort(), decode);

        assertSame(prefill, modelWorkerStatus
                .getRoleStatusMap(RoleType.PREFILL)
                .get(prefill.getIpPort()));
        assertSame(decode, modelWorkerStatus
                .getRoleStatusMap(RoleType.DECODE)
                .get(decode.getIpPort()));
        assertNotSame(
                modelWorkerStatus.getRoleStatusMap(RoleType.PREFILL),
                modelWorkerStatus.getRoleStatusMap(RoleType.DECODE));
        assertTrue(modelWorkerStatus
                .getRoleStatusMap(RoleType.PDFUSION).isEmpty());
    }

    @Test
    void should_return_total_count_across_all_role_maps() {
        modelWorkerStatus.getPdFusionStatusMap().put(
                "127.0.0.1:8080",
                status(RoleType.PDFUSION, "group1", 8080));
        modelWorkerStatus.getDecodeStatusMap().put(
                "127.0.0.1:8081",
                status(RoleType.DECODE, "group2", 8081));
        modelWorkerStatus.getPrefillStatusMap().put(
                "127.0.0.1:8082",
                status(RoleType.PREFILL, "group1", 8082));
        modelWorkerStatus.getFrontendStatusMap().put(
                "127.0.0.1:8083",
                status(RoleType.FRONTEND, "group1", 8083));

        assertEquals(4, modelWorkerStatus.getWorkerTotalCount());
    }

    @Test
    void should_return_zero_total_count_when_all_maps_are_empty() {
        assertEquals(0, modelWorkerStatus.getWorkerTotalCount());
    }

    private void assertRoleMap(
            RoleType role,
            Map<String, WorkerStatus> concreteMap,
            String ipPort) {
        int port = Integer.parseInt(
                ipPort.substring(ipPort.lastIndexOf(':') + 1));
        WorkerStatus status = status(role, "testGroup", port);
        concreteMap.put(ipPort, status);

        Map<String, WorkerStatus> selected =
                modelWorkerStatus.getRoleStatusMap(role);

        assertSame(concreteMap, selected);
        assertEquals(1, selected.size());
        assertSame(status, selected.get(ipPort));
    }

    private static WorkerStatus status(
            RoleType role, String group, int port) {
        return WorkerStatus.createDiscovered(
                role, group, "127.0.0.1", port, port + 1, "test-site");
    }
}
