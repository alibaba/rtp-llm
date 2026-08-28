package org.flexlb.sync.status;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.sync.runner.RunnerTestSupport;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class WorkerDirectoryTest {

    private WorkerDirectory workerDirectory;
    private EndpointRegistry registry;

    @BeforeEach
    void setUp() {
        ConfigService configService = Mockito.mock(ConfigService.class);
        Mockito.when(configService.loadBalanceConfig())
                .thenReturn(new FlexlbConfig());
        registry = RunnerTestSupport.endpointRegistry(configService);
        workerDirectory = new WorkerDirectory(registry);
    }

    @Test
    void should_capture_only_endpoint_generation_matching_group() {
        WorkerStatus matching = status(RoleType.DECODE, "group1", 8080);
        WorkerStatus filtered = status(RoleType.DECODE, "group2", 8081);
        registry.registerPreinitializedEndpoint(
                RoleType.DECODE, matching.getIpPort(), matching);
        registry.registerPreinitializedEndpoint(
                RoleType.DECODE, filtered.getIpPort(), filtered);

        List<WorkerEndpoint.GenerationPin> result =
                workerDirectory.captureEndpoints(
                        RoleType.DECODE, "group1");
        try {
            assertEquals(1, result.size());
            assertSame(matching, result.getFirst().endpoint().getStatus());
            assertEquals(matching.getIpPort(),
                    result.getFirst().endpoint().ipPort());
        } finally {
            closePins(result);
        }
    }

    @Test
    void should_capture_all_endpoint_generations_without_group_filter() {
        WorkerStatus first = status(RoleType.PREFILL, "group1", 8080);
        WorkerStatus second = status(RoleType.PREFILL, "group2", 8081);
        registry.registerPreinitializedEndpoint(
                RoleType.PREFILL, first.getIpPort(), first);
        registry.registerPreinitializedEndpoint(
                RoleType.PREFILL, second.getIpPort(), second);

        List<WorkerEndpoint.GenerationPin> result =
                workerDirectory.captureEndpoints(
                        RoleType.PREFILL, null);
        try {
            assertEquals(2, result.size());
            assertTrue(result.stream().anyMatch(
                    pin -> pin.endpoint().getStatus() == first));
            assertTrue(result.stream().anyMatch(
                    pin -> pin.endpoint().getStatus() == second));
        } finally {
            closePins(result);
        }
    }

    @Test
    void should_return_empty_capture_when_role_registry_is_empty() {
        List<WorkerEndpoint.GenerationPin> result =
                workerDirectory.captureEndpoints(
                        RoleType.PDFUSION, null);

        assertNotNull(result);
        assertTrue(result.isEmpty());
    }

    @Test
    void should_close_filtered_pins_when_no_group_matches() {
        WorkerStatus status = status(RoleType.VIT, "group1", 8080);
        registry.registerPreinitializedEndpoint(
                RoleType.VIT, status.getIpPort(), status);

        List<WorkerEndpoint.GenerationPin> result =
                workerDirectory.captureEndpoints(
                        RoleType.VIT, "nonExistentGroup");

        assertTrue(result.isEmpty());
        assertEquals(List.of(status.getIpPort()),
                workerDirectory.endpointAddresses(
                        RoleType.VIT, null));
    }

    @Test
    void should_exclude_null_group_endpoint_when_group_is_specified() {
        WorkerStatus matching = status(RoleType.DECODE, "groupA", 8080);
        WorkerStatus ungrouped = status(RoleType.DECODE, null, 8081);
        registry.registerPreinitializedEndpoint(
                RoleType.DECODE, matching.getIpPort(), matching);
        registry.registerPreinitializedEndpoint(
                RoleType.DECODE, ungrouped.getIpPort(), ungrouped);

        List<String> result = workerDirectory.endpointAddresses(
                RoleType.DECODE, "groupA");

        assertEquals(List.of(matching.getIpPort()), result);
        assertFalse(result.contains(ungrouped.getIpPort()));
    }

    @Test
    void should_include_grouped_and_ungrouped_endpoints_without_filter() {
        WorkerStatus grouped = status(RoleType.DECODE, "groupA", 8080);
        WorkerStatus ungrouped = status(RoleType.DECODE, null, 8081);
        registry.registerPreinitializedEndpoint(
                RoleType.DECODE, grouped.getIpPort(), grouped);
        registry.registerPreinitializedEndpoint(
                RoleType.DECODE, ungrouped.getIpPort(), ungrouped);

        List<String> result = workerDirectory.endpointAddresses(
                RoleType.DECODE, null);

        assertEquals(2, result.size());
        assertTrue(result.contains(grouped.getIpPort()));
        assertTrue(result.contains(ungrouped.getIpPort()));
    }

    @Test
    void routing_capacity_comes_only_from_published_endpoints() {
        WorkerStatus matching = status(RoleType.DECODE, "group1", 8080);
        WorkerStatus filtered = status(RoleType.DECODE, "group2", 8081);
        var statusMap = workerDirectory.statusMap(RoleType.DECODE);
        statusMap.put(matching.getIpPort(), matching);
        statusMap.put(filtered.getIpPort(), filtered);
        registry.registerPreinitializedEndpoint(
                RoleType.DECODE, matching.getIpPort(), matching);
        registry.registerPreinitializedEndpoint(
                RoleType.DECODE, filtered.getIpPort(), filtered);

        assertEquals(List.of(matching.getIpPort()),
                workerDirectory.endpointAddresses(
                        RoleType.DECODE, "group1"));
        statusMap.remove(filtered.getIpPort());
        assertEquals(2,
                workerDirectory.routingCapacity(RoleType.DECODE));
    }

    @Test
    void should_capture_pd_fusion_and_vit_from_role_specific_registries() {
        assertRoleEndpoint(RoleType.PDFUSION, 8101);
        assertRoleEndpoint(RoleType.VIT, 8102);
    }

    private void assertRoleEndpoint(RoleType role, int port) {
        WorkerStatus status = status(role, null, port);
        WorkerEndpoint registered = registry.registerPreinitializedEndpoint(
                role, status.getIpPort(), status);

        List<WorkerEndpoint.GenerationPin> selected =
                workerDirectory.captureEndpoints(role, null);
        try {
            assertEquals(1, selected.size());
            assertSame(registered, selected.getFirst().endpoint());
        } finally {
            closePins(selected);
        }
    }

    private static WorkerStatus status(
            RoleType role, String group, int port) {
        return RunnerTestSupport.alive(
                role, group, "127.0.0.1", port, port + 1, "test-site");
    }

    private static void closePins(
            List<WorkerEndpoint.GenerationPin> pins) {
        for (WorkerEndpoint.GenerationPin pin : pins) {
            pin.close();
        }
    }

    @Test
    void role_status_maps_are_independent_and_counted_once() {
        WorkerStatus prefill = status(RoleType.PREFILL, "group1", 8201);
        WorkerStatus decode = status(RoleType.DECODE, "group2", 8202);
        workerDirectory.statusMap(RoleType.PREFILL)
                .put(prefill.getIpPort(), prefill);
        workerDirectory.statusMap(RoleType.DECODE)
                .put(decode.getIpPort(), decode);

        assertSame(prefill, workerDirectory.statusMap(RoleType.PREFILL)
                .get(prefill.getIpPort()));
        assertSame(decode, workerDirectory.statusMap(RoleType.DECODE)
                .get(decode.getIpPort()));
        assertNotSame(workerDirectory.statusMap(RoleType.PREFILL),
                workerDirectory.statusMap(RoleType.DECODE));
        assertEquals(2, workerDirectory.discoveredCount());
        assertEquals(1, workerDirectory.discoveredCount(RoleType.PREFILL));
    }

    @Test
    void null_role_has_no_mutable_status_owner() {
        Map<String, WorkerStatus> result = workerDirectory.statusMap(null);

        assertTrue(result.isEmpty());
        assertThrows(UnsupportedOperationException.class,
                () -> result.put("127.0.0.1:8080",
                        status(RoleType.VIT, "group", 8080)));
    }
}
