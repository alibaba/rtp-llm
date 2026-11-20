package org.flexlb.worker.status;

import org.flexlb.config.ConfigService;
import org.flexlb.config.service.RoleType;
import org.flexlb.worker.endpoint.DecodeEndpoint;
import org.flexlb.worker.endpoint.EndpointRegistry;
import org.flexlb.worker.endpoint.PrefillEndpoint;
import org.flexlb.worker.endpoint.WorkerEndpoint;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import java.util.List;
import java.util.Map;

import static org.flexlb.worker.endpoint.EndpointTestSupport.decodeRoutingSnapshot;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Checks endpoint snapshots apply role and group filters and only expose published capacity from the matching worker generation.
 */
class EndpointDiscoveryTest {
    private EndpointRegistry registry;

    @BeforeEach
    void setUp() {
        ConfigService configService = Mockito.mock(ConfigService.class);
        Mockito.when(configService.loadBalanceConfig())
                .thenReturn(org.flexlb.balance.schedule.SchedulingTestConfig.newConfig());
        registry = RunnerTestSupport.endpointRegistry(configService);
    }

    @Test
    void should_capture_only_endpoint_generation_matching_group() {
        WorkerStatus matching = status(RoleType.DECODE, "group1", 8080);
        WorkerStatus filtered = status(RoleType.DECODE, "group2", 8081);
        RunnerTestSupport.publishEndpoint(
                registry, RoleType.DECODE, matching.getLogicalIpPort(), matching);
        RunnerTestSupport.publishEndpoint(
                registry, RoleType.DECODE, filtered.getLogicalIpPort(), filtered);

        List<DecodeEndpoint.SelectionSnapshot> result = decodeRoutingSnapshot(registry, "group1");
        assertEquals(1, result.size());
        {
            WorkerEndpoint selectedEndpoint =
                    registry.getEndpoint(
                            org.flexlb.config.service.RoleType.DECODE, result.getFirst().address());

            assertNotNull(selectedEndpoint);
            assertSame(matching, selectedEndpoint.getStatus());
            assertEquals(matching.getLogicalIpPort(), selectedEndpoint.ipPort());
        }
    }

    @Test
    void should_capture_all_endpoint_generations_without_group_filter() {
        WorkerStatus first = status(RoleType.PREFILL, "group1", 8080);
        WorkerStatus second = status(RoleType.PREFILL, "group2", 8081);
        RunnerTestSupport.publishEndpoint(
                registry, RoleType.PREFILL, first.getLogicalIpPort(), first);
        RunnerTestSupport.publishEndpoint(
                registry, RoleType.PREFILL, second.getLogicalIpPort(), second);

        List<PrefillEndpoint> result =
                registry.<org.flexlb.worker.endpoint.PrefillEndpoint>endpoints(
                        RoleType.PREFILL, null);
        assertEquals(2, result.size());
        assertTrue(result.stream().anyMatch(entry -> entry.getStatus() == first));
        assertTrue(result.stream().anyMatch(entry -> entry.getStatus() == second));
        for (PrefillEndpoint entry : result) {
            {
                WorkerEndpoint selectedEndpoint =
                        registry.getEndpoint(RoleType.PREFILL, entry.ipPort());

                assertNotNull(selectedEndpoint);
                assertSame(entry, selectedEndpoint);
            }
        }
    }

    @Test
    void should_return_empty_snapshot_when_role_registry_is_empty() {
        assertTrue(
                registry.<org.flexlb.worker.endpoint.PrefillEndpoint>endpoints(
                                RoleType.PDFUSION, null)
                        .isEmpty());
        assertTrue(
                registry.endpoints(RoleType.PDFUSION, null).stream()
                        .map(org.flexlb.worker.endpoint.WorkerEndpoint::ipPort)
                        .toList()
                        .isEmpty());
    }

    @Test
    void should_return_no_decode_candidates_when_no_group_matches() {
        WorkerStatus status = status(RoleType.DECODE, "group1", 8080);
        RunnerTestSupport.publishEndpoint(
                registry, RoleType.DECODE, status.getLogicalIpPort(), status);

        assertTrue(decodeRoutingSnapshot(registry, "nonExistentGroup").isEmpty());
        assertEquals(List.of(status.getLogicalIpPort()), decodeAddresses(null));
    }

    @Test
    void should_exclude_null_group_endpoint_when_group_is_specified() {
        WorkerStatus matching = status(RoleType.DECODE, "groupA", 8080);
        WorkerStatus ungrouped = status(RoleType.DECODE, null, 8081);
        RunnerTestSupport.publishEndpoint(
                registry, RoleType.DECODE, matching.getLogicalIpPort(), matching);
        RunnerTestSupport.publishEndpoint(
                registry, RoleType.DECODE, ungrouped.getLogicalIpPort(), ungrouped);

        List<String> result = decodeAddresses("groupA");

        assertEquals(List.of(matching.getLogicalIpPort()), result);
        assertFalse(result.contains(ungrouped.getLogicalIpPort()));
    }

    @Test
    void should_include_grouped_and_ungrouped_endpoints_without_filter() {
        WorkerStatus grouped = status(RoleType.DECODE, "groupA", 8080);
        WorkerStatus ungrouped = status(RoleType.DECODE, null, 8081);
        RunnerTestSupport.publishEndpoint(
                registry, RoleType.DECODE, grouped.getLogicalIpPort(), grouped);
        RunnerTestSupport.publishEndpoint(
                registry, RoleType.DECODE, ungrouped.getLogicalIpPort(), ungrouped);

        List<String> result = decodeAddresses(null);

        assertEquals(2, result.size());
        assertTrue(result.contains(grouped.getLogicalIpPort()));
        assertTrue(result.contains(ungrouped.getLogicalIpPort()));
    }

    @Test
    void routing_capacity_comes_only_from_published_endpoints() {
        WorkerStatus matching = status(RoleType.DECODE, "group1", 8080);
        WorkerStatus filtered = status(RoleType.DECODE, "group2", 8081);
        discover(matching);
        discover(filtered);
        RunnerTestSupport.publishEndpoint(
                registry, RoleType.DECODE, matching.getLogicalIpPort(), matching);
        RunnerTestSupport.publishEndpoint(
                registry, RoleType.DECODE, filtered.getLogicalIpPort(), filtered);

        assertEquals(List.of(matching.getLogicalIpPort()), decodeAddresses("group1"));
        assertEquals(2, registry.counts(RoleType.DECODE).published());
    }

    @Test
    void should_capture_pd_fusion_and_vit_from_role_specific_registries() {
        assertRoleEndpoint(RoleType.PDFUSION, 8101);
        assertRoleEndpoint(RoleType.VIT, 8102);
    }

    @Test
    void role_status_maps_are_independent_and_counted_once() {
        WorkerStatus prefill = status(RoleType.PREFILL, "group1", 8201);
        WorkerStatus decode = status(RoleType.DECODE, "group2", 8202);
        discover(prefill);
        discover(decode);

        assertSame(
                prefill, registry.statusSnapshot(RoleType.PREFILL).get(prefill.getLogicalIpPort()));
        assertSame(decode, registry.statusSnapshot(RoleType.DECODE).get(decode.getLogicalIpPort()));
        assertEquals(2, registry.discoveredCount());
        assertEquals(1, registry.counts(RoleType.PREFILL).discovered());
    }

    @Test
    void status_snapshot_is_immutable() {
        WorkerStatus status = status(RoleType.VIT, "group", 8080);
        discover(status);

        assertEquals(
                Map.of(status.getLogicalIpPort(), status), registry.statusSnapshot(RoleType.VIT));
        assertThrows(
                UnsupportedOperationException.class,
                () -> registry.statusSnapshot(RoleType.VIT).clear());
    }

    private List<String> decodeAddresses(String group) {
        return decodeRoutingSnapshot(registry, group).stream()
                .map(DecodeEndpoint.SelectionSnapshot::address)
                .toList();
    }

    private void assertRoleEndpoint(RoleType role, int port) {
        WorkerStatus status = status(role, null, port);
        WorkerEndpoint registered =
                RunnerTestSupport.publishEndpoint(
                        registry, role, status.getLogicalIpPort(), status);

        assertEquals(
                List.of(status.getLogicalIpPort()),
                registry.endpoints(role, null).stream()
                        .map(org.flexlb.worker.endpoint.WorkerEndpoint::ipPort)
                        .toList());
        {
            WorkerEndpoint selectedEndpoint = registry.getEndpoint(role, status.getLogicalIpPort());

            assertNotNull(selectedEndpoint);
            assertSame(registered, selectedEndpoint);
        }
    }

    private static WorkerStatus status(RoleType role, String group, int port) {
        return RunnerTestSupport.discovered(role, group, "127.0.0.1", port, port + 1, "test-site");
    }

    private void discover(WorkerStatus status) {
        assertSame(
                status,
                registry.currentOrDiscover(
                        status.getRole(), status.getLogicalIpPort(), () -> status));
    }
}
