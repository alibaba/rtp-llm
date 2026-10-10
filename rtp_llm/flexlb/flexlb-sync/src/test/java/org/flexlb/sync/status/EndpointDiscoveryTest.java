package org.flexlb.sync.status;

import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.config.ConfigService;
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
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class EndpointDiscoveryTest {

    private EndpointRegistry registry;

    @BeforeEach
    void setUp() {
        ConfigService configService = Mockito.mock(ConfigService.class);
        Mockito.when(configService.loadBalanceConfig())
                .thenReturn(org.flexlb.balance.scheduler.SchedulingTestConfig.newConfig());
        registry = RunnerTestSupport.endpointRegistry(configService);
    }

    @Test
    void should_capture_only_endpoint_generation_matching_group() {
        WorkerStatus matching = status(RoleType.DECODE, "group1", 8080);
        WorkerStatus filtered = status(RoleType.DECODE, "group2", 8081);
        RunnerTestSupport.publishEndpoint(registry,
                RoleType.DECODE, matching.getIpPort(), matching);
        RunnerTestSupport.publishEndpoint(registry,
                RoleType.DECODE, filtered.getIpPort(), filtered);

        List<DecodeResources.DecodeRoutingView> result =
                registry.decodeRoutingSnapshot("group1");
        assertEquals(1, result.size());
        try (WorkerEndpoint.GenerationPin pin =
                     registry.captureDecodeGeneration(result.getFirst())) {
            assertNotNull(pin);
            assertSame(matching, pin.endpoint().getStatus());
            assertEquals(matching.getIpPort(), pin.endpoint().ipPort());
        }
    }

    @Test
    void should_capture_all_endpoint_generations_without_group_filter() {
        WorkerStatus first = status(RoleType.PREFILL, "group1", 8080);
        WorkerStatus second = status(RoleType.PREFILL, "group2", 8081);
        RunnerTestSupport.publishEndpoint(registry,
                RoleType.PREFILL, first.getIpPort(), first);
        RunnerTestSupport.publishEndpoint(registry,
                RoleType.PREFILL, second.getIpPort(), second);

        List<EndpointRegistry.PrefillRoutingEntry> result =
                registry.prefillRoutingSnapshot(RoleType.PREFILL);
        assertEquals(2, result.size());
        assertTrue(result.stream().anyMatch(entry -> entry.endpoint().getStatus() == first));
        assertTrue(result.stream().anyMatch(entry -> entry.endpoint().getStatus() == second));
        for (EndpointRegistry.PrefillRoutingEntry entry : result) {
            try (WorkerEndpoint.GenerationPin pin =
                         registry.capture(RoleType.PREFILL, entry.address())) {
                assertNotNull(pin);
                assertSame(entry.endpoint(), pin.endpoint());
            }
        }
    }

    @Test
    void should_return_empty_snapshot_when_role_registry_is_empty() {
        assertTrue(registry.prefillRoutingSnapshot(RoleType.PDFUSION).isEmpty());
        assertTrue(registry.endpointAddressSnapshot(RoleType.PDFUSION).isEmpty());
    }

    @Test
    void should_return_no_decode_candidates_when_no_group_matches() {
        WorkerStatus status = status(RoleType.DECODE, "group1", 8080);
        RunnerTestSupport.publishEndpoint(registry,
                RoleType.DECODE, status.getIpPort(), status);

        assertTrue(registry.decodeRoutingSnapshot("nonExistentGroup").isEmpty());
        assertEquals(List.of(status.getIpPort()), decodeAddresses(null));
    }

    @Test
    void should_exclude_null_group_endpoint_when_group_is_specified() {
        WorkerStatus matching = status(RoleType.DECODE, "groupA", 8080);
        WorkerStatus ungrouped = status(RoleType.DECODE, null, 8081);
        RunnerTestSupport.publishEndpoint(registry,
                RoleType.DECODE, matching.getIpPort(), matching);
        RunnerTestSupport.publishEndpoint(registry,
                RoleType.DECODE, ungrouped.getIpPort(), ungrouped);

        List<String> result = decodeAddresses("groupA");

        assertEquals(List.of(matching.getIpPort()), result);
        assertFalse(result.contains(ungrouped.getIpPort()));
    }

    @Test
    void should_include_grouped_and_ungrouped_endpoints_without_filter() {
        WorkerStatus grouped = status(RoleType.DECODE, "groupA", 8080);
        WorkerStatus ungrouped = status(RoleType.DECODE, null, 8081);
        RunnerTestSupport.publishEndpoint(registry,
                RoleType.DECODE, grouped.getIpPort(), grouped);
        RunnerTestSupport.publishEndpoint(registry,
                RoleType.DECODE, ungrouped.getIpPort(), ungrouped);

        List<String> result = decodeAddresses(null);

        assertEquals(2, result.size());
        assertTrue(result.contains(grouped.getIpPort()));
        assertTrue(result.contains(ungrouped.getIpPort()));
    }

    @Test
    void routing_capacity_comes_only_from_published_endpoints() {
        WorkerStatus matching = status(RoleType.DECODE, "group1", 8080);
        WorkerStatus filtered = status(RoleType.DECODE, "group2", 8081);
        discover(matching);
        discover(filtered);
        RunnerTestSupport.publishEndpoint(registry,
                RoleType.DECODE, matching.getIpPort(), matching);
        RunnerTestSupport.publishEndpoint(registry,
                RoleType.DECODE, filtered.getIpPort(), filtered);

        assertEquals(List.of(matching.getIpPort()),
                decodeAddresses("group1"));
        assertEquals(2,
                registry.getEndpointCount(RoleType.DECODE));
    }

    @Test
    void should_capture_pd_fusion_and_vit_from_role_specific_registries() {
        assertRoleEndpoint(RoleType.PDFUSION, 8101);
        assertRoleEndpoint(RoleType.VIT, 8102);
    }

    private List<String> decodeAddresses(String group) {
        return registry.decodeRoutingSnapshot(group).stream()
                .map(DecodeResources.DecodeRoutingView::address).toList();
    }

    private void assertRoleEndpoint(RoleType role, int port) {
        WorkerStatus status = status(role, null, port);
        WorkerEndpoint registered = RunnerTestSupport.publishEndpoint(registry,
                role, status.getIpPort(), status);

        assertEquals(List.of(status.getIpPort()), registry.endpointAddressSnapshot(role));
        try (WorkerEndpoint.GenerationPin pin =
                     registry.capture(role, status.getIpPort())) {
            assertNotNull(pin);
            assertSame(registered, pin.endpoint());
        }
    }

    private static WorkerStatus status(
            RoleType role, String group, int port) {
        return RunnerTestSupport.discovered(
                role, group, "127.0.0.1", port, port + 1, "test-site");
    }

    @Test
    void role_status_maps_are_independent_and_counted_once() {
        WorkerStatus prefill = status(RoleType.PREFILL, "group1", 8201);
        WorkerStatus decode = status(RoleType.DECODE, "group2", 8202);
        discover(prefill);
        discover(decode);

        assertSame(prefill, registry.statusSnapshot(RoleType.PREFILL)
                .get(prefill.getIpPort()));
        assertSame(decode, registry.statusSnapshot(RoleType.DECODE)
                .get(decode.getIpPort()));
        assertEquals(2, registry.discoveredCount());
        assertEquals(1, registry.discoveredCount(RoleType.PREFILL));
    }

    @Test
    void status_snapshot_is_immutable() {
        WorkerStatus status = status(RoleType.VIT, "group", 8080);
        discover(status);

        assertEquals(Map.of(status.getIpPort(), status),
                registry.statusSnapshot(RoleType.VIT));
        assertThrows(UnsupportedOperationException.class,
                () -> registry.statusSnapshot(RoleType.VIT).clear());
    }

    private void discover(WorkerStatus status) {
        assertSame(status, registry.currentOrDiscover(
                status.getRole(), status.getIpPort(), () -> status));
    }
}
