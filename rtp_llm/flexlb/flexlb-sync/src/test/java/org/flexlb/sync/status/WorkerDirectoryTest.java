package org.flexlb.sync.status;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.cache.match.CacheAwareService;
import org.flexlb.config.ConfigService;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusProvider;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.sync.runner.RunnerTestSupport;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import org.mockito.Mockito;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class WorkerDirectoryTest {

    private WorkerDirectory workerDirectory;
    private EndpointRegistry registry;

    private record DecodeMember(WorkerStatus status, WorkerStatusResponse response) {
    }

    private record PrefillMember(WorkerStatus status, WorkerStatusResponse response) {
    }

    @BeforeEach
    void setUp() {
        ConfigService configService = Mockito.mock(ConfigService.class);
        Mockito.when(configService.loadBalanceConfig())
                .thenReturn(org.flexlb.balance.scheduler.SchedulingTestConfig.newConfig());
        registry = RunnerTestSupport.endpointRegistry(configService);
        workerDirectory = new WorkerDirectory(registry);
    }

    @Test
    void should_capture_only_endpoint_generation_matching_group() {
        WorkerStatus matching = status(RoleType.DECODE, "group1", 8080);
        WorkerStatus filtered = status(RoleType.DECODE, "group2", 8081);
        RunnerTestSupport.publishEndpoint(registry,
                RoleType.DECODE, matching.getLogicalIpPort(), matching);
        RunnerTestSupport.publishEndpoint(registry,
                RoleType.DECODE, filtered.getLogicalIpPort(), filtered);

        List<WorkerEndpoint.EndpointHold> result = workerDirectory.holdEndpoints(RoleType.DECODE, "group1");
        try {
            assertEquals(1, result.size());
            assertSame(matching, result.getFirst().endpoint().getStatus());
            assertEquals(matching.getLogicalIpPort(),
                    result.getFirst().endpoint().ipPort());
        } finally {
            closeHolds(result);
        }
    }

    @Test
    void should_capture_all_endpoint_generations_without_group_filter() {
        WorkerStatus first = status(RoleType.PREFILL, "group1", 8080);
        WorkerStatus second = status(RoleType.PREFILL, "group2", 8081);
        RunnerTestSupport.publishEndpoint(registry,
                RoleType.PREFILL, first.getLogicalIpPort(), first);
        RunnerTestSupport.publishEndpoint(registry,
                RoleType.PREFILL, second.getLogicalIpPort(), second);

        List<WorkerEndpoint.EndpointHold> result = workerDirectory.holdEndpoints(RoleType.PREFILL, null);
        try {
            assertEquals(2, result.size());
            assertTrue(result.stream().anyMatch(hold -> hold.endpoint().getStatus() == first));
            assertTrue(result.stream().anyMatch(hold -> hold.endpoint().getStatus() == second));
        } finally {
            closeHolds(result);
        }
    }

    @Test
    void should_return_empty_capture_when_role_registry_is_empty() {
        List<WorkerEndpoint.EndpointHold> result = workerDirectory.holdEndpoints(RoleType.PDFUSION, null);

        assertNotNull(result);
        assertTrue(result.isEmpty());
    }

    @Test
    void should_close_filtered_pins_when_no_group_matches() {
        WorkerStatus status = status(RoleType.VIT, "group1", 8080);
        RunnerTestSupport.publishEndpoint(registry,
                RoleType.VIT, status.getLogicalIpPort(), status);

        List<WorkerEndpoint.EndpointHold> result = workerDirectory.holdEndpoints(RoleType.VIT, "nonExistentGroup");

        assertTrue(result.isEmpty());
        assertEquals(List.of(status.getLogicalIpPort()),
                workerDirectory.endpointAddresses(
                        RoleType.VIT, null));
    }

    @Test
    void should_exclude_null_group_endpoint_when_group_is_specified() {
        WorkerStatus matching = status(RoleType.DECODE, "groupA", 8080);
        WorkerStatus ungrouped = status(RoleType.DECODE, null, 8081);
        RunnerTestSupport.publishEndpoint(registry,
                RoleType.DECODE, matching.getLogicalIpPort(), matching);
        RunnerTestSupport.publishEndpoint(registry,
                RoleType.DECODE, ungrouped.getLogicalIpPort(), ungrouped);

        List<String> result = workerDirectory.endpointAddresses(
                RoleType.DECODE, "groupA");

        assertEquals(List.of(matching.getLogicalIpPort()), result);
        assertFalse(result.contains(ungrouped.getLogicalIpPort()));
    }

    @Test
    void should_include_grouped_and_ungrouped_endpoints_without_filter() {
        WorkerStatus grouped = status(RoleType.DECODE, "groupA", 8080);
        WorkerStatus ungrouped = status(RoleType.DECODE, null, 8081);
        RunnerTestSupport.publishEndpoint(registry,
                RoleType.DECODE, grouped.getLogicalIpPort(), grouped);
        RunnerTestSupport.publishEndpoint(registry,
                RoleType.DECODE, ungrouped.getLogicalIpPort(), ungrouped);

        List<String> result = workerDirectory.endpointAddresses(
                RoleType.DECODE, null);

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
        RunnerTestSupport.publishEndpoint(registry,
                RoleType.DECODE, matching.getLogicalIpPort(), matching);
        RunnerTestSupport.publishEndpoint(registry,
                RoleType.DECODE, filtered.getLogicalIpPort(), filtered);

        assertEquals(List.of(matching.getLogicalIpPort()),
                workerDirectory.endpointAddresses(
                        RoleType.DECODE, "group1"));
        assertEquals(2,
                workerDirectory.routingCapacity(RoleType.DECODE));
    }

    @Test
    void should_capture_pd_fusion_and_vit_from_role_specific_registries() {
        assertRoleEndpoint(RoleType.PDFUSION, 8101);
        assertRoleEndpoint(RoleType.VIT, 8102);
    }

    @Test
    void storesSiblingEnginesUnderTheirDistinctLogicalAddresses() {
        WorkerStatus first = WorkerStatus.createDiscovered(
                RoleType.PREFILL, "group", "127.0.0.1", 8080, 8081,
                "site", null, 0, 2);
        WorkerStatus second = WorkerStatus.createDiscovered(
                RoleType.PREFILL, "group", "127.0.0.1", 8080, 8081,
                "site", null, 1, 2);

        workerDirectory.currentOrDiscover(
                RoleType.PREFILL, first.getLogicalIpPort(), () -> first);
        workerDirectory.currentOrDiscover(
                RoleType.PREFILL, second.getLogicalIpPort(), () -> second);

        Map<String, WorkerStatus> statuses =
                workerDirectory.statusSnapshot(RoleType.PREFILL);
        assertEquals(2, statuses.size());
        assertSame(first, statuses.get("127.0.0.1:8080@0"));
        assertSame(second, statuses.get("127.0.0.1:8080@1"));
    }

    @Test
    void keepsPublishedLogicalEngineWhenSiblingIsNotPublished() {
        WorkerStatus first = WorkerStatus.createDiscovered(
                RoleType.PREFILL, "group", "127.0.0.1", 8080, 8081,
                "site", null, 0, 2);
        WorkerStatus second = WorkerStatus.createDiscovered(
                RoleType.PREFILL, "group", "127.0.0.1", 8080, 8081,
                "site", null, 1, 2);
        discover(first);
        discover(second);
        RunnerTestSupport.publishEndpoint(registry,
                RoleType.PREFILL, first.getLogicalIpPort(), first);

        assertEquals(List.of(first.getLogicalIpPort()),
                workerDirectory.prefillRoutingSnapshot(RoleType.PREFILL)
                        .stream()
                        .map(EndpointRegistry.PrefillRoutingEntry::address)
                        .toList());
    }

    @Test
    void decodeGroupIsRoutableOnlyWhenEveryRankHasAnEndpoint() {
        DecodeMember rankZero = decodeMember("routing-a", "10.0.0.1", 0, 2, 8400);
        DecodeMember rankOne = decodeMember("routing-a", "10.0.0.1", 1, 2, 8401);
        publishDecode(rankZero);

        assertTrue(workerDirectory.decodeRoutingSnapshot("routing-a").isEmpty());
        assertEquals(2, workerDirectory.getWorkerStatuses(RoleType.DECODE, "routing-a").size());

        publishDecode(rankOne);

        assertEquals(2, workerDirectory.decodeRoutingSnapshot("routing-a").size());
    }

    @Test
    void retiringOneRankHidesItsGroupWithoutRemovingWorkerStatuses() {
        DecodeMember rankZero = decodeMember("routing-a", "10.0.0.1", 0, 2, 8410);
        DecodeMember rankOne = decodeMember("routing-a", "10.0.0.1", 1, 2, 8411);
        DecodeMember independent = decodeMember("routing-a", null, 0, 1, 8412);
        publishDecode(rankZero);
        publishDecode(rankOne);
        publishDecode(independent);

        EndpointRegistry.DetachedGeneration detached;
        rankOne.status().lock.lock();
        try {
            detached = workerDirectory.beginRetirement(
                    RoleType.DECODE, rankOne.status().getLogicalIpPort(), rankOne.status());
        } finally {
            rankOne.status().lock.unlock();
        }
        assertNotNull(detached);
        detached.retireAndAwait();

        assertEquals(List.of(independent.status().getLogicalIpPort()),
                workerDirectory.decodeRoutingSnapshot("routing-a").stream().map(view -> view.address()).toList());
        assertEquals(3, workerDirectory.getWorkerStatuses(RoleType.DECODE, "routing-a").size());
        assertSame(rankOne.status(), workerDirectory.statusSnapshot(RoleType.DECODE)
                .get(rankOne.status().getLogicalIpPort()));
    }

    @Test
    void decodeGroupsWithTheSameRanksAreSeparatedByMasterAddress() {
        DecodeMember incomplete = decodeMember("routing-a", "10.0.0.1", 0, 2, 8420);
        DecodeMember completeZero = decodeMember("routing-a", "10.0.0.2", 0, 2, 8421);
        DecodeMember completeOne = decodeMember("routing-a", "10.0.0.2", 1, 2, 8422);
        DecodeMember otherPolicy = decodeMember("routing-b", null, 0, 1, 8423);
        publishDecode(incomplete);
        publishDecode(completeZero);
        publishDecode(completeOne);
        publishDecode(otherPolicy);

        List<String> eligibleAddresses = workerDirectory.decodeRoutingSnapshot("routing-a")
                .stream().map(view -> view.address()).sorted().toList();
        assertEquals(List.of(completeZero.status().getLogicalIpPort(), completeOne.status().getLogicalIpPort())
                .stream().sorted().toList(), eligibleAddresses);
        assertEquals(3, workerDirectory.decodeRoutingSnapshot(null).size());
    }

    @Test
    void duplicateRanksDoNotMakeAGroupRoutable() {
        publishDecode(decodeMember("routing-a", "10.0.0.1", 0, 2, 8430));
        publishDecode(decodeMember("routing-a", "10.0.0.1", 1, 2, 8431));
        publishDecode(decodeMember("routing-a", "10.0.0.1", 1, 2, 8432));

        assertTrue(workerDirectory.decodeRoutingSnapshot("routing-a").isEmpty());
    }

    @Test
    void inconsistentDpSizesDoNotMakeAGroupRoutable() {
        publishDecode(decodeMember("routing-a", "10.0.0.1", 0, 2, 8440));
        publishDecode(decodeMember("routing-a", "10.0.0.1", 1, 3, 8441));
        publishDecode(decodeMember("routing-a", "10.0.0.1", 2, 3, 8442));

        assertTrue(workerDirectory.decodeRoutingSnapshot("routing-a").isEmpty());
    }

    @Test
    void prefillGroupIsRoutableOnlyWhenEveryRankHasAnEndpoint() {
        PrefillMember rankZero = prefillMember("routing-a", "10.0.1.1", 0, 2, 8500);
        PrefillMember rankOne = prefillMember("routing-a", "10.0.1.1", 1, 2, 8501);
        publishPrefill(rankZero);

        assertTrue(workerDirectory.prefillRoutingSnapshot(RoleType.PREFILL).isEmpty());
        assertEquals(2, workerDirectory.getWorkerStatuses(RoleType.PREFILL, "routing-a").size());

        publishPrefill(rankOne);

        assertEquals(2, workerDirectory.prefillRoutingSnapshot(RoleType.PREFILL).size());
    }

    @Test
    void retiringOnePrefillRankRequiresTheWholeGroupToBeRepublished() {
        PrefillMember rankZero = prefillMember("routing-a", "10.0.1.2", 0, 2, 8510);
        PrefillMember rankOne = prefillMember("routing-a", "10.0.1.2", 1, 2, 8511);
        PrefillMember independent = prefillMember("routing-a", null, 0, 1, 8512);
        publishPrefill(rankZero);
        publishPrefill(rankOne);
        publishPrefill(independent);

        assertEquals(3, workerDirectory.prefillRoutingSnapshot(RoleType.PREFILL).size());

        EndpointRegistry.DetachedGeneration detached;
        rankOne.status().lock.lock();
        try {
            detached = workerDirectory.beginRetirement(
                    RoleType.PREFILL, rankOne.status().getLogicalIpPort(), rankOne.status());
        } finally {
            rankOne.status().lock.unlock();
        }
        assertNotNull(detached);
        workerDirectory.completeRetirement(
                RoleType.PREFILL, rankOne.status().getLogicalIpPort(), rankOne.status(),
                detached, Mockito.mock(CacheAwareService.class),
                LoggerFactory.getLogger(WorkerDirectoryTest.class));

        assertEquals(List.of(independent.status().getLogicalIpPort()),
                workerDirectory.prefillRoutingSnapshot(RoleType.PREFILL).stream()
                        .map(EndpointRegistry.PrefillRoutingEntry::address).toList());

        publishPrefill(prefillMember("routing-a", "10.0.1.2", 1, 2, 8511));

        assertEquals(1, workerDirectory.prefillRoutingSnapshot(RoleType.PREFILL).size());

        publishPrefill(prefillMember("routing-a", "10.0.1.2", 0, 2, 8510));

        assertEquals(3, workerDirectory.prefillRoutingSnapshot(RoleType.PREFILL).size());
    }

    @ParameterizedTest
    @EnumSource(value = RoleType.class, names = {"PREFILL", "DECODE", "PDFUSION"})
    void retiresEveryDpMemberBeforeDrainingAndKeepsOtherGroups(RoleType role) {
        WorkerStatus first = publishDpMember(role, "10.0.2.1", 0, 3, 8600);
        WorkerStatus second = publishDpMember(role, "10.0.2.1", 1, 3, 8601);
        WorkerStatus third = publishDpMember(role, "10.0.2.1", 2, 3, 8602);
        WorkerStatus otherGroup = publishDpMember(role, "10.0.2.2", 0, 1, 8603);
        WorkerStatus independent = publishDpMember(role, null, 0, 1, 8604);
        WorkerStatus otherRole =
                publishDpMember(role == RoleType.DECODE ? RoleType.PREFILL : RoleType.DECODE, "10.0.2.1", 0, 1, 8605);
        CacheAwareService cache = Mockito.mock(CacheAwareService.class);

        EndpointRegistry.DetachedGeneration detached;
        first.lock.lock();
        try {
            detached = Mockito.spy(workerDirectory.beginRetirement(role, first.getLogicalIpPort(), first));
        } finally {
            first.lock.unlock();
        }
        AtomicBoolean groupRetiringBeforeDrain = new AtomicBoolean();
        Mockito.doAnswer(invocation -> {
            groupRetiringBeforeDrain.set(List.of(first, second, third)
                    .stream()
                    .allMatch(member -> !member.isActiveGeneration()
                            && workerDirectory.exactEndpoint(role, member.getLogicalIpPort(), member) == null
                            && workerDirectory.statusSnapshot(role).get(member.getLogicalIpPort()) == member
                            && !member.lock.isHeldByCurrentThread()));
            return invocation.callRealMethod();
        }).when(detached).retireAndAwait();

        workerDirectory.completeRetirement(role, first.getLogicalIpPort(), first, detached, cache,
                LoggerFactory.getLogger(WorkerDirectoryTest.class));

        assertTrue(groupRetiringBeforeDrain.get());
        for (WorkerStatus member : List.of(first, second, third)) {
            assertNull(workerDirectory.holdEndpoint(role, member.getLogicalIpPort()));
        }
        assertEquals(Map.of(otherGroup.getLogicalIpPort(), otherGroup, independent.getLogicalIpPort(), independent),
                workerDirectory.statusSnapshot(role));
        assertEquals(2, workerDirectory.routingCapacity(role));
        assertTrue(otherRole.isActiveGeneration());
        assertNotNull(workerDirectory.exactEndpoint(otherRole.getRole(), otherRole.getLogicalIpPort(), otherRole));
        if (role.requiresCacheKeys()) {
            for (WorkerStatus member : List.of(first, second, third)) {
                Mockito.verify(cache).removeEngineBlockCache(member.getLogicalIpPort());
            }
            Mockito.verifyNoMoreInteractions(cache);
        } else {
            Mockito.verifyNoInteractions(cache);
        }
    }

    private WorkerStatus publishDpMember(RoleType role, String masterAddress, long rank, long size, int port) {
        WorkerStatus status = status(role, "routing-a", port);
        discover(status);
        WorkerStatusResponse response = RunnerTestSupport.response(status, true, 1L);
        response.setDpMasterAddress(masterAddress);
        response.setDpRank(rank);
        response.setDpSize(size);
        status.lock.lock();
        try {
            registry.publishPreparedEndpoint(status.getLogicalIpPort(), status,
                    status.prepareNewStatus(status.freezeStatusResponse(response)));
            status.recordSuccessfulPoll(true);
        } finally {
            status.lock.unlock();
        }
        return status;
    }

    @Test
    void prefillGroupsWithTheSameRanksAreSeparatedByMasterAddress() {
        PrefillMember incomplete = prefillMember("routing-a", "10.0.1.3", 0, 2, 8520);
        PrefillMember completeZero = prefillMember("routing-a", "10.0.1.4", 0, 2, 8521);
        PrefillMember completeOne = prefillMember("routing-a", "10.0.1.4", 1, 2, 8522);
        publishPrefill(incomplete);
        publishPrefill(completeZero);
        publishPrefill(completeOne);

        List<String> routableAddresses = workerDirectory.prefillRoutingSnapshot(RoleType.PREFILL)
                .stream().map(EndpointRegistry.PrefillRoutingEntry::address).sorted().toList();
        assertEquals(List.of(completeZero.status().getLogicalIpPort(), completeOne.status().getLogicalIpPort())
                .stream().sorted().toList(), routableAddresses);
    }

    private PrefillMember prefillMember(String group, String masterAddress, long rank, long size, int port) {
        WorkerStatus status = status(RoleType.PREFILL, group, port);
        discover(status);
        WorkerStatusResponse response = RunnerTestSupport.response(status, true, 1L);
        response.setDpMasterAddress(masterAddress);
        response.setDpSize(size);
        response.setDpRank(rank);
        return new PrefillMember(status, response);
    }

    private void publishPrefill(PrefillMember member) {
        WorkerStatus status = member.status();
        status.lock.lock();
        try {
            WorkerStatus.PreparedStatus prepared = status.prepareNewStatus(
                    status.freezeStatusResponse(member.response()));
            registry.publishPreparedEndpoint(status.getLogicalIpPort(), status, prepared);
            status.recordSuccessfulPoll(true);
        } finally {
            status.lock.unlock();
        }
    }

    private DecodeMember decodeMember(String group, String masterAddress, long rank, long size, int port) {
        WorkerStatus status = status(RoleType.DECODE, group, port);
        discover(status);
        WorkerStatusResponse response = RunnerTestSupport.response(status, true, 1L);
        response.setDpMasterAddress(masterAddress);
        response.setDpSize(size);
        response.setDpRank(rank);
        return new DecodeMember(status, response);
    }

    private void publishDecode(DecodeMember member) {
        WorkerStatus status = member.status();
        status.lock.lock();
        try {
            WorkerStatus.PreparedStatus prepared = status.prepareNewStatus(
                    status.freezeStatusResponse(member.response()));
            registry.publishPreparedEndpoint(status.getLogicalIpPort(), status, prepared);
            status.recordSuccessfulPoll(true);
        } finally {
            status.lock.unlock();
        }
    }

    private void assertRoleEndpoint(RoleType role, int port) {
        WorkerStatus status = status(role, null, port);
        WorkerEndpoint registered = RunnerTestSupport.publishEndpoint(registry,
                role, status.getLogicalIpPort(), status);

        List<WorkerEndpoint.EndpointHold> selected = workerDirectory.holdEndpoints(role, null);
        try {
            assertEquals(1, selected.size());
            assertSame(registered, selected.getFirst().endpoint());
        } finally {
            closeHolds(selected);
        }
    }

    private static WorkerStatus status(
            RoleType role, String group, int port) {
        return RunnerTestSupport.discovered(
                role, group, "127.0.0.1", port, port + 1, "test-site");
    }

    private static void closeHolds(List<WorkerEndpoint.EndpointHold> holds) {
        for (WorkerEndpoint.EndpointHold hold : holds) {
            hold.close();
        }
    }

    @Test
    void role_status_maps_are_independent_and_counted_once() {
        WorkerStatus prefill = status(RoleType.PREFILL, "group1", 8201);
        WorkerStatus decode = status(RoleType.DECODE, "group2", 8202);
        discover(prefill);
        discover(decode);

        assertSame(prefill, workerDirectory.statusSnapshot(RoleType.PREFILL)
                .get(prefill.getLogicalIpPort()));
        assertSame(decode, workerDirectory.statusSnapshot(RoleType.DECODE)
                .get(decode.getLogicalIpPort()));
        assertEquals(2, workerDirectory.discoveredCount());
        assertEquals(1, workerDirectory.discoveredCount(RoleType.PREFILL));
    }

    @Test
    void status_snapshot_is_immutable() {
        WorkerStatus status = status(RoleType.VIT, "group", 8080);
        discover(status);

        assertEquals(Map.of(status.getLogicalIpPort(), status),
                workerDirectory.statusSnapshot(RoleType.VIT));
        assertThrows(UnsupportedOperationException.class,
                () -> workerDirectory.statusSnapshot(RoleType.VIT).clear());
    }

    @Test
    void worker_status_provider_returns_discovered_statuses_by_role_and_group() {
        WorkerStatus matching = status(RoleType.PREFILL, "group1", 8301);
        WorkerStatus filtered = status(RoleType.PREFILL, "group2", 8302);
        WorkerStatus otherRole = status(RoleType.DECODE, "group1", 8303);
        discover(matching);
        discover(filtered);
        discover(otherRole);

        WorkerStatusProvider provider = workerDirectory;

        assertEquals(List.of(matching),
                provider.getWorkerStatuses(RoleType.PREFILL, "group1"));
        assertEquals(2,
                provider.getWorkerStatuses(RoleType.PREFILL, null).size());
        assertTrue(provider.getWorkerStatuses(null, "group1").isEmpty());
    }

    private void discover(WorkerStatus status) {
        assertSame(status, workerDirectory.currentOrDiscover(
                status.getRole(), status.getLogicalIpPort(), () -> status));
    }
}
