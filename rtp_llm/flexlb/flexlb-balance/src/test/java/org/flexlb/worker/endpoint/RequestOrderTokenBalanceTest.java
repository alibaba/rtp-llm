package org.flexlb.worker.endpoint;

import org.flexlb.balance.queue.CapacityWaitQueue;
import org.flexlb.balance.queue.OrderedRequestQueue;
import org.flexlb.balance.request.Request;
import org.flexlb.balance.schedule.RequestContext;
import org.flexlb.balance.schedule.SchedulingMetadata;
import org.flexlb.balance.strategy.SelectionResult;
import org.flexlb.balance.strategy.TokenBalancedPrefillStrategy;
import org.flexlb.cache.CacheMatchResult;
import org.flexlb.cache.CacheMatchSource;
import org.flexlb.cache.HostCacheMatch;
import org.flexlb.cache.match.CacheAwareService;
import org.flexlb.cache.metrics.CacheMetricsReporter;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.routing.RoutingConfig;
import org.flexlb.config.service.RoleType;
import org.flexlb.worker.status.WorkerStatusResponse;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Runs real request ordering, token-balanced selection and Endpoint admission together.
 * Waiting requests reconsider current cache leaders while preserving priority and capacity limits.
 */
class RequestOrderTokenBalanceTest {
    @ParameterizedTest
    @CsvSource({"false", "true"})
    void notificationLetsWaitingRequestChooseNewCacheLeaderWithoutCapacityChange(boolean priority) {
        var config = new FlexlbConfig();
        var prefill = config.getRouter().getRoles().getPrefill();
        prefill.getSelection().setType(RoutingConfig.PrefillSelectionType.TOKEN_BALANCE);
        prefill.getAdmission().setMaxInflightRequests(1);
        var oldLeader = endpoint(config, "127.0.0.1");
        var newLeader = endpoint(config, "127.0.0.2");
        try {
            var service = mock(ConfigService.class);
            when(service.loadBalanceConfig()).thenReturn(config);
            var registry = mock(EndpointRegistry.class);
            when(registry.<PrefillEndpoint>endpoints(RoleType.PREFILL, null))
                    .thenReturn(List.of(oldLeader, newLeader));
            when(registry.getEndpoint(RoleType.PREFILL, oldLeader.ipPort())).thenReturn(oldLeader);
            when(registry.getEndpoint(RoleType.PREFILL, newLeader.ipPort())).thenReturn(newLeader);
            var matches =
                    new java.util.concurrent.atomic.AtomicReference<>(
                            new CacheMatchResult(
                                    HostCacheMatch.fromLocalMatches(
                                            Map.of(oldLeader.ipPort(), 80, newLeader.ipPort(), 20)),
                                    CacheMatchSource.LOCAL_SYNC,
                                    0,
                                    1));
            var cache = mock(CacheAwareService.class);
            when(cache.findMatchingEngines(any())).thenAnswer(call -> matches.get());
            var strategy =
                    new TokenBalancedPrefillStrategy(
                            service, registry, cache, mock(CacheMetricsReporter.class));
            var occupied =
                    oldLeader
                            .tryRegisterSelection("occupied", oldLeader.selectionSnapshot(), 0, -1L)
                            .value();
            var oldCapacity = oldLeader.requestCapacity();
            var newCapacity = newLeader.requestCapacity();
            var ordered = new OrderedRequestQueue(priority, false);
            var request = request(config, "waiting", 90);
            ordered.add(request);
            var waiting = new CapacityWaitQueue();
            var blocked = strategy.selectAndRegister(request, null);
            assertEquals(SelectionResult.Status.BLOCKED, blocked.status());
            assertEquals(
                    java.util.Set.of(oldLeader.getStatus().getGenerationId()),
                    blocked.candidateGenerations());
            waiting.park(request, blocked.blocker(), blocked.candidateGenerations());

            matches.set(
                    new CacheMatchResult(
                            HostCacheMatch.fromLocalMatches(
                                    Map.of(oldLeader.ipPort(), 10, newLeader.ipPort(), 90)),
                            CacheMatchSource.LOCAL_SYNC,
                            0,
                            1));
            assertEquals(oldCapacity, oldLeader.requestCapacity());
            assertEquals(newCapacity, newLeader.requestCapacity());
            assertTrue(waiting.capacityChanged(newCapacity));
            assertEquals(
                    List.of(request),
                    ordered.scanForPlanningCandidates(1, 2, waiting::canSchedule));
            assertTrue(waiting.tryStartRetry(request));
            var selected = strategy.selectAndRegister(request, null);
            assertEquals(SelectionResult.Status.SUCCESS, selected.status());
            assertSame(newLeader, selected.value().endpoint());
            assertEquals(
                    1,
                    oldLeader.selectionSnapshot().outstandingRequests(),
                    "the old cache leader is still full");
            waiting.remove(request);
            assertTrue(newLeader.rollbackSelection(selected.value().registration()));
            assertTrue(oldLeader.rollbackSelection(occupied));
            org.junit.jupiter.api.Assertions.assertFalse(waiting.hasWaitingRequests());
        } finally {
            EndpointTestSupport.closeAndWait(oldLeader);
            EndpointTestSupport.closeAndWait(newLeader);
        }
    }

    @ParameterizedTest
    @CsvSource({"false", "true"})
    void failedFallbackLetsTheNextWaiterUseRefreshedCapacity(boolean priority) {
        var config = new FlexlbConfig();
        var prefill = config.getRouter().getRoles().getPrefill();
        prefill.getSelection().setType(RoutingConfig.PrefillSelectionType.TOKEN_BALANCE);
        prefill.getAdmission().setMaxInflightRequests(1);
        var firstLeader = endpoint(config, "127.0.0.1");
        var secondLeader = endpoint(config, "127.0.0.2");
        try {
            var service = mock(ConfigService.class);
            when(service.loadBalanceConfig()).thenReturn(config);
            var registry = mock(EndpointRegistry.class);
            when(registry.<PrefillEndpoint>endpoints(RoleType.PREFILL, null))
                    .thenReturn(List.of(firstLeader, secondLeader));
            when(registry.getEndpoint(RoleType.PREFILL, firstLeader.ipPort()))
                    .thenReturn(firstLeader);
            when(registry.getEndpoint(RoleType.PREFILL, secondLeader.ipPort()))
                    .thenReturn(secondLeader);
            var cache = mock(CacheAwareService.class);
            when(cache.findMatchingEngines(any()))
                    .thenAnswer(
                            call -> {
                                var query =
                                        call.getArgument(0, org.flexlb.cache.CacheMatchQuery.class);
                                var leader =
                                        query.requestId().equals("first")
                                                ? firstLeader
                                                : secondLeader;
                                return new CacheMatchResult(
                                        HostCacheMatch.fromLocalMatches(
                                                Map.of(leader.ipPort(), 80)),
                                        CacheMatchSource.LOCAL_SYNC,
                                        0,
                                        1);
                            });
            var strategy =
                    new TokenBalancedPrefillStrategy(
                            service, registry, cache, mock(CacheMetricsReporter.class));
            var firstOccupied =
                    firstLeader
                            .tryRegisterSelection(
                                    "occupied-a", firstLeader.selectionSnapshot(), 0, -1L)
                            .value();
            var secondOccupied =
                    secondLeader
                            .tryRegisterSelection(
                                    "occupied-b", secondLeader.selectionSnapshot(), 0, -1L)
                            .value();
            var ordered = new OrderedRequestQueue(priority, false);
            var first = request(config, "first", 90);
            var second = request(config, "second", 10);
            ordered.add(first);
            ordered.add(second);
            var waiting = new CapacityWaitQueue();
            for (var request : List.of(first, second)) {
                var blocked = strategy.selectAndRegister(request, null);
                assertEquals(SelectionResult.Status.BLOCKED, blocked.status());
                waiting.park(request, blocked.blocker(), blocked.candidateGenerations());
            }
            assertTrue(secondLeader.rollbackSelection(secondOccupied));
            var capacity = secondLeader.requestCapacity();
            waiting.capacityChanged(capacity);
            assertEquals(
                    List.of(first), ordered.scanForPlanningCandidates(1, 2, waiting::canSchedule));
            assertTrue(waiting.tryStartRetry(first));
            var stillBlocked = strategy.selectAndRegister(first, null);
            assertEquals(SelectionResult.Status.BLOCKED, stillBlocked.status());
            waiting.park(first, stillBlocked.blocker(), stillBlocked.candidateGenerations());
            assertFalse(waiting.canSchedule(second), "the first attempt spent this round's slot");

            long sampledVersion = waiting.workerAdditionVersion();
            waiting.checkCapacity(List.of(firstLeader.requestCapacity(), capacity), sampledVersion);
            ordered.beginScan();
            assertEquals(
                    List.of(second), ordered.scanForPlanningCandidates(1, 2, waiting::canSchedule));
            assertTrue(waiting.tryStartRetry(second));
            assertEquals(1, firstLeader.selectionSnapshot().outstandingRequests());
            var selected = strategy.selectAndRegister(second, null);
            assertEquals(SelectionResult.Status.SUCCESS, selected.status());
            assertSame(secondLeader, selected.value().endpoint());
            waiting.remove(second);
            ordered.remove(second);
            assertTrue(secondLeader.rollbackSelection(selected.value().registration()));
            assertTrue(firstLeader.rollbackSelection(firstOccupied));
            waiting.capacityChanged(firstLeader.requestCapacity());
            ordered.beginScan();
            assertEquals(
                    List.of(first), ordered.scanForPlanningCandidates(1, 2, waiting::canSchedule));
            assertTrue(waiting.tryStartRetry(first));
            var firstSelected = strategy.selectAndRegister(first, null);
            assertEquals(SelectionResult.Status.SUCCESS, firstSelected.status());
            assertSame(firstLeader, firstSelected.value().endpoint());
            waiting.remove(first);
            assertTrue(firstLeader.rollbackSelection(firstSelected.value().registration()));
            assertEquals(0, firstLeader.selectionSnapshot().outstandingRequests());
            assertEquals(0, secondLeader.selectionSnapshot().outstandingRequests());
            org.junit.jupiter.api.Assertions.assertFalse(waiting.hasWaitingRequests());
        } finally {
            EndpointTestSupport.closeAndWait(firstLeader);
            EndpointTestSupport.closeAndWait(secondLeader);
        }
    }

    @ParameterizedTest
    @CsvSource({"false, false", "true, false", "false, true", "true, true"})
    void costOrderComposesWithPriorityAndStrictCacheAdmission(
            boolean priority, boolean engineQueueFull) {
        var config = new FlexlbConfig();
        config.getRouter()
                .getRoles()
                .getPrefill()
                .getSelection()
                .setType(RoutingConfig.PrefillSelectionType.TOKEN_BALANCE);
        config.getRouter()
                .getRoles()
                .getPrefill()
                .getAdmission()
                .setMaxInflightRequests(engineQueueFull ? 128 : 1);
        var leader = endpoint(config, "127.0.0.1");
        var alternate = endpoint(config, "127.0.0.2");
        try {
            var service = mock(ConfigService.class);
            when(service.loadBalanceConfig()).thenReturn(config);
            var registry = mock(EndpointRegistry.class);
            when(registry.<PrefillEndpoint>endpoints(RoleType.PREFILL, null))
                    .thenReturn(List.of(leader, alternate));
            when(registry.getEndpoint(RoleType.PREFILL, leader.ipPort())).thenReturn(leader);
            when(registry.getEndpoint(RoleType.PREFILL, alternate.ipPort())).thenReturn(alternate);
            var cache = mock(CacheAwareService.class);
            when(cache.findMatchingEngines(any()))
                    .thenAnswer(
                            call -> {
                                var query =
                                        call.getArgument(0, org.flexlb.cache.CacheMatchQuery.class);
                                int hit = query.requestId().equals("cheap") ? 80 : 20;
                                return new CacheMatchResult(
                                        HostCacheMatch.fromLocalMatches(
                                                Map.of(
                                                        leader.ipPort(),
                                                        hit,
                                                        alternate.ipPort(),
                                                        10)),
                                        CacheMatchSource.LOCAL_SYNC,
                                        0,
                                        1);
                            });
            var strategy =
                    new TokenBalancedPrefillStrategy(
                            service, registry, cache, mock(CacheMetricsReporter.class));
            var queue = new OrderedRequestQueue(priority, true);
            var expensive = request(config, "expensive", 90);
            var cheap = request(config, "cheap", 10);
            queue.add(expensive);
            queue.add(cheap);
            for (var ticket : queue.estimates(2)) {
                ticket.complete(
                        strategy.estimateQueuedTokens(
                                ticket.request().getRequest(), RoleType.PREFILL, null));
            }
            var first = priority ? expensive : cheap;
            var second = priority ? cheap : expensive;
            assertEquals(List.of(first), queue.scanForPlanningCandidates(1, 10, r -> true));
            var selected = strategy.selectAndRegister(first, null);
            assertEquals(SelectionResult.Status.SUCCESS, selected.status());
            assertSame(leader, selected.value().endpoint());
            assertEquals(priority ? 80L : 20L, selected.value().computeTokens());
            assertTrue(queue.remove(first));
            queue.retry(first);
            if (engineQueueFull) reportWaiting(leader, 11);
            queue.beginScan();
            assertEquals(List.of(second), queue.scanForPlanningCandidates(1, 10, r -> true));
            assertEquals(
                    SelectionResult.Status.BLOCKED,
                    strategy.selectAndRegister(second, null).status());
            assertEquals(0, alternate.selectionSnapshot().outstandingRequests());
            assertTrue(leader.rollbackSelection(selected.value().registration()));
            if (engineQueueFull) {
                assertEquals(
                        SelectionResult.Status.BLOCKED,
                        strategy.selectAndRegister(second, null).status());
                reportWaiting(leader, 0);
            }
            queue.retry(second);
            assertEquals(List.of(second), queue.scanForPlanningCandidates(1, 10, r -> true));
            var resumed = strategy.selectAndRegister(second, null);
            assertEquals(SelectionResult.Status.SUCCESS, resumed.status());
            assertSame(leader, resumed.value().endpoint());
            assertTrue(leader.rollbackSelection(resumed.value().registration()));
            assertTrue(queue.remove(second));
            assertEquals(0, queue.size());
            assertEquals(0, leader.selectionSnapshot().outstandingRequests());
            assertEquals(0, alternate.selectionSnapshot().outstandingRequests());
        } finally {
            EndpointTestSupport.closeAndWait(leader);
            EndpointTestSupport.closeAndWait(alternate);
        }
    }

    private static RequestContext request(FlexlbConfig config, String id, int priority) {
        var context = new RequestContext(config);
        context.setRequest(Request.builder().requestId(id).seqLen(100L).blockSize(1L).build());
        context.setSchedulingMetadata(SchedulingMetadata.of(priority, null, Long.MAX_VALUE, 50));
        return context;
    }

    private static PrefillEndpoint endpoint(FlexlbConfig config, String ip) {
        var endpoint =
                new PrefillEndpoint(
                        EndpointTestSupport.workerStatus(RoleType.PREFILL, ip, 8080, 8081),
                        config,
                        new CapacityNotifier());
        reportWaiting(endpoint, 0);
        return endpoint;
    }

    private static void reportWaiting(PrefillEndpoint endpoint, long waiting) {
        var report = new WorkerStatusResponse();
        report.setAlive(true);
        report.setWaitingQueryLen(waiting);
        EndpointTestSupport.applyStatus(endpoint, report).run();
    }
}
