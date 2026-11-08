package org.flexlb.balance.strategy;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.scheduler.DefaultRouter;
import org.flexlb.balance.schedulev2.V2TestSupport;
import org.flexlb.balance.schedulev2.prefill.PrefillPicker;
import org.flexlb.balance.schedulev2.prefill.PrefillQueueRegistry;
import org.flexlb.balance.schedulev2.request.Request;
import org.flexlb.balance.schedulev2.request.SelectedPrefill;
import org.flexlb.balance.schedulev2.request.Stage;
import org.flexlb.cache.domain.CacheMatchResult;
import org.flexlb.cache.domain.CacheMatchSource;
import org.flexlb.cache.match.CacheAwareService;
import org.flexlb.config.ConfigService;
import org.flexlb.config.DispatcherConfig;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.status.WorkerDirectory;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class PreparedPrefillSelectionTest {

    private final FlexlbConfig config = StrategyTestSupport.config();
    private final ConfigService configService = mock(ConfigService.class);
    private final CacheAwareService cache = mock(CacheAwareService.class);
    private final EndpointRegistry registry;
    private final PrefillQueueRegistry queues;
    private final PrefillPicker picker;
    private final PrefillEndpoint first;
    private final PrefillEndpoint second;

    PreparedPrefillSelectionTest() {
        config.getDispatcher().setType(DispatcherConfig.Type.NON_BATCH_V2);
        when(configService.loadBalanceConfig()).thenReturn(config);
        registry = StrategyTestSupport.endpointRegistry(configService);
        queues = new PrefillQueueRegistry();
        first = publish("10.0.0.1", "10");
        second = publish("10.0.0.2", "15");
        when(cache.findMatchingEngines(any())).thenReturn(CacheMatchResult.empty(CacheMatchSource.LOCAL_SYNC));
        CostBasedPrefillStrategy strategy =
                new CostBasedPrefillStrategy(new WorkerDirectory(registry), cache, mock(EngineHealthReporter.class));
        picker = new PrefillPicker(mock(DefaultRouter.class), strategy);
    }

    @AfterEach
    void close() {
        registry.close();
    }

    @Test
    void twoPreparedRequestsChooseUsingLatestQueueWithoutRepeatingCacheIo() throws Exception {
        Request a = request("1");
        Request b = request("2");
        ExecutorService workers = Executors.newFixedThreadPool(2);
        PrefillInputs aInputs;
        PrefillInputs bInputs;
        try {
            var aFuture = workers.submit(() -> picker.prepare(a.ctx(), RoleType.PREFILL, "g"));
            var bFuture = workers.submit(() -> picker.prepare(b.ctx(), RoleType.PREFILL, "g"));
            aInputs = aFuture.get(5, TimeUnit.SECONDS);
            bInputs = bFuture.get(5, TimeUnit.SECONDS);
        } finally {
            workers.shutdownNow();
        }
        assertEquals(2, aInputs.candidates().size());
        assertEquals(10L,
                aInputs.candidates()
                        .stream()
                        .filter(c -> c.endpoint() == first)
                        .findFirst()
                        .orElseThrow()
                        .predictedMs());
        doThrow(new AssertionError("cache IO must stay in preparation")).when(cache).findMatchingEngines(any());

        try (SelectedRole selected = picker.pick(a, RoleType.PREFILL, "g", aInputs).value()) {
            a.setSelectedPrefill(SelectedPrefill.from(selected, a.seqLen()));
        }
        assertSame(first, a.selectedPrefill().endpoint());
        a.moveTo(Stage.MAIN_WAITING, Stage.PREFILL_WAITING);
        queues.addWaiting(a);

        // B 的材料早已准备好，但最终决策要看到 A 新增的 10ms 排队负载：P1=20ms，P2=15ms。
        try (SelectedRole selected = picker.pick(b, RoleType.PREFILL, "g", bInputs).value()) {
            b.setSelectedPrefill(SelectedPrefill.from(selected, b.seqLen()));
        }
        assertSame(second, b.selectedPrefill().endpoint());
        assertEquals(15L, b.selectedPrefill().serverStatus().getPrefillTime());
        verify(cache, times(2)).findMatchingEngines(any());
    }

    @Test
    void newCandidateRequiresNewMaterialsInsteadOfPredictionOnDecisionThread() {
        Request request = request("1");
        PrefillInputs inputs = picker.prepare(request.ctx(), RoleType.PREFILL, "g");
        List<PrefillCandidateWork> originalCandidates = new ArrayList<>(inputs.candidates());
        PrefillEndpoint added = publish("10.0.0.3", "5");
        assertEquals(originalCandidates, inputs.candidates());
        assertFalse(picker.inputsAreCurrent(inputs, RoleType.PREFILL, "g"));
        PrefillInputs refreshed = picker.prepare(request.ctx(), RoleType.PREFILL, "g");
        assertEquals(3, refreshed.candidates().size());
        assertTrue(picker.inputsAreCurrent(refreshed, RoleType.PREFILL, "g"));
        doThrow(new AssertionError("cache IO must stay in preparation")).when(cache).findMatchingEngines(any());

        assertEquals(PlacementResult.Status.BLOCKED, picker.pick(request, RoleType.PREFILL, "g", inputs).status());
        try (SelectedRole selected = picker.pick(request, RoleType.PREFILL, "g", refreshed).value()) {
            assertSame(added, SelectedPrefill.from(selected, request.seqLen()).endpoint());
        }
        assertEquals(originalCandidates, inputs.candidates());
        verify(cache, times(2)).findMatchingEngines(any());
    }

    @Test
    void preparationFiltersGroupAndRetirementButCountsAllRegisteredGroupMembers() {
        PrefillEndpoint third = publish("10.0.0.3", "20");
        publish("10.0.0.4", "1", "other");
        List<PrefillEndpoint> expected = registry.prefillRoutingSnapshot(RoleType.PREFILL)
                .stream()
                .map(EndpointRegistry.PrefillRoutingEntry::endpoint)
                .filter(endpoint -> endpoint == first || endpoint == third)
                .toList();
        second.close();
        Request request = request("1");

        PrefillInputs inputs = picker.prepare(request.ctx(), RoleType.PREFILL, "g");

        assertEquals(expected, inputs.candidates().stream().map(PrefillCandidateWork::endpoint).toList());
        assertThrows(UnsupportedOperationException.class, () -> inputs.candidates().clear());
        assertTrue(picker.inputsAreCurrent(inputs, RoleType.PREFILL, "g"));
        try (SelectedRole selected = picker.pick(request, RoleType.PREFILL, "g", inputs).value()) {
            assertSame(first, SelectedPrefill.from(selected, request.seqLen()).endpoint());
        }
        var decision = request.ctx().getRoutingTelemetry().routingDecisions().get(RoleType.PREFILL);
        assertEquals(3, decision.totalWorkerCount());
        assertEquals(2, decision.candidateWorkerCount());
        verify(cache).findMatchingEngines(any());
    }

    @Test
    void preparationWithoutGroupKeepsOtherGroupsAndExcludesRetiredEndpoints() {
        PrefillEndpoint other = publish("10.0.0.3", "5", "other");
        first.close();
        Request request = request("1");
        List<PrefillEndpoint> expected = registry.prefillRoutingSnapshot(RoleType.PREFILL)
                .stream()
                .map(EndpointRegistry.PrefillRoutingEntry::endpoint)
                .filter(endpoint -> endpoint == second || endpoint == other)
                .toList();

        PrefillInputs inputs = picker.prepare(request.ctx(), RoleType.PREFILL, null);

        assertEquals(expected, inputs.candidates().stream().map(PrefillCandidateWork::endpoint).toList());
        try (SelectedRole selected = picker.pick(request, RoleType.PREFILL, null, inputs).value()) {
            assertSame(other, SelectedPrefill.from(selected, request.seqLen()).endpoint());
        }
        var decision = request.ctx().getRoutingTelemetry().routingDecisions().get(RoleType.PREFILL);
        assertEquals(3, decision.totalWorkerCount());
        assertEquals(2, decision.candidateWorkerCount());
    }

    @Test
    void preparationKeepsAllBusyCandidatesBeforeCacheMatching() {
        config.getDispatcher().setMaxInflightPerPrefillWorker(1);
        for (PrefillEndpoint endpoint : List.of(first, second)) {
            var status = endpoint.getStatus();
            StrategyTestSupport.publish(status, StrategyTestSupport.response(RoleType.PREFILL, true, 0L, 10_000L,
                    status.appliedStatusCursor().statusVersion() + 1L));
            Request queued = request(endpoint == first ? "10" : "20");
            V2TestSupport.choose(queued, endpoint, 0L, 100L);
            assertTrue(queued.moveTo(Stage.MAIN_WAITING, Stage.PREFILL_WAITING));
            queues.addWaiting(queued);
        }
        Request request = request("1");

        PrefillInputs inputs = picker.prepare(request.ctx(), RoleType.PREFILL, "g");

        assertEquals(2, inputs.candidates().size());
        assertTrue(inputs.candidates().stream().anyMatch(candidate -> candidate.endpoint() == first));
        assertTrue(inputs.candidates().stream().anyMatch(candidate -> candidate.endpoint() == second));
        assertTrue(picker.inputsAreCurrent(inputs, RoleType.PREFILL, "g"));
        verify(cache).findMatchingEngines(any());
    }

    @Test
    void retiringCandidateInvalidatesMaterialsAndDoesNotBlockOtherNodes() {
        Request request = request("1");
        PrefillInputs inputs = picker.prepare(request.ctx(), RoleType.PREFILL, "g");
        assertTrue(picker.inputsAreCurrent(inputs, RoleType.PREFILL, "g"));
        first.close();
        assertFalse(picker.inputsAreCurrent(inputs, RoleType.PREFILL, "g"));
        assertEquals(2, inputs.candidates().size());
        assertTrue(inputs.candidates().stream().anyMatch(candidate -> candidate.endpoint() == first));
        PrefillInputs refreshed = picker.prepare(request.ctx(), RoleType.PREFILL, "g");
        assertEquals(1, refreshed.candidates().size());
        try (SelectedRole selected = picker.pick(request, RoleType.PREFILL, "g", refreshed).value()) {
            assertSame(second, SelectedPrefill.from(selected, request.seqLen()).endpoint());
        }
    }

    private Request request(String id) {
        Request request = V2TestSupport.request(config, id, 50, 16L);
        request.setQueueSeq(Long.parseLong(id));
        return request;
    }

    @Test
    void reorderedMaterialsStillUseEachEndpointsOwnPrediction() {
        Request request = request("1");
        PrefillInputs inputs = picker.prepare(request.ctx(), RoleType.PREFILL, "g");
        PrefillInputs reordered = new PrefillInputs(inputs.cacheMatch(), inputs.candidates().reversed());

        assertTrue(picker.inputsAreCurrent(reordered, RoleType.PREFILL, "g"));
        try (SelectedRole selected = picker.pick(request, RoleType.PREFILL, "g", reordered).value()) {
            assertSame(first, SelectedPrefill.from(selected, request.seqLen()).endpoint());
            assertEquals(10L, selected.serverStatus().getPrefillTime());
        }
    }

    @Test
    void replacedEndpointAtSameAddressRequiresNewMaterials() {
        Request request = request("1");
        PrefillInputs inputs = picker.prepare(request.ctx(), RoleType.PREFILL, "g");
        remove(first);
        PrefillEndpoint replacement = publish("10.0.0.1", "1");

        assertEquals(2, registry.prefillRoutingSnapshot(RoleType.PREFILL).size());
        assertFalse(picker.inputsAreCurrent(inputs, RoleType.PREFILL, "g"));
        assertEquals(PlacementResult.Status.BLOCKED, picker.pick(request, RoleType.PREFILL, "g", inputs).status());
        PrefillInputs refreshed = picker.prepare(request.ctx(), RoleType.PREFILL, "g");
        try (SelectedRole selected = picker.pick(request, RoleType.PREFILL, "g", refreshed).value()) {
            assertSame(replacement, SelectedPrefill.from(selected, request.seqLen()).endpoint());
        }
    }

    @Test
    void removalBetweenValidationAndSelectionStillAllowsRemainingPreparedEndpoint() {
        Request request = request("1");
        PrefillInputs inputs = picker.prepare(request.ctx(), RoleType.PREFILL, "g");
        assertTrue(picker.inputsAreCurrent(inputs, RoleType.PREFILL, "g"));
        remove(first);

        assertFalse(picker.inputsAreCurrent(inputs, RoleType.PREFILL, "g"));
        try (SelectedRole selected = picker.pick(request, RoleType.PREFILL, "g", inputs).value()) {
            assertSame(second, SelectedPrefill.from(selected, request.seqLen()).endpoint());
            assertEquals(15L, selected.serverStatus().getPrefillTime());
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void sameEndpointWithDifferentPreparedAddressIsNotCurrent(boolean reordered) {
        Request request = request("1");
        PrefillInputs inputs = picker.prepare(request.ctx(), RoleType.PREFILL, "g");
        List<PrefillCandidateWork> candidates = new ArrayList<>(inputs.candidates());
        PrefillCandidateWork candidate = candidates.getFirst();
        candidates.set(0, new PrefillCandidateWork("other-address", candidate.endpoint(), candidate.cacheHitTokens(),
                candidate.routingCacheMatchTokens(), candidate.predictedMs()));

        PrefillInputs changed = new PrefillInputs(inputs.cacheMatch(), reordered ? candidates.reversed() : candidates);
        assertFalse(picker.inputsAreCurrent(changed, RoleType.PREFILL, "g"));
    }

    private void remove(PrefillEndpoint endpoint) {
        var status = endpoint.getStatus();
        String address = registry.prefillRoutingSnapshot(RoleType.PREFILL).stream()
                .filter(entry -> entry.endpoint() == endpoint)
                .findFirst().orElseThrow().address();
        EndpointRegistry.DetachedGeneration detached;
        status.lock.lock();
        try {
            detached = registry.detachAndBeginRetirement(RoleType.PREFILL, address, status);
        } finally {
            status.lock.unlock();
        }
        detached.retireAndAwait();
    }

    private PrefillEndpoint publish(String ip, String expression) {
        return publish(ip, expression, "g");
    }

    private PrefillEndpoint publish(String ip, String expression, String group) {
        config.getRouter().getRoles().getPrefill().getExecutionTimeEstimator().setExpression(expression);
        PrefillEndpoint endpoint = (PrefillEndpoint) StrategyTestSupport.publishEndpoint(registry, RoleType.PREFILL,
                ip + ":8080",
                StrategyTestSupport.workerStatus(RoleType.PREFILL, group, ip, 8080, 8081, true, 10_000L, 10_000L));
        V2TestSupport.register(queues, endpoint);
        return endpoint;
    }
}
