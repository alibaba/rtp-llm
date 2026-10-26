package org.flexlb.balance.strategy;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.scheduler.DefaultRouter;
import org.flexlb.balance.schedulev2.V2TestSupport;
import org.flexlb.balance.schedulev2.prefill.PrefillPicker;
import org.flexlb.balance.schedulev2.prefill.PrefillQueueRegistry;
import org.flexlb.balance.schedulev2.request.PrefillChoice;
import org.flexlb.balance.schedulev2.request.Request;
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

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
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
            a.setPrefillChoice(PrefillChoice.from(selected, a.seqLen()));
        }
        assertSame(first, a.prefillChoice().endpoint());
        a.moveTo(Stage.MAIN_WAITING, Stage.PREFILL_WAITING);
        queues.addWaiting(a);

        // B 的材料早已准备好，但最终决策要看到 A 新增的 10ms 排队负载：P1=20ms，P2=15ms。
        try (SelectedRole selected = picker.pick(b, RoleType.PREFILL, "g", bInputs).value()) {
            b.setPrefillChoice(PrefillChoice.from(selected, b.seqLen()));
        }
        assertSame(second, b.prefillChoice().endpoint());
        assertEquals(15L, b.prefillChoice().serverStatus().getPrefillTime());
        verify(cache, times(2)).findMatchingEngines(any());
    }

    @Test
    void newCandidateRequiresNewMaterialsInsteadOfPredictionOnDecisionThread() {
        Request request = request("1");
        PrefillInputs inputs = picker.prepare(request.ctx(), RoleType.PREFILL, "g");
        publish("10.0.0.3", "5");
        doThrow(new AssertionError("cache IO must stay in preparation")).when(cache).findMatchingEngines(any());

        assertEquals(PlacementResult.Status.BLOCKED, picker.pick(request, RoleType.PREFILL, "g", inputs).status());
        verify(cache).findMatchingEngines(any());
    }

    @Test
    void retiringCandidateInvalidatesMaterialsAndDoesNotBlockOtherNodes() {
        Request request = request("1");
        PrefillInputs inputs = picker.prepare(request.ctx(), RoleType.PREFILL, "g");
        assertTrue(picker.inputsAreCurrent(inputs, RoleType.PREFILL, "g"));
        first.close();
        assertFalse(picker.inputsAreCurrent(inputs, RoleType.PREFILL, "g"));
        PrefillInputs refreshed = picker.prepare(request.ctx(), RoleType.PREFILL, "g");
        assertEquals(1, refreshed.candidates().size());
        try (SelectedRole selected = picker.pick(request, RoleType.PREFILL, "g", refreshed).value()) {
            assertSame(second, PrefillChoice.from(selected, request.seqLen()).endpoint());
        }
    }

    private Request request(String id) {
        Request request = V2TestSupport.request(config, id, 50, 16L);
        request.setQueueSeq(Long.parseLong(id));
        return request;
    }

    private PrefillEndpoint publish(String ip, String expression) {
        config.getRouter().getRoles().getPrefill().getExecutionTimeEstimator().setExpression(expression);
        PrefillEndpoint endpoint = (PrefillEndpoint) StrategyTestSupport.publishEndpoint(
                registry, RoleType.PREFILL, ip + ":8080",
                StrategyTestSupport.workerStatus(RoleType.PREFILL, "g", ip, 8080, 8081, true, 10_000L, 10_000L));
        V2TestSupport.register(queues, endpoint);
        return endpoint;
    }
}
