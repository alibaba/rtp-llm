package org.flexlb.service.lease;

import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.PrefillLeaseConfig;
import org.flexlb.service.lease.PrefillLeaseManager.WorkerView;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.anyBoolean;
import static org.mockito.ArgumentMatchers.anyList;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class PrefillTtftLeaseControllerTest {
    private final ConfigService configs = mock(ConfigService.class);
    private final PrefillLeaseManager leases = mock(PrefillLeaseManager.class);
    private final PrefillTtftStatusReader metrics = mock(PrefillTtftStatusReader.class);
    private PrefillLeaseConfig config;
    private PrefillTtftLeaseController controller;
    private List<WorkerView> workers;

    @BeforeEach
    void setUp() {
        FlexlbConfig routing = new FlexlbConfig();
        config = routing.getRouter().getRoles().getPrefill().getChunkLease();
        config.setEnabled(true);
        config.setAutomatic(true);
        config.setGroups(List.of("g"));
        config.setWindowSeconds(1);
        config.setMaxMetricAgeSeconds(1);
        config.setSettleSeconds(0);
        config.setMinSamples(10);
        config.setConsecutiveViolations(2);
        config.setTtftLowerMs(100);
        config.setTtftUpperMs(200);
        when(configs.loadBalanceConfig()).thenReturn(routing);
        workers = List.of(worker("a", true, 20), worker("b", true, 1), worker("idle", false, 0));
        when(leases.status("g")).thenAnswer(ignored -> workers);
        when(leases.changeIfCurrent(anyString(), anyString(), anyList(), anyBoolean(), eq(config)))
                .thenAnswer(call -> call.getArgument(2));
        controller = new PrefillTtftLeaseController(configs, leases, metrics);
    }

    @AfterEach
    void close() {
        controller.close();
    }

    @Test
    void lowTtftRevokesOnlyAfterStableWindowAndConsecutiveViolations() throws Exception {
        measurement(50, 100);
        controller.evaluate("g", config, 1_000);
        verify(metrics, never()).read(eq(config), anyList(), anyLong(), anyLong());
        controller.evaluate("g", config, 4_000);
        verify(leases, never()).changeIfCurrent(anyString(), anyString(), anyList(), anyBoolean(), eq(config));
        controller.evaluate("g", config, 5_000);
        verify(leases).changeIfCurrent("g", PrefillTtftLeaseController.members(workers), List.of("b"), false, config);
        verify(metrics).read(config, List.of("a", "b"), 4_000, 1_000);
    }

    @Test
    void highTtftEnablesAStandbyWorker() throws Exception {
        measurement(250, 100);
        controller.evaluate("g", config, 1_000);
        controller.evaluate("g", config, 4_000);
        controller.evaluate("g", config, 5_000);
        verify(leases).changeIfCurrent("g", PrefillTtftLeaseController.members(workers), List.of("idle"), true, config);
    }

    @Test
    void missingOrInsufficientDataBreaksConsecutiveViolations() throws Exception {
        when(metrics.read(eq(config), anyList(), anyLong(), anyLong()))
                .thenReturn(new PrefillTtftStatusReader.Measurement(50, 100, 4_000))
                .thenThrow(new IllegalStateException("unavailable"))
                .thenReturn(new PrefillTtftStatusReader.Measurement(50, 1, 6_000))
                .thenReturn(new PrefillTtftStatusReader.Measurement(50, 100, 7_000));
        controller.evaluate("g", config, 1_000);
        for (long now : new long[]{4_000, 5_000, 6_000, 7_000}) {
            controller.evaluate("g", config, now);
        }
        verify(leases, never()).changeIfCurrent(anyString(), anyString(), anyList(), anyBoolean(), eq(config));
    }

    @Test
    void changedServingSetRequiresANewWindow() throws Exception {
        measurement(50, 100);
        controller.evaluate("g", config, 1_000);
        controller.evaluate("g", config, 4_000);
        workers = List.of(worker("a", true, 20), worker("b", false, 0), worker("idle", false, 0));
        controller.evaluate("g", config, 5_000);
        assertTrue(controller.status("g").get("result").toString().contains("fresh window"));
        verify(leases, never()).changeIfCurrent(anyString(), anyString(), anyList(), anyBoolean(), eq(config));
    }

    @Test
    void boundsPreventRevokingTheLastServingWorker() throws Exception {
        workers = List.of(worker("a", true, 0), worker("idle", false, 0));
        measurement(50, 100);
        controller.evaluate("g", config, 1_000);
        controller.evaluate("g", config, 4_000);
        controller.evaluate("g", config, 5_000);
        verify(leases, never()).changeIfCurrent(anyString(), anyString(), anyList(), anyBoolean(), eq(config));
        assertTrue(controller.status("g").get("result").toString().contains("bounds"));
    }

    @Test
    void restoresMinimumServingCountWithoutWaitingForSamples() throws Exception {
        workers = List.of(worker("a", false, 0), worker("b", false, 0));
        controller.evaluate("g", config, 1_000);
        verify(leases).changeIfCurrent("g", PrefillTtftLeaseController.members(workers), List.of("a"), true, config);
        verify(metrics, never()).read(eq(config), anyList(), anyLong(), anyLong());
        assertTrue(controller.status("g").get("result").toString().contains("minServingWorkers"));
    }

    @Test
    void configurationChangeWhileQueryRunsDiscardsDecision() throws Exception {
        config.setConsecutiveViolations(1);
        when(metrics.read(eq(config), anyList(), anyLong(), anyLong())).thenAnswer(call -> {
            when(configs.loadBalanceConfig()).thenReturn(new FlexlbConfig());
            return new PrefillTtftStatusReader.Measurement(250, 100, call.getArgument(2));
        });
        controller.evaluate("g", config, 1_000);
        controller.evaluate("g", config, 4_000);
        verify(leases, never()).changeIfCurrent(anyString(), anyString(), anyList(), anyBoolean(), eq(config));
        assertTrue(controller.status("g").get("result").toString().contains("discarded measurement"));
    }

    @Test
    void staleMeasurementCannotChangeNewerLeaseSet() throws Exception {
        measurement(250, 100);
        when(leases.changeIfCurrent(anyString(), anyString(), anyList(), anyBoolean(), eq(config)))
                .thenReturn(List.of());
        controller.evaluate("g", config, 1_000);
        controller.evaluate("g", config, 4_000);
        controller.evaluate("g", config, 5_000);
        assertTrue(controller.status("g").get("result").toString().contains("discarded measurement"));
    }

    @Test
    void batchGrantRespectsMaximumAndAvailableStandbyWorkers() throws Exception {
        config.setScaleUpStep(10);
        config.setMaxServingWorkers(4);
        workers = List.of(worker("a", true, 0), worker("b", true, 0), worker("c", false, 0), worker("d", false, 0),
                worker("e", false, 0));
        measurement(250, 100);
        controller.evaluate("g", config, 1_000);
        controller.evaluate("g", config, 4_000);
        controller.evaluate("g", config, 5_000);
        verify(leases).changeIfCurrent("g", PrefillTtftLeaseController.members(workers), List.of("c", "d"), true,
                config);
    }

    @Test
    void batchRevokeRespectsMinimumAndPendingRequestOrder() throws Exception {
        config.setScaleDownStep(10);
        config.setMinServingWorkers(2);
        workers = List.of(worker("a", true, 10), worker("b", true, 1), worker("c", true, 0), worker("d", true, 20));
        measurement(50, 100);
        controller.evaluate("g", config, 1_000);
        controller.evaluate("g", config, 4_000);
        controller.evaluate("g", config, 5_000);
        verify(leases).changeIfCurrent("g", PrefillTtftLeaseController.members(workers), List.of("c", "b"), false,
                config);
    }

    @Test
    void minimumRecoveryUsesConfiguredStep() {
        config.setScaleUpStep(2);
        config.setMinServingWorkers(3);
        workers = List.of(worker("a", false, 0), worker("b", false, 0), worker("c", false, 0));
        controller.evaluate("g", config, 1_000);
        verify(leases).changeIfCurrent("g", PrefillTtftLeaseController.members(workers), List.of("a", "b"), true,
                config);
    }

    private void measurement(double mean, long samples) throws Exception {
        when(metrics.read(eq(config), anyList(), anyLong(), anyLong()))
                .thenAnswer(call -> new PrefillTtftStatusReader.Measurement(mean, samples, call.getArgument(2)));
    }

    private static WorkerView worker(String address, boolean serving, int pending) {
        return new WorkerView(address, 1, 1, serving, true, true, pending, serving ? "SERVING" : "STANDBY");
    }
}
