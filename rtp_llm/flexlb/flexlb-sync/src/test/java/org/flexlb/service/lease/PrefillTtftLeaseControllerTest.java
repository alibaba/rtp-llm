package org.flexlb.service.lease;

import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.PrefillLeaseConfig;
import org.flexlb.service.lease.PrefillLeaseManager.WorkerView;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.io.IOException;
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
    private final PrefillTtftMetricReader metrics = mock(PrefillTtftMetricReader.class);
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
        when(leases.changeIfCurrent(anyString(), anyString(), anyString(), anyBoolean())).thenReturn(true);
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
        verify(metrics, never()).read(eq(config), anyList(), anyLong());
        controller.evaluate("g", config, 4_000);
        verify(leases, never()).changeIfCurrent(anyString(), anyString(), anyString(), anyBoolean());
        controller.evaluate("g", config, 5_000);
        verify(leases).changeIfCurrent("g", PrefillTtftLeaseController.members(workers), "b", false);
        verify(metrics).read(config, List.of("a", "b"), 4_000);
    }

    @Test
    void highTtftEnablesAStandbyWorker() throws Exception {
        measurement(250, 100);
        controller.evaluate("g", config, 1_000);
        controller.evaluate("g", config, 4_000);
        controller.evaluate("g", config, 5_000);
        verify(leases).changeIfCurrent("g", PrefillTtftLeaseController.members(workers), "idle", true);
    }

    @Test
    void missingOrInsufficientDataBreaksConsecutiveViolations() throws Exception {
        when(metrics.read(eq(config), anyList(), anyLong()))
                .thenReturn(new PrefillTtftMetricReader.Measurement(50, 100, 4_000))
                .thenThrow(new IOException("unavailable"))
                .thenReturn(new PrefillTtftMetricReader.Measurement(50, 1, 6_000))
                .thenReturn(new PrefillTtftMetricReader.Measurement(50, 100, 7_000));
        controller.evaluate("g", config, 1_000);
        for (long now : new long[]{4_000, 5_000, 6_000, 7_000}) {
            controller.evaluate("g", config, now);
        }
        verify(leases, never()).changeIfCurrent(anyString(), anyString(), anyString(), anyBoolean());
    }

    @Test
    void changedServingSetRequiresANewWindow() throws Exception {
        measurement(50, 100);
        controller.evaluate("g", config, 1_000);
        controller.evaluate("g", config, 4_000);
        workers = List.of(worker("a", true, 20), worker("b", false, 0), worker("idle", false, 0));
        controller.evaluate("g", config, 5_000);
        assertTrue(controller.status("g").get("result").toString().contains("fresh window"));
        verify(leases, never()).changeIfCurrent(anyString(), anyString(), anyString(), anyBoolean());
    }

    @Test
    void boundsPreventRevokingTheLastServingWorker() throws Exception {
        workers = List.of(worker("a", true, 0), worker("idle", false, 0));
        measurement(50, 100);
        controller.evaluate("g", config, 1_000);
        controller.evaluate("g", config, 4_000);
        controller.evaluate("g", config, 5_000);
        verify(leases, never()).changeIfCurrent(anyString(), anyString(), anyString(), anyBoolean());
        assertTrue(controller.status("g").get("result").toString().contains("bounds"));
    }

    @Test
    void restoresMinimumServingCountWithoutWaitingForSamples() throws Exception {
        workers = List.of(worker("a", false, 0), worker("b", false, 0));
        controller.evaluate("g", config, 1_000);
        verify(leases).changeIfCurrent("g", PrefillTtftLeaseController.members(workers), "a", true);
        verify(metrics, never()).read(eq(config), anyList(), anyLong());
        assertTrue(controller.status("g").get("result").toString().contains("minServingWorkers"));
    }

    @Test
    void configurationChangeWhileQueryRunsDiscardsDecision() throws Exception {
        config.setConsecutiveViolations(1);
        when(metrics.read(eq(config), anyList(), anyLong())).thenAnswer(call -> {
            when(configs.loadBalanceConfig()).thenReturn(new FlexlbConfig());
            return new PrefillTtftMetricReader.Measurement(250, 100, call.getArgument(2));
        });
        controller.evaluate("g", config, 1_000);
        controller.evaluate("g", config, 4_000);
        verify(leases, never()).changeIfCurrent(anyString(), anyString(), anyString(), anyBoolean());
        assertTrue(controller.status("g").get("result").toString().contains("discarded measurement"));
    }

    @Test
    void staleMeasurementCannotChangeNewerLeaseSet() throws Exception {
        measurement(250, 100);
        when(leases.changeIfCurrent(anyString(), anyString(), anyString(), anyBoolean())).thenReturn(false);
        controller.evaluate("g", config, 1_000);
        controller.evaluate("g", config, 4_000);
        controller.evaluate("g", config, 5_000);
        assertTrue(controller.status("g").get("result").toString().contains("discarded measurement"));
    }

    private void measurement(double mean, double samples) throws Exception {
        when(metrics.read(eq(config), anyList(), anyLong()))
                .thenAnswer(call -> new PrefillTtftMetricReader.Measurement(mean, samples, call.getArgument(2)));
    }

    private static WorkerView worker(String address, boolean serving, int pending) {
        return new WorkerView(address, 1, 1, serving, true, true, pending, serving ? "SERVING" : "STANDBY");
    }
}
