package org.flexlb.service.monitor;

import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.consistency.LBStatusConsistencyService;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.metric.FlexMonitor;
import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.timeout;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class TtftStageAggregatorTest {
    @Test
    void splitsOneEngineTaskAndCountsOnlyDisjointRemoteKv() {
        var telemetry = mock(WorkerStatus.TaskTelemetry.class);
        when(telemetry.engineAddToFirstTokenUs()).thenReturn(5000L);
        when(telemetry.hybridKvAllocWaitUs()).thenReturn(100L);
        when(telemetry.remoteKvWaitMs()).thenReturn(2L);
        when(telemetry.schedulerWaitUs()).thenReturn(300L);
        when(telemetry.prefillInflightWallUs()).thenReturn(2000L);
        when(telemetry.schedulerWaitForPleReadyUs()).thenReturn(120L);

        assertArrayEquals(new long[]{5000, 100, 0, 300, 2000, 2600}, TtftStageAggregator.stages(telemetry, false));
        assertArrayEquals(new long[]{5000, 100, 2000, 300, 2000, 600}, TtftStageAggregator.stages(telemetry, true));
    }

    @Test
    void skipsMissingTotalAndImpossibleTimings() {
        var telemetry = mock(WorkerStatus.TaskTelemetry.class);
        when(telemetry.engineAddToFirstTokenUs()).thenReturn(null, 10L);
        assertNull(TtftStageAggregator.stages(telemetry, false));
        when(telemetry.schedulerWaitUs()).thenReturn(null);
        assertArrayEquals(new long[]{10, 0, 0, 0, 0, 10}, TtftStageAggregator.stages(telemetry, false));
        when(telemetry.schedulerWaitUs()).thenReturn(11L);
        assertNull(TtftStageAggregator.stages(telemetry, false));
    }

    @Test
    void consumesCommittedPrefillTasksOnTheLocalLeaderWithoutWorkerDeployment() {
        var config = mock(ConfigService.class);
        when(config.loadBalanceConfig()).thenReturn(new FlexlbConfig());
        var leadership = mock(LBStatusConsistencyService.class);
        when(leadership.isNeedConsistency()).thenReturn(true);
        var monitor = mock(FlexMonitor.class);
        var observation = mock(WorkerStatus.StatusObservation.class);
        when(observation.role()).thenReturn(RoleType.PREFILL);
        var task = mock(WorkerStatus.TaskObservation.class);
        when(task.requestId()).thenReturn("p-attempt-1");
        var telemetry = mock(WorkerStatus.TaskTelemetry.class);
        when(task.telemetry()).thenReturn(telemetry);
        when(telemetry.engineAddToFirstTokenUs()).thenReturn(100L);
        when(telemetry.schedulerWaitForPleReadyUs()).thenReturn(15L);
        when(observation.finishedTasks()).thenReturn(Map.of("p-attempt-1", task));
        var collector = new TtftStageAggregator(config, leadership, monitor);
        try {
            collector.addWorkerStatus(observation);
            when(leadership.isMaster()).thenReturn(true);
            collector.addWorkerStatus(observation);
            verify(monitor, timeout(2000).times(1)).report(TtftStageReporter.PREFIX + "ple.wait.us", 15d);
        } finally {
            collector.close();
        }
    }
}
