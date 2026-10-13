package org.flexlb.balance.endpoint;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.delivery.DeliveryMetrics;
import org.flexlb.balance.scheduler.RequestRegistry;
import org.flexlb.balance.scheduler.RouteDeliveryStrategy;
import org.flexlb.balance.scheduler.EndpointEventProjector;
import org.flexlb.balance.strategy.CostBasedPrefillStrategy;
import org.flexlb.balance.strategy.SelectedRole;
import org.flexlb.balance.scheduler.PlacementKey;
import org.flexlb.cache.match.CacheAwareService;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.status.WorkerDirectory;
import org.flexlb.config.DispatcherConfig;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.testfixture.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.lang.management.ManagementFactory;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.mockito.Mockito.withSettings;
import static org.mockito.ArgumentMatchers.any;

/** Full-fleet selection uses real endpoint projections, including when every worker is full. */
@Tag("performance-regression")
class PrefillAdmissionFailurePerformanceTest {
    @Test
    void fullFleetSelectionDoesNotBuildAResponsePerWorker() {
        var bean = (com.sun.management.ThreadMXBean) ManagementFactory.getThreadMXBean();
        assertTrue(bean.isThreadAllocatedMemorySupported());
        bean.setThreadAllocatedMemoryEnabled(true);
        long threadId = Thread.currentThread().threadId();
        FlexlbConfig config = new FlexlbConfig();
        config.getScheduler().setType(org.flexlb.config.SchedulerConfig.Type.DIRECT);
        config.setDispatcher(DispatcherConfig.nonBatch());
        config.getRouter().getRoles().getPrefill().getExecutionTimeEstimator().setExpression("sum(computeTokens)");
        config.getDispatcher().setMaxInflightPerPrefillWorker(1);
        var delivery = new RouteDeliveryStrategy(mock(RequestRegistry.class), mock(DeliveryMetrics.class));
        var events = mock(EndpointEventProjector.class);
        var reporter = mock(BatchSchedulerReporter.class);
        long singleWorkerAllocation = 0L;
        int operations = 1_000;
        for (int workerCount : new int[]{1, 64, 512, 1024}) {
            List<EndpointRegistry.PrefillRoutingEntry> directory = new ArrayList<>();
            Map<String, PrefillEndpoint> endpoints = new HashMap<>();
            for (int i = 0; i < workerCount; i++) {
                var status = EndpointTestSupport.workerStatus(RoleType.PREFILL, "worker-" + i, 8080, 8090);
                var endpoint = new PrefillEndpoint(status, config, delivery, events, reporter);
                var observation = new WorkerStatusResponse();
                observation.setAlive(true);
                observation.setRunningQueryLen(1L);
                EndpointTestSupport.applyStatus(endpoint, observation);
                directory.add(new EndpointRegistry.PrefillRoutingEntry(status.getIpPort(), endpoint));
                endpoints.put(status.getIpPort(), endpoint);
            }
            directory = List.copyOf(directory);
            var workers = mock(WorkerDirectory.class, withSettings().stubOnly());
            when(workers.prefillRoutingSnapshot(RoleType.PREFILL)).thenReturn(directory);
            when(workers.captureEndpoint(any(), any())).thenAnswer(call ->
                    endpoints.get(call.<String>getArgument(1)).tryPinGeneration());
            var strategy = new CostBasedPrefillStrategy(workers,
                    mock(CacheAwareService.class, withSettings().stubOnly()),
                    mock(EngineHealthReporter.class, withSettings().stubOnly()));
            var context = new BalanceContext(config);
            var request = new Request();
            request.setRequestId(1L);
            request.setSeqLen(100L);
            request.setBlockCacheKeys(List.of());
            context.setRequest(request);
            PlacementResult<SelectedRole, PlacementKey> result = null;
            for (int i = 0; i < operations; i++) {
                result = strategy.select(context, RoleType.PREFILL, null);
            }
            long before = bean.getThreadAllocatedBytes(threadId);
            long started = System.nanoTime();
            for (int i = 0; i < operations; i++) {
                result = strategy.select(context, RoleType.PREFILL, null);
            }
            long nsPerFailure = (System.nanoTime() - started) / operations;
            long bytesPerFailure = (bean.getThreadAllocatedBytes(threadId) - before) / operations;
            // Unidentified Engine occupancy: 8432, never invented priority provenance.
            assertEquals(8432, result.failure().getCode());
            assertEquals(PlacementResult.Status.BLOCKED, result.status());
            assertTrue(endpoints.containsKey(result.blocker().endpoint()));
            System.out.printf("Prefill failure: workers=%d ns_per_failure=%d bytes_per_failure=%d%n",
                    workerCount, nsPerFailure, bytesPerFailure);
            if (workerCount == 1) { singleWorkerAllocation = bytesPerFailure; }
            assertTrue(bytesPerFailure <= singleWorkerAllocation + workerCount * 128L,
                    "full-fleet prediction must reuse candidate buffers rather than allocate a response per worker");
        }
    }
}
