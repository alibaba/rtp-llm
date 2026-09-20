package org.flexlb.balance.endpoint;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.delivery.DeliveryStrategy;
import org.flexlb.balance.scheduler.EndpointEventProjector;
import org.flexlb.balance.strategy.CostBasedPrefillStrategy;
import org.flexlb.balance.strategy.SelectedRole;
import org.flexlb.config.DispatcherConfig;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.lang.management.ManagementFactory;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;

/** Measures real endpoint aggregation; delivery and event sinks are not invoked in the measured loop. */
@Tag("performance-regression")
class PrefillAdmissionFailurePerformanceTest {
    @Test
    void unchangedFleetDoesNotAllocatePerWorker() throws Throwable {
        var classify = MethodHandles.privateLookupIn(CostBasedPrefillStrategy.class, MethodHandles.lookup())
                .findStatic(CostBasedPrefillStrategy.class, "classifyAdmissionFailure",
                        MethodType.methodType(PlacementResult.class, List.class, int.class, RoleType.class, String.class));
        var bean = (com.sun.management.ThreadMXBean) ManagementFactory.getThreadMXBean();
        assertTrue(bean.isThreadAllocatedMemorySupported());
        bean.setThreadAllocatedMemoryEnabled(true);
        long threadId = Thread.currentThread().threadId();
        FlexlbConfig config = new FlexlbConfig();
        config.setDispatcher(DispatcherConfig.nonBatch());
        config.getDispatcher().setMaxInflightPerPrefillWorker(1);
        var delivery = mock(DeliveryStrategy.class);
        var events = mock(EndpointEventProjector.class);
        var reporter = mock(BatchSchedulerReporter.class);
        long singleWorkerAllocation = 0L;
        for (int workerCount : new int[]{1, 64, 512, 1024}) {
            List<EndpointRegistry.PrefillRoutingEntry> directory = new ArrayList<>();
            for (int i = 0; i < workerCount; i++) {
                var status = EndpointTestSupport.workerStatus(RoleType.PREFILL, "worker-" + i, 8080, 8090);
                var endpoint = new PrefillEndpoint(status, config, delivery, events, reporter);
                var observation = new WorkerStatusResponse();
                observation.setAlive(true);
                observation.setRunningQueryLen(1L);
                EndpointTestSupport.applyStatus(endpoint, observation);
                directory.add(new EndpointRegistry.PrefillRoutingEntry(status.getIpPort(), endpoint));
            }
            directory = List.copyOf(directory);
            PlacementResult<SelectedRole, RoleType> result = null;
            for (int i = 0; i < 10_000; i++) {
                result = (PlacementResult<SelectedRole, RoleType>) classify.invokeExact(directory, 50, RoleType.PREFILL, (String) null);
            }
            long before = bean.getThreadAllocatedBytes(threadId);
            long started = System.nanoTime();
            for (int i = 0; i < 10_000; i++) {
                result = (PlacementResult<SelectedRole, RoleType>) classify.invokeExact(directory, 50, RoleType.PREFILL, (String) null);
            }
            long nsPerFailure = (System.nanoTime() - started) / 10_000;
            long bytesPerFailure = (bean.getThreadAllocatedBytes(threadId) - before) / 10_000;
            // Unidentified Engine occupancy: 8432, never invented priority provenance.
            assertEquals(8432, result.failure().getCode());
            assertEquals(workerCount, result.diagnostics().get("workers"));
            System.out.printf("Prefill failure: workers=%d ns_per_failure=%d bytes_per_failure=%d%n",
                    workerCount, nsPerFailure, bytesPerFailure);
            if (workerCount == 1) { singleWorkerAllocation = bytesPerFailure; }
            assertTrue(bytesPerFailure <= singleWorkerAllocation + 128,
                    "stable endpoint summaries must not allocate one response per worker");
        }
    }
}
