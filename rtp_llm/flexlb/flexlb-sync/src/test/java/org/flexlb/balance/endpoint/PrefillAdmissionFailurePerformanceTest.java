package org.flexlb.balance.endpoint;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.scheduler.AbstractRequestScheduler;
import org.flexlb.balance.strategy.CostBasedPrefillStrategy;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.config.ConfigTestSupport;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
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
        config.setDispatcher(ConfigTestSupport.nonBatch());
        config.getDispatcher().setMaxInflightPerPrefillWorker(1);
        var delivery = org.flexlb.balance.scheduler.SchedulerTestSupport.dispatcher();
        var events = mock(AbstractRequestScheduler.class);
        var reporter = mock(DeliveryMetricsReporter.class);
        long singleWorkerAllocation = 0L;
        for (int workerCount : new int[]{1, 64, 512, 1024}) {
            List<EndpointRegistry.PrefillRoutingEntry> directory = new ArrayList<>();
            for (int i = 0; i < workerCount; i++) {
                var status = EndpointTestSupport.workerStatus(RoleType.PREFILL, "worker-" + i, 8080, 8090);
                var endpoint = EndpointTestSupport.prefill(status, config, delivery, org.flexlb.balance.scheduler.SchedulerTestSupport.repository(events), reporter);
                var observation = new WorkerStatusResponse();
                observation.setAlive(true);
                observation.setRunningQueryLen(1L);
                EndpointTestSupport.applyStatus(endpoint, observation).run();
                directory.add(new EndpointRegistry.PrefillRoutingEntry(status.getIpPort(), endpoint));
            }
            directory = List.copyOf(directory);
            PlacementResult<ServerStatus, RoleType> result = null;
            for (int i = 0; i < 10_000; i++) {
                result = (PlacementResult<ServerStatus, RoleType>) classify.invokeExact(directory, 50, RoleType.PREFILL, (String) null);
            }
            long before = bean.getThreadAllocatedBytes(threadId);
            long started = System.nanoTime();
            for (int i = 0; i < 10_000; i++) {
                result = (PlacementResult<ServerStatus, RoleType>) classify.invokeExact(directory, 50, RoleType.PREFILL, (String) null);
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

    @Test
    void mixedPrioritiesAllocateOnlyTheReturnedSnapshot() {
        var bean = (com.sun.management.ThreadMXBean) ManagementFactory.getThreadMXBean();
        assertTrue(bean.isThreadAllocatedMemorySupported());
        bean.setThreadAllocatedMemoryEnabled(true);
        var state = new PrefillState(new java.util.concurrent.locks.ReentrantLock(), PrefillActiveIndex.ordered(4, java.util.Comparator.comparingLong(
                        org.flexlb.balance.scheduler.RequestRoute::requestId)), System::currentTimeMillis);
        long threadId = Thread.currentThread().threadId();
        for (int round = 0; round < 20_000; round++) {
            state.admissionSummary(round % 100 + 1, 1L);
        }
        long before = bean.getThreadAllocatedBytes(threadId);
        long started = System.nanoTime();
        for (int round = 0; round < 20_000; round++) {
            assertEquals(0L, state.admissionSummary(round % 100 + 1, 1L).occupiedRequests());
        }
        long elapsed = System.nanoTime() - started;
        long bytesPerQuery = (bean.getThreadAllocatedBytes(threadId) - before) / 20_000;
        System.out.printf("Prefill mixed priority: ns_per_query=%d bytes_per_query=%d%n",
                elapsed / 20_000, bytesPerQuery);
        assertTrue(bytesPerQuery <= 80L,
                "mixed priorities may allocate their immutable result, but no extra cache envelope");
    }
}
