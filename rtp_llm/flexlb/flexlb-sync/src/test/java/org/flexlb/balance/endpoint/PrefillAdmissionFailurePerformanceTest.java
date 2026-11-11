package org.flexlb.balance.endpoint;

import org.flexlb.balance.scheduler.AbstractRequestScheduler;
import org.flexlb.config.ConfigTestSupport;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.loadbalance.AdmissionRejectReason;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.lang.management.ManagementFactory;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;

/**
 * Measures selected-worker capacity classification without delivery or event sinks in the loop.
 */
@Tag("performance-regression")
class PrefillAdmissionFailurePerformanceTest {
    @Test
    void unchangedFleetDoesNotAllocatePerWorker() throws Throwable {
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
        for (int workerCount : new int[] {1, 64, 512, 1024}) {
            List<PrefillEndpoint> directory = new ArrayList<>();
            for (int i = 0; i < workerCount; i++) {
                var status =
                        EndpointTestSupport.workerStatus(
                                RoleType.PREFILL, "worker-" + i, 8080, 8090);
                var endpoint =
                        EndpointTestSupport.prefill(
                                status,
                                config,
                                delivery,
                                org.flexlb.balance.scheduler.SchedulerTestSupport.repository(
                                        events),
                                reporter);
                var observation = new WorkerStatusResponse();
                observation.setAlive(true);
                observation.setRunningQueryLen(1L);
                EndpointTestSupport.applyStatus(endpoint, observation).run();
                assertEquals(
                        AdmissionRejectReason.UNSPECIFIED, endpoint.explainAdmissionRejection(50));
                directory.add(endpoint);
            }
            directory = List.copyOf(directory);
            AdmissionRejectReason result = null;
            for (int i = 0; i < 10_000; i++) {
                var selected = directory.get(i % workerCount);
                result = selected.explainAdmissionRejection(50);
            }
            long before = bean.getThreadAllocatedBytes(threadId);
            long started = System.nanoTime();
            for (int i = 0; i < 10_000; i++) {
                var selected = directory.get(i % workerCount);
                result = selected.explainAdmissionRejection(50);
            }
            long nsPerFailure = (System.nanoTime() - started) / 10_000;
            long bytesPerFailure = (bean.getThreadAllocatedBytes(threadId) - before) / 10_000;
            // Unknown Engine occupancy must not be attributed to a priority.
            assertEquals(AdmissionRejectReason.UNSPECIFIED, result);
            System.out.printf(
                    "Prefill failure: workers=%d ns_per_failure=%d bytes_per_failure=%d%n",
                    workerCount, nsPerFailure, bytesPerFailure);
            if (workerCount == 1) {
                singleWorkerAllocation = bytesPerFailure;
            }
            assertTrue(
                    bytesPerFailure <= singleWorkerAllocation + 128,
                    "rejection explanations must not allocate one response per worker");
        }
    }

    @Test
    void repeatedExplanationsDoNotAllocateSnapshots() {
        var bean = (com.sun.management.ThreadMXBean) ManagementFactory.getThreadMXBean();
        assertTrue(bean.isThreadAllocatedMemorySupported());
        bean.setThreadAllocatedMemoryEnabled(true);
        var state =
                org.flexlb.balance.endpoint.EndpointTestSupport.prefillEndpoint(
                        new java.util.concurrent.locks.ReentrantLock(),
                        PrefillRequestQueue.fifo(4),
                        System::currentTimeMillis);
        long threadId = Thread.currentThread().threadId();
        for (int round = 0; round < 20_000; round++) {
            state.explainAdmissionRejection(round % 100 + 1);
        }
        long before = bean.getThreadAllocatedBytes(threadId);
        long started = System.nanoTime();
        for (int round = 0; round < 20_000; round++) {
            assertEquals(
                    AdmissionRejectReason.RESOURCE_EXHAUSTED,
                    state.explainAdmissionRejection(round % 100 + 1));
        }
        long elapsed = System.nanoTime() - started;
        long bytesPerQuery = (bean.getThreadAllocatedBytes(threadId) - before) / 20_000;
        System.out.printf(
                "Prefill mixed priority: ns_per_query=%d bytes_per_query=%d%n",
                elapsed / 20_000, bytesPerQuery);
        assertTrue(
                bytesPerQuery <= 80L, "rejection explanations do not need a statistics snapshot");
    }
}
