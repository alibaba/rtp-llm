package org.flexlb.mockengine;

import java.nio.file.Path;
import java.util.concurrent.*;
import io.grpc.stub.StreamObserver;
import org.flexlb.engine.grpc.EngineRpcService;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import static org.junit.jupiter.api.Assertions.*;

class WorkerStatusMetricsTest {
    @TempDir Path directory;
    @Test void reportsCompletedStatusReadsAndKeepsSnapshotObserversIndependent() throws Exception {
        var scheduler = Executors.newScheduledThreadPool(2);
        var services = new ConcurrentHashMap<Integer, JavaMockEngineCluster.FastRpcService>();
        var service = new JavaMockEngineCluster.FastRpcService("prefill", EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL,
                0, services, scheduler, MockEngineTestSupport.performanceModel(directory, "10"), 100,
                new JavaMockEngineCluster.ClusterStats());
        try {
            int[] responses = {0};
            var observer = new StreamObserver<EngineRpcService.WorkerStatusPB>() {
                public void onNext(EngineRpcService.WorkerStatusPB value) { responses[0]++; }
                public void onError(Throwable error) { fail(error); }
                public void onCompleted() {}
            };
            for (int i=0; i<4; i++) service.getWorkerStatus(EngineRpcService.StatusVersionPB.getDefaultInstance(), observer);
            service.drainTpsWindows();
            var first = service.getMetricsSnapshot();
            assertEquals(4, responses[0]);
            assertEquals(4L, first.get("worker_status_rpcs_total"));
            assertTrue(((Number)first.get("worker_status_qps")).doubleValue() > 0);
            assertTrue(((Number)first.get("worker_status_rt_us")).doubleValue() >= 0);
            assertEquals(first.get("worker_status_qps"), service.getMetricsSnapshot().get("worker_status_qps"));
            service.setFaultConfig(FaultInjectionConfig.builder().statusNoRespond(true).build());
            service.getWorkerStatus(EngineRpcService.StatusVersionPB.getDefaultInstance(), observer);
            assertEquals(4L, service.getMetricsSnapshot().get("worker_status_rpcs_total"));
            assertEquals(MockMetricContract.Unit.MICROSECONDS,
                MockMetricContract.require("rtp_llm_rpc_worker_status_total_rt_us").unit());
        } finally { service.shutdown(); scheduler.shutdownNow(); }
    }
}
