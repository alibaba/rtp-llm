package org.flexlb.engine.grpc.monitor;
import io.grpc.Status;
import io.micrometer.core.instrument.simple.SimpleMeterRegistry;
import org.flexlb.metric.MicrometerFlexMonitor;
import org.flexlb.engine.grpc.EngineRpcService;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
class GrpcReporterTest {
    @Test void successFailureAndRetryAreLogicalCompletionsWithoutZeroFailurePayloads(){
        SimpleMeterRegistry registry=new SimpleMeterRegistry();
        try {
            GrpcReporter reporter=new GrpcReporter(new MicrometerFlexMonitor(registry));reporter.init();reporter.registerRole("127.0.0.1",8081,"PREFILL");
            var response=EngineRpcService.WorkerStatusPB.newBuilder().setAlive(true).setStatusVersion(1).build();
            var first=reporter.start("peer","127.0.0.1",8081,"status",System.nanoTime());first.complete(response.getSerializedSize(),null);first.complete(response.getSerializedSize(),null);
            var failed=reporter.start("peer","127.0.0.1",8081,"status",System.nanoTime());failed.complete(-1,Status.DEADLINE_EXCEEDED.asRuntimeException());
            var retry=reporter.start("peer","127.0.0.1",8081,"status",System.nanoTime());retry.retryStarted();retry.complete(response.getSerializedSize(),null);
            assertEquals(1,registry.find("flexlb.app.grpc.call.count").tags("code","OK","retry","false","role","PREFILL").counter().count());
            assertEquals(1,registry.find("flexlb.app.grpc.call.count").tags("code","DEADLINE_EXCEEDED").counter().count());
            assertEquals(1,registry.find("flexlb.app.grpc.response.size").tags("retry","false").summary().count());
            assertEquals(response.getSerializedSize(),registry.find("flexlb.app.grpc.response.size").tags("retry","false").summary().totalAmount());
            assertEquals(1,registry.find("flexlb.app.grpc.call.count").tags("code","OK","retry","true").counter().count());
        } finally { registry.close(); }
    }
    @Test void steadyRpcRecordingHasBoundedAllocation(){
        var bean=(com.sun.management.ThreadMXBean)java.lang.management.ManagementFactory.getThreadMXBean();
        org.junit.jupiter.api.Assumptions.assumeTrue(bean.isThreadAllocatedMemorySupported());bean.setThreadAllocatedMemoryEnabled(true);
        SimpleMeterRegistry registry=new SimpleMeterRegistry();
        try {
            GrpcReporter reporter=new GrpcReporter(new MicrometerFlexMonitor(registry));reporter.init();
            var response=EngineRpcService.WorkerStatusPB.newBuilder().setAlive(true).build();response.getSerializedSize();
            for(int i=0;i<20000;i++)reporter.start("peer","127.0.0.1",8081,"status",System.nanoTime()).complete(response.getSerializedSize(),null);
            long tid=Thread.currentThread().threadId(),start=bean.getThreadAllocatedBytes(tid);
            for(int i=0;i<100000;i++)reporter.start("peer","127.0.0.1",8081,"status",System.nanoTime()).complete(response.getSerializedSize(),null);
            double bytes=(bean.getThreadAllocatedBytes(tid)-start)/100000.0;
            System.out.println("grpc_metrics_allocated_bytes_per_completion="+bytes);
            assertTrue(bytes<512,"observability allocated "+bytes+" bytes/RPC");
        } finally { registry.close(); }
    }
}
