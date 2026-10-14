package org.flexlb.service.monitor;

import io.micrometer.core.instrument.simple.SimpleMeterRegistry;
import org.flexlb.cache.core.EngineLocalView;
import org.flexlb.cache.core.GlobalCacheIndex;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.metric.MicrometerFlexMonitor;
import org.flexlb.sync.runner.RunnerTestSupport;
import org.flexlb.sync.status.WorkerDirectory;
import org.junit.jupiter.api.Test;
import org.springframework.test.util.ReflectionTestUtils;
import java.util.Map;
import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

class SynchronizationMetricsTest {
    @Test void samplerBindsRoleBeforeFirstRpcAndClearsAbsentWorkerAges() {
        var registry = new SimpleMeterRegistry();
        var monitor = new MicrometerFlexMonitor(registry);
        var directory = mock(WorkerDirectory.class);
        var grpc = mock(EngineGrpcClient.class);
        var local = mock(EngineLocalView.class);
        var global = mock(GlobalCacheIndex.class);
        var status = RunnerTestSupport.alive(RoleType.PREFILL,"g","127.0.0.1",8080,8081,"s");
        for (RoleType role : RoleType.values()) when(directory.statusSnapshot(role)).thenReturn(Map.of());
        when(directory.statusSnapshot(RoleType.PREFILL)).thenReturn(Map.of(status.getIpPort(),status));
        var metrics = new SynchronizationMetrics(monitor,directory,local,global,grpc);
        try {
            metrics.start();metrics.close();
            try { assertTrue(((java.util.concurrent.ExecutorService)ReflectionTestUtils.getField(metrics,"sampler")).awaitTermination(5,java.util.concurrent.TimeUnit.SECONDS)); } catch(InterruptedException e) { throw new AssertionError(e); }
            // Manual samples make absence handling deterministic.
            metrics.ensure(status);
            verify(grpc,atLeastOnce()).registerPeerRole("127.0.0.1",8081,"PREFILL");
            status.statusObservation.started();status.statusObservation.received(100,0,0);status.statusObservation.ready(true);
            ReflectionTestUtils.invokeMethod(metrics,"sample");
            var age = registry.find("flexlb.app.sync.response.age.ms").tags("engineIp","127.0.0.1","service","status").gauge();
            assertTrue(age.value() >= 0);
            when(directory.statusSnapshot(RoleType.PREFILL)).thenReturn(Map.of());
            ReflectionTestUtils.invokeMethod(metrics,"sample");
            assertTrue(Double.isNaN(age.value()));
            assertEquals(0,registry.find("flexlb.app.sync.pending.response.bytes").gauge().value());
        } finally { metrics.close();registry.close(); }
    }
    @Test void responseClockAndCommitClockRemainSeparate() {
        var observation = new org.flexlb.diagnostics.SyncObservation();
        observation.started();observation.received(100,2,3);
        long receipt=observation.responseUs;
        observation.ready(true);
        assertEquals(receipt,observation.lastAliveResponseUs);
        assertEquals(100,observation.payloadBytes);
        observation.applying();observation.done();
        assertEquals(0,observation.payloadBytes);
        assertEquals(receipt,observation.lastAliveResponseUs);
    }
}
