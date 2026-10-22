package org.flexlb.mockengine;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.flexlb.engine.grpc.EngineRpcService;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.concurrent.LinkedBlockingQueue;
import static org.junit.jupiter.api.Assertions.*;
import static org.flexlb.mockengine.MockEngineTestSupport.*;

class ReviewOwnershipRegressionTest {
    @TempDir Path directory;

    private static Object field(Object object, String name) throws Exception {
        var field = object.getClass().getDeclaredField(name);
        field.setAccessible(true);
        return field.get(object);
    }

    @Test void cancelBeforePhaseTwoCannotRecreateReceivedTaskOrPendingCredit() throws Exception {
        var model = performanceModel(directory, "1", 1);
        try (var cluster = MockEngineTestCluster.create(model, 62240, 1, 0)) {
            var service = cluster.prefill(0);
            long id = 7001;
            service.cancel(id);
            var cache = (MockLruBlockCache) field(service, "cache");
            var shape = model.shape(input(id, 512), cache);
            var claim = service.getClass().getDeclaredMethod("schedulePrefillCompletion", List.class, long.class, int.class);
            claim.setAccessible(true);
            assertEquals(true, claim.invoke(service, List.of(shape), 77L, 0));
            assertEquals(0, service.getRunningCount());
            assertEquals(0, ((java.util.concurrent.atomic.AtomicInteger) field(service, "pendingRequests")).get());
            assertEquals(0, service.inflightWorkSnapshot().get("active_prefill_batches"));
            var direct = service.getClass().getDeclaredMethod("admitDirectPrefill", MockPerformanceModel.RequestShape.class);
            direct.setAccessible(true);
            assertEquals(true, direct.invoke(service, shape));
            assertEquals(0, service.getRunningCount());
            assertEquals(0, ((java.util.concurrent.atomic.AtomicInteger) field(service, "pendingRequests")).get());
        }
    }

    @Test void housekeepingPreservesFreshQueuesAndLiveRemoteOwnerBuffers() throws Exception {
        try (var cluster = MockEngineTestCluster.create(performanceModel(directory, "1", 1), 62250, 1, 0)) {
            var service = cluster.prefill(0);
            @SuppressWarnings("unchecked")
            var queues = (Map<Long, LinkedBlockingQueue<EngineRpcService.GenerateOutputsPB>>) field(service, "responseQueues");
            var fresh = new MockResponseQueue();
            queues.put(7002L, fresh);
            service.getRequestStates().put(7002L, "received");
            service.periodicCleanup();
            assertSame(fresh, queues.get(7002L));
            assertEquals("received", service.getRequestStates().get(7002L));
            assertTrue(fresh.offer(EngineRpcService.GenerateOutputsPB.newBuilder().setRequestId(7002).build()));
        }
    }

    @Test void engineCopiesIncludeRuntimeDecodeOverrides() throws Exception {
        var model = performanceModel(directory, "1", 1);
        model.setRuntimeDecode(12, 3, 4);
        var copy = model.forEngine();
        assertEquals(model.tokensPerStep(), copy.tokensPerStep());
        assertEquals(model.decodeModelState(), copy.decodeModelState());
    }

    @Test void responseBufferRetentionIsMeasuredFromItsTerminalFrame() {
        var queue = new MockResponseQueue();
        assertFalse(queue.expiredTerminal(Long.MAX_VALUE, 1));
        queue.offer(EngineRpcService.GenerateOutputsPB.newBuilder().setRequestId(1)
                .setErrorInfo(EngineRpcService.RpcErrorPB.newBuilder().setErrorMessage("terminal")).build());
        assertFalse(queue.expiredTerminal(System.nanoTime(), 60_000));
        assertTrue(queue.expiredTerminal(System.nanoTime() + 2_000_000, 1));
    }
}
