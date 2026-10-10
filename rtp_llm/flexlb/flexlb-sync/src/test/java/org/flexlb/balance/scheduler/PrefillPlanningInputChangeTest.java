package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.EndpointTestSupport;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.master.WorkerStatusTestSupport;
import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.LockSupport;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

/** Planning input changes are independent from locally owned request/work changes. */
class PrefillPlanningInputChangeTest {

    @ParameterizedTest
    @CsvSource({
            "200,100,0,100, 200,100,100,100, 200,100",
            "200,100,100,100, 400,100,100,100, 400,100",
            "0,100,0,0, 0,300,0,0, 300,9223372036854775807",
            "0,0,0,0, 0,0,-1,100, 9223372036854775807,0"
    })
    void newerCapacityOnlyStatusRefreshesCachedConstraints(
            long oldBatchTokens, long oldSeqLen, long oldAvailableKv, long oldTotalKv,
            long newBatchTokens, long newSeqLen, long newAvailableKv, long newTotalKv,
            long expectedTokenCapacity, long expectedKvCapacity) {
        FlexlbConfig config = config();
        WorkerStatus status = status();
        publishInitial(status, response(1, oldBatchTokens, oldSeqLen, oldAvailableKv, oldTotalKv));
        PrefillEndpoint endpoint = endpoint(config, status);
        var before = endpoint.captureRouteProjectionInputs();
        var progress = endpoint.ownership.progress();
        var worker = EndpointTestSupport.batcher(endpoint);
        clearInvocations(worker);

        apply(endpoint, response(2, newBatchTokens, newSeqLen, newAvailableKv, newTotalKv));

        var after = endpoint.captureRouteProjectionInputs();
        assertEquals(2, status.appliedStatusCursor().statusVersion());
        assertNotSame(before, after, "effective capacity changes must invalidate the cached planning source");
        assertEquals(expectedTokenCapacity, after.queue().constraints().batchTokenCapacity());
        assertEquals(expectedKvCapacity, after.queue().constraints().batchKvCapacity());
        assertEquals(progress.ownershipVersion(), endpoint.ownership.progress().ownershipVersion(),
                "capacity-only updates do not create an ownership mutation");
        assertEquals(progress.schedulingInputVersion() + 1,
                endpoint.ownership.progress().schedulingInputVersion());
        verify(worker, times(1)).wake();
        assertEquals(oldAvailableKv <= 0 && oldTotalKv > 0 ? 0 : oldTotalKv <= 0
                ? Long.MAX_VALUE : Math.min(oldAvailableKv, oldTotalKv),
                before.queue().constraints().batchKvCapacity(), "the prior immutable snapshot stays frozen");
    }

    @Test
    void differentRawStatusWithEqualEffectiveConstraintsKeepsCachedSource() {
        FlexlbConfig config = config();
        WorkerStatus status = status();
        publishInitial(status, response(1, 200, 100, 150, 100));
        PrefillEndpoint endpoint = endpoint(config, status);
        var before = endpoint.captureRouteProjectionInputs();
        var progress = endpoint.ownership.progress();
        var worker = EndpointTestSupport.batcher(endpoint);
        clearInvocations(worker);

        apply(endpoint, response(2, 200, 999, 200, 100));

        assertEquals(2, status.appliedStatusCursor().statusVersion());
        assertSame(progress, endpoint.ownership.progress(), "equal normalized constraints need no wake revision");
        assertSame(before, endpoint.captureRouteProjectionInputs());
        verify(worker, never()).signal(anyBoolean());
        verify(worker, never()).wake();
    }

    @Test
    @Timeout(value = 10, unit = TimeUnit.SECONDS)
    void restoredKvWakesAlreadyWaitingHeadWithoutQueueOrEngineActivityChange() throws Exception {
        FlexlbConfig config = config();
        WorkerStatus status = status();
        publishInitial(status, response(1, 1000, 1000, 0, 100));
        PrefillEndpoint endpoint = endpoint(config, status);
        WorkerBatcher worker = EndpointTestSupport.batcher(endpoint);
        CountDownLatch selected = new CountDownLatch(1);
        doAnswer(invocation -> {
            List<RequestRoute> candidates = invocation.getArgument(0);
            RequestRoute exact = candidates.getFirst();
            assertTrue(worker.control(exact, WorkerBatcher.Control.REMOVE));
            selected.countDown();
            return null;
        }).when(worker).processSelection(anyList(), anyString(), any(), any());
        RequestContext context = new RequestContext(config);
        context.data().setRequest(Request.builder().requestId(51L).rawPriority(0).seqLen(10L).build());
        context.data().setSchedulingMetadata(SchedulingMetadata.explicit(0, Long.MAX_VALUE));
        RequestRoute route = SchedulingTestConfig.createRoute(context, null, null, null,
                endpoint, null, null, System.currentTimeMillis());
        worker.start();
        try {
            assertTrue(worker.offer(route));
            awaitKvWait(worker, endpoint);
            var queueBefore = endpoint.ownership.progress().queueVersion();

            apply(endpoint, response(2, 1000, 1000, 100, 100));

            assertTrue(selected.await(3, TimeUnit.SECONDS),
                    "KV recovery must wake the retained head before its request expiration");
            assertEquals(queueBefore + 1, endpoint.ownership.progress().queueVersion(),
                    "the only queue change is removing the resumed exact selection");
            assertEquals(0, endpoint.ownership.progress().queueDepth());
        } finally {
            assertNull(worker.stopAndAwait());
        }
    }

    private static void awaitKvWait(WorkerBatcher worker, PrefillEndpoint endpoint) {
        Condition condition = (Condition) ReflectionTestUtils.getField(worker, "stateChanged");
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(3);
        while (System.nanoTime() < deadline) {
            endpoint.ownership.lock.lock();
            try {
                if ("Prefill KV capacity exhausted".equals(worker.getLatestQueueWaitSnapshot().get("cause"))
                        && endpoint.ownership.lock.hasWaiters(condition)) { return; }
            } finally { endpoint.ownership.lock.unlock(); }
            LockSupport.parkNanos(TimeUnit.MILLISECONDS.toNanos(1));
        }
        fail("worker did not enter the exact KV scheduling-input wait");
    }

    private static FlexlbConfig config() {
        FlexlbConfig config = SchedulingTestConfig.newConfig();
        SchedulingTestConfig.useFifoQueue(config);
        SchedulingTestConfig.useSingleDecision(config);
        SchedulingTestConfig.useBatchDispatcher(config);
        return config;
    }

    private static WorkerStatus status() {
        return WorkerStatus.createDiscovered(RoleType.PREFILL, "planning-input-test",
                "127.0.0.1", 8080, 8090, null);
    }

    private static PrefillEndpoint endpoint(FlexlbConfig config, WorkerStatus status) {
        return EndpointTestSupport.unstartedPrefill(config, status, SchedulerTestSupport.dispatcher(),
                mock(AbstractRequestScheduler.class));
    }

    private static WorkerStatusResponse response(long version, long batchTokens, long seqLen,
                                                 long availableKv, long totalKv) {
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRole(RoleType.PREFILL);
        response.setAlive(true);
        response.setStatusVersion(version);
        response.setMaxBatchTokensSize(batchTokens);
        response.setMaxSeqLen(seqLen);
        response.setAvailableKvCacheTokens(availableKv);
        response.setTotalKvCacheTokens(totalKv);
        return response;
    }

    private static void publishInitial(WorkerStatus status, WorkerStatusResponse response) {
        status.lock.lock();
        try { status.publishPreparedStatus(status.prepareNewStatus(
                WorkerStatusTestSupport.freezeStatusResponse(status, response))); }
        finally { status.lock.unlock(); }
    }

    private static void apply(PrefillEndpoint endpoint, WorkerStatusResponse response) {
        WorkerStatus status = endpoint.getStatus();
        org.flexlb.balance.endpoint.WorkerEndpoint.StatusChange effects;
        status.lock.lock();
        try { effects = endpoint.applyPreparedStatus(status, status.prepareNewStatus(
                WorkerStatusTestSupport.freezeStatusResponse(status, response))); }
        finally { status.lock.unlock(); }
        endpoint.publishStatus(effects);
    }
}
