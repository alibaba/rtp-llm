package org.flexlb.balance.scheduler;

import org.flexlb.balance.resource.DynamicWorkerManager;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.service.monitor.RoutingQueueReporter;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.Mockito.lenient;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Unit tests for {@link QueueingComponent}: queue container semantics
 * (enqueue / requeueHead / capacity) and the permit-gated worker loop.
 *
 * <p>Migrated from the former QueueManagerTest (queue container) and the
 * queue-consumption half of RequestSchedulerTest.
 */
@ExtendWith(MockitoExtension.class)
class QueueingComponentTest {

    @Mock
    private RoutingQueueReporter metrics;
    @Mock
    private ConfigService configService;
    @Mock
    private DynamicWorkerManager dynamicWorkerManager;

    private final List<BalanceContext> consumed = new CopyOnWriteArrayList<>();
    private QueueingComponent queueing;

    @BeforeEach
    void setUp() {
        FlexlbConfig config = new FlexlbConfig();
        config.setMaxQueueSize(10);
        config.setScheduleWorkerSize(1);
        when(configService.loadBalanceConfig()).thenReturn(config);
        queueing = new QueueingComponent(configService, metrics, dynamicWorkerManager, consumed::add);
    }

    @AfterEach
    void tearDown() {
        queueing.shutdown();
    }

    // ==================== Queue container semantics ====================

    @Test
    void enqueue_shouldSucceedAndStampMetadata() {
        BalanceContext ctx = createContext(1L);

        assertTrue(queueing.enqueue(ctx));
        assertTrue(ctx.getEnqueueTime() > 0);
        assertTrue(ctx.getSequenceId() > 0);
        assertEquals(1, queueing.queueSize());
        verify(metrics).reportQueueEntry();
    }

    @Test
    void enqueue_shouldRejectWhenQueueFull() {
        for (int i = 0; i < 10; i++) {
            assertTrue(queueing.enqueue(createContext(i)));
        }

        assertFalse(queueing.enqueue(createContext(11L)));
        assertEquals(10, queueing.queueSize());
        verify(metrics).reportRejected();
    }

    @Test
    void requeueHead_shouldInsertAtFront() throws Exception {
        BalanceContext first = createContext(1L);
        assertTrue(queueing.enqueue(first));

        BalanceContext retried = createContext(2L);
        retried.setFuture(new CompletableFuture<>());
        retried.setEnqueueTime(System.currentTimeMillis());
        queueing.requeueHead(retried);

        // Start the worker: the head-of-queue (retried) request is consumed first
        when(dynamicWorkerManager.tryAcquirePermit(anyLong(), any())).thenReturn(true);
        queueing.start();
        awaitConsumed(2);
        assertEquals(2L, consumed.get(0).getRequestId());
        assertEquals(1L, consumed.get(1).getRequestId());
    }

    @Test
    void requeueHead_shouldCompleteWithErrorWhenQueueFull() {
        for (int i = 0; i < 10; i++) {
            queueing.enqueue(createContext(i));
        }

        BalanceContext ctx = createContext(99L);
        CompletableFuture<Response> future = new CompletableFuture<>();
        ctx.setFuture(future);

        queueing.requeueHead(ctx);

        assertTrue(future.isDone());
        Response response = future.join();
        assertFalse(response.isSuccess());
        assertEquals(StrategyErrorType.QUEUE_FULL.getErrorCode(), response.getCode());
    }

    @Test
    void remove_shouldTakeRequestOutOfQueue() {
        BalanceContext ctx = createContext(1L);
        queueing.enqueue(ctx);
        assertEquals(1, queueing.queueSize());

        queueing.remove(ctx);
        assertEquals(0, queueing.queueSize());
    }

    @Test
    void removeIfQueued_shouldBeSilentNoOpWhenAbsent() {
        BalanceContext ctx = createContext(1L);
        queueing.enqueue(ctx);

        queueing.removeIfQueued(ctx);
        assertEquals(0, queueing.queueSize());

        // Best-effort: removing an already-absent request must not throw.
        queueing.removeIfQueued(ctx);
        assertEquals(0, queueing.queueSize());
    }

    // ==================== Worker loop semantics ====================

    @Test
    void workerLoop_shouldConsumeEnqueuedRequestAndReleasePermit() throws Exception {
        when(dynamicWorkerManager.tryAcquirePermit(anyLong(), any())).thenReturn(true);

        BalanceContext ctx = createContext(1L);
        ctx.setFuture(new CompletableFuture<>());
        assertTrue(queueing.enqueue(ctx));

        queueing.start();
        awaitConsumed(1);
        assertEquals(1L, consumed.get(0).getRequestId());
        assertTrue(ctx.getDequeueTime() > 0);
        verify(metrics).reportQueueWaitingMetric(anyLong());
        verify(dynamicWorkerManager, org.mockito.Mockito.atLeastOnce()).releasePermit();
    }

    @Test
    void workerLoop_shouldNotConsumeWithoutPermit() throws Exception {
        lenient().when(dynamicWorkerManager.tryAcquirePermit(anyLong(), any())).thenReturn(false);

        BalanceContext ctx = createContext(1L);
        ctx.setFuture(new CompletableFuture<>());
        assertTrue(queueing.enqueue(ctx));

        queueing.start();
        Thread.sleep(600);
        assertTrue(consumed.isEmpty());
        assertEquals(1, queueing.queueSize());
    }

    @Test
    void workerLoop_shouldExpireRequestWaitingLongerThanGenerateTimeout() throws Exception {
        when(dynamicWorkerManager.tryAcquirePermit(anyLong(), any())).thenReturn(true);

        BalanceContext ctx = createContext(1L);
        ctx.getRequest().setGenerateTimeout(1); // 1ms — expired by the time the worker takes it
        CompletableFuture<Response> future = new CompletableFuture<>();
        ctx.setFuture(future);
        assertTrue(queueing.enqueue(ctx));
        Thread.sleep(20);

        queueing.start();

        CountDownLatch done = new CountDownLatch(1);
        future.whenComplete((r, t) -> done.countDown());
        assertTrue(done.await(3, TimeUnit.SECONDS));
        assertTrue(future.isCompletedExceptionally());
        try {
            future.join();
        } catch (Exception e) {
            assertNotNull(e.getCause());
            assertEquals(TimeoutException.class, e.getCause().getClass());
        }
        assertTrue(consumed.isEmpty());
    }

    @Test
    void workerLoop_shouldSkipAlreadySettledRequest() throws Exception {
        when(dynamicWorkerManager.tryAcquirePermit(anyLong(), any())).thenReturn(true);

        // First in queue: already settled (e.g. cancelled while queued).
        BalanceContext settled = createContext(1L);
        CompletableFuture<Response> settledFuture = new CompletableFuture<>();
        settledFuture.complete(Response.error(StrategyErrorType.QUEUE_FULL));
        settled.setFuture(settledFuture);
        assertTrue(queueing.enqueue(settled));

        BalanceContext live = createContext(2L);
        live.setFuture(new CompletableFuture<>());
        assertTrue(queueing.enqueue(live));

        queueing.start();
        awaitConsumed(1);
        // The settled request is skipped; only the live one reaches the consumer.
        assertEquals(2L, consumed.get(0).getRequestId());
        Thread.sleep(200);
        assertEquals(1, consumed.size());
        assertEquals(0, queueing.queueSize());
    }

    private void awaitConsumed(int expected) throws InterruptedException {
        long deadline = System.currentTimeMillis() + 3000;
        while (consumed.size() < expected && System.currentTimeMillis() < deadline) {
            Thread.sleep(20);
        }
        assertEquals(expected, consumed.size());
    }

    private BalanceContext createContext(long requestId) {
        BalanceContext ctx = new BalanceContext();
        Request request = new Request();
        request.setRequestId(requestId);
        request.setGenerateTimeout(60_000);
        ctx.setRequest(request);
        return ctx;
    }
}
