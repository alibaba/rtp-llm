package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.scheduler.ScheduledRequest;
import org.flexlb.balance.scheduler.EndpointEventProjector;
import org.flexlb.balance.delivery.DeliveryStrategy;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.scheduler.WorkerBatcher;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.projection.RouteProjection;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.OptionalLong;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/** Canonical queue-order contract exposed by {@link WorkerBatcher}. */
class PrefillGenerationQueueTest {

    private final List<WorkerBatcher> runtimes = new ArrayList<>();
    private FlexlbConfig config;
    private PrefillEndpoint prefillEndpoint;
    private BlockingDeliveryStrategy deliveryStrategy;

    @BeforeEach
    void setUp() {
        config = new FlexlbConfig();
        SchedulingTestConfig.usePriorityQueue(config);
        SchedulingTestConfig.useSingleDecision(config);
        prefillEndpoint = stablePrefillEndpoint();
        deliveryStrategy = new BlockingDeliveryStrategy();
    }

    @AfterEach
    void stopRuntimes() {
        for (WorkerBatcher runtime : runtimes) {
            assertNull(runtime.stopAndAwait());
        }
    }

    @Test
    void priorityOrderIsPriorityDescendingThenActualOfferFifo() {
        WorkerBatcher runtime = runningRuntime();
        long now = System.currentTimeMillis();

        assertTrue(runtime.offer(item(1, 50, now + 5_000, now, 128)));
        assertTrue(runtime.offer(item(2, 70, now + 9_000, now + 100, 128)));
        assertTrue(runtime.offer(item(3, 50, now + 1_000, now + 200, 128)));
        assertTrue(runtime.offer(item(4, 50, now + 5_000, now - 100, 128)));

        WorkerBatcher.QueueSnapshot snapshot =
                runtime.captureQueueSnapshot();
        assertEquals(List.of(2L, 1L, 3L, 4L), requestIds(snapshot.items()));
        assertEquals(4, snapshot.items().size());
        assertEquals(SchedulingTestConfig.useQueueCapacity(config)
                .getMaxWaitingRequestsPerPrefillWorker(), snapshot.queueCapacity());
        assertEquals("test-worker", snapshot.endpointId());
    }

    @Test
    void priorityOrderUsesUniqueOfferSequenceBeforeRequestIdOrCallerTimestamp() {
        WorkerBatcher runtime = runningRuntime();
        long now = System.currentTimeMillis();

        assertTrue(runtime.offer(item(1, 50, now + 9_000, now, 128)));
        assertTrue(runtime.offer(item(2, 50, now + 1_000, now, 128)));
        assertTrue(runtime.offer(item(4, 50, now + 9_000, now, 128)));
        assertTrue(runtime.offer(item(3, 50, now + 9_000, now, 128)));

        assertEquals(List.of(1L, 2L, 4L, 3L), requestIds(
                runtime.captureQueueSnapshot().items()));
    }

    @Test
    void fifoOrderIgnoresPriorityAndCallerTimestamp() {
        SchedulingTestConfig.useFifoQueue(config);
        WorkerBatcher runtime = runningRuntime();
        long now = System.currentTimeMillis();

        assertTrue(runtime.offer(item(1, 30, now + 1_000, now, 128)));
        assertTrue(runtime.offer(item(2, 50, now + 500, now + 100, 128)));
        assertTrue(runtime.offer(item(3, 70, now + 100, now + 200, 128)));

        assertEquals(List.of(1L, 2L, 3L), requestIds(
                runtime.captureQueueSnapshot().items()));
    }

    @Test
    void snapshotsExposeExactCanonicalIdentitiesAndMonotonicMutationVersion() {
        WorkerBatcher runtime = runningRuntime();
        ScheduledRequest first = item(11, 70, Long.MAX_VALUE, 1, 128);
        ScheduledRequest second = item(12, 50, Long.MAX_VALUE, 2, 128);

        WorkerBatcher.QueueSnapshot empty =
                runtime.captureQueueSnapshot();
        assertTrue(runtime.offer(first));
        assertTrue(runtime.offer(second));
        WorkerBatcher.QueueSnapshot offered =
                runtime.captureQueueSnapshot();

        assertTrue(offered.queueVersion() > empty.queueVersion());
        assertSame(first, offered.items().get(0));
        assertSame(second, offered.items().get(1));
        assertEquals(2, runtime.queueSize());
        assertEquals(java.util.Map.of(70, 1, 50, 1),
                runtime.queueSizeByPriority());

        assertTrue(runtime.removeQueued(first, "test exact removal"));
        WorkerBatcher.QueueSnapshot removed =
                runtime.captureQueueSnapshot();
        assertTrue(removed.queueVersion() > offered.queueVersion());
        assertEquals(List.of(second), removed.items());
    }

    @Test
    void removalRequiresTheExactQueuedIdentityEvenForSameRequestId() {
        WorkerBatcher runtime = runningRuntime();
        ScheduledRequest canonical = item(21, 50, Long.MAX_VALUE, 1, 128);
        ScheduledRequest lookalike = item(21, 50, Long.MAX_VALUE, 1, 128);
        assertTrue(runtime.offer(canonical));

        assertFalse(runtime.removeQueued(lookalike, "stale identity"));
        assertEquals(List.of(canonical),
                runtime.captureQueueSnapshot().items());
        assertTrue(runtime.removeQueued(canonical, "canonical identity"));
    }

    private WorkerBatcher runningRuntime() {
        WorkerBatcher runtime = new WorkerBatcher(
                "test-worker",
                prefillEndpoint,
                config,
                deliveryStrategy,
                mock(EndpointEventProjector.class));
        runtimes.add(runtime);
        runtime.start();
        return runtime;
    }

    private ScheduledRequest item(long requestId, int priority, long expiresAtMs,
                           long enqueuedAtMs, long seqLen) {
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(seqLen);
        request.setPriority(priority);
        BalanceContext context = new BalanceContext();
        context.setRequest(request);
        context.setConfig(config);
        context.setSchedulingMetadata(
                SchedulingMetadata.explicit(priority, expiresAtMs));
        return new ScheduledRequest(
                context,
                new CompletableFuture<Response>(),
                null,
                null,
                null,
                prefillEndpoint,
                null,
                null,
                enqueuedAtMs);
    }

    private static List<Long> requestIds(List<ScheduledRequest> items) {
        return items.stream().map(ScheduledRequest::requestId).toList();
    }

    private static PrefillEndpoint stablePrefillEndpoint() {
        PrefillTimePredictor.Evaluator evaluator =
                mock(PrefillTimePredictor.Evaluator.class);
        PrefillTimePredictor predictor = mock(PrefillTimePredictor.class);
        when(predictor.evaluator()).thenReturn(evaluator);
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        when(endpoint.getPredictor()).thenReturn(predictor);
        when(endpoint.getStatus()).thenReturn(WorkerStatus.createDiscovered(
                RoleType.PREFILL,
                "test",
                "127.0.0.1",
                8080,
                8090,
                "test-site"));
        return endpoint;
    }

    /** Holds every exact head ACTIVE so snapshots can exercise live ordering. */
    private static final class BlockingDeliveryStrategy
            implements DeliveryStrategy {

        private final AtomicInteger attempts = new AtomicInteger();
        private final CapacityBoundary.Availability availability =
                new CapacityBoundary.Availability() {
                    @Override
                    public boolean isAvailable() {
                        return false;
                    }

                    @Override
                    public void addListener(Runnable listener) {
                    }

                    @Override
                    public void removeListener(Runnable listener) {
                    }
                };

        @Override
        public Transaction prepare(
                List<ScheduledRequest> candidates,
                PrefillTimePredictor.Evaluator evaluator,
                OptionalLong plannedPrediction) {
            attempts.incrementAndGet();
            return WorkerBatcherTestSupport.boundaryOnly(
                    candidates.getFirst(),
                    CapacityBoundary.unavailable(
                            availability,
                            new RouteProjection.AdmissionBlockSemantics(
                                    "TEST_QUEUE_BLOCK",
                                    RouteProjection.AfterProbeAdmission.BLOCKED,
                                    "TEST_QUEUE_BLOCK",
                                    RoleType.PREFILL)));
        }

        @Override
        public double projectGroupDurationMs(
                List<ScheduledRequest> items,
                PrefillTimePredictor.Evaluator evaluator) {
            return 0.0;
        }

        @Override
        public RouteProjection.DeliveryProjection projectionPolicy() {
            return mock(RouteProjection.DeliveryProjection.class);
        }
    }
}
