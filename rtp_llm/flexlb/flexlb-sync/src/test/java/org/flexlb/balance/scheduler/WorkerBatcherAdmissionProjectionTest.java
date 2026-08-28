package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.delivery.DeliveryContext;
import org.flexlb.balance.delivery.DeliveryItem;
import org.flexlb.balance.delivery.DeliveryLifecyclePort;
import org.flexlb.balance.delivery.DeliveryMetadata;
import org.flexlb.balance.delivery.DeliveryStrategy;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.projection.RouteProjection;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.InternalRuntimeSettings;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.List;
import java.util.OptionalLong;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Route admission block projection switch
 * ({@code FLEXLB_ROUTE_ADMISSION_BLOCK_PROJECTION}, default ON) observed at
 * the exact seam it guards: {@code WorkerBatcher#admissionBlockUnderLock}.
 *
 * <p>Default (projection on): while the active head parks on an unavailable
 * capacity boundary, the route snapshot carries the admission card and
 * RouteAdmissionPolicy BLOCKs the endpoint from that observation.
 *
 * <p>Disabled (projection off): the head still parks on the exact same
 * boundary, but the snapshot never carries the card, so the endpoint is never
 * BLOCKED from an observed admission wait (queue-first form). The switch is
 * read once in the constructor from {@code config.getInternalRuntime()}; the
 * test flips the internal final field reflectively because the env-injecting
 * constructor is package-private to {@code org.flexlb.config}.</p>
 */
class WorkerBatcherAdmissionProjectionTest {

    private final List<WorkerBatcher> runtimes = new ArrayList<>();
    private FlexlbConfig config;
    private PrefillEndpoint prefillEndpoint;
    private ParkingDeliveryStrategy deliveryStrategy;

    @BeforeEach
    void setUp() {
        config = new FlexlbConfig();
        SchedulingTestConfig.usePriorityQueue(config);
        SchedulingTestConfig.useSingleDecision(config);
        prefillEndpoint = stablePrefillEndpoint();
        deliveryStrategy = new ParkingDeliveryStrategy();
    }

    @AfterEach
    void stopRuntimes() {
        for (WorkerBatcher runtime : runtimes) {
            assertNull(runtime.stopAndAwait());
        }
    }

    @Test
    void defaultSettingsProjectAdmissionCardAndBlockTheEndpoint() throws Exception {
        WorkerBatcher runtime = runningRuntime();
        long now = System.currentTimeMillis();
        assertTrue(runtime.offer(item(1, 50, now + 60_000, now, 128)));

        RouteProjection.Inputs inputs = awaitAdmissionCard(runtime);
        assertNotNull(inputs.queue().admissionBlock());
        assertFalse(inputs.queue().activeItems().isEmpty());

        RouteProjection.Candidate candidate = RouteProjection.project(
                inputs,
                probe(),
                RouteProjectionTestSupport.TOKEN_EVALUATOR,
                RouteProjectionTestSupport.ROUTE);
        assertEquals(RouteProjection.Result.State.BLOCKED,
                candidate.projection().state());
        assertFalse(candidate.projection().selectable());
    }

    @Test
    void disabledProjectionNeverBlocksTheEndpointWhileTheHeadStillParks() throws Exception {
        disableAdmissionBlockProjection(config);
        WorkerBatcher runtime = runningRuntime();
        long now = System.currentTimeMillis();
        assertTrue(runtime.offer(item(1, 50, now + 60_000, now, 128)));

        // The head still parks on the exact same unavailable boundary...
        awaitParkedDelivery();

        // ...but no admission card is ever exposed while it parks.
        long deadline = System.currentTimeMillis() + 500;
        while (System.currentTimeMillis() < deadline) {
            assertNull(runtime.captureRouteProjectionInputs().queue().admissionBlock());
            Thread.sleep(10);
        }

        RouteProjection.Inputs inputs = runtime.captureRouteProjectionInputs();
        assertNull(inputs.queue().admissionBlock());
        assertFalse(inputs.queue().activeItems().isEmpty());

        RouteProjection.Candidate candidate = RouteProjection.project(
                inputs,
                probe(),
                RouteProjectionTestSupport.TOKEN_EVALUATOR,
                RouteProjectionTestSupport.ROUTE);
        assertNotEquals(RouteProjection.Result.State.BLOCKED,
                candidate.projection().state());
    }

    private static RouteProjection.Probe probe() {
        return RouteProjectionTestSupport.probe(
                99L, 50, 20L, 0L, RouteProjection.Demand.TTFT_AND_DRAIN);
    }

    private WorkerBatcher runningRuntime() {
        WorkerBatcher runtime = new WorkerBatcher(
                "test-worker",
                prefillEndpoint,
                config,
                deliveryStrategy,
                mock(DeliveryLifecyclePort.class));
        runtimes.add(runtime);
        runtime.start();
        return runtime;
    }

    /** Poll until the parked head's admission card reaches the snapshot. */
    private RouteProjection.Inputs awaitAdmissionCard(WorkerBatcher runtime)
            throws InterruptedException {
        long deadline = System.currentTimeMillis() + 2_000;
        RouteProjection.Inputs latest = runtime.captureRouteProjectionInputs();
        while (latest.queue().admissionBlock() == null
                && System.currentTimeMillis() < deadline) {
            Thread.sleep(10);
            latest = runtime.captureRouteProjectionInputs();
        }
        if (latest.queue().admissionBlock() == null) {
            fail("admission card never reached the snapshot within 2s");
        }
        return latest;
    }

    /** Poll until the policy attempted one delivery (head then parks). */
    private void awaitParkedDelivery() throws InterruptedException {
        long deadline = System.currentTimeMillis() + 2_000;
        while (deliveryStrategy.attempts().get() == 0
                && System.currentTimeMillis() < deadline) {
            Thread.sleep(10);
        }
        if (deliveryStrategy.attempts().get() == 0) {
            fail("group policy never attempted a delivery within 2s");
        }
    }

    /**
     * The switch is a constructor-read final field sourced from
     * {@link InternalRuntimeSettings}, whose env-injecting constructor is
     * package-private to {@code org.flexlb.config}. Writing the final boolean
     * reflectively (non-static final instance field) is the only seam that
     * keeps production classes free of test-only setters.
     */
    private static void disableAdmissionBlockProjection(FlexlbConfig config) {
        try {
            Field field = InternalRuntimeSettings.class.getDeclaredField(
                    "routeAdmissionBlockProjectionEnabled");
            field.setAccessible(true);
            field.setBoolean(config.getInternalRuntime(), false);
        } catch (ReflectiveOperationException failure) {
            throw new IllegalStateException(
                    "test cannot disable route admission block projection",
                    failure);
        }
    }

    private BatchItem item(long requestId, int priority, long expiresAtMs,
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
        return new BatchItem(
                context,
                new CompletableFuture<Response>(),
                null,
                null,
                null,
                prefillEndpoint,
                null,
                null,
                0L,
                enqueuedAtMs);
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

    /**
     * Parks every exact head on one permanently-unavailable capacity boundary
     * with hard-block admission semantics — the same contract
     * {@code PrefillGenerationQueueTest} uses to exercise live ordering.
     */
    private static final class ParkingDeliveryStrategy
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

        AtomicInteger attempts() {
            return attempts;
        }

        @Override
        public <R> R admitAndDeliver(
                List<DeliveryItem> candidates,
                DeliveryMetadata metadata,
                PrefillTimePredictor.Evaluator evaluator,
                OptionalLong plannedPrediction,
                DeliveryContext<R> context) {
            attempts.incrementAndGet();
            return context.commitBoundary(
                    new DeliveryContext.SelectionBoundary(
                            candidates.getFirst(),
                            new CapacityBoundary.Unavailable(
                                    availability,
                                    new RouteProjection.AdmissionBlockSemantics(
                                            "TEST_QUEUE_BLOCK",
                                            RouteProjection.AfterProbeAdmission.BLOCKED,
                                            "TEST_QUEUE_BLOCK"))));
        }

        @Override
        public double projectGroupDurationMs(
                List<DeliveryItem> items,
                PrefillTimePredictor.Evaluator evaluator) {
            return 0.0;
        }

        @Override
        public RouteProjection.DeliveryProjection projectionPolicy() {
            return mock(RouteProjection.DeliveryProjection.class);
        }
    }
}
