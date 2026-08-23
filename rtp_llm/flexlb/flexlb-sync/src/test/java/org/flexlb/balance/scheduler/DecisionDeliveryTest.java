package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;

class DecisionDeliveryTest {

    @Test
    void deliveriesRequireRequestsForTheirOwnMode() {
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        BatchItem batchItem = item(DeliveryMode.BATCH_ENQUEUE, endpoint);
        BatchItem routeItem = item(DeliveryMode.ROUTE_DECISION, endpoint);
        RecordingCallback callback = new RecordingCallback();

        assertThrows(IllegalArgumentException.class, () -> new BatchEnqueueDelivery.Plan(
                List.of(batchItem, routeItem), endpoint, 701, 0, "mixed"));
        assertThrows(IllegalArgumentException.class,
                () -> RouteDecisionDelivery.INSTANCE.deliver(
                        List.of(routeItem, batchItem), callback));
    }

    @Test
    void batchEnqueuePlanRequiresOnePrefillEndpoint() {
        PrefillEndpoint firstEndpoint = mock(PrefillEndpoint.class);
        PrefillEndpoint secondEndpoint = mock(PrefillEndpoint.class);
        BatchItem first = item(DeliveryMode.BATCH_ENQUEUE, firstEndpoint);
        BatchItem second = item(DeliveryMode.BATCH_ENQUEUE, secondEndpoint);

        assertThrows(NullPointerException.class, () -> new BatchEnqueueDelivery.Plan(
                List.of(first), null, 701, 0, "missing_endpoint"));
        assertThrows(IllegalArgumentException.class, () -> new BatchEnqueueDelivery.Plan(
                List.of(first, second), firstEndpoint, 701, 0, "mixed_endpoints"));
    }

    @Test
    void routeDecisionDeliveryRequiresOnePrefillEndpoint() {
        PrefillEndpoint firstEndpoint = mock(PrefillEndpoint.class);
        PrefillEndpoint secondEndpoint = mock(PrefillEndpoint.class);
        BatchItem first = item(DeliveryMode.ROUTE_DECISION, firstEndpoint);
        BatchItem second = item(DeliveryMode.ROUTE_DECISION, secondEndpoint);
        RecordingCallback callback = new RecordingCallback();

        assertThrows(NullPointerException.class,
                () -> RouteDecisionDelivery.INSTANCE.deliver(
                        List.of(item(DeliveryMode.ROUTE_DECISION, null)), callback));
        assertThrows(IllegalArgumentException.class,
                () -> RouteDecisionDelivery.INSTANCE.deliver(
                        List.of(first, second), callback));
    }

    @Test
    void batchEnqueuePlanRequiresPositiveBatchId() {
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        BatchItem batchItem = item(DeliveryMode.BATCH_ENQUEUE, endpoint);

        assertThrows(IllegalArgumentException.class, () -> new BatchEnqueueDelivery.Plan(
                List.of(batchItem), endpoint, 0, 0, "invalid"));
        assertThrows(IllegalArgumentException.class, () -> new BatchEnqueueDelivery.Plan(
                List.of(batchItem), endpoint, -1, 0, "invalid"));
    }

    @Test
    void routeDecisionDeliveryReportsEveryRequestInOrder() {
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        BatchItem first = item(DeliveryMode.ROUTE_DECISION, endpoint);
        BatchItem second = item(DeliveryMode.ROUTE_DECISION, endpoint);
        List<BatchItem> items = List.of(first, second);
        RecordingCallback callback = new RecordingCallback();

        RouteDecisionDelivery.INSTANCE.deliver(items, callback);

        assertEquals(items, callback.delivered);
    }

    @Test
    void routeDecisionDeliveryIsolatesBothDeliveryAndFailureCallbacksPerRequest() {
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        BatchItem first = item(DeliveryMode.ROUTE_DECISION, endpoint);
        BatchItem second = item(DeliveryMode.ROUTE_DECISION, endpoint);
        BatchItem third = item(DeliveryMode.ROUTE_DECISION, endpoint);
        RuntimeException firstFailure = new RuntimeException("first delivery failed");
        RuntimeException secondFailure = new RuntimeException("second delivery failed");
        List<BatchItem> attempts = new ArrayList<>();
        List<BatchItem> delivered = new ArrayList<>();
        List<BatchItem> failed = new ArrayList<>();
        List<Throwable> errors = new ArrayList<>();
        DecisionDelivery.Callback callback = new DecisionDelivery.Callback() {
            @Override
            public void onDelivered(BatchItem item) {
                attempts.add(item);
                if (item == first) {
                    throw firstFailure;
                }
                if (item == second) {
                    throw secondFailure;
                }
                delivered.add(item);
            }

            @Override
            public void onFailure(BatchItem item, Throwable error) {
                failed.add(item);
                errors.add(error);
                if (item == first) {
                    throw new IllegalStateException("failure callback also failed");
                }
            }
        };

        assertDoesNotThrow(() -> RouteDecisionDelivery.INSTANCE.deliver(
                List.of(first, second, third), callback));

        assertEquals(List.of(first, second, third), attempts);
        assertEquals(List.of(third), delivered);
        assertEquals(List.of(first, second), failed);
        assertEquals(2, errors.size());
        assertSame(firstFailure, errors.get(0));
        assertSame(secondFailure, errors.get(1));
    }

    @Test
    void batchEnqueueDeliveryDelegatesPreparedPlanThroughTransportAdapter() {
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        RecordingBatchDispatcher dispatcher = new RecordingBatchDispatcher();
        BatchEnqueueDelivery delivery = new BatchEnqueueDelivery(dispatcher);
        List<BatchItem> items = List.of(
                item(DeliveryMode.BATCH_ENQUEUE, endpoint),
                item(DeliveryMode.BATCH_ENQUEUE, endpoint));
        RecordingCallback callback = new RecordingCallback();
        BatchEnqueueDelivery.Plan plan = new BatchEnqueueDelivery.Plan(
                items, endpoint, 701, 83, "fixed_window");

        BatchDispatcher.SubmissionReserved reserved = assertInstanceOf(
                BatchDispatcher.SubmissionReserved.class,
                delivery.tryReserveSubmission());
        BatchEnqueueDelivery.Submission submission = delivery.prepare(
                plan, reserved.permit(), callback);
        submission.submit();

        assertEquals(1, dispatcher.reservationCount);
        assertEquals(1, dispatcher.submissionCount);
        assertSame(items, dispatcher.items);
        assertSame(endpoint, dispatcher.prefillEndpoint);
        assertEquals(701, dispatcher.batchId);
        assertEquals(83, dispatcher.predictedMs);
        assertEquals("fixed_window", dispatcher.reason);
        assertNotSame(callback, dispatcher.callback);

        dispatcher.callback.onSuccess(items.get(0), 701);
        assertEquals(List.of(), callback.delivered,
                "callbacks remain closed until the lifecycle handoff completes");
        submission.releaseCallbacks();
        assertEquals(List.of(items.get(0)), callback.delivered);
    }

    @Test
    void batchEnqueueAdapterTreatsMismatchedCallbackIdAsUncertain() {
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        RecordingBatchDispatcher dispatcher = new RecordingBatchDispatcher();
        BatchEnqueueDelivery delivery = new BatchEnqueueDelivery(dispatcher);
        BatchItem item = item(DeliveryMode.BATCH_ENQUEUE, endpoint);
        RecordingCallback callback = new RecordingCallback();
        BatchDispatcher.SubmissionReserved reserved = assertInstanceOf(
                BatchDispatcher.SubmissionReserved.class,
                delivery.tryReserveSubmission());
        BatchEnqueueDelivery.Submission submission = delivery.prepare(
                new BatchEnqueueDelivery.Plan(
                        List.of(item), endpoint, 701, 0, "batch_id_fence"),
                reserved.permit(), callback);
        submission.submit();
        submission.releaseCallbacks();

        dispatcher.callback.onSuccess(item, 702);

        assertEquals(List.of(), callback.delivered);
        assertEquals(List.of(item), callback.uncertain);
        assertTrue(callback.uncertainErrors.get(0).getMessage()
                .contains("does not match delivery plan batch id 701"));
    }

    @Test
    void batchEnqueueAdapterMapsTimeoutAndUncertainOutcomes() {
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        RecordingBatchDispatcher dispatcher = new RecordingBatchDispatcher();
        BatchEnqueueDelivery delivery = new BatchEnqueueDelivery(dispatcher);
        BatchItem item = item(DeliveryMode.BATCH_ENQUEUE, endpoint);
        RecordingCallback callback = new RecordingCallback();
        BatchDispatcher.SubmissionReserved reserved = assertInstanceOf(
                BatchDispatcher.SubmissionReserved.class,
                delivery.tryReserveSubmission());
        BatchEnqueueDelivery.Submission submission = delivery.prepare(
                new BatchEnqueueDelivery.Plan(
                        List.of(item), endpoint, 701, 0, "transport_outcomes"),
                reserved.permit(), callback);
        submission.submit();
        submission.releaseCallbacks();
        RuntimeException timeout = new RuntimeException("timeout");
        RuntimeException uncertain = new RuntimeException("uncertain");

        dispatcher.callback.onTimeout(item, timeout);
        dispatcher.callback.onDispatchUncertain(item, 701, uncertain);

        assertEquals(List.of(item), callback.timedOut);
        assertEquals(List.of(timeout), callback.timeoutErrors);
        assertEquals(List.of(item), callback.uncertain);
        assertEquals(List.of(uncertain), callback.uncertainErrors);
    }

    private static BatchItem item(DeliveryMode mode, PrefillEndpoint endpoint) {
        BalanceContext context = new BalanceContext();
        FlexlbConfig config = new FlexlbConfig();
        if (mode == DeliveryMode.ROUTE_DECISION) {
            SchedulingTestConfig.useNonBatchDispatcher(config);
        } else {
            SchedulingTestConfig.useBatchDispatcher(config);
        }
        context.setConfig(config);
        return new BatchItem(context, new CompletableFuture<>(), null,
                null, null, endpoint, null, System.currentTimeMillis());
    }

    private static final class RecordingCallback implements DecisionDelivery.Callback {
        private final List<BatchItem> delivered = new ArrayList<>();
        private final List<BatchItem> timedOut = new ArrayList<>();
        private final List<Throwable> timeoutErrors = new ArrayList<>();
        private final List<BatchItem> uncertain = new ArrayList<>();
        private final List<Throwable> uncertainErrors = new ArrayList<>();

        @Override
        public void onDelivered(BatchItem item) {
            delivered.add(item);
        }

        @Override
        public void onFailure(BatchItem item, Throwable error) {
        }

        @Override
        public void onTimeout(BatchItem item, Throwable error) {
            timedOut.add(item);
            timeoutErrors.add(error);
        }

        @Override
        public void onUncertain(BatchItem item, Throwable error) {
            uncertain.add(item);
            uncertainErrors.add(error);
        }
    }

    private static final class RecordingBatchDispatcher implements BatchDispatcher {
        private int reservationCount;
        private int submissionCount;
        private List<BatchItem> items;
        private PrefillEndpoint prefillEndpoint;
        private long batchId;
        private long predictedMs;
        private String reason;
        private DispatchCallback callback;

        @Override
        public SubmissionReservationResult tryReserveSubmission() {
            reservationCount++;
            return new SubmissionReserved(new SubmissionPermit() {
                private boolean resolved;

                @Override
                public void submit(List<BatchItem> submittedItems,
                                   PrefillEndpoint submittedEndpoint,
                                   long submittedBatchId,
                                   long submittedPredictedMs,
                                   String submittedReason,
                                   DispatchCallback submittedCallback) {
                    if (resolved) {
                        throw new IllegalStateException("submission permit already resolved");
                    }
                    resolved = true;
                    submissionCount++;
                    items = submittedItems;
                    prefillEndpoint = submittedEndpoint;
                    batchId = submittedBatchId;
                    predictedMs = submittedPredictedMs;
                    reason = submittedReason;
                    callback = submittedCallback;
                }

                @Override
                public void release() {
                    if (resolved) {
                        throw new IllegalStateException("submission permit already resolved");
                    }
                    resolved = true;
                }
            });
        }
    }
}
