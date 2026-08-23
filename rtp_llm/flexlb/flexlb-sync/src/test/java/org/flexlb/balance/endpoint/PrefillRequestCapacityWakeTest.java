package org.flexlb.balance.endpoint;

import org.flexlb.balance.scheduler.DecisionGroupHandler;
import org.flexlb.balance.scheduler.AdmittedDecisionGroup;
import org.flexlb.balance.scheduler.BatchItem;
import org.flexlb.balance.scheduler.DecisionGroupMetadata;
import org.flexlb.balance.scheduler.DeliveryCapacityAdmission;
import org.flexlb.balance.scheduler.TestCapacityAdmission;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.NonBatchDispatcherConfig;
import org.flexlb.config.PriorityOrderingConfig;
import org.flexlb.config.QueueSchedulerConfig;
import org.flexlb.config.SingleDecisionConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;

class PrefillRequestCapacityWakeTest {

    @Test
    void releasingRequestSlotWakesCapacityBlockedActiveRequest() throws Exception {
        CountDownLatch delivered = new CountDownLatch(1);
        AtomicInteger deliveryCount = new AtomicInteger();
        AtomicBoolean capacityReservedBeforeCallback = new AtomicBoolean();
        AtomicReference<PrefillEndpoint> endpointRef = new AtomicReference<>();
        DecisionGroupHandler handler = new DecisionGroupHandler() {
            @Override public void onExpired(BatchItem head) { }
            @Override public void onDecisionGroupAdmitted(
                    AdmittedDecisionGroup group, DecisionGroupMetadata meta) {
                deliveryCount.incrementAndGet();
                capacityReservedBeforeCallback.set(
                        endpointRef.get().availableRequestSlots(1) == 0);
                TestCapacityAdmission.complete(group);
                delivered.countDown();
            }
            @Override public void onOfferFailure(BatchItem item, Throwable error) { }
            @Override public void onDeliveryFailure(BatchItem item, Throwable error) { }
        };
        DeliveryCapacityAdmission capacityAdmission = item -> {
            PrefillEndpoint.RequestCapacityReservationAcquisition acquisition =
                    endpointRef.get().acquireRequestCapacityReservation(
                            item.requestId(), 10, 1);
            return switch (acquisition.status()) {
                case CAPACITY_FULL -> new DeliveryCapacityAdmission.CapacityUnavailable(
                        DeliveryCapacityAdmission.CapacityResource.PREFILL_REQUEST,
                        () -> endpointRef.get().availableRequestSlots(1) > 0);
                case REQUEST_ALREADY_TRACKED ->
                        new DeliveryCapacityAdmission.AdmissionFailed(
                                new IllegalStateException(
                                        "request capacity already tracked: "
                                                + item.requestId()));
                case ENDPOINT_RETIRED ->
                        new DeliveryCapacityAdmission.AdmissionFailed(
                                new EndpointGenerationRetiredException(
                                        "endpoint generation retired"));
                case ACQUIRED -> new DeliveryCapacityAdmission.CapacityReserved(
                        new DeliveryCapacityAdmission.ItemCapacityReservation() {
                            @Override public BatchItem item() {
                                return item;
                            }

                            @Override public boolean transferToEndpointLifecycle() {
                                if (!acquisition.reservation().prepareForDelivery()) {
                                    return false;
                                }
                                acquisition.reservation().completePreparedDeliveryTransfer();
                                return true;
                            }

                            @Override public void completeDeliveryHandoff() {
                                acquisition.reservation().completeDeliveryHandoff();
                            }

                            @Override public void release() {
                                acquisition.reservation().release();
                            }
                        });
            };
        };
        PrefillEndpoint endpoint = new PrefillEndpoint(
                workerStatus(), config(), handler, capacityAdmission,
                mock(BatchSchedulerReporter.class));
        endpointRef.set(endpoint);
        try {
            assertTrue(TestCapacityAdmission.commitRouteRequest(endpoint, 1, 10, 1));
            assertEquals(0, endpoint.availableRequestSlots(1));

            long beforeOfferVersion = endpoint.getBatcher().queueVersion();
            assertTrue(endpoint.getBatcher().tryOffer(batchItem(2)));
            long enqueuedVersion = beforeOfferVersion + 1;
            awaitTrue(() -> endpoint.getBatcher().queueSize() == 1
                    && endpoint.getBatcher().queueVersion() == enqueuedVersion);
            assertEquals(enqueuedVersion, endpoint.getBatcher().queueVersion(),
                    "capacity rejection must leave the request ACTIVE");
            assertEquals(0, deliveryCount.get(),
                    "capacity wait must not enter the delivery callback");

            assertTrue(endpoint.releaseRequest(1));
            assertTrue(delivered.await(2, TimeUnit.SECONDS),
                    "releaseRequest must wake the active capacity waiter after leaving its stripe");
            awaitTrue(() -> endpoint.getBatcher().queueSize() == 0);
            assertEquals(1, deliveryCount.get());
            assertTrue(capacityReservedBeforeCallback.get(),
                    "hard request capacity must be reserved before callback entry");
        } finally {
            endpoint.close();
        }
    }

    private static FlexlbConfig config() {
        FlexlbConfig config = new FlexlbConfig();
        QueueSchedulerConfig scheduler = new QueueSchedulerConfig();
        scheduler.setOrdering(new PriorityOrderingConfig());
        scheduler.setDecision(new SingleDecisionConfig());
        scheduler.getCapacity().setMaxOutstandingRequestsGlobal(16);
        NonBatchDispatcherConfig dispatcher = new NonBatchDispatcherConfig();
        dispatcher.setMaxInflightRequestsPerPrefillWorker(1);
        config.setScheduler(scheduler);
        config.setDispatcher(dispatcher);
        return config;
    }

    private static WorkerStatus workerStatus() {
        WorkerStatus status = new WorkerStatus();
        status.setIp("127.0.0.71");
        status.setPort(8071);
        status.setGrpcPort(9071);
        status.setRole(RoleType.PREFILL);
        return status;
    }

    private static BatchItem batchItem(long requestId) {
        long now = System.currentTimeMillis();
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(1);
        request.setPriority(50);
        BalanceContext context = new BalanceContext();
        context.setRequest(request);
        context.setConfig(config());
        context.setSchedulingMetadata(SchedulingMetadata.explicit(50, now + 60_000));
        return new BatchItem(context, new CompletableFuture<>(), null,
                null, null, null, null, now);
    }

    private static void awaitTrue(java.util.function.BooleanSupplier condition)
            throws InterruptedException {
        long deadlineNanos = System.nanoTime() + TimeUnit.SECONDS.toNanos(2);
        while (!condition.getAsBoolean() && System.nanoTime() < deadlineNanos) {
            TimeUnit.MILLISECONDS.sleep(5);
        }
        assertTrue(condition.getAsBoolean(), "condition did not become true within 2 seconds");
    }
}
