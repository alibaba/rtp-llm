package org.flexlb.balance.endpoint;

import org.flexlb.balance.scheduler.DecisionGroupHandler;
import org.flexlb.balance.scheduler.BatchItem;
import org.flexlb.balance.scheduler.DecisionGroupMetadata;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.NonBatchDispatcherConfig;
import org.flexlb.config.PriorityOrderingConfig;
import org.flexlb.config.QueueSchedulerConfig;
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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;

class PrefillRequestCapacityWakeTest {

    @Test
    void releasingRequestSlotWakesCapacityBlockedReadyBacklog() throws Exception {
        CountDownLatch delivered = new CountDownLatch(1);
        PrefillEndpoint endpoint = new PrefillEndpoint(
                workerStatus(), config(), new DecisionGroupHandler() {
                    @Override public void onExpired(BatchItem head) { }
                    @Override public void onDecisionGroupReady(List<BatchItem> items, DecisionGroupMetadata meta) {
                        delivered.countDown();
                    }
                    @Override public void onOfferFailure(BatchItem item, Throwable error) { }
                    @Override public void onDeliveryFailure(BatchItem item, Throwable error) { }
                }, mock(BatchSchedulerReporter.class));
        try {
            assertTrue(endpoint.tryCommitRequest(1, 10, 1));
            assertEquals(0, endpoint.availableRequestSlots(1));

            long beforeOfferVersion = endpoint.getBatcher().queueVersion();
            assertTrue(endpoint.getBatcher().tryOffer(batchItem(2)));
            awaitTrue(() -> endpoint.getBatcher().queueVersion() > beforeOfferVersion + 1);
            assertEquals(1, delivered.getCount());

            assertTrue(endpoint.releaseRequest(1));
            assertTrue(delivered.await(2, TimeUnit.SECONDS),
                    "releaseRequest must signal the ready-only worker after leaving its stripe");
        } finally {
            endpoint.close();
        }
    }

    private static FlexlbConfig config() {
        FlexlbConfig config = new FlexlbConfig();
        QueueSchedulerConfig scheduler = new QueueSchedulerConfig();
        scheduler.setOrdering(new PriorityOrderingConfig());
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
