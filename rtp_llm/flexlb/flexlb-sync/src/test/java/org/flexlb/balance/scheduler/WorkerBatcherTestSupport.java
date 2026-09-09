package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.delivery.DeliveryStrategy;

import java.util.List;

final class WorkerBatcherTestSupport {

    private WorkerBatcherTestSupport() {
    }

    static DeliveryStrategy.Transaction boundaryOnly(
            ScheduledRequest item,
            CapacityBoundary boundary) {
        return new DeliveryStrategy.Transaction() {
            @Override
            public List<ScheduledRequest> items() {
                return List.of();
            }

            @Override
            public ScheduledRequest blockedItem() {
                return item;
            }

            @Override
            public CapacityBoundary blockedResult() {
                return boundary;
            }

            @Override
            public void commitUnderLock() {
                throw new IllegalStateException(
                        "boundary-only preparation cannot commit");
            }

            @Override
            public void handoff(
                    String decisionReason, int remainingQueueDepth) {
                throw new IllegalStateException(
                        "boundary-only preparation cannot hand off");
            }

            @Override
            public void abort(Throwable cause) {
            }

            @Override
            public void close() {
            }
        };
    }
}
