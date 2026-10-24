package org.flexlb.balance.queue;

import org.flexlb.balance.delivery.BlockedRequest;
import org.flexlb.balance.delivery.DeliveryBlockReason;
import org.flexlb.balance.delivery.DeliveryTransaction;
import org.flexlb.balance.estimate.WorkView;
import org.flexlb.balance.scheduler.ScheduledRequest;

import java.util.List;

final class PrefillQueueTestSupport {

    private PrefillQueueTestSupport() {
    }

    static DeliveryTransaction boundaryOnly(ScheduledRequest item, DeliveryBlockReason boundary) {
        return new DeliveryTransaction() {
            @Override
            public List<ScheduledRequest> requests() {
                return List.of();
            }

            @Override
            public BlockedRequest blocked() {
                return new BlockedRequest(item, boundary);
            }

            @Override
            public WorkView commit() {
                throw new IllegalStateException(
                        "boundary-only preparation cannot commit");
            }

            @Override
            public void deliver(String decisionReason, int remainingQueueDepth, WorkView precedingWork) {
                throw new IllegalStateException(
                        "boundary-only preparation cannot hand off");
            }

            @Override
            public void rollbackCommitted(Throwable cause) {
            }

            @Override
            public void close() {
            }
        };
    }
}
