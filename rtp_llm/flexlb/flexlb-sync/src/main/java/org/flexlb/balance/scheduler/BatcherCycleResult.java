package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.delivery.DeliveryMetadata;

import java.util.List;

/** Result of one worker decision cycle. */
sealed interface BatcherCycleResult permits BatcherCycleResult.Outcome,
        BatcherCycleResult.Admitted,
        BatcherCycleResult.CapacityBlocked,
        BatcherCycleResult.AwaitingSchedulingChange {

    enum Outcome implements BatcherCycleResult {
        NO_ACTION,
        QUEUE_CHANGED
    }

    /** The exact ordered prefix which reserved capacity and entered delivery. */
    record Admitted(List<BatchItem> items, DeliveryMetadata metadata)
            implements BatcherCycleResult {
        public Admitted {
            items = List.copyOf(items);
            if (items.isEmpty()) {
                throw new IllegalArgumentException("admitted prefix cannot be empty");
            }
            assert metadata != null : "missing delivery metadata";
        }
    }

    /** The ordered head remains active because this exact resource is full. */
    record CapacityBlocked(
            BatchItem item,
            CapacityBoundary.Unavailable unavailable)
            implements BatcherCycleResult {
        public CapacityBlocked {
            assert item != null : "missing blocked item";
            assert unavailable != null : "missing capacity unavailability";
        }
    }

    enum SchedulingWaitReason {
        COLLECTION_WINDOW,
        PREFILL_KV_CAPACITY
    }

    /**
     * ACTIVE work is waiting for a queue mutation, a worker-status/predictor
     * generation change, or an exact wall-clock deadline. The captured
     * generations close every signal-before-await race.
     */
    record AwaitingSchedulingChange(
            BatchItem head,
            long queueVersion,
            long schedulingInputVersion,
            long wakeAtMs,
            SchedulingWaitReason reason) implements BatcherCycleResult {
        public AwaitingSchedulingChange {
            assert head != null : "missing waiting head";
            assert reason != null : "missing wait reason";
        }
    }
}
