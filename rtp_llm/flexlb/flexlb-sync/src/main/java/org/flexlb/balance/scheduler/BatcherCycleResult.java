package org.flexlb.balance.scheduler;

import java.util.List;
import java.util.Objects;

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
    record Admitted(List<BatchItem> items, DecisionGroupMetadata metadata)
            implements BatcherCycleResult {
        public Admitted {
            items = List.copyOf(items);
            if (items.isEmpty()) {
                throw new IllegalArgumentException("admitted prefix cannot be empty");
            }
            Objects.requireNonNull(metadata, "metadata");
        }
    }

    /** The ordered head remains active because this exact resource is full. */
    record CapacityBlocked(
            BatchItem item,
            DeliveryCapacityAdmission.CapacityUnavailable unavailable)
            implements BatcherCycleResult {
        public CapacityBlocked {
            Objects.requireNonNull(item, "blocked item");
            Objects.requireNonNull(unavailable, "capacity unavailability");
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
            Objects.requireNonNull(head, "waiting head");
            Objects.requireNonNull(reason, "wait reason");
        }
    }
}
