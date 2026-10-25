package org.flexlb.balance.endpoint.prefill;

import org.flexlb.balance.scheduler.ScheduledRequest;

import java.util.Objects;

public record WorkerStatusFact(
        ScheduledRequest item,
        Kind kind,
        long errorCode) {
    public WorkerStatusFact {
        Objects.requireNonNull(kind, "kind");
        if (kind == Kind.ACTIVE && errorCode != 0L) {
            throw new IllegalArgumentException(
                    "an active Prefill fact cannot carry an error code");
        }
    }

    public static WorkerStatusFact active(ScheduledRequest item) {
        return new WorkerStatusFact(item, Kind.ACTIVE, 0L);
    }

    public static WorkerStatusFact terminal(
            ScheduledRequest item, Kind kind, long errorCode) {
        if (kind == Kind.ACTIVE) {
            throw new IllegalArgumentException(
                    "terminal Prefill fact requires a terminal kind");
        }
        return new WorkerStatusFact(item, kind, errorCode);
    }

    public enum Kind {
        ACTIVE,
        COMPLETED,
        FAILED,
        PRIORITY_CANCELED
    }
}
