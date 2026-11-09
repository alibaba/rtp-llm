package org.flexlb.balance.endpoint;

import org.flexlb.balance.endpoint.DecodeEndpoint.ReservationHandle;

public record WorkerStatusFact(Kind kind, ReservationHandle reservation, long errorCode) {

    public static WorkerStatusFact active(ReservationHandle reservation) {
        return new WorkerStatusFact(Kind.ACTIVE, reservation, 0L);
    }

    public static WorkerStatusFact accepted(ReservationHandle reservation) {
        return new WorkerStatusFact(Kind.ACCEPTED, reservation, 0L);
    }

    public static WorkerStatusFact terminal(ReservationHandle reservation, long errorCode) {
        return new WorkerStatusFact(Kind.TERMINAL, reservation, errorCode);
    }

    public enum Kind {
        ACTIVE,
        ACCEPTED,
        TERMINAL
    }
}
