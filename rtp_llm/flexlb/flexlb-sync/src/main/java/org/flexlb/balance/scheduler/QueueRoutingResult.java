package org.flexlb.balance.scheduler;

import org.flexlb.dao.loadbalance.Response;

import java.util.Objects;

/** Result of queue routing before the scheduler publishes ACTIVE ownership. */
public record QueueRoutingResult(
        Status status,
        QueueRouteAdmission admission,
        Response response,
        PlacementKey blocker) {

    public QueueRoutingResult {
        Objects.requireNonNull(status, "status");
        int payloads = (admission == null ? 0 : 1)
                + (response == null ? 0 : 1)
                + (blocker == null ? 0 : 1);
        if (payloads != 1
                || (status == Status.ADMITTED) != (admission != null)
                || (status == Status.REJECTED) != (response != null)
                || (status == Status.BLOCKED) != (blocker != null)) {
            throw new IllegalArgumentException(
                    "queue route status requires its exact payload");
        }
        if (response != null && response.isSuccess()) {
            throw new IllegalArgumentException(
                    "rejected queue route requires a failure response");
        }
    }

    public static QueueRoutingResult admitted(QueueRouteAdmission admission) {
        return new QueueRoutingResult(Status.ADMITTED,
                Objects.requireNonNull(admission, "admission"), null, null);
    }

    public static QueueRoutingResult rejected(Response response) {
        return new QueueRoutingResult(Status.REJECTED, null,
                Objects.requireNonNull(response, "response"), null);
    }

    /** No endpoint is owned; retry after this exact capacity domain changes. */
    public static QueueRoutingResult blocked(PlacementKey blocker) {
        return new QueueRoutingResult(Status.BLOCKED, null, null,
                Objects.requireNonNull(blocker, "blocker"));
    }

    public enum Status {
        ADMITTED,
        REJECTED,
        BLOCKED
    }
}
