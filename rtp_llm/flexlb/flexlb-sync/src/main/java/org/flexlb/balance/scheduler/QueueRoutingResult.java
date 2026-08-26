package org.flexlb.balance.scheduler;

import org.flexlb.dao.loadbalance.Response;

import java.util.Objects;

/** Result of queue routing before the scheduler publishes ACTIVE ownership. */
public sealed interface QueueRoutingResult {

    record Admitted(QueueRouteAdmission admission)
            implements QueueRoutingResult {
        public Admitted {
            Objects.requireNonNull(admission, "admission");
        }
    }

    record Rejected(Response response) implements QueueRoutingResult {
        public Rejected {
            Objects.requireNonNull(response, "response");
            if (response.isSuccess()) {
                throw new IllegalArgumentException(
                        "Rejected queue route requires a failure response");
            }
        }
    }
}
