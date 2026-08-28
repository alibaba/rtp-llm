package org.flexlb.balance.scheduler;

import org.flexlb.dao.loadbalance.Response;


/** Result of queue routing before the scheduler publishes ACTIVE ownership. */
public sealed interface QueueRoutingResult {

    record Admitted(QueueRouteAdmission admission)
            implements QueueRoutingResult {
        public Admitted {
            assert admission != null : "missing queue admission";
        }
    }

    record Rejected(Response response) implements QueueRoutingResult {
        public Rejected {
            assert response != null : "missing queue rejection";
            if (response.isSuccess()) {
                throw new IllegalArgumentException(
                        "Rejected queue route requires a failure response");
            }
        }
    }

    /** No endpoint is owned; retry after this exact capacity domain changes. */
    record Blocked(PlacementKey blocker) implements QueueRoutingResult {
        public Blocked {
            java.util.Objects.requireNonNull(blocker, "blocker");
        }
    }
}
