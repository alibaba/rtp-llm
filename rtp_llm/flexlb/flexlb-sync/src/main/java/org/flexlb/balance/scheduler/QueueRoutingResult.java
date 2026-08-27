package org.flexlb.balance.scheduler;

import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.route.RoleType;

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

    /**
     * No worker for this generic role/group is bindable right now.  The
     * scheduler retains the request and performs a fresh selection after the
     * corresponding lane is signalled or its shared backoff expires.
     *
     * <p>Cache affinity and queue ordering deliberately remain inside their
     * existing owners; neither is projected through this result.</p>
     */
    record Deferred(RoleType role, String group) implements QueueRoutingResult {
        public Deferred {
            if (role == null) {
                throw new IllegalArgumentException(
                        "Deferred queue route requires a blocked role");
            }
        }

    }
}
