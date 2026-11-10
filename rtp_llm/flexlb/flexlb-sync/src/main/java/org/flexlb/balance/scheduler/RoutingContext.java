package org.flexlb.balance.scheduler;

import java.util.Objects;

/**
 * Identifies current routing work for a request and the old route being withdrawn, if any.
 * RequestContext compares these objects by reference so old callbacks cannot complete newer routing work.
 */
public record RoutingContext(RequestContext requestContext, RouteResult routeToWithdraw) {
    public RoutingContext {
        Objects.requireNonNull(requestContext, "requestContext");
    }
}
