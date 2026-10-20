package org.flexlb.balance.scheduler;

/**
 * The Prefill or Decode generation selected for this route retired before the route was delivered.
 */
final class EndpointRetiredException extends IllegalStateException {

    EndpointRetiredException(String role, ScheduledRequest item) {
        super(role + " endpoint generation retired: request_id=" + item.requestId());
    }
}
