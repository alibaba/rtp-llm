package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.Objects;

/**
 * Delivers prepared route decisions through the scheduler response boundary.
 *
 * <p>This class is stateless and allocation-free on the hot path. Resource
 * claims and delivery ownership have already been committed by the scheduler
 * before {@link #deliver} is entered. Each request is isolated: a callback
 * failure is reported for that request and cannot suppress later decisions.
 */
final class RouteDecisionDelivery implements DecisionDelivery<List<BatchItem>> {

    private static final Logger LOGGER = LoggerFactory.getLogger(RouteDecisionDelivery.class);

    static final RouteDecisionDelivery INSTANCE = new RouteDecisionDelivery();

    private RouteDecisionDelivery() {
    }

    @Override
    public void deliver(List<BatchItem> items, Callback callback) {
        Objects.requireNonNull(items, "items");
        Objects.requireNonNull(callback, "callback");
        if (items.isEmpty()) {
            throw new IllegalArgumentException(
                    "route-decision delivery requires at least one request");
        }
        PrefillEndpoint prefillEndpoint = Objects.requireNonNull(
                items.get(0), "items must not contain null").prefillEp();
        Objects.requireNonNull(prefillEndpoint, "prefillEndpoint");
        for (BatchItem item : items) {
            Objects.requireNonNull(item, "items must not contain null");
            if (item.deliveryMode() != DeliveryMode.ROUTE_DECISION) {
                throw new IllegalArgumentException(
                        "route-decision delivery contains a batch-enqueue request");
            }
            if (item.prefillEp() != prefillEndpoint) {
                throw new IllegalArgumentException(
                        "route-decision delivery contains requests for different Prefill endpoints");
            }
            try {
                callback.onDelivered(item);
            } catch (Throwable deliveryFailure) {
                notifyFailure(callback, item, deliveryFailure);
            }
        }
    }

    private static void notifyFailure(Callback callback,
                                      BatchItem item,
                                      Throwable deliveryFailure) {
        try {
            callback.onFailure(item, deliveryFailure);
        } catch (Throwable callbackFailure) {
            LOGGER.error("Route-decision failure callback failed request_id={} cause={}",
                    item.requestId(),
                    deliveryFailure.getMessage(), callbackFailure);
        }
    }
}
