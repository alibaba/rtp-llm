package org.flexlb.balance.projection;

import java.util.List;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Remembers ready groups ahead of a probe for one exact queue and work snapshot.
 */
public final class QueuePrefixCache {
    private final AtomicReference<Entry> entry = new AtomicReference<>();

    Entry read(Object model, RouteProjection.DeliveryProjection delivery, long nowMs) {
        Entry current = entry.get();
        return current != null && current.matches(model, delivery, nowMs) ? current : null;
    }

    void publish(
            Object model,
            RouteProjection.DeliveryProjection delivery,
            long fromMs,
            long untilMs,
            List<Group> groups) {
        if (groups.isEmpty() || fromMs >= untilMs) {
            return;
        }
        Entry current = entry.get();
        if (current != null
                && current.matches(model, delivery, fromMs)
                && current.groups().size() >= groups.size()) {
            return;
        }
        // Contending planners can always use the result they computed locally.
        entry.compareAndSet(
                current, new Entry(model, delivery, fromMs, untilMs, List.copyOf(groups)));
    }

    record Group(int size, long durationMs) {}

    /**
     * Binds the groups to the model, delivery strategy, and valid projection times.
     */
    record Entry(
            Object model,
            RouteProjection.DeliveryProjection delivery,
            long fromMs,
            long untilMs,
            List<Group> groups) {
        boolean matches(
                Object expectedModel,
                RouteProjection.DeliveryProjection expectedDelivery,
                long nowMs) {
            return model == expectedModel
                    && delivery == expectedDelivery
                    && nowMs >= fromMs
                    && nowMs < untilMs;
        }
    }
}
