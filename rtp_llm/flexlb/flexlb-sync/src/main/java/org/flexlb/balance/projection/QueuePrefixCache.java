package org.flexlb.balance.projection;

import java.util.List;

/** Derived memo owned by one endpoint's exact queue/work snapshot, never a global cache. */
public final class QueuePrefixCache {
    record Group(int size, long durationMs) { }
    record Entry(Object model, RouteProjection.DeliveryProjection delivery,
                 long fromMs, long untilMs, List<Group> groups) { }
    private volatile Entry entry;

    Entry read(Object model, RouteProjection.DeliveryProjection delivery, long nowMs) {
        Entry value = entry;
        return value != null && model != null && value.model() == model
                && value.delivery() == delivery && nowMs >= value.fromMs() && nowMs < value.untilMs()
                ? value : null;
    }

    synchronized void publish(Object model, RouteProjection.DeliveryProjection delivery,
                              long nowMs, long untilMs, List<Group> groups) {
        if (model == null || groups.isEmpty() || nowMs >= untilMs) return;
        Entry current = read(model, delivery, nowMs);
        if (current == null || current.groups().size() < groups.size()) {
            entry = new Entry(model, delivery, nowMs, untilMs, List.copyOf(groups));
        }
    }
}
