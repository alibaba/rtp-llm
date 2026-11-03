package org.flexlb.balance.projection;

import java.util.List;
import java.util.concurrent.atomic.AtomicReferenceFieldUpdater;

/** Derived memo owned by one endpoint's exact queue/work snapshot, never a global cache. */
public final class QueuePrefixCache {
    record Group(int size, long durationMs) { }
    record Entry(Object model, RouteProjection.DeliveryProjection delivery,
                 long fromMs, long untilMs, List<Group> groups) { }
    private volatile Entry entry;
    private static final AtomicReferenceFieldUpdater<QueuePrefixCache, Entry> ENTRY =
            AtomicReferenceFieldUpdater.newUpdater(QueuePrefixCache.class, Entry.class, "entry");

    Entry read(Object model, RouteProjection.DeliveryProjection delivery, long nowMs) {
        Entry value = entry;
        return matches(value, model, delivery, nowMs) ? value : null;
    }

    private static boolean matches(Entry value, Object model,
                                   RouteProjection.DeliveryProjection delivery, long nowMs) {
        return value != null && model != null && value.model() == model
                && value.delivery() == delivery && nowMs >= value.fromMs() && nowMs < value.untilMs();
    }

    void publish(Object model, RouteProjection.DeliveryProjection delivery,
                 long nowMs, long untilMs, List<Group> groups) {
        if (model == null || groups.isEmpty() || nowMs >= untilMs) return;
        Entry current = entry;
        if (matches(current, model, delivery, nowMs) && current.groups().size() >= groups.size()) return;
        // A hit performs only volatile reads. Publication is best-effort: a
        // losing planner keeps its correct local result without waiting/retrying.
        ENTRY.compareAndSet(this, current,
                new Entry(model, delivery, nowMs, untilMs, List.copyOf(groups)));
    }
}
