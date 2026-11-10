package org.flexlb.balance.planner;

import java.util.List;
import java.util.OptionalDouble;
import java.util.concurrent.atomic.AtomicInteger;

/** Independent implementation used by both execution and prediction contract tests. */
public final class FirstRequestPlanner implements GroupPlanner {
    public final AtomicInteger selections = new AtomicInteger();

    public <T extends Input> Selection<T> select(
            Iterable<T> ordered, Constraints limits, PrefixPrediction<T> predictor) {
        selections.incrementAndGet();
        var iterator = ordered.iterator();
        if (!iterator.hasNext())
            return new Selection<>(List.of(), 0, 0, Long.MAX_VALUE, false, OptionalDouble.empty());
        T first = iterator.next();
        var items = List.of(first);
        var prediction =
                predictor == null
                        ? OptionalDouble.empty()
                        : OptionalDouble.of(predictor.append(first, items));
        return new Selection<>(
                items, first.seqLen(), first.seqLen(), first.enqueuedAtMs(), false, prediction);
    }

    public String dispatchReason(Selection<?> selection, Constraints limits, long nowMs) {
        return selection.items().isEmpty() ? null : "first_request";
    }

    public long nextCheckAtMs(Selection<?> selection, Constraints limits) {
        return Long.MAX_VALUE;
    }

    public void validateConstraints(Constraints limits) {}
}
