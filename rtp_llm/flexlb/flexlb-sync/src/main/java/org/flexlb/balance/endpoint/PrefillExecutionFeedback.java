package org.flexlb.balance.endpoint;

import org.flexlb.balance.prediction.PrefillBatchFeatures;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

import static com.google.common.math.LongMath.saturatedAdd;

/** Immutable execution evidence for individually delivered requests sharing an engine. */
final class PrefillExecutionFeedback {
    static final int MAX_ITEMS = 4096;
    static final PrefillExecutionFeedback EMPTY = new PrefillExecutionFeedback(List.of(), false);
    static final PrefillExecutionFeedback INVALID = new PrefillExecutionFeedback(List.of(), true);

    private record Item(PrefillBatchFeatures.Item features, long predictionMs, long startMs, long endMs) { }
    private final List<Item> items;
    private final boolean invalid;

    private PrefillExecutionFeedback(List<Item> items, boolean invalid) {
        this.items = items;
        this.invalid = invalid;
    }

    PrefillExecutionFeedback append(PrefillState.TerminalObservation terminal, boolean solitary) {
        if (invalid || items.size() >= MAX_ITEMS || terminal.errorCode() != 0L || terminal.executionTimeMs() <= 0L) {
            return INVALID;
        }
        long end = terminal.endTimeMs();
        if (end <= 0L) {
            if (!solitary || !items.isEmpty()) { return INVALID; }
            end = terminal.executionTimeMs();
        }
        if (end < terminal.executionTimeMs()) { return INVALID; }
        var next = new ArrayList<>(items);
        next.add(new Item(new PrefillBatchFeatures.Item(terminal.inputTokens(), terminal.inputTokens() - terminal.computeTokens()),
                terminal.owner().route.prefillWorkMs(), end - terminal.executionTimeMs(), end));
        return new PrefillExecutionFeedback(List.copyOf(next), false);
    }

    /** Only a drained engine closes the sample: one worker batch may finish across several status reports. */
    PrefillState.BatchCompletion completion() {
        if (invalid || items.isEmpty()) { return null; }
        var intervals = new ArrayList<>(items);
        intervals.sort(Comparator.comparingLong(Item::startMs));
        long duration = 0L, start = intervals.getFirst().startMs(), end = intervals.getFirst().endMs();
        for (var item : intervals) {
            if (item.startMs() > end) {
                duration = saturatedAdd(duration, end - start);
                start = item.startMs();
            }
            end = Math.max(end, item.endMs());
        }
        duration = saturatedAdd(duration, end - start);
        long prediction = 0L;
        for (var item : items) { prediction = saturatedAdd(prediction, item.predictionMs()); }
        return new PrefillState.BatchCompletion(-1L,
                new PrefillBatchFeatures(items.stream().map(Item::features).toList()), prediction, duration, true, true);
    }
}
