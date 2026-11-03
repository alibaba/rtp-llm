package org.flexlb.mock.engine.rtp;

import java.util.*;
import org.flexlb.mock.cache.KvCache;
import org.flexlb.mock.engine.RequestFailure;
import static org.flexlb.mock.engine.RequestFailure.Reason.*;

/**
 * Explicit-group admission, isolated from ordinary FIFO scheduling.
 */
final class RtpGroupScheduler extends RtpSchedulerBase {
    private final ArrayDeque<ArrayDeque<Entry>> groups = new ArrayDeque<>();
    private final Map<Entry, ArrayDeque<Entry>> memberships = new IdentityHashMap<>();
    private boolean preferGroup;

    RtpGroupScheduler(Object lock, RtpConfig config, KvCache cache) { super(lock, config, cache); }

    @Override public List<InputResult> input(List<InputItem> batch) {
        if (closed) throw new RequestFailure(CLOSED, "Engine is closing");
        if (!prefill()) throw new RequestFailure(INVALID, "Explicit batches belong to Prefill");
        batch = List.copyOf(batch);
        var results = new ArrayList<InputResult>();
        var group = new ArrayDeque<Entry>();
        for (var item : batch) {
            try {
                var entry = (Entry) input(item.input(), item.listener());
                waiting.remove(entry); group.addLast(entry);
                results.add(new InputResult(entry, null));
            } catch (RequestFailure failure) { results.add(new InputResult(null, failure)); }
        }
        if (group.size() > config.maxGenerateBatchSize()
                || config.maxInitedKvStreams() > 0 && group.size() > config.maxInitedKvStreams()) {
            waiting.addAll(group); // RTP falls oversized groups back to ordinary FIFO admission.
        } else if (!group.isEmpty()) {
            group.forEach(entry -> memberships.put(entry, group)); groups.addLast(group);
        }
        return List.copyOf(results);
    }

    @Override protected void selectPrefill(List<Entry> selected) {
        boolean groupFirst = !groups.isEmpty() && (preferGroup || waiting.isEmpty());
        if (groupFirst) {
            selectGroup(selected);
            if (selected.isEmpty()) admitPrefill(waiting, selected, config.maxUncachedBatchTokens());
        } else {
            admitPrefill(waiting, selected, config.maxUncachedBatchTokens());
            if (selected.isEmpty()) selectGroup(selected);
        }
        if (!selected.isEmpty()) preferGroup = !memberships.containsKey(selected.getFirst()) && !groups.isEmpty();
        selected.forEach(memberships::remove);
    }

    private void selectGroup(List<Entry> selected) {
        if (groups.isEmpty()) return;
        var group = groups.getFirst();
        admitPrefill(group, selected, 0);
        if (group.isEmpty()) groups.remove(group);
    }

    @Override protected void removeWaiting(Entry entry) {
        super.removeWaiting(entry);
        var group = memberships.remove(entry);
        if (group != null) {
            group.remove(entry);
            if (group.isEmpty()) groups.remove(group);
        }
    }

    @Override protected Set<Entry> waitingEntries() {
        var entries = super.waitingEntries();
        groups.forEach(entries::addAll);
        return entries;
    }
}
