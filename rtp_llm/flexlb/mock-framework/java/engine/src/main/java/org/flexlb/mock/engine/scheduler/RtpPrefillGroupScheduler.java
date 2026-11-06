package org.flexlb.mock.engine.scheduler;

import java.util.*;
import org.flexlb.mock.engine.cache.KvCacheManager;
import org.flexlb.mock.engine.config.RtpConfig;
import static org.flexlb.mock.engine.scheduler.EngineScheduler.RequestFailure.Reason.*;

/** Explicit-group admission, isolated from ordinary FIFO scheduling. */
final class RtpPrefillGroupScheduler extends AbstractRtpScheduler {
    private final ArrayDeque<ArrayDeque<GenerateStream>> groups = new ArrayDeque<>();
    private final Map<GenerateStream, ArrayDeque<GenerateStream>> memberships = new IdentityHashMap<>();
    private boolean preferGroup;

    RtpPrefillGroupScheduler(Object lock, RtpConfig config, KvCacheManager cache) {
        this(lock, config, cache, new TaskHistory());
    }

    RtpPrefillGroupScheduler(Object lock, RtpConfig config, KvCacheManager cache, TaskHistory finished) {
        super(lock, config, cache, finished);
    }

    @Override public List<InputResult> enqueue(List<InputItem> batch) {
        if (closed) throw new RequestFailure(CLOSED, "Engine is closing");
        if (!prefill()) throw new RequestFailure(INVALID, "Explicit batches belong to Prefill");
        batch = List.copyOf(batch);
        var results = new ArrayList<InputResult>();
        var group = new ArrayDeque<GenerateStream>();
        for (var item : batch) {
            try {
                var stream = (GenerateStream) enqueue(item.input(), item.listener());
                waiting.remove(stream); group.addLast(stream);
                results.add(new InputResult(stream, null));
            } catch (RequestFailure failure) { results.add(new InputResult(null, failure)); }
        }
        if (group.size() > config.maxGenerateBatchSize()
                || config.maxInitedKvStreams() > 0 && group.size() > config.maxInitedKvStreams()) {
            waiting.addAll(group); // RTP falls oversized groups back to ordinary FIFO admission.
        } else if (!group.isEmpty()) {
            group.forEach(stream -> memberships.put(stream, group)); groups.addLast(group);
        }
        return List.copyOf(results);
    }

    @Override protected void selectPrefill(List<GenerateStream> selected) {
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

    private void selectGroup(List<GenerateStream> selected) {
        if (groups.isEmpty()) return;
        var group = groups.getFirst();
        admitPrefill(group, selected, 0);
        if (group.isEmpty()) groups.remove(group);
    }

    @Override protected void removeWaiting(GenerateStream stream) {
        super.removeWaiting(stream);
        var group = memberships.remove(stream);
        if (group != null) {
            group.remove(stream);
            if (group.isEmpty()) groups.remove(group);
        }
    }

    @Override protected Set<GenerateStream> waitingStreams() {
        var entries = super.waitingStreams();
        groups.forEach(entries::addAll);
        return entries;
    }
}
