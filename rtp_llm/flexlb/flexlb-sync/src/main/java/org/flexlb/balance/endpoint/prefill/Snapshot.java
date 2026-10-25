package org.flexlb.balance.endpoint.prefill;

import org.flexlb.balance.scheduler.ScheduledRequest;

import java.util.List;

/**
 * Queue and committed work captured at one ownership linearization point.
 */
public record Snapshot(long capturedAtMs, PrefillActiveIndex.Capture active, WorkCapture work) {

    public List<ScheduledRequest> activeItems() {
        return active.items();
    }
}
