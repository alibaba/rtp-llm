package org.flexlb.balance.dp;

import java.util.List;

/**
 * Per-batch context handed to {@link GroupSelector}. Lets selectors that care
 * about the batch contents (cache affinity, length-aware) reason about the
 * requests being placed.
 *
 * <p>V1 selectors (RR) ignore everything in here. The structure exists so future
 * selectors can be added without an API break.
 */
public record BatchHint(List<QueuedRequest> requests, String model, int dpSize) {

    public int size() {
        return requests == null ? 0 : requests.size();
    }
}
