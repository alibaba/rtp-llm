package org.flexlb.cache.domain;

import java.util.Objects;
import java.util.Set;

/** Immutable delta between two full engine cache snapshots. */
public record DiffResult(
        String engineIp,
        Set<Long> addedBlocks,
        Set<Long> removedBlocks) {

    public DiffResult {
        Objects.requireNonNull(engineIp, "engineIp");
        addedBlocks = Set.copyOf(addedBlocks);
        removedBlocks = Set.copyOf(removedBlocks);
    }
}
