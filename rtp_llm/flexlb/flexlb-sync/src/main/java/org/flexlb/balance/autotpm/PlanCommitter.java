package org.flexlb.balance.autotpm;

import org.flexlb.balance.scheduler.BatchItem;
import org.flexlb.balance.scheduler.BatcherContext;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collections;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * Commits an eviction plan via the versioned CAS API of {@link BatcherContext}.
 *
 * <p>Performs an <b>atomic</b> remove-victims + offer-incoming under a single
 * lock (see {@link BatcherContext#tryRemoveAndOffer}). On version mismatch
 * returns a {@link CommitResult#versionMismatch()} so the caller can re-plan.
 *
 * <p>The committer does NOT complete victims' futures — that is the caller's
 * responsibility (it has access to the inflight map / lifecycle cleanup).
 * The removed {@link BatchItem}s are returned via {@link CommitResult#victims()}
 * for the caller to fail with {@code PRIORITY_PREEMPTED}.
 *
 * <p>Stateless and reusable; safe to call concurrently from different threads
 * on different {@link BatcherContext}s. On the same context the underlying
 * CAS is synchronized.
 */
public final class PlanCommitter {

    private static final Logger log = LoggerFactory.getLogger(PlanCommitter.class);

    /**
     * Attempt to commit {@code plan}: remove its victims and offer {@code incoming}.
     *
     * @param plan     eviction plan (carries victim IDs + snapshot version)
     * @param incoming the incoming request item to enqueue after eviction
     * @param ctx      the batcher context to mutate
     * @return commit result; never {@code null}
     */
    public CommitResult execute(PrefillEvictionPlan plan, BatchItem incoming, BatcherContext ctx) {
        if (plan == null || plan.isEmpty()) {
            return CommitResult.empty();
        }
        Set<Long> victimIds = plan.victimRequestIds().stream()
                .collect(Collectors.toSet());
        long expectedVersion = plan.snapshotVersion();

        List<BatchItem> removed = ctx.tryRemoveAndOffer(victimIds, incoming, expectedVersion);
        if (removed == null) {
            log.debug("Eviction CAS failed (version mismatch or still-full). "
                            + "expected_version={} victims={} incoming={}",
                    expectedVersion, victimIds.size(), incoming.requestId());
            return CommitResult.versionMismatch();
        }
        log.info("Eviction committed: {} victims removed, incoming requestId={} offered",
                removed.size(), incoming.requestId());
        return CommitResult.success(removed);
    }

    /**
     * Result of a commit attempt.
     */
    public static final class CommitResult {
        private final boolean success;
        private final List<BatchItem> victims;
        private final String failureReason;

        private CommitResult(boolean success, List<BatchItem> victims, String failureReason) {
            this.success = success;
            this.victims = victims == null ? Collections.emptyList() : victims;
            this.failureReason = failureReason;
        }

        /** CAS succeeded; victims were removed and incoming was offered. */
        public static CommitResult success(List<BatchItem> victims) {
            return new CommitResult(true, victims, null);
        }

        /** Version mismatch or queue still full after eviction; re-plan. */
        public static CommitResult versionMismatch() {
            return new CommitResult(false, Collections.emptyList(), "version_mismatch");
        }

        /** Empty plan — nothing to evict, incoming was not offered here. */
        public static CommitResult empty() {
            return new CommitResult(false, Collections.emptyList(), "empty_plan");
        }

        public boolean isSuccess() { return success; }
        public List<BatchItem> victims() { return victims; }
        public String failureReason() { return failureReason; }
    }
}
