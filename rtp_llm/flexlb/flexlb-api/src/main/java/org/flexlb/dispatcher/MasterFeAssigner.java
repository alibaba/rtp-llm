package org.flexlb.dispatcher;

import org.flexlb.consistency.LBStatusConsistencyService;
import org.flexlb.dao.loadbalance.BatchScheduleTarget;
import org.flexlb.util.Logger;
import org.flexlb.util.RateLimitedWarn;
import org.springframework.beans.factory.ObjectProvider;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.concurrent.TimeUnit;

/**
 * Stamps each batch-schedule target's {@code fe_url} from the single master-side {@link FePool}
 * cursor, so FE selection has exactly one source fleet-wide instead of each dispatcher instance
 * round-robining a private (colliding) cursor.
 *
 * <p>This is the one and only place the stamp is applied, and both local-resolution entry points
 * route through it: the HTTP {@code /batch_schedule} handler
 * ({@link org.flexlb.httpserver.HttpLoadBalanceServer}, which stamps a slave's forwarded request)
 * and the master's own in-process dispatcher ({@link BatchScheduleClient}). Centralizing it is
 * deliberate — the earlier design wired stamping into the HTTP path only, which left the master's
 * in-process resolution (and any consistency-off single node) unstamped, failing every chunk it
 * resolved locally with {@code CHUNK_NO_FE}.
 *
 * <p>Two guards keep this an optimization that never breaks correctness:
 * <ol>
 *   <li>Only when this node resolved the batch locally — it is the elected master, or consistency
 *       is off. A slave that merely forwarded to the master already holds the master's assignment;
 *       re-stamping it with the slave's own cursor would reintroduce the collision this removes.</li>
 *   <li>Only when the {@link FePool} bean exists (this node runs the dispatcher). Absent it, targets
 *       keep {@code fe_url == null} and the dispatcher fails those chunks visibly — FE has exactly
 *       one source, so a misconfigured master (no FE view) surfaces as chunk failures rather than a
 *       silent split onto per-instance cursors.</li>
 * </ol>
 * An empty FE snapshot (pool throws) is swallowed the same way — leaving {@code fe_url} null — so BE
 * assignment (already computed) still returns; the affected chunks then fail visibly in the
 * dispatcher ({@code CHUNK_NO_FE}) rather than aborting the whole schedule.
 *
 * <p><b>Wiring contract:</b> this bean is deliberately an unconditional {@link Component} — do
 * <em>not</em> add {@code @ConditionalOnProperty} to it. {@link org.flexlb.httpserver.HttpLoadBalanceServer}
 * (itself unconditional, present on every flexlb node) takes it as a required constructor
 * dependency; conditioning this bean would break that node's wiring wherever the dispatcher is not
 * enabled. The optional part — the dispatcher's {@link FePool} — is expressed through the
 * {@link ObjectProvider}, not by conditioning this bean: {@link #assign} no-ops when the pool is
 * absent (a master node that does not run the dispatcher).
 */
@Component
public class MasterFeAssigner {

    private final ObjectProvider<FePool> fePoolProvider;
    private final LBStatusConsistencyService consistency;
    /** An empty/failed FE snapshot fails FE assignment on every batch request; cap the WARN at 1/s. */
    private final RateLimitedWarn feAssignWarn = new RateLimitedWarn(1, TimeUnit.SECONDS);

    public MasterFeAssigner(ObjectProvider<FePool> fePoolProvider,
                            LBStatusConsistencyService consistency) {
        this.fePoolProvider = fePoolProvider;
        this.consistency = consistency;
    }

    /**
     * Stamps {@code fe_url} onto each target from the single master cursor when this node resolved
     * the batch locally and a {@link FePool} is wired. A no-op otherwise (a slave's forwarded
     * response, or a node not running the dispatcher), leaving whatever {@code fe_url} the targets
     * already carry — the master's, for a forwarded response — untouched.
     */
    public void assign(List<BatchScheduleTarget> targets) {
        if (targets == null || targets.isEmpty()) {
            return;
        }
        boolean resolvedLocally = !consistency.isNeedConsistency() || consistency.isMaster();
        if (!resolvedLocally) {
            return;
        }
        FePool pool = fePoolProvider.getIfAvailable();
        if (pool == null) {
            return;
        }
        try {
            // One snapshot for the whole batch: nextBatch returns exactly targets.size() urls (so the
            // zip below is 1:1 by construction) or throws before assigning any — an empty snapshot
            // leaves every target's fe_url null rather than stamping a prefix, keeping assignment
            // all-or-nothing per batch. This is a behavior convergence from the earlier per-pick
            // version (pool.next() in a loop), which could leave a partial prefix stamped and the
            // rest null when the snapshot went empty mid-batch; no caller relied on that partial
            // result (fe_url == null fails a chunk with CHUNK_NO_FE either way), so the stricter
            // all-or-nothing contract is safe.
            List<String> feUrls = pool.nextBatch(targets.size());
            for (int i = 0; i < targets.size(); i++) {
                targets.get(i).setFeUrl(feUrls.get(i));
            }
        } catch (IllegalStateException e) {
            // Expected operational failure, not a bug: FePool.nextBatch() throws IllegalStateException
            // when its snapshot is empty (an FE outage / discovery gap). It fires on every request
            // while the pool is empty, so throttle it. The affected chunks fail downstream with
            // CHUNK_NO_FE; the schedule (BE assignment already computed) is not aborted.
            feAssignWarn.warn("[BatchSchedule] FE assignment skipped (empty/failed FE pool); affected "
                    + "chunks fail with CHUNK_NO_FE, no local fallback: {}", e.toString());
        } catch (RuntimeException e) {
            // Unexpected: anything other than the empty-pool contract is a programming error, not an
            // FE outage. Log it loud (ERROR, unthrottled, with stack) so a real bug can't hide behind
            // the throttled "pool empty" WARN — but still swallow rather than abort: BE assignment is
            // already computed and the affected chunks fail visibly with CHUNK_NO_FE (no fallback).
            Logger.error("[BatchSchedule] unexpected error stamping FE assignment (bug?); affected "
                    + "chunks fail with CHUNK_NO_FE, no local fallback", e);
        }
    }
}
