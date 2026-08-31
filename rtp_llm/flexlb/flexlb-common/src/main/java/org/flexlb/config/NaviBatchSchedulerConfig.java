package org.flexlb.config;

import lombok.Getter;
import lombok.Setter;

/**
 * NAVI_BATCH scheduling mode: a cost-model-driven batch scheduler ported from
 * navi_sched's {@code CostScheduler}. Requests are collected within a bounded
 * window and jointly assigned to prefill workers by a projected-gradient-descent
 * optimizer that minimizes a latency/queue objective on the simplex.
 *
 * <p>The numeric fields mirror navi_sched's {@code CostSchedulerConfig}
 * ({@code lambda}/{@code alpha}/{@code alpha_decay}/{@code min_alpha}/
 * {@code max_loop_count}/{@code optimization_time_budget_us}) plus the batching
 * window ({@code windowMs}/{@code maxCount}) that shapes each optimization call.
 */
@Getter
@Setter
public final class NaviBatchSchedulerConfig implements SchedulerConfig {

    /** Objective mix between throughput term (token-weighted) and fairness term. */
    private double naviBatchLambda = 0.5;

    /** Initial projected-gradient step size. */
    private double naviBatchAlpha = 512.0;

    /** Per-iteration multiplicative decay applied to the step size. */
    private double naviBatchAlphaDecay = 1.0;

    /** Floor for the decayed step size. */
    private double naviBatchMinAlpha = 0.0;

    /** Maximum optimizer iterations per scheduling call. */
    private int naviBatchMaxLoopCount = 10;

    /** Soft wall-clock budget for one optimization call, in microseconds. */
    private long naviBatchTimeBudgetUs = 2000;

    /** Collection window before an accumulated batch is scheduled. */
    private long naviBatchWindowMs = 30;

    /** Maximum requests gathered into a single optimization call. */
    private int naviBatchMaxCount = 30;

    /**
     * L2 capacity closed loop: shrink the PGD feasible domain to endpoints
     * with free capacity and add engine slot-free signals as an extra flush
     * trigger. Off by default: it is a behavior change that depends on the
     * engine reporting {@code available_concurrency} (the Java mock engine
     * reports {@code max_prefill_concurrency - running prefill batches};
     * production engines that leave the protobuf field unset read as 0 and
     * would look permanently full). With the flag off the scheduler behaves
     * exactly as before.
     */
    private boolean naviCapacityGatingEnabled = false;

    /**
     * Safety valve for the feasible-domain skip: when every eligible endpoint
     * is observed at capacity, a flushed window requeues into the buffer and
     * waits for the next trigger. Once the oldest requeued request has been
     * stalled for this long, the next flush forces the full endpoint domain
     * (pre-LS behavior) so requests can never be starved by a stale or
     * misleading capacity observation.
     */
    private long naviCapacityStallLimitMs = 300;
}
