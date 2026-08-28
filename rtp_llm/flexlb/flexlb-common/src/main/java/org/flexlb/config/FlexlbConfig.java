package org.flexlb.config;

import com.fasterxml.jackson.annotation.JsonIgnore;
import lombok.Getter;
import lombok.Setter;

/**
 * Public FLEXLB_CONFIG contract, organized by stable responsibility owner.
 * Mutually exclusive behavior is represented by tagged unions so inactive
 * variant fields fail during deserialization.
 */
@Getter
@Setter
public final class FlexlbConfig {

    public static final int CURRENT_SCHEMA_VERSION = 2;

    private int schemaVersion = CURRENT_SCHEMA_VERSION;
    private SchedulerConfig scheduler = new SchedulerConfig();
    private DispatcherConfig dispatcher = new DispatcherConfig();
    private RoutingConfig router = new RoutingConfig();
    private WorkerRegistryConfig workerRegistry = new WorkerRegistryConfig();
    private ObservabilityConfig observability = new ObservabilityConfig();

    @JsonIgnore
    private final InternalRuntimeSettings internalRuntime = new InternalRuntimeSettings();

    @JsonIgnore
    public boolean isDirect() {
        return scheduler.getType() == SchedulerConfig.Type.DIRECT;
    }

    @JsonIgnore
    public boolean isQueue() {
        return scheduler.getType() == SchedulerConfig.Type.QUEUE;
    }

    @JsonIgnore
    public boolean isPriorityOrdering() {
        return isQueue()
                && scheduler.getOrdering().getType()
                == QueueOrderingConfig.Type.PRIORITY;
    }

    /**
     * Whether transient Decode capacity belongs to the delivery boundary.
     *
     * <p>A queue without a preemption policy cannot make a useful placement-time
     * decision when Decode is temporarily full. It should retain the request on
     * its selected Prefill queue and let the exact pre-delivery permit wait for
     * Decode capacity. A configured preemption policy is the only queue policy
     * which needs a placement-time miss in order to plan victim replacement.
     */
    @JsonIgnore
    public boolean defersDecodeCapacityUntilDispatch() {
        return isQueue()
                && queueScheduler().getOrdering().preemptionPolicy().isEmpty();
    }

    /** Resolve the QUEUE decision policy from its single configuration owner. */
    @JsonIgnore
    public DecisionPolicyConfig decisionPolicy() {
        return queueScheduler().getDecision();
    }

    @JsonIgnore
    public boolean isSingleDecision() {
        return isQueue() && queueScheduler().getDecision().getType()
                == DecisionPolicyConfig.Type.SINGLE;
    }

    @JsonIgnore
    public boolean isFixedWindowDecision() {
        return isQueue() && queueScheduler().getDecision().getType()
                == DecisionPolicyConfig.Type.FIXED_WINDOW;
    }

    @JsonIgnore
    public DecisionPolicyConfig fixedWindowDecision() {
        DecisionPolicyConfig policy = decisionPolicy();
        if (policy.getType() == DecisionPolicyConfig.Type.FIXED_WINDOW) {
            return policy;
        }
        throw new IllegalStateException(
                "fixed-window decision configuration is not active");
    }

    @JsonIgnore
    public SchedulerConfig queueScheduler() {
        if (isQueue()) {
            return scheduler;
        }
        throw new IllegalStateException("queue scheduler configuration is not active");
    }

    @JsonIgnore
    public QueueOrderingConfig priorityOrdering() {
        QueueOrderingConfig ordering = queueScheduler().getOrdering();
        if (ordering.getType() == QueueOrderingConfig.Type.PRIORITY) {
            return ordering;
        }
        throw new IllegalStateException("priority ordering configuration is not active");
    }

    @JsonIgnore
    public long effectiveMaxOutputTokensForReservation(long declared) {
        Long maximum = router.getRoles().getDecode().getKvReservation()
                .getMaxOutputTokensForEstimate();
        return maximum == null ? declared : Math.min(declared, maximum);
    }

    /**
     * Resolve one decode reservation estimate without allowing malformed or
     * extreme token counts to wrap negative. A positive endpoint capacity is
     * a final physical cap; zero means the endpoint has not reported a usable
     * capacity yet.
     */
    @JsonIgnore
    public long decodeKvReservationTokens(long inputTokens,
                                          long declaredOutputTokens,
                                          long totalKvCapacity) {
        long normalizedInput = Math.max(0L, inputTokens);
        long normalizedOutput = Math.max(0L, declaredOutputTokens);
        long effectiveOutput = Math.max(0L,
                effectiveMaxOutputTokensForReservation(normalizedOutput));
        long estimated = normalizedInput > Long.MAX_VALUE - effectiveOutput
                ? Long.MAX_VALUE : normalizedInput + effectiveOutput;
        return totalKvCapacity > 0L
                ? Math.min(estimated, totalKvCapacity) : estimated;
    }

    @JsonIgnore
    public int shortestTtftCandidateCount(int workerCount) {
        RoutingConfig.CandidateChoiceConfig choice =
                router.getRoles().getPrefill().getCandidateChoice();
        if (choice.getType()
                != RoutingConfig.CandidateChoiceType.LEAST_RECENTLY_USED_IN_POOL) {
            return 1;
        }
        RoutingConfig.CandidatePoolConfig pool =
                choice.getPool();
        if (pool.getType() == RoutingConfig.CandidatePoolType.FIXED) {
            return Math.max(1, Math.min(pool.getWorkers(), workerCount));
        }
        return Math.max(1, Math.max(pool.getMinimumWorkers(),
                (int) Math.floor(workerCount * pool.getRatio())));
    }
}
