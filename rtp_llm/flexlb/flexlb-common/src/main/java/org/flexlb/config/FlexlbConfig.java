package org.flexlb.config;

import com.fasterxml.jackson.annotation.JsonIgnore;
import lombok.Getter;
import lombok.Setter;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.ResourceMeasureIndicatorEnum;

import static org.flexlb.enums.ResourceMeasureIndicatorEnum.REMAINING_KV_CACHE;
import static org.flexlb.enums.ResourceMeasureIndicatorEnum.PREFILL_PENDING_REQUESTS;

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
    private SchedulerConfig scheduler = new QueueSchedulerConfig();
    private DispatcherConfig dispatcher = new BatchDispatcherConfig();
    private RoutingConfig router = new RoutingConfig();
    private WorkerRegistryConfig workerRegistry = new WorkerRegistryConfig();
    private ObservabilityConfig observability = new ObservabilityConfig();

    @JsonIgnore
    private final InternalRuntimeSettings internalRuntime = new InternalRuntimeSettings();

    @JsonIgnore
    public boolean isDirect() {
        return scheduler instanceof DirectSchedulerConfig;
    }

    @JsonIgnore
    public boolean isQueue() {
        return scheduler instanceof QueueSchedulerConfig;
    }

    @JsonIgnore
    public boolean isPriorityOrdering() {
        return isQueue()
                && ((QueueSchedulerConfig) scheduler).getOrdering() instanceof PriorityOrderingConfig;
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
        if (isDirect()) {
            return false;
        }
        return queueScheduler().getDecision() instanceof SingleDecisionConfig;
    }

    @JsonIgnore
    public boolean isFixedWindowDecision() {
        return !isDirect()
                && queueScheduler().getDecision() instanceof FixedWindowDecisionConfig;
    }

    @JsonIgnore
    public FixedWindowDecisionConfig fixedWindowDecision() {
        DecisionPolicyConfig policy = decisionPolicy();
        if (policy instanceof FixedWindowDecisionConfig fixedWindow) {
            return fixedWindow;
        }
        throw new IllegalStateException(
                "fixed-window decision configuration is not active");
    }

    @JsonIgnore
    public QueueSchedulerConfig queueScheduler() {
        if (scheduler instanceof QueueSchedulerConfig queue) {
            return queue;
        }
        throw new IllegalStateException("queue scheduler configuration is not active");
    }

    @JsonIgnore
    public PriorityOrderingConfig priorityOrdering() {
        QueueOrderingConfig ordering = queueScheduler().getOrdering();
        if (ordering instanceof PriorityOrderingConfig priority) {
            return priority;
        }
        throw new IllegalStateException("priority ordering configuration is not active");
    }

    @JsonIgnore
    public ResourceMeasureIndicatorEnum resourceMeasureFor(RoleType role) {
        return switch (role) {
            case PREFILL, PDFUSION, VIT -> PREFILL_PENDING_REQUESTS;
            case DECODE -> REMAINING_KV_CACHE;
            case FRONTEND -> null;
        };
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
        RoutingConfig.PrefillSelectorConfig selector =
                router.getRoles().getPrefill().getSelector();
        if (!(selector instanceof RoutingConfig.EstimatedTtftSelectorConfig estimated)
                || !(estimated.getCandidateChoice()
                instanceof RoutingConfig.LeastRecentlyUsedInPoolConfig lru)) {
            return 1;
        }
        RoutingConfig.CandidatePoolConfig pool = lru.getPool();
        if (pool instanceof RoutingConfig.FixedCandidatePoolConfig fixed) {
            return Math.max(1, Math.min(fixed.getWorkers(), workerCount));
        }
        RoutingConfig.RatioCandidatePoolConfig ratio =
                (RoutingConfig.RatioCandidatePoolConfig) pool;
        return Math.max(1, Math.max(ratio.getMinimumWorkers(),
                (int) Math.floor(workerCount * ratio.getRatio())));
    }
}
