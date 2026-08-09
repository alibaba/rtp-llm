package org.flexlb.dao;

import lombok.Data;
import lombok.ToString;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.enums.ScheduleModeEnum;
import org.flexlb.util.Prioritized;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * @author zjw
 * description:
 * date: 2025/3/11
 */
@Data
@ToString
public class BalanceContext implements Prioritized {

    //======================== Basic =======================//

    private FlexlbConfig config;

    private Request request;

    private Response response;

    @ToString.Exclude
    private byte[] generateInputPbBytes;

    private volatile ScheduleModeEnum scheduleMode = ScheduleModeEnum.BATCH;

    //======================== Queue ========================//

    private CompletableFuture<Response> future;

    private final AtomicInteger retryCount = new AtomicInteger(0);

    //======================== Meters =======================//

    private long startTime = System.currentTimeMillis();

    /** Monotonic timestamp captured when server-side request processing starts. */
    private long serviceStartNanos = System.nanoTime();

    /**
     * Timestamp (ms) when the request entered the gRPC server pipeline,
     * recorded by {@code GrpcServerTimingInterceptor}. Used to split the
     * total arrival delay into network delay and gRPC server processing time.
     * Remains 0 if the interceptor did not set it (e.g. non-gRPC code path).
     */
    private long grpcEntryTime;

    /** Monotonic counterpart of {@link #grpcEntryTime} for duration measurements. */
    private long grpcEntryNanos;

    /** Monotonic timestamp immediately before the request enters its worker batcher. */
    private long routeSubmittedNanos;

    /** Monotonic timestamp immediately before the batch is dispatched to the engine. */
    private long batchDispatchedNanos;

    private long enqueueTime;

    private long dequeueTime;

    /**
     * Timestamp (ms) when the engine acknowledges the batch in BATCH mode.
     * Set by FlexlbBatchScheduler.onSuccess() when the ACK is received.
     * Used to compute ack_to_response_time_ms in FlexlbServiceImpl.completeSchedule().
     * Remains 0 for non-BATCH paths or when ACK was not received.
     */
    private long ackAtMs;

    /** Monotonic counterpart of {@link #ackAtMs}. */
    private long ackAtNanos;

    private long sequenceId;

    private boolean success = true;

    private String errorMessage;

    //===================== Auto-TPM =================//

    /**
     * Immutable per-request schedule budget (normalized priority, SLO, coarse
     * deadline). {@code null} when Auto-TPM is disabled — every accessor that
     * delegates to budget performs a null check and returns a legacy default.
     */
    private ScheduleBudget budget;

    /**
     * Number of Auto-TPM plan attempts consumed for this request (1-based).
     * 0 when the Auto-TPM path did not schedule it (§19.1 schedule_attempt).
     */
    private int scheduleAttempt;

    /**
     * Auto-TPM plan type that finally placed the request:
     * normal / prefill_evict / decode_evict. Empty when not applicable
     * (§19.1 plan_type).
     */
    private String planType = "";

    /** Cost of the committed eviction plan; 0 for a normal placement (§19.1 plan_cost). */
    private long planCost;

    /** Victims preempted to place this request; 0 for a normal placement (§19.1 victim_count). */
    private int victimCount;

    /**
     * Completed cross-endpoint rescue transfers for this request (Phase 6
     * deadline rescue). 0 = never migrated; a rescue re-entry keeps the
     * original arrival/deadline and bumps this counter.
     */
    private int transferCount;

    /**
     * Prefill endpoint ("ip:httpPort") that the committed plan placed this
     * request on. Empty until a plan commits; used by rescue logging to
     * report the migration target.
     */
    private String scheduledPrefillEndpoint = "";

    /**
     * Prefill endpoint ("ip:httpPort") whose queue just rejected this
     * request's offer (review P1-4): the priority scheduler sets it before a
     * fallback re-route and the prefill strategy skips that worker for
     * exactly one route (cleared on route entry). Null when unset.
     */
    private String excludedPrefillIpPort;

    //===================== Method ===================//

    public long getRequestId() {
        return request.getRequestId();
    }

    /**
     * Normalized Auto-TPM priority of the request (1-100, higher = more
     * important). Delegates to {@link #budget} when set; falls back to
     * {@code request.getPriority()} on the legacy path (Auto-TPM off).
     *
     * <p>Satisfies {@link Prioritized#priority()} for the top-level request
     * queue ({@code QueueManager}'s {@code PriorityBlockingQueue}).
     */
    @Override
    public int priority() {
        return budget != null ? budget.priority() : request.getPriority();
    }

    /**
     * Convenience accessor for the normalized priority (same as
     * {@link #priority()}). Retained for call-site compatibility with
     * Lombok-style getter naming.
     */
    public int getPriority() {
        return priority();
    }

    /**
     * Monotonic enqueue sequence used as the same-priority FIFO tie-break in
     * {@link org.flexlb.util.PriorityOrdering#STRICT}. Set once at enqueue
     * time by {@code QueueManager} via {@code setSequenceId} and never
     * mutated — a re-offer (retry / rescue) keeps the original sequence so
     * the item sorts back to its priority-correct position.
     */
    @Override
    public long enqueueSeq() {
        return sequenceId;
    }

    /**
     * Coarse admission deadline (epoch ms). Delegates to {@link #budget};
     * returns 0 on the legacy path.
     */
    public long getDeadlineMs() {
        return budget != null ? budget.deadlineMs() : 0;
    }

    /**
     * Per-request SLO in ms. Delegates to {@link #budget}; returns 0 on the
     * legacy path.
     */
    public long getRequestSloMs() {
        return budget != null ? budget.requestSloMs() : 0;
    }

    /** Direct accessor for the schedule budget (may be {@code null}). */
    public ScheduleBudget budget() {
        return budget;
    }

    /**
     * Increment retry count
     * @return the new retry count after incrementing
     */
    public int incrementRetryCount() {
        return retryCount.incrementAndGet();
    }

    /**
     * Get current retry count
     */
    public int getRetryCount() {
        return retryCount.get();
    }
}
