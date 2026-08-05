package org.flexlb.balance.autotpm;

import org.flexlb.schedule.grpc.FlexlbScheduleProtocol.CancelReasonPB;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.function.BiConsumer;

/**
 * Executes running preemption: rate-limit check → cancel victim → wait for
 * release → report success or failure.
 *
 * <p>The cancel is delegated to the caller via a {@link BiConsumer} method
 * reference (typically {@code FlexlbBatchScheduler::cancelRequest}). After
 * the cancel is issued, the committer polls {@link DecodeAdmissionTracker}
 * until the victim's reservation is released or the confirm timeout expires.
 *
 * <p>Stateless except for the injected rate limiter — safe to call
 * concurrently from different threads on different victims.
 */
public final class RunningPreemptCommitter {

    private static final Logger log = LoggerFactory.getLogger(RunningPreemptCommitter.class);

    private static final long POLL_INTERVAL_MS = 10L;

    private final BiConsumer<Long, CancelReasonPB> cancelAction;
    private final DecodeAdmissionTracker decodeTracker;
    private final PreemptRateLimiter rateLimiter;

    /**
     * @param cancelAction  method reference to cancel a request
     *                      (e.g. {@code scheduler::cancelRequest})
     * @param decodeTracker the decode admission tracker (for release polling)
     * @param rateLimiter   per-node + global QPS limiter
     */
    public RunningPreemptCommitter(BiConsumer<Long, CancelReasonPB> cancelAction,
                                    DecodeAdmissionTracker decodeTracker,
                                    PreemptRateLimiter rateLimiter) {
        this.cancelAction = cancelAction;
        this.decodeTracker = decodeTracker;
        this.rateLimiter = rateLimiter;
    }

    /**
     * Execute running preemption for a single victim.
     *
     * <ol>
     *   <li>Rate-limit check (per-node + global)</li>
     *   <li>Cancel victim via {@code CANCEL_REASON_PRIORITY_PREEMPTED}</li>
     *   <li>Poll tracker until reservation released or timeout</li>
     * </ol>
     *
     * @param victim           the RUNNING reservation to preempt
     * @param endpointKey      ip:port of the decode endpoint
     * @param confirmTimeoutMs max wait for release (bounded, no infinite wait)
     * @return {@code true} if the victim was cancelled and its reservation released
     *         within the timeout; {@code false} if rate-limited or timed out
     */
    public boolean execute(DecodeReservation victim, String endpointKey,
                           long confirmTimeoutMs) {
        // 1. Rate limit check
        if (!rateLimiter.tryAcquire(endpointKey)) {
            log.debug("Running preempt rate-limited: ep={} reqId={}",
                    endpointKey, victim.requestId());
            return false;
        }

        // 2. Cancel victim
        long requestId = victim.requestId();
        cancelAction.accept(requestId, CancelReasonPB.CANCEL_REASON_PRIORITY_PREEMPTED);
        log.info("Preempting running request: reqId={} pri={} kv={} ep={}",
                requestId, victim.priority(), victim.kvTokensRequired(), endpointKey);

        // 3. Wait for release (poll tracker until victim removed or timeout)
        long deadline = System.currentTimeMillis() + confirmTimeoutMs;
        while (System.currentTimeMillis() < deadline) {
            if (decodeTracker.getReservation(endpointKey, requestId) == null) {
                log.info("Running preempt confirmed: reqId={} released on ep={}",
                        requestId, endpointKey);
                return true;
            }
            try {
                Thread.sleep(POLL_INTERVAL_MS);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                log.warn("Running preempt interrupted while waiting for release: reqId={}",
                        requestId);
                return false;
            }
        }

        // Timeout — victim not released
        log.warn("Running preempt timeout: reqId={} not released within {}ms on ep={}",
                requestId, confirmTimeoutMs, endpointKey);
        return false;
    }
}
