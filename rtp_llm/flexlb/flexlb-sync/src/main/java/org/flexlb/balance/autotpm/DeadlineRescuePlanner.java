package org.flexlb.balance.autotpm;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.scheduler.BatcherContext;
import org.flexlb.balance.scheduler.BatchItem;
import org.flexlb.balance.scheduler.FlexlbBatchScheduler;
import org.flexlb.balance.scheduler.QueueSnapshot;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Phase 5: Periodic scanner that rescues requests approaching their deadline.
 *
 * <p>Scans all prefill endpoint queues for requests in the danger zone
 * ({@code deadline - now < danger_threshold}). For each eligible request:
 * <ol>
 *   <li>CAS remove from the P (prefill) queue via {@link BatcherContext#tryRemove}</li>
 *   <li>Remove from {@link FlexlbBatchScheduler} inflight (bypass duplicate check)</li>
 *   <li>Rollback the D (decode) reservation via {@link DecodeAdmissionTracker#release}</li>
 *   <li>Re-admit through {@link PriorityAdmissionScheduler#submit}</li>
 * </ol>
 *
 * <p>Guarantees:
 * <ul>
 *   <li><b>Migration storm bounded</b>: each request rescued at most
 *       {@code max_transfer} times (tracked per-request).</li>
 *   <li><b>No infinite retry</b>: when re-admission fails, the original
 *       future is completed with {@link StrategyErrorType#DEADLINE_RESCUE_FAILED}.</li>
 *   <li><b>Non-lowest-priority only</b>: P30 (lowest) is never rescued.</li>
 *   <li><b>Scan throughput bounded</b>: at most {@code max_rescue_per_tick}
 *       requests are rescued per scan tick.</li>
 * </ul>
 */
@Component
public class DeadlineRescuePlanner {

    private static final Logger log = LoggerFactory.getLogger(DeadlineRescuePlanner.class);

    private final ConfigService configService;
    private final EndpointRegistry endpointRegistry;
    private final FlexlbBatchScheduler batchScheduler;
    private final PriorityAdmissionScheduler admissionScheduler;
    private final DecodeAdmissionTracker decodeTracker;

    /** Per-request transfer count (requestId → times rescued). */
    private final ConcurrentHashMap<Long, AtomicInteger> transferCount = new ConcurrentHashMap<>();

    @Autowired
    public DeadlineRescuePlanner(ConfigService configService,
                                  EndpointRegistry endpointRegistry,
                                  FlexlbBatchScheduler batchScheduler,
                                  PriorityAdmissionScheduler admissionScheduler) {
        this.configService = configService;
        this.endpointRegistry = endpointRegistry;
        this.batchScheduler = batchScheduler;
        this.admissionScheduler = admissionScheduler;
        this.decodeTracker = batchScheduler.getDecodeAdmissionTracker();
    }

    /**
     * Constructor for testing — allows injecting a specific
     * {@link DecodeAdmissionTracker} (e.g. a mock or real instance).
     */
    DeadlineRescuePlanner(ConfigService configService,
                          EndpointRegistry endpointRegistry,
                          FlexlbBatchScheduler batchScheduler,
                          PriorityAdmissionScheduler admissionScheduler,
                          DecodeAdmissionTracker decodeTracker) {
        this.configService = configService;
        this.endpointRegistry = endpointRegistry;
        this.batchScheduler = batchScheduler;
        this.admissionScheduler = admissionScheduler;
        this.decodeTracker = decodeTracker;
    }

    /**
     * Periodic scan. Called by Spring {@code @Scheduled} every 100 ms.
     *
     * <p>Iterates all prefill endpoints, inspects their queue snapshots,
     * and rescues eligible danger-zone requests.
     */
    @Scheduled(fixedDelay = 100L)
    public void scan() {
        FlexlbConfig config = configService.loadBalanceConfig();
        long dangerThreshold = config.getAutoTpmDangerThresholdMs();
        int maxTransfer = config.getAutoTpmMaxTransfer();
        int maxRescuePerTick = config.getAutoTpmMaxRescuePerTick();
        long now = System.currentTimeMillis();
        int rescued = 0;

        for (PrefillEndpoint ep : endpointRegistry.getPrefillEndpoints().values()) {
            if (rescued >= maxRescuePerTick) {
                break;
            }

            BatcherContext batcherCtx = ep.getBatcherContext();
            QueueSnapshot snapshot = batcherCtx.snapshot();
            long expectedVersion = snapshot.version();

            for (QueueSnapshot.ItemSummary item : snapshot.items()) {
                if (rescued >= maxRescuePerTick) {
                    break;
                }

                long deadlineMs = item.deadlineMs();
                if (deadlineMs <= 0) {
                    continue; // deadline not set, skip
                }

                long remaining = deadlineMs - now;
                if (remaining >= dangerThreshold) {
                    continue; // not in danger zone
                }

                int priority = item.priority();
                if (priority <= 30) {
                    continue; // don't rescue lowest priority (P30)
                }

                long requestId = item.requestId();
                int transfers = getTransferCount(requestId);
                if (transfers >= maxTransfer) {
                    continue; // transfer limit exceeded
                }

                // Attempt rescue
                if (tryRescue(batcherCtx, requestId, expectedVersion)) {
                    rescued++;
                    incrementTransferCount(requestId);
                }
            }
        }

        if (rescued > 0) {
            log.info("Deadline rescue: rescued {} requests in danger zone", rescued);
        }
    }

    /**
     * Execute the rescue flow for a single request.
     *
     * @return {@code true} if the request was successfully removed and
     *         re-admitted; {@code false} on CAS failure or not found
     */
    private boolean tryRescue(BatcherContext batcherCtx, long requestId, long expectedVersion) {
        // Step 1: CAS remove from P queue
        List<BatchItem> removed = batcherCtx.tryRemove(Set.of(requestId), expectedVersion);
        if (removed == null || removed.isEmpty()) {
            // CAS failed (version mismatch) or request not found
            return false;
        }

        BatchItem item = removed.get(0);
        BalanceContext balanceCtx = item.ctx();

        // Step 2: Remove from FlexlbBatchScheduler inflight (bypass duplicate check)
        // This also rolls back any held decode reservation + decode endpoint inflight.
        batchScheduler.removeInflightForRescue(requestId);

        // Step 3: Explicitly release D reservation (belt-and-suspenders —
        // removeInflightForRescue already released it via rollbackOnce,
        // but call explicitly per the spec; release is idempotent).
        DecodeEndpoint decodeEp = item.decodeEp();
        if (decodeEp != null && item.decode() != null) {
            decodeTracker.release(decodeEp.ipPort(), requestId);
        }

        // Step 4: Re-admit through scheduler (may trigger eviction at new endpoint)
        try {
            CompletableFuture<Response> newFuture = admissionScheduler.submit(balanceCtx);
            propagateResult(newFuture, item.future());
            return true;
        } catch (Exception e) {
            log.error("Deadline rescue re-admission threw for requestId={}: {}",
                    requestId, e.getMessage(), e);
            completeRescueError(item.future(),
                    "DEADLINE_RESCUE_FAILED: " + e.getMessage());
            return false;
        }
    }

    /**
     * Propagate the new future's result to the original caller's future.
     *
     * <p>If re-admission succeeds, the original future completes with the
     * new response. If re-admission fails (error response or exception),
     * the original future completes with {@link StrategyErrorType#DEADLINE_RESCUE_FAILED}.
     */
    private void propagateResult(CompletableFuture<Response> newFuture,
                                  CompletableFuture<Response> oldFuture) {
        newFuture.whenComplete((resp, ex) -> {
            if (ex != null) {
                completeRescueError(oldFuture,
                        "DEADLINE_RESCUE_FAILED: " + ex.getMessage());
            } else if (!resp.isSuccess()) {
                completeRescueError(oldFuture,
                        "DEADLINE_RESCUE_FAILED: " + resp.getErrorMessage());
            } else if (!oldFuture.isDone()) {
                oldFuture.complete(resp);
            }
        });
    }

    /**
     * Complete the original future with a DEADLINE_RESCUE_FAILED error.
     * No infinite migration — the request is permanently failed.
     */
    private static void completeRescueError(CompletableFuture<Response> future,
                                            String message) {
        if (future.isDone()) {
            return;
        }
        Response errorResp = Response.error(StrategyErrorType.DEADLINE_RESCUE_FAILED);
        errorResp.setErrorMessage(message);
        future.complete(errorResp);
    }

    private int getTransferCount(long requestId) {
        AtomicInteger count = transferCount.get(requestId);
        return count != null ? count.get() : 0;
    }

    private void incrementTransferCount(long requestId) {
        transferCount.computeIfAbsent(requestId, k -> new AtomicInteger(0))
                .incrementAndGet();
    }

    /**
     * Clean up transfer count for a completed request.
     * Called periodically or on terminal state to prevent map growth.
     */
    public void cleanupTransfers(long requestId) {
        transferCount.remove(requestId);
    }
}
