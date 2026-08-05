package org.flexlb.balance.autotpm;

import org.flexlb.balance.scheduler.FlexlbBatchScheduler;
import org.flexlb.config.ConfigService;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.util.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.util.concurrent.CompletableFuture;

/**
 * Phase 2: Priority-aware admission scheduler for Auto-TPM.
 *
 * <p>Wraps {@link FlexlbBatchScheduler} and sets the absolute deadline
 * ({@code now + sloMs}) on each {@link BalanceContext} before delegation.
 * The deadline is consumed by {@link PriorityDeadlineBatcherAlgorithm}
 * for composite priority + deadline sorting and expiry detection.
 *
 * <p>This is the entry point for requests when {@code flexlbBatchAlgorithm=priority_deadline}
 * is configured. It can be wired in front of {@link FlexlbBatchScheduler}
 * by the route layer.
 */
@Component
public class PriorityAdmissionScheduler {

    private final FlexlbBatchScheduler batchScheduler;
    private final ConfigService configService;

    @Autowired
    public PriorityAdmissionScheduler(FlexlbBatchScheduler batchScheduler,
                                       ConfigService configService) {
        this.batchScheduler = batchScheduler;
        this.configService = configService;
    }

    /**
     * Submit a request with priority-aware admission.
     *
     * <p>Sets the absolute deadline on the context before delegating to
     * the underlying batch scheduler. The deadline is computed as
     * {@code now + resolveSloMs(seqLen)}.
     */
    public CompletableFuture<Response> submit(BalanceContext ctx) {
        if (ctx == null || ctx.getRequest() == null) {
            return batchScheduler.submit(ctx);
        }

        long now = System.currentTimeMillis();
        long sloMs = configService.loadBalanceConfig().resolveSloMs(
                ctx.getRequest().getSeqLen());
        long deadlineMs = now + sloMs;
        ctx.setDeadlineMs(deadlineMs);

        Logger.debug("PriorityAdmissionScheduler submit request_id={} priority={} slo_ms={} deadline_ms={}",
                ctx.getRequestId(), ctx.getPriority(), sloMs, deadlineMs);

        return batchScheduler.submit(ctx);
    }
}
