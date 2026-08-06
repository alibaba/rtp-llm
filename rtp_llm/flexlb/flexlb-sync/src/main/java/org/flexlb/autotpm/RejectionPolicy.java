package org.flexlb.autotpm;

import org.flexlb.balance.scheduler.BatchItem;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;

/**
 * Handles rejection of low-priority requests that have been yielded in the
 * priority queue. When a yielded item's queue deadline expires, it is settled
 * through the existing {@link BatchItem#failExpired()} path (Drop/8400).
 *
 * <p>This class provides a specific rejection completion with a priority-aware
 * message, reusing the {@code NO_AVAILABLE_WORKER} (8400) error code.
 *
 * <p>Guarantees:
 * <ul>
 *   <li>Future completion is idempotent (no-op if already done)</li>
 *   <li>Never calls Engine cancel (no cancel logic in this Stage)</li>
 * </ul>
 */
public final class RejectionPolicy {

    private RejectionPolicy() {}

    /**
     * Complete a yielded item as rejected with 8400 and a priority-aware message.
     *
     * @param item            the batch item to reject
     * @param incomingPriority the priority of the request that caused the yield
     * @return true if the future was completed by this call, false if already done
     */
    public static boolean rejectYielded(BatchItem item, int incomingPriority) {
        if (item.future().isDone()) {
            return false;
        }
        // D10: yielded-queue-deadline clearing is a deadline miss — mark for
        // the scheduler's whenComplete metric hook (deadline_miss.count).
        item.markDeadlineMiss();
        item.rollbackOnce();
        Response errorResp = Response.error(StrategyErrorType.NO_AVAILABLE_WORKER);
        errorResp.setErrorMessage("auto_tpm: yielded for priority=" + incomingPriority);
        return item.future().complete(errorResp);
    }
}
