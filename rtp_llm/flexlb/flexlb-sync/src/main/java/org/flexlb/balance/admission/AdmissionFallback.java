package org.flexlb.balance.admission;

import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;

import java.util.concurrent.CompletableFuture;

/** Optional admission path invoked after ordinary request placement declines. */
@FunctionalInterface
public interface AdmissionFallback {

    /**
     * Attempt to take over one exact request admission.
     *
     * @param context exact request context still owned by the caller
     * @param future exact public response future for that request generation
     * @return {@code false} when no takeover occurred and the caller retains
     *         exclusive responsibility for the future; {@code true} only after
     *         this implementation has taken ownership and is responsible for
     *         publishing either ACTIVE delivery or the terminal response
     */
    boolean tryAdmit(
            BalanceContext context,
            CompletableFuture<Response> future);
}
