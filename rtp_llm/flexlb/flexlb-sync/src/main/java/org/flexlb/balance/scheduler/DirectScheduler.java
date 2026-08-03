package org.flexlb.balance.scheduler;

import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;

import java.util.concurrent.CompletableFuture;

/**
 * Thin-wrapper scheduler for DIRECT mode.
 *
 * <p>Delegates request routing to the existing {@link Router} (DefaultRouter).
 * DIRECT mode has no inflight tracking — the response is produced synchronously
 * and the future is already completed on return. Cancel is a no-op
 * (inherited from {@link AbstractScheduler}).
 */
public class DirectScheduler extends AbstractScheduler {

    private final Router delegate;

    public DirectScheduler(Router delegate) {
        this.delegate = delegate;
    }

    @Override
    public CompletableFuture<Response> submit(BalanceContext ctx) {
        try {
            Response response = delegate.route(ctx);
            return CompletableFuture.completedFuture(response);
        } catch (Exception e) {
            return CompletableFuture.failedFuture(e);
        }
    }
}
