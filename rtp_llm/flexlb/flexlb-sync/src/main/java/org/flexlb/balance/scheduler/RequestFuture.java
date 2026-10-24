package org.flexlb.balance.scheduler;

import org.flexlb.dao.loadbalance.Response;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;

/** External writes remain synchronous and arbitrate through Scheduler; owned writes run outside Context. */
public final class RequestFuture extends CompletableFuture<Response> {
    public enum Completion { RESPONSE, FAILURE, CANCELLATION }
    @FunctionalInterface public interface CompletionTarget {
        boolean complete(Completion completion, Response response, Throwable failure);
    }
    private volatile CompletionTarget target;

    public RequestFuture(CompletionTarget target) { this.target = Objects.requireNonNull(target); }

    @Override public boolean complete(Response response) {
        var current = target;
        return current != null && current.complete(Completion.RESPONSE, response, null);
    }

    @Override public boolean completeExceptionally(Throwable failure) {
        Objects.requireNonNull(failure);
        var current = target;
        return current != null && current.complete(Completion.FAILURE, null, failure);
    }

    @Override public boolean cancel(boolean mayInterruptIfRunning) {
        var current = target;
        return current != null && current.complete(Completion.CANCELLATION, null, null);
    }

    boolean completeOwned(Response response) {
        try { return super.complete(response); } finally { target = null; }
    }
    boolean completeExceptionallyOwned(Throwable failure) {
        try { return super.completeExceptionally(failure); } finally { target = null; }
    }
    boolean cancelOwned() {
        try { return super.cancel(false); } finally { target = null; }
    }
}
