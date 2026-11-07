package org.flexlb.balance.scheduler;

import java.util.function.Function;

/** Captures before/after observations for state-transition assertions without production event wrappers. */
public final class RequestChangeAssertions {
    private RequestChangeAssertions() {}
    public enum UpdateKind { APPLIED, NO_CHANGE, STALE }
    public record Update(UpdateKind kind, ContextResult.Snapshot before, ContextResult.Snapshot after) {}

    public static Update record(RequestContext request,
            Function<RequestContext, ContextResult.Snapshot> change) {
        synchronized (request) {
            var before = request.snapshot();
            var after = change.apply(request);
            return new Update(after == null ? UpdateKind.STALE
                    : after == before ? UpdateKind.NO_CHANGE : UpdateKind.APPLIED,
                    before, after == null ? before : after);
        }
    }
}
