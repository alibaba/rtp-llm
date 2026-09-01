package org.flexlb.balance.strategy;

import org.flexlb.balance.session.SessionPlacementStore;
import org.flexlb.config.RoutingConfig;
import org.flexlb.dao.loadbalance.Request;

import java.util.function.IntFunction;
import java.util.function.IntToLongFunction;

final class SessionAffinityPolicy {
    private SessionAffinityPolicy() {
    }

    static Decision evaluate(Request request,
                             RoutingConfig.SessionAffinityConfig config,
                             SessionPlacementStore store,
                             int candidateCount,
                             IntFunction<String> endpoint,
                             IntToLongFunction score,
                             IntToLongFunction cacheHit,
                             long minScore) {
        if (config == null || request.getSessionSchemaVersion() != Request.SESSION_SCHEMA_VERSION
                || request.getInferenceSessionId() == null
                || request.getInferenceSessionId().isBlank()) {
            return Decision.none(Reason.DISABLED);
        }
        String model = request.getModel();
        String sessionId = request.getInferenceSessionId();
        if (request.getInferenceSessionState() == Request.SessionState.NEW) {
            store.invalidate(model, sessionId);
            return Decision.none(Reason.NEW_SESSION);
        }
        if (request.getInferenceSessionState() != Request.SessionState.ESTABLISHED) {
            return Decision.none(Reason.DISABLED);
        }
        for (int i = 0; i < candidateCount; i++) {
            if (cacheHit.applyAsLong(i) > 0) {
                return Decision.none(Reason.EXACT_CACHE_PRESENT);
            }
        }
        var placement = store.find(model, sessionId, config.getTtlMs());
        if (placement.isEmpty()) {
            return Decision.none(Reason.NO_PLACEMENT);
        }
        long cutoff = saturatedAdd(minScore, config.getMaxExtraTtftMs());
        for (int i = 0; i < candidateCount; i++) {
            if (placement.get().ipPort().equals(endpoint.apply(i))) {
                return score.applyAsLong(i) <= cutoff
                        ? new Decision(i, Reason.SESSION_AFFINITY, cutoff)
                        : Decision.none(Reason.OVER_CAP);
            }
        }
        return Decision.none(Reason.ENDPOINT_UNAVAILABLE);
    }

    private static long saturatedAdd(long left, long right) {
        return left > Long.MAX_VALUE - right ? Long.MAX_VALUE : left + right;
    }

    enum Reason {
        DISABLED,
        NEW_SESSION,
        EXACT_CACHE_PRESENT,
        NO_PLACEMENT,
        ENDPOINT_UNAVAILABLE,
        OVER_CAP,
        SESSION_AFFINITY
    }

    record Decision(int preferredIndex, Reason reason, long scoreCutoffMs) {
        static Decision none(Reason reason) {
            return new Decision(-1, reason, 0L);
        }

        boolean hasPreference() {
            return preferredIndex >= 0;
        }
    }
}
