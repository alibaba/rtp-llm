package org.flexlb.balance.strategy;

import org.flexlb.balance.session.SessionPlacementStore;
import org.flexlb.config.RoutingConfig;
import org.flexlb.dao.loadbalance.Request;

import java.util.function.IntFunction;
import java.util.function.IntToLongFunction;

final class SessionAffinityPolicy {
    private SessionAffinityPolicy() {
    }

    static void initialize(Request request,
                           RoutingConfig.SessionAffinityConfig config,
                           SessionPlacementStore store) {
        if (request.getSessionSchemaVersion() != Request.SESSION_SCHEMA_VERSION
                || request.getInferenceSessionId() == null
                || request.getInferenceSessionId().isBlank()) {
            return;
        }
        synchronized (request) {
            if (request.getSessionPlacementEpoch() >= 0) {
                return;
            }
            String model = request.getModel();
            String sessionId = request.getInferenceSessionId();
            if (request.getInferenceSessionState() == Request.SessionState.NEW) {
                request.setSessionPlacementEpoch(config == null
                        ? store.resetIfPresent(model, sessionId)
                        : store.reset(model, sessionId));
            } else if (config != null
                    && request.getInferenceSessionState() == Request.SessionState.ESTABLISHED) {
                request.setSessionPlacementEpoch(store.currentEpoch(model, sessionId));
            }
        }
    }

    static Decision evaluate(Request request,
                             RoutingConfig.SessionAffinityConfig config,
                             SessionPlacementStore store,
                             int candidateCount,
                             IntFunction<String> endpoint,
                             IntToLongFunction score,
                             IntToLongFunction cacheHit,
                             long minScore) {
        if (request.getSessionSchemaVersion() != Request.SESSION_SCHEMA_VERSION
                || request.getInferenceSessionId() == null
                || request.getInferenceSessionId().isBlank()) {
            return Decision.none(Reason.DISABLED);
        }
        String model = request.getModel();
        String sessionId = request.getInferenceSessionId();
        Request.SessionState state = request.getInferenceSessionState();
        initialize(request, config, store);
        if (state == Request.SessionState.NEW) {
            return Decision.none(Reason.NEW_SESSION);
        }
        if (config == null || state != Request.SessionState.ESTABLISHED) {
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
                        ? new Decision(i, Reason.SESSION_AFFINITY)
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

    record Decision(int preferredIndex, Reason reason) {
        static Decision none(Reason reason) {
            return new Decision(-1, reason);
        }

        boolean hasPreference() {
            return preferredIndex >= 0;
        }
    }
}
