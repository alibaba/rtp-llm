package org.flexlb.balance.session;

import org.flexlb.config.RoutingConfig;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.util.Logger;

public final class SessionPlacementLifecycle {
    private SessionPlacementLifecycle() {
    }

    public static void initialize(Request request,
                                  RoutingConfig.SessionAffinityConfig config,
                                  SessionPlacementStore store) {
        if (request.getSessionSchemaVersion() != Request.SESSION_SCHEMA_VERSION
                || request.getInferenceSessionId() == null
                || request.getInferenceSessionId().isBlank()) {
            return;
        }
        synchronized (request) {
            if (request.getSessionPlacementEpoch() >= 0
                    || request.getInferenceSessionState() == Request.SessionState.UNSPECIFIED) {
                return;
            }
            try {
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
            } catch (RuntimeException exception) {
                request.setSessionPlacementEpoch(-1L);
                request.setInferenceSessionState(Request.SessionState.UNSPECIFIED);
                Logger.warn("Failed to initialize session placement, request_id={}",
                        request.getRequestId(), exception);
            }
        }
    }
}
