package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;

import java.util.List;

/**
 * Immutable result of {@link Router#route}, carrying direct endpoint
 * references so downstream schedulers avoid re-parsing ip:port from a
 * {@link Response} and re-looking-up endpoints in the registry.
 *
 * <p>Success: {@code prefillEp} / {@code decodeEp} are non-null (when the
 * corresponding role was routed), {@code serverStatusList} carries the
 * full strategy-selected metadata (scores, debug info, dpRank, group).
 *
 * <p>Failure: {@code errorType} identifies the failure category (used by
 * {@code QueueScheduler} retry logic), {@code errorMessage} carries the
 * detail message.
 *
 * <p>{@link #toResponse()} bridges to the legacy {@link Response} type
 * for schedulers that still complete futures with {@code Response}.
 */
public record RouteResult(PrefillEndpoint prefillEp,
                          DecodeEndpoint decodeEp,
                          List<ServerStatus> serverStatusList,
                          StrategyErrorType errorType,
                          String errorMessage) {

    /** {@code errorMessage == null} means routing succeeded. */
    public boolean isSuccess() {
        return errorMessage == null;
    }

    public static RouteResult success(PrefillEndpoint prefillEp,
                                      DecodeEndpoint decodeEp,
                                      List<ServerStatus> serverStatusList) {
        return new RouteResult(prefillEp, decodeEp, serverStatusList, null, null);
    }

    public static RouteResult failure(StrategyErrorType errorType, String errorMessage) {
        return new RouteResult(null, null, null, errorType, errorMessage);
    }

    /**
     * Convert to a legacy {@link Response} for futures that still
     * complete with {@code Response} (all three scheduler paths).
     */
    public Response toResponse() {
        if (isSuccess()) {
            Response response = new Response();
            response.setSuccess(true);
            response.setServerStatus(serverStatusList);
            return response;
        }
        Response response = Response.error(errorType);
        if (errorMessage != null) {
            response.setErrorMessage(errorMessage);
        }
        return response;
    }
}
