package org.flexlb.service;

import org.flexlb.consistency.LBStatusConsistencyService;
import org.flexlb.dao.loadbalance.BatchScheduleRequest;
import org.flexlb.dao.loadbalance.BatchScheduleResponse;
import org.flexlb.exception.BatchScheduleTransportException;
import org.flexlb.exception.EngineReadTimeoutException;
import org.flexlb.exception.HttpErrorResponseException;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.transport.GeneralHttpNettyService;
import org.flexlb.util.JsonUtils;
import org.flexlb.util.Logger;
import org.flexlb.util.RateLimitedWarn;
import org.springframework.stereotype.Component;
import reactor.core.publisher.Mono;

import java.net.URI;
import java.util.concurrent.TimeUnit;

/**
 * Resolves a {@code /batch_schedule} request against the master node, regardless
 * of whether the caller runs on the master itself or on a slave that must
 * forward to the elected master.
 *
 * <p>Transport failures (master address unknown, network error talking to master) are
 * signalled as {@link BatchScheduleTransportException}; business failures from the master
 * are returned as a normal response with {@code success=false}.
 */
@Component
public class BatchScheduleCoordinator {

    private static final int MAX_FORWARD_HOPS = 1;

    private final RouteService routeService;
    private final LBStatusConsistencyService consistency;
    private final GeneralHttpNettyService httpNettyService;
    private final EngineHealthReporter engineHealthReporter;
    /** A dead/unknown master fails every forwarded batch request; cap the ERROR stream at 1/s. */
    private final RateLimitedWarn masterUnreachableWarn = new RateLimitedWarn(1, TimeUnit.SECONDS);

    public BatchScheduleCoordinator(RouteService routeService,
                                    LBStatusConsistencyService consistency,
                                    GeneralHttpNettyService httpNettyService,
                                    EngineHealthReporter engineHealthReporter) {
        this.routeService = routeService;
        this.consistency = consistency;
        this.httpNettyService = httpNettyService;
        this.engineHealthReporter = engineHealthReporter;
    }

    public Mono<BatchScheduleResponse> schedule(BatchScheduleRequest request) {
        if (consistency.isNeedConsistency() && !consistency.isMaster()) {
            return forwardToMaster(request);
        }
        return resolveLocally(request, consistency.getMasterHostIpPort());
    }

    private Mono<BatchScheduleResponse> forwardToMaster(BatchScheduleRequest request) {
        String master = consistency.getMasterHostIpPort();
        if (master == null) {
            masterUnreachableWarn.warn("[BatchSchedule] Master unreachable: no elected master");
            engineHealthReporter.reportForwardToMasterResult("LOCAL", "MASTER_NULL");
            return Mono.error(new BatchScheduleTransportException(
                    "master unreachable", "MASTER_NULL"));
        }
        URI uri = URI.create("http://" + master);
        // Match the gRPC forwarder's guard: stale leader views must not form relay loops.
        long incomingHop = Integer.toUnsignedLong(request.getForwardHop());
        String blockedReason = incomingHop >= MAX_FORWARD_HOPS
                ? "FORWARD_HOP_LIMIT"
                : uri.getHost() != null && uri.getHost().equals(consistency.getLocalHostIp())
                        ? "SELF_FORWARD_BLOCKED"
                        : null;
        if (blockedReason != null) {
            masterUnreachableWarn.warn("[BatchSchedule] Forward blocked: reason={}, master={}, hop={}",
                    blockedReason, master, incomingHop);
            engineHealthReporter.reportForwardToMasterResult(uri.getHost(), blockedReason);
            return Mono.error(new BatchScheduleTransportException(
                    "batch schedule forward blocked: " + blockedReason, blockedReason));
        }

        BatchScheduleRequest forwarded = new BatchScheduleRequest();
        forwarded.setBatchCount(request.getBatchCount());
        forwarded.setAssignBe(request.isAssignBe());
        forwarded.setAssignFe(request.isAssignFe());
        forwarded.setForwardHop(Math.toIntExact(incomingHop + 1));
        Logger.debug("[BatchSchedule] Forwarding to master {}: batchCount={}", master, request.getBatchCount());
        return httpNettyService.request(forwarded, uri, "/rtp_llm/batch_schedule", BatchScheduleResponse.class)
                .doOnNext(response -> engineHealthReporter.reportForwardToMasterResult(
                        uri.getHost(), String.valueOf(response.getCode())))
                .onErrorResume(e -> {
                    if (e instanceof BatchScheduleTransportException) {
                        return Mono.error(e);
                    }
                    String errorCode;
                    if (e instanceof HttpErrorResponseException httpError) {
                        if (isUnsupportedBatchEndpoint(httpError.getStatusCode())) {
                            // Rolling upgrade: an older elected master does not expose the new
                            // endpoint yet. The follower has the replicated routing view, so use
                            // the explicit local compatibility path until leadership/version
                            // converges. Other HTTP/transport failures remain fail-closed.
                            String compatibilityCode =
                                    "BATCH_ENDPOINT_UNSUPPORTED_" + httpError.getStatusCode();
                            masterUnreachableWarn.warn("[BatchSchedule] Master {} lacks batch endpoint "
                                            + "(HTTP {}); resolving locally for rolling-upgrade compatibility",
                                    master, httpError.getStatusCode());
                            engineHealthReporter.reportForwardToMasterResult(
                                    uri.getHost(), compatibilityCode);
                            return resolveLocally(request, master);
                        }
                        BatchScheduleResponse businessFailure =
                                JsonUtils.toObjectOrNull(httpError.getBody(), BatchScheduleResponse.class);
                        if (businessFailure != null) {
                            engineHealthReporter.reportForwardToMasterResult(
                                    uri.getHost(), String.valueOf(businessFailure.getCode()));
                            return Mono.just(businessFailure);
                        }
                        // The connection succeeded — master answered with an HTTP error whose
                        // body isn't a business response. Tagging it CONNECT_FAILED would send
                        // an operator chasing the network instead of the master process.
                        errorCode = "HTTP_ERROR";
                    } else {
                        errorCode = e instanceof EngineReadTimeoutException ? "TIMEOUT" : "CONNECT_FAILED";
                    }
                    masterUnreachableWarn.warn("[BatchSchedule] Master unreachable, errorCode={}, err={}",
                            errorCode, e.toString());
                    engineHealthReporter.reportForwardToMasterResult("LOCAL", errorCode);
                    return Mono.error(new BatchScheduleTransportException(
                            "master unreachable: " + errorCode, errorCode));
                });
    }

    private Mono<BatchScheduleResponse> resolveLocally(BatchScheduleRequest request,
                                                       String electedMaster) {
        return routeService.batchSchedule(request)
                .doOnNext(response -> {
                    response.setRealMasterHost(electedMaster);
                    response.setResolvedLocally(true);
                });
    }

    private static boolean isUnsupportedBatchEndpoint(int statusCode) {
        return statusCode == 404 || statusCode == 405 || statusCode == 501;
    }
}
