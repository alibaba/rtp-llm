package org.flexlb.service;

import java.net.URI;
import java.util.concurrent.TimeoutException;

import lombok.AllArgsConstructor;
import lombok.Getter;
import org.flexlb.consistency.LBStatusConsistencyService;
import org.flexlb.dao.loadbalance.BatchScheduleRequest;
import org.flexlb.dao.loadbalance.BatchScheduleResponse;
import org.flexlb.exception.BatchScheduleTransportException;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.transport.GeneralHttpNettyService;
import org.flexlb.util.Logger;
import org.springframework.stereotype.Component;
import reactor.core.publisher.Mono;

/**
 * Resolves a {@code /batch_schedule} request against the master node, regardless
 * of whether the caller runs on the master itself or on a slave that must
 * forward to the elected master.
 *
 * <p>Callers receive an {@link Outcome} carrying the response and its origin
 * ({@link Source#LOCAL} or {@link Source#FORWARDED}). Transport failures
 * (master address unknown, network error talking to master) are signalled as
 * {@link BatchScheduleTransportException}; business failures from the master
 * are returned as a normal {@code Outcome} with {@code response.success=false}.
 */
@Component
public class BatchScheduleCoordinator {

    public enum Source {
        LOCAL, FORWARDED
    }

    @Getter
    @AllArgsConstructor
    public static class Outcome {
        private final BatchScheduleResponse response;
        private final Source source;
    }

    private final RouteService routeService;
    private final LBStatusConsistencyService consistency;
    private final GeneralHttpNettyService httpNettyService;
    private final EngineHealthReporter engineHealthReporter;

    public BatchScheduleCoordinator(RouteService routeService,
                                    LBStatusConsistencyService consistency,
                                    GeneralHttpNettyService httpNettyService,
                                    EngineHealthReporter engineHealthReporter) {
        this.routeService = routeService;
        this.consistency = consistency;
        this.httpNettyService = httpNettyService;
        this.engineHealthReporter = engineHealthReporter;
    }

    public Mono<Outcome> schedule(BatchScheduleRequest request) {
        if (consistency.isNeedConsistency() && !consistency.isMaster()) {
            return forwardToMaster(request);
        }
        return routeService.batchSchedule(request)
                .map(response -> {
                    response.setRealMasterHost(consistency.getMasterHostIpPort());
                    return new Outcome(response, Source.LOCAL);
                });
    }

    private Mono<Outcome> forwardToMaster(BatchScheduleRequest request) {
        String master = consistency.getMasterHostIpPort();
        if (master == null) {
            Logger.error("[BatchSchedule] Master unreachable");
            engineHealthReporter.reportForwardToMasterResult("LOCAL", "MASTER_NULL");
            return Mono.error(new BatchScheduleTransportException(
                    "master unreachable", "MASTER_NULL"));
        }
        Logger.info("[BatchSchedule] Forwarding to master {}: {}", master, request);
        URI uri = URI.create("http://" + master);
        return httpNettyService.request(request, uri, "/rtp_llm/batch_schedule", BatchScheduleResponse.class)
                .doOnNext(response -> engineHealthReporter.reportForwardToMasterResult(
                        uri.getHost(), String.valueOf(response.getCode())))
                .map(response -> new Outcome(response, Source.FORWARDED))
                .onErrorResume(e -> {
                    if (e instanceof BatchScheduleTransportException) {
                        return Mono.error(e);
                    }
                    String errorCode = e instanceof TimeoutException ? "TIMEOUT" : "CONNECT_FAILED";
                    Logger.error("[BatchSchedule] Master unreachable, errorCode={}", errorCode, e);
                    engineHealthReporter.reportForwardToMasterResult("LOCAL", errorCode);
                    return Mono.error(new BatchScheduleTransportException(
                            "master unreachable: " + errorCode, errorCode));
                });
    }
}
