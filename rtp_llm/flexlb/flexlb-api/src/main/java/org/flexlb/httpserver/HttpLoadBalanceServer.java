package org.flexlb.httpserver;

import org.flexlb.balance.scheduler.QueueManager;
import org.flexlb.consistency.LBStatusConsistencyService;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.BatchScheduleContext;
import org.flexlb.dao.loadbalance.BatchScheduleRequest;
import org.flexlb.dao.loadbalance.BatchScheduleResponse;
import org.flexlb.dao.loadbalance.LogLevelUpdateRequest;
import org.flexlb.dao.loadbalance.QueueSnapshotResponse;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.pv.BatchPvLogData;
import org.flexlb.dao.pv.PvLogData;
import org.flexlb.domain.consistency.MasterChangeNotifyReq;
import org.flexlb.domain.consistency.MasterChangeNotifyResp;
import org.flexlb.domain.consistency.SyncLBStatusReq;
import org.flexlb.domain.consistency.SyncLBStatusResp;
import org.flexlb.exception.BatchScheduleTransportException;
import org.flexlb.exception.EngineReadTimeoutException;
import org.flexlb.service.BatchScheduleCoordinator;
import org.flexlb.service.RouteService;
import org.flexlb.service.grace.ActiveRequestCounter;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.transport.GeneralHttpNettyService;
import org.flexlb.util.Logger;
import org.springframework.context.annotation.Bean;
import org.springframework.core.codec.DecodingException;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.server.RouterFunction;
import org.springframework.web.reactive.function.server.ServerRequest;
import org.springframework.web.reactive.function.server.ServerResponse;
import org.springframework.web.server.ServerWebInputException;
import reactor.core.publisher.Mono;

import java.net.URI;
import java.util.function.Function;

import static org.springframework.web.reactive.function.server.RequestPredicates.accept;
import static org.springframework.web.reactive.function.server.RouterFunctions.route;

@Component
public class HttpLoadBalanceServer {
    private final GeneralHttpNettyService generalHttpNettyService;
    private final RouteService routeService;
    private final LBStatusConsistencyService lbStatusConsistencyService;
    private final EngineHealthReporter engineHealthReporter;
    private final QueueManager queueManager;
    private final ActiveRequestCounter activeRequestCounter;
    private final BatchScheduleCoordinator batchScheduleCoordinator;

    public HttpLoadBalanceServer(GeneralHttpNettyService generalHttpNettyService,
                                 RouteService routeService,
                                 LBStatusConsistencyService lbStatusConsistencyService,
                                 EngineHealthReporter engineHealthReporter,
                                 QueueManager queueManager,
                                 ActiveRequestCounter activeRequestCounter,
                                 BatchScheduleCoordinator batchScheduleCoordinator) {
        this.generalHttpNettyService = generalHttpNettyService;
        this.routeService = routeService;
        this.lbStatusConsistencyService = lbStatusConsistencyService;
        this.engineHealthReporter = engineHealthReporter;
        this.queueManager = queueManager;
        this.activeRequestCounter = activeRequestCounter;
        this.batchScheduleCoordinator = batchScheduleCoordinator;
    }

    @Bean
    public RouterFunction<ServerResponse> loadBalancePrefill() {
        return route()
                .POST("/rtp_llm/schedule", accept(MediaType.APPLICATION_JSON),
                        this::scheduleRequest)
                .POST("/rtp_llm/batch_schedule", accept(MediaType.APPLICATION_JSON),
                        this::batchScheduleRequest)
                .POST("/rtp_llm/master/info", accept(MediaType.APPLICATION_JSON),
                        this::responseMasterInfo)
                .POST("/rtp_llm/schedule_snapshot", accept(MediaType.APPLICATION_JSON),
                        this::dumpLBStatus)
                .POST("/rtp_llm/notify_master", accept(MediaType.APPLICATION_JSON),
                        this::notifyParticipant)
                .POST("/rtp_llm/update_log_level", accept(MediaType.APPLICATION_JSON),
                        this::debugMode)
                .GET("/rtp_llm/queue_snapshot", accept(MediaType.APPLICATION_JSON),
                        this::queueSnapshot)
                .build();
    }

    /**
     * Handles load balancing request scheduling.
     *
     * @param request the HTTP request containing the model inference request
     * @return a reactive response containing the load balancing result
     */
    public Mono<ServerResponse> scheduleRequest(ServerRequest request) {
        BalanceContext ctx = new BalanceContext();
        return request.bodyToMono(Request.class)
                .flatMap(req -> {
                    if (req.getRequestId() == 0) {
                        throw new IllegalArgumentException("requestId is 0");
                    }
                    ctx.setRequest(req);
                    return Mono.using(
                            activeRequestCounter::acquire,
                            ignored -> processScheduledRequest(ctx, req),
                            ActiveRequestCounter.RequestToken::close);
                })
                .onErrorResume(e -> handleRequestError(ctx, e))
                .doFinally(signal -> finalizeRequestContext(ctx));
    }

    public Mono<ServerResponse> batchScheduleRequest(ServerRequest request) {
        BatchScheduleContext bctx = new BatchScheduleContext();
        return request.bodyToMono(BatchScheduleRequest.class)
                .switchIfEmpty(Mono.error(new IllegalArgumentException("empty request body")))
                .flatMap(batchRequest -> {
                    bctx.setBatchRequest(batchRequest);
                    return Mono.using(
                            activeRequestCounter::acquire,
                            ignored -> processBatchScheduleRequest(bctx),
                            ActiveRequestCounter.RequestToken::close);
                })
                .onErrorResume(e -> {
                    Logger.error("Batch schedule request processing error", e);
                    // A malformed/empty body is a deterministic client error: a 500 would invite
                    // pointless retries and pollute the server error rate. Anything else escaping
                    // here is a genuine server-side failure and keeps the 500.
                    int status = isClientInputError(e) ? 400 : 500;
                    return batchError(bctx, status, StrategyErrorType.INVALID_REQUEST, e);
                })
                .doFinally(signal -> finalizeBatchContext(bctx));
    }

    private Mono<ServerResponse> processBatchScheduleRequest(BatchScheduleContext bctx) {
        return batchScheduleCoordinator.schedule(bctx.getBatchRequest())
                .flatMap(response -> {
                    bctx.setBatchResponse(response);
                    if (!response.isSuccess()) {
                        Logger.error("[BatchSchedule] failed: {}", response.getErrorMessage());
                    }
                    return json(statusOf(response), response);
                })
                .onErrorResume(BatchScheduleTransportException.class,
                        e -> batchError(bctx, 500, StrategyErrorType.NO_AVAILABLE_WORKER, e));
    }

    /**
     * HTTP status for a business response. INVALID_REQUEST rejections (batch_count out of range,
     * multi-role deployment, null request) are deterministic client errors → 400; every other
     * failure (NO_AVAILABLE_WORKER etc.) is a server-side condition and stays 500.
     */
    private static int statusOf(BatchScheduleResponse response) {
        if (response.isSuccess()) {
            return 200;
        }
        return response.getCode() == StrategyErrorType.INVALID_REQUEST.getErrorCode() ? 400 : 500;
    }

    /** Whether the error is a deterministic client-input failure (body decode / validation). */
    private static boolean isClientInputError(Throwable e) {
        return e instanceof ServerWebInputException
                || e instanceof DecodingException
                || e instanceof IllegalArgumentException;
    }

    /** Builds the error response and stamps it on the context; pv success/error derive from it. */
    private Mono<ServerResponse> batchError(BatchScheduleContext bctx, int httpStatus,
                                            StrategyErrorType type, Throwable e) {
        BatchScheduleResponse errorResponse = BatchScheduleResponse.error(type, e.getMessage());
        bctx.setBatchResponse(errorResponse);
        return json(httpStatus, errorResponse);
    }

    private Mono<ServerResponse> processScheduledRequest(BalanceContext ctx, Request req) {
        engineHealthReporter.reportArriveDelayTime(ctx);

        if (lbStatusConsistencyService.isNeedConsistency() && !lbStatusConsistencyService.isMaster()) {
            return forwardRequestToMaster(ctx, req);
        }

        return routeService.route(ctx)
                .flatMap(response -> handleRoutingResult(ctx, response))
                .doOnCancel(() -> {
                    ctx.setSuccess(false);
                    ctx.setErrorMessage("REQUEST_CANCELLED");
                    routeService.cancel(ctx);
                });
    }

    private Mono<ServerResponse> debugMode(ServerRequest serverRequest) {
        return serverRequest.bodyToMono(LogLevelUpdateRequest.class)
                .flatMap(logLevelUpdateRequest -> {
                    Logger.setGlobalLogLevel(logLevelUpdateRequest.getLogLevel());
                    return ServerResponse.ok()
                            .contentType(MediaType.APPLICATION_JSON)
                            .body(Mono.just("Success! logLevel=" + Logger.getGlobalLogLevel()), String.class);
                }).onErrorResume(e -> {
                    Logger.error("update logLevel error", e);
                    return ServerResponse.status(500)
                            .contentType(MediaType.APPLICATION_JSON)
                            .body(Mono.just(e.getMessage()), String.class);
                });
    }

    private Mono<ServerResponse> responseMasterInfo(ServerRequest request) {
        return request.bodyToMono(Request.class)
                .flatMap((Function<Request, Mono<ServerResponse>>) req -> {
                    Response result = new Response();
                    result.setRealMasterHost(lbStatusConsistencyService.getMasterHostIpPort());
                    result.setQueueLength(queueManager.getQueue().size());
                    result.setCode(200);
                    result.setSuccess(true);
                    return ServerResponse.ok()
                            .contentType(MediaType.APPLICATION_JSON)
                            .body(Mono.just(result), Response.class);
                }).onErrorResume(e -> {
                    Logger.error("responseMasterInfo error", e);
                    Response errorResponse = new Response();
                    errorResponse.setSuccess(false);
                    errorResponse.setCode(500);
                    errorResponse.setErrorMessage(e.getMessage());
                    return ServerResponse.status(500)
                            .contentType(MediaType.APPLICATION_JSON)
                            .body(Mono.just(errorResponse), Response.class);
                });
    }

    public Mono<ServerResponse> notifyParticipant(ServerRequest request) {
        return request.bodyToMono(MasterChangeNotifyReq.class)
                .flatMap(masterChangeNotifyReq -> {
                    MasterChangeNotifyResp resp = lbStatusConsistencyService.handleMasterChange(masterChangeNotifyReq);
                    return ServerResponse.ok()
                            .contentType(MediaType.APPLICATION_JSON)
                            .body(Mono.just(resp), MasterChangeNotifyReq.class);
                }).onErrorResume((Function<Throwable, Mono<ServerResponse>>) e -> {
                    Logger.error("notifyParticipant error", e);
                    return ServerResponse.status(500)
                            .contentType(MediaType.APPLICATION_JSON)
                            .body(Mono.just(e.getMessage()), String.class);
                });
    }

    public Mono<ServerResponse> dumpLBStatus(ServerRequest request) {
        return request.bodyToMono(SyncLBStatusReq.class)
                .flatMap(syncLBStatusReq -> {
                    SyncLBStatusResp resp = lbStatusConsistencyService.dumpLBStatus();
                    return ServerResponse.ok()
                            .contentType(MediaType.APPLICATION_JSON)
                            .body(Mono.just(resp), SyncLBStatusResp.class);
                }).onErrorResume(e -> {
                    Logger.error("dumpLBStatus error", e);
                    return ServerResponse.status(500)
                            .contentType(MediaType.APPLICATION_JSON)
                            .body(Mono.just(e.getMessage()), String.class);
                });
    }

    public Mono<ServerResponse> queueSnapshot(ServerRequest request) {
        try {
            QueueSnapshotResponse response = queueManager.snapshotQueue();
            return ServerResponse.ok()
                    .contentType(MediaType.APPLICATION_JSON)
                    .body(Mono.just(response), QueueSnapshotResponse.class);
        } catch (Exception e) {
            Logger.error("queueSnapshot error", e);
            return ServerResponse.status(500)
                    .contentType(MediaType.APPLICATION_JSON)
                    .body(Mono.just(e.getMessage()), String.class);
        }
    }

    /**
     * Forwards the request to the active master node.
     *
     * @param ctx the balance context
     * @param request the master request to forward
     * @return response from the master node
     */
    private Mono<ServerResponse> forwardRequestToMaster(BalanceContext ctx, Request request) {
        String master = lbStatusConsistencyService.getMasterHostIpPort();
        if (master == null) {
            Logger.error("Master unreachable, routing locally");
            engineHealthReporter.reportForwardToMasterResult("LOCAL", "MASTER_NULL");
            return fallbackToLocalRouting(ctx);
        }
        Logger.info("Forwarding request to master: {}, requestId: {}", master, request.getRequestId());
        URI uri = URI.create("http://" + master);
        return generalHttpNettyService.request(request, uri, "/rtp_llm/schedule", Response.class)
                .flatMap(resp -> {
                            engineHealthReporter.reportForwardToMasterResult(uri.getHost(), String.valueOf(resp.getCode()));
                            return ServerResponse.ok()
                                    .contentType(MediaType.APPLICATION_JSON)
                                    .bodyValue(resp);
                        }
                )
                .onErrorResume(e -> {
                    String errorCode = e instanceof EngineReadTimeoutException ? "TIMEOUT" : "CONNECT_FAILED";
                    Logger.error("[Fallback] Master unreachable, routing locally: {}, errorCode: {}", e.getMessage(), errorCode);
                    engineHealthReporter.reportForwardToMasterResult("LOCAL", errorCode);
                    return fallbackToLocalRouting(ctx);
                });
    }

    private Mono<ServerResponse> fallbackToLocalRouting(BalanceContext ctx) {
        return routeService.route(ctx)
                .flatMap(response -> handleRoutingResult(ctx, response))
                .onErrorResume(e -> {
                    Logger.error("[Fallback] Local routing failed", e);
                    Response errorResponse = Response.error(StrategyErrorType.NO_AVAILABLE_WORKER);
                    return ServerResponse.status(500)
                            .contentType(MediaType.APPLICATION_JSON)
                            .bodyValue(errorResponse);
                });
    }

    /**
     * Processes the routing response and builds the appropriate HTTP response.
     *
     * @param ctx the balance context
     * @param response the routing response
     * @return HTTP response based on routing success or failure
     */
    private Mono<ServerResponse> handleRoutingResult(BalanceContext ctx, Response response) {

        response.setRealMasterHost(lbStatusConsistencyService.getMasterHostIpPort());

        if (response.isSuccess()) {
            return json(200, response);
        }
        Logger.error("Routing failed with error code: {}", response.getErrorMessage());
        ctx.setSuccess(false);
        ctx.setErrorMessage("error_code:" + response.getErrorMessage());
        return json(500, response);
    }

    private Mono<ServerResponse> json(int status, Object body) {
        return ServerResponse.status(status)
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(body);
    }

    /**
     * Handles global request errors.
     *
     * @param ctx the balance context
     * @param throwable the error that occurred
     * @return error response
     */
    private Mono<ServerResponse> handleRequestError(BalanceContext ctx, Throwable throwable) {
        Logger.error("Request processing error", throwable);
        ctx.setSuccess(false);
        ctx.setErrorMessage(throwable.getMessage());

        return ServerResponse.status(500)
                .contentType(MediaType.APPLICATION_JSON)
                .body(Mono.just(throwable.getMessage()), String.class);
    }

    /**
     * Finalizes the request context by reporting metrics.
     *
     * @param ctx the balance context to finalize
     */
    private void finalizeRequestContext(BalanceContext ctx) {
        engineHealthReporter.reportBalancingService(ctx);
        logPvRecord(ctx);
    }

    /**
     * Logs the PV record with appropriate log level based on success status.
     *
     * @param ctx the balance context containing PV log data
     */
    private void logPvRecord(BalanceContext ctx) {
        new PvLogData(ctx).emit();
    }

    /**
     * Finalizes the batch schedule context by reporting metrics and writing the PV log.
     *
     * @param bctx the batch schedule context to finalize
     */
    private void finalizeBatchContext(BatchScheduleContext bctx) {
        engineHealthReporter.reportBatchSchedule(bctx);
        new BatchPvLogData(bctx).emit();
    }

}
