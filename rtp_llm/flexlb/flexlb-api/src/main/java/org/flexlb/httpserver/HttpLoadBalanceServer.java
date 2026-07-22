package org.flexlb.httpserver;

import org.flexlb.balance.scheduler.QueueManager;
import org.flexlb.consistency.LBStatusConsistencyService;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.BatchScheduleContext;
import org.flexlb.dao.loadbalance.BatchScheduleRequest;
import org.flexlb.dao.loadbalance.BatchScheduleResponse;
import org.flexlb.dao.loadbalance.BatchScheduleTarget;
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
import org.flexlb.dispatcher.FePool;
import org.flexlb.exception.BatchScheduleTransportException;
import org.flexlb.exception.EngineReadTimeoutException;
import org.flexlb.service.BatchScheduleCoordinator;
import org.flexlb.service.RouteService;
import org.flexlb.service.grace.ActiveRequestCounter;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.transport.GeneralHttpNettyService;
import org.flexlb.util.Logger;
import org.flexlb.util.RateLimitedWarn;
import org.springframework.beans.factory.ObjectProvider;
import org.springframework.context.annotation.Bean;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.server.RouterFunction;
import org.springframework.web.reactive.function.server.ServerRequest;
import org.springframework.web.reactive.function.server.ServerResponse;
import org.springframework.web.server.ServerWebInputException;
import reactor.core.publisher.Mono;

import java.net.URI;
import java.util.concurrent.TimeUnit;
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
    /**
     * The dispatcher's FE round-robin pool, present only on a node that runs the dispatcher
     * ({@code dispatch.fe-pool-service-id} set, gating the {@code @ConditionalOnProperty} bean).
     * On the elected master this is the single fleet-wide FE cursor whose picks are stamped into
     * {@code /batch_schedule} responses; on a master node without the dispatcher it is absent and
     * responses carry no {@code fe_url}. Injected via {@link ObjectProvider} so the server bean
     * still constructs when the pool is not wired.
     */
    private final ObjectProvider<FePool> fePoolProvider;
    /** An empty FE snapshot fails FE assignment on every batch request; cap the WARN at 1/s. */
    private final RateLimitedWarn feAssignWarn = new RateLimitedWarn(1, TimeUnit.SECONDS);

    public HttpLoadBalanceServer(GeneralHttpNettyService generalHttpNettyService,
                                 RouteService routeService,
                                 LBStatusConsistencyService lbStatusConsistencyService,
                                 EngineHealthReporter engineHealthReporter,
                                 QueueManager queueManager,
                                 ActiveRequestCounter activeRequestCounter,
                                 BatchScheduleCoordinator batchScheduleCoordinator,
                                 ObjectProvider<FePool> fePoolProvider) {
        this.generalHttpNettyService = generalHttpNettyService;
        this.routeService = routeService;
        this.lbStatusConsistencyService = lbStatusConsistencyService;
        this.engineHealthReporter = engineHealthReporter;
        this.queueManager = queueManager;
        this.activeRequestCounter = activeRequestCounter;
        this.batchScheduleCoordinator = batchScheduleCoordinator;
        this.fePoolProvider = fePoolProvider;
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
                .switchIfEmpty(Mono.error(new ServerWebInputException("empty request body")))
                .flatMap(batchRequest -> {
                    bctx.setBatchRequest(batchRequest);
                    return Mono.using(
                            activeRequestCounter::acquire,
                            ignored -> processBatchScheduleRequest(bctx),
                            ActiveRequestCounter.RequestToken::close);
                })
                .onErrorResume(e -> {
                    Logger.error("Batch schedule request processing error", e);
                    return batchError(bctx, errorTypeOf(bctx), e);
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
                    assignFeUrls(response);
                    return json(statusOf(response), response);
                })
                .onErrorResume(BatchScheduleTransportException.class,
                        e -> batchError(bctx, StrategyErrorType.NO_AVAILABLE_WORKER, e));
    }

    /**
     * Stamps each target's {@code fe_url} from the local {@link FePool} so one master-side cursor
     * drives FE selection fleet-wide, replacing each dispatcher round-robining its own pool with
     * a private (colliding) cursor.
     *
     * <p>Two guards keep this an optimization that never breaks correctness:
     * <ol>
     *   <li>Only when this node resolved the batch locally — it is the elected master, or
     *       consistency is off. A slave that merely forwarded to the master already holds the
     *       master's assignment in the response; re-stamping it here with the slave's own cursor
     *       would reintroduce the collision this feature removes.</li>
     *   <li>Only when the {@link FePool} bean exists (this master node also runs the dispatcher).
     *       Absent it, targets keep {@code fe_url == null} and the dispatcher fails those chunks
     *       visibly — FE has exactly one source, so a misconfigured master (no FE view) surfaces
     *       as chunk failures rather than a silent split onto per-instance cursors.</li>
     * </ol>
     * An empty FE snapshot (pool throws) is swallowed the same way — leaving {@code fe_url} null —
     * so BE assignment (already computed) still returns; the affected chunks then fail visibly in
     * the dispatcher rather than aborting the whole schedule response.
     */
    private void assignFeUrls(BatchScheduleResponse response) {
        if (!response.isSuccess() || response.getServerStatus() == null) {
            return;
        }
        boolean resolvedLocally = !lbStatusConsistencyService.isNeedConsistency()
                || lbStatusConsistencyService.isMaster();
        if (!resolvedLocally) {
            return;
        }
        FePool pool = fePoolProvider.getIfAvailable();
        if (pool == null) {
            return;
        }
        try {
            for (BatchScheduleTarget target : response.getServerStatus()) {
                target.setFeUrl(pool.next());
            }
        } catch (RuntimeException e) {
            feAssignWarn.warn("[BatchSchedule] FE assignment skipped, dispatcher will fall back "
                    + "to local pool: {}", e.toString());
        }
    }

    /**
     * HTTP status for a business response. INVALID_REQUEST rejections (batch_count out of range,
     * multi-role deployment, null request) are deterministic client errors → 400; every other
     * failure (NO_AVAILABLE_WORKER etc.) is a server-side condition and stays 500.
     *
     * <p>The body's error type is the single source of truth for the status, because a slave
     * forwarding to the master reconstructs the response from the body alone
     * ({@code BatchScheduleCoordinator#forwardToMaster}) and re-derives the status here. Deriving
     * the two independently would let a forwarded 500 arrive at the caller as a 400 — a retryable
     * server fault reported as "your request is bad".
     */
    private static int statusOf(BatchScheduleResponse response) {
        if (response.isSuccess()) {
            return 200;
        }
        return response.getCode() == StrategyErrorType.INVALID_REQUEST.getErrorCode() ? 400 : 500;
    }

    /**
     * Classifies a failure by the stage it escaped, not by its exception type: everything up to
     * and including the body decode can only fail on what the caller sent, while the same
     * exception type raised once scheduling is under way (an {@link IllegalArgumentException} from
     * a malformed elected-master address, say) is a server fault that must alert and be retried.
     * The decoded request is the marker — the flatMap stamps it before anything else can fail.
     */
    private static StrategyErrorType errorTypeOf(BatchScheduleContext bctx) {
        return bctx.getBatchRequest() == null
                ? StrategyErrorType.INVALID_REQUEST
                : StrategyErrorType.NO_AVAILABLE_WORKER;
    }

    /** Builds the error response and stamps it on the context; pv success/error derive from it. */
    private Mono<ServerResponse> batchError(BatchScheduleContext bctx, StrategyErrorType type, Throwable e) {
        BatchScheduleResponse errorResponse = BatchScheduleResponse.error(type, e.getMessage());
        bctx.setBatchResponse(errorResponse);
        return json(statusOf(errorResponse), errorResponse);
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
