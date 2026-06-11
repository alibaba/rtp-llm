package org.flexlb.httpserver;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;
import org.flexlb.balance.scheduler.QueueManager;
import org.flexlb.config.ConfigService;
import org.flexlb.config.TrafficPolicyConfig;
import org.flexlb.consistency.LBStatusConsistencyService;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.LogLevelUpdateRequest;
import org.flexlb.dao.loadbalance.QueueSnapshotResponse;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.pv.PvLogData;
import org.flexlb.domain.consistency.MasterChangeNotifyReq;
import org.flexlb.domain.consistency.MasterChangeNotifyResp;
import org.flexlb.domain.consistency.SyncLBStatusReq;
import org.flexlb.domain.consistency.SyncLBStatusResp;
import org.flexlb.service.RouteService;
import org.flexlb.service.grace.ActiveRequestCounter;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.service.optimizer.OnlineOptimizerClient;
import org.flexlb.service.optimizer.OnlineOptimizerHooker;
import org.flexlb.transport.GeneralHttpNettyService;
import org.flexlb.util.JsonUtils;
import org.flexlb.util.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.context.annotation.Bean;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.server.RouterFunction;
import org.springframework.web.reactive.function.server.ServerRequest;
import org.springframework.web.reactive.function.server.ServerResponse;
import reactor.core.publisher.Mono;

import java.net.URI;
import java.util.concurrent.TimeoutException;
import java.util.function.Function;

import static org.springframework.web.reactive.function.server.RequestPredicates.accept;
import static org.springframework.web.reactive.function.server.RouterFunctions.route;

@Slf4j
@Component
public class HttpLoadBalanceServer {
    private static final org.slf4j.Logger pvLogger = LoggerFactory.getLogger("pvLogger");
    private static final String AUTHORIZATION_HEADER = "Authorization";
    private static final String X_API_KEY_HEADER = "X-Api-Key";
    private static final String API_KEY_HEADER = "Api-Key";
    private static final String BEARER_PREFIX = "Bearer ";
    private final GeneralHttpNettyService generalHttpNettyService;
    private final RouteService routeService;
    private final LBStatusConsistencyService lbStatusConsistencyService;
    private final EngineHealthReporter engineHealthReporter;
    private final QueueManager queueManager;
    private final ActiveRequestCounter activeRequestCounter;
    private final OnlineOptimizerHooker onlineOptimizerHooker;
    private final ConfigService configService;

    public HttpLoadBalanceServer(GeneralHttpNettyService generalHttpNettyService,
                                 RouteService routeService,
                                 LBStatusConsistencyService lbStatusConsistencyService,
                                 EngineHealthReporter engineHealthReporter,
                                 QueueManager queueManager,
                                 ActiveRequestCounter activeRequestCounter,
                                 OnlineOptimizerHooker onlineOptimizerHooker,
                                 ConfigService configService) {
        this.generalHttpNettyService = generalHttpNettyService;
        this.routeService = routeService;
        this.lbStatusConsistencyService = lbStatusConsistencyService;
        this.engineHealthReporter = engineHealthReporter;
        this.queueManager = queueManager;
        this.activeRequestCounter = activeRequestCounter;
        this.onlineOptimizerHooker = onlineOptimizerHooker;
        this.configService = configService;
    }

    @Bean
    public RouterFunction<ServerResponse> loadBalancePrefill() {
        return route()
                .POST("/rtp_llm/schedule", accept(MediaType.APPLICATION_JSON),
                        this::scheduleRequest)
                .POST("/rtp_llm/master/info", accept(MediaType.APPLICATION_JSON),
                        this::responseMasterInfo)
                .POST("/rtp_llm/schedule_snapshot", accept(MediaType.APPLICATION_JSON),
                        this::dumpLBStatus)
                .POST("/rtp_llm/notify_master", accept(MediaType.APPLICATION_JSON),
                        this::notifyParticipant)
                .POST("/rtp_llm/update_log_level", accept(MediaType.APPLICATION_JSON),
                        this::debugMode)
                .POST("/rtp_llm/update_traffic_policy", accept(MediaType.APPLICATION_JSON),
                        this::updateTrafficPolicy)
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
                    populateApiKeyFromHeaders(req, request);
                    ctx.setRequest(req);
                    return Mono.using(
                            activeRequestCounter::acquire,
                            ignored -> processScheduledRequest(ctx, req),
                            ActiveRequestCounter.RequestToken::close);
                })
                .onErrorResume(e -> handleRequestError(ctx, e))
                .doFinally(signal -> finalizeRequestContext(ctx));
    }

    private void populateApiKeyFromHeaders(Request req, ServerRequest serverRequest) {
        if (StringUtils.isNotBlank(req.getApiKey())) {
            return;
        }

        String apiKey = firstNonBlank(
                serverRequest.headers().firstHeader(X_API_KEY_HEADER),
                serverRequest.headers().firstHeader(API_KEY_HEADER),
                extractBearerToken(serverRequest.headers().firstHeader(AUTHORIZATION_HEADER)));
        req.setApiKey(apiKey);
    }

    private String extractBearerToken(String authorization) {
        if (StringUtils.isBlank(authorization) || !authorization.startsWith(BEARER_PREFIX)) {
            return null;
        }
        return authorization.substring(BEARER_PREFIX.length()).trim();
    }

    private String firstNonBlank(String... values) {
        for (String value : values) {
            if (StringUtils.isNotBlank(value)) {
                return value;
            }
        }
        return null;
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

    private Mono<ServerResponse> updateTrafficPolicy(ServerRequest serverRequest) {
        return serverRequest.bodyToMono(TrafficPolicyConfig.class)
                .flatMap(trafficPolicyConfig -> {
                    configService.updateTrafficPolicy(trafficPolicyConfig);
                    return ServerResponse.ok()
                            .contentType(MediaType.APPLICATION_JSON)
                            .body(Mono.just(trafficPolicyConfig), TrafficPolicyConfig.class);
                }).onErrorResume(e -> {
                    Logger.error("update traffic policy error", e);
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
        Logger.info("Forwarding request to master: {}, request: {}", master, request);
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
                    String errorCode = e instanceof TimeoutException ? "TIMEOUT" : "CONNECT_FAILED";
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
            fireTraceQuery(ctx);
            return buildSuccessResponse(response);
        } else {
            Logger.error("Routing failed with error code: {}", response.getErrorMessage());
            ctx.setSuccess(false);
            ctx.setErrorMessage("error_code:" + response.getErrorMessage());
            return buildErrorResponse(response);
        }
    }

    private void fireTraceQuery(BalanceContext ctx) {
        // Best-effort fire-and-forget. Any failure here MUST NOT propagate to the request path.
        try {
            OnlineOptimizerClient client = onlineOptimizerHooker.getClient();
            if (client == null) {
                return;
            }
            Request req = ctx.getRequest();
            if (req != null && req.getBlockCacheKeys() != null && !req.getBlockCacheKeys().isEmpty()) {
                client.traceQuery(req.getRequestId(), req.getBlockCacheKeys());
            }
        } catch (Throwable t) {
            Logger.warn("fireTraceQuery skipped due to error: {}", t.getMessage());
        }
    }

    /**
     * Builds a successful HTTP response.
     *
     * @param result the master response containing the result
     * @return successful HTTP response
     */
    private Mono<ServerResponse> buildSuccessResponse(Response result) {
        return ServerResponse.ok()
                .contentType(MediaType.APPLICATION_JSON)
                .body(Mono.just(result), Response.class);
    }

    /**
     * Builds an error HTTP response.
     *
     * @param result the master response containing the error
     * @return error HTTP response
     */
    private Mono<ServerResponse> buildErrorResponse(Response result) {
        return ServerResponse.status(500)
                .contentType(MediaType.APPLICATION_JSON)
                .body(Mono.just(result), Response.class);
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

        PvLogData pvLogData = new PvLogData(ctx);

        try {
            String jsonLog = JsonUtils.toStringOrEmpty(pvLogData);
            if (pvLogData.isSuccess()) {
                pvLogger.info(jsonLog);
            } else {
                pvLogger.error(jsonLog);
            }
        } catch (Exception ex) {
            Logger.error("Failed to serialize PV log data", ex);
        }
    }

}
