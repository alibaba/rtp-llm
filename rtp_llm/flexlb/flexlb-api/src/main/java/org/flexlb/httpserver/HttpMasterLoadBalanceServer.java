package org.flexlb.httpserver;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.consistency.LBStatusConsistencyService;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.LogLevelUpdateRequest;
import org.flexlb.dao.loadbalance.MasterRequest;
import org.flexlb.dao.loadbalance.MasterResponse;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.pv.PvLogData;
import org.flexlb.domain.consistency.MasterChangeNotifyReq;
import org.flexlb.domain.consistency.MasterChangeNotifyResp;
import org.flexlb.domain.consistency.SyncLBStatusReq;
import org.flexlb.domain.consistency.SyncLBStatusResp;
import org.flexlb.listener.OnlineListener;
import org.flexlb.listener.ShutdownListener;
import org.flexlb.service.RouteService;
import org.flexlb.service.grace.GracefulShutdownService;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.trace.WhaleSpanUtils;
import org.flexlb.transport.GeneralHttpNettyService;
import org.flexlb.util.HttpRequestUtils;
import org.flexlb.util.JsonUtils;
import org.flexlb.util.LoggingUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Bean;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.server.RouterFunction;
import org.springframework.web.reactive.function.server.ServerRequest;
import org.springframework.web.reactive.function.server.ServerResponse;
import reactor.core.publisher.Mono;

import java.net.URI;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeoutException;
import java.util.function.BiConsumer;
import java.util.function.Function;

import static org.springframework.web.reactive.function.server.RequestPredicates.accept;
import static org.springframework.web.reactive.function.server.RouterFunctions.route;

/**
 * @author saichen
 * description:
 * date: 2025/3/16
 */
@Slf4j
@Component
public class HttpMasterLoadBalanceServer implements ShutdownListener, OnlineListener {
    private static final Logger pvLogger = LoggerFactory.getLogger("pvLogger");
    private final GeneralHttpNettyService generalHttpNettyService;
    private final RouteService routeService;
    private final LBStatusConsistencyService lbStatusConsistencyService;
    private final EngineHealthReporter engineHealthReporter;
    private static final long FORWARD_TIMEOUT_MS = 300;

    public HttpMasterLoadBalanceServer(GeneralHttpNettyService generalHttpNettyService,
                                       RouteService routeService,
                                       LBStatusConsistencyService lbStatusConsistencyService,
                                       AppStateHookServer appStateHookServer, EngineHealthReporter engineHealthReporter) {
        this.generalHttpNettyService = generalHttpNettyService;
        this.routeService = routeService;
        this.lbStatusConsistencyService = lbStatusConsistencyService;
        this.engineHealthReporter = engineHealthReporter;
        appStateHookServer.addOnlineHandler(this);
        GracefulShutdownService.addShutdownListener(this);
    }

    @Autowired
    private WhaleSpanUtils whaleSpanUtils;

    @Bean
    public RouterFunction<ServerResponse> loadBalancePrefill() {
        return route()
                .POST("/rtp_llm/schedule", accept(MediaType.APPLICATION_JSON),
                        this::scheduleRequest)
                .POST("/rtp_llm/master", accept(MediaType.APPLICATION_JSON),
                        this::responseMasterIp)
                .POST("/rtp_llm/schedule_snapshot", accept(MediaType.APPLICATION_JSON),
                        this::dumpLBStatus)
                .POST("/rtp_llm/notify_master", accept(MediaType.APPLICATION_JSON),
                        this::notifyParticipant)
                .POST("/rtp_llm/update_log_level", accept(MediaType.APPLICATION_JSON),
                        this::debugMode)
                .build();
    }

    /**
     * Handles load balancing request scheduling.
     *
     * @param request the HTTP request containing the model inference request
     * @return a reactive response containing the load balancing result
     */
    public Mono<ServerResponse> scheduleRequest(ServerRequest request) {
        BalanceContext ctx = initializeRequestContext(request);

        return request.bodyToMono(MasterRequest.class)
                .flatMap(req -> {
                    ctx.setMasterRequest(req);

                    if (!lbStatusConsistencyService.isMaster()) {
                        return forwardRequestToMaster(ctx, req);
                    }

                    return routeService.route(ctx)
                            .flatMap(result -> handleRoutingResult(ctx, req, result))
                            .doOnCancel(() -> routeService.cancelRequest(ctx));
                })
                .onErrorResume(e -> handleRequestError(ctx, e))
                .doFinally(signal -> finalizeRequestContext(ctx));
    }

    private Mono<ServerResponse> debugMode(ServerRequest serverRequest) {
        return serverRequest.bodyToMono(LogLevelUpdateRequest.class)
                .flatMap(logLevelUpdateRequest -> {
                    LoggingUtils.setGlobalLogLevel(logLevelUpdateRequest.getLogLevel());
                    return ServerResponse.ok()
                            .contentType(MediaType.APPLICATION_JSON)
                            .body(Mono.just("Success! logLevel=" + LoggingUtils.getGlobalLogLevel()), String.class);
                }).onErrorResume(e -> {
                    LoggingUtils.error("update logLevel error", e);
                    return ServerResponse.status(500)
                            .contentType(MediaType.APPLICATION_JSON)
                            .body(Mono.just(e.getMessage()), String.class);
                });
    }

    private Mono<ServerResponse> responseMasterIp(ServerRequest request) {
        BalanceContext balanceContext = new BalanceContext();
        return request.bodyToMono(MasterRequest.class)
                .flatMap((Function<MasterRequest, Mono<ServerResponse>>) req -> {
                    balanceContext.setMasterRequest(req);
                    MasterResponse result = new MasterResponse();
                    result.setRealMasterHost(lbStatusConsistencyService.getMasterHostIpPort());
                    result.setCode(200);
                    result.setSuccess(true);
                    return ServerResponse.ok()
                            .contentType(MediaType.APPLICATION_JSON)
                            .body(Mono.just(result), MasterResponse.class);
                }).onErrorResume(e -> {
                    LoggingUtils.error("selectBestWorker error", e);
                    balanceContext.setSuccess(false);
                    return ServerResponse.status(500)
                            .contentType(MediaType.APPLICATION_JSON)
                            .body(Mono.just(e.getMessage()), String.class);
                });
    }

    private Mono<ServerResponse> forward2Master(URI uri, MasterRequest body) {
        return generalHttpNettyService.request(
                        body,
                        uri,
                        "/rtp_llm/schedule",
                        MasterResponse.class
                )
                .timeout(Duration.ofMillis(FORWARD_TIMEOUT_MS))
                .doOnSubscribe(sub -> LoggingUtils.info("[Forward] 开始异步发送请求"))
                .doOnSuccess(resp -> LoggingUtils.info("[Forward] 收到 Master 响应: {}", resp))
                .doOnError(e -> LoggingUtils.error("[Forward] 转发请求失败", e))
                .flatMap(masterResponse -> ServerResponse.ok()
                        .contentType(MediaType.APPLICATION_JSON)
                        .bodyValue(masterResponse)
                )
                .onErrorResume(TimeoutException.class, e -> {
                    LoggingUtils.info("Request to master timed out");
                    MasterResponse timeoutResponse = MasterResponse.error(StrategyErrorType.CONNECT_TIMEOUT);
                    timeoutResponse.setRealMasterHost(uri.getHost());
                    return ServerResponse.status(504)
                            .contentType(MediaType.APPLICATION_JSON)
                            .bodyValue(timeoutResponse);
                })
                .onErrorResume(e -> {
                    LoggingUtils.info("Failed to forward request to master: {}", e.getMessage());
                    MasterResponse errorResponse = MasterResponse.error(StrategyErrorType.CONNECT_FAILED);
                    errorResponse.setRealMasterHost(uri.getHost());
                    return ServerResponse.status(502)
                            .contentType(MediaType.APPLICATION_JSON)
                            .bodyValue(errorResponse);
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
                    LoggingUtils.error("notifyParticipant error", e);
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
                    LoggingUtils.error("dumpLBStatus error", e);
                    return ServerResponse.status(500)
                            .contentType(MediaType.APPLICATION_JSON)
                            .body(Mono.just(e.getMessage()), String.class);
                });
    }

    @Override
    public void beforeShutdown() {
        try {
            LoggingUtils.warn("handle beforeShutdown.");
            lbStatusConsistencyService.offline();
        } catch (Throwable throwable) {
            LoggingUtils.error("handle beforeShutdown error", throwable);
        }
    }

    @Override
    public void afterStartUp() {
        LoggingUtils.warn("handle afterStartUp.");
        lbStatusConsistencyService.start();
    }

    @Override
    public int priority() {
        return 3;
    }

    /**
     * Initializes the balance context with HTTP headers and trace span.
     *
     * @param request the HTTP request
     * @return initialized balance context
     */
    private BalanceContext initializeRequestContext(ServerRequest request) {
        BalanceContext ctx = buildBalanceContext(request);
        whaleSpanUtils.buildTraceSpan(ctx);
        return ctx;
    }

    /**
     * Forwards the request to the active master node.
     *
     * @param ctx the balance context
     * @param request the master request to forward
     * @return response from the master node
     */
    private Mono<ServerResponse> forwardRequestToMaster(BalanceContext ctx, MasterRequest request) {
        String master = lbStatusConsistencyService.getMasterHostIpPort();
        ctx.getSpan().addEvent("forward_to_master: " + master);
        URI uri = URI.create("http://" + master);
        return forward2Master(uri, request);
    }

    /**
     * Processes the routing result and builds the appropriate HTTP response.
     *
     * @param ctx the balance context
     * @param request the original master request
     * @param result the routing result
     * @return HTTP response based on routing success or failure
     */
    private Mono<ServerResponse> handleRoutingResult(BalanceContext ctx,
                                                     MasterRequest request,
                                                     MasterResponse result) {
        result.setRealMasterHost(lbStatusConsistencyService.getMasterHostIpPort());

        if (result.isSuccess()) {
            ctx.setPvLogData(PvLogData.success(request, result));
            return buildSuccessResponse(result);
        } else {
            LoggingUtils.error("Routing failed with error code: {}", result.getErrorMessage());
            ctx.setSuccess(false);
            ctx.setPvLogData(PvLogData.error(request, "error_code:" + result.getErrorMessage()));
            return buildErrorResponse(result);
        }
    }

    /**
     * Builds a successful HTTP response.
     *
     * @param result the master response containing the result
     * @return successful HTTP response
     */
    private Mono<ServerResponse> buildSuccessResponse(MasterResponse result) {
        return ServerResponse.ok()
                .contentType(MediaType.APPLICATION_JSON)
                .body(Mono.just(result), MasterResponse.class);
    }

    /**
     * Builds an error HTTP response.
     *
     * @param result the master response containing the error
     * @return error HTTP response
     */
    private Mono<ServerResponse> buildErrorResponse(MasterResponse result) {
        return ServerResponse.status(500)
                .contentType(MediaType.APPLICATION_JSON)
                .body(Mono.just(result), MasterResponse.class);
    }

    /**
     * Handles global request errors.
     *
     * @param ctx the balance context
     * @param throwable the error that occurred
     * @return error response
     */
    private Mono<ServerResponse> handleRequestError(BalanceContext ctx, Throwable throwable) {
        LoggingUtils.error("Request processing error", throwable);
        ctx.getSpan().addEvent("request_processing_error");
        ctx.setSuccess(false);

        MasterRequest request = ctx.getMasterRequest();
        ctx.setPvLogData(PvLogData.error(request, throwable.getMessage()));

        return ServerResponse.status(500)
                .contentType(MediaType.APPLICATION_JSON)
                .body(Mono.just(throwable.getMessage()), String.class);
    }

    /**
     * Finalizes the request context by reporting metrics and closing the trace span.
     *
     * @param ctx the balance context to finalize
     */
    private void finalizeRequestContext(BalanceContext ctx) {
        engineHealthReporter.reportBalancingService(ctx);
        ctx.getSpan().endSpan();
        logPvRecord(ctx);
    }

    /**
     * Logs the PV record with appropriate log level based on success status.
     *
     * @param ctx the balance context containing PV log data
     */
    private void logPvRecord(BalanceContext ctx) {
        PvLogData pvLogData = ctx.getPvLogData();
        if (pvLogData == null) {
            return;
        }

        try {
            String jsonLog = JsonUtils.toStringOrEmpty(pvLogData);
            if (pvLogData.isSuccess()) {
                pvLogger.info(jsonLog);
            } else {
                pvLogger.error(jsonLog);
            }
        } catch (Exception ex) {
            LoggingUtils.error("Failed to serialize PV log data", ex);
        }
    }

    private BalanceContext buildBalanceContext(ServerRequest request) {
        BalanceContext ctx = new BalanceContext();
        ServerRequest.Headers httpHeaders = request.headers();
        for (Map.Entry<String, List<String>> entry : httpHeaders.asHttpHeaders().entrySet()) {
            String headerName = entry.getKey();
            List<String> values = entry.getValue();
            if (values == null || values.isEmpty()) {
                continue;
            }
            String headerValue = values.getFirst();
            String lowerCaseHeaderName = headerName.toLowerCase();
            BiConsumer<BalanceContext, String> processor = HttpRequestUtils.HEADER_PROCESSORS.get(lowerCaseHeaderName);
            if (processor != null) {
                processor.accept(ctx, headerValue);
            }
        }
        return ctx;
    }
}