package org.flexlb.httpserver;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.balance.LoadBalanceWrapper;
import org.flexlb.consistency.LBStatusConsistencyService;
import org.flexlb.dao.RequestContext;
import org.flexlb.dao.loadbalance.LogLevelUpdateRequest;
import org.flexlb.dao.loadbalance.MasterRequest;
import org.flexlb.dao.loadbalance.MasterResponse;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.pv.PvLogData;
import org.flexlb.domain.balance.BalanceContext;
import org.flexlb.domain.consistency.MasterChangeNotifyReq;
import org.flexlb.domain.consistency.MasterChangeNotifyResp;
import org.flexlb.domain.consistency.SyncLBStatusReq;
import org.flexlb.domain.consistency.SyncLBStatusResp;
import org.flexlb.listener.OnlineListener;
import org.flexlb.listener.ShutdownListener;
import org.flexlb.service.grace.GracefulShutdownService;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.trace.WhaleSpanUtils;
import org.flexlb.transport.GeneralHttpNettyService;
import org.flexlb.util.HttpRequestUtils;
import org.flexlb.util.IdUtils;
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
    private final LoadBalanceWrapper loadBalanceWrapper;
    private final LBStatusConsistencyService lbStatusConsistencyService;
    private final EngineHealthReporter engineHealthReporter;
    private static final long FORWARD_TIMEOUT_MS = 300;

    public HttpMasterLoadBalanceServer(GeneralHttpNettyService generalHttpNettyService,
                                       LoadBalanceWrapper loadBalanceWrapper,
                                       LBStatusConsistencyService lbStatusConsistencyService,
                                       AppStateHookServer appStateHookServer, EngineHealthReporter engineHealthReporter) {
        this.generalHttpNettyService = generalHttpNettyService;
        this.loadBalanceWrapper = loadBalanceWrapper;
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
                        this::selectRole)
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
                    MasterResponse timeoutResponse = MasterResponse.code(StrategyErrorType.CONNECT_TIMEOUT.getErrorCode());
                    timeoutResponse.setRealMasterHost(uri.getHost());
                    return ServerResponse.status(504)
                            .contentType(MediaType.APPLICATION_JSON)
                            .bodyValue(timeoutResponse);
                })
                .onErrorResume(e -> {
                    LoggingUtils.info("Failed to forward request to master: {}", e.getMessage());
                    MasterResponse errorResponse = MasterResponse.code(StrategyErrorType.CONNECT_FAILED.getErrorCode());
                    errorResponse.setRealMasterHost(uri.getHost());
                    return ServerResponse.status(502)
                            .contentType(MediaType.APPLICATION_JSON)
                            .bodyValue(errorResponse);
                });
    }

    public Mono<ServerResponse> selectRole(ServerRequest request) {
        BalanceContext balanceContext = new BalanceContext();
        RequestContext ctx = buildRequestContext(request);
        balanceContext.setRequestContext(ctx);
        whaleSpanUtils.buildTraceSpan(ctx);
        return request.bodyToMono(MasterRequest.class)
                .flatMap((Function<MasterRequest, Mono<ServerResponse>>) req -> {
                    balanceContext.setMasterRequest(req);
                    if (!lbStatusConsistencyService.isMaster()) {
                        String master = lbStatusConsistencyService.getMasterHostIpPort();
                        ctx.getSpan().addEvent("forward_to_master: " + master);
                        URI uri = URI.create("http://" + master);
                        return forward2Master(uri, req);
                    }
                    // role list
                    MasterResponse result = loadBalanceWrapper.selectEngineWorker(balanceContext);
                    result.setRealMasterHost(lbStatusConsistencyService.getMasterHostIpPort());
                    if (result.isSuccess()) {
                        balanceContext.setPvLogData(PvLogData.success(req, result));
                        return ServerResponse.ok()
                                .contentType(MediaType.APPLICATION_JSON)
                                .body(Mono.just(result), MasterResponse.class);
                    } else {
                        LoggingUtils.error("selectBestWorker error_code:{}", result.getErrorCode());
                        balanceContext.setSuccess(false);
                        balanceContext.setPvLogData(PvLogData.error(req, "error_code:" + result.getErrorCode()));
                        return ServerResponse.status(500)
                                .contentType(MediaType.APPLICATION_JSON)
                                .body(Mono.just(result.getErrorCode()), String.class);
                    }
                }).onErrorResume(e -> {
                    LoggingUtils.error("selectBestWorker error", e);
                    ctx.getSpan().addEvent("selectBestWorker error");
                    balanceContext.setSuccess(false);
                    MasterRequest req = balanceContext.getMasterRequest();
                    balanceContext.setPvLogData(PvLogData.error(req, e.getMessage()));
                    return ServerResponse.status(500)
                            .contentType(MediaType.APPLICATION_JSON)
                            .body(Mono.just(e.getMessage()), String.class);
                })
                .doFinally(s -> {
                    // 汇报监控
                    engineHealthReporter.reportBalancingService(balanceContext);
                    // 记录PV日志
                    PvLogData pvLogData = balanceContext.getPvLogData();
                    //关闭span确保可以上报
                    ctx.getSpan().endSpan();
                    if (pvLogData != null) {
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

    private RequestContext buildRequestContext(ServerRequest request) {
        RequestContext ctx = new RequestContext();
        ServerRequest.Headers httpHeaders = request.headers();
        for (Map.Entry<String, List<String>> entry : httpHeaders.asHttpHeaders().entrySet()) {
            String headerName = entry.getKey();
            List<String> values = entry.getValue();
            if (values == null || values.isEmpty()) {
                continue;
            }
            String headerValue = values.getFirst();
            String lowerCaseHeaderName = headerName.toLowerCase();
            BiConsumer<RequestContext, String> processor = HttpRequestUtils.HEADER_PROCESSORS.get(lowerCaseHeaderName);
            if (processor != null) {
                processor.accept(ctx, headerValue);
            }
        }
        ctx.setRequestId(IdUtils.fastUuid());
        return ctx;
    }
}