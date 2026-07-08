package org.flexlb.httpserver;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.scheduler.FlexlbBatchScheduler;
import org.flexlb.balance.scheduler.QueueManager;
import org.flexlb.config.ConfigService;
import org.flexlb.config.TrafficPolicyConfig;
import org.flexlb.consistency.LBStatusConsistencyService;
import org.flexlb.dao.loadbalance.LogLevelUpdateRequest;
import org.flexlb.dao.loadbalance.QueueSnapshotResponse;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.domain.consistency.MasterChangeNotifyReq;
import org.flexlb.domain.consistency.MasterChangeNotifyResp;
import org.flexlb.domain.consistency.SyncLBStatusReq;
import org.flexlb.domain.consistency.SyncLBStatusResp;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.flexlb.sync.status.ModelWorkerStatus;
import org.flexlb.sync.synchronizer.MasterEngineSynchronizer;
import org.flexlb.util.Logger;
import org.springframework.context.annotation.Bean;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.server.RouterFunction;
import org.springframework.web.reactive.function.server.ServerRequest;
import org.springframework.web.reactive.function.server.ServerResponse;
import reactor.core.publisher.Mono;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

import static org.springframework.web.reactive.function.server.RequestPredicates.accept;
import static org.springframework.web.reactive.function.server.RouterFunctions.route;

@Component
public class HttpLoadBalanceServer {
    private final LBStatusConsistencyService lbStatusConsistencyService;
    private final QueueManager queueManager;
    private final ConfigService configService;
    private final FlexlbBatchScheduler batchScheduler;
    private final EndpointRegistry endpointRegistry;
    private final MasterEngineSynchronizer masterEngineSynchronizer;

    public HttpLoadBalanceServer(LBStatusConsistencyService lbStatusConsistencyService,
                                 QueueManager queueManager,
                                 ConfigService configService,
                                 FlexlbBatchScheduler batchScheduler,
                                 EndpointRegistry endpointRegistry,
                                 @org.springframework.beans.factory.annotation.Autowired(required = false)
                                 MasterEngineSynchronizer masterEngineSynchronizer) {
        this.lbStatusConsistencyService = lbStatusConsistencyService;
        this.queueManager = queueManager;
        this.configService = configService;
        this.batchScheduler = batchScheduler;
        this.endpointRegistry = endpointRegistry;
        this.masterEngineSynchronizer = masterEngineSynchronizer;
    }

    @Bean
    public RouterFunction<ServerResponse> loadBalancePrefill() {
        return route()
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
                .GET("/rtp_llm/inflight_status", accept(MediaType.APPLICATION_JSON),
                        this::inflightStatus)
                .build();
    }

    private Mono<ServerResponse> debugMode(ServerRequest serverRequest) {
        return serverRequest.bodyToMono(LogLevelUpdateRequest.class)
                .flatMap(logLevelUpdateRequest -> {
                    Logger.setLevel(logLevelUpdateRequest.getLogLevel());
                    return ServerResponse.ok()
                            .contentType(MediaType.APPLICATION_JSON)
                            .body(Mono.just("Success! logLevel=" + Logger.getLevel()), String.class);
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

    private Map<String, Response.WorkerRoleSummary> buildWorkerSummary() {
        ModelWorkerStatus modelStatus = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS;
        Map<String, Response.WorkerRoleSummary> summary = new LinkedHashMap<>();
        for (RoleType role : RoleType.values()) {
            Map<String, WorkerStatus> statusMap = modelStatus.getRoleStatusMap(role);
            if (statusMap == null || statusMap.isEmpty()) {
                continue;
            }
            Response.WorkerRoleSummary rs = new Response.WorkerRoleSummary();
            rs.setDiscovered(statusMap.size());
            for (WorkerStatus ws : statusMap.values()) {
                if (ws.isAlive()) {
                    rs.setAlive(rs.getAlive() + 1);
                }
            }
            summary.put(role.getCode(), rs);
        }
        return summary.isEmpty() ? null : summary;
    }

    private Mono<ServerResponse> responseMasterInfo(ServerRequest request) {
        return request.bodyToMono(Request.class)
                .flatMap((Function<Request, Mono<ServerResponse>>) req -> {
                    Response result = new Response();
                    result.setRealMasterHost(lbStatusConsistencyService.getMasterHostIpPort());
                    result.setQueueLength(queueManager.getQueue().size());
                    result.setCode(200);
                    result.setSuccess(true);
                    result.setWorkerSummary(buildWorkerSummary());
                    result.setReady(masterEngineSynchronizer == null || masterEngineSynchronizer.isReady());
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

    public Mono<ServerResponse> inflightStatus(ServerRequest request) {
        try {
            Map<String, Object> result = new LinkedHashMap<>();
            result.put("scheduler_inflight", batchScheduler.getInflightSize());

            List<Map<String, Object>> prefillList = new ArrayList<>();
            for (Map.Entry<String, PrefillEndpoint> entry : endpointRegistry.getPrefillEndpoints().entrySet()) {
                Map<String, Object> ep = new LinkedHashMap<>();
                ep.put("ip_port", entry.getKey());
                ep.put("inflight_batches", entry.getValue().getInflightBatchCount());
                prefillList.add(ep);
            }
            result.put("prefill_endpoints", prefillList);

            List<Map<String, Object>> decodeList = new ArrayList<>();
            for (Map.Entry<String, DecodeEndpoint> entry : endpointRegistry.getDecodeEndpoints().entrySet()) {
                Map<String, Object> ep = new LinkedHashMap<>();
                ep.put("ip_port", entry.getKey());
                ep.put("inflight_requests", entry.getValue().getInflightCount());
                decodeList.add(ep);
            }
            result.put("decode_endpoints", decodeList);

            return ServerResponse.ok()
                    .contentType(MediaType.APPLICATION_JSON)
                    .body(Mono.just(result), Map.class);
        } catch (Exception e) {
            Logger.error("inflightStatus error", e);
            return ServerResponse.status(500)
                    .contentType(MediaType.APPLICATION_JSON)
                    .body(Mono.just(e.getMessage()), String.class);
        }
    }
}
