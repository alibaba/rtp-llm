package org.flexlb.httpserver;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.scheduler.RequestScheduler;
import org.flexlb.balance.scheduler.RequestState;
import org.flexlb.config.ConfigService;
import org.flexlb.config.TrafficPolicyConfig;
import org.flexlb.consistency.LBStatusConsistencyService;
import org.flexlb.dao.loadbalance.BatchScheduleRequest;
import org.flexlb.dao.loadbalance.BatchScheduleResponse;
import org.flexlb.dao.loadbalance.LogLevelUpdateRequest;
import org.flexlb.dao.loadbalance.QueueSnapshotResponse;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.domain.consistency.MasterChangeNotifyReq;
import org.flexlb.domain.consistency.MasterChangeNotifyResp;
import org.flexlb.domain.consistency.SyncLBStatusReq;
import org.flexlb.domain.consistency.SyncLBStatusResp;
import org.flexlb.service.BatchScheduleCoordinator;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.status.WorkerDirectory;
import org.flexlb.sync.synchronizer.MasterEngineSynchronizer;
import org.flexlb.util.JsonUtils;
import org.flexlb.util.Logger;
import org.springframework.context.annotation.Bean;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.server.RouterFunction;
import org.springframework.web.reactive.function.server.ServerRequest;
import org.springframework.web.reactive.function.server.ServerResponse;
import org.springframework.web.server.ServerWebInputException;
import reactor.core.publisher.Mono;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Stream;

import static org.springframework.web.reactive.function.server.RequestPredicates.accept;
import static org.springframework.web.reactive.function.server.RouterFunctions.route;

@Component
public class HttpLoadBalanceServer {
    private static final Path SCHEDULER_SNAPSHOT_DIR =
            Paths.get("/tmp/flexlb-scheduler-snapshots");
    private static final String SCHEDULER_SNAPSHOT_PREFIX = "scheduler-snapshot-";
    private static final int MAX_SNAPSHOT_FILES = 10;

    private final LBStatusConsistencyService lbStatusConsistencyService;
    private final ConfigService configService;
    private final RequestScheduler requestScheduler;
    private final EndpointRegistry endpointRegistry;
    private final WorkerDirectory workerDirectory;
    private final MasterEngineSynchronizer masterEngineSynchronizer;
    private final ServerScheduleLatencyRecorder serverLatencyRecorder;

    private final BatchScheduleCoordinator batchScheduleCoordinator;
    private final EngineHealthReporter engineHealthReporter;

    public HttpLoadBalanceServer(LBStatusConsistencyService lbStatusConsistencyService,
                                 ConfigService configService,
                                 RequestScheduler requestScheduler,
                                 EndpointRegistry endpointRegistry,
                                 WorkerDirectory workerDirectory,
                                 @org.springframework.beans.factory.annotation.Autowired(required = false)
                                 MasterEngineSynchronizer masterEngineSynchronizer,
                                 ServerScheduleLatencyRecorder serverLatencyRecorder,
                                 BatchScheduleCoordinator batchScheduleCoordinator,
                                 EngineHealthReporter engineHealthReporter) {
        this.lbStatusConsistencyService = lbStatusConsistencyService;
        this.configService = configService;
        this.requestScheduler = requestScheduler;
        this.endpointRegistry = endpointRegistry;
        this.workerDirectory = workerDirectory;
        this.masterEngineSynchronizer = masterEngineSynchronizer;
        this.serverLatencyRecorder = serverLatencyRecorder;
        this.batchScheduleCoordinator = batchScheduleCoordinator;
        this.engineHealthReporter = engineHealthReporter;
    }

    @Bean
    public RouterFunction<ServerResponse> loadBalancePrefill() {
        return route()
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
                .POST("/rtp_llm/update_traffic_policy", accept(MediaType.APPLICATION_JSON),
                        this::updateTrafficPolicy)
                .GET("/rtp_llm/queue_snapshot", accept(MediaType.APPLICATION_JSON),
                        this::queueSnapshot)
                .GET("/rtp_llm/inflight_status", accept(MediaType.APPLICATION_JSON),
                        this::inflightStatus)
                .GET("/rtp_llm/server_latency", this::serverLatency)
                .POST("/rtp_llm/server_latency/reset", this::resetServerLatency)
                .build();
    }

    public Mono<ServerResponse> batchScheduleRequest(ServerRequest request) {
        long start = System.currentTimeMillis();
        return request.bodyToMono(BatchScheduleRequest.class)
                .switchIfEmpty(Mono.error(new ServerWebInputException("empty request body")))
                .flatMap(batch -> batchScheduleCoordinator.schedule(batch)
                        .onErrorResume(error -> {
                            Logger.error("Batch scheduling failed", error);
                            return Mono.just(BatchScheduleResponse.error(
                                    StrategyErrorType.NO_AVAILABLE_WORKER, "batch scheduling failed"));
                        })
                        .doOnNext(response -> engineHealthReporter.reportBatchSchedule(batch, response, start)))
                .onErrorResume(error -> {
                    Logger.error("Batch schedule request failed", error);
                    boolean invalid = error instanceof ServerWebInputException;
                    return Mono.just(BatchScheduleResponse.error(invalid
                            ? StrategyErrorType.INVALID_REQUEST : StrategyErrorType.NO_AVAILABLE_WORKER,
                            invalid ? "invalid batch schedule request" : "batch scheduling failed"));
                })
                .flatMap(response -> {
                    int status = response.isSuccess() ? 200
                            : response.getCode() == StrategyErrorType.INVALID_REQUEST.getErrorCode() ? 400 : 500;
                    return json(status, response);
                });
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
        Map<String, Response.WorkerRoleSummary> summary = new LinkedHashMap<>();
        for (RoleType role : RoleType.values()) {
            Map<String, WorkerStatus> statusMap = workerDirectory.statusSnapshot(role);
            int embeddingCount = masterEngineSynchronizer == null ? 0
                    : masterEngineSynchronizer.embeddingWorkerSnapshot(role).size();
            if (statusMap.isEmpty() && embeddingCount == 0) {
                continue;
            }
            Response.WorkerRoleSummary rs = new Response.WorkerRoleSummary();
            rs.setDiscovered(statusMap.size() + embeddingCount);
            rs.setAlive(embeddingCount);
            for (WorkerStatus ws : statusMap.values()) {
                if (ws.pollHealth().reportedAlive()) {
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
                    result.setQueueLength(requestScheduler.getQueuedRequestCount());
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
                            .body(Mono.just(resp), MasterChangeNotifyResp.class);
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
            List<RequestState> snapshot =
                    requestScheduler.snapshotActiveRequests();
            QueueSnapshotResponse response = persistSchedulerSnapshot(snapshot);
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

    private QueueSnapshotResponse persistSchedulerSnapshot(
            List<RequestState> snapshot) throws IOException {
        Files.createDirectories(SCHEDULER_SNAPSHOT_DIR);
        cleanOldSchedulerSnapshots();

        long timestamp = System.currentTimeMillis();
        Path file = SCHEDULER_SNAPSHOT_DIR.resolve(
                SCHEDULER_SNAPSHOT_PREFIX + timestamp + ".json");
        Files.writeString(file, JsonUtils.toFormattedString(snapshot));
        return new QueueSnapshotResponse(
                file.toAbsolutePath().toString(), timestamp, snapshot.size());
    }

    private void cleanOldSchedulerSnapshots() throws IOException {
        List<Path> files;
        try (Stream<Path> entries = Files.list(SCHEDULER_SNAPSHOT_DIR)) {
            files = entries
                    .filter(path -> path.getFileName().toString()
                            .startsWith(SCHEDULER_SNAPSHOT_PREFIX))
                    .sorted()
                    .toList();
        }
        for (int index = 0; index <= files.size() - MAX_SNAPSHOT_FILES; index++) {
            Files.deleteIfExists(files.get(index));
        }
    }

    public Mono<ServerResponse> inflightStatus(ServerRequest request) {
        try {
            Map<String, Object> result = new LinkedHashMap<>();
            result.put("scheduler_inflight", requestScheduler.getInflightSize());
            result.put("decode_max_engine_requests",
                    configService.loadBalanceConfig().getRouter().getRoles()
                            .getDecode().getAvailability().getMaxEngineRequests());

            List<Map<String, Object>> prefillList = new ArrayList<>();
            for (Map.Entry<String, PrefillEndpoint> entry
                    : endpointRegistry.snapshotPrefillEndpoints().entrySet()) {
                PrefillEndpoint endpoint = entry.getValue();
                WorkerStatus.CacheIndexSnapshot cacheIndex =
                        endpoint.getStatus().cacheIndexSnapshot();
                CacheStatus cacheStatus = cacheIndex.cacheStatus();
                Map<String, Object> ep = new LinkedHashMap<>();
                ep.put("ip_port", entry.getKey());
                ep.put("inflight_batches", endpoint.getInflightBatchCount());
                ep.put("inflight_requests", endpoint.getLocallyOwnedRequestCount());
                ep.put("inflight_route_requests",
                        endpoint.getIndividuallyTrackedRequestCount());
                ep.put("cache_version",
                        cacheStatus == null ? -1L : cacheStatus.getVersion());
                ep.put("cache_indexed", cacheIndex.indexInitialized());
                ep.put("cache_indexed_version", cacheIndex.indexedVersion());
                ep.put("cache_key_size",
                        cacheStatus == null ? 0 : cacheStatus.getCacheKeySize());
                prefillList.add(ep);
            }
            result.put("prefill_endpoints", prefillList);

            List<Map<String, Object>> decodeList = new ArrayList<>();
            for (Map.Entry<String, DecodeEndpoint> entry
                    : endpointRegistry.snapshotDecodeEndpoints().entrySet()) {
                DecodeEndpoint.LayeredAdmissionView view =
                        entry.getValue().layeredAdmissionView();
                Map<String, Object> ep = new LinkedHashMap<>();
                ep.put("ip_port", entry.getKey());
                ep.put("reserved_total", view.reserved().size());
                ep.put("master_queued", view.queuedCount());
                ep.put("engine_may_have_seen",
                        Math.max(0, view.reserved().size() - view.queuedCount()));
                ep.put("confirmed_accepted", view.acceptedCount());
                ep.put("confirmed_running", view.runningCount());
                ep.put("total_load", view.routing().totalLoad());
                ep.put("engine_load", view.routing().engineLoad());
                ep.put("active_dispatch_permits", view.activeDispatchPermits());
                ep.put("engine_capacity_used", view.engineCapacityUsed());
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

    private Mono<ServerResponse> serverLatency(ServerRequest request) {
        return ServerResponse.ok()
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(serverLatencyRecorder.snapshot());
    }

    private Mono<ServerResponse> resetServerLatency(ServerRequest request) {
        serverLatencyRecorder.reset();
        return ServerResponse.ok()
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(Map.of("reset", true));
    }
    private Mono<ServerResponse> json(int status, Object body) {
        return ServerResponse.status(status)
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(body);
    }

}
