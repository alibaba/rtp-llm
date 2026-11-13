package org.flexlb.api.server;

import org.flexlb.api.ServerScheduleLatencyRecorder;
import org.flexlb.balance.queue.QueueSnapshotResponse;
import org.flexlb.balance.schedule.RequestRepository;
import org.flexlb.balance.schedule.RequestResult;
import org.flexlb.config.ConfigService;
import org.flexlb.config.routing.TrafficPolicyConfig;
import org.flexlb.config.service.RoleType;
import org.flexlb.master.MasterChangeNotifyReq;
import org.flexlb.master.MasterChangeNotifyResp;
import org.flexlb.master.MasterStatusService;
import org.flexlb.master.SyncLBStatusReq;
import org.flexlb.master.SyncLBStatusResp;
import org.flexlb.util.JsonUtils;
import org.flexlb.util.Logger;
import org.flexlb.util.log.LogLevelUpdateRequest;
import org.flexlb.worker.cache.CacheStatus;
import org.flexlb.worker.endpoint.DecodeEndpoint;
import org.flexlb.worker.endpoint.EndpointRegistry;
import org.flexlb.worker.endpoint.PrefillEndpoint;
import org.flexlb.worker.endpoint.Request;
import org.flexlb.worker.endpoint.Response;
import org.flexlb.worker.status.WorkerStatus;
import org.flexlb.worker.status.WorkerStatusSynchronizer;
import org.springframework.context.annotation.Bean;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.server.RouterFunction;
import org.springframework.web.reactive.function.server.ServerRequest;
import org.springframework.web.reactive.function.server.ServerResponse;
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

/**
 * Exposes HTTP health, configuration and scheduling diagnostics from the live service owners.
 */
@Component
public class HttpLoadBalanceServer {
    private static final Path SCHEDULER_SNAPSHOT_DIR = Paths.get("/tmp/flexlb-scheduler-snapshots");
    private static final String SCHEDULER_SNAPSHOT_PREFIX = "scheduler-snapshot-";

    private static final int MAX_SNAPSHOT_FILES = 10;

    private final MasterStatusService masterStatusService;
    private final ConfigService configService;
    private final RequestRepository requests;
    private final EndpointRegistry endpointRegistry;
    private final WorkerStatusSynchronizer workerStatusSynchronizer;
    private final ServerScheduleLatencyRecorder serverLatencyRecorder;

    private org.flexlb.balance.schedule.RequestScheduler scheduler;

    public HttpLoadBalanceServer(
            MasterStatusService masterStatusService,
            ConfigService configService,
            RequestRepository requests,
            EndpointRegistry endpointRegistry,
            @org.springframework.beans.factory.annotation.Autowired(required = false)
                    WorkerStatusSynchronizer workerStatusSynchronizer,
            ServerScheduleLatencyRecorder serverLatencyRecorder) {
        this.masterStatusService = masterStatusService;
        this.configService = configService;
        this.requests = requests;
        this.endpointRegistry = endpointRegistry;
        this.workerStatusSynchronizer = workerStatusSynchronizer;
        this.serverLatencyRecorder = serverLatencyRecorder;
    }

    @org.springframework.beans.factory.annotation.Autowired
    public void configureScheduling(org.flexlb.balance.schedule.RequestScheduler scheduler) {
        this.scheduler = scheduler;
    }

    @Bean
    public RouterFunction<ServerResponse> loadBalancePrefill() {
        return route().POST(
                        "/rtp_llm/master/info",
                        accept(MediaType.APPLICATION_JSON),
                        this::responseMasterInfo)
                .POST(
                        "/rtp_llm/schedule_snapshot",
                        accept(MediaType.APPLICATION_JSON),
                        this::dumpLBStatus)
                .POST(
                        "/rtp_llm/notify_master",
                        accept(MediaType.APPLICATION_JSON),
                        this::notifyParticipant)
                .POST(
                        "/rtp_llm/update_log_level",
                        accept(MediaType.APPLICATION_JSON),
                        this::debugMode)
                .POST(
                        "/rtp_llm/update_traffic_policy",
                        accept(MediaType.APPLICATION_JSON),
                        this::updateTrafficPolicy)
                .GET(
                        "/rtp_llm/queue_snapshot",
                        accept(MediaType.APPLICATION_JSON),
                        this::queueSnapshot)
                .GET(
                        "/rtp_llm/inflight_status",
                        accept(MediaType.APPLICATION_JSON),
                        this::inflightStatus)
                .GET("/rtp_llm/server_latency", this::serverLatency)
                .POST("/rtp_llm/server_latency/reset", this::resetServerLatency)
                .build();
    }

    public Mono<ServerResponse> notifyParticipant(ServerRequest request) {
        return request.bodyToMono(MasterChangeNotifyReq.class)
                .flatMap(
                        masterChangeNotifyReq -> {
                            MasterChangeNotifyResp resp =
                                    masterStatusService.handleMasterChange(masterChangeNotifyReq);
                            return ServerResponse.ok()
                                    .contentType(MediaType.APPLICATION_JSON)
                                    .body(Mono.just(resp), MasterChangeNotifyResp.class);
                        })
                .onErrorResume(
                        (Function<Throwable, Mono<ServerResponse>>)
                                e -> {
                                    Logger.error("notifyParticipant error", e);
                                    return ServerResponse.status(500)
                                            .contentType(MediaType.APPLICATION_JSON)
                                            .body(Mono.just(e.getMessage()), String.class);
                                });
    }

    public Mono<ServerResponse> dumpLBStatus(ServerRequest request) {
        return request.bodyToMono(SyncLBStatusReq.class)
                .flatMap(
                        syncLBStatusReq -> {
                            SyncLBStatusResp resp = masterStatusService.dumpLBStatus();
                            return ServerResponse.ok()
                                    .contentType(MediaType.APPLICATION_JSON)
                                    .body(Mono.just(resp), SyncLBStatusResp.class);
                        })
                .onErrorResume(
                        e -> {
                            Logger.error("dumpLBStatus error", e);
                            return ServerResponse.status(500)
                                    .contentType(MediaType.APPLICATION_JSON)
                                    .body(Mono.just(e.getMessage()), String.class);
                        });
    }

    public Mono<ServerResponse> queueSnapshot(ServerRequest request) {
        try {
            List<RequestResult> snapshot = requests.snapshotActiveRequests();
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

    public Mono<ServerResponse> inflightStatus(ServerRequest request) {
        try {
            Map<String, Object> result = new LinkedHashMap<>();
            int tracked = requests.liveRequestCount();
            result.put("scheduler_inflight", tracked);
            result.put("scheduler_tracked", tracked);
            result.put(
                    "scheduler_blocked", scheduler == null ? 0 : scheduler.blockedRequestCount());
            result.put(
                    "decode_max_engine_requests",
                    configService
                            .loadBalanceConfig()
                            .getRouter()
                            .getRoles()
                            .getDecode()
                            .getAvailability()
                            .getMaxEngineRequests());

            List<Map<String, Object>> prefillList = new ArrayList<>();
            for (var entry : endpointRegistry.endpoints(RoleType.PREFILL, null)) {
                PrefillEndpoint endpoint = (PrefillEndpoint) entry;
                WorkerStatus.CacheIndexSnapshot cacheIndex =
                        endpoint.getStatus().cacheIndexSnapshot();
                CacheStatus cacheStatus = cacheIndex.cacheStatus();
                Map<String, Object> ep = new LinkedHashMap<>();
                ep.put("ip_port", entry.ipPort());
                var ownership = endpoint.stats();
                ep.put("inflight_requests", ownership.locallyOwnedRequests());
                ep.put("cache_version", cacheStatus == null ? -1L : cacheStatus.getVersion());
                ep.put("cache_indexed", cacheIndex.indexInitialized());
                ep.put("cache_indexed_version", cacheIndex.indexedVersion());
                ep.put("cache_key_size", cacheStatus == null ? 0 : cacheStatus.getCacheKeySize());
                prefillList.add(ep);
            }
            result.put("prefill_endpoints", prefillList);

            List<Map<String, Object>> decodeList = new ArrayList<>();
            for (var entry : endpointRegistry.endpoints(RoleType.DECODE, null)) {
                var view = ((DecodeEndpoint) entry).stats();
                Map<String, Object> ep = new LinkedHashMap<>();
                ep.put("ip_port", entry.ipPort());
                int reservedCount = view.inflight();
                ep.put("reserved_total", reservedCount);
                ep.put("confirmed_accepted", view.accepted());
                ep.put("confirmed_running", view.running());
                ep.put("total_load", view.totalLoad());
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

    private Mono<ServerResponse> debugMode(ServerRequest serverRequest) {
        return serverRequest
                .bodyToMono(LogLevelUpdateRequest.class)
                .flatMap(
                        logLevelUpdateRequest -> {
                            Logger.setLevel(logLevelUpdateRequest.getLogLevel());
                            return ServerResponse.ok()
                                    .contentType(MediaType.APPLICATION_JSON)
                                    .body(
                                            Mono.just("Success! logLevel=" + Logger.getLevel()),
                                            String.class);
                        })
                .onErrorResume(
                        e -> {
                            Logger.error("update logLevel error", e);
                            return ServerResponse.status(500)
                                    .contentType(MediaType.APPLICATION_JSON)
                                    .body(Mono.just(e.getMessage()), String.class);
                        });
    }

    private Mono<ServerResponse> updateTrafficPolicy(ServerRequest serverRequest) {
        return serverRequest
                .bodyToMono(TrafficPolicyConfig.class)
                .flatMap(
                        trafficPolicyConfig -> {
                            configService.updateTrafficPolicy(trafficPolicyConfig);
                            return ServerResponse.ok()
                                    .contentType(MediaType.APPLICATION_JSON)
                                    .body(
                                            Mono.just(trafficPolicyConfig),
                                            TrafficPolicyConfig.class);
                        })
                .onErrorResume(
                        e -> {
                            Logger.error("update traffic policy error", e);
                            return ServerResponse.status(500)
                                    .contentType(MediaType.APPLICATION_JSON)
                                    .body(Mono.just(e.getMessage()), String.class);
                        });
    }

    private Mono<ServerResponse> responseMasterInfo(ServerRequest request) {
        return request.bodyToMono(Request.class)
                .flatMap(
                        (Function<Request, Mono<ServerResponse>>)
                                req -> {
                                    Response result = new Response();
                                    result.setRealMasterHost(
                                            masterStatusService.getMasterHostIpPort());
                                    result.setQueueLength(scheduler.queuedRequestCount());
                                    result.setRequiresGenerateInput(false);
                                    result.setCode(200);
                                    result.setSuccess(true);
                                    result.setWorkerSummary(buildWorkerSummary());
                                    result.setReady(
                                            workerStatusSynchronizer == null
                                                    || workerStatusSynchronizer.isReady());
                                    return ServerResponse.ok()
                                            .contentType(MediaType.APPLICATION_JSON)
                                            .body(Mono.just(result), Response.class);
                                })
                .onErrorResume(
                        e -> {
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

    private Map<String, Response.WorkerRoleSummary> buildWorkerSummary() {
        Map<String, Response.WorkerRoleSummary> summary = new LinkedHashMap<>();
        for (RoleType role : RoleType.values()) {
            Map<String, WorkerStatus> statusMap = endpointRegistry.statusSnapshot(role);
            if (statusMap.isEmpty()) {
                continue;
            }
            Response.WorkerRoleSummary rs = new Response.WorkerRoleSummary();
            rs.setDiscovered(statusMap.size());
            for (WorkerStatus ws : statusMap.values()) {
                if (ws.pollHealth().reportedAlive()) {
                    rs.setAlive(rs.getAlive() + 1);
                }
            }
            summary.put(role.getCode(), rs);
        }
        return summary.isEmpty() ? null : summary;
    }

    private QueueSnapshotResponse persistSchedulerSnapshot(List<RequestResult> snapshot)
            throws IOException {
        Files.createDirectories(SCHEDULER_SNAPSHOT_DIR);
        cleanOldSchedulerSnapshots();

        long timestamp = System.currentTimeMillis();
        Path file = SCHEDULER_SNAPSHOT_DIR.resolve(SCHEDULER_SNAPSHOT_PREFIX + timestamp + ".json");
        Files.writeString(file, JsonUtils.toFormattedString(snapshot));
        return new QueueSnapshotResponse(
                file.toAbsolutePath().toString(), timestamp, snapshot.size());
    }

    private void cleanOldSchedulerSnapshots() throws IOException {
        List<Path> files;
        try (Stream<Path> entries = Files.list(SCHEDULER_SNAPSHOT_DIR)) {
            files =
                    entries.filter(
                                    path ->
                                            path.getFileName()
                                                    .toString()
                                                    .startsWith(SCHEDULER_SNAPSHOT_PREFIX))
                            .sorted()
                            .toList();
        }
        for (int index = 0; index <= files.size() - MAX_SNAPSHOT_FILES; index++) {
            Files.deleteIfExists(files.get(index));
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
}
