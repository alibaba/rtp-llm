package org.flexlb.httpserver;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.scheduler.DiagnosticsProvider;
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
import org.flexlb.service.RouteService;
import org.flexlb.state.DecodeRequestStateView;
import org.flexlb.state.GenerationTriple;
import org.flexlb.state.LedgerTraceView;
import org.flexlb.state.PrefillRequestStateView;
import org.flexlb.sync.shadow.StateShadowBridge;
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
import java.util.Optional;
import java.util.function.Function;

import static org.springframework.web.reactive.function.server.RequestPredicates.accept;
import static org.springframework.web.reactive.function.server.RouterFunctions.route;

@Component
public class HttpLoadBalanceServer {
    private final LBStatusConsistencyService lbStatusConsistencyService;
    private final ConfigService configService;
    private final RouteService routeService;
    private final EndpointRegistry endpointRegistry;
    private final MasterEngineSynchronizer masterEngineSynchronizer;
    private final ServerScheduleLatencyRecorder serverLatencyRecorder;
    private final StateShadowBridge stateShadowBridge;

    public HttpLoadBalanceServer(LBStatusConsistencyService lbStatusConsistencyService,
                                 ConfigService configService,
                                 RouteService routeService,
                                 EndpointRegistry endpointRegistry,
                                 @org.springframework.beans.factory.annotation.Autowired(required = false)
                                 MasterEngineSynchronizer masterEngineSynchronizer,
                                 ServerScheduleLatencyRecorder serverLatencyRecorder) {
        this(lbStatusConsistencyService, configService, routeService, endpointRegistry,
                masterEngineSynchronizer, serverLatencyRecorder, null);
    }

    // 两个构造器并存时 Spring 无法自动选择（无构造器级 @Autowired 会回退找
    // 无参构造器并失败）；必须显式标注生产用构造器。测试可直接 new 任一构造器。
    @org.springframework.beans.factory.annotation.Autowired
    public HttpLoadBalanceServer(LBStatusConsistencyService lbStatusConsistencyService,
                                 ConfigService configService,
                                 RouteService routeService,
                                 EndpointRegistry endpointRegistry,
                                 @org.springframework.beans.factory.annotation.Autowired(required = false)
                                 MasterEngineSynchronizer masterEngineSynchronizer,
                                 ServerScheduleLatencyRecorder serverLatencyRecorder,
                                 @org.springframework.beans.factory.annotation.Autowired(required = false)
                                 StateShadowBridge stateShadowBridge) {
        this.lbStatusConsistencyService = lbStatusConsistencyService;
        this.configService = configService;
        this.routeService = routeService;
        this.endpointRegistry = endpointRegistry;
        this.masterEngineSynchronizer = masterEngineSynchronizer;
        this.serverLatencyRecorder = serverLatencyRecorder;
        this.stateShadowBridge = stateShadowBridge;
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
                .GET("/rtp_llm/server_latency", this::serverLatency)
                .POST("/rtp_llm/server_latency/reset", this::resetServerLatency)
                .GET("/rtp_llm/state_ledger/trace/{requestId}", accept(MediaType.APPLICATION_JSON),
                        this::stateLedgerTrace)
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

    /**
     * Aggregate diagnostics from all {@link DiagnosticsProvider} components
     * (schedulers, inflightStore, endpointRegistry) registered in
     * {@link RouteService#getDiagnosticsProviders()}. This replaces
     * hard-coded calls to {@code queueLength()} / {@code snapshotQueue()}
     * — the HTTP layer no longer needs to know which scheduler owns which
     * diagnostic; it just searches the aggregated map by well-known keys.
     */
    private Map<String, Object> aggregateDiagnostics() {
        Map<String, Object> all = new LinkedHashMap<>();
        for (DiagnosticsProvider provider : routeService.getDiagnosticsProviders()) {
            all.putAll(provider.getDiagnostics());
        }
        return all;
    }

    private Mono<ServerResponse> responseMasterInfo(ServerRequest request) {
        return request.bodyToMono(Request.class)
                .flatMap((Function<Request, Mono<ServerResponse>>) req -> {
                    Response result = new Response();
                    result.setRealMasterHost(lbStatusConsistencyService.getMasterHostIpPort());
                    // Aggregate queue length from DiagnosticsProvider instead of
                    // hard-coded queueLength() call on RouteService.
                    Map<String, Object> masterDiagnostics = aggregateDiagnostics();
                    Object ql = masterDiagnostics.get("queue_length");
                    result.setQueueLength(ql instanceof Integer ? (Integer) ql : 0);
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
            // Extract queue snapshot from DiagnosticsProvider instead of
            // hard-coded snapshotQueue() call on RouteService.
            Map<String, Object> snapshotDiagnostics = aggregateDiagnostics();
            Object qs = snapshotDiagnostics.get("queue_snapshot");
            QueueSnapshotResponse response = qs instanceof QueueSnapshotResponse
                    ? (QueueSnapshotResponse) qs
                    : new QueueSnapshotResponse();
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
            // Aggregate inflight count from DiagnosticsProvider instead of
            // hard-coded globalInflightSize() call on RouteService.
            Map<String, Object> inflightDiag = aggregateDiagnostics();
            Object ac = inflightDiag.get("active_count");
            result.put("scheduler_inflight", ac instanceof Integer ? (Integer) ac : 0);

            List<Map<String, Object>> prefillList = new ArrayList<>();
            for (Map.Entry<String, PrefillEndpoint> entry : endpointRegistry.getPrefillEndpoints().entrySet()) {
                PrefillEndpoint prefill = entry.getValue();
                Map<String, Object> ep = new LinkedHashMap<>();
                ep.put("ip_port", entry.getKey());
                // Ledger per-EP view: activeTotal is the request-level active
                // count (batch members counted individually); field names kept
                // for external script compatibility.
                int active = prefill.prefillActiveRequestCount();
                int engineOwned = (int) prefill.prefillEngineOwnedCount();
                ep.put("inflight_batches", active);
                // Dispatched but not yet acknowledged by the engine
                ep.put("inflight_entries", Math.max(0, active - engineOwned));
                // Engine-acknowledged work, with phase breakdown
                ep.put("engine_work", engineOwned);
                ep.put("engine_waiting", prefill.prefillEngineWaitingCount());
                ep.put("engine_running", prefill.prefillEngineRunningCount());
                prefillList.add(ep);
            }
            result.put("prefill_endpoints", prefillList);

            List<Map<String, Object>> decodeList = new ArrayList<>();
            for (Map.Entry<String, DecodeEndpoint> entry : endpointRegistry.getDecodeEndpoints().entrySet()) {
                DecodeEndpoint decode = entry.getValue();
                Map<String, Object> ep = new LinkedHashMap<>();
                ep.put("ip_port", entry.getKey());
                // Unconfirmed reservations: dispatched but not yet acknowledged by the engine
                ep.put("inflight_requests", decode.decodeInflightCount());
                // Engine-acknowledged work (WAITING/LOADING/RUNNING)
                ep.put("engine_work", decode.decodeEngineWorkCount());
                ep.put("total_load", decode.decodeTotalLoad());
                // Unconfirmed KV reservations: hard (seqLen) vs expected (seqLen + maxNewTokens)
                ep.put("kv_reserved_hard", decode.decodeInflightHardKvReserved());
                ep.put("kv_reserved_expected", decode.decodeInflightExpectedKvReserved());
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

    /**
     * Per-request state-ledger diagnostic storyline (read-only, no auth — same
     * exposure level as the other diagnostics endpoints): P-side active entry
     * (phase / generation binding / batch / timestamps / trace ring) + D-side
     * active entry + both tombstone terminal states (queryable within the
     * tombstone retention window). Data source is the StateShadowBridge's
     * StateLedger read-only view.
     */
    private Mono<ServerResponse> stateLedgerTrace(ServerRequest request) {
        long requestId;
        try {
            requestId = Long.parseLong(request.pathVariable("requestId"));
        } catch (NumberFormatException e) {
            return ServerResponse.status(400)
                    .contentType(MediaType.APPLICATION_JSON)
                    .bodyValue(Map.of("found", false,
                            "error", "invalid requestId (expected decimal long): "
                                    + request.pathVariable("requestId")));
        }
        try {
            StateShadowBridge bridge = stateShadowBridge;
            if (bridge == null || !bridge.isEnabled()) {
                return ServerResponse.ok()
                        .contentType(MediaType.APPLICATION_JSON)
                        .bodyValue(Map.of("request_id", requestId, "found", false,
                                "reason", "state ledger disabled (flexlbStateV2ShadowEnabled=false)"));
            }
            Optional<LedgerTraceView> trace = bridge.ledger().traceOf(requestId);
            if (trace.isEmpty()) {
                // Unknown request or fully expired (no active entry and no tombstone
                // within the retention window).
                return ServerResponse.ok()
                        .contentType(MediaType.APPLICATION_JSON)
                        .bodyValue(Map.of("request_id", requestId, "found", false));
            }
            return ServerResponse.ok()
                    .contentType(MediaType.APPLICATION_JSON)
                    .bodyValue(buildStateLedgerTraceBody(requestId, trace.get()));
        } catch (Exception e) {
            Logger.error("stateLedgerTrace error", e);
            return ServerResponse.status(500)
                    .contentType(MediaType.APPLICATION_JSON)
                    .bodyValue(Map.of("found", false, "error", String.valueOf(e.getMessage())));
        }
    }

    private Map<String, Object> buildStateLedgerTraceBody(long requestId, LedgerTraceView view) {
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("request_id", requestId);
        body.put("found", true);
        body.put("prefill_active", view.prefillActive().map(this::prefillEntryBody).orElse(null));
        body.put("decode_active", view.decodeActive().map(this::decodeEntryBody).orElse(null));
        body.put("prefill_tombstone", view.prefillTombstone().map(this::tombstoneBody).orElse(null));
        body.put("decode_tombstone", view.decodeTombstone().map(this::tombstoneBody).orElse(null));
        return body;
    }

    private Map<String, Object> prefillEntryBody(PrefillRequestStateView entry) {
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("side", "P");
        body.put("phase", entry.phaseName());
        body.put("phase_ordinal", entry.phaseOrdinal());
        body.put("created_at_ms", entry.createdAtMs());
        body.put("batch_id", entry.batchId());
        body.put("binding", bindingBody(entry.binding()));
        body.put("dispatched_at_ms", entry.dispatchedAtMs());
        body.put("engine_owned", entry.engineOwned());
        body.put("kv_tokens_reported", entry.kvTokensReported());
        body.put("last_seen_round", entry.lastSeenRound());
        body.put("last_version", entry.lastVersion());
        body.put("pending_cancel", entry.pendingCancel());
        body.put("trace", entry.trace());
        return body;
    }

    private Map<String, Object> decodeEntryBody(DecodeRequestStateView entry) {
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("side", "D");
        body.put("phase", entry.phaseName());
        body.put("phase_ordinal", entry.phaseOrdinal());
        body.put("created_at_ms", entry.createdAtMs());
        body.put("binding", bindingBody(entry.binding()));
        body.put("reserved_kv", entry.reservedKv());
        body.put("reserved_expected_kv", entry.reservedExpectedKv());
        body.put("engine_owned", entry.engineOwned());
        body.put("kv_tokens_reported", entry.kvTokensReported());
        body.put("last_seen_round", entry.lastSeenRound());
        body.put("last_version", entry.lastVersion());
        body.put("pending_cancel", entry.pendingCancel());
        body.put("trace", entry.trace());
        return body;
    }

    private Map<String, Object> tombstoneBody(LedgerTraceView.TombstoneView tombstone) {
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("request_id", tombstone.requestId());
        body.put("state", tombstone.state().name());
        body.put("reason", tombstone.reason().name());
        body.put("terminal_at_ms", tombstone.terminalAtMs());
        body.put("trace", tombstone.entryTrace());
        return body;
    }

    private Map<String, Object> bindingBody(GenerationTriple binding) {
        if (binding == null) {
            return null;
        }
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("endpoint_id", binding.endpointId());
        body.put("generation", binding.generation());
        body.put("batch_id", binding.batchId());
        return body;
    }
}
