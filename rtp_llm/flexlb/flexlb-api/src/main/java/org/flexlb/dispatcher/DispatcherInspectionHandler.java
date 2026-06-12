package org.flexlb.dispatcher;

import com.alibaba.fastjson2.JSONArray;
import com.alibaba.fastjson2.JSONObject;
import org.flexlb.dao.loadbalance.BatchScheduleTarget;
import org.flexlb.util.Logger;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.core.codec.DecodingException;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.server.ServerRequest;
import org.springframework.web.reactive.function.server.ServerResponse;
import reactor.core.publisher.Mono;

import java.util.List;
import java.util.Optional;

/**
 * Read-only diagnostic endpoints exposing dispatcher-local state and what dispatcher would emit
 * for a hypothetical request.
 *
 * <ul>
 *   <li><b>{@link #snapshot}</b> — {@code GET /dispatcher/_snapshot}. Returns the dispatcher's
 *       current FE pool view in round-robin order with per-host liveness and consecutive-failure
 *       counts. No FE traffic, no master traffic, no side effects.</li>
 *   <li><b>{@link #dryRun}</b> — {@code POST /dispatcher/_dryrun/<spec.path>}. Runs the real
 *       chunk-assembly pipeline ({@link BatchChunkAssembler}) against the request body and
 *       returns the resulting sub-batch bodies as JSON instead of fanning out. With
 *       {@code ?pre_assign=true} (or with config default true) it does call
 *       {@link BatchScheduleClient} for real BE resolution — advancing master's batch RR cursor
 *       just like a production request would, intentional so the dry-run reflects production wire
 *       shape rather than a fake-target reconstruction.</li>
 * </ul>
 *
 * <p>Both endpoints share the dispatcher's enable gate ({@code dispatch.fe-pool-service-id}).
 * A disabled dispatcher does not register either route.
 */
@Component
@ConditionalOnProperty(prefix = "dispatch", name = "fe-pool-service-id")
public class DispatcherInspectionHandler {

    private static final String DRYRUN_URI_PREFIX = "/dispatcher/_dryrun";

    private final DispatchConfig cfg;
    private final DispatcherFePoolRefresher refresher;
    private final FeHealthChecker healthChecker;
    private final BatchScheduleClient batchScheduleClient;

    public DispatcherInspectionHandler(DispatchConfig cfg,
                                       DispatcherFePoolRefresher refresher,
                                       FeHealthChecker healthChecker,
                                       BatchScheduleClient batchScheduleClient) {
        this.cfg = cfg;
        this.refresher = refresher;
        this.healthChecker = healthChecker;
        this.batchScheduleClient = batchScheduleClient;
    }

    // ───────────────────────── snapshot ─────────────────────────

    public Mono<ServerResponse> snapshot(ServerRequest request) {
        List<String> urls = refresher.source().get();
        JSONObject root = new JSONObject();
        JSONObject fePool = new JSONObject();
        fePool.put("serviceId", cfg.getFePoolServiceId());
        fePool.put("size", urls.size());
        JSONArray hosts = new JSONArray();
        for (String url : urls) {
            JSONObject host = new JSONObject();
            host.put("url", url);
            host.put("alive", healthChecker.isAlive(url));
            host.put("consecFails", healthChecker.consecFails(url));
            hosts.add(host);
        }
        fePool.put("hosts", hosts);
        root.put("fePool", fePool);
        return ServerResponse.ok().contentType(MediaType.APPLICATION_JSON).bodyValue(BatchBodyParser.serialize(root));
    }

    // ───────────────────────── dryRun ─────────────────────────

    public Mono<ServerResponse> dryRun(ServerRequest request) {
        String fePath = extractFePath(request);
        BatchEndpointSpec spec = fePath == null ? null : BatchEndpointSpec.BY_PATH.get(fePath);
        if (spec == null) {
            return badRequest("unknown batch endpoint path: " + fePath
                    + ", registered: " + BatchEndpointSpec.BY_PATH.keySet());
        }
        boolean effectivePreAssign = resolvePreAssign(request);
        return request.bodyToMono(byte[].class).defaultIfEmpty(new byte[0]).flatMap(bytes -> {
            JSONObject body = BatchBodyParser.parseObject(bytes);
            if (body == null) {
                return badRequest("expected a JSON object body");
            }
            JSONArray arr = BatchBodyParser.findArrayField(body, spec.getRequestArrayField());
            if (arr == null) {
                return badRequest("missing or non-array field: " + spec.getRequestArrayField());
            }
            return buildDryRunResponse(spec, body, arr, effectivePreAssign);
        }).onErrorResume(this::handleDryRunException);
    }

    private Mono<ServerResponse> handleDryRunException(Throwable e) {
        if (e instanceof DecodingException) {
            return badRequest("malformed JSON body: " + e.getMessage());
        }
        String reason = DispatcherResponses.briefReason(e);
        Logger.warn("dispatcher dry-run unexpected error: {}", reason);
        return DispatcherResponses.error(500, "dryrun_internal_error", reason);
    }

    private String extractFePath(ServerRequest request) {
        String raw = request.uri().getRawPath();
        if (raw == null || !raw.startsWith(DRYRUN_URI_PREFIX)) {
            return null;
        }
        String tail = raw.substring(DRYRUN_URI_PREFIX.length());
        return tail.isEmpty() ? null : tail;
    }

    private boolean resolvePreAssign(ServerRequest request) {
        Optional<String> q = request.queryParam("pre_assign");
        if (q.isEmpty()) {
            return cfg.isPreAssignBe();
        }
        String v = q.get().trim().toLowerCase();
        return switch (v) {
            case "true" -> true;
            case "false" -> false;
            default -> cfg.isPreAssignBe();
        };
    }

    private Mono<ServerResponse> buildDryRunResponse(BatchEndpointSpec spec, JSONObject envelope,
                                                     JSONArray arr, boolean effectivePreAssign) {
        List<JSONArray> chunks = BatchChunkAssembler.split(arr, cfg.getSubBatchSpec());
        List<JSONObject> chunkBodies = BatchChunkAssembler.buildChunkBodies(
                envelope, chunks, spec.getRequestArrayField());
        boolean shouldResolveTargets = effectivePreAssign && !chunks.isEmpty();
        Mono<List<BatchScheduleTarget>> targetsMono = shouldResolveTargets
                ? batchScheduleClient.requestTargets(chunks.size())
                : Mono.just(List.of());
        return targetsMono.map(targets -> {
            BatchChunkAssembler.stampPreAssignedBe(chunkBodies, targets);
            JSONObject out = new JSONObject();
            out.put("path", spec.getPath());
            out.put("splitMode", cfg.getSubBatch());
            out.put("totalItems", arr.size());
            out.put("chunkCount", chunks.size());
            out.put("preAssignConfigDefault", cfg.isPreAssignBe());
            out.put("preAssignEffective", effectivePreAssign);
            JSONArray targetsOut = new JSONArray();
            for (BatchScheduleTarget t : targets) {
                JSONObject addr = new JSONObject();
                addr.put("role", t.getRole() == null ? null : t.getRole().name());
                addr.put("ip", t.getServerIp());
                addr.put("httpPort", t.getHttpPort());
                addr.put("grpcPort", t.getGrpcPort());
                addr.put("arpcPort", t.getArpcPort());
                addr.put("preAssignable", BatchChunkAssembler.isPreAssignable(t));
                targetsOut.add(addr);
            }
            out.put("preAssignTargets", targetsOut);
            JSONArray chunksOut = new JSONArray();
            chunksOut.addAll(chunkBodies);
            out.put("chunks", chunksOut);
            return out;
        }).flatMap(out -> ServerResponse.ok().contentType(MediaType.APPLICATION_JSON).bodyValue(BatchBodyParser.serialize(out)));
    }

    private Mono<ServerResponse> badRequest(String message) {
        return DispatcherResponses.error(400, "invalid_inspection_request", message);
    }
}
