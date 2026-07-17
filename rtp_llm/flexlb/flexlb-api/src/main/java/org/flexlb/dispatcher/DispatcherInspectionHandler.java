package org.flexlb.dispatcher;

import com.alibaba.fastjson2.JSONArray;
import com.alibaba.fastjson2.JSONObject;
import org.flexlb.dao.loadbalance.BatchScheduleTarget;
import org.flexlb.util.Logger;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.server.ServerRequest;
import org.springframework.web.reactive.function.server.ServerResponse;
import reactor.core.publisher.Mono;

import java.util.List;

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
 *       returns the resulting sub-batch bodies as JSON instead of fanning out. Side-effect-free
 *       by default: BE resolution is skipped, so no master traffic and no RR-cursor movement.
 *       Only an explicit {@code ?pre_assign=true} calls {@link BatchScheduleClient} for real BE
 *       resolution — which <em>does</em> advance master's batch RR cursor exactly like a
 *       production request, so use it only when you need the production wire shape and can
 *       accept perturbing live distribution.</li>
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
        return DispatcherResponses.jsonBytes(200, BatchBodyParser.serialize(root));
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
            // Same disposition production uses (BatchEndpointSpec#isSplittableBatch), so dry-run
            // cannot drift from what BatchHandler actually does.
            if (!spec.isSplittableBatch(body, arr)) {
                return passthroughDiagnostic(spec, arr);
            }
            return buildDryRunResponse(spec, body, arr, effectivePreAssign);
        }).onErrorResume(this::handleDryRunException);
    }

    private Mono<ServerResponse> handleDryRunException(Throwable e) {
        // Detail to the log only — the exception text can name internal hosts.
        Logger.warn("dispatcher dry-run unexpected error: {}", DispatcherResponses.briefReason(e));
        return DispatcherResponses.error(500, "dryrun_internal_error", "dry-run failed");
    }

    private String extractFePath(ServerRequest request) {
        // Routed exclusively via POST /dispatcher/_dryrun/**, so the prefix is guaranteed.
        String tail = request.uri().getRawPath().substring(DRYRUN_URI_PREFIX.length());
        return tail.isEmpty() ? null : tail;
    }

    /**
     * Dry-run is side-effect-free by default: resolving BE targets calls master {@code
     * /batch_schedule}, which advances the round-robin cursor and so perturbs the distribution of
     * real traffic. A diagnostic must not do that unless the caller explicitly asks — hence the
     * default is {@code false} regardless of {@link DispatchConfig#isPreAssignBe()}, and only an
     * explicit {@code ?pre_assign=true} opts into the production-accurate (state-advancing) run.
     */
    private boolean resolvePreAssign(ServerRequest request) {
        return request.queryParam("pre_assign")
                .map(v -> "true".equals(v.trim().toLowerCase()))
                .orElse(false);
    }

    private Mono<ServerResponse> buildDryRunResponse(BatchEndpointSpec spec, JSONObject envelope,
                                                     JSONArray arr, boolean effectivePreAssign) {
        List<JSONArray> chunks = BatchChunkAssembler.split(arr, cfg.getSubBatchSpec());
        List<JSONObject> chunkBodies = BatchChunkAssembler.buildChunkBodies(
                envelope, chunks, spec.getRequestArrayField());
        boolean shouldResolveTargets = effectivePreAssign && spec.isPreAssignable() && !chunks.isEmpty();
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
            out.put("preAssignSupported", spec.isPreAssignable());
            out.put("preAssignEffective", effectivePreAssign && spec.isPreAssignable());
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
        }).flatMap(out -> DispatcherResponses.jsonBytes(200, BatchBodyParser.serialize(out)));
    }

    /**
     * Mirror the production path's non-splittable disposition: a registered endpoint whose body
     * is not batch-shaped (e.g. {@code /v1/embeddings} given one multimodal input as
     * {@code List[ContentPart]}) is passthrough-forwarded whole, not split per element. Report
     * that here instead of fabricating per-element chunks.
     */
    private Mono<ServerResponse> passthroughDiagnostic(BatchEndpointSpec spec, JSONArray arr) {
        JSONObject out = new JSONObject();
        out.put("path", spec.getPath());
        out.put("splitMode", cfg.getSubBatch());
        out.put("totalItems", arr == null ? 0 : arr.size());
        out.put("chunkCount", 0);
        out.put("disposition", "passthrough");
        out.put("reason", "request is not splittable for this endpoint; forwarded whole to a single FE");
        return DispatcherResponses.jsonBytes(200, BatchBodyParser.serialize(out));
    }

    private Mono<ServerResponse> badRequest(String message) {
        return DispatcherResponses.error(400, "invalid_inspection_request", message);
    }
}
