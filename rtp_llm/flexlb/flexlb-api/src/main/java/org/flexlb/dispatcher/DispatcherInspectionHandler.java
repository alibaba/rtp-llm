package org.flexlb.dispatcher;

import com.alibaba.fastjson2.JSONArray;
import com.alibaba.fastjson2.JSONObject;
import org.flexlb.config.ConfigService;
import org.flexlb.dao.loadbalance.BatchScheduleTarget;
import org.flexlb.util.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.core.io.buffer.DataBufferLimitException;
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
    private final int maxChunkCount;
    private final long maxResponseBytes;

    @Autowired
    public DispatcherInspectionHandler(DispatchConfig cfg,
                                       DispatcherFePoolRefresher refresher,
                                       FeHealthChecker healthChecker,
                                       BatchScheduleClient batchScheduleClient,
                                       ConfigService configService) {
        this(cfg, refresher, healthChecker, batchScheduleClient,
                configService.loadBalanceConfig().getBatchScheduleMaxCount());
    }

    /** Package-private convenience for focused tests; mirrors the production default. */
    DispatcherInspectionHandler(DispatchConfig cfg,
                                DispatcherFePoolRefresher refresher,
                                FeHealthChecker healthChecker,
                                BatchScheduleClient batchScheduleClient) {
        this(cfg, refresher, healthChecker, batchScheduleClient, 1000);
    }

    DispatcherInspectionHandler(DispatchConfig cfg,
                                DispatcherFePoolRefresher refresher,
                                FeHealthChecker healthChecker,
                                BatchScheduleClient batchScheduleClient,
                                int maxChunkCount) {
        if (maxChunkCount < 1) {
            throw new IllegalArgumentException("maxChunkCount must be >= 1, got " + maxChunkCount);
        }
        this.cfg = cfg;
        this.refresher = refresher;
        this.healthChecker = healthChecker;
        this.batchScheduleClient = batchScheduleClient;
        this.maxChunkCount = maxChunkCount;
        this.maxResponseBytes = cfg.getMaxDryRunResponseBytes();
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
            String generateConfigError = BatchChunkAssembler.validateGenerateConfig(body);
            if (generateConfigError != null) {
                return badRequest(generateConfigError);
            }
            JSONArray arr = BatchBodyParser.findArrayField(body, spec.getRequestArrayField());
            // Same disposition production uses (BatchEndpointSpec#isSplittableBatch), so dry-run
            // cannot drift from what BatchHandler actually does.
            if (!spec.isSplittableBatch(body, arr)) {
                return passthroughDiagnostic(spec, arr);
            }
            String validationError = spec.validateForFanout(body);
            if (validationError != null) {
                return badRequest(validationError);
            }
            int chunkCount = BatchChunkAssembler.chunkCount(arr.size(), cfg.getSubBatchSpec());
            if (chunkCount > maxChunkCount) {
                return DispatcherResponses.error(413, "too_many_sub_batches",
                        "batch produces " + chunkCount + " sub-batches; maximum is "
                                + maxChunkCount + " (BATCH_SCHEDULE_MAX_COUNT)");
            }
            // Reject request-controlled envelope amplification before target resolution (which can
            // advance master's BE cursor) and before allocating any chunk arrays/bodies.
            if (projectedResponseBytes(
                    spec, body, arr, chunkCount, effectivePreAssign, List.of()) > maxResponseBytes) {
                return responseTooLarge();
            }
            return buildDryRunResponse(spec, body, arr, chunkCount, effectivePreAssign);
        }).onErrorResume(this::handleDryRunException);
    }

    private Mono<ServerResponse> handleDryRunException(Throwable e) {
        // Detail to the log only — the exception text can name internal hosts.
        Logger.warn("dispatcher dry-run unexpected error: {}", DispatcherResponses.briefReason(e));
        if (e instanceof DataBufferLimitException) {
            return DispatcherResponses.error(413, "request_body_too_large",
                    "dry-run body exceeds the server limit; see MAX_IN_MEMORY_SIZE");
        }
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
                .map(v -> Boolean.parseBoolean(v.trim()))
                .orElse(false);
    }

    private Mono<ServerResponse> buildDryRunResponse(BatchEndpointSpec spec, JSONObject envelope,
                                                     JSONArray arr, int chunkCount,
                                                     boolean effectivePreAssign) {
        boolean shouldResolveTargets = effectivePreAssign && spec.isPreAssignable() && chunkCount > 0;
        Mono<List<BatchScheduleTarget>> targetsMono = shouldResolveTargets
                // A dry-run only renders BE role_addrs. Never consume the FE cursor for a request
                // that deliberately sends no chunk to the selected FE.
                ? batchScheduleClient.requestTargets(chunkCount, true, false)
                : Mono.just(List.of());
        return targetsMono.flatMap(targets -> {
            // Target strings come from trusted discovery rather than the caller, but they are not
            // length-bounded by the wire type. Account for them before materializing repeated
            // envelopes as well; the final serialization check remains the authoritative backstop.
            if (!targets.isEmpty() && projectedResponseBytes(
                    spec, envelope, arr, chunkCount, effectivePreAssign, targets)
                    > maxResponseBytes) {
                return responseTooLarge();
            }
            List<JSONArray> chunks = BatchChunkAssembler.split(arr, cfg.getSubBatchSpec());
            List<JSONObject> chunkBodies = BatchChunkAssembler.buildChunkBodies(
                    envelope, chunks, spec.getRequestArrayField());
            spec.prepareChunkBodies(envelope, chunkBodies);
            BatchChunkAssembler.stampPreAssignedBe(chunkBodies, targets);
            JSONArray chunksOut = new JSONArray();
            chunksOut.addAll(chunkBodies);
            JSONObject out = dryRunEnvelope(
                    spec, arr.size(), chunkCount, effectivePreAssign, targets, chunksOut);
            byte[] response = BatchBodyParser.serialize(out);
            if (response.length > maxResponseBytes) {
                return responseTooLarge();
            }
            return DispatcherResponses.jsonBytes(200, response);
        });
    }

    /**
     * Exact size projection before chunk materialization. Each chunk differs from one transformed
     * empty-array template only by its slice and, optionally, one dispatcher role-address field.
     * Summing those deltas avoids serializing the repeated envelope {@code chunkCount} times while
     * still counting explicit JSON nulls exactly as the final dry-run serializer does.
     */
    private long projectedResponseBytes(BatchEndpointSpec spec, JSONObject envelope,
                                        JSONArray arr, int chunkCount, boolean effectivePreAssign,
                                        List<BatchScheduleTarget> targets) {
        JSONObject skeleton = dryRunEnvelope(
                spec, arr.size(), chunkCount, effectivePreAssign, targets, new JSONArray());
        long projected = BatchBodyParser.serialize(skeleton).length;
        if (chunkCount == 0) {
            return projected;
        }
        List<JSONObject> templateBodies = BatchChunkAssembler.buildChunkBodies(
                envelope, List.of(new JSONArray()), spec.getRequestArrayField());
        spec.prepareChunkBodies(envelope, templateBodies);
        long templateBytes = BatchBodyParser.serialize(templateBodies.get(0)).length;
        long itemBytes = BatchBodyParser.serialize(arr).length;

        // For k chunks, the sum of their array serializations is the original array size + k - 1
        // (each new pair of [] replaces one comma). Each body then replaces the template's [] with
        // that chunk array. The outer chunks array contributes another k - 1 commas.
        long chunkBodiesBytes = saturatedAdd(
                saturatedMultiply(templateBytes - 2, chunkCount),
                saturatedAdd(itemBytes, chunkCount - 1L));

        int stamped = Math.min(chunkCount, targets.size());
        for (int i = 0; i < stamped; i++) {
            BatchScheduleTarget target = targets.get(i);
            if (BatchChunkAssembler.isPreAssignable(target)) {
                long roleAddrsBytes = BatchBodyParser.serialize(
                        BatchChunkAssembler.preAssignedRoleAddrs(target)).length;
                // The transformed prompt-batch template always has a non-empty generate_config
                // (force_batch is present), so inserting this property adds one comma plus the
                // ASCII `"role_addrs":` key and its array value.
                chunkBodiesBytes = saturatedAdd(chunkBodiesBytes,
                        saturatedAdd(14, roleAddrsBytes));
            }
        }
        return saturatedAdd(projected,
                saturatedAdd(chunkBodiesBytes, chunkCount - 1L));
    }

    private static long saturatedMultiply(long value, long multiplier) {
        if (value > Long.MAX_VALUE / multiplier) {
            return Long.MAX_VALUE;
        }
        return value * multiplier;
    }

    private static long saturatedAdd(long left, long right) {
        if (left > Long.MAX_VALUE - right) {
            return Long.MAX_VALUE;
        }
        return left + right;
    }

    private JSONObject dryRunEnvelope(BatchEndpointSpec spec, int totalItems, int chunkCount,
                                      boolean effectivePreAssign,
                                      List<BatchScheduleTarget> targets, JSONArray chunks) {
        JSONObject out = new JSONObject();
        out.put("path", spec.getPath());
        out.put("splitMode", cfg.getSubBatch());
        out.put("totalItems", totalItems);
        out.put("chunkCount", chunkCount);
        out.put("preAssignConfigDefault", cfg.isPreAssignBe());
        out.put("preAssignSupported", spec.isPreAssignable());
        out.put("preAssignEffective", effectivePreAssign && spec.isPreAssignable());
        JSONArray targetsOut = new JSONArray(targets.size());
        for (BatchScheduleTarget target : targets) {
            JSONObject addr = new JSONObject();
            addr.put("role", target.getRole() == null ? null : target.getRole().name());
            addr.put("ip", target.getServerIp());
            addr.put("httpPort", target.getHttpPort());
            addr.put("grpcPort", target.getGrpcPort());
            addr.put("arpcPort", target.getArpcPort());
            addr.put("preAssignable", BatchChunkAssembler.isPreAssignable(target));
            targetsOut.add(addr);
        }
        out.put("preAssignTargets", targetsOut);
        out.put("chunks", chunks);
        return out;
    }

    private Mono<ServerResponse> responseTooLarge() {
        return DispatcherResponses.error(413, "dryrun_response_too_large",
                "dry-run response exceeds the configured dispatcher limit");
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
