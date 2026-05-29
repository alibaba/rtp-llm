package org.flexlb.dispatcher;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
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
 * for a hypothetical request. Both endpoints share this single handler so the {@code /dispatcher/_*}
 * inspection surface lives in one place — adding {@code _config_dump}, {@code _metrics},
 * {@code _warm_target} later is a new method here, not a new file.
 *
 * <ul>
 *   <li><b>{@link #snapshot}</b> — {@code GET /dispatcher/_snapshot}. Returns the dispatcher's
 *       current FE pool view in round-robin order with per-host liveness and consecutive-failure
 *       counts. No FE traffic, no master traffic, no side effects.</li>
 *   <li><b>{@link #dryRun}</b> — {@code POST /dispatcher/_dryrun/<spec.path>}. Runs the real
 *       chunk-assembly pipeline ({@link BatchSplitter}, {@link BatchChunkBuilder}) against the
 *       request body and returns the resulting sub-batch bodies as JSON instead of fanning out.
 *       With {@code ?pre_assign=true} (or with config default true) it does call
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
    private final ObjectMapper mapper;

    public DispatcherInspectionHandler(DispatchConfig cfg,
                                       DispatcherFePoolRefresher refresher,
                                       FeHealthChecker healthChecker,
                                       BatchScheduleClient batchScheduleClient,
                                       ObjectMapper mapper) {
        this.cfg = cfg;
        this.refresher = refresher;
        this.healthChecker = healthChecker;
        this.batchScheduleClient = batchScheduleClient;
        this.mapper = mapper;
    }

    // ───────────────────────── snapshot ─────────────────────────

    /**
     * Build the snapshot from a single source-of-truth snapshot of the URL list, so the response
     * is internally consistent even if the FE pool refreshes mid-build. Per-host
     * {@code alive}/{@code consecFails} reads still route through the shared {@link FeHealthChecker}
     * concurrent map — a host's counter may tick between two iterations, which is fine for a
     * diagnostic endpoint.
     */
    public Mono<ServerResponse> snapshot(ServerRequest request) {
        List<String> urls = refresher.source().get();
        ObjectNode root = mapper.createObjectNode();
        ObjectNode fePool = root.putObject("fePool");
        fePool.put("serviceId", cfg.getFePoolServiceId());
        fePool.put("size", urls.size());
        ArrayNode hosts = fePool.putArray("hosts");
        for (String url : urls) {
            ObjectNode host = hosts.addObject();
            host.put("url", url);
            host.put("alive", healthChecker.isAlive(url));
            host.put("consecFails", healthChecker.consecFails(url));
        }
        return ServerResponse.ok().contentType(MediaType.APPLICATION_JSON).bodyValue(root);
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
        return request.bodyToMono(JsonNode.class).flatMap(body -> {
            if (!(body instanceof ObjectNode obj)) {
                return badRequest("expected a JSON object body");
            }
            JsonNode arr = obj.get(spec.getRequestArrayField());
            if (arr == null || !arr.isArray()) {
                return badRequest("missing or non-array field: " + spec.getRequestArrayField());
            }
            return buildDryRunResponse(spec, obj, (ArrayNode) arr, effectivePreAssign);
        }).onErrorResume(this::handleDryRunException);
    }

    /**
     * Split caller-facing failure into user error vs server error. A malformed JSON body —
     * surfaced by Spring as {@link DecodingException} — is a 400 the caller can fix by editing
     * the payload. Anything else (NPE inside the assembly pipeline, unexpected coordinator
     * throw, broken contract) is a 500 we log loudly so operators don't mistake it for input
     * validation noise. Dry-run is a diagnostic surface, so mixing both classes into one status
     * would defeat the diagnostic purpose.
     */
    private Mono<ServerResponse> handleDryRunException(Throwable e) {
        if (e instanceof DecodingException) {
            return badRequest("malformed JSON body: " + e.getMessage());
        }
        Logger.warn("dispatcher dry-run unexpected error: {}: {}",
                e.getClass().getSimpleName(), e.getMessage());
        ObjectNode err = mapper.createObjectNode();
        err.put("error", "dryrun_internal_error");
        err.put("message", e.getClass().getSimpleName() + ": " + e.getMessage());
        return ServerResponse.status(500).contentType(MediaType.APPLICATION_JSON).bodyValue(err);
    }

    /**
     * Strip the {@link #DRYRUN_URI_PREFIX} from the raw path and return the trailing FE path.
     * Uses the raw path (pre URL-decode) so a registered path containing %-encoded chars
     * round-trips identically. Returns {@code null} when the URI does not start with the prefix
     * or has no trailing segment — both cases the caller maps to 400.
     */
    private String extractFePath(ServerRequest request) {
        String raw = request.uri().getRawPath();
        if (raw == null || !raw.startsWith(DRYRUN_URI_PREFIX)) {
            return null;
        }
        String tail = raw.substring(DRYRUN_URI_PREFIX.length());
        return tail.isEmpty() ? null : tail;
    }

    /**
     * Resolve effective pre-assign for this dry-run: explicit query param wins, otherwise fall
     * back to the dispatcher's boot-time config. {@code ?pre_assign=true} and
     * {@code ?pre_assign=false} are the only accepted overrides; any other value is silently
     * treated as "use config default" rather than throwing — dry-run is a diagnostic, not a place
     * to fail a request on a typo'd query param.
     */
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

    /**
     * Assembles the dry-run response: split + buildChunkBodies, then optionally call
     * {@link BatchScheduleClient} and stamp. Mirrors {@link GenericBatchHandler}'s empty-array
     * short-circuit: when the input array is empty, never call {@link BatchScheduleClient} with
     * {@code batchCount=0} — dry-run must not have RR-cursor side effects on master for a request
     * that would have been 200-empty on the real path.
     */
    private Mono<ServerResponse> buildDryRunResponse(BatchEndpointSpec spec, ObjectNode envelope,
                                                     ArrayNode arr, boolean effectivePreAssign) {
        List<ArrayNode> chunks = BatchSplitter.split(arr, cfg.subBatchSpec(), mapper);
        List<ObjectNode> chunkBodies = BatchChunkBuilder.buildChunkBodies(
                envelope, chunks, spec.getRequestArrayField());
        boolean shouldResolveTargets = effectivePreAssign && batchScheduleClient != null && !chunks.isEmpty();
        Mono<List<BatchScheduleTarget>> targetsMono = shouldResolveTargets
                ? batchScheduleClient.requestTargets(chunks.size())
                : Mono.just(List.of());
        return targetsMono.map(targets -> {
            BatchChunkBuilder.stampPreAssignedBe(chunkBodies, targets);
            ObjectNode out = mapper.createObjectNode();
            out.put("path", spec.getPath());
            out.put("splitMode", cfg.getSubBatch());
            out.put("totalItems", arr.size());
            out.put("chunkCount", chunks.size());
            out.put("preAssignConfigDefault", cfg.isPreAssignBe());
            out.put("preAssignEffective", effectivePreAssign);
            ArrayNode targetsOut = out.putArray("preAssignTargets");
            for (BatchScheduleTarget t : targets) {
                ObjectNode addr = targetsOut.addObject();
                addr.put("role", t.getRole().name());
                addr.put("ip", t.getServerIp());
                addr.put("httpPort", t.getHttpPort());
                addr.put("grpcPort", t.getGrpcPort());
            }
            ArrayNode chunksOut = out.putArray("chunks");
            chunkBodies.forEach(chunksOut::add);
            return out;
        }).flatMap(out -> ServerResponse.ok().contentType(MediaType.APPLICATION_JSON).bodyValue(out));
    }

    private Mono<ServerResponse> badRequest(String message) {
        ObjectNode err = mapper.createObjectNode();
        err.put("error", "invalid_inspection_request");
        err.put("message", message);
        return ServerResponse.badRequest().contentType(MediaType.APPLICATION_JSON).bodyValue(err);
    }
}
