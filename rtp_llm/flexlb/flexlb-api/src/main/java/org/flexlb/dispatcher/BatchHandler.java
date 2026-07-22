package org.flexlb.dispatcher;

import com.alibaba.fastjson2.JSONArray;
import com.alibaba.fastjson2.JSONObject;
import org.flexlb.dao.loadbalance.BatchScheduleTarget;
import org.flexlb.dao.pv.DispatchPvLogData;
import org.flexlb.util.Logger;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.core.io.buffer.DataBufferLimitException;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.server.ServerRequest;
import org.springframework.web.reactive.function.server.ServerResponse;
import reactor.core.publisher.Mono;
import reactor.core.publisher.SignalType;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Dispatcher batch handler. Reads each batch request body as raw bytes, parses with
 * fastjson2, splits the request array per {@link SubBatchSpec}, builds per-chunk bodies,
 * stamps any pre-assigned BE targets, fans out via {@link FanoutService}, and merges with
 * {@link ResponseMerger}.
 *
 * <p>Status mapping: 400 on a non-JSON-object body, passthrough disposition for registered
 * paths whose body is not a splittable batch, 200 on full or partial success, and on total
 * failure the chunks' shared FE 4xx when they agree on one — 500 otherwise.
 *
 * <p>Single-element batches still fan out as one chunk so partial-failure semantics stay
 * uniform; router-level rejection of non-batch traffic happens upstream.
 */
@Component
@ConditionalOnProperty(prefix = "dispatch", name = "fe-pool-service-id")
public class BatchHandler {

    private final FanoutService fanoutService;
    private final SubBatchSpec subBatch;
    private final BatchScheduleClient batchScheduleClient;
    private final PassthroughClient passthroughClient;
    private final DispatcherMetricsReporter metricsReporter;
    private final boolean preAssignBe;

    public BatchHandler(FanoutService fanoutService,
                        DispatchConfig cfg,
                        BatchScheduleClient batchScheduleClient,
                        PassthroughClient passthroughClient,
                        DispatcherMetricsReporter metricsReporter) {
        this.fanoutService = fanoutService;
        this.subBatch = cfg.getSubBatchSpec();
        this.batchScheduleClient = batchScheduleClient;
        this.passthroughClient = passthroughClient;
        this.metricsReporter = metricsReporter;
        this.preAssignBe = cfg.isPreAssignBe();
    }

    public Mono<ServerResponse> handle(ServerRequest request, BatchEndpointSpec spec) {
        DispatchPvLogData pv = DispatchPvLogData.batch(spec.getPath(), System.currentTimeMillis());
        AtomicBoolean delegatedToPassthrough = new AtomicBoolean(false);
        return request.bodyToMono(byte[].class).defaultIfEmpty(new byte[0]).flatMap(bytes -> {
            JSONObject body = BatchBodyParser.parseObject(bytes);
            if (body == null) {
                return badRequest("expected a JSON object body");
            }
            JSONArray arr = BatchBodyParser.findArrayField(body, spec.getRequestArrayField());
            if (!spec.isSplittableBatch(body, arr)) {
                // Registered path, but this body is not a splittable batch (absent array field,
                // non-batch-shaped array, or a whole-body companion field — see
                // BatchEndpointSpec#isSplittableBatch). Forward verbatim to one FE per the
                // registry contract. PassthroughClient emits its own pv record.
                delegatedToPassthrough.set(true);
                return passthroughClient.forward(request, bytes);
            }
            if (arr.isEmpty()) {
                JSONObject emptyEnvelope = new JSONObject();
                emptyEnvelope.put(spec.getResponseArrayField(), new JSONArray());
                return DispatcherResponses.jsonBytes(200, BatchBodyParser.serialize(emptyEnvelope));
            }
            // Chunk assembly copies generate_config per chunk, so a non-object value (e.g. a
            // string) is a deterministic client error — reject it here instead of letting the
            // JSONException fall into the catch-all and masquerade as a 500 dispatch failure.
            Object generateConfig = body.get("generate_config");
            if (generateConfig != null && !(generateConfig instanceof JSONObject)) {
                return badRequest("generate_config must be a JSON object");
            }
            pv.setTotalItems(arr.size());
            List<JSONArray> chunks = BatchChunkAssembler.split(arr, subBatch);
            pv.setChunkCount(chunks.size());
            List<JSONObject> chunkBodies = BatchChunkAssembler.buildChunkBodies(
                    body, chunks, spec.getRequestArrayField());
            return resolveTargets(chunks.size())
                    .flatMap(targets -> {
                        // BE role_addrs stamping stays gated on the toggle + endpoint support; FE
                        // assignment does not — it is always sourced from the master (below).
                        if (preAssignBe && spec.isPreAssignable()) {
                            BatchChunkAssembler.stampPreAssignedBe(chunkBodies, targets);
                        }
                        // Per-chunk FE the master assigned from its single global cursor. There is
                        // no local fallback by design: a chunk with no master fe_url fails visibly
                        // in fanout, so FE load stays fully attributable to the master cursor and
                        // never splits across a second, per-instance distribution.
                        List<String> preAssignedFeUrls = preAssignedFeUrls(targets);
                        long fanoutStart = System.currentTimeMillis();
                        // Relay the caller's end-to-end headers and query to every chunk: the split
                        // path must not silently change auth/tenancy/tracing semantics relative to
                        // the passthrough path for the same route.
                        return fanoutService.dispatchChunks(spec.getPath(), chunkBodies,
                                        preAssignedFeUrls, spec,
                                        request.headers().asHttpHeaders(), request.uri().getRawQuery())
                                .doOnNext(subs -> metricsReporter.reportFanoutRt(
                                        System.currentTimeMillis() - fanoutStart))
                                .map(subs -> ResponseMerger.merge(subs, spec))
                                .flatMap(merged -> {
                                    pv.setFailedChunks(merged.failedReasons().size());
                                    if (merged.allFailed()) {
                                        return errorResponse(merged);
                                    }
                                    return DispatcherResponses.jsonBytes(
                                            200, BatchBodyParser.serialize(merged.body()));
                                });
                    });
        }).onErrorResume(e -> {
            String errMsg = DispatcherResponses.briefReason(e);
            Logger.warn("dispatcher request failed: spec={}, err={}", spec.getPath(), errMsg);
            pv.setError(errMsg);
            if (e instanceof DataBufferLimitException) {
                // Body over spring.codec.max-in-memory-size is a deterministic client error;
                // a 500 would invite pointless retries and pollute the server error rate.
                return DispatcherResponses.error(413, "request_body_too_large",
                        "batch body exceeds the server limit; see MAX_IN_MEMORY_SIZE");
            }
            // Stable, non-revealing text: the exception message can carry the FE address or
            // upstream response detail, which must not cross the client boundary. The full
            // reason is in the WARN above and in pv.log.
            return DispatcherResponses.error(500, "dispatch_failed", "batch dispatch failed");
        }).doOnNext(resp -> pv.setHttpStatus(resp.rawStatusCode()))
          .doFinally(signal -> {
              if (!delegatedToPassthrough.get()) {
                  finalizePvRecord(pv, signal);
              }
          });
    }

    private void finalizePvRecord(DispatchPvLogData pv, SignalType signal) {
        int status = pv.getHttpStatus();
        String error = pv.getError();
        if (signal == SignalType.CANCEL && status == 0) {
            status = 499;
            error = error != null ? error : "client cancelled";
        }
        pv.finish(status, error);
        pv.emit();
        metricsReporter.reportRequest("batch", pv.getPath(), status, pv.getCostMs());
        if (pv.getChunkCount() > 0) {
            metricsReporter.reportBatchShape(pv.getPath(), pv.getTotalItems(), pv.getChunkCount());
        }
    }

    /**
     * The master's per-chunk FE assignment, index-aligned to {@code targets} (and thus to chunks).
     * A null entry — or an index past a short target list — means "no master FE for this chunk";
     * {@link FanoutService} fails such a chunk visibly rather than picking a local FE, so FE load
     * has exactly one source.
     */
    private static List<String> preAssignedFeUrls(List<BatchScheduleTarget> targets) {
        List<String> feUrls = new ArrayList<>(targets.size());
        for (BatchScheduleTarget target : targets) {
            feUrls.add(target.getFeUrl());
        }
        return feUrls;
    }

    private Mono<ServerResponse> badRequest(String message) {
        return DispatcherResponses.error(400, "invalid_batch_request", message);
    }

    private Mono<ServerResponse> errorResponse(ResponseMerger.MergedResponse merged) {
        JSONObject body = new JSONObject();
        body.put("error", "all_sub_batches_failed");
        // Item units, matching the success-path _partial_failure block; every item failed so
        // failed_count == total_count. total_chunks is sub-batch units.
        int failedItems = merged.failedIndices().size();
        body.put("failed_count", failedItems);
        body.put("total_count", failedItems);
        body.put("total_chunks", merged.totalChunks());
        JSONArray reasons = new JSONArray();
        merged.failedReasons().stream().distinct().forEach(reasons::add);
        body.put("failed_reasons", reasons);
        return DispatcherResponses.jsonBytes(merged.errorStatus(), BatchBodyParser.serialize(body));
    }

    /**
     * Resolves per-chunk targets from the master for every splittable batch. FE selection is
     * sourced solely from the master's single global cursor with no local fallback, so this
     * {@code /batch_schedule} round-trip is now unconditional — it also carries the per-chunk
     * {@code fe_url}, which is never wasted even on endpoints that ignore BE {@code role_addrs}
     * stamping (those merely skip the stamp, see {@link #handle}). All {@link BatchScheduleClient}
     * failure paths collapse to an empty list; the affected chunks then fail visibly in fanout
     * rather than silently falling onto a per-instance FE distribution.
     */
    private Mono<List<BatchScheduleTarget>> resolveTargets(int chunkCount) {
        long start = System.currentTimeMillis();
        return batchScheduleClient.requestTargets(chunkCount)
                .doOnNext(targets -> metricsReporter.reportPreassignRt(
                        System.currentTimeMillis() - start, !targets.isEmpty()));
    }
}
