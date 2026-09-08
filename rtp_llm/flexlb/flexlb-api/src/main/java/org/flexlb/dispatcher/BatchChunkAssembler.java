package org.flexlb.dispatcher;

import com.alibaba.fastjson2.JSON;
import com.alibaba.fastjson2.JSONArray;
import com.alibaba.fastjson2.JSONObject;
import com.alibaba.fastjson2.JSONWriter;
import org.flexlb.dao.loadbalance.BatchScheduleTarget;

import java.util.ArrayList;
import java.util.List;

/**
 * Pure-function helpers for chunk assembly on the dispatcher batch path. Splits the request
 * array per {@link SubBatchSpec}, builds the per-chunk request body (shallow copy of the
 * envelope with the chunk slice swapped in and a fresh {@code generate_config} per chunk),
 * normalizes the legacy {@code generation_config} alias before stamping
 * {@code generate_config.force_batch} (only on the {@code prompt_batch} generation endpoints)
 * and any pre-resolved BE targets.
 *
 * <p>Per-chunk isolation strategy: every chunk gets a shallow-copy of the top-level envelope
 * (so {@code model} and other non-mutated fields share references) and a per-chunk copy of
 * {@code generate_config} (so the per-chunk {@code force_batch} / {@code role_addrs} writes
 * land on private objects). Caller-supplied {@code role_addrs} is reserved and removed at this
 * defense-in-depth boundary; the handler rejects it before assembly. Copying only the small
 * {@code generate_config} sub-object keeps
 * assembly O(chunk_count) instead of O(envelope_size × chunk_count) on the whole tree.
 */
public final class BatchChunkAssembler {

    /** Conservative allowance for dispatcher-owned role_addrs and small rewrite fields. */
    private static final int ROUTING_STAMP_RESERVE_BYTES = 1024;

    private BatchChunkAssembler() {}

    /**
     * Spec-aware split entry point. Dispatches to {@link #splitArray} or {@link #splitByCount}
     * based on {@link SubBatchSpec.Mode}. Empty input returns an empty list.
     */
    public static List<JSONArray> split(JSONArray arr, SubBatchSpec spec) {
        return switch (spec.mode()) {
            case SIZE -> splitArray(arr, spec.value());
            case COUNT -> splitByCount(arr, spec.value());
        };
    }

    /** Returns the number of chunks without allocating them. */
    public static int chunkCount(int total, SubBatchSpec spec) {
        if (total < 0) {
            throw new IllegalArgumentException("total must be >= 0, got " + total);
        }
        if (spec == null || spec.value() < 1) {
            throw new IllegalArgumentException("subBatch spec value must be >= 1");
        }
        if (total == 0) {
            return 0;
        }
        return switch (spec.mode()) {
            // 1 + (n - 1) / size is mathematically ceil(n / size) without int overflow.
            case SIZE -> 1 + (total - 1) / spec.value();
            case COUNT -> Math.min(total, spec.value());
        };
    }

    /**
     * Projects total FE-bound bytes without materializing every repeated envelope. JSON array
     * slices add one bracket/comma byte per split; all non-array fields repeat once per chunk.
     * A small per-chunk reserve covers dispatcher-owned target stamps added after allocation.
     */
    public static long projectedOutboundBytes(JSONObject envelope, JSONArray requestArray,
                                              int chunkCount, BatchEndpointSpec spec) {
        return projectedOutboundBytes(envelope, requestArray, chunkCount, spec, true);
    }

    /**
     * Policy-aware variant whose template exactly matches the force-batch value sent by the
     * production handler.
     */
    public static long projectedOutboundBytes(JSONObject envelope, JSONArray requestArray,
                                              int chunkCount, BatchEndpointSpec spec,
                                              boolean atomicBatchAllowed) {
        if (chunkCount < 1) {
            return 0;
        }
        List<JSONObject> templateBodies = buildChunkBodies(
                envelope, List.of(new JSONArray()), spec.getRequestArrayField(),
                atomicBatchAllowed);
        spec.prepareChunkBodies(envelope, templateBodies);
        JSONWriter.Feature[] features = spec.isFanoutWriteNulls()
                ? new JSONWriter.Feature[] {JSONWriter.Feature.WriteNulls}
                : new JSONWriter.Feature[0];
        long templateBytes = JSON.toJSONBytes(templateBodies.getFirst(), features).length;
        long arrayBytes = JSON.toJSONBytes(requestArray, features).length;
        long repeatedEnvelope = saturatingMultiply(Math.max(0, templateBytes - 2), chunkCount);
        long slicedArrays = saturatingAdd(arrayBytes, chunkCount - 1L);
        long routingReserve = saturatingMultiply(ROUTING_STAMP_RESERVE_BYTES, chunkCount);
        return saturatingAdd(saturatingAdd(repeatedEnvelope, slicedArrays), routingReserve);
    }

    private static long saturatingMultiply(long left, long right) {
        if (left == 0 || right == 0) {
            return 0;
        }
        return left > Long.MAX_VALUE / right ? Long.MAX_VALUE : left * right;
    }

    private static long saturatingAdd(long left, long right) {
        return left > Long.MAX_VALUE - right ? Long.MAX_VALUE : left + right;
    }

    /**
     * Splits into ordered chunks of at most {@code chunkSize}. Last chunk may be shorter.
     * Items are shared by reference with the source array — callers must not mutate them.
     */
    public static List<JSONArray> splitArray(JSONArray arr, int chunkSize) {
        // Not an assert: assertions are disabled by default in production, so a zero would slip
        // through to the chunk-count division and surface as an ArithmeticException instead.
        if (chunkSize < 1) {
            throw new IllegalArgumentException("chunkSize must be >= 1, got " + chunkSize);
        }
        int n = arr.size();
        if (n == 0) {
            return List.of();
        }
        int chunks = 1 + (n - 1) / chunkSize;
        List<JSONArray> out = new ArrayList<>(chunks);
        for (int c = 0; c < chunks; c++) {
            int start = c * chunkSize;
            int end = start + Math.min(chunkSize, n - start);
            JSONArray chunk = new JSONArray(end - start);
            for (int i = start; i < end; i++) {
                chunk.add(arr.get(i));
            }
            out.add(chunk);
        }
        return out;
    }

    /**
     * Splits into at most {@code requestedCount} ordered chunks with the remainder front-loaded
     * onto the leading chunks. If {@code total < requestedCount} the count is clamped to total
     * so no empty chunk is emitted.
     */
    public static List<JSONArray> splitByCount(JSONArray arr, int requestedCount) {
        // See splitArray: a public pure function must fail explicitly, not rely on -ea.
        if (requestedCount < 1) {
            throw new IllegalArgumentException("requestedCount must be >= 1, got " + requestedCount);
        }
        int n = arr.size();
        if (n == 0) {
            return List.of();
        }
        int chunks = Math.min(requestedCount, n);
        int base = n / chunks;
        int remainder = n % chunks;
        List<JSONArray> out = new ArrayList<>(chunks);
        int cursor = 0;
        for (int c = 0; c < chunks; c++) {
            int size = base + (c < remainder ? 1 : 0);
            JSONArray chunk = new JSONArray(size);
            for (int i = 0; i < size; i++) {
                chunk.add(arr.get(cursor + i));
            }
            out.add(chunk);
            cursor += size;
        }
        return out;
    }

    /**
     * Builds per-chunk request bodies. Each is a <em>shallow</em> copy of {@code envelope} with
     * the {@code requestArrayField} replaced by the chunk slice and the effective generation
     * config replaced by a per-chunk copy, then
     * {@code force_batch} stamped per {@link #injectForceBatch} contract. {@code force_batch}
     * is only stamped on the {@code prompt_batch} generation endpoints (root {@code /} and
     * {@code /batch_infer}); it is an rtp_llm generation {@code generate_config} flag with no
     * meaning for the embedding / OpenAI-chat batch shapes, whose chunk bodies carry no
     * {@code generate_config} of their own.
     *
     * <p>{@code generate_config} is copied per chunk because it's the one sub-tree that per-chunk
     * writes ({@code force_batch}, dispatcher-owned {@code role_addrs}) mutate. The legacy
     * {@code generation_config} alias is normalized to that canonical name on generation chunks;
     * otherwise injecting an empty canonical object would make FE ignore every caller setting in
     * the alias. A shallow {@code new JSONObject(sourceGc)} isolates the top-level scalars that get
     * written, and reserved caller routing fields are removed. Every other top-level envelope
     * field is either replaced wholesale ({@code requestArrayField}) or never written per chunk
     * ({@code model}, etc.), so sharing references is safe and cheap.
     */
    public static List<JSONObject> buildChunkBodies(JSONObject envelope, List<JSONArray> chunks,
                                                    String requestArrayField) {
        return buildChunkBodies(envelope, chunks, requestArrayField, true);
    }

    /**
     * Builds chunk bodies while allowing the caller to disable atomic backend batching. Active
     * traffic policies require per-item routing because one chunk can legitimately span multiple
     * worker groups; in that case {@code force_batch=false} overrides any caller value.
     */
    public static List<JSONObject> buildChunkBodies(JSONObject envelope, List<JSONArray> chunks,
                                                    String requestArrayField,
                                                    boolean atomicBatchAllowed) {
        boolean stampForceBatch = BatchEndpointSpec.PROMPT_BATCH_FIELD.equals(requestArrayField);
        String sourceGcKey = effectiveGenerateConfigKey(envelope);
        JSONObject sourceGc = sourceGcKey == null
                ? null : (JSONObject) envelope.get(sourceGcKey);
        List<JSONObject> chunkBodies = new ArrayList<>(chunks.size());
        for (JSONArray chunk : chunks) {
            JSONObject copy = new JSONObject(envelope);
            copy.put(requestArrayField, chunk);
            // Defense in depth: HTTP handlers reject this before assembly, but pure helper users
            // must not accidentally forward a top-level routing override either.
            copy.remove("role_addrs");
            if (sourceGc != null) {
                JSONObject gc = new JSONObject(sourceGc);
                gc.remove("role_addrs");
                copy.put(sourceGcKey, gc);
            }
            if (stampForceBatch) {
                if ("generation_config".equals(sourceGcKey)) {
                    copy.put("generate_config", copy.remove("generation_config"));
                } else {
                    // FE gives generate_config precedence when both spellings are present. Drop
                    // the ignored alias so each chunk has one unambiguous mutable config object.
                    copy.remove("generation_config");
                }
                if (atomicBatchAllowed) {
                    injectForceBatch(copy);
                } else {
                    // RequestExtractor applies top-level GenerateConfig fields after the nested
                    // object, so leaving a caller's top-level force_batch=true would undo this
                    // policy-required fallback.
                    copy.remove("force_batch");
                    ensureGenerateConfig(copy).put("force_batch", false);
                }
            }
            chunkBodies.add(copy);
        }
        return chunkBodies;
    }

    /**
     * Stamps {@code generate_config.force_batch=true} unless the user already supplied either
     * value. A user-supplied {@code force_batch=false} is treated as a legitimate opt-out
     * (e.g. for scheduler interleaving measurements) and must not be overwritten.
     */
    public static void injectForceBatch(JSONObject chunkBody) {
        JSONObject gc = ensureGenerateConfig(chunkBody);
        if (!gc.containsKey("force_batch")) {
            gc.put("force_batch", true);
        }
    }

    /**
     * Whether a target can be stamped into {@code generate_config.role_addrs}: FE's gRPC
     * pre-assignment needs both a gRPC port and a role. Embedding (ARPC-only) or role-less
     * targets are not pre-assignable and fall back to FE's own scheduling.
     */
    public static boolean isPreAssignable(BatchScheduleTarget target) {
        return target.getGrpcPort() != null && target.getRole() != null;
    }

    /**
     * Replaces {@code generate_config.role_addrs} with each chunk's pre-resolved BE target.
     * Per-addr wire shape matches Python {@code rtp_llm.config.generate_config.RoleAddr}:
     * {@code {role, ip, http_port, grpc_port}}. Note {@code ip} (not {@code server_ip} from
     * {@link BatchScheduleTarget}'s wire shape) — the rename matches the FE-side schema.
     *
     * <p>Tolerates a short target list: only the first {@code min(chunkBodies, targets)}
     * chunks get stamped. Replacement is intentional: the dispatcher assignment is
     * authoritative, and an external address must never win by appearing first in the array.
     */
    public static void stampPreAssignedBe(List<JSONObject> chunkBodies,
                                          List<BatchScheduleTarget> targets) {
        if (targets.isEmpty()) {
            return;
        }
        int max = Math.min(chunkBodies.size(), targets.size());
        for (int i = 0; i < max; i++) {
            BatchScheduleTarget target = targets.get(i);
            if (!isPreAssignable(target)) {
                // role_addrs is FE's gRPC pre-assignment mechanism; targets without a gRPC
                // slot (embedding engines) or without a role cannot be pre-assigned through
                // it — skip rather than fail, pre-assignment never blocks traffic.
                continue;
            }
            JSONObject chunkBody = chunkBodies.get(i);
            JSONObject gc = ensureGenerateConfig(chunkBody);
            gc.put("role_addrs", preAssignedRoleAddrs(target));
        }
    }

    /** Builds the exact FE-side wire value used for one authoritative BE assignment. */
    static JSONArray preAssignedRoleAddrs(BatchScheduleTarget target) {
        JSONArray roleAddrs = new JSONArray(1);
        JSONObject addr = new JSONObject();
        addr.put("role", target.getRole().name());
        addr.put("ip", target.getServerIp());
        addr.put("http_port", target.getHttpPort());
        addr.put("grpc_port", target.getGrpcPort());
        roleAddrs.add(addr);
        return roleAddrs;
    }

    /**
     * Validates every generation-config spelling FE accepts, shared by production and dry-run
     * handlers. FE also promotes GenerateConfig fields found at the request root after parsing the
     * nested object, so top-level {@code role_addrs} must be reserved too.
     */
    public static String validateGenerateConfig(JSONObject body) {
        if (body.containsKey("role_addrs")) {
            return "top-level role_addrs is reserved for dispatcher pre-assignment";
        }
        for (String key : List.of("generate_config", "generation_config")) {
            if (!body.containsKey(key)) {
                continue;
            }
            Object value = body.get(key);
            if (!(value instanceof JSONObject gc)) {
                return key + " must be a JSON object";
            }
            if (gc.containsKey("role_addrs")) {
                return key + ".role_addrs is reserved for dispatcher pre-assignment";
            }
        }
        return null;
    }

    /** Canonical config wins exactly as it does in FE; otherwise use the legacy alias. */
    private static String effectiveGenerateConfigKey(JSONObject body) {
        if (body.get("generate_config") instanceof JSONObject) {
            return "generate_config";
        }
        if (body.get("generation_config") instanceof JSONObject) {
            return "generation_config";
        }
        return null;
    }

    private static JSONObject ensureGenerateConfig(JSONObject chunkBody) {
        JSONObject gc = chunkBody.getJSONObject("generate_config");
        if (gc == null) {
            gc = new JSONObject();
            chunkBody.put("generate_config", gc);
        }
        return gc;
    }
}
