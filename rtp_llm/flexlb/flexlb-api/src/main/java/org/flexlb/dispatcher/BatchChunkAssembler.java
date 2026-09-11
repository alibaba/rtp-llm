package org.flexlb.dispatcher;

import com.alibaba.fastjson2.JSONArray;
import com.alibaba.fastjson2.JSONObject;
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

    private BatchChunkAssembler() {}

    /** Ordered slices; count mode spreads the remainder over the first chunks. */
    public static List<JSONArray> split(JSONArray arr, SubBatchSpec spec) {
        int count = chunkCount(arr.size(), spec);
        List<JSONArray> chunks = new ArrayList<>(count);
        int cursor = 0;
        for (int i = 0; i < count; i++) {
            int size = spec.mode() == SubBatchSpec.Mode.SIZE
                    ? Math.min(spec.value(), arr.size() - cursor)
                    : arr.size() / count + (i < arr.size() % count ? 1 : 0);
            chunks.add(new JSONArray(arr.subList(cursor, cursor + size)));
            cursor += size;
        }
        return chunks;
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
     * Exact sum of serialized chunk bodies, shared by production and dry-run. Project with
     * one empty-array template before allocating repeated envelopes or resolving targets.
     */
    public static long projectedChunkBytes(JSONObject envelope, JSONArray requestArray,
                                           int chunkCount, BatchEndpointSpec spec,
                                           boolean atomicBatchAllowed,
                                           List<BatchScheduleTarget> targets) {
        if (chunkCount == 0) {
            return 0;
        }
        JSONObject template = chunkTemplate(envelope, spec, atomicBatchAllowed);
        long templateBytes = BatchBodyParser.serialize(template).length;
        long arrayBytes = BatchBodyParser.serialize(requestArray).length;
        long bytes = saturatingAdd(saturatingMultiply(templateBytes - 2, chunkCount),
                saturatingAdd(arrayBytes, chunkCount - 1L));
        for (int i = 0; i < Math.min(chunkCount, targets.size()); i++) {
            if (isPreAssignable(targets.get(i))) {
                // generate_config already contains force_batch: comma + \"role_addrs\": + value.
                bytes = saturatingAdd(bytes, 14L + BatchBodyParser.serialize(
                        preAssignedRoleAddrs(targets.get(i))).length);
            }
        }
        return bytes;
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

    /** Each chunk owns its envelope and mutable generation config; large read-only fields share references. */
    public static List<JSONObject> buildChunkBodies(JSONObject envelope, List<JSONArray> chunks,
                                                    BatchEndpointSpec spec,
                                                    boolean atomicBatchAllowed) {
        JSONObject template = chunkTemplate(envelope, spec, atomicBatchAllowed);
        String configKey = effectiveGenerateConfigKey(template);
        List<JSONObject> bodies = new ArrayList<>(chunks.size());
        for (JSONArray chunk : chunks) {
            JSONObject body = new JSONObject(template);
            body.put(spec.getRequestArrayField(), chunk);
            if (configKey != null) {
                body.put(configKey, new JSONObject(template.getJSONObject(configKey)));
            }
            bodies.add(body);
        }
        return bodies;
    }

    private static JSONObject chunkTemplate(JSONObject envelope, BatchEndpointSpec spec,
                                            boolean atomicBatchAllowed) {
        JSONObject copy = new JSONObject(envelope);
        copy.put(spec.getRequestArrayField(), new JSONArray());
        copy.remove("role_addrs");
        String configKey = effectiveGenerateConfigKey(envelope);
        if (configKey != null) {
            JSONObject gc = new JSONObject(envelope.getJSONObject(configKey));
            gc.remove("role_addrs");
            copy.put(configKey, gc);
        }
        if (spec.isPreAssignable()) {
            if ("generation_config".equals(configKey)) {
                copy.put("generate_config", copy.remove("generation_config"));
            } else {
                copy.remove("generation_config");
            }
            if (atomicBatchAllowed) {
                injectForceBatch(copy);
            } else {
                // FE promotes top-level config after nested config, so remove that override too.
                copy.remove("force_batch");
                ensureGenerateConfig(copy).put("force_batch", false);
            }
        }
        spec.prepareChunkBody(copy);
        return copy;
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
