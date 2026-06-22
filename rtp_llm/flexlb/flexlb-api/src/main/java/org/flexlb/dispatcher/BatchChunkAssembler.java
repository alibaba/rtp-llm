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
 * stamps {@code generate_config.force_batch} (only on the {@code prompt_batch} generation
 * endpoints) and any pre-resolved BE targets.
 *
 * <p>Per-chunk isolation strategy: every chunk gets a shallow-copy of the top-level envelope
 * (so {@code model} and other non-mutated fields share references) and a per-chunk copy of
 * {@code generate_config} (so the per-chunk {@code force_batch} / {@code role_addrs} writes
 * land on private objects). The gc copy is shallow by default — isolating the top-level scalars
 * that get written — and deep only when the source carries {@code role_addrs}, whose per-chunk
 * append targets a nested array. Copying only the small {@code generate_config} sub-object keeps
 * assembly O(chunk_count) instead of O(envelope_size × chunk_count) on the whole tree.
 */
public final class BatchChunkAssembler {

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

    /**
     * Splits into ordered chunks of at most {@code chunkSize}. Last chunk may be shorter.
     * Items are shared by reference with the source array — callers must not mutate them.
     */
    public static List<JSONArray> splitArray(JSONArray arr, int chunkSize) {
        assert chunkSize >= 1 : "chunkSize must be >= 1, got " + chunkSize;
        int n = arr.size();
        if (n == 0) {
            return List.of();
        }
        int chunks = (n + chunkSize - 1) / chunkSize;
        List<JSONArray> out = new ArrayList<>(chunks);
        for (int c = 0; c < chunks; c++) {
            int start = c * chunkSize;
            int end = Math.min(start + chunkSize, n);
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
        assert requestedCount >= 1 : "requestedCount must be >= 1, got " + requestedCount;
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
     * the {@code requestArrayField} replaced by the chunk slice and {@code generate_config}
     * replaced by a per-chunk copy of the source {@code generate_config}, then
     * {@code force_batch} stamped per {@link #injectForceBatch} contract. {@code force_batch}
     * is only stamped on the {@code prompt_batch} generation endpoints (root {@code /} and
     * {@code /batch_infer}); it is an rtp_llm generation {@code generate_config} flag with no
     * meaning for the embedding / OpenAI-chat batch shapes, whose chunk bodies carry no
     * {@code generate_config} of their own.
     *
     * <p>{@code generate_config} is copied per chunk because it's the one sub-tree that per-chunk
     * writes ({@code force_batch}, {@code role_addrs} append) mutate. A shallow {@code new
     * JSONObject(sourceGc)} isolates only the top-level scalars that get written; when the source
     * carries {@code role_addrs} the per-chunk append targets a nested array, so that case is
     * deep-cloned instead. Every other top-level envelope field is either replaced wholesale
     * ({@code requestArrayField}) or never written per chunk ({@code model}, etc.), so sharing
     * references is safe and cheap. Reparsing the entire envelope per chunk — the prior
     * implementation — was O(envelope_size × chunk_count) and blew up to ~500ms CPU on a 730KB
     * envelope at 100 chunks.
     */
    public static List<JSONObject> buildChunkBodies(JSONObject envelope, List<JSONArray> chunks,
                                                    String requestArrayField) {
        JSONObject sourceGc = envelope.getJSONObject("generate_config");
        boolean gcNeedsDeepCopy = sourceGc != null && sourceGc.containsKey("role_addrs");
        boolean stampForceBatch = BatchEndpointSpec.PROMPT_BATCH_FIELD.equals(requestArrayField);
        List<JSONObject> chunkBodies = new ArrayList<>(chunks.size());
        for (JSONArray chunk : chunks) {
            JSONObject copy = new JSONObject(envelope);
            copy.put(requestArrayField, chunk);
            if (sourceGc != null) {
                copy.put("generate_config", gcNeedsDeepCopy
                        ? BatchBodyParser.deepCopy(sourceGc)
                        : new JSONObject(sourceGc));
            }
            if (stampForceBatch) {
                injectForceBatch(copy);
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
     * Appends each chunk's pre-resolved BE target into {@code generate_config.role_addrs}.
     * Per-addr wire shape matches Python {@code rtp_llm.config.generate_config.RoleAddr}:
     * {@code {role, ip, http_port, grpc_port}}. Note {@code ip} (not {@code server_ip} from
     * {@link BatchScheduleTarget}'s wire shape) — the rename matches the FE-side schema.
     *
     * <p>Tolerates a short target list: only the first {@code min(chunkBodies, targets)}
     * chunks get stamped. User-supplied {@code role_addrs} entries are preserved; the
     * dispatcher's resolved target is appended after them.
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
            JSONArray roleAddrs = gc.getJSONArray("role_addrs");
            if (roleAddrs == null) {
                roleAddrs = new JSONArray();
                gc.put("role_addrs", roleAddrs);
            }
            JSONObject addr = new JSONObject();
            addr.put("role", target.getRole().name());
            addr.put("ip", target.getServerIp());
            addr.put("http_port", target.getHttpPort());
            addr.put("grpc_port", target.getGrpcPort());
            roleAddrs.add(addr);
        }
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
