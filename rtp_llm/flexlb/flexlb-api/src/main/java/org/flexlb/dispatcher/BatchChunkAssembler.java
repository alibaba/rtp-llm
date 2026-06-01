package org.flexlb.dispatcher;

import com.alibaba.fastjson2.JSON;
import com.alibaba.fastjson2.JSONArray;
import com.alibaba.fastjson2.JSONObject;
import org.flexlb.dao.loadbalance.BatchScheduleTarget;
import org.flexlb.dispatcher.SubBatchSpec;

import java.util.ArrayList;
import java.util.List;

/**
 * Pure-function helpers for chunk assembly on the dispatcher batch path. Splits the request
 * array per {@link SubBatchSpec}, builds the per-chunk request body (deep-copy of the envelope
 * with the chunk slice swapped in), stamps {@code generate_config.force_batch} and any
 * pre-resolved BE targets. The mirror of {@code BatchSplitter} + {@code BatchChunkBuilder} on
 * the Jackson side; behavior is intentionally identical so the equivalence test holds.
 *
 * <p>Deep copy uses {@code JSON.parseObject(JSON.toJSONBytes(envelope))} — fastjson2's
 * {@link JSONObject#clone()} is shallow and would let chunks share a single
 * {@code generate_config}, which the per-chunk {@code force_batch} / {@code role_addrs} writes
 * would mutate across siblings. Reparse cost on a typical envelope is microseconds.
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
     * Builds per-chunk request bodies. Each is a deep copy of {@code envelope} with the
     * {@code requestArrayField} replaced by the chunk slice and {@code generate_config.force_batch}
     * stamped per {@link #injectForceBatch} contract.
     */
    public static List<JSONObject> buildChunkBodies(JSONObject envelope, List<JSONArray> chunks,
                                                    String requestArrayField) {
        List<JSONObject> chunkBodies = new ArrayList<>(chunks.size());
        for (JSONArray chunk : chunks) {
            JSONObject copy = deepCopy(envelope);
            copy.put(requestArrayField, chunk);
            injectForceBatch(copy);
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
        JSONObject gc = chunkBody.getJSONObject("generate_config");
        if (gc == null) {
            gc = new JSONObject();
            chunkBody.put("generate_config", gc);
        }
        if (!gc.containsKey("force_batch")) {
            gc.put("force_batch", true);
        }
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
            JSONObject chunkBody = chunkBodies.get(i);
            JSONObject gc = chunkBody.getJSONObject("generate_config");
            if (gc == null) {
                gc = new JSONObject();
                chunkBody.put("generate_config", gc);
            }
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

    private static JSONObject deepCopy(JSONObject source) {
        return JSON.parseObject(JSON.toJSONBytes(source));
    }
}
