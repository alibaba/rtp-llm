package org.flexlb.dispatcher;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.flexlb.dao.loadbalance.BatchScheduleTarget;

import java.util.ArrayList;
import java.util.List;

/**
 * Pure-function helpers for chunk-body assembly: build the per-chunk request body, inject
 * {@code force_batch=true}, stamp pre-assigned BE targets into {@code generate_config.role_addrs}.
 * Shared verbatim by {@link GenericBatchHandler} (real fanout) and {@code DispatcherInspectionHandler}
 * (dry-run inspection) so the inspectable output is byte-equivalent to what production would have
 * sent. Splitting the array itself lives in {@link BatchSplitter#split} — keep this class focused
 * on operations that mutate the resulting chunk bodies.
 *
 * <p>No state, no Spring wiring. All transforms operate on {@code JsonNode} trees and
 * mutate them in place where the parent owns the lifetime; otherwise return fresh nodes
 * built off {@link ObjectMapper}.
 */
final class BatchChunkBuilder {

    private BatchChunkBuilder() {
    }

    /**
     * Builds the per-chunk request body: deep-copies the original envelope (preserving any
     * top-level fields like {@code model}, {@code generate_config}, …), replaces the request
     * array with the chunk slice, and stamps {@code generate_config.force_batch=true} per
     * {@link #injectForceBatch} contract.
     */
    static List<ObjectNode> buildChunkBodies(ObjectNode envelope, List<ArrayNode> chunks,
                                             String requestArrayField) {
        List<ObjectNode> chunkBodies = new ArrayList<>(chunks.size());
        for (ArrayNode chunk : chunks) {
            ObjectNode copy = envelope.deepCopy();
            copy.set(requestArrayField, chunk);
            injectForceBatch(copy);
            chunkBodies.add(copy);
        }
        return chunkBodies;
    }

    /**
     * Stamps {@code generate_config.force_batch=true} into the chunk body unless the user
     * already set it (either value). Matches the legacy ft_proxy convention that the
     * dispatch layer — not the user — tags batch traffic with {@code force_batch} so the
     * per-chunk FE's {@code FIFOScheduler} groups the chunk's prompts into a single
     * scheduling slot. A user-supplied {@code force_batch=false} is a legitimate opt-out
     * (e.g. measuring scheduler interleaving) and must not be overwritten.
     */
    static void injectForceBatch(ObjectNode chunkBody) {
        JsonNode gcNode = chunkBody.get("generate_config");
        ObjectNode gc;
        if (gcNode instanceof ObjectNode existing) {
            gc = existing;
        } else {
            gc = chunkBody.putObject("generate_config");
        }
        if (!gc.has("force_batch")) {
            gc.put("force_batch", true);
        }
    }

    /**
     * Appends each chunk's pre-resolved BE target into {@code generate_config.role_addrs} —
     * the same field FE's existing {@code backend_rpc_server_visitor.route_ips} skips master
     * on when set. Per-addr wire shape matches Python {@code rtp_llm.config.generate_config.RoleAddr}:
     * {@code {role, ip, http_port, grpc_port}}. Note {@code ip} (not {@code server_ip} as in
     * {@link BatchScheduleTarget}'s wire shape) — the rename aligns with the FE-side schema.
     *
     * <p>Tolerates a short target list: only the first {@code min(chunkBodies, targets)}
     * chunks get stamped. User-supplied {@code role_addrs} entries are preserved and the
     * dispatcher's resolved target is appended after them.
     */
    static void stampPreAssignedBe(List<ObjectNode> chunkBodies, List<BatchScheduleTarget> targets) {
        if (targets.isEmpty()) {
            return;
        }
        int max = Math.min(chunkBodies.size(), targets.size());
        for (int i = 0; i < max; i++) {
            BatchScheduleTarget target = targets.get(i);
            ObjectNode chunkBody = chunkBodies.get(i);
            ObjectNode gc = chunkBody.get("generate_config") instanceof ObjectNode existing
                    ? existing
                    : chunkBody.putObject("generate_config");
            ArrayNode roleAddrs = gc.get("role_addrs") instanceof ArrayNode existingAddrs
                    ? existingAddrs
                    : gc.putArray("role_addrs");
            ObjectNode addr = roleAddrs.addObject();
            addr.put("role", target.getRole().name());
            addr.put("ip", target.getServerIp());
            addr.put("http_port", target.getHttpPort());
            addr.put("grpc_port", target.getGrpcPort());
        }
    }
}
