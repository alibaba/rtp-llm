package org.flexlb.balance.scheduler;

import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.engine.grpc.RoleTypeProtoConverter;
import org.flexlb.telemetry.FlexlbTrace;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.concurrent.CompletionException;

import static com.google.common.base.Preconditions.checkArgument;

/** Builds EnqueueBatch requests and validates complete responses against the exact sent members. */
final class BatchEnqueueHelper {
    /** O(1) wire-size bound using the route's exact, immutable endpoint addresses. */
    static long payloadSizeUpperBound(RouteExecution route) {
        return route.ctx().data().getGenerateInputSize()
                + 64L
                + addressSizeUpperBound(route.prefillEndpoint().getIp())
                + (route.decodeEndpoint() == null
                        ? 0L
                        : addressSizeUpperBound(route.decodeEndpoint().getIp()));
    }

    private static long addressSizeUpperBound(String ip) {
        return 64L + (ip == null ? 0L : 3L * ip.length());
    }

    static EngineRpcService.EnqueueBatchRequestPB prepare(
            long batchId, List<Delivery.Member> claims) {
        try {
            var first = claims.getFirst();
            var config = first.owner().data().getConfig();
            var builder =
                    EngineRpcService.EnqueueBatchRequestPB.newBuilder()
                            .setBatchId(batchId)
                            .setFetchAttachTimeoutMs(
                                    config.getDispatcher().getFetchAttachTimeoutMs());
            var addresses = new RoleAddresses();
            long firstRank = first.route().prefillDpRank();
            var firstSlot =
                    EngineRpcService.EnqueueBatchDpSlotPB.newBuilder().setDpRank((int) firstRank);
            Map<Long, EngineRpcService.EnqueueBatchDpSlotPB.Builder> slots = null;
            for (Delivery.Member claim : claims) {
                var item = claim.route();
                long rank = item.prefillDpRank();
                var slot = firstSlot;
                if (rank != firstRank) {
                    if (slots == null) {
                        slots = new TreeMap<>();
                        slots.put(firstRank, firstSlot);
                    }
                    slot =
                            slots.computeIfAbsent(
                                    rank,
                                    key ->
                                            EngineRpcService.EnqueueBatchDpSlotPB.newBuilder()
                                                    .setDpRank(key.intValue()));
                }
                slot.addRequests(
                        EngineRpcService.EnqueueBatchExternalInputPB.newBuilder()
                                .setInput(input(item, addresses)));
            }
            if (slots == null) {
                builder.addDpSlots(firstSlot);
            } else {
                slots.values().forEach(builder::addDpSlots);
            }
            return builder.build();
        } catch (Exception failure) {
            throw new IllegalArgumentException(
                    "Batch request build failed: " + failure.getMessage(), failure);
        }
    }

    /** Remove abandoned members from an already complete immutable payload without parsing again. */
    static EngineRpcService.EnqueueBatchRequestPB retain(
            EngineRpcService.EnqueueBatchRequestPB prepared, List<Delivery.Member> sending) {
        var ids = new HashSet<String>();
        for (Delivery.Member claim : sending) {
            ids.add(claim.route().requestId());
        }
        var filtered = prepared.toBuilder().clearDpSlots();
        for (var slot : prepared.getDpSlotsList()) {
            var selected = slot.toBuilder().clearRequests();
            for (var member : slot.getRequestsList()) {
                if (ids.contains(org.flexlb.engine.grpc.RequestId.parse(member.getInput()))) {
                    selected.addRequests(member);
                }
            }
            if (selected.getRequestsCount() > 0) {
                filtered.addDpSlots(selected);
            }
        }
        return filtered.build();
    }

    /** Results have the exact sending-claim order. One malformed member invalidates the entire ACK. */
    static List<Delivery.Result> validate(
            long batchId,
            List<Delivery.Member> claims,
            EngineRpcService.EnqueueBatchResponsePB response,
            Throwable failure) {
        if (failure != null || response == null || response.getBatchId() != batchId) {
            Throwable problem =
                    failure instanceof CompletionException && failure.getCause() != null
                            ? failure.getCause()
                            : failure;
            if (problem == null) {
                problem =
                        new IllegalStateException(
                                "Invalid EnqueueBatch response for batch " + batchId);
            }
            return Collections.nCopies(claims.size(), Delivery.Result.uncertain(problem));
        }
        var expected = HashMap.<String, Integer>newHashMap(claims.size());
        var results = new Delivery.Result[claims.size()];
        var violations = new ArrayList<String>();
        for (int i = 0; i < claims.size(); i++) {
            String id = claims.get(i).route().requestId();
            if (expected.put(id, i) != null) {
                violations.add("duplicate sending request_id=" + id);
            }
        }
        for (var error : response.getErrorsList()) {
            long code = error.hasErrorInfo() ? error.getErrorInfo().getErrorCode() : 0L;
            String message =
                    error.hasErrorInfo()
                            ? error.getErrorInfo().getErrorMessage()
                            : "missing error_info";
            acknowledge(
                    expected,
                    results,
                    org.flexlb.engine.grpc.RequestId.parse(error),
                    Delivery.Result.prefillRejected(
                            new RuntimeException(
                                    "EnqueueBatch rejected request "
                                            + org.flexlb.engine.grpc.RequestId.parse(error)
                                            + " error_code="
                                            + code
                                            + ": "
                                            + message)),
                    violations);
        }
        for (var success : response.getSuccessesList()) {
            acknowledge(
                    expected,
                    results,
                    org.flexlb.engine.grpc.RequestId.parse(success),
                    Delivery.Result.delivered(),
                    violations);
        }
        expected.forEach(
                (id, index) -> {
                    if (index >= 0 && results[index] == null) {
                        violations.add("response is missing request_id=" + id);
                    }
                });
        if (!violations.isEmpty()) {
            return Collections.nCopies(
                    claims.size(),
                    Delivery.Result.uncertain(
                            new IllegalStateException(
                                    "Malformed EnqueueBatch response: "
                                            + String.join("; ", violations))));
        }
        return List.of(results);
    }

    private static void acknowledge(
            Map<String, Integer> expected,
            Delivery.Result[] results,
            String id,
            Delivery.Result result,
            List<String> violations) {
        String kind = result.status() == Delivery.Result.Status.DELIVERED ? "success" : "error";
        Integer index = expected.get(id);
        if (index == null || index < 0) {
            violations.add(kind + " references unknown request_id=" + id);
            int unknownKind = result.status() == Delivery.Result.Status.DELIVERED ? -1 : -2;
            if (index == null) {
                expected.put(id, unknownKind);
            } else {
                violations.add(
                        index == unknownKind
                                ? "duplicate " + kind + " for request_id=" + id
                                : "request_id appears in both success and error: " + id);
            }
            return;
        }
        var previous = results[index];
        if (previous != null) {
            violations.add(
                    previous.status() == result.status()
                            ? "duplicate " + kind + " for request_id=" + id
                            : "request_id appears in both success and error: " + id);
        } else {
            results[index] = result;
        }
    }

    private static EngineRpcService.GenerateInputPB input(
            RouteExecution item, RoleAddresses addresses) throws Exception {
        var data = item.ctx().data();
        var prepared = data.getGenerateInput();
        if (prepared == null) {
            throw new IllegalArgumentException(
                    "generateInputPb is missing for request " + item.requestId());
        }
        var input = prepared.toBuilder();
        checkArgument(
                java.util.Objects.equals(
                        org.flexlb.engine.grpc.RequestId.parse(input), item.requestId()),
                "request_id mismatch between schedule request and GenerateInputPB");
        if (FlexlbTrace.isEnabled() && data.getTraceContext() != null) {
            try {
                var carrier = FlexlbTrace.inject(data.getTraceContext());
                String parent = carrier.get("traceparent");
                if (parent != null && !parent.isEmpty()) {
                    input.getRequestInfoBuilder()
                            .setTraceContext(
                                    input.getRequestInfo().getTraceContext().toBuilder()
                                            .setTraceparent(parent)
                                            .setTracestate(carrier.getOrDefault("tracestate", "")));
                }
            } catch (Throwable ignored) {
            }
        }
        var config = input.getGenerateConfigBuilder().clearRoleAddrs();
        var prefill = item.prefill();
        var decode = item.decode();
        if (prefill != null) {
            addresses.prefill = role(prefill, addresses.prefill);
            config.addRoleAddrs(addresses.prefill);
        }
        if (decode != null) {
            addresses.decode = role(decode, addresses.decode);
            config.addRoleAddrs(addresses.decode);
        }
        return input.setPriority(item.priority()).build();
    }

    private static EngineRpcService.RoleAddrPB role(
            ServerStatus status, EngineRpcService.RoleAddrPB cached) {
        var role = status.getRole();
        if (cached != null
                && cached.getRoleStr().equals(role.getCode())
                && cached.getIp().equals(status.getServerIp())
                && cached.getHttpPort() == status.getHttpPort()
                && cached.getGrpcPort() == status.getGrpcPort()) {
            return cached;
        }
        return EngineRpcService.RoleAddrPB.newBuilder()
                .setRole(RoleTypeProtoConverter.toRoleAddrProto(role))
                .setRoleStr(role.getCode())
                .setIp(status.getServerIp())
                .setHttpPort(status.getHttpPort())
                .setGrpcPort(status.getGrpcPort())
                .build();
    }

    private static final class RoleAddresses {
        private EngineRpcService.RoleAddrPB prefill;
        private EngineRpcService.RoleAddrPB decode;
    }
}
