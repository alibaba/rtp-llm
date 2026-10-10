package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.DeliveryResult;
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

/** Complete payload preparation and atomic whole-response validation; no request lifecycle effects. */
final class BatchCodec {
    static EngineRpcService.EnqueueBatchRequestPB prepare(long batchId, List<DeliveryClaim> claims) {
        try {
            var builder = EngineRpcService.EnqueueBatchRequestPB.newBuilder().setBatchId(batchId)
                    .setFetchAttachTimeoutMs(claims.getFirst().owner().data().getConfig().getDispatcher().getFetchAttachTimeoutMs());
            var addresses = new RoleAddresses();
            long firstRank = claims.getFirst().item().prefill().getDpRank();
            var firstSlot = EngineRpcService.EnqueueBatchDpSlotPB.newBuilder().setDpRank((int) firstRank);
            Map<Long, EngineRpcService.EnqueueBatchDpSlotPB.Builder> slots = null;
            for (DeliveryClaim claim : claims) {
                var item = claim.item();
                long rank = item.prefill().getDpRank();
                var slot = firstSlot;
                if (rank != firstRank) {
                    if (slots == null) { slots = new TreeMap<>(); slots.put(firstRank, firstSlot); }
                    slot = slots.computeIfAbsent(rank, key -> EngineRpcService.EnqueueBatchDpSlotPB.newBuilder().setDpRank(key.intValue()));
                }
                slot.addRequests(EngineRpcService.EnqueueBatchExternalInputPB.newBuilder().setInput(input(item, addresses)));
            }
            if (slots == null) { builder.addDpSlots(firstSlot); }
            else { slots.values().forEach(builder::addDpSlots); }
            return builder.build();
        } catch (Exception failure) {
            throw new IllegalArgumentException("Batch request build failed: " + failure.getMessage(), failure);
        }
    }

    /** Remove abandoned members from an already complete immutable payload without parsing again. */
    static EngineRpcService.EnqueueBatchRequestPB retain(EngineRpcService.EnqueueBatchRequestPB prepared, List<DeliveryClaim> sending) {
        var ids = new HashSet<Long>();
        for (DeliveryClaim claim : sending) { ids.add(claim.item().requestId()); }
        var filtered = prepared.toBuilder().clearDpSlots();
        for (var slot : prepared.getDpSlotsList()) {
            var selected = slot.toBuilder().clearRequests();
            for (var member : slot.getRequestsList()) {
                if (ids.contains(member.getInput().getRequestId())) { selected.addRequests(member); }
            }
            if (selected.getRequestsCount() > 0) { filtered.addDpSlots(selected); }
        }
        return filtered.build();
    }

    /** Results have the exact sending-claim order. One malformed member invalidates the entire ACK. */
    static List<DeliveryResult> validate(long batchId, List<DeliveryClaim> claims,
            EngineRpcService.EnqueueBatchResponsePB response, Throwable failure) {
        if (failure != null || response == null || response.getBatchId() != batchId) {
            Throwable problem = failure instanceof CompletionException && failure.getCause() != null ? failure.getCause() : failure;
            if (problem == null) { problem = new IllegalStateException("Invalid EnqueueBatch response for batch " + batchId); }
            return Collections.nCopies(claims.size(), DeliveryResult.uncertain(problem));
        }
        var expected = HashMap.<Long, Integer>newHashMap(claims.size());
        var results = new DeliveryResult[claims.size()];
        var violations = new ArrayList<String>();
        for (int i = 0; i < claims.size(); i++) {
            long id = claims.get(i).item().requestId();
            if (expected.put(id, i) != null) { violations.add("duplicate sending request_id=" + id); }
        }
        for (var error : response.getErrorsList()) {
            long code = error.hasErrorInfo() ? error.getErrorInfo().getErrorCode() : 0L;
            String message = error.hasErrorInfo() ? error.getErrorInfo().getErrorMessage() : "missing error_info";
            acknowledge(expected, results, error.getRequestId(), DeliveryResult.prefillRejected(new RuntimeException(
                    "EnqueueBatch rejected request " + error.getRequestId() + " error_code=" + code + ": " + message)), violations);
        }
        for (var success : response.getSuccessesList()) {
            acknowledge(expected, results, success.getRequestId(), DeliveryResult.delivered(), violations);
        }
        expected.forEach((id, index) -> { if (index >= 0 && results[index] == null) { violations.add("response is missing request_id=" + id); } });
        if (!violations.isEmpty()) {
            return Collections.nCopies(claims.size(), DeliveryResult.uncertain(new IllegalStateException(
                    "Malformed EnqueueBatch response: " + String.join("; ", violations))));
        }
        return List.of(results);
    }

    private static void acknowledge(Map<Long, Integer> expected, DeliveryResult[] results, long id,
            DeliveryResult result, List<String> violations) {
        String kind = result.status() == DeliveryResult.Status.DELIVERED ? "success" : "error";
        Integer index = expected.get(id);
        if (index == null || index < 0) {
            violations.add(kind + " references unknown request_id=" + id);
            int unknownKind = result.status() == DeliveryResult.Status.DELIVERED ? -1 : -2;
            if (index == null) { expected.put(id, unknownKind); }
            else { violations.add(index == unknownKind ? "duplicate " + kind + " for request_id=" + id
                    : "request_id appears in both success and error: " + id); }
            return;
        }
        var previous = results[index];
        if (previous != null) {
            violations.add(previous.status() == result.status() ? "duplicate " + kind + " for request_id=" + id
                    : "request_id appears in both success and error: " + id);
        } else { results[index] = result; }
    }

    private static EngineRpcService.GenerateInputPB input(RequestRoute item, RoleAddresses addresses) throws Exception {
        var prepared = item.ctx().data().getGenerateInput();
        if (prepared == null) { throw new IllegalArgumentException("generateInputPb is missing for request " + item.requestId()); }
        var input = prepared.toBuilder();
        checkArgument(input.getRequestId() == item.requestId(), "request_id mismatch between schedule request and GenerateInputPB");
        if (FlexlbTrace.isEnabled() && item.ctx().data().getTraceContext() != null) {
            try {
                var carrier = FlexlbTrace.inject(item.ctx().data().getTraceContext());
                String parent = carrier.get("traceparent");
                if (parent != null && !parent.isEmpty()) {
                    input.getRequestInfoBuilder().setTraceContext(input.getRequestInfo().getTraceContext().toBuilder()
                            .setTraceparent(parent).setTracestate(carrier.getOrDefault("tracestate", "")));
                }
            } catch (Throwable ignored) { }
        }
        var config = input.getGenerateConfigBuilder().clearRoleAddrs();
        if (item.prefill() != null) { addresses.prefill = role(item.prefill(), addresses.prefill); config.addRoleAddrs(addresses.prefill); }
        if (item.decode() != null) { addresses.decode = role(item.decode(), addresses.decode); config.addRoleAddrs(addresses.decode); }
        return input.setPriority(item.priority()).build();
    }

    private static EngineRpcService.RoleAddrPB role(ServerStatus status, EngineRpcService.RoleAddrPB cached) {
        var role = status.getRole();
        if (cached != null && cached.getRoleStr().equals(role.getCode()) && cached.getIp().equals(status.getServerIp())
                && cached.getHttpPort() == status.getHttpPort() && cached.getGrpcPort() == status.getGrpcPort()) { return cached; }
        return EngineRpcService.RoleAddrPB.newBuilder().setRole(RoleTypeProtoConverter.toLegacyProto(role))
                .setRoleStr(role.getCode()).setIp(status.getServerIp()).setHttpPort(status.getHttpPort())
                .setGrpcPort(status.getGrpcPort()).build();
    }

    private static final class RoleAddresses {
        private EngineRpcService.RoleAddrPB prefill;
        private EngineRpcService.RoleAddrPB decode;
    }
}
