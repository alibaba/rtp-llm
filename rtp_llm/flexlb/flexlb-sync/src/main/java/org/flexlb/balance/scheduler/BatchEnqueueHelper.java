package org.flexlb.balance.scheduler;

import com.google.protobuf.ByteString;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import com.google.protobuf.InvalidProtocolBufferException;
import com.google.protobuf.UnknownFieldSet;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.engine.grpc.EngineRpcService.GenerateInputPB;
import org.flexlb.engine.grpc.RoleTypeProtoConverter;
import org.flexlb.telemetry.FlexlbTrace;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.concurrent.CompletionException;

import static com.google.common.base.Preconditions.checkArgument;

/**
 * Parses immutable request bytes and builds EnqueueBatch payloads for exact selected members.
 * Validates the complete response against the sent members; owns no request or resource state.
 */
final class BatchEnqueueHelper {
    /**
     * Validate input without allocating a decoded token array; Engine sees the same protobuf wire fields.
     */
    static GenerateInputPB parseInput(ByteString bytes) throws InvalidProtocolBufferException {
        try {
            CodedInputStream input = bytes.newCodedInput();
            var metadata = GenerateInputPB.newBuilder();
            var tokens = new Tokens();
            int retainedStart = 0;
            int tagStart = 0;
            for (int tag; (tag = input.readTag()) != 0; tagStart = input.getTotalBytesRead()) {
                if (tag == 16 || tag == 18) {
                    mergeMetadata(metadata, bytes, retainedStart, tagStart);
                    tokens.readField(tag, input, bytes);
                    retainedStart = input.getTotalBytesRead();
                    continue;
                }
                if (!input.skipField(tag)) {
                    input.checkLastTagWas(0);
                }
            }
            input.checkLastTagWas(0);
            mergeMetadata(metadata, bytes, retainedStart, bytes.size());
            var parsed = metadata.build();
            ByteString encoded = tokens.finish();
            if (!encoded.isEmpty() && bytes.asReadOnlyByteBufferList().size() > 1) {
                // A segmented input may use a copying stream decoder for metadata. Give
                // tokens their own compact storage so they cannot pin copied large fields.
                var compact = ByteString.newOutput(encoded.size());
                encoded.writeTo(compact);
                encoded = compact.toByteString();
            }
            if (!encoded.isEmpty()) {
                // Unknown fields use the standard protobuf writer. On the Engine this is the
                // known packed token_ids field again; no schema or RPC method changes.
                var field = UnknownFieldSet.Field.newBuilder().addLengthDelimited(encoded).build();
                parsed =
                        parsed.toBuilder()
                                .mergeUnknownFields(
                                        UnknownFieldSet.newBuilder().mergeField(2, field).build())
                                .build();
            }
            return parsed;
        } catch (InvalidProtocolBufferException malformed) {
            // Diagnostics can outlive dispatch. Never let a parse failure pin a partial
            // tensor/message.
            throw malformed.setUnfinishedMessage(null);
        } catch (IOException failure) {
            throw new InvalidProtocolBufferException(failure);
        }
    }

    /** O(1) wire-size bound using the route's exact, immutable endpoint addresses. */
    static long payloadSizeUpperBound(RouteResult route) {
        return route.ctx().data().getGenerateInputSize()
                + 64L
                + addressSizeUpperBound(route.prefillEndpoint().getIp())
                + (route.decodeEndpoint() == null
                        ? 0L
                        : addressSizeUpperBound(route.decodeEndpoint().getIp()));
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

    private static void mergeMetadata(
            GenerateInputPB.Builder metadata, ByteString bytes, int start, int end)
            throws IOException {
        if (start == end) {
            return;
        }
        var input = bytes.substring(start, end).newCodedInput();
        // ByteString is immutable: tensors and unknown byte fields can share its storage.
        input.enableAliasing(true);
        metadata.mergeFrom(input);
        input.checkLastTagWas(0);
    }

    private static long addressSizeUpperBound(String ip) {
        return 64L + (ip == null ? 0L : 3L * ip.length());
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

    private static EngineRpcService.GenerateInputPB input(RouteResult item, RoleAddresses addresses)
            throws Exception {
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
        var prefill = item.role(RoleType.PREFILL);
        var decode = item.role(RoleType.DECODE);
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
            RoleSelection selected, EngineRpcService.RoleAddrPB cached) {
        var role = selected.role();
        if (cached != null
                && cached.getRoleStr().equals(role.getCode())
                && cached.getIp().equals(selected.ip())
                && cached.getHttpPort() == selected.httpPort()
                && cached.getGrpcPort() == selected.grpcPort()) {
            return cached;
        }
        return EngineRpcService.RoleAddrPB.newBuilder()
                .setRole(RoleTypeProtoConverter.toRoleAddrProto(role))
                .setRoleStr(role.getCode())
                .setIp(selected.ip())
                .setHttpPort(selected.httpPort())
                .setGrpcPort(selected.grpcPort())
                .build();
    }

    /**
     * Combines packed and unpacked token fields while preserving the original token sequence.
     */
    private static final class Tokens {
        private ByteString packed;
        private ByteString.Output combined;
        private CodedOutputStream writer;

        private void readField(int tag, CodedInputStream input, ByteString bytes)
                throws IOException {
            if (tag == 16) {
                writer().writeInt32NoTag(input.readInt32());
                return;
            }
            int length = input.readRawVarint32();
            int previousLimit = input.pushLimit(length);
            int start = input.getTotalBytesRead();
            while (input.getBytesUntilLimit() > 0) {
                input.readInt32();
            }
            input.popLimit(previousLimit);
            ByteString chunk = bytes.substring(start, input.getTotalBytesRead());
            if (packed == null && writer == null) {
                packed = chunk;
            } else {
                writer().writeRawBytes(chunk);
            }
        }

        private CodedOutputStream writer() throws IOException {
            if (writer == null) {
                combined = ByteString.newOutput(packed == null ? 32 : Math.max(32, packed.size()));
                writer = CodedOutputStream.newInstance(combined, 64);
                if (packed != null) {
                    writer.writeRawBytes(packed);
                    packed = null;
                }
            }
            return writer;
        }

        private ByteString finish() throws IOException {
            if (writer == null) {
                return packed == null ? ByteString.EMPTY : packed;
            }
            writer.flush();
            return combined.toByteString();
        }
    }

    private static final class RoleAddresses {
        private EngineRpcService.RoleAddrPB prefill;
        private EngineRpcService.RoleAddrPB decode;
    }
}
